"""IMDb ``/technical`` specifications: fetch (optional) and parse (always).

IMDb has had a WAF in front of the site since April 2026, so this app never scrapes
it. Two routes in:

1. **Paste** the page source. Always available, needs no configuration, and is the
   documented path.
2. **Fetch** through a service the operator controls (``IMDB_FETCHER_URL``) — Byparr,
   FlareSolverr, a browserless instance, a proxy, anything that takes a URL and returns
   page source. Three wire contracts are spoken; see :func:`_plan_request`.

   IMDb answers a fresh browser with a JavaScript bot check, so a fetched page is
   examined for one and asked for again rather than parsed: answering the check earns
   the browser a token, and a fetcher that keeps a session serves the real page on a
   later attempt. ``IMDB_FETCHER_ATTEMPTS`` bounds that.

Parsing tries three layouts in order, because IMDb has shipped all three and a saved
page could be any of them:

1. The ``__NEXT_DATA__`` JSON island — structured, so no HTML parsing at all.
2. Current server-rendered markup, keyed on ``data-testid="title-techspec_*"``.
3. The pre-2020 ``<td class="label">`` table.

There is no HTML parser dependency: the JSON island covers the common case, and the
markup fallbacks only need to find a labelled block and strip tags out of it, which
regex does adequately for a page we neither trust nor need to render.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx2

from app.cache import TTLCache
from app.config import Settings
from app.models import TechnicalSpecs
from app.providers import ProviderDisabled, ProviderUnavailable
from app.sources.parsing import clean

# IMDb's key for each row -> the TechnicalSpecs field it fills.
SPEC_FIELDS: dict[str, str] = {
    "runtime": "runtimes",
    "soundmix": "sound_mixes",
    "aspectratio": "aspect_ratios",
    "camera": "cameras",
    "laboratory": "laboratories",
    "negative_format": "negative_formats",
    "cinematographic_process": "cinematographic_processes",
    "printed_film_format": "printed_formats",
    "film_length": "film_lengths",
    "colorations": "colors",
    "color": "colors",
}

# The same rows as they are labelled in the two markup layouts.
SPEC_LABELS: dict[str, str] = {
    "runtime": "runtimes",
    "sound mix": "sound_mixes",
    "aspect ratio": "aspect_ratios",
    "camera": "cameras",
    "cameras": "cameras",
    "laboratory": "laboratories",
    "negative format": "negative_formats",
    "cinematographic process": "cinematographic_processes",
    "printed film format": "printed_formats",
    "film length": "film_lengths",
    "color": "colors",
    "colorations": "colors",
}

_SPEC_FIELD_NAMES = frozenset(TechnicalSpecs.model_fields)
_MAX_ITEMS_PER_FIELD = 12
_MAX_ITEM_CHARS = 200

_NEXT_DATA = re.compile(
    r'<script[^>]+id="__NEXT_DATA__"[^>]*>(?P<json>.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
_TESTID_BLOCK = re.compile(
    # ``[^>]*>`` finishes the element that carries the testid, so the attributes that
    # follow it do not end up in the row's text as ``class="…"``.
    r'data-testid="title-techspec_(?P<key>[a-z_]+)"[^>]*>'
    r'(?P<body>.*?)(?=data-testid="title-techspec_|</ul>|</section>)',
    re.DOTALL | re.IGNORECASE,
)
_LEGACY_ROW = re.compile(
    r'<td\s+class="label"[^>]*>(?P<label>.*?)</td>(?P<body>.*?)</tr>',
    re.DOTALL | re.IGNORECASE,
)
_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_BLOCK_END = re.compile(r"</(?:li|span|div|td|p|a|h\d)\s*>|<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]*>")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Row values that carry no information and would otherwise pollute the lists.
_NOISE = {
    "",
    "see more",
    "see less",
    "edit",
    "add",
    "contribute to this page",
    "technical specs",
    "technical specifications",
}


def _text_items(fragment: str) -> list[str]:
    """Strip markup, keeping each list item as its own string."""
    fragment = _SCRIPT_OR_STYLE.sub(" ", fragment)
    fragment = _BLOCK_END.sub("\x00", fragment)
    fragment = _TAG.sub("", fragment)
    text = html.unescape(fragment)
    return [item for part in text.split("\x00") if (item := clean(part))]


def _squash(text: str) -> str:
    """Casefolded, with punctuation and spacing removed.

    Labels reach us in two spellings — the visible ``Sound mix`` and the JSON key
    ``soundmix`` — and both have to be recognised as the same row heading.
    """
    return _NON_ALNUM.sub("", text.casefold())


def _keep(items: list[str], *, label: str | None = None) -> list[str]:
    """Drop noise, the row's own label, and duplicates, preserving order."""
    banned = {_squash(entry) for entry in _NOISE}
    if label:
        banned.add(_squash(label))

    kept: list[str] = []
    for item in items:
        candidate = clean(item).rstrip(":")
        folded = candidate.casefold()
        if not candidate or _squash(candidate) in banned or len(candidate) > _MAX_ITEM_CHARS:
            continue
        if folded in {existing.casefold() for existing in kept}:
            continue
        kept.append(candidate)
        if len(kept) >= _MAX_ITEMS_PER_FIELD:
            break
    return kept


def _strings_in(value: object, *, depth: int = 0) -> list[str]:
    """Every string leaf in a JSON subtree, in document order."""
    if depth > 8:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for entry in value for item in _strings_in(entry, depth=depth + 1)]
    if isinstance(value, dict):
        found: list[str] = []
        for key, entry in value.items():
            # Skip GraphQL bookkeeping and URLs, which are never spec text.
            if key in {"__typename", "id", "url", "href", "originalTitleText"}:
                continue
            found.extend(_strings_in(entry, depth=depth + 1))
        return found
    return []


def _from_next_data(page: str) -> dict[str, list[str]]:
    """Read the spec rows out of the ``__NEXT_DATA__`` island.

    The path to them has moved between IMDb releases, so instead of walking a fixed
    path this looks for any object whose ``id`` is a known spec key and harvests the
    strings beneath it.
    """
    match = _NEXT_DATA.search(page)
    if not match:
        return {}
    try:
        payload = json.loads(match.group("json"))
    except ValueError:
        return {}

    collected: dict[str, list[str]] = {}

    def visit(node: object, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(node, list):
            for entry in node:
                visit(entry, depth + 1)
            return
        if not isinstance(node, dict):
            return

        node_id = node.get("id")
        field = SPEC_FIELDS.get(node_id) if isinstance(node_id, str) else None
        if field:
            body: object = node
            for key in ("section", "items", "rows"):
                if isinstance(nested := node.get(key), dict | list):
                    body = nested
                    break
            # The heading is only worth suppressing when it is part of what was
            # harvested. A nested section never contains it, and IMDb's Color row
            # says literally "Color" — dropping that would lose the only value.
            label = str(node.get("name") or node_id) if body is node else None
            values = _keep(_strings_in(body), label=label)
            if values:
                collected.setdefault(field, []).extend(values)

        for entry in node.values():
            visit(entry, depth + 1)

    visit(payload)
    return collected


def _from_testid_markup(page: str) -> dict[str, list[str]]:
    collected: dict[str, list[str]] = {}
    for match in _TESTID_BLOCK.finditer(page):
        key = match.group("key").casefold()
        field = SPEC_FIELDS.get(key)
        if not field:
            continue
        # The block starts at the testid attribute, so it contains the row's own
        # visible label ("Negative format") before the values. The key is the same
        # word with the punctuation removed, which is enough to recognise it.
        items = _keep(_text_items(match.group("body")), label=key)
        if items:
            collected.setdefault(field, []).extend(items)
    return collected


def _from_legacy_table(page: str) -> dict[str, list[str]]:
    collected: dict[str, list[str]] = {}
    for match in _LEGACY_ROW.finditer(page):
        label = clean(_TAG.sub("", match.group("label"))).rstrip(":")
        field = SPEC_LABELS.get(label.casefold())
        if not field:
            continue
        items = _keep(_text_items(match.group("body")), label=label)
        if items:
            collected.setdefault(field, []).extend(items)
    return collected


def parse_technical(page: str) -> TechnicalSpecs:
    """Parse an IMDb ``/technical`` page into :class:`TechnicalSpecs`.

    Never raises: an unrecognised page yields empty specs, and the caller decides
    whether that is worth telling the user about (it is — the advice just falls back
    to source-only heuristics).
    """
    if not page or not page.strip():
        return TechnicalSpecs()

    merged: dict[str, list[str]] = {}
    for extractor in (_from_next_data, _from_testid_markup, _from_legacy_table):
        for field, values in extractor(page).items():
            merged.setdefault(field, []).extend(values)
        # Stop at the first layout that produced the format rows we actually need.
        if merged.get("negative_formats") or merged.get("aspect_ratios"):
            break

    return TechnicalSpecs(
        **{field: _keep(values) for field, values in merged.items() if field in _SPEC_FIELD_NAMES}
    )


class ImdbTechnicalProvider:
    """Retrieves ``/technical`` page source through an operator-run fetcher."""

    def __init__(self, settings: Settings, *, client: httpx2.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client
        # Previewing the specs and then asking for advice must not cost two fetches.
        self._cache: TTLCache[TechnicalSpecs] = TTLCache(
            ttl_seconds=float(settings.cache_ttl_seconds)
        )

    @property
    def enabled(self) -> bool:
        return self._settings.imdb_fetcher_enabled

    async def fetch(self, imdb_id: str) -> str:
        """Return the page source for ``tt…``'s technical page.

        The request is shaped by :func:`_plan_request`. Either raw HTML or a JSON
        envelope is accepted — ``solution.response`` for Byparr and FlareSolverr, or a
        top-level ``content`` / ``html`` / ``body`` / ``data`` / ``result`` / ``text``
        key — which covers browserless, most scraping APIs, and a twenty-line Worker.

        A page that is a bot check rather than IMDb is asked for again, and reported as
        such if every attempt is met with one.
        """
        if not self.enabled:
            raise ProviderDisabled(
                "No IMDb fetcher is configured. Paste the /technical page source instead."
            )
        if not re.fullmatch(r"tt\d{5,}", imdb_id):
            raise ProviderUnavailable(f"{imdb_id!r} is not an IMDb title id.")

        plan = _plan_request(self._settings, f"https://www.imdb.com/title/{imdb_id}/technical/")
        attempts = self._settings.imdb_fetcher_attempts

        page = ""
        for _ in range(attempts):
            page = await self._fetch_once(plan)
            if not _looks_like_a_challenge(page):
                return page

        tries = f"{attempts} attempt" + ("s" if attempts != 1 else "")
        raise ProviderUnavailable(
            "The IMDb fetcher was served a bot check instead of the technical page.",
            detail=f"{tries} met {_challenge_flavour(page)}. Pasting the source always works.",
        )

    async def _fetch_once(self, plan: _FetchRequest) -> str:
        """One call to the fetcher, reduced to page source."""

        async def send(client: httpx2.AsyncClient) -> httpx2.Response:
            return await client.request(
                plan.method,
                plan.url,
                params=plan.params,
                headers=plan.headers,
                json=plan.payload,
            )

        timeout = self._settings.imdb_fetcher_timeout_seconds
        try:
            if self._client is not None:
                response = await send(self._client)
            else:
                async with httpx2.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    response = await send(client)
        except httpx2.HTTPError as exc:
            raise ProviderUnavailable(
                "Could not reach the IMDb fetcher.", detail=type(exc).__name__
            ) from exc

        if response.status_code >= 400:
            raise ProviderUnavailable(
                f"The IMDb fetcher returned HTTP {response.status_code}.",
                detail=_error_detail(response),
            )

        body = response.text
        if "json" in response.headers.get("content-type", "").casefold():
            if failure := _envelope_failure(body):
                raise ProviderUnavailable(f"The IMDb fetcher reported: {failure}")
            body = _unwrap_json_envelope(body) or body
        if not body.strip():
            raise ProviderUnavailable("The IMDb fetcher returned an empty body.")
        return body

    async def fetch_specs(self, imdb_id: str) -> TechnicalSpecs:
        async def parse() -> TechnicalSpecs:
            return parse_technical(await self.fetch(imdb_id))

        return await self._cache.get_or_set(imdb_id, parse)


@dataclass(frozen=True, slots=True)
class _FetchRequest:
    """One outbound call to the operator's fetcher, fully shaped."""

    kind: str
    method: str
    url: str
    params: dict[str, str]
    headers: dict[str, str]
    payload: dict[str, Any] | None


# The endpoint names each service answers on. A configured URL ending in one of these
# is that service rather than a fetcher someone wrote; "" is a bare host:port, which is
# how a browserless instance usually gets written down.
_BROWSERLESS_ENDPOINTS = frozenset({"", "content", "unblock"})
_FLARESOLVERR_ENDPOINTS = frozenset({"v1"})

# Byparr and FlareSolverr report a failure in the envelope, at HTTP 200. Anything else
# in ``status`` is a failure worth repeating to the user.
_ENVELOPE_OK = frozenset({"ok", "success"})

# Browserless announces itself as HeadlessChrome, which is the first thing a WAF looks
# at, so the browser it drives is given a plausible desktop identity instead. The
# version ages harmlessly: it only has to be a version that exists.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
)

# FlareSolverr keeps one browser per session name and creates it on first use, so the
# token its browser earns answering IMDb's bot check is still there for the next
# attempt — which is what makes retrying worthwhile. The TTL stops a name from holding
# a browser open forever. Byparr has no sessions and ignores both fields.
_FLARESOLVERR_SESSION = "graindamage-imdb"
_FLARESOLVERR_SESSION_TTL_MINUTES = 30

# What a bot check standing in for the page looks like. Every marker was checked
# against a real /technical page, which contains none of them.
_CHALLENGE_MARKERS = (
    "awswafcookiedomainlist",
    "awswafintegration",
    "gokuprops",
    "human verification",
    "cf_chl_opt",
    "cf-browser-verification",
    "challenge-platform",
    "checking your browser",
    "just a moment",
)
_AWS_WAF_MARKERS = ("awswaf", "gokuprops", "human verification")

# The interstitials are a couple of kilobytes; IMDb's own page is a megabyte. Only the
# head is searched, which is where a challenge keeps everything.
_CHALLENGE_WINDOW = 20_000


def _endpoint_name(url: httpx2.URL) -> str:
    """The last path segment, so a reverse-proxied ``/browserless/content`` still counts."""
    return url.path.strip("/").rsplit("/", 1)[-1].casefold()


def _fetcher_kind(mode: str, endpoint_name: str) -> str:
    """Which contract to speak: the configured mode, or the endpoint name's own answer."""
    if mode != "auto":
        return mode
    if endpoint_name in _FLARESOLVERR_ENDPOINTS:
        return "flaresolverr"
    if endpoint_name in _BROWSERLESS_ENDPOINTS:
        return "browserless"
    return "query"


def _budget_ms(settings: Settings) -> int:
    """90% of the HTTP timeout, in milliseconds.

    The browser is given slightly less time than the connection to it, so a slow page
    comes back as the service's own error rather than as a severed connection.
    """
    return max(1000, int(settings.imdb_fetcher_timeout_seconds * 900))


def _plan_request(settings: Settings, target: str) -> _FetchRequest:
    """Shape the fetcher call for whichever contract the configured URL speaks.

    ``query`` — a proxy or Worker of your own::

        GET {IMDB_FETCHER_URL}?url={target}      Authorization: Bearer {token}

    ``browserless`` — a bare ``host:port``, ``/content`` or ``/unblock``::

        POST {IMDB_FETCHER_URL}?token={token}    {"url": "{target}", …}

    ``flaresolverr`` — Byparr or FlareSolverr, at ``/v1``::

        POST {IMDB_FETCHER_URL}                  {"cmd": "request.get", "url": "{target}"}

    None of the three can be configured into the others' shape — browserless has no
    ``?url=`` route, FlareSolverr takes a command rather than a URL parameter — so the
    difference has to live here. ``IMDB_FETCHER_MODE`` forces one when the endpoint
    name guesses wrong.
    """
    url = httpx2.URL(str(settings.imdb_fetcher_url))
    token = settings.imdb_fetcher_token
    # Whatever the operator put in the URL is kept: an API key already in the query
    # string must survive the parameters this adds.
    params: dict[str, str] = dict(url.params)
    headers = {"Accept": "text/html,application/json", "User-Agent": settings.user_agent}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    endpoint_name = _endpoint_name(url)
    kind = _fetcher_kind(settings.imdb_fetcher_mode, endpoint_name)

    if kind == "query":
        return _FetchRequest(kind, "GET", str(url), {**params, "url": target}, headers, None)

    if kind == "flaresolverr":
        # Byparr implements FlareSolverr's API: one command, and the page comes back in
        # solution.response. maxTimeout is milliseconds (Byparr reads anything >= 1000
        # as ms too), and it covers solving the challenge as well as loading the page.
        # session and session_ttl_minutes are what let a retry benefit from the token
        # the previous attempt's browser earned; see _FLARESOLVERR_SESSION.
        command = {
            "cmd": "request.get",
            "url": target,
            "maxTimeout": _budget_ms(settings),
            "session": _FLARESOLVERR_SESSION,
            "session_ttl_minutes": _FLARESOLVERR_SESSION_TTL_MINUTES,
        }
        return _FetchRequest(kind, "POST", str(url), params, headers, command)

    # Browserless. /content is the endpoint; a bare host:port means the operator wrote
    # down the instance rather than the route.
    endpoint = str(url) if endpoint_name else str(url.copy_with(path="/content", query=None))
    if token:
        # browserless v1 only reads the query parameter; v2 reads either. The bearer
        # header stays on for whatever reverse proxy may be in front of it.
        params["token"] = token

    if endpoint_name == "unblock":
        # /unblock drives a stealth browser for WAF-protected pages and returns JSON;
        # "content" is what asks for the HTML in it. It takes no gotoOptions.
        payload: dict[str, Any] = {"url": target, "content": True}
        return _FetchRequest(kind, "POST", endpoint, params, headers, payload)

    payload = {
        "url": target,
        # The parser reads __NEXT_DATA__, which is in the initial HTML, so there is
        # nothing to gain by waiting for network idle — that only waits for the ads.
        "gotoOptions": {"waitUntil": "domcontentloaded", "timeout": _budget_ms(settings)},
        "userAgent": {"userAgent": _BROWSER_USER_AGENT},
        "setExtraHTTPHeaders": {"Accept-Language": "en-US,en;q=0.9"},
        # Return whatever loaded rather than nothing at all when a wait times out.
        "bestAttempt": True,
    }
    return _FetchRequest(kind, "POST", endpoint, params, headers, payload)


def _looks_like_a_challenge(page: str) -> bool:
    """Whether the fetcher was served a bot check instead of IMDb.

    A page carrying IMDb's own data is never one, whatever else it mentions, so the
    markers only decide for pages that have no specs in them anyway.
    """
    if "__NEXT_DATA__" in page or "title-techspec" in page:
        return False
    head = page[:_CHALLENGE_WINDOW].casefold()
    return any(marker in head for marker in _CHALLENGE_MARKERS)


def _challenge_flavour(page: str) -> str:
    """Whose bot check it was, so the reader knows what stopped them."""
    head = page[:_CHALLENGE_WINDOW].casefold()
    if any(marker in head for marker in _AWS_WAF_MARKERS):
        return "IMDb's AWS WAF JavaScript challenge"
    return "a JavaScript bot check"


def _error_detail(response: httpx2.Response) -> str | None:
    """The service's own explanation for an HTTP error, if it gave one.

    Byparr answers 408 with ``{"detail": "Timed out while … solving the challenge"}``,
    which is worth more to the reader than the status code alone.
    """
    try:
        payload: Any = json.loads(response.text)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("detail", "message", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:200]
    return None


def _envelope_failure(body: str) -> str | None:
    """The failure a FlareSolverr-shaped envelope reports at HTTP 200, if any.

    Byparr and FlareSolverr answer ``{"status": "error", "message": …}`` for some
    failures, which would otherwise reach the parser as an unrecognisable page and be
    reported as a film with no technical specifications.
    """
    try:
        payload: Any = json.loads(body)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    if not isinstance(status, str) or status.casefold() in _ENVELOPE_OK:
        return None
    for key in ("message", "detail", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:200]
    return status


def _unwrap_json_envelope(body: str) -> str | None:
    try:
        payload: Any = json.loads(body)
    except ValueError:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        # FlareSolverr and Byparr nest the page one level down, in solution.response.
        solution = payload.get("solution")
        if isinstance(solution, dict):
            nested = solution.get("response")
            if isinstance(nested, str) and nested.strip():
                return nested
        for key in ("content", "html", "body", "data", "result", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None
