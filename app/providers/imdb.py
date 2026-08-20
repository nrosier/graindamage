"""IMDb ``/technical`` specifications: fetch (optional) and parse (always).

IMDb has had a WAF in front of the site since April 2026, so this app never scrapes
it. Two routes in:

1. **Paste** the page source. Always available, needs no configuration, and is the
   documented path.
2. **Fetch** through a service the operator controls (``IMDB_FETCHER_URL``) — a proxy,
   a browserless instance, anything that takes a URL and returns page source.

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

        The fetcher is called as ``GET {IMDB_FETCHER_URL}?url={imdb_url}`` with an
        optional bearer token. Either raw HTML or a JSON envelope with a
        ``content`` / ``html`` / ``body`` / ``data`` key is accepted, which covers
        browserless, most scraping APIs, and a twenty-line Worker.
        """
        if not self.enabled:
            raise ProviderDisabled(
                "No IMDb fetcher is configured. Paste the /technical page source instead."
            )
        if not re.fullmatch(r"tt\d{5,}", imdb_id):
            raise ProviderUnavailable(f"{imdb_id!r} is not an IMDb title id.")

        target = f"https://www.imdb.com/title/{imdb_id}/technical/"
        headers = {"Accept": "text/html,application/json", "User-Agent": self._settings.user_agent}
        if self._settings.imdb_fetcher_token:
            headers["Authorization"] = f"Bearer {self._settings.imdb_fetcher_token}"

        url = str(self._settings.imdb_fetcher_url)
        timeout = self._settings.imdb_fetcher_timeout_seconds
        try:
            if self._client is not None:
                response = await self._client.get(url, params={"url": target}, headers=headers)
            else:
                async with httpx2.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    response = await client.get(url, params={"url": target}, headers=headers)
        except httpx2.HTTPError as exc:
            raise ProviderUnavailable(
                "Could not reach the IMDb fetcher.", detail=type(exc).__name__
            ) from exc

        if response.status_code >= 400:
            raise ProviderUnavailable(f"The IMDb fetcher returned HTTP {response.status_code}.")

        body = response.text
        if "json" in response.headers.get("content-type", "").casefold():
            body = _unwrap_json_envelope(body) or body
        if not body.strip():
            raise ProviderUnavailable("The IMDb fetcher returned an empty body.")
        return body

    async def fetch_specs(self, imdb_id: str) -> TechnicalSpecs:
        async def parse() -> TechnicalSpecs:
            return parse_technical(await self.fetch(imdb_id))

        return await self._cache.get_or_set(imdb_id, parse)


def _unwrap_json_envelope(body: str) -> str | None:
    try:
        payload: Any = json.loads(body)
    except ValueError:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("content", "html", "body", "data", "result", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None
