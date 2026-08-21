"""IMDb ``/technical`` specifications: read the rows out of whatever the user has.

IMDb has had a WAF in front of the site since April 2026 which answers an automated
request with a JavaScript bot check, so this app does not fetch the page at all. The
rows arrive one of two ways, and neither is a scrape:

1. **Looked up** through Gemini, which is asked for the film's technical rows — see
   :meth:`app.providers.gemini.GeminiClient.technical_specs`. Labelled as a lookup in
   the UI, because a language model's recollection of a page is not the page.
2. **Pasted** by the user, who has the page open anyway. Both the formatted text you
   get from selecting the page and the HTML you get from view-source are understood.

Parsing tries four layouts in order, because IMDb has shipped three of them and a
paste could be any of them:

1. The ``__NEXT_DATA__`` JSON island — structured, so no HTML parsing at all.
2. Current server-rendered markup, keyed on ``data-testid="title-techspec_*"``.
3. The pre-2020 ``<td class="label">`` table.
4. Formatted text: a label, then its values, as the page reads on screen.

There is no HTML parser dependency: the JSON island covers the common case, and the
markup fallbacks only need to find a labelled block and strip tags out of it, which
regex does adequately for a page we neither trust nor need to render.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable, Mapping

from app.models import TechnicalSpecs
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


# Labels as they read on screen, squashed, plus the spellings the JSON keys use.
_LABEL_LOOKUP: dict[str, str] = {
    **{_squash(key): field for key, field in SPEC_FIELDS.items()},
    **{_squash(label): field for label, field in SPEC_LABELS.items()},
    "colour": "colors",
    "colours": "colors",
    "colorations": "colors",
}

# A pasted page carries the furniture around the rows too. These end the current row.
_TEXT_TERMINATORS = frozenset(
    _squash(line)
    for line in (
        "more to explore",
        "contribute to this page",
        "suggest an edit or add missing content",
        "back to top",
        "learn more about contributing",
        "photos",
        "storyline",
        "details",
        "box office",
        "related news",
        "user reviews",
        "more like this",
        "recently viewed",
    )
)

# " · " is how IMDb joins values on one line; a tab is how a copied table arrives.
_TEXT_VALUE_SPLIT = re.compile(r"\t+|\s+·\s+|\s{2,}")
_WORD = re.compile(r"\S+")
# "Printed film format" and "Cinematographic process" are the longest headings.
_MAX_LABEL_WORDS = 3
# Technical rows are terse. Anything this wordy is prose that followed the table.
_MAX_VALUE_WORDS = 12


def _field_for_label(text: str) -> str | None:
    """The TechnicalSpecs field a heading names, allowing a stray plural."""
    squashed = _squash(text)
    if not squashed:
        return None
    return _LABEL_LOOKUP.get(squashed) or _LABEL_LOOKUP.get(squashed.rstrip("s"))


def _split_heading(line: str) -> tuple[str, str, str] | None:
    """Split ``Aspect ratio  2.39 : 1`` into its field, its heading and the rest.

    The heading is matched a word at a time from the start of the line rather than
    by looking for a separator, because a copied row may put its values behind a
    colon, a tab, two spaces or nothing at all.
    """
    words = list(_WORD.finditer(line))
    for count in range(min(_MAX_LABEL_WORDS, len(words)), 0, -1):
        heading = line[words[0].start() : words[count - 1].end()]
        if (field := _field_for_label(heading)) is not None:
            return field, heading, line[words[count - 1].end() :]
    return None


def _from_plain_text(page: str) -> dict[str, list[str]]:
    """Read the rows out of the page as it reads on screen, not as it is marked up.

    This is the paste path: selecting the technical page and copying it gives a
    heading followed by its values, either on the same line or on the lines below.
    Values are only ever collected for a heading that has been recognised, so the
    furniture that comes with a whole-page paste has nowhere to land.
    """
    collected: dict[str, list[str]] = {}
    field: str | None = None

    for raw in page.splitlines():
        line = raw.strip()
        # Markup means one of the HTML extractors should have handled this, and tag
        # soup would otherwise be harvested as if it were values.
        if not line or "<" in line or ">" in line:
            field = None
            continue
        if _squash(line) in _TEXT_TERMINATORS:
            field = None
            continue

        heading = _split_heading(line)
        # IMDb's colour row reads "Color / Color": a bare heading that reopens the row
        # already being read is that row's value, and the only one it has.
        if heading is not None and heading[0] == field and not heading[2].strip():
            heading = None

        if heading is not None:
            field, label, tail = heading[0], heading[1], heading[2]
            values = _TEXT_VALUE_SPLIT.split(tail.strip()) if tail.strip() else []
        elif field is not None:
            if len(line.split()) > _MAX_VALUE_WORDS:
                field = None
                continue
            label, values = None, _TEXT_VALUE_SPLIT.split(line)
        else:
            continue

        if kept := _keep(values, label=label):
            collected.setdefault(field, []).extend(kept)

    return collected


def specs_from_rows(rows: Mapping[str, Iterable[str]]) -> TechnicalSpecs:
    """Build specs from rows that did not come from a page — a lookup's answer.

    Held to the same limits as a parse: known fields only, no duplicates, nothing
    longer than a row value plausibly is.
    """
    return TechnicalSpecs(
        **{
            field: kept
            for field, values in rows.items()
            if field in _SPEC_FIELD_NAMES
            and (kept := _keep([value for value in values if isinstance(value, str)]))
        }
    )


def parse_technical(page: str) -> TechnicalSpecs:
    """Parse an IMDb ``/technical`` page into :class:`TechnicalSpecs`.

    Accepts either the page source or the page's formatted text, since a user with
    the page open may reasonably produce either.

    Never raises: an unrecognised page yields empty specs, and the caller decides
    whether that is worth telling the user about (it is — the advice just falls back
    to source-only heuristics).
    """
    if not page or not page.strip():
        return TechnicalSpecs()

    merged: dict[str, list[str]] = {}
    extractors = (_from_next_data, _from_testid_markup, _from_legacy_table, _from_plain_text)
    for extractor in extractors:
        for field, values in extractor(page).items():
            merged.setdefault(field, []).extend(values)
        # Stop at the first layout that produced the format rows we actually need.
        if merged.get("negative_formats") or merged.get("aspect_ratios"):
            break

    return TechnicalSpecs(
        **{field: _keep(values) for field, values in merged.items() if field in _SPEC_FIELD_NAMES}
    )
