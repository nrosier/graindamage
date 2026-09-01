"""Reading a film's title and year out of the name of a file.

Library filenames are a folk format: ``Blade.Runner.1982.2160p.UHD.BluRay.DV.TrueHD
.7.1-FraMeSToR.mkv``. The title is everything before the year, and everything after it
describes the *release*, not the film. That single observation does most of the work
here; the rest is the awkward cases.

Two of those are worth naming, because they are why the year is not simply "the first
four digits that look like a year":

* ``Blade.Runner.2049.2017.2160p`` — two candidates, and the release year is the second.
* ``2001.A.Space.Odyssey.1968`` — two candidates, and the first one is the title.

So the year is the *last* candidate that is not the whole beginning of the name, and it
has to be a year that has actually happened. That last rule is what keeps
``Blade.Runner.2049.2160p.BluRay`` — a real and common shape — from being read as a film
from 2049 called "Blade Runner": 2049 is rejected, no year is found, and the title comes
back as "Blade Runner 2049", which is the correct answer.

Nothing here is certain, which is why both front-ends always show what they guessed and
offer to be told otherwise. :func:`best_hit` is the other half of that: given the search
results, it picks the one the filename's year agrees with.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.models import MovieHit

# Hyphens split too: it is how a release group hangs off the end (``x265-RARBG``), and
# a title that loses one to a space (``Spider-Man`` → ``Spider Man``) still searches.
_SEPARATORS = re.compile(r"[\s._+\-()\[\]{}]+")

# Accented characters survive tokenising, so "Amélie" is searched for as itself.
_YEARISH = re.compile(r"^(?:18|19|20)\d{2}$")

# The oldest surviving films are from 1888; anything later than next year is not a
# release date, it is part of a title.
_MIN_YEAR = 1888

# A stem that says nothing about the film. Ripping tools produce all of these, and the
# containing directory is then the only place a title can come from.
# The plurals are in there because a directory called ``Movies`` is a library root,
# never a title, and a file called ``movie.mkv`` inside it has to look somewhere else.
_CONTENTLESS_WORDS = frozenset(
    {
        "movie", "movies", "film", "films", "video", "videos", "main", "index",
        "output", "untitled", "default", "feature", "media",
    }
)  # fmt: skip
_CONTENTLESS_SHAPE = re.compile(
    r"^(?:title|track|part|disc|disk|dvd|cd|vts|video_ts|bd|chapter)[\s._-]*\d*$|^\d{1,2}$",
    re.IGNORECASE,
)

# Tags that describe the file rather than the film. Only ever stripped from the *tail*,
# and only when no year was found — generic English words are deliberately absent, so
# "The Final Cut" and "Special Edition" survive in a title that contains them.
_RELEASE_TAGS = frozenset(
    {
        # resolution and scan
        "480p", "576p", "720p", "1080p", "1080i", "2160p", "4320p", "4k", "8k",
        "uhd", "hd", "sd", "fhd", "qhd", "hdready",
        # source
        "bluray", "blu", "ray", "bdrip", "brrip", "bdremux", "remux", "web", "webrip",
        "webdl", "dl", "hdtv", "pdtv", "dvdrip", "dvd5", "dvd9", "hdrip", "hdcam", "cam",
        "telesync", "telecine", "screener", "scr", "r5", "vhsrip", "uhdbd", "bdmv", "iso",
        # streaming services, which appear as the source
        "amzn", "nf", "dsnp", "hmax", "atvp", "hulu", "itunes", "pcok", "stan", "crav",
        # codec and depth
        "x264", "x265", "h264", "h265", "hevc", "avc", "av1", "xvid", "divx", "vp9",
        "mpeg2", "mpeg4", "10bit", "8bit", "12bit", "hi10p", "hi10",
        # dynamic range
        "hdr", "hdr10", "hdr10plus", "dv", "dovi", "dolbyvision", "sdr", "hlg", "pq",
        # audio
        "dts", "dtshd", "dtsx", "truehd", "atmos", "ac3", "eac3", "dd", "ddp", "aac",
        "flac", "mp3", "opus", "lpcm", "pcm", "ma", "commentary", "dual", "multi",
        "subbed", "dubbed", "sub", "subs",
        # release furniture
        "proper", "repack", "internal", "limited", "extended", "unrated", "uncut",
        "remastered", "restored", "imax", "criterion", "hybrid", "retail", "rerip",
        "readnfo", "nfo", "custom", "rip",
    }
)  # fmt: skip

_SINGLE_LETTER = re.compile(r"^[A-Za-z]$")


@dataclass(frozen=True, slots=True)
class NameGuess:
    """What a filename appears to say, and which name it was read from."""

    title: str
    year: int | None
    source: str = "filename"

    @property
    def query(self) -> str:
        """What to search TMDB for. The year is a filter, not part of the title."""
        return self.title

    def describe(self) -> str:
        named = f"{self.title} ({self.year})" if self.year else self.title
        return f"{named} — read from the {self.source}"


def guess_name(path: Path, *, max_year: int | None = None) -> NameGuess:
    """Guess a film's title and year from ``path``.

    Falls back to the parent directory's name when the file's own is contentless, which
    is what ``Blade Runner (1982)/movie.mkv`` needs.
    """
    ceiling = max_year if max_year is not None else datetime.now(UTC).year + 1

    stem, source = _best_name(path)
    tokens = _tokens(stem)
    if not tokens:
        return NameGuess(title="", year=None, source=source)

    index = _year_index(tokens, ceiling)
    if index is not None:
        return NameGuess(title=" ".join(tokens[:index]), year=int(tokens[index]), source=source)

    return NameGuess(title=" ".join(_without_release_tail(tokens, stem)), year=None, source=source)


# --- matching the guess against what was searched for -----------------------


def best_hit(hits: Sequence[MovieHit], year: int | None) -> MovieHit:
    """The hit the filename's year agrees with, or TMDB's own first answer."""
    if year is not None:
        matching = next((hit for hit in hits if hit.year == year), None)
        if matching is not None:
            return matching
    return hits[0]


def filename_first(hits: Sequence[MovieHit], year: int | None) -> list[MovieHit]:
    """``hits`` with :func:`best_hit` moved to the front, TMDB's order otherwise kept.

    The terminal puts its cursor on that row; a page has no cursor, so it puts the row
    first instead. Same rule, so the two front-ends agree about which film a filename
    is claiming.
    """
    if not hits:
        return []
    best = best_hit(hits, year)
    return [best, *(hit for hit in hits if hit is not best)]


# --- internals --------------------------------------------------------------


def _best_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if not _is_contentless(stem):
        return stem, "filename"
    parent = path.parent.name
    if parent and not _is_contentless(parent):
        return parent, "directory"
    return stem, "filename"


def _is_contentless(name: str) -> bool:
    folded = name.strip().casefold()
    return not folded or folded in _CONTENTLESS_WORDS or bool(_CONTENTLESS_SHAPE.match(folded))


def _tokens(name: str) -> list[str]:
    return [token for token in _SEPARATORS.split(name) if token]


def _year_index(tokens: list[str], ceiling: int) -> int | None:
    """The index of the release year, or ``None``.

    The last candidate wins, and index 0 never does: a name that *starts* with a year
    is a film called after one.
    """
    candidates = [
        index
        for index, token in enumerate(tokens)
        if index > 0 and _YEARISH.match(token) and _MIN_YEAR <= int(token) <= ceiling
    ]
    return candidates[-1] if candidates else None


def _without_release_tail(tokens: list[str], stem: str) -> list[str]:
    """Strip release furniture off the end, leaving at least one token behind."""
    kept = list(tokens)

    # A release group is the last thing in the name and a hyphen puts it there. Only
    # trust that when the name has already proved to be a release name, or "Spider-Man"
    # would lose half of itself.
    if len(kept) > 1 and _looks_like_a_release(kept) and re.search(r"-[^\s._+\-()\[\]{}]+$", stem):
        kept.pop()

    while len(kept) > 1 and _is_tail_furniture(kept):
        kept.pop()
    return kept


def _looks_like_a_release(tokens: list[str]) -> bool:
    return any(token.casefold() in _RELEASE_TAGS for token in tokens)


def _is_tail_furniture(kept: list[str]) -> bool:
    """Is the last token about the release rather than about the film?

    A named tag always is. A stray digit or letter only is when what comes before it is
    furniture too — which is the difference between the ``7 1`` of ``TrueHD 7.1`` and the
    ``13`` of ``Apollo 13``, or between the ``264`` of a split ``H.264`` and the ``2049``
    of a title whose year was rejected as being in the future.
    """
    if kept[-1].casefold() in _RELEASE_TAGS:
        return True
    return _is_stray(kept[-1]) and len(kept) > 1 and _is_furniture(kept[-2])


def _is_stray(token: str) -> bool:
    """A bare number or a single letter: meaningless without its neighbours."""
    return token.isdigit() or bool(_SINGLE_LETTER.match(token))


def _is_furniture(token: str) -> bool:
    return token.casefold() in _RELEASE_TAGS or _is_stray(token)
