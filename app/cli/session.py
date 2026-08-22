"""What one run shares, whichever way it was asked for.

A run gathers its inputs two ways — from flags, in :mod:`app.cli.app`, or from the menus
in :mod:`app.cli.wizard` — and both need the same handful of things: a :class:`Context`
holding the clients to ask, the technical rows with a note about where they came from,
and one exception each for *the user stopped* and *this run cannot start*. They live here
so that neither of those two modules has to import the other.

Nothing here prompts, prints, or writes a file. The wording of the warnings is here
because both paths must say the same thing about a missing key: what is degraded, and
what to do about it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.cli.probe import Runner
from app.cli.prompts import Terminal
from app.config import Settings, get_settings
from app.models import MovieHit, SpecsSource, TechnicalSpecs
from app.providers.gemini import GeminiClient
from app.providers.imdb import parse_technical
from app.providers.tmdb import TmdbClient

# A pasted page is a megabyte or so; anything past this is not a technical page.
MAX_SPECS_CHARS = 4_000_000

IMDB_ID = re.compile(r"tt\d{5,}")

GUESSED_FROM_YEAR = (
    "Grain was guessed from the release year rather than read from a negative format, "
    "so treat it as a guess."
)
NO_ROWS_FOUND = "Gemini did not have the technical rows for this film."
NO_TMDB_KEY = (
    "No TMDB key (TMDB_API_KEY), so the film was not looked up. Settings below come "
    "from the file alone; pass --imdb-id to get the technical rows anyway."
)
NO_VIDEO_TRACK = "ffprobe found no video track, so the resolution and depth are unknown."


class Abort(RuntimeError):
    """The user chose to stop. Nothing is written."""


class SetupProblem(RuntimeError):
    """Something the run needs is missing, unreadable, or in the way."""


@dataclass(slots=True)
class Context:
    """Everything the run talks to, so a test can hand it doubles.

    The clients are built once and live for the process, which is what makes their TTL
    caches worth anything: a retyped search, a second look at the same film, or a step
    revisited from the wizard's review screen costs no upstream request.
    """

    settings: Settings
    terminal: Terminal
    tmdb: TmdbClient
    gemini: GeminiClient
    runner: Runner | None = None


def make_context(
    settings: Settings | None = None,
    *,
    terminal: Terminal | None = None,
    runner: Runner | None = None,
) -> Context:
    resolved = settings or get_settings()
    return Context(
        settings=resolved,
        terminal=terminal or Terminal(),
        tmdb=TmdbClient(resolved),
        gemini=GeminiClient(resolved),
        runner=runner,
    )


# --- the film ---------------------------------------------------------------


def is_imdb_id(text: str) -> bool:
    """Whether ``text`` is a title id and nothing else — what ``--imdb-id`` accepts."""
    return bool(IMDB_ID.fullmatch(text.strip()))


def imdb_id_in(text: str) -> str | None:
    """The first title id anywhere in ``text``, so that pasting the address works too.

    Someone asked for an IMDb id is much more likely to have the page open than the id
    written down, and ``https://www.imdb.com/title/tt0083658/`` contains the answer.
    """
    found = IMDB_ID.search(text)
    return found.group(0) if found else None


def technical_url(imdb_id: str) -> str:
    """The page the rows are on, which is the one thing that is always worth printing."""
    return f"https://www.imdb.com/title/{imdb_id}/technical/"


def film_detail(hit: MovieHit, year: int | None) -> str:
    """The right-hand half of a film's row: its year, and whether the file agrees."""
    detail = str(hit.year) if hit.year else "year unknown"
    return f"{detail}  · matches the filename" if year and hit.year == year else detail


def best_hit(hits: list[MovieHit], year: int | None) -> MovieHit:
    """The hit the filename's year agrees with, or TMDB's own first answer."""
    if year is not None:
        matching = next((hit for hit in hits if hit.year == year), None)
        if matching is not None:
            return matching
    return hits[0]


# --- the technical rows ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoundSpecs:
    """The technical rows, where they came from, and what to say about it.

    The default is the honest empty answer: no rows, no source, nothing to caveat. Grain
    then falls back to the release year, and whoever built it says so in a warning.
    """

    specs: TechnicalSpecs = field(default_factory=TechnicalSpecs)
    source: SpecsSource | None = None
    caveat: str | None = None
    # A menu row has about thirty characters of detail before the terminal cuts it off,
    # and a look-up's caveat is three sentences. When the two cannot be the same words,
    # this is the short one; a paste says everything it needs to in its caveat already.
    label: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def describe(self) -> str:
        """One short phrase naming where these rows came from."""
        if self.source is None:
            return "none — grain will be guessed from the release year"
        return self.label or self.caveat or self.source.value


def specs_from_paste(text: str) -> FoundSpecs:
    """Parse what someone pasted into the box. An empty paste is an answer, not a fault."""
    if not text.strip():
        return FoundSpecs()
    specs = parse_technical(text[:MAX_SPECS_CHARS])
    if specs.is_empty:
        return FoundSpecs(
            specs, warnings=(f"Nothing recognisable in what you pasted. {GUESSED_FROM_YEAR}",)
        )
    return FoundSpecs(specs, source=SpecsSource.PASTED, caveat="pasted from the technical page")


def specs_from_file(raw: Path) -> FoundSpecs:
    """Read the technical page out of a file. Unreadable is a :class:`SetupProblem`."""
    path = raw.expanduser()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SetupProblem(f"{path}: {exc.strerror or exc}.") from exc

    specs = parse_technical(text[:MAX_SPECS_CHARS])
    if specs.is_empty:
        return FoundSpecs(
            specs,
            warnings=(
                f"Nothing recognisable in {path.name} — no negative format, no aspect "
                f"ratio. {GUESSED_FROM_YEAR}",
            ),
        )
    return FoundSpecs(specs, source=SpecsSource.PASTED, caveat=f"pasted from {path.name}")
