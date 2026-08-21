"""The command line: one file in, two files out.

``graindamage /movies/Blade.Runner.1982.2160p.mkv`` reads the title and year out of the
name, runs ``ffprobe`` on the file, offers the TMDB hits in a list you arrow through,
looks the IMDb technical rows up through Gemini, and writes a HandBrake preset and an
FFmpeg script beside the film. The advice is the web app's advice: same rules engine,
same optional Gemini review, same validation, same wording for the caveats. This module
only gathers the inputs and lays the answer out.

Three habits keep it honest:

* **Refuse rather than guess.** A missing file, or a missing ``ffprobe``, exits 2 rather
  than falling back to 1080p defaults — settings for the wrong source are worse than no
  settings at all.
* **Nothing is overwritten** without ``--force``, and that is checked before any network
  call, so a re-run says so at once instead of after a Gemini round trip.
* **Prompts and progress go to stderr, the report to stdout**, so
  ``graindamage film.mkv > notes.txt`` still shows you the menu.

Exit codes: ``0`` written, ``1`` you stopped it, ``2`` setup or usage, ``130`` Ctrl-C.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app import __version__
from app.advice import finish_advice
from app.cli.filename import NameGuess, guess_name
from app.cli.outputs import OutputExists, Written, refuse_existing, write_outputs
from app.cli.probe import ProbeFailed, Runner, probe
from app.cli.prompts import Choice, Terminal
from app.cli.report import render_report
from app.config import Settings, get_settings
from app.models import (
    Advice,
    Encoder,
    EncodeRequest,
    GrainLevel,
    Movie,
    MovieHit,
    SizePreference,
    SourceReport,
    SpecsSource,
    SpeedPreference,
    TechnicalSpecs,
)
from app.providers import ProviderError
from app.providers.gemini import GeminiClient
from app.providers.imdb import parse_technical
from app.providers.tmdb import TmdbClient

EXIT_OK = 0
EXIT_ABORTED = 1
EXIT_SETUP = 2
EXIT_INTERRUPTED = 130

# A pasted page is a megabyte or so; anything past this is not a technical page.
MAX_SPECS_CHARS = 4_000_000

_IMDB_ID = re.compile(r"tt\d{5,}")
_MIN_YEAR, _MAX_YEAR = 1888, 2100

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


class Extra(StrEnum):
    """The two rows on the film menu that are not films."""

    RETYPE = "retype"
    ABORT = "abort"


class Abort(RuntimeError):
    """The user chose to stop. Nothing is written."""


class SetupProblem(RuntimeError):
    """Something the run needs is missing, unreadable, or in the way."""


@dataclass(slots=True)
class Context:
    """Everything the run talks to, so a test can hand it doubles.

    The clients are built once and live for the process, which is what makes their TTL
    caches worth anything: a retyped search or a second look at the same film costs no
    upstream request.
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


# --- the argument surface ---------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graindamage",
        description="Grain-aware AV1 / x265 settings for one video file.",
        epilog=(
            "Writes <name>.graindamage.json (a HandBrake preset holding both encoders) "
            "and <name>.graindamage.sh (the FFmpeg command) beside the file."
        ),
    )
    parser.add_argument("path", type=Path, help="the video file to advise on")
    parser.add_argument("--version", action="version", version=f"graindamage {__version__}")

    film = parser.add_argument_group("the film")
    film.add_argument("--title", help="skip the filename guess and search for this")
    film.add_argument("--year", type=_year_arg, help="the release year, if the name lacks one")
    film.add_argument(
        "--imdb-id",
        type=_imdb_arg,
        metavar="ttNNNNNNN",
        help="use this title id for the technical rows instead of TMDB's",
    )
    film.add_argument(
        "--specs",
        type=Path,
        metavar="FILE",
        help="read IMDb's technical specifications from a file instead of asking Gemini",
    )

    encode = parser.add_argument_group("the encode")
    encode.add_argument(
        "--encoder",
        choices=[member.value for member in Encoder],
        help="which encoder goes first: the live command and the first preset",
    )
    encode.add_argument(
        "--speed",
        choices=[member.value for member in SpeedPreference],
        default=SpeedPreference.BALANCED.value,
        help="how much CPU time you will spend (default: balanced)",
    )
    encode.add_argument(
        "--size",
        choices=[member.value for member in SizePreference],
        default=SizePreference.BALANCED.value,
        help="where to sit on the size/fidelity curve (default: balanced)",
    )
    encode.add_argument(
        "--grain",
        choices=[member.value for member in GrainLevel],
        help="override the inferred grain level",
    )
    encode.add_argument(
        "--bit-depth", type=int, choices=[8, 10, 12], help="force the encoding bit depth"
    )

    run = parser.add_argument_group("this run")
    run.add_argument("--outdir", type=Path, help="write the two files here instead")
    run.add_argument("--force", action="store_true", help="replace files that are already there")
    run.add_argument(
        "--no-gemini",
        dest="gemini",
        action="store_false",
        help="no Gemini at all: no review and no technical look-up",
    )
    run.add_argument(
        "-y", "--yes", action="store_true", help="take the best film match without asking"
    )
    run.add_argument("--ffprobe", default="ffprobe", help="path to ffprobe (default: ffprobe)")
    run.add_argument(
        "-q", "--quiet", action="store_true", help="print only the paths that were written"
    )
    return parser


def _imdb_arg(value: str) -> str:
    text = value.strip()
    if not _IMDB_ID.fullmatch(text):
        raise argparse.ArgumentTypeError("an IMDb title id looks like tt0083658")
    return text


def _year_arg(value: str) -> int:
    text = value.strip()
    if not text.isdigit() or not _MIN_YEAR <= int(text) <= _MAX_YEAR:
        raise argparse.ArgumentTypeError(f"a release year between {_MIN_YEAR} and {_MAX_YEAR}")
    return int(text)


# --- the run ----------------------------------------------------------------


def main(argv: Sequence[str] | None = None, *, context: Context | None = None) -> int:
    """Parse the arguments, do the work, and turn what happened into an exit code."""
    args = build_parser().parse_args(argv)
    ctx = context or make_context()
    try:
        return asyncio.run(_run(args, ctx))
    except Abort:
        ctx.terminal.write("Stopped. Nothing was written.")
        return EXIT_ABORTED
    except KeyboardInterrupt:  # pragma: no cover - needs a real signal
        ctx.terminal.write("Interrupted. Nothing was written.")
        return EXIT_INTERRUPTED
    except SetupProblem as exc:
        ctx.terminal.write(f"graindamage: {exc}")
        return EXIT_SETUP


async def _run(args: argparse.Namespace, context: Context) -> int:
    terminal = context.terminal
    path = _resolve_input(args.path)
    out_dir = (args.outdir or path.parent).expanduser()
    stem = path.stem

    if not args.force:
        try:
            refuse_existing(out_dir, stem)
        except OutputExists as exc:
            raise SetupProblem(str(exc)) from exc

    report = await _probe(path, args, context)
    warnings = list(report.warnings)
    if not report.has_video:
        warnings.append(NO_VIDEO_TRACK)

    guess = _guess(path, args)
    if not args.quiet:
        terminal.write(f"File     {path.name}")
        terminal.write(f"Film     {guess.describe()}")

    movie, movie_warning = await _find_movie(context, guess, assume_yes=args.yes)
    if movie_warning:
        warnings.append(movie_warning)

    imdb_id = args.imdb_id or (movie.imdb_id if movie else None)
    found = await _find_specs(context, args, imdb_id=imdb_id, movie=movie)
    warnings.extend(found.warnings)

    request = EncodeRequest(
        movie=movie,
        specs=found.specs,
        specs_source=found.source,
        source=report.media,
        source_tool=report.tool,
        grain_override=GrainLevel(args.grain) if args.grain else None,
        bit_depth_override=args.bit_depth,
        speed=SpeedPreference(args.speed),
        size=SizePreference(args.size),
        input_path=path.name,
        output_stem=stem,
        fallback_year=guess.year,
    )

    advice = await finish_advice(
        request,
        annotator=context.gemini if args.gemini and context.gemini.enabled else None,
        warnings=warnings,
    )
    if args.encoder:
        _prefer(advice, Encoder(args.encoder))

    written = _write(
        advice, request, stem=stem, out_dir=out_dir, media_dir=path.parent, found=found, movie=movie
    )
    _report(advice, request, movie=movie, found=found, written=written, quiet=args.quiet)
    return EXIT_OK


# --- the file ---------------------------------------------------------------


def _resolve_input(raw: Path) -> Path:
    path = raw.expanduser()
    if path.is_dir():
        raise SetupProblem(f"{path} is a directory — point me at one file.")
    if not path.exists():
        raise SetupProblem(f"{path} is not there.")
    return path


async def _probe(path: Path, args: argparse.Namespace, context: Context) -> SourceReport:
    """The file's own numbers, or nothing: this is the whole reason for the run."""
    try:
        return await probe(path, executable=args.ffprobe, runner=context.runner)
    except ProbeFailed as exc:
        raise SetupProblem(str(exc)) from exc


def _guess(path: Path, args: argparse.Namespace) -> NameGuess:
    guessed = guess_name(path)
    if args.title or args.year:
        return NameGuess(
            title=args.title or guessed.title,
            year=args.year or guessed.year,
            source="command line" if args.title else guessed.source,
        )
    return guessed


# --- the film ---------------------------------------------------------------


async def _find_movie(
    context: Context, guess: NameGuess, *, assume_yes: bool
) -> tuple[Movie | None, str | None]:
    """Search, offer the hits, and follow the pick up with a details request.

    A film-less run is a working run — the rules engine only needs the file — so every
    failure here degrades to ``(None, warning)``. Only the user saying *stop* aborts.
    """
    tmdb = context.tmdb
    if not tmdb.enabled:
        return None, NO_TMDB_KEY

    query = guess.query or _ask_title(context.terminal)
    while True:
        try:
            hits = await tmdb.search(query)
        except ProviderError as exc:
            return None, f"{exc.message} Continuing without the film's details."

        if assume_yes:
            if not hits:
                return None, f'Nothing on TMDB for "{query}". Continuing without the film.'
            return await _details(tmdb, _best(hits, guess.year))

        if not hits:
            context.terminal.write(f'Nothing on TMDB for "{query}".')
            query = _ask_title(context.terminal)
            continue

        chosen = context.terminal.select("Which film is this?", _film_choices(hits, guess.year))
        if chosen is None or chosen.value is Extra.ABORT:
            raise Abort
        if chosen.value is Extra.RETYPE:
            query = _ask_title(context.terminal)
            continue
        return await _details(tmdb, chosen.value)


def _film_choices(hits: Sequence[MovieHit], year: int | None) -> list[Choice[MovieHit | Extra]]:
    rows: list[Choice[MovieHit | Extra]] = []
    for hit in hits:
        detail = str(hit.year) if hit.year else "year unknown"
        if year and hit.year == year:
            detail = f"{detail}  · matches the filename"
        rows.append(Choice(value=hit, label=hit.display_title, detail=detail))
    rows.append(Choice(value=Extra.RETYPE, label="Wrong film — let me type the title"))
    rows.append(Choice(value=Extra.ABORT, label="Stop, and write nothing"))
    return rows


def _best(hits: Sequence[MovieHit], year: int | None) -> MovieHit:
    """The hit the filename's year agrees with, or TMDB's own first answer."""
    if year is not None:
        matching = next((hit for hit in hits if hit.year == year), None)
        if matching is not None:
            return matching
    return hits[0]


async def _details(tmdb: TmdbClient, hit: MovieHit) -> tuple[Movie | None, str | None]:
    try:
        return await tmdb.get_movie(hit.tmdb_id), None
    except ProviderError as exc:
        return None, f"{exc.message} Continuing without {hit.title}'s details."


def _ask_title(terminal: Terminal) -> str:
    answer = terminal.ask("Title to search for (empty to stop):")
    if not answer:
        raise Abort
    return answer


# --- the technical rows ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoundSpecs:
    """The technical rows, where they came from, and what to say about it."""

    specs: TechnicalSpecs
    source: SpecsSource | None = None
    caveat: str | None = None
    warnings: tuple[str, ...] = ()


async def _find_specs(
    context: Context, args: argparse.Namespace, *, imdb_id: str | None, movie: Movie | None
) -> FoundSpecs:
    """A paste always wins; otherwise ask Gemini, and offer a paste if it does not know."""
    if args.specs is not None:
        return _specs_from_file(args.specs)

    gemini = context.gemini
    if not (imdb_id and args.gemini and gemini.enabled):
        return FoundSpecs(TechnicalSpecs(), warnings=(_why_no_lookup(args, imdb_id, gemini),))

    try:
        lookup = await gemini.technical_specs(
            imdb_id,
            title=movie.title if movie else None,
            year=movie.year if movie else None,
        )
    except ProviderError as exc:
        return FoundSpecs(TechnicalSpecs(), warnings=(f"{exc.message} {GUESSED_FROM_YEAR}",))

    if not lookup.specs.is_empty:
        return FoundSpecs(lookup.specs, source=SpecsSource.GEMINI, caveat=lookup.caveat)

    pasted = _offer_paste(context.terminal, imdb_id) if _can_prompt(context.terminal, args) else ""
    if not pasted.strip():
        return FoundSpecs(TechnicalSpecs(), warnings=(f"{NO_ROWS_FOUND} {GUESSED_FROM_YEAR}",))

    specs = parse_technical(pasted[:MAX_SPECS_CHARS])
    if specs.is_empty:
        return FoundSpecs(
            specs,
            warnings=(f"Nothing recognisable in what you pasted. {GUESSED_FROM_YEAR}",),
        )
    return FoundSpecs(specs, source=SpecsSource.PASTED, caveat="pasted from the technical page")


def _specs_from_file(raw: Path) -> FoundSpecs:
    path = raw.expanduser()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SetupProblem(f"--specs {path}: {exc.strerror or exc}.") from exc

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


def _why_no_lookup(args: argparse.Namespace, imdb_id: str | None, gemini: GeminiClient) -> str:
    if not args.gemini:
        return f"--no-gemini, so the technical rows were not looked up. {GUESSED_FROM_YEAR}"
    if not gemini.enabled:
        return (
            "No Gemini key (GEMINI_API_KEY), so the technical rows were not looked up. "
            f"Pass --specs FILE with the technical page in it. {GUESSED_FROM_YEAR}"
        )
    if not imdb_id:
        return (
            "No IMDb title id for this film, so its technical rows could not be looked "
            f"up. Pass --imdb-id ttNNNNNNN. {GUESSED_FROM_YEAR}"
        )
    return GUESSED_FROM_YEAR  # pragma: no cover - the three above are the only ways here


def _can_prompt(terminal: Terminal, args: argparse.Namespace) -> bool:
    return terminal.interactive and not args.yes


def _offer_paste(terminal: Terminal, imdb_id: str) -> str:
    """Print the technical page's address and open a box to paste it into."""
    terminal.write()
    terminal.write(NO_ROWS_FOUND)
    terminal.write(f"  https://www.imdb.com/title/{imdb_id}/technical/")
    terminal.write("Select the specifications there and paste them below.")
    return terminal.paste("Paste, then Ctrl-D. Ctrl-D on its own guesses grain from the year.")


# --- the answer ------------------------------------------------------------


def _prefer(advice: Advice, encoder: Encoder) -> None:
    """Put one encoder first: it becomes the live command and the first preset."""
    advice.plans.sort(key=lambda plan: plan.encoder is not encoder)


def _write(
    advice: Advice,
    request: EncodeRequest,
    *,
    stem: str,
    out_dir: Path,
    media_dir: Path,
    found: FoundSpecs,
    movie: Movie | None,
) -> Written:
    try:
        return write_outputs(
            advice,
            request,
            stem=stem,
            directory=out_dir,
            media_dir=media_dir,
            movie=movie,
            specs_caveat=found.caveat,
            force=True,  # the refusal already happened, before anything was asked
        )
    except (OutputExists, OSError) as exc:
        raise SetupProblem(f"Could not write beside {media_dir}: {exc}") from exc


def _report(
    advice: Advice,
    request: EncodeRequest,
    *,
    movie: Movie | None,
    found: FoundSpecs,
    written: Written,
    quiet: bool,
) -> None:
    if quiet:
        for path in written.paths:
            sys.stdout.write(f"{path}\n")
        return
    sys.stdout.write(
        render_report(
            advice, request, movie=movie, specs_caveat=found.caveat, written=written.paths
        )
    )
