"""The command line: one file in, two files out.

``graindamage /media/Blade.Runner.1982.2160p.mkv`` is the whole interface. On a terminal
it walks you through it — :mod:`app.cli.wizard` asks which film, where the technical rows
should come from, and what kind of encode, each as a list you move a cursor through. With
``--yes``, no TTY, or ``--no-menu``, the same run takes its answers from the flags
instead, which is what a script wants. Either way it reads the title out of the filename,
runs ``ffprobe`` on the file, and writes a HandBrake preset and an FFmpeg script beside
the film.

The advice is the web app's advice: same rules engine, same optional Gemini review, same
validation, same wording for the caveats. This module only decides what goes in and lays
what comes out down.

Three habits keep it honest:

* **Refuse rather than guess.** A missing file, or a missing ``ffprobe``, exits 2 rather
  than falling back to 1080p defaults — settings for the wrong source are worse than no
  settings at all.
* **Nothing is overwritten** without ``--force``. On the flag path that is a refusal
  before any network call, so a re-run says so at once rather than after a Gemini round
  trip; in the menus it is the first question asked, for the same reason.
* **Prompts and progress go to stderr, the report to stdout**, so
  ``graindamage film.mkv > notes.txt`` still shows you the menu.

Exit codes: ``0`` written, ``1`` you stopped it, ``2`` setup or usage, ``130`` Ctrl-C.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from app import __version__
from app.advice import finish_advice
from app.cli.filename import NameGuess, guess_name
from app.cli.outputs import (
    OutputExists,
    Written,
    preset_text,
    refuse_existing,
    script_text,
    write_outputs,
)
from app.cli.probe import ProbeFailed, probe
from app.cli.prompts import Choice, Terminal
from app.cli.report import render_report
from app.cli.session import (
    GUESSED_FROM_YEAR,
    NO_ROWS_FOUND,
    NO_TMDB_KEY,
    NO_VIDEO_TRACK,
    Abort,
    Context,
    FoundSpecs,
    SetupProblem,
    best_hit,
    film_detail,
    is_imdb_id,
    make_context,
    specs_from_file,
    specs_from_paste,
    technical_url,
)
from app.cli.wizard import Answers, answers_from, choose_input
from app.cli.wizard import run as run_wizard
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
)
from app.providers import ProviderError
from app.providers.gemini import GeminiClient
from app.providers.tmdb import TmdbClient

EXIT_OK = 0
EXIT_ABORTED = 1
EXIT_SETUP = 2
EXIT_INTERRUPTED = 130

_MIN_YEAR, _MAX_YEAR = 1888, 2100

NO_FILE = "Which film? Give me the path to one video file: graindamage /media/film.mkv"

# Why a run ended with no paths. Both go in the report's tail, under "Not written".
NOT_ASKED = "You asked for the settings only — everything above is complete."
WRITE_FAILED = (
    "The settings above are complete — pass --outdir DIR to put the files somewhere writable."
)


class Extra(StrEnum):
    """The two rows on the flag path's film menu that are not films."""

    RETYPE = "retype"
    ABORT = "abort"


class Emit(StrEnum):
    """The one document ``--print`` puts on stdout, in place of the report."""

    PRESET = "preset"
    SCRIPT = "script"


# --- the argument surface ---------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graindamage",
        description="Grain-aware AV1 / x265 settings for one video file.",
        epilog=(
            "With a terminal and no --yes, every step is a menu: arrows move, enter "
            "chooses, q stops. Writes <name>.graindamage.json (a HandBrake preset "
            "holding both encoders) and <name>.graindamage.sh (the FFmpeg command) "
            "beside the file — or, with --no-write, prints the settings and both "
            "commands and leaves the directory alone."
        ),
    )
    parser.add_argument(
        "path", type=Path, nargs="?", help="the video file to advise on (asked for if omitted)"
    )
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
        "--no-write",
        dest="write",
        action="store_false",
        help="print the settings and both commands, and write no files",
    )
    run.add_argument(
        "--print",
        dest="emit",
        choices=[member.value for member in Emit],
        help="put one document on stdout instead of in a file, and write nothing",
    )
    run.add_argument(
        "--no-gemini",
        dest="gemini",
        action="store_false",
        help="no Gemini at all: no review and no technical look-up",
    )
    run.add_argument(
        "--no-menu",
        dest="menu",
        action="store_false",
        help="take the answers from these flags rather than asking step by step",
    )
    run.add_argument(
        "-y", "--yes", action="store_true", help="no questions: the best film match, no menus"
    )
    run.add_argument("--ffprobe", default="ffprobe", help="path to ffprobe (default: ffprobe)")
    run.add_argument(
        "-q", "--quiet", action="store_true", help="print only the paths that were written"
    )
    return parser


def _imdb_arg(value: str) -> str:
    text = value.strip()
    if not is_imdb_id(text):
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
    """Gather, advise, write, report. The gathering is the only part with two shapes."""
    _settle(args)
    terminal = context.terminal
    path = _input_path(args, terminal)
    stem = path.stem
    menus = _menus_wanted(args, terminal)

    if args.write and not (args.force or menus):
        try:
            refuse_existing((args.outdir or path.parent).expanduser(), stem)
        except OutputExists as exc:
            raise SetupProblem(str(exc)) from exc

    report = await _probe(path, args, context)
    guess = _guess(path, args)

    warnings = list(report.warnings)
    if not report.has_video:
        warnings.append(NO_VIDEO_TRACK)

    answers = answers_from(
        args, out_dir=(args.outdir or path.parent).expanduser(), found=_given(args)
    )
    answers.source_warnings = tuple(warnings)
    answers.review = bool(args.gemini and context.gemini.enabled)

    if menus:
        answers = await run_wizard(
            context, answers, path=path, report=report, guess=guess, gemini=args.gemini
        )
    else:
        if not args.quiet:
            terminal.write(f"File     {path.name}")
            terminal.write(f"Film     {guess.describe()}")
        await _gather(args, context, answers, guess=guess)

    request = _request(answers, path=path, report=report, guess=guess)
    if answers.review and not args.quiet:
        terminal.write("Asking Gemini to review the plan…")
    advice = await finish_advice(
        request,
        annotator=context.gemini if answers.review else None,
        warnings=answers.warnings,
    )
    if answers.encoder:
        _prefer(advice, answers.encoder)

    if args.emit is not None:
        _emit(Emit(args.emit), advice, request, answers, media_dir=path.parent)
        return EXIT_OK

    written, unwritten = _write(advice, request, answers, media_dir=path.parent)
    _report(
        advice,
        request,
        answers,
        written=written,
        unwritten=unwritten,
        quiet=args.quiet,
        terminal=terminal,
    )
    # The advice always arrives now. The exit code is about the files: asking for them and
    # getting none is a failure, however complete the report above it is.
    return EXIT_OK if written is not None or not answers.write else EXIT_SETUP


def _settle(args: argparse.Namespace) -> None:
    """``--print`` is one document on stdout, so it implies no report and no files."""
    if args.emit is not None:
        args.quiet = True
        args.write = False


def _menus_wanted(args: argparse.Namespace, terminal: Terminal) -> bool:
    """Whether to walk the steps. ``--yes`` and ``--quiet`` both mean *do not ask me*."""
    return bool(args.menu and terminal.interactive and not (args.yes or args.quiet))


def _request(
    answers: Answers, *, path: Path, report: SourceReport, guess: NameGuess
) -> EncodeRequest:
    return EncodeRequest(
        movie=answers.movie,
        specs=answers.found.specs,
        specs_source=answers.found.source,
        source=report.media,
        source_tool=report.tool,
        grain_override=answers.grain,
        bit_depth_override=answers.bit_depth,
        speed=answers.speed,
        size=answers.size,
        input_path=path.name,
        output_stem=path.stem,
        fallback_year=guess.year,
    )


# --- the file ---------------------------------------------------------------


def _input_path(args: argparse.Namespace, terminal: Terminal) -> Path:
    """The file to advise on: the argument, or a menu of what is lying about."""
    if args.path is not None:
        return _resolve_input(args.path)
    if not terminal.interactive:
        raise SetupProblem(NO_FILE)
    return _resolve_input(choose_input(terminal))


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


# --- gathering from the flags ----------------------------------------------


async def _gather(
    args: argparse.Namespace, context: Context, answers: Answers, *, guess: NameGuess
) -> None:
    """The flag path: one search, one details request, one look-up, no review screen."""
    movie, warning = await _find_movie(context, guess, assume_yes=args.yes)
    answers.movie = movie
    answers.film_warning = warning
    answers.imdb_id = args.imdb_id or (movie.imdb_id if movie else None)

    if args.specs is None:
        answers.found = await _find_specs(
            args, context, imdb_id=answers.imdb_id, movie=answers.movie
        )


def _given(args: argparse.Namespace) -> FoundSpecs:
    """``--specs FILE``, read before anything is asked. It wins over every look-up."""
    if args.specs is None:
        return FoundSpecs()
    try:
        return specs_from_file(args.specs)
    except SetupProblem as exc:
        raise SetupProblem(f"--specs {exc}") from exc


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
            return await _details(tmdb, best_hit(hits, guess.year))

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
    rows: list[Choice[MovieHit | Extra]] = [
        Choice(value=hit, label=hit.display_title, detail=film_detail(hit, year)) for hit in hits
    ]
    rows.append(Choice(value=Extra.RETYPE, label="Wrong film — let me type the title"))
    rows.append(Choice(value=Extra.ABORT, label="Stop, and write nothing"))
    return rows


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


async def _find_specs(
    args: argparse.Namespace, context: Context, *, imdb_id: str | None, movie: Movie | None
) -> FoundSpecs:
    """Ask Gemini, and offer a paste if it does not know. No id or no key: say why."""
    gemini = context.gemini
    if not (imdb_id and args.gemini and gemini.enabled):
        return FoundSpecs(warnings=(_why_no_lookup(args, imdb_id, gemini),))

    try:
        lookup = await gemini.technical_specs(
            imdb_id,
            title=movie.title if movie else None,
            year=movie.year if movie else None,
        )
    except ProviderError as exc:
        return FoundSpecs(warnings=(f"{exc.message} {GUESSED_FROM_YEAR}",))

    if not lookup.specs.is_empty:
        return FoundSpecs(
            lookup.specs,
            source=SpecsSource.GEMINI,
            caveat=lookup.caveat,
            label=lookup.summary,
        )

    if not (context.terminal.interactive and not args.yes):
        return FoundSpecs(warnings=(f"{NO_ROWS_FOUND} {GUESSED_FROM_YEAR}",))

    found = specs_from_paste(_offer_paste(context.terminal, imdb_id))
    if found.source is None and not found.warnings:
        return FoundSpecs(warnings=(f"{NO_ROWS_FOUND} {GUESSED_FROM_YEAR}",))
    return found


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


def _offer_paste(terminal: Terminal, imdb_id: str) -> str:
    """Print the technical page's address and open a box to paste it into."""
    terminal.write()
    terminal.write(NO_ROWS_FOUND)
    terminal.write(f"  {technical_url(imdb_id)}")
    terminal.write("Select the specifications there and paste them below.")
    return terminal.paste("Paste, then Ctrl-D. Ctrl-D on its own guesses grain from the year.")


# --- the answer ------------------------------------------------------------


def _prefer(advice: Advice, encoder: Encoder) -> None:
    """Put one encoder first: it becomes the live command and the first preset."""
    advice.plans.sort(key=lambda plan: plan.encoder is not encoder)


def _write(
    advice: Advice, request: EncodeRequest, answers: Answers, *, media_dir: Path
) -> tuple[Written | None, str | None]:
    """Write the two files, or say why not.

    Nothing raised here: a directory that will not take the files must not cost the
    settings, which by this point have been probed for, looked up and reasoned about.
    """
    if not answers.write:
        return None, NOT_ASKED
    try:
        written = write_outputs(
            advice,
            request,
            stem=request.output_stem,
            directory=answers.out_dir,
            media_dir=media_dir,
            movie=answers.movie,
            specs_caveat=answers.found.caveat,
            force=True,  # the refusal already happened, before anything was asked
        )
    except (OutputExists, OSError) as exc:
        return None, f"Could not write beside {answers.out_dir}: {exc}\n{WRITE_FAILED}"
    return written, None


def _emit(
    kind: Emit, advice: Advice, request: EncodeRequest, answers: Answers, *, media_dir: Path
) -> None:
    """One document on stdout, and nothing else on it: ``--print``.

    The script always names the film's directory here. One written beside the film finds
    it from its own location, but one that came down a pipe has no location to ask.
    """
    if kind is Emit.PRESET:
        sys.stdout.write(preset_text(advice, request))
        return
    sys.stdout.write(
        script_text(
            advice,
            request,
            movie=answers.movie,
            specs_caveat=answers.found.caveat,
            media_dir=str(media_dir),
        )
    )


def _report(
    advice: Advice,
    request: EncodeRequest,
    answers: Answers,
    *,
    written: Written | None,
    unwritten: str | None,
    quiet: bool,
    terminal: Terminal,
) -> None:
    if quiet:
        for path in written.paths if written else ():
            sys.stdout.write(f"{path}\n")
        if unwritten:
            # No report to carry it, so the reason goes where the other diagnostics go.
            terminal.write(f"graindamage: {unwritten}")
        return
    sys.stdout.write(
        render_report(
            advice,
            request,
            movie=answers.movie,
            specs_caveat=answers.found.caveat,
            written=written.paths if written else (),
            unwritten=unwritten,
        )
    )
