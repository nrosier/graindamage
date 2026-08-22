"""The menu-driven front-end: one command, one file, four steps.

``graindamage /media/film.mkv`` on a terminal walks the whole job in order — which film,
where the technical rows come from, what kind of encode, and then write — with every
question a list you move a cursor through. Nothing else has to be remembered: the flags
in :mod:`app.cli.app` all have a row here, and the file's own numbers are already in hand
because ``ffprobe`` ran before the first question.

It only decides what goes into the :class:`~app.models.EncodeRequest`. The answer is
built, reviewed, validated and written by exactly the same code as a flag-driven run, so
the two front-ends cannot come to different conclusions about the same film.

Four habits:

* **The first row is always the way forward.** Enter four times is a complete run, so
  trusting the defaults is never punished, and the cursor starts on the answer the file
  itself suggests — the hit whose year matches the filename, the setting in force now.
* **Every step can be gone back to.** The last screen returns to any of the three before
  it, and an answer already given is kept rather than asked for again. Changing the film
  drops rows that were looked up *for the old one*, because those are now about the wrong
  film; a set you pasted yourself is kept, and named, so you can see and replace it.
* **Every menu can stop the run**, and stopping writes nothing at all.
* **A missing key removes rows, never the run.** With no TMDB key there is no search but
  still an IMDb id to type; with no Gemini key there is no look-up but still the page to
  paste. The last row of every list is the honest one: carry on without it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from app import __version__
from app.advice import infer_grain
from app.cli.filename import NameGuess
from app.cli.outputs import PRESET_SUFFIX, SCRIPT_SUFFIX, existing_outputs
from app.cli.prompts import Choice, Terminal
from app.cli.report import source_summary
from app.cli.session import (
    GUESSED_FROM_YEAR,
    NO_ROWS_FOUND,
    Abort,
    Context,
    FoundSpecs,
    SetupProblem,
    film_detail,
    imdb_id_in,
    specs_from_file,
    specs_from_paste,
    technical_url,
)
from app.models import (
    Encoder,
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

STEPS = 4

# Wide enough for "Encoder order" and for "let it be inferred", so that every menu
# reads as two columns rather than a ragged list.
_LABEL = 14
_VALUE = 18

NO_FILM_CHOSEN = (
    "You chose to carry on without identifying the film, so the settings come from the file alone."
)
NO_ROWS_CHOSEN = f"You skipped the technical rows. {GUESSED_FROM_YEAR}"
NOTHING_PASTED = f"No technical rows for this film. {GUESSED_FROM_YEAR}"
NO_SEARCH = "No TMDB key (TMDB_API_KEY), so there is nothing to search — type an IMDb id instead."


class Film(StrEnum):
    """The rows on the film menu that are not films."""

    RETYPE = "retype"
    IMDB = "imdb"
    NONE = "none"
    STOP = "stop"


class Rows(StrEnum):
    """The ways to get a film's technical specifications."""

    KEEP = "keep"
    ASK = "ask"
    PASTE = "paste"
    FILE = "file"
    IMDB = "imdb"
    SKIP = "skip"
    STOP = "stop"


class Encode(StrEnum):
    """The encode hub: one row per setting, plus the way out of it."""

    DONE = "done"
    SPEED = "speed"
    SIZE = "size"
    GRAIN = "grain"
    DEPTH = "depth"
    ORDER = "order"
    REVIEW = "review"
    STOP = "stop"


class Ready(StrEnum):
    """The last screen: write, show, or go back to any step that led here."""

    WRITE = "write"
    SHOW = "show"
    FILM = "film"
    ROWS = "rows"
    ENCODE = "encode"
    STOP = "stop"


class Clash(StrEnum):
    """What to do about files that are already there."""

    REPLACE = "replace"
    ELSEWHERE = "elsewhere"
    STOP = "stop"


_SPEED_DETAIL: dict[SpeedPreference, str] = {
    SpeedPreference.QUALITY: "SVT-AV1 preset 3, x265 slower — the most CPU time",
    SpeedPreference.BALANCED: "SVT-AV1 preset 4, x265 slow",
    SpeedPreference.FAST: "SVT-AV1 preset 6, x265 medium — quicker, at some fidelity",
}
_SIZE_DETAIL: dict[SizePreference, str] = {
    SizePreference.ARCHIVAL: "CRF −2: bigger files, closer to the source",
    SizePreference.BALANCED: "the CRF the resolution anchors at",
    SizePreference.COMPACT: "CRF +2: smaller files, visibly so on grain",
}
_GRAIN_DETAIL: dict[GrainLevel, str] = {
    GrainLevel.NONE: "clean — no grain parameters at all",
    GrainLevel.LIGHT: "film-grain 4, real grain coded",
    GrainLevel.MODERATE: "film-grain 8, x265 tune grain",
    GrainLevel.HEAVY: "film-grain 12, denoised and re-synthesised",
    GrainLevel.EXTREME: "film-grain 20, denoised and re-synthesised",
}

# The rows worth showing back: the four the grain heuristic actually reads.
_SPEC_ROWS: tuple[tuple[str, str], ...] = (
    ("negative_formats", "Negative format"),
    ("cinematographic_processes", "Process"),
    ("printed_formats", "Printed format"),
    ("aspect_ratios", "Aspect ratio"),
)


@dataclass(slots=True)
class Answers:
    """Everything the wizard decides, and the only thing it hands back.

    The warnings are kept in the field that owns them rather than in one list, so that
    going back to a step and answering differently replaces its warning instead of
    appending a second one.
    """

    out_dir: Path
    force: bool = False
    write: bool = True
    source_warnings: tuple[str, ...] = ()
    movie: Movie | None = None
    imdb_id: str | None = None
    film_warning: str | None = None
    found: FoundSpecs = field(default_factory=FoundSpecs)
    speed: SpeedPreference = SpeedPreference.BALANCED
    size: SizePreference = SizePreference.BALANCED
    encoder: Encoder | None = None
    grain: GrainLevel | None = None
    bit_depth: int | None = None
    review: bool = False

    @property
    def warnings(self) -> list[str]:
        """Everything the report should say about what this run had to do without."""
        film = [self.film_warning] if self.film_warning else []
        return [*self.source_warnings, *film, *self.found.warnings]


def answers_from(args: argparse.Namespace, *, out_dir: Path, found: FoundSpecs) -> Answers:
    """The flags as the wizard's starting position: every one of them is a row here."""
    return Answers(
        out_dir=out_dir,
        force=bool(args.force),
        write=bool(args.write),
        movie=None,
        imdb_id=args.imdb_id,
        found=found,
        speed=SpeedPreference(args.speed),
        size=SizePreference(args.size),
        encoder=Encoder(args.encoder) if args.encoder else None,
        grain=GrainLevel(args.grain) if args.grain else None,
        bit_depth=args.bit_depth,
    )


async def run(
    context: Context,
    answers: Answers,
    *,
    path: Path,
    report: SourceReport,
    guess: NameGuess,
    gemini: bool = True,
) -> Answers:
    """Walk the four steps and return what they decided. Raises :class:`Abort` to stop."""
    wizard = Wizard(context, answers, path=path, report=report, guess=guess, gemini=gemini)
    return await wizard.run()


class Wizard:
    """One run's worth of questions. Holds the answers so far and the terminal to ask on."""

    def __init__(
        self,
        context: Context,
        answers: Answers,
        *,
        path: Path,
        report: SourceReport,
        guess: NameGuess,
        gemini: bool = True,
    ) -> None:
        self._context = context
        self._terminal = context.terminal
        self._answers = answers
        self._path = path
        self._report = report
        self._guess = guess
        self._stem = path.stem
        self._pinned = answers.imdb_id  # --imdb-id, or one typed in: it beats TMDB's
        self._gemini = gemini and context.gemini.enabled
        answers.review = answers.review and self._gemini

    async def run(self) -> Answers:
        self._preamble()
        self._clash_step()
        await self._film_step()
        await self._rows_step()
        self._encode_step()

        while True:
            match self._ready_step():
                case Ready.WRITE:
                    self._answers.write = True
                    return self._answers
                case Ready.SHOW:
                    self._answers.write = False
                    return self._answers
                case Ready.FILM:
                    await self._film_step()
                    await self._rows_step()
                case Ready.ROWS:
                    await self._rows_step()
                case Ready.ENCODE:
                    self._encode_step()
                case Ready.STOP:
                    raise Abort

    # --- the plumbing ------------------------------------------------------

    def _pick[T](self, prompt: str, rows: Sequence[Choice[T]], *, start: int = 0) -> T:
        """Offer ``rows``. Esc, ``q`` and Ctrl-C stop the run, as the footer says."""
        chosen = self._terminal.select(prompt, rows, start=start)
        if chosen is None:
            raise Abort
        return chosen.value

    def _setting[T](self, prompt: str, rows: Sequence[Choice[T]], current: T) -> T:
        """A sub-menu whose cursor starts on the value in force now."""
        start = next((index for index, row in enumerate(rows) if row.value == current), 0)
        return self._pick(prompt, rows, start=start)

    def _say(self, text: str = "") -> None:
        self._terminal.write(text)

    def _note(self, text: str) -> None:
        self._say(f"  {text}")

    @property
    def _year(self) -> int | None:
        """The release year, from whichever of the two knows it."""
        movie = self._answers.movie
        return (movie.year if movie else None) or self._guess.year

    # --- before the first question -----------------------------------------

    def _preamble(self) -> None:
        self._say(f"graindamage {__version__} — {STEPS} steps to two files.")
        self._say()
        self._say(f"File     {self._path.name}")
        self._say(f"Source   {source_summary(self._report.media)}")
        self._say(f"Name     {self._guess.describe()}")

    def _clash_step(self) -> None:
        """Ask about files in the way now, rather than after a look-up and a wait."""
        while not self._answers.force:
            present = existing_outputs(self._answers.out_dir, self._stem)
            if not present:
                return

            self._say()
            self._say(f"{', '.join(path.name for path in present)} is already there.")
            match self._pick(
                "Something is in the way",
                [
                    Choice(
                        value=Clash.REPLACE, label="Replace it", detail="write over the old one"
                    ),
                    Choice(value=Clash.ELSEWHERE, label="Write into another directory"),
                    Choice(value=Clash.STOP, label="Stop, and write nothing"),
                ],
            ):
                case Clash.REPLACE:
                    self._answers.force = True
                case Clash.ELSEWHERE:
                    if typed := self._terminal.ask("Directory to write into:"):
                        self._answers.out_dir = Path(typed).expanduser()
                case Clash.STOP:
                    raise Abort

    # --- step 1: the film ---------------------------------------------------

    async def _film_step(self) -> None:
        answers = self._answers
        before = answers.imdb_id
        query = self._guess.query

        while True:
            hits = await self._search(query) if query else []
            rows: list[Choice[MovieHit | Film]] = [
                Choice(
                    value=hit, label=hit.display_title, detail=film_detail(hit, self._guess.year)
                )
                for hit in hits
            ]
            start = next(
                (index for index, hit in enumerate(hits) if hit.year == self._guess.year), 0
            )
            if self._context.tmdb.enabled:
                rows.append(
                    Choice(
                        value=Film.RETYPE,
                        label="Search for another title",
                        detail="the filename is only a guess",
                    )
                )
            rows.append(
                Choice(
                    value=Film.IMDB,
                    label="Type the film's IMDb id",
                    detail=f"tt… — {answers.imdb_id or 'enough on its own for the rows'}",
                )
            )
            rows.append(
                Choice(
                    value=Film.NONE,
                    label="No film — settings from the file alone",
                    detail="grain then comes from the release year",
                )
            )
            rows.append(Choice(value=Film.STOP, label="Stop, and write nothing"))

            chosen = self._pick(_step(1, "Which film is this?"), rows, start=start)
            if isinstance(chosen, MovieHit):
                await self._details(chosen)
                break
            match chosen:
                case Film.RETYPE:
                    query = self._terminal.ask("Title to search for:") or query
                case Film.IMDB:
                    if self._ask_imdb_id():
                        break
                case Film.NONE:
                    answers.movie, answers.film_warning = None, NO_FILM_CHOSEN
                    break
                case Film.STOP:
                    raise Abort

        # Rows looked up for the film we have just replaced are about the wrong film.
        if answers.imdb_id != before and answers.found.source is SpecsSource.GEMINI:
            answers.found = FoundSpecs()

    async def _search(self, query: str) -> list[MovieHit]:
        """TMDB's hits, or an empty list and a line saying why there are none."""
        if not self._context.tmdb.enabled:
            self._say()
            self._say(NO_SEARCH)
            return []

        self._say()
        self._say(f'Searching TMDB for "{query}"…')
        try:
            hits = await self._context.tmdb.search(query)
        except ProviderError as exc:
            self._note(exc.message)
            return []
        if not hits:
            self._note(f'Nothing on TMDB for "{query}".')
        return hits

    async def _details(self, hit: MovieHit) -> None:
        """Follow a pick up with the details request, which is what carries the IMDb id."""
        answers = self._answers
        try:
            answers.movie = await self._context.tmdb.get_movie(hit.tmdb_id)
            answers.film_warning = None
        except ProviderError as exc:
            answers.movie = None
            answers.film_warning = f"{exc.message} Continuing without {hit.title}'s details."
            self._note(answers.film_warning)
        answers.imdb_id = self._pinned or (answers.movie.imdb_id if answers.movie else None)

    def _ask_imdb_id(self) -> bool:
        """Take a title id, or the address of a page with one in it.

        ``False`` puts the menu back up rather than moving on, because nothing was
        answered — an empty line is *never mind*, and a line with no id in it is a typo
        worth another go at.
        """
        typed = self._terminal.ask("IMDb id or the address of the film's page:")
        found = imdb_id_in(typed)
        if found is None:
            if typed:
                self._note("That has no IMDb title id in it — they look like tt0083658.")
            return False
        self._pinned = found
        self._answers.imdb_id = found
        if self._answers.movie is None:
            self._answers.film_warning = (
                f"No film details — {found} was used for the technical rows only."
            )
        return True

    # --- step 2: the technical rows ----------------------------------------

    async def _rows_step(self) -> None:
        answers = self._answers
        asked = False

        while True:
            self._describe_rows()
            imdb_id = answers.imdb_id
            rows: list[Choice[Rows]] = []
            if answers.found.source is not None:
                rows.append(
                    Choice(
                        value=Rows.KEEP,
                        label="Keep these rows",
                        detail=answers.found.describe,
                    )
                )
            elif imdb_id and self._gemini and not asked:
                rows.append(
                    Choice(
                        value=Rows.ASK,
                        label="Ask Gemini for the rows",
                        detail=f"{self._context.settings.gemini_model} — a look-up, not IMDb",
                    )
                )
            rows.append(
                Choice(
                    value=Rows.PASTE,
                    label="Paste the technical page yourself",
                    detail=technical_url(imdb_id) if imdb_id else "IMDb's own words, always",
                )
            )
            rows.append(Choice(value=Rows.FILE, label="Read the page out of a file"))
            if not imdb_id:
                rows.append(
                    Choice(
                        value=Rows.IMDB,
                        label="Type the film's IMDb id",
                        detail="so the page can be linked and looked up",
                    )
                )
            rows.append(
                Choice(
                    value=Rows.SKIP,
                    label="Skip — guess the grain from the release year",
                    detail=f"{self._year}" if self._year else "and the year is unknown too",
                )
            )
            rows.append(Choice(value=Rows.STOP, label="Stop, and write nothing"))

            match self._pick(_step(2, "The film's technical specifications"), rows):
                case Rows.KEEP:
                    return
                case Rows.ASK:
                    asked = True
                    answers.found = await self._ask_gemini(imdb_id or "")
                case Rows.PASTE:
                    self._take_paste()
                    return
                case Rows.FILE:
                    if self._take_file():
                        return
                case Rows.IMDB:
                    self._ask_imdb_id()
                case Rows.SKIP:
                    if not answers.found.warnings:
                        answers.found = FoundSpecs(warnings=(NO_ROWS_CHOSEN,))
                    return
                case Rows.STOP:
                    raise Abort

    def _describe_rows(self) -> None:
        """What is known so far, and the address of the page that would settle it."""
        answers = self._answers
        self._say()
        if answers.found.source is None:
            self._say(
                "The aspect ratio, negative format and process are what decide the grain settings."
            )
        else:
            self._say(f"Technical rows — {answers.found.describe}")
        for line in _spec_lines(answers.found.specs):
            self._note(line)
        if answers.imdb_id:
            self._note(technical_url(answers.imdb_id))

    async def _ask_gemini(self, imdb_id: str) -> FoundSpecs:
        """The look-up. Empty rows are a success — an honest blank asks for a paste."""
        movie = self._answers.movie
        self._say("Asking Gemini for the technical rows…")
        try:
            lookup = await self._context.gemini.technical_specs(
                imdb_id,
                title=movie.title if movie else None,
                year=movie.year if movie else None,
            )
        except ProviderError as exc:
            self._note(exc.message)
            return FoundSpecs(warnings=(f"{exc.message} {GUESSED_FROM_YEAR}",))

        if lookup.specs.is_empty:
            self._note(NO_ROWS_FOUND)
            return FoundSpecs(warnings=(f"{NO_ROWS_FOUND} {GUESSED_FROM_YEAR}",))
        return FoundSpecs(
            lookup.specs,
            source=SpecsSource.GEMINI,
            caveat=lookup.caveat,
            label=lookup.summary,
        )

    def _take_paste(self) -> None:
        """Print the address, open the box, and keep whatever came out of it."""
        answers = self._answers
        self._say()
        if answers.imdb_id:
            self._say("Open the page, select the specifications, copy them:")
            self._note(technical_url(answers.imdb_id))
        pasted = specs_from_paste(
            self._terminal.paste("Paste them below, then Ctrl-D. Ctrl-D on its own skips.")
        )

        if pasted.source is not None:
            answers.found = pasted
            for line in _spec_lines(pasted.specs):
                self._note(line)
            return
        for warning in pasted.warnings:
            self._note(warning)
        if answers.found.source is not None:
            self._note("Keeping the rows you already had.")
            return
        if answers.found.warnings:
            return  # a look-up already said why there are no rows; that reason stands
        answers.found = pasted if pasted.warnings else FoundSpecs(warnings=(NOTHING_PASTED,))

    def _take_file(self) -> bool:
        """Read the page out of a file. ``False`` asks the question again."""
        typed = self._terminal.ask("Path to the file:")
        if not typed:
            return False
        try:
            found = specs_from_file(Path(typed))
        except SetupProblem as exc:
            self._note(str(exc))
            return False

        self._answers.found = found
        if found.source is None:
            for warning in found.warnings:
                self._note(warning)
            return False
        return True

    # --- step 3: the encode -------------------------------------------------

    def _encode_step(self) -> None:
        answers = self._answers
        while True:
            rows: list[Choice[Encode]] = [
                Choice(
                    value=Encode.DONE,
                    label=_pad("These are fine"),
                    detail="on to the last step",
                ),
                Choice(
                    value=Encode.SPEED,
                    label=_pad("Speed"),
                    detail=f"{answers.speed.value} — {_SPEED_DETAIL[answers.speed]}",
                ),
                Choice(
                    value=Encode.SIZE,
                    label=_pad("Size"),
                    detail=f"{answers.size.value} — {_SIZE_DETAIL[answers.size]}",
                ),
                Choice(value=Encode.GRAIN, label=_pad("Grain"), detail=self._grain_line()),
                Choice(value=Encode.DEPTH, label=_pad("Bit depth"), detail=self._depth_line()),
                Choice(
                    value=Encode.ORDER,
                    label=_pad("Encoder order"),
                    detail=f"{self._first_encoder().label} first",
                ),
            ]
            if self._gemini:
                rows.append(
                    Choice(
                        value=Encode.REVIEW,
                        label=_pad("Gemini"),
                        detail="deciding — the rules below are its proposal"
                        if answers.review
                        else "off — the rules engine's plan, unreviewed",
                    )
                )
            rows.append(Choice(value=Encode.STOP, label=_pad("Stop, and write nothing")))

            match self._pick(_step(3, "The encode"), rows):
                case Encode.DONE:
                    return
                case Encode.SPEED:
                    answers.speed = self._setting(
                        "How much CPU time will you spend?",
                        [
                            Choice(
                                value=speed,
                                label=_pad(speed.value, _VALUE),
                                detail=_SPEED_DETAIL[speed],
                            )
                            for speed in SpeedPreference
                        ],
                        answers.speed,
                    )
                case Encode.SIZE:
                    answers.size = self._setting(
                        "Where on the size/fidelity curve?",
                        [
                            Choice(
                                value=size,
                                label=_pad(size.value, _VALUE),
                                detail=_SIZE_DETAIL[size],
                            )
                            for size in SizePreference
                        ],
                        answers.size,
                    )
                case Encode.GRAIN:
                    answers.grain = self._setting(
                        "How grainy is this film?",
                        [
                            Choice(
                                value=None,
                                label=_pad("let it be inferred", _VALUE),
                                detail=self._grain_line(inferred=True),
                            ),
                            *(
                                Choice(
                                    value=level,
                                    label=_pad(level.value, _VALUE),
                                    detail=_GRAIN_DETAIL[level],
                                )
                                for level in GrainLevel
                            ),
                        ],
                        answers.grain,
                    )
                case Encode.DEPTH:
                    answers.bit_depth = self._setting(
                        "What bit depth should the encode be?",
                        [
                            Choice(
                                value=None,
                                label=_pad("from the source", _VALUE),
                                detail=self._depth_line(),
                            ),
                            Choice(
                                value=10,
                                label=_pad("10-bit", _VALUE),
                                detail="what a film source wants",
                            ),
                            Choice(
                                value=8,
                                label=_pad("8-bit", _VALUE),
                                detail="only for a player that needs it",
                            ),
                            Choice(value=12, label=_pad("12-bit", _VALUE)),
                        ],
                        answers.bit_depth,
                    )
                case Encode.ORDER:
                    answers.encoder = self._setting(
                        "Which encoder leads — the live command, and the first preset?",
                        [
                            Choice(
                                value=Encoder.SVT_AV1,
                                label=_pad(Encoder.SVT_AV1.label, _VALUE),
                                detail="half the size, and it can synthesise grain back",
                            ),
                            Choice(
                                value=Encoder.X265,
                                label=_pad(Encoder.X265.label, _VALUE),
                                detail="plays on anything, and codes every grain particle",
                            ),
                        ],
                        self._first_encoder(),
                    )
                case Encode.REVIEW:
                    answers.review = self._setting(
                        "Let Gemini decide the settings?",
                        [
                            Choice(
                                value=True,
                                label="yes",
                                detail="it reads the film and the rules' proposal, then decides",
                            ),
                            Choice(
                                value=False,
                                label="no",
                                detail="the rules engine's tables and nothing else",
                            ),
                        ],
                        answers.review,
                    )
                case Encode.STOP:
                    raise Abort

    def _first_encoder(self) -> Encoder:
        return self._answers.encoder or Encoder.SVT_AV1

    def _grain_line(self, *, inferred: bool = False) -> str:
        """What grain the rules would call this film, and on what evidence."""
        profile = infer_grain(
            self._answers.found.specs,
            self._report.media,
            year=self._year,
            override=None if inferred else self._answers.grain,
        )
        origin = f" ({profile.origin_format})" if profile.origin_format else ""
        how = (
            "set by hand"
            if profile.user_override
            else f"inferred, confidence {profile.confidence:.0%}"
        )
        return f"{profile.level.value}{origin} — {how}"

    def _depth_line(self) -> str:
        depth = self._answers.bit_depth
        if depth is not None:
            return f"{depth}-bit, set by hand"
        video = self._report.media.video
        source = f"{video.bit_depth}-bit" if video and video.bit_depth else "unknown depth"
        return f"from the source — {source}"

    # --- step 4: write ------------------------------------------------------

    def _ready_step(self) -> Ready:
        answers = self._answers
        movie = answers.movie
        film = movie.display_title if movie else "not identified"
        if movie and movie.year:
            film = f"{film} ({movie.year})"
        if answers.imdb_id:
            film = f"{film} · {answers.imdb_id}"
        encode = ", ".join(
            [
                f"{answers.size.value} size",
                f"{answers.speed.value} speed",
                f"{self._first_encoder().label} first",
                f"Gemini {'deciding' if answers.review else 'off'}"
                if self._gemini
                else "no Gemini",
            ]
        )

        self._say()
        self._say("Ready.")
        self._note(f"Film     {film}")
        self._note(f"Rows     {answers.found.describe}")
        self._note(f"Grain    {self._grain_line()}")
        self._note(f"Encode   {encode}")
        if answers.write:
            for suffix in (PRESET_SUFFIX, SCRIPT_SUFFIX):
                self._note(f"Write    {answers.out_dir / f'{self._stem}{suffix}'}")
        else:
            self._note("Write    nothing — the settings are printed instead")

        return self._pick(
            _step(4, "Write the two files?"),
            [
                Choice(
                    value=Ready.WRITE,
                    label="Write them",
                    detail="a HandBrake preset and an FFmpeg script",
                ),
                Choice(
                    value=Ready.SHOW,
                    label="Show them instead",
                    detail="print the settings and both commands, and write nothing",
                ),
                Choice(value=Ready.FILM, label="Back to the film"),
                Choice(value=Ready.ROWS, label="Back to the technical rows"),
                Choice(value=Ready.ENCODE, label="Back to the encode settings"),
                Choice(value=Ready.STOP, label="Stop, and write nothing"),
            ],
            # --no-write asked for the settings, so that is the row Enter should land on.
            start=0 if answers.write else 1,
        )


# --- the file, when the command was given none -------------------------------

VIDEO_SUFFIXES = frozenset(
    {
        ".mkv",
        ".mp4",
        ".m4v",
        ".mov",
        ".avi",
        ".ts",
        ".m2ts",
        ".mts",
        ".webm",
        ".mpg",
        ".mpeg",
        ".vob",
        ".wmv",
        ".flv",
        ".ogv",
    }
)

# Where a film is likely to be mounted in a container that has no argument to go on.
LIKELY_ROOTS = ("/media", "/movies", "/films", "/video", "/videos", "/data", "/mnt")

_MAX_FOUND = 20
_MAX_DEPTH = 3


def find_videos(root: Path, *, depth: int = _MAX_DEPTH, limit: int = _MAX_FOUND) -> list[Path]:
    """Video files at or under ``root``, breadth-first and capped.

    Breadth-first because a mount point's own files are the likely answer and its
    twentieth subdirectory is not, and capped because this exists to fill a menu — a
    library of nine thousand films is browsed by typing a path, not by scrolling.
    """
    found: list[Path] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue and len(found) < limit:
        directory, level = queue.pop(0)
        try:
            entries = sorted(directory.iterdir())
        except OSError:  # unreadable, or gone since the listing above
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if level < depth:
                    queue.append((entry, level + 1))
            elif entry.suffix.lower() in VIDEO_SUFFIXES:
                found.append(entry)
                if len(found) >= limit:
                    break
    return found


def likely_videos(cwd: Path) -> list[Path]:
    """What to offer someone who ran the command with no argument at all."""
    found = find_videos(cwd)
    for name in LIKELY_ROOTS:
        if len(found) >= _MAX_FOUND:
            break
        root = Path(name)
        if root.is_dir() and root != cwd:
            found.extend(find_videos(root, limit=_MAX_FOUND - len(found)))
    return found


def choose_input(terminal: Terminal, *, cwd: Path | None = None) -> Path:
    """No argument and a terminal: offer what is lying about, or take a typed path."""
    here = cwd or Path.cwd()
    while True:
        found = likely_videos(here)
        rows: list[Choice[Path | None]] = [
            Choice(value=path, label=_relative(path, here), detail=_size(path)) for path in found
        ]
        rows.append(Choice(value=None, label="Type the path to a file", detail="or a mount point"))
        prompt = (
            "Which file? These are the ones I can see" if found else "I cannot see any video files"
        )
        chosen = terminal.select(prompt, rows)
        if chosen is None:
            raise Abort
        if chosen.value is not None:
            return chosen.value

        typed = terminal.ask("Path to the file:")
        if not typed:
            raise Abort
        path = Path(typed).expanduser()
        if path.is_file():
            return path
        if path.is_dir():
            terminal.write(f"  {path} is a directory — looking in it.")
            here = path
            continue
        terminal.write(f"  {path} is not there.")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _size(path: Path) -> str:
    """How big it is. A film is gigabytes; a sample or a trailer is not, and ``0.0 GB``
    beside a real file reads like a fault."""
    try:
        size = path.stat().st_size
    except OSError:  # pragma: no cover - it was there a moment ago
        return ""
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.1f} GB"
    return f"{size / 1_000_000:.0f} MB"


# --- shared bits of typesetting ---------------------------------------------


def _step(number: int, title: str) -> str:
    return f"Step {number} of {STEPS} · {title}"


def _pad(label: str, width: int = _LABEL) -> str:
    return f"{label:<{width}}"


def _spec_lines(specs: TechnicalSpecs) -> list[str]:
    """The rows that decide the grain, one line each, for showing back."""
    lines: list[str] = []
    for attribute, label in _SPEC_ROWS:
        values: list[str] = getattr(specs, attribute)
        if values:
            lines.append(f"{label:<17}{', '.join(values)}")
    return lines
