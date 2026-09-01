"""The menu walk-through: four steps, and every one of them a list.

The terminal here answers as a person would — a row number, a label, or nothing at all,
which is Enter and lands on whatever row the cursor started on. That last case is the
one worth defending: pressing Enter four times has to be a complete, sensible run, and
it has to pick the film the filename agrees with rather than TMDB's first answer.

Everything is offline: the same mock transports and stub ``ffprobe`` as
``test_cli_run.py``, which is where the doubles live.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.cli.app import EXIT_ABORTED, EXIT_OK, EXIT_SETUP, main
from app.cli.prompts import Choice, Terminal
from app.cli.wizard import choose_input
from tests.support import FILM_NAME, film_in, make_settings
from tests.test_cli_run import (
    NO_ROWS,
    PASTED_PAGE,
    context,
    gemini_transport,
    keyed,
    paper,
    tmdb_transport,
    written_files,
)


class Wizarding(Terminal):
    """A terminal that answers menus from a script, and Enter once the script runs out.

    A pick is a row number, or a piece of a row's label — which keeps the tests readable
    and, more usefully, makes them fail loudly when a row they name stops existing.
    ``None``, or a script that has run out, is Enter: whichever row the cursor is on.
    """

    def __init__(
        self,
        picks: Sequence[int | str | None] = (),
        *,
        typed: Sequence[str] = (),
        pasted: str = "",
    ) -> None:
        self.err = io.StringIO()
        super().__init__(stdin=io.StringIO(), stderr=self.err)
        self.picks = list(picks)
        self.typed = list(typed)
        self.pasted = pasted
        self.menus: list[list[str]] = []

    @property
    def interactive(self) -> bool:
        return True

    def select[T](
        self, prompt: str, choices: Sequence[Choice[T]], *, start: int = 0
    ) -> Choice[T] | None:
        labels = [choice.label for choice in choices]
        self.menus.append(labels)
        self.write(prompt)
        for index, choice in enumerate(choices):
            mark = "▸" if index == start else " "
            self.write(f" {mark} {choice.label}  {choice.detail or ''}")

        wanted = self.picks.pop(0) if self.picks else None
        if wanted is None:
            return choices[start]
        if isinstance(wanted, int):
            return choices[wanted]
        found = next(
            (index for index, label in enumerate(labels) if wanted.lower() in label.lower()), None
        )
        assert found is not None, f"no row like {wanted!r} in {labels}"
        return choices[found]

    def ask(self, prompt: str) -> str:
        self.write(prompt)
        return self.typed.pop(0) if self.typed else ""

    def paste(self, prompt: str) -> str:
        self.write(prompt)
        return self.pasted

    def menu_count(self, label: str) -> int:
        """How many times a menu containing ``label`` was drawn."""
        return sum(1 for menu in self.menus if any(label in row for row in menu))


# --- pressing enter ---------------------------------------------------------


def test_enter_through_every_step_writes_the_two_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal = Wizarding()
    film = film_in(tmp_path)

    assert main([str(film)], context=context(terminal=terminal)) == EXIT_OK

    preset, script = written_files(tmp_path, film)
    assert len(json.loads(preset.read_text())["PresetList"]) == 2
    assert script.read_text().startswith("#!/usr/bin/env bash")

    prompts = terminal.err.getvalue()
    for step in ("Step 1 of 4", "Step 2 of 4", "Step 3 of 4", "Step 4 of 4"):
        assert step in prompts
    assert "Blade Runner · 1982 · Ridley Scott" in capsys.readouterr().out


def test_the_cursor_starts_on_the_film_the_filename_agrees_with(tmp_path: Path) -> None:
    """TMDB's first answer is the 2049 sequel; the file says 1982, so Enter means 1982."""
    client, seen = tmdb_transport()

    main([str(film_in(tmp_path))], context=context(terminal=Wizarding(), tmdb=client))

    assert seen[1].url.path.endswith("/movie/78")


def test_the_rows_are_asked_for_once_and_then_offered_to_keep(tmp_path: Path) -> None:
    terminal = Wizarding()
    client, seen = gemini_transport()

    assert (
        main([str(film_in(tmp_path))], context=context(terminal=terminal, gemini=client)) == EXIT_OK
    )

    lookups = [r for r in seen if b"Fetch the technical specifications" in r.content]
    assert len(lookups) == 1
    assert terminal.menu_count("Keep these rows") == 1
    # A row is cut off at the terminal's width, so it shows the short form of the caveat
    # rather than the three sentences the report gets.
    assert "Keep these rows  Gemini web search · confidence medium" in terminal.err.getvalue()


def test_every_menu_stops_the_run_and_writes_nothing(tmp_path: Path) -> None:
    terminal = Wizarding(
        ["Blade Runner", "Ask Gemini", "Keep these rows", "These are fine", "Stop"]
    )
    film = film_in(tmp_path)

    assert main([str(film)], context=context(terminal=terminal)) == EXIT_ABORTED
    assert not any(path.exists() for path in written_files(tmp_path, film))
    assert "Stopped. Nothing was written." in terminal.err.getvalue()


# --- step 2: where the rows come from --------------------------------------


def test_the_page_can_be_pasted_instead_of_looked_up(tmp_path: Path) -> None:
    """The address is printed, the box is opened, and no look-up is made at all."""
    terminal = Wizarding(["Blade Runner", "Paste the technical page"], pasted=PASTED_PAGE)
    client, seen = gemini_transport()
    film = film_in(tmp_path)

    assert main([str(film)], context=context(terminal=terminal, gemini=client)) == EXIT_OK

    assert "https://www.imdb.com/title/tt0083658/technical/" in terminal.err.getvalue()
    assert not any(b"Fetch the technical specifications" in r.content for r in seen)
    assert (
        "# Specs:  pasted from the technical page" in written_files(tmp_path, film)[1].read_text()
    )


def test_a_look_up_with_nothing_in_it_falls_through_to_the_paste(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal = Wizarding(pasted=PASTED_PAGE)
    film = film_in(tmp_path)

    code = main(
        [str(film)], context=context(terminal=terminal, gemini=gemini_transport(NO_ROWS)[0])
    )

    assert code == EXIT_OK
    assert terminal.menu_count("Ask Gemini for the rows") == 1
    assert (
        "# Specs:  pasted from the technical page" in written_files(tmp_path, film)[1].read_text()
    )


def test_skipping_the_rows_says_so_in_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal = Wizarding(["Blade Runner", "Skip"])

    assert main([str(film_in(tmp_path))], context=context(terminal=terminal)) == EXIT_OK

    printed = capsys.readouterr().out
    assert "You skipped the technical rows" in printed
    assert "guessed from the release year" in printed


def test_the_rows_can_be_read_out_of_a_file(tmp_path: Path) -> None:
    page = tmp_path / "technical.txt"
    page.write_text(PASTED_PAGE, encoding="utf-8")
    terminal = Wizarding(["Blade Runner", "Read the page out of a file"], typed=[str(page)])
    film = film_in(tmp_path)

    assert main([str(film)], context=context(terminal=terminal)) == EXIT_OK
    assert "# Specs:  pasted from technical.txt" in written_files(tmp_path, film)[1].read_text()


def test_a_file_that_is_not_there_is_said_so_and_asked_again(tmp_path: Path) -> None:
    """A typo in step 2 costs a line, not the run."""
    terminal = Wizarding(
        ["Blade Runner", "Read the page out of a file", "Skip"], typed=[str(tmp_path / "nope.txt")]
    )

    assert main([str(film_in(tmp_path))], context=context(terminal=terminal)) == EXIT_OK
    assert "nope.txt" in terminal.err.getvalue()


# --- step 1 without a TMDB key ---------------------------------------------


def test_an_imdb_id_can_be_typed_when_there_is_nothing_to_search(tmp_path: Path) -> None:
    """No TMDB key removes the hits, not the run: the id is enough for the rows."""
    terminal = Wizarding(
        ["Type the film's IMDb id"], typed=["https://www.imdb.com/title/tt0083658/"]
    )
    client, seen = gemini_transport()

    code = main(
        [str(film_in(tmp_path))],
        context=context(make_settings(gemini_api_key="k"), terminal=terminal, gemini=client),
    )

    assert code == EXIT_OK
    assert b"tt0083658" in seen[0].content
    assert "TMDB_API_KEY" in terminal.err.getvalue()


def test_with_no_key_the_search_row_is_gone_rather_than_dead(tmp_path: Path) -> None:
    """A row that cannot do anything is worse than a missing one: Enter would loop."""
    terminal = Wizarding(["No film"])

    assert main([str(film_in(tmp_path))], context=context(make_settings(), terminal=terminal)) == (
        EXIT_OK
    )

    assert terminal.menus[0] == [
        "Type the film's IMDb id",
        "No film — settings from the file alone",
        "Stop, and write nothing",
    ]


def test_a_line_with_no_id_in_it_puts_the_film_menu_back_up(tmp_path: Path) -> None:
    terminal = Wizarding(["Type the film's IMDb id", "No film"], typed=["the one with the unicorn"])

    assert main([str(film_in(tmp_path))], context=context(make_settings(), terminal=terminal)) == (
        EXIT_OK
    )

    # That row is step 1's alone, so counting it counts how often step 1 was drawn.
    assert terminal.menu_count("No film — settings from the file alone") == 2
    assert "they look like tt0083658" in terminal.err.getvalue()


def test_carrying_on_with_no_film_at_all_still_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal = Wizarding(["No film"])
    film = film_in(tmp_path)

    assert main([str(film)], context=context(terminal=terminal)) == EXIT_OK

    printed = capsys.readouterr().out
    assert "No film identified" in printed
    assert "carry on without identifying the film" in printed
    assert all(path.exists() for path in written_files(tmp_path, film))


# --- step 3 and step 4 ------------------------------------------------------


def test_a_setting_changed_in_the_hub_reaches_the_advice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal = Wizarding(
        ["Blade Runner", "Ask Gemini", "Keep these rows", "Grain", "heavy", "These are fine"]
    )

    assert main([str(film_in(tmp_path))], context=context(terminal=terminal)) == EXIT_OK
    assert "Grain    heavy" in capsys.readouterr().out


def test_the_last_step_goes_back_to_the_encode_settings(tmp_path: Path) -> None:
    terminal = Wizarding(
        [
            "Blade Runner",
            "Ask Gemini",
            "Keep these rows",
            "These are fine",
            "Back to the encode settings",
            "Encoder order",
            "x265",
            "These are fine",
            "Write them",
        ]
    )
    film = film_in(tmp_path)

    assert main([str(film)], context=context(terminal=terminal)) == EXIT_OK

    # The last screen came round a second time, and the hub redrew after the change so
    # that the row shows the value that is now in force.
    assert terminal.menu_count("Back to the encode settings") == 2
    assert terminal.menu_count("Encoder order") == 3

    preset = json.loads(written_files(tmp_path, film)[0].read_text())
    assert "x265" in preset["PresetList"][0]["PresetName"]


def test_the_show_row_prints_the_settings_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal = Wizarding(
        ["Blade Runner", "Ask Gemini", "Keep these rows", "These are fine", "Show them instead"]
    )
    film = film_in(tmp_path)

    assert main([str(film)], context=context(terminal=terminal)) == EXIT_OK

    assert not any(path.exists() for path in written_files(tmp_path, film))
    printed = capsys.readouterr().out
    assert "HandBrake  HandBrakeCLI -i" in printed
    assert "You asked for the settings only" in printed


def test_no_write_puts_the_cursor_on_the_show_row(tmp_path: Path) -> None:
    """--no-write has already answered step 4, so four Enters must not write anything."""
    terminal = Wizarding()
    film = film_in(tmp_path)

    assert main([str(film), "--no-write"], context=context(terminal=terminal)) == EXIT_OK

    assert not any(path.exists() for path in written_files(tmp_path, film))
    drawn = terminal.err.getvalue()
    assert "Write    nothing — the settings are printed instead" in drawn
    assert "▸ Show them instead" in drawn


def test_files_in_the_way_are_the_first_thing_asked_about(tmp_path: Path) -> None:
    """And answering "somewhere else" leaves the one that was in the way alone."""
    film = film_in(tmp_path)
    _, script = written_files(tmp_path, film)
    script.write_text("# one I edited myself\n")
    elsewhere = tmp_path / "presets"
    terminal = Wizarding(["another directory"], typed=[str(elsewhere)])

    assert main([str(film)], context=context(terminal=terminal)) == EXIT_OK

    assert terminal.menus[0] == [
        "Replace it",
        "Write into another directory",
        "Stop, and write nothing",
    ]
    assert script.read_text() == "# one I edited myself\n"
    assert all(path.exists() for path in written_files(elsewhere, film))


def test_replacing_them_is_the_first_row(tmp_path: Path) -> None:
    film = film_in(tmp_path)
    _, script = written_files(tmp_path, film)
    script.write_text("# stale\n")
    terminal = Wizarding()

    assert main([str(film)], context=context(terminal=terminal)) == EXIT_OK
    assert "stale" not in script.read_text()


# --- the file, when there was no argument ----------------------------------


def test_choosing_a_file_from_the_menu(tmp_path: Path) -> None:
    film = film_in(tmp_path)
    terminal = Wizarding([FILM_NAME])

    assert choose_input(terminal, cwd=tmp_path) == film


def test_the_menu_row_carries_the_size(tmp_path: Path) -> None:
    """Which unit is chosen is ``library.human_size``'s business; that the row shows it
    at all is this menu's, because size is how you tell a film from its trailer."""
    small = tmp_path / "Trailer.mkv"
    with small.open("wb") as handle:
        handle.truncate(40_000_000)
    big = tmp_path / "Feature.mkv"
    with big.open("wb") as handle:
        handle.truncate(8_100_000_000)
    terminal = Wizarding(["Feature"])

    assert choose_input(terminal, cwd=tmp_path) == big

    drawn = terminal.err.getvalue()
    assert "Feature.mkv  8.1 GB" in drawn
    assert "Trailer.mkv  40 MB" in drawn


def test_a_typed_path_that_is_not_there_is_asked_again(tmp_path: Path) -> None:
    film = film_in(tmp_path)
    terminal = Wizarding(["Type the path", FILM_NAME], typed=[str(tmp_path / "nope.mkv")])

    assert choose_input(terminal, cwd=tmp_path) == film
    assert "is not there" in terminal.err.getvalue()


def test_no_argument_offers_what_it_can_see(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    film = film_in(tmp_path)
    monkeypatch.chdir(tmp_path)
    terminal = Wizarding([FILM_NAME])

    assert main([], context=context(keyed(), terminal=terminal)) == EXIT_OK
    assert all(path.exists() for path in written_files(tmp_path, film))


def test_no_argument_and_no_terminal_is_a_setup_problem(tmp_path: Path) -> None:
    terminal, err = paper()

    assert main([], context=context(terminal=terminal)) == EXIT_SETUP
    assert "path to one video file" in err.getvalue()
