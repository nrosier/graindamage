"""The whole command, from a filename to two files on disk.

Every run here is offline: the providers get mock transports, ``ffprobe`` gets a stub
runner returning a real fixture, and the terminal gets two string buffers. What is being
tested is the wiring and the refusals — that the filename becomes a search, that the
menu's last row really does write nothing, that nothing is replaced without ``--force``,
and that a run with no keys at all still ends with a preset and a script.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx2
import pytest

from app.cli.app import EXIT_ABORTED, EXIT_OK, EXIT_SETUP, main
from app.cli.outputs import PRESET_SUFFIX, SCRIPT_SUFFIX
from app.cli.probe import Runner
from app.cli.prompts import Choice, Terminal
from app.cli.session import Context
from app.config import Settings
from app.providers.gemini import GeminiClient
from app.providers.tmdb import TmdbClient
from tests.support import fixture, json_response, make_settings, mock_client

FILM_NAME = "Blade.Runner.1982.2160p.BluRay.x265-GRP.mkv"
FFPROBE_REPORT = fixture("ffprobe_uhd_hdr.json")

# The technical page as it reads on screen, which is what a person pastes.
PASTED_PAGE = """Technical specifications
Aspect ratio
2.20 : 1
Negative format
35 mm
Cinematographic process
Super 35
"""

SEARCH_PAYLOAD: dict[str, Any] = {
    "results": [
        {"id": 335984, "title": "Blade Runner 2049", "release_date": "2017-10-04"},
        {"id": 78, "title": "Blade Runner", "release_date": "1982-06-25"},
    ]
}
MOVIE_PAYLOAD: dict[str, Any] = {
    "id": 78,
    "title": "Blade Runner",
    "release_date": "1982-06-25",
    "runtime": 117,
    "external_ids": {"imdb_id": "tt0083658"},
    "credits": {"crew": [{"name": "Ridley Scott", "job": "Director"}]},
}
SPECS_ANSWER: dict[str, Any] = {
    "negative_formats": ["35 mm"],
    "cinematographic_processes": ["Super 35"],
    "aspect_ratios": ["2.20 : 1"],
    "confidence": "medium",
}
NO_ROWS: dict[str, Any] = {"confidence": "low"}


# --- the doubles ------------------------------------------------------------


def film_in(directory: Path, name: str = FILM_NAME) -> Path:
    path = directory / name
    path.write_bytes(b"not really a film, and ffprobe is a stub")
    return path


def probing(code: int = 0, out: str = FFPROBE_REPORT, err: str = "") -> Runner:
    async def runner(args: Sequence[str], timeout: float) -> tuple[int, str, str]:
        return code, out, err

    return runner


def tmdb_transport(
    search: Any = None, movie: Any = None
) -> tuple[httpx2.AsyncClient, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if "/search/" in request.url.path:
            return json_response(SEARCH_PAYLOAD if search is None else search)
        return json_response(MOVIE_PAYLOAD if movie is None else movie)

    return mock_client(handler), seen


def gemini_transport(specs: Any = None) -> tuple[httpx2.AsyncClient, list[httpx2.Request]]:
    """Answers Gemini's two calls, told apart by the question each one asks."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        answer = SPECS_ANSWER if specs is None else specs
        if b"Fetch the technical specifications" not in request.content:
            answer = {"summary": "Reviewed for this film."}
        return httpx2.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": json.dumps(answer)}]},
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    return mock_client(handler), seen


def paper(typed: str = "") -> tuple[Terminal, io.StringIO]:
    """A terminal on two buffers: no TTY, so menus are numbered."""
    err = io.StringIO()
    return Terminal(stdin=io.StringIO(typed), stderr=err), err


class Pasting(Terminal):
    """A terminal that answers as a person at a keyboard would, without needing one.

    ``interactive`` is what decides whether a paste is offered at all, and it cannot be
    true for a pair of string buffers — so this says so directly, and stands in for the
    two prompts a real one would draw.
    """

    def __init__(self, pasted: str) -> None:
        self.err = io.StringIO()
        super().__init__(stdin=io.StringIO(), stderr=self.err)
        self.pasted = pasted

    @property
    def interactive(self) -> bool:
        return True

    def select[T](
        self, prompt: str, choices: Sequence[Choice[T]], *, start: int = 0
    ) -> Choice[T] | None:
        self.write(prompt)
        return choices[0] if choices else None

    def paste(self, prompt: str) -> str:
        self.write(prompt)
        return self.pasted


def keyed(**overrides: Any) -> Settings:
    return make_settings(
        **{"tmdb_api_key": "tmdb-key", "gemini_api_key": "gemini-key", **overrides}
    )


def context(
    settings: Settings | None = None,
    *,
    typed: str = "",
    terminal: Terminal | None = None,
    tmdb: httpx2.AsyncClient | None = None,
    gemini: httpx2.AsyncClient | None = None,
    runner: Runner | None = None,
) -> Context:
    resolved = settings or keyed()
    return Context(
        settings=resolved,
        terminal=terminal if terminal is not None else paper(typed)[0],
        tmdb=TmdbClient(resolved, client=tmdb if tmdb is not None else tmdb_transport()[0]),
        gemini=GeminiClient(
            resolved, client=gemini if gemini is not None else gemini_transport()[0]
        ),
        runner=runner or probing(),
    )


def written_files(directory: Path, film: Path) -> tuple[Path, Path]:
    return (
        directory / f"{film.stem}{PRESET_SUFFIX}",
        directory / f"{film.stem}{SCRIPT_SUFFIX}",
    )


# --- the happy path ---------------------------------------------------------


def test_a_film_becomes_a_preset_and_a_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    film = film_in(tmp_path)

    assert main([str(film), "--yes"], context=context()) == EXIT_OK

    preset, script = written_files(tmp_path, film)
    document = json.loads(preset.read_text())
    assert len(document["PresetList"]) == 2
    assert script.read_text().startswith("#!/usr/bin/env bash")

    printed = capsys.readouterr().out
    assert "Blade Runner · 1982 · Ridley Scott" in printed
    assert "https://www.imdb.com/title/tt0083658/technical/" in printed
    assert "AV1 (SVT-AV1)" in printed and "x265 (HEVC)" in printed
    assert str(preset) in printed and str(script) in printed


def test_the_filename_is_what_gets_searched_for(tmp_path: Path) -> None:
    client, seen = tmdb_transport()

    main([str(film_in(tmp_path)), "--yes"], context=context(tmdb=client))

    assert dict(seen[0].url.params)["query"] == "Blade Runner"


def test_a_title_on_the_command_line_beats_the_filename(tmp_path: Path) -> None:
    client, seen = tmdb_transport()

    main(
        [str(film_in(tmp_path)), "--yes", "--title", "Blade Runner 2049"],
        context=context(tmdb=client),
    )

    assert dict(seen[0].url.params)["query"] == "Blade Runner 2049"


def test_yes_takes_the_hit_the_filename_agrees_with(tmp_path: Path) -> None:
    """TMDB's first answer is the 2049 sequel; the file says 1982, so 1982 wins."""
    client, seen = tmdb_transport()

    main([str(film_in(tmp_path)), "--yes"], context=context(tmdb=client))

    assert seen[1].url.path.endswith("/movie/78")


def test_the_report_is_the_only_thing_on_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Progress and prompts go to stderr, so a redirected report stays clean."""
    terminal, err = paper()
    film = film_in(tmp_path)

    main([str(film), "--yes"], context=context(terminal=terminal))

    assert "File     " in err.getvalue()
    assert "File     " not in capsys.readouterr().out


def test_quiet_prints_the_two_paths_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    film = film_in(tmp_path)

    main([str(film), "--yes", "--quiet"], context=context())

    assert capsys.readouterr().out.splitlines() == [str(p) for p in written_files(tmp_path, film)]


# --- the menu ---------------------------------------------------------------


def test_the_menu_offers_the_hits_then_retype_then_stop(tmp_path: Path) -> None:
    terminal, err = paper("2\n")

    assert main([str(film_in(tmp_path))], context=context(terminal=terminal)) == EXIT_OK

    printed = err.getvalue()
    assert "1) Blade Runner 2049  2017" in printed
    assert "2) Blade Runner  1982  · matches the filename" in printed
    assert "3) Wrong film — let me type the title" in printed
    assert "4) Stop, and write nothing" in printed


def test_the_last_row_stops_the_run_and_writes_nothing(tmp_path: Path) -> None:
    terminal, err = paper("4\n")
    film = film_in(tmp_path)

    assert main([str(film)], context=context(terminal=terminal)) == EXIT_ABORTED

    assert not any(path.exists() for path in written_files(tmp_path, film))
    assert "Stopped. Nothing was written." in err.getvalue()


def test_the_retype_row_searches_again(tmp_path: Path) -> None:
    """Row three is "wrong film": it takes a title and comes back with a fresh menu."""
    client, seen = tmdb_transport()
    terminal, _ = paper("3\nSomething Else\n2\n")

    assert (
        main([str(film_in(tmp_path))], context=context(terminal=terminal, tmdb=client)) == EXIT_OK
    )

    queries = [
        dict(request.url.params).get("query") for request in seen if "/search/" in request.url.path
    ]
    assert queries == ["Blade Runner", "Something Else"]


def test_an_empty_title_at_the_retype_prompt_stops(tmp_path: Path) -> None:
    terminal, _ = paper("3\n\n")
    film = film_in(tmp_path)

    assert main([str(film)], context=context(terminal=terminal)) == EXIT_ABORTED
    assert not any(path.exists() for path in written_files(tmp_path, film))


def test_nothing_found_asks_for_another_title(tmp_path: Path) -> None:
    client, _ = tmdb_transport(search={"results": []})
    terminal, err = paper("Blade Runner\n")
    film = film_in(tmp_path)

    assert main([str(film)], context=context(terminal=terminal, tmdb=client)) == EXIT_ABORTED
    assert 'Nothing on TMDB for "Blade Runner".' in err.getvalue()


# --- degrading rather than failing -----------------------------------------


def test_without_a_tmdb_key_the_film_is_skipped_but_the_files_are_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(gemini_api_key="gemini-key")
    film = film_in(tmp_path)

    assert main([str(film)], context=context(settings)) == EXIT_OK

    printed = capsys.readouterr().out
    assert "No film identified" in printed
    assert "TMDB_API_KEY" in printed
    assert all(path.exists() for path in written_files(tmp_path, film))


def test_an_imdb_id_alone_still_buys_the_technical_rows(tmp_path: Path) -> None:
    settings = make_settings(gemini_api_key="gemini-key")
    client, seen = gemini_transport()

    code = main(
        [str(film_in(tmp_path)), "--imdb-id", "tt0083658"], context=context(settings, gemini=client)
    )

    assert code == EXIT_OK
    assert b"tt0083658" in seen[0].content


def test_no_gemini_means_no_request_at_all(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client, seen = gemini_transport()

    code = main([str(film_in(tmp_path)), "--yes", "--no-gemini"], context=context(gemini=client))

    assert code == EXIT_OK
    assert seen == []
    assert "--no-gemini" in capsys.readouterr().out


def test_a_tmdb_that_is_down_costs_a_warning_not_the_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return json_response({"status_message": "Service unavailable"}, 503)

    film = film_in(tmp_path)
    code = main([str(film), "--yes"], context=context(tmdb=mock_client(handler)))

    assert code == EXIT_OK
    assert "Continuing without the film's details." in capsys.readouterr().out
    assert all(path.exists() for path in written_files(tmp_path, film))


def test_the_year_in_the_filename_is_what_grain_falls_back_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no film and no rows, the only era hint left is the name of the file."""
    settings = make_settings()  # no keys at all
    film = film_in(tmp_path, "Some.Film.1965.1080p.BluRay.x264-GRP.mkv")

    assert main([str(film)], context=context(settings)) == EXIT_OK

    printed = capsys.readouterr().out
    assert "Grain    moderate (35 mm)" in printed
    assert "A 1965 release was almost certainly 35 mm" in printed


def test_the_year_flag_reaches_the_grain_estimate_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    film = film_in(tmp_path, "Some.Film.1080p.BluRay.x264-GRP.mkv")

    code = main([str(film), "--year", "2019"], context=context(make_settings()))

    assert code == EXIT_OK
    assert "A 2019 release was most likely shot digitally" in capsys.readouterr().out


# --- the technical rows ----------------------------------------------------


def test_a_specs_file_wins_and_is_never_looked_up(tmp_path: Path) -> None:
    pasted = tmp_path / "technical.txt"
    pasted.write_text(PASTED_PAGE, encoding="utf-8")
    client, seen = gemini_transport()
    film = film_in(tmp_path)

    code = main([str(film), "--yes", "--specs", str(pasted)], context=context(gemini=client))

    assert code == EXIT_OK
    assert not any(b"Fetch the technical specifications" in request.content for request in seen)
    assert "# Specs:  pasted from technical.txt" in written_files(tmp_path, film)[1].read_text()


def test_a_specs_file_that_says_nothing_is_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pasted = tmp_path / "technical.txt"
    pasted.write_text("just some notes I made\n", encoding="utf-8")

    code = main([str(film_in(tmp_path)), "--yes", "--specs", str(pasted)], context=context())

    assert code == EXIT_OK
    printed = capsys.readouterr().out
    assert "Nothing recognisable in technical.txt" in printed
    assert "guessed from the release year" in printed


def test_a_missing_specs_file_stops_the_run(tmp_path: Path) -> None:
    terminal, err = paper()

    code = main(
        [str(film_in(tmp_path)), "--yes", "--specs", str(tmp_path / "nope.txt")],
        context=context(terminal=terminal),
    )

    assert code == EXIT_SETUP
    assert "--specs" in err.getvalue()


def test_rows_gemini_does_not_have_are_offered_as_a_paste(tmp_path: Path) -> None:
    terminal = Pasting(PASTED_PAGE)
    film = film_in(tmp_path)

    code = main(
        [str(film)], context=context(terminal=terminal, gemini=gemini_transport(NO_ROWS)[0])
    )

    assert code == EXIT_OK
    assert "https://www.imdb.com/title/tt0083658/technical/" in terminal.err.getvalue()
    assert (
        "# Specs:  pasted from the technical page" in written_files(tmp_path, film)[1].read_text()
    )


def test_an_empty_paste_continues_with_grain_guessed_from_the_year(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-D on an empty box is an answer: carry on, and say the grain is a guess."""
    terminal = Pasting("")

    code = main(
        [str(film_in(tmp_path))],
        context=context(terminal=terminal, gemini=gemini_transport(NO_ROWS)[0]),
    )

    assert code == EXIT_OK
    printed = capsys.readouterr().out
    assert "Gemini did not have the technical rows" in printed
    assert "guessed from the release year" in printed


# --- the overrides ---------------------------------------------------------


def test_the_encoder_flag_decides_which_command_runs(tmp_path: Path) -> None:
    film = film_in(tmp_path)

    main([str(film), "--yes", "--encoder", "x265"], context=context())

    preset, script = written_files(tmp_path, film)
    assert "x265" in json.loads(preset.read_text())["PresetList"][0]["PresetName"]
    live = [line for line in script.read_text().splitlines() if line.startswith("ffmpeg")]
    assert "libx265" in live[0]


def test_the_preferences_reach_the_advice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            str(film_in(tmp_path)),
            "--yes",
            "--grain",
            "heavy",
            "--speed",
            "quality",
            "--size",
            "archival",
        ],
        context=context(),
    )

    assert code == EXIT_OK
    assert "Grain    heavy" in capsys.readouterr().out


def test_an_output_directory_of_its_own_still_points_at_the_film(tmp_path: Path) -> None:
    film = film_in(tmp_path)
    elsewhere = tmp_path / "presets"

    assert main([str(film), "--yes", "--outdir", str(elsewhere)], context=context()) == EXIT_OK

    preset, script = written_files(elsewhere, film)
    assert preset.exists()
    assert f"cd -- {tmp_path}" in script.read_text()


# --- refusing ---------------------------------------------------------------


def test_files_that_are_already_there_stop_the_run_before_anything_is_asked(
    tmp_path: Path,
) -> None:
    """The check comes first, so a re-run costs no request and no waiting."""
    film = film_in(tmp_path)
    _, script = written_files(tmp_path, film)
    script.write_text("# one I edited myself\n")
    client, seen = tmdb_transport()
    terminal, err = paper()

    code = main([str(film), "--yes"], context=context(terminal=terminal, tmdb=client))

    assert code == EXIT_SETUP
    assert "--force" in err.getvalue()
    assert seen == []
    assert script.read_text() == "# one I edited myself\n"


def test_force_replaces_them(tmp_path: Path) -> None:
    film = film_in(tmp_path)
    _, script = written_files(tmp_path, film)
    script.write_text("# stale\n")

    assert main([str(film), "--yes", "--force"], context=context()) == EXIT_OK
    assert "stale" not in script.read_text()


def test_a_file_that_is_not_there_is_a_setup_problem(tmp_path: Path) -> None:
    terminal, err = paper()

    code = main([str(tmp_path / "nope.mkv"), "--yes"], context=context(terminal=terminal))

    assert code == EXIT_SETUP
    assert "is not there" in err.getvalue()


def test_a_directory_is_a_setup_problem(tmp_path: Path) -> None:
    terminal, err = paper()

    assert main([str(tmp_path), "--yes"], context=context(terminal=terminal)) == EXIT_SETUP
    assert "is a directory" in err.getvalue()


def test_an_ffprobe_that_refuses_the_file_stops_the_run(tmp_path: Path) -> None:
    terminal, err = paper()
    film = film_in(tmp_path)

    code = main(
        [str(film), "--yes"],
        context=context(terminal=terminal, runner=probing(1, "", "moov atom not found")),
    )

    assert code == EXIT_SETUP
    assert "moov atom not found" in err.getvalue()
    assert not any(path.exists() for path in written_files(tmp_path, film))


def test_an_imdb_id_that_is_not_one_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main([str(film_in(tmp_path)), "--imdb-id", "0083658"], context=context())

    assert raised.value.code == EXIT_SETUP
