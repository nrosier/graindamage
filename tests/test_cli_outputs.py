"""The two files written beside the film.

One HandBrake document holding both presets, and one bash script with the first plan
live and the other commented out. The properties worth pinning down are that the presets
come from the same renderer the web app's download uses, that the script is runnable
from anywhere, and that neither file is ever replaced by accident.
"""

from __future__ import annotations

import json
import os
import shlex
import stat
from pathlib import Path

import pytest

from app.advice import finish_advice, render_handbrake_preset
from app.cli.outputs import (
    PRESET_SUFFIX,
    SCRIPT_SUFFIX,
    OutputExists,
    existing_outputs,
    preset_document,
    script_text,
    write_outputs,
)
from app.models import Advice, EncodeRequest
from tests.support import movie, request_for, run


@pytest.fixture
def advised() -> tuple[Advice, EncodeRequest]:
    """Baseline advice with commands attached — no key, no network, two plans."""
    request = request_for()
    return run(finish_advice(request)), request


# --- the HandBrake document -------------------------------------------------


def test_one_document_carries_a_preset_for_every_plan(
    advised: tuple[Advice, EncodeRequest],
) -> None:
    advice, request = advised

    document = preset_document(advice, request)

    assert len(document["PresetList"]) == len(advice.plans) == 2
    assert document["VersionMajor"] == 1
    names = [preset["PresetName"] for preset in document["PresetList"]]
    assert len(set(names)) == 2


def test_each_preset_is_the_one_the_web_app_would_hand_over(
    advised: tuple[Advice, EncodeRequest],
) -> None:
    """Only the envelopes are merged, so the two front-ends cannot drift apart."""
    advice, request = advised

    merged = preset_document(advice, request)["PresetList"]
    alone = [
        render_handbrake_preset(advice, request, plan)["PresetList"][0] for plan in advice.plans
    ]

    assert merged == alone


# --- the script -------------------------------------------------------------


def test_the_first_plan_runs_and_the_rest_are_offered(
    advised: tuple[Advice, EncodeRequest],
) -> None:
    advice, request = advised

    lines = script_text(advice, request, movie=movie()).splitlines()
    commands = [line for line in lines if "ffmpeg -i" in line]

    assert lines[0] == "#!/usr/bin/env bash"
    assert "set -euo pipefail" in lines
    assert commands[0] == advice.plans[0].ffmpeg_command
    assert commands[1] == f"# {advice.plans[1].ffmpeg_command}"


def test_the_header_says_which_film_and_how_grainy(
    advised: tuple[Advice, EncodeRequest],
) -> None:
    advice, request = advised

    text = script_text(advice, request, movie=movie(), specs_caveat="pasted from IMDb")

    assert "FFmpeg commands for Blade Runner" in text
    assert f"# Grain:  {advice.grain.level.value}" in text
    assert "# Specs:  pasted from IMDb" in text
    assert "1920×1080" in text  # the source summary the report shows too


def test_a_summary_with_newlines_in_it_stays_one_comment(
    advised: tuple[Advice, EncodeRequest],
) -> None:
    """A model's summary is prose, and a comment is a line: the header must not break."""
    advice, request = advised
    advice.summary = "Two lines\nof summary\n\nand a gap."

    text = script_text(advice, request)

    assert "# Two lines of summary and a gap." in text
    assert all(line.startswith("#") or not line.strip() for line in text.splitlines()[:12])


def test_a_script_beside_the_film_finds_its_own_directory(
    advised: tuple[Advice, EncodeRequest],
) -> None:
    """The container's path for the film is not the host's, so no path is embedded."""
    advice, request = advised

    text = script_text(advice, request)

    assert 'cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")"' in text
    assert request.input_path in text


def test_a_script_written_elsewhere_names_the_film_directory(
    advised: tuple[Advice, EncodeRequest],
) -> None:
    advice, request = advised

    text = script_text(advice, request, media_dir="/movies/Blade Runner (1982)")

    assert "cd -- '/movies/Blade Runner (1982)'" in text


# --- writing ----------------------------------------------------------------


def test_both_files_land_beside_the_film(
    advised: tuple[Advice, EncodeRequest], tmp_path: Path
) -> None:
    advice, request = advised

    written = write_outputs(
        advice, request, stem="film", directory=tmp_path, media_dir=tmp_path, movie=movie()
    )

    assert written.preset == tmp_path / f"film{PRESET_SUFFIX}"
    assert written.script == tmp_path / f"film{SCRIPT_SUFFIX}"
    assert json.loads(written.preset.read_text())["PresetList"]
    assert written.script.read_text().startswith("#!/usr/bin/env bash")


def test_the_script_is_executable(advised: tuple[Advice, EncodeRequest], tmp_path: Path) -> None:
    advice, request = advised

    written = write_outputs(advice, request, stem="film", directory=tmp_path, media_dir=tmp_path)

    mode = written.script.stat().st_mode
    assert mode & stat.S_IXUSR
    assert os.access(written.script, os.X_OK)


def test_an_output_directory_of_its_own_is_created(
    advised: tuple[Advice, EncodeRequest], tmp_path: Path
) -> None:
    advice, request = advised
    elsewhere = tmp_path / "presets" / "graindamage"

    written = write_outputs(
        advice, request, stem="film", directory=elsewhere, media_dir=tmp_path / "movies"
    )

    assert written.preset.parent == elsewhere
    assert f"cd -- {shlex.quote(str(tmp_path / 'movies'))}" in written.script.read_text()


def test_nothing_is_replaced_without_being_asked(
    advised: tuple[Advice, EncodeRequest], tmp_path: Path
) -> None:
    advice, request = advised
    (tmp_path / f"film{SCRIPT_SUFFIX}").write_text("# one I edited myself\n")

    with pytest.raises(OutputExists, match="--force"):
        write_outputs(advice, request, stem="film", directory=tmp_path, media_dir=tmp_path)

    assert (tmp_path / f"film{SCRIPT_SUFFIX}").read_text() == "# one I edited myself\n"
    assert not (tmp_path / f"film{PRESET_SUFFIX}").exists()


def test_force_replaces_both(advised: tuple[Advice, EncodeRequest], tmp_path: Path) -> None:
    advice, request = advised
    (tmp_path / f"film{SCRIPT_SUFFIX}").write_text("# stale\n")

    written = write_outputs(
        advice, request, stem="film", directory=tmp_path, media_dir=tmp_path, force=True
    )

    assert "stale" not in written.script.read_text()
    assert existing_outputs(tmp_path, "film") == list(written.paths)
