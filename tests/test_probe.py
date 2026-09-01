"""Running ffprobe, and every way that can go wrong.

The subprocess sits behind an injected ``runner``, so all of the failure handling is
tested without ffmpeg installed. Two tests do start a real process — the missing binary
and the timeout — because those paths are entirely about what
:func:`~app.probe.run_ffprobe` does with the operating system's answer, and a fake
runner would only be testing itself.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.models import SourceTool
from app.probe import FFPROBE_ARGS, ProbeFailed, Runner, probe, run_ffprobe
from tests.support import fixture, run

FILM = Path("/movies/Blade.Runner.1982.2160p.mkv")


def runner_returning(code: int, out: str = "", err: str = "") -> tuple[Runner, list[Sequence[str]]]:
    """A stub runner with the argument lists it was called with."""
    calls: list[Sequence[str]] = []

    async def runner(args: Sequence[str], timeout: float) -> tuple[int, str, str]:
        calls.append(list(args))
        return code, out, err

    return runner, calls


def test_a_good_probe_becomes_a_source_report() -> None:
    runner, calls = runner_returning(0, fixture("ffprobe_uhd_hdr.json"))

    report = run(probe(FILM, runner=runner))

    assert report.tool is SourceTool.FFPROBE
    assert report.has_video
    assert report.media.video is not None
    assert report.media.video.height == 2160
    assert calls == [["ffprobe", *FFPROBE_ARGS, str(FILM)]]


def test_the_executable_can_be_pointed_somewhere_else() -> None:
    runner, calls = runner_returning(0, fixture("ffprobe_uhd_hdr.json"))

    run(probe(FILM, executable="/opt/bin/ffprobe", runner=runner))

    assert calls[0][0] == "/opt/bin/ffprobe"


def test_a_filename_starting_with_a_dash_is_not_offered_as_a_flag() -> None:
    runner, calls = runner_returning(0, fixture("ffprobe_uhd_hdr.json"))

    run(probe(Path("-weird.mkv"), runner=runner))

    assert calls[0][-1] == "./-weird.mkv"


def test_a_non_zero_exit_carries_the_end_of_stderr() -> None:
    runner, _ = runner_returning(1, "", "line one\n\nmoov atom not found\nInvalid data found")

    with pytest.raises(ProbeFailed) as raised:
        run(probe(FILM, runner=runner))

    message = str(raised.value)
    assert "exited 1 on Blade.Runner.1982.2160p.mkv" in message
    assert "moov atom not found / Invalid data found" in message


def test_a_silent_success_is_still_a_failure() -> None:
    """Exit 0 with no output means nothing was described, whatever it claims."""
    runner, _ = runner_returning(0, "   \n")

    with pytest.raises(ProbeFailed, match="with nothing at all"):
        run(probe(FILM, runner=runner))


def test_output_that_is_not_ffprobe_json_is_reported_as_unreadable() -> None:
    runner, _ = runner_returning(0, "this is not json, nor is it mkvinfo")

    with pytest.raises(ProbeFailed, match="could not be read"):
        run(probe(FILM, runner=runner))


def test_json_without_streams_is_reported_as_unreadable() -> None:
    runner, _ = runner_returning(0, '{"note": "valid json, wrong shape"}')

    with pytest.raises(ProbeFailed, match="could not be read"):
        run(probe(FILM, runner=runner))


# --- the real subprocess ----------------------------------------------------


def test_a_missing_binary_says_how_to_get_one() -> None:
    with pytest.raises(ProbeFailed) as raised:
        run(run_ffprobe(["graindamage-no-such-binary", "-version"], 5.0))

    message = str(raised.value)
    assert "was not found" in message
    assert "Install ffmpeg" in message and "--ffprobe" in message


@pytest.mark.skipif(shutil.which("sleep") is None, reason="needs /bin/sleep")
def test_a_probe_that_never_finishes_is_killed() -> None:
    sleep = shutil.which("sleep")
    assert sleep is not None

    with pytest.raises(ProbeFailed, match=r"did not finish within 0\.05s"):
        run(run_ffprobe([sleep, "10"], 0.05))
