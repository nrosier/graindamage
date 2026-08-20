"""Format sniffing: users paste whatever their tooling produced, unlabelled."""

from __future__ import annotations

import pytest

from app.models import SourceTool
from app.sources import (
    MAX_INPUT_CHARS,
    PARSERS,
    UnknownSourceFormat,
    detect,
    parse_source,
)
from tests.support import fixture

FIXTURE_TOOLS = [
    ("ffprobe_uhd_hdr.json", SourceTool.FFPROBE),
    ("mkvinfo_uhd_hdr.txt", SourceTool.MKVINFO),
    ("mkvinfo_sparse.txt", SourceTool.MKVINFO),
    ("mediainfo_1080p_film.txt", SourceTool.MEDIAINFO),
]


@pytest.mark.parametrize(("name", "tool"), FIXTURE_TOOLS)
def test_each_fixture_detects_as_its_own_tool(name: str, tool: SourceTool) -> None:
    assert detect(fixture(name)) is tool


@pytest.mark.parametrize(("name", "tool"), FIXTURE_TOOLS)
def test_parse_source_dispatches_and_records_the_tool(name: str, tool: SourceTool) -> None:
    report = parse_source(fixture(name))

    assert report.tool is tool
    assert report.has_video


def test_every_tool_has_a_parser() -> None:
    # A new SourceTool without an entry here would KeyError at request time.
    assert set(PARSERS) == set(SourceTool)


def test_leading_blank_lines_and_indentation_do_not_confuse_detection() -> None:
    padded = "\n\n   " + fixture("ffprobe_uhd_hdr.json")
    assert detect(padded) is SourceTool.FFPROBE


def test_an_explicit_tool_overrides_the_sniff() -> None:
    # The UI's manual override: MediaInfo text forced through the MediaInfo parser
    # even in the (hypothetical) case where a sniff would disagree.
    report = parse_source(fixture("mediainfo_1080p_film.txt"), tool=SourceTool.MEDIAINFO)
    assert report.tool is SourceTool.MEDIAINFO


def test_forcing_the_wrong_parser_fails_loudly() -> None:
    with pytest.raises(ValueError):
        parse_source(fixture("mediainfo_1080p_film.txt"), tool=SourceTool.FFPROBE)


@pytest.mark.parametrize("text", ["", "   \n\t ", "hello", "some notes about my film"])
def test_unrecognised_text_is_refused_with_the_commands_to_run(text: str) -> None:
    with pytest.raises(UnknownSourceFormat) as caught:
        parse_source(text)

    # The error is the only instruction some users will read, so it carries the
    # exact commands rather than just naming the three tools.
    message = str(caught.value)
    assert "-show_streams" in message
    assert "mkvinfo FILE" in message
    assert "mediainfo FILE" in message


def test_detect_returns_none_rather_than_raising() -> None:
    assert detect("") is None
    assert detect("hello") is None


def test_an_oversized_paste_is_refused_before_parsing() -> None:
    # -show_frames on a feature film is tens of megabytes; refusing it is cheaper
    # than letting json.loads chew through it.
    with pytest.raises(UnknownSourceFormat, match="the limit is"):
        parse_source('{"streams": [' + " " * MAX_INPUT_CHARS + "]}")
