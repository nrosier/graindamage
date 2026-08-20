"""The tolerant number parsers in :mod:`app.sources.parsing`."""

from __future__ import annotations

import pytest

from app.sources.parsing import (
    clean,
    is_missing,
    normalise_codec,
    parse_bitrate,
    parse_bool,
    parse_duration,
    parse_float,
    parse_frame_rate,
    parse_int,
    parse_positive_int,
    parse_size,
    ungroup,
)


def test_clean_collapses_exotic_whitespace() -> None:
    assert clean("  Dolby\u00a0Digital\tPlus \n") == "Dolby Digital Plus"
    assert clean(None) == ""
    assert clean(1920) == "1920"


def test_ungroup_removes_thousands_separators_only() -> None:
    assert ungroup("1 920") == "1920"
    assert ungroup("3,196") == "3196"
    # Not a thousands separator: the group is not three digits long.
    assert ungroup("2 h 8 min") == "2 h 8 min"
    assert ungroup("4:2:0") == "4:2:0"


def test_parse_float_survives_grouped_digits() -> None:
    # The bug this guards: "1 920 pixels" parsed as 1.0, so every MediaInfo width
    # became one pixel wide and the resolution class was always SD.
    assert parse_float("1 920 pixels") == 1920.0
    assert parse_float("1 411.2 kb/s") == 1411.2


# MediaInfo picks its digit separator by locale: NBSP, narrow NBSP or thin space. The
# fixture files all use a plain space, so these three are only covered here.
@pytest.mark.parametrize("space", ["\u00a0", "\u202f", "\u2009"])
def test_parse_float_survives_unicode_group_separators(space: str) -> None:
    assert parse_float(f"1{space}920 pixels") == 1920.0
    assert parse_bitrate(f"3{space}502 kb/s") == 3_502_000


def test_parse_float_handles_fractions_and_missing() -> None:
    assert parse_float("24000/1001") == pytest.approx(23.976, abs=1e-3)
    assert parse_float("34000/50000") == 0.68
    assert parse_float("1/0") == 1.0  # division by zero falls through to the number
    assert parse_float("N/A") is None
    assert parse_float("") is None
    assert parse_float(None) is None


def test_parse_int_rounds_and_positive_int_rejects_zero() -> None:
    assert parse_int("23.976") == 24
    assert parse_int("-3") == -3
    assert parse_positive_int("0") is None
    assert parse_positive_int("-3") is None
    assert parse_positive_int("10") == 10


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("23.976 (24000/1001) FPS", 24000 / 1001),
        ("24000/1001", 24000 / 1001),
        ("23.976", 23.976),
        ("0", None),
        ("unknown", None),
    ],
)
def test_parse_frame_rate(text: str, expected: float | None) -> None:
    result = parse_frame_rate(text)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("58120000", 58_120_000),
        ("30.0 Mb/s", 30_000_000),
        ("3 502 kb/s", 3_502_000),
        ("1 411.2 kb/s", 1_411_200),
        ("640 kbps", 640_000),
        ("0", None),
        ("N/A", None),
    ],
)
def test_parse_bitrate(text: str, expected: int | None) -> None:
    assert parse_bitrate(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("56000000000", 56_000_000_000),
        ("24.5 GiB", 26_306_674_688),
        ("1.36 MiB", 1_426_063),
        ("500 MB", 500_000_000),
        ("2 048 bytes", 2048),
        ("-", None),
    ],
)
def test_parse_size(text: str, expected: int | None) -> None:
    assert parse_size(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("00:42:39.573000000", 2559.573),
        ("02:01:01.000000000", 7261.0),
        ("2559.573s (00:42:39.573)", 2559.573),
        ("1 h 57 min", 7020.0),
        ("2 h 8 min", 7680.0),
        ("142 min", 8520.0),
        ("128.043", 128.043),
        ("unspecified", None),
    ],
)
def test_parse_duration(text: str, expected: float | None) -> None:
    result = parse_duration(text)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_is_missing_and_parse_bool() -> None:
    assert is_missing("N/A")
    assert is_missing("  none  ")
    assert not is_missing("0")
    assert parse_bool("Yes") is True
    assert parse_bool("1") is True
    assert parse_bool("no") is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("V_MPEGH/ISO/HEVC", "hevc"),
        ("hevc", "hevc"),
        ("V_MPEG4/ISO/AVC", "h264"),
        ("AVC", "h264"),
        ("V_AV1", "av1"),
        ("A_TRUEHD", "truehd"),
        ("DTS-HD Master Audio", "dts-hd ma"),
        ("DTS", "dts"),
        # "plus" has to be matched before the "Dolby Digital" it contains.
        ("Dolby Digital Plus", "eac3"),
        ("Dolby Digital", "ac3"),
        ("S_TEXT/UTF8", "subrip"),
        ("S_HDMV/PGS", "pgs"),
        ("", None),
        (None, None),
    ],
)
def test_normalise_codec(value: str | None, expected: str | None) -> None:
    assert normalise_codec(value) == expected


def test_normalise_codec_passes_unknown_names_through_folded() -> None:
    assert normalise_codec("Some Future Codec") == "some future codec"
