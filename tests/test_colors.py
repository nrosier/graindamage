"""Colour-signalling translation: three input spellings, two encoder spellings."""

from __future__ import annotations

import pytest

from app.sources.colors import (
    CODE_BY_MATRIX,
    CODE_BY_PRIMARIES,
    CODE_BY_TRANSFER,
    MATRIX_BY_CODE,
    PRIMARIES_BY_CODE,
    RANGE_BY_CODE,
    TRANSFER_BY_CODE,
    chroma_from_subsampling_pair,
    mastering_display_from_named_primaries,
    normalise_matrix,
    normalise_primaries,
    normalise_range,
    normalise_transfer,
    pix_fmt_details,
)


def test_code_tables_are_exact_inverses() -> None:
    # SVT-AV1 is handed CODE_BY_*; x265 is handed the names. A one-way drift here
    # would signal HDR to one encoder and not the other.
    assert {code: name for name, code in CODE_BY_PRIMARIES.items()} == PRIMARIES_BY_CODE
    assert {code: name for name, code in CODE_BY_TRANSFER.items()} == TRANSFER_BY_CODE
    assert {code: name for name, code in CODE_BY_MATRIX.items()} == MATRIX_BY_CODE


def test_hdr_and_wide_gamut_code_points() -> None:
    # The four that decide whether an encode is tagged HDR at all.
    assert PRIMARIES_BY_CODE[9] == "bt2020"
    assert TRANSFER_BY_CODE[16] == "smpte2084"
    assert TRANSFER_BY_CODE[18] == "arib-std-b67"
    assert MATRIX_BY_CODE[9] == "bt2020nc"


def test_matroska_range_codes() -> None:
    assert RANGE_BY_CODE == {1: "tv", 2: "pc"}


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("bt2020", "bt2020"),  # already an FFmpeg name (ffprobe)
        ("BT.2020", "bt2020"),  # MediaInfo label
        ("Display P3", "smpte432"),
        ("DCI P3", "smpte431"),
        ("nonsense", None),
    ],
)
def test_normalise_primaries(label: str, expected: str | None) -> None:
    assert normalise_primaries(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("smpte2084", "smpte2084"),
        ("PQ", "smpte2084"),
        ("HLG", "arib-std-b67"),
        ("BT.709", "bt709"),
        ("nonsense", None),
    ],
)
def test_normalise_transfer(label: str, expected: str | None) -> None:
    assert normalise_transfer(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("bt2020nc", "bt2020nc"),
        ("BT.2020 non-constant", "bt2020nc"),
        ("BT.2020 constant", "bt2020c"),
        ("nonsense", None),
    ],
)
def test_normalise_matrix(label: str, expected: str | None) -> None:
    assert normalise_matrix(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("tv", "tv"),
        ("Limited", "tv"),
        ("mpeg", "tv"),
        ("pc", "pc"),
        ("Full", "pc"),
        ("jpeg", "pc"),
        ("", None),
    ],
)
def test_normalise_range(label: str, expected: str | None) -> None:
    assert normalise_range(label) == expected


@pytest.mark.parametrize(
    ("pix_fmt", "expected"),
    [
        ("yuv420p", (8, "4:2:0")),
        ("yuv420p10le", (10, "4:2:0")),
        ("yuv422p10le", (10, "4:2:2")),
        ("yuv444p12le", (12, "4:4:4")),
        ("p010le", (10, "4:2:0")),
        # Not in the table: depth and subsampling come from the name's shape.
        ("yuv420p14le", (14, "4:2:0")),
        ("yuv410p", (8, "4:1:0")),
        ("rgb24", (None, None)),
    ],
)
def test_pix_fmt_details(pix_fmt: str, expected: tuple[int | None, str | None]) -> None:
    assert pix_fmt_details(pix_fmt) == expected


@pytest.mark.parametrize(
    ("pair", "expected"),
    [((1, 1), "4:2:0"), ((1, 0), "4:2:2"), ((0, 0), "4:4:4"), ((2, 0), "4:1:1"), ((3, 3), None)],
)
def test_chroma_from_subsampling_pair(pair: tuple[int, int], expected: str | None) -> None:
    assert chroma_from_subsampling_pair(*pair) == expected


def test_mastering_display_from_named_primaries() -> None:
    """MediaInfo names the primary set and gives only the luminance numerically."""
    display = mastering_display_from_named_primaries(
        "bt2020", max_luminance=1000.0, min_luminance=0.0001
    )

    assert display is not None
    assert (display.red_x, display.red_y) == (0.708, 0.292)
    assert (display.white_x, display.white_y) == (0.3127, 0.3290)
    assert display.max_luminance == 1000.0
    assert display.min_luminance == 0.0001


def test_mastering_display_from_unknown_primaries_is_none() -> None:
    # Better no metadata than invented coordinates: a wrong master-display string
    # makes a display tone-map to the wrong volume.
    assert (
        mastering_display_from_named_primaries("film", max_luminance=1000.0, min_luminance=0.0001)
        is None
    )
