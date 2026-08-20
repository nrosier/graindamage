"""Colour-signalling translation between the three input tools and two encoders.

Four spellings of the same facts are in play:

* **ffprobe** emits H.273 *names* the way FFmpeg spells them (``bt2020nc``).
* **mkvinfo** emits the raw H.273 *integers* stored in the Matroska ``Colour`` element.
* **MediaInfo** emits human labels (``BT.2020 non-constant``, ``PQ``).
* **x265** takes FFmpeg's names; **SVT-AV1** takes the integers.

So FFmpeg's names are the canonical internal form, and everything converts to or
from them here rather than in each parser.
"""

from __future__ import annotations

from app.models import MasteringDisplay

# --- H.273 integers <-> FFmpeg names ----------------------------------------

PRIMARIES_BY_CODE: dict[int, str] = {
    1: "bt709",
    4: "bt470m",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "film",
    9: "bt2020",
    10: "smpte428",
    11: "smpte431",
    12: "smpte432",
    22: "jedec-p22",
}

TRANSFER_BY_CODE: dict[int, str] = {
    1: "bt709",
    4: "bt470m",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "linear",
    9: "log100",
    10: "log316",
    11: "iec61966-2-4",
    12: "bt1361e",
    13: "iec61966-2-1",
    14: "bt2020-10",
    15: "bt2020-12",
    16: "smpte2084",
    17: "smpte428",
    18: "arib-std-b67",
}

MATRIX_BY_CODE: dict[int, str] = {
    0: "gbr",
    1: "bt709",
    4: "fcc",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "ycgco",
    9: "bt2020nc",
    10: "bt2020c",
    11: "smpte2085",
    12: "chroma-derived-nc",
    13: "chroma-derived-c",
    14: "ictcp",
}

# Matroska ColourRange: 1 = broadcast (limited), 2 = full.
RANGE_BY_CODE: dict[int, str] = {1: "tv", 2: "pc"}

CODE_BY_PRIMARIES: dict[str, int] = {name: code for code, name in PRIMARIES_BY_CODE.items()}
CODE_BY_TRANSFER: dict[str, int] = {name: code for code, name in TRANSFER_BY_CODE.items()}
CODE_BY_MATRIX: dict[str, int] = {name: code for code, name in MATRIX_BY_CODE.items()}

# --- MediaInfo labels -> FFmpeg names ---------------------------------------

_MEDIAINFO_PRIMARIES: dict[str, str] = {
    "bt.709": "bt709",
    "bt.601 ntsc": "smpte170m",
    "bt.601 pal": "bt470bg",
    "bt.470 system m": "bt470m",
    "bt.470 system b/g": "bt470bg",
    "bt.2020": "bt2020",
    "smpte 240m": "smpte240m",
    "smpte 428m": "smpte428",
    "dci p3": "smpte431",
    "display p3": "smpte432",
    "p3-dci": "smpte431",
    "p3-d65": "smpte432",
    "generic film": "film",
}

_MEDIAINFO_TRANSFER: dict[str, str] = {
    "bt.709": "bt709",
    "bt.601": "smpte170m",
    "bt.470 system m": "bt470m",
    "bt.470 system b/g": "bt470bg",
    "smpte 240m": "smpte240m",
    "smpte 428m": "smpte428",
    "linear": "linear",
    "pq": "smpte2084",
    "hlg": "arib-std-b67",
    "srgb/sycc": "iec61966-2-1",
    "xvycc": "iec61966-2-4",
    "bt.2020 (10-bit)": "bt2020-10",
    "bt.2020 (12-bit)": "bt2020-12",
}

_MEDIAINFO_MATRIX: dict[str, str] = {
    "bt.709": "bt709",
    "bt.601": "smpte170m",
    "bt.470 system b/g": "bt470bg",
    "bt.2020 non-constant": "bt2020nc",
    "bt.2020 constant": "bt2020c",
    "smpte 240m": "smpte240m",
    "fcc 73.682": "fcc",
    "ycgco": "ycgco",
    "identity": "gbr",
    "derived non-constant": "chroma-derived-nc",
    "derived constant": "chroma-derived-c",
}

_MEDIAINFO_RANGE: dict[str, str] = {"limited": "tv", "full": "pc"}

# Well-known primary sets, for the MediaInfo case where the mastering display is
# reported by name ("Display P3") and only the luminance is numeric.
NAMED_PRIMARY_SETS: dict[str, tuple[float, float, float, float, float, float, float, float]] = {
    # red_x, red_y, green_x, green_y, blue_x, blue_y, white_x, white_y
    "bt2020": (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290),
    "smpte431": (0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3140, 0.3510),
    "smpte432": (0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290),
    "bt709": (0.640, 0.330, 0.300, 0.600, 0.150, 0.060, 0.3127, 0.3290),
}


def _normalise_label(label: str) -> str:
    return " ".join(label.strip().casefold().split())


def primaries_from_label(label: str) -> str | None:
    return _MEDIAINFO_PRIMARIES.get(_normalise_label(label))


def transfer_from_label(label: str) -> str | None:
    return _MEDIAINFO_TRANSFER.get(_normalise_label(label))


def matrix_from_label(label: str) -> str | None:
    return _MEDIAINFO_MATRIX.get(_normalise_label(label))


def range_from_label(label: str) -> str | None:
    return _MEDIAINFO_RANGE.get(_normalise_label(label))


def normalise_primaries(value: str) -> str | None:
    """Accept either spelling — parsers get ffprobe names and MediaInfo labels."""
    lowered = value.strip().casefold()
    if lowered in CODE_BY_PRIMARIES:
        return lowered
    return primaries_from_label(value)


def normalise_transfer(value: str) -> str | None:
    lowered = value.strip().casefold()
    if lowered in CODE_BY_TRANSFER:
        return lowered
    return transfer_from_label(value)


def normalise_matrix(value: str) -> str | None:
    lowered = value.strip().casefold()
    if lowered in CODE_BY_MATRIX:
        return lowered
    return matrix_from_label(value)


def normalise_range(value: str) -> str | None:
    lowered = value.strip().casefold()
    if lowered in {"tv", "limited", "mpeg"}:
        return "tv"
    if lowered in {"pc", "full", "jpeg"}:
        return "pc"
    return None


def mastering_display_from_named_primaries(
    primaries: str,
    *,
    max_luminance: float,
    min_luminance: float,
) -> MasteringDisplay | None:
    """Rebuild ST 2086 metadata when only a primary-set *name* was reported."""
    coords = NAMED_PRIMARY_SETS.get(primaries)
    if coords is None:
        return None
    red_x, red_y, green_x, green_y, blue_x, blue_y, white_x, white_y = coords
    return MasteringDisplay(
        red_x=red_x,
        red_y=red_y,
        green_x=green_x,
        green_y=green_y,
        blue_x=blue_x,
        blue_y=blue_y,
        white_x=white_x,
        white_y=white_y,
        max_luminance=max_luminance,
        min_luminance=min_luminance,
    )


# --- pixel formats ----------------------------------------------------------

# FFmpeg pixel format -> (bit depth, chroma subsampling). Only the formats that
# turn up in real film sources are listed; anything else falls back to a regex.
PIX_FMTS: dict[str, tuple[int, str]] = {
    "yuv420p": (8, "4:2:0"),
    "yuvj420p": (8, "4:2:0"),
    "yuv420p10le": (10, "4:2:0"),
    "yuv420p12le": (12, "4:2:0"),
    "yuv422p": (8, "4:2:2"),
    "yuvj422p": (8, "4:2:2"),
    "yuv422p10le": (10, "4:2:2"),
    "yuv422p12le": (12, "4:2:2"),
    "yuv444p": (8, "4:4:4"),
    "yuv444p10le": (10, "4:4:4"),
    "yuv444p12le": (12, "4:4:4"),
    "p010le": (10, "4:2:0"),
    "gbrp": (8, "4:4:4"),
}


def pix_fmt_details(pix_fmt: str) -> tuple[int | None, str | None]:
    """Bit depth and chroma subsampling for a pixel format name."""
    known = PIX_FMTS.get(pix_fmt.strip().casefold())
    if known:
        return known

    lowered = pix_fmt.strip().casefold()
    depth: int | None = None
    for candidate in (16, 14, 12, 10, 9):
        if f"p{candidate}" in lowered or f"{candidate}le" in lowered:
            depth = candidate
            break
    else:
        if lowered.startswith(("yuv", "gbr", "gray")):
            depth = 8

    subsampling: str | None = None
    for marker, value in (("420", "4:2:0"), ("422", "4:2:2"), ("444", "4:4:4"), ("410", "4:1:0")):
        if marker in lowered:
            subsampling = value
            break
    return depth, subsampling


def chroma_from_subsampling_pair(horizontal: int, vertical: int) -> str | None:
    """Matroska stores chroma siting as horizontal/vertical subsampling factors."""
    return {
        (0, 0): "4:4:4",
        (1, 0): "4:2:2",
        (1, 1): "4:2:0",
        (2, 0): "4:1:1",
    }.get((horizontal, vertical))
