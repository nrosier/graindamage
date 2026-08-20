"""Number and unit parsing shared by the three source parsers.

The tools disagree about everything: ffprobe prints ``"24000/1001"`` and raw seconds,
mkvinfo prints ``00:42:39.573000000``, MediaInfo prints ``2 h 8 min`` and separates
thousands with a non-breaking space. Rather than three sets of near-identical regexes,
each quantity gets one tolerant parser here.

All of these return ``None`` rather than raising: a source paste is user input, and a
field we cannot read should degrade to "unknown" and a warning, never to a 500.
"""

from __future__ import annotations

import re
from fractions import Fraction

# MediaInfo groups digits with NBSP / narrow NBSP / thin space depending on locale.
# Spelled as escapes because these characters are invisible in an editor: NBSP,
# narrow NBSP, thin, figure and hair space.
_SPACES = dict.fromkeys(map(ord, "\u00a0\u202f\u2009\u2007\u200a"), " ")

_NOT_AVAILABLE = {"", "n/a", "na", "none", "unknown", "unspecified", "-", "--"}

_BITRATE_UNITS: dict[str, float] = {
    "b/s": 1.0,
    "bps": 1.0,
    "bit/s": 1.0,
    "kb/s": 1_000.0,
    "kbps": 1_000.0,
    "kbit/s": 1_000.0,
    "mb/s": 1_000_000.0,
    "mbps": 1_000_000.0,
    "mbit/s": 1_000_000.0,
    "gb/s": 1_000_000_000.0,
}

_SIZE_UNITS: dict[str, float] = {
    "b": 1.0,
    "byte": 1.0,
    "bytes": 1.0,
    "kb": 1_000.0,
    "kib": 1_024.0,
    "mb": 1_000_000.0,
    "mib": 1_048_576.0,
    "gb": 1_000_000_000.0,
    "gib": 1_073_741_824.0,
    "tb": 1_000_000_000_000.0,
    "tib": 1_099_511_627_776.0,
}

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# A thousands separator: a space or comma sitting between a digit and a group of
# exactly three more. MediaInfo writes every large number this way ("1 920 pixels",
# "1 411.2 kb/s"), and without this a 1920-pixel width parses as 1.
_GROUPED = re.compile(r"(?<=\d)[ ,](?=\d{3}(?:\D|$))")
_CLOCK = re.compile(r"(\d+):([0-5]?\d):([0-5]?\d(?:\.\d+)?)")
_HUMAN_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*(h|hour|hours|min|mn|s|sec|ms)\b", re.IGNORECASE)
_HUMAN_UNITS: dict[str, float] = {
    "h": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "min": 60.0,
    "mn": 60.0,
    "s": 1.0,
    "sec": 1.0,
    "ms": 0.001,
}


def clean(value: object) -> str:
    """Collapse odd whitespace and trim — the first step for every text field."""
    if value is None:
        return ""
    text = str(value).translate(_SPACES)
    return " ".join(text.split())


def ungroup(text: str) -> str:
    """Remove thousands separators so the number survives ``_NUMBER``.

    Applied to numeric fields only, never to labels: it is deliberately narrow so
    that ``2 h 8 min`` and ``4:2:0`` pass through untouched.
    """
    return _GROUPED.sub("", text)


def is_missing(value: object) -> bool:
    return clean(value).casefold() in _NOT_AVAILABLE


def parse_float(value: object) -> float | None:
    """First number in the text, or ``None``. Handles ``a/b`` fractions."""
    text = clean(value)
    if is_missing(text):
        return None

    if "/" in text:
        left, _, right = text.partition("/")
        try:
            return float(Fraction(left.strip()) / Fraction(right.strip()))
        except (ValueError, ZeroDivisionError, ArithmeticError):
            pass

    match = _NUMBER.search(ungroup(text))
    return float(match.group()) if match else None


def parse_int(value: object) -> int | None:
    number = parse_float(value)
    return round(number) if number is not None else None


def parse_positive_int(value: object) -> int | None:
    number = parse_int(value)
    return number if number is not None and number > 0 else None


def parse_frame_rate(value: object) -> float | None:
    """``23.976 (24000/1001) FPS``, ``24000/1001`` or ``23.976`` — all fine.

    The parenthesised exact ratio is preferred when present: it is what tells
    24000/1001 apart from a rounded 23.98.
    """
    text = clean(value)
    if is_missing(text):
        return None

    ratio = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if ratio:
        numerator, denominator = int(ratio.group(1)), int(ratio.group(2))
        if denominator and numerator:
            return numerator / denominator

    rate = parse_float(text)
    return rate if rate and rate > 0 else None


def parse_bitrate(value: object) -> int | None:
    """Bits per second from ``30.0 Mb/s``, ``1 411.2 kb/s`` or a bare integer."""
    text = clean(value)
    if is_missing(text):
        return None

    number = parse_float(text)
    if number is None:
        return None

    unit_match = re.search(r"(g|m|k)?b(?:it)?(?:/s|ps)", text, re.IGNORECASE)
    multiplier = _BITRATE_UNITS.get(unit_match.group().casefold(), 1.0) if unit_match else 1.0
    result = round(number * multiplier)
    return result if result > 0 else None


def parse_size(value: object) -> int | None:
    """Bytes from ``30.0 GiB``, ``1.36 MiB`` or a bare integer."""
    text = clean(value)
    if is_missing(text):
        return None

    number = parse_float(text)
    if number is None:
        return None

    unit_match = re.search(r"\b([kmgt]i?b|bytes?)\b", text, re.IGNORECASE)
    multiplier = _SIZE_UNITS.get(unit_match.group(1).casefold(), 1.0) if unit_match else 1.0
    result = round(number * multiplier)
    return result if result > 0 else None


def parse_duration(value: object) -> float | None:
    """Seconds from a clock string, a human string, or bare seconds.

    ``00:42:39.573000000`` · ``2559.573s (00:42:39.573)`` · ``2 h 8 min`` ·
    ``128.043``. mkvinfo's nanosecond-padded clock and MediaInfo's prose both land
    here, so the clock form is tried first and prose second.
    """
    text = clean(value)
    if is_missing(text):
        return None

    clock = _CLOCK.search(text)
    if clock:
        hours, minutes, seconds = clock.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    human = _HUMAN_DURATION.findall(text)
    if human:
        total = sum(float(amount) * _HUMAN_UNITS[unit.casefold()] for amount, unit in human)
        if total > 0:
            return total

    seconds = parse_float(text)
    return seconds if seconds and seconds > 0 else None


def parse_bool(value: object) -> bool:
    return clean(value).casefold() in {"1", "true", "yes", "y", "on"}


def normalise_codec(value: object) -> str | None:
    """Reduce a codec name or Matroska/MediaInfo codec id to one canonical token.

    The rules engine only ever asks coarse questions of this — "is the source
    already AV1?", "is it an intra-frame master?" — so the goal is a stable token,
    not a faithful reproduction of the tool's spelling.
    """
    text = clean(value).casefold()
    if not text:
        return None

    table: tuple[tuple[tuple[str, ...], str], ...] = (
        (("v_av1", "av01", "av1"), "av1"),
        (("v_mpegh/iso/hevc", "hevc", "h.265", "h265", "x265", "hvc1", "hev1", "dvhe"), "hevc"),
        (("v_mpeg4/iso/avc", "avc", "h.264", "h264", "x264", "avc1"), "h264"),
        (("v_mpeg2", "mpeg-2 video", "mpeg2video", "mpeg2"), "mpeg2video"),
        (("v_ms/vfw/fourcc", "vc-1", "vc1", "wvc1"), "vc1"),
        (("v_vp9", "vp9"), "vp9"),
        (("prores", "apch", "apcn"), "prores"),
        (("dnxhd", "dnxhr", "vc-3"), "dnxhd"),
        (("ffv1",), "ffv1"),
        (("v_uncompressed", "rawvideo", "v210"), "rawvideo"),
        (("a_truehd", "truehd", "mlp fba"), "truehd"),
        # DTS-HD MA is lossless and DTS core is not, so they must stay distinct.
        (("dts-hd ma", "dts-hd master", "dts xll", "dts_hd_ma"), "dts-hd ma"),
        (("a_dts", "dts-hd", "dts"), "dts"),
        # MediaInfo reports a "Commercial name" in preference to a format id, so the
        # marketing spellings have to be here too — and "plus" must be tested first.
        (("a_eac3", "e-ac-3", "eac3", "ec-3", "dolby digital plus"), "eac3"),
        (("a_ac3", "ac-3", "ac3", "dolby digital"), "ac3"),
        (("a_flac", "flac"), "flac"),
        (("a_opus", "opus"), "opus"),
        (("a_aac", "aac"), "aac"),
        (("a_pcm", "pcm"), "pcm"),
        (("s_text/utf8", "subrip", "srt"), "subrip"),
        (("s_text/ass", "s_text/ssa", "ass", "ssa"), "ass"),
        (("s_hdmv/pgs", "pgs"), "pgs"),
        (("s_vobsub", "vobsub", "dvd_subtitle"), "dvdsub"),
    )
    for needles, canonical in table:
        if any(needle in text for needle in needles):
            return canonical
    return text
