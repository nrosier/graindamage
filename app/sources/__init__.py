"""Source-file description parsers: ffprobe, mkvinfo and MediaInfo.

Users paste whatever their tooling produced, so nothing asks them which format it is.
Detection is by structure, not by a dropdown:

* ffprobe — JSON, so it is unambiguous from the first character.
* mkvinfo — an EBML tree of ``|  + Label: value`` lines.
* MediaInfo — ``Section`` headers over ``Label  : value`` rows.

All three converge on :class:`~app.models.SourceReport`, whose ``warnings`` list is the
honest part: the tools do not report the same fields, and the rules engine has to know
which numbers it was actually given. See each module for what its tool cannot tell us.
"""

from __future__ import annotations

from collections.abc import Callable

from app.models import SourceReport, SourceTool
from app.sources import ffprobe, mediainfo, mkvinfo

Parser = Callable[[str], SourceReport]

PARSERS: dict[SourceTool, Parser] = {
    SourceTool.FFPROBE: ffprobe.parse,
    SourceTool.MKVINFO: mkvinfo.parse,
    SourceTool.MEDIAINFO: mediainfo.parse,
}

# Ordered by how decisive the sniff is: JSON first, then the two text formats.
_DETECTORS: tuple[tuple[SourceTool, Callable[[str], bool]], ...] = (
    (SourceTool.FFPROBE, ffprobe.looks_like_ffprobe),
    (SourceTool.MKVINFO, mkvinfo.looks_like_mkvinfo),
    (SourceTool.MEDIAINFO, mediainfo.looks_like_mediainfo),
)

MAX_INPUT_CHARS = 512_000

_HELP = (
    "Paste one of: ffprobe JSON (ffprobe -v quiet -print_format json -show_format "
    "-show_streams FILE), mkvinfo output (mkvinfo FILE), or a MediaInfo report "
    "(mediainfo FILE)."
)


class UnknownSourceFormat(ValueError):
    """The paste matched none of the three formats."""


def detect(text: str) -> SourceTool | None:
    """Which tool produced this text, or ``None`` if nothing matches."""
    if not text.strip():
        return None
    return next((tool for tool, matches in _DETECTORS if matches(text)), None)


def parse_source(text: str, *, tool: SourceTool | None = None) -> SourceReport:
    """Parse pasted tool output into a :class:`SourceReport`.

    Args:
        text: The pasted output.
        tool: Force a parser instead of sniffing — used by tests and by the UI's
            manual override when a paste is ambiguous.

    Raises:
        UnknownSourceFormat: The text is empty, too large, or unrecognised.
        ValueError: The format was recognised but the content was unusable; the
            message names what was missing.
    """
    if not text.strip():
        raise UnknownSourceFormat("Nothing pasted. " + _HELP)
    if len(text) > MAX_INPUT_CHARS:
        raise UnknownSourceFormat(
            f"That paste is {len(text) // 1000} kB; the limit is "
            f"{MAX_INPUT_CHARS // 1000} kB. Drop -show_frames / -show_packets and "
            "paste just the streams and format sections."
        )

    chosen = tool or detect(text)
    if chosen is None:
        raise UnknownSourceFormat("Could not tell what produced that text. " + _HELP)

    return PARSERS[chosen](text)


__all__ = [
    "MAX_INPUT_CHARS",
    "PARSERS",
    "UnknownSourceFormat",
    "detect",
    "parse_source",
]
