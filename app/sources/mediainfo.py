"""Parse MediaInfo's text output (the default ``mediainfo file.mkv`` report).

MediaInfo is the best of the three for a human to read and the most forgiving to
paste: it works on any container, always states bit depth and chroma subsampling, and
names HDR characteristics in plain language (``PQ``, ``BT.2020``) which
:mod:`app.sources.colors` maps back to FFmpeg's spelling.

Its cost is precision. Numbers are formatted for people — ``3 840 pixels`` with a
non-breaking space, ``2 h 8 min`` with the seconds thrown away, ``30.0 Mb/s`` rounded
to three digits — so durations and bitrates come back slightly approximate. That is
harmless for choosing a CRF and is noted in the report.

The format is ``Section`` headers followed by ``Label  : value`` rows. Sections repeat
per track (``Audio #1``, ``Audio #2``), and the label column position varies with the
longest label in the file, so rows are split on the first colon rather than a column.
"""

from __future__ import annotations

import re

from app.models import (
    AudioTrack,
    SourceMedia,
    SourceReport,
    SourceTool,
    SubtitleTrack,
    VideoTrack,
)
from app.sources.colors import (
    mastering_display_from_named_primaries,
    normalise_matrix,
    normalise_primaries,
    normalise_range,
    normalise_transfer,
    pix_fmt_details,
)
from app.sources.parsing import (
    clean,
    normalise_codec,
    parse_bitrate,
    parse_duration,
    parse_float,
    parse_frame_rate,
    parse_positive_int,
    parse_size,
)

_SECTION_KINDS = ("general", "video", "audio", "text", "menu", "image", "other")

# "Mastering display luminance : min: 0.0001 cd/m2, max: 1000 cd/m2"
_LUMINANCE = re.compile(
    r"min\s*:\s*(?P<min>[\d.]+)\s*cd/m2?.*?max\s*:\s*(?P<max>[\d.]+)\s*cd/m2?",
    re.IGNORECASE | re.DOTALL,
)


class Section:
    """One ``General`` / ``Video`` / ``Audio`` block, as an ordered label map."""

    def __init__(self, kind: str, ordinal: int) -> None:
        self.kind = kind
        self.ordinal = ordinal
        self.rows: dict[str, str] = {}

    def add(self, label: str, value: str) -> None:
        # Duplicate labels happen ("Duration" appears formatted and raw); the first
        # is MediaInfo's preferred rendering, so it wins.
        self.rows.setdefault(label.casefold(), value)

    def get(self, *labels: str) -> str | None:
        for label in labels:
            value = self.rows.get(label.casefold())
            if value:
                return value
        return None

    def has_any(self, *needles: str) -> bool:
        haystack = " ".join(self.rows.values()).casefold()
        return any(needle.casefold() in haystack for needle in needles)


def looks_like_mediainfo(text: str) -> bool:
    lines = [line.rstrip() for line in text.splitlines()]
    has_section = any(
        line.strip().casefold().split("#")[0].strip() in _SECTION_KINDS and ":" not in line
        for line in lines
        if line.strip()
    )
    # The label/value separator is a colon with whitespace on its left, which is
    # what distinguishes MediaInfo rows from mkvinfo's "+ Label: value".
    has_rows = sum(bool(re.match(r"^[A-Za-z][\w /()'.*-]{2,}\s{2,}:\s", line)) for line in lines)
    return has_section and has_rows >= 3


def _split_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    current: Section | None = None
    counts: dict[str, int] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # A section header is a bare kind, optionally numbered: "Video", "Audio #2".
        header = clean(line).split("#")[0].strip().casefold()
        if ":" not in line and header in _SECTION_KINDS:
            counts[header] = counts.get(header, 0) + 1
            current = Section(header, counts[header])
            sections.append(current)
            continue

        label, separator, value = line.partition(":")
        if separator and label.strip() and current is not None:
            current.add(clean(label), clean(value))

    return sections


def _video_track(section: Section) -> VideoTrack:
    pix_fmt = None
    bit_depth = parse_positive_int(section.get("Bit depth"))
    chroma = section.get("Chroma subsampling")
    if chroma:
        chroma = clean(chroma).split()[0]  # "4:2:0 (Type 2)" -> "4:2:0"

    # MediaInfo has no pixel-format field; synthesise the FFmpeg name it implies so
    # downstream code has one to show.
    if bit_depth and chroma:
        suffix = "" if bit_depth == 8 else f"{bit_depth}le"
        pix_fmt = f"yuv{chroma.replace(':', '')}p{suffix}"
        if pix_fmt_details(pix_fmt) == (None, None):
            pix_fmt = None

    scan = section.get("Scan type")
    track = VideoTrack(
        index=section.ordinal,
        codec=normalise_codec(section.get("Format", "Codec ID")),
        profile=section.get("Format profile"),
        width=parse_positive_int(section.get("Width")),
        height=parse_positive_int(section.get("Height")),
        display_aspect_ratio=section.get("Display aspect ratio"),
        frame_rate=parse_frame_rate(
            section.get("Frame rate", "Original frame rate", "Nominal frame rate")
        ),
        frame_rate_mode=(section.get("Frame rate mode") or "").casefold() or None,
        bit_depth=bit_depth,
        chroma_subsampling=chroma,
        pix_fmt=pix_fmt,
        scan_type=clean(scan).casefold() if scan else None,
        bitrate_bps=parse_bitrate(section.get("Bit rate", "Nominal bit rate", "Maximum bit rate")),
    )

    if primaries := section.get("Color primaries", "Colour primaries"):
        track.color_primaries = normalise_primaries(primaries)
    if transfer := section.get("Transfer characteristics"):
        track.color_transfer = normalise_transfer(transfer)
    if matrix := section.get("Matrix coefficients"):
        track.color_matrix = normalise_matrix(matrix)
    if color_range := section.get("Color range", "Colour range"):
        track.color_range = normalise_range(color_range)

    track.max_cll = parse_positive_int(section.get("Maximum Content Light Level", "MaxCLL"))
    track.max_fall = parse_positive_int(section.get("Maximum Frame-Average Light Level", "MaxFALL"))

    # Mastering display primaries arrive as a name, not coordinates, so the standard
    # xy values for that set are substituted and the luminance read from its own row.
    mastering_primaries = section.get(
        "Mastering display color primaries", "Mastering display colour primaries"
    )
    luminance = section.get("Mastering display luminance")
    if mastering_primaries and luminance:
        canonical = normalise_primaries(mastering_primaries)
        bounds = _LUMINANCE.search(luminance)
        if canonical and bounds:
            track.mastering_display = mastering_display_from_named_primaries(
                canonical,
                max_luminance=float(bounds.group("max")),
                min_luminance=float(bounds.group("min")),
            )

    hdr_format = section.get("HDR format") or ""
    track.dolby_vision = "dolby vision" in hdr_format.casefold()
    track.hdr10_plus = "hdr10+" in hdr_format.casefold()

    return track


def _audio_track(section: Section) -> AudioTrack:
    channels = parse_positive_int(section.get("Channel(s)", "Channels"))
    sample_rate = parse_float(section.get("Sampling rate"))
    # MediaInfo prints "48.0 kHz"; parse_float sees 48.0, so scale by the unit.
    if (
        sample_rate
        and sample_rate < 1000
        and "khz" in (section.get("Sampling rate") or "").casefold()
    ):
        sample_rate *= 1000

    return AudioTrack(
        index=section.ordinal,
        codec=normalise_codec(
            section.get("Commercial name", "Format profile", "Format", "Codec ID")
        ),
        channels=channels,
        channel_layout=section.get("Channel layout"),
        sample_rate=int(sample_rate) if sample_rate else None,
        bitrate_bps=parse_bitrate(section.get("Bit rate", "Nominal bit rate")),
        language=section.get("Language"),
        title=section.get("Title"),
        default=(section.get("Default") or "Yes").casefold().startswith("y"),
    )


def _subtitle_track(section: Section) -> SubtitleTrack:
    return SubtitleTrack(
        index=section.ordinal,
        # Codec ID first here, unlike video: a text subtitle's Format is the character
        # encoding ("UTF-8"), and only S_TEXT/UTF8 identifies it as SRT.
        codec=normalise_codec(section.get("Codec ID", "Format")),
        language=section.get("Language"),
        title=section.get("Title"),
        forced=(section.get("Forced") or "No").casefold().startswith("y"),
    )


def parse(text: str) -> SourceReport:
    """Build a :class:`SourceReport` from MediaInfo text output.

    Raises:
        ValueError: no MediaInfo sections could be recognised.
    """
    sections = _split_sections(text)
    if not sections:
        raise ValueError(
            "No MediaInfo sections found — expected a 'General' or 'Video' header "
            "followed by 'Label  : value' rows."
        )

    general = next((s for s in sections if s.kind == "general"), None)
    warnings: list[str] = []

    media = SourceMedia(
        container=general.get("Format") if general else None,
        duration_seconds=parse_duration(general.get("Duration")) if general else None,
        size_bytes=parse_size(general.get("File size")) if general else None,
        overall_bitrate_bps=parse_bitrate(general.get("Overall bit rate")) if general else None,
    )

    for section in sections:
        if section.kind == "video" and media.video is None:
            media.video = _video_track(section)
        elif section.kind == "audio":
            media.audio.append(_audio_track(section))
        elif section.kind == "text":
            media.subtitles.append(_subtitle_track(section))

    if media.video is None:
        warnings.append("No Video section found in that MediaInfo report.")
    else:
        if media.duration_seconds is None:
            warnings.append("No duration reported, so no size estimate can be given.")
        if media.video.bitrate_bps is None and media.overall_bitrate_bps is None:
            warnings.append(
                "No bitrate reported, so the source's quality cannot be judged and "
                "the CRF stays conservative."
            )
        if media.video.bit_depth is None:
            warnings.append("No bit depth reported; set it below if you know it.")

    if media.duration_seconds is not None:
        warnings.append(
            "MediaInfo rounds durations and bitrates for readability, so the size "
            "estimate is approximate. ffprobe gives exact figures."
        )

    return SourceReport(tool=SourceTool.MEDIAINFO, media=media, warnings=warnings)
