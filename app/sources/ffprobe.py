"""Parse ``ffprobe -show_format -show_streams -print_format json`` output.

This is the preferred input: it is the only one of the three tools that reports pixel
format (hence bit depth *and* chroma subsampling) for any container, and it carries
HDR10 mastering metadata as side data.

Accepted shapes: the full object with ``format`` and ``streams``, a bare
``{"streams": [...]}`` from a narrower ``-show_entries``, or a top-level list of
streams. Missing keys are tolerated; a paste with no video stream produces a warning
rather than an error, since audio-only advice is still refusable politely upstream.
"""

from __future__ import annotations

import json
from typing import Any

from app.models import (
    AudioTrack,
    MasteringDisplay,
    SourceMedia,
    SourceReport,
    SourceTool,
    SubtitleTrack,
    VideoTrack,
)
from app.sources.colors import (
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

# ffprobe reports interlacing through field_order; everything but "progressive"
# and "unknown" describes a field order, which means interlaced.
_INTERLACED_FIELD_ORDERS = {"tt", "bb", "tb", "bt", "interlaced"}


def looks_like_ffprobe(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        return False
    return '"streams"' in text or '"codec_type"' in text or '"format"' in text


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _tag(stream: dict[str, Any], *names: str) -> str | None:
    """Look a value up in ``tags``, case-insensitively.

    Matroska statistics tags (``BPS``, ``BPS-eng``, ``NUMBER_OF_BYTES``) are how a
    remux reports per-track bitrate, and their case varies by muxer version.
    """
    tags = {key.casefold(): value for key, value in _as_dict(stream.get("tags")).items()}
    for name in names:
        for key, value in tags.items():
            matches = key == name.casefold() or key.startswith(f"{name.casefold()}-")
            if matches and not isinstance(value, dict | list):
                return clean(value)
    return None


def _stream_bitrate(stream: dict[str, Any], duration: float | None) -> int | None:
    direct = parse_bitrate(stream.get("bit_rate"))
    if direct:
        return direct
    tagged = parse_bitrate(_tag(stream, "BPS"))
    if tagged:
        return tagged
    # Last resort: bytes / duration, which remuxes record even when BPS is absent.
    size = parse_size(_tag(stream, "NUMBER_OF_BYTES"))
    span = parse_duration(_tag(stream, "DURATION")) or duration
    if size and span:
        return int(size * 8 / span)
    return None


def _mastering_display(side_data: dict[str, Any]) -> MasteringDisplay | None:
    """ffprobe emits these as ``"34000/50000"`` fractions in most builds and as
    plain floats in some; :func:`parse_float` handles both."""
    fields = (
        "red_x",
        "red_y",
        "green_x",
        "green_y",
        "blue_x",
        "blue_y",
        "white_point_x",
        "white_point_y",
        "max_luminance",
        "min_luminance",
    )
    values: dict[str, float] = {}
    for field in fields:
        parsed = parse_float(side_data.get(field))
        if parsed is None:
            return None
        values[field] = parsed

    return MasteringDisplay(
        red_x=values["red_x"],
        red_y=values["red_y"],
        green_x=values["green_x"],
        green_y=values["green_y"],
        blue_x=values["blue_x"],
        blue_y=values["blue_y"],
        white_x=values["white_point_x"],
        white_y=values["white_point_y"],
        max_luminance=values["max_luminance"],
        min_luminance=values["min_luminance"],
    )


def _video_track(stream: dict[str, Any], duration: float | None) -> VideoTrack:
    pix_fmt = clean(stream.get("pix_fmt")) or None
    depth, chroma = pix_fmt_details(pix_fmt) if pix_fmt else (None, None)
    depth = depth or parse_positive_int(stream.get("bits_per_raw_sample"))

    field_order = clean(stream.get("field_order")).casefold()
    scan_type: str | None = None
    if field_order == "progressive":
        scan_type = "progressive"
    elif field_order in _INTERLACED_FIELD_ORDERS:
        scan_type = "interlaced"

    track = VideoTrack(
        index=parse_positive_int(stream.get("index")) or 0,
        codec=normalise_codec(stream.get("codec_name")),
        profile=clean(stream.get("profile")) or None,
        width=parse_positive_int(stream.get("width"))
        or parse_positive_int(stream.get("coded_width")),
        height=parse_positive_int(stream.get("height"))
        or parse_positive_int(stream.get("coded_height")),
        display_aspect_ratio=clean(stream.get("display_aspect_ratio")) or None,
        # r_frame_rate is the container's base rate; avg_frame_rate can be skewed
        # by a trailing partial GOP, so it is only the fallback.
        frame_rate=parse_frame_rate(stream.get("r_frame_rate"))
        or parse_frame_rate(stream.get("avg_frame_rate")),
        bit_depth=depth,
        chroma_subsampling=chroma,
        pix_fmt=pix_fmt,
        scan_type=scan_type,
        bitrate_bps=_stream_bitrate(stream, duration),
    )

    if primaries := clean(stream.get("color_primaries")):
        track.color_primaries = normalise_primaries(primaries)
    if transfer := clean(stream.get("color_transfer")):
        track.color_transfer = normalise_transfer(transfer)
    if matrix := clean(stream.get("color_space")):
        track.color_matrix = normalise_matrix(matrix)
    if color_range := clean(stream.get("color_range")):
        track.color_range = normalise_range(color_range)

    for entry in _as_list(stream.get("side_data_list")):
        side_data = _as_dict(entry)
        kind = clean(side_data.get("side_data_type")).casefold()
        if kind.startswith("mastering display"):
            track.mastering_display = _mastering_display(side_data)
        elif kind.startswith("content light level"):
            track.max_cll = parse_positive_int(side_data.get("max_content"))
            track.max_fall = parse_positive_int(side_data.get("max_average"))
        elif "dovi" in kind or "dolby vision" in kind:
            track.dolby_vision = True
        elif "2094-40" in kind or "hdr dynamic" in kind:
            track.hdr10_plus = True

    codec_tag = clean(stream.get("codec_tag_string")).casefold()
    if codec_tag in {"dvhe", "dvh1", "dav1"}:
        track.dolby_vision = True

    return track


def _audio_track(stream: dict[str, Any], duration: float | None) -> AudioTrack:
    return AudioTrack(
        index=parse_positive_int(stream.get("index")) or 0,
        codec=normalise_codec(clean(stream.get("profile")) or stream.get("codec_name")),
        channels=parse_positive_int(stream.get("channels")),
        channel_layout=clean(stream.get("channel_layout")) or None,
        sample_rate=parse_positive_int(stream.get("sample_rate")),
        bitrate_bps=_stream_bitrate(stream, duration),
        language=_tag(stream, "language"),
        title=_tag(stream, "title"),
        default=bool(_as_dict(stream.get("disposition")).get("default")),
    )


def _subtitle_track(stream: dict[str, Any]) -> SubtitleTrack:
    disposition = _as_dict(stream.get("disposition"))
    return SubtitleTrack(
        index=parse_positive_int(stream.get("index")) or 0,
        codec=normalise_codec(stream.get("codec_name")),
        language=_tag(stream, "language"),
        title=_tag(stream, "title"),
        forced=bool(disposition.get("forced")),
    )


def parse(text: str) -> SourceReport:
    """Build a :class:`SourceReport` from ffprobe JSON.

    Raises:
        ValueError: the text is not JSON, or is JSON of an unusable shape.
    """
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"That does not parse as ffprobe JSON ({exc}).") from exc

    if isinstance(payload, list):
        payload = {"streams": payload}
    if not isinstance(payload, dict):
        raise ValueError("Expected an ffprobe JSON object with 'streams' and/or 'format'.")

    fmt = _as_dict(payload.get("format"))
    # Only objects can be streams. Coercing scalars to {} instead would make
    # ``[1, 2, 3]`` look like three unreadable streams rather than unusable JSON.
    streams = [row for row in _as_list(payload.get("streams")) if isinstance(row, dict)]
    if not streams and not fmt:
        raise ValueError("No 'streams' or 'format' section found in that JSON.")

    warnings: list[str] = []
    duration = parse_duration(fmt.get("duration"))

    media = SourceMedia(
        container=clean(fmt.get("format_name")) or None,
        duration_seconds=duration,
        size_bytes=parse_size(fmt.get("size")),
        overall_bitrate_bps=parse_bitrate(fmt.get("bit_rate")),
    )

    for stream in streams:
        kind = clean(stream.get("codec_type")).casefold()
        if kind == "video":
            # Cover art is stored as a video stream; ignore the still-image ones.
            if _as_dict(stream.get("disposition")).get("attached_pic"):
                continue
            if media.video is None:
                media.video = _video_track(stream, duration)
        elif kind == "audio":
            media.audio.append(_audio_track(stream, duration))
        elif kind == "subtitle":
            media.subtitles.append(_subtitle_track(stream))

    if media.video is None:
        warnings.append("No video stream found in that ffprobe output.")
    else:
        if media.video.bit_depth is None:
            warnings.append(
                "No pixel format or bits_per_raw_sample, so bit depth is unknown — "
                "run ffprobe with -show_streams, or set it manually below."
            )
        if media.video.bitrate_bps is None and media.overall_bitrate_bps is None:
            warnings.append(
                "No bitrate reported. Add -show_format so the source's quality can be "
                "judged, or expect a conservative CRF."
            )
        if media.video.frame_rate is None:
            warnings.append("No frame rate reported; keyframe interval falls back to 24 fps.")

    if media.duration_seconds is None and media.overall_bitrate_bps is None:
        warnings.append("No -show_format section, so duration and overall bitrate are unknown.")

    return SourceReport(tool=SourceTool.FFPROBE, media=media, warnings=warnings)
