"""Parse ``mkvinfo`` output — MKVToolNix's Matroska structure dump.

Why support it at all, given ffprobe exists: for an MKV remux (which is what most
sources are) mkvinfo is the authoritative reader of what the *container* declares,
including the full ``Colour`` element — primaries, transfer, matrix, range, MaxCLL,
MaxFALL and ST 2086 mastering metadata — plus per-track bitrate from the ``BPS``
statistics tags mkvmerge writes by default.

Where it is genuinely weaker, and why the report carries warnings: ``BitsPerChannel``
and ``ChromaSubsampling`` are *optional* Matroska elements that many muxers omit, and
bit depth is a primary input to the CRF decision. When they are missing we fall back
to the profile mkvinfo annotates onto ``Codec's private data`` (``HEVC profile:
Main 10`` implies 10-bit) and say so. And it reads Matroska only — an MP4 or a ProRes
master has to go through ffprobe or MediaInfo.

The output is an indentation tree (``|  + Label: value``). Absolute indentation has
shifted between MKVToolNix releases, so the parser builds a tree from *relative*
depth rather than matching fixed column positions.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction

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
    MATRIX_BY_CODE,
    PRIMARIES_BY_CODE,
    RANGE_BY_CODE,
    TRANSFER_BY_CODE,
    chroma_from_subsampling_pair,
)
from app.sources.parsing import (
    clean,
    normalise_codec,
    parse_bitrate,
    parse_duration,
    parse_float,
    parse_frame_rate,
    parse_int,
    parse_positive_int,
    parse_size,
)

# "|  + Label: value", with an optional "(mkvinfo)" prefix that some builds print.
_LINE = re.compile(r"^(?:\(mkvinfo\)\s*)?(?P<prefix>[|\s]*)\+\s?(?P<body>.*?)\s*$")

# mkvinfo annotates codec private data with the stream profile, e.g.
# "Codec's private data: size 2299 (HEVC profile: Main 10 @L5.1)".
_PROFILE_IN_PRIVATE = re.compile(
    r"profile:\s*(?P<profile>[^@)]+?)\s*(?:@\s*L?(?P<level>[\d.]+))?\)"
)


def looks_like_mkvinfo(text: str) -> bool:
    lowered = text.casefold()
    markers = ("+ ebml head", "+ segment", "|+ tracks", "+ track type:", "document type: matroska")
    return sum(marker in lowered for marker in markers) >= 2


@dataclass(slots=True)
class Node:
    """One line of the dump, with the lines nested under it."""

    label: str
    value: str = ""
    children: list[Node] = field(default_factory=list)

    def find(self, *labels: str) -> Node | None:
        wanted = {label.casefold() for label in labels}
        return next((child for child in self.children if child.label.casefold() in wanted), None)

    def find_all(self, *labels: str) -> list[Node]:
        wanted = {label.casefold() for label in labels}
        return [child for child in self.children if child.label.casefold() in wanted]

    def get(self, *labels: str) -> str | None:
        node = self.find(*labels)
        return node.value or None if node else None

    def walk(self) -> Iterator[Node]:
        for child in self.children:
            yield child
            yield from child.walk()

    def search(self, *labels: str) -> Node | None:
        """First descendant with one of these labels, at any depth."""
        wanted = {label.casefold() for label in labels}
        return next((node for node in self.walk() if node.label.casefold() in wanted), None)

    def search_all(self, *labels: str) -> list[Node]:
        wanted = {label.casefold() for label in labels}
        return [node for node in self.walk() if node.label.casefold() in wanted]


def _build_tree(text: str) -> Node:
    root = Node("root")
    stack: list[tuple[int, Node]] = [(-1, root)]

    for raw in text.splitlines():
        match = _LINE.match(raw)
        if not match or not match.group("body"):
            continue

        depth = len(match.group("prefix"))
        label, separator, value = match.group("body").partition(":")
        # mkvinfo quotes flag names: '"Default track" flag: 1' -> 'Default track flag'.
        label = clean(label.replace('"', ""))
        value = clean(value) if separator else ""

        # Every line is pushed, master or leaf: nesting comes from relative depth
        # alone, so a master that carries a value ("Segment: size 341298") keeps it.
        while stack and stack[-1][0] >= depth:
            stack.pop()
        node = Node(label=label, value=value)
        stack[-1][1].children.append(node)
        stack.append((depth, node))

    return root


def _display_aspect_ratio(video: Node) -> str | None:
    """Matroska stores display size, not a ratio; reduce it to one.

    ``Display unit`` 0 means the values are pixels, which is the only unit that can
    be turned into an aspect ratio without knowing the display's own geometry.
    """
    unit = parse_int(video.get("Display unit"))
    if unit not in (None, 0):
        return None
    width = parse_positive_int(video.get("Display width"))
    height = parse_positive_int(video.get("Display height"))
    if not (width and height):
        return None
    ratio = Fraction(width, height).limit_denominator(1000)
    return f"{ratio.numerator}:{ratio.denominator}"


def _mastering_display(colour: Node) -> MasteringDisplay | None:
    node = colour.find(
        "Video colour mastering metadata",
        "Mastering metadata",
        "Colour mastering metadata",
    )
    if node is None:
        return None

    coordinates: dict[str, float] = {}
    for name in ("Red", "Green", "Blue", "White"):
        for axis in ("x", "y"):
            value = parse_float(
                node.get(
                    f"{name} colour coordinate {axis}",
                    f"{name} color coordinate {axis}",
                )
            )
            if value is None:
                return None
            coordinates[f"{name.casefold()}_{axis}"] = value

    max_luminance = parse_float(node.get("Maximum luminance"))
    min_luminance = parse_float(node.get("Minimum luminance"))
    if max_luminance is None or min_luminance is None:
        return None

    return MasteringDisplay(
        red_x=coordinates["red_x"],
        red_y=coordinates["red_y"],
        green_x=coordinates["green_x"],
        green_y=coordinates["green_y"],
        blue_x=coordinates["blue_x"],
        blue_y=coordinates["blue_y"],
        white_x=coordinates["white_x"],
        white_y=coordinates["white_y"],
        max_luminance=max_luminance,
        min_luminance=min_luminance,
    )


def _apply_colour(track: VideoTrack, colour: Node) -> None:
    if (code := parse_int(colour.get("Colour primaries", "Color primaries"))) is not None:
        track.color_primaries = PRIMARIES_BY_CODE.get(code)
    if (code := parse_int(colour.get("Colour transfer", "Color transfer"))) is not None:
        track.color_transfer = TRANSFER_BY_CODE.get(code)
    if (
        code := parse_int(colour.get("Colour matrix coefficients", "Color matrix coefficients"))
    ) is not None:
        track.color_matrix = MATRIX_BY_CODE.get(code)
    if (code := parse_int(colour.get("Colour range", "Color range"))) is not None:
        track.color_range = RANGE_BY_CODE.get(code)

    track.bit_depth = parse_positive_int(colour.get("Bits per channel"))
    horizontal = parse_int(colour.get("Chroma subsampling horizontal"))
    vertical = parse_int(colour.get("Chroma subsampling vertical"))
    if horizontal is not None and vertical is not None:
        track.chroma_subsampling = chroma_from_subsampling_pair(horizontal, vertical)

    track.max_cll = parse_positive_int(colour.get("Maximum content light"))
    track.max_fall = parse_positive_int(colour.get("Maximum frame light"))
    track.mastering_display = _mastering_display(colour)


def _profile_from_private_data(track_node: Node) -> tuple[str | None, int | None]:
    """Read the profile mkvinfo annotates onto codec private data.

    This is the fallback that rescues bit depth when ``BitsPerChannel`` is absent:
    ``Main 10`` and ``Main 12`` name their depth, and any HEVC/AVC profile without a
    number is 8-bit.
    """
    private = track_node.get("Codec's private data", "Codec private data")
    if not private:
        return None, None

    match = _PROFILE_IN_PRIVATE.search(private)
    if not match:
        return None, None

    profile = clean(match.group("profile"))
    level = match.group("level")
    label = f"{profile}@L{level}" if level else profile

    depth: int | None = None
    if (found := re.search(r"\b(10|12)\b", profile)) is not None:
        depth = int(found.group(1))
    elif re.search(r"\b(main|high|baseline|extended)\b", profile, re.IGNORECASE):
        depth = 8
    return label, depth


def _video_track(track_node: Node, number: int | None) -> VideoTrack:
    video = track_node.find("Video track") or Node("Video track")
    profile, profile_depth = _profile_from_private_data(track_node)

    track = VideoTrack(
        index=number,
        codec=normalise_codec(track_node.get("Codec ID")),
        profile=profile,
        width=parse_positive_int(video.get("Pixel width")),
        height=parse_positive_int(video.get("Pixel height")),
        display_aspect_ratio=_display_aspect_ratio(video),
        frame_rate=_frame_rate(track_node),
        # Matroska has no scan-type element in practice; interlacing shows up as
        # a FlagInterlaced of 1 ("Interlaced" in newer mkvinfo).
        scan_type=_scan_type(video, track_node),
    )

    if colour := video.find("Colour", "Color", "Video colour information"):
        _apply_colour(track, colour)
    if track.bit_depth is None:
        track.bit_depth = profile_depth

    codec_id = (track_node.get("Codec ID") or "").casefold()
    track.dolby_vision = "dvhe" in codec_id or "dvh1" in codec_id

    return track


def _frame_rate(track_node: Node) -> float | None:
    """``Default duration: 00:00:00.041708333 (23.976 frames/fields per second…)``.

    The parenthesised rate is what mkvinfo computed and is preferred; otherwise the
    reciprocal of the default duration gives the same answer.
    """
    default_duration = track_node.get("Default duration")
    if not default_duration:
        return None

    parenthesised = re.search(r"\(([\d.]+)\s*frames", default_duration)
    if parenthesised:
        return parse_frame_rate(parenthesised.group(1))

    seconds = parse_duration(default_duration.split("(")[0])
    return 1 / seconds if seconds else None


def _scan_type(video: Node, track_node: Node) -> str | None:
    labels = ("Interlaced", "Interlaced flag", "Video interlaced flag")
    flag = video.get(*labels) or track_node.get(*labels)
    if flag is None:
        return None
    lowered = flag.casefold()
    if "progress" in lowered or lowered.startswith("2"):
        return "progressive"
    if "interlac" in lowered or lowered.startswith("1"):
        return "interlaced"
    return None


def _audio_track(track_node: Node, number: int | None) -> AudioTrack:
    audio = track_node.find("Audio track") or Node("Audio track")
    sample_rate = parse_float(audio.get("Sampling frequency"))
    return AudioTrack(
        index=number,
        codec=normalise_codec(track_node.get("Codec ID")),
        channels=parse_positive_int(audio.get("Channels")),
        sample_rate=int(sample_rate) if sample_rate else None,
        language=track_node.get("Language", "Language (IETF BCP 47)"),
        title=track_node.get("Name"),
        default=(track_node.get("Default track flag") or "1") != "0",
    )


def _subtitle_track(track_node: Node, number: int | None) -> SubtitleTrack:
    return SubtitleTrack(
        index=number,
        codec=normalise_codec(track_node.get("Codec ID")),
        language=track_node.get("Language", "Language (IETF BCP 47)"),
        title=track_node.get("Name"),
        forced=(track_node.get("Forced display flag", "Forced track flag") or "0") != "0",
    )


def _statistics_tags(root: Node) -> dict[int | None, dict[str, str]]:
    """Collect mkvmerge's ``BPS`` / ``NUMBER_OF_BYTES`` tags per track UID.

    A tag block with no ``Targets`` applies to the whole file and is filed under
    ``None``.
    """
    collected: dict[int | None, dict[str, str]] = {}
    for tags in root.search_all("Tags"):
        for tag in tags.find_all("Tag"):
            targets = tag.find("Targets")
            uid = parse_int(targets.get("Track UID")) if targets else None
            bucket = collected.setdefault(uid, {})
            for simple in tag.find_all("Simple"):
                name = simple.get("Name")
                value = simple.get("String", "Binary")
                if name and value:
                    bucket[name.upper()] = value
    return collected


def _apply_statistics(
    track: VideoTrack | AudioTrack,
    uid: int | None,
    tags: dict[int | None, dict[str, str]],
    duration: float | None,
) -> None:
    stats = tags.get(uid) or tags.get(None) or {}
    if not stats:
        return

    if bitrate := parse_bitrate(stats.get("BPS")):
        track.bitrate_bps = bitrate
        return

    size = parse_size(stats.get("NUMBER_OF_BYTES"))
    span = parse_duration(stats.get("DURATION")) or duration
    if size and span:
        track.bitrate_bps = int(size * 8 / span)


def parse(text: str) -> SourceReport:
    """Build a :class:`SourceReport` from an mkvinfo dump.

    Raises:
        ValueError: no Matroska tree could be recognised in the text.
    """
    root = _build_tree(text)
    if not root.children:
        raise ValueError("No mkvinfo tree found — expected lines like '|+ Tracks'.")

    tracks_node = root.search("Tracks")
    info_node = root.search("Segment information")
    if tracks_node is None and info_node is None:
        raise ValueError(
            "That looks like mkvinfo output but has neither a 'Tracks' nor a "
            "'Segment information' section. Run mkvinfo without --summary."
        )

    warnings: list[str] = []
    duration = parse_duration(info_node.get("Duration")) if info_node else None

    # "+ Segment: size 34129857693" is the closest thing to a file size mkvinfo gives.
    segment = root.find("Segment")
    size_bytes = parse_size(segment.value) if segment and segment.value else None

    ebml_head = root.find("EBML head")
    media = SourceMedia(
        container=(ebml_head.get("Document type") if ebml_head else None) or "matroska",
        duration_seconds=duration,
        size_bytes=size_bytes,
    )
    if size_bytes and duration:
        media.overall_bitrate_bps = int(size_bytes * 8 / duration)

    tags = _statistics_tags(root)
    track_nodes = tracks_node.find_all("Track") if tracks_node else []

    for track_node in track_nodes:
        kind = (track_node.get("Track type") or "").casefold()
        number = parse_positive_int(track_node.get("Track number"))
        uid = parse_int(track_node.get("Track UID"))

        if kind.startswith("video") and media.video is None:
            video = _video_track(track_node, number)
            _apply_statistics(video, uid, tags, duration)
            media.video = video
        elif kind.startswith("audio"):
            audio = _audio_track(track_node, number)
            _apply_statistics(audio, uid, tags, duration)
            media.audio.append(audio)
        elif kind.startswith("subtitle"):
            media.subtitles.append(_subtitle_track(track_node, number))

    if media.video is None:
        warnings.append("No video track found in that mkvinfo output.")
    else:
        if media.video.bit_depth is None:
            warnings.append(
                "mkvinfo reported no bit depth: Matroska's BitsPerChannel element is "
                "optional and this file omits it, and the codec profile was not "
                "annotated either. Set the bit depth below, or paste ffprobe output."
            )
        if media.video.chroma_subsampling is None:
            warnings.append(
                "No chroma subsampling in the container (also an optional Matroska "
                "element) — assuming 4:2:0, which is right for essentially every "
                "consumer source."
            )
        if media.video.bitrate_bps is None:
            warnings.append(
                "No BPS statistics tag for the video track, so the source's own "
                "bitrate is unknown and the CRF stays conservative. mkvmerge writes "
                "these tags by default; a file without them was muxed elsewhere."
            )
        if media.video.frame_rate is None:
            warnings.append(
                "No default duration on the video track, so the frame rate is unknown "
                "(keyframe interval falls back to 24 fps)."
            )

    return SourceReport(tool=SourceTool.MKVINFO, media=media, warnings=warnings)
