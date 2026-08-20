"""The mkvinfo parser.

mkvinfo is authoritative about what a Matroska container *declares* but two of the
elements we need most — ``BitsPerChannel`` and ``ChromaSubsampling`` — are optional,
so the interesting cases are the sparse ones and the warnings they produce.
"""

from __future__ import annotations

import pytest

from app.models import SourceReport, SourceTool
from app.sources import mkvinfo
from tests.support import fixture


def test_tool_and_container(mkvinfo_report: SourceReport) -> None:
    assert mkvinfo_report.tool is SourceTool.MKVINFO
    # From the EBML head's document type, not guessed from the filename.
    assert mkvinfo_report.media.container == "matroska"


def test_segment_gives_duration_size_and_a_derived_overall_bitrate(
    mkvinfo_report: SourceReport,
) -> None:
    media = mkvinfo_report.media

    assert media.duration_seconds == pytest.approx(7261.0)
    # "+ Segment: size 52428800000" — the master element's own value.
    assert media.size_bytes == 52_428_800_000
    # mkvinfo reports no overall bitrate, so it comes from size over duration.
    assert media.overall_bitrate_bps == 57_764_825


def test_video_geometry_and_derived_aspect_ratio(mkvinfo_report: SourceReport) -> None:
    video = mkvinfo_report.media.video
    assert video is not None

    assert video.codec == "hevc"
    assert (video.width, video.height) == (3840, 2160)
    # Matroska stores a display *size*; it is reduced to a ratio.
    assert video.display_aspect_ratio == "16:9"
    assert video.frame_rate == pytest.approx(23.976)
    # "Interlaced: 2" is FlagInterlaced=progressive, not "two fields".
    assert video.scan_type == "progressive"
    assert not video.is_interlaced


def test_profile_is_read_off_the_codec_private_data_annotation(
    mkvinfo_report: SourceReport,
) -> None:
    video = mkvinfo_report.media.video
    assert video is not None
    assert video.profile == "Main 10@L5.1"


def test_colour_element_gives_full_hdr10_signalling(mkvinfo_report: SourceReport) -> None:
    video = mkvinfo_report.media.video
    assert video is not None

    # H.273 code points, translated to the FFmpeg spellings used internally.
    assert video.color_primaries == "bt2020"
    assert video.color_transfer == "smpte2084"
    assert video.color_matrix == "bt2020nc"
    assert video.color_range == "tv"
    assert video.bit_depth == 10
    assert video.chroma_subsampling == "4:2:0"
    assert video.max_cll == 1000
    assert video.max_fall == 400
    assert video.is_hdr


def test_mastering_metadata_is_read_in_real_units(mkvinfo_report: SourceReport) -> None:
    video = mkvinfo_report.media.video
    assert video is not None
    display = video.mastering_display
    assert display is not None

    assert (display.red_x, display.red_y) == (0.68, 0.32)
    assert (display.green_x, display.green_y) == (0.265, 0.69)
    assert display.max_luminance == 1000.0
    assert display.min_luminance == 0.0001


def test_per_track_bitrate_comes_from_the_bps_tags(mkvinfo_report: SourceReport) -> None:
    media = mkvinfo_report.media
    assert media.video is not None

    # Tags are matched to tracks by Track UID, not by order.
    assert media.video.bitrate_bps == 58_120_000
    assert media.video.bits_per_pixel == pytest.approx(0.292, abs=1e-3)
    assert media.audio[0].bitrate_bps == 4_200_000


def test_audio_and_subtitle_tracks(mkvinfo_report: SourceReport) -> None:
    media = mkvinfo_report.media

    assert [track.codec for track in media.audio] == ["truehd"]
    assert media.audio[0].channels == 8
    assert media.audio[0].sample_rate == 48000
    assert media.audio[0].language == "eng"
    assert media.audio[0].title == "TrueHD Atmos 7.1"
    assert media.audio[0].default

    assert [track.codec for track in media.subtitles] == ["subrip"]
    assert not media.subtitles[0].forced


def test_a_complete_dump_produces_no_warnings(mkvinfo_report: SourceReport) -> None:
    assert mkvinfo_report.warnings == []


def test_sparse_dump_reads_what_is_there() -> None:
    report = mkvinfo.parse(fixture("mkvinfo_sparse.txt"))
    video = report.media.video
    assert video is not None

    assert video.codec == "h264"
    assert (video.width, video.height) == (1920, 816)
    # 1920x816 reduces to 40:17 — a 2.35:1 letterboxed transfer.
    assert video.display_aspect_ratio == "40:17"
    assert report.media.duration_seconds == pytest.approx(6750.0)
    assert report.media.overall_bitrate_bps == 10_180_663
    assert report.media.audio[0].codec == "ac3"
    # No "Default track" flag line at all: Matroska's default for the flag is 1.
    assert report.media.audio[0].default


def test_sparse_dump_names_every_gap_it_leaves() -> None:
    report = mkvinfo.parse(fixture("mkvinfo_sparse.txt"))
    video = report.media.video
    assert video is not None

    assert video.bit_depth is None
    assert video.chroma_subsampling is None
    assert video.bitrate_bps is None
    assert video.frame_rate is None
    assert video.profile is None

    # One warning each, and no more: the four things a CRF decision actually wants.
    assert len(report.warnings) == 4
    joined = " ".join(report.warnings)
    assert "BitsPerChannel" in joined
    assert "chroma subsampling" in joined
    assert "BPS statistics tag" in joined
    assert "frame rate is unknown" in joined


def test_line_prefixed_builds_parse_identically(mkvinfo_report: SourceReport) -> None:
    # Older MKVToolNix prints "(mkvinfo) |  + Pixel width: 3840". Stripping the prefix
    # has to leave the relative indentation intact or the tree flattens.
    prefixed = "\n".join(
        f"(mkvinfo) {line}" for line in fixture("mkvinfo_uhd_hdr.txt").splitlines()
    )

    report = mkvinfo.parse(prefixed)

    assert report.media == mkvinfo_report.media
    assert report.warnings == []


def test_display_unit_other_than_pixels_yields_no_aspect_ratio() -> None:
    # Display unit 3 is "aspect ratio"; 1920x816 would then not be a pixel count.
    text = fixture("mkvinfo_uhd_hdr.txt").replace("Display unit: 0", "Display unit: 3")

    report = mkvinfo.parse(text)

    assert report.media.video is not None
    assert report.media.video.display_aspect_ratio is None


def test_partial_mastering_metadata_is_discarded() -> None:
    # A master-display string built from seven of eight coordinates would tell a
    # display the wrong volume, so it is all or nothing.
    text = fixture("mkvinfo_uhd_hdr.txt").replace("Blue colour coordinate y: 0.06", "")

    report = mkvinfo.parse(text)

    assert report.media.video is not None
    assert report.media.video.mastering_display is None
    # The rest of the Colour element still applies.
    assert report.media.video.color_transfer == "smpte2084"


def test_interlaced_flag_is_honoured() -> None:
    text = fixture("mkvinfo_uhd_hdr.txt").replace("Interlaced: 2", "Interlaced: 1")

    report = mkvinfo.parse(text)

    assert report.media.video is not None
    assert report.media.video.scan_type == "interlaced"
    assert report.media.video.is_interlaced


def test_bit_depth_falls_back_to_the_codec_profile() -> None:
    text = fixture("mkvinfo_uhd_hdr.txt").replace("Bits per channel: 10", "")

    report = mkvinfo.parse(text)

    assert report.media.video is not None
    # "HEVC profile: Main 10 @L5.1" is enough to know it is 10-bit.
    assert report.media.video.bit_depth == 10
    assert report.warnings == []


def test_summary_output_is_refused_with_an_explanation() -> None:
    # mkvinfo --summary prints a flat "Track 1: video" listing with no tree.
    with pytest.raises(ValueError, match="Tracks"):
        mkvinfo.parse("+ EBML head\n|+ Document type: matroska\n+ Segment: size 10\n")


def test_text_with_no_tree_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="No mkvinfo tree"):
        mkvinfo.parse("this is not mkvinfo output")


def test_looks_like_mkvinfo() -> None:
    assert mkvinfo.looks_like_mkvinfo(fixture("mkvinfo_uhd_hdr.txt"))
    assert mkvinfo.looks_like_mkvinfo(fixture("mkvinfo_sparse.txt"))
    assert not mkvinfo.looks_like_mkvinfo(fixture("ffprobe_uhd_hdr.json"))
    assert not mkvinfo.looks_like_mkvinfo(fixture("mediainfo_1080p_film.txt"))
    # One marker is not enough — "+ Segment" alone appears in plenty of prose.
    assert not mkvinfo.looks_like_mkvinfo("+ Segment: size 10")
