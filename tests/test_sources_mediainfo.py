"""The MediaInfo text parser.

MediaInfo formats its numbers for people — grouped digits, rounded bitrates, durations
truncated to whole minutes — so the parser's job is to undo that formatting without
inventing precision. Several of these tests exist because it once did not.
"""

from __future__ import annotations

import textwrap

import pytest

from app.models import SourceReport, SourceTool
from app.sources import mediainfo
from tests.support import fixture

HDR_REPORT = textwrap.dedent(
    """\
    General
    Format                                   : Matroska
    Duration                                 : 2 h 8 min
    File size                                : 52.1 GiB

    Video
    Format                                   : HEVC
    Format profile                           : Main 10@L5.1@High
    Width                                    : 3 840 pixels
    Height                                   : 2 160 pixels
    Frame rate                               : 23.976 (24000/1001) FPS
    Bit rate                                 : 58.1 Mb/s
    Chroma subsampling                       : 4:2:0 (Type 2)
    Bit depth                                : 10 bits
    Scan type                                : Progressive
    HDR format                               : SMPTE ST 2094-40, HDR10+ Profile B
    Color range                              : Limited
    Color primaries                          : BT.2020
    Transfer characteristics                 : PQ
    Matrix coefficients                      : BT.2020 non-constant
    Mastering display color primaries        : BT.2020
    Mastering display luminance              : min: 0.0001 cd/m2, max: 1000 cd/m2
    Maximum Content Light Level              : 1234 cd/m2
    Maximum Frame-Average Light Level        : 321 cd/m2
    """
)


def test_tool_and_general_section(mediainfo_report: SourceReport) -> None:
    media = mediainfo_report.media

    assert mediainfo_report.tool is SourceTool.MEDIAINFO
    assert media.container == "Matroska"
    # "24.5 GiB" — binary units, so not 24.5e9.
    assert media.size_bytes == 26_306_674_688
    # "1 h 57 min": the seconds MediaInfo dropped are gone for good.
    assert media.duration_seconds == pytest.approx(7020.0)
    assert media.overall_bitrate_bps == 30_000_000


def test_grouped_digits_do_not_truncate_the_resolution(mediainfo_report: SourceReport) -> None:
    video = mediainfo_report.media.video
    assert video is not None

    # The bug this guards: "1 920 pixels" parsed as 1, so every MediaInfo paste
    # looked like a 1x1 SD source and got an absurd CRF.
    assert (video.width, video.height) == (1920, 1080)
    assert mediainfo_report.media.resolution_label == "1080p"


def test_video_track_details(mediainfo_report: SourceReport) -> None:
    video = mediainfo_report.media.video
    assert video is not None

    assert video.codec == "h264"
    assert video.profile == "High@L4.1"
    assert video.display_aspect_ratio == "1.85:1"
    assert video.frame_rate == pytest.approx(24000 / 1001)
    assert video.frame_rate_mode == "constant"
    assert video.bit_depth == 8
    assert video.chroma_subsampling == "4:2:0"
    assert video.scan_type == "progressive"
    assert not video.is_interlaced
    assert video.bitrate_bps == 28_000_000
    # MediaInfo's own "Bits/(Pixel*Frame) : 0.564" agrees.
    assert video.bits_per_pixel == pytest.approx(0.563, abs=1e-3)


def test_pixel_format_is_synthesised_from_depth_and_subsampling(
    mediainfo_report: SourceReport,
) -> None:
    video = mediainfo_report.media.video
    assert video is not None
    # MediaInfo has no pix_fmt row; 8-bit 4:2:0 implies this name.
    assert video.pix_fmt == "yuv420p"


def test_sdr_colour_labels_become_ffmpeg_spellings(mediainfo_report: SourceReport) -> None:
    video = mediainfo_report.media.video
    assert video is not None

    assert video.color_primaries == "bt709"
    assert video.color_transfer == "bt709"
    assert video.color_matrix == "bt709"
    assert video.color_range == "tv"
    assert not video.is_hdr
    assert not video.is_wide_gamut


def test_audio_codec_comes_from_the_commercial_name(mediainfo_report: SourceReport) -> None:
    audio = mediainfo_report.media.audio

    # Format is "DTS XLL", which does not say lossless; the commercial name does.
    assert [track.codec for track in audio] == ["dts-hd ma", "ac3"]
    assert audio[0].is_lossless
    assert not audio[1].is_lossless


def test_audio_track_details(mediainfo_report: SourceReport) -> None:
    first, second = mediainfo_report.media.audio

    assert first.channels == 6
    assert first.channel_layout == "C L R Ls Rs LFE"
    # "48.0 kHz" scaled by its unit, not taken as 48 Hz.
    assert first.sample_rate == 48000
    assert first.bitrate_bps == 3_502_000
    assert first.language == "English"
    assert first.title == "DTS-HD MA 5.1"
    assert first.default

    assert second.bitrate_bps == 640_000
    assert second.title == "Commentary"
    assert not second.default


def test_subtitle_codec_prefers_codec_id_over_format(mediainfo_report: SourceReport) -> None:
    subtitles = mediainfo_report.media.subtitles

    # Format is "UTF-8", which is the character encoding, not the subtitle format.
    assert [track.codec for track in subtitles] == ["subrip"]
    assert subtitles[0].title == "English SDH"
    assert not subtitles[0].forced


def test_the_only_warning_is_about_rounding(mediainfo_report: SourceReport) -> None:
    # Everything a CRF needs is present; the one caveat is MediaInfo's own rounding.
    assert len(mediainfo_report.warnings) == 1
    assert "rounds durations and bitrates" in mediainfo_report.warnings[0]


def test_menu_section_does_not_become_a_track(mediainfo_report: SourceReport) -> None:
    # Its chapter rows ("00:07:12.291 : en:Chapter 02") look like label/value pairs.
    media = mediainfo_report.media
    assert len(media.audio) == 2
    assert len(media.subtitles) == 1


def test_hdr10_plus_report() -> None:
    report = mediainfo.parse(HDR_REPORT)
    video = report.media.video
    assert video is not None

    assert video.codec == "hevc"
    assert (video.width, video.height) == (3840, 2160)
    assert video.bit_depth == 10
    assert video.pix_fmt == "yuv420p10le"
    # "4:2:0 (Type 2)" — the parenthesised siting is dropped.
    assert video.chroma_subsampling == "4:2:0"
    assert video.color_primaries == "bt2020"
    assert video.color_transfer == "smpte2084"
    assert video.color_matrix == "bt2020nc"
    assert video.is_hdr
    assert video.hdr10_plus
    assert not video.dolby_vision
    assert video.max_cll == 1234
    assert video.max_fall == 321


def test_named_mastering_primaries_become_coordinates() -> None:
    report = mediainfo.parse(HDR_REPORT)
    assert report.media.video is not None
    display = report.media.video.mastering_display
    assert display is not None

    # MediaInfo names the primary set instead of listing xy pairs, so the standard
    # BT.2020 coordinates are substituted — note these are the *container* values
    # (0.708/0.292), not the rounded ones a grader typed into an ffprobe fixture.
    assert (display.red_x, display.red_y) == (0.708, 0.292)
    assert display.max_luminance == 1000.0
    assert display.min_luminance == 0.0001
    assert display.to_x265().startswith("G(8500,39850)")


def test_dolby_vision_is_read_from_the_hdr_format_row() -> None:
    text = HDR_REPORT.replace(
        "SMPTE ST 2094-40, HDR10+ Profile B",
        "Dolby Vision, Version 1.0, dvhe.08.06, BL+RPU",
    )

    report = mediainfo.parse(text)

    assert report.media.video is not None
    assert report.media.video.dolby_vision
    assert not report.media.video.hdr10_plus


def test_interlaced_scan_type() -> None:
    text = HDR_REPORT.replace(
        "Scan type                                : Progressive", "Scan type : MBAFF"
    )

    report = mediainfo.parse(text)

    assert report.media.video is not None
    assert report.media.video.scan_type == "mbaff"


def test_missing_general_section_is_reported_not_fatal() -> None:
    text = HDR_REPORT.split("Video", 1)[1]

    report = mediainfo.parse(f"Video{text}")

    assert report.media.video is not None
    assert report.media.duration_seconds is None
    assert report.media.container is None
    assert any("no size estimate" in warning for warning in report.warnings)
    # No duration, so no rounding caveat to give.
    assert not any("rounds durations" in warning for warning in report.warnings)


def test_a_report_with_no_video_section_warns() -> None:
    text = textwrap.dedent(
        """\
        General
        Format                                   : FLAC
        Duration                                 : 42 min

        Audio
        Format                                   : FLAC
        Channel(s)                               : 2 channels
        Sampling rate                            : 96.0 kHz
        """
    )

    report = mediainfo.parse(text)

    assert report.media.video is None
    assert not report.has_video
    assert report.media.audio[0].sample_rate == 96000
    assert any("No Video section" in warning for warning in report.warnings)


def test_missing_bit_depth_and_bitrate_are_each_flagged() -> None:
    text = "\n".join(
        line
        for line in HDR_REPORT.splitlines()
        if not line.startswith(("Bit depth", "Bit rate", "File size"))
    )

    report = mediainfo.parse(text)

    assert report.media.video is not None
    assert report.media.video.bit_depth is None
    assert report.media.video.pix_fmt is None
    joined = " ".join(report.warnings)
    assert "No bit depth" in joined
    assert "quality cannot be judged" in joined


def test_text_without_sections_is_refused() -> None:
    with pytest.raises(ValueError, match="No MediaInfo sections"):
        mediainfo.parse("Label  : value\nAnother  : value\n")


def test_looks_like_mediainfo() -> None:
    assert mediainfo.looks_like_mediainfo(fixture("mediainfo_1080p_film.txt"))
    assert mediainfo.looks_like_mediainfo(HDR_REPORT)
    assert not mediainfo.looks_like_mediainfo(fixture("ffprobe_uhd_hdr.json"))
    assert not mediainfo.looks_like_mediainfo(fixture("mkvinfo_uhd_hdr.txt"))
    # A section header with no rows under it is not a report.
    assert not mediainfo.looks_like_mediainfo("Video\n")
