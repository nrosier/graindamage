"""The ffprobe JSON parser — the preferred input, so the strictest expectations."""

from __future__ import annotations

import json

import pytest

from app.models import SourceReport, SourceTool
from app.sources import ffprobe
from tests.support import fixture


def test_tool_is_recorded(ffprobe_report: SourceReport) -> None:
    assert ffprobe_report.tool is SourceTool.FFPROBE
    assert ffprobe_report.has_video


def test_container_level_facts(ffprobe_report: SourceReport) -> None:
    media = ffprobe_report.media

    assert media.container == "matroska,webm"
    assert media.duration_seconds == pytest.approx(7261.0)
    assert media.size_bytes == 56_000_000_000
    assert media.overall_bitrate_bps == 61_700_000


def test_video_track_geometry_and_depth(ffprobe_report: SourceReport) -> None:
    video = ffprobe_report.media.video
    assert video is not None

    assert video.codec == "hevc"
    assert video.profile == "Main 10"
    assert (video.width, video.height) == (3840, 2160)
    assert video.display_aspect_ratio == "16:9"
    assert video.frame_rate == pytest.approx(24000 / 1001)
    assert video.pix_fmt == "yuv420p10le"
    # From the pixel format, not from bits_per_raw_sample — both agree here.
    assert video.bit_depth == 10
    assert video.chroma_subsampling == "4:2:0"
    assert video.scan_type == "progressive"
    assert not video.is_interlaced


def test_video_bitrate_comes_from_the_matroska_bps_tag(ffprobe_report: SourceReport) -> None:
    video = ffprobe_report.media.video
    assert video is not None

    # The stream has no bit_rate field; BPS-eng is what a remux leaves behind.
    assert video.bitrate_bps == 58_120_000
    assert video.bits_per_pixel == pytest.approx(0.292, abs=1e-3)


def test_hdr10_signalling_survives(ffprobe_report: SourceReport) -> None:
    video = ffprobe_report.media.video
    assert video is not None

    assert video.color_primaries == "bt2020"
    assert video.color_transfer == "smpte2084"
    assert video.color_matrix == "bt2020nc"
    assert video.color_range == "tv"
    assert video.is_hdr
    assert video.is_wide_gamut
    assert not video.dolby_vision
    assert not video.hdr10_plus


def test_mastering_display_fractions_become_real_units(ffprobe_report: SourceReport) -> None:
    video = ffprobe_report.media.video
    assert video is not None
    display = video.mastering_display
    assert display is not None

    # "34000/50000" -> 0.68, "10000000/10000" -> 1000 cd/m².
    assert (display.red_x, display.red_y) == (0.68, 0.32)
    assert (display.green_x, display.green_y) == (0.265, 0.69)
    assert (display.blue_x, display.blue_y) == (0.15, 0.06)
    assert (display.white_x, display.white_y) == (0.3127, 0.329)
    assert display.max_luminance == 1000.0
    assert display.min_luminance == 0.0001
    assert video.max_cll == 1000
    assert video.max_fall == 400


def test_audio_tracks_keep_codec_channels_and_language(ffprobe_report: SourceReport) -> None:
    audio = ffprobe_report.media.audio

    assert [track.codec for track in audio] == ["truehd", "ac3"]
    assert [track.channels for track in audio] == [8, 6]
    assert [track.language for track in audio] == ["eng", "fra"]
    assert audio[0].is_lossless
    assert not audio[1].is_lossless
    # No BPS tag and no bit_rate on the TrueHD track, so it stays honest.
    assert audio[0].bitrate_bps is None
    assert audio[1].bitrate_bps == 640_000


def test_subtitles_are_collected(ffprobe_report: SourceReport) -> None:
    assert [track.codec for track in ffprobe_report.media.subtitles] == ["subrip"]


def test_cover_art_is_not_mistaken_for_the_video_track(ffprobe_report: SourceReport) -> None:
    # The fixture's last stream is an mjpeg with disposition.attached_pic set. Taking
    # it as the video track would report a 600x900 still and an SD-class encode.
    video = ffprobe_report.media.video
    assert video is not None
    assert video.codec == "hevc"


def test_complete_output_produces_no_warnings(ffprobe_report: SourceReport) -> None:
    assert ffprobe_report.warnings == []


def test_streams_only_paste_warns_about_what_is_missing() -> None:
    payload = json.loads(fixture("ffprobe_uhd_hdr.json"))
    del payload["format"]

    report = ffprobe.parse(json.dumps(payload))

    assert report.media.duration_seconds is None
    # The BPS tag still gives a per-track bitrate, so only the format section is missed.
    assert report.media.video is not None
    assert report.media.video.bitrate_bps == 58_120_000
    assert any("show_format" in warning for warning in report.warnings)


def test_missing_pixel_format_falls_back_to_bits_per_raw_sample() -> None:
    payload = json.loads(fixture("ffprobe_uhd_hdr.json"))
    del payload["streams"][0]["pix_fmt"]

    report = ffprobe.parse(json.dumps(payload))

    assert report.media.video is not None
    assert report.media.video.bit_depth == 10
    assert report.media.video.chroma_subsampling is None
    assert report.warnings == []


def test_no_depth_at_all_is_reported() -> None:
    payload = json.loads(fixture("ffprobe_uhd_hdr.json"))
    del payload["streams"][0]["pix_fmt"]
    del payload["streams"][0]["bits_per_raw_sample"]

    report = ffprobe.parse(json.dumps(payload))

    assert report.media.video is not None
    assert report.media.video.bit_depth is None
    assert any("bit depth is unknown" in warning for warning in report.warnings)


def test_interlaced_field_order_is_recognised() -> None:
    payload = json.loads(fixture("ffprobe_uhd_hdr.json"))
    payload["streams"][0]["field_order"] = "tt"

    report = ffprobe.parse(json.dumps(payload))

    assert report.media.video is not None
    assert report.media.video.scan_type == "interlaced"
    assert report.media.video.is_interlaced


def test_dolby_vision_from_codec_tag() -> None:
    payload = json.loads(fixture("ffprobe_uhd_hdr.json"))
    payload["streams"][0]["codec_tag_string"] = "dvhe"

    report = ffprobe.parse(json.dumps(payload))

    assert report.media.video is not None
    assert report.media.video.dolby_vision


def test_a_bare_stream_list_is_accepted() -> None:
    payload = json.loads(fixture("ffprobe_uhd_hdr.json"))

    report = ffprobe.parse(json.dumps(payload["streams"]))

    assert report.media.video is not None
    assert report.media.video.width == 3840
    assert report.media.container is None


def test_audio_only_paste_warns_rather_than_raising() -> None:
    payload = {"streams": [{"codec_type": "audio", "codec_name": "flac", "channels": 2}]}

    report = ffprobe.parse(json.dumps(payload))

    assert report.media.video is None
    assert not report.has_video
    assert any("No video stream" in warning for warning in report.warnings)


@pytest.mark.parametrize(
    "text",
    ["not json at all", "{", '{"nothing": true}', "[1, 2, 3]"],
)
def test_unusable_json_raises_value_error(text: str) -> None:
    with pytest.raises(ValueError):
        ffprobe.parse(text)


def test_looks_like_ffprobe() -> None:
    assert ffprobe.looks_like_ffprobe(fixture("ffprobe_uhd_hdr.json"))
    assert ffprobe.looks_like_ffprobe('[{"codec_type": "video"}]')
    assert not ffprobe.looks_like_ffprobe(fixture("mkvinfo_uhd_hdr.txt"))
    assert not ffprobe.looks_like_ffprobe(fixture("mediainfo_1080p_film.txt"))
    assert not ffprobe.looks_like_ffprobe('{"unrelated": 1}')
