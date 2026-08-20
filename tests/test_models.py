"""The derived properties other modules make decisions on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import (
    Advice,
    AudioTrack,
    Encoder,
    EncoderPlan,
    GrainLevel,
    GrainProfile,
    MasteringDisplay,
    Movie,
    MovieHit,
    SourceMedia,
    TechnicalSpecs,
    VideoTrack,
)
from tests.support import video_track

BT2020_DISPLAY = MasteringDisplay(
    red_x=0.68,
    red_y=0.32,
    green_x=0.265,
    green_y=0.69,
    blue_x=0.15,
    blue_y=0.06,
    white_x=0.3127,
    white_y=0.329,
    max_luminance=1000.0,
    min_luminance=0.0001,
)


def test_mastering_display_to_x265_uses_encoder_units() -> None:
    # x265 wants 0.00002 units for xy and 0.0001 for luminance, and the element
    # order is G, B, R, WP — not R first.
    assert BT2020_DISPLAY.to_x265() == (
        "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    )


def test_mastering_display_to_svt_av1_uses_real_units() -> None:
    assert BT2020_DISPLAY.to_svt_av1() == (
        "G(0.265,0.69)B(0.15,0.06)R(0.68,0.32)WP(0.3127,0.329)L(1000,0.0001)"
    )


def test_video_track_hdr_and_gamut_flags() -> None:
    sdr = video_track()
    assert not sdr.is_hdr
    assert not sdr.is_wide_gamut

    pq = video_track(color_transfer="smpte2084", color_primaries="bt2020")
    assert pq.is_hdr
    assert pq.is_wide_gamut

    hlg = video_track(color_transfer="arib-std-b67")
    assert hlg.is_hdr


def test_video_track_interlacing() -> None:
    assert video_track(scan_type="interlaced").is_interlaced
    assert video_track(scan_type="Interlaced").is_interlaced
    assert not video_track(scan_type="progressive").is_interlaced
    assert not video_track(scan_type=None).is_interlaced


def test_bits_per_pixel_is_bitrate_over_pixels_per_frame() -> None:
    track = video_track()

    assert track.pixels == 1920 * 1080
    # 20 Mb/s over 1920x1080 at 23.976 fps: comfortably disc-grade.
    assert track.bits_per_pixel == pytest.approx(0.4022, abs=1e-4)


@pytest.mark.parametrize(
    "overrides",
    [{"bitrate_bps": None}, {"frame_rate": None}, {"width": None}, {"height": None}],
)
def test_bits_per_pixel_is_none_when_an_input_is_missing(overrides: dict[str, None]) -> None:
    assert video_track(**overrides).bits_per_pixel is None


@pytest.mark.parametrize(
    ("height", "label"),
    [(480, "SD"), (576, "SD"), (720, "720p"), (1080, "1080p"), (1440, "1440p"), (2160, "2160p")],
)
def test_resolution_label(height: int, label: str) -> None:
    assert SourceMedia(video=video_track(height=height)).resolution_label == label


def test_resolution_label_needs_a_video_track() -> None:
    assert SourceMedia().resolution_label is None
    assert SourceMedia(video=VideoTrack()).resolution_label is None


def test_grain_level_rank_ordering_and_clamping() -> None:
    assert GrainLevel.NONE.rank < GrainLevel.LIGHT.rank < GrainLevel.MODERATE.rank
    assert GrainLevel.MODERATE.rank < GrainLevel.HEAVY.rank < GrainLevel.EXTREME.rank
    assert GrainLevel.from_rank(2) is GrainLevel.MODERATE
    assert GrainLevel.from_rank(-5) is GrainLevel.NONE
    assert GrainLevel.from_rank(99) is GrainLevel.EXTREME


@pytest.mark.parametrize(("crf", "label"), [(27.0, "27"), (26.5, "26.5"), (30, "30")])
def test_crf_label_drops_a_pointless_decimal(crf: float, label: str) -> None:
    plan = EncoderPlan(encoder=Encoder.X265, crf=crf, preset="slow")
    assert plan.crf_label == label


def test_params_string_is_the_encoder_wire_format() -> None:
    plan = EncoderPlan(
        encoder=Encoder.SVT_AV1,
        crf=30,
        preset="5",
        params={"tune": "0", "keyint": "240", "film-grain": "12"},
    )
    assert plan.params_string == "tune=0:keyint=240:film-grain=12"


def test_estimated_size_note() -> None:
    assert (
        EncoderPlan(
            encoder=Encoder.X265, crf=21, preset="slow", estimated_bitrate_bps=12_345_678
        ).estimated_size_note
        == "~12.3 Mb/s video"
    )
    assert EncoderPlan(encoder=Encoder.X265, crf=21, preset="slow").estimated_size_note is None


@pytest.mark.parametrize(
    ("codec", "lossless"),
    [
        ("truehd", True),
        ("dts-hd ma", True),
        ("flac", True),
        ("pcm", True),
        ("dts", False),
        ("eac3", False),
        (None, False),
    ],
)
def test_audio_track_is_lossless(codec: str | None, lossless: bool) -> None:
    assert AudioTrack(codec=codec).is_lossless is lossless


def test_technical_specs_all_format_text_is_the_grain_haystack() -> None:
    specs = TechnicalSpecs(
        negative_formats=["35 mm"],
        cinematographic_processes=["Super 35"],
        printed_formats=["70 mm (blow-up)"],
        cameras=["Panaflex Camera"],
        film_lengths=["3,196 m"],
        # Deliberately not in the haystack: a sound mix or a lab says nothing about
        # grain, and matching "70 mm 6-Track" as a negative format would be wrong.
        sound_mixes=["70 mm 6-Track"],
        laboratories=["Technicolor"],
    )

    haystack = specs.all_format_text()

    assert "35 mm" in haystack
    assert "super 35" in haystack
    assert "70 mm (blow-up)" in haystack
    assert "panaflex" in haystack
    assert "6-track" not in haystack
    assert "technicolor" not in haystack


def test_technical_specs_is_empty() -> None:
    assert TechnicalSpecs().is_empty
    assert not TechnicalSpecs(runtimes=["142 min"]).is_empty


def test_movie_hit_display_title_mentions_the_original_only_when_it_differs() -> None:
    assert MovieHit(tmdb_id=1, title="Alien").display_title == "Alien"
    assert (
        MovieHit(tmdb_id=1, title="Come and See", original_title="Idi i smotri").display_title
        == "Come and See (Idi i smotri)"
    )
    assert MovieHit(tmdb_id=1, title="Alien", original_title="Alien").display_title == "Alien"


def test_movie_imdb_urls() -> None:
    with_id = Movie(tmdb_id=78, title="Blade Runner", imdb_id="tt0083658")
    assert with_id.imdb_url == "https://www.imdb.com/title/tt0083658/"
    assert with_id.imdb_technical_url == "https://www.imdb.com/title/tt0083658/technical/"

    without = Movie(tmdb_id=78, title="Blade Runner")
    assert without.imdb_url is None
    assert without.imdb_technical_url is None


def test_advice_requires_at_least_one_plan() -> None:
    # A plan-less Advice would render an empty results page; fail loudly instead.
    with pytest.raises(ValidationError, match="at least one encoder plan"):
        Advice(grain=GrainProfile(level=GrainLevel.MODERATE, confidence=0.8))


def test_advice_plan_for() -> None:
    svt = EncoderPlan(encoder=Encoder.SVT_AV1, crf=30, preset="5")
    advice = Advice(grain=GrainProfile(level=GrainLevel.MODERATE, confidence=0.8), plans=[svt])

    assert advice.plan_for(Encoder.SVT_AV1) is svt
    assert advice.plan_for(Encoder.X265) is None


def test_grain_confidence_is_a_probability() -> None:
    with pytest.raises(ValidationError):
        GrainProfile(level=GrainLevel.LIGHT, confidence=1.5)


def test_encoder_labels() -> None:
    assert Encoder.SVT_AV1.label == "AV1 (SVT-AV1)"
    assert Encoder.X265.label == "x265 (HEVC)"
