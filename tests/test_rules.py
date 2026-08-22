"""The deterministic baseline: the numbers, and the reasons attached to them.

These tests assert the *arithmetic*, not just that a plan came back: an anchor plus
named signed adjustments is the whole contract this module offers the UI, and a
silently drifting anchor is exactly the bug a test suite should catch.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.advice.rules import (
    _resolution_class,
    build_advice,
    estimate_bitrate,
    parse_aspect_ratio,
)
from app.models import (
    AudioTrack,
    Encoder,
    EncoderPlan,
    GrainLevel,
    SizePreference,
    SourceMedia,
    SpeedPreference,
    TechnicalSpecs,
)
from app.sources import parse_source
from tests.support import fixture, media, movie, request_for, specs, video_track


def plans(**overrides: Any) -> tuple[EncoderPlan, EncoderPlan]:
    """The two baseline plans for a request, in (SVT-AV1, x265) order."""
    advice = build_advice(request_for(**overrides))
    svt = advice.plan_for(Encoder.SVT_AV1)
    x265 = advice.plan_for(Encoder.X265)
    assert svt is not None and x265 is not None
    return svt, x265


# --- resolution classes -----------------------------------------------------


@pytest.mark.parametrize(
    ("height", "expected"),
    [(480, "SD"), (576, "SD"), (720, "720p"), (1080, "1080p"), (1200, "1440p"), (2160, "2160p")],
)
def test_resolution_class(height: int, expected: str) -> None:
    assert _resolution_class(video_track(height=height)) == expected


def test_resolution_class_defaults_to_1080p_when_unknown() -> None:
    # The commonest case, and the one whose anchor is least wrong if guessed.
    assert _resolution_class(None) == "1080p"
    assert _resolution_class(video_track(height=None)) == "1080p"


# --- the baseline plan ------------------------------------------------------


def test_both_plans_are_produced_with_the_expected_anchors() -> None:
    advice = build_advice(request_for())

    assert [plan.encoder for plan in advice.plans] == [Encoder.SVT_AV1, Encoder.X265]
    svt, x265 = advice.plans
    # 1080p anchors 28 and 21, each less 1 for moderate grain.
    assert svt.crf == 27.0
    assert x265.crf == 20.0
    assert svt.preset == "4"
    assert x265.preset == "slow"


def test_the_first_adjustment_is_the_resolution_baseline() -> None:
    svt, x265 = plans()

    assert svt.adjustments[0].label == "1080p baseline"
    assert svt.adjustments[0].delta == 28.0
    assert x265.adjustments[0].delta == 21.0
    # Anchor plus every delta is exactly the CRF shown: no hidden rounding.
    assert sum(a.delta for a in svt.adjustments) == svt.crf
    assert sum(a.delta for a in x265.adjustments) == x265.crf


def test_moderate_grain_costs_one_crf_point_on_both_encoders() -> None:
    svt, x265 = plans()

    for plan in (svt, x265):
        grain_adjustment = next(a for a in plan.adjustments if a.label == "moderate grain")
        assert grain_adjustment.delta == -1.0


def test_heavy_grain_costs_x265_more_than_av1() -> None:
    # AV1 can synthesise grain; x265 has to code every particle, so it pays more.
    svt, x265 = plans(grain_override=GrainLevel.HEAVY)

    assert svt.crf == 27.0
    assert x265.crf == 19.0


def test_extreme_grain_costs_x265_three_points() -> None:
    svt, x265 = plans(grain_override=GrainLevel.EXTREME)

    assert svt.crf == 27.0
    assert x265.crf == 18.0


def test_grainless_digital_sources_get_a_crf_point_back() -> None:
    svt, x265 = plans(grain_override=GrainLevel.NONE)

    assert svt.crf == 29.0
    assert x265.crf == 22.0


def test_light_grain_moves_nothing() -> None:
    svt, x265 = plans(grain_override=GrainLevel.LIGHT)

    assert svt.crf == 28.0
    assert x265.crf == 21.0
    assert not any(a.label.endswith("grain") for a in svt.adjustments)


# --- source quality ---------------------------------------------------------


@pytest.mark.parametrize(
    ("bitrate", "delta"),
    [
        (20_000_000, 0.0),  # 0.402 bpp — disc
        (5_000_000, 0.5),  # 0.101 bpp — good web release
        (3_000_000, 1.5),  # 0.060 bpp — modestly compressed
        (1_500_000, 2.5),  # 0.030 bpp — heavily compressed
    ],
)
def test_a_weaker_source_raises_the_crf(bitrate: int, delta: float) -> None:
    svt, _ = plans(source=media(video=video_track(bitrate_bps=bitrate)))

    applied = next((a.delta for a in svt.adjustments if a.label == "source quality"), 0.0)
    assert applied == delta
    assert svt.crf == 27.0 + delta


def test_a_disc_grade_source_is_explained_rather_than_adjusted() -> None:
    svt, _ = plans()

    assert not any(a.label == "source quality" for a in svt.adjustments)
    assert any("disc-grade" in reason for reason in svt.rationale)


def test_an_unknown_source_bitrate_is_admitted() -> None:
    svt, _ = plans(source=media(video=video_track(bitrate_bps=None)))

    assert svt.crf == 27.0
    assert any("bitrate unknown" in reason for reason in svt.rationale)


def test_a_heavily_compressed_source_is_warned_about_once() -> None:
    advice = build_advice(request_for(source=media(video=video_track(bitrate_bps=1_500_000))))

    # The warning belongs to the request, not to each of the two plans.
    matching = [w for w in advice.warnings if "already heavily compressed" in w]
    assert len(matching) == 1


# --- HDR --------------------------------------------------------------------


def test_hdr_earns_a_crf_point_on_both_plans() -> None:
    advice = build_advice(request_for(source=parse_source(fixture("ffprobe_uhd_hdr.json")).media))
    svt, x265 = advice.plans

    # 2160p anchors 32 and 23, each less 1 for grain and 1 for HDR.
    assert svt.crf == 30.0
    assert x265.crf == 21.0
    for plan in (svt, x265):
        assert next(a for a in plan.adjustments if a.label == "HDR").delta == -1.0


def test_2160p_takes_one_preset_step_faster_unless_quality_was_asked_for() -> None:
    uhd = parse_source(fixture("ffprobe_uhd_hdr.json")).media

    balanced, _ = plans(source=uhd)
    quality, _ = plans(source=uhd, speed=SpeedPreference.QUALITY)

    assert balanced.preset == "5"
    assert any("keeps encode time sane" in reason for reason in balanced.rationale)
    assert quality.preset == "3"


# --- speed and size preferences ---------------------------------------------


@pytest.mark.parametrize(
    ("speed", "svt_preset", "x265_preset"),
    [
        (SpeedPreference.QUALITY, "3", "slower"),
        (SpeedPreference.BALANCED, "4", "slow"),
        (SpeedPreference.FAST, "6", "medium"),
    ],
)
def test_speed_preference_picks_the_preset(
    speed: SpeedPreference, svt_preset: str, x265_preset: str
) -> None:
    svt, x265 = plans(speed=speed)

    assert svt.preset == svt_preset
    assert x265.preset == x265_preset


def test_heavy_grain_slows_both_presets_by_one_step() -> None:
    svt, x265 = plans(grain_override=GrainLevel.HEAVY)

    assert svt.preset == "3"
    assert x265.preset == "slower"
    assert any("hold grain together" in reason for reason in svt.rationale)
    assert any("rate-distortion search" in reason for reason in x265.rationale)


def test_an_explicit_fast_request_is_not_quietly_slowed_down() -> None:
    svt, x265 = plans(grain_override=GrainLevel.HEAVY, speed=SpeedPreference.FAST)

    assert svt.preset == "6"
    assert x265.preset == "medium"


@pytest.mark.parametrize(
    ("size", "delta"),
    [
        (SizePreference.ARCHIVAL, -2.0),
        (SizePreference.BALANCED, 0.0),
        (SizePreference.COMPACT, 2.0),
    ],
)
def test_size_preference_moves_the_crf(size: SizePreference, delta: float) -> None:
    svt, x265 = plans(size=size)

    assert svt.crf == 27.0 + delta
    assert x265.crf == 20.0 + delta


def test_the_crf_is_clamped_to_each_encoders_sane_range() -> None:
    # SD + extreme grain + archival + a starved source pushes x265 below its floor.
    svt, x265 = plans(
        source=media(video=video_track(width=720, height=480, bitrate_bps=20_000_000)),
        grain_override=GrainLevel.EXTREME,
        size=SizePreference.ARCHIVAL,
    )

    assert svt.crf == 21.0  # 24 - 1 - 2, still inside 15..45
    assert x265.crf == 14.0  # 19 - 3 - 2 = 14, exactly the floor

    compact = build_advice(
        request_for(
            source=media(video=video_track(height=2160, width=3840, bitrate_bps=1_000_000)),
            grain_override=GrainLevel.NONE,
            size=SizePreference.COMPACT,
        )
    )
    x265_compact = compact.plan_for(Encoder.X265)
    assert x265_compact is not None
    # 23 + 1 + 2.5 + 2 = 28.5, under the 32 ceiling, so no clamp — but the AV1 side
    # would be 37.5 and is likewise legal. Both stay inside their limits.
    assert x265_compact.crf == 28.5


# --- parameters -------------------------------------------------------------


def test_keyframe_interval_is_ten_seconds_of_the_source_frame_rate() -> None:
    svt, x265 = plans()

    assert svt.params["keyint"] == "240"
    assert x265.params["keyint"] == "240"
    # x265 wants a minimum too, or scene cuts alone decide keyframe placement.
    assert x265.params["min-keyint"] == "24"


def test_an_unknown_frame_rate_falls_back_to_24_fps() -> None:
    svt, _ = plans(source=media(video=video_track(frame_rate=None)))

    assert svt.params["keyint"] == "240"


def test_svt_av1_always_tunes_for_subjective_quality() -> None:
    svt, _ = plans()

    # tune=1 (the default) optimises PSNR, which prefers smoothing grain away.
    assert svt.params["tune"] == "0"
    assert svt.params["scd"] == "1"


def test_synthesis_is_a_floor_until_grain_becomes_the_image() -> None:
    light, _ = plans(grain_override=GrainLevel.LIGHT)
    moderate, _ = plans(grain_override=GrainLevel.MODERATE)
    heavy, _ = plans(grain_override=GrainLevel.HEAVY)
    extreme, _ = plans(grain_override=GrainLevel.EXTREME)

    # Below heavy the film's own grain is still coded, and synthesis only fills in what
    # the quantiser flattened — which is what film-grain-denoise=0 buys.
    assert (light.params["film-grain"], light.params["film-grain-denoise"]) == ("4", "0")
    assert (moderate.params["film-grain"], moderate.params["film-grain-denoise"]) == ("8", "0")
    # At these levels grain *is* the image, so it is removed, coded clean, and put back.
    assert (heavy.params["film-grain"], heavy.params["film-grain-denoise"]) == ("12", "1")
    assert (extreme.params["film-grain"], extreme.params["film-grain-denoise"]) == ("20", "1")

    assert any("film-grain-denoise=0" in reason for reason in moderate.rationale)
    assert any("film-grain-denoise=1" in reason for reason in heavy.rationale)


def test_no_synthetic_grain_over_a_source_that_may_not_have_any() -> None:
    digital, _ = plans(specs=specs(negative_formats=["Digital"], cinematographic_processes=[]))
    assert "film-grain" not in digital.params

    # Light grain, but nothing said whether it came off a negative: adding grain here
    # would be inventing it.
    undecided = build_advice(request_for(movie=None, specs=specs_with_nothing()))
    svt = undecided.plan_for(Encoder.SVT_AV1)
    assert svt is not None
    assert undecided.grain.level is GrainLevel.LIGHT
    assert "film-grain" not in svt.params

    # A level set by hand does count as film: the user is saying there is grain to keep.
    forced, _ = plans(specs=specs_with_nothing(), grain_override=GrainLevel.MODERATE)
    assert forced.params["film-grain"] == "8"


def test_loop_restoration_is_off_only_where_the_real_grain_is_coded() -> None:
    moderate, _ = plans(grain_override=GrainLevel.MODERATE)
    heavy, _ = plans(grain_override=GrainLevel.HEAVY)

    assert moderate.params["enable-restoration"] == "0"
    # Denoised first, so there is no grain left for the Wiener filter to eat.
    assert "enable-restoration" not in heavy.params


def test_film_grain_never_lands_above_the_preset_svt_av1_warns_about() -> None:
    uhd = media(video=video_track(width=3840, height=2160))
    grainy, _ = plans(source=uhd, speed=SpeedPreference.FAST, grain_override=GrainLevel.HEAVY)
    clean, _ = plans(
        source=uhd,
        speed=SpeedPreference.FAST,
        specs=specs(negative_formats=["Digital"], cinematographic_processes=[]),
    )

    # preset 6 for fast, +1 for 2160p — but SVT-AV1 calls film-grain above 6 a
    # debugging-only compute overhead, and the grain matters more than the last step.
    assert grainy.params["film-grain"] == "12"
    assert grainy.preset == "6"
    assert any("preset 6" in reason for reason in grainy.rationale)
    # Nothing to protect, nothing to cap.
    assert clean.preset == "7"


def test_the_size_estimate_only_falls_where_the_grain_is_really_removed() -> None:
    moderate, _ = plans(grain_override=GrainLevel.MODERATE)
    heavy, _ = plans(grain_override=GrainLevel.HEAVY)
    assert moderate.estimated_bitrate_bps and heavy.estimated_bitrate_bps

    # Heavier grain, same CRF, yet a smaller estimate — because heavy is denoised and
    # re-synthesised, while moderate still codes every particle.
    assert heavy.crf == moderate.crf
    assert heavy.estimated_bitrate_bps < moderate.estimated_bitrate_bps
    assert moderate.estimated_bitrate_bps == estimate_bitrate(
        Encoder.SVT_AV1,
        moderate.crf,
        video_track(),
        GrainLevel.MODERATE,
        synthesised=False,
    )


def test_x265_tunes_for_grain_from_moderate_upwards() -> None:
    light, light_x265 = plans(grain_override=GrainLevel.LIGHT)
    _, moderate_x265 = plans(grain_override=GrainLevel.MODERATE)
    _, heavy_x265 = plans(grain_override=GrainLevel.HEAVY)

    assert light_x265.tune is None
    # Without tune grain, SAO is turned off by hand: it is a smoothing filter.
    assert light_x265.params["sao"] == "0"
    assert light_x265.params["rc-lookahead"] == "48"
    assert light is not None

    assert moderate_x265.tune == "grain"
    assert moderate_x265.params["aq-strength"] == "0.9"
    assert moderate_x265.params["rc-lookahead"] == "60"
    # tune grain sets aq-mode 0; auto-variance AQ is put back deliberately.
    assert moderate_x265.params["aq-mode"] == "3"
    assert "sao" not in moderate_x265.params

    assert heavy_x265.params["aq-strength"] == "0.8"


def test_the_two_things_tune_grain_does_not_do_are_set_by_hand() -> None:
    _, moderate_x265 = plans(grain_override=GrainLevel.MODERATE)

    # Measured against x265 4.2: tune grain leaves qcomp at 0.60 and the deblocking
    # offsets at 0:0, whatever its reputation says.
    assert moderate_x265.params["qcomp"] == "0.8"
    # One value, not "-1:-1" — params_string joins on ':', and libx265 would read the
    # second half as a parameter name and drop the parameter after it.
    assert moderate_x265.params["deblock"] == "-1"
    # The whole string has to survive the join: one "=" per colon-separated segment.
    assert all(part.count("=") == 1 for part in moderate_x265.params_string.split(":"))
    assert moderate_x265.params["strong-intra-smoothing"] == "0"
    assert not any("raises qcomp" in reason for reason in moderate_x265.rationale)
    assert any("psy-rdoq to 10" in reason for reason in moderate_x265.rationale)


def test_a_coarse_stock_negative_is_tuned_harder_than_a_late_one() -> None:
    old_svt, old_x265 = plans(movie=movie(year=1962))
    late_svt, late_x265 = plans(movie=movie(year=1998))

    # The era moves the tuning, not the grain level, so the CRF ladder is untouched.
    assert old_svt.crf == late_svt.crf
    assert old_svt.params["film-grain"] == "12"
    assert late_svt.params["film-grain"] == "8"
    assert old_svt.params["qp-scale-compress-strength"] == "2"
    assert "qp-scale-compress-strength" not in late_svt.params
    assert (old_x265.params["qcomp"], old_x265.params["deblock"]) == ("0.85", "-2")
    assert (late_x265.params["qcomp"], late_x265.params["deblock"]) == ("0.8", "-1")
    assert any("1962" in reason for reason in old_svt.rationale)


def test_the_era_ladder_works_from_a_parsed_year_alone() -> None:
    svt, x265 = plans(movie=None, fallback_year=1965)

    assert svt.params["qp-scale-compress-strength"] == "2"
    assert x265.params["deblock"] == "-2"


def test_a_digital_film_gets_no_era_tuning_whatever_its_year() -> None:
    # Nothing here is photochemical, so "old" says nothing about grain.
    svt, x265 = plans(
        movie=movie(year=1968),
        specs=specs(negative_formats=["Digital"], cinematographic_processes=[]),
    )

    assert "qp-scale-compress-strength" not in svt.params
    assert "qcomp" not in x265.params


def test_sdr_colour_signalling_reaches_both_encoders_in_their_own_dialect() -> None:
    svt, x265 = plans()

    # SVT-AV1 takes H.273 code points...
    assert svt.params["color-primaries"] == "1"
    assert svt.params["transfer-characteristics"] == "1"
    assert svt.params["matrix-coefficients"] == "1"
    assert svt.params["color-range"] == "0"
    # ...x265 takes the names.
    assert x265.params["colorprim"] == "bt709"
    assert x265.params["transfer"] == "bt709"
    assert x265.params["colormatrix"] == "bt709"
    assert x265.params["range"] == "limited"


def test_full_range_is_signalled_as_such() -> None:
    svt, x265 = plans(source=media(video=video_track(color_range="pc")))

    assert svt.params["color-range"] == "1"
    assert x265.params["range"] == "full"


def test_hdr10_metadata_reaches_both_encoders() -> None:
    uhd = parse_source(fixture("ffprobe_uhd_hdr.json")).media
    svt, x265 = plans(source=uhd)

    assert svt.params["color-primaries"] == "9"
    assert svt.params["transfer-characteristics"] == "16"
    assert svt.params["matrix-coefficients"] == "9"
    assert svt.params["mastering-display"] == (
        "G(0.265,0.69)B(0.15,0.06)R(0.68,0.32)WP(0.3127,0.329)L(1000,0.0001)"
    )
    assert svt.params["content-light"] == "1000,400"

    assert x265.params["hdr10"] == "1"
    assert x265.params["hdr10-opt"] == "1"
    # Without repeat-headers the HDR metadata only appears once, and a player that
    # seeks past the first frame shows the film in the wrong colours.
    assert x265.params["repeat-headers"] == "1"
    assert x265.params["master-display"] == (
        "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    )
    assert x265.params["max-cll"] == "1000,400"


def test_no_video_track_means_no_colour_parameters() -> None:
    svt, x265 = plans(source=SourceMedia())

    assert "color-primaries" not in svt.params
    assert "colorprim" not in x265.params


def test_a_bit_depth_override_is_applied_to_the_source() -> None:
    request = request_for(source=media(video=video_track(bit_depth=None)), bit_depth_override=10)

    build_advice(request)

    assert request.source.video is not None
    assert request.source.video.bit_depth == 10


# --- estimates --------------------------------------------------------------


def test_estimated_bitrates_show_av1_costing_roughly_half() -> None:
    svt, x265 = plans()

    assert svt.estimated_bitrate_bps == 3_884_115
    assert x265.estimated_bitrate_bps == 7_311_276
    assert svt.estimated_bitrate_bps < x265.estimated_bitrate_bps / 1.5


def test_grain_synthesis_lowers_the_estimate() -> None:
    coded = estimate_bitrate(
        Encoder.SVT_AV1, 27, video_track(), GrainLevel.HEAVY, synthesised=False
    )
    synthesised = estimate_bitrate(
        Encoder.SVT_AV1, 27, video_track(), GrainLevel.HEAVY, synthesised=True
    )
    assert coded is not None and synthesised is not None
    assert synthesised < coded


def test_a_higher_crf_estimates_a_smaller_file() -> None:
    low = estimate_bitrate(Encoder.X265, 18, video_track(), GrainLevel.MODERATE, synthesised=False)
    high = estimate_bitrate(Encoder.X265, 24, video_track(), GrainLevel.MODERATE, synthesised=False)
    assert low is not None and high is not None
    assert high < low


def test_no_estimate_without_a_frame_size() -> None:
    assert estimate_bitrate(Encoder.X265, 20, None, GrainLevel.LIGHT, synthesised=False) is None
    assert (
        estimate_bitrate(
            Encoder.X265, 20, video_track(width=None), GrainLevel.LIGHT, synthesised=False
        )
        is None
    )


# --- cross-cutting notes ----------------------------------------------------


def test_letterboxing_is_detected_from_imdbs_ratio() -> None:
    # IMDb says 2.39:1, the file is 16:9 — so there are bars top and bottom.
    advice = build_advice(request_for())

    assert any("letterboxed" in note for note in advice.notes)
    assert any("cropdetect" in note for note in advice.notes)


def test_an_already_cropped_file_is_told_not_to_crop_again() -> None:
    advice = build_advice(
        request_for(
            specs=specs(aspect_ratios=["1.85 : 1"]),
            source=media(video=video_track(width=1920, height=816, display_aspect_ratio=None)),
        )
    )

    assert any("Do not crop again" in note for note in advice.notes)


def test_a_matching_ratio_says_nothing_about_cropping() -> None:
    advice = build_advice(
        request_for(
            specs=specs(aspect_ratios=["1.78 : 1"]),
            source=media(video=video_track(display_aspect_ratio="16:9")),
        )
    )

    assert not any("crop" in note for note in advice.notes)


def test_interlaced_sources_are_warned_about() -> None:
    advice = build_advice(request_for(source=media(video=video_track(scan_type="interlaced"))))

    assert any("Deinterlace before encoding" in warning for warning in advice.warnings)


def test_an_av1_source_is_told_that_re_encoding_is_pointless() -> None:
    advice = build_advice(request_for(source=media(video=video_track(codec="av1"))))

    assert any("already AV1" in warning for warning in advice.warnings)


def test_high_chroma_sources_are_told_the_plans_assume_420() -> None:
    advice = build_advice(request_for(source=media(video=video_track(chroma_subsampling="4:4:4"))))

    assert any("4:4:4" in note for note in advice.notes)


def test_dolby_vision_and_hdr10_plus_losses_are_stated() -> None:
    dv = build_advice(request_for(source=media(video=video_track(dolby_vision=True))))
    plus = build_advice(request_for(source=media(video=video_track(hdr10_plus=True))))

    # Losing the RPU silently is the kind of thing you notice a week later.
    assert any("dovi_tool" in warning for warning in dv.warnings)
    assert any("hdr10plus_tool" in note for note in plus.notes)


def test_lossless_audio_is_flagged_for_stream_copying() -> None:
    advice = build_advice(request_for())

    assert any("-c:a copy" in note for note in advice.notes)


def test_lossy_audio_alone_earns_no_note() -> None:
    advice = build_advice(
        request_for(source=media(audio=[AudioTrack(index=1, codec="ac3", channels=6)]))
    )

    assert not any("-c:a copy" in note for note in advice.notes)


def test_a_guessed_grain_level_is_declared_a_guess() -> None:
    advice = build_advice(
        request_for(specs=specs(negative_formats=[], cinematographic_processes=[]))
    )

    assert advice.grain.confidence < 0.5
    assert any("guessed at, not established" in warning for warning in advice.warnings)


def test_a_confident_grain_level_earns_no_warning() -> None:
    advice = build_advice(request_for())

    assert advice.grain.confidence >= 0.5
    assert not any("guessed at" in warning for warning in advice.warnings)


def test_a_request_with_no_video_track_still_returns_usable_defaults() -> None:
    advice = build_advice(request_for(source=SourceMedia()))

    assert len(advice.plans) == 2
    assert advice.plans[0].crf == 27.0  # the 1080p assumption
    assert any("No video track was described" in warning for warning in advice.warnings)
    # Nothing about the file is known, so nothing about the file is claimed.
    assert advice.notes == []


def test_the_summary_names_the_film_the_class_and_the_origin() -> None:
    advice = build_advice(request_for())

    # The rules always write one, even though the model permits None for a model answer.
    assert advice.summary is not None
    assert advice.summary.startswith("Blade Runner: 1080p source from Super 35 mm, moderate grain")


def test_the_summary_copes_without_a_film() -> None:
    advice = build_advice(
        request_for(movie=None, specs=specs(negative_formats=[], cinematographic_processes=[]))
    )

    assert advice.summary is not None
    assert advice.summary.startswith("This source: 1080p source from an unknown origin format")


# --- the fallback year ------------------------------------------------------


def test_a_fallback_year_stands_in_for_a_film_nobody_looked_up() -> None:
    """The CLI reads a year out of the filename; with no film, it is all grain has."""
    advice = build_advice(request_for(movie=None, specs=specs_with_nothing(), fallback_year=1965))

    assert advice.grain.level is GrainLevel.MODERATE
    assert "1965" in " ".join(advice.grain.reasons)


def test_a_looked_up_year_beats_a_parsed_one() -> None:
    advice = build_advice(
        request_for(movie=movie(year=1965), specs=specs_with_nothing(), fallback_year=2019)
    )

    assert "1965" in " ".join(advice.grain.reasons)


def specs_with_nothing() -> TechnicalSpecs:
    """Specs that say nothing about format, so the year is the only thing left."""
    return specs(negative_formats=[], cinematographic_processes=[], printed_formats=[])


# --- aspect ratios ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2.39 : 1", 2.39),
        ("16:9", pytest.approx(1.7778, abs=1e-4)),
        ("1.85 : 1 (intended)", 1.85),
        ("2.35/1", 2.35),
        ("1920x1080", pytest.approx(1.7778, abs=1e-4)),
        ("1.85", 1.85),
        ("2.39 : 0", None),
        ("open matte", None),
    ],
)
def test_parse_aspect_ratio(text: str, expected: float | None) -> None:
    assert parse_aspect_ratio(text) == expected
