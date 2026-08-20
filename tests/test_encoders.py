"""Command and preset rendering.

Assertions are on ``shlex.split`` tokens rather than on the command string, because
the point of building these as argument lists is that quoting is the shell's problem
and not ours — a test that matched raw substrings would pass on a broken quote.
"""

from __future__ import annotations

import shlex

import pytest

from app.advice.encoders import (
    DEINTERLACE_FILTER,
    attach_commands,
    output_name,
    render_ffmpeg,
    render_handbrake,
    render_handbrake_preset,
)
from app.advice.rules import build_advice
from app.models import Advice, Encoder, EncodeRequest, EncoderPlan, GrainLevel
from app.sources import parse_source
from tests.support import fixture, media, request_for, video_track


def built(**overrides: object) -> tuple[Advice, EncodeRequest]:
    request = request_for(**overrides)
    return attach_commands(build_advice(request), request), request


def tokens(command: str) -> list[str]:
    return shlex.split(command)


def flag(command: str, name: str) -> str | None:
    """The value following ``name``, or ``None`` if the flag is absent."""
    parts = tokens(command)
    return parts[parts.index(name) + 1] if name in parts else None


# --- ffmpeg -----------------------------------------------------------------


def test_ffmpeg_command_shape() -> None:
    advice, request = built()
    plan = advice.plans[0]
    command = render_ffmpeg(plan, request)
    parts = tokens(command)

    assert parts[0] == "ffmpeg"
    assert flag(command, "-i") == "input.mkv"
    # -map 0 or the extra audio and subtitle tracks are silently dropped.
    assert flag(command, "-map") == "0"
    assert flag(command, "-c:v") == "libsvtav1"
    assert flag(command, "-crf") == "27"
    assert flag(command, "-preset") == "4"
    assert flag(command, "-pix_fmt") == "yuv420p10le"
    assert flag(command, "-c:a") == "copy"
    assert flag(command, "-c:s") == "copy"
    assert parts[-1] == "blade-runner-1982-1080p.av1.mkv"


def test_ffmpeg_x265_command_uses_its_own_flags() -> None:
    advice, request = built()
    plan = advice.plans[1]
    command = render_ffmpeg(plan, request)

    assert flag(command, "-c:v") == "libx265"
    assert flag(command, "-preset") == "slow"
    assert flag(command, "-tune") == "grain"
    assert tokens(command)[-1] == "blade-runner-1982-1080p.x265.mkv"


def test_the_params_string_is_one_argument() -> None:
    # Colons and equals signs inside one token; splitting it would hand ffmpeg
    # "keyint=240" as a filename.
    advice, request = built()
    svt, x265 = advice.plans

    svt_params = flag(render_ffmpeg(svt, request), "-svtav1-params")
    assert svt_params == svt.params_string
    assert svt_params is not None and svt_params.startswith("tune=0:keyint=240")

    assert flag(render_ffmpeg(x265, request), "-x265-params") == x265.params_string


def test_hdr_mastering_values_survive_quoting() -> None:
    advice, request = built(source=parse_source(fixture("ffprobe_uhd_hdr.json")).media)
    x265 = advice.plans[1]

    params = flag(render_ffmpeg(x265, request), "-x265-params")

    # Parentheses and commas would be eaten by a shell if the value were interpolated.
    assert params is not None
    assert "master-display=G(13250,34500)" in params
    assert "max-cll=1000,400" in params


def test_colour_signalling_is_written_to_the_container_too() -> None:
    advice, request = built(source=parse_source(fixture("ffprobe_uhd_hdr.json")).media)
    command = render_ffmpeg(advice.plans[0], request)

    # A correctly coded stream with no container tags still shows wrong colours in
    # a good number of players.
    assert flag(command, "-color_primaries") == "bt2020"
    assert flag(command, "-color_trc") == "smpte2084"
    assert flag(command, "-colorspace") == "bt2020nc"
    assert flag(command, "-color_range") == "tv"


def test_interlaced_sources_get_a_deinterlace_filter() -> None:
    advice, request = built(source=media(video=video_track(scan_type="interlaced")))
    command = render_ffmpeg(advice.plans[0], request)

    assert flag(command, "-vf") == DEINTERLACE_FILTER
    # The filter has to come before the encoder settings it feeds.
    parts = tokens(command)
    assert parts.index("-vf") < parts.index("-c:v")


def test_progressive_sources_get_no_filter() -> None:
    advice, request = built()
    assert "-vf" not in tokens(render_ffmpeg(advice.plans[0], request))


def test_a_path_with_spaces_and_quotes_stays_one_argument() -> None:
    nasty = "/media/films/Blade Runner (1982) [remux]/it's here.mkv"
    advice, request = built(input_path=nasty)

    command = render_ffmpeg(advice.plans[0], request)

    assert flag(command, "-i") == nasty
    assert (
        tokens(render_handbrake(advice.plans[0], request))[
            tokens(render_handbrake(advice.plans[0], request)).index("-i") + 1
        ]
        == nasty
    )


def test_a_plan_without_parameters_omits_the_params_flag() -> None:
    plan = EncoderPlan(encoder=Encoder.X265, crf=20, preset="slow")

    command = render_ffmpeg(plan, request_for())

    assert "-x265-params" not in tokens(command)


# --- HandBrake --------------------------------------------------------------


def test_handbrake_command_shape() -> None:
    advice, request = built()
    command = render_handbrake(advice.plans[0], request)

    assert tokens(command)[0] == "HandBrakeCLI"
    assert flag(command, "-o") == "blade-runner-1982-1080p.av1.mkv"
    assert flag(command, "--format") == "av_mkv"
    # HandBrake picks depth by encoder name rather than by pixel format.
    assert flag(command, "-e") == "svt_av1_10bit"
    assert flag(command, "-q") == "27"
    assert flag(command, "--encoder-preset") == "4"
    assert flag(command, "--encopts") == advice.plans[0].params_string


def test_handbrake_passes_every_track_through_untouched() -> None:
    advice, request = built()
    command = render_handbrake(advice.plans[0], request)
    parts = tokens(command)

    # Left to itself HandBrake keeps one AAC track and the "best" subtitle, which
    # is not what anyone re-encoding a remux wants.
    assert "--all-audio" in parts
    assert flag(command, "--aencoder") == "copy"
    assert flag(command, "--audio-fallback") == "ac3"
    assert "--all-subtitles" in parts
    assert flag(command, "--subtitle-lang-list") == "any"
    assert "--markers" in parts


def test_handbrake_never_crops_silently() -> None:
    advice, request = built()

    # Even though the advice says this film is letterboxed: cropping changes the
    # framing, so it stays the user's decision.
    assert any("letterboxed" in note for note in advice.notes)
    assert flag(render_handbrake(advice.plans[0], request), "--crop-mode") == "none"


def test_handbrake_x265_tune_is_passed_separately() -> None:
    advice, request = built()
    command = render_handbrake(advice.plans[1], request)

    assert flag(command, "-e") == "x265_10bit"
    assert flag(command, "--encoder-tune") == "grain"


def test_handbrake_deinterlaces_with_comb_detection() -> None:
    advice, request = built(source=media(video=video_track(scan_type="interlaced")))
    parts = tokens(render_handbrake(advice.plans[0], request))

    assert "--comb-detect" in parts
    assert "--decomb" in parts


# --- presets ----------------------------------------------------------------


def test_handbrake_preset_is_importable_shaped() -> None:
    advice, request = built()
    document = render_handbrake_preset(advice, request, advice.plans[0])

    assert document["VersionMajor"] == 1
    assert len(document["PresetList"]) == 1
    preset = document["PresetList"][0]
    assert preset["PresetName"] == "graindamage — Blade Runner (AV1 (SVT-AV1))"
    assert preset["Type"] == 1
    assert preset["FileFormat"] == "av_mkv"
    assert preset["VideoEncoder"] == "svt_av1_10bit"
    assert preset["VideoQualityType"] == 2
    assert preset["VideoQualitySlider"] == 27.0
    assert preset["VideoPreset"] == "4"
    assert preset["VideoOptionExtra"] == advice.plans[0].params_string
    assert preset["PictureWidth"] == 1920
    assert preset["PictureHeight"] == 1080


def test_preset_description_carries_the_reasoning() -> None:
    advice, request = built()
    preset = render_handbrake_preset(advice, request, advice.plans[1])["PresetList"][0]

    assert preset["PresetDescription"] == ("moderate grain · Super 35 mm · CRF 20 · preset slow")
    assert preset["VideoTune"] == "grain"


def test_preset_name_is_truncated_rather_than_rejected() -> None:
    advice, request = built(movie=None)
    preset = render_handbrake_preset(advice, request, advice.plans[0])["PresetList"][0]

    assert preset["PresetName"] == "graindamage — source (AV1 (SVT-AV1))"
    assert len(preset["PresetName"]) <= 80


def test_preset_enables_autocrop_only_when_the_advice_says_letterboxed() -> None:
    letterboxed, request = built()
    preset = render_handbrake_preset(letterboxed, request, letterboxed.plans[0])["PresetList"][0]
    assert preset["PictureCropMode"] == 2

    # A file already as wide as IMDb's 2.39:1 has no bars to find.
    scope, request = built(source=media(video=video_track(display_aspect_ratio="2.39:1")))
    assert not any("letterboxed" in note for note in scope.notes)
    preset = render_handbrake_preset(scope, request, scope.plans[0])["PresetList"][0]
    assert preset["PictureCropMode"] == 0


def test_preset_deinterlace_filter_follows_the_source() -> None:
    progressive, request = built()
    preset = render_handbrake_preset(progressive, request, progressive.plans[0])["PresetList"][0]
    assert preset["PictureDeinterlaceFilter"] == "off"
    assert preset["PictureCombDetectPreset"] == "off"

    interlaced, request = built(source=media(video=video_track(scan_type="interlaced")))
    preset = render_handbrake_preset(interlaced, request, interlaced.plans[0])["PresetList"][0]
    assert preset["PictureDeinterlaceFilter"] == "decomb"
    assert preset["PictureCombDetectPreset"] == "default"


def test_preset_omits_dimensions_it_does_not_know() -> None:
    advice, request = built(source=media(video=video_track(width=None, height=None)))
    preset = render_handbrake_preset(advice, request, advice.plans[0])["PresetList"][0]

    assert "PictureWidth" not in preset
    assert "PictureHeight" not in preset


def test_preset_copies_lossless_audio_by_default() -> None:
    advice, request = built()
    preset = render_handbrake_preset(advice, request, advice.plans[0])["PresetList"][0]

    assert "copy:truehd" in preset["AudioCopyMask"]
    assert "copy:dtshd" in preset["AudioCopyMask"]
    assert preset["AudioList"][0]["AudioEncoder"] == "copy"
    assert preset["AudioEncoderFallback"] == "ac3"


# --- naming and attachment --------------------------------------------------


@pytest.mark.parametrize(("encoder", "suffix"), [(Encoder.SVT_AV1, "av1"), (Encoder.X265, "x265")])
def test_output_name_marks_which_encoder_made_the_file(encoder: Encoder, suffix: str) -> None:
    plan = EncoderPlan(encoder=encoder, crf=20, preset="slow")

    assert output_name(request_for(), plan) == f"blade-runner-1982-1080p.{suffix}.mkv"


def test_attach_commands_fills_in_both_front_ends_for_every_plan() -> None:
    request = request_for()
    advice = build_advice(request)

    # The rules engine knows nothing about command lines; rendering is a later step.
    assert all(plan.ffmpeg_command == "" for plan in advice.plans)

    attach_commands(advice, request)

    for plan in advice.plans:
        assert plan.ffmpeg_command.startswith("ffmpeg ")
        assert plan.handbrake_command.startswith("HandBrakeCLI ")


def test_the_two_front_ends_agree_on_quality_and_preset() -> None:
    # Different flags, same encode: -crf/-q and -preset/--encoder-preset.
    advice, request = built(grain_override=GrainLevel.HEAVY)

    for plan in advice.plans:
        ffmpeg = render_ffmpeg(plan, request)
        handbrake = render_handbrake(plan, request)
        assert flag(ffmpeg, "-crf") == flag(handbrake, "-q") == plan.crf_label
        assert flag(ffmpeg, "-preset") == flag(handbrake, "--encoder-preset") == plan.preset
        assert flag(ffmpeg, "-tune") == flag(handbrake, "--encoder-tune") == plan.tune
