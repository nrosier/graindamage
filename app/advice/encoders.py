"""Turn a plan into something you can paste into a terminal.

Two front-ends for the same two encoders, because people use both: FFmpeg for the
copy-paste crowd and HandBrake for everyone who wants a GUI queue. The parameters are
identical; only the spelling differs.

Commands are assembled as argument *lists* and joined with :func:`shlex.join`, never
by string interpolation. HDR mastering-display values contain parentheses and commas
that a shell would otherwise eat, and the input path is user-supplied — quoting it
correctly is not optional.
"""

from __future__ import annotations

import shlex
from typing import Any

from app.models import Advice, Encoder, EncodeRequest, EncoderPlan

FFMPEG_ENCODER: dict[Encoder, str] = {
    Encoder.SVT_AV1: "libsvtav1",
    Encoder.X265: "libx265",
}

# HandBrake exposes depth as a separate encoder rather than a pixel format.
HANDBRAKE_ENCODER: dict[Encoder, str] = {
    Encoder.SVT_AV1: "svt_av1_10bit",
    Encoder.X265: "x265_10bit",
}

PARAMS_FLAG: dict[Encoder, str] = {
    Encoder.SVT_AV1: "-svtav1-params",
    Encoder.X265: "-x265-params",
}

OUTPUT_SUFFIX: dict[Encoder, str] = {
    Encoder.SVT_AV1: "av1",
    Encoder.X265: "x265",
}

# bwdif over yadif: better vertical detail, and send_field keeps the full frame rate,
# which matters because interlaced sources are the ones with motion to preserve.
DEINTERLACE_FILTER = "bwdif=mode=send_field"


def output_name(request: EncodeRequest, plan: EncoderPlan) -> str:
    return f"{request.output_stem}.{OUTPUT_SUFFIX[plan.encoder]}.mkv"


def render_ffmpeg(plan: EncoderPlan, request: EncodeRequest) -> str:
    video = request.source.video
    args: list[str] = ["ffmpeg", "-i", request.input_path, "-map", "0"]

    if video is not None and video.is_interlaced:
        args += ["-vf", DEINTERLACE_FILTER]

    args += [
        "-c:v",
        FFMPEG_ENCODER[plan.encoder],
        "-crf",
        plan.crf_label,
        "-preset",
        plan.preset,
    ]
    if plan.tune:
        args += ["-tune", plan.tune]
    args += ["-pix_fmt", plan.pixel_format]

    if plan.params:
        args += [PARAMS_FLAG[plan.encoder], plan.params_string]

    # Tag the container as well as the bitstream: a correct stream with no container
    # signalling still plays back with the wrong colours in a lot of players.
    if video is not None:
        if video.color_primaries:
            args += ["-color_primaries", video.color_primaries]
        if video.color_transfer:
            args += ["-color_trc", video.color_transfer]
        if video.color_matrix:
            args += ["-colorspace", video.color_matrix]
        if video.color_range:
            args += ["-color_range", video.color_range]

    args += ["-c:a", "copy", "-c:s", "copy", output_name(request, plan)]
    return shlex.join(args)


def render_handbrake(plan: EncoderPlan, request: EncodeRequest) -> str:
    video = request.source.video
    args: list[str] = [
        "HandBrakeCLI",
        "-i",
        request.input_path,
        "-o",
        output_name(request, plan),
        "--format",
        "av_mkv",
        "-e",
        HANDBRAKE_ENCODER[plan.encoder],
        "-q",
        plan.crf_label,
        "--encoder-preset",
        plan.preset,
    ]
    if plan.tune:
        args += ["--encoder-tune", plan.tune]
    if plan.params:
        args += ["--encopts", plan.params_string]

    args += [
        # HandBrake defaults to one AAC track and the "best" subtitle; for an archive
        # re-encode you almost always want everything passed through untouched.
        "--all-audio",
        "--aencoder",
        "copy",
        "--audio-copy-mask",
        "truehd,dtshd,dts,ac3,eac3,flac,aac,mp3",
        "--audio-fallback",
        "ac3",
        "--all-subtitles",
        "--subtitle-lang-list",
        "any",
        "--markers",
    ]

    if video is not None and video.is_interlaced:
        args += ["--comb-detect", "--decomb"]
    # Cropping is a judgement call about the film's intended framing, so it is never
    # applied silently; the notes explain when to switch this to "auto".
    args += ["--crop-mode", "none"]

    return shlex.join(args)


def render_handbrake_preset(
    advice: Advice, request: EncodeRequest, plan: EncoderPlan
) -> dict[str, Any]:
    """A HandBrake ``.json`` preset (schema as of HandBrake 1.7).

    Importable through *Presets → Import from file*, which beats retyping an
    ``--encopts`` string into the GUI.
    """
    video = request.source.video
    title = request.movie.title if request.movie else "source"
    name = f"graindamage — {title} ({plan.encoder.label})"

    preset: dict[str, Any] = {
        "PresetName": name[:80],
        "PresetDescription": (
            f"{advice.grain.level.value} grain"
            + (f" · {advice.grain.origin_format}" if advice.grain.origin_format else "")
            + f" · CRF {plan.crf_label} · preset {plan.preset}"
        ),
        "Type": 1,  # a user preset rather than a built-in
        "Default": False,
        "FileFormat": "av_mkv",
        "Mp4HttpOptimize": False,
        "ChapterMarkers": True,
        "VideoEncoder": HANDBRAKE_ENCODER[plan.encoder],
        "VideoQualityType": 2,  # constant quality
        "VideoQualitySlider": round(plan.crf, 1),
        "VideoPreset": plan.preset,
        "VideoTune": plan.tune or "",
        "VideoProfile": "auto",
        "VideoLevel": "auto",
        "VideoOptionExtra": plan.params_string,
        "VideoFramerateMode": "vfr",
        "VideoFramerate": "auto",
        "VideoScaler": "swscale",
        "PictureCropMode": 0 if not _should_autocrop(advice) else 2,
        "PictureDeinterlaceFilter": (
            "decomb" if video is not None and video.is_interlaced else "off"
        ),
        "PictureDeinterlacePreset": "default",
        "PictureCombDetectPreset": (
            "default" if video is not None and video.is_interlaced else "off"
        ),
        "AudioTrackSelectionBehavior": "all",
        "AudioSecondaryEncoderMode": False,
        "AudioCopyMask": [
            "copy:truehd",
            "copy:dtshd",
            "copy:dts",
            "copy:eac3",
            "copy:ac3",
            "copy:flac",
            "copy:aac",
            "copy:mp3",
        ],
        "AudioEncoderFallback": "ac3",
        "AudioList": [
            {
                "AudioEncoder": "copy",
                "AudioTrackQualityEnable": False,
                "AudioMixdown": "none",
                "AudioSamplerate": "auto",
                "AudioBitrate": "auto",
                "AudioTrackDRCSlider": 0.0,
                "AudioTrackGainSlider": 0.0,
            }
        ],
        "SubtitleTrackSelectionBehavior": "all",
        "SubtitleAddForeignAudioSearch": True,
        "SubtitleBurnBehavior": "none",
    }

    if video is not None and video.width and video.height:
        preset["PictureWidth"] = video.width
        preset["PictureHeight"] = video.height

    return {
        "PresetList": [preset],
        "VersionMajor": 1,
        "VersionMinor": 0,
        "VersionMicro": 0,
    }


def _should_autocrop(advice: Advice) -> bool:
    return any("letterboxed" in note for note in advice.notes)


def attach_commands(advice: Advice, request: EncodeRequest) -> Advice:
    """Fill in every plan's rendered commands, in place."""
    for plan in advice.plans:
        plan.ffmpeg_command = render_ffmpeg(plan, request)
        plan.handbrake_command = render_handbrake(plan, request)
    return advice
