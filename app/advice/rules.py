"""The deterministic baseline: CRF, preset and encoder parameters, with reasons.

This module is the product's floor. It has no API keys, no network and no model
behind it, so an unconfigured container still produces real settings — Gemini in
milestone 5 only ever *annotates* what comes out of here, and is validated against it.

Every number is reached by starting from a resolution-dependent anchor and applying
named, signed adjustments, which are kept on the plan (:class:`Adjustment`) rather
than folded away. That is deliberate: "CRF 26" is an opinion, and the user is better
served by seeing "28 base, −1 for 35 mm grain, −1 for HDR gradients" than by being
handed a number to trust.

The anchors are conventional community values for 10-bit encodes of film sources, not
measurements: SVT-AV1 sits roughly 6–8 CRF points above x265 for comparable quality
because the two scales are unrelated.
"""

from __future__ import annotations

import re

from app.advice.grain import infer_grain
from app.models import (
    Adjustment,
    Advice,
    AdviceSource,
    Encoder,
    EncodeRequest,
    EncoderPlan,
    GrainLevel,
    GrainProfile,
    SizePreference,
    SpeedPreference,
    VideoTrack,
)
from app.sources.colors import CODE_BY_MATRIX, CODE_BY_PRIMARIES, CODE_BY_TRANSFER

# --- anchors ----------------------------------------------------------------

_BASE_CRF: dict[Encoder, dict[str, float]] = {
    # Higher resolutions tolerate a higher CRF: at the same viewing distance each
    # coding error covers less of the screen.
    Encoder.SVT_AV1: {"SD": 24, "720p": 26, "1080p": 28, "1440p": 30, "2160p": 32},
    Encoder.X265: {"SD": 19, "720p": 20, "1080p": 21, "1440p": 22, "2160p": 23},
}

_CRF_LIMITS: dict[Encoder, tuple[float, float]] = {
    # Well inside each encoder's legal range; outside these the result is either
    # visually lossless at absurd size or obviously broken.
    Encoder.SVT_AV1: (15.0, 45.0),
    Encoder.X265: (14.0, 32.0),
}

# x265 codes every grain particle, so grain costs it real bitrate. SVT-AV1 can
# synthesise grain instead, which is why its penalties are smaller at the top end.
_GRAIN_CRF_DELTA: dict[Encoder, dict[GrainLevel, float]] = {
    Encoder.SVT_AV1: {
        GrainLevel.NONE: 1.0,
        GrainLevel.LIGHT: 0.0,
        GrainLevel.MODERATE: -1.0,
        GrainLevel.HEAVY: -1.0,
        GrainLevel.EXTREME: -1.0,
    },
    Encoder.X265: {
        GrainLevel.NONE: 1.0,
        GrainLevel.LIGHT: 0.0,
        GrainLevel.MODERATE: -1.0,
        GrainLevel.HEAVY: -2.0,
        GrainLevel.EXTREME: -3.0,
    },
}

# Film-grain synthesis strength, only used where grain dominates the image.
_FILM_GRAIN_STRENGTH: dict[GrainLevel, int] = {
    GrainLevel.HEAVY: 12,
    GrainLevel.EXTREME: 20,
}

_SIZE_CRF_DELTA: dict[SizePreference, float] = {
    SizePreference.ARCHIVAL: -2.0,
    SizePreference.BALANCED: 0.0,
    SizePreference.COMPACT: 2.0,
}

_SVT_PRESETS: dict[SpeedPreference, int] = {
    SpeedPreference.QUALITY: 3,
    SpeedPreference.BALANCED: 4,
    SpeedPreference.FAST: 6,
}

_X265_PRESETS: dict[SpeedPreference, str] = {
    SpeedPreference.QUALITY: "slower",
    SpeedPreference.BALANCED: "slow",
    SpeedPreference.FAST: "medium",
}
_X265_PRESET_LADDER = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)

# Reference points for the size estimate: bits per pixel per frame at the anchor CRF.
_BPP_ANCHOR: dict[Encoder, tuple[float, float, float]] = {
    # (bpp at anchor, anchor CRF, bitrate ratio per +1 CRF)
    Encoder.SVT_AV1: (0.055, 28.0, 0.88),
    Encoder.X265: (0.100, 21.0, 0.85),
}

_GRAIN_BITRATE_FACTOR: dict[GrainLevel, float] = {
    GrainLevel.NONE: 1.0,
    GrainLevel.LIGHT: 1.08,
    GrainLevel.MODERATE: 1.25,
    GrainLevel.HEAVY: 1.55,
    GrainLevel.EXTREME: 1.90,
}

DEFAULT_FRAME_RATE = 24.0
KEYFRAME_SECONDS = 10

_ASPECT_RATIO = re.compile(r"(\d+(?:\.\d+)?)\s*[:/x]\s*(\d+(?:\.\d+)?)")


def parse_aspect_ratio(text: str) -> float | None:
    """``2.39 : 1``, ``16:9``, ``1.85 : 1 (intended)`` — all to a single float."""
    match = _ASPECT_RATIO.search(text)
    if match:
        width, height = float(match.group(1)), float(match.group(2))
        return width / height if height else None
    # A bare decimal such as "1.85" is also how IMDb sometimes writes it.
    bare = re.fullmatch(r"\s*(\d\.\d+)\s*", text)
    return float(bare.group(1)) if bare else None


# --- helpers ----------------------------------------------------------------


def _resolution_class(video: VideoTrack | None) -> str:
    if video is None or not video.height:
        return "1080p"  # the safest assumption when nothing is known
    height = video.height
    if height <= 576:
        return "SD"
    if height <= 720:
        return "720p"
    if height <= 1080:
        return "1080p"
    if height <= 1440:
        return "1440p"
    return "2160p"


def _source_quality_delta(video: VideoTrack | None) -> tuple[float, str | None, str | None]:
    """CRF nudge, explanation, and optional warning based on the source's bitrate.

    Re-encoding a heavily compressed file cannot recover what it lost, so chasing its
    remaining noise with a low CRF only reprints someone else's artefacts at a larger
    size.
    """
    bpp = video.bits_per_pixel if video else None
    if bpp is None:
        return 0.0, "Source bitrate unknown, so no quality adjustment was applied.", None
    if bpp >= 0.15:
        return (
            0.0,
            f"Source is {bpp:.3f} bits/pixel — disc-grade, so it is worth encoding faithfully.",
            None,
        )
    if bpp >= 0.08:
        return 0.5, f"Source is {bpp:.3f} bits/pixel — a good web release.", None
    if bpp >= 0.04:
        return (
            1.5,
            f"Source is only {bpp:.3f} bits/pixel, so its own detail is already limited.",
            "The source is modestly compressed; a lower CRF would mostly preserve its "
            "compression artefacts, not detail.",
        )
    return (
        2.5,
        f"Source is {bpp:.3f} bits/pixel — heavily compressed.",
        "This source is already heavily compressed. Re-encoding it will lose more than "
        "it saves; keep the original unless you need the space.",
    )


def _keyint(video: VideoTrack | None) -> int:
    frame_rate = (video.frame_rate if video else None) or DEFAULT_FRAME_RATE
    return round(frame_rate * KEYFRAME_SECONDS)


def _shift_preset(preset: str, steps: int) -> str:
    """Move along x265's preset ladder, clamped at both ends."""
    try:
        index = _X265_PRESET_LADDER.index(preset)
    except ValueError:
        return preset
    return _X265_PRESET_LADDER[max(0, min(index + steps, len(_X265_PRESET_LADDER) - 1))]


def estimate_bitrate(
    encoder: Encoder, crf: float, video: VideoTrack | None, grain: GrainLevel, *, synthesised: bool
) -> int | None:
    """A rough bits/second figure, good enough to compare the two plans.

    Deliberately crude — one exponential fit through a community anchor point. It
    exists so the user can see that AV1 lands at roughly half the size, not so anyone
    can plan a disc against it.
    """
    if video is None or not video.pixels:
        return None

    bpp_anchor, crf_anchor, ratio = _BPP_ANCHOR[encoder]
    frame_rate = video.frame_rate or DEFAULT_FRAME_RATE
    bpp = bpp_anchor * (ratio ** (crf - crf_anchor))

    factor = _GRAIN_BITRATE_FACTOR[grain]
    if synthesised:
        # A denoised signal plus synthesis parameters costs far less than coded grain.
        factor = min(factor, 1.2)
    return int(bpp * factor * video.pixels * frame_rate)


# --- plan construction ------------------------------------------------------


def _colour_params(encoder: Encoder, video: VideoTrack | None) -> dict[str, str]:
    """Carry the source's colour signalling into the encode.

    Dropping this is the classic way to end up with a washed-out or fluorescent HDR
    encode: the pixels survive but nothing tells the display what they mean.
    """
    if video is None:
        return {}

    params: dict[str, str] = {}
    if encoder is Encoder.X265:
        if video.color_primaries:
            params["colorprim"] = video.color_primaries
        if video.color_transfer:
            params["transfer"] = video.color_transfer
        if video.color_matrix:
            params["colormatrix"] = video.color_matrix
        if video.color_range:
            params["range"] = "full" if video.color_range == "pc" else "limited"
        if video.is_hdr:
            params["hdr10"] = "1"
            params["hdr10-opt"] = "1"
            params["repeat-headers"] = "1"
            if video.mastering_display:
                params["master-display"] = video.mastering_display.to_x265()
            if video.max_cll is not None:
                params["max-cll"] = f"{video.max_cll},{video.max_fall or 0}"
        return params

    # SVT-AV1 takes the H.273 code points rather than names.
    if video.color_primaries and (code := CODE_BY_PRIMARIES.get(video.color_primaries)):
        params["color-primaries"] = str(code)
    if video.color_transfer and (code := CODE_BY_TRANSFER.get(video.color_transfer)):
        params["transfer-characteristics"] = str(code)
    if video.color_matrix and (code := CODE_BY_MATRIX.get(video.color_matrix)):
        params["matrix-coefficients"] = str(code)
    if video.color_range:
        params["color-range"] = "1" if video.color_range == "pc" else "0"
    if video.is_hdr:
        if video.mastering_display:
            params["mastering-display"] = video.mastering_display.to_svt_av1()
        if video.max_cll is not None:
            params["content-light"] = f"{video.max_cll},{video.max_fall or 0}"
    return params


def _svt_av1_plan(request: EncodeRequest, grain: GrainProfile) -> EncoderPlan:
    video = request.source.video
    resolution = _resolution_class(video)
    base = _BASE_CRF[Encoder.SVT_AV1][resolution]

    adjustments: list[Adjustment] = []
    rationale: list[str] = []

    grain_delta = _GRAIN_CRF_DELTA[Encoder.SVT_AV1][grain.level]
    if grain_delta:
        adjustments.append(
            Adjustment(
                label=f"{grain.level.value} grain",
                delta=grain_delta,
                detail="Grain is expensive detail; a lower CRF stops it turning into blocking.",
            )
        )

    if video is not None and video.is_hdr:
        adjustments.append(
            Adjustment(
                label="HDR",
                delta=-1.0,
                detail="PQ/HLG gradients band before SDR ones do, so give them more bits.",
            )
        )

    quality_delta, quality_detail, _ = _source_quality_delta(video)
    if quality_delta:
        adjustments.append(
            Adjustment(label="source quality", delta=quality_delta, detail=quality_detail)
        )
    elif quality_detail:
        rationale.append(quality_detail)

    size_delta = _SIZE_CRF_DELTA[request.size]
    if size_delta:
        adjustments.append(
            Adjustment(label=f"{request.size.value} target", delta=size_delta, detail=None)
        )

    low, high = _CRF_LIMITS[Encoder.SVT_AV1]
    crf = max(low, min(base + sum(a.delta for a in adjustments), high))

    preset = _SVT_PRESETS[request.speed]
    if grain.level.rank >= GrainLevel.HEAVY.rank and request.speed is not SpeedPreference.FAST:
        preset = max(2, preset - 1)
        rationale.append(
            f"Preset {preset} rather than {_SVT_PRESETS[request.speed]}: slower presets "
            "hold grain together instead of averaging it away."
        )
    if resolution == "2160p" and request.speed is not SpeedPreference.QUALITY:
        preset += 1
        rationale.append(f"Preset {preset} at 2160p — a step faster keeps encode time sane.")

    params: dict[str, str] = {
        # tune=0 optimises for subjective quality; the default (1) optimises PSNR,
        # which systematically prefers smoothing film grain away.
        "tune": "0",
        "keyint": str(_keyint(video)),
        "scd": "1",
    }

    strength = _FILM_GRAIN_STRENGTH.get(grain.level)
    synthesised = strength is not None
    if strength is not None:
        params["film-grain"] = str(strength)
        params["film-grain-denoise"] = "1"
        rationale.append(
            f"film-grain={strength} with the denoiser on: SVT-AV1 removes the grain, "
            "codes the clean image, and re-synthesises grain at playback. That is the "
            "single largest saving available on a grainy film. If you would rather code "
            "the real grain, set film-grain-denoise=0 and drop film-grain to about "
            f"{max(2, strength // 3)} — otherwise you get coded grain *and* synthesised "
            "grain on top."
        )
    elif grain.level is not GrainLevel.NONE:
        rationale.append(
            "No film-grain synthesis at this grain level: synthesised grain is uniform, "
            "and on light or moderate grain that reads as noise laid over the picture "
            "rather than as the picture's own texture."
        )

    params.update(_colour_params(Encoder.SVT_AV1, video))

    rationale.append(
        "10-bit output regardless of the source's depth: it costs a few percent, "
        "removes banding in skies and dissolves, and every AV1 decoder supports it."
    )

    return EncoderPlan(
        encoder=Encoder.SVT_AV1,
        crf=crf,
        preset=str(preset),
        params=params,
        adjustments=[
            Adjustment(label=f"{resolution} baseline", delta=base, detail=None),
            *adjustments,
        ],
        rationale=rationale,
        estimated_bitrate_bps=estimate_bitrate(
            Encoder.SVT_AV1, crf, video, grain.level, synthesised=synthesised
        ),
    )


def _x265_plan(request: EncodeRequest, grain: GrainProfile) -> EncoderPlan:
    video = request.source.video
    resolution = _resolution_class(video)
    base = _BASE_CRF[Encoder.X265][resolution]

    adjustments: list[Adjustment] = []
    rationale: list[str] = []

    grain_delta = _GRAIN_CRF_DELTA[Encoder.X265][grain.level]
    if grain_delta:
        adjustments.append(
            Adjustment(
                label=f"{grain.level.value} grain",
                delta=grain_delta,
                detail="x265 has no grain synthesis, so every particle has to be coded.",
            )
        )

    if video is not None and video.is_hdr:
        adjustments.append(
            Adjustment(
                label="HDR",
                delta=-1.0,
                detail="PQ/HLG gradients band before SDR ones do, so give them more bits.",
            )
        )

    quality_delta, quality_detail, _ = _source_quality_delta(video)
    if quality_delta:
        adjustments.append(
            Adjustment(label="source quality", delta=quality_delta, detail=quality_detail)
        )

    size_delta = _SIZE_CRF_DELTA[request.size]
    if size_delta:
        adjustments.append(
            Adjustment(label=f"{request.size.value} target", delta=size_delta, detail=None)
        )

    low, high = _CRF_LIMITS[Encoder.X265]
    crf = max(low, min(base + sum(a.delta for a in adjustments), high))

    preset = _X265_PRESETS[request.speed]
    if grain.level.rank >= GrainLevel.HEAVY.rank and request.speed is not SpeedPreference.FAST:
        preset = _shift_preset(preset, 1)
        rationale.append(f"Preset {preset}: grain needs the extra rate-distortion search.")

    tune: str | None = None
    params: dict[str, str] = {
        "keyint": str(_keyint(video)),
        "min-keyint": str(max(1, _keyint(video) // 10)),
    }

    if grain.level.rank >= GrainLevel.MODERATE.rank:
        tune = "grain"
        rationale.append(
            "--tune grain is the whole point of x265 on this film: it raises qcomp, "
            "turns off SAO and cu-tree, and loosens deblocking, all of which stop the "
            "encoder treating grain as noise to be removed."
        )
        # tune grain sets aq-mode 0; auto-variance AQ still helps dark grainy scenes,
        # and explicit params are applied after the tune, so this wins.
        params["aq-mode"] = "3"
        params["aq-strength"] = "0.8" if grain.level.rank >= GrainLevel.HEAVY.rank else "0.9"
        params["rc-lookahead"] = "60"
    else:
        params["aq-mode"] = "3"
        params["sao"] = "0"
        params["rc-lookahead"] = "48"
        rationale.append(
            "SAO off even without --tune grain: it is a smoothing filter, and on film "
            "sources it costs more texture than it saves bits."
        )

    params.update(_colour_params(Encoder.X265, video))

    rationale.append(
        "Use a 10-bit x265 build (x265_10bit / -pix_fmt yuv420p10le). 10-bit HEVC is "
        "both more efficient and free of the banding an 8-bit encode adds."
    )

    return EncoderPlan(
        encoder=Encoder.X265,
        crf=crf,
        preset=preset,
        tune=tune,
        params=params,
        adjustments=[
            Adjustment(label=f"{resolution} baseline", delta=base, detail=None),
            *adjustments,
        ],
        rationale=rationale,
        estimated_bitrate_bps=estimate_bitrate(
            Encoder.X265, crf, video, grain.level, synthesised=False
        ),
    )


def _cross_cutting_notes(
    request: EncodeRequest, grain: GrainProfile
) -> tuple[list[str], list[str]]:
    """Advice that is not a CRF: cropping, deinterlacing, audio, and outright "don't"."""
    notes: list[str] = []
    warnings: list[str] = []
    video = request.source.video

    if video is None:
        warnings.append(
            "No video track was described, so these settings come from defaults rather "
            "than from your file. Paste ffprobe, mkvinfo or MediaInfo output for real advice."
        )
        return notes, warnings

    # Said once, here, rather than in both plans' rationale.
    if (quality_warning := _source_quality_delta(video)[2]) is not None:
        warnings.append(quality_warning)

    # IMDb's aspect ratio against the file's: a wider film in a narrower frame is
    # letterboxed, and those black bars are pure wasted bitrate.
    imdb_ratios = [
        ratio
        for text in request.specs.aspect_ratios
        if (ratio := parse_aspect_ratio(text)) is not None
    ]
    file_ratio = (
        parse_aspect_ratio(video.display_aspect_ratio) if video.display_aspect_ratio else None
    )
    if file_ratio is None and video.width and video.height:
        file_ratio = video.width / video.height
    if imdb_ratios and file_ratio:
        widest = max(imdb_ratios)
        if widest > file_ratio + 0.05:
            notes.append(
                f"IMDb lists {widest:.2f}:1 but the file is {file_ratio:.2f}:1, so it is "
                "letterboxed. Crop the bars — ffmpeg -vf cropdetect, or HandBrake's "
                "automatic crop — before encoding; black bars cost bitrate and hurt "
                "nothing but your file size."
            )
        elif widest < file_ratio - 0.05:
            notes.append(
                f"The file is wider ({file_ratio:.2f}:1) than IMDb's {widest:.2f}:1, so it "
                "has probably already been cropped. Do not crop again."
            )

    if video.is_interlaced:
        warnings.append(
            "The source is interlaced. Deinterlace before encoding (ffmpeg -vf "
            "bwdif=mode=send_field, or HandBrake's Decomb) — neither AV1 nor HEVC has "
            "interlaced coding tools, and encoding fields as frames looks terrible."
        )

    if video.codec == "av1":
        warnings.append(
            "The source is already AV1. Re-encoding it will lose quality for very little "
            "space; only do this if you need a different resolution or container."
        )

    if video.chroma_subsampling in {"4:2:2", "4:4:4"}:
        notes.append(
            f"The source is {video.chroma_subsampling}. Both plans assume 4:2:0 output, "
            "which is what players expect; keep the original if this is a grading master."
        )

    if video.dolby_vision:
        warnings.append(
            "Dolby Vision is present. Neither plan preserves the DV RPU — you will get "
            "the HDR10 base layer only. Use dovi_tool to extract and re-inject the RPU "
            "if DV matters to you."
        )
    if video.hdr10_plus:
        notes.append(
            "HDR10+ dynamic metadata is present and will be dropped. hdr10plus_tool can "
            "extract and re-inject it; x265 also accepts --dhdr10-info."
        )

    if request.source.audio:
        lossless = [track for track in request.source.audio if track.is_lossless]
        if lossless:
            notes.append(
                f"{len(lossless)} lossless audio track(s). Copy audio through (-c:a copy) "
                "unless size is the point — re-encoding it is where quality quietly goes, "
                "and it is usually a small share of the file."
            )

    if grain.confidence < 0.5:
        warnings.append(
            f"Grain was guessed at, not established (confidence {grain.confidence:.0%}). "
            "Look at a still from a dark, flat shot and set the grain level by hand if "
            "this is wrong — it moves the CRF by several points."
        )

    return notes, warnings


def build_advice(request: EncodeRequest) -> Advice:
    """Produce the baseline advice for a request. Never raises, never calls out."""
    grain = infer_grain(
        request.specs,
        request.source,
        year=request.movie.year if request.movie else None,
        override=request.grain_override,
    )

    if request.bit_depth_override and request.source.video:
        request.source.video.bit_depth = request.bit_depth_override

    notes, warnings = _cross_cutting_notes(request, grain)

    return Advice(
        source=AdviceSource.BASELINE,
        grain=grain,
        plans=[_svt_av1_plan(request, grain), _x265_plan(request, grain)],
        summary=_summary(request, grain),
        notes=notes,
        warnings=warnings,
    )


def _summary(request: EncodeRequest, grain: GrainProfile) -> str:
    title = request.movie.title if request.movie else "This source"
    resolution = _resolution_class(request.source.video)
    origin = grain.origin_format or "an unknown origin format"
    return (
        f"{title}: {resolution} source from {origin}, {grain.level.value} grain. "
        "Settings below are tuned to keep that texture rather than average it out."
    )
