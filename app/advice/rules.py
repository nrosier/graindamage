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
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class _Synthesis:
    """How SVT-AV1 should handle the grain: replace it, or shore it up."""

    strength: int
    # True denoises the picture and re-synthesises grain at playback — the largest
    # saving there is, at the cost of a *uniform* grain field. False leaves the real
    # grain in the picture to be coded and uses synthesis only as a floor, filling in
    # where the quantiser flattened it.
    denoise: bool


# Only ever applied to a photochemical source: synthesising grain over a digitally
# acquired image lays noise on a picture that never had any.
_FILM_GRAIN: dict[GrainLevel, _Synthesis] = {
    GrainLevel.LIGHT: _Synthesis(4, denoise=False),
    GrainLevel.MODERATE: _Synthesis(8, denoise=False),
    # At these levels grain *is* the image, and coding every particle of it is what
    # makes a grainy film enormous. Denoise-and-resynthesise is the right trade.
    GrainLevel.HEAVY: _Synthesis(12, denoise=True),
    GrainLevel.EXTREME: _Synthesis(20, denoise=True),
}

# The highest CRF at which coding the real grain is still worth attempting. Above it the
# encoder has been told to keep every grain particle and then given no bits to do it
# with: real grain is high-frequency spatial noise, the bit budget runs out, and what
# was meant to be grain becomes swirling blocks in motion. So the denoiser and the CRF
# are one decision. A plan that codes real grain is capped here; a plan that wants a
# higher CRF has to denoise and re-synthesise instead.
REAL_GRAIN_CRF_CEILING = 27.0

# Finer grain is cheaper to code, so a large-format negative's tight grain holds together
# a little further up the scale than 35 mm's does. Heavy and extreme never reach this
# table: at those levels the grain is replaced rather than coded.
_CEILING_BONUS: dict[GrainLevel, float] = {GrainLevel.LIGHT: 2.0}

# What the strength becomes when a level that would have coded its own grain flips to
# replacing it: a replacement field carries the whole look, where a floor only fills in
# what the quantiser flattened. Monotone with the levels that already replace (12, 20).
_REPLACEMENT_STRENGTH: dict[GrainLevel, int] = {
    GrainLevel.LIGHT: 6,
    GrainLevel.MODERATE: 10,
}

# Stocks got finer, and dupe negatives and optical printing got rarer. A pre-1970
# negative carries coarser, higher-contrast grain than a late-1990s one of the same
# format, so the same "35 mm" row means more grain to protect.
_COARSE_STOCK_YEAR = 1970
_COARSE_STOCK_BONUS = 4

# SVT-AV1 warns that film-grain above this preset "produces a significant compute
# overhead" and is for debugging only. Measured against SVT-AV1 4.2.0, not folklore.
_MAX_FILM_GRAIN_PRESET = 6

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
        # The picture being coded has had its grain taken out, so it costs about what a
        # grainless one costs; the grain comes back as a handful of synthesis
        # parameters. Modelling it any dearer than that hides the saving, which is the
        # whole reason for choosing to re-synthesise.
        factor = _GRAIN_BITRATE_FACTOR[GrainLevel.NONE]
    return int(bpp * factor * video.pixels * frame_rate)


# --- plan construction ------------------------------------------------------


def _release_year(request: EncodeRequest) -> int | None:
    """The film's year, from whichever of the two places knows it.

    A looked-up year beats a parsed one: ``fallback_year`` is whatever the CLI read out
    of the filename, which is right often enough to be useful and wrong often enough
    that TMDB wins when it answered.
    """
    if request.movie is not None:
        return request.movie.year
    return request.fallback_year


def _is_coarse_stock(request: EncodeRequest, grain: GrainProfile) -> bool:
    """Whether this is an early photochemical source, which grains differently.

    The format row says how big the negative was; the year says what was coated on it
    and how many optical generations sit between it and the scan. Both have to point at
    film for the era to mean anything — a 2020 digital film released in a year we do not
    know is not a 1962 negative.
    """
    if not grain.is_photochemical:
        return False
    year = _release_year(request)
    return year is not None and year <= _COARSE_STOCK_YEAR


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

    synthesis = _FILM_GRAIN.get(grain.level) if grain.is_photochemical else None

    ceiling = REAL_GRAIN_CRF_CEILING + _CEILING_BONUS.get(grain.level, 0.0)

    if synthesis is not None and not synthesis.denoise and crf > ceiling:
        # Coding the real grain and then starving it are two halves of one mistake, and
        # this is where the two decisions meet. Which way out to take depends on whether
        # this film's own grain is worth the bitrate: a compact target says the smaller
        # file matters more, and a source whose detail is already limited has grain that
        # is half compression artefact anyway. Either way it is better replaced than
        # faithfully coded. Otherwise the grain is the point, and the bitrate is found.
        thin_source = quality_delta >= 1.5
        if request.size is SizePreference.COMPACT or thin_source:
            synthesis = _Synthesis(
                _REPLACEMENT_STRENGTH.get(grain.level, synthesis.strength), denoise=True
            )
            because = (
                "this source's own detail is already limited, so what is left of its "
                "grain is half compression artefact"
                if thin_source
                else "a compact target asks for the smaller file"
            )
            rationale.append(
                f"CRF {crf:g} is too high to code this film's own grain, and {because} — "
                "so the denoiser goes on and the grain is re-synthesised at playback "
                "instead. That keeps the CRF, at the cost of a uniform grain field in "
                "place of this negative's own."
            )
        else:
            adjustments.append(
                Adjustment(
                    label="real grain ceiling",
                    delta=ceiling - crf,
                    detail=(
                        f"Coded grain needs the bitrate to carry it; above CRF "
                        f"{ceiling:g} it breaks up in motion instead."
                    ),
                )
            )
            rationale.append(
                f"CRF {ceiling:g} rather than {crf:g}: this film's own "
                "grain is being coded rather than replaced, and grain is high-frequency "
                "noise that a higher CRF cannot afford — it would break up into blocks "
                "in motion. The file is larger for it. Ask for a compact target instead "
                "and the grain is denoised and re-synthesised, which is the other way to "
                "spend less."
            )
            crf = ceiling

    coarse = synthesis is not None and _is_coarse_stock(request, grain)

    if synthesis is not None:
        if preset > _MAX_FILM_GRAIN_PRESET:
            rationale.append(
                f"Preset {_MAX_FILM_GRAIN_PRESET} rather than {preset}: SVT-AV1 warns that "
                "film-grain above preset 6 is a large compute overhead meant for debugging, "
                "and the grain settings matter more here than the last step of speed."
            )
            preset = _MAX_FILM_GRAIN_PRESET

        strength = synthesis.strength + (_COARSE_STOCK_BONUS if coarse else 0)
        params["film-grain"] = str(strength)
        params["film-grain-denoise"] = "1" if synthesis.denoise else "0"

        if synthesis.denoise:
            rationale.append(
                f"film-grain={strength} with film-grain-denoise=1: SVT-AV1 removes the "
                "grain, codes the clean image, and re-synthesises grain at playback. On a "
                "negative this coarse that is the single largest saving available. The "
                "grain it puts back is uniform, though, so if this film's grain varies by "
                "shot — a blow-up, a mixed-format shoot — set film-grain-denoise=0, drop "
                f"film-grain to about {max(2, strength // 3)}, and bring the CRF down to "
                f"{REAL_GRAIN_CRF_CEILING:g} or below, because grain that is coded rather "
                "than replaced has to be paid for."
            )
        else:
            rationale.append(
                f"film-grain={strength} with film-grain-denoise=0: the film's own grain "
                "stays in the picture and is coded, and synthesis only acts as a floor "
                "where the quantiser flattened it. Turning the denoiser on would be "
                "smaller, but it replaces this negative's grain with a uniform field."
            )
            # AV1's loop restoration is a Wiener / self-guided filter fitted per unit —
            # a denoiser inside the loop. Where the grain is being coded rather than
            # replaced, it spends bits smoothing away what was just paid for. CDEF is
            # left alone deliberately: it is also a deringer, and dropping it trades
            # grain for visible artefacts at these CRFs.
            params["enable-restoration"] = "0"
            rationale.append(
                "enable-restoration=0: AV1's loop restoration is a Wiener filter applied "
                "inside the coding loop, so on a source whose grain is being coded it "
                "smooths away texture the encode has already paid for."
            )

    if coarse:
        # x265 gets the same idea as qcomp below; this is SVT-AV1's version of it.
        params["qp-scale-compress-strength"] = "2"
        rationale.append(
            f"qp-scale-compress-strength=2 for a {_release_year(request)} negative: it "
            "flattens the QP scale between temporal layers, so the B-frames are not "
            "quantised much harder than the keyframes they inherit their grain from. "
            "Coarse early stock is where that difference shows first."
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
            # Only denoise-and-resynthesise makes a grainy film cheap; synthesis used as
            # a floor still codes every particle, so the grain factor stands.
            Encoder.SVT_AV1,
            crf,
            video,
            grain.level,
            synthesised=synthesis is not None and synthesis.denoise,
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
        coarse = _is_coarse_stock(request, grain)
        tune = "grain"
        rationale.append(
            "--tune grain is the whole point of x265 on this film: it raises psy-rd to "
            "4.0 and psy-rdoq to 10, and turns off SAO, cu-tree, AQ and rskip, all of "
            "which stop the encoder treating grain as noise to be removed. What it does "
            "*not* do, despite its reputation, is change qcomp or the deblocking "
            "offsets — measured against x265 4.2, both stay at their defaults — so the "
            "two below are set by hand."
        )
        # tune grain sets aq-mode 0; auto-variance AQ still helps dark grainy scenes,
        # and explicit params are applied after the tune, so this wins.
        params["aq-mode"] = "3"
        params["aq-strength"] = "0.8" if grain.level.rank >= GrainLevel.HEAVY.rank else "0.9"
        params["rc-lookahead"] = "60"

        params["qcomp"] = "0.85" if coarse else "0.8"
        rationale.append(
            f"qcomp={params['qcomp']} against the 0.60 default: a flatter quantiser "
            "curve spends closer to the same quality on the complex frames, and on a "
            "grainy film every frame is a complex frame."
            + (
                " Higher still here, because coarse early stock varies far more from shot to shot."
                if coarse
                else ""
            )
        )

        # One value, not "-1:-1": params_string joins on ':', and libx265 reads the
        # second half of a colon-separated value as a parameter name — which silently
        # swallows whatever comes after it. A single value sets both offsets anyway.
        params["deblock"] = "-2" if coarse else "-1"
        rationale.append(
            f"deblock={params['deblock']} (both offsets): the in-loop deblocking filter "
            "does not know the difference between a block edge and grain, and at its "
            "default strength it takes the grain with it."
        )

        if grain.is_photochemical:
            params["strong-intra-smoothing"] = "0"
            rationale.append(
                "strong-intra-smoothing=0: it bilinear-smooths intra prediction across "
                "large flat blocks, which is exactly where fine grain lives — skies, "
                "walls, dissolves."
            )
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
        year=_release_year(request),
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
