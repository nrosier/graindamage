"""Validate encoder settings that came from a language model.

The output of this app is a command a user will paste into a shell. A model that
hallucinates a plausible-looking flag produces an encode that dies twenty minutes in;
a model that is *steered* — by a title, by pasted page source, by anything upstream —
could otherwise put whatever it likes into that command. So nothing Gemini returns
reaches a command line without passing through here.

The policy is an allowlist, not a denylist: a parameter is dropped unless it is named
below *and* its value is in range. Everything dropped is reported, so the UI can say
what was ignored rather than silently disagreeing with the model.

What is checked here is whether a setting *exists and is legal*, never whether it is a
good idea. The model has the film, the technical rows and the parse of the source file,
so it is better placed to weigh those than any table in this repository; a parameter is
refused only when the encoder itself would refuse it.

Values are also character-restricted. Commands are built with :func:`shlex.join`, so
this is defence in depth rather than the only barrier — but a parameter value has no
legitimate reason to contain a backtick.

Colons are barred for a plainer reason: :attr:`EncoderPlan.params_string` joins the
parameters with ``:`` into the one string that ``-x265-params``, ``-svtav1-params`` and
HandBrake's ``--encopts`` take. A colon inside a value does not merely fail — libx265
reads the far half as a parameter name (``Unknown option: -1:sao``) and silently drops
whatever came after it. No parameter allowed below needs one; ``deblock`` takes a single
value that x265 applies to both offsets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import Encoder
from app.sources.colors import CODE_BY_MATRIX, CODE_BY_PRIMARIES, CODE_BY_TRANSFER

# Every legitimate encoder parameter value is made of these — note the absent colon.
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.,()+/=-]{1,120}$")
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9-]{0,40}$")

MAX_PARAMS = 40


@dataclass(frozen=True, slots=True)
class IntRange:
    low: int
    high: int

    def accepts(self, value: str) -> bool:
        try:
            return self.low <= int(value) <= self.high
        except ValueError:
            return False


@dataclass(frozen=True, slots=True)
class FloatRange:
    low: float
    high: float

    def accepts(self, value: str) -> bool:
        try:
            return self.low <= float(value) <= self.high
        except ValueError:
            return False


@dataclass(frozen=True, slots=True)
class Choice:
    values: frozenset[str]

    def accepts(self, value: str) -> bool:
        return value.casefold() in self.values


@dataclass(frozen=True, slots=True)
class Pattern:
    regex: re.Pattern[str]

    def accepts(self, value: str) -> bool:
        return bool(self.regex.fullmatch(value))


Rule = IntRange | FloatRange | Choice | Pattern

_BOOL = IntRange(0, 1)
_MASTERING = Pattern(
    re.compile(
        r"G\([\d.]+,[\d.]+\)B\([\d.]+,[\d.]+\)R\([\d.]+,[\d.]+\)WP\([\d.]+,[\d.]+\)L\([\d.]+,[\d.]+\)"
    )
)
_LIGHT_LEVEL = Pattern(re.compile(r"\d{1,6},\d{1,6}"))

# --- SVT-AV1 ----------------------------------------------------------------

# Ranges below were read off SVT-AV1 4.2.0's own refusals, not off a wiki. Two
# absences are deliberate:
#
# * ``enable-hdr`` does not exist — SVT-AV1 answers "Error parsing option enable-hdr",
#   exactly as it does for an invented key. HDR is signalled through
#   ``mastering-display``, ``content-light`` and the three colour parameters.
# * ``tune`` stops at 2 although 3-5 parse. Tune 3 (IQ) aborts outright on a
#   random-access encode, and tune 5 (VMAF) unsharp-masks the input, which is the
#   opposite of what this tool is for.
SVT_AV1_PARAMS: dict[str, Rule] = {
    "tune": IntRange(0, 2),
    "keyint": IntRange(-2, 10_000),
    "scd": _BOOL,
    "film-grain": IntRange(0, 50),
    "film-grain-denoise": _BOOL,
    "aq-mode": IntRange(0, 2),
    "enable-tf": IntRange(0, 2),
    "tf-strength": IntRange(0, 4),
    "enable-overlays": _BOOL,
    "enable-dlf": IntRange(0, 2),
    "enable-cdef": _BOOL,
    "enable-restoration": _BOOL,
    "enable-qm": _BOOL,
    "qm-min": IntRange(0, 15),
    "qm-max": IntRange(0, 15),
    "sharpness": IntRange(-7, 7),
    # The grain-relevant rate-control knobs: how flat the QP scale is between temporal
    # layers, how much extra goes to dark frames, and the variance-boost family.
    "qp-scale-compress-strength": IntRange(0, 3),
    "luminance-qp-bias": IntRange(0, 100),
    "enable-variance-boost": _BOOL,
    "variance-boost-strength": IntRange(1, 4),
    "variance-octile": IntRange(1, 8),
    "variance-boost-curve": IntRange(0, 2),
    "irefresh-type": IntRange(1, 2),
    "lookahead": IntRange(-1, 120),
    "lp": IntRange(0, 128),
    "pin": _BOOL,
    "fast-decode": IntRange(0, 2),
    "tile-columns": IntRange(0, 6),
    "tile-rows": IntRange(0, 6),
    "input-depth": Choice(frozenset({"8", "10"})),
    "color-primaries": IntRange(0, 255),
    "transfer-characteristics": IntRange(0, 255),
    "matrix-coefficients": IntRange(0, 255),
    "color-range": _BOOL,
    "chroma-sample-position": IntRange(0, 3),
    "mastering-display": _MASTERING,
    "content-light": _LIGHT_LEVEL,
    "scm": IntRange(0, 2),
}

# --- x265 -------------------------------------------------------------------

X265_PARAMS: dict[str, Rule] = {
    "keyint": IntRange(-1, 10_000),
    "min-keyint": IntRange(0, 10_000),
    "open-gop": _BOOL,
    "scenecut": IntRange(0, 10_000),
    "aq-mode": IntRange(0, 4),
    "aq-strength": FloatRange(0.0, 3.0),
    "qcomp": FloatRange(0.5, 1.0),
    "cbqpoffs": IntRange(-12, 12),
    "crqpoffs": IntRange(-12, 12),
    "ipratio": FloatRange(1.0, 2.0),
    "pbratio": FloatRange(1.0, 2.0),
    "psy-rd": FloatRange(0.0, 5.0),
    "psy-rdoq": FloatRange(0.0, 50.0),
    "rd": IntRange(1, 6),
    "rdoq-level": IntRange(0, 2),
    "rc-lookahead": IntRange(0, 250),
    "ref": IntRange(1, 16),
    "bframes": IntRange(0, 16),
    "b-adapt": IntRange(0, 2),
    "me": IntRange(0, 5),
    "subme": IntRange(0, 7),
    "merange": IntRange(0, 32_768),
    # One number, applied to both the tC and beta offsets. The two-value "-1:-1" form
    # cannot be used here: see the colon note in this module's docstring.
    "deblock": IntRange(-6, 6),
    "sao": _BOOL,
    "limit-sao": _BOOL,
    "selective-sao": IntRange(0, 4),
    "cutree": _BOOL,
    "strong-intra-smoothing": _BOOL,
    "tu-intra-depth": IntRange(1, 4),
    "tu-inter-depth": IntRange(1, 4),
    "limit-refs": IntRange(0, 3),
    "limit-modes": _BOOL,
    "early-skip": _BOOL,
    "rect": _BOOL,
    "amp": _BOOL,
    "weightb": _BOOL,
    "weightp": _BOOL,
    "hdr10": _BOOL,
    "hdr10-opt": _BOOL,
    "repeat-headers": _BOOL,
    "aud": _BOOL,
    "hrd": _BOOL,
    "high-tier": _BOOL,
    "level-idc": Pattern(re.compile(r"\d(?:\.\d)?|\d{2,3}")),
    "colorprim": Choice(frozenset(CODE_BY_PRIMARIES)),
    "transfer": Choice(frozenset(CODE_BY_TRANSFER)),
    "colormatrix": Choice(frozenset(CODE_BY_MATRIX)),
    "range": Choice(frozenset({"limited", "full"})),
    "master-display": Pattern(
        re.compile(r"G\(\d+,\d+\)B\(\d+,\d+\)R\(\d+,\d+\)WP\(\d+,\d+\)L\(\d+,\d+\)")
    ),
    "max-cll": _LIGHT_LEVEL,
    "chromaloc": IntRange(0, 5),
    "frame-threads": IntRange(0, 16),
}

ALLOWED_PARAMS: dict[Encoder, dict[str, Rule]] = {
    Encoder.SVT_AV1: SVT_AV1_PARAMS,
    Encoder.X265: X265_PARAMS,
}

# Parameters derived from the source's own colour signalling. A model has no business
# changing these — it cannot see the file — so they are restored after validation.
PROTECTED_PARAMS: dict[Encoder, frozenset[str]] = {
    Encoder.SVT_AV1: frozenset(
        {
            "color-primaries",
            "transfer-characteristics",
            "matrix-coefficients",
            "color-range",
            "mastering-display",
            "content-light",
        }
    ),
    Encoder.X265: frozenset(
        {
            "colorprim",
            "transfer",
            "colormatrix",
            "range",
            "hdr10",
            "hdr10-opt",
            "master-display",
            "max-cll",
        }
    ),
}

X265_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    }
)
X265_TUNES = frozenset({"grain", "psnr", "ssim", "fastdecode", "zerolatency", "animation"})

CRF_LIMITS: dict[Encoder, tuple[float, float]] = {
    Encoder.SVT_AV1: (10.0, 55.0),
    Encoder.X265: (10.0, 40.0),
}


@dataclass(slots=True)
class ValidationResult:
    params: dict[str, str]
    rejected: list[str]


def validate_params(encoder: Encoder, params: dict[str, str]) -> ValidationResult:
    """Keep the parameters that are allowed and in range; report the rest."""
    allowed = ALLOWED_PARAMS[encoder]
    kept: dict[str, str] = {}
    rejected: list[str] = []

    for raw_name, raw_value in list(params.items())[:MAX_PARAMS]:
        name = str(raw_name).strip().casefold()
        value = str(raw_value).strip()

        if not _SAFE_NAME.fullmatch(name):
            rejected.append(f"{raw_name!r} (not a parameter name)")
            continue
        if not _SAFE_VALUE.fullmatch(value):
            rejected.append(f"{name} (unsafe or empty value)")
            continue
        rule = allowed.get(name)
        if rule is None:
            rejected.append(f"{name} (not a recognised {encoder.value} parameter)")
            continue
        if not rule.accepts(value):
            rejected.append(f"{name}={value} (out of range)")
            continue
        kept[name] = value

    if len(params) > MAX_PARAMS:
        rejected.append(f"{len(params) - MAX_PARAMS} further parameters (over the limit)")

    return ValidationResult(params=kept, rejected=rejected)


def validate_crf(encoder: Encoder, crf: float) -> tuple[float, str | None]:
    """Clamp a CRF to the encoder's usable range, and only to that.

    How far a model may move the CRF from the rules engine's proposal is deliberately
    not bounded: the model is the one that has read the film, its technical rows and the
    parse of the source, and a table keyed on resolution and negative format is a
    starting point rather than a verdict. What stays enforced is the encoder's own legal
    range, because a number outside it is not a judgement — it is a broken command.
    """
    low, high = CRF_LIMITS[encoder]
    bounded = max(low, min(float(crf), high))

    if abs(bounded - float(crf)) < 0.01:
        return round(bounded, 1), None
    return (
        round(bounded, 1),
        f"CRF {crf:g} was clamped to {bounded:g} — outside {encoder.value}'s usable "
        f"range of {low:g}–{high:g}.",
    )


def validate_preset(encoder: Encoder, preset: str, baseline: str) -> tuple[str, str | None]:
    """SVT-AV1 presets are numbers 0–13; x265's are names."""
    candidate = str(preset).strip().casefold()
    if encoder is Encoder.SVT_AV1:
        if candidate.isdigit() and 0 <= int(candidate) <= 13:
            return candidate, None
        return baseline, f"preset {preset!r} is not an SVT-AV1 preset (0–13)."
    if candidate in X265_PRESETS:
        return candidate, None
    return baseline, f"preset {preset!r} is not an x265 preset."


def validate_tune(encoder: Encoder, tune: str | None) -> tuple[str | None, str | None]:
    if tune is None or not str(tune).strip():
        return None, None
    candidate = str(tune).strip().casefold()
    if encoder is Encoder.SVT_AV1:
        # SVT-AV1's tune is numeric and lives in the parameter string instead.
        return None, f"tune {tune!r} ignored: SVT-AV1 takes tune as a parameter, not a name."
    if candidate in X265_TUNES:
        return candidate, None
    return None, f"tune {tune!r} is not an x265 tune."
