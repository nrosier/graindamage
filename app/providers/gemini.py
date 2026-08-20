"""Gemini as a second opinion on the deterministic plan — never as the only opinion.

The rules engine has already produced complete, usable settings by the time this
module runs. Gemini's job is narrow: look at the film, the source and the baseline,
and say where the baseline is wrong. It is asked for structured output
(``responseMimeType: application/json`` plus a ``responseSchema``), and everything it
returns passes through :mod:`app.advice.validate` before it can reach a command line.

What the model is *allowed* to change:

* CRF, within :data:`app.advice.validate.MAX_CRF_DRIFT` of the baseline.
* Preset and (for x265) tune, from the real ladders.
* Allowlisted encoder parameters — it can add or change them, but not remove them.
  Losing ``keyint`` or a colour tag because a model omitted it from its answer is a
  worse failure than being unable to drop ``sao=0``.
* The prose: summary, notes, warnings, per-plan rationale.

What it can never change: the colour and HDR parameters. Those are read off the
source file, which the model cannot see, so it has nothing to contribute and a
corrupted ``master-display`` string would silently ruin the encode.

Any failure at all — no key, timeout, refusal, truncated JSON, schema mismatch —
returns the baseline with a warning. There is no path where a Gemini problem costs
the user their advice.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import httpx2

from app.advice.rules import estimate_bitrate
from app.advice.validate import (
    MAX_CRF_DRIFT,
    PROTECTED_PARAMS,
    SVT_AV1_PARAMS,
    X265_PARAMS,
    X265_PRESETS,
    X265_TUNES,
    validate_crf,
    validate_params,
    validate_preset,
    validate_tune,
)
from app.cache import TTLCache
from app.config import Settings
from app.models import (
    Adjustment,
    Advice,
    AdviceSource,
    Encoder,
    EncodeRequest,
    EncoderPlan,
)
from app.providers import ProviderDisabled, ProviderUnavailable

# --- limits on what the model can put on the page ---------------------------

MAX_SUMMARY_CHARS = 400
MAX_LINE_CHARS = 400
MAX_NOTES = 8
MAX_WARNINGS = 6
MAX_RATIONALE = 6
MAX_OVERVIEW_CHARS = 600

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")

# --- the structured-output contract -----------------------------------------

# Gemini's responseSchema is an OpenAPI 3.0 subset with no additionalProperties, so
# the parameter map is carried as a list of name/value pairs rather than an object.
_PARAM_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "name": {"type": "STRING", "description": "Parameter name, e.g. aq-strength"},
        "value": {"type": "STRING", "description": "Parameter value as a string"},
    },
    "required": ["name", "value"],
}

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "encoder": {"type": "STRING", "enum": [Encoder.SVT_AV1.value, Encoder.X265.value]},
        "crf": {"type": "NUMBER", "description": "Constant-quality value for this encoder"},
        "preset": {"type": "STRING", "description": "0-13 for SVT-AV1, a name for x265"},
        "tune": {
            "type": "STRING",
            "description": "x265 tune name, or an empty string. Never set for SVT-AV1.",
        },
        "params": {"type": "ARRAY", "items": _PARAM_SCHEMA},
        "rationale": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Only points the baseline rationale does not already make",
        },
    },
    "required": ["encoder", "crf", "preset", "rationale"],
}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING", "description": "One sentence on this specific film"},
        "notes": {"type": "ARRAY", "items": {"type": "STRING"}},
        "warnings": {"type": "ARRAY", "items": {"type": "STRING"}},
        "plans": {"type": "ARRAY", "items": _PLAN_SCHEMA},
    },
    "required": ["summary", "plans"],
}

SYSTEM_PROMPT = f"""\
You are a video encoding engineer reviewing a proposed archival re-encode of a film. \
You are given the film's production details, IMDb technical rows, a parse of the \
user's source file, and a baseline plan produced by a deterministic rules engine.

Your job is to correct the baseline where it is wrong for this particular film, and to \
say what the rules engine cannot know. The baseline is usually close. Returning it \
almost unchanged, with one or two genuinely film-specific observations, is a good \
answer; inventing changes to look useful is not.

Rules you must follow:

1. Return both encoder plans, using the exact encoder ids given.
2. Keep each CRF within {MAX_CRF_DRIFT:g} points of the baseline CRF for that encoder. \
Anything further is clamped and your reasoning is discarded.
3. SVT-AV1 presets are integers 0-13 (lower is slower). x265 presets are names: \
{", ".join(sorted(X265_PRESETS))}. x265 tunes are: {", ".join(sorted(X265_TUNES))}. \
SVT-AV1 has no named tune — its tune is the numeric `tune` parameter.
4. Only these SVT-AV1 parameters exist for you: {", ".join(sorted(SVT_AV1_PARAMS))}.
5. Only these x265 parameters exist for you: {", ".join(sorted(X265_PARAMS))}.
6. Do not return colour, HDR, mastering-display or content-light parameters. They are \
read from the source file and will be overwritten with the file's own values.
7. Parameters you omit keep their baseline values. You cannot delete a parameter, only \
change or add one.
8. Prose must be specific and plain. No marketing adjectives, no restating the \
baseline's own rationale, no explaining what CRF is. If you have nothing to add for a \
plan, return an empty rationale list.
9. Grain is the priority: this tool exists to stop film grain being smoothed into \
mush. If your suggestion trades grain for size, say so explicitly.
10. If the source looks like a bad candidate for re-encoding at all, put that in \
warnings rather than quietly encoding it anyway.
"""


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _prose(value: object, limit: int = MAX_LINE_CHARS) -> str | None:
    """A model-supplied string, made safe to put in a list on a page."""
    if not isinstance(value, str):
        return None
    text = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub("", value)).strip()
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _lines(value: object, *, limit: int, seen: set[str]) -> list[str]:
    """Deduplicated, capped, cleaned prose lines."""
    out: list[str] = []
    for item in _as_list(value):
        line = _prose(item)
        if line is None:
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= limit:
            break
    return out


def _params_from_pairs(value: object) -> dict[str, str]:
    """The name/value array from the schema, back into a mapping."""
    params: dict[str, str] = {}
    for entry in _as_list(value):
        row = _as_dict(entry)
        name, raw = row.get("name"), row.get("value")
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(raw, bool):
            params[name] = "1" if raw else "0"
        elif isinstance(raw, int | float):
            params[name] = f"{raw:g}"
        elif isinstance(raw, str):
            params[name] = raw
    return params


def _uses_synthesis(plan: EncoderPlan) -> bool:
    strength = plan.params.get("film-grain")
    if not strength or strength == "0":
        return False
    # Without the denoiser the real grain is coded too, so there is no saving.
    return plan.params.get("film-grain-denoise") != "0"


class GeminiClient:
    """Structured encoding advice from Gemini, validated against the baseline."""

    def __init__(self, settings: Settings, *, client: httpx2.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client
        # The API payload is cached, not the merged Advice: Advice objects are mutated
        # downstream (commands are attached to their plans), so they must not be shared.
        self._cache: TTLCache[dict[str, Any]] = TTLCache(
            ttl_seconds=float(settings.cache_ttl_seconds)
        )

    @property
    def enabled(self) -> bool:
        return self._settings.gemini_enabled

    async def annotate(self, request: EncodeRequest, baseline: Advice) -> Advice:
        """Return the baseline reviewed by Gemini, or the baseline plus a warning.

        Never raises. A model that is unreachable, unhelpful or wrong costs the user
        nothing except the annotation they were hoping for.
        """
        if not self.enabled:
            return baseline

        try:
            payload = await self._advice_payload(request, baseline)
        except ProviderUnavailable as exc:
            reviewed = baseline.model_copy(deep=True)
            detail = f" ({exc.detail})" if exc.detail else ""
            reviewed.warnings.append(
                f"{exc.message}{detail} The settings below are the deterministic "
                "baseline, which is complete on its own."
            )
            return reviewed

        return merge_advice(baseline, payload, request)

    # --- internals ---------------------------------------------------------

    async def _advice_payload(self, request: EncodeRequest, baseline: Advice) -> dict[str, Any]:
        context = build_context(request, baseline)
        body = json.dumps(context, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(
            f"{self._settings.gemini_model}|{self._settings.gemini_temperature}|{body}".encode()
        ).hexdigest()
        return await self._cache.get_or_set(key, lambda: self._generate(body))

    async def _generate(self, user_text: str) -> dict[str, Any]:
        if not self._settings.gemini_api_key:
            raise ProviderDisabled("Gemini advice is not configured. Set GEMINI_API_KEY.")

        url = (
            f"{self._settings.gemini_base_url.rstrip('/')}"
            f"/models/{self._settings.gemini_model}:generateContent"
        )
        # Header auth rather than ?key=, so the key cannot end up in a proxy's log.
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self._settings.user_agent,
            "x-goog-api-key": self._settings.gemini_api_key,
        }
        request_body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature": self._settings.gemini_temperature,
                "maxOutputTokens": self._settings.gemini_max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
            },
        }

        try:
            if self._client is not None:
                response = await self._client.post(url, json=request_body, headers=headers)
            else:
                async with httpx2.AsyncClient(
                    timeout=self._settings.gemini_timeout_seconds
                ) as client:
                    response = await client.post(url, json=request_body, headers=headers)
        except httpx2.HTTPError as exc:
            raise ProviderUnavailable("Could not reach Gemini.", detail=type(exc).__name__) from exc

        if response.status_code in {400, 403}:
            raise ProviderUnavailable(
                f"Gemini rejected the request ({response.status_code}) — usually an "
                "invalid API key or a model name your key cannot use."
            )
        if response.status_code == 404:
            raise ProviderUnavailable(
                f"Gemini has no model named {self._settings.gemini_model} (404)."
            )
        if response.status_code == 429:
            raise ProviderUnavailable("Gemini rate limit reached — try again shortly (429).")
        if response.status_code >= 400:
            raise ProviderUnavailable(f"Gemini returned HTTP {response.status_code}.")

        try:
            envelope = response.json()
        except ValueError as exc:
            raise ProviderUnavailable("Gemini returned a response that was not JSON.") from exc

        return _extract_payload(_as_dict(envelope))


def _extract_payload(envelope: dict[str, Any]) -> dict[str, Any]:
    """Pull the JSON object out of Gemini's ``candidates[0].content.parts``."""
    block_reason = _as_dict(envelope.get("promptFeedback")).get("blockReason")
    if isinstance(block_reason, str) and block_reason:
        raise ProviderUnavailable(f"Gemini declined to answer ({block_reason}).")

    candidates = _as_list(envelope.get("candidates"))
    if not candidates:
        raise ProviderUnavailable("Gemini returned no candidates.")

    candidate = _as_dict(candidates[0])
    finish = candidate.get("finishReason")
    if finish == "MAX_TOKENS":
        raise ProviderUnavailable(
            "Gemini's answer was cut off by the output token limit — raise "
            "GEMINI_MAX_OUTPUT_TOKENS."
        )
    if isinstance(finish, str) and finish not in {"STOP", "MAX_TOKENS", ""}:
        raise ProviderUnavailable(f"Gemini stopped early ({finish}).")

    chunks: list[str] = []
    for raw in _as_list(_as_dict(candidate.get("content")).get("parts")):
        # Thinking parts and function calls have no "text"; skip them silently.
        if isinstance(chunk := _as_dict(raw).get("text"), str):
            chunks.append(chunk)
    text = "".join(chunks)
    if not text.strip():
        raise ProviderUnavailable("Gemini returned an empty answer.")

    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ProviderUnavailable("Gemini's answer was not the JSON it was asked for.") from exc
    if not isinstance(payload, dict):
        raise ProviderUnavailable("Gemini's answer was not a JSON object.")
    return payload


# --- prompt context ---------------------------------------------------------


def build_context(request: EncodeRequest, baseline: Advice) -> dict[str, Any]:
    """The facts the model gets. Compact on purpose: no keys, no paths, no filenames.

    The input path is deliberately excluded — it is a local filesystem path that the
    model has no use for and no business seeing.
    """
    context: dict[str, Any] = {
        "preferences": {"speed": request.speed.value, "size": request.size.value},
        "grain_estimate": {
            "level": baseline.grain.level.value,
            "confidence": round(baseline.grain.confidence, 2),
            "origin_format": baseline.grain.origin_format,
            "reasons": baseline.grain.reasons,
            "set_by_user": baseline.grain.user_override,
        },
        "baseline_plans": [
            {
                "encoder": plan.encoder.value,
                "crf": plan.crf,
                "preset": plan.preset,
                "tune": plan.tune,
                "params": plan.params,
                "rationale": plan.rationale,
                "estimated_bitrate_bps": plan.estimated_bitrate_bps,
            }
            for plan in baseline.plans
        ],
        "baseline_notes": baseline.notes,
        "baseline_warnings": baseline.warnings,
    }

    if (movie := request.movie) is not None:
        context["film"] = {
            "title": movie.title,
            "original_title": movie.original_title,
            "year": movie.year,
            "directors": movie.directors,
            "genres": movie.genres,
            "countries": movie.countries,
            "runtime_minutes": movie.runtime_minutes,
            "overview": (movie.overview or "")[:MAX_OVERVIEW_CHARS] or None,
        }

    specs = {
        name: value
        for name in type(request.specs).model_fields
        if (value := getattr(request.specs, name))
    }
    if specs:
        context["imdb_technical"] = specs

    media = request.source
    source: dict[str, Any] = {
        "container": media.container,
        "duration_seconds": media.duration_seconds,
        "size_bytes": media.size_bytes,
        "overall_bitrate_bps": media.overall_bitrate_bps,
        "resolution_class": media.resolution_label,
        "parsed_from": request.source_tool.value if request.source_tool else None,
    }
    if (video := media.video) is not None:
        source["video"] = {
            "codec": video.codec,
            "profile": video.profile,
            "width": video.width,
            "height": video.height,
            "display_aspect_ratio": video.display_aspect_ratio,
            "frame_rate": video.frame_rate,
            "bit_depth": video.bit_depth,
            "chroma_subsampling": video.chroma_subsampling,
            "scan_type": video.scan_type,
            "bitrate_bps": video.bitrate_bps,
            "bits_per_pixel": (
                round(bpp, 4) if (bpp := video.bits_per_pixel) is not None else None
            ),
            "color_primaries": video.color_primaries,
            "color_transfer": video.color_transfer,
            "color_matrix": video.color_matrix,
            "color_range": video.color_range,
            "hdr": video.is_hdr,
            "dolby_vision": video.dolby_vision,
            "hdr10_plus": video.hdr10_plus,
            "has_mastering_display": video.mastering_display is not None,
            "max_cll": video.max_cll,
        }
    if media.audio:
        source["audio"] = [
            {
                "codec": track.codec,
                "channels": track.channels,
                "language": track.language,
                "lossless": track.is_lossless,
                "bitrate_bps": track.bitrate_bps,
            }
            for track in media.audio[:8]
        ]
    if media.subtitles:
        source["subtitle_count"] = len(media.subtitles)
    context["source"] = source

    return context


# --- merging ----------------------------------------------------------------


def merge_advice(baseline: Advice, payload: dict[str, Any], request: EncodeRequest) -> Advice:
    """Fold a validated model payload into the baseline. Never raises."""
    reviewed = baseline.model_copy(deep=True)
    rejected: list[str] = []

    if (summary := _prose(payload.get("summary"), MAX_SUMMARY_CHARS)) is not None:
        reviewed.summary = summary

    reviewed.notes.extend(
        _lines(
            payload.get("notes"),
            limit=MAX_NOTES,
            seen={line.casefold() for line in reviewed.notes},
        )
    )
    reviewed.warnings.extend(
        _lines(
            payload.get("warnings"),
            limit=MAX_WARNINGS,
            seen={line.casefold() for line in reviewed.warnings},
        )
    )

    handled: set[Encoder] = set()
    for raw in _as_list(payload.get("plans")):
        row = _as_dict(raw)
        encoder = _encoder_from(row.get("encoder"))
        if encoder is None:
            rejected.append(f"a plan for unknown encoder {row.get('encoder')!r}")
            continue
        if encoder in handled:
            rejected.append(f"a second {encoder.value} plan")
            continue
        plan = reviewed.plan_for(encoder)
        if plan is None:
            rejected.append(f"a {encoder.value} plan the baseline does not have")
            continue
        handled.add(encoder)
        rejected.extend(_apply_plan(plan, row, baseline, request))

    reviewed.source = AdviceSource.GEMINI
    reviewed.rejected_flags = rejected
    return reviewed


def _encoder_from(value: object) -> Encoder | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().casefold().replace("_", "-")
    aliases = {
        "svt-av1": Encoder.SVT_AV1,
        "av1": Encoder.SVT_AV1,
        "libsvtav1": Encoder.SVT_AV1,
        "x265": Encoder.X265,
        "hevc": Encoder.X265,
        "libx265": Encoder.X265,
        "h265": Encoder.X265,
    }
    return aliases.get(candidate)


def _apply_plan(
    plan: EncoderPlan, row: dict[str, Any], baseline: Advice, request: EncodeRequest
) -> list[str]:
    """Validate one model plan onto its baseline plan, in place."""
    rejected: list[str] = []
    encoder = plan.encoder
    original_crf = plan.crf

    raw_crf = row.get("crf")
    if isinstance(raw_crf, int | float) and not isinstance(raw_crf, bool):
        crf, complaint = validate_crf(encoder, float(raw_crf), original_crf)
        if complaint:
            rejected.append(complaint)
        plan.crf = crf
    elif raw_crf is not None:
        rejected.append(f"CRF {raw_crf!r} (not a number)")

    if (raw_preset := row.get("preset")) is not None:
        preset, complaint = validate_preset(encoder, str(raw_preset), plan.preset)
        if complaint:
            rejected.append(complaint)
        plan.preset = preset

    if "tune" in row:
        tune, complaint = validate_tune(encoder, row.get("tune"))
        if complaint:
            rejected.append(complaint)
        # An x265 tune of None means the model explicitly cleared it, which is a
        # legitimate call — grain tuning is not always right.
        elif encoder is Encoder.X265:
            plan.tune = tune

    result = validate_params(encoder, _params_from_pairs(row.get("params")))
    rejected.extend(result.rejected)
    plan.params.update(result.params)

    # The source's own colour signalling always wins: the model has not seen the file.
    if (baseline_plan := baseline.plan_for(encoder)) is not None:
        for name in PROTECTED_PARAMS[encoder]:
            if name in baseline_plan.params:
                plan.params[name] = baseline_plan.params[name]
            else:
                plan.params.pop(name, None)

    plan.rationale.extend(
        _lines(
            row.get("rationale"),
            limit=MAX_RATIONALE,
            seen={line.casefold() for line in plan.rationale},
        )
    )

    if abs(plan.crf - original_crf) >= 0.01:
        plan.adjustments.append(
            Adjustment(
                label="Gemini",
                delta=round(plan.crf - original_crf, 1),
                detail="Gemini's adjustment for this particular film.",
            )
        )
        plan.estimated_bitrate_bps = estimate_bitrate(
            encoder,
            plan.crf,
            request.source.video,
            baseline.grain.level,
            synthesised=_uses_synthesis(plan),
        )
    elif _uses_synthesis(plan) != _uses_synthesis_of(baseline, encoder):
        plan.estimated_bitrate_bps = estimate_bitrate(
            encoder,
            plan.crf,
            request.source.video,
            baseline.grain.level,
            synthesised=_uses_synthesis(plan),
        )

    return rejected


def _uses_synthesis_of(advice: Advice, encoder: Encoder) -> bool:
    plan = advice.plan_for(encoder)
    return _uses_synthesis(plan) if plan is not None else False


__all__ = [
    "RESPONSE_SCHEMA",
    "SYSTEM_PROMPT",
    "GeminiClient",
    "build_context",
    "merge_advice",
]
