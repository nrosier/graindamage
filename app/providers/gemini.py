"""Gemini decides the settings. Nothing here proposes them first.

This is where the advice is made. The model is given the film's production details, the
IMDb technical rows, a full parse of the user's source file, the preferences and a
heuristic grain estimate, and it answers with the whole thing: CRF, the named steps that
add up to it, preset, tune, the complete parameter set, the grain profile it settled on,
and the prose. That answer *is* the advice — it is validated, not edited.

That is a deliberate reversal. :mod:`app.advice.rules` used to produce a plan first,
send it here as a proposal, and receive the model's answer folded back on top of it, so
every parameter the model did not mention silently kept its table value and the model's
CRF was shown as one delta away from a table. The tables now do one job: they are the
answer when there is no model at all, labelled as such by
:data:`~app.advice.pipeline.TABLES_ONLY_NOTE`. They are never sent, and they never fill
a gap in what came back.

It is asked for structured output (``responseMimeType: application/json`` plus a
``responseSchema``), and everything it returns passes through
:mod:`app.advice.validate` before it can reach a command line.

What is checked, and what is left alone:

* CRF is held to the encoder's own legal range and nothing tighter. A number outside it
  is a broken command; a number inside it is a judgement, and the model is the only
  party here that has read the film.
* Presets, tunes and parameter names and ranges come from the real allowlists. A
  rejected one is reported to the user rather than quietly swapped.
* The account of the CRF is the model's own labelled factors, and they are checked to
  add up to the number they claim to explain.
* Colour and HDR parameters are read off the source file and forced over whatever came
  back: the model cannot see the file, so it has nothing to contribute and a corrupted
  ``master-display`` string would silently ruin the encode. ``keyint`` is the one
  parameter supplied when the model omits it — ten seconds of frames is arithmetic, and
  a command line that lost its keyframe interval is a worse failure than a convention
  applied on the model's behalf.
* Coherence is reported, never overruled: a plan that codes real grain at a CRF too high
  to carry it gets a warning naming both halves, because that pairing is the commonest
  way for good reasoning about each setting separately to add up to a bad encode.

Any failure at all — no key, timeout, refusal, truncated JSON, an answer with no usable
plan in it — comes back as a :class:`~app.advice.pipeline.Decision` carrying the reason
and no advice, and the pipeline falls back to the tables with that reason shown. There is
no path where a Gemini problem costs the user their settings, and none where it quietly
half-costs them.

This module has one other job: :meth:`GeminiClient.technical_specs` asks for a film's
IMDb ``/technical`` rows, because IMDb's WAF means the app cannot read that page
itself. That answer is a recollection, not a scrape, so it is labelled as one
everywhere it appears — on the page, and in the context this module later sends back.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx2

from app.advice.pipeline import Decision
from app.advice.rules import (
    REAL_GRAIN_CRF_CEILING,
    colour_params,
    estimate_bitrate,
    grain_for,
    keyint_for,
)
from app.advice.validate import (
    CRF_LIMITS,
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
    GrainLevel,
    GrainProfile,
    SpecsSource,
    TechnicalSpecs,
    VideoTrack,
)
from app.providers import (
    ProviderDisabled,
    ProviderRateLimited,
    ProviderRejected,
    ProviderUnavailable,
)
from app.providers.imdb import SPEC_LABELS, specs_from_rows

# --- limits on what the model can put on the page ---------------------------

MAX_SUMMARY_CHARS = 400
MAX_LINE_CHARS = 400
MAX_NOTES = 8
MAX_WARNINGS = 6
MAX_RATIONALE = 6
MAX_ADJUSTMENTS = 8
MAX_GRAIN_REASONS = 6
MAX_LABEL_CHARS = 60
MAX_OVERVIEW_CHARS = 600

# A CRF account is arithmetic, so it either adds up or it does not; this is the slack
# allowed for the model writing 0.1 as 0.09999.
ACCOUNT_TOLERANCE = 0.05

# What a preset falls back to when the model returns one the encoder does not have. Not a
# proposal — the model was not shown these, and a plan that reaches here has already been
# reported to the user as having named a preset that does not exist.
_DEFAULT_PRESET: dict[Encoder, str] = {Encoder.SVT_AV1: "5", Encoder.X265: "slow"}

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_CODE_FENCE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL)

# Appended to a system prompt when the API will not let us ask for JSON properly.
_JSON_ONLY = (
    "Reply with a single JSON object matching the fields described above, and "
    "nothing else: no prose before or after it, no markdown code fence."
)
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

# The account of the CRF, which is printed to the user as a table that has to total.
_ADJUSTMENT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING", "description": 'Short name, e.g. "2160p starting point"'},
        "delta": {
            "type": "NUMBER",
            "description": "The first factor's whole value; every later one signed",
        },
        "detail": {"type": "STRING", "description": "One sentence on why, or omitted"},
    },
    "required": ["label", "delta"],
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
        "params": {
            "type": "ARRAY",
            "items": _PARAM_SCHEMA,
            "description": "Every parameter this encode needs — omitted means absent",
        },
        "adjustments": {
            "type": "ARRAY",
            "items": _ADJUSTMENT_SCHEMA,
            "description": "The named factors that sum to crf, in the order you applied them",
        },
        "rationale": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Why these settings, for this film. The only explanation there is.",
        },
    },
    "required": ["encoder", "crf", "preset", "params", "adjustments", "rationale"],
}

_GRAIN_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "level": {"type": "STRING", "enum": [level.value for level in GrainLevel]},
        "confidence": {"type": "NUMBER", "description": "0.0-1.0, honestly"},
        "origin_format": {
            "type": "STRING",
            "description": 'What the picture came off, e.g. "35 mm", "16 mm", "digital"',
        },
        "reasons": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "The evidence for this level, from the rows and the source parse",
        },
    },
    "required": ["level", "confidence", "reasons"],
}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING", "description": "One sentence on this specific film"},
        "grain": _GRAIN_SCHEMA,
        "notes": {"type": "ARRAY", "items": {"type": "STRING"}},
        "warnings": {"type": "ARRAY", "items": {"type": "STRING"}},
        "plans": {"type": "ARRAY", "items": _PLAN_SCHEMA},
    },
    "required": ["summary", "grain", "plans"],
}

# One row per TechnicalSpecs field, described by the label IMDb prints above it.
_ROW_DESCRIPTIONS: dict[str, str] = {
    "negative_formats": 'Negative format, e.g. "35 mm", "65 mm", "Digital"',
    "cinematographic_processes": 'Cinematographic process, e.g. "Super 35", "Spherical"',
    "printed_formats": 'Printed film format, e.g. "35 mm", "D-Cinema"',
    "cameras": "Camera and lens names, verbatim",
    "aspect_ratios": 'Aspect ratio, exactly as IMDb writes it, e.g. "2.39 : 1"',
    "film_lengths": 'Film length, with its unit, e.g. "3,024 m"',
    "colors": 'Color, e.g. "Color", "Black and White"',
    "laboratories": "Laboratory names, with their locations",
    "sound_mixes": 'Sound mix, e.g. "Dolby Digital", "DTS"',
    "runtimes": "Runtime as IMDb lists it",
}

TECHNICAL_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        **{
            field: {"type": "ARRAY", "items": {"type": "STRING"}, "description": description}
            for field, description in _ROW_DESCRIPTIONS.items()
        },
        "confidence": {
            "type": "STRING",
            "enum": ["high", "medium", "low"],
            "description": "How sure you are that these are the rows IMDb actually lists",
        },
    },
    "required": ["confidence"],
}

TECHNICAL_PROMPT = f"""\
You look up the technical specifications a film has on its IMDb /technical page and \
return them as data. You are given a title, a year and an IMDb title id, and the id is \
authoritative: if the title you know for that id differs, answer for the id.

Return only the rows IMDb lists: {", ".join(sorted(SPEC_LABELS))}.

Rules you must follow:

1. An empty list is the right answer for a row you do not know. These rows decide how a \
film's grain is encoded, so a plausible-looking guess is worse than nothing — a missing \
row falls back to a documented heuristic, while a wrong one silently misdirects it.
2. Values verbatim, in IMDb's own spelling and spacing: "2.39 : 1", not "2.39:1"; \
"35 mm", not "35mm". One list entry per row value.
3. No prose, no commentary, no explanation, no uncertainty hedges inside the values.
4. Report confidence honestly: "high" only for a film whose page you clearly recall or \
have just read, "low" if you are reconstructing it from what the film is generally known \
to be. Do not inflate it.
5. Never invent a camera, laboratory or process to fill a row out.
"""

SYSTEM_PROMPT = f"""\
You are the video encoding engineer deciding how a film should be re-encoded. You are \
given the film's production details, its IMDb technical rows, a full parse of the user's \
source file, the user's preferences, and a heuristic grain estimate made from those rows.

Nothing else has decided anything. There is no proposal to edit and no default behind \
you: what you return is what the user encodes with, so every field has to be one you \
would defend. Decide what this particular film needs, and say why.

Rules you must follow:

1. Return both encoder plans, using the exact encoder ids given.
2. Put each CRF where this film needs it: {CRF_LIMITS[Encoder.SVT_AV1][0]:g}-\
{CRF_LIMITS[Encoder.SVT_AV1][1]:g} for SVT-AV1, {CRF_LIMITS[Encoder.X265][0]:g}-\
{CRF_LIMITS[Encoder.X265][1]:g} for x265. Say what the number costs or saves in size, \
because the user sees an estimate move with it.
3. Grain and CRF are one decision, not two, and separating them is the failure this tool \
exists to prevent. Real grain is high-frequency spatial noise, so if you leave it in the \
picture to be coded — SVT-AV1 film-grain-denoise=0, or x265, which has no synthesis and \
always codes it — you have to fund it. For SVT-AV1 that means about CRF \
{REAL_GRAIN_CRF_CEILING:g} or below at 2160p and lower again at 1080p, where a grain \
particle covers fewer pixels. Above that the bit budget runs out mid-frame and the grain \
breaks up into swirling blocks in motion, which is worse than either honest choice. If \
the size the user asked for will not pay for coded grain, denoise and re-synthesise \
instead: film-grain-denoise=1 with a strength that carries the whole look rather than a \
floor, and keep the higher CRF. What you must not return is real grain coded at a CRF \
that cannot carry it. If this particular film is the exception — a fine-grained late \
camera negative, a source already denoised by whoever mastered it — say so in the \
rationale and your numbers stand.
4. SVT-AV1 presets are integers 0-13 (lower is slower). x265 presets are names: \
{", ".join(sorted(X265_PRESETS))}. x265 tunes are: {", ".join(sorted(X265_TUNES))}. \
SVT-AV1 has no named tune — its tune is the numeric `tune` parameter, where 0 optimises \
for how the picture looks and 1 optimises PSNR, which systematically prefers smoothing \
grain away.
5. Only these SVT-AV1 parameters exist for you: {", ".join(sorted(SVT_AV1_PARAMS))}.
6. Only these x265 parameters exist for you: {", ".join(sorted(X265_PARAMS))}.
7. Do not return colour, HDR, mastering-display or content-light parameters. They are \
read from the source file and will be overwritten with the file's own values.
8. **Return the complete parameter set for each encoder.** There is nothing behind your \
answer: a parameter you omit is absent from the command line, not left at some default \
someone else chose. That includes the ones that are easy to forget because they are \
usually just there — keyint, and SVT-AV1's tune. The single exception is rule 7's colour \
signalling, which is filled in from the file. If you want a feature off, set its off \
value; omitting it does not turn it off.
9. Return `adjustments`: the named factors that add up to `crf`. The first is the value \
you started from, every later one a signed move away from it, and the total must equal \
`crf`. This is printed to the user as the account of the number — a CRF with no account \
is a number they cannot argue with — so the labels have to name real reasons \
("2160p starting point", "35 mm grain", "coded grain ceiling"), not restate the arithmetic.
10. Return `grain`: the level, what the picture came off, your confidence and the \
evidence. You are given a heuristic estimate; it is a guess from the technical rows and \
the source parse, and you may disagree with it — say so in the reasons if you do. When it \
is marked set_by_user, the level is the user's and is not yours to change.
11. Prose must be specific and plain. No marketing adjectives, no explaining what CRF is, \
no restating the numbers you have just given. The rationale is the only explanation the \
user gets for a plan, so each one needs at least the grain decision and the CRF in it.
12. Grain is the priority: this tool exists to stop film grain being smoothed into \
mush. If your settings trade grain for size, say so explicitly.
13. Answer with grain settings, not only a CRF. For a source shot on film, decide and \
justify, per encoder:
  - SVT-AV1: the film-grain strength, and above all film-grain-denoise — 1 denoises the \
picture and replaces its grain with a uniform synthetic field, 0 leaves the real grain \
to be coded and uses synthesis only as a floor where the quantiser flattened it. Say \
which of the two this film wants and why. Then the in-loop filters that smooth grain \
(enable-restoration, enable-cdef, enable-dlf), the rate control that decides whether \
grainy frames keep their bits (qp-scale-compress-strength, luminance-qp-bias, \
enable-variance-boost with variance-boost-strength and variance-octile), and sharpness, \
enable-tf and tune.
  - x265: tune grain raises psy-rd to 4.0 and psy-rdoq to 10 and turns off SAO, cu-tree, \
AQ and rskip — but it does not touch qcomp or the deblocking offsets, so move those \
yourself if you want them moved. Then aq-mode and aq-strength, sao / selective-sao / \
limit-sao, strong-intra-smoothing, psy-rd and psy-rdoq, rd and rdoq-level.
A plan that gives a CRF and no grain settings is not an answer for a film source.
14. Older films need their own answer, and the year alone is not it: stock, dupe \
negatives and optical printing are what decide the grain. A pre-1970 negative, a \
blow-up, and anything printed through a dupe carry coarser, higher-contrast grain than a \
late-1990s camera negative, and older restorations often carry scanner noise and gate \
weave on top of the real grain. Use the year, country, director and format rows you were \
given to say which of those this film is, then tune for that case — including whether \
its grain should be coded or synthesised rather than either by default.
15. No parameter value may contain a colon. The parameters are joined with ':' into one \
-svtav1-params / -x265-params string, and a colon inside a value makes the encoder read \
the far half as a parameter name and drop the one after it. Write deblock=-1, which x265 \
applies to both offsets, never deblock=-1:-1.
16. If the source looks like a bad candidate for re-encoding at all, put that in \
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


@dataclass(frozen=True, slots=True)
class SpecsLookup:
    """Technical rows Gemini recalled, with how much it trusts them.

    The confidence travels with the rows because it is the difference between "these
    are IMDb's rows" and "these are what the film is generally said to be", and only
    the user can decide whether that is good enough to encode from.
    """

    specs: TechnicalSpecs
    confidence: str
    grounded: bool

    @property
    def caveat(self) -> str:
        """One line for the page: where these rows came from, and what to do about it."""
        how = "found by searching the web" if self.grounded else "recalled from training data"
        return (
            f"These rows were {how} by Gemini, not read from IMDb — confidence "
            f"{self.confidence}. Check them against the technical page, and paste it "
            "below if anything is wrong."
        )

    @property
    def summary(self) -> str:
        """The same fact in a few words, for a list row with no room for a sentence."""
        how = "web search" if self.grounded else "recall"
        return f"Gemini {how} · confidence {self.confidence}"


def _uses_synthesis(plan: EncoderPlan) -> bool:
    strength = plan.params.get("film-grain")
    if not strength or strength == "0":
        return False
    # Without the denoiser the real grain is coded too, so there is no saving.
    return plan.params.get("film-grain-denoise") != "0"


class GeminiClient:
    """The settings, decided by Gemini and validated before they reach a command line."""

    def __init__(self, settings: Settings, *, client: httpx2.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client
        # The API payload is cached, not the Advice built from it: Advice objects are
        # mutated downstream (commands are attached to their plans), so must not be shared.
        self._cache: TTLCache[dict[str, Any]] = TTLCache(
            ttl_seconds=float(settings.cache_ttl_seconds)
        )

    @property
    def enabled(self) -> bool:
        return self._settings.gemini_enabled

    async def decide(self, request: EncodeRequest) -> Decision:
        """The settings Gemini decided on, or the reason there are none.

        Never raises. A model that is unreachable, unhelpful or wrong costs the user the
        decision they were hoping for and nothing else: the pipeline falls back to the
        tables and shows the sentence returned here alongside them.
        """
        if not self.enabled:
            return Decision()

        try:
            payload = await self._advice_payload(request)
        except ProviderUnavailable as exc:
            detail = f" ({exc.detail})" if exc.detail else ""
            return Decision(problem=f"{exc.message}{detail}")

        return advice_from(payload, request)

    async def technical_specs(
        self, imdb_id: str, *, title: str | None = None, year: int | None = None
    ) -> SpecsLookup:
        """Ask Gemini for the film's IMDb ``/technical`` rows.

        Raises :class:`ProviderDisabled` without a key and :class:`ProviderUnavailable`
        if the lookup fails. A model that answers honestly that it does not know the
        page returns empty rows, which is a success — the caller falls back to the
        paste box and the year-based heuristic.
        """
        if not self.enabled:
            raise ProviderDisabled(
                "Looking up the technical rows needs a Gemini key (GEMINI_API_KEY). "
                "Paste the technical page instead — that always works."
            )

        grounded = self._settings.gemini_web_grounding
        try:
            payload = await self._specs_payload(imdb_id, title, year, grounded=grounded)
        except (ProviderRejected, ProviderRateLimited):
            if not grounded:
                raise
            # Both refusals are about the search tool rather than the question, and both
            # are answered the same way: ask again without it. A model may simply not
            # have grounding; or it has it and the grounding quota — which is metered
            # separately from, and far more tightly than, plain generateContent — is
            # spent. Recalled rows are worth less than searched ones, and the lookup
            # says so, but they are worth much more than a 429.
            grounded = False
            payload = await self._specs_payload(imdb_id, title, year, grounded=False)

        confidence = payload.get("confidence")
        return SpecsLookup(
            specs=specs_from_rows(
                {field: _as_list(payload.get(field)) for field in TechnicalSpecs.model_fields}
            ),
            confidence=confidence if confidence in {"high", "medium", "low"} else "low",
            grounded=grounded,
        )

    # --- internals ---------------------------------------------------------

    async def _specs_payload(
        self, imdb_id: str, title: str | None, year: int | None, *, grounded: bool
    ) -> dict[str, Any]:
        named = f"{title} ({year})" if title and year else title or "the film"
        question = (
            f"Fetch the technical specifications of {named}, IMDb id {imdb_id}, from "
            f"IMDb: https://www.imdb.com/title/{imdb_id}/technical/"
        )
        key = hashlib.sha256(
            f"specs|{self._settings.gemini_model}|{grounded}|{question}".encode()
        ).hexdigest()
        return await self._cache.get_or_set(
            key,
            lambda: self._generate(
                question,
                system=TECHNICAL_PROMPT,
                # Grounding and structured output are mutually exclusive, so a grounded
                # lookup asks for JSON in words and is parsed leniently.
                schema=None if grounded else TECHNICAL_SCHEMA,
                tools=[{"google_search": {}}] if grounded else None,
                # A lookup is recall, not judgement: nothing here wants variation.
                temperature=0.0,
            ),
        )

    async def _advice_payload(self, request: EncodeRequest) -> dict[str, Any]:
        context = build_context(request)
        body = json.dumps(context, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(
            f"{self._settings.gemini_model}|{self._settings.gemini_temperature}|{body}".encode()
        ).hexdigest()
        return await self._cache.get_or_set(
            key, lambda: self._generate(body, system=SYSTEM_PROMPT, schema=RESPONSE_SCHEMA)
        )

    async def _generate(
        self,
        user_text: str,
        *,
        system: str,
        schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """POST one ``generateContent`` call and return the JSON object it answered.

        With ``schema`` the model is held to structured output. With ``tools`` it is
        not — the API refuses both at once — so the JSON is dug out of whatever prose
        came with it instead.
        """
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
        generation: dict[str, Any] = {
            "temperature": (
                self._settings.gemini_temperature if temperature is None else temperature
            ),
            "maxOutputTokens": self._settings.gemini_max_output_tokens,
        }
        if schema is not None:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = schema

        instruction = system if schema is not None else f"{system}\n{_JSON_ONLY}"
        request_body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": generation,
        }
        if tools:
            request_body["tools"] = tools

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
            raise ProviderRejected(
                f"Gemini rejected the request ({response.status_code}) — usually an "
                "invalid API key, a model name your key cannot use, or a feature this "
                "model does not have."
            )
        if response.status_code == 404:
            raise ProviderUnavailable(
                f"Gemini has no model named {self._settings.gemini_model} (404)."
            )
        if response.status_code == 429:
            raise ProviderRateLimited(
                "Gemini rate limit reached — try again shortly (429).",
                detail=_quota_detail(response),
            )
        if response.status_code >= 400:
            raise ProviderUnavailable(f"Gemini returned HTTP {response.status_code}.")

        try:
            envelope = response.json()
        except ValueError as exc:
            raise ProviderUnavailable("Gemini returned a response that was not JSON.") from exc

        return _extract_payload(_as_dict(envelope), lenient=schema is None)


def _quota_detail(response: httpx2.Response) -> str | None:
    """Which allowance ran out, if the 429 body says.

    Worth digging out because Gemini meters its features separately: a key with plenty
    of ``generateContent`` left can still be out of search-grounding requests, and
    "rate limit reached" on its own reads as though the whole key were spent.
    """
    try:
        error = _as_dict(_as_dict(response.json()).get("error"))
    except ValueError:
        return None

    for detail in _as_list(error.get("details")):
        for violation in _as_list(_as_dict(detail).get("violations")):
            named = _as_dict(violation)
            if (quota := _prose(named.get("quotaId") or named.get("quotaMetric"))) is not None:
                return quota
    return _prose(error.get("message"))


def _extract_payload(envelope: dict[str, Any], *, lenient: bool = False) -> dict[str, Any]:
    """Pull the JSON object out of Gemini's ``candidates[0].content.parts``.

    ``lenient`` is for answers that could not be schema-constrained: the object is
    fished out of a code fence or a sentence rather than being the whole reply.
    """
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
        payload = json.loads(_only_json(text) if lenient else text)
    except ValueError as exc:
        raise ProviderUnavailable("Gemini's answer was not the JSON it was asked for.") from exc
    if not isinstance(payload, dict):
        raise ProviderUnavailable("Gemini's answer was not a JSON object.")
    return payload


def _only_json(text: str) -> str:
    """The outermost JSON object in a reply that was allowed to say more than JSON."""
    if fenced := _CODE_FENCE.search(text):
        text = fenced.group("body")
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text


# --- prompt context ---------------------------------------------------------


_SPECS_PROVENANCE: dict[SpecsSource | None, str] = {
    SpecsSource.PASTED: "pasted from IMDb's technical page by the user — authoritative",
    SpecsSource.GEMINI: (
        "recalled by a language model that was asked for the IMDb page, not read from "
        "IMDb — treat as unverified and say so if a row looks wrong for this film"
    ),
}


def build_context(request: EncodeRequest) -> dict[str, Any]:
    """The facts the model decides on. Compact on purpose: no keys, no paths, no filenames.

    Everything known about the film, its negative and the source file goes in, because
    the model is the one weighing them against each other. Two things stay out. The input
    path is a local filesystem path the model has no use for and no business seeing. And
    there is no plan in here: a proposal to react to is an anchor, and the whole point of
    this call is that the decision has not been made yet.

    The grain estimate is the one thing that looks like an answer, and it is labelled as
    the guess it is — the model is free to disagree with it, and told so.
    """
    grain = grain_for(request)
    context: dict[str, Any] = {
        "preferences": {"speed": request.speed.value, "size": request.size.value},
        "grain_estimate": {
            "made_by": (
                "heuristic over the technical rows and the source parse — a guess to "
                "weigh, not a decision"
            ),
            "level": grain.level.value,
            "confidence": round(grain.confidence, 2),
            "origin_format": grain.origin_format,
            # Spelled out rather than left to be inferred from origin_format, because
            # the grain rules the model is asked to follow all hinge on it.
            "photochemical": grain.is_photochemical,
            "reasons": grain.reasons,
            "set_by_user": grain.user_override,
        },
        # Stated rather than enforced: the model is told the number so it can decide with
        # it instead of tripping over it. A plan that goes above it and says why in the
        # rationale is accepted as it stands, with a warning naming both halves.
        "limits": {"svt_av1_real_grain_crf_ceiling": REAL_GRAIN_CRF_CEILING},
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
    elif request.fallback_year is not None:
        # No film was looked up — the CLI's --year, or the year in the filename, is then
        # the only thing the era half of the grain rules has to work from.
        context["release_year"] = request.fallback_year

    specs = {
        name: value
        for name in type(request.specs).model_fields
        if (value := getattr(request.specs, name))
    }
    if specs:
        context["imdb_technical"] = specs
        # Where the rows came from changes how far they can be leant on, and the model
        # is not told to trust its own earlier recollection as though it were IMDb.
        if (origin := _SPECS_PROVENANCE.get(request.specs_source)) is not None:
            context["imdb_technical_source"] = origin

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
            # The model sets keyint itself, so it is given the convention rather than
            # left to work out ten seconds of frames from the frame rate.
            "ten_second_keyframe_interval": keyint_for(video),
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


# --- building the advice ----------------------------------------------------

NO_USABLE_PLAN = (
    "Gemini's answer had no usable encoder plan in it, so it decided nothing for this film."
)


def advice_from(payload: dict[str, Any], request: EncodeRequest) -> Decision:
    """Build the advice out of a model payload alone. Never raises.

    Every field comes from the answer. Nothing is inherited, defaulted from a table or
    merged with anything, and where the answer is unusable the fact is reported rather
    than papered over: a rejected preset or parameter goes to ``rejected_flags`` and is
    shown to the user, and an answer with no usable plan in it at all is returned as a
    problem, so the pipeline falls back to the tables *visibly* instead of blending them
    in field by field.
    """
    rejected: list[str] = []
    grain = _grain_from(payload.get("grain"), request, rejected)

    decided: dict[Encoder, EncoderPlan] = {}
    for raw in _as_list(payload.get("plans")):
        row = _as_dict(raw)
        encoder = _encoder_from(row.get("encoder"))
        if encoder is None:
            rejected.append(f"a plan for unknown encoder {row.get('encoder')!r}")
            continue
        if encoder in decided:
            rejected.append(f"a second {encoder.value} plan")
            continue
        if (plan := _plan_from(encoder, row, request, grain, rejected)) is not None:
            decided[encoder] = plan

    if not decided:
        return Decision(problem=NO_USABLE_PLAN)

    # Declaration order, not the order the model happened to answer in, so the page and
    # the report put the two encoders in the same places from one run to the next.
    plans = [decided[encoder] for encoder in Encoder if encoder in decided]
    rejected.extend(f"no {e.value} plan in the answer" for e in Encoder if e not in decided)

    warnings = _lines(payload.get("warnings"), limit=MAX_WARNINGS, seen=set())
    for plan in plans:
        if (complaint := _starved_grain(plan)) is not None:
            warnings.append(complaint)

    return Decision(
        advice=Advice(
            source=AdviceSource.GEMINI,
            grain=grain,
            plans=plans,
            summary=_prose(payload.get("summary"), MAX_SUMMARY_CHARS),
            notes=_lines(payload.get("notes"), limit=MAX_NOTES, seen=set()),
            warnings=warnings,
            rejected_flags=rejected,
        )
    )


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


def _plan_from(
    encoder: Encoder,
    row: dict[str, Any],
    request: EncodeRequest,
    grain: GrainProfile,
    rejected: list[str],
) -> EncoderPlan | None:
    """One validated plan, or ``None`` if there is no plan here to validate.

    A CRF is the one field with nothing sensible to fall back on: a preset or a parameter
    the encoder refuses can be reported and dropped, but a plan with no number in it is
    not a plan, and inventing one would be exactly the table-shaped guess this module
    stopped making.
    """
    raw_crf = row.get("crf")
    if not isinstance(raw_crf, int | float) or isinstance(raw_crf, bool):
        rejected.append(f"the {encoder.value} plan, whose CRF was {raw_crf!r} and not a number")
        return None

    crf, complaint = validate_crf(encoder, float(raw_crf))
    clamped = complaint is not None
    if complaint:
        rejected.append(complaint)

    preset, complaint = validate_preset(
        encoder, str(row.get("preset", "")), _DEFAULT_PRESET[encoder]
    )
    if complaint:
        rejected.append(complaint)

    tune, complaint = validate_tune(encoder, row.get("tune"))
    if complaint:
        rejected.append(complaint)

    result = validate_params(encoder, _params_from_pairs(row.get("params")))
    rejected.extend(result.rejected)

    plan = EncoderPlan(
        encoder=encoder,
        crf=crf,
        preset=preset,
        # SVT-AV1's tune is the numeric parameter, so a name there is never carried.
        tune=tune if encoder is Encoder.X265 else None,
        params=_file_derived(encoder, result.params, request.source.video),
        adjustments=_adjustments_from(
            row.get("adjustments"), encoder, crf, clamped=clamped, rejected=rejected
        ),
        rationale=_lines(row.get("rationale"), limit=MAX_RATIONALE, seen=set()),
    )
    plan.estimated_bitrate_bps = estimate_bitrate(
        encoder, crf, request.source.video, grain.level, synthesised=_uses_synthesis(plan)
    )
    return plan


def _file_derived(
    encoder: Encoder, params: dict[str, str], video: VideoTrack | None
) -> dict[str, str]:
    """The model's parameters, with the two things it could not know put in.

    Colour and HDR signalling is read off the source and forced over whatever came back:
    the model has not seen the file, so a value it returned there is a guess at best and
    a corrupted ``master-display`` string at worst, and either silently ruins the encode.

    ``keyint`` is the one parameter supplied when the model left it out. Ten seconds of
    frames is arithmetic on the frame rate rather than a judgement, and a command line
    with no keyframe interval in it is a worse failure than a convention applied on the
    model's behalf. Nothing else is defaulted — an omitted parameter is an absent one.
    """
    settled = dict(params)
    for name in PROTECTED_PARAMS[encoder]:
        settled.pop(name, None)
    settled.update(colour_params(encoder, video))
    settled.setdefault("keyint", str(keyint_for(video)))
    return settled


def _adjustments_from(
    value: object,
    encoder: Encoder,
    crf: float,
    *,
    clamped: bool,
    rejected: list[str],
) -> list[Adjustment]:
    """The model's own account of its CRF, checked to add up to the number it explains.

    The user is shown these as a table that totals to the final CRF, so factors that do
    not add up are not a rounding curiosity — they are a table that reads as broken. The
    difference is shown as its own row rather than absorbed anywhere: when the CRF was
    clamped to the encoder's range, that row is the true reason and says so; otherwise
    the model's arithmetic slipped, and the row and a rejection both say that instead.
    """
    factors: list[Adjustment] = []
    for entry in _as_list(value):
        row = _as_dict(entry)
        delta = row.get("delta")
        label = _prose(row.get("label"), MAX_LABEL_CHARS)
        if label is None or not isinstance(delta, int | float) or isinstance(delta, bool):
            continue
        factors.append(
            Adjustment(label=label, delta=round(float(delta), 1), detail=_prose(row.get("detail")))
        )
        if len(factors) >= MAX_ADJUSTMENTS:
            break

    if not factors:
        rejected.append(f"the {encoder.value} plan's account of its CRF — no usable factors")
        return [Adjustment(label="Gemini's CRF for this film", delta=crf, detail=None)]

    residual = round(crf - sum(factor.delta for factor in factors), 1)
    if abs(residual) < ACCOUNT_TOLERANCE:
        return factors

    if clamped:
        low, high = CRF_LIMITS[encoder]
        factors.append(
            Adjustment(
                label=f"held to {encoder.value}'s range",
                delta=residual,
                detail=f"{encoder.value} only takes a CRF between {low:g} and {high:g}.",
            )
        )
    else:
        rejected.append(
            f"the {encoder.value} plan's account of its CRF — the factors came to "
            f"{crf - residual:g}, not the {crf:g} it asked for"
        )
        factors.append(
            Adjustment(
                label="unaccounted for",
                delta=residual,
                detail="Gemini's own factors did not add up to its CRF; this is the difference.",
            )
        )
    return factors


def _grain_from(value: object, request: EncodeRequest, rejected: list[str]) -> GrainProfile:
    """The grain profile the model settled on — unless the user set the level by hand.

    A hand-set level is not the model's to revisit, so that case returns the estimate
    whole (which already reports itself as an override at full confidence). Otherwise
    everything here is the model's, with one exception: ``origin_format`` falls back to
    the estimate's, because that comes off the negative-format row rather than out of a
    table, and losing it would quietly reclassify a film source as maybe-digital and
    suppress the grain handling this tool exists for.
    """
    estimate = grain_for(request)
    if estimate.user_override:
        return estimate

    row = _as_dict(value)
    raw_level = row.get("level")
    level = _GRAIN_LEVELS.get(str(raw_level).strip().casefold())
    if level is None:
        rejected.append(f"grain level {raw_level!r}, so the heuristic estimate stands")
        level = estimate.level

    raw_confidence = row.get("confidence")
    confidence = (
        max(0.0, min(float(raw_confidence), 1.0))
        if isinstance(raw_confidence, int | float) and not isinstance(raw_confidence, bool)
        else estimate.confidence
    )

    return GrainProfile(
        level=level,
        confidence=round(confidence, 2),
        reasons=_lines(row.get("reasons"), limit=MAX_GRAIN_REASONS, seen=set()),
        origin_format=_prose(row.get("origin_format"), MAX_LABEL_CHARS) or estimate.origin_format,
    )


_GRAIN_LEVELS: dict[str, GrainLevel] = {level.value: level for level in GrainLevel}


def _starved_grain(plan: EncoderPlan) -> str | None:
    """Report — never rewrite — a plan that codes real grain it cannot afford.

    The model decides the settings, so this does not touch its numbers. But coding the
    negative's own grain at a CRF with no budget for it is the one pairing where sound
    reasoning about each setting on its own adds up to an encode that visibly breaks in
    motion, so the user is told both halves and can judge the model's reasoning for
    themselves.
    """
    if plan.encoder is not Encoder.SVT_AV1:
        return None
    strength = plan.params.get("film-grain")
    if not strength or strength == "0":
        return None
    if plan.params.get("film-grain-denoise") != "0":
        return None
    if plan.crf <= REAL_GRAIN_CRF_CEILING:
        return None
    return (
        f"SVT-AV1 is set to code this film's own grain (film-grain-denoise=0) at CRF "
        f"{plan.crf:g}, above the {REAL_GRAIN_CRF_CEILING:g} that coded grain usually "
        "needs — check the SVT-AV1 rationale for why this film is an exception. If it is "
        f"not, either bring the CRF to {REAL_GRAIN_CRF_CEILING:g} or set "
        "film-grain-denoise=1 and let the grain be re-synthesised instead."
    )


__all__ = [
    "RESPONSE_SCHEMA",
    "SYSTEM_PROMPT",
    "GeminiClient",
    "advice_from",
    "build_context",
]
