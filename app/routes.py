"""HTTP routes. Three steps, no session, no database.

The whole flow is stateless: step 2's form carries the film's ids as hidden fields and
the pasted text in its textareas, so ``/advise`` can rebuild everything it needs from
one POST. Re-asking TMDB or re-running the technical-specs lookup is free because both
are cached, which means a reload or a tweaked preference costs no upstream requests.

Errors are rendered, not raised. Every failure below returns HTTP 200 with an error
partial, because HTMX does not swap non-2xx responses by default — a 502 here would
leave the user looking at an unchanged page and no explanation. The exception is
``/healthz``, which is machine-facing.

Nothing in a route ever blocks the answer: a dead TMDB, a failed specs lookup or an
unparseable paste each degrade to advice built from whatever survived, with a warning
saying what was lost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.datastructures import FormData

from app import __version__
from app.advice import finish_advice, render_handbrake_preset
from app.config import Settings
from app.models import (
    Advice,
    Encoder,
    EncodeRequest,
    GrainLevel,
    Movie,
    SizePreference,
    SourceTool,
    SpecsSource,
    SpeedPreference,
    TechnicalSpecs,
)
from app.providers import ProviderDisabled, ProviderError
from app.providers.gemini import GeminiClient
from app.providers.imdb import parse_technical
from app.providers.tmdb import TmdbClient
from app.sources import MAX_INPUT_CHARS, UnknownSourceFormat, parse_source

# The paste box takes either the formatted specifications or the whole page source,
# and a source paste is well over a megabyte once the JSON island is included.
MAX_TECHNICAL_CHARS = 4_000_000
MAX_PATH_CHARS = 240
MAX_QUERY_CHARS = 200

# A lookup that honestly does not know beats one that invents rows, but the user
# still has to be told what to do next.
NO_ROWS_FOUND = (
    "Gemini did not have the technical rows for this title. Open the technical page "
    "linked above and paste the specifications to get grain-accurate advice."
)

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(slots=True)
class Services:
    """The long-lived collaborators. Built once per app so their caches survive."""

    settings: Settings
    templates: Jinja2Templates
    tmdb: TmdbClient
    gemini: GeminiClient


# --- form handling ----------------------------------------------------------


def _text(form: FormData, name: str, limit: int) -> str:
    value = form.get(name)
    if not isinstance(value, str):
        return ""
    return value[:limit]


def _one_line(value: str, limit: int) -> str:
    """A single line with no control characters — these end up in a shell command."""
    return _CONTROL.sub("", value).strip()[:limit]


def _enum_or_none[E: StrEnum](form: FormData, name: str, kind: type[E]) -> E | None:
    """Form selects use an empty value for "work it out yourself"."""
    raw = form.get(name)
    if not isinstance(raw, str) or raw.strip() in {"", "auto"}:
        return None
    try:
        return kind(raw.strip())
    except ValueError:
        return None


def _enum_or_default[E: StrEnum](form: FormData, name: str, kind: type[E], default: E) -> E:
    return _enum_or_none(form, name, kind) or default


def _int_or_none(value: object) -> int | None:
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _bit_depth(value: object) -> int | None:
    """Only the depths an encoder can actually be told about."""
    depth = _int_or_none(value)
    return depth if depth in {8, 10, 12} else None


def _slug(text: str) -> str:
    return _SLUG_STRIP.sub("-", text.casefold()).strip("-") or "output"


def _checked(form: FormData, name: str) -> bool:
    return isinstance(form.get(name), str)


@dataclass(slots=True)
class AdviceInputs:
    """Everything the step-2 form submits, cleaned up."""

    tmdb_id: int | None = None
    imdb_id: str | None = None
    technical_source: str = ""
    source_text: str = ""
    source_tool: SourceTool | None = None
    grain_override: GrainLevel | None = None
    bit_depth_override: int | None = None
    speed: SpeedPreference = SpeedPreference.BALANCED
    size: SizePreference = SizePreference.BALANCED
    input_path: str = ""
    output_stem: str = ""
    use_gemini: bool = True
    encoder: Encoder = Encoder.SVT_AV1

    @classmethod
    def from_form(cls, form: FormData) -> AdviceInputs:
        imdb_id = _one_line(_text(form, "imdb_id", 40), 40)
        return cls(
            tmdb_id=_int_or_none(form.get("tmdb_id")),
            imdb_id=imdb_id if re.fullmatch(r"tt\d{5,}", imdb_id) else None,
            technical_source=_text(form, "technical_source", MAX_TECHNICAL_CHARS),
            source_text=_text(form, "source_text", MAX_INPUT_CHARS + 1),
            source_tool=_enum_or_none(form, "source_tool", SourceTool),
            grain_override=_enum_or_none(form, "grain_override", GrainLevel),
            bit_depth_override=_bit_depth(form.get("bit_depth_override")),
            speed=_enum_or_default(form, "speed", SpeedPreference, SpeedPreference.BALANCED),
            size=_enum_or_default(form, "size", SizePreference, SizePreference.BALANCED),
            input_path=_one_line(_text(form, "input_path", MAX_PATH_CHARS), MAX_PATH_CHARS),
            output_stem=_one_line(_text(form, "output_stem", MAX_PATH_CHARS), MAX_PATH_CHARS),
            use_gemini=_checked(form, "use_gemini"),
            encoder=_enum_or_default(form, "encoder", Encoder, Encoder.SVT_AV1),
        )

    def hidden_fields(self) -> dict[str, str]:
        """The subset a plain (non-HTMX) form needs to re-post for the preset download.

        The pasted specifications are deliberately excluded: a source paste is up to a
        megabyte, and the rows it produces are already cached against the IMDb id.

        ``use_gemini`` is carried through so the downloaded preset matches the numbers
        on screen — the model's answer is cached against the same context, so this
        costs nothing and skipping it would quietly hand over different settings.
        """
        fields = {
            "source_text": self.source_text,
            "speed": self.speed.value,
            "size": self.size.value,
            "input_path": self.input_path,
            "output_stem": self.output_stem,
        }
        if self.use_gemini:
            fields["use_gemini"] = "on"
        if self.tmdb_id is not None:
            fields["tmdb_id"] = str(self.tmdb_id)
        if self.imdb_id:
            fields["imdb_id"] = self.imdb_id
        if self.source_tool is not None:
            fields["source_tool"] = self.source_tool.value
        if self.grain_override is not None:
            fields["grain_override"] = self.grain_override.value
        if self.bit_depth_override is not None:
            fields["bit_depth_override"] = str(self.bit_depth_override)
        return fields


@dataclass(slots=True)
class Assembled:
    """A request ready for the rules engine, plus what went wrong getting there."""

    request: EncodeRequest
    warnings: list[str] = field(default_factory=list)
    movie: Movie | None = None


# --- assembly ---------------------------------------------------------------


async def _resolve_movie(
    services: Services, inputs: AdviceInputs
) -> tuple[Movie | None, str | None]:
    if inputs.tmdb_id is None:
        return None, None
    try:
        return await services.tmdb.get_movie(inputs.tmdb_id), None
    except ProviderError as exc:
        return None, f"{exc.message} The film's details are missing from this answer."


async def _lookup_specs(
    services: Services, imdb_id: str, *, title: str | None = None, year: int | None = None
) -> tuple[TechnicalSpecs, str | None, str | None]:
    """Ask Gemini for a film's technical rows: the rows, a caveat, an error.

    Exactly one of the caveat and the error is ever set. The caveat is not a failure —
    it says where the rows came from, which the user needs in order to trust them.
    """
    try:
        lookup = await services.gemini.technical_specs(imdb_id, title=title, year=year)
    except ProviderError as exc:
        return TechnicalSpecs(), None, exc.message
    if lookup.specs.is_empty:
        return lookup.specs, NO_ROWS_FOUND, None
    return lookup.specs, lookup.caveat, None


async def _resolve_specs(
    services: Services, inputs: AdviceInputs, imdb_id: str | None
) -> tuple[TechnicalSpecs, SpecsSource | None, str | None]:
    """Pasted specifications win; otherwise ask Gemini for them."""
    if inputs.technical_source.strip():
        specs = parse_technical(inputs.technical_source)
        if specs.is_empty:
            return (
                specs,
                None,
                (
                    "Nothing recognisable was found in the pasted specifications — no "
                    "negative format, no aspect ratio. Grain had to be guessed from the "
                    "release year."
                ),
            )
        return specs, SpecsSource.PASTED, None

    if imdb_id and services.gemini.enabled:
        specs, _, failure = await _lookup_specs(services, imdb_id)
        if failure is not None:
            return (
                TechnicalSpecs(),
                None,
                (f"{failure} Paste the film's technical specifications for grain-accurate advice."),
            )
        return specs, SpecsSource.GEMINI if not specs.is_empty else None, None
    return TechnicalSpecs(), None, None


async def _assemble(services: Services, inputs: AdviceInputs) -> Assembled:
    warnings: list[str] = []

    movie, movie_warning = await _resolve_movie(services, inputs)
    if movie_warning:
        warnings.append(movie_warning)

    imdb_id = inputs.imdb_id or (movie.imdb_id if movie else None)
    specs, specs_source, specs_warning = await _resolve_specs(services, inputs, imdb_id)
    if specs_warning:
        warnings.append(specs_warning)

    request = EncodeRequest(
        movie=movie,
        specs=specs,
        specs_source=specs_source,
        grain_override=inputs.grain_override,
        bit_depth_override=inputs.bit_depth_override,
        speed=inputs.speed,
        size=inputs.size,
    )

    if inputs.source_text.strip():
        try:
            report = parse_source(inputs.source_text, tool=inputs.source_tool)
        except (UnknownSourceFormat, ValueError) as exc:
            warnings.append(f"{exc} Settings below use 1080p defaults instead of your file.")
        else:
            request.source = report.media
            request.source_tool = report.tool
            warnings.extend(report.warnings)

    stem = inputs.output_stem or _default_stem(movie, request)
    request.output_stem = stem
    request.input_path = inputs.input_path or _default_input_path(request)

    return Assembled(request=request, warnings=warnings, movie=movie)


def _default_stem(movie: Movie | None, request: EncodeRequest) -> str:
    if movie is None:
        return "output"
    parts = [_slug(movie.title)]
    if movie.year:
        parts.append(str(movie.year))
    if label := request.source.resolution_label:
        parts.append(label.casefold())
    return "-".join(parts)


def _default_input_path(request: EncodeRequest) -> str:
    container = (request.source.container or "").casefold()
    return "input.mp4" if container in {"mp4", "mov", "m4v", "isom"} else "input.mkv"


async def _advise(services: Services, inputs: AdviceInputs) -> tuple[Advice, Assembled]:
    """Baseline advice, optionally reviewed by Gemini, with commands attached."""
    assembled = await _assemble(services, inputs)
    advice = await finish_advice(
        assembled.request,
        annotator=services.gemini if inputs.use_gemini and services.gemini.enabled else None,
        warnings=assembled.warnings,
    )
    return advice, assembled


# --- router -----------------------------------------------------------------


def create_router(services: Services) -> APIRouter:
    router = APIRouter()
    templates = services.templates
    settings = services.settings

    def render(request: Request, name: str, context: dict[str, Any]) -> HTMLResponse:
        return templates.TemplateResponse(request, name, context)

    def error(
        request: Request, message: str, *, detail: str | None = None, help_text: str | None = None
    ) -> HTMLResponse:
        # HTTP 200 on purpose: see the module docstring.
        return render(
            request,
            "partials/error.html",
            {"message": message, "detail": detail, "help_text": help_text},
        )

    @router.get("/healthz")
    async def healthz() -> dict[str, object]:
        """Liveness probe, also used by the container HEALTHCHECK."""
        return {
            "status": "ok",
            "version": __version__,
            "capabilities": settings.capabilities(),
        }

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return render(
            request,
            "index.html",
            {"capabilities": settings.capabilities(), "step": 1},
        )

    @router.post("/search", response_class=HTMLResponse)
    async def search(
        request: Request,
        query: Annotated[str, Form()] = "",
    ) -> HTMLResponse:
        term = _one_line(query, MAX_QUERY_CHARS)
        if not term:
            return error(request, "Type a title to search for.")

        try:
            hits = await services.tmdb.search(term)
        except ProviderDisabled as exc:
            return error(
                request,
                exc.message,
                help_text=(
                    "You can still get settings without TMDB — the film lookup only "
                    "supplies the IMDb id that finds the negative format."
                ),
            )
        except ProviderError as exc:
            return error(request, exc.message, detail=exc.detail)

        return render(
            request,
            "partials/search_results.html",
            {"query": term, "hits": hits, "step": 1, "oob": True},
        )

    @router.post("/pick", response_class=HTMLResponse)
    async def pick(
        request: Request,
        tmdb_id: Annotated[int, Form()],
    ) -> HTMLResponse:
        try:
            movie = await services.tmdb.get_movie(tmdb_id)
        except ProviderError as exc:
            return error(request, exc.message, detail=exc.detail)

        specs = TechnicalSpecs()
        specs_caveat: str | None = None
        specs_error: str | None = None
        if movie.imdb_id and services.gemini.enabled:
            specs, specs_caveat, specs_error = await _lookup_specs(
                services, movie.imdb_id, title=movie.title, year=movie.year
            )

        return render(
            request,
            "partials/film.html",
            {
                "movie": movie,
                "specs": specs,
                "specs_caveat": specs_caveat,
                "specs_error": specs_error,
                "capabilities": settings.capabilities(),
                "grain_levels": list(GrainLevel),
                "source_tools": list(SourceTool),
                "speeds": list(SpeedPreference),
                "sizes": list(SizePreference),
                "step": 2,
                "oob": True,
            },
        )

    @router.post("/technical", response_class=HTMLResponse)
    async def technical(
        request: Request,
        imdb_id: Annotated[str, Form()] = "",
        tmdb_id: Annotated[str, Form()] = "",
    ) -> HTMLResponse:
        title_id = _one_line(imdb_id, 40)
        if not re.fullmatch(r"tt\d{5,}", title_id):
            return error(
                request,
                "That is not an IMDb title id.",
                help_text="Ids look like tt0083658 and are in the film's IMDb URL.",
            )

        # The title and year make the lookup answer for the right film when the model
        # knows it by name. They are a bonus: the id alone is enough to ask with.
        title: str | None = None
        year: int | None = None
        if (numeric := _one_line(tmdb_id, 20)).isdigit():
            try:
                film = await services.tmdb.get_movie(int(numeric))
            except ProviderError:
                pass
            else:
                title, year = film.title, film.year

        specs, caveat, failure = await _lookup_specs(services, title_id, title=title, year=year)
        if failure is not None:
            return error(
                request,
                failure,
                help_text=(
                    "Open the film's technical page, select the specifications, and paste "
                    "them into the box below instead."
                ),
            )
        return render(
            request,
            "partials/technical.html",
            {
                "specs": specs,
                "imdb_id": title_id,
                "specs_caveat": caveat,
                "specs_error": None,
            },
        )

    @router.post("/advise", response_class=HTMLResponse)
    async def advise(request: Request) -> HTMLResponse:
        form = await request.form()
        inputs = AdviceInputs.from_form(form)
        advice, assembled = await _advise(services, inputs)

        return render(
            request,
            "partials/advice.html",
            {
                "advice": advice,
                "request_data": assembled.request,
                "movie": assembled.movie,
                "hidden_fields": inputs.hidden_fields(),
                "gemini_enabled": services.gemini.enabled,
                "gemini_requested": inputs.use_gemini,
                "step": 3,
                "oob": True,
            },
        )

    @router.post("/preset")
    async def preset(request: Request) -> JSONResponse:
        """The HandBrake ``.json`` preset for one plan, as a download."""
        form = await request.form()
        inputs = AdviceInputs.from_form(form)
        advice, assembled = await _advise(services, inputs)

        plan = advice.plan_for(inputs.encoder) or advice.plans[0]
        body = render_handbrake_preset(advice, assembled.request, plan)
        filename = _UNSAFE_FILENAME.sub(
            "-", f"graindamage-{assembled.request.output_stem}-{plan.encoder.value}.json"
        )
        return JSONResponse(
            body,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return router
