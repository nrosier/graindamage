"""The finished advice as terminal text.

This is the CLI's equivalent of ``partials/advice.html`` and it shows the same things in
the same order, because the reasoning is the point: a CRF with no account of how it was
reached is a number you cannot argue with. Warnings come before notes, and the grain
line always says where the technical rows came from.

Plain text, no colour. The output is as likely to end up in a pipe or a log as on a
screen, and it stands on its own: both commands are here, so a run that wrote no files
still told you everything.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from app.models import Advice, AdviceSource, EncodeRequest, EncoderPlan, Movie

INDENT = "  "
# Wide enough for "HandBrake", which is the longest label a plan line takes.
LABEL = 9


def render_report(
    advice: Advice,
    request: EncodeRequest,
    *,
    movie: Movie | None = None,
    specs_caveat: str | None = None,
    written: Iterable[Path] = (),
    unwritten: str | None = None,
) -> str:
    """The whole answer, ready to print.

    ``unwritten`` is why there are no paths — asked for, or refused by the directory. It
    ends the report so that a redirected one says for itself that nothing was written.
    """
    return "\n".join(
        _lines(
            advice,
            request,
            movie=movie,
            specs_caveat=specs_caveat,
            written=written,
            unwritten=unwritten,
        )
    )


def _lines(
    advice: Advice,
    request: EncodeRequest,
    *,
    movie: Movie | None,
    specs_caveat: str | None,
    written: Iterable[Path],
    unwritten: str | None,
) -> Iterator[str]:
    yield ""
    yield from _film(movie)
    yield ""
    yield from _facts(advice, request, specs_caveat)

    for plan in advice.plans:
        yield ""
        yield from _plan(plan)

    yield from _bullets("Warnings", advice.warnings, mark="!")
    yield from _bullets("Notes", advice.notes, mark="·")
    yield from _bullets("Rejected by validation", advice.rejected_flags, mark="×")

    paths = list(written)
    if paths:
        yield ""
        yield "Written"
        for path in paths:
            yield f"{INDENT}{path}"
    if unwritten:
        yield ""
        yield "Not written"
        for line in unwritten.splitlines():
            yield f"{INDENT}{line}"
    yield ""


def _film(movie: Movie | None) -> Iterator[str]:
    if movie is None:
        yield "No film identified — settings come from the source file alone."
        return

    parts = [movie.display_title]
    if movie.year:
        parts.append(str(movie.year))
    if movie.directors:
        parts.append(", ".join(movie.directors[:2]))
    yield " · ".join(parts)
    if url := movie.imdb_technical_url:
        yield f"{INDENT}{url}"


def _facts(advice: Advice, request: EncodeRequest, specs_caveat: str | None) -> Iterator[str]:
    yield f"Source   {request.source.describe()}"
    # Who decided, on its own line: everything below reads the same either way, and the
    # difference between a decision made for this film and a table lookup is the single
    # most important thing to know before acting on any of it. The notes carry the rest.
    yield f"Settings {_decided_by(advice.source)}"

    grain = advice.grain
    origin = f" ({grain.origin_format})" if grain.origin_format else ""
    overridden = " — set by hand" if grain.user_override else ""
    yield f"Grain    {grain.level.value}{origin}, confidence {grain.confidence:.0%}{overridden}"
    for reason in grain.reasons[:4]:
        yield f"{INDENT}       · {reason}"

    if specs_caveat:
        yield f"Specs    {specs_caveat}"
    if advice.summary:
        yield ""
        yield advice.summary


def _decided_by(source: AdviceSource) -> str:
    if source is AdviceSource.GEMINI:
        return "decided by Gemini for this film"
    return "from tables — nothing read this film (see Notes)"


def _plan(plan: EncoderPlan) -> Iterator[str]:
    headline = [f"CRF {plan.crf_label}", f"preset {plan.preset}"]
    if plan.tune:
        headline.append(f"tune {plan.tune}")
    headline.append(plan.pixel_format)
    if note := plan.estimated_size_note:
        headline.append(note)

    yield f"{plan.encoder.label}   {' · '.join(headline)}"
    for adjustment in plan.adjustments:
        yield f"{INDENT}{adjustment.delta:+g}  {adjustment.label}"
    if plan.params:
        yield _field("params", plan.params_string)
    if plan.ffmpeg_command:
        yield _field("ffmpeg", plan.ffmpeg_command)
    if plan.handbrake_command:
        yield _field("HandBrake", plan.handbrake_command)


def _field(label: str, value: str) -> str:
    """One labelled line under a plan, all three labels in the same column."""
    return f"{INDENT}{label:<{LABEL}}  {value}"


def _bullets(heading: str, items: list[str], *, mark: str) -> Iterator[str]:
    if not items:
        return
    yield ""
    yield heading
    for item in items:
        yield f"{INDENT}{mark} {item}"
