"""The finished advice as terminal text.

This is the CLI's equivalent of ``partials/advice.html`` and it shows the same things in
the same order, because the reasoning is the point: a CRF with no account of how it was
reached is a number you cannot argue with. Warnings come before notes, and the grain
line always says where the technical rows came from.

Plain text, no colour. The output is as likely to end up in a pipe or a log as on a
screen, and the two files written beside the film are the actual deliverable.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from app.models import Advice, EncodeRequest, EncoderPlan, Movie, SourceMedia

INDENT = "  "


def render_report(
    advice: Advice,
    request: EncodeRequest,
    *,
    movie: Movie | None = None,
    specs_caveat: str | None = None,
    written: Iterable[Path] = (),
) -> str:
    """The whole answer, ready to print."""
    return "\n".join(
        _lines(advice, request, movie=movie, specs_caveat=specs_caveat, written=written)
    )


def _lines(
    advice: Advice,
    request: EncodeRequest,
    *,
    movie: Movie | None,
    specs_caveat: str | None,
    written: Iterable[Path],
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
    yield f"Source   {source_summary(request.source)}"

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


def source_summary(media: SourceMedia) -> str:
    """One line describing the file, shared with the header of the written script."""
    parts: list[str] = []
    video = media.video
    if video is not None:
        if video.width and video.height:
            parts.append(f"{video.width}×{video.height}")
        if video.codec:
            parts.append(video.codec)
        if video.bit_depth:
            parts.append(f"{video.bit_depth}-bit")
        if video.is_hdr:
            parts.append("HDR")
        if video.dolby_vision:
            parts.append("Dolby Vision")
        if video.is_interlaced:
            parts.append("interlaced")
        if video.frame_rate:
            parts.append(f"{video.frame_rate:.3f} fps")
        if video.bitrate_bps:
            parts.append(f"{video.bitrate_bps / 1_000_000:.1f} Mb/s")
    else:
        parts.append("no video track")

    if media.duration_seconds:
        parts.append(_duration(media.duration_seconds))
    if media.size_bytes:
        parts.append(f"{media.size_bytes / 1_000_000_000:.1f} GB")
    if media.audio:
        parts.append(f"{len(media.audio)} audio")
    if media.subtitles:
        parts.append(f"{len(media.subtitles)} subtitle")
    return " · ".join(parts)


def _duration(seconds: float) -> str:
    minutes, _ = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


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
        yield f"{INDENT}params  {plan.params_string}"
    if plan.ffmpeg_command:
        yield f"{INDENT}{plan.ffmpeg_command}"


def _bullets(heading: str, items: list[str], *, mark: str) -> Iterator[str]:
    if not items:
        return
    yield ""
    yield heading
    for item in items:
        yield f"{INDENT}{mark} {item}"
