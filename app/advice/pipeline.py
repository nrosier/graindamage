"""The last four steps of producing advice, shared by both front-ends.

The web page and the CLI assemble a :class:`~app.models.EncodeRequest` in completely
different ways — one from a form, one from a filename and a probe — but from there on
they must agree exactly: same rules, same optional review, same disclosure about where
the technical rows came from, same rendered commands. Keeping that tail in one place is
what stops the two from drifting apart, and :data:`LOOKED_UP_NOTE` in particular has to
say the same thing in both.

:class:`Annotator` is a protocol rather than an import of
:class:`~app.providers.gemini.GeminiClient`, so this module stays free of provider
imports — the provider already imports :mod:`app.advice.validate`, and importing it
back would be a cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.advice.encoders import attach_commands
from app.advice.rules import build_advice
from app.models import Advice, EncodeRequest, SpecsSource

# Looked-up rows are a model's recollection of IMDb, and grain is the one decision they
# drive. Saying so on the answer itself is the difference between advice you can check
# and advice you have to take on faith.
LOOKED_UP_NOTE = (
    "The technical rows behind the grain estimate were looked up by Gemini, not "
    "read from IMDb. Check them on the technical page if the grain matters."
)


class Annotator(Protocol):
    """Anything that can review a plan — in practice the Gemini client."""

    async def annotate(self, request: EncodeRequest, advice: Advice) -> Advice: ...


async def finish_advice(
    request: EncodeRequest,
    *,
    annotator: Annotator | None = None,
    warnings: Sequence[str] = (),
) -> Advice:
    """Deterministic plans, optionally reviewed, with commands and warnings attached.

    ``warnings`` are the ones collected while assembling the request — a dead TMDB, an
    unparseable paste. They go first because they explain why the rules had to assume
    things, which the rules' own warnings then refer to.
    """
    advice = build_advice(request)

    if annotator is not None:
        advice = await annotator.annotate(request, advice)

    if request.specs_source is SpecsSource.GEMINI:
        advice.notes.append(LOOKED_UP_NOTE)

    advice.warnings = [*warnings, *advice.warnings]
    attach_commands(advice, request)
    return advice
