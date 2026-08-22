"""The last four steps of producing advice, shared by both front-ends.

The web page and the CLI assemble a :class:`~app.models.EncodeRequest` in completely
different ways — one from a form, one from a filename and a probe — but from there on
they must agree exactly: same decision, same disclosure about where the technical rows
came from, same rendered commands. Keeping that tail in one place is what stops the two
from drifting apart, and :data:`LOOKED_UP_NOTE` and :data:`TABLES_ONLY_NOTE` in
particular have to say the same thing in both.

The decision itself belongs to the model. :class:`Decider` is asked for a complete
answer, not for edits to one: the facts go out, settings come back, and what comes back
*is* the advice. :func:`~app.advice.rules.build_advice` is what you get when there is no
model at all — no key, review declined, or a call that failed — and that case is labelled
on the page rather than blended into the other one.

:class:`Decider` is a protocol rather than an import of
:class:`~app.providers.gemini.GeminiClient`, so this module stays free of provider
imports — the provider already imports :mod:`app.advice.validate`, and importing it
back would be a cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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

# The other half of the same honesty. These settings came out of tables keyed on three
# rows, and the user is entitled to know that nothing weighed this film against them.
TABLES_ONLY_NOTE = (
    "These settings were not decided for this film. They come from tables keyed on "
    "resolution, negative format and release year — nothing in that path read the film, "
    "its technical rows or your source file. A sound starting point, and no more than "
    "that."
)


@dataclass(frozen=True, slots=True)
class Decision:
    """What the model decided, or why it decided nothing.

    Both fields empty means it was never asked. A ``problem`` with no ``advice`` is a
    model that was asked and could not answer — the sentence is shown to the user,
    because the settings they are about to read are not the ones they asked for.
    """

    advice: Advice | None = None
    problem: str | None = None


class Decider(Protocol):
    """Anything that can decide the settings — in practice the Gemini client."""

    async def decide(self, request: EncodeRequest) -> Decision: ...


async def finish_advice(
    request: EncodeRequest,
    *,
    decider: Decider | None = None,
    warnings: Sequence[str] = (),
) -> Advice:
    """The decided settings — or the tables, labelled — with commands and warnings on.

    ``warnings`` are the ones collected while assembling the request — a dead TMDB, an
    unparseable paste. They go first because they explain why some of the facts are
    missing, which the warnings after them then refer to.
    """
    decision = await decider.decide(request) if decider is not None else Decision()

    advice = decision.advice
    if advice is None:
        # Nothing read this film, so the answer says so before it says anything else.
        advice = build_advice(request)
        advice.notes.insert(0, TABLES_ONLY_NOTE)
        if decision.problem:
            advice.warnings.insert(0, decision.problem)

    if request.specs_source is SpecsSource.GEMINI:
        advice.notes.append(LOOKED_UP_NOTE)

    advice.warnings = [*warnings, *advice.warnings]
    attach_commands(advice, request)
    return advice
