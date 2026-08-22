"""Turning a film and a file into settings.

Three layers, in dependency order:

* :mod:`app.advice.grain` — how grainy is this film, and why.
* :mod:`app.advice.rules` — the table-derived plans, for when there is no model to
  decide. No keys, no network, always works.
* :mod:`app.advice.encoders` — those plans as FFmpeg / HandBrake commands and presets.
* :mod:`app.advice.pipeline` — the tail both the web page and the CLI run, in one place.

:mod:`app.advice.validate` guards the boundary where a language model's suggestions
enter, and is used by :mod:`app.providers.gemini` rather than by anything here.
"""

from __future__ import annotations

from app.advice.encoders import (
    attach_commands,
    output_name,
    render_ffmpeg,
    render_handbrake,
    render_handbrake_preset,
)
from app.advice.grain import infer_grain
from app.advice.pipeline import (
    LOOKED_UP_NOTE,
    TABLES_ONLY_NOTE,
    Decider,
    Decision,
    finish_advice,
)
from app.advice.rules import build_advice, grain_for, parse_aspect_ratio

__all__ = [
    "LOOKED_UP_NOTE",
    "TABLES_ONLY_NOTE",
    "Decider",
    "Decision",
    "attach_commands",
    "build_advice",
    "finish_advice",
    "grain_for",
    "infer_grain",
    "output_name",
    "parse_aspect_ratio",
    "render_ffmpeg",
    "render_handbrake",
    "render_handbrake_preset",
]
