"""The two files the CLI leaves behind.

``<stem>.graindamage.json`` is a HandBrake preset document holding **both** encoders, so
one *Presets → Import from file* offers AV1 and x265 and you pick in the queue.

``<stem>.graindamage.sh`` is the FFmpeg half: the first plan live, the second commented
out underneath it. It ``cd``s to the film's directory and then names the film by its
basename rather than its full path — a script written inside a container that embedded
``/media/film.mkv`` would not run on the host that mounted it.

Nothing is overwritten without being asked twice: :func:`write_outputs` raises
:class:`OutputExists` rather than replacing a file, because the thing next to a film
called ``*.graindamage.sh`` may well be one you edited.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import __version__
from app.advice import render_handbrake_preset
from app.cli.report import source_summary
from app.models import Advice, EncodeRequest, Movie

# A comment is one line by definition, and a filename or a model's summary can
# contain a newline. Collapsing whitespace is what keeps the header a header.
_WHITESPACE = re.compile(r"\s+")

PRESET_SUFFIX = ".graindamage.json"
SCRIPT_SUFFIX = ".graindamage.sh"
SCRIPT_MODE = 0o755


class OutputExists(RuntimeError):
    """A file we were about to write is already there."""


@dataclass(frozen=True, slots=True)
class Written:
    preset: Path
    script: Path

    @property
    def paths(self) -> tuple[Path, Path]:
        return self.preset, self.script


def existing_outputs(directory: Path, stem: str) -> list[Path]:
    """Which of the two files are already there."""
    candidates = (directory / f"{stem}{PRESET_SUFFIX}", directory / f"{stem}{SCRIPT_SUFFIX}")
    return [path for path in candidates if path.exists()]


def refuse_existing(directory: Path, stem: str) -> None:
    """Raise :class:`OutputExists` if writing here would replace something.

    The CLI calls this before it probes or asks anything, so a re-run says so at once
    rather than after a Gemini round trip.
    """
    present = existing_outputs(directory, stem)
    if present:
        names = ", ".join(path.name for path in present)
        raise OutputExists(f"{names} already there. Pass --force to replace it.")


def preset_document(advice: Advice, request: EncodeRequest) -> dict[str, Any]:
    """One HandBrake document, one preset per plan.

    Each plan is rendered by the same :func:`~app.advice.render_handbrake_preset` the web
    app's download uses; only the envelopes are merged, so the two front-ends cannot
    disagree about what a preset contains.
    """
    documents = [render_handbrake_preset(advice, request, plan) for plan in advice.plans]
    presets: list[Any] = [preset for document in documents for preset in document["PresetList"]]
    return {**documents[0], "PresetList": presets}


def preset_text(advice: Advice, request: EncodeRequest) -> str:
    """:func:`preset_document` as the exact bytes the file holds.

    ``--print preset`` and the written file both come through here, so the one that goes
    down a pipe cannot drift from the one on disk.
    """
    return json.dumps(preset_document(advice, request), indent=2) + "\n"


def script_text(
    advice: Advice,
    request: EncodeRequest,
    *,
    movie: Movie | None = None,
    specs_caveat: str | None = None,
    media_dir: str | None = None,
) -> str:
    """A runnable bash script for the first plan, with the rest offered as comments.

    ``media_dir`` is the film's directory, and is only written out when the script will
    not be sitting in it; otherwise the script finds it from its own location.
    """
    title = movie.display_title if movie else request.input_path
    grain = advice.grain
    origin = f" ({grain.origin_format})" if grain.origin_format else ""

    lines = [
        "#!/usr/bin/env bash",
        _comment(title, label=f"graindamage {__version__} — FFmpeg commands for "),
        "#",
        _comment(source_summary(request.source), label="Source: "),
        _comment(
            f"{grain.level.value}{origin}, confidence {grain.confidence:.0%}", label="Grain:  "
        ),
    ]
    lines += [_comment(reason, label="Grain:  · ") for reason in grain.reasons[:3]]
    if specs_caveat:
        lines.append(_comment(specs_caveat, label="Specs:  "))
    if advice.summary:
        lines.append(_comment(advice.summary))
    lines += [
        "#",
        "# The first command runs; any alternative below it is commented out. Swap the",
        "# leading '#' to choose the other encoder. ffmpeg refuses to overwrite an",
        "# existing output unless you add -y, so re-running this is safe.",
        "",
        "set -euo pipefail",
        "",
    ]

    where = shlex.quote(media_dir) if media_dir else '"$(dirname -- "${BASH_SOURCE[0]:-$0}")"'
    lines += [f"cd -- {where}", ""]

    for position, plan in enumerate(advice.plans):
        headline = [f"CRF {plan.crf_label}", f"preset {plan.preset}"]
        if plan.tune:
            headline.append(f"tune {plan.tune}")
        if note := plan.estimated_size_note:
            headline.append(note)
        lines.append(_comment(f"{plan.encoder.label} — {', '.join(headline)}"))
        prefix = "" if position == 0 else "# "
        lines.append(f"{prefix}{plan.ffmpeg_command}")
        lines.append("")

    return "\n".join(lines)


def _comment(text: str, *, label: str = "") -> str:
    """One comment line, whatever whitespace was in ``text``.

    The label is written as given — it is ours, and its padding is what lines the header
    up. Everything after it is collapsed, because a film's title, a model's summary and a
    filename can all contain a newline, and a comment is one line by definition.
    """
    return f"# {label}{_WHITESPACE.sub(' ', text).strip()}"


def write_outputs(
    advice: Advice,
    request: EncodeRequest,
    *,
    stem: str,
    directory: Path,
    media_dir: Path,
    movie: Movie | None = None,
    specs_caveat: str | None = None,
    force: bool = False,
) -> Written:
    """Write both files into ``directory``, refusing to clobber unless ``force``."""
    preset_path = directory / f"{stem}{PRESET_SUFFIX}"
    script_path = directory / f"{stem}{SCRIPT_SUFFIX}"

    if not force:
        refuse_existing(directory, stem)

    directory.mkdir(parents=True, exist_ok=True)
    preset_path.write_text(preset_text(advice, request), encoding="utf-8")
    script_path.write_text(
        script_text(
            advice,
            request,
            movie=movie,
            specs_caveat=specs_caveat,
            media_dir=None if directory.resolve() == media_dir.resolve() else str(media_dir),
        ),
        encoding="utf-8",
    )
    script_path.chmod(SCRIPT_MODE)
    return Written(preset=preset_path, script=script_path)
