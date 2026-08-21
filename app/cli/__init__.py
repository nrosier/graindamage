"""The command-line front-end.

A second way in to the same advice: give it a file and it reads the title out of the
name, ``ffprobe``s the file, asks who the film is, and writes a HandBrake preset and an
FFmpeg script beside it. :mod:`app.cli.app` holds the flow; the rest is one concern each
— :mod:`~app.cli.filename` reads names, :mod:`~app.cli.probe` runs ffprobe,
:mod:`~app.cli.prompts` talks to the terminal, :mod:`~app.cli.report` and
:mod:`~app.cli.outputs` write the answer down.

Nothing here is imported by the web app, and only :mod:`app.cli.probe` starts a process.
"""

from __future__ import annotations

from app.cli.app import main

__all__ = ["main"]
