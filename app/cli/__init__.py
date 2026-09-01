"""The command-line front-end.

A second way in to the same advice, and one command wide: ``graindamage film.mkv`` reads
the title out of the name, ``ffprobe``s the file, walks you through who the film is and
where its technical rows come from, and writes a HandBrake preset and an FFmpeg script
beside it.

:mod:`app.cli.app` holds the flow — argument surface, the flag path, and the writing —
and :mod:`app.cli.wizard` holds the menus a person gets instead of the flags. The two
would import each other, so what they share (the clients, the prose, the specs a paste
or a file turned into) lives in :mod:`app.cli.session`. The rest is one concern each:
:mod:`~app.cli.prompts` moves a cursor over a list, :mod:`~app.cli.report` and
:mod:`~app.cli.outputs` write the answer down.

Everything here needs a terminal. What does not — reading a title out of a filename,
running ``ffprobe``, finding video files on a mount — sits in :mod:`app.filename`,
:mod:`app.probe` and :mod:`app.library`, because the web front-end does those too.
"""

from __future__ import annotations

from app.cli.app import main

__all__ = ["main"]
