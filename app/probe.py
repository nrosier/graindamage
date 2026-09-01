"""Running ``ffprobe`` on a file and parsing what it says.

Both front-ends read files this way — the CLI on the path it was given, the web app on a
file picked out of the library — and this is the one place in the project that starts a
process. It stays out of :mod:`app.sources` so that importing the parsers can never imply
the ability to execute anything.

``create_subprocess_exec`` takes an argument list and no shell, so a filename full of
spaces, quotes or semicolons is just a filename. The subprocess itself sits behind an
injected ``runner`` so the tests do not need ffmpeg installed to exercise every failure.

:func:`probe_text` and :func:`probe` are the same call, split where the web app needs to
get in: it caches and re-serves ffprobe's own JSON, because that text is what its step-2
form carries, and parses it separately.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from app.models import SourceReport, SourceTool
from app.sources import UnknownSourceFormat, parse_source

# -v error keeps the banner and the progress noise off stderr, so anything that does
# arrive there is a real complaint worth showing the user.
FFPROBE_ARGS = ("-v", "error", "-print_format", "json", "-show_format", "-show_streams")

DEFAULT_TIMEOUT_SECONDS = 60.0

# (exit code, stdout, stderr)
Runner = Callable[[Sequence[str], float], Awaitable[tuple[int, str, str]]]


class ProbeFailed(RuntimeError):
    """``ffprobe`` is missing, refused the file, or said something unreadable."""


async def probe_text(
    path: Path,
    *,
    executable: str = "ffprobe",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> str:
    """What ``ffprobe`` says about ``path``, verbatim, or :class:`ProbeFailed`."""
    args = [executable, *FFPROBE_ARGS, _as_argument(path)]
    code, out, err = await (runner or run_ffprobe)(args, timeout)

    if code != 0:
        raise ProbeFailed(f"{executable} exited {code} on {path.name}.{_detail(err)}")
    if not out.strip():
        raise ProbeFailed(f"{executable} described {path.name} with nothing at all.{_detail(err)}")
    return out


async def probe(
    path: Path,
    *,
    executable: str = "ffprobe",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> SourceReport:
    """Describe ``path`` by asking ``ffprobe`` about it."""
    out = await probe_text(path, executable=executable, timeout=timeout, runner=runner)
    return read_probe_text(out, executable=executable)


def read_probe_text(out: str, *, executable: str = "ffprobe") -> SourceReport:
    """Parse ffprobe's JSON, blaming ffprobe rather than the parsers when it will not."""
    try:
        return parse_source(out, tool=SourceTool.FFPROBE)
    except (UnknownSourceFormat, ValueError) as exc:
        raise ProbeFailed(f"{executable} output could not be read: {exc}") from exc


async def run_ffprobe(args: Sequence[str], timeout: float) -> tuple[int, str, str]:
    """Run ``args`` to completion, or kill it and say so."""
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ProbeFailed(
            f"{args[0]} was not found. Install ffmpeg (which ships ffprobe), or point "
            f"--ffprobe at the binary."
        ) from exc
    except OSError as exc:
        raise ProbeFailed(f"{args[0]} could not be started: {exc}") from exc

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise ProbeFailed(f"{args[0]} did not finish within {timeout:g}s.") from None

    return process.returncode or 0, _decode(stdout), _decode(stderr)


def _as_argument(path: Path) -> str:
    """``ffprobe`` reads a leading dash as a flag, so never hand it a bare one."""
    text = str(path)
    return f"./{text}" if text.startswith("-") else text


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _detail(stderr: str) -> str:
    """The last few lines of stderr, which is where ffprobe puts the reason."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return f" {' / '.join(lines[-3:])}" if lines else ""
