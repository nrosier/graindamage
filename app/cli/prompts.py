"""Terminal interaction: an arrow-key list, a typed line, a pasted block.

There is no dependency here for the same reason there is none anywhere else in this
project — a list you can move a cursor through is a hundred lines of :mod:`termios`, and
those hundred lines never break on a version bump.

Three ideas keep it honest:

* **The loop is pure.** :func:`run_menu` takes decoded key names and a draw callback and
  returns an index, so every key can be tested without a terminal. Only
  :meth:`Terminal.select` touches ``termios``.
* **It redraws in place** by moving the cursor up rather than clearing the screen, so
  what you did before the prompt is still above it afterwards. Lines are truncated to
  the terminal's width, because a wrapped line would make that arithmetic wrong.
* **Prompts go to stderr.** The report is stdout's job, so ``graindamage film.mkv >
  notes.txt`` still shows you the menu.

Without a TTY — ``docker run`` with no ``-it``, a pipe, a platform with no ``termios`` —
everything falls back to a numbered prompt that reads one line.
"""

from __future__ import annotations

import os
import select
import shutil
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TextIO

try:  # POSIX only; the fallback is the numbered prompt.
    import termios
    import tty

    RAW_CAPABLE = True
except ImportError:  # pragma: no cover - Windows
    RAW_CAPABLE = False

UP = "up"
DOWN = "down"
ENTER = "enter"
ABORT = "abort"
DIGIT = "#"  # "#3" is the third row

CURSOR = "▸"

_ESCAPE_TIMEOUT = 0.05
_MAX_BAD_ANSWERS = 3


@dataclass(frozen=True, slots=True)
class Choice[T]:
    """One row of a menu, and whatever the caller wants back for it."""

    value: T
    label: str
    detail: str | None = None


def run_menu(count: int, keys: Iterable[str], *, draw: Callable[[int], None]) -> int | None:
    """Move a cursor over ``count`` rows. Returns the chosen index, or ``None`` to abort.

    The wrap-around is deliberate: with *Abort* as the last row, one press of ↑ from the
    top is the fastest way out.
    """
    if count <= 0:
        return None

    index = 0
    draw(index)
    for key in keys:
        if key == UP:
            index = (index - 1) % count
        elif key == DOWN:
            index = (index + 1) % count
        elif key.startswith(DIGIT):
            wanted = int(key[1:]) - 1
            if not 0 <= wanted < count:
                continue
            index = wanted
        elif key == ENTER:
            return index
        elif key == ABORT:
            return None
        else:
            continue
        draw(index)
    return None


def decode_keys(read: Callable[[int], bytes], *, waiting: Callable[[], bool]) -> Iterator[str]:
    """Turn raw bytes into key names.

    ``waiting`` answers "is there more input right now", which is the only way to tell a
    bare Escape from the start of an arrow key — they are the same first byte.
    """
    while True:
        first = read(1)
        if not first:
            return
        if first in {b"\r", b"\n"}:
            yield ENTER
        elif first in {b"q", b"Q", b"\x03", b"\x04"}:
            yield ABORT
        elif first in {b"k", b"K"}:
            yield UP
        elif first in {b"j", b"J"}:
            yield DOWN
        elif first.isdigit() and first != b"0":
            yield f"{DIGIT}{first.decode()}"
        elif first == b"\x1b":
            if not waiting():
                yield ABORT
                continue
            rest = read(2)
            if rest in {b"[A", b"OA"}:
                yield UP
            elif rest in {b"[B", b"OB"}:
                yield DOWN
            else:
                yield "other"
        else:
            yield "other"


class Terminal:
    """Where the prompts read and write. Swappable, so the tests can watch."""

    def __init__(self, *, stdin: TextIO | None = None, stderr: TextIO | None = None) -> None:
        self._stdin = stdin if stdin is not None else sys.stdin
        self._err = stderr if stderr is not None else sys.stderr

    # --- plumbing ----------------------------------------------------------

    @property
    def interactive(self) -> bool:
        """Can a cursor be moved here at all?"""
        if not RAW_CAPABLE:
            return False
        try:
            return bool(self._stdin.isatty() and self._err.isatty())
        except ValueError:  # a closed stream
            return False

    @property
    def width(self) -> int:
        return max(40, shutil.get_terminal_size(fallback=(80, 24)).columns)

    def write(self, text: str = "") -> None:
        self._err.write(f"{text}\n")
        self._err.flush()

    # --- prompts -----------------------------------------------------------

    def select[T](self, prompt: str, choices: Sequence[Choice[T]]) -> Choice[T] | None:
        """Offer ``choices``. Returns the chosen one, or ``None`` if the user aborted."""
        if not choices:
            return None
        self.write()
        self.write(prompt)
        index = (
            self._select_interactively(choices)
            if self.interactive
            else self._select_by_number(choices)
        )
        return choices[index] if index is not None else None

    def ask(self, prompt: str) -> str:
        """One line of typed input. Empty on EOF, which callers treat as "never mind"."""
        self._err.write(f"{prompt} ")
        self._err.flush()
        line = self._stdin.readline()
        return line.strip()

    def paste(self, prompt: str) -> str:
        """Everything up to EOF. Only ever called when a person is there to type it."""
        self.write(prompt)
        try:
            return self._stdin.read()
        except (EOFError, KeyboardInterrupt):
            return ""

    # --- the two ways to choose -------------------------------------------

    def _select_interactively[T](self, choices: Sequence[Choice[T]]) -> int | None:
        fd = self._stdin.fileno()
        saved = termios.tcgetattr(fd)
        draw = self._painter(choices)
        try:
            tty.setcbreak(fd)
            keys = decode_keys(lambda n: os.read(fd, n), waiting=lambda: _input_waiting(fd))
            try:
                return run_menu(len(choices), keys, draw=draw)
            except KeyboardInterrupt:
                return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            self.write()

    def _painter[T](self, choices: Sequence[Choice[T]]) -> Callable[[int], None]:
        """A draw function that overwrites its own previous output."""
        printed = 0

        def draw(selected: int) -> None:
            nonlocal printed
            if printed:
                self._err.write(f"\x1b[{printed}A")
            lines = [self._row(choice, index == selected) for index, choice in enumerate(choices)]
            lines.append("  ↑↓ move · 1-9 jump · enter choose · q abort")
            for line in lines:
                self._err.write(f"\x1b[2K{line[: self.width - 1]}\n")
            self._err.flush()
            printed = len(lines)

        return draw

    def _row[T](self, choice: Choice[T], selected: bool) -> str:
        mark = f" {CURSOR} " if selected else "   "
        detail = f"  {choice.detail}" if choice.detail else ""
        return f"{mark}{choice.label}{detail}"

    def _select_by_number[T](self, choices: Sequence[Choice[T]]) -> int | None:
        for index, choice in enumerate(choices, start=1):
            detail = f"  {choice.detail}" if choice.detail else ""
            self.write(f"  {index}) {choice.label}{detail}")

        for _ in range(_MAX_BAD_ANSWERS):
            answer = self.ask(f"Choice [1-{len(choices)}, enter for 1]:")
            if not answer:
                return 0
            if answer.isdigit() and 1 <= int(answer) <= len(choices):
                return int(answer) - 1
            self.write(f"  {answer!r} is not one of them.")
        return None


def _input_waiting(fd: int) -> bool:
    ready, _, _ = select.select([fd], [], [], _ESCAPE_TIMEOUT)
    return bool(ready)
