"""Video files on a mount: what is there, and what a request is allowed to name.

The CLI takes a path from the command line, where the user already has whatever access
the shell gives them. The web app cannot: a ``path`` in a form is a string a stranger
typed, so every one of them goes through :func:`inside` first, and the answer is a path
under :data:`LIBRARY_ROOT` or an exception.

``/mnt`` is the whole of it. There is no setting, because a configurable list of
directories to serve over HTTP is a decision better made once, here, than per
deployment — and a container mounts its library wherever the compose file says.

Resolving before comparing is the point of :func:`inside`: a symlink inside the root
pointing at ``/etc`` resolves to ``/etc``, which is not under the root, and is refused.
The cost is that a library which *is* a tree of symlinks pointing elsewhere cannot be
browsed. That is the right way round — the alternative serves the whole filesystem to
anyone who can create a link.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The one directory the web app will open. Not configurable on purpose; see above.
LIBRARY_ROOT = Path("/mnt")

VIDEO_SUFFIXES = frozenset(
    {
        ".mkv",
        ".mp4",
        ".m4v",
        ".mov",
        ".avi",
        ".ts",
        ".m2ts",
        ".mts",
        ".webm",
        ".mpg",
        ".mpeg",
        ".vob",
        ".wmv",
        ".flv",
        ".ogv",
    }
)

# Where a film is likely to be mounted in a container that has no argument to go on.
# The CLI looks through all of these; the web app only ever opens LIBRARY_ROOT.
LIKELY_ROOTS = ("/media", "/movies", "/films", "/video", "/videos", "/data", "/mnt")

MAX_FOUND = 20
_MAX_DEPTH = 3

# One page of a directory. A library with nine thousand films in one flat directory is
# navigated by typing a path, not by scrolling, and the browser says how many it dropped.
MAX_ENTRIES = 400


class OutsideLibrary(ValueError):
    """A path that is not under :data:`LIBRARY_ROOT` at all."""


class NotUsable(ValueError):
    """A path under the root that is not the kind of thing wanted.

    Told apart from :class:`OutsideLibrary` because the two need different words: one is
    a refusal to look outside the library, the other is "that is a directory, not a film".
    """


@dataclass(frozen=True, slots=True)
class Entry:
    """One row of a listing: a subdirectory to open, or a film to pick."""

    path: Path
    name: str
    is_dir: bool
    size: str = ""


@dataclass(frozen=True, slots=True)
class Listing:
    """One directory, ready to render: where it is, how to leave, and what is in it."""

    path: Path
    parent: Path | None
    crumbs: tuple[tuple[str, Path], ...]
    entries: tuple[Entry, ...]
    dropped: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.entries


# --- what a request may name -------------------------------------------------


def inside(root: Path, raw: str) -> Path:
    """``raw`` as a real path under ``root``, or :class:`OutsideLibrary`.

    An empty string is the root itself, which is what an unparameterised browse means.
    """
    resolved = Path(raw.strip() or str(root)).expanduser().resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise OutsideLibrary(f"{resolved} is outside {root}.")
    return resolved


def directory_in(root: Path, raw: str) -> Path:
    path = inside(root, raw)
    if not path.is_dir():
        raise NotUsable(f"{path} is not a directory.")
    return path


def video_in(root: Path, raw: str) -> Path:
    """The one path ``ffprobe`` may be pointed at: a video file under ``root``."""
    path = inside(root, raw)
    if not path.is_file():
        raise NotUsable(f"{path} is not a file.")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise NotUsable(f"{path.name} is not a video file — {', '.join(sorted(VIDEO_SUFFIXES))}.")
    return path


# --- what is there -----------------------------------------------------------


def list_directory(root: Path, path: Path) -> Listing:
    """``path``'s subdirectories and video files, directories first, capped.

    Anything else — text files, artwork, the sample directory's contents — is left out
    rather than shown greyed: this list exists to be picked from.
    """
    directories: list[Entry] = []
    files: list[Entry] = []
    dropped = 0

    for entry in sorted(path.iterdir(), key=lambda item: item.name.casefold()):
        if entry.name.startswith("."):
            continue
        if len(directories) + len(files) >= MAX_ENTRIES:
            dropped += 1
            continue
        if entry.is_dir():
            directories.append(Entry(path=entry, name=entry.name, is_dir=True))
        elif entry.suffix.lower() in VIDEO_SUFFIXES:
            files.append(Entry(path=entry, name=entry.name, is_dir=False, size=human_size(entry)))

    return Listing(
        path=path,
        parent=path.parent if path != root.resolve() else None,
        crumbs=_crumbs(root, path),
        entries=(*directories, *files),
        dropped=dropped,
    )


def _crumbs(root: Path, path: Path) -> tuple[tuple[str, Path], ...]:
    """The trail from the root to ``path``, each step a place to go back to."""
    base = root.resolve()
    trail: list[tuple[str, Path]] = [(str(root), base)]
    walked = base
    for part in path.relative_to(base).parts:
        walked = walked / part
        trail.append((part, walked))
    return tuple(trail)


def find_videos(root: Path, *, depth: int = _MAX_DEPTH, limit: int = MAX_FOUND) -> list[Path]:
    """Video files at or under ``root``, breadth-first and capped.

    Breadth-first because a mount point's own files are the likely answer and its
    twentieth subdirectory is not, and capped because this exists to fill a menu — a
    library of nine thousand films is browsed by typing a path, not by scrolling.
    """
    found: list[Path] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue and len(found) < limit:
        directory, level = queue.pop(0)
        try:
            entries = sorted(directory.iterdir())
        except OSError:  # unreadable, or gone since the listing above
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if level < depth:
                    queue.append((entry, level + 1))
            elif entry.suffix.lower() in VIDEO_SUFFIXES:
                found.append(entry)
                if len(found) >= limit:
                    break
    return found


def human_size(path: Path) -> str:
    """How big it is. A film is gigabytes; a sample or a trailer is not, and ``0.0 GB``
    beside a real file reads like a fault."""
    try:
        size = path.stat().st_size
    except OSError:  # pragma: no cover - it was there a moment ago
        return ""
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.1f} GB"
    return f"{size / 1_000_000:.0f} MB"
