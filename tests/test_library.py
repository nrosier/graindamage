"""The library mount: what is on it, and what a form is allowed to name.

This module is the security surface of the web file picker, so most of it is refusals.
The one that matters is the symlink: a path is resolved *before* it is compared against
the root, so a link inside the library pointing at ``/etc`` is refused rather than
followed. Everything else — a directory where a film was wanted, a ``.txt`` where a
video was wanted — is the same idea one step further in.

The listing tests use sparse files (``truncate``), because a test that really wrote
eight gigabytes to check a unit label would deserve to be slow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.library import (
    MAX_ENTRIES,
    VIDEO_SUFFIXES,
    NotUsable,
    OutsideLibrary,
    directory_in,
    find_videos,
    human_size,
    inside,
    list_directory,
    video_in,
)


def library(tmp_path: Path) -> Path:
    """A root of its own, so a test can never be handed ``tmp_path``'s neighbours."""
    root = tmp_path / "library"
    root.mkdir()
    return root


def sized(path: Path, size: int) -> Path:
    """A file that claims ``size`` bytes without occupying them."""
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


# --- what a request may name -------------------------------------------------


def test_the_root_itself_is_what_an_empty_path_means(tmp_path: Path) -> None:
    root = library(tmp_path)

    assert inside(root, "") == root.resolve()
    assert inside(root, "   ") == root.resolve()


def test_a_path_inside_the_root_is_allowed(tmp_path: Path) -> None:
    root = library(tmp_path)
    wanted = root / "Blade Runner (1982)"
    wanted.mkdir()

    assert inside(root, str(wanted)) == wanted.resolve()


def test_a_path_above_the_root_is_refused(tmp_path: Path) -> None:
    root = library(tmp_path)

    with pytest.raises(OutsideLibrary):
        inside(root, str(tmp_path))


def test_an_absolute_path_somewhere_else_entirely_is_refused(tmp_path: Path) -> None:
    root = library(tmp_path)

    with pytest.raises(OutsideLibrary):
        inside(root, "/etc/passwd")


def test_a_traversal_out_of_the_root_is_refused(tmp_path: Path) -> None:
    """``..`` is resolved before it is judged, so it cannot be smuggled through."""
    root = library(tmp_path)
    (root / "films").mkdir()

    with pytest.raises(OutsideLibrary):
        inside(root, str(root / "films" / ".." / ".." / "elsewhere"))


def test_a_symlink_pointing_out_of_the_root_is_refused(tmp_path: Path) -> None:
    """The whole point of resolving first. A link is followed, then refused for where
    it lands — not accepted for where it sits."""
    root = library(tmp_path)
    outside = tmp_path / "secrets"
    outside.mkdir()
    (outside / "Film.mkv").touch()
    (root / "escape").symlink_to(outside)

    with pytest.raises(OutsideLibrary):
        directory_in(root, str(root / "escape"))
    with pytest.raises(OutsideLibrary):
        video_in(root, str(root / "escape" / "Film.mkv"))


def test_a_path_that_is_not_there_is_not_a_directory(tmp_path: Path) -> None:
    root = library(tmp_path)

    with pytest.raises(NotUsable):
        directory_in(root, str(root / "gone"))


def test_a_file_where_a_directory_was_wanted_is_refused(tmp_path: Path) -> None:
    root = library(tmp_path)
    (root / "Film.mkv").touch()

    with pytest.raises(NotUsable):
        directory_in(root, str(root / "Film.mkv"))


def test_a_directory_where_a_video_was_wanted_is_refused(tmp_path: Path) -> None:
    root = library(tmp_path)
    (root / "films").mkdir()

    with pytest.raises(NotUsable):
        video_in(root, str(root / "films"))


def test_something_that_is_not_a_video_is_refused(tmp_path: Path) -> None:
    """So no route can hand ``ffprobe`` an arbitrary file under the mount."""
    root = library(tmp_path)
    (root / "notes.txt").touch()

    with pytest.raises(NotUsable) as raised:
        video_in(root, str(root / "notes.txt"))

    # The refusal says which containers there are, because the next question is always
    # "well what does it take, then".
    assert ".mkv" in str(raised.value)


def test_the_case_of_the_suffix_does_not_matter(tmp_path: Path) -> None:
    root = library(tmp_path)
    film = root / "FILM.MKV"
    film.touch()

    assert video_in(root, str(film)) == film.resolve()


def test_every_offered_suffix_is_actually_accepted(tmp_path: Path) -> None:
    root = library(tmp_path)

    for suffix in VIDEO_SUFFIXES:
        film = root / f"Film{suffix}"
        film.touch()
        assert video_in(root, str(film)) == film.resolve(), suffix


# --- what is there -----------------------------------------------------------


def test_directories_come_before_files(tmp_path: Path) -> None:
    """A browse is a walk downwards, so the way down is above the things to pick."""
    root = library(tmp_path)
    (root / "Zulu.mkv").touch()
    (root / "Aliens").mkdir()
    (root / "Brazil.mkv").touch()

    listing = list_directory(root, root.resolve())

    assert [entry.name for entry in listing.entries] == ["Aliens", "Brazil.mkv", "Zulu.mkv"]
    assert [entry.is_dir for entry in listing.entries] == [True, False, False]


def test_anything_that_is_not_a_film_is_left_out(tmp_path: Path) -> None:
    root = library(tmp_path)
    (root / ".hidden").mkdir()
    (root / ".Film.mkv").touch()
    (root / "poster.jpg").touch()
    (root / "Film.mkv").touch()

    listing = list_directory(root, root.resolve())

    assert [entry.name for entry in listing.entries] == ["Film.mkv"]


def test_a_row_is_sized_in_whichever_unit_says_something(tmp_path: Path) -> None:
    """A film is gigabytes, a trailer is not, and `0.0 GB` beside a file reads as a fault."""
    root = library(tmp_path)
    sized(root / "Trailer.mkv", 40_000_000)
    sized(root / "Feature.mkv", 8_100_000_000)

    sizes = {entry.name: entry.size for entry in list_directory(root, root.resolve()).entries}

    assert sizes == {"Feature.mkv": "8.1 GB", "Trailer.mkv": "40 MB"}


def test_a_directory_row_carries_no_size(tmp_path: Path) -> None:
    """Adding up a directory means walking it, and the answer would not be worth it."""
    root = library(tmp_path)
    (root / "Aliens").mkdir()

    assert list_directory(root, root.resolve()).entries[0].size == ""


def test_the_root_has_nowhere_up_to_go(tmp_path: Path) -> None:
    root = library(tmp_path)

    assert list_directory(root, root.resolve()).parent is None


def test_below_the_root_there_is_a_way_back(tmp_path: Path) -> None:
    root = library(tmp_path)
    (root / "films").mkdir()

    listing = list_directory(root, (root / "films").resolve())

    assert listing.parent == root.resolve()


def test_the_crumbs_lead_from_the_root_to_here(tmp_path: Path) -> None:
    """The first crumb is the root as it is written, which in production is ``/mnt``."""
    root = library(tmp_path)
    here = root / "films" / "1982"
    here.mkdir(parents=True)

    crumbs = list_directory(root, here.resolve()).crumbs

    assert [name for name, _ in crumbs] == [str(root), "films", "1982"]
    assert [path for _, path in crumbs] == [
        root.resolve(),
        (root / "films").resolve(),
        here.resolve(),
    ]


def test_an_empty_directory_says_so_rather_than_looking_broken(tmp_path: Path) -> None:
    root = library(tmp_path)
    (root / "poster.jpg").touch()

    assert list_directory(root, root.resolve()).is_empty


def test_a_directory_past_the_cap_says_how_much_it_dropped(tmp_path: Path) -> None:
    root = library(tmp_path)
    for index in range(MAX_ENTRIES + 12):
        (root / f"Film {index:04d}.mkv").touch()

    listing = list_directory(root, root.resolve())

    assert len(listing.entries) == MAX_ENTRIES
    assert listing.dropped == 12


# --- finding films with nothing to go on -------------------------------------


def test_find_videos_looks_breadth_first(tmp_path: Path) -> None:
    """A mount point's own films are the likely answer; its twentieth subdirectory is not."""
    root = library(tmp_path)
    (root / "Top.mkv").touch()
    (root / "films" / "1982").mkdir(parents=True)
    (root / "films" / "Middle.mkv").touch()
    (root / "films" / "1982" / "Deep.mkv").touch()

    found = [path.name for path in find_videos(root)]

    assert found == ["Top.mkv", "Middle.mkv", "Deep.mkv"]


def test_find_videos_stops_at_the_depth_it_was_given(tmp_path: Path) -> None:
    root = library(tmp_path)
    buried = root / "a" / "b" / "c"
    buried.mkdir(parents=True)
    (buried / "Film.mkv").touch()

    assert find_videos(root, depth=2) == []
    assert [path.name for path in find_videos(root, depth=3)] == ["Film.mkv"]


def test_find_videos_stops_at_the_limit(tmp_path: Path) -> None:
    root = library(tmp_path)
    for index in range(30):
        (root / f"Film {index:02d}.mkv").touch()

    assert len(find_videos(root, limit=5)) == 5


def test_find_videos_skips_what_it_cannot_read(tmp_path: Path) -> None:
    """An unreadable directory costs that directory, not the search."""
    root = library(tmp_path)
    (root / "Film.mkv").touch()
    shut = root / "shut"
    shut.mkdir()
    (shut / "Hidden.mkv").touch()
    shut.chmod(0o000)
    try:
        assert [path.name for path in find_videos(root)] == ["Film.mkv"]
    finally:
        shut.chmod(0o755)


# --- sizes -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (40_000_000, "40 MB"),
        (999_000_000, "999 MB"),
        (1_000_000_000, "1.0 GB"),
        (8_100_000_000, "8.1 GB"),
        (64_000_000_000, "64.0 GB"),
    ],
)
def test_human_size_switches_unit_where_the_number_stops_saying_anything(
    tmp_path: Path, size: int, expected: str
) -> None:
    assert human_size(sized(library(tmp_path) / "Film.mkv", size)) == expected


def test_a_file_that_went_away_is_sizeless_rather_than_an_error(tmp_path: Path) -> None:
    assert human_size(library(tmp_path) / "never-existed.mkv") == ""
