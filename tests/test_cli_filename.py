"""Reading a title and a year out of a filename.

The table below is the specification: each row is a shape that actually turns up in a
library, and the two ambiguous-year cases are the reason the rule is "the last candidate
that has already happened" rather than "the first thing that looks like a year".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.cli.filename import guess_name

# (filename, expected title, expected year)
NAMES: list[tuple[str, str, int | None]] = [
    # The common shape: title, year, then everything about the release.
    ("Blade.Runner.1982.2160p.UHD.BluRay.DV.TrueHD.7.1-FraMeSToR.mkv", "Blade Runner", 1982),
    ("The_Godfather_1972_REMASTERED_1080p_BluRay_x265.mkv", "The Godfather", 1972),
    ("Movie Title (2019) [1080p] [WEBRip].mkv", "Movie Title", 2019),
    ("Se7en.1995.mkv", "Se7en", 1995),
    ("Amélie.2001.1080p.BluRay.mkv", "Amélie", 2001),
    ("Spider-Man.2002.1080p.x264-GROUP.mkv", "Spider Man", 2002),
    ("Apollo.13.1995.1080p.mkv", "Apollo 13", 1995),
    ("The.Final.Cut.2004.720p.WEBRip.mkv", "The Final Cut", 2004),
    # Two candidates, and the release year is the second one.
    ("Blade.Runner.2049.2017.2160p.BluRay.x265-GRP.mkv", "Blade Runner 2049", 2017),
    ("2012.2009.1080p.BluRay.mkv", "2012", 2009),
    # Two candidates, and the first one is the title.
    ("2001.A.Space.Odyssey.1968.1080p.mkv", "2001 A Space Odyssey", 1968),
    ("1917.2019.2160p.UHD.BluRay.mkv", "1917", 2019),
    # A year that has not happened is part of the title, not a release date.
    ("Blade.Runner.2049.2160p.BluRay.x265.mkv", "Blade Runner 2049", None),
    # No year at all: strip the release furniture off the tail instead.
    ("Some.Film.1080p.BluRay.x264-GROUP.mkv", "Some Film", None),
    ("The.Final.Cut.1080p.HDR.DTS-HD.MA.5.1-CtrlHD.mkv", "The Final Cut", None),
    ("Film.Name.4K.HDR10.Atmos.mkv", "Film Name", None),
    ("1917.1080p.BluRay.mkv", "1917", None),
    # Nothing to strip, and nothing left over.
    ("Nosferatu.mkv", "Nosferatu", None),
    ("movie.mkv", "movie", None),
]


@pytest.mark.parametrize(("name", "title", "year"), NAMES)
def test_a_filename_says_what_the_film_is(name: str, title: str, year: int | None) -> None:
    guess = guess_name(Path("/movies") / name, max_year=2026)

    assert (guess.title, guess.year) == (title, year)
    assert guess.source == "filename"


def test_the_year_ceiling_moves_with_the_calendar() -> None:
    """A 2024 release is a release; in 2023 it would have been part of the title.

    A rejected year is kept rather than stripped, which is the same rule that gets
    ``Blade Runner 2049`` right: if it is not a release date it is part of the name.
    """
    assert guess_name(Path("Dune.Part.Two.2024.mkv"), max_year=2026).year == 2024

    earlier = guess_name(Path("Dune.Part.Two.2024.mkv"), max_year=2023)
    assert (earlier.title, earlier.year) == ("Dune Part Two 2024", None)


def test_a_number_in_a_title_survives_a_name_with_no_year() -> None:
    """``Apollo 13`` keeps its 13; ``TrueHD 7.1`` keeps neither its 7 nor its 1."""
    assert guess_name(Path("Apollo.13.1080p.BluRay.mkv")).title == "Apollo 13"
    assert guess_name(Path("Malcolm.X.2160p.HDR.mkv")).title == "Malcolm X"
    assert guess_name(Path("Heat.1080p.TrueHD.7.1.H.264-GRP.mkv")).title == "Heat"


def test_a_contentless_name_falls_back_to_the_directory() -> None:
    guess = guess_name(Path("/movies/Blade Runner (1982)/movie.mkv"), max_year=2026)

    assert (guess.title, guess.year) == ("Blade Runner", 1982)
    assert guess.source == "directory"


@pytest.mark.parametrize("stem", ["title00", "track_3", "VTS_01", "disc1", "07", "untitled"])
def test_the_names_a_ripper_produces_are_all_contentless(stem: str) -> None:
    guess = guess_name(Path(f"/movies/The Thing (1982)/{stem}.mkv"), max_year=2026)

    assert (guess.title, guess.year, guess.source) == ("The Thing", 1982, "directory")


def test_a_contentless_directory_leaves_the_filename_alone() -> None:
    """Both names being useless is not a reason to prefer the useless one."""
    guess = guess_name(Path("/movies/disc1/movie.mkv"), max_year=2026)

    assert (guess.title, guess.source) == ("movie", "filename")


def test_the_query_is_the_title_and_the_year_is_only_a_filter() -> None:
    guess = guess_name(Path("Blade.Runner.1982.2160p.mkv"), max_year=2026)

    assert guess.query == "Blade Runner"
    assert guess.describe() == "Blade Runner (1982) — read from the filename"
    assert guess_name(Path("Nosferatu.mkv")).describe() == "Nosferatu — read from the filename"
