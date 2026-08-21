"""IMDb ``/technical``: reading the rows out of the four shapes they arrive in.

Nothing here touches the network — the app never requests IMDb. Three of the four
layouts are pages IMDb has shipped, and the fourth is the one a user actually produces:
the formatted text you get by selecting the specifications and copying them.
"""

from __future__ import annotations

import pytest

from app.providers.imdb import parse_technical, specs_from_rows
from tests.support import fixture

# --- the JSON island (current IMDb) -----------------------------------------


def test_the_next_data_island_yields_every_row() -> None:
    specs = parse_technical(fixture("imdb_technical_next_data.html"))

    assert specs.negative_formats == ["35 mm"]
    assert specs.cinematographic_processes == [
        "Panavision (anamorphic)",
        "Super 35",
        "Digital Intermediate (4K)",
    ]
    assert specs.aspect_ratios == ["2.39 : 1"]
    assert specs.runtimes == ["1 hour 57 minutes", "1 hour 57 minutes (Final Cut)"]
    assert specs.sound_mixes == ["70 mm 6-Track", "Dolby Stereo"]
    assert specs.printed_formats == ["35 mm", "70 mm (blow-up)"]
    assert specs.film_lengths == ["3,196 m"]
    assert specs.laboratories == ["Technicolor, Hollywood (CA), USA"]
    assert not specs.is_empty


def test_attributes_beside_a_row_title_are_kept() -> None:
    # IMDb hangs the rental house off the camera row; it is part of the answer.
    specs = parse_technical(fixture("imdb_technical_next_data.html"))

    assert specs.cameras == ["Panaflex Camera", "Joe Dunton & Company"]


def test_a_row_whose_value_repeats_its_heading_survives() -> None:
    # The Color row says literally "Color". Suppressing the heading inside a nested
    # section would throw the only value away.
    assert parse_technical(fixture("imdb_technical_next_data.html")).colors == ["Color"]


# --- the current server-rendered markup -------------------------------------


def test_the_testid_markup_yields_the_rows_without_their_labels() -> None:
    specs = parse_technical(fixture("imdb_technical_markup.html"))

    assert specs.runtimes == ["1 hour 57 minutes"]
    assert specs.aspect_ratios == ["2.20 : 1"]
    assert specs.negative_formats == ["35 mm"]
    assert specs.sound_mixes == ["Dolby Stereo"]
    # "Panavision&nbsp;(anamorphic)" — the entity and the exotic space both go.
    assert specs.cinematographic_processes == ["Panavision (anamorphic)", "Super 35"]
    assert specs.cameras == ["Panaflex Camera, Panavision Lenses"]


def test_markup_attributes_never_become_values() -> None:
    text = " ".join(parse_technical(fixture("imdb_technical_markup.html")).all_format_text())

    assert "ipc-metadata-list" not in text
    assert "class=" not in text


def test_a_broken_json_island_falls_through_to_the_markup() -> None:
    page = (
        '<script id="__NEXT_DATA__" type="application/json">{ this is not json</script>'
        + fixture("imdb_technical_markup.html")
    )

    assert parse_technical(page).negative_formats == ["35 mm"]


# --- the pre-2020 table ------------------------------------------------------


def test_the_legacy_table_still_parses() -> None:
    specs = parse_technical(fixture("imdb_technical_legacy.html"))

    assert specs.runtimes == ["142 min"]
    assert specs.sound_mixes == ["Mono"]
    assert specs.aspect_ratios == ["1.37 : 1"]
    assert specs.cameras == ["Konvas 2M, Lomo Lenses"]
    assert specs.laboratories == ["Mosfilm, Moscow, USSR"]
    assert specs.negative_formats == ["35 mm"]
    assert specs.cinematographic_processes == ["Spherical"]
    assert specs.printed_formats == ["35 mm"]


def test_the_island_wins_when_a_page_carries_both_layouts() -> None:
    # A saved page can contain the island and a rendered fallback; the island is the
    # one that was not mangled by whatever saved it.
    page = """
    <script id="__NEXT_DATA__" type="application/json">
      {"categories": [{"id": "negative_format", "name": "Negative format",
        "section": {"items": [{"id": "nf-1", "rowTitle": "16 mm"}]}}]}
    </script>
    <table><tr><td class="label">Negative Format</td><td>35 mm</td></tr></table>
    """

    assert parse_technical(page).negative_formats == ["16 mm"]


# --- robustness --------------------------------------------------------------


@pytest.mark.parametrize("page", ["", "   ", "<html><body>Not IMDb at all</body></html>"])
def test_an_unusable_page_gives_empty_specs_rather_than_an_error(page: str) -> None:
    # The advice degrades to source-only heuristics; a raised exception would take the
    # whole request with it.
    specs = parse_technical(page)

    assert specs.is_empty
    assert specs.negative_formats == []


def test_navigation_noise_is_not_a_specification() -> None:
    page = """
    <table><tr><td class="label">Negative Format</td>
    <td>35 mm<br>See more<br>Edit</td></tr></table>
    """

    assert parse_technical(page).negative_formats == ["35 mm"]


def test_repeated_values_are_collapsed_case_insensitively() -> None:
    page = """
    <table><tr><td class="label">Negative Format</td>
    <td>35 mm<br>35 MM<br>Super 35</td></tr></table>
    """

    assert parse_technical(page).negative_formats == ["35 mm", "Super 35"]


def test_a_row_with_too_many_values_is_capped() -> None:
    values = "<br>".join(f"{index} min" for index in range(1, 30))
    page = f'<table><tr><td class="label">Runtime</td><td>{values}</td></tr></table>'

    assert len(parse_technical(page).runtimes) == 12


def test_an_absurdly_long_value_is_dropped() -> None:
    # A row that swallowed half the page is markup we misread, not a camera.
    page = f'<table><tr><td class="label">Camera</td><td>{"Arriflex " * 40}</td></tr></table>'

    assert parse_technical(page).cameras == []


def test_unknown_rows_are_ignored() -> None:
    page = """
    <table>
      <tr><td class="label">Filming Dates</td><td>March 1981</td></tr>
      <tr><td class="label">Negative Format</td><td>35 mm</td></tr>
    </table>
    """

    specs = parse_technical(page)

    assert specs.negative_formats == ["35 mm"]
    assert specs.runtimes == []


# --- the formatted text, which is what a user actually pastes ----------------

# The technical page as it reads on screen, furniture and all.
PASTED_PAGE = """Blade Runner
Technical specifications
Edit
Runtime
1 hour 57 minutes
Sound mix
Dolby Stereo
70 mm 6-Track
Color
Color
Aspect ratio
2.20 : 1
Camera
Panavision Panaflex, Joe Dunton Cooke Lenses
Negative format
35 mm
Cinematographic process
Super 35
Printed film format
70 mm (blow-up)
Film length
3,175 m
Laboratory
Technicolor, Hollywood, USA
More to explore
Recently viewed
"""


def test_the_formatted_specifications_parse_like_a_page() -> None:
    specs = parse_technical(PASTED_PAGE)

    assert specs.negative_formats == ["35 mm"]
    assert specs.cinematographic_processes == ["Super 35"]
    assert specs.printed_formats == ["70 mm (blow-up)"]
    assert specs.aspect_ratios == ["2.20 : 1"]
    assert specs.sound_mixes == ["Dolby Stereo", "70 mm 6-Track"]
    assert specs.film_lengths == ["3,175 m"]
    assert specs.laboratories == ["Technicolor, Hollywood, USA"]
    assert specs.runtimes == ["1 hour 57 minutes"]


def test_the_colour_row_that_repeats_its_own_heading_keeps_its_value() -> None:
    """IMDb's colour row reads "Color / Color", and the second one is the answer."""
    assert parse_technical(PASTED_PAGE).colors == ["Color"]


@pytest.mark.parametrize(
    "line",
    [
        "Aspect ratio: 2.39 : 1",
        "Aspect ratio\t2.39 : 1",
        "Aspect ratio  2.39 : 1",
        "Aspect ratio 2.39 : 1",
        "Aspect Ratio:\t2.39 : 1",
        "aspect ratios  2.39 : 1",
    ],
)
def test_a_heading_and_its_value_on_one_line(line: str) -> None:
    assert parse_technical(line).aspect_ratios == ["2.39 : 1"]


@pytest.mark.parametrize("separator", ["\t", " · ", "   "])
def test_values_sharing_a_line_are_separated(separator: str) -> None:
    page = f"Sound mix{separator}Dolby Digital{separator}DTS{separator}SDDS"

    assert parse_technical(page).sound_mixes == ["Dolby Digital", "DTS", "SDDS"]


def test_page_furniture_after_the_rows_is_not_a_value() -> None:
    specs = parse_technical(PASTED_PAGE)

    assert "Recently viewed" not in specs.laboratories
    assert "Edit" not in specs.runtimes
    assert "Blade Runner" not in specs.runtimes


def test_prose_after_a_row_ends_it() -> None:
    page = (
        "Negative format\n35 mm\n"
        "This article about a film is a stub and you can help by expanding it today.\n"
    )

    specs = parse_technical(page)

    assert specs.negative_formats == ["35 mm"]


def test_text_before_any_heading_is_never_a_value() -> None:
    page = "Sign in\nWatchlist\nBlade Runner (1982)\nRuntime\n1 hour 57 minutes\n"

    specs = parse_technical(page)

    assert specs.runtimes == ["1 hour 57 minutes"]
    assert specs.negative_formats == []


def test_a_script_that_mentions_a_row_name_is_not_a_row() -> None:
    page = "<script>var runtime = '117 min'; var aspectRatio = '2.20 : 1';</script>"

    assert parse_technical(page).is_empty


def test_the_markup_layouts_still_win_over_the_text_reading() -> None:
    """A pasted page source has to parse as markup, not as prose that contains tags."""
    specs = parse_technical(fixture("imdb_technical_markup.html"))

    assert specs.negative_formats
    assert not any("<" in value for value in specs.negative_formats)


# --- rows that never came from a page at all ---------------------------------


def test_rows_from_a_lookup_are_held_to_the_same_limits() -> None:
    specs = specs_from_rows(
        {
            "aspect_ratios": ["2.39 : 1", "2.39 : 1", "  2.39 : 1  "],
            "sound_mixes": [f"Mix {index}" for index in range(20)],
            "cameras": ["x" * 500],
            "laboratories": ["Deluxe, London, UK"],
        }
    )

    assert specs.aspect_ratios == ["2.39 : 1"]
    assert len(specs.sound_mixes) == 12
    assert specs.cameras == []
    assert specs.laboratories == ["Deluxe, London, UK"]


def test_rows_a_lookup_invented_a_field_name_for_are_dropped() -> None:
    specs = specs_from_rows({"resolution": ["4K"], "negative_formats": ["16 mm"]})

    assert specs.negative_formats == ["16 mm"]
    assert not hasattr(specs, "resolution")


def test_values_that_are_not_strings_are_ignored() -> None:
    specs = specs_from_rows({"runtimes": [117, None, "1 hour 57 minutes"]})  # type: ignore[list-item]

    assert specs.runtimes == ["1 hour 57 minutes"]


def test_no_rows_at_all_is_empty_specs_not_an_error() -> None:
    assert specs_from_rows({}).is_empty
