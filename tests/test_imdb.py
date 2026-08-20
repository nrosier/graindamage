"""IMDb ``/technical``: parsing three page layouts, and the optional fetcher.

The parser is the part that matters — a pasted page is the documented path and the
only one that works without configuration — so it gets a fixture per layout IMDb has
shipped. The fetcher tests are about the contract an operator has to implement.
"""

from __future__ import annotations

from typing import Any

import httpx2
import pytest

from app.providers import ProviderDisabled, ProviderUnavailable
from app.providers.imdb import ImdbTechnicalProvider, parse_technical
from tests.support import Handler, fixture, make_settings, mock_client, recording_handler, run

FETCHER_URL = "https://fetcher.example/get"
TECHNICAL_URL = "https://www.imdb.com/title/tt0083658/technical/"


def provider(handler: Handler, **overrides: Any) -> ImdbTechnicalProvider:
    settings = make_settings(**{"imdb_fetcher_url": FETCHER_URL, **overrides})
    return ImdbTechnicalProvider(settings, client=mock_client(handler))


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


# --- the fetcher -------------------------------------------------------------


def test_without_a_fetcher_url_the_paste_path_is_named() -> None:
    client = ImdbTechnicalProvider(
        make_settings(), client=mock_client(lambda request: httpx2.Response(200, text="x"))
    )

    assert not client.enabled
    with pytest.raises(ProviderDisabled, match="Paste the /technical page source instead"):
        run(client.fetch("tt0083658"))


@pytest.mark.parametrize("imdb_id", ["nope", "tt123", "0083658", "tt0083658/technical", ""])
def test_only_a_real_title_id_is_ever_requested(imdb_id: str) -> None:
    handler, seen = recording_handler(httpx2.Response(200, text="<html></html>"))

    with pytest.raises(ProviderUnavailable, match="is not an IMDb title id"):
        run(provider(handler).fetch(imdb_id))
    assert seen == []


def test_the_fetcher_is_asked_for_the_technical_url() -> None:
    handler, seen = recording_handler(httpx2.Response(200, text="<html>page</html>"))

    assert run(provider(handler).fetch("tt0083658")) == "<html>page</html>"

    assert str(seen[0].url).startswith(FETCHER_URL)
    assert dict(seen[0].url.params) == {"url": TECHNICAL_URL}
    assert seen[0].headers["user-agent"].startswith("graindamage/")
    assert "authorization" not in seen[0].headers


def test_a_configured_token_is_sent_as_a_bearer() -> None:
    handler, seen = recording_handler(httpx2.Response(200, text="<html>page</html>"))

    run(provider(handler, imdb_fetcher_token="s3cret").fetch("tt0083658"))

    assert seen[0].headers["authorization"] == "Bearer s3cret"


@pytest.mark.parametrize("key", ["content", "html", "body", "data", "result", "text"])
def test_a_json_envelope_is_unwrapped(key: str) -> None:
    # browserless, most scraping APIs and a hand-written Worker all differ here.
    handler, _ = recording_handler(httpx2.Response(200, json={key: "<html>page</html>"}))

    assert run(provider(handler).fetch("tt0083658")) == "<html>page</html>"


def test_a_bare_json_string_is_accepted_too() -> None:
    handler, _ = recording_handler(httpx2.Response(200, json="<html>page</html>"))

    assert run(provider(handler).fetch("tt0083658")) == "<html>page</html>"


def test_an_unrecognised_json_envelope_is_passed_through_as_text() -> None:
    # Better to hand the parser something it will report as empty than to guess.
    handler, _ = recording_handler(httpx2.Response(200, json={"pageSource": "<html/>"}))

    assert run(provider(handler).fetch("tt0083658")) == '{"pageSource":"<html/>"}'


def test_html_that_looks_like_json_is_left_alone() -> None:
    body = '{"content": "not really json-typed"}'
    handler, _ = recording_handler(
        httpx2.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})
    )

    assert run(provider(handler).fetch("tt0083658")) == body


def test_an_http_failure_from_the_fetcher_is_reported() -> None:
    handler, _ = recording_handler(httpx2.Response(502, text="Bad Gateway"))

    with pytest.raises(ProviderUnavailable, match="returned HTTP 502"):
        run(provider(handler).fetch("tt0083658"))


def test_an_empty_body_is_a_failure_not_an_empty_page() -> None:
    handler, _ = recording_handler(httpx2.Response(200, text="   \n  "))

    with pytest.raises(ProviderUnavailable, match="returned an empty body"):
        run(provider(handler).fetch("tt0083658"))


def test_a_transport_failure_names_the_kind() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("too slow")

    with pytest.raises(ProviderUnavailable) as caught:
        run(provider(handler).fetch("tt0083658"))

    assert caught.value.message == "Could not reach the IMDb fetcher."
    assert caught.value.detail == "ReadTimeout"


def test_fetch_specs_parses_and_caches() -> None:
    handler, seen = recording_handler(
        httpx2.Response(200, text=fixture("imdb_technical_next_data.html"))
    )
    client = provider(handler)

    first = run(client.fetch_specs("tt0083658"))
    second = run(client.fetch_specs("tt0083658"))

    assert first.negative_formats == ["35 mm"]
    # Previewing the specs and then asking for advice must not cost two fetches.
    assert first is second
    assert len(seen) == 1
