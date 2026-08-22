"""The HTTP surface: three steps, plus the two ways it is allowed to degrade.

Two properties matter more than any individual assertion here, and most of these
tests exist to pin one of them down:

* **A failure is rendered, not raised.** Every error path has to answer HTTP 200
  with an explanation, because HTMX does not swap a non-2xx response into the page.
* **Nothing blocks the answer.** A dead TMDB, a specs look-up that knows nothing,
  an unparseable paste and a silent Gemini each cost a warning, never the settings.

Providers reach a mock transport, so no test here needs the network — and
:func:`make_settings` keeps a developer's real ``.env`` out of the app under test.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from app import __version__
from tests.support import fixture, json_response, make_app, make_settings, mock_client

FFPROBE_REPORT = fixture("ffprobe_uhd_hdr.json")
TECHNICAL_PAGE = fixture("imdb_technical_next_data.html")

# What a technical-specs look-up answers with.
SPECS_ANSWER: dict[str, Any] = {
    "negative_formats": ["35 mm"],
    "cinematographic_processes": ["Super 35"],
    "aspect_ratios": ["2.20 : 1"],
    "sound_mixes": ["Dolby Stereo"],
    "confidence": "medium",
}

SEARCH_PAYLOAD: dict[str, Any] = {
    "results": [
        {
            "id": 78,
            "title": "Blade Runner",
            "release_date": "1982-06-25",
            "overview": "A blade runner must pursue and terminate four replicants.",
            "poster_path": "/63N9uy8nd9j7Eog2axPQ8lbr3Wj.jpg",
        },
        {"id": 335984, "title": "Blade Runner 2049", "release_date": "2017-10-04"},
    ]
}

MOVIE_PAYLOAD: dict[str, Any] = {
    "id": 78,
    "title": "Blade Runner",
    "release_date": "1982-06-25",
    "runtime": 117,
    "genres": [{"id": 878, "name": "Science Fiction"}],
    "external_ids": {"imdb_id": "tt0083658"},
    "credits": {"crew": [{"name": "Ridley Scott", "job": "Director"}]},
}


def transport(
    response: httpx2.Response | None = None,
    *,
    search: httpx2.Response | None = None,
) -> tuple[httpx2.AsyncClient, list[httpx2.Request]]:
    """A recording transport. ``search`` answers TMDB's search path separately."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if search is not None and "/search/" in request.url.path:
            return search
        return response if response is not None else json_response({})

    return mock_client(handler), seen


def tmdb_transport(
    movie: httpx2.Response | None = None,
    search: httpx2.Response | None = None,
) -> tuple[httpx2.AsyncClient, list[httpx2.Request]]:
    return transport(
        movie if movie is not None else json_response(MOVIE_PAYLOAD),
        search=search if search is not None else json_response(SEARCH_PAYLOAD),
    )


def gemini_envelope(payload: Any) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": json.dumps(payload)}]},
                    "finishReason": "STOP",
                }
            ]
        },
    )


def gemini_transport(
    answer: Any = None,
    *,
    specs: Any = None,
    response: httpx2.Response | None = None,
) -> tuple[httpx2.AsyncClient, list[httpx2.Request]]:
    """A transport answering Gemini's two calls: the specs look-up and the review.

    They go to the same endpoint, so they are told apart by the question asked.
    """
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if response is not None:
            return response
        if b"Fetch the technical specifications" in request.content:
            return gemini_envelope(SPECS_ANSWER if specs is None else specs)
        return gemini_envelope({"summary": "Reviewed for this film."} if answer is None else answer)

    return mock_client(handler), seen


def client_for(
    *,
    tmdb: httpx2.AsyncClient | None = None,
    gemini: httpx2.AsyncClient | None = None,
    **overrides: Any,
) -> TestClient:
    settings = make_settings(**overrides)
    return TestClient(make_app(settings, tmdb=tmdb, gemini=gemini))


@pytest.fixture
def client() -> TestClient:
    """A client with no integrations configured — the fresh-container case."""
    return client_for()


# --- the container itself ----------------------------------------------------


def test_healthz_reports_version_and_capabilities(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["capabilities"] == {
        "tmdb_search": False,
        "gemini_advice": False,
        "imdb_technical_lookup": False,
        # The rules engine has no configuration, so an unkeyed container is still useful.
        "baseline_rules": True,
    }


def test_capabilities_follow_configuration() -> None:
    body = client_for(tmdb_api_key="tmdb-key", gemini_api_key="gemini-key").get("/healthz").json()

    # The look-up rides on the Gemini key: there is nothing else to configure.
    assert body["capabilities"] == {
        "tmdb_search": True,
        "gemini_advice": True,
        "imdb_technical_lookup": True,
        "baseline_rules": True,
    }


def test_index_renders_without_any_keys(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "graindamage" in response.text
    assert "set TMDB_API_KEY to enable" in response.text


def test_index_serves_static_assets(client: TestClient) -> None:
    # HTMX is vendored rather than loaded from a CDN, so it has to be served here.
    for path in ("/static/css/app.css", "/static/js/htmx.min.js"):
        assert client.get(path).status_code == 200, path


# --- step 1: search ----------------------------------------------------------


def test_search_without_a_key_says_so_and_offers_the_alternative(client: TestClient) -> None:
    response = client.post("/search", data={"query": "Blade Runner"})

    assert response.status_code == 200
    assert "TMDB search is not configured. Set TMDB_API_KEY." in response.text
    assert "You can still get settings without TMDB" in response.text


def test_a_blank_query_costs_no_upstream_request() -> None:
    tmdb, seen = tmdb_transport()
    response = client_for(tmdb=tmdb, tmdb_api_key="key").post("/search", data={"query": "   "})

    assert "Type a title to search for." in response.text
    assert seen == []


def test_search_lists_the_matches_as_a_partial() -> None:
    tmdb, _ = tmdb_transport()

    response = client_for(tmdb=tmdb, tmdb_api_key="key").post(
        "/search", data={"query": "Blade Runner"}
    )

    assert response.status_code == 200
    assert "2 matches for" in response.text
    assert "Blade Runner 2049" in response.text
    assert 'name="tmdb_id" value="78"' in response.text
    assert 'hx-post="/pick"' in response.text
    # A partial, not a full page — and it refreshes the step indicator out of band.
    assert "<!doctype html>" not in response.text.lower()
    assert 'hx-swap-oob="true"' in response.text


def test_a_search_with_no_matches_suggests_what_to_try() -> None:
    tmdb, _ = tmdb_transport(search=json_response({"results": []}))

    response = client_for(tmdb=tmdb, tmdb_api_key="key").post("/search", data={"query": "zzz"})

    assert "Nothing found" in response.text
    assert "original-language" in response.text


def test_an_upstream_failure_is_still_an_http_200() -> None:
    tmdb, _ = tmdb_transport(search=json_response({}, 503))

    response = client_for(tmdb=tmdb, tmdb_api_key="key").post("/search", data={"query": "blade"})

    # A 502 here would leave HTMX with nothing to swap and the user with no explanation.
    assert response.status_code == 200
    assert "TMDB returned HTTP 503." in response.text


# --- step 2: pick the film ---------------------------------------------------


def test_picking_a_film_renders_the_source_form() -> None:
    tmdb, _ = tmdb_transport()

    response = client_for(tmdb=tmdb, tmdb_api_key="key").post("/pick", data={"tmdb_id": "78"})

    assert response.status_code == 200
    assert 'hx-post="/advise"' in response.text
    assert 'name="tmdb_id" value="78"' in response.text
    assert 'name="imdb_id" value="tt0083658"' in response.text
    assert 'name="technical_source"' in response.text
    assert 'name="source_text"' in response.text
    assert "Ridley Scott" in response.text
    assert "Get settings" in response.text


def test_the_form_names_all_three_source_tools() -> None:
    tmdb, _ = tmdb_transport()

    text = client_for(tmdb=tmdb, tmdb_api_key="key").post("/pick", data={"tmdb_id": "78"}).text

    assert "ffprobe -v quiet -print_format json -show_format -show_streams FILE" in text
    assert "mkvinfo FILE" in text
    assert "mediainfo FILE" in text


@pytest.mark.parametrize("configured", [True, False])
def test_the_gemini_checkbox_appears_only_when_a_key_is_set(configured: bool) -> None:
    tmdb, _ = tmdb_transport()
    client = client_for(
        tmdb=tmdb, tmdb_api_key="key", gemini_api_key="gemini-key" if configured else None
    )

    text = client.post("/pick", data={"tmdb_id": "78"}).text

    assert ('name="use_gemini"' in text) is configured


@pytest.mark.parametrize("configured", [True, False])
def test_the_lookup_button_appears_only_with_a_gemini_key(configured: bool) -> None:
    tmdb, _ = tmdb_transport()
    gemini, _ = gemini_transport()
    client = client_for(
        tmdb=tmdb,
        gemini=gemini,
        tmdb_api_key="key",
        gemini_api_key="key" if configured else None,
    )

    text = client.post("/pick", data={"tmdb_id": "78"}).text

    assert ("Look these up with Gemini" in text) is configured
    # The paste box and the link to the page are there either way.
    assert "Paste the technical specifications from IMDb" in text
    assert "https://www.imdb.com/title/tt0083658/technical/" in text


def test_picking_a_film_shows_the_rows_the_lookup_returned() -> None:
    tmdb, _ = tmdb_transport()
    gemini, seen = gemini_transport()
    client = client_for(tmdb=tmdb, gemini=gemini, tmdb_api_key="key", gemini_api_key="key")

    text = client.post("/pick", data={"tmdb_id": "78"}).text

    assert "Negative format" in text
    assert "35 mm" in text
    assert len(seen) == 1
    # Rows that were recalled rather than read must say so on the page.
    assert "not read from IMDb" in text
    assert "confidence medium" in text


def test_the_lookup_is_asked_about_the_film_the_user_picked() -> None:
    tmdb, _ = tmdb_transport()
    gemini, seen = gemini_transport()
    client = client_for(tmdb=tmdb, gemini=gemini, tmdb_api_key="key", gemini_api_key="key")

    client.post("/pick", data={"tmdb_id": "78"})

    asked = seen[0].content
    assert b"Blade Runner (1982)" in asked
    assert b"tt0083658" in asked
    # No key in the query string; it travels as a header.
    assert "key" not in seen[0].url.params


def test_a_failed_lookup_does_not_cost_the_form() -> None:
    tmdb, _ = tmdb_transport()
    gemini, _ = gemini_transport(response=httpx2.Response(502, text="Bad Gateway"))
    client = client_for(tmdb=tmdb, gemini=gemini, tmdb_api_key="key", gemini_api_key="key")

    response = client.post("/pick", data={"tmdb_id": "78"})

    assert response.status_code == 200
    assert "Gemini returned HTTP 502." in response.text
    # The paste box is the path that always works, so the form must survive.
    assert "Get settings" in response.text


def test_a_tmdb_failure_while_picking_is_reported() -> None:
    tmdb, _ = tmdb_transport(json_response({}, 500))

    response = client_for(tmdb=tmdb, tmdb_api_key="key").post("/pick", data={"tmdb_id": "78"})

    assert response.status_code == 200
    assert "TMDB returned HTTP 500." in response.text


# --- the technical panel -----------------------------------------------------


@pytest.mark.parametrize("imdb_id", ["", "nope", "tt123", "0083658", "tt0083658/technical"])
def test_only_a_real_title_id_is_looked_up(imdb_id: str) -> None:
    gemini, seen = gemini_transport()
    client = client_for(gemini=gemini, gemini_api_key="key")

    response = client.post("/technical", data={"imdb_id": imdb_id})

    assert "That is not an IMDb title id." in response.text
    assert "Ids look like tt0083658" in response.text
    assert seen == []


def test_without_a_key_the_panel_points_at_the_paste_box(client: TestClient) -> None:
    response = client.post("/technical", data={"imdb_id": "tt0083658"})

    assert response.status_code == 200
    assert "needs a Gemini key" in response.text
    assert "paste" in response.text


def test_the_looked_up_rows_are_rendered() -> None:
    gemini, _ = gemini_transport()

    text = (
        client_for(gemini=gemini, gemini_api_key="key")
        .post("/technical", data={"imdb_id": "tt0083658"})
        .text
    )

    assert "Negative format" in text
    assert "35 mm" in text
    assert "Sound mix" in text
    assert "Dolby Stereo" in text


def test_a_lookup_that_knows_nothing_asks_for_a_paste() -> None:
    gemini, _ = gemini_transport(specs={"confidence": "low"})

    text = (
        client_for(gemini=gemini, gemini_api_key="key")
        .post("/technical", data={"imdb_id": "tt0083658"})
        .text
    )

    assert "did not have the technical rows" in text
    assert "paste the specifications" in text


# --- step 3: the advice ------------------------------------------------------


def test_advice_covers_both_encoders_and_both_front_ends() -> None:
    tmdb, _ = tmdb_transport()
    client = client_for(tmdb=tmdb, tmdb_api_key="key")

    response = client.post(
        "/advise",
        data={"tmdb_id": "78", "source_text": FFPROBE_REPORT, "technical_source": TECHNICAL_PAGE},
    )

    assert response.status_code == 200
    text = response.text
    assert "libsvtav1" in text
    assert "libx265" in text
    assert text.count("HandBrakeCLI") == 2
    # Read from the file, not assumed: 4K HDR, and the stem comes from the film.
    assert "3840×2160" in text
    assert "HDR" in text
    assert "blade-runner-1982-2160p.av1.mkv" in text
    assert "ffmpeg -i input.mkv -map 0" in text
    assert "deterministic rules" in text
    assert "Read these first" not in text


def test_advice_needs_no_film_and_no_source(client: TestClient) -> None:
    # The empty form is a real path: someone who just wants sane 1080p defaults.
    response = client.post("/advise", data={})

    assert response.status_code == 200
    assert "libsvtav1" in response.text
    assert "output.av1.mkv" in response.text


def test_an_unparseable_paste_falls_back_to_defaults_with_a_warning(client: TestClient) -> None:
    response = client.post("/advise", data={"source_text": "this is not a media report"})

    assert response.status_code == 200
    assert "Settings below use 1080p defaults instead of your file." in response.text
    assert "libsvtav1" in response.text


def test_a_pasted_page_with_nothing_in_it_is_reported(client: TestClient) -> None:
    response = client.post(
        "/advise", data={"technical_source": "<html><body>Not IMDb at all</body></html>"}
    )

    assert "Nothing recognisable was found in the pasted specifications" in response.text
    assert "libsvtav1" in response.text


def test_pasted_specifications_beat_the_lookup() -> None:
    gemini, seen = gemini_transport()
    client = client_for(gemini=gemini, gemini_api_key="key")

    response = client.post(
        "/advise", data={"imdb_id": "tt0083658", "technical_source": TECHNICAL_PAGE}
    )

    # What the user pasted is what they meant; asking a model over it would be worse.
    assert seen == []
    # And it was the pasted rows that decided the grain, not the release year.
    assert "Super 35 mm" in response.text
    assert "Nothing recognisable" not in response.text
    assert "looked up by Gemini" not in response.text


def test_the_answer_says_when_its_rows_were_looked_up() -> None:
    gemini, _ = gemini_transport()
    client = client_for(gemini=gemini, gemini_api_key="key")

    response = client.post("/advise", data={"imdb_id": "tt0083658"})

    assert "looked up by Gemini, not read from IMDb" in response.text
    # The rows still did their job: Super 35 is a 35 mm negative, so grain is expected.
    assert "libsvtav1" in response.text


def test_a_grain_override_is_labelled_as_the_users_own(client: TestClient) -> None:
    response = client.post("/advise", data={"grain_override": "extreme"})

    assert "extreme grain" in response.text
    assert "set by you" in response.text
    assert "% confident" not in response.text


def test_a_dead_tmdb_does_not_block_the_answer() -> None:
    tmdb, _ = tmdb_transport(json_response({}, 500))
    client = client_for(tmdb=tmdb, tmdb_api_key="key")

    response = client.post("/advise", data={"tmdb_id": "78", "source_text": FFPROBE_REPORT})

    assert response.status_code == 200
    # Rendered through Jinja, so the apostrophe arrives as an entity.
    assert "details are missing from this answer." in response.text
    assert "libsvtav1" in response.text
    # Without the film there is no title to name the output after.
    assert "output.av1.mkv" in response.text


def test_a_failed_lookup_does_not_block_the_answer() -> None:
    gemini, _ = gemini_transport(response=httpx2.Response(502, text="Bad Gateway"))
    client = client_for(gemini=gemini, gemini_api_key="key")

    response = client.post("/advise", data={"imdb_id": "tt0083658"})

    assert "Paste the film" in response.text
    assert "for grain-accurate advice." in response.text
    assert "libsvtav1" in response.text


def test_gemini_is_only_asked_when_the_box_is_ticked() -> None:
    gemini, seen = gemini_transport()
    client = client_for(gemini=gemini, gemini_api_key="key")

    unticked = client.post("/advise", data={"source_text": FFPROBE_REPORT})
    assert seen == []
    assert "deterministic rules" in unticked.text

    ticked = client.post("/advise", data={"source_text": FFPROBE_REPORT, "use_gemini": "on"})
    assert len(seen) == 1
    assert "decided by Gemini" in ticked.text
    assert "Reviewed for this film." in ticked.text


def test_a_silent_gemini_leaves_the_baseline_standing() -> None:
    gemini, _ = gemini_transport(response=json_response({"error": "nope"}, 500))
    client = client_for(gemini=gemini, gemini_api_key="key")

    response = client.post("/advise", data={"source_text": FFPROBE_REPORT, "use_gemini": "on"})

    assert response.status_code == 200
    assert "deterministic rules" in response.text
    assert "Gemini was asked but did not contribute" in response.text
    assert "libsvtav1" in response.text


def test_the_preset_form_carries_the_answer_forward() -> None:
    tmdb, _ = tmdb_transport()
    client = client_for(tmdb=tmdb, tmdb_api_key="key")

    text = client.post(
        "/advise",
        data={
            "tmdb_id": "78",
            "source_text": FFPROBE_REPORT,
            "technical_source": TECHNICAL_PAGE,
            "grain_override": "heavy",
        },
    ).text

    assert 'name="source_text"' in text
    assert 'name="tmdb_id" value="78"' in text
    assert 'name="grain_override" value="heavy"' in text
    assert 'name="encoder" value="x265"' in text
    # The pasted page is up to a megabyte and is already cached against the IMDb id.
    assert 'name="technical_source"' not in text


def test_a_hostile_input_path_is_quoted_not_interpolated(client: TestClient) -> None:
    response = client.post("/advise", data={"input_path": "my movie.mkv; rm -rf /"})

    # shlex.join, so the whole thing stays one argument to ffmpeg.
    assert "&#39;my movie.mkv; rm -rf /&#39;" in response.text


def test_control_characters_never_reach_a_command(client: TestClient) -> None:
    response = client.post("/advise", data={"input_path": "a\nb\x01.mkv"})

    assert "-i ab.mkv" in response.text


def test_an_absurd_path_is_truncated(client: TestClient) -> None:
    response = client.post("/advise", data={"input_path": "x" * 400})

    assert "x" * 240 in response.text
    assert "x" * 241 not in response.text


# --- the preset download -----------------------------------------------------


def test_the_preset_comes_back_as_a_download() -> None:
    tmdb, _ = tmdb_transport()
    client = client_for(tmdb=tmdb, tmdb_api_key="key")

    response = client.post(
        "/preset", data={"tmdb_id": "78", "source_text": FFPROBE_REPORT, "encoder": "x265"}
    )

    assert response.status_code == 200
    assert response.headers["content-disposition"] == (
        'attachment; filename="graindamage-blade-runner-1982-2160p-x265.json"'
    )
    body = response.json()
    preset = body["PresetList"][0]
    assert preset["VideoEncoder"] == "x265_10bit"
    assert preset["PictureWidth"] == 3840
    assert "Blade Runner" in preset["PresetName"]


def test_the_preset_defaults_to_the_av1_plan(client: TestClient) -> None:
    response = client.post("/preset", data={"source_text": FFPROBE_REPORT})

    assert response.headers["content-disposition"] == (
        'attachment; filename="graindamage-output-svt-av1.json"'
    )
    assert response.json()["PresetList"][0]["VideoEncoder"] == "svt_av1_10bit"


def test_an_unknown_encoder_falls_back_rather_than_failing(client: TestClient) -> None:
    response = client.post("/preset", data={"encoder": "vp9"})

    assert response.status_code == 200
    assert response.json()["PresetList"][0]["VideoEncoder"] == "svt_av1_10bit"


def test_a_hostile_output_stem_cannot_shape_the_filename(client: TestClient) -> None:
    response = client.post("/preset", data={"output_stem": "../../etc/passwd"})

    disposition = response.headers["content-disposition"]
    assert disposition == 'attachment; filename="graindamage-..-..-etc-passwd-svt-av1.json"'
    assert "/" not in disposition
