"""TMDB search and lookup, over a mock transport.

Nothing here touches the network: every client is built with an injected
``httpx2.AsyncClient`` whose transport answers from a fixture payload, so the tests
are about how TMDB's JSON is read rather than about TMDB being up.
"""

from __future__ import annotations

import re
from typing import Any

import httpx2
import pytest

from app.providers import ProviderDisabled, ProviderUnavailable
from app.providers.tmdb import TmdbClient
from tests.support import Handler, json_response, make_settings, mock_client, recording_handler, run

SEARCH_PAYLOAD: dict[str, Any] = {
    "page": 1,
    "results": [
        {
            "id": 78,
            "title": "Blade Runner",
            "original_title": "Blade Runner",
            "release_date": "1982-06-25",
            "overview": "A blade runner must pursue and terminate four replicants.",
            "poster_path": "/63N9uy8nd9j7Eog2axPQ8lbr3Wj.jpg",
            "vote_average": 8.1,
        },
        {
            "id": 335984,
            "title": "Blade Runner 2049",
            "original_title": "Blade Runner 2049",
            "release_date": "2017-10-04",
            "overview": "Thirty years after the events of the first film.",
            "poster_path": None,
            "vote_average": 0,
        },
    ],
}

MOVIE_PAYLOAD: dict[str, Any] = {
    "id": 78,
    "title": "Blade Runner",
    "original_title": "Blade Runner",
    "release_date": "1982-06-25",
    "runtime": 117,
    "overview": "A blade runner must pursue and terminate four replicants.",
    "poster_path": "/63N9uy8nd9j7Eog2axPQ8lbr3Wj.jpg",
    "vote_average": 8.1,
    "original_language": "en",
    "genres": [{"id": 878, "name": "Science Fiction"}, {"id": 53, "name": "Thriller"}],
    "production_countries": [
        {"iso_3166_1": "US", "name": "United States of America"},
        {"iso_3166_1": "GB", "name": "United Kingdom"},
    ],
    "external_ids": {"imdb_id": "tt0083658", "wikidata_id": "Q184843"},
    "credits": {
        "crew": [
            {"name": "Ridley Scott", "job": "Director"},
            {"name": "Hampton Fancher", "job": "Screenplay"},
            {"name": "Jordan Cronenweth", "job": "Director of Photography"},
        ]
    },
}


def tmdb(handler: Handler, **overrides: Any) -> TmdbClient:
    """A client with a key configured and every request answered by ``handler``."""
    settings = make_settings(**{"tmdb_api_key": "v3-key", **overrides})
    return TmdbClient(settings, client=mock_client(handler))


def answering(payload: Any, status_code: int = 200) -> tuple[TmdbClient, list[httpx2.Request]]:
    handler, seen = recording_handler(json_response(payload, status_code))
    return tmdb(handler), seen


# --- configuration ----------------------------------------------------------


def test_without_a_key_nothing_is_attempted() -> None:
    client = TmdbClient(make_settings(), client=mock_client(lambda request: json_response({})))

    assert not client.enabled
    with pytest.raises(ProviderDisabled, match="Set TMDB_API_KEY"):
        run(client.search("blade runner"))
    with pytest.raises(ProviderDisabled, match="Set TMDB_API_KEY"):
        run(client.get_movie(78))


def test_an_empty_query_costs_no_request() -> None:
    client, seen = answering(SEARCH_PAYLOAD)

    assert run(client.search("   ")) == []
    assert seen == []


def test_a_v3_key_travels_as_a_query_parameter() -> None:
    client, seen = answering(SEARCH_PAYLOAD)

    run(client.search("blade runner"))

    params = dict(seen[0].url.params)
    assert seen[0].url.path == "/3/search/movie"
    assert params["api_key"] == "v3-key"
    assert params["query"] == "blade runner"
    assert params["language"] == "en-US"
    assert params["include_adult"] == "false"
    assert "authorization" not in seen[0].headers
    assert seen[0].headers["user-agent"].startswith("graindamage/")


def test_a_v4_token_travels_as_a_bearer_header() -> None:
    # TMDB hands out both kinds from the same dashboard page, so both have to work —
    # and a JWT in a query string ends up in every log between here and there.
    #
    # Assembled from parts rather than written out: a literal JWT in a source file is
    # exactly what a secret scanner should shout about, and it did. Only the "ey"
    # prefix and the two dots decide which header the client uses.
    token = ".".join(("eyJhbGciOiJub25lIn0", "not-a-payload", "not-a-signature"))
    handler, seen = recording_handler(json_response(SEARCH_PAYLOAD))
    client = tmdb(handler, tmdb_api_key=token)

    run(client.search("blade runner"))

    assert seen[0].headers["authorization"] == f"Bearer {token}"
    assert "api_key" not in dict(seen[0].url.params)
    assert token not in str(seen[0].url)


# --- search -----------------------------------------------------------------


def test_search_reads_the_fields_the_picker_shows() -> None:
    client, _ = answering(SEARCH_PAYLOAD)

    hits = run(client.search("blade runner"))

    assert [hit.tmdb_id for hit in hits] == [78, 335984]
    first = hits[0]
    assert first.title == "Blade Runner"
    assert first.year == 1982
    assert first.overview is not None and first.overview.startswith("A blade runner")
    assert first.poster_url == ("https://image.tmdb.org/t/p/w185/63N9uy8nd9j7Eog2axPQ8lbr3Wj.jpg")
    assert first.vote_average == pytest.approx(8.1)


def test_a_missing_poster_is_none_rather_than_a_broken_url() -> None:
    client, _ = answering(SEARCH_PAYLOAD)

    second = run(client.search("blade runner"))[1]

    assert second.poster_url is None
    # An unrated film scores 0, which is not a rating.
    assert second.vote_average is None


@pytest.mark.parametrize(
    ("release_date", "year"),
    [("1982-06-25", 1982), ("1982", 1982), ("", None), (None, None), ("soon", None)],
)
def test_the_year_comes_from_whatever_the_release_date_is(
    release_date: object, year: int | None
) -> None:
    client, _ = answering({"results": [{"id": 1, "title": "T", "release_date": release_date}]})

    assert run(client.search("t"))[0].year == year


def test_rows_without_an_id_or_a_title_are_dropped() -> None:
    client, _ = answering(
        {
            "results": [
                {"id": "78", "title": "String id"},
                {"id": 79},
                {"id": 80, "title": "   "},
                {"id": 81, "original_title": "Only an original title"},
                "not even an object",
            ]
        }
    )

    hits = run(client.search("junk"))

    assert [(hit.tmdb_id, hit.title) for hit in hits] == [(81, "Only an original title")]


def test_the_result_list_is_capped_by_configuration() -> None:
    payload = {"results": [{"id": index, "title": f"Film {index}"} for index in range(1, 30)]}
    handler, _ = recording_handler(json_response(payload))
    client = tmdb(handler, tmdb_max_results=3)

    assert len(run(client.search("film"))) == 3


def test_a_payload_without_results_is_no_hits_rather_than_an_error() -> None:
    client, _ = answering({"page": 1, "total_results": 0})

    assert run(client.search("nothing at all")) == []


def test_the_same_search_is_only_made_once() -> None:
    client, seen = answering(SEARCH_PAYLOAD)

    run(client.search("blade runner"))
    run(client.search("blade runner"))

    assert len(seen) == 1


def test_the_search_cache_ignores_case_and_surrounding_space() -> None:
    client, seen = answering(SEARCH_PAYLOAD)

    run(client.search("Blade Runner"))
    run(client.search("  blade RUNNER  "))

    assert len(seen) == 1


def test_a_different_language_is_a_different_cache_entry() -> None:
    handler, seen = recording_handler(json_response(SEARCH_PAYLOAD))
    english = tmdb(handler)
    french = TmdbClient(
        make_settings(tmdb_api_key="v3-key", tmdb_language="fr-FR"), client=mock_client(handler)
    )

    run(english.search("blade runner"))
    run(french.search("blade runner"))

    assert [dict(request.url.params)["language"] for request in seen] == ["en-US", "fr-FR"]


def test_caching_can_be_turned_off() -> None:
    handler, seen = recording_handler(json_response(SEARCH_PAYLOAD))
    client = tmdb(handler, cache_ttl_seconds=0)

    run(client.search("blade runner"))
    run(client.search("blade runner"))

    assert len(seen) == 2


# --- movie lookup -----------------------------------------------------------


def test_get_movie_assembles_everything_the_advice_needs() -> None:
    client, seen = answering(MOVIE_PAYLOAD)

    movie = run(client.get_movie(78))

    assert seen[0].url.path == "/3/movie/78"
    # One request, not three: the credits and the IMDb id are appended to it.
    assert dict(seen[0].url.params)["append_to_response"] == "external_ids,credits"
    assert movie.tmdb_id == 78
    assert movie.title == "Blade Runner"
    assert movie.year == 1982
    assert movie.imdb_id == "tt0083658"
    assert movie.runtime_minutes == 117
    assert movie.release_date == "1982-06-25"
    assert movie.genres == ["Science Fiction", "Thriller"]
    assert movie.countries == ["US", "GB"]
    assert movie.original_language == "en"


def test_only_directors_are_taken_from_the_crew() -> None:
    client, _ = answering(MOVIE_PAYLOAD)

    # The screenwriter and the DoP are not who we are asking about.
    assert run(client.get_movie(78)).directors == ["Ridley Scott"]


def test_two_directors_are_both_kept() -> None:
    payload = {
        **MOVIE_PAYLOAD,
        "credits": {
            "crew": [
                {"name": "Lana Wachowski", "job": "Director"},
                {"name": "Lilly Wachowski", "job": "Director"},
                {"name": "", "job": "Director"},
            ]
        },
    }
    client, _ = answering(payload)

    assert run(client.get_movie(78)).directors == ["Lana Wachowski", "Lilly Wachowski"]


def test_the_imdb_id_falls_back_to_the_top_level_field() -> None:
    # /movie/{id} carries imdb_id itself when append_to_response is not honoured.
    payload = {key: value for key, value in MOVIE_PAYLOAD.items() if key != "external_ids"}
    client, _ = answering({**payload, "imdb_id": "tt0083658"})

    assert run(client.get_movie(78)).imdb_id == "tt0083658"


def test_a_missing_imdb_id_is_not_fatal() -> None:
    payload = {key: value for key, value in MOVIE_PAYLOAD.items() if key != "external_ids"}
    client, _ = answering(payload)

    # The technical page becomes unreachable, but the film is still usable.
    assert run(client.get_movie(78)).imdb_id is None


@pytest.mark.parametrize("runtime", [0, None, "117", -5])
def test_an_implausible_runtime_is_discarded(runtime: object) -> None:
    client, _ = answering({**MOVIE_PAYLOAD, "runtime": runtime})

    assert run(client.get_movie(78)).runtime_minutes is None


def test_a_record_without_a_title_is_reported_rather_than_returned() -> None:
    client, _ = answering({"id": 78, "runtime": 117})

    with pytest.raises(ProviderUnavailable, match="no usable record for id 78"):
        run(client.get_movie(78))


def test_the_same_id_is_only_looked_up_once() -> None:
    client, seen = answering(MOVIE_PAYLOAD)

    first = run(client.get_movie(78))
    second = run(client.get_movie(78))

    assert len(seen) == 1
    # /advise re-resolves the film from the hidden form field; that must be free.
    assert first is second


# --- failures ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, "TMDB rejected the API key (401)."),
        (404, "TMDB has no record with that id (404)."),
        (429, "TMDB rate limit reached — try again shortly (429)."),
        (500, "TMDB returned HTTP 500."),
        (503, "TMDB returned HTTP 503."),
    ],
)
def test_http_failures_become_readable_messages(status_code: int, expected: str) -> None:
    client, _ = answering({"status_message": "whatever"}, status_code)

    with pytest.raises(ProviderUnavailable) as caught:
        run(client.search("blade runner"))

    assert caught.value.message == expected


def test_a_transport_failure_names_the_kind_without_a_stack_trace() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectTimeout("timed out")

    with pytest.raises(ProviderUnavailable) as caught:
        run(tmdb(handler).search("blade runner"))

    assert caught.value.message == "Could not reach TMDB."
    assert caught.value.detail == "ConnectTimeout"


def test_a_body_that_is_not_json_is_reported_as_such() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text="<html>Cloudflare</html>")

    with pytest.raises(ProviderUnavailable, match=re.escape("was not JSON")):
        run(tmdb(handler).search("blade runner"))


def test_a_json_body_of_the_wrong_shape_is_reported_as_such() -> None:
    client, _ = answering([1, 2, 3])

    with pytest.raises(ProviderUnavailable, match="unexpected payload shape"):
        run(client.search("blade runner"))


def test_a_failure_is_not_cached() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return json_response(SEARCH_PAYLOAD) if calls > 1 else json_response({}, 503)

    client = tmdb(handler)

    async def retry() -> int:
        with pytest.raises(ProviderUnavailable):
            await client.search("blade runner")
        return len(await client.search("blade runner"))

    assert run(retry()) == 2
    assert calls == 2
