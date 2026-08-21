"""IMDb ``/technical``: parsing three page layouts, and the optional fetcher.

The parser is the part that matters — a pasted page is the documented path and the
only one that works without configuration — so it gets a fixture per layout IMDb has
shipped. The fetcher tests are about the contract an operator has to implement.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

from app.providers import ProviderDisabled, ProviderUnavailable
from app.providers.imdb import ImdbTechnicalProvider, parse_technical
from tests.support import Handler, fixture, make_settings, mock_client, recording_handler, run

FETCHER_URL = "https://fetcher.example/get"
BROWSERLESS_URL = "http://192.168.1.2:3579/content"
BYPARR_URL = "http://192.168.1.2:8191/v1"
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


# --- the fetcher: browserless ------------------------------------------------


def sent_body(request: httpx2.Request) -> dict[str, Any]:
    payload: Any = json.loads(request.content)
    assert isinstance(payload, dict)
    return payload


def test_a_browserless_url_is_posted_to_with_the_target_in_the_body() -> None:
    handler, seen = recording_handler(httpx2.Response(200, text="<html>page</html>"))
    client = provider(handler, imdb_fetcher_url=BROWSERLESS_URL)

    assert run(client.fetch("tt0083658")) == "<html>page</html>"

    request = seen[0]
    # browserless serves HTML on POST-with-a-body only. There is no ?url= route on it
    # to configure, so recognising the endpoint is the app's job.
    assert request.method == "POST"
    assert str(request.url) == BROWSERLESS_URL
    assert request.headers["content-type"] == "application/json"
    assert sent_body(request)["url"] == TECHNICAL_URL


def test_browserless_waits_for_the_document_rather_than_for_the_ad_trackers() -> None:
    handler, seen = recording_handler(httpx2.Response(200, text="<html>page</html>"))
    client = provider(handler, imdb_fetcher_url=BROWSERLESS_URL, imdb_fetcher_timeout_seconds=20.0)

    run(client.fetch("tt0083658"))

    goto = sent_body(seen[0])["gotoOptions"]
    # __NEXT_DATA__ is in the initial HTML; networkidle would only wait for IMDb's ads.
    assert goto["waitUntil"] == "domcontentloaded"
    # 90% of the HTTP timeout, in ms: Chrome reports the failure rather than having the
    # connection cut from under it.
    assert goto["timeout"] == 18_000


def test_a_bare_host_and_port_means_the_content_endpoint() -> None:
    # How anyone writes their instance down: the port, and nothing after it.
    handler, seen = recording_handler(httpx2.Response(200, text="<html>page</html>"))

    run(provider(handler, imdb_fetcher_url="http://192.168.1.2:3579").fetch("tt0083658"))

    assert str(seen[0].url) == BROWSERLESS_URL


def test_browserless_takes_its_token_in_the_query_string_as_well() -> None:
    handler, seen = recording_handler(httpx2.Response(200, text="<html>page</html>"))
    client = provider(handler, imdb_fetcher_url=BROWSERLESS_URL, imdb_fetcher_token="s3cret")

    run(client.fetch("tt0083658"))

    # v1 reads only ?token=, v2 reads either, and a reverse proxy in front may want the
    # header — so both carry it.
    assert dict(seen[0].url.params)["token"] == "s3cret"
    assert seen[0].headers["authorization"] == "Bearer s3cret"


def test_the_unblock_endpoint_is_asked_for_content_and_unwrapped() -> None:
    handler, seen = recording_handler(
        httpx2.Response(200, json={"content": "<html>page</html>", "cookies": [], "ttl": 0})
    )
    client = provider(handler, imdb_fetcher_url="http://192.168.1.2:3579/unblock")

    assert run(client.fetch("tt0083658")) == "<html>page</html>"

    # /unblock drives the stealth browser and answers in JSON; it takes no gotoOptions.
    assert sent_body(seen[0]) == {"url": TECHNICAL_URL, "content": True}


def test_a_query_string_already_on_the_fetcher_url_survives() -> None:
    handler, seen = recording_handler(httpx2.Response(200, text="<html>page</html>"))

    run(provider(handler, imdb_fetcher_url=f"{FETCHER_URL}?api_key=k").fetch("tt0083658"))

    assert dict(seen[0].url.params) == {"api_key": "k", "url": TECHNICAL_URL}


@pytest.mark.parametrize(
    ("mode", "url", "method"),
    [
        ("auto", BROWSERLESS_URL, "POST"),
        ("auto", FETCHER_URL, "GET"),
        # When the endpoint name guesses wrong, the setting decides.
        ("query", BROWSERLESS_URL, "GET"),
        ("browserless", FETCHER_URL, "POST"),
        ("auto", BYPARR_URL, "POST"),
        ("query", BYPARR_URL, "GET"),
        ("flaresolverr", FETCHER_URL, "POST"),
    ],
)
def test_the_mode_setting_overrides_the_endpoint_name(mode: str, url: str, method: str) -> None:
    handler, seen = recording_handler(httpx2.Response(200, text="<html>page</html>"))
    client = provider(handler, imdb_fetcher_url=url, imdb_fetcher_mode=mode)

    run(client.fetch("tt0083658"))

    assert seen[0].method == method
    # The target travels in the query string one way and in the body the other.
    assert ("url" in dict(seen[0].url.params)) == (method == "GET")


def test_browserless_is_given_a_plausible_browser_identity() -> None:
    handler, seen = recording_handler(httpx2.Response(200, text="<html>page</html>"))

    run(provider(handler, imdb_fetcher_url=BROWSERLESS_URL).fetch("tt0083658"))

    body = sent_body(seen[0])
    # The Chrome browserless drives announces itself as HeadlessChrome, and IMDb's WAF
    # answers that with a challenge page before anything else happens. The app's own
    # User-Agent identifies the app to the operator; it is not what IMDb should see.
    user_agent = body["userAgent"]["userAgent"]
    assert user_agent.startswith("Mozilla/5.0")
    assert "Headless" not in user_agent
    assert body["setExtraHTTPHeaders"] == {"Accept-Language": "en-US,en;q=0.9"}
    # A wait that times out should still yield the page it managed to load.
    assert body["bestAttempt"] is True


# --- the fetcher: Byparr and FlareSolverr ------------------------------------


def byparr_answer(page: str = "<html>page</html>") -> httpx2.Response:
    """A Byparr ``/v1`` reply, shaped as its own LinkResponse model."""
    return httpx2.Response(
        200,
        json={
            "status": "ok",
            "message": "Success",
            "solution": {
                "url": TECHNICAL_URL,
                "status": 200,
                "cookies": [],
                "userAgent": "Mozilla/5.0",
                "headers": {},
                "response": page,
                "contentType": "text/html",
            },
            "startTimestamp": 1,
            "endTimestamp": 2,
            "version": "2.0.0",
        },
    )


def test_byparr_is_sent_the_flaresolverr_command() -> None:
    handler, seen = recording_handler(byparr_answer())
    client = provider(handler, imdb_fetcher_url=BYPARR_URL)

    # The page arrives one level down, in solution.response.
    assert run(client.fetch("tt0083658")) == "<html>page</html>"

    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == BYPARR_URL
    # maxTimeout covers solving the challenge as well as loading the page, so it gets
    # 90% of the HTTP timeout (in ms) and gives up before the connection does.
    assert sent_body(request) == {
        "cmd": "request.get",
        "url": TECHNICAL_URL,
        "maxTimeout": 18_000,
        # FlareSolverr keeps a browser per session name, so the token earned answering
        # IMDb's challenge survives into the next attempt. Byparr ignores both fields.
        "session": "graindamage-imdb",
        "session_ttl_minutes": 30,
    }


def test_a_byparr_failure_envelope_is_repeated_rather_than_parsed() -> None:
    # status "error" arrives with HTTP 200. Handing that to the parser would report a
    # film with no technical specifications, which is a different thing entirely.
    handler, _ = recording_handler(
        httpx2.Response(
            200,
            json={
                "status": "error",
                "message": "Invalid request",
                "solution": {"url": TECHNICAL_URL, "status": 500, "response": ""},
            },
        )
    )

    with pytest.raises(ProviderUnavailable, match="reported: Invalid request"):
        run(provider(handler, imdb_fetcher_url=BYPARR_URL).fetch("tt0083658"))


def test_the_services_own_explanation_survives_an_http_error() -> None:
    # Byparr answers 408 when a challenge outlasts maxTimeout, and explains itself in
    # the body — worth more to the reader than the status code alone.
    handler, _ = recording_handler(
        httpx2.Response(
            408, json={"detail": "Timed out while loading the page or solving the challenge"}
        )
    )

    with pytest.raises(ProviderUnavailable) as caught:
        run(provider(handler, imdb_fetcher_url=BYPARR_URL).fetch("tt0083658"))

    assert caught.value.message == "The IMDb fetcher returned HTTP 408."
    assert caught.value.detail is not None
    assert caught.value.detail.startswith("Timed out while loading")


def test_a_flaresolverr_page_is_preferred_over_a_top_level_key() -> None:
    # Some deployments sit behind a wrapper that adds its own envelope; the nested
    # solution is the real page either way.
    handler, _ = recording_handler(
        httpx2.Response(
            200,
            json={
                "status": "ok",
                "message": "Success",
                "data": "<html>wrapper</html>",
                "solution": {"url": TECHNICAL_URL, "status": 200, "response": "<html>page</html>"},
            },
        )
    )

    assert run(provider(handler, imdb_fetcher_url=BYPARR_URL).fetch("tt0083658")) == (
        "<html>page</html>"
    )


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


# --- the fetcher: a bot check instead of the page ----------------------------

# What IMDb serves a fresh browser: the challenge script sets a token and reloads, so
# the page itself never contains any specifications.
AWS_WAF_CHALLENGE = """<!DOCTYPE html><html lang="en"><head><title></title>
<script>window.awsWafCookieDomainList = ['imdb.com']; window.gokuProps = {"key":"AQID"};</script>
<script src="https://x.token.awswaf.com/x/challenge.js"></script></head>
<body><div id="challenge-container"></div>
<script>AwsWafIntegration.getToken().then(() => window.location.reload(true));</script>
</body></html>"""


def sequence_handler(*responses: httpx2.Response) -> tuple[Handler, list[httpx2.Request]]:
    """A handler that answers with each response in turn, repeating the last."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return responses[min(len(seen) - 1, len(responses) - 1)]

    return handler, seen


def test_a_bot_check_is_asked_again_and_the_second_answer_is_kept() -> None:
    # Answering the challenge earns the fetcher's browser a token, which a fetcher with
    # a session still holds next time — so the retry is the whole point.
    handler, seen = sequence_handler(
        byparr_answer(AWS_WAF_CHALLENGE),
        byparr_answer('<html><script id="__NEXT_DATA__">{}</script></html>'),
    )

    page = run(provider(handler, imdb_fetcher_url=BYPARR_URL).fetch("tt0083658"))

    assert "__NEXT_DATA__" in page
    assert len(seen) == 2


def test_a_bot_check_every_time_is_reported_rather_than_parsed() -> None:
    # Parsing it would report a film with no technical specifications, which is a
    # different thing entirely — and would be cached as if it were true.
    handler, seen = sequence_handler(byparr_answer(AWS_WAF_CHALLENGE))

    with pytest.raises(ProviderUnavailable) as caught:
        run(
            provider(handler, imdb_fetcher_url=BYPARR_URL, imdb_fetcher_attempts=2).fetch(
                "tt0083658"
            )
        )

    assert len(seen) == 2
    assert caught.value.message.endswith("bot check instead of the technical page.")
    assert caught.value.detail is not None
    assert caught.value.detail.startswith("2 attempts met IMDb's AWS WAF")


def test_one_attempt_is_one_request() -> None:
    handler, seen = sequence_handler(byparr_answer(AWS_WAF_CHALLENGE))

    with pytest.raises(ProviderUnavailable, match="bot check"):
        run(
            provider(handler, imdb_fetcher_url=BYPARR_URL, imdb_fetcher_attempts=1).fetch(
                "tt0083658"
            )
        )

    assert len(seen) == 1


def test_a_page_with_specs_in_it_is_never_called_a_bot_check() -> None:
    # A real page that happens to mention a challenge script is still the real page.
    page = f'<html><script id="__NEXT_DATA__">{{}}</script>{AWS_WAF_CHALLENGE}</html>'
    handler, seen = sequence_handler(byparr_answer(page))

    assert run(provider(handler, imdb_fetcher_url=BYPARR_URL).fetch("tt0083658")) == page
    assert len(seen) == 1


def test_a_cloudflare_interstitial_is_recognised_too() -> None:
    handler, _ = sequence_handler(
        httpx2.Response(200, text="<html><head><title>Just a moment...</title></head></html>")
    )

    with pytest.raises(ProviderUnavailable) as caught:
        run(provider(handler, imdb_fetcher_attempts=1).fetch("tt0083658"))

    assert caught.value.detail is not None
    assert caught.value.detail.startswith("1 attempt met a JavaScript bot check")
