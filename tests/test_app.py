from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.config import Settings
from app.main import create_app


@pytest.fixture
def client() -> TestClient:
    """A client with no integrations configured — the fresh-container case."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        tmdb_api_key=None,
        gemini_api_key=None,
        imdb_fetcher_url=None,
    )
    return TestClient(create_app(settings))


def test_healthz_reports_version_and_capabilities(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["capabilities"] == {
        "tmdb_search": False,
        "gemini_advice": False,
        "imdb_technical_fetcher": False,
    }


def test_index_renders_without_any_keys(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "graindamage" in response.text
    assert "set TMDB_API_KEY to enable" in response.text


def test_index_serves_static_assets(client: TestClient) -> None:
    for path in ("/static/css/app.css", "/static/js/htmx.min.js"):
        assert client.get(path).status_code == 200, path


def test_search_returns_placeholder_partial(client: TestClient) -> None:
    response = client.post("/search", data={"query": "Blade Runner"})

    assert response.status_code == 200
    assert "Blade Runner" in response.text
    # A partial, not a full page.
    assert "<!doctype html>" not in response.text.lower()


def test_capabilities_follow_configuration() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        tmdb_api_key="tmdb-key",
        gemini_api_key="gemini-key",
        imdb_fetcher_url="https://fetcher.example/get",
    )

    body = TestClient(create_app(settings)).get("/healthz").json()

    assert body["capabilities"] == {
        "tmdb_search": True,
        "gemini_advice": True,
        "imdb_technical_fetcher": True,
    }
