"""Builders shared by the test modules.

Two things are centralised here because getting them wrong quietly weakens every test
that uses them:

* **No network, ever.** Providers take an injected ``httpx2.AsyncClient``, so
  :func:`mock_client` is the only way a test reaches an HTTP verb.
* **No ambient environment.** :func:`make_settings` passes ``_env_file=None`` so a
  developer's real ``.env`` cannot turn an integration on mid-test.

There is no pytest-asyncio in the dependency set, so async code is driven through
:func:`run`, which is one ``asyncio.run`` per call — fine, because nothing here shares
an event loop between awaits.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import httpx2
from fastapi import FastAPI

from app.config import Settings
from app.main import create_app
from app.models import (
    AudioTrack,
    EncodeRequest,
    Movie,
    SourceMedia,
    SubtitleTrack,
    TechnicalSpecs,
    VideoTrack,
)
from app.providers.gemini import GeminiClient
from app.providers.tmdb import TmdbClient

FIXTURES = Path(__file__).resolve().parent / "fixtures"

Handler = Callable[[httpx2.Request], httpx2.Response]


def fixture(name: str) -> str:
    """The text of ``tests/fixtures/<name>``."""
    return (FIXTURES / name).read_text(encoding="utf-8")


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Drive one coroutine to completion."""
    return asyncio.run(coroutine)


def make_settings(**overrides: Any) -> Settings:
    """Settings with nothing configured unless the test asks for it."""
    defaults: dict[str, Any] = {
        "tmdb_api_key": None,
        "gemini_api_key": None,
        "cache_ttl_seconds": 3600,
    }
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        **{**defaults, **overrides},
    )


def mock_client(handler: Handler) -> httpx2.AsyncClient:
    """An ``AsyncClient`` whose every request is answered by ``handler``."""
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def json_response(payload: Any, status_code: int = 200) -> httpx2.Response:
    return httpx2.Response(status_code, json=payload)


def recording_handler(response: httpx2.Response) -> tuple[Handler, list[httpx2.Request]]:
    """A handler that always returns ``response``, plus the list it records into."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return response

    return handler, seen


# --- domain builders --------------------------------------------------------


def video_track(**overrides: Any) -> VideoTrack:
    """A 1080p 23.976 fps 20 Mb/s track: 0.402 bits per pixel, i.e. disc-grade."""
    defaults: dict[str, Any] = {
        "index": 0,
        "codec": "hevc",
        "width": 1920,
        "height": 1080,
        "frame_rate": 24000 / 1001,
        "bit_depth": 10,
        "chroma_subsampling": "4:2:0",
        "pix_fmt": "yuv420p10le",
        "scan_type": "progressive",
        "bitrate_bps": 20_000_000,
        "color_primaries": "bt709",
        "color_transfer": "bt709",
        "color_matrix": "bt709",
        "color_range": "tv",
    }
    return VideoTrack(**{**defaults, **overrides})


def media(**overrides: Any) -> SourceMedia:
    """A 26 GB 1080p Matroska rip. Pass ``video=None`` for a file with no video track."""
    defaults: dict[str, Any] = {
        "container": "matroska,webm",
        "duration_seconds": 7020.0,
        "size_bytes": 26_306_674_688,
        "overall_bitrate_bps": 30_000_000,
        "video": video_track(),
        "audio": [AudioTrack(index=1, codec="truehd", channels=6, language="eng", default=True)],
        "subtitles": [SubtitleTrack(index=2, codec="subrip", language="eng")],
    }
    return SourceMedia(**{**defaults, **overrides})


def movie(**overrides: Any) -> Movie:
    defaults: dict[str, Any] = {
        "tmdb_id": 78,
        "title": "Blade Runner",
        "year": 1982,
        "imdb_id": "tt0083658",
        "directors": ["Ridley Scott"],
        "genres": ["Science Fiction"],
    }
    return Movie(**{**defaults, **overrides})


def specs(**overrides: Any) -> TechnicalSpecs:
    defaults: dict[str, Any] = {
        "negative_formats": ["35 mm"],
        "cinematographic_processes": ["Panavision (anamorphic)", "Super 35"],
        "aspect_ratios": ["2.39 : 1"],
    }
    return TechnicalSpecs(**{**defaults, **overrides})


def request_for(**overrides: Any) -> EncodeRequest:
    defaults: dict[str, Any] = {
        "movie": movie(),
        "specs": specs(),
        "source": media(),
        "input_path": "input.mkv",
        "output_stem": "blade-runner-1982-1080p",
    }
    return EncodeRequest(**{**defaults, **overrides})


# --- app builder ------------------------------------------------------------


def make_app(
    settings: Settings,
    *,
    tmdb: httpx2.AsyncClient | None = None,
    gemini: httpx2.AsyncClient | None = None,
) -> FastAPI:
    """The real app, with mock transports pushed into whichever providers a test uses.

    The router reads ``services.<provider>`` per request, so replacing them after
    construction is enough — and it keeps ``create_app`` itself under test.
    """
    app = create_app(settings)
    services = app.state.services
    if tmdb is not None:
        services.tmdb = TmdbClient(settings, client=tmdb)
    if gemini is not None:
        services.gemini = GeminiClient(settings, client=gemini)
    return app
