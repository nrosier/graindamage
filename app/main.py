"""FastAPI application factory: settings in, wired app out.

The providers are built once per app rather than per request, because their TTL
caches live inside them — a per-request client would cache nothing. They hold no
connections between calls, so there is nothing to close.

Everything that answers an HTTP request lives in :mod:`app.routes`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.config import Settings, get_settings
from app.providers.gemini import GeminiClient
from app.providers.tmdb import TmdbClient
from app.routes import Services, create_router

BASE_DIR = Path(__file__).resolve().parent


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="graindamage",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
    )

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.globals["version"] = __version__

    services = Services(
        settings=settings,
        templates=templates,
        tmdb=TmdbClient(settings),
        gemini=GeminiClient(settings),
    )

    app.state.settings = settings
    app.state.templates = templates
    app.state.services = services

    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.include_router(create_router(services))

    return app


app = create_app()
