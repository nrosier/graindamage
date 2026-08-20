"""FastAPI application: routes, templates and static assets.

Milestone 1 is the skeleton — search and advice land in later milestones. The
placeholder partial in :func:`search` is deliberate: it keeps the HTMX wiring
end-to-end testable before a provider exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.config import Settings, get_settings

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

    app.state.settings = settings
    app.state.templates = templates
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        """Liveness probe, also used by the container HEALTHCHECK."""
        return {
            "status": "ok",
            "version": __version__,
            "capabilities": settings.capabilities(),
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"capabilities": settings.capabilities()},
        )

    @app.post("/search", response_class=HTMLResponse)
    async def search(
        request: Request,
        query: Annotated[str, Form()] = "",
    ) -> HTMLResponse:
        # TODO(milestone 2): replace with the TMDB provider.
        return templates.TemplateResponse(
            request,
            "partials/search_results.html",
            {
                "query": query.strip(),
                "tmdb_enabled": settings.tmdb_enabled,
            },
        )

    return app


app = create_app()
