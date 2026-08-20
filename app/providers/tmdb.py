"""TMDB title search and detail lookup.

IMDb has no public API, so search runs against TMDB and each pick resolves its
``imdb_id`` through ``external_ids`` — which is what makes the IMDb ``/technical``
URL reachable later. Both v3 API keys and v4 bearer tokens are accepted because
TMDB's own dashboard hands out either one.

Results are cached: the same query typed twice, or a pick re-submitted when the
step-2 form is posted again, must not cost another upstream request.
"""

from __future__ import annotations

from typing import Any

import httpx2

from app.cache import TTLCache
from app.config import Settings
from app.models import Movie, MovieHit
from app.providers import ProviderDisabled, ProviderUnavailable


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _year(release_date: object) -> int | None:
    if isinstance(release_date, str) and len(release_date) >= 4 and release_date[:4].isdigit():
        return int(release_date[:4])
    return None


def _clean_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


class TmdbClient:
    """A thin, cached wrapper over the two TMDB endpoints we need."""

    def __init__(self, settings: Settings, *, client: httpx2.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client
        self._search_cache: TTLCache[list[MovieHit]] = TTLCache(
            ttl_seconds=float(settings.cache_ttl_seconds)
        )
        self._movie_cache: TTLCache[Movie] = TTLCache(ttl_seconds=float(settings.cache_ttl_seconds))

    @property
    def enabled(self) -> bool:
        return self._settings.tmdb_enabled

    async def search(self, query: str) -> list[MovieHit]:
        query = query.strip()
        if not self.enabled:
            raise ProviderDisabled("TMDB search is not configured. Set TMDB_API_KEY.")
        if not query:
            return []

        key = (query.casefold(), self._settings.tmdb_language)
        return await self._search_cache.get_or_set(key, lambda: self._search_uncached(query))

    async def get_movie(self, tmdb_id: int) -> Movie:
        if not self.enabled:
            raise ProviderDisabled("TMDB lookup is not configured. Set TMDB_API_KEY.")
        key = (tmdb_id, self._settings.tmdb_language)
        return await self._movie_cache.get_or_set(key, lambda: self._movie_uncached(tmdb_id))

    # --- internals ---------------------------------------------------------

    async def _search_uncached(self, query: str) -> list[MovieHit]:
        payload = await self._get(
            "/search/movie",
            {
                "query": query,
                "language": self._settings.tmdb_language,
                "include_adult": "false",
            },
        )
        results = _as_list(payload.get("results"))[: self._settings.tmdb_max_results]
        return [hit for hit in (self._to_hit(_as_dict(row)) for row in results) if hit]

    async def _movie_uncached(self, tmdb_id: int) -> Movie:
        payload = await self._get(
            f"/movie/{tmdb_id}",
            {
                "language": self._settings.tmdb_language,
                "append_to_response": "external_ids,credits",
            },
        )
        hit = self._to_hit(payload)
        if hit is None:
            raise ProviderUnavailable(f"TMDB returned no usable record for id {tmdb_id}.")

        credits = _as_dict(payload.get("credits"))
        directors = [
            name
            for member in _as_list(credits.get("crew"))
            if _as_dict(member).get("job") == "Director"
            and (name := _clean_str(_as_dict(member).get("name")))
        ]
        runtime = payload.get("runtime")

        return Movie(
            **hit.model_dump(),
            imdb_id=_clean_str(_as_dict(payload.get("external_ids")).get("imdb_id"))
            or _clean_str(payload.get("imdb_id")),
            runtime_minutes=runtime if isinstance(runtime, int) and runtime > 0 else None,
            release_date=_clean_str(payload.get("release_date")),
            genres=[
                name
                for genre in _as_list(payload.get("genres"))
                if (name := _clean_str(_as_dict(genre).get("name")))
            ],
            directors=directors,
            countries=[
                name
                for country in _as_list(payload.get("production_countries"))
                if (name := _clean_str(_as_dict(country).get("iso_3166_1")))
            ],
            original_language=_clean_str(payload.get("original_language")),
        )

    def _to_hit(self, row: dict[str, Any]) -> MovieHit | None:
        tmdb_id = row.get("id")
        title = _clean_str(row.get("title")) or _clean_str(row.get("original_title"))
        if not isinstance(tmdb_id, int) or not title:
            return None

        poster_path = _clean_str(row.get("poster_path"))
        vote = row.get("vote_average")
        return MovieHit(
            tmdb_id=tmdb_id,
            title=title,
            original_title=_clean_str(row.get("original_title")),
            year=_year(row.get("release_date")),
            overview=_clean_str(row.get("overview")),
            poster_url=(
                f"{self._settings.tmdb_image_base_url}/{self._settings.tmdb_poster_size}{poster_path}"
                if poster_path
                else None
            ),
            vote_average=float(vote) if isinstance(vote, int | float) and vote else None,
        )

    async def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        key = self._settings.tmdb_api_key or ""
        headers = {"Accept": "application/json", "User-Agent": self._settings.user_agent}

        # v4 tokens are JWTs and go in the header; v3 keys are a query parameter.
        if key.startswith("ey") and key.count(".") == 2:
            headers["Authorization"] = f"Bearer {key}"
        else:
            params = {**params, "api_key": key}

        url = f"{self._settings.tmdb_base_url.rstrip('/')}{path}"
        try:
            if self._client is not None:
                response = await self._client.get(url, params=params, headers=headers)
            else:
                async with httpx2.AsyncClient(
                    timeout=self._settings.tmdb_timeout_seconds
                ) as client:
                    response = await client.get(url, params=params, headers=headers)
        except httpx2.HTTPError as exc:
            raise ProviderUnavailable("Could not reach TMDB.", detail=type(exc).__name__) from exc

        if response.status_code == 401:
            raise ProviderUnavailable("TMDB rejected the API key (401).")
        if response.status_code == 404:
            raise ProviderUnavailable("TMDB has no record with that id (404).")
        if response.status_code == 429:
            raise ProviderUnavailable("TMDB rate limit reached — try again shortly (429).")
        if response.status_code >= 400:
            raise ProviderUnavailable(f"TMDB returned HTTP {response.status_code}.")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderUnavailable("TMDB returned a response that was not JSON.") from exc
        if not isinstance(payload, dict):
            raise ProviderUnavailable("TMDB returned an unexpected payload shape.")
        return payload
