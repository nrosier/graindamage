"""Runtime configuration, read from the environment (or a local .env file).

Every integration is optional: the app must start and serve a useful page with no
keys at all, so that a fresh container never dies on missing configuration. What is
missing is reported through :meth:`Settings.capabilities` instead.

The deterministic rules engine has no configuration and no dependencies, which is
why ``baseline_rules`` is always reported as available — an unkeyed container still
produces real encoding settings.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- app ---------------------------------------------------------------
    debug: bool = False
    host: str = "0.0.0.0"  # the container listens on all interfaces by design
    port: int = 8080
    user_agent: str = "graindamage/0.7 (+https://github.com/nrosier/graindamage)"

    # --- TMDB: title search and metadata ----------------------------------
    tmdb_api_key: str | None = None
    tmdb_language: str = "en-US"
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    tmdb_image_base_url: str = "https://image.tmdb.org/t/p"
    tmdb_poster_size: str = "w185"
    tmdb_timeout_seconds: float = 10.0
    tmdb_max_results: int = Field(default=8, ge=1, le=20)

    # --- Google Gemini: encoding advice -----------------------------------
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.7-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout_seconds: float = 45.0
    gemini_max_output_tokens: int = Field(default=2048, ge=256)
    gemini_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- IMDb technical specs ---------------------------------------------
    # Pasting the /technical page source always works. Optionally, a fetcher you
    # control (proxy, browserless, cookie-bearing service) can retrieve it instead;
    # it receives the IMDb URL and returns the page source.
    imdb_fetcher_url: str | None = None
    imdb_fetcher_token: str | None = None
    imdb_fetcher_timeout_seconds: float = 20.0

    # --- caching ----------------------------------------------------------
    cache_ttl_seconds: int = Field(default=3600, ge=0)

    @property
    def tmdb_enabled(self) -> bool:
        return bool(self.tmdb_api_key)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def imdb_fetcher_enabled(self) -> bool:
        return bool(self.imdb_fetcher_url)

    def capabilities(self) -> dict[str, bool]:
        """Which optional integrations are wired up, for /healthz and the UI."""
        return {
            "tmdb_search": self.tmdb_enabled,
            "gemini_advice": self.gemini_enabled,
            "imdb_technical_fetcher": self.imdb_fetcher_enabled,
            "baseline_rules": True,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
