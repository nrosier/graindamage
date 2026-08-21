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
    user_agent: str = "graindamage/0.8 (+https://github.com/nrosier/graindamage)"

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
    # IMDb is behind a WAF that answers automated requests with a JavaScript bot
    # check, so this app does not fetch the page at all. The rows come from Gemini
    # (which is asked for the film's /technical rows) or from a paste, and the link
    # to the real page is always on screen so a paste is one copy away.
    #
    # Grounding lets that lookup search the web instead of answering from memory,
    # which is the difference between IMDb's rows and something that looks like
    # them. It is dropped automatically if the model turns out not to support it.
    gemini_web_grounding: bool = True

    # --- caching ----------------------------------------------------------
    cache_ttl_seconds: int = Field(default=3600, ge=0)

    @property
    def tmdb_enabled(self) -> bool:
        return bool(self.tmdb_api_key)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    def capabilities(self) -> dict[str, bool]:
        """Which optional integrations are wired up, for /healthz and the UI."""
        return {
            "tmdb_search": self.tmdb_enabled,
            "gemini_advice": self.gemini_enabled,
            # The technical rows are looked up through Gemini, so they ride on its key.
            "imdb_technical_lookup": self.gemini_enabled,
            "baseline_rules": True,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
