"""Outbound integrations: TMDB, the IMDb ``/technical`` page, and Gemini.

Every provider raises :class:`ProviderError` for anything the user should see. The
distinction that matters at the route layer is *disabled* (no key configured — show
setup instructions) versus *failed* (upstream said no — show a retry), so those are
separate subclasses rather than a status code the templates would have to interpret.
"""

from __future__ import annotations


class ProviderError(Exception):
    """A failure with a message safe to render — never contains credentials."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ProviderDisabled(ProviderError):
    """The integration has no credentials, so it was never attempted."""


class ProviderUnavailable(ProviderError):
    """The integration was attempted and failed (network, status, or bad payload)."""


__all__ = ["ProviderDisabled", "ProviderError", "ProviderUnavailable"]
