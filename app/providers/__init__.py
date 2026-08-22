"""Outbound integrations: TMDB and Gemini.

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


class ProviderRejected(ProviderUnavailable):
    """The request itself was refused — a bad key, or a feature this model lacks.

    Separate because retrying it unchanged is pointless, while retrying it *changed*
    (without an optional feature the model turned out not to support) is exactly right.
    """


class ProviderRateLimited(ProviderUnavailable):
    """The request was refused for quota, not for content.

    Separate from :class:`ProviderRejected` because the two are retryable in opposite
    ways: a rejection means *this request* is wrong and will stay wrong, while a rate
    limit means this request was fine and something else was over its allowance. That
    matters when the allowance belongs to one optional part of the call — a metered
    tool, say — because the same request without that part draws on a different one.
    """


__all__ = [
    "ProviderDisabled",
    "ProviderError",
    "ProviderRateLimited",
    "ProviderRejected",
    "ProviderUnavailable",
]
