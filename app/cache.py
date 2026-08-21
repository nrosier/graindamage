"""A small async TTL cache — enough for one container, deliberately not more.

TMDB rate-limits and Gemini costs money — and the technical-specs look-up is a
Gemini call too — so repeated identical requests should not reach either of them.
There is no Redis here: a single process serves the whole app, so a dict with an
eviction pass is the right size.

Concurrent misses on the same key are coalesced behind a per-key lock, so ten
simultaneous searches for the same title make one upstream call, not ten.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass, field

MAX_ENTRIES = 512


@dataclass(slots=True)
class _Entry[T]:
    value: T
    expires_at: float


@dataclass(slots=True)
class TTLCache[T]:
    """Keys must be hashable; values are stored as-is and never copied."""

    ttl_seconds: float
    max_entries: int = MAX_ENTRIES
    _entries: dict[Hashable, _Entry[T]] = field(default_factory=dict)
    _locks: dict[Hashable, asyncio.Lock] = field(default_factory=dict)

    def get(self, key: Hashable) -> T | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            self._entries.pop(key, None)
            return None
        return entry.value

    def set(self, key: Hashable, value: T) -> None:
        if self.ttl_seconds <= 0:
            return
        self._evict_if_needed()
        self._entries[key] = _Entry(value=value, expires_at=time.monotonic() + self.ttl_seconds)

    async def get_or_set(self, key: Hashable, factory: Callable[[], Awaitable[T]]) -> T:
        """Return the cached value, or await ``factory`` once and cache the result.

        Failures are not cached: a transient upstream error should not be replayed
        for the whole TTL.
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            # A concurrent caller may have filled it while we waited.
            cached = self.get(key)
            if cached is not None:
                return cached
            value = await factory()
            self.set(key, value)
        self._locks.pop(key, None)
        return value

    def clear(self) -> None:
        self._entries.clear()
        self._locks.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def _evict_if_needed(self) -> None:
        if len(self._entries) < self.max_entries:
            return
        now = time.monotonic()
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            del self._entries[key]
        # Still full of live entries: drop the oldest insertions (dicts keep order).
        while len(self._entries) >= self.max_entries:
            self._entries.pop(next(iter(self._entries)))
