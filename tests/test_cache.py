"""The TTL cache every provider sits behind.

The interesting behaviour is not storage, it is the promises the providers rely on:
one upstream call per key even under concurrency, and no caching of failures.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from app.cache import MAX_ENTRIES, TTLCache
from tests.support import run


def test_a_missing_key_reads_as_none() -> None:
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)

    assert cache.get("absent") is None
    assert len(cache) == 0


def test_a_stored_value_comes_back_unchanged() -> None:
    cache: TTLCache[list[int]] = TTLCache(ttl_seconds=60)
    value = [1, 2, 3]

    cache.set("key", value)

    # Values are stored, not copied: the providers cache immutable results.
    assert cache.get("key") is value


def test_an_empty_result_is_still_a_cached_result() -> None:
    # A search with no hits is a legitimate answer and must not be re-fetched, so
    # the miss test has to be ``is None`` rather than falsiness.
    cache: TTLCache[list[str]] = TTLCache(ttl_seconds=60)

    cache.set("nothing found", [])

    assert cache.get("nothing found") == []
    assert len(cache) == 1


def test_an_expired_entry_is_dropped_on_read() -> None:
    cache: TTLCache[str] = TTLCache(ttl_seconds=0.01)
    cache.set("key", "value")

    time.sleep(0.02)

    assert cache.get("key") is None
    # Not merely hidden — the entry is gone, so the dict cannot grow without bound.
    assert len(cache) == 0


def test_a_zero_ttl_disables_caching_entirely() -> None:
    # CACHE_TTL_SECONDS=0 is the documented way to turn caching off in development.
    cache: TTLCache[str] = TTLCache(ttl_seconds=0)

    cache.set("key", "value")

    assert cache.get("key") is None
    assert len(cache) == 0


def test_zero_ttl_still_calls_the_factory_every_time() -> None:
    cache: TTLCache[str] = TTLCache(ttl_seconds=0)
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        return "value"

    assert run(cache.get_or_set("key", factory)) == "value"
    assert run(cache.get_or_set("key", factory)) == "value"
    assert calls == 2


def test_get_or_set_calls_the_factory_once_per_key() -> None:
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    calls: list[str] = []

    async def build(name: str) -> str:
        calls.append(name)
        return name.upper()

    assert run(cache.get_or_set("a", lambda: build("a"))) == "A"
    assert run(cache.get_or_set("a", lambda: build("a"))) == "A"
    assert run(cache.get_or_set("b", lambda: build("b"))) == "B"

    assert calls == ["a", "b"]


def test_a_failing_factory_is_not_cached() -> None:
    # A timeout must not be replayed for the whole hour.
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("upstream said no")
        return "value"

    async def twice() -> str:
        # Both calls share one event loop on purpose: the per-key lock outlives a
        # failure, and a lock is bound to the loop that first awaited it.
        with contextlib.suppress(RuntimeError):
            await cache.get_or_set("key", flaky)
        return await cache.get_or_set("key", flaky)

    assert run(twice()) == "value"
    assert attempts == 2


def test_concurrent_misses_make_one_upstream_call() -> None:
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    calls = 0

    async def slow() -> str:
        nonlocal calls
        calls += 1
        # Yield, so every other waiter is queued behind the lock before this returns.
        await asyncio.sleep(0)
        return "value"

    async def stampede() -> list[str]:
        return await asyncio.gather(*(cache.get_or_set("key", slow) for _ in range(10)))

    assert run(stampede()) == ["value"] * 10
    assert calls == 1


def test_the_per_key_lock_is_released_after_use() -> None:
    # Locks are keyed like entries, so leaking them would leak memory per query.
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)

    async def factory() -> str:
        return "value"

    run(cache.get_or_set("key", factory))

    assert cache._locks == {}


def test_keys_of_different_shapes_do_not_collide() -> None:
    # TmdbClient keys on (query, language) and GeminiClient on a hash string.
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)

    cache.set(("blade runner", "en-US"), "english")
    cache.set(("blade runner", "fr-FR"), "french")
    cache.set("blade runner", "bare")

    assert cache.get(("blade runner", "en-US")) == "english"
    assert cache.get(("blade runner", "fr-FR")) == "french"
    assert cache.get("blade runner") == "bare"


def test_the_oldest_entries_are_evicted_when_full() -> None:
    cache: TTLCache[int] = TTLCache(ttl_seconds=60, max_entries=3)

    for index in range(5):
        cache.set(index, index)

    assert len(cache) == 3
    assert cache.get(0) is None
    assert cache.get(1) is None
    assert [cache.get(index) for index in (2, 3, 4)] == [2, 3, 4]


def test_the_default_ceiling_is_a_ceiling() -> None:
    cache: TTLCache[int] = TTLCache(ttl_seconds=60)

    for index in range(MAX_ENTRIES + 20):
        cache.set(index, index)

    assert len(cache) <= MAX_ENTRIES


def test_clear_empties_entries_and_locks() -> None:
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    cache.set("key", "value")

    cache.clear()

    assert len(cache) == 0
    assert cache._locks == {}
