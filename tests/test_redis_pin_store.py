# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RedisPinStore adapter mechanics and its config → backend wiring.

An injected fake async client keeps these hermetic — no live Redis and no
``fakeredis`` dependency. Building the L2 exercises the lazy-import path without
requiring ``redis`` to be installed, because the real client is only created on
first ``get``/``put``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from switchyard.lib.backends.latency_service_llm_backend import _build_affinity_l2
from switchyard.lib.config.latency_service_backend_config import (
    LatencyServiceBackendConfig,
    LatencyServiceEndpoint,
)
from switchyard.lib.redis_pin_store import RedisPinStore


class _FakeAsyncRedis:
    """Minimal async Redis double capturing set() arguments."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.set_calls: list[tuple[str, str, int | None]] = []

    async def get(self, key: str) -> str | None:
        return self.kv.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.set_calls.append((key, value, ex))
        self.kv[key] = value


# --- adapter mechanics ------------------------------------------------------


async def test_put_writes_prefixed_key_with_ttl() -> None:
    fake = _FakeAsyncRedis()
    store = RedisPinStore("redis://x", ttl_seconds=123, key_prefix="p:", client=fake)
    await store.put("abc", "model-A")
    assert fake.set_calls == [("p:abc", "model-A", 123)]


async def test_get_reads_prefixed_key_and_roundtrips() -> None:
    fake = _FakeAsyncRedis()
    store = RedisPinStore("redis://x", key_prefix="p:", client=fake)
    assert await store.get("abc") is None
    await store.put("abc", "model-B")
    assert await store.get("abc") == "model-B"


def test_invalid_ttl_rejected() -> None:
    with pytest.raises(ValueError):
        RedisPinStore("redis://x", ttl_seconds=0)


async def test_aclose_closes_client_once() -> None:
    """aclose releases the client's pool; repeated calls don't double-close."""

    class _ClosableFake(_FakeAsyncRedis):
        def __init__(self) -> None:
            super().__init__()
            self.closes = 0

        async def aclose(self) -> None:
            self.closes += 1

    fake = _ClosableFake()
    store = RedisPinStore("redis://x", client=fake)
    await store.aclose()
    await store.aclose()  # client reference already released → no-op
    assert fake.closes == 1


async def test_aclose_before_first_use_is_noop() -> None:
    """Closing a store that never built a client must not import or connect."""
    await RedisPinStore("redis://x").aclose()


# --- config validation + backend wiring -------------------------------------


def _redis_config(**overrides: object) -> LatencyServiceBackendConfig:
    base: dict[str, object] = {
        "endpoints": [LatencyServiceEndpoint(model="m")],
        "session_affinity": True,
        "affinity_store": "redis",
        "affinity_store_url": "redis://cache:6379/0",
    }
    base.update(overrides)
    return LatencyServiceBackendConfig(**base)  # type: ignore[arg-type]


def test_redis_store_requires_url() -> None:
    with pytest.raises(ValidationError):
        _redis_config(affinity_store_url=None)


def test_redis_store_requires_session_affinity() -> None:
    with pytest.raises(ValidationError):
        _redis_config(session_affinity=False)


def test_memory_is_the_default_store() -> None:
    cfg = LatencyServiceBackendConfig(endpoints=[LatencyServiceEndpoint(model="m")])
    assert cfg.affinity_store == "memory"
    assert _build_affinity_l2(cfg) is None


def test_build_affinity_l2_threads_config_into_redis_store() -> None:
    cfg = _redis_config(affinity_store_ttl_seconds=42, affinity_key_prefix="k:")
    store = _build_affinity_l2(cfg)
    assert isinstance(store, RedisPinStore)
    assert store._ttl_seconds == 42
    assert store._key_prefix == "k:"
