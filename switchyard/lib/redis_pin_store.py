# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Redis-backed shared L2 store for session-affinity pins.

Implements :class:`~switchyard.lib.affinity_pin_store.AffinityPinStore` against a
standalone Redis so pins are visible to every Switchyard worker/pod and survive
pod churn. Keys are namespaced ``{key_prefix}{session_key}`` and expire after
``ttl_seconds``; the latency backend re-pins on every successful turn, so an
active conversation slides its own TTL.

The ``redis`` dependency is optional (``switchyard[affinity-redis]``) and is
imported lazily, so the default install never pulls it in. Socket timeouts are
bounded: a slow or unreachable store fails fast into ``SessionAffinity``'s
best-effort (fail-open) path rather than blocking the request.
"""

from __future__ import annotations

from typing import Any

#: Bounded socket timeout (seconds). The store is expected to be colocated with
#: the workers, so a healthy round-trip is sub-millisecond — 100 ms is ~100×
#: headroom. The cap bounds how long a *degraded* store can stall a request
#: before fail-open kicks in (and before SessionAffinity's circuit breaker
#: stops attempting the store at all).
_DEFAULT_SOCKET_TIMEOUT_S = 0.1


class RedisPinStore:
    """Shared, persistent pin store backed by a standalone Redis.

    Args:
        url: Redis connection URL (e.g. ``"redis://host:6379/0"``).
        ttl_seconds: Expiry applied to each pin write. Must be > 0.
        key_prefix: Namespace prepended to every key.
        socket_timeout: Per-operation and connect timeout, in seconds.
        client: Pre-built async Redis client. Primarily for tests; when ``None``
            the client is created lazily from ``url`` on first use.
    """

    def __init__(
        self,
        url: str,
        *,
        ttl_seconds: int = 3_600,
        key_prefix: str = "swyd:pin:",
        socket_timeout: float = _DEFAULT_SOCKET_TIMEOUT_S,
        client: Any = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._url = url
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self._socket_timeout = socket_timeout
        self._client = client

    def _get_client(self) -> Any:
        """Return the async Redis client, building it lazily on first use."""
        if self._client is None:
            # Lazy import keeps ``redis`` an optional dependency.
            from redis.asyncio import from_url

            # redis-py's from_url is untyped when the optional extra is
            # installed; without it the import is opaque and the ignore is
            # unused — suppress both cases so strict mypy stays green either way.
            self._client = from_url(  # type: ignore[no-untyped-call,unused-ignore]
                self._url,
                decode_responses=True,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_timeout,
            )
        return self._client

    def _redis_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    async def get(self, key: str) -> str | None:
        """Return the pinned value for ``key``, or ``None`` if absent/expired."""
        value = await self._get_client().get(self._redis_key(key))
        return value if value is None else str(value)

    async def put(self, key: str, value: str) -> None:
        """Pin ``key`` to ``value`` with the configured TTL."""
        await self._get_client().set(self._redis_key(key), value, ex=self._ttl_seconds)

    async def aclose(self) -> None:
        """Close the client and its connection pool. Safe to call repeatedly.

        Releases the client reference first, so a second call is a no-op (a
        later ``get``/``put`` would lazily rebuild — but the store is only
        closed at backend shutdown).
        """
        client = self._client
        if client is None:
            return
        self._client = None
        await client.aclose()


__all__ = ["RedisPinStore"]
