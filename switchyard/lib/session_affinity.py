# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-conversation decision pin shared across routers (sticky routing)."""

from __future__ import annotations

import inspect
import logging
import time
from typing import TYPE_CHECKING

from switchyard.lib.conversation_turn import conversation_turn_number
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.session_cache import SessionCache
from switchyard.lib.session_key import session_key_from_body
from switchyard_rust.core import ChatRequest

if TYPE_CHECKING:
    from collections.abc import Callable

    from switchyard.lib.affinity_pin_store import AffinityPinStore

logger = logging.getLogger(__name__)

#: Consecutive L2 failures that open the circuit breaker.
_L2_BREAKER_THRESHOLD = 3

#: Seconds the breaker stays open before allowing a recovery probe.
_L2_BREAKER_COOLDOWN_S = 10.0

#: ``ProxyContext.metadata`` key memoizing the per-request session key, so
#: callers that consult affinity more than once don't re-hash the (growing)
#: request body.
CTX_SESSION_KEY = "_session_affinity_key"


class SessionAffinity:
    """Pins a routing decision per conversation and reuses it on later turns.

    The shared building block both routing paths use for the *common* part of
    stickiness: derive a stable conversation key (system prompt + first user
    message, memoized on the request context) and store/look up a pinned value
    in a bounded LRU. The pinned value is opaque — a tier label for the
    classifier router, an endpoint id for the latency backend. Each caller keeps
    its own *policy* (when to write a pin, whether to honor one) on top.

    The in-process ``SessionCache`` is the L1. An optional ``l2`` implementing
    :class:`~switchyard.lib.affinity_pin_store.AffinityPinStore` is a shared,
    out-of-process store (e.g. Redis): pins are read through it on an L1 miss
    (and warmed back into L1) and written through it, so a conversation stays
    pinned across workers and pod churn. The L2 is **best-effort** — any error
    or timeout falls back to L1 / unpinned and never breaks routing — and sits
    behind a circuit breaker: after ``l2_breaker_threshold`` consecutive
    failures, L2 operations are skipped (no network attempt, zero added
    latency) for ``l2_breaker_cooldown_s``; the first operation after the
    cooldown is the recovery probe, whose success closes the breaker.

    A disabled instance no-ops (and never hashes the body). Not thread-safe —
    touched only on the request path.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_sessions: int = 10_000,
        warmup_turns: int = 0,
        l2: AffinityPinStore | None = None,
        l2_breaker_threshold: int = _L2_BREAKER_THRESHOLD,
        l2_breaker_cooldown_s: float = _L2_BREAKER_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if warmup_turns < 0:
            raise ValueError("warmup_turns must be >= 0")
        if l2_breaker_threshold < 1:
            raise ValueError("l2_breaker_threshold must be >= 1")
        if l2_breaker_cooldown_s <= 0:
            raise ValueError("l2_breaker_cooldown_s must be > 0")
        self._enabled = enabled
        self._warmup_turns = warmup_turns
        self._pins: SessionCache = SessionCache(max_sessions)
        #: Optional shared/persistent L2 store; ``None`` keeps behavior L1-only.
        self._l2 = l2
        # L2 observability: hits = pins rescued from the shared store after an
        # L1 miss (cross-worker reuse); errors = fail-open get/put operations
        # (the alerting signal for a silently degraded store). Event-loop only,
        # like the pins themselves — no lock.
        self._l2_hits = 0
        self._l2_errors = 0
        # Circuit breaker over the L2: a streak of consecutive failures opens
        # it, and while open every operation is skipped without a network
        # attempt, so a store outage stops taxing requests with timeout waits.
        # ``clock`` is injectable for tests; monotonic so wall-clock jumps
        # can't wedge the breaker.
        self._l2_breaker_threshold = l2_breaker_threshold
        self._l2_breaker_cooldown_s = l2_breaker_cooldown_s
        self._clock = clock
        self._l2_failure_streak = 0
        self._l2_skip_until = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def max_sessions(self) -> int:
        """Configured maximum number of pinned conversations."""
        return self._pins.max_sessions

    @property
    def warmup_turns(self) -> int:
        """Number of initial turns that cannot read or write affinity pins."""
        return self._warmup_turns

    @property
    def l2_enabled(self) -> bool:
        """Whether a shared (L2) pin store is configured."""
        return self._l2 is not None

    @property
    def l2_hits(self) -> int:
        """Pins resolved from the shared store after an in-process (L1) miss."""
        return self._l2_hits

    @property
    def l2_errors(self) -> int:
        """Shared-store operations (get or put) that failed open."""
        return self._l2_errors

    @property
    def l2_breaker_open(self) -> bool:
        """Whether the breaker is currently skipping shared-store operations."""
        return self._l2 is not None and self._clock() < self._l2_skip_until

    def _l2_skips(self) -> bool:
        """Breaker gate checked before every L2 operation.

        ``True`` while the cooldown is running — skip the store without a
        network attempt. Once the cooldown elapses this returns ``False``, so
        the next operation goes through as the recovery probe (concurrent
        in-flight requests at that instant may each probe; every probe is
        bounded by the store's socket timeout, so the burst is harmless).
        """
        return self._clock() < self._l2_skip_until

    def _record_l2_failure(self, what: str) -> None:
        """Count one fail-open L2 operation and open the breaker at the streak."""
        self._l2_errors += 1
        self._l2_failure_streak += 1
        if self._l2_failure_streak < self._l2_breaker_threshold:
            logger.warning("session-affinity L2 %s", what, exc_info=True)
            return
        self._l2_skip_until = self._clock() + self._l2_breaker_cooldown_s
        logger.warning(
            "session-affinity L2 %s; breaker open for %.1fs (failure streak %d)",
            what,
            self._l2_breaker_cooldown_s,
            self._l2_failure_streak,
            exc_info=True,
        )

    def _record_l2_success(self) -> None:
        """Reset the failure streak; close the breaker after a successful probe."""
        if self._l2_failure_streak >= self._l2_breaker_threshold:
            logger.info("session-affinity L2 recovered; breaker closed")
        self._l2_failure_streak = 0
        self._l2_skip_until = 0.0

    async def pinned(self, ctx: ProxyContext, request: ChatRequest) -> str | None:
        """Return the value pinned to ``request``'s conversation, or ``None``.

        Checks L1 first; on a miss (and only when an L2 is configured) consults
        the shared store and warms the hit back into L1 so later turns on this
        worker skip the round-trip. An L2 error is swallowed — the conversation
        is treated as unpinned rather than failing the request — and while the
        breaker is open the store isn't consulted at all.
        """
        if not self._enabled or self._is_warmup_turn(request):
            return None
        key = self._session_key(ctx, request)
        value = self._pins.get(key)
        if value is not None or self._l2 is None or self._l2_skips():
            return value
        try:
            value = await self._l2.get(key)
        except Exception:
            self._record_l2_failure("get failed; routing without a pin")
            return None
        self._record_l2_success()
        if value is not None:
            self._l2_hits += 1
            self._pins.put(key, value)
        return value

    async def pin(self, ctx: ProxyContext, request: ChatRequest, value: str) -> None:
        """Pin ``request``'s conversation to ``value`` (no-op when disabled).

        Writes through to the L2 when configured; an L2 error is swallowed so
        the pin stays worker-local rather than failing the request, and while
        the breaker is open the write is skipped entirely.
        """
        if not self._enabled or self._is_warmup_turn(request):
            return
        key = self._session_key(ctx, request)
        self._pins.put(key, value)
        if self._l2 is None or self._l2_skips():
            return
        try:
            await self._l2.put(key, value)
        except Exception:
            self._record_l2_failure("put failed; pin is worker-local only")
            return
        self._record_l2_success()

    async def aclose(self) -> None:
        """Release the shared store's resources (no-op for L1-only instances).

        Duck-typed: closes the L2 only if it exposes ``aclose()`` (the protocol
        doesn't require one). Best-effort like every other L2 interaction — a
        close failure is logged, never raised, so backend teardown can't wedge.
        """
        if self._l2 is None:
            return
        closer = getattr(self._l2, "aclose", None)
        if not callable(closer):
            return
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning("session-affinity L2 close failed", exc_info=True)

    def __len__(self) -> int:
        """Number of conversations currently pinned."""
        return len(self._pins)

    def _session_key(self, ctx: ProxyContext, request: ChatRequest) -> str:
        """Derive the conversation key once per request, memoized on ``ctx``."""
        cached = ctx.metadata.get(CTX_SESSION_KEY)
        if isinstance(cached, str):
            return cached
        key = session_key_from_body(request.body)
        ctx.metadata[CTX_SESSION_KEY] = key
        return key

    def _is_warmup_turn(self, request: ChatRequest) -> bool:
        """Return whether this request is still inside the no-stick warmup."""
        return conversation_turn_number(request) <= self._warmup_turns


__all__ = ["CTX_SESSION_KEY", "SessionAffinity"]
