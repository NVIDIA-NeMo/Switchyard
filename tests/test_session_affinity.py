# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SessionAffinity's L1 (in-process) + optional L2 (shared) tiering.

The latency backend and classifier router integration tests cover affinity
*policy*; these tests cover the store mechanics in isolation: write-through,
read-through with L1 warming, and best-effort (fail-open) degradation when the
L2 errors.
"""

from __future__ import annotations

from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.session_affinity import SessionAffinity
from switchyard_rust.core import ChatRequest


def _request(text: str = "hi") -> ChatRequest:
    """A turn-1 request whose first user message anchors a distinct session."""
    return ChatRequest.openai_chat(
        {"model": "incoming-model", "messages": [{"role": "user", "content": text}]}
    )


class _DictL2:
    """In-memory AffinityPinStore double with call counters."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.gets = 0
        self.puts = 0

    async def get(self, key: str) -> str | None:
        self.gets += 1
        return self.store.get(key)

    async def put(self, key: str, value: str) -> None:
        self.puts += 1
        self.store[key] = value


class _BrokenL2:
    """AffinityPinStore double that always raises, to exercise fail-open."""

    async def get(self, key: str) -> str | None:
        raise RuntimeError("l2 unavailable")

    async def put(self, key: str, value: str) -> None:
        raise RuntimeError("l2 unavailable")


async def test_no_l2_still_pins_via_l1() -> None:
    """The async interface works L1-only when no L2 is configured."""
    affinity = SessionAffinity(enabled=True)
    req = _request("task")
    await affinity.pin(ProxyContext(), req, "model-A")
    assert await affinity.pinned(ProxyContext(), req) == "model-A"


async def test_pin_writes_through_to_l2() -> None:
    """A pin lands in both L1 and the shared L2."""
    l2 = _DictL2()
    affinity = SessionAffinity(enabled=True, l2=l2)
    req = _request("task")
    await affinity.pin(ProxyContext(), req, "model-A")
    assert l2.puts == 1
    assert list(l2.store.values()) == ["model-A"]


async def test_l1_miss_reads_through_l2_and_warms_l1() -> None:
    """A pin written by one worker is visible to another via the shared L2,
    and the read-through warms the reader's L1 so later turns skip the round-trip."""
    l2 = _DictL2()
    worker_a = SessionAffinity(enabled=True, l2=l2)
    worker_b = SessionAffinity(enabled=True, l2=l2)
    req = _request("shared task")

    # Worker A serves the first turn and pins.
    await worker_a.pin(ProxyContext(), req, "model-A")

    # Worker B has never seen this conversation: L1 miss → L2 read-through.
    assert await worker_b.pinned(ProxyContext(), req) == "model-A"
    assert l2.gets == 1
    assert worker_b.l2_hits == 1

    # L1 is now warm on worker B: a second lookup does not hit L2 again.
    assert await worker_b.pinned(ProxyContext(), req) == "model-A"
    assert l2.gets == 1
    assert worker_b.l2_hits == 1


async def test_l2_get_failure_is_fail_open() -> None:
    """An L2 read error routes as unpinned rather than raising, and is counted."""
    affinity = SessionAffinity(enabled=True, l2=_BrokenL2())
    assert await affinity.pinned(ProxyContext(), _request("task")) is None
    assert affinity.l2_errors == 1


async def test_l2_put_failure_is_fail_open_and_keeps_l1_pin() -> None:
    """An L2 write error does not raise; the pin still holds in the local L1."""
    affinity = SessionAffinity(enabled=True, l2=_BrokenL2())
    req = _request("task")
    await affinity.pin(ProxyContext(), req, "model-A")  # must not raise
    assert affinity.l2_errors == 1
    # L1 retained the pin, so a same-worker lookup still resolves (no L2 read).
    assert await affinity.pinned(ProxyContext(), req) == "model-A"
    assert affinity.l2_errors == 1  # the L1 hit never consulted the broken L2


async def test_disabled_never_touches_l2() -> None:
    """A disabled instance no-ops without reading or writing the shared store."""
    l2 = _DictL2()
    affinity = SessionAffinity(enabled=False, l2=l2)
    req = _request("task")
    await affinity.pin(ProxyContext(), req, "model-A")
    assert await affinity.pinned(ProxyContext(), req) is None
    assert l2.gets == 0
    assert l2.puts == 0


async def test_warmup_turn_never_touches_l2() -> None:
    """Turns inside the warmup window neither read nor write the shared store."""
    l2 = _DictL2()
    affinity = SessionAffinity(enabled=True, warmup_turns=1, l2=l2)
    req = _request("task")  # turn 1, within warmup_turns=1
    await affinity.pin(ProxyContext(), req, "model-A")
    assert await affinity.pinned(ProxyContext(), req) is None
    assert l2.gets == 0
    assert l2.puts == 0


async def test_aclose_delegates_to_l2() -> None:
    """Closing the affinity coordinator releases the shared store."""

    class _ClosableL2(_DictL2):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    l2 = _ClosableL2()
    affinity = SessionAffinity(enabled=True, l2=l2)
    await affinity.aclose()
    assert l2.closed


async def test_aclose_is_noop_without_l2_or_without_aclose() -> None:
    """L1-only instances and L2s lacking aclose() close without error."""
    await SessionAffinity(enabled=True).aclose()
    await SessionAffinity(enabled=True, l2=_DictL2()).aclose()  # no aclose method


async def test_aclose_swallows_l2_close_errors() -> None:
    """A failing store close is logged, never raised — teardown can't wedge."""

    class _BrokenClose(_DictL2):
        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    await SessionAffinity(enabled=True, l2=_BrokenClose()).aclose()  # must not raise


# ---------------------------------------------------------------------------
# L2 circuit breaker
# ---------------------------------------------------------------------------


class _Clock:
    """Manually-advanced monotonic clock for deterministic breaker tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FlakyL2(_BrokenL2):
    """Failing store that counts attempts and can be healed mid-test."""

    def __init__(self) -> None:
        self.attempts = 0
        self.healed = False
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        self.attempts += 1
        if not self.healed:
            raise RuntimeError("l2 unavailable")
        return self.store.get(key)

    async def put(self, key: str, value: str) -> None:
        self.attempts += 1
        if not self.healed:
            raise RuntimeError("l2 unavailable")
        self.store[key] = value


def _breaker_affinity(l2: _FlakyL2, clock: _Clock) -> SessionAffinity:
    return SessionAffinity(
        enabled=True,
        l2=l2,
        l2_breaker_threshold=3,
        l2_breaker_cooldown_s=10.0,
        clock=clock,
    )


async def test_l2_breaker_opens_after_streak_and_skips_operations() -> None:
    """Threshold consecutive failures open the breaker; further operations
    never reach the store while the cooldown runs."""
    l2, clock = _FlakyL2(), _Clock()
    affinity = _breaker_affinity(l2, clock)

    for i in range(3):
        await affinity.pin(ProxyContext(), _request(f"t{i}"), "model-A")
    assert l2.attempts == 3
    assert affinity.l2_breaker_open

    # Open breaker: neither reads (distinct sessions = L1 misses) nor writes
    # touch the store.
    assert await affinity.pinned(ProxyContext(), _request("t-new")) is None
    await affinity.pin(ProxyContext(), _request("t-newer"), "model-A")
    assert l2.attempts == 3
    assert affinity.l2_errors == 3


async def test_l2_breaker_probes_after_cooldown_and_closes_on_success() -> None:
    """The first operation after the cooldown is the probe; success closes the
    breaker and normal read/write-through resumes."""
    l2, clock = _FlakyL2(), _Clock()
    affinity = _breaker_affinity(l2, clock)

    for i in range(3):
        await affinity.pin(ProxyContext(), _request(f"t{i}"), "model-A")
    l2.healed = True
    clock.now = 10.0  # cooldown elapsed → next op is the probe

    await affinity.pin(ProxyContext(), _request("probe"), "model-A")

    assert not affinity.l2_breaker_open
    assert l2.attempts == 4
    # Fully closed: later operations go through again.
    await affinity.pin(ProxyContext(), _request("after"), "model-A")
    assert l2.attempts == 5


async def test_l2_breaker_failed_probe_rearms_the_cooldown() -> None:
    """A failing probe re-opens the breaker for another full cooldown."""
    l2, clock = _FlakyL2(), _Clock()
    affinity = _breaker_affinity(l2, clock)

    for i in range(3):
        await affinity.pin(ProxyContext(), _request(f"t{i}"), "model-A")
    clock.now = 10.0  # still broken: the probe fails

    await affinity.pin(ProxyContext(), _request("probe"), "model-A")

    assert l2.attempts == 4
    assert affinity.l2_breaker_open
    # Mid-cooldown operations stay skipped.
    clock.now = 19.9
    await affinity.pin(ProxyContext(), _request("mid"), "model-A")
    assert l2.attempts == 4


async def test_l2_non_consecutive_failures_do_not_open_the_breaker() -> None:
    """A success resets the streak: intermittent blips never trip the breaker."""
    l2, clock = _FlakyL2(), _Clock()
    affinity = _breaker_affinity(l2, clock)

    await affinity.pin(ProxyContext(), _request("t0"), "model-A")
    await affinity.pin(ProxyContext(), _request("t1"), "model-A")
    l2.healed = True
    await affinity.pin(ProxyContext(), _request("t2"), "model-A")  # success resets
    l2.healed = False
    await affinity.pin(ProxyContext(), _request("t3"), "model-A")
    await affinity.pin(ProxyContext(), _request("t4"), "model-A")

    assert not affinity.l2_breaker_open
    assert affinity.l2_errors == 4


async def test_l2_breaker_config_validated() -> None:
    """Breaker knobs reject nonsensical values."""
    import pytest

    with pytest.raises(ValueError):
        SessionAffinity(enabled=True, l2_breaker_threshold=0)
    with pytest.raises(ValueError):
        SessionAffinity(enabled=True, l2_breaker_cooldown_s=0)
