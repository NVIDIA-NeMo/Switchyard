# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable async L2 store for session-affinity pins.

The in-process Rust ``SessionCache`` stays the L1 inside
:class:`switchyard.lib.session_affinity.SessionAffinity`. An optional object
implementing this protocol is the L2: a shared, out-of-process store (e.g.
Redis) that lets pins be read by every Switchyard worker/pod and survive pod
churn, so a conversation keeps hitting the same upstream even when its turns
land on different replicas.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AffinityPinStore(Protocol):
    """Shared, out-of-process store for conversation→decision pins.

    The pinned value is opaque to the store (an endpoint id for the latency
    backend, a tier label for the classifier router). Implementations are
    called on the request path, so they should be fast, and
    :class:`SessionAffinity` treats them as **best-effort**: an error or timeout
    must never break request routing (the caller falls back to L1 / unpinned).
    """

    async def get(self, key: str) -> str | None:
        """Return the value pinned to ``key``, or ``None`` if absent."""
        ...

    async def put(self, key: str, value: str) -> None:
        """Pin ``key`` to ``value``. May apply a TTL or eviction policy."""
        ...


__all__ = ["AffinityPinStore"]
