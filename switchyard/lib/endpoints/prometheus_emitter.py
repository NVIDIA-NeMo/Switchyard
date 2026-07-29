# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide table for objects that contribute extra ``/metrics`` lines.

The accumulator-derived exposition rendered by
:func:`switchyard.lib.prometheus_exposition.render_prometheus` covers
request flow (counts, tokens, latency). Components that own state which
is not request-derived register an emitter here so their lines appear on
the same ``/metrics`` scrape rather than a sidecar URL.

Single-process table by design: a Switchyard server is one process,
emitters are owned by component lifetimes, and the table is
write-once-read-many across startup.
"""

from __future__ import annotations

from collections.abc import Callable

PrometheusEmitter = Callable[[], list[str]]
"""A no-arg callable returning Prometheus exposition lines (no trailing newline).

Each call snapshots the emitter's current state. The table composes
output in registration order; emitters must not assume any ordering
relative to other emitters or to the accumulator-derived block.
"""

_EMITTERS: list[PrometheusEmitter] = []


def register(emitter: PrometheusEmitter) -> None:
    """Register an emitter. Idempotent — re-registering is a no-op."""
    if emitter not in _EMITTERS:
        _EMITTERS.append(emitter)


def unregister(emitter: PrometheusEmitter) -> None:
    """Remove a previously-registered emitter. No-op if not present.

    Backends call this from their ``shutdown()`` hook so a re-built chain
    does not leave a stale closure pointing at a torn-down backend.
    """
    try:
        _EMITTERS.remove(emitter)
    except ValueError:
        pass


def render() -> str:
    """Compose registered emitter output as Prometheus exposition text.

    Returns an empty string when no emitter is registered, so callers can
    unconditionally concatenate the result to the accumulator-derived
    exposition without producing trailing whitespace artefacts.
    """
    lines: list[str] = []
    for emitter in _EMITTERS:
        lines.extend(emitter())
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _clear_for_tests() -> None:
    """Drop every registered emitter — test fixtures only."""
    _EMITTERS.clear()


__all__ = ["PrometheusEmitter", "register", "unregister", "render"]
