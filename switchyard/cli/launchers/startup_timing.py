# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in per-stage startup timing for ``switchyard launch``.

Set ``SWITCHYARD_STARTUP_TIMING=1`` to print where launch startup time goes
before the child agent (``claude`` / ``codex``) is spawned. The launcher calls
:func:`mark` at a handful of fixed points and :func:`dump` once, right before it
hands off to the child. Each printed line is the wall time spent *between* two
marks, so a slow line names the slow phase — for example a slow
``chain built (incl. backend-format probe)`` points at the AUTO format probe,
not the proxy.

Timing starts at launch dispatch, so it covers config resolution, chain build
(including the AUTO backend-format probe), proxy boot, and child hand-off. The
one-time Python import/bootstrap cost happens before dispatch and is not counted
here; measure it on its own with ``switchyard launch claude --model X --dry-run``.

Off by default and cheap when off: :func:`mark`/:func:`dump` return immediately
unless the env var is set, so this adds nothing to a normal launch. State is a
module-level list because a launch is a single, short-lived process — there is
one launch per ``switchyard`` invocation.
"""

from __future__ import annotations

import os
import sys
import time

_ENV_VAR = "SWITCHYARD_STARTUP_TIMING"
_FALSEY = {"", "0", "false", "no"}

# (label, perf_counter timestamp) for each point reached during startup.
_marks: list[tuple[str, float]] = []


def enabled() -> bool:
    """Return whether ``SWITCHYARD_STARTUP_TIMING`` is set to a truthy value."""
    return os.environ.get(_ENV_VAR, "").strip().lower() not in _FALSEY


def mark(label: str) -> None:
    """Record that startup reached *label*. No-op unless timing is enabled."""
    if enabled():
        _marks.append((label, time.perf_counter()))


def dump(stream: object = None) -> None:
    """Print the per-stage breakdown to stderr, then reset. No-op when disabled.

    Each line is the time between consecutive marks; the last line is the total
    from the first mark to the last.
    """
    if not enabled() or len(_marks) < 2:
        _marks.clear()
        return
    out = stream if stream is not None else sys.stderr
    start = _marks[0][1]
    prev = start
    lines = [f"switchyard startup timing ({_ENV_VAR}):"]
    for label, stamp in _marks[1:]:
        lines.append(f"  {(stamp - prev) * 1000:8.1f} ms  {label}")
        prev = stamp
    lines.append(f"  {'-' * 8}")
    lines.append(f"  {(prev - start) * 1000:8.1f} ms  total (launch invoked -> child spawn)")
    print("\n".join(lines), file=out)  # type: ignore[arg-type]
    _marks.clear()
