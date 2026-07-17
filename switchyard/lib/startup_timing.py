# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage startup timing for ``switchyard launch``.

Turn it on with ``--startup-timing`` or ``SWITCHYARD_STARTUP_TIMING=1``; it does
nothing when off. ``mark()`` records a checkpoint during startup, and ``dump()``
prints the time between checkpoints to stderr just before the agent starts. It
times only switchyard's own startup work, not Python's one-time import cost.
"""

import os
import sys
import time
from typing import TextIO

# On unless SWITCHYARD_STARTUP_TIMING is unset or "0", or once --startup-timing calls
# enable(). Read once at import — the env var does not change during a launch.
enabled: bool = os.environ.get("SWITCHYARD_STARTUP_TIMING", "0") != "0"

# (label, perf_counter timestamp) for each point reached during startup.
_marks: list[tuple[str, float]] = []


def enable() -> None:
    """Turn timing on for this process (called by the ``--startup-timing`` flag)."""
    global enabled
    enabled = True


def mark(label: str) -> None:
    """Record that startup reached *label*. No-op unless timing is enabled."""
    if enabled:
        _marks.append((label, time.perf_counter()))


def dump(stream: TextIO | None = None) -> None:
    """Print the per-stage breakdown to stderr, then reset. No-op when disabled.

    Each line is the time between consecutive marks; the last line is the total
    from the first mark to the last.
    """
    if not enabled or len(_marks) < 2:
        _marks.clear()
        return
    out = stream if stream is not None else sys.stderr
    start = _marks[0][1]
    prev = start
    lines = ["switchyard startup timing:"]
    for label, stamp in _marks[1:]:
        lines.append(f"  {(stamp - prev) * 1000:8.1f} ms  {label}")
        prev = stamp
    lines.append(f"  {'-' * 8}")
    lines.append(f"  {(prev - start) * 1000:8.1f} ms  total (launch invoked -> child spawn)")
    print("\n".join(lines), file=out)
    _marks.clear()
