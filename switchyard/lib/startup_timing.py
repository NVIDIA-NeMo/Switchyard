# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in granular startup timing for ``switchyard launch``.

Prints a per-stage breakdown of where launch startup time goes, right before
the child agent (``claude``) is spawned. Enable it either way:

* ``switchyard launch claude --startup-timing`` (the CLI flag calls
  :func:`enable`), or
* ``SWITCHYARD_STARTUP_TIMING=1 switchyard launch claude`` (the env var).

If neither is set it does not trigger at all — :func:`mark`/:func:`dump` return
immediately, so a normal launch pays nothing.

The breakdown is granular: the launcher and the backend-format resolver both
call :func:`mark` at fixed points, so each backend-format probe
(``/v1/chat/completions``, ``/v1/messages``, ``/v1/responses``) shows up as its
own line. A slow line names the slow phase — for example three slow ``probe:``
lines mean a slow upstream made the AUTO format detection stack timeouts.

Timing starts at launch dispatch, so it covers config resolution, chain build
(including the AUTO backend-format probe), proxy boot, and child hand-off. The
one-time Python import/bootstrap cost happens before dispatch and is not counted
here; measure it on its own with ``switchyard launch claude --model X --dry-run``.

State is module-level because a launch is a single, short-lived process — there
is one launch per ``switchyard`` invocation.
"""

import os
import sys
import time
from typing import TextIO

_ENV_VAR = "SWITCHYARD_STARTUP_TIMING"
_FALSEY = {"", "0", "false", "no"}

# Set by the --startup-timing CLI flag via enable(). ORed with the env var so
# either source turns timing on.
_forced = False

# (label, perf_counter timestamp) for each point reached during startup.
_marks: list[tuple[str, float]] = []


def enable() -> None:
    """Turn timing on for this process (called by the ``--startup-timing`` flag)."""
    global _forced
    _forced = True


def enabled() -> bool:
    """Return whether timing is on — the CLI flag or a truthy env var enables it."""
    if _forced:
        return True
    return os.environ.get(_ENV_VAR, "").strip().lower() not in _FALSEY


def mark(label: str) -> None:
    """Record that startup reached *label*. No-op unless timing is enabled."""
    if enabled():
        _marks.append((label, time.perf_counter()))


def dump(stream: TextIO | None = None) -> None:
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
    lines = ["switchyard startup timing:"]
    for label, stamp in _marks[1:]:
        lines.append(f"  {(stamp - prev) * 1000:8.1f} ms  {label}")
        prev = stamp
    lines.append(f"  {'-' * 8}")
    lines.append(f"  {(prev - start) * 1000:8.1f} ms  total (launch invoked -> child spawn)")
    print("\n".join(lines), file=out)
    _marks.clear()
