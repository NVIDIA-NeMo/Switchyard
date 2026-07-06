# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory-to-prompt extractors for the dimension-collector calibration corpus.

Each extractor turns one harness run (currently: Harbor episode dirs) into a
stream of :class:`ScoredPrompt` records the calibration runner consumes. Two
scoring scopes per prompt:

* ``full`` — the concatenated message blob the proxy actually sees in
  production. System prompt, prior tool results, and tool calls all in.
* ``latest`` — the most recent user-role text content only. Cleaner signal
  isolation; useful as a counterpoint to ``full``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredPrompt:
    """Normalized prompt record consumed by the calibration runner.

    Fields are designed to be report-sliceable downstream: every analytical
    cut in the report (per-source, per-run, per-turn-position) reads from
    these columns and nothing else.
    """

    run_id: str
    source: str         # label from the manifest entry (e.g. "stage_router-settle-aware")
    task: str           # task id (e.g. "book-portfolio-analysis")
    turn_idx: int       # 0-based turn within the trajectory
    full: str           # full prompt blob (system + history + latest user)
    latest: str         # latest user-role text content only
