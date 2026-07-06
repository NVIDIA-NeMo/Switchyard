# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor episode dirs → ScoredPrompt stream.

Harbor lays out one directory per task with an ``agent/`` subtree::

    <run-root>/
      <task-id>__<suffix>/
        agent/
          episode-0/prompt.txt   ← LLM-inbound prompt for turn 0
          episode-1/prompt.txt
          ...

Switchyard's ``switchyard verify --route-profile ...`` writes Harbor TBLite
runs at ``benchmark/tb_runs/<run-name>/jobs/<run-name>/`` — that's the
directory to point the manifest at.

The ``source`` field on emitted records comes from the manifest entry's
``source:`` key (defaults to ``"harbor"``). Use distinct source labels on
different runs (e.g. ``stage_router-settle-aware``, ``all-opus``) so the by-source slice in
the report compares like-with-like.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from . import ScoredPrompt

_EPISODE_DIR_PATTERN = re.compile(r"^episode-(\d+)$")
DEFAULT_SOURCE = "harbor"


def extract(
    run_id: str,
    run_root: Path,
    *,
    source: str = DEFAULT_SOURCE,
) -> Iterator[ScoredPrompt]:
    """Walk a Harbor run directory and yield one ScoredPrompt per episode."""
    for task_dir in sorted(run_root.iterdir()):
        if not task_dir.is_dir():
            continue
        agent_dir = task_dir / "agent"
        if not agent_dir.is_dir():
            continue
        task = task_dir.name.split("__", 1)[0]
        for episode_dir, idx in _ordered_episode_dirs(agent_dir):
            prompt_path = episode_dir / "prompt.txt"
            if not prompt_path.is_file():
                continue
            try:
                blob = prompt_path.read_text()
            except OSError:
                continue
            if not blob.strip():
                continue
            yield ScoredPrompt(
                run_id=run_id,
                source=source,
                task=task,
                turn_idx=idx,
                full=blob,
                # Harbor prompts are pre-concatenated blobs (system framing +
                # tool history + latest turn), not a structured messages list.
                # ``latest`` falls back to the same blob.
                latest=blob,
            )


def _ordered_episode_dirs(agent_dir: Path) -> Iterator[tuple[Path, int]]:
    """Yield (episode_dir, turn_idx) ordered by numeric episode index."""
    ordered: list[tuple[int, Path]] = []
    for child in agent_dir.iterdir():
        if not child.is_dir():
            continue
        match = _EPISODE_DIR_PATTERN.match(child.name)
        if not match:
            continue
        ordered.append((int(match.group(1)), child))
    ordered.sort(key=lambda pair: pair[0])
    for idx, path in ordered:
        yield path, idx
