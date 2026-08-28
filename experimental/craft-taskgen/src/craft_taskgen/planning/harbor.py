# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor invocation + trial-result parsing shared by planning-task tooling."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_harbor(
    dataset_dir: Path,
    trials_dir: Path,
    model: str,
    agent: str,
    api_base: str,
    task_filter: str | None,
) -> None:
    """Invoke `harbor run` on a dataset, writing trial outputs to ``trials_dir``."""
    trials_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "harbor",
        "run",
        "-p",
        str(dataset_dir),
        "-a",
        agent,
        "--agent-kwarg",
        f"api_base={api_base}",
        "-m",
        model,
        "-o",
        str(trials_dir),
        "-n",
        "1",
    ]
    if task_filter:
        cmd.extend(["--filter", task_filter])
    logger.info("harbor: %s", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"harbor run failed with returncode {proc.returncode}")


def parse_trial_result(trial_dir: Path) -> dict[str, Any]:
    """Extract F2P/P2P scores from a trial directory.

    Prefers the detailed ``verifier/results.json`` layout written by the
    planning adapter's score.py. Falls back to result.json's binary reward.
    """
    detailed = trial_dir / "verifier" / "results.json"
    if detailed.exists():
        data = json.loads(detailed.read_text())
        if "f2p" in data and isinstance(data["f2p"], dict):
            return {
                "f2p_score": float(data["f2p"].get("score", 0.0)),
                "p2p_score": float(data["p2p"].get("score", 1.0)),
                "reward": float(data.get("reward", 0.0)),
                "failed_f2p": list(data["f2p"].get("failed_tests", [])),
                "failed_p2p": list(data["p2p"].get("failed_tests", [])),
            }
        if "fail_to_pass" in data:
            total = max(int(data.get("total_functional", data.get("fail_to_pass", 0))), 1)
            passed = int(data.get("fail_to_pass", 0))
            return {
                "f2p_score": passed / total,
                "p2p_score": 1.0 if int(data.get("regression_failures", 0)) == 0 else 0.0,
                "reward": float(data.get("reward", 0.0)),
                "failed_f2p": list(data.get("still_failing_tests", [])),
                "failed_p2p": list(data.get("regression_tests", [])),
            }
    result_json = trial_dir / "result.json"
    if result_json.exists():
        data = json.loads(result_json.read_text())
        reward = float(((data.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0.0))
        return {
            "f2p_score": reward,
            "p2p_score": 1.0 if reward == 1.0 else 0.0,
            "reward": reward,
            "failed_f2p": [],
            "failed_p2p": [],
        }
    return {
        "f2p_score": 0.0,
        "p2p_score": 0.0,
        "reward": 0.0,
        "failed_f2p": [],
        "failed_p2p": [],
    }


def resolve_harbor_output_dir(trials_dir: Path) -> Path:
    """Harbor writes outputs under ``trials_dir/{timestamp}/``.

    Given the path passed to ``harbor run -o``, return the path that actually
    contains the ``{task}__{suffix}`` trial directories. If ``trials_dir``
    already holds trial dirs directly, return it unchanged.
    """
    if not trials_dir.is_dir():
        return trials_dir
    children = [c for c in trials_dir.iterdir() if c.is_dir()]
    if not children:
        return trials_dir

    # Timestamp dirs (harbor's job dirs) themselves contain trial subdirs that
    # have their own result.json. A flat layout has result.json directly on
    # each child. Distinguish by checking whether any child has a grandchild
    # with result.json -- if so, that child is a timestamp (job) dir.
    def _has_trial_grandchild(child: Path) -> bool:
        if not child.is_dir():
            return False
        for gc in child.iterdir():
            if gc.is_dir() and (gc / "result.json").exists():
                return True
        return False

    timestamp_children = [c for c in children if _has_trial_grandchild(c)]
    if timestamp_children:
        return max(timestamp_children, key=lambda c: c.stat().st_mtime)
    return trials_dir


def collect_trial_dirs(trials_dir: Path) -> dict[str, Path]:
    """Map task_name -> newest trial dir.

    Harbor names trials ``{task_name}__{suffix}``. If multiple trials exist
    (retries), return the most recently modified. Handles both layouts:
    flat ``trials_dir/{task}__xxx/`` and timestamped
    ``trials_dir/{timestamp}/{task}__xxx/``.
    """
    effective = resolve_harbor_output_dir(trials_dir)
    by_task: dict[str, Path] = {}
    for child in effective.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if "__" not in name:
            continue
        base = name.rsplit("__", 1)[0]
        prev = by_task.get(base)
        if prev is None or child.stat().st_mtime > prev.stat().st_mtime:
            by_task[base] = child
    return by_task
