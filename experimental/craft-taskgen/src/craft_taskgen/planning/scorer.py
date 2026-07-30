# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning-task scorer.

Runs two planner agent trials (Opus, Haiku by default) on each candidate's
Harbor planning task, feeds each plan through a fixed Sonnet implementer,
and records the per-condition F2P/P2P scores plus the planner-A-minus-B
delta back onto the candidate JSON.

This tool does NOT classify tasks into planning / not_planning / ambiguous.
It only measures. Tagging tasks as "planning_task" is a downstream decision
made after inspecting the empirical distribution across the full candidate
set.

Per candidate:

  1. Build a planner dataset (wrap_pipeline --mode planner).
  2. Run harbor with planner A (Opus 4.6), capture plan.md per task.
  3. Run harbor with planner B (Haiku 4.5), capture plan.md per task.
  4. Build two implementer datasets, each injecting one planner's plans.
  5. Run harbor with Sonnet 4.6 on each.
  6. Parse F2P/P2P per task, compute delta (planner_a - planner_b).
  7. Write the measurements into the candidate's ``planning_scores`` block.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from craft_taskgen.planning.harbor import (
    collect_trial_dirs,
    parse_trial_result,
    resolve_harbor_output_dir,
    run_harbor,
)

logger = logging.getLogger(__name__)


DEFAULT_PLANNER_A = "aws/anthropic/bedrock-claude-opus-4-6"
DEFAULT_PLANNER_B = "aws/anthropic/claude-haiku-4-5-v1"
DEFAULT_IMPLEMENTER = "aws/anthropic/bedrock-claude-sonnet-4-6"

_WRAP_PIPELINE = Path(__file__).resolve().parents[3] / "planning" / "wrap_pipeline.py"


def _default_api_base() -> str:
    """Inference gateway base URL, from `OPENAI_BASE_URL`. No endpoint is hardcoded."""
    return os.environ.get("OPENAI_BASE_URL", "")


@dataclass
class ScorerConfig:
    planner_a_model: str = DEFAULT_PLANNER_A
    planner_b_model: str = DEFAULT_PLANNER_B
    implementer_model: str = DEFAULT_IMPLEMENTER
    api_base: str = field(default_factory=_default_api_base)
    agent: str = "claude-code"


@dataclass
class Score:
    task_name: str
    planner_a_f2p: float
    planner_b_f2p: float
    planner_a_p2p: float
    planner_b_p2p: float
    delta: float
    planner_a_trial: str = ""
    planner_b_trial: str = ""
    errors: list[str] = field(default_factory=list)


def _run_wrap_pipeline(
    source_dataset: Path,
    output_dir: Path,
    mode: str,
    plans_dir: Path | None,
    task_filter: str | None,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    cmd = [
        sys.executable,
        str(_WRAP_PIPELINE),
        "--source",
        str(source_dataset),
        "--mode",
        mode,
        "--output",
        str(output_dir),
    ]
    if plans_dir is not None:
        cmd.extend(["--plans-dir", str(plans_dir)])
    if task_filter:
        cmd.extend(["--filter", task_filter])
    logger.info("wrap_pipeline (%s): %s", mode, " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"wrap_pipeline {mode} failed ({proc.returncode}): {proc.stderr[:500]}")


def _score_trials(task_names: list[str], trials_dir: Path) -> dict[str, dict[str, Any]]:
    trials = collect_trial_dirs(trials_dir)
    out: dict[str, dict[str, Any]] = {}
    for name in task_names:
        trial = trials.get(name)
        if trial is None:
            out[name] = {"f2p_score": 0.0, "p2p_score": 0.0, "trial_dir": ""}
            continue
        parsed = parse_trial_result(trial)
        out[name] = {
            "f2p_score": float(parsed["f2p_score"]),
            "p2p_score": float(parsed["p2p_score"]),
            "trial_dir": str(trial),
        }
    return out


def score_tasks(
    task_names: list[str],
    planner_a_trials: Path,
    planner_b_trials: Path,
) -> list[Score]:
    a_scores = _score_trials(task_names, planner_a_trials)
    b_scores = _score_trials(task_names, planner_b_trials)
    out: list[Score] = []
    for name in task_names:
        a = a_scores[name]
        b = b_scores[name]
        out.append(
            Score(
                task_name=name,
                planner_a_f2p=a["f2p_score"],
                planner_b_f2p=b["f2p_score"],
                planner_a_p2p=a["p2p_score"],
                planner_b_p2p=b["p2p_score"],
                delta=a["f2p_score"] - b["f2p_score"],
                planner_a_trial=str(a["trial_dir"]),
                planner_b_trial=str(b["trial_dir"]),
            )
        )
    return out


def write_back(
    candidates_dir: Path,
    scores: list[Score],
    cfg: ScorerConfig,
) -> None:
    """Merge each Score into its candidate JSON under ``planning_scores``.

    Does NOT set a ``planning_task`` flag; tagging is a separate step.
    """
    by_task_name: dict[str, Path] = {}
    for p in candidates_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        tn = data.get("task_name") or p.stem
        by_task_name[tn] = p

    for s in scores:
        path = by_task_name.get(s.task_name) or (candidates_dir / f"{s.task_name}.json")
        if not path.exists():
            logger.warning("no candidate JSON for %s", s.task_name)
            continue
        candidate = json.loads(path.read_text())
        candidate["planning_scores"] = {
            "planner_a_model": cfg.planner_a_model,
            "planner_b_model": cfg.planner_b_model,
            "implementer_model": cfg.implementer_model,
            "planner_a_f2p": round(s.planner_a_f2p, 3),
            "planner_b_f2p": round(s.planner_b_f2p, 3),
            "planner_a_p2p": round(s.planner_a_p2p, 3),
            "planner_b_p2p": round(s.planner_b_p2p, 3),
            "delta": round(s.delta, 3),
            "planner_a_trial": s.planner_a_trial,
            "planner_b_trial": s.planner_b_trial,
        }
        path.write_text(json.dumps(candidate, indent=2) + "\n")


def run_score(
    candidates_dir: str,
    dataset_dir: str,
    work_dir: str,
    *,
    planner_a_model: str = DEFAULT_PLANNER_A,
    planner_b_model: str = DEFAULT_PLANNER_B,
    implementer_model: str = DEFAULT_IMPLEMENTER,
    api_base: str | None = None,
    agent: str = "claude-code",
    task_filter: str | None = None,
    skip_harbor: bool = False,
    skip_synth: bool = False,
) -> dict[str, Any]:
    """Synth gold plan + run two planner trials + two implementer trials
    per candidate, record measurements. No tagging, no thresholds.
    """
    from craft_taskgen.planning.synth import run_synth

    cand_path = Path(candidates_dir)
    src_path = Path(dataset_dir)
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    if not cand_path.is_dir():
        raise FileNotFoundError(f"candidates_dir not found: {candidates_dir}")
    if not src_path.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    cfg = ScorerConfig(
        planner_a_model=planner_a_model,
        planner_b_model=planner_b_model,
        implementer_model=implementer_model,
        api_base=api_base or _default_api_base(),
        agent=agent,
    )

    all_names: list[str] = []
    for p in cand_path.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        tn = data.get("task_name") or p.stem
        all_names.append(tn)
    if task_filter:
        task_names = [n for n in all_names if n == task_filter or task_filter in n]
    else:
        task_names = all_names
    if not task_names:
        raise FileNotFoundError(f"no candidate JSONs matched in {candidates_dir} (filter={task_filter!r})")

    synth_result: dict[str, Any] = {"synthesized": [], "skipped": [], "failed": []}
    if not skip_synth:
        logger.info("synth: generating gold plans for %d candidate(s)", len(task_names))
        synth_result = run_synth(candidates_dir=candidates_dir, filter_task=task_filter)

    planner_dataset = work_path / "planner-dataset"
    plans_a_trials = work_path / "plans-a"
    plans_b_trials = work_path / "plans-b"
    impl_a_dataset = work_path / "impl-a-dataset"
    impl_b_dataset = work_path / "impl-b-dataset"
    impl_a_trials = work_path / "impl-a-trials"
    impl_b_trials = work_path / "impl-b-trials"

    if not skip_harbor:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set (source .env for NVIDIA gateway creds)")

        _run_wrap_pipeline(src_path, planner_dataset, "planner", None, task_filter)
        run_harbor(planner_dataset, plans_a_trials, planner_a_model, agent, cfg.api_base, task_filter)
        run_harbor(planner_dataset, plans_b_trials, planner_b_model, agent, cfg.api_base, task_filter)

        plans_a_resolved = resolve_harbor_output_dir(plans_a_trials)
        plans_b_resolved = resolve_harbor_output_dir(plans_b_trials)

        _run_wrap_pipeline(src_path, impl_a_dataset, "implementer", plans_a_resolved, task_filter)
        _run_wrap_pipeline(src_path, impl_b_dataset, "implementer", plans_b_resolved, task_filter)

        run_harbor(impl_a_dataset, impl_a_trials, implementer_model, agent, cfg.api_base, task_filter)
        run_harbor(impl_b_dataset, impl_b_trials, implementer_model, agent, cfg.api_base, task_filter)

    scores = score_tasks(task_names, impl_a_trials, impl_b_trials)
    write_back(cand_path, scores, cfg)

    return {
        "total": len(scores),
        "synth": synth_result,
        "scores": [
            {
                "task_name": s.task_name,
                "planner_a_f2p": round(s.planner_a_f2p, 3),
                "planner_b_f2p": round(s.planner_b_f2p, 3),
                "delta": round(s.delta, 3),
                "planner_a_trial": s.planner_a_trial,
                "planner_b_trial": s.planner_b_trial,
            }
            for s in scores
        ],
    }
