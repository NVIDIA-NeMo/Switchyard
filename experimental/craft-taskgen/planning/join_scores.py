#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join a planner job dir and an implementer job dir into a single report.

Input: two harbor job dirs from the iterative planning pipeline.
  - Planner job dir: trials contain plan.md + planner reward (file recall).
  - Implementer job dir: trials contain F2P reward.

Output (written to --output):
  - reward.txt: single float, mean F2P across all implementer trials.
  - results.json: full per-task breakdown.
  - report.md: human-readable table.

Trials are joined by task_name (from result.json), not trial dir name,
because harbor appends a random suffix per trial.

Usage:
    python planning/05_join_scores.py \\
        --planner-job jobs/e2e-foo/planner/<harbor_subdir> \\
        --impl-job    jobs/e2e-foo/implementer/<harbor_subdir> \\
        --output      jobs/e2e-foo/results
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_job_dir(path: Path) -> Path:
    """Accept either a harbor parent dir (single timestamped child) or the inner dir."""
    if not path.exists():
        raise FileNotFoundError(f"job dir does not exist: {path}")
    if (path / "job.log").exists():
        return path
    children = [c for c in path.iterdir() if c.is_dir() and (c / "job.log").exists()]
    if len(children) == 1:
        return children[0]
    if len(children) == 0:
        raise FileNotFoundError(f"no harbor job dir found under {path}")
    raise ValueError(
        f"{path} has multiple harbor job dirs; pass the specific one: {[c.name for c in children]}"
    )


def _index_trials(job_dir: Path) -> dict[str, Path]:
    """Map task_name -> trial dir, using result.json as the source of truth."""
    index: dict[str, Path] = {}
    for child in sorted(job_dir.iterdir()):
        if not child.is_dir():
            continue
        result_path = child / "result.json"
        if not result_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            logger.warning("skipping %s: invalid result.json", child.name)
            continue
        task_name = result.get("task_name")
        if not task_name:
            continue
        if task_name in index:
            logger.warning("duplicate task %s in %s, keeping first", task_name, job_dir)
            continue
        index[task_name] = child
    return index


def _read_reward(trial_dir: Path) -> float | None:
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text())
    except json.JSONDecodeError:
        return None
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    return float(reward) if reward is not None else None


def _read_verifier_details(trial_dir: Path) -> dict:
    details_path = trial_dir / "verifier" / "results.json"
    if not details_path.exists():
        return {}
    try:
        return json.loads(details_path.read_text())
    except json.JSONDecodeError:
        return {}


def _extract_f2p_p2p(details: dict) -> tuple[dict, dict]:
    """Normalize the F2P/P2P section across the old flat and new nested formats."""
    f2p_block = details.get("f2p")
    if isinstance(f2p_block, dict):
        f2p_detail = {
            "passed": f2p_block.get("passed"),
            "total": f2p_block.get("total"),
            "score": f2p_block.get("score"),
            "failed_tests": f2p_block.get("failed_tests") or [],
        }
    else:
        passed = details.get("fail_to_pass")
        total = details.get("total_functional")
        score = passed / total if passed is not None and total not in (None, 0) else None
        f2p_detail = {
            "passed": passed,
            "total": total,
            "score": round(score, 3) if score is not None else None,
            "failed_tests": details.get("still_failing_tests") or [],
        }

    p2p_block = details.get("p2p")
    if isinstance(p2p_block, dict):
        p2p_detail = {
            "passed": p2p_block.get("passed"),
            "total": p2p_block.get("total"),
            "score": p2p_block.get("score"),
            "failed_tests": p2p_block.get("failed_tests") or [],
        }
    else:
        p2p_detail = {
            "passed": None,
            "total": None,
            "score": None,
            "failed_tests": details.get("regression_tests") or [],
        }
    return f2p_detail, p2p_detail


def _read_agent_config(trial_dir: Path) -> dict:
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return {}
    try:
        result = json.loads(result_path.read_text())
    except json.JSONDecodeError:
        return {}
    config = result.get("config") or {}
    agent = config.get("agent") or {}
    return {
        "name": agent.get("name"),
        "model": agent.get("model_name"),
        "kwargs": agent.get("kwargs") or {},
    }


def join_scores(planner_job: Path, impl_job: Path, output_dir: Path) -> dict:
    planner_job = _resolve_job_dir(planner_job)
    impl_job = _resolve_job_dir(impl_job)

    planner_trials = _index_trials(planner_job)
    impl_trials = _index_trials(impl_job)

    all_tasks = sorted(set(planner_trials) | set(impl_trials))
    if not all_tasks:
        raise RuntimeError(f"no task trials found under {planner_job} or {impl_job}")

    planner_config = next((_read_agent_config(t) for t in planner_trials.values()), {})
    impl_config = next((_read_agent_config(t) for t in impl_trials.values()), {})

    per_task: dict[str, dict] = {}
    for task in all_tasks:
        planner_trial = planner_trials.get(task)
        impl_trial = impl_trials.get(task)

        plan_recall = _read_reward(planner_trial) if planner_trial else None
        reward = _read_reward(impl_trial) if impl_trial else None
        impl_details = _read_verifier_details(impl_trial) if impl_trial else {}
        f2p_detail, p2p_detail = _extract_f2p_p2p(impl_details)

        per_task[task] = {
            "plan_recall": plan_recall,
            "reward": reward if reward is not None else 0.0,
            "f2p": f2p_detail,
            "p2p": p2p_detail,
            "planner_ok": planner_trial is not None and plan_recall is not None,
            "impl_ok": impl_trial is not None and reward is not None,
        }

    reward_values = [v["reward"] for v in per_task.values()]
    f2p_fractions = [v["f2p"]["score"] for v in per_task.values() if v["f2p"]["score"] is not None]
    p2p_fractions = [v["p2p"]["score"] for v in per_task.values() if v["p2p"]["score"] is not None]
    recall_values = [v["plan_recall"] for v in per_task.values() if v["plan_recall"] is not None]

    summary = {
        "pass_rate": round(statistics.mean(reward_values), 4) if reward_values else 0.0,
        "n_passed": sum(1 for v in reward_values if v >= 1.0),
        "n_tasks": len(all_tasks),
        "mean_f2p_fraction": (round(statistics.mean(f2p_fractions), 4) if f2p_fractions else None),
        "mean_p2p_fraction": (round(statistics.mean(p2p_fractions), 4) if p2p_fractions else None),
        "mean_plan_recall": (round(statistics.mean(recall_values), 4) if recall_values else None),
        "n_planner_ok": sum(1 for v in per_task.values() if v["planner_ok"]),
        "n_impl_ok": sum(1 for v in per_task.values() if v["impl_ok"]),
    }

    report = {
        "summary": summary,
        "planner": planner_config,
        "implementer": impl_config,
        "job_dirs": {
            "planner": str(planner_job),
            "implementer": str(impl_job),
        },
        "per_task": per_task,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reward.txt").write_text(f"{summary['pass_rate']:.4f}\n")
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "report.md").write_text(_render_markdown(report))

    return report


def _render_markdown(report: dict) -> str:
    summary = report["summary"]
    planner = report["planner"]
    impl = report["implementer"]

    def _pct(value: float | None) -> str:
        return f"{value:.1%}" if value is not None else "n/a"

    lines = [
        "# Iterative Planning E2E Report",
        "",
        f"- Planner: `{planner.get('name')}` / `{planner.get('model')}`",
        f"- Implementer: `{impl.get('name')}` / `{impl.get('model')}`",
        f"- Tasks: {summary['n_tasks']} "
        f"(planner ok: {summary['n_planner_ok']}, impl ok: {summary['n_impl_ok']})",
        f"- **Pass rate (binary reward): {summary['pass_rate']:.4f} "
        f"({summary['n_passed']}/{summary['n_tasks']})**",
        f"- Mean F2P fraction: {_pct(summary['mean_f2p_fraction'])}",
        f"- Mean P2P fraction: {_pct(summary['mean_p2p_fraction'])}",
        "- Mean plan recall: "
        + (f"{summary['mean_plan_recall']:.4f}" if summary["mean_plan_recall"] is not None else "n/a"),
        "",
        "| Task | Reward | F2P | P2P | Plan recall | Regressions | Planner | Impl |",
        "|------|--------|-----|-----|-------------|-------------|---------|------|",
    ]
    for task, v in sorted(report["per_task"].items()):
        reward = f"{v['reward']:.1f}"
        f2p = v["f2p"]
        f2p_str = (
            f"{f2p['passed']}/{f2p['total']} ({f2p['score']:.1%})"
            if f2p["passed"] is not None and f2p["total"] is not None and f2p["score"] is not None
            else "-"
        )
        p2p = v["p2p"]
        p2p_str = (
            f"{p2p['passed']}/{p2p['total']} ({p2p['score']:.1%})"
            if p2p["passed"] is not None and p2p["total"] is not None and p2p["score"] is not None
            else "-"
        )
        recall = f"{v['plan_recall']:.3f}" if v["plan_recall"] is not None else "-"
        regressions = len(p2p.get("failed_tests") or [])
        planner_ok = "ok" if v["planner_ok"] else "FAIL"
        impl_ok = "ok" if v["impl_ok"] else "FAIL"
        lines.append(
            f"| {task} | {reward} | {f2p_str} | {p2p_str} | {recall} | "
            f"{regressions} | {planner_ok} | {impl_ok} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-job", type=Path, required=True)
    parser.add_argument("--impl-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = join_scores(args.planner_job, args.impl_job, args.output)
    summary = report["summary"]
    print(
        f"pass rate: {summary['pass_rate']:.4f} "
        f"({summary['n_passed']}/{summary['n_tasks']}) "
        f"| impl ok: {summary['n_impl_ok']}/{summary['n_tasks']} "
        f"| wrote: {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
