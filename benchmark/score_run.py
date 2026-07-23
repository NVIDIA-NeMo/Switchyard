#!/usr/bin/env python3
"""Score a benchmark run directory via the stage-router Rust scorer.

Reads trajectory.json from each completed task, feeds each tool-use turn through:
  DimensionCollector.process(ctx, request)  →  pick_capable_first / pick_efficient_first

The picker functions replicate live routing exactly: overrides, scorer, and fall_open.

Usage:
    uv run python benchmark/score_run.py --run benchmark/tb_runs/<run-name>
    uv run python benchmark/score_run.py --run benchmark/tb_runs/<run-name> \\
        --output /tmp/scores.jsonl --threshold 0.20 --window 3
"""
import argparse
import asyncio
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from switchyard.lib.processors.stage_router.dimensions import from_signal
from switchyard.lib.processors.stage_router.picker import (
    CAPABLE,
    pick_capable_first,
    pick_efficient_first,
)
from switchyard.lib.processors.stage_router.scorer import score as scorer_score
from switchyard_rust.components import DimensionCollector, get_tool_result_signal
from switchyard_rust.core import ChatRequest, ProxyContext

RECENT_WINDOW = 3


def _tool_use_id(message: str) -> str:
    parts = message.split()
    return parts[-1] if len(parts) >= 2 else f"tu_{id(message)}"


async def score_trajectory(
    traj: dict,
    reward: float | None,
    task_name: str,
    trial_name: str,
    run_id: str,
    recent_window: int,
    confidence_threshold: float,
) -> list[dict]:
    steps = traj.get("steps", [])

    # One DimensionCollector per task — accumulates state across turns
    dc = DimensionCollector(recent_window=recent_window)
    await dc.startup()

    rows: list[dict] = []
    messages: list[dict] = []

    for s in steps:
        if s["source"] == "user" and not (s.get("extra") or {}).get("is_sidechain"):
            messages.append({"role": "user", "content": [{"type": "text", "text": s["message"]}]})
            break

    if not messages:
        return rows

    for s in steps:
        if s["source"] != "agent":
            continue
        extra = s.get("extra") or {}
        if extra.get("is_sidechain"):
            continue

        tool_name = extra.get("tool_use_name")
        if not tool_name:
            text = s.get("message", "")
            if text:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            continue

        tool_use_id = _tool_use_id(s.get("message", ""))
        raw_args = extra.get("raw_arguments") or {}
        is_error = bool(extra.get("tool_result_is_error", False))

        metadata = extra.get("metadata") or {}
        raw_result = metadata.get("raw_tool_result") or {}
        content = raw_result.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)

        messages.append({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": raw_args}],
        })
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error}],
        })

        # Build Anthropic ChatRequest and process once through DimensionCollector
        ctx = ProxyContext()
        request = ChatRequest.anthropic({
            "model": "claude-opus-4-8",
            "max_tokens": 8096,
            "messages": messages,
        })
        await dc.process(ctx, request)

        signal = get_tool_result_signal(ctx)
        if signal is None:
            continue

        # Raw score for analysis: wrong signals score positive (→CAPABLE),
        # progress signals negative (→EFFICIENT). Picker-independent, so the
        # histogram shows the capable/efficient separation directly.
        dims = from_signal(signal)
        sr = scorer_score(dims)  # fixed weights; threshold dials corroboration

        # Actual picker decisions on the same ctx — both just read the signal,
        # no extra dc.process() calls needed
        tier_cf = await pick_capable_first(ctx, confidence_threshold)
        tier_ef = await pick_efficient_first(ctx, confidence_threshold)

        rows.append({
            "task_name":        task_name,
            "trial_name":       trial_name,
            "run_id":           run_id,
            "reward":           reward,
            "turn_depth":       signal.turn_depth,
            "score":            sr.score,
            "confidence":       sr.confidence,
            "tool_name":        tool_name,
            "is_error":         is_error,
            "write_count":      signal.write_count,
            "edit_count":       signal.edit_count,
            "read_count":       signal.read_count,
            "no_error_streak":  signal.no_error_streak,
            "pure_bash_streak": signal.pure_bash_streak,
            "tests_passed":     signal.tests_passed,
            "pick_cf":          tier_cf,   # CAPABLE=1, EFFICIENT=0
            "pick_ef":          tier_ef,
        })

    return rows


def band(score: float, threshold: float) -> str:
    if score >= threshold:
        return "strong_clear"
    if score > 0:
        return "strong_uncertain"
    if score <= -threshold:
        return "weak_clear"
    if score < 0:
        return "weak_uncertain"
    return "zero"


def write_per_task_csv(all_rows: list[dict], csv_path: Path, threshold: float) -> None:
    groups: dict[tuple, list] = defaultdict(list)
    for r in all_rows:
        groups[(r["run_id"], r["task_name"])].append(r)

    fieldnames = [
        "run_id", "task_name", "reward", "n_turns",
        "mean_score", "mean_confidence",
        "pct_strong_clear", "pct_strong_uncertain", "pct_weak_uncertain", "pct_weak_clear",
        "opus_pct_cf", "nemotron_pct_cf", "opus_pct_ef", "nemotron_pct_ef",
    ]
    rows_out = []
    for (run_id, task_name), turns in sorted(groups.items()):
        n = len(turns)
        scores = [t["score"] for t in turns]
        confs  = [t["confidence"] for t in turns]
        reward = turns[0]["reward"]
        bands  = [band(s, threshold) for s in scores]
        n_sc = bands.count("strong_clear")
        n_su = bands.count("strong_uncertain")
        n_wu = bands.count("weak_uncertain")
        n_wc = bands.count("weak_clear")
        n_cf_opus = sum(1 for t in turns if t["pick_cf"] == CAPABLE)
        n_ef_opus = sum(1 for t in turns if t["pick_ef"] == CAPABLE)
        rows_out.append({
            "run_id":               run_id,
            "task_name":            task_name,
            "reward":               reward,
            "n_turns":              n,
            "mean_score":           round(mean(scores), 4),
            "mean_confidence":      round(mean(confs), 4),
            "pct_strong_clear":     round(n_sc / n, 4),
            "pct_strong_uncertain": round(n_su / n, 4),
            "pct_weak_uncertain":   round(n_wu / n, 4),
            "pct_weak_clear":       round(n_wc / n, 4),
            "opus_pct_cf":          round(n_cf_opus / n, 4),
            "nemotron_pct_cf":      round(1 - n_cf_opus / n, 4),
            "opus_pct_ef":          round(n_ef_opus / n, 4),
            "nemotron_pct_ef":      round(1 - n_ef_opus / n, 4),
        })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Per-task CSV → {csv_path}  ({len(rows_out)} rows)")


async def score_run(run_dir: Path, recent_window: int, threshold: float) -> list[dict]:
    jobs_dirs = list(run_dir.glob("jobs/*/")) or [run_dir]
    run_id = run_dir.name

    all_rows: list[dict] = []
    for jobs_dir in jobs_dirs:
        task_dirs = [d for d in jobs_dir.iterdir() if d.is_dir() and d.name != "verifier"]
        print(f"{jobs_dir.name}: {len(task_dirs)} task dirs")

        for task_dir in sorted(task_dirs):
            traj_path   = task_dir / "agent" / "trajectory.json"
            result_path = task_dir / "result.json"
            if not traj_path.exists():
                continue

            traj = json.loads(traj_path.read_text())
            reward: float | None = None
            task_name = task_dir.name
            if result_path.exists():
                res = json.loads(result_path.read_text())
                task_name = res.get("task_name") or task_name
                r = (res.get("verifier_result") or {}).get("rewards", {}).get("reward")
                reward = float(r) if r is not None else None

            rows = await score_trajectory(
                traj, reward, task_name, task_dir.name, run_id, recent_window, threshold
            )
            all_rows.extend(rows)
            print(f"  {task_dir.name}: {len(rows)} turns, reward={reward}")

    return all_rows


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",       required=True, help="Path to run directory under benchmark/tb_runs/")
    parser.add_argument("--output",    help="Per-turn JSONL (default: /tmp/<run-name>-scores.jsonl)")
    parser.add_argument("--csv",       help="Per-task CSV (default: /tmp/<run-name>-per-task.csv)")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--window",    type=int,   default=RECENT_WINDOW)
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    jsonl_out = Path(args.output or f"/tmp/{run_dir.name}-scores.jsonl")
    csv_out   = Path(args.csv    or f"/tmp/{run_dir.name}-per-task.csv")

    all_rows = await score_run(run_dir, args.window, args.threshold)

    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_out, "w") as fh:
        for row in all_rows:
            fh.write(json.dumps(row) + "\n")
    print(f"Per-turn JSONL → {jsonl_out}  ({len(all_rows)} rows)")

    if all_rows:
        write_per_task_csv(all_rows, csv_out, args.threshold)
        total = len(all_rows)
        scores = [r["score"] for r in all_rows]
        cf_opus = sum(1 for r in all_rows if r["pick_cf"] == CAPABLE)
        ef_opus = sum(1 for r in all_rows if r["pick_ef"] == CAPABLE)
        print(f"\nGlobal summary ({total} turns, threshold={args.threshold}):")
        for bn in ["strong_clear", "strong_uncertain", "weak_uncertain", "weak_clear"]:
            n = sum(1 for s in scores if band(s, args.threshold) == bn)
            print(f"  {bn:<22} {n:>5}  ({100*n/total:.1f}%)")
        print(f"  cf → Opus: {cf_opus}/{total} ({100*cf_opus/total:.1f}%)")
        print(f"  ef → Opus: {ef_opus}/{total} ({100*ef_opus/total:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
