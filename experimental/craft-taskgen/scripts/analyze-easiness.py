# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prototype: score harbor-lab trajectories against the proposed easiness heuristics.

Two modes:

  --mode fixtures (default)
    Reads a deep-dive fixture bundle (tmp2/dd-fixtures-20260422/) with
    pre-captured harbor_lab/hlab_*.md outputs. Small cohort, quick turnaround.

  --mode run_dir
    Reads a live pipeline run directory (e.g. harbor-tasks/craft-tools-v4/runs/<ts>/)
    and its `state.json`, shells out to harbor-lab for each task that made it to
    Opus triage, and scores the trajectory. Designed to run on the VM with live
    artifacts. Needs `harbor-lab` on PATH (or $HARBOR_LAB pointing to the binary)
    and the jobs/ tree still present.

Either mode produces objective flags derived from already-captured harbor-lab
outputs (tool-sequence, edits, metrics). No LLM calls, no pipeline integration
— this is exploratory analysis to calibrate thresholds before wiring into the
pipeline as `TaskState.easiness_flag_efficiency` (task i).

Heuristics (objective, reproducible). **All are gated on reward=1.0** — a
failing trial can't be "too easy" regardless of trajectory shape.

Calibrated against the 2026-04-17 run (43 tasks, 29 eligible):

STRONG (rare recipe-writing — calibrated to be never-firing on legitimate work):
    no_exploration    : total Grep+Read < 10.  (Cohort min was 9; <10 picks
                        up tasks that essentially didn't explore at all.)
    zero_iteration    : num_pytest_runs == 0.  (The "run-once-to-confirm"
                        pattern at the end is normal; >= 1 means the agent
                        at least verified its work.)

MEDIUM (statistical outliers, p10 thresholds):
    fast_wall_time    : wall <= global p10 of the cohort.
    low_turns         : total tool calls <= global p10 of the cohort.

DROPPED (didn't pan out on real data):
    few_edits         : num_edits <= 2 — fires on 100% false positives in
                        calibration (minimal patches to complex tasks).

Auto-reject rule:
    auto_reject = reward == 1.0 AND (
                    no_exploration OR zero_iteration
                    OR (fast_wall_time AND low_turns)
                  )

Soft flag rule (individual p10 without corroboration — human review queue):
    soft_flag   = reward == 1.0 AND NOT auto_reject AND (fast_wall_time OR low_turns)

Usage:
    uv run python scripts/analyze-easiness.py \\
        --fixtures tmp2/dd-fixtures-20260422 \\
        --output easiness_analysis.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrajectoryStats:
    task_id: str
    bucket: str
    opus_score: str
    num_edits: int
    num_tool_calls: int
    num_greps: int
    num_reads: int
    num_bash: int
    num_pytest_runs: int
    exploration_before_first_edit: int
    wall_time_s: float | None
    reward: float | None
    f2p_passed: int | None
    f2p_total: int | None


def _read_or_empty(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


_EDITS_COUNT_RE = re.compile(r"\(claude-code,\s*(?P<n>\d+)\s*edits?\)")
_TOOL_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*(?P<tool>\[?\w+\]?)\s*\|\s*(?P<args>.*?)\s*\|$")
_WALL_TIME_RE = re.compile(r"Mean agent execution\s*\|\s*([\d.]+)s")


def _parse_tool_sequence(md: str) -> tuple[list[tuple[str, str]], int]:
    """Return [(tool, args), ...] and the total-calls count from the header."""
    calls: list[tuple[str, str]] = []
    total_calls = 0
    total_m = re.search(r"\(claude-code,\s*(?P<n>\d+)\s*calls?\)", md)
    if total_m:
        total_calls = int(total_m.group("n"))
    for line in md.splitlines():
        m = _TOOL_ROW_RE.match(line)
        if m:
            calls.append((m.group("tool"), m.group("args")))
    return calls, total_calls


def _is_pytest_bash(args: str) -> bool:
    a = args.lower()
    return "pytest" in a


def _stats_from_outputs(
    *,
    task_id: str,
    bucket: str,
    opus_score: str,
    edits_md: str,
    tool_seq_md: str,
    metrics_md: str,
    reward_json_text: str,
) -> TrajectoryStats:
    """Parse harbor-lab output + reward.json text into a TrajectoryStats. The
    source of each string (fixture file vs live subprocess) is the caller's
    concern."""
    m = _EDITS_COUNT_RE.search(edits_md)
    num_edits = int(m.group("n")) if m else 0

    calls, total_calls = _parse_tool_sequence(tool_seq_md)

    num_greps = sum(1 for t, _ in calls if t == "Grep")
    num_reads = sum(1 for t, _ in calls if t == "Read")
    num_bash = sum(1 for t, _ in calls if t == "Bash")
    num_pytest = sum(1 for t, a in calls if t == "Bash" and _is_pytest_bash(a))

    exploration_before_first_edit = 0
    for t, _ in calls:
        if t == "Edit":
            break
        if t in ("Grep", "Read"):
            exploration_before_first_edit += 1

    wall_m = _WALL_TIME_RE.search(metrics_md)
    wall_time_s = float(wall_m.group(1)) if wall_m else None

    reward: float | None = None
    f2p_passed: int | None = None
    f2p_total: int | None = None
    if reward_json_text.strip():
        try:
            d = json.loads(reward_json_text)
            reward = d.get("reward")
            f2p_passed = d.get("f2p_passed")
            f2p_total = d.get("f2p_total")
        except json.JSONDecodeError:
            pass

    return TrajectoryStats(
        task_id=task_id,
        bucket=bucket,
        opus_score=opus_score,
        num_edits=num_edits,
        num_tool_calls=total_calls,
        num_greps=num_greps,
        num_reads=num_reads,
        num_bash=num_bash,
        num_pytest_runs=num_pytest,
        exploration_before_first_edit=exploration_before_first_edit,
        wall_time_s=wall_time_s,
        reward=reward,
        f2p_passed=f2p_passed,
        f2p_total=f2p_total,
    )


def stats_for_sample(sample_dir: Path) -> TrajectoryStats:
    """Fixture-mode: read pre-captured harbor_lab/hlab_*.md + trial/verifier/reward.json."""
    info = json.loads(_read_or_empty(sample_dir / "fixture_info.json") or "{}")
    return _stats_from_outputs(
        task_id=info.get("task_id", sample_dir.name),
        bucket=info.get("bucket", "?"),
        opus_score=info.get("opus_score", ""),
        edits_md=_read_or_empty(sample_dir / "harbor_lab" / "hlab_edits.md"),
        tool_seq_md=_read_or_empty(sample_dir / "harbor_lab" / "hlab_tool_sequence.md"),
        metrics_md=_read_or_empty(sample_dir / "harbor_lab" / "hlab_metrics.md"),
        reward_json_text=_read_or_empty(sample_dir / "trial" / "verifier" / "reward.json"),
    )


def _resolve_harbor_lab() -> str:
    env = os.environ.get("HARBOR_LAB")
    if env and os.access(env, os.X_OK):
        return env
    on_path = shutil.which("harbor-lab")
    if on_path:
        return on_path
    # Known checkout locations: $HOME/Documents/vscode/harbor-lab (macOS dev
    # machines) and /data/projects/harbor-lab (craftbench VMs). Both use
    # uv-managed .venv/bin/harbor-lab.
    candidates = [
        os.path.expanduser("~/Documents/vscode/harbor-lab/.venv/bin/harbor-lab"),
        "/data/projects/harbor-lab/.venv/bin/harbor-lab",
    ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise RuntimeError(
        "harbor-lab binary not found. Install harbor-lab, set $HARBOR_LAB, "
        "or put it on PATH. Tried: $HARBOR_LAB, PATH, and "
        f"{candidates}"
    )


def _run_hlab(bin_path: str, *args: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run([bin_path, *args], capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except (subprocess.TimeoutExpired, OSError):
        return ""


def stats_for_live_task(task_id: str, task_state: dict, hlab_bin: str) -> TrajectoryStats | None:
    """Run-dir mode: shell out to harbor-lab to capture stats from a live trial dir.

    Returns None if the task has no Opus trial or if harbor-lab can't find it.
    """
    opus_trial_dir = task_state.get("opus_trial_dir", "")
    if not opus_trial_dir:
        return None
    # harbor-lab takes the JOB dir (parent of the trial dir)
    job_dir = str(Path(opus_trial_dir).parent)
    if not Path(job_dir).is_dir():
        return None

    edits_md = _run_hlab(hlab_bin, "edits", f"{job_dir}/", "--format", "markdown")
    # IMPORTANT: no --tail — we need the full trajectory to count
    # Grep+Read calls before the first Edit, which appears near the start.
    tool_seq_md = _run_hlab(hlab_bin, "tool-sequence", f"{job_dir}/", "--text", "--format", "markdown")
    metrics_md = _run_hlab(hlab_bin, "metrics", f"{job_dir}/", "--format", "markdown")

    reward_json_text = ""
    reward_path = Path(opus_trial_dir) / "verifier" / "reward.json"
    if reward_path.is_file():
        try:
            reward_json_text = reward_path.read_text(errors="replace")
        except OSError:
            pass

    stage = task_state.get("stage", "")
    return _stats_from_outputs(
        task_id=task_id,
        bucket=stage,
        opus_score=task_state.get("opus_score", ""),
        edits_md=edits_md,
        tool_seq_md=tool_seq_md,
        metrics_md=metrics_md,
        reward_json_text=reward_json_text,
    )


def _percentile(values: list[float], p: float) -> float | None:
    """Simple no-interp percentile. Returns None on empty input."""
    vs = sorted(v for v in values if v is not None)
    if not vs:
        return None
    idx = int(max(0, min(len(vs) - 1, round(p / 100.0 * (len(vs) - 1)))))
    return vs[idx]


def classify(stats: TrajectoryStats, wall_p10: float | None, turns_p10: float | None) -> dict:
    """Apply the proposed easiness heuristics. All gated on reward=1.0 —
    a failing trial cannot be "too easy" regardless of trajectory shape.
    Returns flag booleans + human-readable reasons.

    `wall_p10` and `turns_p10` are the 10th-percentile thresholds from the
    cohort (only the bottom decile trips the medium flags — calibrated
    against the 2026-04-17 run where p20 over-flagged at 41%).
    """
    flags: dict = {
        "no_exploration": False,
        "zero_iteration": False,
        "fast_wall_time": False,
        "low_turns": False,
        "strong": False,
        "medium_combined": False,
        "auto_reject_candidate": False,
        "soft_flag_only": False,
    }
    reasons: list[str] = []

    if stats.reward != 1.0:
        return {"flags": flags, "reasons": ["reward != 1.0, skipped"]}

    total_exploration = stats.num_greps + stats.num_reads
    no_exploration = total_exploration < 10
    flags["no_exploration"] = no_exploration
    if no_exploration:
        reasons.append(f"only {total_exploration} Grep+Read (<10)")

    zero_iteration = stats.num_pytest_runs == 0
    flags["zero_iteration"] = zero_iteration
    if zero_iteration:
        reasons.append("no pytest runs in trajectory")

    fast_wall = False
    if stats.wall_time_s is not None and wall_p10 is not None:
        fast_wall = stats.wall_time_s <= wall_p10
        if fast_wall:
            reasons.append(f"wall {stats.wall_time_s:.0f}s <= p10 {wall_p10:.0f}s")
    flags["fast_wall_time"] = fast_wall

    low_turns = False
    if stats.num_tool_calls and turns_p10 is not None:
        low_turns = stats.num_tool_calls <= turns_p10
        if low_turns:
            reasons.append(f"{stats.num_tool_calls} calls <= p10 {turns_p10:.0f}")
    flags["low_turns"] = low_turns

    flags["strong"] = no_exploration or zero_iteration
    flags["medium_combined"] = fast_wall and low_turns
    flags["auto_reject_candidate"] = flags["strong"] or flags["medium_combined"]
    flags["soft_flag_only"] = not flags["auto_reject_candidate"] and (fast_wall or low_turns)

    return {"flags": flags, "reasons": reasons}


def _load_fixture_mode_stats(fixtures_dir: Path) -> list[TrajectoryStats]:
    samples_dir = fixtures_dir / "samples"
    if not samples_dir.is_dir():
        print(f"ERROR: {samples_dir} not found", file=sys.stderr)
        sys.exit(1)
    return [stats_for_sample(d) for d in sorted(samples_dir.iterdir()) if d.is_dir()]


def _load_run_dir_stats(run_dir: Path) -> list[TrajectoryStats]:
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        print(f"ERROR: {state_path} not found", file=sys.stderr)
        sys.exit(1)
    try:
        hlab_bin = _resolve_harbor_lab()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {state_path}...")
    state = json.loads(state_path.read_text())
    tasks = state.get("tasks", {})

    # Eligible tasks: made it to Opus triage or past. Ignore evaluated/rejected earlier.
    eligible_stages = {
        "opus_smoke_tested",
        "opus_triaged",
        "haiku_smoke_tested",
        "accepted",
        "needs_fix",
    }
    eligible = [
        (tid, t) for tid, t in tasks.items() if t.get("stage") in eligible_stages and t.get("opus_trial_dir")
    ]
    print(f"Scoring {len(eligible)} tasks via harbor-lab at {hlab_bin}")
    print(f"  eligible stages: {sorted(eligible_stages)}")

    out: list[TrajectoryStats] = []
    for i, (tid, tstate) in enumerate(eligible, 1):
        if i % 10 == 0:
            print(f"  ... {i}/{len(eligible)}")
        s = stats_for_live_task(tid, tstate, hlab_bin)
        if s is not None:
            out.append(s)
    print(f"Scored {len(out)} tasks")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["fixtures", "run_dir"],
        default="fixtures",
        help="fixtures: read pre-captured bundle. run_dir: shell out to harbor-lab on a live run.",
    )
    parser.add_argument(
        "--fixtures",
        default="tmp2/dd-fixtures-20260422",
        help="(mode=fixtures) Path to the deep-dive fixture bundle",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="(mode=run_dir) Path to a pipeline run dir, e.g. harbor-tasks/craft-tools-v4/runs/<ts>/",
    )
    parser.add_argument(
        "--output",
        default="easiness_analysis.csv",
        help="CSV path (empty to skip)",
    )
    args = parser.parse_args()

    if args.mode == "fixtures":
        stats_list = _load_fixture_mode_stats(Path(args.fixtures))
    else:
        if not args.run_dir:
            print("ERROR: --mode run_dir requires --run-dir <path>", file=sys.stderr)
            return 1
        stats_list = _load_run_dir_stats(Path(args.run_dir))

    # Global percentile thresholds. Medium-flag threshold is p10 (bottom
    # decile) — p20 over-flagged in calibration against the 2026-04-17 run.
    # Only the `reward=1.0` cohort contributes to the distribution, since
    # those are the tasks the flags actually apply to.
    eligible_stats = [s for s in stats_list if s.reward == 1.0]
    wall_values = [s.wall_time_s for s in eligible_stats if s.wall_time_s is not None]
    turn_values = [float(s.num_tool_calls) for s in eligible_stats if s.num_tool_calls]
    wall_p10 = _percentile(wall_values, 10)
    turns_p10 = _percentile(turn_values, 10)

    wall_med = _percentile(wall_values, 50)
    turns_med = _percentile(turn_values, 50)
    print(f"Cohort: {len(stats_list)} samples ({len(eligible_stats)} eligible reward=1.0)")
    if wall_values and wall_p10 is not None and wall_med is not None:
        print(
            f"  wall_time_s: min={min(wall_values):.0f}, p10={wall_p10:.0f}, "
            f"median={wall_med:.0f}, max={max(wall_values):.0f}"
        )
    if turn_values and turns_p10 is not None and turns_med is not None:
        print(
            f"  tool_calls : min={int(min(turn_values))}, p10={int(turns_p10)}, "
            f"median={int(turns_med)}, max={int(max(turn_values))}"
        )
    print()

    rows: list[dict] = []
    for s in stats_list:
        c = classify(s, wall_p10, turns_p10)
        flagged = c["flags"].get("auto_reject_candidate", False)
        soft = c["flags"].get("soft_flag_only", False)
        eligible = s.reward == 1.0
        if not eligible:
            marker = "skip-r!=1"
        elif flagged:
            marker = "AUTO-REJECT"
        elif soft:
            marker = "soft-flag"
        else:
            marker = "clean"

        wall_str = f"{s.wall_time_s:.0f}s" if s.wall_time_s is not None else "?"
        print(f"  [{marker:>11}] {s.bucket:<22} {s.task_id}")
        print(
            f"    score={s.opus_score}  reward={s.reward}  edits={s.num_edits}  "
            f"calls={s.num_tool_calls}  greps={s.num_greps}  reads={s.num_reads}  "
            f"pytest={s.num_pytest_runs}  explore_pre_edit={s.exploration_before_first_edit}  "
            f"wall={wall_str}"
        )
        if c["reasons"]:
            print(f"    flags: {'; '.join(c['reasons'])}")

        rows.append(
            {
                **asdict(s),
                **{f"flag_{k}": v for k, v in c["flags"].items()},
                "reasons": "; ".join(c["reasons"]),
                "marker": marker,
            }
        )

    # Summary (eligible = reward=1.0 only — easiness heuristics don't apply to failing trials)
    n_total = len(rows)
    n_eligible = sum(1 for r in rows if r["marker"] != "skip-r!=1")
    n_auto = sum(1 for r in rows if r["marker"] == "AUTO-REJECT")
    n_soft = sum(1 for r in rows if r["marker"] == "soft-flag")
    n_clean = sum(1 for r in rows if r["marker"] == "clean")
    print()
    print(
        f"Summary (of {n_total} total): {n_eligible} eligible (reward=1.0) → "
        f"{n_auto} auto-reject, {n_soft} soft-flag, {n_clean} clean"
    )

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
