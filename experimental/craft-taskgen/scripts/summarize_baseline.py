# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""summarize_baseline.py — F2P / P2P / %resolved report from Harbor trial dirs.

Walks one or more trial roots, groups trials by model, and emits a markdown
table per model. Convention: infra-fail trials (no `verifier/reward.json`,
unparseable, or missing-from-cohort) count as `reward=0, f2p=0, p2p=0`.

Default cohort = `references/v2b-tasks.txt` (resolved relative to this
script's parent directory). The denominator is fixed by the cohort size;
any task not seen for a model counts as an infra fail.

Override with `--cohort-tasks-file PATH` (one task name per line) to use a
different cohort, `--exclude TASKID[,TASKID...]` to drop tasks ad-hoc from
whichever cohort is in effect, or `--no-cohort-check` to disable cohort
filling and use trials-seen as the denominator.

Multi-trial input: when multiple iter dirs of the same model are passed,
all trials per (model, task) are aggregated. Per-task pass rate =
passes_t / n_iters_expected, with missing trials counted as fails. The
headline `% resolved` is the mean of per-task pass rates; `±` is the
terminal-bench-style stderr √(Σ p̂_t(1−p̂_t)) / n_tasks. F2P / P2P also
report a per-iter sample std alongside their means.

Gateway-flake detection: for each baseline's aggregate `result.json`, the
script reads `exception_stats` and classifies each excepted trial. Trials
killed by upstream gateway issues (codex hitting a model-capacity error or
exhausting stream-retries on an org TPM cap; claude-code surfacing a
non-null `api_error_status`) are flagged loudly because their pass/fail
verdicts do not reflect model capability — they should be rerun. Other
exception classes (agent-timeout, verifier-timeout, infra) are noted but
not flagged as gateway issues.

Efficiency metrics (turns, tokens, wall-clock) are read from
`agent/trajectory.json::final_metrics`. Codex and opencode write that file
natively; **claude-code requires** running `harbor-lab rebuild-trajectories
<job_dir>` first to populate it from `claude-code.txt`. The script warns
loudly when claude-code trials lack trajectory.json so the operator knows
to rebuild. Without trajectory.json, those rows show `—` for efficiency
cells (the underlying scoring still works).

Usage:
  uv run python scripts/summarize_baseline.py <root>... \\
      [--alias RUN_DIR=LABEL]... [--cohort-tasks-file PATH] [--exclude TASKID,...] \\
      [--no-cohort-check]

Examples:
  # Default — pin to current canonical cohort, missing tasks counted as infra-fail
  uv run python scripts/summarize_baseline.py <root>

  # Same canonical cohort but drop two tasks ad-hoc for this run
  uv run python scripts/summarize_baseline.py <root> \\
      --exclude t2v3-FA800f-colmodernvbert-multimodal-integration,t2v3-LE282c-robomme-env-integration

  # Override cohort entirely (e.g. score against the historical 92-task list)
  uv run python scripts/summarize_baseline.py <root> --cohort-tasks-file path/to/other-cohort.txt

  # No cohort filling — denominator = trials seen
  uv run python scripts/summarize_baseline.py <root> --no-cohort-check

  # vllm-served models report generic model_info.name="model"; alias to a clean label
  uv run python scripts/summarize_baseline.py <root> \\
      --alias v2b-newmodel-opencode=newmodel-v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Default cohort lives in <repo>/references/. Resolved at runtime so the
# script works from any cwd inside the checkout.
DEFAULT_COHORT_FILE = Path(__file__).resolve().parent.parent / "references" / "v2b-tasks.txt"

# Default post-hoc skip list — tests that were added to a task's
# tests/f2p_skip.txt AFTER baseline runs were performed. Applied at score
# time so historical runs can be rescored as if the skip had been in
# effect, without rerunning trials.
DEFAULT_POSTHOC_SKIPS_FILE = Path(__file__).resolve().parent.parent / "references" / "v2b-posthoc-skips.txt"

# vllm-served models report `model_info.name = "model"` and don't disambiguate
# distinct deployments; fall back to the run-group dir name in that case.
GENERIC_MODEL_NAMES = {"", "model", "vllm/model", "default", "unknown"}

# Narrow patterns indicating a trial was killed by an upstream gateway/inference
# issue rather than a real model failure. Each pattern lives in JSON envelopes
# the agent emits, so they don't false-positive on tool-error strings.
#
# codex: emits `"turn.failed"` after exhausting its (low) default retry budget
#        on these specific upstream messages.
# claude-code: handles 529/429 transparently with 10 retries; only records a
#        non-null `api_error_status` if all retries were exhausted.
# opencode: no validated patterns yet — leave for future calibration.
# Anchored on the opening `"` so the pattern is tied to a JSON string field (not
# free text in tool output), but doesn't require a closing `"` since the
# upstream messages have trailing text (e.g. "...at capacity. Please try a different model.").
_CODEX_TURN_FAILED = '"turn.failed"'
_CODEX_GATEWAY_MSGS = (
    '"Selected model is at capacity',  # gpt-5.x model capacity, not retried
    '"Rate limit reached for ',  # org TPM cap, stream_max_retries exhausted
    '"stream disconnected before completion: Rate limit',  # same root cause, different surface
)
_CLAUDE_API_ERROR_PATTERNS = (
    '"api_error_status":"overloaded_error"',
    '"api_error_status":"rate_limit_error"',
)

# Efficiency metrics extracted per trial for the optional --csv output and the
# per-model efficiency table.
#
# Primary source: `agent/trajectory.json::final_metrics` — uniform schema
# across all three agents. Codex and opencode write trajectory.json natively
# during the run. Claude-code only writes `agent/claude-code.txt` (NDJSON);
# trajectory.json must be rebuilt from it via:
#
#     harbor-lab rebuild-trajectories <job_dir>
#
# (See the harbor-lab CLI — `metrics`, `rebuild-trajectories`,
# `tool-sequence`, etc.) Without that step, claude-code trials show `—` for
# all efficiency metrics. The script warns loudly when claude-code trajectory.json
# is missing so the operator knows to rebuild.
#
# Fallback sources (used when trajectory.json is missing):
#   - tokens: `result.json::agent_result.n_{input,cache,output}_tokens`
#     (tools-side schema; claude-code writes nulls here too)
#     OR `reward.json::process_metrics.*_tokens` (search-side schema)
#   - wall_clock_sec: `result.json::agent_execution.{started,finished}_at` diff
#     OR `process_metrics.execution_time_sec`
#
# tool_call_count is NOT in `final_metrics` directly — `_read_trajectory_metrics`
# derives it by walking the per-step `tool_calls` lists and summing their
# lengths (a single step can carry multiple parallel tool calls).
_EFFICIENCY_KEYS = (
    "agent_steps",  # turns
    "tool_call_count",
    "input_tokens",
    "cached_tokens",
    "output_tokens",
    "wall_clock_sec",
)


def _read_trajectory_metrics(trial_dir: Path) -> dict:
    """Pull per-trial token + step + tool counts from `agent/trajectory.json`.

    Returns a dict with keys from `final_metrics` plus a derived
    `tool_call_count` (sum of `len(step.tool_calls)` across all steps).
    Returns {} if file is missing or unparseable.

    final_metrics schema (uniform across agents):
        total_prompt_tokens     — input tokens
        total_completion_tokens — output tokens
        total_cached_tokens     — cached tokens
        total_steps             — agent steps (turns)

    `tool_call_count` isn't in final_metrics so we walk the steps list. A
    single step can contain multiple tool_calls (claude-code parallel
    invocations), so we sum list-lengths rather than counting steps.

    Codex and opencode write trajectory.json natively. Claude-code requires
    `harbor-lab rebuild-trajectories <job_dir>` to populate it from
    claude-code.txt.
    """
    p = trial_dir / "agent" / "trajectory.json"
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out = dict(d.get("final_metrics") or {})
    out["tool_call_count"] = sum(len(s.get("tool_calls") or []) for s in d.get("steps") or [])
    return out


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _wall_clock_sec(result: dict, process_metrics: dict) -> float | None:
    """Wall-clock for the agent's solve attempt.

    Prefer `result.json::agent_execution.{started_at,finished_at}` because it
    excludes environment_setup + verifier overhead and is present on every
    agent. Falls back to `process_metrics.execution_time_sec` (opencode/codex
    populate it; claude-code does not).
    """
    ax = result.get("agent_execution") or {}
    start = _parse_iso(ax.get("started_at"))
    end = _parse_iso(ax.get("finished_at"))
    if start and end:
        return (end - start).total_seconds()
    raw = process_metrics.get("execution_time_sec")
    return float(raw) if raw is not None else None


def find_trial_dirs(roots: list[Path]) -> list[Path]:
    """Yield every trial dir (containing `verifier/reward.json` + `result.json`).

    Follows symlinks so that aggregator dirs assembled from `ln -s` of
    individual iter dirs (e.g. `evals-final/<model>/iter1 -> /path/to/baseline-...`)
    are walked correctly. We use `os.walk(followlinks=True)` because
    `pathlib.Path.rglob` does not recurse into symlinked directories on
    Python 3.12.
    """
    import os as _os

    out: list[Path] = []
    seen_real: set[str] = set()  # cycle protection (resolved real path)
    for root in roots:
        if not root.exists():
            print(f"WARN: missing root {root}", file=sys.stderr)
            continue
        if (root / "verifier" / "reward.json").exists() and (root / "result.json").exists():
            out.append(root)
            continue
        for dirpath, _dirnames, filenames in _os.walk(root, followlinks=True):
            # Cycle guard via real-path memoization.
            real = _os.path.realpath(dirpath)
            if real in seen_real:
                continue
            seen_real.add(real)
            if "reward.json" in filenames:
                # `dirpath` is the verifier/ dir → trial = parent of verifier/.
                if Path(dirpath).name != "verifier":
                    continue
                trial = Path(dirpath).parent
                if (trial / "result.json").exists():
                    out.append(trial)
    return out


def load_trial(
    trial_dir: Path,
    aliases: dict[str, str],
    posthoc_skips: dict[str, set[str]] | None = None,
) -> dict | None:
    """Read one trial. Returns None if reward.json or result.json is unparseable.

    posthoc_skips: optional `task_id -> {test_paths}` map. Tests in this map
    are rescored as if they had been in the task's tests/f2p_skip.txt at
    run time: failing tests are removed (decrementing f2p_total), passing
    tests decrement both f2p_passed and f2p_total. The 'resolved' flag is
    recomputed from the post-skip f2p/p2p counts.
    """
    try:
        reward = json.loads((trial_dir / "verifier" / "reward.json").read_text())
        result = json.loads((trial_dir / "result.json").read_text())
        # f2p_failed/p2p_failed moved out of reward.json (harbor>=0.13.1
        # pydantic validation rejects list values). Merge them back in from
        # the side-car if present so downstream readers see one combined view.
        details_path = trial_dir / "verifier" / "reward-details.json"
        if details_path.exists():
            try:
                details = json.loads(details_path.read_text())
                for key in ("f2p_failed", "p2p_failed"):
                    if key in details and key not in reward:
                        reward[key] = details[key]
            except (OSError, json.JSONDecodeError):
                pass
    except (OSError, json.JSONDecodeError):
        return None
    task_name = result.get("task_name") or trial_dir.name.rsplit("-", 1)[0]
    agent_info = result.get("agent_info") or {}
    model_info = agent_info.get("model_info") or {}
    raw_model = model_info.get("name") or agent_info.get("name") or ""

    try:
        run_group = trial_dir.parent.parent.name
    except (AttributeError, IndexError):
        run_group = ""

    if run_group in aliases:
        model = aliases[run_group]
    elif raw_model in GENERIC_MODEL_NAMES:
        model = run_group or "unknown"
    else:
        model = raw_model

    f2p_passed = int(reward.get("f2p_passed") or 0)
    f2p_total = int(reward.get("f2p_total") or 0)
    p2p_passed = int(reward.get("p2p_passed") or 0)
    p2p_total = int(reward.get("p2p_total") or 0)
    resolved = bool(reward.get("resolved"))
    posthoc_applied = 0

    skip_set = (posthoc_skips or {}).get(task_name) or set()
    if skip_set:
        f2p_failed = list(reward.get("f2p_failed") or [])
        for test_path in skip_set:
            if test_path in f2p_failed:
                # Test was failing — removing from skip list shrinks the
                # denominator without changing pass count.
                f2p_failed.remove(test_path)
                f2p_total -= 1
                posthoc_applied += 1
            # If test wasn't in f2p_failed, it was either passing or already
            # skipped at run time. We can't distinguish without re-reading
            # the source-of-truth fail_to_pass.txt — but for our intended
            # use (rescoring runs that predate the skip-list update), the
            # interesting case is `was failing → now skipped`. Other paths
            # are no-ops on the score, so we leave them alone.
        # Recompute resolved: all remaining F2P pass + no P2P regression.
        if f2p_total > 0:
            resolved = (f2p_passed >= f2p_total) and (p2p_passed >= p2p_total)
        # Guard against pathological zero-total post-skip: if all F2P got
        # skipped, we can't make a meaningful pass call — leave resolved
        # as it was reported.

    process_metrics = reward.get("process_metrics") or {}
    agent_result = result.get("agent_result") or {}
    rec = {
        "task": task_name,
        "model": model,
        "resolved": resolved,
        "f2p_passed": f2p_passed,
        "f2p_total": f2p_total,
        "p2p_passed": p2p_passed,
        "p2p_total": p2p_total,
        "posthoc_skips_applied": posthoc_applied,
    }
    # Fallback efficiency sources (used if trajectory.json is missing).
    rec["input_tokens"] = agent_result.get("n_input_tokens") or process_metrics.get("input_tokens")
    rec["cached_tokens"] = agent_result.get("n_cache_tokens") or process_metrics.get("cached_tokens")
    rec["output_tokens"] = agent_result.get("n_output_tokens") or process_metrics.get("output_tokens")
    rec["agent_steps"] = process_metrics.get("agent_steps")
    rec["tool_call_count"] = process_metrics.get("tool_call_count")
    rec["wall_clock_sec"] = _wall_clock_sec(result, process_metrics)

    # Preferred source: agent/trajectory.json::final_metrics — uniform across
    # codex/opencode/claude-code (the latter only after `harbor-lab
    # rebuild-trajectories` has been run). Overrides the fallbacks above when
    # present so the data is consistent across agents.
    fm = _read_trajectory_metrics(trial_dir)
    rec["has_trajectory"] = bool(fm)
    if fm:
        if fm.get("total_prompt_tokens") is not None:
            rec["input_tokens"] = fm["total_prompt_tokens"]
        if fm.get("total_completion_tokens") is not None:
            rec["output_tokens"] = fm["total_completion_tokens"]
        if fm.get("total_cached_tokens") is not None:
            rec["cached_tokens"] = fm["total_cached_tokens"]
        if fm.get("total_steps") is not None:
            rec["agent_steps"] = fm["total_steps"]
        if fm.get("tool_call_count") is not None:
            rec["tool_call_count"] = fm["tool_call_count"]

    # Agent name — used to surface a helpful warning if claude-code trials
    # have no trajectory.json (the user needs to run harbor-lab to rebuild).
    agent_name = ""
    cfg = trial_dir / "config.json"
    if cfg.is_file():
        try:
            agent_name = (json.loads(cfg.read_text()).get("agent") or {}).get("name") or ""
        except (OSError, json.JSONDecodeError):
            agent_name = ""
    rec["agent_name"] = agent_name
    return rec


def _classify_exception(trial_dir: Path, agent_name: str, exception_type: str) -> str:
    """Classify an excepted trial. Returns one of:
    'gateway-flake'      — trial killed by upstream gateway/capacity/rate-limit
    'agent-timeout'      — agent hit the wall-clock trial cap
    'verifier-timeout'   — verifier (test runner) timed out
    'infra'              — RuntimeError before agent ran (docker build fail, etc.)
    'agent-exit-other'   — non-zero exit not matching gateway pattern
    'other'              — unrecognized exception type
    """
    if exception_type == "AgentTimeoutError":
        return "agent-timeout"
    if exception_type == "VerifierTimeoutError":
        return "verifier-timeout"
    if exception_type == "RuntimeError":
        return "infra"
    if exception_type != "NonZeroAgentExitCodeError":
        return "other"

    # Agent crashed; look at agent log for upstream-error signature.
    if agent_name == "codex":
        log = trial_dir / "agent" / "codex.txt"
    elif agent_name == "claude-code":
        log = trial_dir / "agent" / "claude-code.txt"
    elif agent_name == "opencode":
        log = trial_dir / "agent" / "opencode.txt"
    else:
        return "agent-exit-other"

    if not log.is_file():
        return "agent-exit-other"

    try:
        content = log.read_text(errors="replace")
    except OSError:
        return "agent-exit-other"

    if agent_name == "codex":
        if _CODEX_TURN_FAILED in content and any(m in content for m in _CODEX_GATEWAY_MSGS):
            return "gateway-flake"
    elif agent_name == "claude-code":
        if any(p in content for p in _CLAUDE_API_ERROR_PATTERNS):
            return "gateway-flake"
    # opencode: no validated patterns; fall through.
    return "agent-exit-other"


def _trial_passed_verifier(trial_dir: Path) -> bool | None:
    """Return True if verifier marked this trial resolved, False if not, None if unknown."""
    rj = trial_dir / "verifier" / "reward.json"
    if not rj.is_file():
        return None
    try:
        return bool(json.loads(rj.read_text()).get("resolved"))
    except (OSError, json.JSONDecodeError):
        return None


def _detect_baseline_exceptions(
    baseline_dir: Path, aliases: dict[str, str]
) -> dict[str, list[tuple[str, str, str, bool | None]]]:
    """Read baseline_dir/result.json and return per-model lists of excepted trials.

    Returns: model -> [(trial_id, exception_type, classification, verifier_resolved), ...]
    where verifier_resolved is True if the trial got reward=1 despite the agent
    exception, False if reward=0, None if no reward.json was produced.
    """
    rj = baseline_dir / "result.json"
    if not rj.is_file():
        return {}
    try:
        data = json.loads(rj.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    # Resolve model name the same way load_trial does, but cheaper: we just need
    # the run_group label and apply aliases. Fall back to the run_group name.
    try:
        run_group = baseline_dir.parent.name
    except (AttributeError, IndexError):
        run_group = ""
    model = aliases.get(run_group, run_group or "unknown")

    out: dict[str, list[tuple[str, str, str, bool | None]]] = {model: []}
    for ev in (data.get("stats", {}).get("evals", {}) or {}).values():
        for exc_type, trial_ids in (ev.get("exception_stats") or {}).items():
            for tid in trial_ids or []:
                trial_dir = baseline_dir / tid
                # Read agent name from this trial's config.json — most reliable
                # source even when reward.json is missing.
                agent_name = ""
                cfg = trial_dir / "config.json"
                if cfg.is_file():
                    try:
                        cfg_data = json.loads(cfg.read_text())
                        agent_name = (cfg_data.get("agent") or {}).get("name") or ""
                    except (OSError, json.JSONDecodeError):
                        pass
                klass = _classify_exception(trial_dir, agent_name, exc_type)
                resolved = _trial_passed_verifier(trial_dir)
                out[model].append((tid, exc_type, klass, resolved))
    return out


def _read_cohort_file(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def _read_posthoc_skips_file(path: Path) -> dict[str, set[str]]:
    """Parse a post-hoc skips file. Format: <task_id>\\t<test_path> per line.

    Returns: task_id -> {test_paths}
    """
    out: dict[str, set[str]] = defaultdict(set)
    for ln in path.read_text().splitlines():
        ln = ln.rstrip()
        if not ln or ln.startswith("#"):
            continue
        # Split on the first tab; fall back to first run of whitespace if no tab.
        if "\t" in ln:
            task_id, test_path = ln.split("\t", 1)
        else:
            parts = ln.split(None, 1)
            if len(parts) != 2:
                continue
            task_id, test_path = parts
        out[task_id.strip()].add(test_path.strip())
    return dict(out)


def _parse_excludes(values: list[str]) -> set[str]:
    out: set[str] = set()
    for v in values:
        for tok in v.split(","):
            tok = tok.strip()
            if tok:
                out.add(tok)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("roots", nargs="+", type=Path, help="Trial dirs OR parent dirs containing trials.")
    ap.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="RUN_DIR=LABEL",
        help="Override model label for trials whose run-group dir "
        "(parent.parent.name) matches RUN_DIR. Repeatable. Useful "
        "when model_info.name is generic (vllm-served).",
    )
    ap.add_argument(
        "--cohort-tasks-file",
        type=Path,
        default=None,
        help="Cohort definition file (one task name per line; '#' comments OK). "
        f"Default: references/{DEFAULT_COHORT_FILE.name}",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="TASKID[,TASKID...]",
        help="Drop these task IDs from the cohort for this run. Repeatable; "
        "each value may be a comma-separated list. Useful for ad-hoc 'what if "
        "we drop these?' analyses without committing a new cohort file.",
    )
    ap.add_argument(
        "--no-cohort-check",
        action="store_true",
        help="Disable cohort filling. Denominator becomes trials seen "
        "per model (no fixed-cohort infra-fail accounting).",
    )
    ap.add_argument(
        "--posthoc-skips-file",
        type=Path,
        default=None,
        help="Post-hoc F2P test skip list (task_id<TAB>test_path per line). Tests "
        "listed here are rescored as if they had been in the task's "
        "tests/f2p_skip.txt at run time — useful for rescoring historical baselines "
        f"after the skip list is amended. Default: references/{DEFAULT_POSTHOC_SKIPS_FILE.name}",
    )
    ap.add_argument(
        "--no-posthoc-skips",
        action="store_true",
        help="Disable post-hoc skip rescoring. Read reward.json verbatim.",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write per-task CSV with one row per (model, task): resolved, "
        "f2p/p2p counts, agent_steps, tool_call_count, input/cached/output "
        "tokens, wall_clock_sec. Includes infra-fail rows (zero-filled) so "
        "every cohort task is represented per model.",
    )
    ap.add_argument(
        "--no-efficiency-table",
        action="store_true",
        help="Skip the per-model efficiency markdown table (turns / tools / "
        "tokens / wall-clock). Useful when only the headline reward summary "
        "is wanted.",
    )
    args = ap.parse_args()

    aliases: dict[str, str] = {}
    for a in args.alias:
        if "=" not in a:
            print(f"WARN: malformed --alias {a!r}; need RUN_DIR=LABEL", file=sys.stderr)
            continue
        k, v = a.split("=", 1)
        aliases[k.strip()] = v.strip()

    excludes = _parse_excludes(args.exclude)

    posthoc_skips: dict[str, set[str]] = {}
    posthoc_label = ""
    if not args.no_posthoc_skips:
        if args.posthoc_skips_file and args.no_posthoc_skips:
            print(
                "ERROR: --no-posthoc-skips and --posthoc-skips-file are mutually exclusive",
                file=sys.stderr,
            )
            return 2
        posthoc_path = args.posthoc_skips_file or DEFAULT_POSTHOC_SKIPS_FILE
        if posthoc_path.is_file():
            posthoc_skips = _read_posthoc_skips_file(posthoc_path)
            n_tests = sum(len(s) for s in posthoc_skips.values())
            posthoc_label = f"{posthoc_path} ({len(posthoc_skips)} tasks, {n_tests} tests)"
        elif args.posthoc_skips_file is not None:
            print(f"ERROR: --posthoc-skips-file {posthoc_path} not found", file=sys.stderr)
            return 2
        # Default file missing is fine; just no rescoring.

    cohort: set[str] | None = None
    cohort_label = ""
    if args.no_cohort_check:
        if args.cohort_tasks_file:
            print(
                "ERROR: --no-cohort-check and --cohort-tasks-file are mutually exclusive",
                file=sys.stderr,
            )
            return 2
        if excludes:
            print(
                "ERROR: --no-cohort-check and --exclude are mutually exclusive (no cohort to exclude from)",
                file=sys.stderr,
            )
            return 2
    else:
        cohort_path = args.cohort_tasks_file or DEFAULT_COHORT_FILE
        if not cohort_path.is_file():
            print(f"ERROR: cohort file {cohort_path} not found", file=sys.stderr)
            return 2
        cohort = _read_cohort_file(cohort_path)
        cohort_label = f"{cohort_path} ({len(cohort)} tasks)"
        if excludes:
            unknown = excludes - cohort
            if unknown:
                print(
                    f"WARN: --exclude tasks not present in cohort: {sorted(unknown)}",
                    file=sys.stderr,
                )
            applied = excludes & cohort
            cohort -= applied
            cohort_label += f", excludes: {len(applied)}"

    trial_dirs = find_trial_dirs(args.roots)
    print(f"Found {len(trial_dirs)} trial dirs across {len(args.roots)} root(s)", file=sys.stderr)

    # Accumulate every trial per (model, task). Multiple iter dirs of the same
    # model contribute multiple recs for the same task; the headline then
    # averages per-task pass rates over n_iters (matching the
    # terminal-bench k=N convention). For single-iter input the behavior is
    # the same as before — each task has one rec.
    by_model_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    skipped = 0
    for td in trial_dirs:
        rec = load_trial(td, aliases, posthoc_skips=posthoc_skips)
        if rec is None:
            skipped += 1
            continue
        rec["trial_dir"] = str(td)
        by_model_task[(rec["model"], rec["task"])].append(rec)
    if skipped:
        print(f"Skipped {skipped} unreadable trials", file=sys.stderr)

    by_model: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for (model, task), recs in by_model_task.items():
        by_model[model][task] = recs

    if not by_model:
        print("ERROR: no valid trials found", file=sys.stderr)
        return 2

    rows = []
    for model in sorted(by_model):
        trials = by_model[model]
        seen = set(trials)
        if cohort is not None:
            target = cohort
            missing = sorted(target - seen)
            extra = sorted(seen - target)
        else:
            target = seen
            missing = []
            extra = []

        # Per-task aggregation: each cohort task contributes one pass-rate
        # value (passes_t / n_iters_expected). Headline % resolved = mean
        # over tasks of those pass rates.
        #
        # Missing-trial policy: any expected trial that didn't produce a
        # result counts as a fail (contributes 0 to passes, 1 to denominator).
        # n_iters_expected = max recs observed across tasks for this model —
        # i.e., we infer the intended trial count per task from the densest
        # task and treat sparser tasks as having missing trials = fails.
        n_iters_expected = max((len(r) for r in trials.values()), default=1)
        infra = 0
        n_total_trials = 0
        n_missing_trials = 0
        per_task_pass_rates: list[float] = []
        per_task_f2p: list[float] = []
        per_task_p2p: list[float] = []
        for t in target:
            recs = trials.get(t) or []
            n_t_actual = len(recs)
            if n_t_actual == 0:
                # Cohort task with zero trials → all missing = all zeros.
                infra += 1
                n_missing_trials += n_iters_expected
                per_task_pass_rates.append(0.0)
                per_task_f2p.append(0.0)
                per_task_p2p.append(0.0)
                continue
            n_total_trials += n_t_actual
            n_missing_trials += max(0, n_iters_expected - n_t_actual)
            # Sum then divide by n_iters_expected: missing trials contribute 0.
            per_task_pass_rates.append(sum(1 for r in recs if r["resolved"]) / n_iters_expected)
            per_task_f2p.append(
                sum(r["f2p_passed"] / r["f2p_total"] if r["f2p_total"] else 0.0 for r in recs)
                / n_iters_expected
            )
            per_task_p2p.append(
                sum(r["p2p_passed"] / r["p2p_total"] if r["p2p_total"] else 1.0 for r in recs)
                / n_iters_expected
            )
        n_tasks = len(per_task_pass_rates)
        # tbench-style stderr: predicted std of one new iter's accuracy under
        # task-independent Bernoulli sampling, with p_t = passes_t / n_iters_t.
        # Verified against the published terminal-bench leaderboard
        # (codex/gpt-5.5: 82.0% ± 2.2 reproduced exactly by this formula).
        # NOT a CI on the mean — for that, divide by sqrt(n_iters).
        sum_pq = sum(p * (1 - p) for p in per_task_pass_rates)
        stderr_pct = (math.sqrt(sum_pq) / n_tasks * 100) if n_tasks else 0.0
        # F2P / P2P are continuous in [0,1]; the binary-pinning pathology
        # that motivates the tbench formula doesn't apply, so we report
        # plain sample std across the per-iter mean values. Group recs
        # by iter dir (parent of trial_dir) to compute one F2P / P2P
        # value per iter.
        recs_by_iter: dict[str, list[dict]] = defaultdict(list)
        for task_recs in trials.values():
            for r in task_recs:
                iter_id = str(Path(r.get("trial_dir", "")).parent) if r.get("trial_dir") else ""
                recs_by_iter[iter_id].append(r)
        per_iter_f2p_means: list[float] = []
        per_iter_p2p_means: list[float] = []
        for iter_recs in recs_by_iter.values():
            if not iter_recs:
                continue
            f2p_vals = [r["f2p_passed"] / r["f2p_total"] if r["f2p_total"] else 0.0 for r in iter_recs]
            p2p_vals = [r["p2p_passed"] / r["p2p_total"] if r["p2p_total"] else 1.0 for r in iter_recs]
            per_iter_f2p_means.append(sum(f2p_vals) / len(f2p_vals))
            per_iter_p2p_means.append(sum(p2p_vals) / len(p2p_vals))
        f2p_std = statistics.stdev(per_iter_f2p_means) if len(per_iter_f2p_means) > 1 else 0.0
        p2p_std = statistics.stdev(per_iter_p2p_means) if len(per_iter_p2p_means) > 1 else 0.0
        rows.append(
            {
                "model": model,
                "n_tasks": n_tasks,
                "n_total_trials": n_total_trials,
                "n_missing_trials": n_missing_trials,
                "n_iters_expected": n_iters_expected,
                "infra": infra,
                "f2p": (sum(per_task_f2p) / n_tasks) if n_tasks else 0.0,
                "p2p": (sum(per_task_p2p) / n_tasks) if n_tasks else 0.0,
                "f2p_std": f2p_std,
                "p2p_std": p2p_std,
                "resolved": (sum(per_task_pass_rates) / n_tasks * 100) if n_tasks else 0.0,
                "stderr": stderr_pct,
                "missing": len(missing),
                "extra": len(extra),
            }
        )

    print()
    if cohort_label:
        print(f"# Baseline summary — cohort: **{cohort_label}**")
    else:
        print("# Baseline summary — `--no-cohort-check` (denominator = trials seen per model)")
    if posthoc_label:
        print(f"  Post-hoc skips applied: {posthoc_label}")
    print()
    header = (
        "| Model | F2P (mean ± iter-std) | P2P (mean ± iter-std) "
        "| % resolved ± tbench | #tasks | #iters | #trials | #missing | #infra | #out-of-cohort |"
    )
    print(header)
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        f2p_cell = f"{r['f2p']:.3f} ± {r['f2p_std']:.3f}"
        p2p_cell = f"{r['p2p']:.3f} ± {r['p2p_std']:.3f}"
        resolved_cell = f"{r['resolved']:.1f}% ± {r['stderr']:.2f}"
        print(
            f"| `{r['model']}` | {f2p_cell} | {p2p_cell} | {resolved_cell} | "
            f"{r['n_tasks']} | {r['n_iters_expected']} | {r['n_total_trials']} | "
            f"{r['n_missing_trials']} | {r['infra']} | {r['extra']} |"
        )
    print()

    # Efficiency table — mean per-task across non-missing trials. Infra-fail
    # rows (missing trials) are excluded from these means since zero-filling
    # turn-counts and tokens would be meaningless. The `n` column shows how
    # many trials contributed to each model's row.
    if not args.no_efficiency_table:
        eff_rows = []
        for model in sorted(by_model):
            trials = by_model[model]
            # Flat list across all (task, trial) pairs — efficiency means are
            # over individual trials, so multi-iter input contributes more
            # observations per task.
            all_recs = [r for recs in trials.values() for r in recs]
            row = {"model": model}
            for key in _EFFICIENCY_KEYS:
                vals = [r[key] for r in all_recs if r.get(key) is not None]
                row[key] = sum(vals) / len(vals) if vals else None
                row[f"{key}_n"] = len(vals)
            row["n_max"] = max((row[f"{k}_n"] for k in _EFFICIENCY_KEYS), default=0)
            eff_rows.append(row)
        if any(r["n_max"] > 0 for r in eff_rows):
            print("## Efficiency (mean per task across non-missing trials)")
            print()
            print("| Model | Turns | Tools | Input tok | Cached tok | Output tok | Wall (s) | n |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|")
            for r in eff_rows:

                def fmt(v, decimals=0):
                    return "—" if v is None else f"{v:,.{decimals}f}"

                print(
                    f"| `{r['model']}` | {fmt(r['agent_steps'])} | {fmt(r['tool_call_count'])} | "
                    f"{fmt(r['input_tokens'])} | {fmt(r['cached_tokens'])} | {fmt(r['output_tokens'])} | "
                    f"{fmt(r['wall_clock_sec'], 1)} | {r['n_max']} |"
                )
            print()

    for r in rows:
        if cohort and r["missing"]:
            print(
                f"⚠ `{r['model']}` is missing {r['missing']} of {len(cohort)} cohort tasks "
                f"(counted as infra-fail with reward=0/f2p=0/p2p=0).",
                file=sys.stderr,
            )
        if cohort and r["extra"]:
            print(
                f"ℹ `{r['model']}` has {r['extra']} trials outside the cohort "
                f"(ignored in headline; pass --no-cohort-check to include).",
                file=sys.stderr,
            )

    # Surface trials whose trajectory.json is missing — efficiency cells will
    # fall back to whatever's in result.json/reward.json, which for claude-code
    # means nulls everywhere. The fix is `harbor-lab rebuild-trajectories
    # <job_dir>`. Group by (model, agent) and call out claude-code specifically
    # since that's the case that needs operator action.
    missing_trajectory: dict[tuple[str, str], int] = defaultdict(int)
    for (mdl, _task), recs in by_model_task.items():
        for rec in recs:
            if rec.get("has_trajectory") is False:
                agent = rec.get("agent_name") or "?"
                missing_trajectory[(mdl, agent)] += 1
    if missing_trajectory:
        print("", file=sys.stderr)
        print(
            "⚠ Missing trajectory.json — efficiency metrics (turns, tokens) won't be populated for these:",
            file=sys.stderr,
        )
        for (mdl, agent), n in sorted(missing_trajectory.items()):
            note = " — fix: `harbor-lab rebuild-trajectories <job_dir>`" if agent == "claude-code" else ""
            print(f"  {mdl} (agent={agent}): {n} trial(s){note}", file=sys.stderr)

    # Surface which trials were impacted by post-hoc skips and whether the
    # rescoring flipped a verdict (resolved=False before, but f2p/p2p shape
    # implies resolved=True after). Operators want to know if the skip-list
    # update actually moved any numbers — a no-op rescore is still useful
    # to record but shouldn't be silent.
    if posthoc_skips:
        flips_per_model: dict[str, set[str]] = defaultdict(set)
        applied_per_model: dict[str, int] = defaultdict(int)
        for (model, task), recs in by_model_task.items():
            for rec in recs:
                n_applied = rec.get("posthoc_skips_applied", 0)
                if n_applied:
                    applied_per_model[model] += n_applied
                    # Flip detection: rec["resolved"] is post-skip; the verdict
                    # would have been False pre-skip if the skipped tests were
                    # in f2p_failed (since we only decrement total when they
                    # were failing). Confirm that's the case by checking that
                    # the post-skip f2p_passed >= f2p_total and there was at
                    # least one skip applied for this trial. With multi-iter
                    # input, dedupe by task — we want "this task had a flip in
                    # at least one iter," not multiple counts of the same task.
                    if rec["resolved"] and (rec["f2p_passed"] >= rec["f2p_total"]):
                        flips_per_model[model].add(task)
        if applied_per_model:
            print("", file=sys.stderr)
            print("ℹ Post-hoc skip impact:", file=sys.stderr)
            for model in sorted(applied_per_model):
                n_apply = applied_per_model[model]
                flips = flips_per_model.get(model, [])
                flip_note = (
                    f", flipped {len(flips)} task(s) to resolved: {', '.join(sorted(flips))}" if flips else ""
                )
                print(f"  {model}: {n_apply} skip(s) applied{flip_note}", file=sys.stderr)

    # Walk each baseline dir's aggregate result.json for excepted trials and
    # classify each. Excepted means the agent or verifier raised an exception
    # somewhere in the trial pipeline; it does NOT necessarily mean the trial
    # failed verification — sometimes the agent had already made enough code
    # edits before crashing that the verifier still passes (reward=1.0).
    baseline_dirs = {td.parent for td in trial_dirs}
    excepted_per_model: dict[str, list[tuple[str, str, str, bool | None]]] = defaultdict(list)
    for bd in sorted(baseline_dirs):
        per_model = _detect_baseline_exceptions(bd, aliases)
        for model, items in per_model.items():
            excepted_per_model[model].extend(items)

    # Per-model breakdown by classification, with verifier outcome tracked too.
    # classification → ('display label', is_gateway_flake)
    klass_meta = {
        "gateway-flake": ("gateway-flake", True),
        "agent-timeout": ("agent-timeout", False),
        "verifier-timeout": ("verifier-timeout", False),
        "infra": ("infra (e.g. docker build)", False),
        "agent-exit-other": ("agent exit ≠0 (other)", False),
        "other": ("other", False),
    }

    if any(excepted_per_model.values()):
        print("", file=sys.stderr)
        print("# Excepted trials breakdown", file=sys.stderr)
        print(
            "  An exception during a trial does not always mean the verifier scored it as failed —",
            file=sys.stderr,
        )
        print(
            "  the agent may have made enough edits before crashing that tests still pass.",
            file=sys.stderr,
        )
        print(
            "  Counts below show: <total>  (verifier-resolved / verifier-failed / no-verdict).",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        for model in sorted(excepted_per_model):
            items = excepted_per_model[model]
            if not items:
                continue
            by_class: dict[str, list[tuple[str, str, bool | None]]] = defaultdict(list)
            for tid, exc_type, klass, resolved in items:
                by_class[klass].append((tid, exc_type, resolved))
            print(f"  {model}:", file=sys.stderr)
            for klass, group in sorted(by_class.items(), key=lambda kv: kv[0]):
                label, _is_gw = klass_meta.get(klass, (klass, False))
                n = len(group)
                passed = sum(1 for _, _, r in group if r is True)
                failed = sum(1 for _, _, r in group if r is False)
                no_verdict = sum(1 for _, _, r in group if r is None)
                print(
                    f"    {label}: {n}  ({passed} pass / {failed} fail / {no_verdict} no-verdict)",
                    file=sys.stderr,
                )

    # Loud warning ONLY for gateway-flakes — these are the ones the operator
    # should rerun. Other classes are visible in the breakdown above but don't
    # warrant the same treatment (timeouts and infra fails are real signal,
    # not measurement noise).
    gw_flakes_per_model = {
        m: [it for it in items if it[2] == "gateway-flake"] for m, items in excepted_per_model.items()
    }
    if any(gw_flakes_per_model.values()):
        print("", file=sys.stderr)
        print(
            "⚠️⚠️⚠️ GATEWAY-FLAKE DETECTED — trials killed by upstream gateway",
            file=sys.stderr,
        )
        print(
            "    capacity/rate-limit, not real model failures. Their pass/fail",
            file=sys.stderr,
        )
        print(
            "    verdicts don't reflect model capability — RERUN these for",
            file=sys.stderr,
        )
        print("    accurate measurement:", file=sys.stderr)
        for model, flakes in sorted(gw_flakes_per_model.items()):
            if not flakes:
                continue
            print(f"  {model} ({len(flakes)}):", file=sys.stderr)
            for tid, exc_type, _, resolved in flakes:
                resolved_note = (
                    "verifier still passed"
                    if resolved
                    else "verifier failed"
                    if resolved is False
                    else "no verifier verdict"
                )
                print(f"    • {tid}  [{exc_type}, {resolved_note}]", file=sys.stderr)

    # Per-task CSV. One row per (model, task) over the cohort (or trials-seen
    # if --no-cohort-check). Infra-fail rows are zero-filled on f2p/p2p so
    # downstream tooling sees every cohort task per model — efficiency
    # metrics are left blank since zero-filling them would bias means.
    if args.csv:
        target_tasks = cohort if cohort is not None else None
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "model",
                    "task",
                    "trial_dir",
                    "resolved",
                    "f2p_passed",
                    "f2p_total",
                    "p2p_passed",
                    "p2p_total",
                    "posthoc_skips_applied",
                    "agent_steps",
                    "tool_call_count",
                    "input_tokens",
                    "cached_tokens",
                    "output_tokens",
                    "wall_clock_sec",
                    "infra_fail",
                ]
            )
            for model in sorted(by_model):
                trials = by_model[model]
                tasks_to_emit = target_tasks if target_tasks is not None else set(trials)
                for task in sorted(tasks_to_emit):
                    recs = trials.get(task) or []
                    if not recs:
                        # Infra fail / missing trial: emit one zero-filled row
                        # per missing cohort task (no trial_dir).
                        w.writerow(
                            [
                                model,
                                task,
                                "",
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                1,
                            ]
                        )
                        continue
                    for rec in recs:
                        w.writerow(
                            [
                                model,
                                task,
                                rec.get("trial_dir", ""),
                                int(rec["resolved"]),
                                rec["f2p_passed"],
                                rec["f2p_total"],
                                rec["p2p_passed"],
                                rec["p2p_total"],
                                rec.get("posthoc_skips_applied", 0),
                                rec.get("agent_steps") if rec.get("agent_steps") is not None else "",
                                rec.get("tool_call_count") if rec.get("tool_call_count") is not None else "",
                                rec.get("input_tokens") if rec.get("input_tokens") is not None else "",
                                rec.get("cached_tokens") if rec.get("cached_tokens") is not None else "",
                                rec.get("output_tokens") if rec.get("output_tokens") is not None else "",
                                rec.get("wall_clock_sec") if rec.get("wall_clock_sec") is not None else "",
                                0,
                            ]
                        )
        print(f"Wrote per-task CSV → {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
