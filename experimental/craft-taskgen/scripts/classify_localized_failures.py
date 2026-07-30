# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""classify_localized_failures.py — automatic failure-mode typology for
fully-localized e2e failures.

Population: every (model, task) trial pair where the e2e agent achieved
`exam_file_recall == 1.0` (saw every gold-edited file) AND `e2e_resolved == 0`
(failed verification). 64 such pairs across codex55 / opus47 / haiku45 /
qwen36 on the v2b cohort.

For each pair, this script extracts a structured per-trial record by reading
the trial's `verifier/reward.json`, `verifier/test-stdout.txt`, the agent's
trajectory (codex `apply_patch` payloads, claude-code `Edit`/`Write` tool
calls, opencode `edit`/`write`), and the gold patch. From those, it computes
rule-based failure-mode flags and a primary failure class.

Failure-mode flags (multi-label):

  PATCH_APPLY_FAILED      — verifier output indicates patch couldn't be applied
                            (rare in practice — agents call apply tools directly,
                            not raw `git apply`)
  P2P_REGRESSION          — at least one previously-passing test now fails
  F2P_PARTIAL             — flipped some F2P tests but not all
  F2P_NONE                — no F2P tests now pass (no forward progress)
  TYPE_OR_IMPORT_ERROR    — pytest collected zero tests OR exited with
                            ImportError (signals broken module-level code)
  COMMITTED_NARROWER      — committed_file_recall < 1.0 despite
                            examined_file_recall = 1.0
                            (saw all the right files, edited fewer than gold)
  COMMITTED_BROADER       — agent edited files NOT in the gold set
                            (over-eager fix touching unrelated code)
  AGENT_DIFF_TINY         — agent's diff is <30% the size of gold patch
                            (insufficient edit volume — likely wrong approach)
  AGENT_DIFF_HUGE         — agent's diff is >300% gold patch size
                            (over-engineering / unrelated changes)
  EXIT_NONZERO            — agent process exited non-zero before completing

A primary class is assigned by precedence:
  PATCH_APPLY_FAILED > TYPE_OR_IMPORT_ERROR > P2P_REGRESSION > F2P_NONE >
  F2P_PARTIAL > (none — fallback)

Output: per-trial JSON records in --output-jsonl, one record per line. Plus a
tabular summary (CSV) of class distribution per model.

Usage:
  uv run python scripts/classify_localized_failures.py \\
      --implicit-csvs /tmp/e2e-implicit-search.csv /tmp/e2e-opus.csv \\
                      /tmp/e2e-haiku.csv /tmp/e2e-qwen.csv \\
      --e2e-roots /tmp/e2e-codex-full /tmp/e2e-opus /tmp/e2e-haiku /tmp/e2e-qwen \\
      --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v2b \\
      --output-jsonl docs/data/v2b-localized-failure-typology.jsonl \\
      --output-summary docs/data/v2b-localized-failure-summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Failure-mode codes
PATCH_APPLY_FAILED = "PATCH_APPLY_FAILED"
P2P_REGRESSION = "P2P_REGRESSION"
F2P_PARTIAL = "F2P_PARTIAL"
F2P_NONE = "F2P_NONE"
TYPE_OR_IMPORT_ERROR = "TYPE_OR_IMPORT_ERROR"
COMMITTED_NARROWER = "COMMITTED_NARROWER"
COMMITTED_BROADER = "COMMITTED_BROADER"
AGENT_DIFF_TINY = "AGENT_DIFF_TINY"
AGENT_DIFF_HUGE = "AGENT_DIFF_HUGE"
EXIT_NONZERO = "EXIT_NONZERO"

PRIMARY_CLASS_PRECEDENCE = [
    PATCH_APPLY_FAILED,
    TYPE_OR_IMPORT_ERROR,
    P2P_REGRESSION,
    F2P_NONE,
    F2P_PARTIAL,
]

ALL_FLAGS = [
    PATCH_APPLY_FAILED,
    P2P_REGRESSION,
    F2P_PARTIAL,
    F2P_NONE,
    TYPE_OR_IMPORT_ERROR,
    COMMITTED_NARROWER,
    COMMITTED_BROADER,
    AGENT_DIFF_TINY,
    AGENT_DIFF_HUGE,
    EXIT_NONZERO,
]


# ---------------------------------------------------------------------------
# Diff size estimation
# ---------------------------------------------------------------------------


def _gold_patch_size(task_dir: Path) -> int:
    """Lines added + removed in the gold patch."""
    p = task_dir / "solution" / "changes.patch"
    if not p.is_file():
        return 0
    n = 0
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- ")):
            n += 1
    return n


def _agent_diff_size_from_trajectory(traj: dict) -> int:
    """Estimate the agent's total edit volume in lines added+removed.

    Three sources, by tool:
      codex apply_patch: `input` is the codex DSL — count `+`/`-` lines
      claude-code Edit/Write/MultiEdit — count newlines in old/new strings
      opencode edit/write — count newlines in oldString/newString/content
    """
    n = 0
    for step in traj.get("steps") or []:
        for tc in step.get("tool_calls") or []:
            fn = tc.get("function_name", "")
            args = tc.get("arguments") or {}
            if fn == "apply_patch":
                # codex DSL
                for line in (args.get("input") or "").splitlines():
                    if line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- ")):
                        n += 1
                # unified-diff fallback
                for line in (args.get("patch") or "").splitlines():
                    if line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- ")):
                        n += 1
            elif fn in ("Edit", "MultiEdit"):
                for k in ("old_string", "new_string"):
                    s = args.get(k) or ""
                    n += s.count("\n") + (1 if s else 0)
            elif fn == "Write":
                s = args.get("content") or ""
                n += s.count("\n") + (1 if s else 0)
            elif fn in ("edit", "write"):
                for k in ("oldString", "newString", "content"):
                    s = args.get(k) or ""
                    n += s.count("\n") + (1 if s else 0)
    return n


def _load_trajectory(trial_dir: Path) -> dict | None:
    """Load ATIF trajectory or adapt claude-code.txt. Same as in score_e2e_implicit_search.py
    but inlined here for self-containment."""
    p = trial_dir / "agent" / "trajectory.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    cc = trial_dir / "agent" / "claude-code.txt"
    if cc.is_file():
        try:
            return _adapt_claude_code(cc)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _adapt_claude_code(p: Path) -> dict:
    steps: list[dict] = []
    pending: dict[str, dict] = {}
    sid = 0
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "assistant":
                msg = ev.get("message") or {}
                content = msg.get("content") or []
                text_buf: list[str] = []
                for blk in content:
                    btype = blk.get("type")
                    if btype == "text":
                        text_buf.append(blk.get("text") or "")
                    elif btype == "tool_use":
                        sid += 1
                        step = {
                            "step_id": sid,
                            "source": "agent",
                            "message": "\n".join(text_buf),
                            "tool_calls": [
                                {
                                    "tool_call_id": blk.get("id"),
                                    "function_name": blk.get("name", ""),
                                    "arguments": blk.get("input") or {},
                                }
                            ],
                        }
                        text_buf = []
                        pending[blk.get("id") or ""] = step
                        steps.append(step)
            elif ev.get("type") == "user":
                msg = ev.get("message") or {}
                content = msg.get("content") or []
                if isinstance(content, list):
                    for blk in content:
                        if blk.get("type") == "tool_result":
                            tcid = blk.get("tool_use_id")
                            payload = blk.get("content")
                            if isinstance(payload, list):
                                txt = "\n".join(p.get("text", "") for p in payload if isinstance(p, dict))
                            else:
                                txt = str(payload or "")
                            if tcid in pending:
                                pending[tcid].setdefault("observation", {}).setdefault("results", []).append(
                                    {"source_call_id": tcid, "content": txt}
                                )
    return {
        "schema_version": "claude-code-adapted",
        "agent": {"name": "claude-code"},
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Verifier output parsing
# ---------------------------------------------------------------------------

_PATCH_APPLY_FAIL_RE = re.compile(
    r"(error: patch failed|patch does not apply|fatal: corrupt patch|cannot apply patch)",
    re.IGNORECASE,
)
_IMPORT_ERROR_RE = re.compile(r"^(ImportError|ModuleNotFoundError|SyntaxError):", re.MULTILINE)
_COLLECTION_ERROR_RE = re.compile(r"errors? during collection|0 items collected", re.IGNORECASE)


def _check_verifier_output(test_stdout: str) -> tuple[bool, bool]:
    """Return (patch_apply_failed, type_or_import_error) flags."""
    patch_apply = bool(_PATCH_APPLY_FAIL_RE.search(test_stdout))
    type_or_import = bool(_IMPORT_ERROR_RE.search(test_stdout)) or bool(
        _COLLECTION_ERROR_RE.search(test_stdout)
    )
    return (patch_apply, type_or_import)


# ---------------------------------------------------------------------------
# Per-trial classifier
# ---------------------------------------------------------------------------


@dataclass
class TrialRecord:
    model: str
    task: str
    trial_path: str

    # Verifier outcome
    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0
    f2p_failed: list[str] = field(default_factory=list)
    p2p_failed: list[str] = field(default_factory=list)

    # Implicit-search numbers (carried from input CSV)
    exam_file_recall: float = 0.0
    exam_function_recall: float = 0.0
    comm_file_recall: float = 0.0
    comm_function_recall: float = 0.0
    n_examined_files: int = 0
    n_committed_files: int = 0
    gold_n_files: int = 0
    gold_n_functions: int = 0

    # Diff sizes
    gold_diff_lines: int = 0
    agent_diff_lines: int = 0
    agent_diff_ratio: float = 0.0  # agent / gold

    # Flags
    flags: list[str] = field(default_factory=list)
    primary_class: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def _classify(rec: TrialRecord) -> None:
    """Apply classification rules. Mutates `rec.flags` and `rec.primary_class`."""
    flags: set[str] = set()

    # F2P-based outcome flags
    if rec.f2p_total == 0:
        flags.add(TYPE_OR_IMPORT_ERROR)  # collected nothing — likely import-time crash
    else:
        if rec.f2p_passed == 0:
            flags.add(F2P_NONE)
        elif rec.f2p_passed < rec.f2p_total:
            flags.add(F2P_PARTIAL)
    # P2P regression: known failures > 0
    if rec.p2p_failed:
        flags.add(P2P_REGRESSION)

    # Localization width
    if rec.exam_file_recall >= 1.0 and rec.comm_file_recall < 1.0:
        flags.add(COMMITTED_NARROWER)
    if rec.n_committed_files > rec.gold_n_files:
        flags.add(COMMITTED_BROADER)

    # Diff-size flags
    if rec.gold_diff_lines > 0:
        ratio = rec.agent_diff_lines / rec.gold_diff_lines
        rec.agent_diff_ratio = round(ratio, 3)
        if ratio < 0.30:
            flags.add(AGENT_DIFF_TINY)
        elif ratio > 3.0:
            flags.add(AGENT_DIFF_HUGE)

    rec.flags = sorted(flags)

    # Primary class by precedence (first match wins)
    for c in PRIMARY_CLASS_PRECEDENCE:
        if c in flags:
            rec.primary_class = c
            return
    rec.primary_class = "UNCLASSIFIED"


# ---------------------------------------------------------------------------
# Population builder
# ---------------------------------------------------------------------------


def _build_trial_record(csv_row: dict, trial_dir: Path, task_dir: Path | None) -> TrialRecord | None:
    """Read all the artifacts and build a TrialRecord."""
    rj_path = trial_dir / "verifier" / "reward.json"
    if not rj_path.is_file():
        return None
    try:
        rj = json.loads(rj_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    # f2p_failed/p2p_failed live in reward-details.json under the new
    # SCORE_PY_TEMPLATE (split out for harbor>=0.13.1 pydantic compat).
    details_path = trial_dir / "verifier" / "reward-details.json"
    if details_path.is_file():
        try:
            details = json.loads(details_path.read_text())
            for key in ("f2p_failed", "p2p_failed"):
                if key in details and key not in rj:
                    rj[key] = details[key]
        except (OSError, json.JSONDecodeError):
            pass

    # Implicit-search numbers from the CSV row
    rec = TrialRecord(
        model=csv_row["agent"] + "/" + csv_row["model"][:80],
        task=csv_row["task"],
        trial_path=str(trial_dir),
        f2p_passed=int(rj.get("f2p_passed") or 0),
        f2p_total=int(rj.get("f2p_total") or 0),
        p2p_passed=int(rj.get("p2p_passed") or 0),
        p2p_total=int(rj.get("p2p_total") or 0),
        f2p_failed=list(rj.get("f2p_failed") or []),
        p2p_failed=list(rj.get("p2p_failed") or []),
        exam_file_recall=float(csv_row["exam_file_recall"] or 0),
        exam_function_recall=float(csv_row["exam_function_recall"] or 0),
        comm_file_recall=float(csv_row["comm_file_recall"] or 0),
        comm_function_recall=float(csv_row["comm_function_recall"] or 0),
        n_examined_files=int(csv_row["n_examined_files"] or 0),
        n_committed_files=int(csv_row["n_committed_files"] or 0),
        gold_n_files=int(csv_row["gold_n_files"] or 0),
        gold_n_functions=int(csv_row["gold_n_functions"] or 0),
    )

    # Gold diff size from task_dir
    if task_dir is not None and task_dir.is_dir():
        rec.gold_diff_lines = _gold_patch_size(task_dir)

    # Agent diff size from trajectory
    traj = _load_trajectory(trial_dir)
    if traj is not None:
        rec.agent_diff_lines = _agent_diff_size_from_trajectory(traj)

    # Verifier-output text flags
    test_stdout = trial_dir / "verifier" / "test-stdout.txt"
    if test_stdout.is_file():
        try:
            text = test_stdout.read_text(errors="replace")
            patch_fail, type_err = _check_verifier_output(text)
            if patch_fail:
                rec.flags.append(PATCH_APPLY_FAILED)
            if type_err:
                rec.flags.append(TYPE_OR_IMPORT_ERROR)
        except OSError:
            pass

    _classify(rec)
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--implicit-csvs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more CSVs from score_e2e_implicit_search.py (per-model).",
    )
    ap.add_argument(
        "--e2e-roots",
        nargs="+",
        type=Path,
        required=True,
        help="One or more roots holding the e2e job dirs (matched against trial_dir column).",
    )
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="v2b harbor-tasks dir (for solution/changes.patch lookup).",
    )
    ap.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Per-trial structured records, one JSON per line.",
    )
    ap.add_argument(
        "--output-summary",
        type=Path,
        required=True,
        help="Aggregate distribution CSV.",
    )
    ap.add_argument(
        "--include-all-failures",
        action="store_true",
        help="If set, include all e2e_resolved=0 trials (not only fully-localized ones). "
        "Default: only include trials with exam_file_recall=1.0.",
    )
    args = ap.parse_args()

    # Pre-resolve trial_dir → absolute path. The trial_dir column is a relative
    # path under one of the e2e-roots; we walk all roots to find the right one.
    def _find_trial(trial_path: str) -> Path | None:
        for root in args.e2e_roots:
            cand = root / "jobs" / trial_path
            if cand.is_dir():
                return cand
            cand2 = root / trial_path
            if cand2.is_dir():
                return cand2
        return None

    records: list[TrialRecord] = []
    skipped = 0
    for csv_path in args.implicit_csvs:
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row["e2e_resolved"] != "0":
                    continue
                if not args.include_all_failures and float(row["exam_file_recall"] or 0) < 1.0:
                    continue
                trial_dir = _find_trial(row["trial_dir"])
                if trial_dir is None:
                    skipped += 1
                    continue
                task_dir = args.tasks_dir / row["task"]
                rec = _build_trial_record(row, trial_dir, task_dir)
                if rec is None:
                    skipped += 1
                    continue
                records.append(rec)

    print(f"Built {len(records)} trial records ({skipped} skipped)", file=sys.stderr)

    # Write per-trial JSONL
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), default=str) + "\n")
    print(f"Wrote per-trial records → {args.output_jsonl}", file=sys.stderr)

    # Aggregate distribution
    by_model_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_model_flag: dict[str, Counter[str]] = defaultdict(Counter)
    overall_class: Counter[str] = Counter()
    overall_flag: Counter[str] = Counter()
    by_model_n: Counter[str] = Counter()
    for rec in records:
        m = rec.model
        by_model_n[m] += 1
        by_model_class[m][rec.primary_class] += 1
        overall_class[rec.primary_class] += 1
        for f in rec.flags:
            by_model_flag[m][f] += 1
            overall_flag[f] += 1

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("w", newline="") as fh:
        w = csv.writer(fh)
        # Header: model, n_trials, primary class counts, flag counts
        models = sorted(by_model_n)
        header = ["row_kind", "label"] + models + ["overall"]
        w.writerow(header)

        # Trial counts per model
        w.writerow(["trial_count", "n"] + [by_model_n[m] for m in models] + [sum(by_model_n.values())])

        # Primary class distribution
        for cls in PRIMARY_CLASS_PRECEDENCE + ["UNCLASSIFIED"]:
            row = ["primary_class", cls]
            for m in models:
                row.append(by_model_class[m].get(cls, 0))
            row.append(overall_class.get(cls, 0))
            w.writerow(row)

        # Flag distribution (multi-label, so doesn't sum to 100%)
        for flag in ALL_FLAGS:
            row = ["flag", flag]
            for m in models:
                row.append(by_model_flag[m].get(flag, 0))
            row.append(overall_flag.get(flag, 0))
            w.writerow(row)

    print(f"Wrote summary → {args.output_summary}", file=sys.stderr)

    # Brief stdout summary
    print(f"\n=== Primary failure-class distribution (n={sum(by_model_n.values())}) ===")
    for cls in PRIMARY_CLASS_PRECEDENCE + ["UNCLASSIFIED"]:
        n = overall_class.get(cls, 0)
        if n == 0:
            continue
        pct = 100.0 * n / max(1, sum(by_model_n.values()))
        print(f"  {cls:<28} {n:>3}  ({pct:>5.1f}%)")
    print("\n=== Flag distribution (multi-label) ===")
    for flag in ALL_FLAGS:
        n = overall_flag.get(flag, 0)
        if n == 0:
            continue
        pct = 100.0 * n / max(1, sum(by_model_n.values()))
        print(f"  {flag:<28} {n:>3}  ({pct:>5.1f}%)")
    print("\n=== Per-model primary-class breakdown ===")
    print(f"  {'model':<60} " + " ".join(f"{c[:10]:>10}" for c in PRIMARY_CLASS_PRECEDENCE))
    for m in models:
        cells = " ".join(f"{by_model_class[m].get(c, 0):>10}" for c in PRIMARY_CLASS_PRECEDENCE)
        print(f"  {m:<60} {cells}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
