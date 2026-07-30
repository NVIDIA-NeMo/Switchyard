# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""deep_dive_e2e_trajectories.py — exhaustive per-trial analyses of e2e
trajectories on the v2b cohort.

Twelve analyses (A1-A12) producing a single per-trial enriched CSV and a
summary markdown. The 12 angles are documented in
`docs/analyses/may01-search-vs-e2e-paper-framing.md` and the response thread that
prompted them. Briefly:

  A1 — 2x2 read-enough × resolved confusion matrix (Jiantao's framing)
  A2 — total examined-files distribution by outcome
  A3 — "read beyond gold" — adjacent files / test files / docs files
  A4 — "effective gold" derived from successful-agent behavior, rescore
  A5 — probe-then-edit ratio (Read tool calls vs Edit/apply_patch)
  A6 — edit-locality scaling: examined files vs gold-patch size
  A7 — failure-mode attribution to task properties
  A8 — edit/test iteration count
  A9 — subtask decomposition (TodoWrite/task) usage
  A10 — thrashing signatures: same file written multiple times, edits reverted
  A11 — intermediate-state replay (skipped: reconstruction infeasible from
        trajectory alone without running tests)
  A12 — self-introspection: did the agent's text mention the right files
        before editing?

Inputs: the four per-model implicit-search CSVs + the e2e trial dirs +
patch-gold + v2b harbor-tasks dir. Outputs:

  docs/data/v2b-deep-dive-per-trial.csv      — every trial × every metric
  docs/data/v2b-deep-dive-summary.md         — aggregate findings, per-model
                                               splits, top examples per
                                               analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Tool-name conventions (carried from score_e2e_implicit_search.py)
# ---------------------------------------------------------------------------

CODEX_BASH = "exec_command"
CODEX_PATCH = "apply_patch"
CODEX_STDIN = "write_stdin"

CC_BASH = "Bash"
CC_READ = "Read"
CC_EDIT = "Edit"
CC_WRITE = "Write"
CC_MULTI_EDIT = "MultiEdit"
CC_NOTEBOOK = "NotebookEdit"
CC_GREP = "Grep"
CC_GLOB = "Glob"
CC_TODO = "TodoWrite"

OC_BASH = "bash"
OC_READ = "read"
OC_EDIT = "edit"
OC_WRITE = "write"
OC_GREP = "grep"
OC_GLOB = "glob"
OC_TASK = "task"
OC_TODO = "todowrite"

EDIT_TOOLS = {CODEX_PATCH, CC_EDIT, CC_WRITE, CC_MULTI_EDIT, CC_NOTEBOOK, OC_EDIT, OC_WRITE}
READ_TOOLS = {CC_READ, OC_READ}  # codex's exec_command is multi-purpose; counted via bash
BASH_TOOLS = {CODEX_BASH, CC_BASH, OC_BASH}
GREP_GLOB_TOOLS = {CC_GREP, CC_GLOB, OC_GREP, OC_GLOB}
TODO_TOOLS = {CC_TODO, OC_TODO, OC_TASK}

# ---------------------------------------------------------------------------
# Trajectory loaders (copy of score_e2e_implicit_search.py adapter)
# ---------------------------------------------------------------------------


def _load_trajectory(trial_dir: Path) -> dict | None:
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
    return {"schema_version": "claude-code-adapted", "agent": {"name": "claude-code"}, "steps": steps}


# ---------------------------------------------------------------------------
# Utility: file path normalization
# ---------------------------------------------------------------------------

_STRIP_PREFIXES = ("/repo/", "repo/", "/code/", "code/", "./")


def _normalize_file(path: str) -> str:
    p = path.strip().lower()
    for prefix in _STRIP_PREFIXES:
        if p.startswith(prefix):
            p = p[len(prefix) :]
            break
    return p.rstrip("/")


_PATH_LIKE_RE = re.compile(r"(?:[\w./_-]+/)+[\w_-]+\.\w+")


def _paths_in(text: str) -> set[str]:
    out: set[str] = set()
    for m in _PATH_LIKE_RE.finditer(text or ""):
        c = m.group(0)
        if c.startswith(("http://", "https://")):
            continue
        out.add(_normalize_file(c))
    return out


# ---------------------------------------------------------------------------
# Per-trial extractor: collect all the signals from one trial
# ---------------------------------------------------------------------------


@dataclass
class TrialSignals:
    """Everything we need from one trial. Most are computed once here; A1-A12
    pivot off these fields plus the implicit-search CSV row."""

    # Identity
    model: str = ""
    task: str = ""
    trial_dir: str = ""
    e2e_resolved: int = 0  # 0 or 1

    # Outcome details
    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0

    # Carried from implicit-search CSV
    exam_file_recall: float = 0.0
    exam_function_recall: float = 0.0
    comm_file_recall: float = 0.0
    n_examined_files: int = 0
    n_committed_files: int = 0
    gold_n_files: int = 0
    gold_n_functions: int = 0

    # NEW signals from trajectory walk
    n_steps: int = 0
    n_read_calls: int = 0  # explicit Read/read tool
    n_bash_calls: int = 0
    n_edit_calls: int = 0  # Edit/Write/apply_patch/edit/write
    n_grep_glob_calls: int = 0
    n_todo_calls: int = 0  # TodoWrite/todowrite/task
    # Multi-edit on same file (thrashing signal)
    files_edited_once: int = 0
    files_edited_2plus: int = 0  # potential thrashing
    # Test invocations: Bash commands matching pytest/pytest-xdist/python -m unittest
    n_test_invocations: int = 0
    # First-edit step / first-test-invocation step (probe-then-edit ratio)
    first_edit_step: int = 0  # 0 if no edits ever
    first_test_step: int = 0
    # Total LOC of agent edits (approx.) — same as classify_localized_failures
    agent_diff_lines: int = 0
    # Files examined that are NOT in gold (over-broad)
    n_examined_nongold: int = 0
    # Files examined matching test patterns
    n_examined_test_files: int = 0
    # Files examined matching docs / README
    n_examined_doc_files: int = 0
    # Self-introspection: did the agent's text mention any gold function name
    # BEFORE the first edit?
    pre_edit_mention_count: int = 0  # gold functions mentioned in agent text before first edit
    pre_edit_diagnosis_text: str = ""  # excerpt of agent text from before first edit (max 600 chars)

    # Gold-patch size (lines added+removed)
    gold_diff_lines: int = 0


def _gold_patch_size(task_dir: Path) -> int:
    p = task_dir / "solution" / "changes.patch"
    if not p.is_file():
        return 0
    n = 0
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- ")):
            n += 1
    return n


def _committed_file_in_step(fn: str, args: dict) -> str | None:
    """Return the file path the agent committed to, if any."""
    if fn == CODEX_PATCH:
        # Codex DSL — first '*** Update File:' or '*** Add File:'
        text = (args.get("input") or "") + "\n" + (args.get("patch") or "")
        m = re.search(r"^\*\*\* (?:Update|Add|Delete) File:\s*(\S+)\s*$", text, re.MULTILINE)
        if m:
            return _normalize_file(m.group(1))
    elif fn in (CC_EDIT, CC_WRITE, CC_MULTI_EDIT, CC_NOTEBOOK):
        fp = args.get("file_path") or args.get("notebook_path")
        if fp:
            return _normalize_file(fp)
    elif fn in (OC_EDIT, OC_WRITE):
        fp = args.get("filePath")
        if fp:
            return _normalize_file(fp)
    return None


_TEST_INVOKE_RE = re.compile(r"\b(?:pytest|python\s+-m\s+(?:pytest|unittest)|tox|nose2)\b")
_TEST_FILE_RE = re.compile(r"(?:^|/)(?:test_[\w_]+|conftest|tests?)\.py$|(?:^|/)tests?/")
_DOC_FILE_RE = re.compile(r"\.(?:md|rst|txt)$|(?:^|/)docs?/", re.IGNORECASE)


def _agent_diff_size(traj: dict) -> int:
    n = 0
    for step in traj.get("steps") or []:
        for tc in step.get("tool_calls") or []:
            fn = tc.get("function_name", "")
            args = tc.get("arguments") or {}
            if fn == CODEX_PATCH:
                for line in (args.get("input") or "").splitlines():
                    if line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- ")):
                        n += 1
                for line in (args.get("patch") or "").splitlines():
                    if line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- ")):
                        n += 1
            elif fn in (CC_EDIT, CC_MULTI_EDIT):
                for k in ("old_string", "new_string"):
                    s = args.get(k) or ""
                    n += s.count("\n") + (1 if s else 0)
            elif fn == CC_WRITE:
                s = args.get("content") or ""
                n += s.count("\n") + (1 if s else 0)
            elif fn in (OC_EDIT, OC_WRITE):
                for k in ("oldString", "newString", "content"):
                    s = args.get(k) or ""
                    n += s.count("\n") + (1 if s else 0)
    return n


def _build_trial_signals(
    csv_row: dict, trial_dir: Path, task_dir: Path | None, gold_files: set[str], gold_funcs: list[str]
) -> TrialSignals:
    """Walk one trial's trajectory and gather all the per-trial signals."""
    rj_path = trial_dir / "verifier" / "reward.json"
    rj: dict = {}
    if rj_path.is_file():
        try:
            rj = json.loads(rj_path.read_text())
        except (OSError, json.JSONDecodeError):
            rj = {}

    sig = TrialSignals(
        model=csv_row["agent"] + "/" + csv_row["model"][:80],
        task=csv_row["task"],
        trial_dir=str(trial_dir),
        e2e_resolved=1 if csv_row["e2e_resolved"] == "1" else 0,
        f2p_passed=int(rj.get("f2p_passed") or 0),
        f2p_total=int(rj.get("f2p_total") or 0),
        p2p_passed=int(rj.get("p2p_passed") or 0),
        p2p_total=int(rj.get("p2p_total") or 0),
        exam_file_recall=float(csv_row["exam_file_recall"] or 0),
        exam_function_recall=float(csv_row["exam_function_recall"] or 0),
        comm_file_recall=float(csv_row["comm_file_recall"] or 0),
        n_examined_files=int(csv_row["n_examined_files"] or 0),
        n_committed_files=int(csv_row["n_committed_files"] or 0),
        gold_n_files=int(csv_row["gold_n_files"] or 0),
        gold_n_functions=int(csv_row["gold_n_functions"] or 0),
    )
    if task_dir is not None and task_dir.is_dir():
        sig.gold_diff_lines = _gold_patch_size(task_dir)

    traj = _load_trajectory(trial_dir)
    if traj is None:
        return sig

    sig.agent_diff_lines = _agent_diff_size(traj)
    steps = traj.get("steps") or []
    sig.n_steps = len(steps)

    # Collect: all examined files (path-extracted), all committed files (per
    # tool call), tool counts, first-edit step, first-test step, and pre-edit
    # text.
    examined_files: set[str] = set()
    edits_per_file: Counter = Counter()
    pre_edit_text_buf: list[str] = []
    first_edit_step = 0
    first_test_step = 0
    gold_func_bare = {gf.strip().rsplit(".", 1)[-1] for gf in gold_funcs if gf.strip()}

    for i, step in enumerate(steps, 1):
        for tc in step.get("tool_calls") or []:
            fn = tc.get("function_name", "")
            args = tc.get("arguments") or {}

            if fn in EDIT_TOOLS:
                sig.n_edit_calls += 1
                committed = _committed_file_in_step(fn, args)
                if committed:
                    edits_per_file[committed] += 1
                if first_edit_step == 0:
                    first_edit_step = i
            elif fn in READ_TOOLS:
                sig.n_read_calls += 1
                fp = args.get("file_path") or args.get("filePath")
                if fp:
                    examined_files.add(_normalize_file(fp))
            elif fn in BASH_TOOLS:
                sig.n_bash_calls += 1
                cmd = args.get("command") or args.get("cmd") or ""
                examined_files.update(_paths_in(cmd))
                if _TEST_INVOKE_RE.search(cmd):
                    sig.n_test_invocations += 1
                    if first_test_step == 0:
                        first_test_step = i
            elif fn in GREP_GLOB_TOOLS:
                sig.n_grep_glob_calls += 1
            elif fn in TODO_TOOLS:
                sig.n_todo_calls += 1

        # Observation paths (rg/grep output)
        obs = step.get("observation") or {}
        for r in obs.get("results") or []:
            content = r.get("content") or ""
            examined_files.update(_paths_in(content))

        # Agent text up to (but not including) the first edit
        msg = step.get("message") or ""
        if first_edit_step == 0 and msg:
            pre_edit_text_buf.append(msg)

    # Files-edited-multiple-times (thrashing)
    sig.files_edited_once = sum(1 for c in edits_per_file.values() if c == 1)
    sig.files_edited_2plus = sum(1 for c in edits_per_file.values() if c >= 2)

    sig.first_edit_step = first_edit_step
    sig.first_test_step = first_test_step

    # Examined-file analytics
    sig.n_examined_nongold = len(examined_files - gold_files)
    sig.n_examined_test_files = sum(1 for f in examined_files if _TEST_FILE_RE.search(f))
    sig.n_examined_doc_files = sum(1 for f in examined_files if _DOC_FILE_RE.search(f))

    # Self-introspection: how many gold function names appear in pre-edit text
    pre_edit_full = "\n".join(pre_edit_text_buf)
    if pre_edit_full and gold_func_bare:
        seen = set()
        for bare in gold_func_bare:
            if bare and re.search(rf"\b{re.escape(bare)}\b", pre_edit_full):
                seen.add(bare)
        sig.pre_edit_mention_count = len(seen)
        sig.pre_edit_diagnosis_text = pre_edit_full[-600:]

    return sig


# ---------------------------------------------------------------------------
# Cross-trial analyses (operate on the population of TrialSignals)
# ---------------------------------------------------------------------------


def _safe_mean(xs: list[float]) -> float:
    return st.mean(xs) if xs else 0.0


def analysis_a1_2x2(sigs: list[TrialSignals], threshold: float = 0.8) -> dict:
    """A1: read-enough × resolved confusion matrix.

    'Read-enough' = exam_file_recall >= threshold."""
    out: dict = {"by_model": {}, "overall": {}}
    by_model: dict[str, dict] = defaultdict(lambda: dict(rt=0, rf=0, nt=0, nf=0))
    for s in sigs:
        cell_read = s.exam_file_recall >= threshold
        cell_succ = s.e2e_resolved == 1
        key = (
            "rt"
            if (cell_read and cell_succ)
            else "rf"
            if (cell_read and not cell_succ)
            else "nt"
            if (not cell_read and cell_succ)
            else "nf"
        )
        by_model[s.model][key] += 1
    for m in sorted(by_model):
        c = by_model[m]
        c["read_enough_rate"] = (c["rt"] + c["rf"]) / max(1, sum(c[k] for k in ("rt", "rf", "nt", "nf")))
        c["resolved_given_read_enough"] = c["rt"] / max(1, c["rt"] + c["rf"])
        c["resolved_given_read_poor"] = c["nt"] / max(1, c["nt"] + c["nf"])
        out["by_model"][m] = c
    # Overall
    overall = {"rt": 0, "rf": 0, "nt": 0, "nf": 0}
    for c in by_model.values():
        for k in ("rt", "rf", "nt", "nf"):
            overall[k] += c[k]
    overall["resolved_given_read_enough"] = overall["rt"] / max(1, overall["rt"] + overall["rf"])
    overall["resolved_given_read_poor"] = overall["nt"] / max(1, overall["nt"] + overall["nf"])
    out["overall"] = overall
    out["threshold"] = threshold
    return out


def analysis_a2_total_files(sigs: list[TrialSignals]) -> dict:
    """A2: total examined-files distribution by outcome, per model."""
    out: dict = {"by_model": {}}
    by_model: dict[str, dict] = defaultdict(lambda: {"resolved": [], "failed": []})
    for s in sigs:
        bucket = "resolved" if s.e2e_resolved == 1 else "failed"
        by_model[s.model][bucket].append(s.n_examined_files)
    for m in sorted(by_model):
        d = by_model[m]
        out["by_model"][m] = {
            "resolved_mean": _safe_mean(d["resolved"]),
            "resolved_median": st.median(d["resolved"]) if d["resolved"] else 0,
            "resolved_n": len(d["resolved"]),
            "failed_mean": _safe_mean(d["failed"]),
            "failed_median": st.median(d["failed"]) if d["failed"] else 0,
            "failed_n": len(d["failed"]),
        }
    return out


def analysis_a3_read_beyond_gold(sigs: list[TrialSignals]) -> dict:
    """A3: did the agent read adjacent files (test files, docs, files outside gold)?"""
    out: dict = {"by_model_outcome": {}}
    by: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in sigs:
        out_label = "resolved" if s.e2e_resolved == 1 else "failed"
        by[(s.model, out_label)]["nongold"].append(s.n_examined_nongold)
        by[(s.model, out_label)]["test"].append(s.n_examined_test_files)
        by[(s.model, out_label)]["doc"].append(s.n_examined_doc_files)
    for (m, o), data in by.items():
        out["by_model_outcome"][f"{m}/{o}"] = {k: _safe_mean(v) for k, v in data.items()}
    return out


def analysis_a4_effective_gold(
    sigs: list[TrialSignals], all_examined: dict[str, dict[str, set[str]]]
) -> dict:
    """A4: "effective gold" = union of files read by every successful trial of
    each task. Per-task: how many extra files are in effective gold beyond
    patch-derived gold? Then rescore failed trials against effective gold."""
    out = {"per_task_extras": [], "summary": {}}
    extras_sizes = []
    for task, by_outcome in all_examined.items():
        succ_examined = by_outcome.get("resolved_examined_union") or set()
        gold_files = by_outcome.get("gold_files") or set()
        if not succ_examined:
            continue
        # Files all successful agents read but aren't in gold
        adjacent = succ_examined - gold_files
        extras_sizes.append(len(adjacent))
        out["per_task_extras"].append(
            {
                "task": task,
                "gold_n": len(gold_files),
                "succ_examined_n": len(succ_examined),
                "adjacent_n": len(adjacent),
            }
        )
    if extras_sizes:
        out["summary"] = {
            "tasks_with_resolved": len(extras_sizes),
            "mean_adjacent_files": _safe_mean(extras_sizes),
            "median_adjacent_files": st.median(extras_sizes),
        }
    return out


def analysis_a5_probe_then_edit(sigs: list[TrialSignals]) -> dict:
    """A5: probe-then-edit ratio. How many steps before first edit?"""
    out: dict = {"by_model_outcome": {}}
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_ratio: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in sigs:
        if s.first_edit_step == 0:
            continue
        out_label = "resolved" if s.e2e_resolved == 1 else "failed"
        # Probe-fraction: steps before first edit / total steps
        ratio = s.first_edit_step / max(1, s.n_steps)
        by[(s.model, out_label)].append(s.first_edit_step)
        by_ratio[(s.model, out_label)].append(ratio)
    for (m, o), vals in by.items():
        out["by_model_outcome"][f"{m}/{o}"] = {
            "mean_first_edit_step": _safe_mean(vals),
            "mean_probe_fraction": _safe_mean(by_ratio[(m, o)]),
            "n": len(vals),
        }
    return out


def analysis_a6_edit_locality(sigs: list[TrialSignals]) -> dict:
    """A6: examined-files vs gold-patch size. Do agents adapt search to scope?"""
    # Pearson correlation per model, plus scatter data
    import math

    out: dict = {"by_model": {}}
    by: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for s in sigs:
        if s.gold_diff_lines and s.n_examined_files:
            by[s.model].append((s.gold_diff_lines, s.n_examined_files))
    for m, pairs in by.items():
        if len(pairs) < 3:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in pairs)
        denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        deny = math.sqrt(sum((y - my) ** 2 for y in ys))
        r = num / (denx * deny) if (denx and deny) else 0.0
        out["by_model"][m] = {"pearson_r": round(r, 3), "n": len(pairs), "mean_examined": round(my, 1)}
    return out


def analysis_a7_failure_modes_by_task(sigs: list[TrialSignals]) -> dict:
    """A7: what task properties correlate with F2P_NONE vs F2P_PARTIAL?
    Within the failed-AND-fully-localized subset only."""
    failures = [s for s in sigs if s.e2e_resolved == 0 and s.exam_file_recall >= 1.0]
    out: dict = {"n": len(failures), "buckets": {}}
    # Partition into F2P_NONE vs F2P_PARTIAL
    f2p_none = [s for s in failures if s.f2p_total > 0 and s.f2p_passed == 0]
    f2p_partial = [s for s in failures if s.f2p_total > 0 and 0 < s.f2p_passed < s.f2p_total]
    # Mean task properties per bucket
    for label, bucket in [("F2P_NONE", f2p_none), ("F2P_PARTIAL", f2p_partial)]:
        if not bucket:
            continue
        out["buckets"][label] = {
            "n": len(bucket),
            "mean_gold_n_files": _safe_mean([s.gold_n_files for s in bucket]),
            "mean_gold_diff_lines": _safe_mean([s.gold_diff_lines for s in bucket]),
            "mean_gold_n_functions": _safe_mean([s.gold_n_functions for s in bucket]),
            "mean_agent_diff_lines": _safe_mean([s.agent_diff_lines for s in bucket]),
        }
    return out


def analysis_a8_iteration(sigs: list[TrialSignals]) -> dict:
    """A8: edit-test iteration count. n_test_invocations + n_edit_calls."""
    out: dict = {"by_model_outcome": {}}
    by: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in sigs:
        out_label = "resolved" if s.e2e_resolved == 1 else "failed"
        by[(s.model, out_label)]["edits"].append(s.n_edit_calls)
        by[(s.model, out_label)]["tests"].append(s.n_test_invocations)
        by[(s.model, out_label)]["steps"].append(s.n_steps)
    for (m, o), data in by.items():
        out["by_model_outcome"][f"{m}/{o}"] = {
            "mean_edits": _safe_mean(data["edits"]),
            "mean_test_runs": _safe_mean(data["tests"]),
            "mean_steps": _safe_mean(data["steps"]),
            "n": len(data["edits"]),
        }
    return out


def analysis_a9_decomposition(sigs: list[TrialSignals]) -> dict:
    """A9: TodoWrite/task usage by outcome."""
    out: dict = {"by_model_outcome": {}}
    by: dict[tuple[str, str], list[int]] = defaultdict(list)
    for s in sigs:
        out_label = "resolved" if s.e2e_resolved == 1 else "failed"
        by[(s.model, out_label)].append(s.n_todo_calls)
    for (m, o), vals in by.items():
        out["by_model_outcome"][f"{m}/{o}"] = {
            "mean_todo_calls": _safe_mean(vals),
            "frac_with_decomposition": sum(1 for v in vals if v > 0) / max(1, len(vals)),
            "n": len(vals),
        }
    return out


def analysis_a10_thrashing(sigs: list[TrialSignals]) -> dict:
    """A10: files edited multiple times — possible thrashing."""
    out: dict = {"by_model_outcome": {}}
    by: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in sigs:
        out_label = "resolved" if s.e2e_resolved == 1 else "failed"
        total_edited = s.files_edited_once + s.files_edited_2plus
        thrash_rate = s.files_edited_2plus / total_edited if total_edited else 0
        by[(s.model, out_label)]["thrash_rate"].append(thrash_rate)
        by[(s.model, out_label)]["files_2plus"].append(s.files_edited_2plus)
    for (m, o), data in by.items():
        out["by_model_outcome"][f"{m}/{o}"] = {
            "mean_thrash_rate": _safe_mean(data["thrash_rate"]),
            "mean_files_edited_2plus": _safe_mean(data["files_2plus"]),
            "n": len(data["thrash_rate"]),
        }
    return out


def analysis_a12_self_introspection(sigs: list[TrialSignals]) -> dict:
    """A12: did the agent's pre-edit text mention any gold function name?"""
    out: dict = {"by_model_outcome": {}}
    by: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in sigs:
        out_label = "resolved" if s.e2e_resolved == 1 else "failed"
        by[(s.model, out_label)]["mention_count"].append(s.pre_edit_mention_count)
        by[(s.model, out_label)]["any_mention"].append(1 if s.pre_edit_mention_count else 0)
    for (m, o), data in by.items():
        out["by_model_outcome"][f"{m}/{o}"] = {
            "mean_mention_count": _safe_mean(data["mention_count"]),
            "frac_any_mention": _safe_mean(data["any_mention"]),
            "n": len(data["mention_count"]),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--implicit-csvs", nargs="+", type=Path, required=True, help="Per-model implicit-search CSVs."
    )
    ap.add_argument(
        "--e2e-roots", nargs="+", type=Path, required=True, help="One or more dirs holding e2e job dirs."
    )
    ap.add_argument("--patch-gold", type=Path, required=True, help="references/v2b-patch-gold.json")
    ap.add_argument(
        "--tasks-dir", type=Path, required=True, help="v2b harbor-tasks dir (for solution/changes.patch)."
    )
    ap.add_argument("--output-csv", type=Path, required=True, help="Per-trial enriched CSV.")
    ap.add_argument("--output-md", type=Path, required=True, help="Aggregate findings markdown.")
    args = ap.parse_args()

    patch_gold = json.loads(args.patch_gold.read_text())
    print(f"Loaded patch-gold for {len(patch_gold)} v2b tasks", file=sys.stderr)

    def _find_trial(trial_path: str) -> Path | None:
        for root in args.e2e_roots:
            cand = root / "jobs" / trial_path
            if cand.is_dir():
                return cand
            cand2 = root / trial_path
            if cand2.is_dir():
                return cand2
        return None

    sigs: list[TrialSignals] = []
    skipped = 0
    for csv_path in args.implicit_csvs:
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                trial_dir = _find_trial(row["trial_dir"])
                if trial_dir is None:
                    skipped += 1
                    continue
                gold = patch_gold.get(row["task"])
                if gold is None:
                    skipped += 1
                    continue
                gold_files = {_normalize_file(f) for f in (gold.get("files") or [])}
                gold_funcs = list(gold.get("functions") or [])
                task_dir = args.tasks_dir / row["task"]
                sig = _build_trial_signals(row, trial_dir, task_dir, gold_files, gold_funcs)
                sigs.append(sig)
    print(f"Built {len(sigs)} trial signal records ({skipped} skipped)", file=sys.stderr)

    # Per-trial CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fh:
        if sigs:
            w = csv.DictWriter(fh, fieldnames=list(asdict(sigs[0])))
            w.writeheader()
            for s in sigs:
                w.writerow(asdict(s))
    print(f"Wrote per-trial CSV → {args.output_csv}", file=sys.stderr)

    # Effective-gold builder for A4 (needs full examined-file sets per
    # successful trial). Approximation: use n_examined_files etc.; the union
    # itself requires re-walking trajectories. For v1 we use what we already
    # have aggregated.
    a4 = {"summary": {}, "per_task_extras": []}
    by_task: dict[str, dict[str, list[TrialSignals]]] = defaultdict(lambda: {"r": [], "f": []})
    for s in sigs:
        by_task[s.task]["r" if s.e2e_resolved else "f"].append(s)
    extras_sizes = []
    for task, d in by_task.items():
        if not d["r"]:
            continue
        # Approximation: mean examined-files among successful trials,
        # minus gold-n-files, gives a (rough) "extra adjacent" count
        succ_mean_exam = _safe_mean([s.n_examined_files for s in d["r"]])
        gold_n = d["r"][0].gold_n_files
        adjacent = max(0, succ_mean_exam - gold_n)
        extras_sizes.append(adjacent)
        a4["per_task_extras"].append(
            {
                "task": task,
                "gold_n": gold_n,
                "succ_mean_examined": round(succ_mean_exam, 1),
                "approx_adjacent": round(adjacent, 1),
            }
        )
    if extras_sizes:
        a4["summary"] = {
            "tasks_with_resolved": len(extras_sizes),
            "mean_approx_adjacent": round(_safe_mean(extras_sizes), 1),
            "median_approx_adjacent": round(st.median(extras_sizes), 1),
        }

    findings = {
        "n_trials": len(sigs),
        "n_models": len({s.model for s in sigs}),
        "a1_2x2_thr0.8": analysis_a1_2x2(sigs, threshold=0.8),
        "a1_2x2_thr1.0": analysis_a1_2x2(sigs, threshold=1.0),
        "a2_total_files": analysis_a2_total_files(sigs),
        "a3_read_beyond_gold": analysis_a3_read_beyond_gold(sigs),
        "a4_effective_gold": a4,
        "a5_probe_then_edit": analysis_a5_probe_then_edit(sigs),
        "a6_edit_locality": analysis_a6_edit_locality(sigs),
        "a7_failure_mode_by_task": analysis_a7_failure_modes_by_task(sigs),
        "a8_iteration": analysis_a8_iteration(sigs),
        "a9_decomposition": analysis_a9_decomposition(sigs),
        "a10_thrashing": analysis_a10_thrashing(sigs),
        "a12_self_introspection": analysis_a12_self_introspection(sigs),
    }

    # Dump structured findings beside the markdown for downstream tooling
    json_out = args.output_md.with_suffix(".json")
    json_out.write_text(json.dumps(findings, indent=2, default=str))
    print(f"Wrote structured findings → {json_out}", file=sys.stderr)

    # Brief stdout summary
    print()
    print("=== A1 — read-enough × resolved (threshold=0.8) ===")
    o = findings["a1_2x2_thr0.8"]["overall"]
    print(f"  read-enough∩resolved={o['rt']}, read-enough∩failed={o['rf']},")
    print(f"  read-poor∩resolved={o['nt']}, read-poor∩failed={o['nf']}")
    p_re = o["resolved_given_read_enough"]
    p_rp = o["resolved_given_read_poor"]
    print(f"  P(resolved | read-enough) = {p_re:.3f}")
    print(f"  P(resolved | read-poor)   = {p_rp:.3f}")

    print("\n=== A6 — edit-locality scaling (Pearson r per model) ===")
    for m, d in findings["a6_edit_locality"]["by_model"].items():
        print(f"  {m[:60]:<60}  r={d['pearson_r']:+.3f}  n={d['n']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
