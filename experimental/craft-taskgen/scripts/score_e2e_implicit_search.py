# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""score_e2e_implicit_search.py — extract & score implicit localization in e2e trajectories.

The premise: when a coding agent solves an end-to-end task, it implicitly
searches the codebase along the way (via Read, Grep, Bash `cat`/`sed`/`rg`,
etc.). This script extracts what files+functions the agent referenced
during a trial — and scores that against the same patch-derived gold used
for the dedicated search task. Output mirrors the search per-trial CSV so
the two corpora can be compared directly.

Two implicit-search scores per trial:
  - examined_*  — files+functions the agent looked at in any way (broad).
                  Comparable to a search agent's submission, which can
                  include "looked at, didn't commit to."
  - committed_* — files+functions the agent ran `apply_patch` against.
                  Strict, oracle-like: the agent committed to editing these.

Supports three agent transcript formats:
  - codex (ATIF trajectory.json with `exec_command` / `apply_patch` / `write_stdin`)
  - opencode (ATIF trajectory.json with `bash` / `read` / `edit` / `write` /
    `task` — note that `task` delegates to a subagent whose tool calls are
    invisible from the top-level trajectory)
  - claude-code (raw NDJSON `agent/claude-code.txt` adapted to ATIF shape
    in-process; covers `Bash` / `Read` / `Edit` / `Write` / `Grep` / `Glob` /
    `MultiEdit` / `NotebookEdit`)

Per-tool dispatch in `_extract_files_from_tool_call` distinguishes oracle
signals (`Read.file_path`, `read.filePath`, `Edit.file_path`, `edit.filePath`,
codex `apply_patch` payloads) from heuristic signals (file paths in bash
command strings or observation output).

Usage:
  uv run python scripts/score_e2e_implicit_search.py \\
      --e2e-roots <e2e-job-dir>... \\
      --patch-gold references/v2b-patch-gold.json \\
      --output /tmp/e2e-implicit-search.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Scoring primitives — same as rescore_search_against_patch_gold.py.
# Inlined rather than imported so this script runs standalone.
# ---------------------------------------------------------------------------

_STRIP_PREFIXES = ("/repo/", "repo/", "/code/", "code/", "./")


def _normalize_file(path: str) -> str:
    p = path.strip().lower()
    for prefix in _STRIP_PREFIXES:
        if p.startswith(prefix):
            p = p[len(prefix) :]
            break
    return p.rstrip("/")


def _normalize_function_tail(name: str) -> str:
    parts = name.strip().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else name.strip()


def _exact_or_tail_match_count(predicted: set[str], gold: set[str]) -> int:
    matched_exact = predicted & gold
    rem_pred = predicted - matched_exact
    rem_gold = gold - matched_exact
    pred_tails = Counter(_normalize_function_tail(p) for p in rem_pred)
    gold_tails = Counter(_normalize_function_tail(g) for g in rem_gold)
    return len(matched_exact) + sum((pred_tails & gold_tails).values())


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _score(pred_files: set[str], pred_funcs: set[str], gold_files: set[str], gold_funcs: set[str]) -> dict:
    file_hits = len(pred_files & gold_files)
    file_recall = file_hits / len(gold_files) if gold_files else 1.0
    file_precision = file_hits / len(pred_files) if pred_files else 0.0
    file_f1 = _f1(file_precision, file_recall)
    file_iou_d = len(pred_files | gold_files)
    file_iou = file_hits / file_iou_d if file_iou_d > 0 else 1.0

    func_recall_hits = _exact_or_tail_match_count(pred_funcs, gold_funcs)
    func_recall = func_recall_hits / len(gold_funcs) if gold_funcs else 1.0
    func_precision_hits = _exact_or_tail_match_count(gold_funcs, pred_funcs)
    func_precision = func_precision_hits / len(pred_funcs) if pred_funcs else 0.0
    func_f1 = _f1(func_precision, func_recall)
    pred_tails = {_normalize_function_tail(f) for f in pred_funcs}
    gold_tails = {_normalize_function_tail(f) for f in gold_funcs}
    func_iou_d = len(pred_tails | gold_tails)
    func_iou = len(pred_tails & gold_tails) / func_iou_d if func_iou_d > 0 else 1.0

    has_files = bool(gold_files)
    has_funcs = bool(gold_funcs)
    if has_files and has_funcs:
        nav_score = 0.5 * file_f1 + 0.5 * func_f1
    elif has_files:
        nav_score = file_f1
    elif has_funcs:
        nav_score = func_f1
    else:
        nav_score = 1.0

    return {
        "file_recall": round(file_recall, 4),
        "file_precision": round(file_precision, 4),
        "file_f1": round(file_f1, 4),
        "file_iou": round(file_iou, 4),
        "function_recall": round(func_recall, 4),
        "function_precision": round(func_precision, 4),
        "function_f1": round(func_f1, 4),
        "function_iou": round(func_iou, 4),
        "navigation_score": round(nav_score, 4),
    }


# ---------------------------------------------------------------------------
# ATIF trajectory extraction — codex
# ---------------------------------------------------------------------------
#
# Codex's `exec_command` tool runs an arbitrary bash command. We extract files
# the agent explicitly references (cat/sed/nl/head/tail/rg/ls/cd) plus any
# files surfaced in observation output. `apply_patch` is the strongest
# committed signal — its `arguments.patch` is a unified diff, so the file
# headers tell us exactly which files the agent committed to editing.

# Bash commands whose first positional arg is a file path we want to capture
_FILE_OPENING_CMDS = {"cat", "sed", "nl", "head", "tail", "less", "more", "wc", "rg", "grep", "awk"}

# Regex for `+++ b/<path>` and `--- a/<path>` in unified diffs (apply_patch payload)
_DIFF_FILE_RE = re.compile(r"^[\+\-]{3} (?:[ab]/)?([^\s\n]+)$", re.MULTILINE)

# Codex's apply_patch DSL uses `*** Update File: <path>` / `*** Add File: <path>`
# / `*** Delete File: <path>` instead of unified-diff headers. The payload is in
# the `input` field rather than `patch`.
_CODEX_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File:\s*(\S+)\s*$", re.MULTILINE)

# Regex for absolute repo paths (we strip leading /code or /repo); also catches
# patterns like `tests/test_foo.py` and `pkg/sub/mod.py`
_PATH_LIKE_RE = re.compile(r"(?:[\w./_-]+/)+[\w_-]+\.\w+")


def _paths_in_text(text: str) -> set[str]:
    """Extract file-pathy substrings from a text blob.

    Conservative: only matches strings with at least one `/` and a file extension.
    Strips repo-root prefixes via `_normalize_file`.
    """
    out: set[str] = set()
    for m in _PATH_LIKE_RE.finditer(text):
        candidate = m.group(0)
        # Filter out obviously non-source paths (URLs, .so/.dylib build artifacts)
        if candidate.startswith(("http://", "https://")):
            continue
        out.add(_normalize_file(candidate))
    return out


def _files_from_apply_patch(args: dict) -> set[str]:
    """Extract files from an apply_patch tool call.

    Two formats observed:
      - codex `input` field: `*** Update File: <path>` blocks (codex's own DSL)
      - generic `patch` field: unified diff with `+++ b/<path>` headers

    We try both — empty strings parse to empty sets, so this is safe to OR."""
    out: set[str] = set()
    patch_text = args.get("patch", "") or ""
    input_text = args.get("input", "") or ""
    for m in _DIFF_FILE_RE.finditer(patch_text):
        path = m.group(1)
        if path != "/dev/null":
            out.add(_normalize_file(path))
    for m in _CODEX_PATCH_FILE_RE.finditer(input_text):
        path = m.group(1)
        out.add(_normalize_file(path))
    # Some apply_patch implementations also accept a `files` array; cover both.
    for f in args.get("files") or []:
        if isinstance(f, str):
            out.add(_normalize_file(f))
    return out


def _files_from_exec_command(args: dict) -> set[str]:
    """Heuristic: extract file paths from a bash command line.

    Strategy: tokenize the cmd, take any token that looks like a path
    (has a `/` and an extension OR is a known relative file). We then run the
    full string through `_paths_in_text` for completeness — duplicate hits
    are deduped by the set return.
    """
    cmd = args.get("cmd", "") or ""
    out = _paths_in_text(cmd)
    return out


def _gold_func_name(gf: str) -> str:
    """Bare method/function name (last dot segment).
    `pkg.mod.Class.method` -> `method`. `pkg.mod.func` -> `func`."""
    return gf.strip().rsplit(".", 1)[-1]


def _gold_func_file(gf: str, gold_files_norm: set[str]) -> str | None:
    """Find the gold file whose path matches a dotted module prefix of the gold
    function. Tries progressively shorter prefixes (drop method, drop class).
    Returns None if no match — caller treats those as bare-name-only."""
    parts = gf.strip().split(".")
    for cut in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:cut])
        candidate = _normalize_file(prefix.replace(".", "/") + ".py")
        if candidate in gold_files_norm:
            return candidate
    return None


def _extract_files_from_tool_call(fn_name: str, args: dict) -> tuple[set[str], set[str], str]:
    """Per-tool dispatch. Returns (examined_files, committed_files, text_blob).

    `text_blob` is whatever string content of the tool call we want to scan for
    function-name mentions later.

    Tool naming conventions:
      codex:        exec_command, apply_patch, write_stdin
      claude-code:  Bash, Read, Edit, Write, NotebookEdit, Grep, Glob,
                    MultiEdit, ...
      opencode:     bash, read, edit, write, glob, grep, ... (lowercased)
                    plus `task` for subagent delegation (which is opaque —
                    we can't see inside it).

    Committed-files signals:
      - codex apply_patch: paths in `*** Update File: <path>` (DSL) or
        `+++ b/<path>` (unified diff)
      - claude-code Edit/Write/NotebookEdit/MultiEdit: `file_path`
      - opencode edit/write: `filePath`

    Examined-files signals (oracle, no inference):
      - claude-code Read: `file_path`
      - opencode read: `filePath`
      - claude-code Grep/Glob: `pattern` is a glob, not a path; skipped.
                                We rely on observations for those.

    Examined-files signals (heuristic — extracted from text):
      - codex exec_command.cmd
      - claude-code Bash.command
      - opencode bash.command
      - any other tool's args (json-serialize and regex-match path-like strings)
    """
    examined: set[str] = set()
    committed: set[str] = set()
    text: str = ""

    fn = fn_name  # alias for brevity

    # --- Bash variants (codex exec_command, claude-code Bash, opencode bash) ---
    if fn == "exec_command":
        cmd = args.get("cmd", "") or ""
        examined.update(_paths_in_text(cmd))
        text = cmd
    elif fn in ("Bash", "bash"):
        cmd = args.get("command", "") or ""
        examined.update(_paths_in_text(cmd))
        text = cmd

    # --- Apply-patch (codex only) ---
    elif fn == "apply_patch":
        files = _files_from_apply_patch(args)
        examined.update(files)
        committed.update(files)
        text = (args.get("patch") or "") + "\n" + (args.get("input") or "")

    # --- File reads (oracle examined-only) ---
    elif fn == "Read":  # claude-code
        fp = args.get("file_path") or ""
        if fp:
            examined.add(_normalize_file(fp))
    elif fn == "read":  # opencode
        fp = args.get("filePath") or ""
        if fp:
            examined.add(_normalize_file(fp))

    # --- File edits (oracle committed) ---
    elif fn in ("Edit", "Write", "MultiEdit", "NotebookEdit"):  # claude-code
        fp = args.get("file_path") or args.get("notebook_path") or ""
        if fp:
            normalized = _normalize_file(fp)
            examined.add(normalized)
            committed.add(normalized)
        # `old_string`/`new_string`/`content` may name functions
        text = " ".join(str(args.get(k) or "") for k in ("old_string", "new_string", "content", "new_source"))
    elif fn in ("edit", "write"):  # opencode
        fp = args.get("filePath") or ""
        if fp:
            normalized = _normalize_file(fp)
            examined.add(normalized)
            committed.add(normalized)
        text = " ".join(str(args.get(k) or "") for k in ("oldString", "newString", "content"))

    # --- Glob/Grep (claude-code) — pattern alone, observations carry the hits ---
    elif fn in ("Glob", "Grep", "glob", "grep"):
        # No oracle signal from args; fall through to observation scanning below.
        text = json.dumps(args, default=str)

    # --- Subagent delegation (opencode `task`) — opaque, but capture the prompt ---
    elif fn == "task":
        # The subagent's tool calls aren't visible. Best we can do is capture
        # the prompt text so any explicit file/function names mentioned there
        # get credit.
        text = json.dumps(args, default=str)
        examined.update(_paths_in_text(text))

    # --- Catch-all: scan json-serialized args for paths ---
    else:
        blob = json.dumps(args, default=str)
        examined.update(_paths_in_text(blob))
        text = blob

    return examined, committed, text


def _extract_signals(
    traj: dict, gold_funcs_set: set[str], gold_files_norm: set[str], agent_name: str
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return (examined_files, examined_funcs, committed_files, committed_funcs).

    Generic: dispatches per-tool via `_extract_files_from_tool_call`. Function
    matching uses bare-name + file-co-location filter (see
    `_gold_func_file` for the file lookup strategy).

    `agent_name` is purely informational here; tool naming is what determines
    parsing. claude-code transcripts (which lack ATIF) are routed through
    `_atif_from_claude_code_ndjson` upstream so this function sees a uniform
    ATIF-shaped dict.
    """
    _ = agent_name  # kept for future per-agent quirks
    examined_files: set[str] = set()
    committed_files: set[str] = set()

    gold_func_tuples: list[tuple[str, str, str | None]] = [
        (gf, _gold_func_name(gf), _gold_func_file(gf, gold_files_norm)) for gf in gold_funcs_set
    ]
    raw_func_mentions: set[str] = set()

    def _scan_text_for_func_names(text: str) -> None:
        if not text:
            return
        for gf, bare, _ in gold_func_tuples:
            if bare and re.search(rf"\b{re.escape(bare)}\b", text):
                raw_func_mentions.add(gf)

    for step in traj.get("steps") or []:
        for tc in step.get("tool_calls") or []:
            fn = tc.get("function_name", "")
            args = tc.get("arguments") or {}
            ex, cm, text = _extract_files_from_tool_call(fn, args)
            examined_files.update(ex)
            committed_files.update(cm)
            if text:
                _scan_text_for_func_names(text)

        obs = step.get("observation") or {}
        for r in obs.get("results") or []:
            content = r.get("content", "") or ""
            examined_files.update(_paths_in_text(content))
            _scan_text_for_func_names(content)

        msg = step.get("message") or ""
        _scan_text_for_func_names(msg)

    examined_funcs: set[str] = set()
    committed_funcs: set[str] = set()
    for gf, _, file_path in gold_func_tuples:
        if gf not in raw_func_mentions:
            continue
        if file_path is None:
            examined_funcs.add(gf)
            committed_funcs.add(gf)
        else:
            if file_path in examined_files:
                examined_funcs.add(gf)
            if file_path in committed_files:
                committed_funcs.add(gf)

    return examined_files, examined_funcs, committed_files, committed_funcs


# ---------------------------------------------------------------------------
# Claude-code transcript adapter
# ---------------------------------------------------------------------------
#
# claude-code writes its raw NDJSON transcript to agent/claude-code.txt but
# does NOT produce an ATIF trajectory.json unless `harbor-lab rebuild-trajectories`
# was run. Rather than require that pre-processing, we adapt the NDJSON in-place
# to an ATIF-shaped dict that the rest of this script can consume.
#
# claude-code NDJSON shape (relevant subset):
#   {"type": "system", ...}                       — session init
#   {"type": "user", "message": {...}}            — user prompt / tool result
#   {"type": "assistant", "message": {
#       "content": [
#           {"type": "thinking", ...},
#           {"type": "text", "text": ...},
#           {"type": "tool_use", "name": ..., "input": {...}, "id": ...}
#       ]}}
#   {"type": "result", ...}                       — final summary
#
# tool_result blocks come back in subsequent user messages with type tool_result
# and a content payload matching the tool_use id.


def _adapt_claude_code_transcript(claude_txt_path: Path) -> dict:
    """Read agent/claude-code.txt and return an ATIF-shaped dict with `agent`,
    `steps`. Each tool_use becomes a step with one `tool_calls` entry and the
    matching tool_result becomes that step's `observation.results`."""
    steps: list[dict] = []
    pending_tools: dict[str, dict] = {}  # tool_use_id -> step (so we can attach observation)
    step_id = 0

    with claude_txt_path.open() as f:
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
                # We may have multiple blocks per message: a thinking, a text,
                # several tool_uses. Emit one step per tool_use; merge any text
                # message into a sibling step that has source=agent, message=text.
                text_buf: list[str] = []
                for blk in content:
                    btype = blk.get("type")
                    if btype == "text":
                        text_buf.append(blk.get("text") or "")
                    elif btype == "thinking":
                        # Don't expose thinking text to function-name matching —
                        # it's internal monologue, not real signal about what
                        # files the agent actually examined.
                        continue
                    elif btype == "tool_use":
                        step_id += 1
                        step = {
                            "step_id": step_id,
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
                        pending_tools[blk.get("id") or ""] = step
                        steps.append(step)
                # If text-only assistant (no tool_use), record as its own step
                if text_buf and not any(blk.get("type") == "tool_use" for blk in content):
                    step_id += 1
                    steps.append(
                        {
                            "step_id": step_id,
                            "source": "agent",
                            "message": "\n".join(text_buf),
                        }
                    )
            elif ev.get("type") == "user":
                msg = ev.get("message") or {}
                content = msg.get("content") or []
                # tool_result blocks: attach to the corresponding tool_use's step
                if isinstance(content, list):
                    for blk in content:
                        if blk.get("type") == "tool_result":
                            tcid = blk.get("tool_use_id")
                            payload = blk.get("content")
                            # `content` is sometimes a string, sometimes a list
                            # of {type: text, text: ...} blocks
                            if isinstance(payload, list):
                                txt = "\n".join(p.get("text", "") for p in payload if isinstance(p, dict))
                            else:
                                txt = str(payload or "")
                            if tcid in pending_tools:
                                step = pending_tools[tcid]
                                step.setdefault("observation", {}).setdefault("results", []).append(
                                    {"source_call_id": tcid, "content": txt}
                                )
    return {
        "schema_version": "claude-code-adapted",
        "agent": {"name": "claude-code"},
        "steps": steps,
    }


def _load_trajectory(trial_dir: Path) -> dict | None:
    """Load an ATIF trajectory, or adapt claude-code.txt to ATIF shape."""
    p = trial_dir / "agent" / "trajectory.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    cc = trial_dir / "agent" / "claude-code.txt"
    if cc.is_file():
        try:
            return _adapt_claude_code_transcript(cc)
        except (OSError, json.JSONDecodeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Trial walking
# ---------------------------------------------------------------------------


def _find_trial_dirs(roots: list[Path]) -> list[Path]:
    """Find trial dirs that have either trajectory.json (ATIF — codex/opencode)
    or claude-code.txt (NDJSON — claude-code/opus). Both must have a sibling
    result.json so we can read trial metadata."""
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            print(f"WARN: missing root {root}", file=sys.stderr)
            continue
        for pattern in ("agent/trajectory.json", "agent/claude-code.txt"):
            for entry in root.rglob(pattern):
                trial = entry.parent.parent
                if trial in seen:
                    continue
                if (trial / "result.json").is_file():
                    seen.add(trial)
                    out.append(trial)
    return out


def _trial_metadata(trial_dir: Path) -> tuple[str, str, str, dict]:
    """Return (agent_name, model_name_with_effort, task_name, reward_dict).
    reward_dict can be {} if the verifier didn't run."""
    try:
        r = json.loads((trial_dir / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        return ("unknown", "unknown", trial_dir.name.rsplit("-", 1)[0], {})
    ac = (r.get("config") or {}).get("agent") or {}
    name = ac.get("name") or "unknown"
    model = ac.get("model_name") or "unknown"
    effort = (ac.get("kwargs") or {}).get("reasoning_effort")
    if effort and model != "unknown":
        model = f"{model} / effort={effort}"
    task_name = r.get("task_name") or trial_dir.name.rsplit("-", 1)[0]
    rj_path = trial_dir / "verifier" / "reward.json"
    rj: dict = {}
    if rj_path.is_file():
        try:
            rj = json.loads(rj_path.read_text())
        except (OSError, json.JSONDecodeError):
            rj = {}
    return (name, model, task_name, rj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


CSV_HEADER = [
    "agent",
    "model",
    "task",
    "trial_dir",
    # Implicit-search signal
    "n_examined_files",
    "n_examined_funcs",
    "n_committed_files",
    "n_committed_funcs",
    # Examined scores
    "exam_file_recall",
    "exam_file_precision",
    "exam_file_f1",
    "exam_function_recall",
    "exam_function_precision",
    "exam_function_f1",
    "exam_navigation_score",
    # Committed scores
    "comm_file_recall",
    "comm_file_precision",
    "comm_file_f1",
    "comm_function_recall",
    "comm_function_precision",
    "comm_function_f1",
    "comm_navigation_score",
    # E2E outcome (carried for joining)
    "e2e_resolved",
    "e2e_f2p_passed",
    "e2e_f2p_total",
    "e2e_p2p_passed",
    "e2e_p2p_total",
    # Gold sizes for context
    "gold_n_files",
    "gold_n_functions",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--e2e-roots",
        nargs="+",
        type=Path,
        required=True,
        help="One or more e2e job dirs (each contains <trial>/agent/trajectory.json).",
    )
    ap.add_argument(
        "--patch-gold",
        type=Path,
        required=True,
        help="Patch-derived gold JSON (output of extract_v2b_patch_gold.py).",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    patch_gold = json.loads(args.patch_gold.read_text())
    print(f"Loaded patch-gold for {len(patch_gold)} tasks", file=sys.stderr)

    trials = _find_trial_dirs(args.e2e_roots)
    print(f"Found {len(trials)} e2e trials", file=sys.stderr)

    n_no_gold = 0
    n_no_traj = 0
    n_scored = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for td in sorted(trials, key=lambda p: str(p)):
            agent, model, task, rj = _trial_metadata(td)
            gold = patch_gold.get(task)
            if gold is None:
                n_no_gold += 1
                continue
            traj = _load_trajectory(td)
            if traj is None:
                n_no_traj += 1
                continue

            gold_files = {_normalize_file(f) for f in (gold.get("files") or [])}
            gold_funcs = set(gold.get("functions") or [])

            traj_agent = ((traj.get("agent") or {}).get("name") or "").lower()
            ex_files, ex_funcs, cm_files, cm_funcs = _extract_signals(
                traj, gold_funcs, gold_files, traj_agent
            )
            ex_score = _score(ex_files, ex_funcs, gold_files, gold_funcs)
            cm_score = _score(cm_files, cm_funcs, gold_files, gold_funcs)
            n_scored += 1

            w.writerow(
                [
                    agent,
                    model,
                    task,
                    str(td.relative_to(td.parents[2]) if len(td.parents) >= 2 else td.name),
                    len(ex_files),
                    len(ex_funcs),
                    len(cm_files),
                    len(cm_funcs),
                    ex_score["file_recall"],
                    ex_score["file_precision"],
                    ex_score["file_f1"],
                    ex_score["function_recall"],
                    ex_score["function_precision"],
                    ex_score["function_f1"],
                    ex_score["navigation_score"],
                    cm_score["file_recall"],
                    cm_score["file_precision"],
                    cm_score["file_f1"],
                    cm_score["function_recall"],
                    cm_score["function_precision"],
                    cm_score["function_f1"],
                    cm_score["navigation_score"],
                    int(bool(rj.get("resolved"))) if rj else "",
                    rj.get("f2p_passed", "") if rj else "",
                    rj.get("f2p_total", "") if rj else "",
                    rj.get("p2p_passed", "") if rj else "",
                    rj.get("p2p_total", "") if rj else "",
                    len(gold_files),
                    len(gold_funcs),
                ]
            )

    print(
        f"\nWrote {n_scored} trial rows → {args.output}\n"
        f"  trials with no patch-gold for task: {n_no_gold}\n"
        f"  trials with unparseable trajectory:  {n_no_traj}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
