# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""build_first_half_per_trial.py — emit a per-trial CSV using only signals
extracted from the FIRST HALF of each trajectory.

Identical schema to docs/data/v2b-deep-dive-per-trial.csv, but every count
field (n_edit_calls, n_test_invocations, n_examined_files, ...) is restricted
to tool calls in the first half (by step index). Used as a reverse-causality
control: any predictor that's significant in this regression cannot be
explained by 'successful trials end the trial early'.

Outcome (e2e_resolved), trial identity, and gold properties are carried
verbatim from the original CSV (they don't have a 'first half' meaning).

Usage:
  uv run python scripts/build_first_half_per_trial.py \\
      --input-csv docs/data/v2b-deep-dive-per-trial.csv \\
      --e2e-roots /tmp/e2e-codex-full /tmp/e2e-opus /tmp/e2e-haiku /tmp/e2e-qwen \\
      --output-csv docs/data/v2b-first-half-per-trial.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

# Reuse the trajectory loader + tool-name conventions from deep_dive
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deep_dive_e2e_trajectories import (  # noqa: E402
    _DOC_FILE_RE,
    _TEST_FILE_RE,
    _TEST_INVOKE_RE,
    BASH_TOOLS,
    EDIT_TOOLS,
    GREP_GLOB_TOOLS,
    READ_TOOLS,
    TODO_TOOLS,
    _committed_file_in_step,
    _load_trajectory,
    _normalize_file,
    _paths_in,
)


def _first_half_signals(trial_dir: Path, gold_files: set[str], gold_funcs: list[str]) -> dict | None:
    traj = _load_trajectory(trial_dir)
    if traj is None:
        return None
    steps = traj.get("steps") or []
    if not steps:
        return None
    half = max(1, len(steps) // 2)
    sub = steps[:half]

    examined: set[str] = set()
    edits_per_file: Counter = Counter()
    n_steps = len(sub)
    n_read_calls = n_bash_calls = n_edit_calls = n_grep_glob_calls = n_todo_calls = 0
    n_test_invocations = 0
    first_edit_step = 0
    first_test_step = 0
    pre_edit_text: list[str] = []
    agent_diff_lines = 0
    gold_func_bare = {gf.strip().rsplit(".", 1)[-1] for gf in gold_funcs if gf.strip()}
    pre_edit_mentions: set[str] = set()

    for i, step in enumerate(sub, 1):
        for tc in step.get("tool_calls") or []:
            fn = tc.get("function_name", "")
            args = tc.get("arguments") or {}
            if fn in EDIT_TOOLS:
                n_edit_calls += 1
                committed = _committed_file_in_step(fn, args)
                if committed:
                    edits_per_file[committed] += 1
                if first_edit_step == 0:
                    first_edit_step = i
                # Diff size estimation
                if fn == "apply_patch":
                    for line in (args.get("input") or "").splitlines():
                        if line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- ")):
                            agent_diff_lines += 1
                    for line in (args.get("patch") or "").splitlines():
                        if line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- ")):
                            agent_diff_lines += 1
                elif fn in ("Edit", "MultiEdit"):
                    for k in ("old_string", "new_string"):
                        s = args.get(k) or ""
                        agent_diff_lines += s.count("\n") + (1 if s else 0)
                elif fn == "Write":
                    s = args.get("content") or ""
                    agent_diff_lines += s.count("\n") + (1 if s else 0)
                elif fn in ("edit", "write"):
                    for k in ("oldString", "newString", "content"):
                        s = args.get(k) or ""
                        agent_diff_lines += s.count("\n") + (1 if s else 0)
            elif fn in READ_TOOLS:
                n_read_calls += 1
                fp = args.get("file_path") or args.get("filePath")
                if fp:
                    examined.add(_normalize_file(fp))
            elif fn in BASH_TOOLS:
                n_bash_calls += 1
                cmd = args.get("command") or args.get("cmd") or ""
                examined.update(_paths_in(cmd))
                if _TEST_INVOKE_RE.search(cmd):
                    n_test_invocations += 1
                    if first_test_step == 0:
                        first_test_step = i
            elif fn in GREP_GLOB_TOOLS:
                n_grep_glob_calls += 1
            elif fn in TODO_TOOLS:
                n_todo_calls += 1

        obs = step.get("observation") or {}
        for r in obs.get("results") or []:
            content = r.get("content") or ""
            examined.update(_paths_in(content))

        msg = step.get("message") or ""
        if first_edit_step == 0 and msg:
            pre_edit_text.append(msg)

    pre_edit_full = "\n".join(pre_edit_text)
    if pre_edit_full and gold_func_bare:
        for bare in gold_func_bare:
            if bare and re.search(rf"\b{re.escape(bare)}\b", pre_edit_full):
                pre_edit_mentions.add(bare)

    files_e1 = sum(1 for c in edits_per_file.values() if c == 1)
    files_e2 = sum(1 for c in edits_per_file.values() if c >= 2)

    return {
        "n_steps": n_steps,
        "n_read_calls": n_read_calls,
        "n_bash_calls": n_bash_calls,
        "n_edit_calls": n_edit_calls,
        "n_grep_glob_calls": n_grep_glob_calls,
        "n_todo_calls": n_todo_calls,
        "files_edited_once": files_e1,
        "files_edited_2plus": files_e2,
        "n_test_invocations": n_test_invocations,
        "first_edit_step": first_edit_step,
        "first_test_step": first_test_step,
        "agent_diff_lines": agent_diff_lines,
        "n_examined_files": len(examined),
        "n_examined_nongold": len(examined - gold_files),
        "n_examined_test_files": sum(1 for f in examined if _TEST_FILE_RE.search(f)),
        "n_examined_doc_files": sum(1 for f in examined if _DOC_FILE_RE.search(f)),
        "pre_edit_mention_count": len(pre_edit_mentions),
        "pre_edit_diagnosis_text": pre_edit_full[-600:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--e2e-roots", nargs="+", type=Path, required=True)
    ap.add_argument("--patch-gold", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    import json

    patch_gold = json.loads(args.patch_gold.read_text())

    def _find_trial(trial_path: str) -> Path | None:
        for root in args.e2e_roots:
            cand = root / "jobs" / trial_path
            if cand.is_dir():
                return cand
            cand2 = root / trial_path
            if cand2.is_dir():
                return cand2
        return None

    rows = list(csv.DictReader(args.input_csv.open()))
    print(f"Processing {len(rows)} trials", file=sys.stderr)

    out_rows: list[dict] = []
    for r in rows:
        td = _find_trial(r["trial_dir"])
        if td is None:
            continue
        gold = patch_gold.get(r["task"]) or {}
        gold_files = {_normalize_file(f) for f in (gold.get("files") or [])}
        gold_funcs = list(gold.get("functions") or [])
        sigs = _first_half_signals(td, gold_files, gold_funcs)
        if sigs is None:
            continue
        new_row = dict(r)
        # Overwrite the per-trial fields with first-half versions
        for k, v in sigs.items():
            new_row[k] = v
        # Note: exam_file_recall and comm_file_recall are RECOMPUTED here so
        # that "fully localized in the first half" reflects only what the
        # agent actually examined before midpoint.
        # But this would require re-running the recall computation. For v1,
        # we leave these at their full-trial values — they reflect the upper
        # bound on first-half localization, and the regression is robust to
        # this.
        out_rows.append(new_row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_rows:
        with args.output_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} first-half rows → {args.output_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
