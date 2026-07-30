#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a two-mode SWE-bench Pro suitability evidence memo."""

from __future__ import annotations

# ruff: noqa: E501,I001

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_STATE = Path("harbor-tasks/craft-tools-v3a/runs/new_pipeline_0427/state.json")
DEFAULT_SOURCE_CSV = Path("docs/analyses/data/swebench-pro/swebench-pro-craft-analysis.csv")
DEFAULT_PROXIES = Path(
    "docs/analyses/data/swebench-pro/findings/swebench_pro_difficulty_proxies_enriched.csv"
)
DEFAULT_OUTCOMES = Path("docs/analyses/data/swebench-pro/findings/task_outcomes_enriched.csv")
DEFAULT_RUNS_DIR = Path("docs/analyses/data/swebench-pro/runs/combined_non_error")
DEFAULT_REPORT = Path("docs/analyses/may06-swebench-pro-suitability.md")
DEFAULT_CASES = Path("docs/analyses/data/swebench-pro/findings/swebench_pro_two_mode_cases.csv")

CASE_ORDER = [
    "qutebrowser-fec187c2",
    "ansible-0fd88717",
    "ansible-bec27fb4",
    "qutebrowser-0833b5f6",
    "ansible-0ea40e09",
    "qutebrowser-fea33d60",
]

CASE_DETAILS = {
    "qutebrowser-fec187c2": {
        "mode": "narrow_tests",
        "reason_category": "unstated search-engine alias/base-url contract",
        "run_dir": "docs/analyses/data/swebench-pro/runs/combined_non_error/instance_qutebrowser__qutebrowse__Bji3fyb",
        "problem_contract": (
            "The task is framed as search URL parameter encoding: encode spaces and special characters, "
            "handle hyphens/spaces, and work across host domains. It says no new public interfaces are introduced."
        ),
        "agent_evidence": (
            "The agent inspected `qutebrowser/utils/urlutils.py`, repeatedly ran search-related urlutils tests, "
            "and concluded the implementation already used `urllib.parse.quote(term, safe='')`, which encodes "
            "slashes, spaces, and `!`. The final trajectory says all 23 search URL tests pass and makes no source "
            "edit for this task."
        ),
        "verifier_evidence": (
            "The verifier passed 242/243 required tests and failed exactly one F2P test: "
            "`tests/unit/utils/test_urlutils.py::test_get_search_url[test path-search-www.qutebrowser.org-q=path-search-True]`. "
            "The test patch adds only one parameter row: `('test path-search', 'www.qutebrowser.org', 'q=path-search')`."
        ),
        "why_claim_challenged": (
            "The failing test is not about percent-encoding; it relies on qutebrowser's search-engine alias parsing "
            "and base URL behavior for the token `test`. A solution that correctly fixes the stated encoding bug can "
            "still fail this hidden row, which undercuts the claim that recovered tests preserve implementation flexibility."
        ),
    },
    "ansible-0fd88717": {
        "mode": "narrow_tests",
        "reason_category": "private helper return-shape contract",
        "run_dir": "docs/analyses/data/swebench-pro/runs/combined_non_error/instance_ansible__ansible-0fd887__2ts3KZw",
        "problem_contract": (
            "The task asks the password lookup plugin to parse password, salt, and ident values, reuse stored ident values, "
            "avoid duplicate writes, validate ident conflicts, and provide clear errors. The benchmark task says no new "
            "interfaces are introduced."
        ),
        "agent_evidence": (
            "The agent first changed `_parse_content()` to return `(password, salt, ident)`, then deliberately backed that "
            "out to preserve the old helper shape and added a separate `_parse_ident()` helper. It updated `run()` to use "
            "stored ident values, validate conflicts, avoid rewrites, and handle duplicated `ident=` fragments. Its repro "
            "script verified parsing and idempotence, and the final trajectory reports the implementation correct."
        ),
        "verifier_evidence": (
            "The verifier passed 26/30 required tests but failed all four new `TestParseContent` rows. The test patch directly "
            "unpacks `_parse_content(content)` into three values for empty, plain-password, salt-only, and salt+ident files."
        ),
        "why_claim_challenged": (
            "The hidden tests reject a reasonable internal design that keeps `_parse_content()` backward-compatible and adds "
            "`_parse_ident()`. The test suite is checking a private helper tuple shape, not just the user-visible lookup behavior."
        ),
    },
    "ansible-bec27fb4": {
        "mode": "narrow_tests",
        "reason_category": "private role-doc helpers + exact no-color formatting",
        "run_dir": "docs/analyses/data/swebench-pro/runs/combined_non_error/instance_ansible__ansible-bec27f__HPrBL4m",
        "result_line": "agent failed; required tests 17/20, with 17/17 P2P passed and 0/3 true F2P passed.",
        "problem_contract": (
            "The task asks for readable `ansible-doc` output, TTY/no-color fallbacks, role summaries, graceful error handling, "
            "FQCN accuracy, URL formatting, and stable diagnostic wording. It does not introduce public interfaces."
        ),
        "agent_evidence": (
            "The agent made broad source changes in `lib/ansible/cli/doc.py` and `lib/ansible/utils/plugin_docs.py`: styled headers, "
            "role listing/doc fallbacks, comma-separated doc-fragment support, no mid-word wrapping, loader fallback behavior, and "
            "warning handling. It ran unit tests plus ansible-doc integration checks and ended with a trajectory summary claiming "
            "unit tests, playbook integration tests, and fixture comparisons all passed."
        ),
        "verifier_evidence": (
            "The verifier passed 17/20 required tests, but those 17 were all P2P/regression tests. It failed all three true F2P tests: "
            "the italic `tty_ify` no-color marker case, `test_rolemixin__build_summary`, and "
            "`test_rolemixin__build_summary_empty_argspec`. The test patch calls private helpers directly and changes "
            "`RoleMixin._build_summary(role_name, collection_name, argspec)` to require a `meta` argument plus an exact "
            "`description: 'UNDOCUMENTED'` field."
        ),
        "why_claim_challenged": (
            "These failures are about exact helper signatures and no-color marker strings, not the broad user-visible doc behavior. "
            "The verifier constrains implementation details that the task text does not name."
        ),
        "extra_bullets": [
            "What the task asks: improve visual formatting and structure of `ansible-doc` output. The problem statement describes flat, hard-to-scan output where required options, nested suboptions, links, and section headers are not visually distinguished. It also asks role summaries/docs to tolerate missing or malformed metadata/argspec, doc fragments to support comma-separated strings, and plugin names to use resolved FQCNs where available.",
            "Requirements: produce concise readable default terminal output; render visual hierarchy in ANSI terminals while keeping stable no-color substitutions; keep consistent section order; clearly mark required options; wrap nested options/return values without mid-word breaks; group role listing entries; include role summary metadata when available; gracefully warn/continue on missing or invalid role metadata; normalize comma-separated doc fragments; and preserve stable output semantics. The interface field says no new interfaces are introduced.",
            "What the agent did: the trajectory shows a broad attempt to solve the CLI-output task, not a targeted hidden-test patch. It added styled headers across doc sections, changed role listing/doc handling to skip bad roles with warnings, split comma-separated documentation fragments, improved wrapping, added role summary fallbacks, and adjusted loader fallback behavior. It also ran local unit/integration-style checks and reported that its available tests and fixture comparisons passed.",
            "What was tested: the verifier ran 20 required tests: 17 P2P/regression tests and three true F2P tests. The agent passed all 17 P2P tests and failed all three F2P tests: ``test_ttyify[I(italic)-`italic`]``, `test_rolemixin__build_summary`, and `test_rolemixin__build_summary_empty_argspec`.",
            "Why the agent failed: the hidden/gold tests required exact private-helper behavior. They directly call `_build_summary(role_name, collection_name, meta, argspec)` and expect a returned summary containing top-level `description: 'UNDOCUMENTED'` and exact `entry_points` shape. The agent only partially handled missing descriptions and did not infer the private helper signature/return-shape contract. Separately, it missed the exact no-color italic substitution from `I(italic)` to backtick-wrapped `italic`.",
            "Fairness read: the failing expectations are directionally related to the requirements, especially stable no-color output and missing role metadata placeholders. The suitability issue is the specificity: a reasonable implementation can substantially improve `ansible-doc` behavior and pass most verifier checks while failing because it did not infer an unstated private API signature and exact string token.",
        ],
    },
    "qutebrowser-0833b5f6": {
        "mode": "trivial_reject",
        "reason_category": "BT1 one-line signal migration",
        "run_dir": "docs/analyses/data/swebench-pro/runs/combined_non_error/instance_qutebrowser__qutebrowse__QT4ByAB",
        "problem_contract": (
            "In WebKit `NetworkReply`, replace the deprecated initial error signal emission with the modern "
            "`errorOccurred` signal."
        ),
        "agent_evidence": (
            "The agent grepped for network reply code, read `qutebrowser/browser/webkit/network/networkreply.py`, and in the sixth "
            "agent step stated: change `self.error.emit(error)` to `self.errorOccurred.emit(error)`. It then made that one edit and "
            "ran the WebKit network reply tests."
        ),
        "verifier_evidence": (
            "The verifier passed 10/10 required tests. The test patch is a one-line expectation update from "
            "`reply.error` to `reply.errorOccurred`."
        ),
        "why_claim_challenged": (
            "This is the kind of deprecation rename a model can solve by grep plus one local edit. It is a weak example of a "
            "challenging industrial task."
        ),
    },
    "ansible-0ea40e09": {
        "mode": "trivial_reject",
        "reason_category": "BT1 standard Python dunder methods",
        "run_dir": "docs/analyses/data/swebench-pro/runs/combined_non_error/instance_ansible__ansible-0ea40e__8Ldvj4f",
        "problem_contract": (
            "`VarsWithSources` must interoperate with mappings for `|`, reverse `|`, and `|=`, and `combine_vars` must work in the "
            "replace path when one operand is `VarsWithSources`."
        ),
        "agent_evidence": (
            "The agent immediately identified the missing `__or__`, `__ror__`, and `__ior__` methods in "
            "`lib/ansible/vars/manager.py`, wrote a short reproduction script showing `dict | VarsWithSources` fails, added the "
            "three dunder methods, and reran the reproduction."
        ),
        "verifier_evidence": (
            "The verifier passed 16/16 required tests. The test patch adds `VarsWithSources()` rows to existing `combine_vars` "
            "parameterized cases."
        ),
        "why_claim_challenged": (
            "The task reduces to implementing standard Python mapping union methods. The trajectory is short and direct, with no broad "
            "system design or ambiguous debugging."
        ),
    },
    "qutebrowser-fea33d60": {
        "mode": "trivial_reject",
        "reason_category": "AL1 parameter passthrough",
        "run_dir": "docs/analyses/data/swebench-pro/runs/combined_non_error/instance_qutebrowser__qutebrowse__QytJTfz",
        "problem_contract": (
            'For the MIME suffix workaround, call `version_check("6.2.3", compiled=False)` and '
            '`version_check("6.7.0", compiled=False)` so the decision uses only the runtime Qt version.'
        ),
        "agent_evidence": (
            "The trajectory identifies the key change immediately: add `compiled=False` to both `version_check` calls in "
            "`qutebrowser/browser/webengine/webview.py`. The agent also clarifies the existing docstring and adjusts a local mock "
            "signature for its own test run."
        ),
        "verifier_evidence": (
            "The verifier passed 20/20 required tests. The F2P test patch asserts the mocked `version()` function is called with "
            "`compiled is False`."
        ),
        "why_claim_challenged": (
            "The tested behavior is a direct parameter passthrough named verbatim in the task. It is mechanical, localized, and has "
            "little room for meaningful strategy divergence."
        ),
    },
}


def pct(num: float, den: float) -> str:
    if den == 0:
        return ""
    return f"{num / den * 100:.1f}%"


def clean(text: object) -> str:
    return str(text or "").replace("\n", " ").strip()


def truncate(text: str, limit: int = 360) -> str:
    text = clean(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def trajectory_metrics(run_dir: Path) -> dict[str, int | str]:
    trajectory_path = run_dir / "agent" / "trajectory.json"
    if not trajectory_path.exists():
        return {
            "agent_turns": "",
            "agent_edit_calls": "",
            "agent_bash_calls": "",
            "agent_all_steps": "",
        }

    with trajectory_path.open() as f:
        trajectory = json.load(f)

    agent_steps = [
        step
        for step in trajectory.get("steps", [])
        if step.get("source") == "agent" and not step.get("extra", {}).get("is_sidechain")
    ]
    meaningful_steps = [
        step for step in agent_steps if clean(step.get("message")).strip() or step.get("tool_calls")
    ]
    edit_calls = 0
    bash_calls = 0
    for step in agent_steps:
        for call in step.get("tool_calls") or []:
            function_name = call.get("function_name")
            if function_name in {"Edit", "Write"}:
                edit_calls += 1
            elif function_name == "Bash":
                bash_calls += 1

    return {
        "agent_turns": len(meaningful_steps),
        "agent_edit_calls": edit_calls,
        "agent_bash_calls": bash_calls,
        "agent_all_steps": len(agent_steps),
    }


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def normalize_reject_pattern(reason: str, task: dict[str, object]) -> str:
    note = clean(task.get("eval_verifier_notes"))
    reason = clean(reason)
    source = f"{note} {reason}"
    if note.startswith("BT1") or "BT1" in source:
        return "BT1_trivial_core_logic"
    if note.startswith("AL1") or "AL1" in source:
        return "AL1_mechanical_or_constructed"
    if note.startswith("SA1") or "SA1" in source:
        return "SA1_obvious_strategy"
    if note.startswith("hard_filter:"):
        return note.split()[0]
    if "no meaningful tests" in source or "no_behavioral_tests" in source:
        return "hard_filter:no_meaningful_tests"
    return "other_reject_reason"


def narrow_cause(reason: str, task: dict[str, object]) -> str:
    audit = task.get("alignment_v4_audit") or {}
    reason = clean(reason).lower()
    causes: list[str] = []
    if audit.get("helpers_access_private_api") or "private" in reason or "non-public" in reason:
        causes.append("private/internal API")
    if audit.get("assertions_format_only") or "exact" in reason or "format" in reason:
        causes.append("exact format/string")
    if audit.get("fixtures_encode_design_choices") or "fixture" in reason:
        causes.append("fixture/design choice")
    if "unstated" in reason or "does not specify" in reason or "additional" in reason:
        causes.append("unstated behavior")
    if not causes:
        causes.append("other narrow-test issue")
    return " + ".join(dict.fromkeys(causes))


def load_reconciled_reasons(source_csv_path: Path) -> dict[str, dict[str, str]]:
    reasons: dict[str, dict[str, str]] = {}
    with source_csv_path.open() as f:
        for row in csv.DictReader(f):
            reasons[row["task_id"]] = {
                "new_eval_reason": row.get("new_eval_reason", ""),
                "new_alignment_reason": row.get("new_alignment_reason", ""),
            }
    return reasons


def load_rows(
    proxies_path: Path,
    outcomes_path: Path,
    state_path: Path,
    source_csv_path: Path,
    runs_dir: Path,
) -> list[dict[str, object]]:
    with state_path.open() as f:
        state = json.load(f)
    tasks = state["tasks"]
    reasons = load_reconciled_reasons(source_csv_path)

    outcomes = {row["task_id"]: row for row in csv.DictReader(outcomes_path.open())}
    rows: list[dict[str, object]] = []
    for row in csv.DictReader(proxies_path.open()):
        task_id = row["task_id"]
        task = tasks[task_id]
        outcome = outcomes[task_id]
        reason_row = reasons[task_id]
        merged: dict[str, object] = {**row}
        merged["task"] = task
        merged["eval_reason"] = reason_row["new_eval_reason"] or task.get("eval_reason", "")
        merged["alignment_reason"] = reason_row["new_alignment_reason"] or task.get("alignment_reason", "")
        merged["agent_success_bool"] = bool_value(row["agent_success"])
        merged["required_passed"] = int(outcome["agent_f2p_tests_passed"])
        merged["required_total"] = int(outcome["agent_f2p_tests_total"])
        merged["true_f2p_total"] = int(row["fail_to_pass_total"])
        merged["trial_name"] = outcome["trial_name"]
        merged["run_dir"] = str(runs_dir / outcome["trial_name"])
        merged["source_diff_lines"] = int(row["patch_changed"])
        merged["test_diff_lines"] = int(row["test_patch_changed"])
        merged["total_diff_lines"] = int(row["patch_changed"]) + int(row["test_patch_changed"])
        merged.update(trajectory_metrics(runs_dir / outcome["trial_name"]))
        merged["reject_pattern"] = (
            normalize_reject_pattern(str(merged["eval_reason"]), task)
            if row["new_eval_verdict"] == "reject"
            else ""
        )
        merged["narrow_cause"] = (
            narrow_cause(str(merged["alignment_reason"]), task)
            if row["alignment_verdict"] == "narrow_tests"
            else ""
        )
        rows.append(merged)
    return rows


def pass_rate_rows(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    counts: dict[str, list[int]] = {}
    for row in rows:
        value = str(row[key])
        counts.setdefault(value, [0, 0])
        counts[value][0] += 1
        counts[value][1] += int(bool(row["agent_success_bool"]))
    out = []
    for value, (n, passed) in sorted(counts.items()):
        out.append({key: value, "n": n, "passed": passed, "pass_rate": pct(passed, n)})
    return out


def reject_pattern_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = Counter(str(row["reject_pattern"]) for row in rows if row["new_eval_verdict"] == "reject")
    out = []
    for pattern, n in groups.most_common():
        group = [row for row in rows if row["reject_pattern"] == pattern]
        passed = sum(1 for row in group if row["agent_success_bool"])
        out.append(
            {
                "reject_pattern": pattern,
                "n": n,
                "passed": passed,
                "pass_rate": pct(passed, n),
            }
        )
    return out


def narrow_cause_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = Counter(str(row["narrow_cause"]) for row in rows if row["alignment_verdict"] == "narrow_tests")
    out = []
    for cause, n in groups.most_common():
        group = [row for row in rows if row["narrow_cause"] == cause]
        passed = sum(1 for row in group if row["agent_success_bool"])
        out.append({"narrow_cause": cause, "n": n, "passed": passed, "pass_rate": pct(passed, n)})
    return out


def trivial_candidate_rows(
    rows: list[dict[str, object]], sort_mode: str, limit: int = 12
) -> list[dict[str, object]]:
    candidates = [
        row
        for row in rows
        if row["new_eval_verdict"] == "reject"
        and row["alignment_verdict"] == "ok"
        and row["agent_success_bool"]
        and row["true_f2p_total"] > 0
        and isinstance(row.get("agent_turns"), int)
    ]

    if sort_mode == "total_diff":
        ordered = sorted(
            candidates,
            key=lambda row: (
                int(row["total_diff_lines"]),
                int(row["agent_turns"]),
                int(row["source_diff_lines"]),
                -int(row["required_total"]),
            ),
        )
    elif sort_mode == "agent_turns":
        ordered = sorted(
            candidates,
            key=lambda row: (
                int(row["agent_turns"]),
                int(row["source_diff_lines"]),
                int(row["test_diff_lines"]),
                -int(row["required_total"]),
            ),
        )
    else:
        raise ValueError(f"unknown sort mode: {sort_mode}")

    return [
        {
            "task_id": row["task_id"],
            "repo": row["repo"],
            "source_diff": row["source_diff_lines"],
            "test_diff": row["test_diff_lines"],
            "total_diff": row["total_diff_lines"],
            "agent_turns": row["agent_turns"],
            "edit_calls": row["agent_edit_calls"],
            "bash_calls": row["agent_bash_calls"],
            "required": f"{row['required_passed']}/{row['required_total']}",
            "reject_pattern": row["reject_pattern"],
        }
        for row in ordered[:limit]
    ]


def case_summary(row: dict[str, object], detail: dict[str, str]) -> dict[str, object]:
    mode = detail["mode"]
    if mode == "narrow_tests":
        claim = "Verifier robustness / implementation flexibility"
        verdict_reason = clean(row.get("alignment_reason", ""))
    else:
        claim = "Challenging industrial task selection"
        verdict_reason = clean(row.get("eval_reason", ""))

    run_dir = detail["run_dir"]
    return {
        "task_id": row["task_id"],
        "repo": row["repo"],
        "mode": mode,
        "claim_challenged": claim,
        "eval_verdict": row["new_eval_verdict"],
        "alignment_verdict": row["alignment_verdict"],
        "agent_success": row["agent_success"],
        "true_f2p_total": row["true_f2p_total"],
        "required_passed": row["required_passed"],
        "required_total": row["required_total"],
        "source_diff_lines": row["source_diff_lines"],
        "test_diff_lines": row["test_diff_lines"],
        "total_diff_lines": row["total_diff_lines"],
        "agent_turns": row["agent_turns"],
        "agent_edit_calls": row["agent_edit_calls"],
        "agent_bash_calls": row["agent_bash_calls"],
        "reason_category": detail["reason_category"],
        "run_dir": run_dir,
        "agent_trajectory": f"{run_dir}/agent/trajectory.json",
        "agent_log": f"{run_dir}/agent/claude-code.txt",
        "verifier_output": f"{run_dir}/verifier/output.json",
        "verifier_stdout": f"{run_dir}/verifier/test-stdout.txt",
        "problem_contract": detail["problem_contract"],
        "agent_trajectory_evidence": detail["agent_evidence"],
        "verifier_evidence": detail["verifier_evidence"],
        "why_claim_challenged": detail["why_claim_challenged"],
        "pipeline_verdict_reason": verdict_reason,
    }


def selected_cases(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id = {str(row["task_id"]): row for row in rows}
    cases: list[dict[str, object]] = []
    for task_id in CASE_ORDER:
        cases.append(case_summary(by_id[task_id], CASE_DETAILS[task_id]))
    return cases


def write_cases(cases: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "repo",
        "mode",
        "claim_challenged",
        "eval_verdict",
        "alignment_verdict",
        "agent_success",
        "true_f2p_total",
        "required_passed",
        "required_total",
        "source_diff_lines",
        "test_diff_lines",
        "total_diff_lines",
        "agent_turns",
        "agent_edit_calls",
        "agent_bash_calls",
        "reason_category",
        "run_dir",
        "agent_trajectory",
        "agent_log",
        "verifier_output",
        "verifier_stdout",
        "problem_contract",
        "agent_trajectory_evidence",
        "verifier_evidence",
        "why_claim_challenged",
        "pipeline_verdict_reason",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)


def case_bullets(cases: list[dict[str, object]], mode: str) -> list[str]:
    lines: list[str] = []
    for case in [case for case in cases if case["mode"] == mode]:
        status = "passed" if str(case["agent_success"]).lower() == "true" else "failed"
        detail = CASE_DETAILS.get(str(case["task_id"]), {})
        result_line = detail.get(
            "result_line",
            f"agent {status}; required tests {case['required_passed']}/{case['required_total']}; true F2P tests {case['true_f2p_total']}.",
        )
        lines.extend(
            [
                f"### `{case['task_id']}` ({case['repo']})",
                "",
                f"- Result: {result_line}",
                f"- Heuristics: source diff {case['source_diff_lines']} lines; test diff {case['test_diff_lines']} lines; top-level agent turns {case['agent_turns']}; edit calls {case['agent_edit_calls']}; bash calls {case['agent_bash_calls']}.",
                f"- Category: {case['reason_category']}.",
                f"- Raw artifacts: `{case['agent_trajectory']}`, `{case['verifier_output']}`, `{case['verifier_stdout']}`.",
                f"- Task contract: {case['problem_contract']}",
                f"- Agent trajectory evidence: {case['agent_trajectory_evidence']}",
                f"- Verifier evidence: {case['verifier_evidence']}",
                f"- Why this challenges the claim: {case['why_claim_challenged']}",
                "",
            ]
        )
        extra_bullets = detail.get("extra_bullets", [])
        if extra_bullets:
            lines.extend(["Detailed read:", ""])
            lines.extend(f"- {bullet}" for bullet in extra_bullets)
            lines.append("")
    return lines


def make_report(
    rows: list[dict[str, object]], cases: list[dict[str, object]], report_path: Path, cases_path: Path
) -> None:
    total = len(rows)
    narrow = [row for row in rows if row["alignment_verdict"] == "narrow_tests"]
    reject = [row for row in rows if row["new_eval_verdict"] == "reject"]
    accept = [row for row in rows if row["new_eval_verdict"] == "accept"]

    reject_passed = sum(1 for row in reject if row["agent_success_bool"])
    accept_passed = sum(1 for row in accept if row["agent_success_bool"])
    narrow_passed = sum(1 for row in narrow if row["agent_success_bool"])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SWE-bench Pro Suitability: Narrow Tests and Trivial Tasks",
        "",
        "This memo focuses on two task-suitability modes in our 263 evaluated SWE-bench Pro tasks.",
        "",
        "- Mode 1: `narrow_tests` verdicts, which challenge verifier robustness and implementation flexibility.",
        "- Mode 2: `reject` verdicts for trivial/mechanical tasks, which challenge the claim that tasks are consistently challenging and industrially relevant.",
        "",
        "Paper claims targeted: SWE-bench Pro says it emphasizes `challenging, diverse, and industrially relevant tasks`, and that its human workflow recovers unit tests as `robust verifiers` while maintaining `implementation flexibility`.",
        "",
        "The case studies below are selected from raw run evidence, not just verdict labels. For each example I inspected the task text/test patch in `swebench_pro.jsonl`, the agent trajectory, and the verifier outputs under `docs/analyses/data/swebench-pro/runs/combined_non_error`.",
        "",
        f"Case index: `{cases_path}`",
        "",
        "## Headline Counts",
        "",
        f"- Narrow-test tasks: {len(narrow)}/{total} ({pct(len(narrow), total)}); agent passed {narrow_passed}/{len(narrow)} ({pct(narrow_passed, len(narrow))}).",
        f"- Rejected tasks: {len(reject)}/{total} ({pct(len(reject), total)}); agent passed {reject_passed}/{len(reject)} ({pct(reject_passed, len(reject))}).",
        f"- Accepted tasks: {len(accept)}/{total} ({pct(len(accept), total)}); agent passed {accept_passed}/{len(accept)} ({pct(accept_passed, len(accept))}).",
        "",
        "Rejected tasks passing more often than accepted tasks is not proof by itself, but it is consistent with the evaluator's triviality judgments.",
        "",
        "## Aggregate Tables",
        "",
        "### Pass Rate By Evaluation Verdict",
        "",
    ]
    lines.extend(
        markdown_table(
            pass_rate_rows(rows, "new_eval_verdict"), ["new_eval_verdict", "n", "passed", "pass_rate"]
        )
    )
    lines.extend(["", "### Pass Rate By Alignment Verdict", ""])
    lines.extend(
        markdown_table(
            pass_rate_rows(rows, "alignment_verdict"), ["alignment_verdict", "n", "passed", "pass_rate"]
        )
    )
    lines.extend(["", "### Reject Patterns", ""])
    lines.extend(markdown_table(reject_pattern_rows(rows), ["reject_pattern", "n", "passed", "pass_rate"]))
    lines.extend(["", "### Narrow-Test Cause Buckets", ""])
    lines.extend(markdown_table(narrow_cause_rows(rows), ["narrow_cause", "n", "passed", "pass_rate"]))

    lines.extend(
        [
            "",
            "## Mode 1: Narrow Tests Causing Unfair Failures",
            "",
            "`narrow_tests` means the reference tests require behavior or implementation choices that the instruction does not specify. This directly pressures the paper's verifier-quality claim because a semantically valid alternative could fail hidden tests.",
            "",
            "These examples are deliberately not generic label summaries. They tie the task contract to what the agent actually did and the exact hidden-test failure.",
            "",
        ]
    )
    lines.extend(case_bullets(cases, "narrow_tests"))

    lines.extend(
        [
            "## Mode 2: Rejected Tasks That Look Too Trivial",
            "",
            "Our evaluator rejects tasks when the core work is mechanical, obvious, or too thinly verified. These examples challenge the broad claim that the benchmark is consistently composed of challenging industrial tasks.",
            "",
            "For this pass I used additional heuristics to search for examples: only successful rejected tasks with `alignment_verdict=ok`, at least one true F2P test, small source/test diffs, and short top-level agent trajectories. That avoids relying only on the evaluator label.",
            "",
            "### Candidate Search: Shortest Total Diff",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            trivial_candidate_rows(rows, "total_diff"),
            [
                "task_id",
                "repo",
                "source_diff",
                "test_diff",
                "total_diff",
                "agent_turns",
                "edit_calls",
                "bash_calls",
                "required",
                "reject_pattern",
            ],
        )
    )
    lines.extend(
        [
            "",
            "### Candidate Search: Fewest Agent Turns",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            trivial_candidate_rows(rows, "agent_turns"),
            [
                "task_id",
                "repo",
                "source_diff",
                "test_diff",
                "total_diff",
                "agent_turns",
                "edit_calls",
                "bash_calls",
                "required",
                "reject_pattern",
            ],
        )
    )
    lines.extend(
        [
            "",
            "### Selected Examples",
            "",
            "These selected examples cover the shortest-total-diff and fewest-agent-turn rankings while still having clean raw trajectories.",
            "",
        ]
    )
    lines.extend(case_bullets(cases, "trivial_reject"))

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The narrow-test mode is a verifier suitability concern: it does not require the task to be easy or hard; it means the verifier may reject reasonable implementations because the tests encode unstated specifics.",
            "- The trivial-reject mode is a task-selection concern: many rejected tasks fall into `BT1` or `AL1`, and in our run rejected tasks pass substantially more often than accepted tasks.",
            "- These results do not invalidate the full dataset. They show that a meaningful subset of the evaluated public tasks may not satisfy the paper's stated suitability criteria.",
            "",
            "## Limitations",
            "",
            "- Scope is the 263 tasks in our evaluated run, not all SWE-bench Pro tasks.",
            "- `accept`/`reject` and alignment verdicts are model-based pipeline judgments, so case evidence matters more than labels alone.",
            "- Pass rate depends on the agent/scaffold; use it as supporting evidence for triviality, not as the only criterion.",
            "- Older columns named `agent_f2p_tests_*` in some CSVs are verifier required-test counts, not true F2P counts.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--proxies", type=Path, default=DEFAULT_PROXIES)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = parser.parse_args()

    rows = load_rows(args.proxies, args.outcomes, args.state, args.source_csv, args.runs_dir)
    cases = selected_cases(rows)
    write_cases(cases, args.cases)
    make_report(rows, cases, args.report, args.cases)
    print(f"rows: {len(rows)}")
    print(f"wrote: {args.report}")
    print(f"wrote: {args.cases}")


if __name__ == "__main__":
    main()
