# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-off calibration: replay the new direct-API deep-dive judge on historical trials.

Two modes:

  --mode fixtures (default)
    Reads a fixture bundle with pre-captured harbor-lab outputs, e.g.
    tmp2/dd-fixtures-20260422/ (10 curated samples). Small cohort, fast
    turnaround, no harbor-lab dependency.

    samples/<bucket>__<task_id>/
      instruction.md
      tests/...
      trial/verifier/{reward.json,verify_full_output.txt}
      harbor_lab/hlab_{errors,edits,tool_sequence,metrics}.md
      diagnostics/NNN_triage_opus.md          # historical deep-dive verdict
      fixture_info.json

  --mode run_dir
    Reads a live pipeline run directory (e.g. harbor-tasks/craft-tools-v4/
    runs/<ts>/) and its state.json, shells out to harbor-lab for each task
    that made it to Opus triage, and compares the new direct-API verdict
    against diagnostics/NNN_triage_opus.md in the task dir. Designed to
    run on the VM (craftbench02) where jobs/ and harbor-lab are available.
    Bigger cohort (~40+ tasks), higher wall cost (~1-2 min / task).

In either mode the script calls `llm_judge.judge(prompt, DEEP_DIVE_SCHEMA,
model=LLM_STEP_MODEL)` per task and writes a per-sample + optional
per-pair CSV comparing new verdicts against historical ones.

Purpose: directional validation that the new direct-API deep dive produces
verdicts comparable to the `claude -p` version it replaces. Phase A:
eyeball per-failure agreement, spot-check regressions. Not statistical.

IMPORTANT — this script emits RAW DD classifications. The production
pipeline applies two filters inside `_run_triage_one` AFTER the judges
return and BEFORE the action-routing logic:

  1. Skip-filter (`_load_skipped_tests`): drops classifications for tests
     already in `f2p_skip.txt` / `p2p_skip.txt` (already-decided skips).
  2. Reward-filter (`_load_actually_failed_tests`): drops classifications
     on tests that actually PASSED in the trial.

Calibration CSV counts can therefore include classifications the live
pipeline would drop before acting. When cross-referencing against
production behavior (e.g. "how many tests would get skipped"), subtract
the already-skipped set per task (read `tests/f2p_skip.txt`,
`tests/p2p_skip.txt` from the run dir). There is currently NO F2P/P2P
scope-membership filter post-DD — the LLM's prompt-level scoping rule
is the only thing telling it not to classify out-of-scope tests, and
it isn't perfectly obeyed. Future work: add a deterministic scope
filter in `_run_triage_one` alongside the two above.

Usage:
    # Fixture mode (default, 10 samples):
    uv run python scripts/calibrate-deep-dive.py \\
        --output deep_dive_calibration.csv \\
        --concurrency 3

    # Single-sample dry run:
    uv run python scripts/calibrate-deep-dive.py \\
        --only scope_reject_A__fiftyone-8359f434 --verbose

    # Live run-dir mode (craftbench02, full cohort):
    uv run python scripts/calibrate-deep-dive.py \\
        --mode run_dir \\
        --run-dir harbor-tasks/craft-tools-v4/runs/2026-04-17-015355/ \\
        --output deep_dive_apr17.csv --dump-pairs deep_dive_apr17_pairs.csv \\
        --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

import craft_taskgen.config as _cfg
from craft_taskgen import llm_judge
from craft_taskgen.prompts import (
    DEEP_DIVE_SCHEMA,
    FAIRNESS_REVIEW_SCHEMA,
    REVIEWER_SCHEMA,
    deep_dive_prompt,
    fairness_review_prompt,
    skeptical_reviewer_prompt,
)
from craft_taskgen.steps import (
    _DEEP_DIVE_EDITS_CAP_CHARS,
    _DEEP_DIVE_HARBOR_LAB_CAP_CHARS,
    _DEEP_DIVE_TEST_BODY_CAP_CHARS,
    _DEEP_DIVE_VERIFY_TAIL_CHARS,
    _cap,
    _fetch_deep_dive_context,
)

load_dotenv()


def _read_or_empty(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def load_sample_context(sample_dir: Path) -> dict:
    """Pre-assemble deep-dive inputs from a fixture sample dir.

    Matches `_fetch_deep_dive_context`'s output shape so we can feed the
    same `deep_dive_prompt` builder. Harbor-lab output comes from
    pre-captured markdown rather than live subprocess calls.
    """
    instruction_md = _read_or_empty(sample_dir / "instruction.md")

    verify_text = _read_or_empty(sample_dir / "trial" / "verifier" / "verify_full_output.txt")
    verify_tail = _cap(verify_text, _DEEP_DIVE_VERIFY_TAIL_CHARS) if verify_text else ""

    reward_json = _read_or_empty(sample_dir / "trial" / "verifier" / "reward.json")

    tests_dir = sample_dir / "tests"
    f2p = _read_or_empty(tests_dir / "fail_to_pass.txt")
    p2p = _read_or_empty(tests_dir / "pass_to_pass.txt")
    f2p_skip = _read_or_empty(tests_dir / "f2p_skip.txt")
    p2p_skip = _read_or_empty(tests_dir / "p2p_skip.txt")

    bodies: list[tuple[str, str]] = []
    postmerge_dir = tests_dir / "postmerge_tests"
    if postmerge_dir.is_dir():
        remaining = _DEEP_DIVE_TEST_BODY_CAP_CHARS
        for p in sorted(postmerge_dir.rglob("*.py")):
            if remaining <= 0:
                break
            rel = str(p.relative_to(postmerge_dir))
            body = _read_or_empty(p)
            if len(body) > remaining:
                body = _cap(body, remaining)
            bodies.append((rel, body))
            remaining -= len(body)

    hlab_dir = sample_dir / "harbor_lab"
    hlab_errors = _cap(_read_or_empty(hlab_dir / "hlab_errors.md"), _DEEP_DIVE_HARBOR_LAB_CAP_CHARS)
    hlab_edits = _cap(_read_or_empty(hlab_dir / "hlab_edits.md"), _DEEP_DIVE_EDITS_CAP_CHARS)
    hlab_tool_seq = _cap(_read_or_empty(hlab_dir / "hlab_tool_sequence.md"), _DEEP_DIVE_HARBOR_LAB_CAP_CHARS)
    hlab_metrics = _cap(_read_or_empty(hlab_dir / "hlab_metrics.md"), _DEEP_DIVE_HARBOR_LAB_CAP_CHARS)

    return {
        "instruction_md": instruction_md,
        "reward_json": reward_json,
        "verify_output_tail": verify_tail,
        "postmerge_test_bodies": bodies,
        "f2p_tests": f2p,
        "p2p_tests": p2p,
        "f2p_skip": f2p_skip,
        "p2p_skip": p2p_skip,
        "harbor_lab_errors": hlab_errors,
        "harbor_lab_edits": hlab_edits,
        "harbor_lab_tool_sequence": hlab_tool_seq,
        "harbor_lab_metrics": hlab_metrics,
    }


_TRIAGE_FAILURE_RE = re.compile(
    r"###\s+(?P<test>[^\n]+)\n"
    r"- \*\*Classification:\*\*\s+(?P<cls>\S+)\s*\n"
    r"(?:- \*\*Evidence:\*\*\s+(?P<evidence>.+?)(?=\n\n|\n### |\n## |\Z))?",
    re.MULTILINE | re.DOTALL,
)


def parse_historical_triage(md: str) -> list[dict]:
    """Extract `{test_name, classification, evidence}` entries from diagnostics/NNN_triage_opus.md."""
    out: list[dict] = []
    for m in _TRIAGE_FAILURE_RE.finditer(md):
        out.append(
            {
                "test_name": m.group("test").strip(),
                "classification": m.group("cls").strip(),
                "evidence": (m.group("evidence") or "").strip(),
            }
        )
    return out


def latest_triage_file(sample_dir: Path) -> Path | None:
    """Return the highest-numbered triage_opus.md in diagnostics/."""
    diag_dir = sample_dir / "diagnostics"
    if not diag_dir.is_dir():
        return None
    candidates = sorted(diag_dir.glob("*_triage_opus.md"))
    return candidates[-1] if candidates else None


def latest_review_file(sample_dir: Path) -> Path | None:
    """Return the highest-numbered review_opus.md in diagnostics/."""
    diag_dir = sample_dir / "diagnostics"
    if not diag_dir.is_dir():
        return None
    candidates = sorted(diag_dir.glob("*_review_opus.md"))
    return candidates[-1] if candidates else None


_REVIEW_CHALLENGE_RE = re.compile(
    r"###\s+(?P<test>[^\n]+)\n"
    r"- \*\*Original:\*\*\s+(?P<orig>\S+)\s*\n"
    r"- \*\*Revised:\*\*\s+(?P<revised>\S+)",
    re.MULTILINE,
)


def parse_historical_review(md: str) -> dict[str, str]:
    """Extract `{test_name: revised_classification}` from diagnostics/NNN_review_opus.md.
    Tests not listed in the Challenges section keep their deep-dive classification."""
    out: dict[str, str] = {}
    for m in _REVIEW_CHALLENGE_RE.finditer(md):
        out[m.group("test").strip()] = m.group("revised").strip()
    return out


def apply_reclassifications(failures: list[dict], reclass_map: dict[str, str]) -> list[dict]:
    """Return a new failures list with classifications overridden by reclass_map where
    the test name matches (exact match). Preserves original entry otherwise."""
    out: list[dict] = []
    for f in failures:
        name = f.get("test_name", "")
        new_class = reclass_map.get(name)
        if new_class and new_class != f.get("classification"):
            f = dict(f)  # copy
            f["classification"] = new_class
        out.append(f)
    return out


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{100 * num / denom:.0f}%"


def _normalize_test_name(name: str) -> str:
    """For agreement comparison, use the part after the final '::'.

    Historical triage files often list fully-qualified paths
    (`tests/foo.py::test_bar` or `path::TestClass::test_bar`), while the
    new judge sometimes returns just the method name. Match on the final
    segment so format differences don't show up as false disagreements.
    """
    if not name:
        return ""
    return name.split("::")[-1].strip()


def _compare_verdicts(new_failures: list[dict], historical_failures: list[dict]) -> dict:
    """Compare new-judge vs historical per-failure classifications using
    basename-normalized test names. Returns a dict with matched/mismatched
    counts, itemized drift, and a full list of pairs (one per test seen in
    either verdict) for optional dump-pairs export.
    """
    # basename -> full failure dict
    new_by_basename: dict[str, dict] = {}
    for f in new_failures:
        new_by_basename[_normalize_test_name(f.get("test_name", ""))] = f

    hist_by_basename: dict[str, dict] = {}
    for f in historical_failures:
        hist_by_basename[_normalize_test_name(f.get("test_name", ""))] = f

    common = set(new_by_basename) & set(hist_by_basename)
    only_new = set(new_by_basename) - set(hist_by_basename)
    only_hist = set(hist_by_basename) - set(new_by_basename)

    same_class = 0
    drift: list[str] = []
    pairs: list[dict] = []

    for base in sorted(common | only_new | only_hist):
        new_f = new_by_basename.get(base, {})
        hist_f = hist_by_basename.get(base, {})
        new_c = new_f.get("classification", "")
        hist_c = hist_f.get("classification", "")

        if base in common and new_c == hist_c:
            same_class += 1
            status = "agree"
        elif base in common:
            drift.append(f"{base}: old={hist_c} → new={new_c}")
            status = "drift"
        elif base in only_new:
            status = "only_new"
        else:
            status = "only_hist"

        pairs.append(
            {
                "test_basename": base,
                "status": status,
                "historical_classification": hist_c or "MISSING",
                "historical_evidence": hist_f.get("evidence", ""),
                "historical_test_name": hist_f.get("test_name", ""),
                "new_classification": new_c or "MISSING",
                "new_evidence": new_f.get("evidence", ""),
                "new_test_name": new_f.get("test_name", ""),
            }
        )

    return {
        "common_tests": len(common),
        "same_class": same_class,
        "different_class": len(common) - same_class,
        "only_new": sorted(only_new),
        "only_hist": sorted(only_hist),
        "drift": drift,
        "pairs": pairs,
    }


def _latest_triage_in_dir(diag_dir: Path) -> Path | None:
    """Return the highest-numbered *_triage_opus.md inside diag_dir."""
    if not diag_dir.is_dir():
        return None
    candidates = sorted(diag_dir.glob("*_triage_opus.md"))
    return candidates[-1] if candidates else None


# Skip-worthy per reviewer-prompt convention (broader than steps.py SKIP_WORTHY
# which uses just the narrow auto-skip set — reviewer workflow STEP 4 treats
# instruction_scope as skip-worthy too).
_DUAL_DD_SKIP_SET = frozenset(
    {"test_format_only", "test_not_relevant", "instruction_missing_symbol", "instruction_scope"}
)


def _dual_dd_agreement_status(primary_class: str, secondary_class: str) -> str:
    """Classify a pair of DD verdicts by action-level agreement:

    - agree_exact: identical classification string
    - agree_skip: both classifications are skip-worthy (action agrees: skip)
    - agree_nonskip: both are non-skip (action agrees: keep test counted)
    - disagree_across: one skip, the other non-skip (action disagrees)
    - missing: one side did not classify this test at all
    """
    if not primary_class and not secondary_class:
        return "missing"
    if not primary_class or not secondary_class:
        return "missing"
    if primary_class == secondary_class:
        return "agree_exact"
    p_skip = primary_class in _DUAL_DD_SKIP_SET
    s_skip = secondary_class in _DUAL_DD_SKIP_SET
    if p_skip and s_skip:
        return "agree_skip"
    if not p_skip and not s_skip:
        return "agree_nonskip"
    return "disagree_across"


async def _judge_dual_dd(
    *,
    task_id: str,
    bucket: str,
    opus_score: str,
    stage: str,
    triage_rounds: str | int,
    context: dict,
    model_primary: str,
    model_secondary: str,
    verbose: bool,
    header_note: str = "",
) -> dict:
    """Run deep dive twice in parallel with two different models; emit
    per-test inter-model agreement stats. Skips reviewer entirely — this mode
    is calibrating whether cross-family dual DD can stand in for the reviewer
    layer."""
    prompt = deep_dive_prompt(**{k: v for k, v in context.items() if k != "harbor_lab_tool_sequence_full"})

    lines: list[str] = []
    lines.append(f"=== {task_id} ({bucket}) ===")
    note = f"  {header_note}, " if header_note else "  "
    lines.append(f"{note}score: {opus_score}, prompt len: {len(prompt)} chars")
    lines.append(f"  primary: {model_primary}")
    lines.append(f"  secondary: {model_secondary}")

    try:
        primary_result, secondary_result = await asyncio.gather(
            llm_judge.judge(prompt=prompt, schema=DEEP_DIVE_SCHEMA, model=model_primary),
            llm_judge.judge(prompt=prompt, schema=DEEP_DIVE_SCHEMA, model=model_secondary),
        )
    except Exception as e:
        lines.append(f"  ERROR (dual DD): {type(e).__name__}: {e}")
        if verbose:
            print("\n".join(lines))
        return {"task_id": task_id, "bucket": bucket, "error": f"{type(e).__name__}: {e}"}

    p_failures = primary_result.result.get("failures", [])
    s_failures = secondary_result.result.get("failures", [])

    p_by_base = {}
    for f in p_failures:
        p_by_base[_normalize_test_name(f.get("test_name", ""))] = f
    s_by_base = {}
    for f in s_failures:
        s_by_base[_normalize_test_name(f.get("test_name", ""))] = f

    all_bases = set(p_by_base) | set(s_by_base)
    pairs: list[dict] = []
    agree_exact = agree_skip = agree_nonskip = disagree_across = only_primary = only_secondary = 0
    for base in sorted(all_bases):
        p_f = p_by_base.get(base, {})
        s_f = s_by_base.get(base, {})
        p_c = p_f.get("classification", "")
        s_c = s_f.get("classification", "")
        status = _dual_dd_agreement_status(p_c, s_c)
        if status == "agree_exact":
            agree_exact += 1
        elif status == "agree_skip":
            agree_skip += 1
        elif status == "agree_nonskip":
            agree_nonskip += 1
        elif status == "disagree_across":
            disagree_across += 1
        elif status == "missing":
            if p_c and not s_c:
                only_primary += 1
            elif s_c and not p_c:
                only_secondary += 1

        pairs.append(
            {
                "test_basename": base,
                "status": status,
                "primary_classification": p_c or "MISSING",
                "secondary_classification": s_c or "MISSING",
                "primary_evidence": (p_f.get("evidence", "") or "")[:500],
                "secondary_evidence": (s_f.get("evidence", "") or "")[:500],
                "primary_test_name": p_f.get("test_name", ""),
                "secondary_test_name": s_f.get("test_name", ""),
            }
        )

    common = agree_exact + agree_skip + agree_nonskip + disagree_across
    action_agree = agree_exact + agree_skip + agree_nonskip  # merge-compatible
    p_assess = primary_result.result.get("overall_assessment", "")[:80]
    s_assess = secondary_result.result.get("overall_assessment", "")[:80]
    lines.append(f"  primary DD: {len(p_failures)} failures — {p_assess!r}")
    lines.append(f"  secondary DD: {len(s_failures)} failures — {s_assess!r}")
    lines.append(
        f"  ACTION-AGREE: {action_agree}/{common} ({_pct(action_agree, common)}) "
        f"[exact {agree_exact}, skip {agree_skip}, nonskip {agree_nonskip}]"
    )
    lines.append(
        f"  DISAGREE_ACROSS: {disagree_across}/{common} ({_pct(disagree_across, common)}) "
        f"[only_primary {only_primary}, only_secondary {only_secondary}]"
    )

    if verbose:
        print("\n".join(lines))

    return {
        "task_id": task_id,
        "bucket": bucket,
        "opus_score": opus_score,
        "stage": stage,
        "triage_rounds": triage_rounds,
        "primary_model": model_primary,
        "secondary_model": model_secondary,
        "primary_n_failures": len(p_failures),
        "secondary_n_failures": len(s_failures),
        "primary_assessment": primary_result.result.get("overall_assessment", "")[:200],
        "secondary_assessment": secondary_result.result.get("overall_assessment", "")[:200],
        "common_tests": common,
        "agree_exact": agree_exact,
        "agree_skip": agree_skip,
        "agree_nonskip": agree_nonskip,
        "disagree_across": disagree_across,
        "only_primary": only_primary,
        "only_secondary": only_secondary,
        "action_agree_pct": _pct(action_agree, common),
        "disagree_across_pct": _pct(disagree_across, common),
        "pairs": pairs,
        "primary_tokens_in": primary_result.usage.get("input_tokens", 0),
        "primary_tokens_out": primary_result.usage.get("output_tokens", 0),
        "primary_latency_s": round(primary_result.latency_s, 2),
        "secondary_tokens_in": secondary_result.usage.get("input_tokens", 0),
        "secondary_tokens_out": secondary_result.usage.get("output_tokens", 0),
        "secondary_latency_s": round(secondary_result.latency_s, 2),
        "error": "",
    }


async def _judge_and_compare(
    *,
    task_id: str,
    bucket: str,
    opus_score: str,
    stage: str,
    triage_rounds: str | int,
    context: dict,
    historical_file: Path | None,
    review_historical_file: Path | None,
    model: str,
    reviewer_model: str,
    skip_reviewer: bool,
    verbose: bool,
    header_note: str = "",
) -> dict:
    """Run Opus deep-dive + Gemini reviewer (matching production triage) and
    compare the FINAL per-failure classification (post-reviewer reclass) against
    the historical final (deep-dive classifications overridden by
    016_review_opus.md reclassifications)."""
    prompt = deep_dive_prompt(**{k: v for k, v in context.items() if k != "harbor_lab_tool_sequence_full"})

    lines: list[str] = []
    lines.append(f"=== {task_id} ({bucket}) ===")
    note = f"  {header_note}, " if header_note else "  "
    lines.append(f"{note}score: {opus_score}, prompt len: {len(prompt)} chars")

    try:
        dd_result = await llm_judge.judge(
            prompt=prompt,
            schema=DEEP_DIVE_SCHEMA,
            model=model,
        )
    except Exception as e:
        lines.append(f"  ERROR (deep dive): {type(e).__name__}: {e}")
        if verbose:
            print("\n".join(lines))
        return {"task_id": task_id, "bucket": bucket, "error": f"{type(e).__name__}: {e}"}

    dd_output = dd_result.result
    dd_failures = dd_output.get("failures", [])

    # Reviewer pass — matches production: Gemini gets the deep-dive output plus
    # context, reclassifies via `challenges[]`. Final classification overrides
    # the deep-dive one where reviewer disagrees.
    rev_tokens_in = rev_tokens_out = 0
    rev_latency_s = 0.0
    rev_n_reclassifications = 0
    new_final_failures = list(dd_failures)

    if not skip_reviewer:
        reviewer_prompt_text = skeptical_reviewer_prompt(
            instruction_md=context.get("instruction_md", ""),
            reference_test_bodies=context.get("postmerge_test_bodies", []),
            deep_dive_output=dd_output,
            score=opus_score,
            verify_output_tail=context.get("verify_output_tail", ""),
            trajectory_tail="",  # easiness check only — doesn't affect classification
        )
        try:
            rev_result = await llm_judge.judge(
                prompt=reviewer_prompt_text,
                schema=REVIEWER_SCHEMA,
                model=reviewer_model,
            )
            rev_tokens_in = rev_result.usage.get("input_tokens", 0)
            rev_tokens_out = rev_result.usage.get("output_tokens", 0)
            rev_latency_s = round(rev_result.latency_s, 2)
            # Apply challenges[] reclassifications
            reclass_map: dict[str, str] = {}
            for c in rev_result.result.get("challenges", []):
                tname = c.get("test_name", "")
                orig = c.get("original_classification", "")
                revised = c.get("revised_classification", "")
                if tname and revised and revised != orig:
                    reclass_map[tname] = revised
            new_final_failures = apply_reclassifications(dd_failures, reclass_map)
            rev_n_reclassifications = len(reclass_map)
        except Exception as e:
            lines.append(f"  WARNING (reviewer failed, using deep-dive only): {type(e).__name__}: {e}")

    # Historical final: deep-dive classification + review reclassifications.
    historical_md = _read_or_empty(historical_file) if historical_file else ""
    hist_dd_failures = parse_historical_triage(historical_md)
    review_md = _read_or_empty(review_historical_file) if review_historical_file else ""
    hist_reclass_map = parse_historical_review(review_md)
    historical_final = apply_reclassifications(hist_dd_failures, hist_reclass_map)
    hist_n_reclassifications = sum(
        1 for f in hist_dd_failures if hist_reclass_map.get(f["test_name"]) not in (None, f["classification"])
    )

    # Raw deep-dive-only comparison (for backwards-compat reporting)
    cmp_raw = _compare_verdicts(dd_failures, hist_dd_failures)
    # Final classification comparison (the one that actually matters)
    cmp_final = _compare_verdicts(new_final_failures, historical_final)

    lines.append(f"  new DD: {len(dd_failures)} failures — {dd_output.get('overall_assessment', '')[:100]!r}")
    lines.append(
        f"  new reviewer reclassifications: {rev_n_reclassifications} | "
        f"historical reviewer reclassifications: {hist_n_reclassifications}"
    )
    dd_name = historical_file.name if historical_file else "none"
    rv_name = review_historical_file.name if review_historical_file else "none"
    lines.append(f"  historical DD: {len(hist_dd_failures)} failures ({dd_name}); review: {rv_name}")
    lines.append(
        f"  RAW (DD vs DD)    : agree {cmp_raw['same_class']}/{cmp_raw['common_tests']} "
        f"({_pct(cmp_raw['same_class'], cmp_raw['common_tests'])}), drift {cmp_raw['different_class']}"
    )
    lines.append(
        f"  FINAL (post-review): agree {cmp_final['same_class']}/{cmp_final['common_tests']} "
        f"({_pct(cmp_final['same_class'], cmp_final['common_tests'])}), drift {cmp_final['different_class']}"
    )
    if cmp_final["drift"]:
        for row in cmp_final["drift"][:8]:
            lines.append(f"    final drift: {row}")
        if len(cmp_final["drift"]) > 8:
            lines.append(f"    ...{len(cmp_final['drift']) - 8} more")

    if verbose:
        print("\n".join(lines))

    return {
        "task_id": task_id,
        "bucket": bucket,
        "opus_score": opus_score,
        "stage": stage,
        "triage_rounds": triage_rounds,
        "new_dd_n_failures": len(dd_failures),
        "new_reviewer_reclassifications": rev_n_reclassifications,
        "new_assessment": dd_output.get("overall_assessment", "")[:200],
        "historical_dd_file": historical_file.name if historical_file else "",
        "historical_review_file": review_historical_file.name if review_historical_file else "",
        "historical_dd_n_failures": len(hist_dd_failures),
        "historical_reviewer_reclassifications": hist_n_reclassifications,
        # RAW = deep-dive vs deep-dive, no reviewer involvement
        "raw_common_tests": cmp_raw["common_tests"],
        "raw_same_class": cmp_raw["same_class"],
        "raw_agreement_pct": _pct(cmp_raw["same_class"], cmp_raw["common_tests"]),
        # FINAL = post-reviewer classification on both sides (matches production)
        "final_common_tests": cmp_final["common_tests"],
        "final_same_class": cmp_final["same_class"],
        "final_different_class": cmp_final["different_class"],
        "final_only_new_n": len(cmp_final["only_new"]),
        "final_only_hist_n": len(cmp_final["only_hist"]),
        "final_agreement_pct": _pct(cmp_final["same_class"], cmp_final["common_tests"]),
        "final_drift": "; ".join(cmp_final["drift"]),
        "final_only_new": "; ".join(cmp_final["only_new"]),
        "final_only_hist": "; ".join(cmp_final["only_hist"]),
        "pairs": cmp_final["pairs"],
        "dd_tokens_in": dd_result.usage.get("input_tokens", 0),
        "dd_tokens_out": dd_result.usage.get("output_tokens", 0),
        "dd_latency_s": round(dd_result.latency_s, 2),
        "rev_tokens_in": rev_tokens_in,
        "rev_tokens_out": rev_tokens_out,
        "rev_latency_s": rev_latency_s,
        "error": "",
    }


async def judge_sample(
    sample_dir: Path,
    *,
    model: str,
    reviewer_model: str,
    skip_reviewer: bool,
    verbose: bool,
    secondary_model: str = "",
) -> dict:
    """Fixture-mode entry: load pre-captured context + historical triage from the bundle."""
    info = json.loads(_read_or_empty(sample_dir / "fixture_info.json") or "{}")
    if secondary_model:
        return await _judge_dual_dd(
            task_id=info.get("task_id", sample_dir.name),
            bucket=info.get("bucket", "?"),
            opus_score=info.get("opus_score", ""),
            stage=info.get("stage", ""),
            triage_rounds=info.get("triage_rounds", ""),
            context=load_sample_context(sample_dir),
            model_primary=model,
            model_secondary=secondary_model,
            verbose=verbose,
        )
    return await _judge_and_compare(
        task_id=info.get("task_id", sample_dir.name),
        bucket=info.get("bucket", "?"),
        opus_score=info.get("opus_score", ""),
        stage=info.get("stage", ""),
        triage_rounds=info.get("triage_rounds", ""),
        context=load_sample_context(sample_dir),
        historical_file=latest_triage_file(sample_dir),
        review_historical_file=latest_review_file(sample_dir),
        model=model,
        reviewer_model=reviewer_model,
        skip_reviewer=skip_reviewer,
        verbose=verbose,
    )


async def judge_task_from_state(
    task_id: str,
    task_state: dict,
    *,
    model: str,
    reviewer_model: str,
    skip_reviewer: bool,
    verbose: bool,
    secondary_model: str = "",
) -> dict | None:
    """Run-dir mode entry: read state.json task + live harbor-lab + diagnostics/*.md
    from the task_dir. Returns None if the task doesn't have enough evidence to
    calibrate (no opus_trial_dir, or no historical triage_opus.md).
    """
    task_dir = task_state.get("task_dir", "")
    opus_trial_dir = task_state.get("opus_trial_dir", "")
    if not task_dir or not opus_trial_dir:
        return None
    if not Path(task_dir).is_dir() or not Path(opus_trial_dir).is_dir():
        return None

    historical_file = _latest_triage_in_dir(Path(task_dir) / "diagnostics")
    if historical_file is None:
        return None  # nothing historical to compare against

    diag_dir = Path(task_dir) / "diagnostics"
    review_candidates = sorted(diag_dir.glob("*_review_opus.md"))
    review_historical_file = review_candidates[-1] if review_candidates else None

    # Shell out to live harbor-lab via the pipeline's own context-fetch helper.
    context = await _fetch_deep_dive_context(task_dir, opus_trial_dir)

    if secondary_model:
        return await _judge_dual_dd(
            task_id=task_id,
            bucket=task_state.get("stage", "?"),
            opus_score=task_state.get("opus_score", ""),
            stage=task_state.get("stage", ""),
            triage_rounds="",
            context=context,
            model_primary=model,
            model_secondary=secondary_model,
            verbose=verbose,
            header_note=f"hist={historical_file.name}",
        )
    return await _judge_and_compare(
        task_id=task_id,
        bucket=task_state.get("stage", "?"),
        opus_score=task_state.get("opus_score", ""),
        stage=task_state.get("stage", ""),
        triage_rounds="",
        context=context,
        historical_file=historical_file,
        review_historical_file=review_historical_file,
        model=model,
        reviewer_model=reviewer_model,
        skip_reviewer=skip_reviewer,
        verbose=verbose,
        header_note=f"hist={historical_file.name}",
    )


async def _run_fixture_mode(args: argparse.Namespace, model: str) -> list[dict]:
    fixtures_root = Path(args.fixtures).resolve()
    samples_dir = fixtures_root / "samples"
    if not samples_dir.is_dir():
        print(f"ERROR: {samples_dir} not found", file=sys.stderr)
        sys.exit(1)

    sample_dirs = sorted(d for d in samples_dir.iterdir() if d.is_dir())
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        sample_dirs = [d for d in sample_dirs if d.name in wanted]
        if not sample_dirs:
            print(f"ERROR: none of --only names matched: {wanted}", file=sys.stderr)
            sys.exit(1)
    if args.limit:
        sample_dirs = sample_dirs[: args.limit]

    reviewer_model = args.reviewer_model or "openai/gcp/google/gemini-3.1-pro-preview"
    secondary_model = args.secondary_model or ""
    print(f"Fixture mode: {len(sample_dirs)} samples")
    print(f"  deep-dive primary: {model}")
    if secondary_model:
        print(f"  deep-dive secondary: {secondary_model}  (dual-DD mode, reviewer SKIPPED)")
    else:
        print(f"  reviewer model: {'SKIPPED' if args.skip_reviewer else reviewer_model}")

    sem = asyncio.Semaphore(args.concurrency)

    async def _run(d: Path) -> dict:
        async with sem:
            return await judge_sample(
                d,
                model=model,
                reviewer_model=reviewer_model,
                skip_reviewer=args.skip_reviewer,
                verbose=args.verbose,
                secondary_model=secondary_model,
            )

    return list(await asyncio.gather(*[_run(d) for d in sample_dirs]))


async def _run_rundir_mode(args: argparse.Namespace, model: str) -> list[dict]:
    if not args.run_dir:
        print("ERROR: --mode run_dir requires --run-dir <path>", file=sys.stderr)
        sys.exit(1)
    state_path = Path(args.run_dir) / "state.json"
    if not state_path.is_file():
        print(f"ERROR: {state_path} not found", file=sys.stderr)
        sys.exit(1)

    state = json.loads(state_path.read_text())
    tasks = state.get("tasks", {})

    # Eligible: task reached Opus triage (has opus_trial_dir + a historical
    # triage_opus.md somewhere in diagnostics/). Filter in judge_task_from_state
    # returns None for ineligible.
    eligible_stages = {"opus_triaged", "accepted", "needs_fix"}
    eligible_items = [(tid, t) for tid, t in tasks.items() if t.get("stage") in eligible_stages]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        eligible_items = [(tid, t) for tid, t in eligible_items if tid in wanted]
    if args.limit:
        eligible_items = eligible_items[: args.limit]

    reviewer_model = args.reviewer_model or "openai/gcp/google/gemini-3.1-pro-preview"
    secondary_model = args.secondary_model or ""
    print(f"Run-dir mode: {state_path}")
    print(f"  candidates: {len(eligible_items)} tasks (stages: {sorted(eligible_stages)})")
    print(f"  deep-dive primary: {model}")
    if secondary_model:
        print(f"  deep-dive secondary: {secondary_model}  (dual-DD mode, reviewer SKIPPED)")
    else:
        print(f"  reviewer model: {'SKIPPED' if args.skip_reviewer else reviewer_model}")

    sem = asyncio.Semaphore(args.concurrency)

    async def _run(task_id: str, task_state: dict) -> dict | None:
        async with sem:
            return await judge_task_from_state(
                task_id,
                task_state,
                model=model,
                reviewer_model=reviewer_model,
                skip_reviewer=args.skip_reviewer,
                verbose=args.verbose,
                secondary_model=secondary_model,
            )

    raw = await asyncio.gather(*[_run(tid, t) for tid, t in eligible_items])
    results: list[dict] = [r for r in raw if r is not None]
    skipped = len(raw) - len(results)
    if skipped:
        print(f"  Skipped {skipped} tasks (missing trial_dir or historical triage_opus.md)")
    return results


async def _judge_fairness_reviewer_sample(sample_dir: Path, bucket: str, task_id: str, model: str) -> dict:
    """Run the new fairness reviewer on a single fixture or task dir.

    Input layout is the same as the deep-dive fixture path: any
    directory containing `instruction.md`, `tests/`, `trial/verifier/`,
    `harbor_lab/` (or whatever load_sample_context expects).
    """
    try:
        context = load_sample_context(sample_dir)
    except Exception as e:
        return {
            "task_id": task_id,
            "bucket": bucket,
            "error": f"context load failed: {type(e).__name__}: {e}",
        }

    prompt = fairness_review_prompt(
        instruction_md=context["instruction_md"],
        reward_json=context["reward_json"],
        verify_output_tail=context["verify_output_tail"],
        postmerge_test_bodies=context["postmerge_test_bodies"],
        harbor_lab_errors=context["harbor_lab_errors"],
        harbor_lab_edits=context["harbor_lab_edits"],
        harbor_lab_tool_sequence=context["harbor_lab_tool_sequence"],
        f2p_tests=context["f2p_tests"],
        p2p_tests=context["p2p_tests"],
    )
    try:
        res = await llm_judge.judge(prompt=prompt, schema=FAIRNESS_REVIEW_SCHEMA, model=model)
    except Exception as e:
        return {
            "task_id": task_id,
            "bucket": bucket,
            "error": f"judge call failed: {type(e).__name__}: {e}",
        }

    severity = res.result.get("severity", "") or ""
    quote = (res.result.get("evidence_quote", "") or "").strip()
    test = (res.result.get("evidence_test", "") or "").strip()
    return {
        "task_id": task_id,
        "bucket": bucket,
        "severity": severity,
        "reason": (res.result.get("reason", "") or "")[:300],
        "evidence_quote": quote[:200],
        "evidence_test": test,
        "has_both_evidence": bool(quote) and bool(test),
        "would_trigger_regen": severity == "major" and bool(quote) and bool(test),
        "model": res.model,
        "latency_s": round(res.latency_s, 3),
        "tokens_in": res.usage.get("input_tokens", 0),
        "tokens_out": res.usage.get("output_tokens", 0),
        "error": "",
    }


async def _judge_fairness_reviewer_live(
    task_id: str,
    task_state: dict,
    model: str,
    verbose: bool,
) -> dict | None:
    """Live run_dir path: read state.json task entry, shell out to harbor-lab
    for full context, run fairness reviewer.

    Returns None for tasks without a completed Opus trial. Mirrors
    judge_task_from_state's path selection so the eligibility filter
    matches the existing DD calibration behavior.
    """
    task_dir = task_state.get("task_dir", "")
    opus_trial_dir = task_state.get("opus_trial_dir", "")
    stage = task_state.get("stage", "?")
    if not task_dir or not opus_trial_dir:
        return None
    if not Path(task_dir).is_dir() or not Path(opus_trial_dir).is_dir():
        return None

    context = await _fetch_deep_dive_context(task_dir, opus_trial_dir)
    if verbose:
        print(f"  ... {stage}/{task_id}")

    prompt = fairness_review_prompt(
        instruction_md=context["instruction_md"],
        reward_json=context["reward_json"],
        verify_output_tail=context["verify_output_tail"],
        postmerge_test_bodies=context["postmerge_test_bodies"],
        harbor_lab_errors=context["harbor_lab_errors"],
        harbor_lab_edits=context["harbor_lab_edits"],
        harbor_lab_tool_sequence=context["harbor_lab_tool_sequence"],
        f2p_tests=context["f2p_tests"],
        p2p_tests=context["p2p_tests"],
    )
    try:
        res = await llm_judge.judge(prompt=prompt, schema=FAIRNESS_REVIEW_SCHEMA, model=model)
    except Exception as e:
        return {
            "task_id": task_id,
            "bucket": stage,
            "error": f"judge call failed: {type(e).__name__}: {e}",
        }

    severity = res.result.get("severity", "") or ""
    quote = (res.result.get("evidence_quote", "") or "").strip()
    test = (res.result.get("evidence_test", "") or "").strip()
    return {
        "task_id": task_id,
        "bucket": stage,
        "severity": severity,
        "reason": (res.result.get("reason", "") or "")[:300],
        "evidence_quote": quote[:200],
        "evidence_test": test,
        "has_both_evidence": bool(quote) and bool(test),
        "would_trigger_regen": severity == "major" and bool(quote) and bool(test),
        "model": res.model,
        "latency_s": round(res.latency_s, 3),
        "tokens_in": res.usage.get("input_tokens", 0),
        "tokens_out": res.usage.get("output_tokens", 0),
        "error": "",
    }


async def _run_fairness_reviewer_calibration(args: argparse.Namespace, model: str) -> list[dict]:
    """Drive the fairness-reviewer calibration in either fixtures or run_dir mode.

    For fixtures: iterate `samples/<bucket>__<task>/` dirs directly.
    For run_dir: read `state.json` and iterate tasks whose stage shows
    they reached Opus triage (has opus_trial_dir). Live harbor-lab is
    shelled out via `_fetch_deep_dive_context` for full context.
    """
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def _fixture_one(sample_dir: Path, bucket: str, task_id: str) -> dict:
        async with sem:
            if args.verbose:
                print(f"  ... {bucket}/{task_id}")
            return await _judge_fairness_reviewer_sample(sample_dir, bucket, task_id, model)

    async def _live_one(task_id: str, task_state: dict) -> dict | None:
        async with sem:
            return await _judge_fairness_reviewer_live(task_id, task_state, model, args.verbose)

    tasks = []
    if args.mode == "fixtures":
        samples_root = Path(args.fixtures) / "samples"
        if not samples_root.is_dir():
            print(f"ERROR: no fixtures dir at {samples_root}")
            return []
        items = sorted(samples_root.iterdir())
        for p in items:
            if not p.is_dir():
                continue
            name = p.name
            bucket, _, task_id = name.partition("__")
            task_id = task_id or name
            if args.only and name not in args.only.split(","):
                continue
            tasks.append(_fixture_one(p, bucket, task_id))
        if args.limit:
            tasks = tasks[: args.limit]
        print(f"Running fairness-reviewer on {len(tasks)} sample(s), model={model}")
        out = await asyncio.gather(*tasks)
        return list(out)

    # run_dir mode — live state.json + harbor-lab
    state_path = Path(args.run_dir) / "state.json"
    if not state_path.is_file():
        print(f"ERROR: {state_path} not found")
        return []
    state = json.loads(state_path.read_text())
    eligible_stages = {"opus_triaged", "accepted", "needs_fix", "rejected"}
    items = [
        (tid, t)
        for tid, t in state.get("tasks", {}).items()
        if t.get("stage") in eligible_stages and t.get("opus_trial_dir")
    ]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        items = [(tid, t) for tid, t in items if tid in wanted]
    if args.limit:
        items = items[: args.limit]
    print(f"Running fairness-reviewer on {len(items)} task(s) from {state_path}, model={model}")
    raw = await asyncio.gather(*[_live_one(tid, t) for tid, t in items])
    results = [r for r in raw if r is not None]
    skipped = len(raw) - len(results)
    if skipped:
        print(f"  Skipped {skipped} tasks (missing task_dir or trial_dir)")
    return results


async def main_async(args: argparse.Namespace) -> int:
    model = args.model or _cfg.LLM_STEP_MODEL
    dual_dd = bool(args.secondary_model)

    if args.judge == "fairness-reviewer":
        reviewer_model = args.model or _cfg.LLM_ALIGNMENT_MODEL
        results = await _run_fairness_reviewer_calibration(args, reviewer_model)
        if not results:
            return 1
        # Severity distribution overall + per bucket
        from collections import Counter

        total = len(results)
        succeeded = [r for r in results if not r.get("error")]
        sev_counts = Counter(r.get("severity", "") or "(missing)" for r in succeeded)
        triggered = sum(1 for r in succeeded if r.get("would_trigger_regen"))
        print(f"\n{len(succeeded)}/{total} samples judged (fairness-reviewer mode).")
        print("  Severity distribution:")
        for sev, n in sev_counts.most_common():
            print(f"    {sev}: {n}")
        print(f"  Would auto-regen (major + both evidence fields): {triggered}/{len(succeeded)}")
        by_bucket: dict[str, Counter] = {}
        for r in succeeded:
            by_bucket.setdefault(r["bucket"], Counter())[r.get("severity", "") or "(missing)"] += 1
        print("\n  Per-bucket severity distribution:")
        for bucket in sorted(by_bucket):
            sev = by_bucket[bucket]
            line = ", ".join(f"{k}={v}" for k, v in sev.most_common())
            print(f"    {bucket}: {line}")
        for r in results:
            if r.get("error"):
                print(f"  ERROR {r['bucket']}/{r['task_id']}: {r['error']}")
        if args.output:
            fieldnames = [
                "task_id",
                "bucket",
                "severity",
                "would_trigger_regen",
                "has_both_evidence",
                "evidence_quote",
                "evidence_test",
                "reason",
                "model",
                "latency_s",
                "tokens_in",
                "tokens_out",
                "error",
            ]
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            print(f"\nWrote per-sample CSV: {args.output}")
        return 0 if all(not r.get("error") for r in results) else 1

    if args.mode == "fixtures":
        results = await _run_fixture_mode(args, model)
    else:
        results = await _run_rundir_mode(args, model)

    total = len(results)
    succeeded = [r for r in results if not r.get("error")]
    if succeeded and dual_dd:
        common = sum(r["common_tests"] for r in succeeded)
        agree_exact = sum(r["agree_exact"] for r in succeeded)
        agree_skip = sum(r["agree_skip"] for r in succeeded)
        agree_nonskip = sum(r["agree_nonskip"] for r in succeeded)
        disagree = sum(r["disagree_across"] for r in succeeded)
        only_p = sum(r["only_primary"] for r in succeeded)
        only_s = sum(r["only_secondary"] for r in succeeded)
        action_agree = agree_exact + agree_skip + agree_nonskip
        print(
            f"\n{len(succeeded)}/{total} samples judged (dual-DD mode)."
            f"\n  ACTION-AGREE: {action_agree}/{common} ({_pct(action_agree, common)}) "
            f"[exact {agree_exact}, skip {agree_skip}, nonskip {agree_nonskip}]"
            f"\n  DISAGREE_ACROSS: {disagree}/{common} ({_pct(disagree, common)})"
            f"\n  only_primary: {only_p}, only_secondary: {only_s}"
        )
        for r in sorted(succeeded, key=lambda x: x["bucket"]):
            print(
                f"  {r['bucket']:<22} {r['task_id']:<36} "
                f"common={r['common_tests']}, exact={r['agree_exact']}, "
                f"skip-agree={r['agree_skip']}, nonskip-agree={r['agree_nonskip']}, "
                f"ACROSS={r['disagree_across']} "
                f"(only_p={r['only_primary']}, only_s={r['only_secondary']})"
            )
    elif succeeded:
        raw_same = sum(r["raw_same_class"] for r in succeeded)
        raw_common = sum(r["raw_common_tests"] for r in succeeded)
        final_same = sum(r["final_same_class"] for r in succeeded)
        final_common = sum(r["final_common_tests"] for r in succeeded)
        final_only_new = sum(r["final_only_new_n"] for r in succeeded)
        final_only_hist = sum(r["final_only_hist_n"] for r in succeeded)
        print(
            f"\n{len(succeeded)}/{total} samples judged."
            f"\n  RAW (DD-only)    : agree {raw_same}/{raw_common} ({_pct(raw_same, raw_common)})"
            f"\n  FINAL (post-review): agree {final_same}/{final_common} ({_pct(final_same, final_common)}), "
            f"only-new {final_only_new}, only-hist {final_only_hist}"
        )
        for r in sorted(succeeded, key=lambda x: x["bucket"]):
            print(
                f"  {r['bucket']:<22} {r['task_id']:<36} "
                f"RAW {r['raw_same_class']}/{r['raw_common_tests']} ({r['raw_agreement_pct']}), "
                f"FINAL {r['final_same_class']}/{r['final_common_tests']} ({r['final_agreement_pct']}), "
                f"rev_reclass: new={r['new_reviewer_reclassifications']}/"
                f"hist={r['historical_reviewer_reclassifications']}"
            )
    for r in results:
        if r.get("error"):
            print(f"  ERROR {r['bucket']:<22} {r['task_id']}: {r['error']}")

    if args.output:
        # The per-sample summary CSV excludes the `pairs` list (goes to
        # --dump-pairs) — it doesn't round-trip cleanly through csv.
        excluded = {"pairs"}
        fieldnames = [k for k in results[0].keys() if k not in excluded] if results else []
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: v for k, v in r.items() if k not in excluded})
        print(f"\nWrote per-sample CSV: {args.output}")

    if args.dump_pairs:
        if dual_dd:
            pair_fieldnames = [
                "task_id",
                "bucket",
                "opus_score",
                "test_basename",
                "status",
                "primary_classification",
                "secondary_classification",
                "primary_evidence",
                "secondary_evidence",
                "primary_test_name",
                "secondary_test_name",
            ]
        else:
            pair_fieldnames = [
                "task_id",
                "bucket",
                "opus_score",
                "test_basename",
                "status",
                "historical_classification",
                "new_classification",
                "historical_evidence",
                "new_evidence",
                "historical_test_name",
                "new_test_name",
            ]
        with open(args.dump_pairs, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pair_fieldnames)
            writer.writeheader()
            for r in results:
                if r.get("error"):
                    continue
                for pair in r.get("pairs", []):
                    writer.writerow(
                        {
                            "task_id": r["task_id"],
                            "bucket": r["bucket"],
                            "opus_score": r.get("opus_score", ""),
                            **pair,
                        }
                    )
        n_pairs = sum(len(r.get("pairs", [])) for r in results if not r.get("error"))
        print(f"Wrote per-pair CSV: {args.dump_pairs} ({n_pairs} rows)")

    return 0 if all(not r["error"] for r in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["fixtures", "run_dir"],
        default="fixtures",
        help="fixtures: pre-captured bundle. run_dir: shell out to harbor-lab on a live run.",
    )
    parser.add_argument(
        "--judge",
        choices=["deep-dive", "fairness-reviewer"],
        default="deep-dive",
        help=(
            "deep-dive (default): legacy per-test DD calibration vs historical "
            "triage/review output. fairness-reviewer: invoke the new triage-time "
            "fairness reviewer and emit per-sample severity distribution."
        ),
    )
    parser.add_argument(
        "--fixtures",
        default="tmp2/dd-fixtures-20260422",
        help="(mode=fixtures) Path to the fixture bundle (contains samples/ and README.md)",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="(mode=run_dir) Path to a pipeline run dir, e.g. harbor-tasks/craft-tools-v4/runs/<ts>/",
    )
    parser.add_argument(
        "--output",
        default="deep_dive_calibration.csv",
        help="CSV path for per-sample results (empty to skip)",
    )
    parser.add_argument(
        "--dump-pairs",
        default="",
        help=(
            "Optional CSV path. Writes one row per test seen in either verdict — "
            "{task_id, bucket, test_basename, status (agree/drift/only_new/only_hist), "
            "historical_classification, new_classification, historical_evidence, "
            "new_evidence}. Feeds the PR C 50-task human audit."
        ),
    )
    parser.add_argument("--only", help="Comma-separated sample names to restrict the run to")
    parser.add_argument("--limit", type=int, help="Process only the first N samples")
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel judge calls (default: 3)")
    parser.add_argument(
        "--model",
        help="Override _cfg.LLM_STEP_MODEL for deep-dive (e.g. 'azure/anthropic/claude-opus-4-6')",
    )
    parser.add_argument(
        "--reviewer-model",
        help="Legacy skeptical-reviewer model (replay-only; production no "
        "longer runs a reviewer). Defaults to Gemini-3.1-Pro to match the "
        "Apr 16-19 2026 historical cohort.",
    )
    parser.add_argument(
        "--skip-reviewer",
        action="store_true",
        help="Deep-dive only — skip the reviewer pass. Useful for iterating quickly on "
        "the deep-dive prompt, but the FINAL classification number is meaningless.",
    )
    parser.add_argument(
        "--secondary-model",
        default="",
        help="Dual-DD mode: run deep dive twice in parallel with --model (primary) and "
        "--secondary-model (secondary), skip reviewer entirely. Per-test inter-model "
        "agreement statuses: agree_exact / agree_skip / agree_nonskip / disagree_across. "
        "Example: --secondary-model openai/us/azure/openai/gpt-5.4",
    )
    parser.add_argument("--verbose", action="store_true", help="Per-sample progress + disagreement detail")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
