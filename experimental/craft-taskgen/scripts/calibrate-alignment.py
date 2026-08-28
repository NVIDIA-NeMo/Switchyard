# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-off calibration: run the alignment judge on a historical cohort.

Reads an input CSV (default: bigtest.csv at repo root) with columns:

    task_id, repo, commit_sha, subject, pr_url, task_dir, run_dir,
    instruction_md, instruction_words, hardness_verdict, hardness_band,
    final_stage, haiku_score, opus_score, easiness_concern

For each row where instruction_md is non-empty AND the repo is locally cloned
under `repos/` AND the commit sha is reachable, pre-assembles the alignment
judge's context (reference test bodies at the commit sha + unified PR diff
via merge_base..sha), calls the alignment judge via `llm_judge.judge()`, and
writes an output CSV with the original columns plus:

    alignment_verdict, alignment_reason, alignment_leakage_evidence,
    alignment_v4_fixtures, alignment_v4_helpers, alignment_v4_assertions,
    alignment_attempts, alignment_latency_s, alignment_tokens_in,
    alignment_tokens_out, alignment_skip_reason

Purpose: see how the new alignment judge classifies a
big heterogeneous cohort of historical instructions. Compare against the
old hardness verdict as a *relationship check*, NOT ground truth — the
old hardness skill was known to be noisy / confounded.

Usage:
    uv run python scripts/calibrate-alignment.py \\
        --input bigtest.csv \\
        --output bigtest_with_alignment.csv \\
        --sample 100 --concurrency 5

N-parallel build+alignment calibration (--n>1 with --mode=full):

    uv run python scripts/calibrate-alignment.py \\
        --input bigtest.csv --output bigtest_n2.csv \\
        --mode full --n 2 --sample 20 --concurrency 5

When ``--n > 1`` (capped at 4) and ``--mode=full``, runs N independent
build+alignment candidate loops per row concurrently — mirroring the
production orchestrator. Picks a winner uniformly at random among
passers (matching ``_select_winner``). Adds aggregate columns:
``n_candidates, candidate_verdicts, candidate_regen_counts, n_passed,
winner_cand_id, mean_jaccard, min_jaccard, max_jaccard``.

Jaccard distribution columns are diagnostic only and **not sufficient**
for spotting leakage on their own — instruction leakage is a structural
property (presence of internal symbols) that token-set intersection
doesn't capture cleanly. Calibration deliverable includes manual
spot-checks of representative high- and low-jaccard pairs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import subprocess
import sys
import time
from typing import Any

from dotenv import load_dotenv

import craft_taskgen.config as _cfg
from craft_taskgen import llm_judge
from craft_taskgen.config import PipelineContext
from craft_taskgen.prompts import (
    ALIGNMENT_SCHEMA,
    BUILD_SCHEMA,
    EVALUATE_SCHEMA,
    alignment_judge_prompt,
    build_task_prompt,
    evaluate_candidate_prompt,
)
from craft_taskgen.steps import (
    _BUILD_DIFF_BYTE_CAP,
    _build_alignment_feedback,
    _fetch_build_context,
    _fetch_evaluate_context,
    _list_commit_test_files_sync,
    _validate_build_output,
)

load_dotenv()


def fetch_reference_tests_and_diff(
    repo: str, commit_sha: str, merge_base_sha: str
) -> tuple[list[tuple[str, str]], str, str | None]:
    """Pre-assemble reference test bodies + PR diff, matching _fetch_alignment_context.

    Returns (tests, diff, skip_reason). If skip_reason is non-None, caller
    should skip this row.
    """
    repo_path = os.path.join("repos", repo)
    if not os.path.isdir(repo_path):
        return [], "", f"repo_not_cloned:{repo}"

    # Verify commit reachable
    try:
        subprocess.check_output(
            ["git", "-C", repo_path, "cat-file", "-t", commit_sha],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return [], "", f"sha_not_reachable:{commit_sha[:10]}"

    # If merge_base_sha wasn't in the CSV, derive it as the first parent.
    if not merge_base_sha:
        try:
            merge_base_sha = subprocess.check_output(
                ["git", "-C", repo_path, "rev-parse", f"{commit_sha}^"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except subprocess.CalledProcessError:
            return [], "", "no_parent_sha"

    test_paths = _list_commit_test_files_sync(repo_path, merge_base_sha, commit_sha)
    reference_test_bodies: list[tuple[str, str]] = []
    for rel_path in test_paths:
        try:
            body = subprocess.check_output(
                ["git", "-C", repo_path, "show", f"{commit_sha}:{rel_path}"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            continue
        reference_test_bodies.append((rel_path, body))

    try:
        diff = subprocess.check_output(
            ["git", "-C", repo_path, "diff", merge_base_sha, commit_sha],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return [], "", "diff_failed"

    if len(diff) > _BUILD_DIFF_BYTE_CAP:
        half = _BUILD_DIFF_BYTE_CAP // 2
        omitted_lines = diff.count("\n") - diff[:half].count("\n") - diff[-half:].count("\n")
        omitted_bytes = len(diff) - _BUILD_DIFF_BYTE_CAP
        marker = f"\n\n[...truncated {omitted_lines} lines ({omitted_bytes:,} bytes) omitted...]\n\n"
        diff = diff[:half] + marker + diff[-half:]

    return reference_test_bodies, diff, None


STEP_MODEL = os.environ.get("LLM_STEP_MODEL", "aws/anthropic/bedrock-claude-opus-4-6")
ALIGNMENT_MODEL = os.environ.get("LLM_ALIGNMENT_MODEL", "openai/us/azure/openai/gpt-5.4")


async def _resolve_merge_base(repo: str, commit_sha: str) -> str | None:
    """Return the first-parent SHA of the commit, or None if unreachable."""
    repo_path = os.path.join("repos", repo)
    if not os.path.isdir(repo_path):
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", repo_path, "rev-parse", f"{commit_sha}^"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError:
        return None


async def _run_alignment_passes(
    instruction_md: str,
    reference_test_bodies: list[tuple[str, str]],
    diff: str,
    sem: asyncio.Semaphore,
    max_retries: int = _cfg.ALIGNMENT_MAX_RETRIES,
) -> tuple[list[dict], dict | None]:
    """Run alignment judge up to ``max_retries`` times, stopping on first `ok`.

    Returns (attempts, accept_result). accept_result is the `ok`-result dict
    if any attempt said ok, else None.
    """
    attempts: list[dict] = []
    accept_result = None
    prompt = alignment_judge_prompt(
        instruction_md=instruction_md,
        reference_test_bodies=reference_test_bodies,
        diff=diff,
    )
    for i in range(max_retries):
        async with sem:
            try:
                res = await llm_judge.judge(prompt=prompt, schema=ALIGNMENT_SCHEMA, model=ALIGNMENT_MODEL)
            except Exception as err:
                attempts.append({"attempt": i + 1, "error": f"{type(err).__name__}: {err}"})
                continue
        verdict = res.result.get("verdict", "")
        attempts.append(
            {
                "attempt": i + 1,
                "verdict": verdict,
                "reason": res.result.get("reason", ""),
                "v4_audit": res.result.get("v4_audit", {}) or {},
                "leakage_evidence": res.result.get("leakage_evidence", []) or [],
                "tokens_in": res.usage.get("input_tokens", 0),
                "tokens_out": res.usage.get("output_tokens", 0),
                "latency_s": res.latency_s,
            }
        )
        if verdict == "ok":
            accept_result = res.result
            break
    return attempts, accept_result


async def run_full_pipeline_row(
    row: dict[str, str],
    sem: asyncio.Semaphore,
    progress: dict[str, int],
    ctx: PipelineContext,
    verbose_fh=None,
    enable_feedback: bool = True,
    skip_eval: bool = False,
    max_rebuilds: int = _cfg.MAX_BUILD_REGENS_PER_CANDIDATE,
    max_alignment_retries: int = _cfg.ALIGNMENT_MAX_RETRIES,
) -> dict[str, Any]:
    """End-to-end: evaluate → (if accept) build → alignment (+ 1 regen).

    Produces all the alignment-only columns PLUS new_eval_*, new_build_*,
    new_regen_*, new_instruction_md columns capturing our refactored
    pipeline's output for this historical candidate.
    """
    out = dict(row)
    out.update(
        {
            "new_eval_verdict": "",
            "new_eval_reason": "",
            "new_eval_reject_pattern": "",
            "new_instruction_sketch": "",
            "new_build_words": "",
            "new_instruction_md": "",
            "new_regen_count": "0",
            "alignment_verdict": "",
            "alignment_reason": "",
            "alignment_leakage_evidence": "",
            "alignment_v4_fixtures": "",
            "alignment_v4_helpers": "",
            "alignment_v4_assertions": "",
            "alignment_attempts": "",
            "alignment_latency_s": "",
            "alignment_tokens_in": "",
            "alignment_tokens_out": "",
            "alignment_skip_reason": "",
        }
    )

    repo = row["repo"]
    sha = row["commit_sha"]
    merge_base = await _resolve_merge_base(repo, sha)
    if merge_base is None:
        out["alignment_skip_reason"] = f"sha_not_reachable:{sha[:10]}"
        progress["done"] += 1
        return out

    # ---------------- EVALUATE ----------------
    if skip_eval:
        # Bypass eval entirely — feed build+align with empty eval_reason and
        # an instruction_sketch derived from the row's subject. Used when
        # we want to measure build+align variance in isolation, without
        # eval drift / per-call non-determinism polluting the signal.
        out["new_eval_verdict"] = "accept"
        out["new_eval_reason"] = "(eval skipped — pre-screened cohort)"
        out["new_eval_reject_pattern"] = ""
        out["new_instruction_sketch"] = row.get("subject", "")
    else:
        try:
            diff_stat, diff, readme = await asyncio.to_thread(_fetch_evaluate_context, repo, sha, merge_base)
        except Exception as err:
            out["alignment_skip_reason"] = f"eval_ctx_fail:{err!s:.80}"
            progress["done"] += 1
            return out

        eval_prompt = evaluate_candidate_prompt(
            repo=repo,
            sha=sha,
            subject=row["subject"],
            diff_stat=diff_stat,
            diff=diff,
            readme_excerpt=readme,
        )
        async with sem:
            try:
                eval_res = await llm_judge.judge(prompt=eval_prompt, schema=EVALUATE_SCHEMA, model=STEP_MODEL)
            except Exception as err:
                out["alignment_skip_reason"] = f"eval_judge_err:{type(err).__name__}:{err!s:.80}"
                progress["done"] += 1
                return out

        out["new_eval_verdict"] = eval_res.result.get("verdict", "")
        out["new_eval_reason"] = eval_res.result.get("reason", "")
        out["new_eval_reject_pattern"] = eval_res.result.get("reject_pattern", "")
        out["new_instruction_sketch"] = eval_res.result.get("instruction_sketch", "")

        if out["new_eval_verdict"] != "accept":
            progress["done"] += 1
            print(
                f"[{progress['done']}/{progress['total']}] {row['task_id']}: "
                f"eval={out['new_eval_verdict']} (stop)",
                file=sys.stderr,
            )
            return out

    # ---------------- BUILD + possibly one ALIGNMENT-feedback regen ----------------
    current_instruction: str = ""
    regen_count = 0
    feedback = ""
    attempts: list[dict] = []
    accept_result = None

    while True:
        try:
            bctx = await asyncio.to_thread(_fetch_build_context, repo, sha, merge_base, ctx)
        except Exception as err:
            out["alignment_skip_reason"] = f"build_ctx_fail:{err!s:.80}"
            progress["done"] += 1
            return out

        template_lines = bctx["instruction_template"].splitlines()
        template_first_line = template_lines[0] if template_lines else ""

        # Inner loop mirrors the live pipeline: 1 build regen on validation fail.
        build_output: dict | None = None
        build_validation_error: str | None = None
        for build_attempt in range(2):
            build_prompt = build_task_prompt(
                repo=repo,
                sha=sha,
                merge_base_sha=merge_base,
                subject=row["subject"],
                eval_reason=out["new_eval_reason"],
                instruction_sketch=out["new_instruction_sketch"],
                repo_map=bctx["repo_map"],
                diff=bctx["diff"],
                reference_test_bodies=bctx["reference_test_bodies"],
                instruction_template=bctx["instruction_template"],
                instruction_example=bctx["instruction_example"],
                alignment_feedback=feedback,
            )
            sys_prompt = None
            if build_attempt == 1 and build_validation_error is not None:
                sys_prompt = (
                    f"Previous response failed validation: {build_validation_error}. "
                    "Regenerate to fix that specific problem. Return only the JSON "
                    "object per the schema."
                )
            async with sem:
                try:
                    build_res = await llm_judge.judge(
                        prompt=build_prompt,
                        schema=BUILD_SCHEMA,
                        model=STEP_MODEL,
                        system_prompt=sys_prompt,
                    )
                except Exception as err:
                    out["alignment_skip_reason"] = f"build_judge_err:{type(err).__name__}:{err!s:.80}"
                    progress["done"] += 1
                    return out

            build_output = build_res.result
            build_validation_error = _validate_build_output(
                build_output,
                task_dir="",
                instruction_template_first_line=template_first_line,
            )
            if build_validation_error is None:
                break  # validated

        if build_validation_error is not None:
            out["alignment_skip_reason"] = f"build_validation:{build_validation_error[:80]}"
            progress["done"] += 1
            return out

        assert build_output is not None
        current_instruction = build_output["instruction_md"]

        # Run alignment on the fresh instruction
        attempts, accept_result = await _run_alignment_passes(
            instruction_md=current_instruction,
            reference_test_bodies=bctx["reference_test_bodies"],
            diff=bctx["diff"],
            sem=sem,
            max_retries=max_alignment_retries,
        )

        if accept_result is not None:
            break  # alignment ok, done

        # No accept. Decide whether to regen.
        if not enable_feedback or regen_count >= max_rebuilds:
            break
        actionable = any(a.get("verdict") in ("leaked", "narrow_tests") for a in attempts)
        if not actionable:
            break

        # Regenerate with feedback
        feedback = _build_alignment_feedback(attempts)
        regen_count += 1

    # ---------------- Record results ----------------
    out["new_build_words"] = str(len(current_instruction.split()))
    out["new_instruction_md"] = current_instruction
    out["new_regen_count"] = str(regen_count)
    out["alignment_attempts"] = str(len(attempts))

    if accept_result is not None:
        out["alignment_verdict"] = "ok"
        out["alignment_reason"] = accept_result.get("reason", "")
        v4 = accept_result.get("v4_audit", {}) or {}
        out["alignment_v4_fixtures"] = str(v4.get("fixtures_encode_design_choices", ""))
        out["alignment_v4_helpers"] = str(v4.get("helpers_access_private_api", ""))
        out["alignment_v4_assertions"] = str(v4.get("assertions_format_only", ""))
    elif attempts:
        last = attempts[-1]
        out["alignment_verdict"] = last.get("verdict", "error")
        out["alignment_reason"] = last.get("reason", last.get("error", ""))
        out["alignment_leakage_evidence"] = json.dumps(last.get("leakage_evidence", []), ensure_ascii=False)
        v4 = last.get("v4_audit", {}) or {}
        out["alignment_v4_fixtures"] = str(v4.get("fixtures_encode_design_choices", ""))
        out["alignment_v4_helpers"] = str(v4.get("helpers_access_private_api", ""))
        out["alignment_v4_assertions"] = str(v4.get("assertions_format_only", ""))

    # Aggregate token/latency across all LLM calls this row made
    if attempts:
        out["alignment_latency_s"] = f"{sum(a.get('latency_s', 0) for a in attempts):.2f}"
        out["alignment_tokens_in"] = str(sum(a.get("tokens_in", 0) for a in attempts))
        out["alignment_tokens_out"] = str(sum(a.get("tokens_out", 0) for a in attempts))

    progress["done"] += 1
    print(
        f"[{progress['done']}/{progress['total']}] {row['task_id']}: "
        f"eval=accept build={out['new_build_words']}w "
        f"align={out['alignment_verdict']} regen={regen_count}",
        file=sys.stderr,
    )

    if verbose_fh is not None:
        verbose_fh.write(
            f"\n{'=' * 80}\n"
            f"task_id:   {row['task_id']}\n"
            f"repo/sha:  {repo}/{sha[:12]}\n"
            f"subject:   {row['subject'][:80]}\n"
            f"OLD pipeline:  hardness={row.get('hardness_verdict', '')} "
            f"final={row.get('final_stage', '')}\n"
            f"NEW evaluate:  {out['new_eval_verdict']}  reason={out['new_eval_reason'][:200]}\n"
            f"NEW build:     words={out['new_build_words']}  regen_count={regen_count}\n"
            f"--- new instruction_md ---\n{current_instruction}\n"
            f"NEW alignment: verdict={out['alignment_verdict']}  attempts={len(attempts)}\n"
            f"  reason: {out['alignment_reason'][:400]}\n"
        )
        if out["alignment_leakage_evidence"]:
            verbose_fh.write(f"  leakage_evidence: {out['alignment_leakage_evidence'][:500]}\n")
        verbose_fh.flush()

    return out


async def judge_row(
    row: dict[str, str],
    sem: asyncio.Semaphore,
    progress: dict[str, int],
    verbose_fh=None,
) -> dict[str, Any]:
    """Run alignment judge on a single CSV row. Returns output-row dict."""
    out = dict(row)
    out.update(
        {
            "alignment_verdict": "",
            "alignment_reason": "",
            "alignment_leakage_evidence": "",
            "alignment_v4_fixtures": "",
            "alignment_v4_helpers": "",
            "alignment_v4_assertions": "",
            "alignment_attempts": "",
            "alignment_latency_s": "",
            "alignment_tokens_in": "",
            "alignment_tokens_out": "",
            "alignment_skip_reason": "",
        }
    )

    instruction_md = row.get("instruction_md", "").strip()
    if not instruction_md:
        out["alignment_skip_reason"] = "no_instruction"
        return out

    # Fetch git context
    tests, diff, skip = await asyncio.to_thread(
        fetch_reference_tests_and_diff, row["repo"], row["commit_sha"], ""
    )
    if skip:
        out["alignment_skip_reason"] = skip
        return out
    if not tests:
        out["alignment_skip_reason"] = "no_reference_tests"
        return out

    prompt = alignment_judge_prompt(
        instruction_md=instruction_md,
        reference_test_bodies=tests,
        diff=diff,
    )

    async with sem:
        try:
            judge_result = await llm_judge.judge(
                prompt=prompt,
                schema=ALIGNMENT_SCHEMA,
                model=os.environ.get("LLM_ALIGNMENT_MODEL", "openai/us/azure/openai/gpt-5.4"),
            )
        except Exception as err:
            out["alignment_skip_reason"] = f"judge_error:{type(err).__name__}:{err!s:.100}"
            progress["done"] += 1
            print(
                f"[{progress['done']}/{progress['total']}] {row['task_id']}: ERROR ({err})",
                file=sys.stderr,
            )
            return out

    result = judge_result.result
    v4 = result.get("v4_audit") or {}
    evidence = result.get("leakage_evidence") or []
    out["alignment_verdict"] = result.get("verdict", "")
    out["alignment_reason"] = result.get("reason", "")
    out["alignment_leakage_evidence"] = json.dumps(evidence, ensure_ascii=False)
    out["alignment_v4_fixtures"] = str(v4.get("fixtures_encode_design_choices", ""))
    out["alignment_v4_helpers"] = str(v4.get("helpers_access_private_api", ""))
    out["alignment_v4_assertions"] = str(v4.get("assertions_format_only", ""))
    out["alignment_attempts"] = "1"
    out["alignment_latency_s"] = f"{judge_result.latency_s:.2f}"
    out["alignment_tokens_in"] = str(judge_result.usage.get("input_tokens", 0))
    out["alignment_tokens_out"] = str(judge_result.usage.get("output_tokens", 0))
    progress["done"] += 1
    print(
        f"[{progress['done']}/{progress['total']}] {row['task_id']}: "
        f"{out['alignment_verdict']} ({judge_result.latency_s:.1f}s)"
    )

    if verbose_fh is not None:
        verbose_fh.write(
            f"\n{'=' * 80}\n"
            f"task_id:        {row['task_id']}\n"
            f"repo:           {row['repo']}\n"
            f"commit_sha:     {row['commit_sha'][:12]}\n"
            f"subject:        {row['subject'][:80]}\n"
            f"old_hardness:   {row.get('hardness_verdict', '')}  band={row.get('hardness_band', '')}\n"
            f"instruction_words: {row.get('instruction_words', '')}\n"
            f"--- instruction_md ---\n{instruction_md}\n"
            f"--- new alignment ---\n"
            f"verdict:        {out['alignment_verdict']}\n"
            f"reason:         {out['alignment_reason']}\n"
        )
        if evidence:
            verbose_fh.write("leakage_evidence:\n")
            for q in evidence:
                verbose_fh.write(f"  - {q}\n")
        verbose_fh.write(
            f"v4: fixtures={out['alignment_v4_fixtures']} "
            f"helpers={out['alignment_v4_helpers']} "
            f"assertions={out['alignment_v4_assertions']}\n"
            f"latency: {judge_result.latency_s:.2f}s  "
            f"tokens: {out['alignment_tokens_in']}/{out['alignment_tokens_out']}\n"
        )
        verbose_fh.flush()
    return out


def sample_rows(rows: list[dict], n: int, stratify_by: str = "hardness_verdict") -> list[dict]:
    """Stratified random sample. For calibration we want proportional
    representation across old-hardness verdicts so we can see relationship."""
    # In alignment-only mode we re-judge existing instructions, so rows must
    # carry instruction_md. In full mode we generate instructions fresh from
    # repo+sha, so an empty instruction_md is fine — only the repo presence
    # matters. The caller passes `mode` indirectly via this filter relaxation.
    has_instruction = [r for r in rows if r.get("instruction_md", "").strip()]
    eligible = has_instruction if has_instruction else list(rows)
    if n <= 0 or n >= len(eligible):
        # Still apply the local-repo filter so we don't try rows whose
        # repo isn't cloned.
        local_repos = set(os.listdir("repos")) if os.path.isdir("repos") else set()
        return [r for r in eligible if r["repo"] in local_repos] or eligible

    # Restrict to rows where the repo is locally present.
    local_repos = set(os.listdir("repos")) if os.path.isdir("repos") else set()
    eligible = [r for r in eligible if r["repo"] in local_repos]
    if n >= len(eligible):
        return eligible

    # Stratify by hardness verdict
    buckets: dict[str, list[dict]] = {}
    for r in eligible:
        buckets.setdefault(r.get(stratify_by, ""), []).append(r)

    sample: list[dict] = []
    total_eligible = sum(len(b) for b in buckets.values())
    for bucket, items in buckets.items():
        share = max(1, round(n * len(items) / total_eligible))
        random.shuffle(items)
        sample.extend(items[:share])
    random.shuffle(sample)
    return sample[:n]


def _pairwise_jaccard(strings: list[str]) -> dict:
    """Pairwise token-set Jaccard across candidate instruction_md payloads.

    Returns ``{min, mean, max, n_pairs}``. Local helper — production code
    never computes diversity (see plan: diversity is a one-time research
    question answered offline, not a per-task pipeline concern). Jaccard
    alone is insufficient for spotting leakage; calibration deliverable
    pairs this with manual spot-checks.
    """
    if len(strings) < 2:
        return {"min": 1.0, "mean": 1.0, "max": 1.0, "n_pairs": 0}
    token_sets = [set(s.split()) for s in strings]
    pairs: list[float] = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            a, b = token_sets[i], token_sets[j]
            union = a | b
            pairs.append(len(a & b) / len(union) if union else 1.0)
    return {
        "min": min(pairs),
        "mean": sum(pairs) / len(pairs),
        "max": max(pairs),
        "n_pairs": len(pairs),
    }


async def _run_full_pipeline_row_n_times(
    row: dict[str, str],
    sem: asyncio.Semaphore,
    progress: dict[str, int],
    ctx: PipelineContext,
    verbose_fh,
    enable_feedback: bool,
    n: int,
    skip_eval: bool = False,
    max_rebuilds: int = _cfg.MAX_BUILD_REGENS_PER_CANDIDATE,
    max_alignment_retries: int = _cfg.ALIGNMENT_MAX_RETRIES,
) -> dict[str, Any]:
    """Run ``run_full_pipeline_row`` N times concurrently per row, then merge.

    Models the N-parallel build+alignment orchestrator's behavior over an
    historical-cohort row. Picks a passing candidate uniformly at random
    (matching ``_select_winner``) and surfaces its single-row columns
    plus N-aggregate columns.
    """
    results = await asyncio.gather(
        *(
            run_full_pipeline_row(
                row,
                sem,
                progress,
                ctx,
                verbose_fh,
                enable_feedback=enable_feedback,
                skip_eval=skip_eval,
                max_rebuilds=max_rebuilds,
                max_alignment_retries=max_alignment_retries,
            )
            for _ in range(n)
        )
    )

    verdicts = [r.get("alignment_verdict", "") for r in results]
    regen_counts = [r.get("new_regen_count", "0") for r in results]
    instructions = [r.get("new_instruction_md", "") for r in results]
    passers_idx = [i for i, v in enumerate(verdicts) if v == "ok"]

    # Pick winner uniformly at random among passers, matching _select_winner.
    if passers_idx:
        winner_idx = random.choice(passers_idx)
    else:
        # No passer: keep cand0 for the single-row columns so the output
        # row still shows a representative verdict/reason for analysis.
        winner_idx = 0

    out = dict(results[winner_idx])

    diversity = _pairwise_jaccard([s for s in instructions if s])
    out["n_candidates"] = str(n)
    out["candidate_verdicts"] = ";".join(verdicts)
    out["candidate_regen_counts"] = ";".join(regen_counts)
    out["n_passed"] = str(len(passers_idx))
    out["winner_cand_id"] = str(winner_idx)
    out["mean_jaccard"] = f"{diversity['mean']:.3f}"
    out["min_jaccard"] = f"{diversity['min']:.3f}"
    out["max_jaccard"] = f"{diversity['max']:.3f}"
    # Surface every candidate's full instruction + reason for manual deep-dive.
    # Critical for tiny calibration runs where jaccard alone is uninformative;
    # the human reads each candidate to verify regen actually improves output.
    for i, r in enumerate(results):
        out[f"cand{i}_instruction"] = r.get("new_instruction_md", "")
        out[f"cand{i}_verdict"] = r.get("alignment_verdict", "")
        out[f"cand{i}_reason"] = r.get("alignment_reason", "")
        out[f"cand{i}_regen_count"] = r.get("new_regen_count", "0")
        out[f"cand{i}_leakage"] = r.get("alignment_leakage_evidence", "")
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="bigtest.csv")
    parser.add_argument("--output", default="bigtest_with_alignment.csv")
    parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help="Number of rows to sample (stratified by hardness_verdict). 0 = all eligible.",
    )
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample")
    parser.add_argument(
        "--verbose-log",
        default="",
        help="Optional path to write per-row detailed debug info "
        "(prompt-size, verdict, reason, evidence). For post-run inspection.",
    )
    parser.add_argument(
        "--mode",
        choices=["alignment-only", "full"],
        default="alignment-only",
        help=(
            "alignment-only: run alignment judge on each row's existing "
            "instruction_md (default). full: run new evaluate → build → "
            "alignment (with one regen on actionable rejection) on each "
            "row's repo/sha from scratch, producing fresh NEW instructions."
        ),
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="In full mode, disable the alignment→build regen feedback loop "
        "(useful for measuring first-pass quality in isolation).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="In full mode, bypass the evaluate step entirely. Build runs "
        "directly using each row's repo/sha/subject, with empty eval_reason "
        "and the row's subject as instruction_sketch. Use when measuring "
        "build+alignment variance in isolation on a pre-screened cohort "
        "(e.g., rerun-accepts-v2) where eval-step variance would just add "
        "noise to the build/align signal.",
    )
    parser.add_argument(
        "--max-rebuilds",
        type=int,
        default=_cfg.MAX_BUILD_REGENS_PER_CANDIDATE,
        help=f"Maximum number of alignment-feedback-driven Build regens per "
        f"candidate (default {_cfg.MAX_BUILD_REGENS_PER_CANDIDATE}, matches production). "
        f"Set to 1 to reproduce the prior Phase-1 default; bump to test whether "
        f"deeper redraft chains rescue more tasks.",
    )
    parser.add_argument(
        "--max-alignment-retries",
        type=int,
        default=_cfg.ALIGNMENT_MAX_RETRIES,
        help=f"Number of retention retries on the alignment judge "
        f"(default {_cfg.ALIGNMENT_MAX_RETRIES}, matches production). Set to 1 to "
        f"disable retention bias entirely (single roll, no early-exit-on-any-ok).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=_cfg.BUILD_N_CANDIDATES,
        help=(
            f"Number of parallel build+alignment candidate loops per row in "
            f"full mode (default {_cfg.BUILD_N_CANDIDATES}, matches production; capped at 4). "
            f"With n>1, runs N independent candidates per row and picks a winner "
            f"uniformly at random among passers — mirrors the production "
            f"orchestrator. Adds Jaccard "
            "distribution columns for diversity analysis. Manual spot-check "
            "of representative pairs is still required — Jaccard alone is "
            "insufficient for diagnosing leakage."
        ),
    )
    args = parser.parse_args()

    if args.n < 1:
        print(f"WARNING: --n={args.n} < 1, clamping to 1", file=sys.stderr)
        args.n = 1
    elif args.n > 4:
        print(f"WARNING: --n={args.n} > 4, clamping to 4 (cost guardrail)", file=sys.stderr)
        args.n = 4
    if args.n > 1 and args.mode != "full":
        parser.error("--n > 1 requires --mode=full (alignment-only mode is single-roll)")

    verbose_fh = open(args.verbose_log, "w") if args.verbose_log else None

    random.seed(args.seed)

    with open(args.input) as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {args.input}", file=sys.stderr)

    sampled = sample_rows(rows, args.sample)
    print(f"Sampled {len(sampled)} eligible rows (local repo + instruction_md)", file=sys.stderr)

    # Print sample distribution
    from collections import Counter

    dist = Counter(r.get("hardness_verdict", "") for r in sampled)
    print(f"  distribution by hardness_verdict: {dict(dist)}", file=sys.stderr)

    progress = {"done": 0, "total": len(sampled)}
    sem = asyncio.Semaphore(args.concurrency)
    ctx = PipelineContext()

    start = time.monotonic()
    if args.mode == "full":
        n_label = f", N={args.n}" if args.n > 1 else ""
        print(
            f"Mode: full pipeline (evaluate → build → alignment"
            f"{'' if args.no_feedback else ' + 1 regen on actionable'}{n_label})",
            file=sys.stderr,
        )
        if args.n > 1:
            # Each row spawns N independent build+align candidates concurrently
            # (mirrors the production orchestrator). Update progress total to
            # account for the multiplied call count.
            progress["total"] = len(sampled) * args.n
            results = await asyncio.gather(
                *(
                    _run_full_pipeline_row_n_times(
                        r,
                        sem,
                        progress,
                        ctx,
                        verbose_fh,
                        enable_feedback=not args.no_feedback,
                        n=args.n,
                        skip_eval=args.skip_eval,
                        max_rebuilds=args.max_rebuilds,
                        max_alignment_retries=args.max_alignment_retries,
                    )
                    for r in sampled
                )
            )
        else:
            results = await asyncio.gather(
                *(
                    run_full_pipeline_row(
                        r,
                        sem,
                        progress,
                        ctx,
                        verbose_fh,
                        enable_feedback=not args.no_feedback,
                        skip_eval=args.skip_eval,
                        max_rebuilds=args.max_rebuilds,
                        max_alignment_retries=args.max_alignment_retries,
                    )
                    for r in sampled
                )
            )
    else:
        print("Mode: alignment-only (judge existing instruction_md)", file=sys.stderr)
        results = await asyncio.gather(*(judge_row(r, sem, progress, verbose_fh) for r in sampled))
    wall = time.monotonic() - start

    # Write output CSV with union of original + new columns
    extra_cols = [
        "alignment_verdict",
        "alignment_reason",
        "alignment_leakage_evidence",
        "alignment_v4_fixtures",
        "alignment_v4_helpers",
        "alignment_v4_assertions",
        "alignment_attempts",
        "alignment_latency_s",
        "alignment_tokens_in",
        "alignment_tokens_out",
        "alignment_skip_reason",
    ]
    if args.mode == "full":
        extra_cols = [
            "new_eval_verdict",
            "new_eval_reason",
            "new_eval_reject_pattern",
            "new_instruction_sketch",
            "new_build_words",
            "new_instruction_md",
            "new_regen_count",
        ] + extra_cols
        if args.n > 1:
            extra_cols += [
                "n_candidates",
                "candidate_verdicts",
                "candidate_regen_counts",
                "n_passed",
                "winner_cand_id",
                "mean_jaccard",
                "min_jaccard",
                "max_jaccard",
            ]
            # Per-candidate columns for manual deep-dive in combination with
            # the jaccard distribution (does jaccard correlate with what a
            # human reading the instructions would say about diversity?).
            for i in range(args.n):
                extra_cols += [
                    f"cand{i}_instruction",
                    f"cand{i}_verdict",
                    f"cand{i}_reason",
                    f"cand{i}_regen_count",
                    f"cand{i}_leakage",
                ]
    fieldnames = list(rows[0].keys()) + extra_cols
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    if verbose_fh is not None:
        verbose_fh.close()

    # Summary stats
    processed = [r for r in results if r.get("alignment_latency_s")]
    latencies = [float(r["alignment_latency_s"]) for r in processed]
    tokens_in = sum(int(r["alignment_tokens_in"]) for r in processed)
    tokens_out = sum(int(r["alignment_tokens_out"]) for r in processed)

    print("", file=sys.stderr)
    print(f"Wrote {len(results)} rows to {args.output}", file=sys.stderr)
    print(f"Wall clock:         {wall:.1f}s ({wall / 60:.1f} min)", file=sys.stderr)
    if processed:
        print(
            f"Sum of call time:   {sum(latencies):.0f}s   "
            f"(speedup at c={args.concurrency}: {sum(latencies) / wall:.1f}x)",
            file=sys.stderr,
        )
        print(
            f"Per-call latency:   avg={sum(latencies) / len(latencies):.1f}s  "
            f"min={min(latencies):.1f}s  max={max(latencies):.1f}s",
            file=sys.stderr,
        )
        print(f"Tokens:             in={tokens_in:,}  out={tokens_out:,}", file=sys.stderr)
    print("", file=sys.stderr)

    verdict_dist = Counter(r.get("alignment_verdict", "(skipped)") for r in results)
    print(f"alignment_verdict distribution: {dict(verdict_dist)}", file=sys.stderr)
    skip_dist = Counter(r.get("alignment_skip_reason", "") for r in results if r.get("alignment_skip_reason"))
    if skip_dist:
        print(f"skip_reason distribution: {dict(skip_dist)}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
