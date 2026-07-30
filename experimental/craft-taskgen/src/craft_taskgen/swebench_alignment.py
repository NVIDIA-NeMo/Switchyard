# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the alignment judge over imported SWE-bench Pro candidates."""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import craft_taskgen.config as _cfg
from craft_taskgen import llm_judge
from craft_taskgen.config import PipelineState, Stage
from craft_taskgen.prompts import ALIGNMENT_SCHEMA, alignment_judge_prompt
from craft_taskgen.steps import _BUILD_DIFF_BYTE_CAP, _list_commit_test_files_sync


@dataclass
class AlignmentCandidate:
    task_id: str
    repo: str
    sha: str
    merge_base_sha: str
    source_task_id: str
    problem_statement: str
    requirements: str = ""
    interface: str = ""


def _task_dir_slug(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text.strip())
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:40] or "task"


def _instruction_from_metadata(
    source_metadata: dict[str, Any],
    *,
    include_requirements: bool,
    include_interface: bool,
) -> tuple[str, str]:
    problem_statement = str(source_metadata.get("problem_statement", "")).strip()
    if not problem_statement:
        raise ValueError("missing candidate_data.source_metadata.problem_statement")

    sections = [problem_statement]
    sources = ["problem_statement"]

    requirements = str(source_metadata.get("requirements", "")).strip()
    if include_requirements and requirements:
        sections.append(f"## Requirements\n{requirements}")
        sources.append("requirements")

    interface = str(source_metadata.get("interface", "")).strip()
    if include_interface and interface:
        sections.append(f"## Interface\n{interface}")
        sources.append("interface")

    return "\n\n".join(sections), "+".join(sources)


def _candidate_instruction(
    candidate: AlignmentCandidate,
    *,
    include_requirements: bool,
    include_interface: bool,
) -> tuple[str, str]:
    source_metadata = {
        "problem_statement": candidate.problem_statement,
        "requirements": candidate.requirements,
        "interface": candidate.interface,
    }
    return _instruction_from_metadata(
        source_metadata,
        include_requirements=include_requirements,
        include_interface=include_interface,
    )


def _materialize_promoted_task(
    state: PipelineState,
    task,
    *,
    include_requirements: bool = False,
    include_interface: bool = False,
) -> str:
    candidate_data = task.candidate_data or {}
    source_metadata = candidate_data.get("source_metadata", {}) if isinstance(candidate_data, dict) else {}
    instruction_md, _ = _instruction_from_metadata(
        source_metadata,
        include_requirements=include_requirements,
        include_interface=include_interface,
    )

    slug = _task_dir_slug(task.description or task.task_id)
    task_dir = task.task_dir or os.path.join(state.run_dir, f"t2v3-{task.task_id}-{slug}")
    os.makedirs(task_dir, exist_ok=True)

    instruction_path = os.path.join(task_dir, "instruction.md")
    Path(instruction_path).write_text(instruction_md)

    task.task_dir = task_dir
    task.instruction_words = len(instruction_md.split())
    return task_dir


def _promote_existing_ok_tasks(
    state: PipelineState,
    *,
    include_requirements: bool = False,
    include_interface: bool = False,
) -> None:
    promoted = 0
    skipped_non_promising = 0
    skipped_non_ok = 0
    for task in state.tasks.values():
        if task.alignment_verdict != "ok":
            skipped_non_ok += 1
            continue
        if task.stage not in (Stage.PROMISING, Stage.BUILT):
            skipped_non_promising += 1
            continue
        _materialize_promoted_task(
            state,
            task,
            include_requirements=include_requirements,
            include_interface=include_interface,
        )
        task.stage = Stage.ALIGNMENT_CHECKED
        promoted += 1

    print(
        "Promotion summary: "
        f"{promoted} promoted to {Stage.ALIGNMENT_CHECKED.value}, "
        f"{skipped_non_promising} skipped due to stage, "
        f"{skipped_non_ok} skipped due to non-ok alignment"
    )


def _expand_candidate_files(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    return sorted(set(files))


def _load_candidates(
    paths: list[str],
    repo_filter: str | None,
    instance_id: str | None,
) -> list[AlignmentCandidate]:
    wanted_repo = repo_filter.lower() if repo_filter else None
    out: list[AlignmentCandidate] = []

    for path in paths:
        data = json.loads(Path(path).read_text())
        if "repo" not in data:
            raise ValueError(f"{path}: missing repo")
        if "candidates" not in data:
            raise ValueError(f"{path}: missing candidates")

        repo = str(data["repo"]).strip()
        if not repo:
            raise ValueError(f"{path}: repo is empty")
        if not isinstance(data["candidates"], list):
            raise ValueError(f"{path}: candidates must be a list")
        if wanted_repo and wanted_repo != repo.lower():
            continue

        for idx, cand in enumerate(data["candidates"]):
            if not isinstance(cand, dict):
                raise ValueError(f"{path}: candidates[{idx}] must be an object")
            if "source_metadata" not in cand:
                raise ValueError(f"{path}: candidates[{idx}] missing source_metadata")
            if "source_task_id" not in cand:
                raise ValueError(f"{path}: candidates[{idx}] missing source_task_id")
            if "sha" not in cand:
                raise ValueError(f"{path}: candidates[{idx}] missing sha")
            if "merge_base_sha" not in cand:
                raise ValueError(f"{path}: candidates[{idx}] missing merge_base_sha")

            source_metadata = cand["source_metadata"]
            if not isinstance(source_metadata, dict):
                raise ValueError(f"{path}: candidates[{idx}].source_metadata must be an object")
            if "problem_statement" not in source_metadata:
                raise ValueError(f"{path}: candidates[{idx}].source_metadata missing problem_statement")

            source_task_id = str(cand["source_task_id"]).strip()
            sha = str(cand["sha"]).strip()
            merge_base_sha = str(cand["merge_base_sha"]).strip()
            problem_statement = str(source_metadata["problem_statement"]).strip()
            requirements = str(source_metadata.get("requirements", "")).strip()
            interface = str(source_metadata.get("interface", "")).strip()
            if not source_task_id:
                raise ValueError(f"{path}: candidates[{idx}].source_task_id is empty")
            if not sha:
                raise ValueError(f"{path}: candidates[{idx}].sha is empty")
            if not merge_base_sha:
                raise ValueError(f"{path}: candidates[{idx}].merge_base_sha is empty")
            if not problem_statement:
                raise ValueError(f"{path}: candidates[{idx}].source_metadata.problem_statement is empty")
            if instance_id and source_task_id != instance_id:
                continue
            out.append(
                AlignmentCandidate(
                    task_id="",
                    repo=repo,
                    sha=sha,
                    merge_base_sha=merge_base_sha,
                    source_task_id=source_task_id,
                    problem_statement=problem_statement,
                    requirements=requirements,
                    interface=interface,
                )
            )
    return out


def _load_candidates_from_state(
    state_path: str,
    repo_filter: str | None,
    instance_id: str | None,
) -> tuple[PipelineState, list[AlignmentCandidate]]:
    state = PipelineState.load(state_path)
    wanted_repo = repo_filter.lower() if repo_filter else None
    out: list[AlignmentCandidate] = []

    for task_id, task in state.tasks.items():
        candidate_data = task.candidate_data or {}
        if "source_metadata" not in candidate_data:
            raise ValueError(f"{task_id}: missing candidate_data.source_metadata")
        if "source_task_id" not in candidate_data:
            raise ValueError(f"{task_id}: missing candidate_data.source_task_id")

        source_metadata = candidate_data["source_metadata"]
        if not isinstance(source_metadata, dict):
            raise ValueError(f"{task_id}: candidate_data.source_metadata must be an object")
        if "problem_statement" not in source_metadata:
            raise ValueError(f"{task_id}: candidate_data.source_metadata missing problem_statement")

        source_task_id = str(candidate_data["source_task_id"]).strip()
        problem_statement = str(source_metadata["problem_statement"]).strip()
        requirements = str(source_metadata.get("requirements", "")).strip()
        interface = str(source_metadata.get("interface", "")).strip()
        if not source_task_id:
            raise ValueError(f"{task_id}: candidate_data.source_task_id is empty")
        if not problem_statement:
            raise ValueError(f"{task_id}: candidate_data.source_metadata.problem_statement is empty")
        if wanted_repo and wanted_repo != task.repo.lower():
            continue
        if instance_id and source_task_id != instance_id:
            continue
        out.append(
            AlignmentCandidate(
                task_id=task_id,
                repo=task.repo,
                sha=task.commit_sha,
                merge_base_sha=task.merge_base_sha,
                source_task_id=source_task_id,
                problem_statement=problem_statement,
                requirements=requirements,
                interface=interface,
            )
        )
    return state, out


def _truncate_diff(diff: str) -> tuple[str, bool]:
    if len(diff) <= _BUILD_DIFF_BYTE_CAP:
        return diff, False
    half = _BUILD_DIFF_BYTE_CAP // 2
    omitted_lines = diff.count("\n") - diff[:half].count("\n") - diff[-half:].count("\n")
    omitted_bytes = len(diff) - _BUILD_DIFF_BYTE_CAP
    marker = f"\n\n[...truncated {omitted_lines} lines ({omitted_bytes:,} bytes) omitted...]\n\n"
    return diff[:half] + marker + diff[-half:], True


def _build_context(candidate: AlignmentCandidate, repos_dir: str) -> dict[str, Any]:
    if not candidate.problem_statement:
        raise ValueError("missing source_metadata.problem_statement")
    if not candidate.sha:
        raise ValueError("missing candidate sha")
    if not candidate.merge_base_sha:
        raise ValueError("missing candidate merge_base_sha")

    repo_path = os.path.join(repos_dir, candidate.repo)
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        raise ValueError(f"missing git clone at {repo_path}")

    test_paths = _list_commit_test_files_sync(repo_path, candidate.merge_base_sha, candidate.sha)
    reference_test_bodies: list[tuple[str, str]] = []
    for rel_path in test_paths:
        try:
            body = subprocess.check_output(
                ["git", "-C", repo_path, "show", f"{candidate.sha}:{rel_path}"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            continue
        reference_test_bodies.append((rel_path, body))

    try:
        diff = subprocess.check_output(
            ["git", "-C", repo_path, "diff", candidate.merge_base_sha, candidate.sha],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as err:
        raise ValueError(f"git diff failed: {err.output[:200]}") from err

    diff, diff_truncated = _truncate_diff(diff)
    return {
        "instruction_md": candidate.problem_statement,
        "reference_test_bodies": reference_test_bodies,
        "diff": diff,
        "diff_truncated": diff_truncated,
    }


async def _judge_one(
    candidate: AlignmentCandidate,
    repos_dir: str,
    model: str,
    dry_run: bool,
    include_requirements: bool,
    include_interface: bool,
) -> dict[str, Any]:
    result = {
        "task_id": candidate.task_id,
        "source_task_id": candidate.source_task_id,
        "repo": candidate.repo,
        "commit_sha": candidate.sha,
        "merge_base_sha": candidate.merge_base_sha,
    }

    try:
        context = await asyncio.to_thread(_build_context, candidate, repos_dir)
    except Exception as err:
        result["status"] = "context_error"
        result["error"] = str(err)
        return result

    if not context["reference_test_bodies"]:
        result["status"] = "skipped"
        result["error"] = "no reference test files found"
        result["reference_test_count"] = 0
        result["reference_test_paths"] = []
        return result

    instruction_md, instruction_source = _candidate_instruction(
        candidate,
        include_requirements=include_requirements,
        include_interface=include_interface,
    )
    prompt = alignment_judge_prompt(
        instruction_md=instruction_md,
        reference_test_bodies=context["reference_test_bodies"],
        diff=context["diff"],
    )

    if dry_run:
        result.update(
            {
                "status": "dry_run",
                "instruction_source": instruction_source,
                "requirements_included": "requirements" in instruction_source.split("+"),
                "interface_included": "interface" in instruction_source.split("+"),
                "reference_test_count": len(context["reference_test_bodies"]),
                "reference_test_paths": [path for path, _ in context["reference_test_bodies"]],
                "diff_truncated": context["diff_truncated"],
                "prompt": prompt,
            }
        )
        return result

    try:
        judged = await llm_judge.judge(prompt=prompt, schema=ALIGNMENT_SCHEMA, model=model)
    except Exception as err:
        result["status"] = "judge_error"
        result["error"] = str(err)
        result["reference_test_count"] = len(context["reference_test_bodies"])
        result["reference_test_paths"] = [path for path, _ in context["reference_test_bodies"]]
        return result

    payload = judged.result
    result.update(
        {
            "status": "ok",
            "instruction_source": instruction_source,
            "requirements_included": "requirements" in instruction_source.split("+"),
            "interface_included": "interface" in instruction_source.split("+"),
            "reference_test_count": len(context["reference_test_bodies"]),
            "reference_test_paths": [path for path, _ in context["reference_test_bodies"]],
            "diff_truncated": context["diff_truncated"],
            "verdict": payload.get("verdict", ""),
            "reason": payload.get("reason", ""),
            "leakage_evidence": payload.get("leakage_evidence", []),
            "v4_audit": payload.get("v4_audit", {}),
            "model": judged.model,
            "tokens_in": judged.usage.get("input_tokens", 0),
            "tokens_out": judged.usage.get("output_tokens", 0),
            "latency_s": judged.latency_s,
        }
    )
    return result


async def _run(args: argparse.Namespace) -> list[dict[str, Any]]:
    state: PipelineState | None = None
    candidate_files: list[str] = []
    if args.state_json:
        state, candidates = _load_candidates_from_state(args.state_json, args.repo, args.instance_id)
        source_label = f"state {args.state_json}"
    else:
        candidate_files = _expand_candidate_files(args.candidates)
        if not candidate_files:
            raise RuntimeError("no candidate files matched")
        candidates = _load_candidates(candidate_files, args.repo, args.instance_id)
        source_label = f"{len(candidate_files)} file(s)"
    if args.limit:
        candidates = candidates[: args.limit]
    if args.promote_existing_ok:
        if state is None:
            raise RuntimeError("--promote-existing-ok requires --state-json")
        _promote_existing_ok_tasks(
            state,
            include_requirements=args.include_requirements,
            include_interface=args.include_interface,
        )
        state.save(args.state_json)
        print(f"Updated state: {args.state_json}")
        return []
    if not candidates:
        raise RuntimeError("no matching candidates found")

    print(f"Loaded {len(candidates)} candidate(s) from {source_label}")
    if args.dry_run:
        print(f"Building alignment prompts only (concurrency={args.concurrency}, model={args.model})")
    else:
        print(f"Running alignment judge (concurrency={args.concurrency}, model={args.model})")
    if args.include_interface:
        print("Including source_metadata.interface in alignment instructions when present")
    if args.include_requirements:
        print("Including source_metadata.requirements in alignment instructions when present")

    sem = asyncio.Semaphore(args.concurrency)
    progress_lock = asyncio.Lock()
    completed = 0

    async def _wrapped(candidate: AlignmentCandidate) -> dict[str, Any]:
        nonlocal completed
        async with sem:
            result = await _judge_one(
                candidate,
                args.repos_dir,
                args.model,
                args.dry_run,
                args.include_requirements,
                args.include_interface,
            )
        async with progress_lock:
            completed += 1
            label = candidate.source_task_id or f"{candidate.repo}@{candidate.sha[:8]}"
            status = result.get("verdict") if result.get("status") == "ok" else result.get("status")
            print(f"[{completed}/{len(candidates)}] {label} -> {status}")
        return result

    results = await asyncio.gather(*[_wrapped(candidate) for candidate in candidates])
    if state is not None:
        _apply_results_to_state(
            state,
            results,
            promote_ok=args.promote_ok,
            include_requirements=args.include_requirements,
            include_interface=args.include_interface,
        )
        state.save(args.state_json)
        print(f"Updated state: {args.state_json}")
    return results


def _apply_results_to_state(
    state: PipelineState,
    results: list[dict[str, Any]],
    *,
    promote_ok: bool = False,
    include_requirements: bool = False,
    include_interface: bool = False,
) -> None:
    promoted = 0
    skipped_non_promising = 0
    skipped_non_ok = 0
    for row in results:
        task_id = row.get("task_id", "")
        if not task_id or task_id not in state.tasks:
            continue
        task = state.tasks[task_id]
        status = row.get("status", "")
        if status == "ok":
            task.alignment_verdict = row.get("verdict", "")
            task.alignment_reason = row.get("reason", "")
            task.alignment_v4_audit = row.get("v4_audit", {}) or {}
            task.alignment_attempts = [
                {
                    "attempt": 1,
                    "verdict": row.get("verdict", ""),
                    "reason": row.get("reason", ""),
                    "v4_audit": row.get("v4_audit", {}) or {},
                    "leakage_evidence": row.get("leakage_evidence", []) or [],
                    "tokens_in": row.get("tokens_in", 0),
                    "tokens_out": row.get("tokens_out", 0),
                    "latency_s": row.get("latency_s", 0.0),
                    "model": row.get("model", ""),
                }
            ]
        else:
            task.alignment_verdict = status
            task.alignment_reason = row.get("error", "")
            task.alignment_v4_audit = {}
            task.alignment_attempts = [{"attempt": 1, "verdict": status, "reason": row.get("error", "")}]

        if not promote_ok:
            continue
        if status != "ok" or row.get("verdict") != "ok":
            skipped_non_ok += 1
            continue
        if task.stage not in (Stage.PROMISING, Stage.BUILT):
            skipped_non_promising += 1
            continue
        _materialize_promoted_task(
            state,
            task,
            include_requirements=include_requirements,
            include_interface=include_interface,
        )
        task.stage = Stage.ALIGNMENT_CHECKED
        promoted += 1

    if promote_ok:
        print(
            "Promotion summary: "
            f"{promoted} promoted to {Stage.ALIGNMENT_CHECKED.value}, "
            f"{skipped_non_promising} skipped due to stage, "
            f"{skipped_non_ok} skipped due to non-ok alignment"
        )


def _print_summary(results: list[dict[str, Any]]) -> None:
    statuses = Counter(row.get("status", "?") for row in results)
    verdicts = Counter(row.get("verdict", "") for row in results if row.get("status") == "ok")

    print(f"Processed: {len(results)}")
    for verdict in ("ok", "vague", "narrow_tests", "leaked", "misaligned"):
        if verdicts[verdict]:
            print(f"  {verdict}: {verdicts[verdict]}")
    for status in ("dry_run", "skipped", "context_error", "judge_error"):
        if statuses[status]:
            print(f"  {status}: {statuses[status]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidates", nargs="+", help="Candidate JSON files or glob patterns.")
    group.add_argument("--state-json", help="Pipeline state.json to read and update in place.")
    parser.add_argument("--output", default=None, help="Optional output JSONL path.")
    parser.add_argument("--repos-dir", default="repos", help="Local git clone directory.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum candidates to process (0 = all).")
    parser.add_argument("--repo", default=None, help="Filter by short repo name.")
    parser.add_argument("--instance-id", default=None, help="Filter by source_task_id.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=_cfg.LLM_CONCURRENCY,
        help="Max concurrent judge calls.",
    )
    parser.add_argument(
        "--model",
        default=_cfg.LLM_ALIGNMENT_MODEL,
        help="Alignment-judge model override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts but skip judge calls. Useful for inspecting a single instance.",
    )
    parser.add_argument(
        "--include-interface",
        action="store_true",
        help=(
            "Append source_metadata.interface to the problem statement before running the alignment judge."
        ),
    )
    parser.add_argument(
        "--include-requirements",
        action="store_true",
        help=(
            "Append source_metadata.requirements to the problem statement before running the alignment judge."
        ),
    )
    parser.add_argument(
        "--promote-ok",
        action="store_true",
        help=(
            "When used with --state-json, materialize task_dir + instruction.md and "
            "promote only PROMISING/BUILT tasks with alignment verdict 'ok' to ALIGNMENT_CHECKED."
        ),
    )
    parser.add_argument(
        "--promote-existing-ok",
        action="store_true",
        help=(
            "When used with --state-json, skip judging and promote already-stored "
            "alignment_verdict='ok' tasks from PROMISING/BUILT to ALIGNMENT_CHECKED."
        ),
    )
    args = parser.parse_args()

    if (args.promote_ok or args.promote_existing_ok) and not args.state_json:
        print("ERROR: --promote-ok/--promote-existing-ok require --state-json", file=sys.stderr)
        return 1
    if args.promote_ok and args.promote_existing_ok:
        print("ERROR: choose only one of --promote-ok or --promote-existing-ok", file=sys.stderr)
        return 1
    if (args.promote_ok or args.promote_existing_ok) and args.dry_run:
        print("ERROR: promotion flags cannot be used with --dry-run", file=sys.stderr)
        return 1

    try:
        results = asyncio.run(_run(args))
    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for row in results:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    if args.dry_run and len(results) == 1 and results[0].get("status") == "dry_run":
        print()
        print(results[0]["prompt"])

    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
