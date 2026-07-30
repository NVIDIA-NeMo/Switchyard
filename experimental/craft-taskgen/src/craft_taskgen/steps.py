# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline step implementations — evaluate, build, alignment, docker, smoke, triage."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import craft_taskgen.config as _cfg
from craft_taskgen import llm_judge, rubrics
from craft_taskgen.claude_cli import (
    _fix_docker_or_shelve_async,
    _fix_f2p_p2p_classify_or_shelve_async,
    run_claude_async,
    save_state_locked,
)
from craft_taskgen.config import (
    PipelineContext,
    PipelineState,
    Stage,
)
from craft_taskgen.diagnostics import _next_diagnostic_path, _write_diagnostic
from craft_taskgen.docker import (
    run_docker_build_async,
    run_f2p_p2p_classify_async,
    run_score_check_async,
)
from craft_taskgen.prompts import (
    BUILD_SCHEMA,
    DEEP_DIVE_SCHEMA,
    EVALUATE_SCHEMA,
    FAIRNESS_REVIEW_SCHEMA,
    SCORE_PY_TEMPLATE,
    SOLVE_SH_TEMPLATE,
    build_dockerfile_prompt,
    build_task_prompt,
    deep_dive_prompt,
    easiness_triage_feedback_block,
    evaluate_candidate_prompt,
    fairness_review_prompt,
    reviewer_triage_feedback_block,
)
from craft_taskgen.runner import _run_smoke_async
from craft_taskgen.task_format import strip_instruction_boilerplate


def _parse_score_ratio(score_str: str) -> float | None:
    """Parse score string to a ratio. Handles both 'N/M' and 'F2P N/M, P2P N/M' formats."""
    import re

    if not score_str or "/" not in str(score_str):
        return None
    try:
        # Find all N/M pairs in the string
        pairs = re.findall(r"(\d+)/(\d+)", str(score_str))
        if not pairs:
            return None
        total_passed = sum(int(p[0]) for p in pairs)
        total_tests = sum(int(p[1]) for p in pairs)
        return total_passed / total_tests if total_tests > 0 else None
    except (ValueError, IndexError):
        return None


def _snapshot_task_files(task_dir: str) -> dict[str, int]:
    """Hash every file under task_dir (relative path → content hash).

    Used to detect what the fix agent changed vs. what the pipeline wrote
    (diagnostics/ gets filtered in the diff step, not here).
    """
    hashes: dict[str, int] = {}
    if not task_dir:
        return hashes
    for root, _dirs, files in os.walk(task_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, task_dir)
            try:
                with open(fpath, "rb") as fh:
                    hashes[rel] = hash(fh.read())
            except OSError:
                pass
    return hashes


def _diff_task_files(task_dir: str, pre_hashes: dict[str, int]) -> set[str]:
    """Return basenames of files that differ from pre_hashes snapshot.

    Skips the diagnostics/ directory: the pipeline writes its own fix log
    there via fix attempt helpers, so it would always appear "changed" and
    defeat the skip-only fast-path.
    """
    changed: set[str] = set()
    if not task_dir:
        return changed
    for root, _dirs, files in os.walk(task_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, task_dir)
            if rel.split(os.sep)[0] == "diagnostics":
                continue
            try:
                with open(fpath, "rb") as fh:
                    new_hash = hash(fh.read())
            except OSError:
                continue
            if rel not in pre_hashes or pre_hashes[rel] != new_hash:
                changed.add(os.path.basename(rel))
    return changed


SKIP_ONLY_FILES = frozenset({"f2p_skip.txt", "p2p_skip.txt"})


# --- Deterministic easiness --------------------------------------------------
# A task solved with `grep_read <= _EASINESS_GREP_READ_MAX` exploration calls
# gets flagged for human review. Fires only on reward == 1.0 (a failed trial
# can't be "too easy"). See docs/reference/easiness-heuristics.md for calibration.
# Threshold calibrated against recent bulk cohorts: at <= 5 the flag catches
# only genuinely minimal trajectories, with an expected flag rate of ~10-15%
# of accepted tasks. `pytest_runs == 0` was considered as a second trigger
# but dropped as noisy — agents can legitimately verify via runners we can't
# detect cheaply (unittest, custom shell scripts, ast.parse sanity checks).
_EASINESS_GREP_READ_MAX = 5

# Parses rows emitted by `harbor-lab tool-sequence` — format `| N | Tool | args |`.
_EASINESS_TOOL_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*(?P<tool>\[?\w+\]?)\s*\|\s*(?P<args>.*?)\s*\|$",
    re.MULTILINE,
)


def _count_tool_ops(tool_sequence_md: str) -> dict[str, int]:
    """Parse a harbor-lab tool_sequence block into per-tool counts.

    Returns a dict with at least `grep_read` and `pytest_runs` keys. Extra
    tool types are included for diagnostic logging; callers only need the
    grep_read count under the current rubric. pytest_runs is retained for
    observability (dashboard / human review) but no longer flags on its own.
    """
    counts: dict[str, int] = {"grep_read": 0, "pytest_runs": 0}
    for m in _EASINESS_TOOL_ROW_RE.finditer(tool_sequence_md or ""):
        tool = m.group("tool")
        args = m.group("args")
        if tool in ("Grep", "Read"):
            counts["grep_read"] += 1
        elif tool == "Bash" and "pytest" in args.lower():
            counts["pytest_runs"] += 1
        counts[tool] = counts.get(tool, 0) + 1
    return counts


def _deterministic_easiness(tool_sequence_md: str, reward_json: str) -> tuple[bool, str]:
    """Compute whether the trial looks suspiciously easy based on tool-call
    counts alone. Returns `(flag, reason)`. Does NOT auto-reject — callers
    route the flag into `needs_human_review`.

    Single trigger: `grep_read <= _EASINESS_GREP_READ_MAX` on reward==1.0
    trials. Agent solved without meaningfully exploring the codebase.
    """
    if not reward_json:
        return False, ""
    try:
        reward = float(json.loads(reward_json).get("reward", 0.0))
    except (ValueError, json.JSONDecodeError):
        return False, ""
    if reward < 1.0:
        return False, ""
    counts = _count_tool_ops(tool_sequence_md)
    if counts["grep_read"] <= _EASINESS_GREP_READ_MAX:
        return True, f"grep_read={counts['grep_read']} (<={_EASINESS_GREP_READ_MAX})"
    return False, ""


def _is_skip_only_change(changed_files: set[str]) -> bool:
    """True if the only files changed are f2p_skip.txt / p2p_skip.txt."""
    return bool(changed_files) and changed_files <= SKIP_ONLY_FILES


def _classify_fix(changed_files: set[str]) -> str:
    """Classify what the fix agent changed into one of four routing types.

    Returns:
        "skip_only"   — only f2p_skip.txt / p2p_skip.txt changed → re-score existing trial
        "dockerfile"  — only Dockerfile changed → jump straight to smoke
        "instruction" — only instruction.md changed → re-run alignment, then jump to smoke
        "other"       — anything else (or empty) → full rebuild from alignment
    """
    if _is_skip_only_change(changed_files):
        return "skip_only"
    if changed_files == {"Dockerfile"}:
        return "dockerfile"
    if changed_files == {"instruction.md"}:
        return "instruction"
    return "other"


def _load_skipped_tests(task_dir: str) -> set[str]:
    """Load test names already excluded from scoring via f2p_skip.txt / p2p_skip.txt.

    These tests still appear as FAILED in pytest output, so the deep-dive agent
    picks them up and reports them as failures every iteration. Filter them out
    before the reviewer / fix agent see them — otherwise the pipeline loops
    polishing the skip file cosmetically while the underlying pytest output is
    unchanged.

    Format: `test_path::test_name | optional reason` — blank lines and `#`
    comments ignored.

    Both the full test ID (``path::name``) and the short name (after ``:`:``)
    are indexed so the filter matches regardless of whether the deep-dive agent
    reports the full pytest path or just the function name.
    """
    skipped: set[str] = set()
    if not task_dir:
        return skipped
    for fname in ("f2p_skip.txt", "p2p_skip.txt"):
        path = os.path.join(task_dir, "tests", fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    test_id = line.split("|")[0].strip()
                    if test_id:
                        skipped.add(test_id)
                        if "::" in test_id:
                            skipped.add(test_id.rsplit("::", 1)[-1])
        except OSError:
            pass
    return skipped


def _load_actually_failed_tests(trial_dir: str) -> set[str] | None:
    """Load the set of tests that actually failed in this trial.

    Returns a set containing every failed test_id plus its short name (after
    `::`) for robust matching against whatever shape the deep-dive agent used
    in its `failures[]` output.

    Primary source: `reward-details.json` `f2p_failed` + `p2p_failed` arrays
    (current SCORE_PY_TEMPLATE — split out so reward.json stays numeric-only
    for harbor>=0.13.1 pydantic validation). Fallbacks, in order:
      - older `reward.json` that still embeds the arrays directly
      - parse FAILED lines from `verify_full_output.txt`
    so trials produced by any era of the template still classify correctly,
    instead of filtering every DD classification against an empty set.

    Returns None only when no source is available.
    """
    if not trial_dir:
        return None

    def _expand(test_ids: set[str]) -> set[str]:
        out: set[str] = set()
        for tid in test_ids:
            if not tid:
                continue
            out.add(tid)
            if "::" in tid:
                out.add(tid.rsplit("::", 1)[-1])
        return out

    for fname in ("reward-details.json", "reward.json"):
        path = os.path.join(trial_dir, "verifier", fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if "f2p_failed" in data or "p2p_failed" in data:
            raw: set[str] = set()
            for key in ("f2p_failed", "p2p_failed"):
                raw.update(data.get(key, []) or [])
            return _expand(raw)

    verify_output = os.path.join(trial_dir, "verifier", "verify_full_output.txt")
    if not os.path.isfile(verify_output):
        return None
    try:
        with open(verify_output) as f:
            full_output = f.read()
    except OSError:
        return None
    raw = {m.group(1) for m in re.finditer(r"^(\S+::\S+)\s+FAILED", full_output, re.MULTILINE)}
    return _expand(raw)


@contextlib.asynccontextmanager
async def _mark_in_progress(task, step_name: str, state: PipelineState, state_file: str):
    """Set in_progress_step at entry and save, clear on exit and save."""
    task.in_progress_step = step_name
    await save_state_locked(state, state_file)
    try:
        yield
    finally:
        task.in_progress_step = ""
        await save_state_locked(state, state_file)


def _rescore_trial(task_dir: str, trial_dir: str) -> bool:
    """Re-run score.py against existing trial output with updated skip files.

    Copies score.py, fail_to_pass.txt, pass_to_pass.txt, and any skip files
    from the task's tests/ dir into the trial's verifier dir, then runs
    score.py with paths rewritten to local filesystem.
    """
    import shutil
    import tempfile

    verify_output = os.path.join(trial_dir, "verifier", "verify_full_output.txt")
    if not os.path.isfile(verify_output):
        return False

    tests_dir = os.path.join(task_dir, "tests")
    if not os.path.isdir(tests_dir):
        return False

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_tests = os.path.join(tmpdir, "tests")
            tmp_logs = os.path.join(tmpdir, "logs", "verifier")
            os.makedirs(tmp_tests)
            os.makedirs(tmp_logs)

            # Copy test list files and skip files
            for fname in (
                "fail_to_pass.txt",
                "pass_to_pass.txt",
                "f2p_skip.txt",
                "p2p_skip.txt",
            ):
                src = os.path.join(tests_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(tmp_tests, fname))

            # Copy existing verify output
            shutil.copy2(verify_output, os.path.join(tmp_logs, "verify_full_output.txt"))

            # Write score.py with rewritten paths
            score_code = SCORE_PY_TEMPLATE.replace("/tests/", f"{tmp_tests}/")
            score_code = score_code.replace("/logs/verifier", tmp_logs)
            score_path = os.path.join(tmpdir, "score.py")
            with open(score_path, "w") as f:
                f.write(score_code)

            result = subprocess.run(
                ["python3", score_path],
                capture_output=True,
                text=True,
                cwd=tmpdir,
                timeout=30,
            )
            if result.returncode != 0:
                print(f"    -> Re-score error: {result.stderr[:200]}")
                return False

            # Copy updated reward.json back to trial
            reward_src = os.path.join(tmp_logs, "reward.json")
            reward_dst = os.path.join(trial_dir, "verifier", "reward.json")
            if os.path.isfile(reward_src):
                shutil.copy2(reward_src, reward_dst)
                return True
            return False
    except Exception as e:
        print(f"    -> Re-score exception: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 1: Select candidates
# ---------------------------------------------------------------------------


def select_candidates(
    candidate_files: list[str],
    top_per_repo: int = 5,
    skip_per_repo: int = 0,
    max_total: int = 30,
) -> list[dict]:
    """Pull top candidates from mining output, filter for feature-shaped commits.

    `top_per_repo=0` means no per-repo cap.
    `max_total=0` means no global cap.
    """
    from craft_taskgen.prefilters import prefilter_candidate

    all_candidates = []
    prefilter_rejects = 0
    for fpath in candidate_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to read candidate file {fpath}: {e}") from e
        repo = Path(fpath).stem
        repo_path = os.path.join("repos", repo)
        if not os.path.isdir(repo_path):
            print(
                f"  WARNING: candidate file {Path(fpath).name} → repo name '{repo}' "
                f"but repos/{repo}/ does not exist. "
                f"Rename the file to match the cloned repo directory "
                f"(e.g. candidates/networkx.json → repos/networkx)."
            )
        repo_candidates = data.get("candidates", [])[skip_per_repo:]
        if top_per_repo > 0:
            repo_candidates = repo_candidates[:top_per_repo]
        for c in repo_candidates:
            if c.get("score", 0) <= 0:
                continue
            # Pre-filter before expensive LLM eval
            reject_reason = prefilter_candidate(c)
            if reject_reason:
                prefilter_rejects += 1
                continue
            sha = c["sha"]
            base_sha = c["base_sha"]
            merge_base_sha = c.get("merge_base_sha", "")
            if not sha or not base_sha:
                raise ValueError(
                    f"Candidate in {fpath} has empty sha or base_sha: sha={sha!r}, base_sha={base_sha!r}"
                )
            if not merge_base_sha:
                raise ValueError(
                    f"Candidate in {fpath} has empty merge_base_sha — "
                    f"re-run the miner to regenerate candidates."
                )
            # These SHAs/refs flow into git subprocess argv downstream. The calls
            # use list-form subprocess (no shell), so the real risk is argument
            # injection: a value starting with '-' would be read as a git option.
            # Require a commit-ish token that starts alphanumeric and contains
            # only safe ref characters, which blocks option-injection and stray
            # whitespace/metacharacters (CWE-88).
            for _label, _val in (("sha", sha), ("base_sha", base_sha), ("merge_base_sha", merge_base_sha)):
                if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._/-]*", _val):
                    raise ValueError(
                        f"Candidate in {fpath} has malformed {_label}={_val!r} "
                        f"(must start alphanumeric and contain only [0-9A-Za-z._/-])."
                    )
            all_candidates.append(
                {
                    "repo": repo,
                    "sha": sha,
                    "base_sha": base_sha,
                    "merge_base_sha": merge_base_sha,
                    "subject": c.get("subject", ""),
                    "score": c["score"],
                    "src_files": len(c.get("source_files", [])),
                    "test_files": len(c.get("test_files", [])),
                    "src_lines": c.get("source_lines_changed", 0),
                    "_raw": {"repo": repo, **c},
                }
            )

    if prefilter_rejects:
        print(f"  Pre-filtered {prefilter_rejects} candidates (docs/CI/formatting/version-bump)")
    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    if max_total > 0:
        return all_candidates[:max_total]
    return all_candidates


# ---------------------------------------------------------------------------
# Step 2: Evaluate candidates
# ---------------------------------------------------------------------------


_DIFF_BYTE_CAP = 40_000  # head/tail-truncate diffs larger than this for Evaluate
_README_LINE_CAP = 50


def _fetch_evaluate_context(repo: str, sha: str, merge_base_sha: str) -> tuple[str, str, str]:
    """Pre-assemble diff_stat, diff, and README excerpt for the evaluate prompt.

    The direct-API judge has no Bash/Read tools — context must be collected
    in Python. We use `git diff merge_base..sha` rather than `git show sha`
    because for merge commits `git show` returns only the merge message
    (empty diff body), while `git diff merge_base..sha` shows the actual
    combined change the PR introduced. Diff is head/tail-truncated at
    _DIFF_BYTE_CAP with a marker.
    """
    repo_dir = f"repos/{repo}"
    try:
        diff_stat = subprocess.check_output(
            ["git", "-C", repo_dir, "diff", "--stat", merge_base_sha, sha],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        err = e.output.decode("utf-8", errors="replace") if isinstance(e.output, bytes) else str(e.output)
        diff_stat = f"[git diff --stat failed: {err[:200]}]"

    try:
        diff = subprocess.check_output(
            ["git", "-C", repo_dir, "diff", merge_base_sha, sha],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        err = e.output.decode("utf-8", errors="replace") if isinstance(e.output, bytes) else str(e.output)
        diff = f"[git diff failed: {err[:200]}]"

    if len(diff) > _DIFF_BYTE_CAP:
        half = _DIFF_BYTE_CAP // 2
        omitted_lines = diff.count("\n") - diff[:half].count("\n") - diff[-half:].count("\n")
        omitted_bytes = len(diff) - _DIFF_BYTE_CAP
        marker = f"\n\n[...truncated {omitted_lines} lines ({omitted_bytes:,} bytes) omitted...]\n\n"
        diff = diff[:half] + marker + diff[-half:]

    readme_excerpt = ""
    for readme_name in ("README.md", "README.rst", "README"):
        readme_path = Path(repo_dir) / readme_name
        if readme_path.exists():
            try:
                lines = readme_path.read_text(errors="replace").splitlines()[:_README_LINE_CAP]
                readme_excerpt = "\n".join(lines)
            except OSError:
                pass
            break

    return diff_stat, diff, readme_excerpt


async def step_evaluate(state: PipelineState, state_file: str, concurrency: int = 4) -> None:
    """Evaluate CANDIDATE tasks via direct-API judge. Advances to PROMISING or REJECTED."""
    candidates = [t for t in state.tasks.values() if t.stage == Stage.CANDIDATE]
    if not candidates:
        print("No candidates to evaluate.")
        return

    sem = asyncio.Semaphore(concurrency)

    async def _evaluate_one(i: int, task) -> None:
        async with sem, _mark_in_progress(task, "evaluate", state, state_file):
            print(f"  [{i}/{len(candidates)}] {task.repo}/{task.commit_sha[:8]} — {task.description[:60]}")

            # Pre-assemble context in Python (direct-API judge has no tools).
            diff_stat, diff, readme = await asyncio.to_thread(
                _fetch_evaluate_context, task.repo, task.commit_sha, task.merge_base_sha
            )
            prompt = evaluate_candidate_prompt(
                repo=task.repo,
                sha=task.commit_sha,
                subject=task.description,
                diff_stat=diff_stat,
                diff=diff,
                readme_excerpt=readme,
            )

            try:
                judge_result = await llm_judge.judge(
                    prompt=prompt,
                    schema=EVALUATE_SCHEMA,
                    model=_cfg.LLM_STEP_MODEL,
                )
            except Exception as err:
                print(f"    ERROR: {err}")
                task.eval_verdict = "ERROR"
                task.eval_reason = f"{type(err).__name__}: {err}"
                task.stage = Stage.EVALUATED
            else:
                output = judge_result.result
                task.eval_verdict = output.get("verdict", "ERROR")
                task.eval_reason = output.get("reason", "")
                task.eval_instruction_sketch = output.get("instruction_sketch", "")
                task.eval_verifier_notes = output.get("reject_pattern", "")
                task.llm_usage.setdefault("evaluate", []).append(
                    {
                        "tokens_in": judge_result.usage.get("input_tokens", 0),
                        "tokens_out": judge_result.usage.get("output_tokens", 0),
                        "tokens_cached": judge_result.usage.get("cached_tokens", 0),
                        "model": judge_result.model,
                        "latency_s": round(judge_result.latency_s, 3),
                    }
                )

                if task.eval_verdict == "accept":
                    task.stage = Stage.PROMISING
                    print(f"    -> ACCEPT: {task.eval_reason[:80]}")
                else:
                    task.stage = Stage.REJECTED
                    pattern = output.get("reject_pattern", "")
                    pattern_suffix = f" [{pattern}]" if pattern else ""
                    print(f"    -> REJECT{pattern_suffix}: {task.eval_reason[:80]}")

            task.iteration_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "step": "evaluate",
                    "verdict": task.eval_verdict,
                    "reason": task.eval_reason,
                }
            )

            await save_state_locked(state, state_file)

    print(f"Evaluating {len(candidates)} candidates (concurrency={concurrency})...")
    # return_exceptions=True so one task's unhandled error (e.g. a UnicodeDecodeError
    # from a non-UTF-8 diff) doesn't tear down the whole asyncio.gather and lose the
    # other candidates. Mark the crashed task as rejected with the exception detail.
    results = await asyncio.gather(
        *[_evaluate_one(i, t) for i, t in enumerate(candidates, 1)],
        return_exceptions=True,
    )
    for (_i, task), r in zip(enumerate(candidates, 1), results):
        if isinstance(r, BaseException):
            task.stage = Stage.REJECTED
            task.eval_verdict = "error"
            task.eval_reason = f"evaluate crashed: {type(r).__name__}: {r}"[:400]
            print(f"  [evaluate ERROR] {task.task_id}: {type(r).__name__}: {r}")
    await save_state_locked(state, state_file)


# ---------------------------------------------------------------------------
# Step 3: Build task packages
# ---------------------------------------------------------------------------


def _generate_task_id(repo: str, commit_sha: str) -> str:
    """Generate a deterministic task ID from repo + sha. No scanning, no races."""
    prefix = repo.replace("-", "")[:2].upper()
    return f"{prefix}{commit_sha[:4]}"


def _is_test_file(path: str) -> bool:
    """Return True if path is a pytest test file.

    Matches test_*.py, *_test.py, *.test.py, and files in tests/ or test/ directories.
    Intentionally excludes conftest.py — that is pytest infrastructure, not a test
    file, and must remain in changes.patch so the oracle can apply it cleanly.
    """
    if not path.endswith(".py"):
        return False
    base = os.path.basename(path)
    if base == "conftest.py":
        return False
    if base.startswith("test_") or base.endswith("_test.py") or base.endswith(".test.py"):
        return True
    # file lives inside a tests/ or test/ directory anywhere in the path
    parts = path.replace("\\", "/").split("/")
    return any(part.lower() in ("tests", "test") for part in parts[:-1])


def _generate_solve_sh(repo: str, merge_base_sha: str, commit_sha: str, task_dir: str) -> tuple[bool, str]:
    """Mechanically generate solution/solve.sh and solution/changes.patch from git diff.

    changes.patch excludes test files that were added/modified by the commit.
    Those files are handled by the postmerge overlay in the classify step, and
    including them causes git-apply conflicts (file already exists at target state).

    Returns (ok, error_message).
    """
    repo_path = os.path.join("repos", repo)
    solution_dir = os.path.join(task_dir, "solution")
    os.makedirs(solution_dir, exist_ok=True)

    # Find Added/Modified test files to exclude from the patch.
    # The postmerge overlay places them at the post-commit state before solve.sh
    # runs, so including them in changes.patch causes git-apply to fail.
    try:
        names_result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only", "--diff-filter=AM", merge_base_sha, commit_sha],
            capture_output=True,
            text=True,
            timeout=60,
        )
        test_files = (
            [f.strip() for f in names_result.stdout.splitlines() if f.strip() and _is_test_file(f.strip())]
            if names_result.returncode == 0
            else []
        )
    except (subprocess.TimeoutExpired, OSError):
        test_files = []

    diff_cmd = ["git", "-C", repo_path, "diff", "--binary", merge_base_sha, commit_sha]
    if test_files:
        diff_cmd.append("--")
        diff_cmd.extend(f":(exclude){f}" for f in test_files)

    try:
        result = subprocess.run(
            diff_cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"git diff failed: {e}"

    if result.returncode != 0:
        return False, f"git diff exited {result.returncode}: {result.stderr.strip()}"

    patch_content = result.stdout
    if not patch_content.strip():
        return False, f"git diff {merge_base_sha}..{commit_sha} produced an empty patch"

    try:
        with open(os.path.join(solution_dir, "changes.patch"), "w") as f:
            f.write(patch_content)

        solve_sh_path = os.path.join(solution_dir, "solve.sh")
        with open(solve_sh_path, "w") as f:
            f.write(SOLVE_SH_TEMPLATE)
        os.chmod(solve_sh_path, 0o755)
    except OSError as e:
        return False, f"Failed to write solution files: {e}"

    return True, ""


def _find_task_dir_by_id(task_id: str, run_dir: str) -> str | None:
    """Find task directory matching a pre-assigned ID (t2v3-{task_id}-*) in run_dir."""
    if not os.path.isdir(run_dir):
        return None
    for entry in Path(run_dir).iterdir():
        if entry.is_dir() and entry.name.startswith(f"t2v3-{task_id}-"):
            if os.path.isfile(os.path.join(str(entry), "instruction.md")):
                return str(entry)
    return None


def _check_claude_error(result: dict, step_label: str) -> str:
    """Return a non-empty error string if result signals a Claude failure, else empty string.

    Covers two distinct failure modes:
    - "error" key: run_claude_async infrastructure error (timeout, JSON parse, etc.)
    - "is_error": Claude CLI exit signal (max_turns hit, internal error)
    """
    if "error" in result:
        detail = result.get("error_detail", "")
        status = result.get("api_error_status")
        parts = [f"{step_label} Claude error: {result['error']}"]
        if status:
            parts.append(f"status={status}")
        if detail:
            parts.append(detail)
        return " ".join(parts)
    if result.get("is_error"):
        subtype = result.get("subtype", "unknown")
        detail = result.get("error_detail", "")
        status = result.get("api_error_status")
        parts = [f"{step_label} is_error subtype={subtype}"]
        if status:
            parts.append(f"status={status}")
        if detail:
            parts.append(detail)
        return " ".join(parts)
    return ""


def _write_task_toml(task_dir: str, task_id: str, profile: _cfg.PipelineProfile) -> None:
    """Write task.toml from profile values — all resource/timeout fields are profile-controlled."""
    allow = str(profile.task_allow_internet).lower()
    content = f"""version = "1.0"

[metadata]
name = "{task_id}"
difficulty = "hard"

[verifier]
timeout_sec = {profile.task_verifier_timeout}

[agent]
timeout_sec = {profile.task_agent_timeout}

[environment]
build_timeout_sec = {float(profile.task_build_timeout)}
cpus = {profile.task_cpus}
memory_mb = {profile.task_memory_mb}
storage_mb = {profile.task_storage_mb}
gpus = {profile.task_gpus}
allow_internet = {allow}
mcp_servers = []

[environment.env]
ANTHROPIC_API_KEY = "${{ANTHROPIC_API_KEY}}"
ANTHROPIC_BASE_URL = "${{ANTHROPIC_BASE_URL}}"
OPENAI_API_KEY = "${{OPENAI_API_KEY}}"
OPENAI_BASE_URL = "${{OPENAI_BASE_URL}}"

[solution.env]
"""
    with open(os.path.join(task_dir, "task.toml"), "w") as f:
        f.write(content)


# Regex patterns parse the pipeline-generated Dockerfile to populate
# candidate.json fields (python version, github org/repo, clone dir, parent SHA).
# Tolerant of BuildKit platform flags and URLs at end-of-file.
_DOCKERFILE_FROM_RE = re.compile(r"^FROM\s+(?:--\S+\s+)*python:(\d+\.\d+)", re.MULTILINE)
_DOCKERFILE_CLONE_REPO_RE = re.compile(
    r"git\s+clone\b[^\n]*?https://github\.com/([\w.\-]+/[\w.\-]+?)(?:\.git)?(?:[\s\\]|$)",
)
_DOCKERFILE_CLONE_DIR_RE = re.compile(r"git\s+clone\b[^\n]*?https://\S+\s+(/\S+)")
_DOCKERFILE_CHECKOUT_RE = re.compile(r"git\s+checkout\s+([0-9a-f]{7,40})\b")


def _write_candidate_json(task: _cfg.TaskState) -> None:
    """Write candidate.json sidecar in the shape planning adapters consume.

    Called after F2P/P2P classification, when every field is authoritative:
    the Dockerfile has been built (so sha256 is stable) and the test lists
    are populated.
    """
    dockerfile_path = os.path.join(task.task_dir, "environment", "Dockerfile")
    dockerfile_bytes = Path(dockerfile_path).read_bytes()
    dockerfile_text = dockerfile_bytes.decode()

    from_match = _DOCKERFILE_FROM_RE.search(dockerfile_text)
    if not from_match:
        raise ValueError(f"Dockerfile missing 'FROM python:X.Y' line: {dockerfile_path}")
    python_version = from_match.group(1)

    clone_repo_match = _DOCKERFILE_CLONE_REPO_RE.search(dockerfile_text)
    if not clone_repo_match:
        raise ValueError(f"Dockerfile missing 'git clone https://github.com/<repo>': {dockerfile_path}")
    repo_full = clone_repo_match.group(1)

    clone_dir_match = _DOCKERFILE_CLONE_DIR_RE.search(dockerfile_text)
    repo_dir = clone_dir_match.group(1).rstrip("/") if clone_dir_match else "/code"

    checkout_match = _DOCKERFILE_CHECKOUT_RE.search(dockerfile_text)
    if not checkout_match:
        raise ValueError(f"Dockerfile missing 'git checkout <sha>': {dockerfile_path}")
    parent_sha = checkout_match.group(1)

    # abbrev is the 2nd dash-separated segment of the task dir name: t2v3-<ABBREV>-...
    task_dir_name = Path(task.task_dir).name
    parts = task_dir_name.split("-")
    abbrev = parts[1] if len(parts) > 1 else task_dir_name

    instruction_path = os.path.join(task.task_dir, "instruction.md")
    spec = Path(instruction_path).read_text() if os.path.isfile(instruction_path) else ""

    candidate = {
        "task_name": task_dir_name,
        "abbrev": abbrev,
        "repo": repo_full,
        "parent_sha": parent_sha,
        "spec": spec,
        "docker": {
            "mode": "verbatim",
            "python": python_version,
            "repo_dir": repo_dir,
            "source_dockerfile_sha256": hashlib.sha256(dockerfile_bytes).hexdigest(),
        },
        "fail_to_pass": list(task.f2p_tests),
        "pass_to_pass": list(task.p2p_tests),
    }
    with open(os.path.join(task.task_dir, "candidate.json"), "w") as f:
        json.dump(candidate, f, indent=2, sort_keys=True)


def _save_build_diagnostic(run_dir: str, tid: str, result: dict) -> None:
    """Write Claude's raw result to a file so build failures can be diagnosed."""
    path = os.path.join(run_dir, f"build_failed_{tid}.txt")
    try:
        with open(path, "w") as f:
            f.write(f"num_turns: {result.get('num_turns', '?')}\n")
            f.write(f"subtype: {result.get('subtype', '')}\n")
            f.write(f"is_error: {result.get('is_error', False)}\n\n")
            f.write(result.get("result", "")[:8000])
        print(f"    -> Diagnostic saved: {path}")
    except OSError:
        pass


_BUILD_DIFF_BYTE_CAP = 80_000
_BUILD_REPO_MAP_MAX_CHARS = 60_000
_TASK_SLUG_RE = re.compile(r"[^a-z0-9\-]")


def _fetch_build_context(
    repo: str,
    sha: str,
    merge_base_sha: str,
    ctx: PipelineContext,
) -> dict:
    """Pre-assemble all context for the Build judge.

    Returns a dict with repo_map, diff, reference_test_bodies, template
    content, and example content. Truncates diff at 80 KB. Reads test file
    bodies at the commit SHA so the builder sees post-fix tests (what the
    verifier will run against).
    """
    from craft_taskgen.mining.repo_map import build_repo_map

    repo_path = os.path.join("repos", repo)

    # Repo map (aider-style PageRank-ranked tags).
    try:
        repo_map = build_repo_map(repo_path, max_chars=_BUILD_REPO_MAP_MAX_CHARS)
    except Exception as err:
        repo_map = f"[repo_map build failed: {type(err).__name__}: {err}]"

    # Full PR diff (merge_base..sha). Use `git diff merge_base sha` rather
    # than `git show sha` (empty body on merge commits) or `git show A..B`
    # (walks commits individually, not a unified diff).
    try:
        diff = subprocess.check_output(
            ["git", "-C", repo_path, "diff", merge_base_sha, sha],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as err:
        out = (
            err.output.decode("utf-8", errors="replace") if isinstance(err.output, bytes) else str(err.output)
        )
        diff = f"[git diff {merge_base_sha} {sha} failed: {out[:200]}]"

    if len(diff) > _BUILD_DIFF_BYTE_CAP:
        half = _BUILD_DIFF_BYTE_CAP // 2
        omitted_lines = diff.count("\n") - diff[:half].count("\n") - diff[-half:].count("\n")
        omitted_bytes = len(diff) - _BUILD_DIFF_BYTE_CAP
        marker = f"\n\n[...truncated {omitted_lines} lines ({omitted_bytes:,} bytes) omitted...]\n\n"
        diff = diff[:half] + marker + diff[-half:]

    # Reference test file bodies at the commit SHA.
    test_paths = _list_commit_test_files_sync(repo_path, merge_base_sha, sha)
    reference_test_bodies: list[tuple[str, str]] = []
    for rel_path in test_paths:
        try:
            body = subprocess.check_output(
                ["git", "-C", repo_path, "show", f"{sha}:{rel_path}"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            continue
        reference_test_bodies.append((rel_path, body))

    # Instruction template + example at construction time.
    try:
        instruction_template = Path(ctx.instruction_template).read_text()
    except OSError as err:
        instruction_template = f"[template read failed: {err}]"
    try:
        instruction_example = Path(ctx.template_task_dir, "instruction.md").read_text()
    except OSError as err:
        instruction_example = f"[example read failed: {err}]"

    return {
        "repo_map": repo_map,
        "diff": diff,
        "reference_test_bodies": reference_test_bodies,
        "instruction_template": instruction_template,
        "instruction_example": instruction_example,
    }


def _list_commit_test_files_sync(repo_path: str, merge_base_sha: str, sha: str) -> list[str]:
    """Synchronous equivalent of _find_commit_test_files (runs in asyncio.to_thread)."""
    try:
        diff_names = subprocess.check_output(
            ["git", "-C", repo_path, "diff", "--name-only", merge_base_sha, sha],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace")
    except subprocess.CalledProcessError:
        return []
    test_paths: list[str] = []
    for line in diff_names.splitlines():
        line = line.strip()
        if not line or not line.endswith(".py"):
            continue
        parts = line.split("/")
        is_test = any(p == "tests" or p == "test" or p.startswith("test_") for p in parts) or parts[
            -1
        ].startswith("test_")
        if is_test:
            test_paths.append(line)
    return test_paths


def _sanitize_task_slug(raw: str, fallback: str) -> str:
    slug = raw.strip().lower().replace("_", "-").replace(" ", "-")
    slug = _TASK_SLUG_RE.sub("", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:40] if slug else fallback


def _validate_build_output(output: dict, task_dir: str, instruction_template_first_line: str) -> str | None:
    """Return None on success, or a human-readable error message describing
    why the output is invalid — used to feed a regen retry.
    """
    instruction_md = output.get("instruction_md", "")
    if not instruction_md.strip():
        return "instruction_md is empty"
    if not output.get("task_slug", "").strip():
        return "task_slug is empty"
    if instruction_template_first_line and not instruction_md.startswith(
        instruction_template_first_line.strip()
    ):
        return (
            f"instruction_md must start with the template's first line: "
            f"{instruction_template_first_line.strip()!r}"
        )
    word_count = len(instruction_md.split())
    if word_count > rubrics.INSTRUCTION_WORD_HARD_MAX:
        return f"instruction_md is {word_count} words; hard max is {rubrics.INSTRUCTION_WORD_HARD_MAX}"
    return None


@dataclass
class _BuildOutcome:
    """Pure result of a single Build attempt — no task mutation, no state save.

    The wrapper (`_run_build_one` or `_one_candidate_loop`) translates this
    into TaskState mutations + stage transitions. Outcome discriminator:

      ok              — instruction.md written; task_dir + usage populated
      context_fail    — _fetch_build_context raised; route to EVALUATED
      judge_fail      — llm_judge.judge raised on both attempts; route to EVALUATED
      validation_fail — both attempts produced unparseable/invalid output; route to NEEDS_FIX
    """

    outcome: str
    task_dir: str = ""
    instruction_md: str = ""
    instruction_words: int = 0
    slug: str = ""
    usage_entry: dict = field(default_factory=dict)
    iteration_entry: dict = field(default_factory=dict)
    error_message: str = ""


async def _build_instruction(
    task,
    run_dir: str,
    *,
    feedback: str,
    cand_id: int | None,
    log_prefix: str = "build",
) -> _BuildOutcome:
    """Pure Build core. Does not mutate `task` or save state.

    Performs context fetch + LLM judge call (with one validation regen),
    writes instruction.md to either `t2v3-{tid}-{slug}` (cand_id=None,
    legacy single-build path) or `t2v3-{tid}-cand{cand_id}` (fanout path).
    Returns a _BuildOutcome the caller translates into task mutations.

    `feedback` replaces what used to be `task.alignment_feedback` — pass
    `""` for fresh build, the alignment-feedback string for a regen.
    """
    os.makedirs(run_dir, exist_ok=True)
    tid = _generate_task_id(task.repo, task.commit_sha)
    print(f"  [{log_prefix}] {task.repo}/{task.commit_sha[:8]} (ID: {tid})")

    ctx = PipelineContext()
    try:
        context = await asyncio.to_thread(
            _fetch_build_context,
            task.repo,
            task.commit_sha,
            task.merge_base_sha,
            ctx,
        )
    except Exception as err:
        msg = f"build context assembly: {err}"
        print(f"    ERROR: build context assembly failed: {type(err).__name__}: {err}")
        return _BuildOutcome(outcome="context_fail", error_message=msg)

    prompt = build_task_prompt(
        repo=task.repo,
        sha=task.commit_sha,
        merge_base_sha=task.merge_base_sha,
        subject=task.description,
        eval_reason=task.eval_reason,
        instruction_sketch=task.eval_instruction_sketch,
        repo_map=context["repo_map"],
        diff=context["diff"],
        reference_test_bodies=context["reference_test_bodies"],
        instruction_template=context["instruction_template"],
        instruction_example=context["instruction_example"],
        alignment_feedback=feedback,
    )

    template_lines = context["instruction_template"].splitlines()
    template_first_line = template_lines[0] if template_lines else ""

    output: dict | None = None
    validation_error: str | None = None
    last_err: Exception | None = None
    judge_result = None

    for attempt in range(2):  # one initial try, one regen on failure
        try:
            judge_result = await llm_judge.judge(
                prompt=prompt,
                schema=BUILD_SCHEMA,
                model=_cfg.LLM_STEP_MODEL,
                system_prompt=(
                    f"The run directory is: {run_dir}. You must return a single JSON "
                    f"object per the schema. Do not attempt to write files — the "
                    f"pipeline writes instruction.md from your response."
                )
                if attempt == 0
                else (
                    f"The run directory is: {run_dir}. Previous response validation "
                    f"failed: {validation_error}. Regenerate and fix that specific "
                    f"problem. Return only the JSON object per the schema."
                ),
            )
        except Exception as err:
            last_err = err
            break
        output = judge_result.result
        validation_error = _validate_build_output(
            output, task_dir="", instruction_template_first_line=template_first_line
        )
        if validation_error is None:
            break
        print(f"    -> validation failed (attempt {attempt + 1}/2): {validation_error}")

    if last_err is not None:
        msg = f"build judge: {last_err}"
        print(f"    ERROR: {type(last_err).__name__}: {last_err}")
        return _BuildOutcome(outcome="judge_fail", error_message=msg)

    if validation_error is not None:
        msg = f"build validation failed after regen: {validation_error}"
        print(f"    -> NEEDS_FIX: {validation_error}")
        return _BuildOutcome(
            outcome="validation_fail",
            error_message=msg,
            iteration_entry={
                "timestamp": datetime.now().isoformat(),
                "step": "build_failed",
                "reason": validation_error,
            },
        )

    assert output is not None and judge_result is not None
    slug = _sanitize_task_slug(
        output.get("task_slug", ""),
        fallback=re.sub(r"[^a-z0-9]+", "-", task.description.lower())[:30].strip("-") or "task",
    )
    if cand_id is None:
        task_dir = os.path.join(run_dir, f"t2v3-{tid}-{slug}")
    else:
        task_dir = os.path.join(run_dir, f"t2v3-{tid}-cand{cand_id}")
    os.makedirs(task_dir, exist_ok=True)
    instruction_path = os.path.join(task_dir, "instruction.md")
    with open(instruction_path, "w") as f:
        f.write(output["instruction_md"])
    strip_instruction_boilerplate(instruction_path)
    with open(instruction_path) as f:
        instruction_md = f.read()
    instruction_words = len(instruction_md.split())

    usage_entry = {
        "tokens_in": judge_result.usage.get("input_tokens", 0),
        "tokens_out": judge_result.usage.get("output_tokens", 0),
        "tokens_cached": judge_result.usage.get("cached_tokens", 0),
        "model": judge_result.model,
        "latency_s": round(judge_result.latency_s, 3),
    }
    iteration_entry = {
        "timestamp": datetime.now().isoformat(),
        "step": "build",
        "task_dir": task_dir,
        "instruction_words": instruction_words,
    }
    print(f"    -> BUILT: {task_dir} ({instruction_words} words)")
    return _BuildOutcome(
        outcome="ok",
        task_dir=task_dir,
        instruction_md=instruction_md,
        instruction_words=instruction_words,
        slug=slug,
        usage_entry=usage_entry,
        iteration_entry=iteration_entry,
    )


async def _run_build_one(task, state: PipelineState, state_file: str) -> None:
    """Build a single task package via direct-API judge.

    Thin wrapper around `_build_instruction` (the pure core). Translates
    the outcome into TaskState mutations + stage transitions. One regen
    on parse/schema/validation failure (handled inside the core); if
    the regen also fails, the task is shelved NEEDS_FIX. No deeper fix
    loop — upstream stages (alignment, reviewer) have their own gates.
    """
    async with _mark_in_progress(task, "build", state, state_file):
        outcome = await _build_instruction(
            task,
            state.run_dir,
            feedback=task.alignment_feedback,
            cand_id=None,
        )

        if outcome.outcome == "context_fail":
            task.stage = Stage.EVALUATED
            task.needs_human_review = True
            task.human_review_reason = outcome.error_message
            await save_state_locked(state, state_file)
            return
        if outcome.outcome == "judge_fail":
            task.stage = Stage.EVALUATED
            task.needs_human_review = True
            task.human_review_reason = outcome.error_message
            await save_state_locked(state, state_file)
            return
        if outcome.outcome == "validation_fail":
            task.stage = Stage.NEEDS_FIX
            task.needs_human_review = True
            task.human_review_reason = outcome.error_message
            if outcome.iteration_entry:
                task.iteration_log.append(outcome.iteration_entry)
            await save_state_locked(state, state_file)
            return

        # Success
        task.task_dir = outcome.task_dir
        task.instruction_words = outcome.instruction_words
        task.llm_usage.setdefault("build", []).append(outcome.usage_entry)
        task.stage = Stage.BUILT
        task.iteration_log.append(outcome.iteration_entry)
        await save_state_locked(state, state_file)
        return


async def step_build(state: PipelineState, state_file: str, concurrency: int = 4) -> None:
    """Combined Build + Alignment step for PROMISING candidates.

    Thin wrapper for ``--from-step build`` resume. Delegates to the
    parallel-candidate orchestrator (``_run_build_align_candidates``),
    which runs N concurrent build+alignment loops per task and selects
    a passing winner. Tasks emerge at ``ALIGNMENT_CHECKED`` (success)
    or ``REJECTED`` / ``NEEDS_FIX`` (terminal failure).

    Note: the ``concurrency`` parameter is retained for backwards
    compatibility with the ``--from-step`` CLI but is ignored — sem
    sizing comes from ``_cfg.LLM_CONCURRENCY`` / ``_cfg.DOCKER_CONCURRENCY``
    / ``_cfg.SMOKE_CONCURRENCY`` so this wrapper matches the main
    pipeline path in pipeline.py.
    """
    from craft_taskgen.config import MAX_PROMISING_PER_REPO

    del concurrency  # ignored — see docstring; sems sized from _cfg

    all_promising = [t for t in state.tasks.values() if t.stage == Stage.PROMISING]
    repo_counts: dict[str, int] = {}
    promising = []
    for t in all_promising:
        count = repo_counts.get(t.repo, 0)
        if count < MAX_PROMISING_PER_REPO:
            promising.append(t)
            repo_counts[t.repo] = count + 1
        else:
            t.stage = Stage.REJECTED
            t.eval_reason = f"Skipped: repo {t.repo} already has {MAX_PROMISING_PER_REPO} candidates"
            print(f"  Skipping {t.task_id} (repo cap: {t.repo} has {MAX_PROMISING_PER_REPO} already)")
    if not promising:
        print("No PROMISING candidates to build.")
        return

    # Match pipeline.py:282-294 exactly so the wrapper and full-pipeline
    # paths share identical sem semantics.
    candidate_sem_size = max(_cfg.LLM_CONCURRENCY, _cfg.LLM_CONCURRENCY * 2)
    if candidate_sem_size < _cfg.BUILD_N_CANDIDATES:
        candidate_sem_size = _cfg.BUILD_N_CANDIDATES
    sems = {
        "llm": asyncio.Semaphore(_cfg.LLM_CONCURRENCY),
        "docker": asyncio.Semaphore(_cfg.DOCKER_CONCURRENCY),
        "smoke": asyncio.Semaphore(_cfg.SMOKE_CONCURRENCY),
        "candidate": asyncio.Semaphore(candidate_sem_size),
    }

    print(
        f"Build+Alignment for {len(promising)} tasks "
        f"(llm={_cfg.LLM_CONCURRENCY}, candidate={candidate_sem_size}, N={_cfg.BUILD_N_CANDIDATES})..."
    )
    await asyncio.gather(*[_run_build_align_candidates(t, state, state_file, sems) for t in promising])


# ---------------------------------------------------------------------------
# Step 4a: Alignment judge (cross-family audit of instruction ↔ tests)
# ---------------------------------------------------------------------------


def _fetch_alignment_context(
    task_dir: str,
    repo: str,
    sha: str,
    merge_base_sha: str,
) -> dict:
    """Pre-assemble instruction.md text + reference test bodies + PR diff.

    Alignment judge sees the whole picture in one prompt: what Build produced,
    what tests will verify, and what the PR actually changed. Uses the same
    git-diff-over-git-show fix so merge commits produce real unified diffs.
    """
    instruction_path = os.path.join(task_dir, "instruction.md")
    try:
        instruction_md = Path(instruction_path).read_text()
    except OSError as err:
        instruction_md = f"[instruction.md read failed: {err}]"

    repo_path = os.path.join("repos", repo)
    test_paths = _list_commit_test_files_sync(repo_path, merge_base_sha, sha)
    reference_test_bodies: list[tuple[str, str]] = []
    for rel_path in test_paths:
        try:
            body = subprocess.check_output(
                ["git", "-C", repo_path, "show", f"{sha}:{rel_path}"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            continue
        reference_test_bodies.append((rel_path, body))

    try:
        diff = subprocess.check_output(
            ["git", "-C", repo_path, "diff", merge_base_sha, sha],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as err:
        out = (
            err.output.decode("utf-8", errors="replace") if isinstance(err.output, bytes) else str(err.output)
        )
        diff = f"[git diff {merge_base_sha} {sha} failed: {out[:200]}]"

    if len(diff) > _BUILD_DIFF_BYTE_CAP:
        half = _BUILD_DIFF_BYTE_CAP // 2
        omitted_lines = diff.count("\n") - diff[:half].count("\n") - diff[-half:].count("\n")
        omitted_bytes = len(diff) - _BUILD_DIFF_BYTE_CAP
        marker = f"\n\n[...truncated {omitted_lines} lines ({omitted_bytes:,} bytes) omitted...]\n\n"
        diff = diff[:half] + marker + diff[-half:]

    return {
        "instruction_md": instruction_md,
        "reference_test_bodies": reference_test_bodies,
        "diff": diff,
    }


@dataclass
class _AlignmentRetryResult:
    """Pure result of an alignment retention-retry pass — no task mutation, no save.

    The wrapper translates this into TaskState mutations + stage transitions.
    `context_error` is non-None iff `_fetch_alignment_context` raised; in
    that case `attempts`/`accept_result`/`usage_entries` are empty.
    """

    attempts: list[dict] = field(default_factory=list)
    usage_entries: list[dict] = field(default_factory=list)
    accept_result: dict | None = None
    context_error: str = ""


async def _run_alignment_retry(
    task_dir: str,
    repo: str,
    commit_sha: str,
    merge_base_sha: str,
    *,
    log_prefix: str = "alignment",
) -> _AlignmentRetryResult:
    """Pure alignment retention-retry core. Does not mutate any task or save state.

    Fetches context (instruction.md + reference test bodies + PR diff) then
    runs up to `_cfg.ALIGNMENT_MAX_RETRIES` independent judge calls. Accepts
    on the first `verdict == "ok"`. Returns the full attempt log + cumulative
    usage entries; the caller decides how to translate to task state.
    """
    from craft_taskgen.prompts import ALIGNMENT_SCHEMA, alignment_judge_prompt

    print(f"  [{log_prefix}] {task_dir}")

    try:
        context = await asyncio.to_thread(
            _fetch_alignment_context,
            task_dir,
            repo,
            commit_sha,
            merge_base_sha,
        )
    except Exception as err:
        msg = f"alignment context assembly: {err}"
        print(f"    ERROR: alignment context assembly failed: {type(err).__name__}: {err}")
        return _AlignmentRetryResult(context_error=msg)

    prompt = alignment_judge_prompt(
        instruction_md=context["instruction_md"],
        reference_test_bodies=context["reference_test_bodies"],
        diff=context["diff"],
    )

    attempts: list[dict] = []
    usage_entries: list[dict] = []
    accept_result: dict | None = None

    for attempt_idx in range(_cfg.ALIGNMENT_MAX_RETRIES):
        try:
            judge_result = await llm_judge.judge(
                prompt=prompt,
                schema=ALIGNMENT_SCHEMA,
                model=_cfg.LLM_ALIGNMENT_MODEL,
            )
        except Exception as err:
            print(f"    ERROR on attempt {attempt_idx + 1}: {type(err).__name__}: {err}")
            attempts.append(
                {
                    "attempt": attempt_idx + 1,
                    "error": f"{type(err).__name__}: {err}",
                }
            )
            continue

        verdict = judge_result.result.get("verdict", "")
        reason = judge_result.result.get("reason", "")
        v4_audit = judge_result.result.get("v4_audit", {}) or {}
        leakage_evidence = judge_result.result.get("leakage_evidence", []) or []

        attempts.append(
            {
                "attempt": attempt_idx + 1,
                "verdict": verdict,
                "reason": reason,
                "v4_audit": v4_audit,
                "leakage_evidence": leakage_evidence,
                "tokens_in": judge_result.usage.get("input_tokens", 0),
                "tokens_out": judge_result.usage.get("output_tokens", 0),
                "latency_s": round(judge_result.latency_s, 3),
                "model": judge_result.model,
            }
        )
        usage_entries.append(
            {
                "tokens_in": judge_result.usage.get("input_tokens", 0),
                "tokens_out": judge_result.usage.get("output_tokens", 0),
                "tokens_cached": judge_result.usage.get("cached_tokens", 0),
                "model": judge_result.model,
                "latency_s": round(judge_result.latency_s, 3),
            }
        )
        if verdict == "ok":
            accept_result = judge_result.result
            break

    return _AlignmentRetryResult(
        attempts=attempts,
        usage_entries=usage_entries,
        accept_result=accept_result,
    )


async def _run_alignment_one(task, state: PipelineState, state_file: str) -> None:
    """Cross-family alignment judge: audit instruction ↔ reference-test alignment.

    Thin wrapper around `_run_alignment_retry` (the pure core). Translates
    the result into TaskState mutations + stage transitions.

    Runs after Build, before assemble_task_dir_artifacts. Uses a different
    model family than Build (Opus → Build; GPT-5.4 → alignment) to mitigate
    self-preference bias when judging output.

    Retention-biased retry: up to 3 independent attempts. Keep if any
    returns `ok`. Otherwise reject — or, if `leaked`/`narrow_tests` on any
    attempt and `alignment_regen_count == 0`, route back to Build with
    feedback for one regen.
    """
    async with _mark_in_progress(task, "alignment", state, state_file):
        result = await _run_alignment_retry(
            task.task_dir,
            task.repo,
            task.commit_sha,
            task.merge_base_sha,
        )

        if result.context_error:
            task.needs_human_review = True
            task.human_review_reason = result.context_error
            task.stage = Stage.NEEDS_FIX
            await save_state_locked(state, state_file)
            return

        attempts = result.attempts
        accept_result = result.accept_result

        for usage_entry in result.usage_entries:
            task.llm_usage.setdefault("alignment", []).append(usage_entry)

        task.alignment_attempts = attempts

        if accept_result is not None:
            task.alignment_verdict = "ok"
            task.alignment_reason = accept_result.get("reason", "")
            task.alignment_v4_audit = accept_result.get("v4_audit", {}) or {}
            task.alignment_feedback = ""  # clear any stale feedback after acceptance
            task.stage = Stage.ALIGNMENT_CHECKED
            attempts_count = len(attempts)
            print(f"    -> OK (attempt {attempts_count}/{_cfg.ALIGNMENT_MAX_RETRIES})")
        else:
            verdicts = [a.get("verdict", a.get("error", "?")) for a in attempts]
            task.alignment_verdict = attempts[-1].get("verdict", "error") if attempts else "error"
            task.alignment_reason = "; ".join(
                f"[{a.get('attempt')}] {a.get('verdict', 'err')}: {a.get('reason', a.get('error', ''))[:200]}"
                for a in attempts
            )
            task.alignment_v4_audit = attempts[-1].get("v4_audit", {}) if attempts else {}

            # Bounded feedback loop: if alignment rejected with `leaked` or
            # `narrow_tests` on any attempt AND we haven't used our regen
            # budget (1 per task), route back to Build with the evidence.
            # Otherwise final-reject. `vague`/`misaligned` don't trigger
            # regen — those usually mean Build's output is structurally
            # broken, not fixable by naming-fewer-things.
            # If every attempt ended in an exception (gateway timeout,
            # auth failure, parse error), there's no real alignment
            # verdict — this is an infra problem, not a task problem.
            # Route to NEEDS_FIX for retry/review rather than REJECTED,
            # matching the build/evaluate paths' handling of infra
            # exceptions.
            all_errors = attempts and all(a.get("error") for a in attempts)
            actionable = any(a.get("verdict") in ("leaked", "narrow_tests") for a in attempts)
            if all_errors:
                task.stage = Stage.NEEDS_FIX
                task.needs_human_review = True
                task.human_review_reason = f"alignment judge: all attempts failed ({verdicts})"
                print(f"    -> NEEDS_FIX (alignment infra errors: {verdicts})")
            elif actionable and task.alignment_regen_count == 0:
                task.alignment_feedback = _build_alignment_feedback(attempts)
                task.alignment_regen_count = 1
                task.stage = Stage.PROMISING  # re-enter Build
                print(f"    -> REGEN (feedback: {len(task.alignment_feedback)} chars, verdicts: {verdicts})")
            else:
                task.stage = Stage.REJECTED
                print(f"    -> REJECT (all attempts non-ok: {verdicts})")

        task.iteration_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": "alignment",
                "verdict": task.alignment_verdict,
                "attempts": len(attempts),
            }
        )

        await save_state_locked(state, state_file)


@dataclass
class CandidateResult:
    """Pure result of one independent build → align → (regen) → align loop.

    The orchestrator translates this into TaskState mutations after winner
    selection. `outcome` discriminator:

      pass       — alignment accepted; instruction.md ready in `task_dir`
      reject     — alignment rejected non-actionably (vague/misaligned) or
                   leaked/narrow_tests after regen; not a candidate for winning
      needs_fix  — infrastructure failure (build context, judge exception,
                   alignment context, or all-error retry); needs human review
    """

    cand_id: int
    outcome: str
    task_dir: str = ""
    instruction_md: str = ""
    instruction_words: int = 0
    slug: str = ""
    build_usage: list[dict] = field(default_factory=list)
    alignment_usage: list[dict] = field(default_factory=list)
    alignment_attempts: list[dict] = field(default_factory=list)
    alignment_feedback: str = ""
    alignment_regen_count: int = 0
    alignment_verdict: str = ""
    alignment_reason: str = ""
    alignment_v4_audit: dict = field(default_factory=dict)
    iteration_log: list[dict] = field(default_factory=list)
    needs_human_review_reason: str = ""


def _build_outcome_to_candidate_outcome(build_outcome_kind: str) -> str:
    """Map a _BuildOutcome.outcome to a CandidateResult.outcome."""
    if build_outcome_kind == "ok":
        return "pass"
    # context_fail / judge_fail / validation_fail are all infra-ish; route to needs_fix.
    return "needs_fix"


def _summarize_alignment_reason(attempts: list[dict]) -> str:
    return "; ".join(
        f"[{a.get('attempt')}] {a.get('verdict', 'err')}: {a.get('reason', a.get('error', ''))[:200]}"
        for a in attempts
    )


def _is_actionable(attempts: list[dict]) -> bool:
    """True if any attempt's verdict is leaked or narrow_tests (regen-eligible)."""
    return any(a.get("verdict") in ("leaked", "narrow_tests") for a in attempts)


def _all_errors(attempts: list[dict]) -> bool:
    """True if every attempt errored (no real verdict obtained)."""
    return bool(attempts) and all(a.get("error") for a in attempts)


async def _one_candidate_loop(
    cand_id: int,
    task,
    run_dir: str,
    candidate_sem: asyncio.Semaphore,
) -> CandidateResult:
    """Run one independent build → alignment → (rebuild on actionable rejection) loop.

    Pure: does not mutate `task` or save state. Returns a CandidateResult
    that the orchestrator translates into task mutations after winner
    selection.

    Each candidate writes to ``f"{run_dir}/t2v3-{tid}-cand{cand_id}"``. The
    rebuild loop runs up to ``_cfg.MAX_BUILD_REGENS_PER_CANDIDATE`` times,
    each rebuild overwriting the same cand-dir's instruction.md and using
    cumulative leakage feedback in the prompt. On winner promotion, the
    cand-dir is renamed to the canonical slug-based path.
    """
    log_prefix = f"build cand{cand_id}"
    align_log_prefix = f"align cand{cand_id}"

    async with candidate_sem:
        # --- Initial build ---
        build = await _build_instruction(task, run_dir, feedback="", cand_id=cand_id, log_prefix=log_prefix)
        if build.outcome != "ok":
            return CandidateResult(
                cand_id=cand_id,
                outcome=_build_outcome_to_candidate_outcome(build.outcome),
                build_usage=[build.usage_entry] if build.usage_entry else [],
                iteration_log=[build.iteration_entry] if build.iteration_entry else [],
                needs_human_review_reason=build.error_message,
            )

        task_dir = build.task_dir
        instruction_md = build.instruction_md
        instruction_words = build.instruction_words
        slug = build.slug
        build_usage = [build.usage_entry]
        iteration_log = [build.iteration_entry]

        attempts: list[dict] = []  # cumulative across all alignment rounds
        alignment_usage: list[dict] = []
        feedback = ""
        regen_count = 0

        # Initial alignment evaluation
        align_result = await _run_alignment_retry(
            task_dir,
            task.repo,
            task.commit_sha,
            task.merge_base_sha,
            log_prefix=align_log_prefix,
        )

        # Rebuild loop: continues while assessor returns actionable rejection
        # AND we have rebuild budget. Each iteration's body merges this
        # round's attempts, decides whether to rebuild, and re-aligns.
        while True:
            if align_result.context_error:
                return CandidateResult(
                    cand_id=cand_id,
                    outcome="needs_fix",
                    task_dir=task_dir,
                    instruction_md=instruction_md,
                    instruction_words=instruction_words,
                    slug=slug,
                    build_usage=build_usage,
                    alignment_usage=alignment_usage,
                    alignment_attempts=attempts,
                    alignment_feedback=feedback,
                    alignment_regen_count=regen_count,
                    iteration_log=iteration_log,
                    needs_human_review_reason=align_result.context_error,
                )

            # Merge this round's attempts into running list (offset numbering)
            offset = len(attempts)
            for a in align_result.attempts:
                a["attempt"] = a["attempt"] + offset
            attempts.extend(align_result.attempts)
            alignment_usage.extend(align_result.usage_entries)

            # Pass — break out, emit pass below
            if align_result.accept_result is not None:
                accept_result = align_result.accept_result
                break

            # Rejection — decide whether to rebuild or terminate
            this_round = align_result.attempts
            if _all_errors(this_round):
                verdicts = [a.get("verdict", a.get("error", "?")) for a in this_round]
                print(f"    -> [{align_log_prefix}] NEEDS_FIX (all attempts errored: {verdicts})")
                return CandidateResult(
                    cand_id=cand_id,
                    outcome="needs_fix",
                    task_dir=task_dir,
                    instruction_md=instruction_md,
                    instruction_words=instruction_words,
                    slug=slug,
                    build_usage=build_usage,
                    alignment_usage=alignment_usage,
                    alignment_attempts=attempts,
                    alignment_feedback=feedback,
                    alignment_regen_count=regen_count,
                    alignment_verdict=attempts[-1].get("verdict", "error") if attempts else "error",
                    alignment_reason=_summarize_alignment_reason(attempts),
                    alignment_v4_audit=attempts[-1].get("v4_audit", {}) if attempts else {},
                    iteration_log=iteration_log,
                    needs_human_review_reason=f"alignment judge: all attempts failed ({verdicts})",
                )

            if not _is_actionable(this_round):
                verdicts = [a.get("verdict", a.get("error", "?")) for a in this_round]
                print(f"    -> [{align_log_prefix}] REJECT (non-actionable: {verdicts})")
                return CandidateResult(
                    cand_id=cand_id,
                    outcome="reject",
                    task_dir=task_dir,
                    instruction_md=instruction_md,
                    instruction_words=instruction_words,
                    slug=slug,
                    build_usage=build_usage,
                    alignment_usage=alignment_usage,
                    alignment_attempts=attempts,
                    alignment_feedback=feedback,
                    alignment_regen_count=regen_count,
                    alignment_verdict=attempts[-1].get("verdict", "error"),
                    alignment_reason=_summarize_alignment_reason(attempts),
                    alignment_v4_audit=attempts[-1].get("v4_audit", {}),
                    iteration_log=iteration_log,
                )

            if regen_count >= _cfg.MAX_BUILD_REGENS_PER_CANDIDATE:
                verdicts = [a.get("verdict", a.get("error", "?")) for a in this_round]
                print(
                    f"    -> [{align_log_prefix}] REJECT (rebuild budget exhausted "
                    f"at r={regen_count}: {verdicts})"
                )
                return CandidateResult(
                    cand_id=cand_id,
                    outcome="reject",
                    task_dir=task_dir,
                    instruction_md=instruction_md,
                    instruction_words=instruction_words,
                    slug=slug,
                    build_usage=build_usage,
                    alignment_usage=alignment_usage,
                    alignment_attempts=attempts,
                    alignment_feedback=feedback,
                    alignment_regen_count=regen_count,
                    alignment_verdict=attempts[-1].get("verdict", "error"),
                    alignment_reason=_summarize_alignment_reason(attempts),
                    alignment_v4_audit=attempts[-1].get("v4_audit", {}),
                    iteration_log=iteration_log,
                )

            # Actionable rejection + budget left: rebuild with cumulative feedback
            feedback = _build_alignment_feedback(attempts)
            regen_count += 1
            verdicts = [a.get("verdict", a.get("error", "?")) for a in this_round]
            print(
                f"    -> [{align_log_prefix}] REGEN #{regen_count} "
                f"(feedback: {len(feedback)} chars, verdicts: {verdicts})"
            )
            rebuild = await _build_instruction(
                task,
                run_dir,
                feedback=feedback,
                cand_id=cand_id,
                log_prefix=f"{log_prefix} regen{regen_count}",
            )
            if rebuild.outcome != "ok":
                return CandidateResult(
                    cand_id=cand_id,
                    outcome=_build_outcome_to_candidate_outcome(rebuild.outcome),
                    task_dir=task_dir,
                    instruction_md=instruction_md,
                    instruction_words=instruction_words,
                    slug=slug,
                    build_usage=build_usage + ([rebuild.usage_entry] if rebuild.usage_entry else []),
                    alignment_usage=alignment_usage,
                    alignment_attempts=attempts,
                    alignment_feedback=feedback,
                    alignment_regen_count=regen_count,
                    alignment_verdict=attempts[-1].get("verdict", "error"),
                    alignment_reason=_summarize_alignment_reason(attempts),
                    alignment_v4_audit=attempts[-1].get("v4_audit", {}),
                    iteration_log=iteration_log
                    + ([rebuild.iteration_entry] if rebuild.iteration_entry else []),
                    needs_human_review_reason=rebuild.error_message,
                )

            build_usage.append(rebuild.usage_entry)
            iteration_log.append(rebuild.iteration_entry)
            task_dir = rebuild.task_dir
            instruction_md = rebuild.instruction_md
            instruction_words = rebuild.instruction_words
            slug = rebuild.slug

            # Re-align on the new instruction (loop continues)
            align_result = await _run_alignment_retry(
                task_dir,
                task.repo,
                task.commit_sha,
                task.merge_base_sha,
                log_prefix=f"{align_log_prefix} regen{regen_count}",
            )

        # --- Out of loop: accept_result is set (passed) ---
        label = "OK" if regen_count == 0 else f"OK after regen #{regen_count}"
        print(f"    -> [{align_log_prefix}] {label} (attempt {len(attempts)})")
        return CandidateResult(
            cand_id=cand_id,
            outcome="pass",
            task_dir=task_dir,
            instruction_md=instruction_md,
            instruction_words=instruction_words,
            slug=slug,
            build_usage=build_usage,
            alignment_usage=alignment_usage,
            alignment_attempts=attempts,
            alignment_feedback="",
            alignment_regen_count=regen_count,
            alignment_verdict="ok",
            alignment_reason=accept_result.get("reason", ""),
            alignment_v4_audit=accept_result.get("v4_audit", {}) or {},
            iteration_log=iteration_log
            + [
                {
                    "timestamp": datetime.now().isoformat(),
                    "step": "alignment",
                    "verdict": "ok",
                    "attempts": len(attempts),
                    "cand_id": cand_id,
                    "regen_count": regen_count,
                }
            ],
        )


def _select_winner(passers: list[CandidateResult]) -> CandidateResult:
    """Pick a winner among passing candidates.

    Today: uniform random. Named as a separate helper so the policy is
    swappable later (e.g., to "shortest instruction wins" if calibration
    shows shorter instructions leak less). No config knob today.
    """
    import random

    return random.choice(passers)


def _candidate_loser_summary(result: CandidateResult) -> dict:
    """Compact dict for `task.build_align_losers` — audit-only, ≤500 chars reason."""
    return {
        "cand_id": result.cand_id,
        "outcome": result.outcome,
        "verdict": result.alignment_verdict,
        "short_reason": (result.alignment_reason or "")[:500],
        "regen_count": result.alignment_regen_count,
        "instruction_words": result.instruction_words,
    }


def _cleanup_loser_dirs(losers: list[CandidateResult]) -> None:
    """Best-effort `shutil.rmtree` for each non-winner candidate dir."""
    import shutil

    for r in losers:
        if r.task_dir and os.path.isdir(r.task_dir):
            shutil.rmtree(r.task_dir, ignore_errors=True)


def _cleanup_orphan_cand_dirs(run_dir: str, tid: str) -> None:
    """Defensive cleanup at task entry: remove any stray cand{N} dirs from a prior crash."""
    import shutil

    if not os.path.isdir(run_dir):
        return
    pattern_prefix = f"t2v3-{tid}-cand"
    for entry in os.listdir(run_dir):
        if entry.startswith(pattern_prefix):
            full = os.path.join(run_dir, entry)
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)


def _promote_winner_to_task(task, winner: CandidateResult, run_dir: str, tid: str) -> str:
    """Rename the winner's cand-dir to its canonical slug-based path and return new path.

    Side-effect: ``os.rename`` on disk. Caller is responsible for setting
    ``task.task_dir`` and other fields.
    """
    canonical = os.path.join(run_dir, f"t2v3-{tid}-{winner.slug}")
    if winner.task_dir != canonical:
        # If the canonical path already exists from a prior partial run,
        # remove it before rename so os.rename succeeds atomically.
        if os.path.isdir(canonical):
            import shutil

            shutil.rmtree(canonical, ignore_errors=True)
        os.rename(winner.task_dir, canonical)
    return canonical


async def _run_build_align_candidates(
    task,
    state: PipelineState,
    state_file: str,
    sems: dict,
) -> None:
    """Orchestrate N parallel build+alignment candidate loops, then promote a winner.

    Fresh-entry path (Stage.PROMISING / Stage.EVALUATED). Spawns
    ``_cfg.BUILD_N_CANDIDATES`` independent candidates concurrently; each
    runs build → align → (regen on leaked/narrow_tests) → re-align. After
    all candidates finish, picks a passing one uniformly at random,
    promotes its dir to the canonical slug-based path, records loser
    summaries, and cleans up loser dirs.

    Stage transitions:
      pass → ALIGNMENT_CHECKED
      no passers, all needs_fix → NEEDS_FIX (needs_human_review=True)
      no passers, mixed/reject → REJECTED
    """
    async with _mark_in_progress(task, "build_align", state, state_file):
        run_dir = state.run_dir
        os.makedirs(run_dir, exist_ok=True)
        tid = _generate_task_id(task.repo, task.commit_sha)
        _cleanup_orphan_cand_dirs(run_dir, tid)

        n = max(1, _cfg.BUILD_N_CANDIDATES)
        candidate_sem = sems["candidate"]
        print(f"  [build_align] {task.repo}/{task.commit_sha[:8]} (ID: {tid}, N={n})")

        gathered = await asyncio.gather(
            *[_one_candidate_loop(i, task, run_dir, candidate_sem) for i in range(n)],
            return_exceptions=True,
        )

        # Coerce exceptions into synthetic needs_fix results
        results: list[CandidateResult] = []
        for i, item in enumerate(gathered):
            if isinstance(item, BaseException):
                print(f"    ERROR cand{i}: {type(item).__name__}: {item}")
                results.append(
                    CandidateResult(
                        cand_id=i,
                        outcome="needs_fix",
                        needs_human_review_reason=f"cand{i} crashed: {type(item).__name__}: {item}",
                    )
                )
            else:
                results.append(item)

        passers = [r for r in results if r.outcome == "pass"]
        if passers:
            winner = _select_winner(passers)
            losers = [r for r in results if r is not winner]

            canonical = _promote_winner_to_task(task, winner, run_dir, tid)

            # Apply winner fields to task
            task.task_dir = canonical
            task.instruction_words = winner.instruction_words
            for usage in winner.build_usage:
                task.llm_usage.setdefault("build", []).append(usage)
            for usage in winner.alignment_usage:
                task.llm_usage.setdefault("alignment", []).append(usage)
            # Loser usage entries also count toward cost reporting
            for r in losers:
                for usage in r.build_usage:
                    task.llm_usage.setdefault("build", []).append(usage)
                for usage in r.alignment_usage:
                    task.llm_usage.setdefault("alignment", []).append(usage)
            task.alignment_attempts = winner.alignment_attempts
            task.alignment_feedback = ""
            task.alignment_regen_count = winner.alignment_regen_count
            task.alignment_verdict = "ok"
            task.alignment_reason = winner.alignment_reason
            task.alignment_v4_audit = winner.alignment_v4_audit
            task.build_align_losers = [_candidate_loser_summary(r) for r in losers]
            for entry in winner.iteration_log:
                task.iteration_log.append(entry)
            task.iteration_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "step": "build_align",
                    "n_candidates": n,
                    "n_passed": len(passers),
                    "selected_cand_id": winner.cand_id,
                    "verdict": "ok",
                }
            )
            task.stage = Stage.ALIGNMENT_CHECKED
            print(
                f"    -> ALIGNMENT_CHECKED (winner=cand{winner.cand_id}, "
                f"passers={len(passers)}/{n}, regen={winner.alignment_regen_count})"
            )

            _cleanup_loser_dirs(losers)
            await save_state_locked(state, state_file)
            return

        # No passers — terminal failure. Distinguish needs_fix vs reject.
        for r in results:
            for usage in r.build_usage:
                task.llm_usage.setdefault("build", []).append(usage)
            for usage in r.alignment_usage:
                task.llm_usage.setdefault("alignment", []).append(usage)
        task.build_align_losers = [_candidate_loser_summary(r) for r in results]

        # Telemetry parity with the success path: surface per-candidate
        # alignment attempts, a representative verdict, and the v4_audit
        # so dashboards / status / post-hoc analysis can see why this
        # task failed alignment without having to re-derive from
        # build_align_losers + reason text.
        aggregated_attempts: list[dict] = []
        for r in results:
            for a in r.alignment_attempts:
                a_copy = dict(a)
                a_copy["cand_id"] = r.cand_id
                aggregated_attempts.append(a_copy)
        task.alignment_attempts = aggregated_attempts
        # Use first non-empty verdict / v4_audit as representative; falls
        # through to empty if every candidate had infra-only failures.
        representative = next((r for r in results if r.alignment_verdict), results[0] if results else None)
        if representative is not None:
            task.alignment_verdict = representative.alignment_verdict
            task.alignment_v4_audit = representative.alignment_v4_audit

        any_needs_fix = any(r.outcome == "needs_fix" for r in results)
        if any_needs_fix:
            review_reasons = [r.needs_human_review_reason for r in results if r.needs_human_review_reason]
            task.stage = Stage.NEEDS_FIX
            task.needs_human_review = True
            task.human_review_reason = "; ".join(review_reasons) or "build_align: all candidates failed"
            print(f"    -> NEEDS_FIX (no passers, n={n})")
        else:
            task.stage = Stage.REJECTED
            task.alignment_reason = " | ".join(
                f"cand{r.cand_id}: {r.alignment_reason or r.alignment_verdict or 'no_attempts'}"
                for r in results
            )
            print(f"    -> REJECTED (no passers, n={n})")

        stage_str = task.stage.value if hasattr(task.stage, "value") else str(task.stage)
        task.iteration_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": "build_align",
                "n_candidates": n,
                "n_passed": 0,
                "outcome": stage_str,
                "verdict": task.alignment_verdict,
                "reason": (task.alignment_reason or task.human_review_reason or "")[:500],
            }
        )
        _cleanup_loser_dirs(results)
        await save_state_locked(state, state_file)


async def _run_alignment_only_for_triage(
    task,
    state: PipelineState,
    state_file: str,
    sems: dict,
) -> None:
    """Stage.BUILT entry path (post-triage Build regen) — alignment-only, no fanout.

    Triage's `_run_triage_build_regen` produces exactly one new
    instruction.md in ``task.task_dir``, then sets ``task.stage = Stage.BUILT``.
    The pipeline loop re-enters here. Run alignment once on that
    instruction; do NOT trigger another Build regen on leaked/narrow_tests
    (the triage path's instruction is what it is).
    """
    async with _mark_in_progress(task, "alignment", state, state_file):
        async with sems["llm"]:
            result = await _run_alignment_retry(
                task.task_dir,
                task.repo,
                task.commit_sha,
                task.merge_base_sha,
            )

        if result.context_error:
            task.needs_human_review = True
            task.human_review_reason = result.context_error
            task.stage = Stage.NEEDS_FIX
            await save_state_locked(state, state_file)
            return

        for usage in result.usage_entries:
            task.llm_usage.setdefault("alignment", []).append(usage)
        task.alignment_attempts = result.attempts

        if result.accept_result is not None:
            task.alignment_verdict = "ok"
            task.alignment_reason = result.accept_result.get("reason", "")
            task.alignment_v4_audit = result.accept_result.get("v4_audit", {}) or {}
            task.alignment_feedback = ""
            task.stage = Stage.ALIGNMENT_CHECKED
            print(f"    -> OK (post-triage alignment, attempts={len(result.attempts)})")
        else:
            verdicts = [a.get("verdict", a.get("error", "?")) for a in result.attempts]
            task.alignment_verdict = (
                result.attempts[-1].get("verdict", "error") if result.attempts else "error"
            )
            task.alignment_reason = _summarize_alignment_reason(result.attempts)
            task.alignment_v4_audit = result.attempts[-1].get("v4_audit", {}) if result.attempts else {}

            all_errors = bool(result.attempts) and all(a.get("error") for a in result.attempts)
            if all_errors:
                task.stage = Stage.NEEDS_FIX
                task.needs_human_review = True
                task.human_review_reason = f"alignment judge: all attempts failed ({verdicts})"
                print(f"    -> NEEDS_FIX (post-triage alignment infra errors: {verdicts})")
            else:
                # Post-triage instruction failed alignment; do NOT re-regen build.
                task.stage = Stage.REJECTED
                print(f"    -> REJECT (post-triage alignment: {verdicts})")

        task.iteration_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": "alignment_post_triage",
                "verdict": task.alignment_verdict,
                "attempts": len(result.attempts),
            }
        )
        await save_state_locked(state, state_file)


_DEEP_DIVE_VERIFY_TAIL_CHARS = 12_000
_DEEP_DIVE_HARBOR_LAB_CAP_CHARS = 12_000
_DEEP_DIVE_EDITS_CAP_CHARS = 30_000
_DEEP_DIVE_TEST_BODY_CAP_CHARS = 60_000
_DEEP_DIVE_SUBCOMMAND_TIMEOUT_S = 60


def _resolve_harbor_lab_bin() -> str:
    """Locate the harbor-lab binary. Raises RuntimeError if it cannot be found.

    Preference order: HARBOR_LAB env var, system PATH, then a list of known
    checkout locations (macOS dev machines, craftbench VMs).
    """
    import shutil

    env_bin = os.environ.get("HARBOR_LAB")
    if env_bin and os.access(env_bin, os.X_OK):
        return env_bin

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
    for pinned in candidates:
        if os.path.isfile(pinned) and os.access(pinned, os.X_OK):
            return pinned

    raise RuntimeError(
        "harbor-lab binary not found. Install harbor-lab (clone it and run `uv sync`) "
        "and either put its `.venv/bin/harbor-lab` on PATH or export HARBOR_LAB=<path>."
    )


async def _run_harbor_lab(bin_path: str, *args: str) -> str:
    """Run `harbor-lab <args...>` and return stdout. Never raises on nonzero exit —
    the stderr/stdout are returned so the LLM can see partial output."""
    try:
        proc = await asyncio.create_subprocess_exec(
            bin_path,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        return f"[harbor-lab invocation failed: {e}]"

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_DEEP_DIVE_SUBCOMMAND_TIMEOUT_S)
    except asyncio.TimeoutError:
        # asyncio.wait_for cancels communicate() but leaves the child process
        # running. Without an explicit kill/wait, a slow or hung harbor-lab
        # accumulates as a zombie on every timeout (deep-dive fans out 5×).
        proc.kill()
        await proc.wait()
        return f"[harbor-lab {' '.join(args)} timed out after {_DEEP_DIVE_SUBCOMMAND_TIMEOUT_S}s]"

    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace").strip()
    if proc.returncode != 0 and err:
        out = f"{out}\n[stderr]\n{err}"
    return out


def _cap(text: str, limit: int) -> str:
    """Head/tail-truncate text to `limit` chars with a truncation marker."""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head - 64
    if tail <= 0:
        return text[:limit]
    dropped = len(text) - head - tail
    return f"{text[:head]}\n... [truncated {dropped} chars] ...\n{text[-tail:]}"


async def _fetch_deep_dive_context(task_dir: str, trial_dir: str) -> dict:
    """Pre-assemble every piece of evidence the deep-dive judge needs.

    Direct-API judge has no shell / Read tools, so all harbor-lab output and
    file contents must be captured in Python. Subcommands run concurrently;
    file reads happen in a thread to keep the event loop unblocked.
    """
    try:
        hlab_bin = _resolve_harbor_lab_bin()
    except RuntimeError as e:
        hlab_bin = None
        hlab_error = str(e)
    else:
        hlab_error = ""

    job_dir = str(Path(trial_dir).parent) if trial_dir else ""

    async def _hlab(*args: str) -> str:
        if hlab_bin is None:
            return f"[harbor-lab unavailable: {hlab_error}]"
        if not job_dir:
            return "[no trial/job dir on task — skipping harbor-lab]"
        return await _run_harbor_lab(hlab_bin, *args)

    def _read_text(path: Path, cap: int | None = None) -> str:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return ""
        return _cap(text, cap) if cap else text

    def _load_files() -> dict:
        """Read all the static files in one thread hop so _fetch remains async."""
        out = {
            "instruction_md": "",
            "reward_json": "",
            "verify_output_tail": "",
            "postmerge_test_bodies": [],
            "f2p_tests": "",
            "p2p_tests": "",
            "f2p_skip": "",
            "p2p_skip": "",
        }
        out["instruction_md"] = _read_text(Path(task_dir, "instruction.md"))

        reward_path = Path(trial_dir, "verifier", "reward.json")
        out["reward_json"] = _read_text(reward_path)

        verify_path = Path(trial_dir, "verifier", "verify_full_output.txt")
        verify_text = _read_text(verify_path)
        out["verify_output_tail"] = _cap(verify_text, _DEEP_DIVE_VERIFY_TAIL_CHARS) if verify_text else ""

        tests_dir = Path(task_dir, "tests")
        out["f2p_tests"] = _read_text(tests_dir / "fail_to_pass.txt")
        out["p2p_tests"] = _read_text(tests_dir / "pass_to_pass.txt")
        out["f2p_skip"] = _read_text(tests_dir / "f2p_skip.txt")
        out["p2p_skip"] = _read_text(tests_dir / "p2p_skip.txt")

        postmerge_dir = tests_dir / "postmerge_tests"
        if postmerge_dir.is_dir():
            bodies: list[tuple[str, str]] = []
            remaining = _DEEP_DIVE_TEST_BODY_CAP_CHARS
            for p in sorted(postmerge_dir.rglob("*.py")):
                if remaining <= 0:
                    break
                rel = str(p.relative_to(postmerge_dir))
                body = _read_text(p)
                if len(body) > remaining:
                    body = _cap(body, remaining)
                bodies.append((rel, body))
                remaining -= len(body)
            out["postmerge_test_bodies"] = bodies
        return out

    static_task = asyncio.create_task(asyncio.to_thread(_load_files))
    errors_task = asyncio.create_task(_hlab("errors", f"{job_dir}/", "--format", "markdown"))
    edits_task = asyncio.create_task(_hlab("edits", f"{job_dir}/", "--format", "markdown"))
    edits_verbose_task = asyncio.create_task(
        _hlab("edits", f"{job_dir}/", "--verbose", "--format", "markdown")
    )
    # Two tool-sequence fetches:
    #  - `tool_seq_tail` (--tail 10, with text) → last-10 trajectory for the
    #    DD prompt. Bounded for prompt brevity.
    #  - `tool_seq_full` (no --tail) → full trajectory for the deterministic
    #    easiness check, which needs total Grep+Read and pytest counts across
    #    the whole run. The tailed version undercounts: on long trials all
    #    Reads often happen before the last 10 calls, producing false
    #    `grep_read=0` flags.
    tool_seq_tail_task = asyncio.create_task(
        _hlab("tool-sequence", f"{job_dir}/", "--tail", "10", "--text", "--format", "markdown")
    )
    tool_seq_full_task = asyncio.create_task(_hlab("tool-sequence", f"{job_dir}/", "--format", "markdown"))
    metrics_task = asyncio.create_task(_hlab("metrics", f"{job_dir}/", "--format", "markdown"))

    (
        static,
        errors_md,
        edits_md,
        edits_verbose_md,
        tool_seq_tail_md,
        tool_seq_full_md,
        metrics_md,
    ) = await asyncio.gather(
        static_task,
        errors_task,
        edits_task,
        edits_verbose_task,
        tool_seq_tail_task,
        tool_seq_full_task,
        metrics_task,
    )

    edits_combined = edits_md.strip()
    if edits_verbose_md.strip():
        edits_combined = (
            f"{edits_combined}\n\n---\n### Edits (verbose)\n\n"
            f"{_cap(edits_verbose_md, _DEEP_DIVE_EDITS_CAP_CHARS)}"
        )

    return {
        **static,
        "harbor_lab_errors": _cap(errors_md, _DEEP_DIVE_HARBOR_LAB_CAP_CHARS),
        "harbor_lab_edits": _cap(edits_combined, _DEEP_DIVE_EDITS_CAP_CHARS),
        "harbor_lab_tool_sequence": _cap(tool_seq_tail_md, _DEEP_DIVE_HARBOR_LAB_CAP_CHARS),
        # Full (un-tailed) tool sequence — deterministic easiness counts from
        # here so totals aren't truncated by the prompt-brevity tail.
        "harbor_lab_tool_sequence_full": tool_seq_full_md,
        "harbor_lab_metrics": _cap(metrics_md, _DEEP_DIVE_HARBOR_LAB_CAP_CHARS),
    }


def _build_alignment_feedback(attempts: list[dict]) -> str:
    """Summarize alignment-judge rejections into feedback for a Build regen.

    Collects unique leakage_evidence quotes across all attempts plus the most
    recent non-ok reason. Bounded in length so it doesn't balloon the Build
    prompt.
    """
    evidence_seen: set[str] = set()
    evidence: list[str] = []
    for a in attempts:
        for quote in a.get("leakage_evidence", []) or []:
            if quote and quote not in evidence_seen:
                evidence_seen.add(quote)
                evidence.append(quote.strip()[:300])
    non_ok = [a for a in attempts if a.get("verdict") and a["verdict"] != "ok"]
    last_reason = (non_ok[-1].get("reason", "").strip()[:500]) if non_ok else ""
    verdict = non_ok[-1].get("verdict", "unknown") if non_ok else "unknown"

    lines = [
        f"The previous instruction.md was rejected by the alignment judge (verdict: {verdict}).",
        f"Reason: {last_reason}",
    ]
    if evidence:
        lines.append(
            "\nSpecific phrases the judge flagged as leakage — do NOT include "
            "these or equivalents in the regenerated instruction:"
        )
        for quote in evidence[:8]:
            lines.append(f'  - "{quote}"')
        lines.append(
            "\nRewrite each flagged concept in behavioral terms. Public class "
            "names and module paths that reference tests import are OK; "
            "private helpers, internal file paths, exact constants from the "
            "diff, and implementation-recipe language are not."
        )
    return "\n".join(lines)


async def step_alignment(state: PipelineState, state_file: str, concurrency: int = 4) -> None:
    """Alignment-only step for BUILT tasks (post-triage Build regen).

    Thin wrapper for ``--from-step alignment`` resume. Delegates to
    ``_run_alignment_only_for_triage`` which runs alignment without
    fanout (the triage path produced exactly one new instruction.md
    per task; there's nothing to fan out).

    Note: the ``concurrency`` parameter is retained for backwards
    compatibility with the ``--from-step`` CLI but is ignored — sem
    sizing comes from ``_cfg.*`` so this wrapper matches the main
    pipeline path in pipeline.py.
    """
    del concurrency  # ignored — see docstring; sems sized from _cfg

    built = [t for t in state.tasks.values() if t.stage == Stage.BUILT]
    if not built:
        print("No BUILT tasks to alignment-check.")
        return

    candidate_sem_size = max(_cfg.LLM_CONCURRENCY, _cfg.LLM_CONCURRENCY * 2)
    if candidate_sem_size < _cfg.BUILD_N_CANDIDATES:
        candidate_sem_size = _cfg.BUILD_N_CANDIDATES
    sems = {
        "llm": asyncio.Semaphore(_cfg.LLM_CONCURRENCY),
        "docker": asyncio.Semaphore(_cfg.DOCKER_CONCURRENCY),
        "smoke": asyncio.Semaphore(_cfg.SMOKE_CONCURRENCY),
        "candidate": asyncio.Semaphore(candidate_sem_size),
    }

    print(f"Alignment-checking {len(built)} BUILT (post-triage) tasks (llm={_cfg.LLM_CONCURRENCY})...")
    await asyncio.gather(*[_run_alignment_only_for_triage(t, state, state_file, sems) for t in built])


def _dockerfile_mtime(task_dir: str) -> float:
    path = os.path.join(task_dir, "environment", "Dockerfile")
    return os.path.getmtime(path) if os.path.isfile(path) else 0


def _has_dockerfile(task_dir: str) -> bool:
    """Return True if environment/Dockerfile already exists in the task directory."""
    return os.path.isfile(os.path.join(task_dir, "environment", "Dockerfile"))


# ---------------------------------------------------------------------------
# Step 4: Find tests and generate solve.sh (ALIGNMENT_CHECKED → TESTS_DISCOVERED)
# ---------------------------------------------------------------------------


async def _run_assemble_task_dir_artifacts_one(task, state: PipelineState, state_file: str) -> None:
    """Assemble all mechanical task artifacts: task.toml, solve.sh, and postmerge test files.

    No LLM call. Moves from ALIGNMENT_CHECKED to TESTS_DISCOVERED.
    Docker build happens separately in _run_docker_classify_one.
    """
    async with _mark_in_progress(task, "assemble_task_dir_artifacts", state, state_file):
        print(f"  [assemble_task_dir_artifacts] {task.task_dir}")

        # Generate solution/solve.sh and solution/changes.patch from git diff.
        solve_ok, solve_err = _generate_solve_sh(
            task.repo, task.merge_base_sha, task.commit_sha, task.task_dir
        )
        if not solve_ok:
            print(f"    -> FAILED: solve.sh generation failed: {solve_err}")
            task.needs_human_review = True
            task.human_review_reason = f"solve.sh generation failed: {solve_err}"
            task.stage = Stage.NEEDS_FIX
            task.iteration_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "step": "assemble_task_dir_artifacts_failed",
                    "reason": f"solve.sh: {solve_err}",
                }
            )
            await save_state_locked(state, state_file)
            return

        # Discover test files changed by the commit.
        test_paths = await _find_commit_test_files(task.repo, task.commit_sha, task.merge_base_sha)
        if test_paths is None:
            task.needs_human_review = True
            task.human_review_reason = "Git infrastructure failure: could not list commit diff files"
            task.stage = Stage.NEEDS_FIX
            await save_state_locked(state, state_file)
            return
        if not test_paths:
            task.needs_human_review = True
            task.human_review_reason = "No test files found in commit diff"
            task.stage = Stage.NEEDS_FIX
            await save_state_locked(state, state_file)
            return

        # Extract postmerge test files so Docker classify step can overlay them.
        extracted = await _extract_postmerge_tests(task.repo, task.commit_sha, task.task_dir, test_paths)
        if extracted < len(test_paths) and extracted > 0:
            task.iteration_log.append(
                {
                    "step": "postmerge_extract_partial",
                    "extracted": extracted,
                    "total": len(test_paths),
                }
            )
            await save_state_locked(state, state_file)
        if extracted == 0:
            task.needs_human_review = True
            task.human_review_reason = "All postmerge test files failed to extract from git"
            task.stage = Stage.NEEDS_FIX
            await save_state_locked(state, state_file)
            return

        profile = _cfg.PipelineProfile(
            **{k: v for k, v in state.profile_data.items() if k in _cfg.PipelineProfile.__dataclass_fields__}
        )
        _write_task_toml(task.task_dir, _generate_task_id(task.repo, task.commit_sha), profile)
        task.stage = Stage.TESTS_DISCOVERED
        print(f"    -> TESTS_DISCOVERED: {len(test_paths)} test file(s), solve.sh ready")
        task.iteration_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": "assemble_task_dir_artifacts",
                "test_files": len(test_paths),
            }
        )
        await save_state_locked(state, state_file)


async def step_assemble_task_dir_artifacts(
    state: PipelineState, state_file: str, concurrency: int = 4
) -> None:
    """Assemble mechanical task artifacts for ALIGNMENT_CHECKED tasks (thin --from-step wrapper)."""
    checked = [t for t in state.tasks.values() if t.stage == Stage.ALIGNMENT_CHECKED]
    if not checked:
        print("No ALIGNMENT_CHECKED tasks for assemble_task_dir_artifacts.")
        return

    sem = asyncio.Semaphore(concurrency)

    async def _wrap(task):
        async with sem:
            await _run_assemble_task_dir_artifacts_one(task, state, state_file)

    print(f"Assembling task artifacts for {len(checked)} tasks (concurrency={concurrency})...")
    await asyncio.gather(*[_wrap(t) for t in checked])


# ---------------------------------------------------------------------------
# Step 4c: Build Dockerfile (TESTS_DISCOVERED → DOCKERFILE_BUILT)
# ---------------------------------------------------------------------------


async def _run_build_dockerfile_one(task, state: PipelineState, state_file: str) -> None:
    """Create or regenerate environment/Dockerfile via Claude.

    Advances stage to DOCKERFILE_BUILT on success. Sets NEEDS_FIX on failure.
    """
    async with _mark_in_progress(task, "build_dockerfile", state, state_file):
        print(f"  [build_dockerfile] {task.task_dir}")

        prompt = build_dockerfile_prompt(
            task.repo,
            task.commit_sha,
            task.merge_base_sha,
            task.task_dir,
        )
        result = await run_claude_async(
            prompt,
            max_turns=40,
            allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
            model=_cfg.LLM_STEP_MODEL or None,
        )

        if err := _check_claude_error(result, "build_dockerfile"):
            print(f"    -> FAILED: {err}")
            task.needs_human_review = True
            task.human_review_reason = err
            task.stage = Stage.NEEDS_FIX
            await save_state_locked(state, state_file)
            return

        if not _has_dockerfile(task.task_dir):
            print("    -> FAILED: Claude ran but environment/Dockerfile was not created")
            task.needs_human_review = True
            task.human_review_reason = "build_dockerfile step: Dockerfile not created by Claude"
            task.stage = Stage.NEEDS_FIX
            await save_state_locked(state, state_file)
            return

        task.stage = Stage.DOCKERFILE_BUILT
        print("    -> Dockerfile created")
        task.iteration_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": "build_dockerfile",
            }
        )


# ---------------------------------------------------------------------------
# Step 5: Docker build + F2P/P2P classification (DOCKERFILE_BUILT → F2P_P2P_CLASSIFIED)
# ---------------------------------------------------------------------------


async def _run_docker_classify_one(task, state: PipelineState, state_file: str) -> None:
    """Build Docker image and run 2-pass overlay/oracle classification.

    Assumes test files are already discovered and in tests/postmerge_tests/
    (done by _run_assemble_task_dir_artifacts_one). Reads test paths from disk.
    """
    async with _mark_in_progress(task, "docker_classify", state, state_file):
        print(f"  [docker_classify] {task.task_dir}")

        # Reconstruct test_paths list from postmerge_tests/ directory on disk.
        # This avoids storing them in state and works correctly after pipeline restarts.
        postmerge_dir = os.path.join(task.task_dir, "tests", "postmerge_tests")
        test_paths: list[str] = []
        if os.path.isdir(postmerge_dir):
            for root, _dirs, files in os.walk(postmerge_dir):
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    rel = os.path.relpath(os.path.join(root, fname), postmerge_dir)
                    test_paths.append(rel)

        if not test_paths:
            task.needs_human_review = True
            task.human_review_reason = (
                "docker_classify: no test files found in tests/postmerge_tests/ — "
                "run from find_tests stage to re-discover"
            )
            task.stage = Stage.NEEDS_FIX
            await save_state_locked(state, state_file)
            return

        image_built = False
        dockerfile_mtime = _dockerfile_mtime(task.task_dir)

        while True:
            need_rebuild = not image_built or _dockerfile_mtime(task.task_dir) != dockerfile_mtime
            if need_rebuild:
                print("    Building Docker image...")
                build_ok, build_output = await run_docker_build_async(task.task_dir)
                if not build_ok:
                    issue = f"Docker build failed:\n{build_output[-500:]}"
                    print("    -> BUILD FAILED, attempting fix...")
                    task.iteration_log.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "step": "docker_build_fail",
                            "fix_attempt": task.fix_attempts,
                        }
                    )
                    _fix_ok = await _fix_docker_or_shelve_async(task, issue, "Docker build failed")
                    if task.task_dir:
                        strip_instruction_boilerplate(os.path.join(task.task_dir, "instruction.md"))
                    if _fix_ok:
                        continue
                    break
                image_built = True
                dockerfile_mtime = _dockerfile_mtime(task.task_dir)

            print(f"    Running 2-run classification ({len(test_paths)} test files)...")
            f2p_tests, p2p_tests, classify_output = await run_f2p_p2p_classify_async(
                task.task_dir, test_paths
            )

            if f2p_tests is None:
                # Default values; overridden by specific error branches below.
                step_label = "classify_error"
                fix_label = "Classification: unknown error"
                issue = (
                    f"F2P/P2P classification failed with unknown error.\n"
                    f"Test files: {test_paths}\n"
                    f"Output:\n{classify_output[-500:]}"
                )
                if classify_output.startswith("TIMEOUT:"):
                    step_label = "classify_timeout"
                    fix_label = "Classification: Docker timed out"
                    issue = (
                        "F2P/P2P classification timed out — Docker container exceeded time limit.\n"
                        "This is likely an infrastructure issue, not a task problem. "
                        "Check Docker resource limits or retry.\n"
                        f"Test files: {test_paths}\n"
                        f"Output:\n{classify_output}"
                    )
                elif classify_output.startswith("SOLVE_FAIL:"):
                    step_label = "F2P/P2P Classify: Solution Patch Failed"
                    fix_label = "Classification: patch failed to apply"
                    issue = (
                        "F2P/P2P classification failed: solve.sh (changes.patch) did not apply cleanly.\n"
                        "The patch may conflict with the repo state in the Docker image. "
                        "Check that the Dockerfile clones at the correct base SHA and that "
                        "changes.patch was generated from the right SHAs.\n"
                        f"Test files: {test_paths}\n"
                        f"Output:\n{classify_output[-500:]}"
                    )
                elif classify_output.startswith("OVERLAY_REGRESSION:"):
                    print("    -> Classification: test regressed (passed overlay, failed oracle) — rejecting")
                    task.stage = Stage.REJECTED
                    task.issues.append(
                        {
                            "type": "overlay_regression",
                            "test": "",
                            "description": (
                                "F2P/P2P classification failed: at least one test passed on pre-merge code "
                                "(overlay) but failed after the fix (oracle). This indicates the commit "
                                "introduced a regression in an existing test — the task is invalid.\n"
                                f"Test files: {test_paths}"
                            ),
                            "classification": "overlay_regression",
                        }
                    )
                    await save_state_locked(state, state_file)
                    return
                elif classify_output.startswith("OVERLAY_UNCOLLECTED:"):
                    print(
                        "    -> Classification: test not collected in overlay "
                        "(imports new API not yet in baseline) — rejecting"
                    )
                    task.stage = Stage.REJECTED
                    task.issues.append(
                        {
                            "type": "overlay_uncollected",
                            "test": "",
                            "description": (
                                "F2P/P2P classification failed: new test file imports symbols "
                                "that don't exist in the pre-merge codebase. The test cannot be "
                                "collected during the overlay run. This usually means the test "
                                "directly imports the new feature being added — not suitable for "
                                "fail-to-pass classification.\n"
                                f"Test files: {test_paths}"
                            ),
                            "classification": "overlay_uncollected",
                        }
                    )
                    await save_state_locked(state, state_file)
                    return
                elif classify_output.startswith("ORACLE_ZERO:"):
                    step_label = "classify_oracle_zero"
                    fix_label = "Classification: oracle run passed 0 tests"
                    issue = (
                        "F2P/P2P classification failed: the oracle run (with solution applied) "
                        "passed 0 tests. The solution patch may be incomplete, or the test "
                        "discovery may be targeting the wrong files.\n"
                        f"Test files: {test_paths}\n"
                        f"Output:\n{classify_output[-500:]}"
                    )

                task.iteration_log.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": step_label,
                        "fix_attempt": task.fix_attempts,
                        "output": classify_output[-200:],
                    }
                )
                print(f"    -> {fix_label}, attempting fix...")
                _fix_ok = await _fix_f2p_p2p_classify_or_shelve_async(task, issue, fix_label)
                if task.task_dir:
                    strip_instruction_boilerplate(os.path.join(task.task_dir, "instruction.md"))
                if _fix_ok:
                    image_built = False  # force Docker rebuild after fix
                    continue
                break
            else:
                tests_dir = os.path.join(task.task_dir, "tests")

                # Preserve any existing skip files across test list regeneration
                f2p_skip_path = os.path.join(tests_dir, "f2p_skip.txt")
                f2p_skip_backup = None
                if os.path.isfile(f2p_skip_path):
                    with open(f2p_skip_path) as ef:
                        f2p_skip_backup = ef.read()

                p2p_skip_path = os.path.join(tests_dir, "p2p_skip.txt")
                p2p_skip_backup = None
                if os.path.isfile(p2p_skip_path):
                    with open(p2p_skip_path) as ef:
                        p2p_skip_backup = ef.read()

                try:
                    with open(os.path.join(tests_dir, "fail_to_pass.txt"), "w") as f:
                        f.write("\n".join(f2p_tests) + ("\n" if f2p_tests else ""))

                    with open(os.path.join(tests_dir, "pass_to_pass.txt"), "w") as f:
                        f.write("\n".join(p2p_tests) + ("\n" if p2p_tests else ""))

                    with open(os.path.join(tests_dir, "score.py"), "w") as f:
                        f.write(SCORE_PY_TEMPLATE)

                    with open(os.path.join(tests_dir, "test.sh"), "w") as f:
                        f.write(
                            "#!/usr/bin/env bash\n"
                            "set -uo pipefail\n"
                            "mkdir -p /logs/verifier\n"
                            "# Overlay postmerge test files so newly-added tests exist for scoring\n"
                            "if [ -d /tests/postmerge_tests ]; then\n"
                            "    find /tests/postmerge_tests -type f | while IFS= read -r f; do\n"
                            '        rel="${f#/tests/postmerge_tests/}"\n'
                            '        mkdir -p "/code/$(dirname "$rel")"\n'
                            '        cp "$f" "/code/$rel"\n'
                            "    done\n"
                            "fi\n"
                            "cd /code\n"
                            "python3 -m pytest -v --tb=no --continue-on-collection-errors \\\n"
                            "    $(cat /tests/fail_to_pass.txt /tests/pass_to_pass.txt 2>/dev/null) \\\n"
                            "    2>&1 | tee /logs/verifier/verify_full_output.txt || true\n"
                            "python3 /tests/score.py\n"
                        )
                except OSError as e:
                    task.needs_human_review = True
                    task.human_review_reason = f"Failed to write test list files: {e}"
                    task.stage = Stage.NEEDS_FIX
                    await save_state_locked(state, state_file)
                    return

                # Restore skip files if they existed before regeneration
                if f2p_skip_backup is not None:
                    with open(f2p_skip_path, "w") as ef:
                        ef.write(f2p_skip_backup)
                if p2p_skip_backup is not None:
                    with open(p2p_skip_path, "w") as ef:
                        ef.write(p2p_skip_backup)

                task.f2p_tests = f2p_tests
                task.p2p_tests = p2p_tests
                task.stage = Stage.F2P_P2P_CLASSIFIED
                _write_candidate_json(task)

                print(f"    -> CLASSIFIED: {len(f2p_tests)} F2P, {len(p2p_tests)} P2P")
                task.iteration_log.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "docker_classify",
                        "f2p_count": len(f2p_tests),
                        "p2p_count": len(p2p_tests),
                    }
                )

                f2p_lines = "".join(f"- `{t}`\n" for t in f2p_tests)
                p2p_lines = "".join(f"- `{t}`\n" for t in p2p_tests)
                diag_content = (
                    f"# F2P/P2P Classification\n\n"
                    f"**Commit:** {task.commit_sha}\n"
                    f"**Test files scanned:** {len(test_paths)}\n\n"
                    f"**Fail-to-Pass ({len(f2p_tests)}):**\n{f2p_lines}"
                    f"\n**Pass-to-Pass ({len(p2p_tests)}):**\n{p2p_lines}"
                )
                _write_diagnostic(_next_diagnostic_path(task.task_dir, "docker_classify"), diag_content)
                break

        await save_state_locked(state, state_file)


async def step_build_dockerfile(state: PipelineState, state_file: str, concurrency: int = 4) -> None:
    """Build Dockerfile for TESTS_DISCOVERED tasks (thin wrapper for --from-step resume)."""
    tasks = [t for t in state.tasks.values() if t.stage == Stage.TESTS_DISCOVERED]
    if not tasks:
        print("No TESTS_DISCOVERED tasks for build_dockerfile.")
        return

    sem = asyncio.Semaphore(concurrency)

    async def _wrap(task):
        async with sem:
            await _run_build_dockerfile_one(task, state, state_file)

    print(f"Building Dockerfiles for {len(tasks)} tasks (concurrency={concurrency})...")
    await asyncio.gather(*[_wrap(t) for t in tasks])


async def step_docker_classify(state: PipelineState, state_file: str, concurrency: int = 4) -> None:
    """Run docker_classify for DOCKERFILE_BUILT tasks (thin wrapper for --from-step resume)."""
    built = [t for t in state.tasks.values() if t.stage == Stage.DOCKERFILE_BUILT]
    if not built:
        print("No DOCKERFILE_BUILT tasks to classify.")
        return

    sem = asyncio.Semaphore(concurrency)

    async def _wrap(task):
        async with sem:
            await _run_docker_classify_one(task, state, state_file)

    print(f"Docker-classifying {len(built)} tasks (concurrency={concurrency})...")
    await asyncio.gather(*[_wrap(t) for t in built])


# ---------------------------------------------------------------------------
# Step 5: F2P/P2P classification (replaces docker validate)
# ---------------------------------------------------------------------------


async def _find_commit_test_files(repo: str, commit_sha: str, merge_base_sha: str) -> list[str] | None:
    """Return test file paths (relative to repo root) changed between merge_base_sha and commit_sha.

    Uses git diff <merge_base_sha> <commit_sha> (required for merge commits, which produce
    no output with git show --name-only).

    Returns None on git infrastructure failure (timeout/OSError), as distinct from
    returning [] when git succeeds but no test files were changed.
    """
    repo_path = os.path.join("repos", repo)
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            repo_path,
            "diff",
            merge_base_sha,
            commit_sha,
            "--name-only",
            "--diff-filter=AM",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except (asyncio.TimeoutError, OSError) as e:
        print(f"    WARNING: git infrastructure failure for {repo}@{commit_sha}: {e}")
        return None

    if proc.returncode != 0:
        print(f"    WARNING: git exited {proc.returncode} for {repo}@{commit_sha}: {stderr.decode().strip()}")
        return None

    return [
        line.strip() for line in stdout.decode().splitlines() if line.strip() and _is_test_file(line.strip())
    ]


async def _extract_postmerge_tests(repo: str, commit_sha: str, task_dir: str, test_paths: list[str]) -> int:
    """Copy test files at commit_sha into tests/postmerge_tests/.

    Stored under tests/ so Harbor uploads them to /tests/postmerge_tests/ at verification
    time only — the agent never sees them during its run.

    Returns the number of files successfully extracted.
    """
    postmerge_dir = os.path.join(task_dir, "tests", "postmerge_tests")
    repo_path = os.path.join("repos", repo)
    # Pre-create all unique parent directories before spawning git subprocesses
    unique_dirs = {os.path.dirname(os.path.join(postmerge_dir, p)) for p in test_paths}
    for d in unique_dirs:
        os.makedirs(d, exist_ok=True)

    async def _extract_one(rel_path: str) -> tuple[str, str] | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                repo_path,
                "show",
                f"{commit_sha}:{rel_path}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode == 0:
                with open(os.path.join(postmerge_dir, rel_path), "w") as f:
                    f.write(stdout.decode())
                return None
            else:
                return (rel_path, stderr.decode().strip() or "file not found at commit")
        except (asyncio.TimeoutError, OSError) as e:
            print(f"    WARNING: failed to extract {rel_path}: {e}")
            return (rel_path, str(e))

    results = await asyncio.gather(*[_extract_one(p) for p in test_paths])
    skipped = [r for r in results if r is not None]
    if skipped:
        print(f"    WARNING: {len(skipped)}/{len(test_paths)} postmerge test files could not be extracted:")
        for path, reason in skipped:
            print(f"      - {path} ({reason})")
    return len(test_paths) - len(skipped)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Step 6 (was 7): Oracle check (hard gate — blocks on failure)
# ---------------------------------------------------------------------------


async def _run_oracle_check_one(task, state: PipelineState, state_file: str) -> None:
    """Apply solve.sh, run score.py. Block pipeline if not resolved."""
    async with _mark_in_progress(task, "oracle_check", state, state_file):
        print(f"  [oracle_check] {task.task_dir}")

        result = await run_score_check_async(task.task_dir, apply_solution=True)

        if "error" in result:
            task.oracle_flagged = True
            task.oracle_flag_reason = f"Score check error: {result['error']}"
            task.needs_human_review = True
            task.stage = Stage.NEEDS_FIX
            print(f"    -> ERROR: {result['error']} — blocked for human review")
            task.iteration_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "step": "oracle_check",
                    "oracle_resolved": False,
                    "error": result["error"],
                }
            )
            diag_content = (
                f"# Oracle Check\n\n"
                f"**Result:** ERROR\n"
                f"**Error:** {result['error']}\n"
                f"**Output:**\n```\n{result.get('output', '')}\n```\n"
            )
        else:
            resolved = result["resolved"]
            f2p_score = result["f2p_score"]
            p2p_score = result["p2p_score"]
            f2p_passed = result["f2p_passed"]
            f2p_total = result["f2p_total"]
            p2p_passed = result["p2p_passed"]
            p2p_total = result["p2p_total"]

            task.oracle_resolved = resolved
            task.oracle_f2p_score = f2p_score
            task.oracle_p2p_score = p2p_score

            diag_content = (
                f"# Oracle Check\n\n"
                f"**Resolved:** {resolved}\n"
                f"**F2P score:** {f2p_score:.2f} ({f2p_passed}/{f2p_total})\n"
                f"**P2P score:** {p2p_score:.2f} ({p2p_passed}/{p2p_total})\n"
            )

            if not resolved:
                task.oracle_flagged = True
                task.oracle_flag_reason = (
                    f"Oracle not resolved: f2p={f2p_score:.2f} ({f2p_passed}/{f2p_total}), "
                    f"p2p={p2p_score:.2f} ({p2p_passed}/{p2p_total})"
                )
                task.needs_human_review = True
                task.stage = Stage.NEEDS_FIX  # blocking — stops pipeline
                print(f"    -> NOT RESOLVED: {task.oracle_flag_reason}")
                print("    -> Blocked for human review (edit tests/*.txt and --from-step oracle_check)")
                diag_content += f"\n**Flag reason:** {task.oracle_flag_reason}\n"
            else:
                task.stage = Stage.ORACLE_CHECKED
                print(f"    -> RESOLVED: f2p={f2p_score:.2f}, p2p={p2p_score:.2f}")

            task.iteration_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "step": "oracle_check",
                    "oracle_resolved": resolved,
                    "oracle_f2p_score": f2p_score,
                    "oracle_p2p_score": p2p_score,
                }
            )

        diag_path = _next_diagnostic_path(task.task_dir, "oracle_check")
        _write_diagnostic(diag_path, diag_content)
        await save_state_locked(state, state_file)


async def step_oracle_check(state: PipelineState, state_file: str, concurrency: int = 4) -> None:
    """Oracle check for F2P_P2P_CLASSIFIED tasks. Hard gate — blocks pipeline if not resolved."""
    checked = [t for t in state.tasks.values() if t.stage == Stage.F2P_P2P_CLASSIFIED]
    if not checked:
        print("No F2P_P2P_CLASSIFIED tasks for oracle check.")
        return

    sem = asyncio.Semaphore(concurrency)

    async def _wrap(task):
        async with sem:
            await _run_oracle_check_one(task, state, state_file)

    print(f"Oracle-checking {len(checked)} tasks (concurrency={concurrency})...")
    await asyncio.gather(*[_wrap(t) for t in checked])


# ---------------------------------------------------------------------------
# Generic smoke test and triage steps
# ---------------------------------------------------------------------------


async def _run_smoke_one(
    task,
    state: PipelineState,
    state_file: str,
    *,
    model: str,
    label: str,
    score_attr: str,
    trial_attr: str,
    next_stage: Stage,
    agent: str = "claude-code",
    reasoning_effort: str = "",
) -> None:
    """Run smoke test on a single task with retry logic."""
    async with _mark_in_progress(task, f"{label}_smoke", state, state_file):
        print(f"  [{label}_smoke] {task.task_dir}")

        diag = {}
        for attempt in range(_cfg.MAX_SMOKE_RETRIES + 1):
            diag = await _run_smoke_async(task, model, label, agent=agent, reasoning_effort=reasoning_effort)

            is_retryable = diag.get("infra_failure") or diag.get("no_trial")
            if not is_retryable:
                break  # real result (success, timeout, or task-level failure)

            if attempt < _cfg.MAX_SMOKE_RETRIES:
                kind = "infra failure" if diag.get("infra_failure") else "no trial dir"
                wait = 60 * (attempt + 1)
                print(f"    -> {kind} (attempt {attempt + 1}), retrying in {wait}s...")
                task.iteration_log.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": f"{label}_smoke_retry",
                        "attempt": attempt + 1,
                        "exception": str(diag.get("exception", ""))[:200],
                    }
                )
                await asyncio.sleep(wait)

        # Route final result
        if diag.get("infra_failure") or diag.get("no_trial"):
            setattr(task, score_attr, "infra_failure" if diag.get("infra_failure") else "error")
            setattr(task, trial_attr, diag.get("trial_dir", ""))
            task.stage = Stage.NEEDS_FIX
            task.needs_human_review = True
            task.human_review_reason = (
                f"{label} smoke infra/no_trial after {_cfg.MAX_SMOKE_RETRIES + 1} attempts. "
                f"Exception: {diag.get('exception', 'none')[:100]}"
            )
        elif diag.get("timeout"):
            setattr(task, score_attr, "error")
            task.stage = Stage.NEEDS_FIX
            task.needs_human_review = True
            task.human_review_reason = f"{label} smoke timed out (>30 min)"
        else:
            setattr(task, score_attr, diag.get("score_detail", diag.get("score", "?/?")))
            setattr(task, trial_attr, diag.get("trial_dir", ""))
            task.stage = next_stage

        task.iteration_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": f"{label}_smoke",
                "score": getattr(task, score_attr, ""),
                "opus_score": task.opus_score,
                "fix_attempts": task.fix_attempts,
            }
        )
        await save_state_locked(state, state_file)


async def _step_smoke(
    state: PipelineState,
    state_file: str,
    *,
    from_stage: Stage,
    model: str,
    label: str,
    score_attr: str,
    trial_attr: str,
    next_stage: Stage,
    concurrency: int = 4,
    agent: str = "claude-code",
    reasoning_effort: str = "",
) -> None:
    """Generic smoke test step — async with semaphore."""
    tasks = [t for t in state.tasks.values() if t.stage == from_stage]
    if not tasks:
        print(f"No {from_stage.value} tasks for {label} smoke test.")
        return

    sem = asyncio.Semaphore(concurrency)

    async def _wrap(task):
        async with sem:
            await _run_smoke_one(
                task,
                state,
                state_file,
                model=model,
                label=label,
                score_attr=score_attr,
                trial_attr=trial_attr,
                next_stage=next_stage,
                agent=agent,
                reasoning_effort=reasoning_effort,
            )

    print(f"{label} smoke-testing {len(tasks)} tasks (concurrency={concurrency})...")
    await asyncio.gather(*[_wrap(t) for t in tasks])


async def _run_triage_build_regen(task, label: str, fixable_issues: list[dict]) -> bool:
    """Re-run the Build step with triage feedback to produce a revised instruction.

    Called by `_run_triage_one` when the fairness reviewer returns
    `severity=major` with both an evidence quote and a named failing
    test. Assembles the reviewer's evidence as a `<triage_feedback>`
    block, re-fetches Build context (repo map, diff, etc.), and invokes
    the Build judge. Writes the new `instruction.md` if successful.

    Returns True on success (caller should route task back through
    alignment + smoke). Returns False if the regen failed or the regen
    budget is exhausted — in either case the task is shelved to
    NEEDS_FIX.
    """
    if task.triage_regen_count >= _cfg.MAX_TRIAGE_REGENS:
        print(f"    -> MAX TRIAGE REGENS ({_cfg.MAX_TRIAGE_REGENS}) reached, shelving")
        task.stage = Stage.NEEDS_FIX
        task.needs_human_review = True
        task.human_review_reason = f"{label} triage: {_cfg.MAX_TRIAGE_REGENS} regen(s) exhausted"
        return False

    # Dispatch feedback template based on what triggered the regen.
    # `_run_triage_one` packs either a reviewer verdict (classification=
    # "reviewer_unfairness") or an easiness signal (classification=
    # "easiness_too_prescriptive") into the single fixable_issue.
    first = fixable_issues[0] if fixable_issues else {}
    if first.get("classification") == "easiness_too_prescriptive":
        grep_read = int(first.get("_easiness_grep_read", 0) or 0)
        easiness_reason_str = first.get("_easiness_reason") or first.get("description", "")
        triage_feedback = easiness_triage_feedback_block(
            grep_read_count=grep_read, easiness_reason=easiness_reason_str
        )
    else:
        # Reviewer path — description encoded as
        # "{reason}\n\nInstruction quote: {quote}".
        reviewer_test = first.get("test", "")
        desc_raw = first.get("description", "") or ""
        if "\n\nInstruction quote:" in desc_raw:
            reviewer_reason, _, reviewer_quote = desc_raw.partition("\n\nInstruction quote:")
        else:
            reviewer_reason, reviewer_quote = desc_raw, ""
        triage_feedback = reviewer_triage_feedback_block(
            reason=reviewer_reason, quote=reviewer_quote, test=reviewer_test
        )

    ctx = PipelineContext()
    try:
        context = await asyncio.to_thread(
            _fetch_build_context,
            task.repo,
            task.commit_sha,
            task.merge_base_sha,
            ctx,
        )
    except Exception as err:
        print(f"    ERROR: Build context fetch failed during triage regen: {type(err).__name__}: {err}")
        task.stage = Stage.NEEDS_FIX
        task.needs_human_review = True
        task.human_review_reason = f"{label} triage regen: context fetch: {err}"
        return False

    prompt = build_task_prompt(
        repo=task.repo,
        sha=task.commit_sha,
        merge_base_sha=task.merge_base_sha,
        subject=task.description,
        eval_reason=task.eval_reason,
        instruction_sketch=task.eval_instruction_sketch,
        repo_map=context["repo_map"],
        diff=context["diff"],
        reference_test_bodies=context["reference_test_bodies"],
        instruction_template=context["instruction_template"],
        instruction_example=context["instruction_example"],
        alignment_feedback=task.alignment_feedback,
        triage_feedback=triage_feedback,
    )

    try:
        judge_result = await llm_judge.judge(prompt=prompt, schema=BUILD_SCHEMA, model=_cfg.LLM_STEP_MODEL)
    except Exception as err:
        print(f"    ERROR: Build regen LLM failed: {type(err).__name__}: {err}")
        task.stage = Stage.NEEDS_FIX
        task.needs_human_review = True
        task.human_review_reason = f"{label} triage regen: build LLM: {err}"
        return False

    output = judge_result.result
    instruction_md = output.get("instruction_md", "") if isinstance(output, dict) else ""
    if not instruction_md or not task.task_dir:
        print("    ERROR: Build regen returned empty instruction_md")
        task.stage = Stage.NEEDS_FIX
        task.needs_human_review = True
        task.human_review_reason = f"{label} triage regen: empty instruction"
        return False

    instruction_path = os.path.join(task.task_dir, "instruction.md")
    with open(instruction_path, "w") as f:
        f.write(instruction_md)
    strip_instruction_boilerplate(instruction_path)

    task.triage_regen_count += 1
    task.llm_usage.setdefault("build_regen", []).append(
        {
            "tokens_in": judge_result.usage.get("input_tokens", 0),
            "tokens_out": judge_result.usage.get("output_tokens", 0),
            "tokens_cached": judge_result.usage.get("cached_tokens", 0),
            "model": judge_result.model,
            "latency_s": round(judge_result.latency_s, 3),
        }
    )
    print(
        f"    -> Build regen complete (regen #{task.triage_regen_count}/"
        f"{_cfg.MAX_TRIAGE_REGENS}), routing back through alignment"
    )
    return True


async def _run_fairness_review_one(task, deep_dive_context: dict, label: str) -> dict | None:
    """Cross-family fairness-review step.

    Runs in parallel with the Opus deep-dive (skip/keep) and asks GPT-5.x
    one question at the task level: does the trial failure stem from an
    unfair instruction under-specification? Returns the parsed JSON
    result or ``None`` on failure (caller treats `None` as no concern).

    Severity enum: `none` / `minor` / `major`. `major` is gated on both
    an `evidence_quote` (verbatim instruction sentence) and an
    `evidence_test` (failing reference test dependent on an unstated
    detail). Prompt is instructed to downgrade to `minor`/`none` when
    either is missing.

    Model: `LLM_ALIGNMENT_MODEL` (defaults to GPT-5.4, cross-family from
    Opus-generated instruction).
    """
    prompt = fairness_review_prompt(
        instruction_md=deep_dive_context["instruction_md"],
        reward_json=deep_dive_context["reward_json"],
        verify_output_tail=deep_dive_context["verify_output_tail"],
        postmerge_test_bodies=deep_dive_context["postmerge_test_bodies"],
        harbor_lab_errors=deep_dive_context["harbor_lab_errors"],
        harbor_lab_edits=deep_dive_context["harbor_lab_edits"],
        harbor_lab_tool_sequence=deep_dive_context["harbor_lab_tool_sequence"],
        f2p_tests=deep_dive_context["f2p_tests"],
        p2p_tests=deep_dive_context["p2p_tests"],
    )
    try:
        result = await llm_judge.judge(
            prompt=prompt, schema=FAIRNESS_REVIEW_SCHEMA, model=_cfg.LLM_ALIGNMENT_MODEL
        )
    except Exception as e:
        print(f"    WARN: fairness review failed ({type(e).__name__}: {e}); skipping")
        return None

    task.llm_usage.setdefault("fairness_review", []).append(
        {
            "tokens_in": result.usage.get("input_tokens", 0),
            "tokens_out": result.usage.get("output_tokens", 0),
            "tokens_cached": result.usage.get("cached_tokens", 0),
            "model": result.model,
            "latency_s": round(result.latency_s, 3),
        }
    )
    return result.result


async def _run_triage_one(
    task,
    state: PipelineState,
    state_file: str,
    *,
    label: str,
    score_attr: str,
    trial_attr: str,
    accept_stage: Stage,
    reset_fix_budget: bool = False,
) -> None:
    """Triage a single task: Opus per-test skip/keep + GPT-5.x fairness review.

    Two parallel judges, separate questions:

    - **Opus DD** (``LLM_STEP_MODEL``): per-test verdict — ``skip`` or
      ``keep`` — on each failing reference test. Opus is trusted for
      this call (monotonic with its alignment judgment and trained on
      CRAFT's V4 rubric). Skip verdicts get written to
      ``f2p_skip.txt`` deterministically, then the trial is rescored
      against the modified skip set.
    - **Fairness review** (``LLM_ALIGNMENT_MODEL``, cross-family):
      single task-level severity verdict (``none``/``minor``/``major``).
      Only ``severity=major`` with both a verbatim instruction quote
      AND a named failing test triggers Build regen. Everything else
      sets ``reviewer_concern_flag`` (soft signal) and the task
      continues on Opus's per-test verdict.

    Reward==1 fast path: no LLM calls. Deterministic easiness check
    only, then accept.
    """
    async with _mark_in_progress(task, f"{label}_triage", state, state_file):
        score = getattr(task, score_attr, "?")
        print(f"  [{label}_triage] {task.task_dir} ({label}: {score})")

        trial_dir = getattr(task, trial_attr, "")
        if not trial_dir or not os.path.isdir(trial_dir):
            print("    -> No trial directory, skipping deep dive")
            task.stage = Stage.NEEDS_FIX
            task.needs_human_review = True
            task.human_review_reason = f"No {label} trial directory"
            await save_state_locked(state, state_file)
            return

        if reset_fix_budget:
            task.fix_attempts = 0

        # Build triage history from previous iterations
        triage_history = ""
        if task.fix_history:
            triage_history = "Previous fix attempts:\n"
            for j, h in enumerate(task.fix_history):
                text = h.get("summary", h) if isinstance(h, dict) else str(h)[:200]
                triage_history += f"  Attempt {j + 1}: {text}\n"
        if task.issues:
            prev_classifications = {}
            for iss in task.issues:
                name = iss.get("test", "?")
                cls = iss.get("classification", "?")
                prev_classifications[name] = cls
            if prev_classifications:
                triage_history += "Previous classifications:\n"
                for name, cls in prev_classifications.items():
                    triage_history += f"  {name}: {cls}\n"

        task.issues = []
        # Reset per-pass reviewer fields so stale values don't leak between
        # retries on the same task.
        task.reviewer_concern_flag = False
        task.reviewer_concern_severity = ""
        task.reviewer_concern_reason = ""
        task.reviewer_concern_evidence_quote = ""
        task.reviewer_concern_evidence_test = ""
        task.dd_failure_count = 0
        task.dd_dropped_by_skip_filter = 0
        task.dd_dropped_by_reward_filter = 0
        task.in_progress_step = f"{label}_deep_dive"
        await save_state_locked(state, state_file)

        deep_dive_context = await _fetch_deep_dive_context(task.task_dir, trial_dir)

        # Reward==1 fast path: nothing to triage. Run deterministic
        # easiness only (trajectory counts, no LLM) and accept.
        easiness_flag, easiness_reason = _deterministic_easiness(
            deep_dive_context.get("harbor_lab_tool_sequence_full")
            or deep_dive_context["harbor_lab_tool_sequence"],
            deep_dive_context["reward_json"],
        )
        task.easiness_flag = easiness_flag
        task.easiness_reason = easiness_reason
        try:
            reward_value = (
                float(json.loads(deep_dive_context["reward_json"]).get("reward", 0.0))
                if deep_dive_context["reward_json"]
                else None
            )
        except (ValueError, json.JSONDecodeError):
            reward_value = None
        if reward_value is not None and reward_value >= 1.0:
            # On reward=1 + easiness_flag, prefer one shot at rewriting
            # the instruction at a higher level of abstraction before
            # accepting. The agent solved with <= _EASINESS_GREP_READ_MAX
            # exploration calls, which usually means the instruction
            # named the file/class/fix strategy directly. Mirror the
            # reviewer-regen path: build a fixable_issues wrapper with
            # easiness-specific framing, route through
            # _run_triage_build_regen. If the regen budget is exhausted
            # (second pass still too easy), shelve as NEEDS_FIX — the
            # task is structurally too easy, not just phrased too
            # prescriptively.
            if easiness_flag:
                if task.triage_regen_count < _cfg.MAX_TRIAGE_REGENS:
                    print(
                        f"    -> Easiness flag + reward=1.0; routing to Build regen with "
                        f"prescriptive-instruction feedback "
                        f"(regen {task.triage_regen_count + 1}/{_cfg.MAX_TRIAGE_REGENS})"
                    )
                    # Parse grep_read count from reason (e.g. "grep_read=1 (<=5)")
                    grep_read_count = 0
                    try:
                        import re as _re

                        m = _re.search(r"grep_read=(\d+)", easiness_reason or "")
                        if m:
                            grep_read_count = int(m.group(1))
                    except Exception:
                        pass
                    fixable_issues = [
                        {
                            "type": "easiness_too_prescriptive",
                            "test": "(easiness — no specific test)",
                            "description": easiness_reason or f"grep_read={grep_read_count}",
                            "classification": "easiness_too_prescriptive",
                            "_easiness_grep_read": grep_read_count,
                            "_easiness_reason": easiness_reason,
                        }
                    ]
                    task.in_progress_step = f"{label}_build_regen"
                    await save_state_locked(state, state_file)
                    regen_ok = await _run_triage_build_regen(task, label, fixable_issues)
                    if regen_ok:
                        task.pending_fix_type = "instruction"
                        task.stage = Stage.BUILT
                    task.iteration_log.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "step": f"{label}_build_regen",
                            "regen_attempt": task.triage_regen_count,
                            "regen_ok": regen_ok,
                            "trigger": "easiness",
                            "easiness_reason": easiness_reason,
                            "opus_score": getattr(task, score_attr, ""),
                        }
                    )
                    await save_state_locked(state, state_file)
                    return
                # Regen budget exhausted; task is structurally too easy.
                task.stage = Stage.NEEDS_FIX
                task.needs_human_review = True
                task.human_review_reason = f"easiness=too_easy ({easiness_reason}); regen budget exhausted"
                print(f"    -> NEEDS_FIX: easiness flag persisted after regen ({easiness_reason})")
                task.iteration_log.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": f"{label}_too_easy_post_regen",
                        "opus_score": task.opus_score,
                        "easiness_flag": easiness_flag,
                        "easiness_reason": easiness_reason,
                    }
                )
                await save_state_locked(state, state_file)
                return

            # reward=1, no easiness flag → clean accept.
            task.stage = accept_stage
            print(
                f"    -> {accept_stage.value.upper()} "
                f"({label}={getattr(task, score_attr, '?')}, reward=1.0 — no failures to triage)"
            )
            task.iteration_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "step": f"{label}_accepted_reward_1",
                    "opus_score": task.opus_score,
                    "easiness_flag": easiness_flag,
                    "easiness_reason": easiness_reason,
                }
            )
            await save_state_locked(state, state_file)
            return

        # Run Opus per-test DD and cross-family fairness review in parallel.
        dd_prompt = deep_dive_prompt(
            instruction_md=deep_dive_context["instruction_md"],
            reward_json=deep_dive_context["reward_json"],
            verify_output_tail=deep_dive_context["verify_output_tail"],
            postmerge_test_bodies=deep_dive_context["postmerge_test_bodies"],
            harbor_lab_errors=deep_dive_context["harbor_lab_errors"],
            harbor_lab_edits=deep_dive_context["harbor_lab_edits"],
            harbor_lab_tool_sequence=deep_dive_context["harbor_lab_tool_sequence"],
            harbor_lab_metrics=deep_dive_context["harbor_lab_metrics"],
            f2p_tests=deep_dive_context["f2p_tests"],
            p2p_tests=deep_dive_context["p2p_tests"],
            f2p_skip=deep_dive_context["f2p_skip"],
            p2p_skip=deep_dive_context["p2p_skip"],
            triage_history=triage_history,
        )
        try:
            dd_judge, reviewer_result = await asyncio.gather(
                llm_judge.judge(prompt=dd_prompt, schema=DEEP_DIVE_SCHEMA, model=_cfg.LLM_STEP_MODEL),
                _run_fairness_review_one(task, deep_dive_context, label),
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"    ERROR: {err}")
            task.stage = Stage.NEEDS_FIX
            task.needs_human_review = True
            task.human_review_reason = f"{label} deep dive failed: {err}"
            await save_state_locked(state, state_file)
            return

        task.llm_usage.setdefault(f"deep_dive_{label.lower()}", []).append(
            {
                "tokens_in": dd_judge.usage.get("input_tokens", 0),
                "tokens_out": dd_judge.usage.get("output_tokens", 0),
                "tokens_cached": dd_judge.usage.get("cached_tokens", 0),
                "model": dd_judge.model,
                "latency_s": round(dd_judge.latency_s, 3),
            }
        )

        output = dict(dd_judge.result)
        raw_failures = output.get("failures", [])
        task.dd_failure_count = len(raw_failures)

        # Populate reviewer_concern_* fields. The flag is set whenever the
        # review emits anything other than `none`; the regen trigger
        # (below) fires only when severity=major with BOTH evidence
        # fields populated.
        reviewer_severity = ""
        reviewer_reason = ""
        reviewer_quote = ""
        reviewer_test = ""
        if reviewer_result is not None:
            reviewer_severity = reviewer_result.get("severity", "") or ""
            reviewer_reason = reviewer_result.get("reason", "") or ""
            reviewer_quote = (reviewer_result.get("evidence_quote", "") or "").strip()
            reviewer_test = (reviewer_result.get("evidence_test", "") or "").strip()
        task.reviewer_concern_severity = reviewer_severity
        task.reviewer_concern_reason = reviewer_reason
        task.reviewer_concern_evidence_quote = reviewer_quote
        task.reviewer_concern_evidence_test = reviewer_test
        task.reviewer_concern_flag = reviewer_severity not in ("", "none")
        reviewer_triggers_regen = (
            reviewer_severity == "major" and bool(reviewer_quote) and bool(reviewer_test)
        )

        # Drop DD verdicts on tests already excluded via f2p_skip.txt /
        # p2p_skip.txt — the verifier still runs them and pytest still
        # reports FAILED, so the judge keeps flagging them every
        # iteration. Dropping here breaks the loop.
        already_skipped = _load_skipped_tests(task.task_dir)
        if already_skipped:
            failures = [f for f in raw_failures if f.get("test_name") not in already_skipped]
            dropped = len(raw_failures) - len(failures)
            task.dd_dropped_by_skip_filter = dropped
            print(
                f"    -> Skip-filter: {len(raw_failures)} raw -> {len(failures)} after filter "
                f"({dropped} already in skip files, {len(already_skipped)} skip entries loaded)"
            )
        else:
            failures = raw_failures
            print(f"    -> Skip-filter: {len(raw_failures)} raw (no skip files yet)")

        # Drop verdicts on tests that didn't actually fail in this trial.
        # The DD prompt tells the judge to only classify failing tests,
        # but the judge has been observed classifying passing tests
        # whose bodies look "unfair" per the rubric. Cross-check against
        # reward.json as the authoritative list of actually-failed tests.
        actually_failed = _load_actually_failed_tests(trial_dir)
        if actually_failed is not None:
            before = len(failures)
            failures = [f for f in failures if f.get("test_name", "") in actually_failed]
            dropped = before - len(failures)
            task.dd_dropped_by_reward_filter = dropped
            if dropped > 0:
                print(
                    f"    -> Reward.json filter: dropped {dropped} verdict(s) on tests that actually passed"
                )

        output["failures"] = failures

        # Log DD findings + reviewer verdict
        task.iteration_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": f"{label}_deep_dive",
                "assessment": output.get("overall_assessment", ""),
                "failures": [
                    {"test": f.get("test_name", ""), "classification": f.get("classification", "")}
                    for f in failures
                ],
                "opus_score": task.opus_score,
                "dd_failure_count": task.dd_failure_count,
                "dropped_by_skip_filter": task.dd_dropped_by_skip_filter,
                "dropped_by_reward_filter": task.dd_dropped_by_reward_filter,
                "reviewer_severity": task.reviewer_concern_severity,
                "reviewer_concern_flag": task.reviewer_concern_flag,
                "reviewer_triggers_regen": reviewer_triggers_regen,
                "easiness_flag": task.easiness_flag,
                "easiness_reason": task.easiness_reason,
            }
        )

        # Write Opus DD diagnostic
        if task.task_dir:
            diag_content = (
                f"# {label} Deep Dive (Opus skip/keep)\n\n"
                f"**Score:** {score}\n"
                f"**Assessment:** {output.get('overall_assessment', '')}\n\n"
                f"## Verdicts\n\n"
            )
            for f in failures:
                diag_content += (
                    f"### {f.get('test_name', '?')}\n"
                    f"- **Verdict:** {f.get('classification', '?')}\n"
                    f"- **Evidence:** {f.get('evidence', 'none')}\n\n"
                )
            diag_path = _next_diagnostic_path(task.task_dir, f"triage_{label.lower()}")
            _write_diagnostic(diag_path, diag_content)

            # Write fairness-review diagnostic (whether or not it triggered)
            if reviewer_result is not None:
                review_content = (
                    f"# {label} Fairness Review\n\n"
                    f"**Model:** `{_cfg.LLM_ALIGNMENT_MODEL}`\n"
                    f"**Severity:** `{reviewer_severity or '(none reported)'}`\n"
                    f"**Auto-regen triggered:** {reviewer_triggers_regen}\n\n"
                    f"## Reason\n\n{reviewer_reason or '(empty)'}\n\n"
                    f"## Evidence quote\n\n{reviewer_quote or '(empty)'}\n\n"
                    f"## Evidence test\n\n{reviewer_test or '(empty)'}\n"
                )
                review_path = _next_diagnostic_path(task.task_dir, f"fairness_review_{label.lower()}")
                _write_diagnostic(review_path, review_content)

        # If reviewer has a strong well-evidenced claim that the instruction
        # is unfair, prefer Build regen over per-test skips. Rationale: when
        # the reviewer can produce a verbatim quote + named failing test,
        # the root cause is under-specification; skipping tests is a bandaid
        # that masks the instruction-level issue. Falls back to skip path on
        # the next triage pass if regen hits the MAX_TRIAGE_REGENS cap.
        if reviewer_triggers_regen:
            print(
                f"    -> Reviewer severity=major + evidence; routing to Build regen "
                f"(regen {task.triage_regen_count + 1}/{_cfg.MAX_TRIAGE_REGENS})"
            )
            fixable_issues = [
                {
                    "type": "reviewer_unfairness",
                    "test": reviewer_test,
                    "description": f"{reviewer_reason}\n\nInstruction quote: {reviewer_quote}",
                    "classification": "reviewer_unfairness",
                }
            ]
            task.in_progress_step = f"{label}_build_regen"
            await save_state_locked(state, state_file)
            regen_ok = await _run_triage_build_regen(task, label, fixable_issues)
            if regen_ok:
                task.pending_fix_type = "instruction"
                task.stage = Stage.BUILT
            task.iteration_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "step": f"{label}_build_regen",
                    "regen_attempt": task.triage_regen_count,
                    "regen_ok": regen_ok,
                    "reviewer_severity": reviewer_severity,
                    "reviewer_evidence_quote": reviewer_quote[:160],
                    "reviewer_evidence_test": reviewer_test,
                    "opus_score": getattr(task, score_attr, ""),
                }
            )
            await save_state_locked(state, state_file)
            return

        # No reviewer-driven regen — partition DD verdicts into skip / keep
        # buckets for the per-test action path.
        skip_tests: list[dict] = []
        keep_tests: list[dict] = []
        for failure in failures:
            classification = failure.get("classification", "keep")
            task.issues.append(
                {
                    "type": classification,
                    "test": failure.get("test_name", ""),
                    "description": failure.get("evidence", ""),
                    "classification": classification,
                }
            )
            if classification == "skip":
                skip_tests.append(failure)
            else:
                keep_tests.append(failure)

        # Thin-F2P guard runs BEFORE the skip writes: if applying all
        # skip verdicts would leave ≤ 1 F2P test, reject outright rather
        # than write the skip file and accept a thinly-scored task.
        # Placing this check after a rescore-accept branch would let a
        # reward=1.0 post-skip result slip through with one scored F2P.
        if skip_tests and task.f2p_tests:
            skippable_f2p = sum(1 for f in skip_tests if f.get("test_name", "") in task.f2p_tests)
            remaining_f2p = len(task.f2p_tests) - skippable_f2p
            if remaining_f2p <= 1:
                task.stage = Stage.REJECTED
                task.needs_human_review = True
                task.human_review_reason = (
                    f"Only {remaining_f2p} F2P test(s) would remain after skipping "
                    f"{skippable_f2p} unfair tests — too thin to keep"
                )
                print(
                    f"    -> REJECTED: {remaining_f2p} F2P remaining after "
                    f"skipping {skippable_f2p} unfair (need >= 2)"
                )
                task.iteration_log.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": f"{label}_rejected_thin_f2p",
                        "f2p_total": len(task.f2p_tests),
                        "f2p_skippable": skippable_f2p,
                        "f2p_remaining": remaining_f2p,
                    }
                )
                await save_state_locked(state, state_file)
                return

        # Apply Opus's skip verdicts — deterministic, per-test, cheap.
        # Append to f2p_skip.txt (never touches postmerge_tests/ or source,
        # so the change survives regeneration).
        if skip_tests and task.task_dir:
            skip_file = os.path.join(task.task_dir, "tests", "f2p_skip.txt")
            os.makedirs(os.path.dirname(skip_file), exist_ok=True)
            existing: set[str] = set()
            if os.path.isfile(skip_file):
                with open(skip_file) as sf:
                    for line in sf:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            existing.add(line.split("|")[0].strip())
            new_lines: list[str] = []
            f2p_set = set(task.f2p_tests or [])
            for f in skip_tests:
                tname = f.get("test_name", "")
                if not tname or tname in existing:
                    continue
                if f2p_set and tname not in f2p_set:
                    continue  # only auto-skip F2P tests; P2P skips are unsafe
                evidence = (f.get("evidence") or "")[:100].replace("\n", " ")
                new_lines.append(f"{tname} | skip: {evidence}")
                existing.add(tname)
            if new_lines:
                with open(skip_file, "a") as sf:
                    if existing - set(new_lines):  # file had prior entries
                        sf.write("\n")
                    sf.write("\n".join(new_lines) + "\n")
                print(f"    -> Auto-wrote {len(new_lines)} test(s) to f2p_skip.txt (Opus skip verdict)")

        # Re-score the trial against the updated skip set if there were
        # any new skip verdicts. If reward reaches 1.0 after skipping,
        # the task accepts outright — reviewer's concern becomes moot
        # (though `reviewer_concern_flag` remains populated as a soft
        # signal for human batch review).
        if skip_tests and task.task_dir and _rescore_trial(task.task_dir, trial_dir):
            reward_path = os.path.join(trial_dir, "verifier", "reward.json")
            try:
                with open(reward_path) as rf:
                    reward_data = json.load(rf)
                f2p_p = reward_data.get("f2p_passed", "?")
                f2p_t = reward_data.get("f2p_total", "?")
                p2p_p = reward_data.get("p2p_passed", "?")
                p2p_t = reward_data.get("p2p_total", "?")
                new_score = f"F2P {f2p_p}/{f2p_t}, P2P {p2p_p}/{p2p_t}"
                setattr(task, score_attr, new_score)
                new_reward = reward_data.get("reward", 0.0)
                skipped = reward_data.get("f2p_skipped", 0)
                print(
                    f"    -> Auto-skip re-score: {new_score}, reward={new_reward}, {skipped} test(s) skipped"
                )
                if new_reward == 1.0:
                    task.stage = accept_stage
                    print(f"    -> {accept_stage.value.upper()} after auto-skip re-score")
                    task.iteration_log.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "step": f"{label}_accepted",
                            "opus_score": task.opus_score,
                            "skipped_tests": [f.get("test_name", "") for f in skip_tests],
                            "reviewer_concern_flag": task.reviewer_concern_flag,
                            "reviewer_severity": task.reviewer_concern_severity,
                        }
                    )
                    await save_state_locked(state, state_file)
                    return
            except (OSError, json.JSONDecodeError, KeyError) as e:
                print(f"    -> Re-score read failed: {e}, continuing")

        # (Thin-F2P guard already ran before the skip writes above. Reaching
        # this point means skipping leaves > 1 F2P and the rescore either
        # didn't run or didn't reach reward=1.0.)

        # Skips didn't resolve the failure, F2P is thick enough, and the
        # reviewer either didn't fire or fired without enough evidence to
        # trigger a regen. Accept: `keep` verdicts are genuine capability
        # gaps, task discriminates.
        task.stage = accept_stage
        print(
            f"    -> {accept_stage.value.upper()} "
            f"({label}={getattr(task, score_attr, '?')}, "
            f"{len(skip_tests)} skipped / {len(keep_tests)} kept as genuine)"
        )
        task.iteration_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": f"{label}_accepted",
                "opus_score": task.opus_score,
                "issues": [
                    {"test": iss.get("test", ""), "classification": iss.get("classification", "")}
                    for iss in task.issues
                ],
                "reviewer_concern_flag": task.reviewer_concern_flag,
                "reviewer_severity": task.reviewer_concern_severity,
            }
        )
        await save_state_locked(state, state_file)


async def _step_triage(
    state: PipelineState,
    state_file: str,
    *,
    from_stage: Stage,
    label: str,
    score_attr: str,
    trial_attr: str,
    accept_stage: Stage,
    reset_fix_budget: bool = False,
    concurrency: int = 4,
) -> None:
    """Generic triage step — async with semaphore. Deep-dive trials, classify failures, auto-fix."""
    tasks = [t for t in state.tasks.values() if t.stage == from_stage]
    if not tasks:
        print(f"No {from_stage.value} tasks to triage.")
        return

    sem = asyncio.Semaphore(concurrency)

    async def _wrap(task):
        async with sem:
            await _run_triage_one(
                task,
                state,
                state_file,
                label=label,
                score_attr=score_attr,
                trial_attr=trial_attr,
                accept_stage=accept_stage,
                reset_fix_budget=reset_fix_budget,
            )

    print(f"{label}-triaging {len(tasks)} tasks (concurrency={concurrency})...")
    await asyncio.gather(*[_wrap(t) for t in tasks])


# ---------------------------------------------------------------------------
# Per-task independent pipeline
# ---------------------------------------------------------------------------


def _compare_and_accept(task) -> None:
    """Finalize a task after Opus triage. No Haiku comparison (step dropped) —
    just set ACCEPTED and surface any soft flags the reviewer raised.

    Prior versions ran Haiku smoke + rank-inversion / both-zero gates here.
    Apr 17 2026 cohort (43 tasks, ~500 cumulative across all bulk runs) fired
    0 rank-inversions. Dropped as paying ~20 min/task for an unused gate.
    """
    task.stage = Stage.ACCEPTED
    if task.easiness_flag:
        print(f"    -> FLAG: easiness concern (deterministic) — {task.easiness_reason}")
    print(f"    -> ACCEPTED (Opus={task.opus_score})")

    task.iteration_log.append(
        {
            "timestamp": datetime.now().isoformat(),
            "step": "comparison",
            "opus_score": task.opus_score,
            "outcome": task.stage.value,
        }
    )


async def run_task_pipeline(
    task,
    state: PipelineState,
    state_file: str,
    sems: dict[str, asyncio.Semaphore],
) -> None:
    """Run a single task through the full pipeline independently (build through accept).

    Wraps the stage loop in a nested helper + try/except so that per-task
    docker images are removed once the task reaches a terminal stage
    (ACCEPTED / REJECTED / NEEDS_FIX) or the coroutine crashes (the outer
    gather catches the exception and marks NEEDS_FIX — same effective
    outcome). Cancellation (pipeline teardown) skips cleanup so resumed runs
    keep image caches.
    """
    from craft_taskgen.config import MAX_FIX_ATTEMPTS
    from craft_taskgen.docker import cleanup_task_images

    max_iterations = MAX_FIX_ATTEMPTS + 2

    async def _pipeline_body():
        for _ in range(max_iterations):
            # Build + Alignment as a single combined step. The orchestrator
            # runs N parallel build → align → (regen-once) candidate loops
            # for fresh entries (PROMISING/EVALUATED), or alignment-only
            # for post-triage entries (Stage.BUILT — see below).
            if task.stage in (Stage.PROMISING, Stage.EVALUATED):
                await _run_build_align_candidates(task, state, state_file, sems)
                if task.stage != Stage.ALIGNMENT_CHECKED:
                    # Terminal stage already set by orchestrator (REJECTED or
                    # NEEDS_FIX). For NEEDS_FIX caused by build infra failures,
                    # bump fix_attempts so MAX_FIX_ATTEMPTS still gates retries.
                    if task.stage == Stage.NEEDS_FIX:
                        task.fix_attempts = (task.fix_attempts or 0) + 1
                    return

            if task.stage == Stage.BUILT:
                # Triage-induced Build regen path: triage's
                # _run_triage_build_regen produced a single new instruction
                # in task.task_dir; run alignment-only, no fanout.
                await _run_alignment_only_for_triage(task, state, state_file, sems)
                if task.stage != Stage.ALIGNMENT_CHECKED:
                    return

            if task.stage == Stage.ALIGNMENT_CHECKED:
                if task.pending_fix_type == "instruction" and os.path.isfile(
                    os.path.join(task.task_dir, "task.toml")
                ):
                    # Instruction-only fix: alignment validated the new instruction.
                    # Skip find_tests/dockerfile/classify/oracle — all produce identical
                    # results since git diff, Dockerfile, and solve.sh are unchanged.
                    # Guard: only fast-path if artifacts already exist on disk.
                    task.pending_fix_type = ""
                    task.stage = Stage.ORACLE_CHECKED
                else:
                    async with sems["docker"]:
                        await _run_assemble_task_dir_artifacts_one(task, state, state_file)
                    if task.stage != Stage.TESTS_DISCOVERED:
                        return

            if task.stage == Stage.TESTS_DISCOVERED:
                async with sems["llm"]:
                    await _run_build_dockerfile_one(task, state, state_file)
                if task.stage != Stage.DOCKERFILE_BUILT:
                    return

            if task.stage == Stage.DOCKERFILE_BUILT:
                async with sems["docker"]:
                    await _run_docker_classify_one(task, state, state_file)
                if task.stage != Stage.F2P_P2P_CLASSIFIED:
                    return

            if task.stage == Stage.F2P_P2P_CLASSIFIED:
                async with sems["docker"]:
                    await _run_oracle_check_one(task, state, state_file)
                if task.stage != Stage.ORACLE_CHECKED:
                    return

            # Opus first — primary quality gate with fix loop
            if task.stage == Stage.ORACLE_CHECKED:
                async with sems["smoke"]:
                    await _run_smoke_one(
                        task,
                        state,
                        state_file,
                        model=_cfg.SMOKE_MODEL,
                        label="Opus",
                        score_attr="opus_score",
                        trial_attr="opus_trial_dir",
                        next_stage=Stage.OPUS_SMOKE_TESTED,
                        agent=_cfg.SMOKE_AGENT,
                        reasoning_effort=_cfg.SMOKE_REASONING_EFFORT,
                    )
                if task.stage != Stage.OPUS_SMOKE_TESTED:
                    return

            if task.stage == Stage.OPUS_SMOKE_TESTED:
                task.fix_attempts = 0  # fresh budget for triage fixes
                async with sems["llm"]:
                    await _run_triage_one(
                        task,
                        state,
                        state_file,
                        label="Opus",
                        score_attr="opus_score",
                        trial_attr="opus_trial_dir",
                        accept_stage=Stage.OPUS_TRIAGED,
                    )
                if task.stage in (Stage.BUILT, Stage.ORACLE_CHECKED, Stage.ALIGNMENT_CHECKED):
                    continue  # triage set a fix stage — restart loop
                if task.stage != Stage.OPUS_TRIAGED:
                    return

            # Finalize — Haiku step was dropped (0 rank-inversions observed on the
            # Apr 17 2026 cohort; efficiency heuristics cover "too easy").
            if task.stage == Stage.OPUS_TRIAGED:
                _compare_and_accept(task)
                await save_state_locked(state, state_file)
                await _generate_summary(task, state, state_file, sems["llm"])
                return

            return  # unexpected stage

    try:
        await _pipeline_body()
    except asyncio.CancelledError:
        raise  # pipeline teardown — preserve image caches for resume
    except Exception:
        # pipeline.py's gather(return_exceptions=True) will mark this NEEDS_FIX.
        await cleanup_task_images(task.task_dir)
        raise

    if task.stage in (Stage.ACCEPTED, Stage.REJECTED, Stage.NEEDS_FIX):
        await cleanup_task_images(task.task_dir)


_SUMMARY_DIAG_PER_FILE_CAP_CHARS = 3000
_SUMMARY_DIAG_TOTAL_CAP_CHARS = 40_000


async def _generate_summary(
    task,
    state: PipelineState,
    state_file: str,
    sem: asyncio.Semaphore,
) -> None:
    """Generate a narrative summary from the diagnostics/ directory.

    Direct-API: reads every diagnostic file, caps each to keep the prompt
    bounded (summary context can grow large on deeply-iterated tasks),
    then dispatches to llm_judge.judge.
    """
    diag_dir = os.path.join(task.task_dir, "diagnostics") if task.task_dir else ""
    if not diag_dir or not os.path.isdir(diag_dir):
        return

    md_files = sorted(f for f in os.listdir(diag_dir) if f.endswith(".md"))
    if not md_files:
        return

    def _load_diagnostics() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        remaining = _SUMMARY_DIAG_TOTAL_CAP_CHARS
        for name in md_files:
            if remaining <= 0:
                break
            try:
                body = Path(diag_dir, name).read_text(errors="replace")
            except OSError:
                continue
            if len(body) > _SUMMARY_DIAG_PER_FILE_CAP_CHARS:
                body = body[:_SUMMARY_DIAG_PER_FILE_CAP_CHARS] + "\n[truncated]"
            if len(body) > remaining:
                body = body[:remaining] + "\n[truncated]"
            out.append((name, body))
            remaining -= len(body)
        return out

    diagnostics = await asyncio.to_thread(_load_diagnostics)

    from craft_taskgen.prompts import SUMMARY_SCHEMA, task_summary_prompt

    prompt = task_summary_prompt(diagnostics, task.stage.value)

    async with sem:
        try:
            judge_result = await llm_judge.judge(
                prompt=prompt,
                schema=SUMMARY_SCHEMA,
                model=_cfg.LLM_STEP_MODEL,
                max_tokens=256,
            )
        except Exception as e:
            print(f"    -> Summary ERROR ({type(e).__name__}: {e})")
            return

    summary_text = (judge_result.result.get("summary") or "").strip()
    if summary_text:
        task.summary = summary_text
        task.llm_usage.setdefault("summary", []).append(
            {
                "tokens_in": judge_result.usage.get("input_tokens", 0),
                "tokens_out": judge_result.usage.get("output_tokens", 0),
                "tokens_cached": judge_result.usage.get("cached_tokens", 0),
                "model": judge_result.model,
                "latency_s": round(judge_result.latency_s, 3),
            }
        )
        # Write to diagnostics as the final file
        diag_path = _next_diagnostic_path(task.task_dir, "summary")
        _write_diagnostic(diag_path, f"# Task Summary\n\n{summary_text}\n")
        print(f"    -> Summary: {summary_text[:100]}...")
        await save_state_locked(state, state_file)


# ---------------------------------------------------------------------------
# Concrete step wrappers
# ---------------------------------------------------------------------------


async def step_opus_smoke(state: PipelineState, state_file: str, concurrency: int = 4) -> None:
    """Agent smoke test — primary quality gate (configurable agent/model)."""
    await _step_smoke(
        state,
        state_file,
        from_stage=Stage.ORACLE_CHECKED,
        model=_cfg.SMOKE_MODEL,
        label="Opus",
        score_attr="opus_score",
        trial_attr="opus_trial_dir",
        next_stage=Stage.OPUS_SMOKE_TESTED,
        concurrency=concurrency,
        agent=_cfg.SMOKE_AGENT,
        reasoning_effort=_cfg.SMOKE_REASONING_EFFORT,
    )


async def step_opus_triage(state: PipelineState, state_file: str, concurrency: int = 4) -> None:
    """Opus triage — deep dive + reviewer with fix loop."""
    await _step_triage(
        state,
        state_file,
        from_stage=Stage.OPUS_SMOKE_TESTED,
        label="Opus",
        score_attr="opus_score",
        trial_attr="opus_trial_dir",
        accept_stage=Stage.OPUS_TRIAGED,
        reset_fix_budget=True,
        concurrency=concurrency,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def step_report(state: PipelineState) -> None:
    """Print summary report of pipeline state."""
    stages: dict[str, int] = {}
    for task in state.tasks.values():
        stage_name = task.stage.value
        stages[stage_name] = stages.get(stage_name, 0) + 1

    print("=" * 60)
    print("Pipeline Report")
    print("=" * 60)
    print(f"Total tasks: {len(state.tasks)}")
    print()
    for stage_name in [
        "accepted",
        "opus_triaged",
        "opus_smoke_tested",
        "needs_fix",
        "rejected",
        "alignment_checked",
        "built",
        "promising",
        "evaluated",
        "candidate",
    ]:
        count = stages.get(stage_name, 0)
        if count > 0:
            print(f"  {stage_name.upper():20s}: {count}")
    print()

    accepted = [t for t in state.tasks.values() if t.stage == Stage.ACCEPTED]
    if accepted:
        print("ACCEPTED tasks:")
        for t in accepted:
            print(f"  {t.task_dir or t.task_id:50s} Opus={t.opus_score}")
        print()

    flagged = [t for t in accepted if t.easiness_flag]
    if flagged:
        print(f"  NOTE: {len(flagged)} accepted tasks flagged for easiness review:")
        for t in flagged:
            print(f"    {t.task_id}: {t.easiness_reason}")
        print()

    needs_fix = [t for t in state.tasks.values() if t.stage == Stage.NEEDS_FIX]
    if needs_fix:
        print("NEEDS_FIX (human review required):")
        for t in needs_fix:
            print(f"  {t.task_id:20s} {t.human_review_reason[:80]}")
        print()
