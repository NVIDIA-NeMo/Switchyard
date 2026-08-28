# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest SWE-Bench-Pro public PRs into the calibrate-alignment.py CSV format.

Pulls instances from the HuggingFace dataset ``ScaleAI/SWE-bench_Pro`` (public
test split, 731 instances across 11 repos), filters by ``--repos``, samples up
to ``--limit-per-repo`` per repo, and for each instance synthesizes a local
commit by applying SWE-Pro's ``patch`` + ``test_patch`` onto ``base_commit``.

The synthesized commit gives us a real SHA the pipeline can ``git diff``
against (calibrate-alignment.py's build context fetcher uses
``git diff merge_base..sha``). We use a per-instance scratch branch under
``swepro-staging/`` so we can rerun without polluting upstream branches.

Output CSV columns match what calibrate-alignment.py expects in ``--mode full``:

    task_id, repo, commit_sha, base_sha, merge_base_sha, subject, pr_url,
    instruction_md, hardness_verdict

``instruction_md`` is left blank (build will produce a fresh one).
``hardness_verdict`` is set to "unknown" so the stratified sampler doesn't
explode on missing buckets.

Usage:
    uv run python scripts/swepro/ingest.py \\
        --repos qutebrowser \\
        --limit-per-repo 10 \\
        --output swepro_input.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

REPOS_DIR = Path("repos")
SCRATCH_BRANCH_PREFIX = "swepro-staging/"


def _run_git(
    repo_path: Path,
    *args: str,
    check: bool = True,
    input_data: bytes | None = None,
) -> tuple[int, str, str]:
    """Run a git command; return (returncode, stdout, stderr)."""
    res = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        input=input_data,
    )
    if check and res.returncode != 0:
        sys.stderr.write(
            f"git {' '.join(args)} failed in {repo_path}:\n"
            f"  stdout: {res.stdout.decode('utf-8', errors='replace')[:300]}\n"
            f"  stderr: {res.stderr.decode('utf-8', errors='replace')[:300]}\n"
        )
        raise subprocess.CalledProcessError(res.returncode, ["git", *args])
    return (
        res.returncode,
        res.stdout.decode("utf-8", errors="replace"),
        res.stderr.decode("utf-8", errors="replace"),
    )


def _short_repo(swepro_repo: str) -> str:
    """Map ``owner/name`` → ``name`` (our local clones drop the owner prefix)."""
    return swepro_repo.split("/", 1)[1] if "/" in swepro_repo else swepro_repo


def _ensure_base_commit_present(repo_path: Path, base_commit: str) -> bool:
    rc, _, _ = _run_git(repo_path, "cat-file", "-e", f"{base_commit}^{{commit}}", check=False)
    return rc == 0


def _apply_patch(repo_path: Path, patch_text: str) -> tuple[bool, str]:
    """Apply a unified diff with ``git apply --3way``. Returns (ok, error_msg)."""
    if not patch_text.strip():
        return True, ""
    res = subprocess.run(
        ["git", "-C", str(repo_path), "apply", "--3way", "--allow-empty", "--whitespace=nowarn", "-"],
        input=patch_text.encode("utf-8"),
        capture_output=True,
    )
    if res.returncode != 0:
        return False, res.stderr.decode("utf-8", errors="replace")[:500]
    return True, ""


def _stage_and_commit(repo_path: Path, message: str) -> str:
    """Stage all changes, commit with ``message``. Returns the new HEAD SHA."""
    _run_git(repo_path, "add", "-A")
    # Allow empty in case the patch + test_patch combined into a no-op (shouldn't happen but defensive).
    _run_git(
        repo_path,
        "-c",
        "user.email=swepro-ingest@local",
        "-c",
        "user.name=swepro-ingest",
        "commit",
        "--allow-empty",
        "-m",
        message,
    )
    _, sha, _ = _run_git(repo_path, "rev-parse", "HEAD")
    return sha.strip()


def _materialize_instance(
    repo_path: Path,
    instance_id: str,
    base_commit: str,
    patch: str,
    test_patch: str,
    subject: str,
) -> tuple[str | None, str | None, str]:
    """Create a synthetic commit on a scratch branch.

    Returns (commit_sha, merge_base_sha, error). On success, error is "".
    On failure, commit_sha is None and error explains why.
    """
    if not _ensure_base_commit_present(repo_path, base_commit):
        return None, None, f"base_commit {base_commit[:10]} not present in local repo"

    branch = f"{SCRATCH_BRANCH_PREFIX}{instance_id}"
    try:
        _run_git(repo_path, "checkout", "-q", "--detach", base_commit)
    except subprocess.CalledProcessError:
        return None, None, f"failed to checkout base_commit {base_commit[:10]}"

    _run_git(repo_path, "branch", "-q", "-D", branch, check=False)
    _run_git(repo_path, "checkout", "-q", "-b", branch)

    ok, err = _apply_patch(repo_path, patch)
    if not ok:
        return None, None, f"patch apply failed: {err}"
    ok, err = _apply_patch(repo_path, test_patch)
    if not ok:
        return None, None, f"test_patch apply failed: {err}"

    new_sha = _stage_and_commit(repo_path, f"swepro:{instance_id}\n\n{subject}")
    return new_sha, base_commit, ""


def _load_dataset(name: str = "ScaleAI/SWE-bench_Pro", split: str = "test"):
    from datasets import load_dataset  # type: ignore

    return load_dataset(name, split=split)


def _derive_pr_url(swepro_repo: str, instance_id: str) -> str:
    """SWE-Pro instance_id often encodes the PR number; if it does, build a URL.

    Format conventions vary across SWE-Bench-Pro releases; common forms:
      - ``owner__repo-1234``
      - ``owner__repo_1234``
      - ``owner__repo-issue-1234``

    If we can't extract a number, we fall back to the repo URL.
    """
    base = f"https://github.com/{swepro_repo}"
    parts = instance_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return f"{base}/pull/{parts[1]}"
    parts = instance_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return f"{base}/pull/{parts[1]}"
    return base


def _derive_subject(problem_statement: str, instance_id: str) -> str:
    """Use the first non-empty informative line of problem_statement as the subject.

    SWE-Pro's problem_statement often follows a "Title: <foo>\n..." format. We
    strip a leading "Title:" label and prefer the line *after* it when the
    label appears on its own line. Falls back to the first non-empty line.
    """
    lines = [line.strip().lstrip("#").strip() for line in problem_statement.splitlines()]
    for i, s in enumerate(lines):
        if not s:
            continue
        if s.lower().startswith("title:"):
            after = s.split(":", 1)[1].strip()
            if after:
                return after[:200]
            for nxt in lines[i + 1 :]:
                if nxt:
                    return nxt[:200]
            continue
        return s[:200]
    return instance_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repos",
        default="qutebrowser",
        help=(
            "Comma-separated short repo names (e.g. 'qutebrowser,ansible'). "
            "Matched against SWE-Pro's owner/name field by suffix."
        ),
    )
    parser.add_argument(
        "--limit-per-repo",
        type=int,
        default=10,
        help="Max instances per repo to attempt ingestion for (default 10).",
    )
    parser.add_argument(
        "--output",
        default="swepro_input.csv",
        help="Output CSV path (default: swepro_input.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling within a repo (default: 42).",
    )
    parser.add_argument(
        "--dataset",
        default="ScaleAI/SWE-bench_Pro",
        help="HuggingFace dataset name (default: ScaleAI/SWE-bench_Pro).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test).",
    )
    args = parser.parse_args()

    target_repos = {r.strip() for r in args.repos.split(",") if r.strip()}
    if not target_repos:
        parser.error("--repos cannot be empty")

    if not REPOS_DIR.is_dir():
        parser.error(f"repos directory not found at {REPOS_DIR.resolve()}")
    available = set(os.listdir(REPOS_DIR))
    missing = target_repos - available
    if missing:
        sys.stderr.write(
            f"WARNING: requested repos not cloned under repos/: {sorted(missing)}. "
            f"Clone them first or remove from --repos.\n"
        )
        target_repos &= available

    if not target_repos:
        sys.stderr.write("ERROR: no requested repos are present locally; nothing to do.\n")
        sys.exit(2)

    sys.stderr.write(f"Loading {args.dataset} ({args.split} split)...\n")
    ds = _load_dataset(args.dataset, args.split)
    sys.stderr.write(f"  loaded {len(ds)} instances\n")

    import random

    rng = random.Random(args.seed)

    rows_out: list[dict] = []
    stats = {"attempted": 0, "patched": 0, "skipped_apply": 0, "skipped_base_missing": 0}
    skip_log: list[tuple[str, str]] = []

    for short in sorted(target_repos):
        per_repo = [rec for rec in ds if _short_repo(rec["repo"]) == short]
        sys.stderr.write(f"  {short}: {len(per_repo)} candidate instances in dataset\n")
        rng.shuffle(per_repo)
        repo_path = REPOS_DIR / short
        attempted_for_repo = 0

        for rec in per_repo:
            if attempted_for_repo >= args.limit_per_repo:
                break
            attempted_for_repo += 1
            stats["attempted"] += 1

            instance_id = rec["instance_id"]
            base_commit = rec["base_commit"]
            patch = rec.get("patch", "") or ""
            test_patch = rec.get("test_patch", "") or ""
            problem_statement = rec.get("problem_statement", "") or ""

            subject = _derive_subject(problem_statement, instance_id)
            pr_url = _derive_pr_url(rec["repo"], instance_id)

            commit_sha, merge_base_sha, err = _materialize_instance(
                repo_path,
                instance_id=instance_id,
                base_commit=base_commit,
                patch=patch,
                test_patch=test_patch,
                subject=subject,
            )
            if commit_sha is None:
                if "not present" in (err or ""):
                    stats["skipped_base_missing"] += 1
                else:
                    stats["skipped_apply"] += 1
                skip_log.append((instance_id, err or "unknown"))
                continue

            stats["patched"] += 1
            rows_out.append(
                {
                    "task_id": instance_id,
                    "repo": short,
                    "commit_sha": commit_sha,
                    "base_sha": merge_base_sha or "",
                    "merge_base_sha": merge_base_sha or "",
                    "subject": subject,
                    "pr_url": pr_url,
                    "instruction_md": "",
                    "hardness_verdict": "unknown",
                }
            )

    fieldnames = [
        "task_id",
        "repo",
        "commit_sha",
        "base_sha",
        "merge_base_sha",
        "subject",
        "pr_url",
        "instruction_md",
        "hardness_verdict",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    sys.stderr.write(
        f"\nWrote {len(rows_out)} rows to {args.output}\n"
        f"  attempted:            {stats['attempted']}\n"
        f"  patched successfully: {stats['patched']}\n"
        f"  base_commit missing:  {stats['skipped_base_missing']}\n"
        f"  patch apply failed:   {stats['skipped_apply']}\n"
    )
    if skip_log:
        sys.stderr.write("\nSkipped instances:\n")
        for iid, reason in skip_log[:20]:
            sys.stderr.write(f"  {iid}: {reason[:160]}\n")
        if len(skip_log) > 20:
            sys.stderr.write(f"  ... and {len(skip_log) - 20} more\n")


if __name__ == "__main__":
    main()
