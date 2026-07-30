#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mine GitHub PRs for hard tool-orchestration task candidates.

Walks a repo's merged PRs and scores them against difficulty signals
derived from our H-rule criteria (see rubrics.py) and SWE-bench Pro's methodology.

Usage:
    craft-taskgen-mine repos/dramatiq --top 20
    craft-taskgen-mine repos/scrapy --after 2025-10-01 --top 30
    craft-taskgen-mine repos/click --out candidates/click.json

Or run directly:
    python -m craft_taskgen.miner repos/dramatiq --top 20

Output format:
    {
      "repo": "repos/dramatiq",
      "after": null,
      "n_prs_scanned": 1234,
      "n_candidates": 20,
      "candidates": [
        {
          "sha": "abc123...",        # merge commit SHA (lands on main)
          "base_sha": "def456...",   # base branch HEAD before the PR
          "subject": "Add retry middleware ...",
          "author": "Alice",
          "date": "2026-01-15T10:30:00Z",
          "source_files": ["dramatiq/middleware.py", ...],
          "test_files": ["tests/test_middleware.py", ...],
          "other_files": [],
          "source_lines_changed": 147,
          "test_lines_changed": 89,
          "packages_touched": 2,
          "package_names": ["dramatiq", "examples"],
          "has_test_patch": true,
          "is_multi_file": true,
          "is_multi_package": true,
          "is_nontrivial_source": true,
          "is_nontrivial_tests": true,
          "is_refactoring": false,
          "has_iteration_signal": false,
          "score": 11.0,
          "score_breakdown": {
            "multi_file_5plus": 3.0,
            "multi_package_2": 1.5,
            "source_100plus_lines": 2.0,
            "tests_50plus_lines": 1.5,
            "test_source_ratio_high": 1.0
          }
        }
      ]
    }

The pipeline consumes these candidate JSONs via:
    craft-taskgen run --candidates candidates/*.json

Scoring heuristics (all structural, no LLM):
    - has_test_patch: hard gate (no tests = reject)
    - multi_file: 2+ source files = 1-3 pts
    - multi_package: 2+ top-level dirs = 1.5-2 pts
    - source_lines_changed: 30+ = 1-2 pts
    - test_lines_changed: 10+ = 1-1.5 pts
    - test/source ratio: >= 0.5 = 1 pt
    - iteration signal: commit message hints = 1 pt
    - iteration cluster: nearby commits on same files = 1-2 pts
    - refactoring penalty: rename/format/lint subjects = -2 pts
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def git(repo: Path, *args: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git {' '.join(args)} timed out after {timeout}s in {repo}")
    except OSError as e:
        raise RuntimeError(f"git {' '.join(args)} failed to start in {repo}: {e}")
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr[:200]}")
    return result.stdout


def _run_gh_api(
    argv: list[str], *, timeout_s: int = 60, max_attempts: int = 3
) -> subprocess.CompletedProcess:
    """Run `gh api ...` with bounded retries on transient failures.

    Some VM network paths see 5–10% intermittent IPv4 dial timeouts to GitHub's
    anycast endpoint. A batch mine does 50+ calls sequentially, so without
    retries one transient failure kills the whole run. Backoff 2s/5s/10s.
    Persistent auth / rate-limit errors still surface as RuntimeError after
    attempts are exhausted.
    """
    backoff_s = [2, 5, 10]
    last_err = ""
    for attempt in range(max_attempts):
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            last_err = f"subprocess timeout ({timeout_s}s)"
        except OSError as e:
            raise RuntimeError(f"gh api failed to start: {e}") from e
        else:
            if result.returncode == 0:
                return result
            last_err = result.stderr[:200].strip() or f"exit {result.returncode}"
        if attempt + 1 < max_attempts:
            print(
                f"  gh api transient failure (attempt {attempt + 1}/{max_attempts}): {last_err[:120]}",
                file=sys.stderr,
            )
            time.sleep(backoff_s[attempt])
    raise RuntimeError(f"gh api failed after {max_attempts} attempts: {last_err}")


def get_prs(github_repo: str, after: str | None = None, max_count: int = 500) -> list[dict]:
    """Get merged PRs from GitHub API via gh CLI.

    sha = pr["merge_commit_sha"] — the commit that lands on main after merge.
    Guaranteed present in a local clone regardless of squash vs regular merge.
    Raises RuntimeError on gh failure — no fallback to commit walking.
    """
    results: list[dict] = []
    page = 1

    while len(results) < max_count:
        argv = [
            "gh",
            "api",
            f"repos/{github_repo}/pulls",
            "--method",
            "GET",
            "-f",
            "state=closed",
            "-f",
            "per_page=100",
            "-F",
            f"page={page}",
        ]
        try:
            result = _run_gh_api(argv, timeout_s=60)
        except RuntimeError as e:
            raise RuntimeError(f"gh api failed for {github_repo} page {page}: {e}") from e

        try:
            prs = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"gh api returned non-JSON for {github_repo} page {page} "
                f"(rate limit or auth failure? run: gh auth login). "
                f"First 200 chars: {result.stdout[:200]!r}"
            )

        if isinstance(prs, dict):
            raise RuntimeError(
                f"gh api returned an error for {github_repo} page {page}: {prs.get('message', prs)}"
            )

        if not prs:
            break

        page_had_results = False
        for pr in prs:
            if not pr.get("merged_at"):
                continue  # skip closed-but-not-merged PRs

            # GitHub sorts by updated_at, not merged_at — a 2023 PR that got a new
            # comment today appears on page 1. Skip old PRs but keep scanning the
            # page; only stop paging when the entire page yields nothing new.
            if after and pr["merged_at"][:10] < after[:10]:
                continue

            if pr["merge_commit_sha"] is None:
                continue  # squash-merged with deleted branch — SHA unavailable

            base = pr.get("base") or {}
            base_sha = base.get("sha")
            if not base_sha:
                continue  # malformed PR response — base SHA unavailable
            user = pr.get("user") or {}
            results.append(
                {
                    "sha": pr["merge_commit_sha"],
                    "base_sha": base_sha,
                    "subject": pr.get("title", ""),
                    "author": user.get("login", "unknown"),
                    "date": pr["merged_at"],
                    "pr_number": pr.get("number", 0),
                }
            )
            page_had_results = True
            if len(results) >= max_count:
                break

        if not page_had_results or len(results) >= max_count:
            break

        page += 1

    return results


def get_diff_stats(repo: Path, base_sha: str, sha: str) -> dict:
    """Get per-file diff stats for the PR's actual changes.

    Uses merge-base rather than base_sha directly so that PRs where the author
    merged or rebased main mid-review don't inflate the diff with unrelated commits.
    """
    merge_base = git(repo, "merge-base", base_sha, sha).strip()
    raw = git(repo, "diff", "--numstat", merge_base, sha)
    files = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added = int(parts[0]) if parts[0] != "-" else 0
        deleted = int(parts[1]) if parts[1] != "-" else 0
        filepath = parts[2]
        files.append({"path": filepath, "added": added, "deleted": deleted})
    return {"files": files, "merge_base_sha": merge_base}


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

# Patterns that suggest test files
TEST_PATTERNS = re.compile(
    r"(^test[_/]|/test[_/]|/tests[_/]|_test\.py$|test_.*\.py$|"
    r"\.test\.[jt]sx?$|\.spec\.[jt]sx?$|__tests__/|/testing/)",
    re.IGNORECASE,
)

# Patterns that suggest docs/config (not interesting)
BORING_PATTERNS = re.compile(
    r"(\.md$|\.rst$|\.txt$|\.cfg$|\.ini$|\.toml$|\.ya?ml$|\.json$|"
    r"\.lock$|changelog|license|readme|\.github/|\.gitignore)",
    re.IGNORECASE,
)

# Patterns that suggest refactoring (renames, formatting)
REFACTOR_SUBJECTS = re.compile(
    r"(rename|refactor|format|style|lint|typo|cleanup|clean up|whitespace|"
    r"black|isort|ruff|flake8|pep8|type.?hint|annotation|docstring)",
    re.IGNORECASE,
)

# Patterns suggesting iteration / difficulty
ITERATION_SUBJECTS = re.compile(
    r"(revert|fix.?up|actually|oops|forgot|missing|also.?fix|"
    r"follow.?up|amend|correction|broke|regress|retry)",
    re.IGNORECASE,
)


def is_test_file(path: str) -> bool:
    return bool(TEST_PATTERNS.search(path))


def is_boring_file(path: str) -> bool:
    return bool(BORING_PATTERNS.search(path))


def is_python_source(path: str) -> bool:
    return path.endswith(".py") and not is_test_file(path) and not is_boring_file(path)


EXCLUDED_PACKAGES = {"tests", "test", "testing", "examples", "example", "docs", "doc", "benchmarks", "bench"}


def get_top_package(path: str) -> str | None:
    """Extract the top-level package/directory from a file path.

    Excludes tests/, examples/, docs/ -- these aren't real packages and
    inflate multi-package scores.
    """
    parts = Path(path).parts
    if len(parts) < 2:
        return None
    pkg = parts[0]
    if pkg.lower() in EXCLUDED_PACKAGES:
        return None
    return pkg


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    sha: str
    base_sha: str
    merge_base_sha: str
    subject: str
    author: str
    date: str

    # Raw metrics
    source_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    other_files: list[str] = field(default_factory=list)
    source_lines_changed: int = 0
    test_lines_changed: int = 0
    packages_touched: int = 0
    package_names: list[str] = field(default_factory=list)

    # Scoring signals
    has_test_patch: bool = False
    is_multi_file: bool = False
    is_multi_package: bool = False
    is_nontrivial_source: bool = False
    is_nontrivial_tests: bool = False
    is_refactoring: bool = False
    has_iteration_signal: bool = False

    # Composite score (higher = more promising)
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)

    def compute_score(self) -> None:
        """Score this candidate against our difficulty signals."""
        breakdown = {}

        # Hard gate: must have test changes
        if not self.has_test_patch:
            self.score = 0.0
            breakdown["gate_no_tests"] = "REJECT"
            self.score_breakdown = breakdown
            return

        # Hard gate: must not be pure refactoring
        if self.is_refactoring and self.source_lines_changed < 50:
            self.score = 0.0
            breakdown["gate_refactoring"] = "REJECT"
            self.score_breakdown = breakdown
            return

        score = 0.0

        # Multi-file source changes (3+ source files)
        n_src = len(self.source_files)
        if n_src >= 5:
            score += 3.0
            breakdown["multi_file_5plus"] = 3.0
        elif n_src >= 3:
            score += 2.0
            breakdown["multi_file_3plus"] = 2.0
        elif n_src >= 2:
            score += 1.0
            breakdown["multi_file_2"] = 1.0

        # Multi-package (2+ top-level dirs touched by source changes)
        if self.packages_touched >= 3:
            score += 2.0
            breakdown["multi_package_3plus"] = 2.0
        elif self.packages_touched >= 2:
            score += 1.5
            breakdown["multi_package_2"] = 1.5

        # Nontrivial source changes (>30 lines)
        if self.source_lines_changed >= 100:
            score += 2.0
            breakdown["source_100plus_lines"] = 2.0
        elif self.source_lines_changed >= 30:
            score += 1.0
            breakdown["source_30plus_lines"] = 1.0

        # Nontrivial test changes (>10 lines) -- verifier quality signal
        if self.test_lines_changed >= 50:
            score += 1.5
            breakdown["tests_50plus_lines"] = 1.5
        elif self.test_lines_changed >= 10:
            score += 1.0
            breakdown["tests_10plus_lines"] = 1.0

        # Test-to-source ratio: substantial tests relative to source
        # suggests the verification itself was nontrivial
        if self.source_lines_changed > 0:
            ratio = self.test_lines_changed / self.source_lines_changed
            if ratio >= 0.5:
                score += 1.0
                breakdown["test_source_ratio_high"] = 1.0

        # Iteration signal from commit message
        if self.has_iteration_signal:
            score += 1.0
            breakdown["iteration_signal"] = 1.0

        # Penalty for likely-refactoring commit messages
        if self.is_refactoring:
            score -= 2.0
            breakdown["refactoring_penalty"] = -2.0

        self.score = max(score, 0.0)
        self.score_breakdown = breakdown


def analyze_pr(repo: Path, pr: dict) -> Candidate:
    """Analyze a single PR and produce a scored candidate."""
    sha = pr["sha"]
    base_sha = pr["base_sha"]
    stats = get_diff_stats(repo, base_sha, sha)

    candidate = Candidate(
        sha=sha,
        base_sha=base_sha,
        merge_base_sha=stats["merge_base_sha"],
        subject=pr["subject"],
        author=pr["author"],
        date=pr["date"],
    )

    source_packages = set()

    for f in stats["files"]:
        path = f["path"]
        lines = f["added"] + f["deleted"]

        if is_test_file(path):
            candidate.test_files.append(path)
            candidate.test_lines_changed += lines
        elif is_boring_file(path):
            candidate.other_files.append(path)
        elif is_python_source(path):
            candidate.source_files.append(path)
            candidate.source_lines_changed += lines
            pkg = get_top_package(path)
            if pkg:
                source_packages.add(pkg)
        else:
            # Non-Python source (JS, Go, etc.) -- still count as source
            candidate.source_files.append(path)
            candidate.source_lines_changed += lines
            pkg = get_top_package(path)
            if pkg:
                source_packages.add(pkg)

    candidate.packages_touched = len(source_packages)
    candidate.package_names = sorted(source_packages)
    candidate.has_test_patch = len(candidate.test_files) > 0
    candidate.is_multi_file = len(candidate.source_files) >= 3
    candidate.is_multi_package = candidate.packages_touched >= 2
    candidate.is_nontrivial_source = candidate.source_lines_changed >= 30
    candidate.is_nontrivial_tests = candidate.test_lines_changed >= 10
    candidate.is_refactoring = bool(REFACTOR_SUBJECTS.search(pr["subject"]))
    candidate.has_iteration_signal = bool(ITERATION_SUBJECTS.search(pr["subject"]))

    candidate.compute_score()
    return candidate


# ---------------------------------------------------------------------------
# Nearby-commit clustering (iteration detection)
# ---------------------------------------------------------------------------


def detect_iteration_clusters(candidates: list[Candidate], window_hours: int = 48) -> list[Candidate]:
    """Boost scores for PRs that are part of an iteration cluster.

    If multiple PRs from the same author touch overlapping files within
    a time window, they likely represent iteration on a hard problem.
    """
    # Sort by date
    dated = []
    for c in candidates:
        try:
            dt = datetime.fromisoformat(c.date.replace("Z", "+00:00"))
        except ValueError:
            continue
        dated.append((dt, c))
    dated.sort(key=lambda x: x[0])

    window = timedelta(hours=window_hours)

    for i, (dt_i, ci) in enumerate(dated):
        if ci.score == 0:
            continue
        src_set_i = set(ci.source_files)
        if not src_set_i:
            continue

        cluster_size = 0
        for j in range(i + 1, len(dated)):
            dt_j, cj = dated[j]
            if dt_j - dt_i > window:
                break
            if cj.author != ci.author:
                continue
            src_set_j = set(cj.source_files)
            if src_set_i & src_set_j:  # overlapping files
                cluster_size += 1

        if cluster_size >= 2:
            ci.score += 2.0
            ci.score_breakdown["iteration_cluster_3plus"] = 2.0
        elif cluster_size >= 1:
            ci.score += 1.0
            ci.score_breakdown["iteration_cluster_2"] = 1.0

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def mine_repo(repo_path: Path, github_repo: str, after: str | None, top_n: int) -> tuple[list[dict], int]:
    """Mine a repo for hard task candidates via GitHub PRs.

    Returns (top_candidates, n_prs_scanned).
    """
    print(f"Scanning {repo_path.name} ({github_repo})...", file=sys.stderr)

    prs = get_prs(github_repo, after=after)
    n_prs = len(prs)
    print(f"  {n_prs} PRs to analyze", file=sys.stderr)

    candidates = []
    for i, pr in enumerate(prs):
        if i % 200 == 0 and i > 0:
            print(f"  ...analyzed {i}/{n_prs}", file=sys.stderr)
        try:
            c = analyze_pr(repo_path, pr)
            if c.score > 0:
                candidates.append(c)
        except RuntimeError:
            # Expected: base_sha not present in local clone (shallow clone, force-push)
            continue
        except Exception as e:
            print(
                f"  WARN: PR #{pr.get('pr_number')} ({pr.get('sha', '')[:8]}) failed unexpectedly: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            continue

    print(f"  {len(candidates)} candidates with score > 0", file=sys.stderr)

    # Cluster detection for iteration bonus
    candidates = detect_iteration_clusters(candidates)

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Take top N
    top = candidates[:top_n]

    print(
        f"  Top {len(top)} candidates (score range: {top[0].score:.1f} - {top[-1].score:.1f})"
        if top
        else "  No candidates found",
        file=sys.stderr,
    )

    return [asdict(c) for c in top], n_prs


def _load_repos_csv(csv_path: Path) -> list[dict]:
    """Load repo list from craft-repos.csv. Returns list of {short_name, github_repo, ...}."""
    import csv

    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _clone_or_find_repo(github_repo: str, repos_dir: Path) -> Path | None:
    """Find a cloned repo in repos_dir, or clone it. Returns repo path or None on failure."""
    # Try common directory naming conventions
    owner, name = github_repo.split("/")
    for candidate in [name, f"{owner}__{name}", github_repo.replace("/", "_")]:
        repo_path = repos_dir / candidate
        if repo_path.is_dir() and (repo_path / ".git").is_dir():
            return repo_path

    # Not found — clone it
    repo_path = repos_dir / name
    print(f"  Cloning {github_repo} -> {repo_path}...", file=sys.stderr)
    try:
        result = subprocess.run(
            ["git", "clone", f"https://github.com/{github_repo}.git", str(repo_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("  Clone timed out after 300s", file=sys.stderr)
        return None
    if result.returncode != 0:
        print(f"  Clone failed (exit {result.returncode}): {result.stderr[:200]}", file=sys.stderr)
        return None
    return repo_path if repo_path.is_dir() else None


def main():
    parser = argparse.ArgumentParser(
        description="Mine GitHub PRs for hard tool-orchestration task candidates."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("repo", nargs="?", type=Path, help="Path to a single cloned git repo")
    group.add_argument(
        "--repos-csv",
        type=Path,
        help="CSV file with repo list (default: references/craft-repos.csv). Mines all repos.",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=None,
        help="Directory containing cloned repos (for --repos-csv mode). Repos not found here are cloned.",
    )
    parser.add_argument(
        "--after",
        type=str,
        default=None,
        help="Only consider PRs merged after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top candidates to output per repo (default: 20)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON file (single repo) or directory (--repos-csv mode)",
    )
    args = parser.parse_args()

    if args.after:
        try:
            datetime.strptime(args.after, "%Y-%m-%d")
        except ValueError:
            parser.error("--after must be in YYYY-MM-DD format (e.g. 2025-10-01)")

    if args.repos_csv:
        # Batch mode: mine all repos from CSV
        repos_dir = args.repos_dir or Path("repos")
        repos_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(args.out) if args.out else Path("candidates")
        out_dir.mkdir(parents=True, exist_ok=True)

        repos = _load_repos_csv(args.repos_csv)
        print(f"Mining {len(repos)} repos from {args.repos_csv}", file=sys.stderr)

        mined = 0
        for entry in repos:
            short_name = entry["short_name"]
            github_repo = entry["github_repo"]
            out_file = out_dir / f"{short_name}.json"

            print(f"\n[{short_name}] ({github_repo})", file=sys.stderr)
            if out_file.exists():
                print(f"  SKIP: {out_file} already exists", file=sys.stderr)
                mined += 1
                continue
            repo_path = _clone_or_find_repo(github_repo, repos_dir)
            if not repo_path:
                print("  SKIP: could not find or clone repo", file=sys.stderr)
                continue

            results, n_prs = mine_repo(repo_path, github_repo, args.after, args.top)
            output = {
                "repo": short_name,
                "github_repo": github_repo,
                "after": args.after,
                "n_prs_scanned": n_prs,
                "n_candidates": len(results),
                "candidates": results,
            }
            with open(out_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  -> {out_file} ({len(results)} candidates)", file=sys.stderr)
            mined += 1

        print(f"\nDone: mined {mined}/{len(repos)} repos -> {out_dir}/", file=sys.stderr)
    else:
        # Single repo mode
        repo_path = args.repo

        if not repo_path.is_dir():
            # Accept owner/repo slug — clone to repos/{name} automatically
            slug_match = re.fullmatch(r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", str(repo_path))
            if not slug_match:
                print(f"Error: {repo_path} is not a directory and is not an owner/repo slug", file=sys.stderr)
                sys.exit(1)
            github_repo = f"{slug_match.group(1)}/{slug_match.group(2)}"
            cloned = _clone_or_find_repo(github_repo, Path("repos"))
            if not cloned:
                print(f"Error: failed to clone {github_repo}", file=sys.stderr)
                sys.exit(1)
            repo_path = cloned
        else:
            # Derive github_repo from git remote of existing clone
            remote_result = subprocess.run(
                ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if remote_result.returncode != 0:
                print("Error: could not get remote URL — is this a GitHub repo?", file=sys.stderr)
                sys.exit(1)
            remote_url = remote_result.stdout.strip()
            m = re.search(r"github\.com[:/]([^/]+/[^/?#]+?)(?:\.git)?$", remote_url)
            if not m:
                print(f"Error: remote URL is not a GitHub URL: {remote_url}", file=sys.stderr)
                sys.exit(1)
            github_repo = m.group(1)

        results, n_prs = mine_repo(repo_path, github_repo, args.after, args.top)
        output = {
            "repo": str(repo_path),
            "github_repo": github_repo,
            "after": args.after,
            "n_prs_scanned": n_prs,
            "n_candidates": len(results),
            "candidates": results,
        }

        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Wrote {len(results)} candidates to {args.out}", file=sys.stderr)
        else:
            json.dump(output, sys.stdout, indent=2)
            print(file=sys.stdout)


if __name__ == "__main__":
    main()
