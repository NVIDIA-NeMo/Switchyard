# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""extract_v2b_patch_gold.py — derive search-style gold from v2b solution patches.

Walks `<v2b-tasks-dir>/*/solution/changes.patch` for each task, parses the diff
to extract the set of files modified, and uses AST analysis on the pre-patch
source (cloned per the task's environment/Dockerfile) to determine which
functions contain the changed lines. Emits a JSON file shaped like the
search-side `gold_answer.json` for use as a canonical, patch-derived gold for
cross-summarizer correlation analysis.

Output JSON shape (one entry per task):
{
  "t2v3-PI628c-embedding-display-fix": {
    "files": ["pixeltable/catalog/catalog.py", ...],
    "functions": ["pixeltable.catalog.catalog.Catalog.construct_tvp", ...],
    "alt_files": [],
    "alt_functions": [],
    "n_hunks": 12,
    "repo_url": "https://github.com/pixeltable/pixeltable.git",
    "repo_commit": "8d51ca4a9bcf11644880721674043a76e15af961",
    "skipped_files": []  // non-Python or unparseable
  },
  ...
}

Repos are cloned to `--repos-cache <dir>` (default /tmp/v2b-repos-cache) and
checked out at the pre-patch commit. Cache is keyed by (repo, commit) so
re-runs are fast. Network access required on first run.

Usage:
  uv run python scripts/extract_v2b_patch_gold.py \\
      --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v2b \\
      --output references/v2b-patch-gold.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------


@dataclass
class Hunk:
    file: str  # post-patch path (b/<path>)
    pre_start: int  # line number in pre-patch file (1-based)
    pre_len: int


def _parse_patch(patch_text: str) -> list[Hunk]:
    """Parse a unified diff. Returns one Hunk per `@@` block. Skips file additions
    (where pre-len is 0 because the file didn't exist) — those have no pre-patch
    state to AST-walk.
    """
    out: list[Hunk] = []
    current_file: str | None = None
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            # `+++ b/<path>` or `+++ /dev/null`
            path = line[4:].strip()
            if path == "/dev/null":
                current_file = None
            elif path.startswith("b/"):
                current_file = path[2:]
            else:
                current_file = path
            continue
        if line.startswith("@@") and current_file:
            # @@ -A,B +C,D @@
            m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
            if not m:
                continue
            pre_start = int(m.group(1))
            pre_len = int(m.group(2)) if m.group(2) else 1
            if pre_len == 0:
                # New file or pure addition with no pre-patch lines; can't
                # locate inside a function.
                continue
            out.append(Hunk(file=current_file, pre_start=pre_start, pre_len=pre_len))
    return out


# ---------------------------------------------------------------------------
# Dockerfile parsing — extract repo URL + pre-patch commit
# ---------------------------------------------------------------------------

_GIT_CLONE_RE = re.compile(r"git clone(?:\s+--[\w=:\-]+)*\s+(https?://[^\s]+\.git)")
_GIT_CHECKOUT_RE = re.compile(r"git checkout\s+([0-9a-f]{40})")


def _parse_dockerfile_repo(dockerfile_text: str) -> tuple[str, str] | None:
    """Return (repo_url, commit_sha) from the Dockerfile, or None if absent."""
    # Collapse line continuations so multi-line RUN blocks are parseable.
    collapsed = re.sub(r"\\\s*\n", " ", dockerfile_text)
    clone_m = _GIT_CLONE_RE.search(collapsed)
    co_m = _GIT_CHECKOUT_RE.search(collapsed)
    if not clone_m or not co_m:
        return None
    return (clone_m.group(1), co_m.group(1))


def _repo_name_from_url(url: str) -> str:
    """github.com/foo/bar.git -> bar"""
    name = url.rstrip("/").rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".git") else name


# ---------------------------------------------------------------------------
# Repo cloning + checkout
# ---------------------------------------------------------------------------


def _ensure_repo_at_commit(url: str, commit: str, cache_dir: Path) -> Path:
    """Clone (or reuse) the repo, fetch the commit, check it out into a
    per-(repo, commit) worktree. Returns the worktree path.

    Layout:
      <cache>/<repo>/.git           — bare-ish shared clone (full history fetched)
      <cache>/<repo>/<commit[:12]>/  — checkout of the specific commit
    """
    repo_name = _repo_name_from_url(url)
    repo_dir = cache_dir / repo_name
    worktree_dir = cache_dir / f"{repo_name}__{commit[:12]}"

    if worktree_dir.is_dir() and (worktree_dir / ".git").exists():
        # Cached. Trust it.
        return worktree_dir

    if not repo_dir.is_dir():
        print(f"  cloning {url} → {repo_dir}", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", url, str(repo_dir)],
            check=True,
            capture_output=True,
        )
    else:
        # Make sure we have the commit. If not, fetch.
        rc = subprocess.run(
            ["git", "-C", str(repo_dir), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
        )
        if rc.returncode != 0:
            print(f"  fetching commit {commit[:12]} for {repo_name}", file=sys.stderr)
            subprocess.run(
                ["git", "-C", str(repo_dir), "fetch", "--no-tags", "origin", commit],
                check=False,  # commit might not be on origin's branches
                capture_output=True,
            )
            # Last-resort: full fetch.
            rc2 = subprocess.run(
                ["git", "-C", str(repo_dir), "cat-file", "-e", f"{commit}^{{commit}}"],
                capture_output=True,
            )
            if rc2.returncode != 0:
                subprocess.run(
                    ["git", "-C", str(repo_dir), "fetch", "--all", "--tags"],
                    check=False,
                    capture_output=True,
                )

    # Now check out the commit into a fresh worktree (lighter than re-cloning).
    print(f"  checking out {commit[:12]} → {worktree_dir.name}", file=sys.stderr)
    subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "add", "--detach", str(worktree_dir), commit],
        check=True,
        capture_output=True,
    )
    return worktree_dir


# ---------------------------------------------------------------------------
# Function extraction via AST
# ---------------------------------------------------------------------------


@dataclass
class FuncRange:
    qualname: str  # e.g. "Catalog.construct_tvp" or "module_func"
    lineno: int
    end_lineno: int


def _walk_funcs(tree: ast.AST, prefix: list[str] = []) -> list[FuncRange]:  # noqa: B006
    """Flatten all function defs in a module to (qualname, start_line, end_line) tuples.
    Includes nested classes; methods get the dotted ClassName.method form.
    """
    out: list[FuncRange] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qn = ".".join(prefix + [node.name])
            out.append(FuncRange(qualname=qn, lineno=node.lineno, end_lineno=node.end_lineno or node.lineno))
        elif isinstance(node, ast.ClassDef):
            out.extend(_walk_funcs(node, prefix + [node.name]))
    return out


def _module_path_from_file(file_path: str) -> str:
    """pixeltable/catalog/catalog.py -> pixeltable.catalog.catalog. Strip `.py`."""
    if file_path.endswith(".py"):
        file_path = file_path[:-3]
    return file_path.replace("/", ".")


def _funcs_for_hunks(repo_root: Path, file_path: str, hunks: list[Hunk]) -> list[str]:
    """For each hunk, find the enclosing function in the pre-patch source.
    Returns dotted-qualified function names (`module.Class.method`).
    Empty if the file isn't Python or can't be parsed.
    """
    src_file = repo_root / file_path
    if not src_file.is_file() or not file_path.endswith(".py"):
        return []
    try:
        tree = ast.parse(src_file.read_text(errors="replace"), filename=str(src_file))
    except (SyntaxError, ValueError):
        return []
    module = _module_path_from_file(file_path)
    funcs = _walk_funcs(tree)
    matched: set[str] = set()
    for h in hunks:
        # The hunk's `pre_start` is 1-based and refers to the first line of
        # context. The first changed line is typically `pre_start + (context lines)`,
        # but we don't have the context structure here without re-parsing the
        # hunk body. Approximation: any function whose line range overlaps the
        # hunk range is "touched". Generous but correct for the use case.
        h_lo, h_hi = h.pre_start, h.pre_start + h.pre_len - 1
        for f in funcs:
            if f.lineno <= h_hi and f.end_lineno >= h_lo:
                matched.add(f"{module}.{f.qualname}")
    return sorted(matched)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class TaskGold:
    files: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    alt_files: list[str] = field(default_factory=list)
    alt_functions: list[str] = field(default_factory=list)
    n_hunks: int = 0
    repo_url: str = ""
    repo_commit: str = ""
    skipped_files: list[str] = field(default_factory=list)


def extract_one(task_dir: Path, cache_dir: Path) -> TaskGold | None:
    patch_file = task_dir / "solution" / "changes.patch"
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not patch_file.is_file() or not dockerfile.is_file():
        return None

    repo_info = _parse_dockerfile_repo(dockerfile.read_text())
    if repo_info is None:
        print(f"WARN: {task_dir.name}: no git clone+checkout in Dockerfile", file=sys.stderr)
        return None
    url, commit = repo_info

    hunks = _parse_patch(patch_file.read_text())
    if not hunks:
        print(f"WARN: {task_dir.name}: no parseable hunks in changes.patch", file=sys.stderr)
        return None

    # Group hunks by file
    by_file: dict[str, list[Hunk]] = {}
    for h in hunks:
        by_file.setdefault(h.file, []).append(h)

    repo_root = _ensure_repo_at_commit(url, commit, cache_dir)

    gold = TaskGold(
        n_hunks=len(hunks),
        repo_url=url,
        repo_commit=commit,
    )
    all_funcs: set[str] = set()
    for fp, fh in sorted(by_file.items()):
        if fp.endswith(".py"):
            funcs = _funcs_for_hunks(repo_root, fp, fh)
            if not funcs:
                gold.skipped_files.append(fp)
            else:
                all_funcs.update(funcs)
        else:
            gold.skipped_files.append(fp)
        gold.files.append(fp)

    gold.files = sorted(set(gold.files))
    gold.functions = sorted(all_funcs)
    return gold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Directory containing v2b task subdirs (e.g. craft-taskgen-v2b/).",
    )
    ap.add_argument("--output", type=Path, required=True, help="JSON output path.")
    ap.add_argument(
        "--repos-cache",
        type=Path,
        default=Path("/tmp/v2b-repos-cache"),
        help="Directory for cached repo clones.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, stop after processing this many tasks (for quick smoke tests).",
    )
    args = ap.parse_args()

    args.repos_cache.mkdir(parents=True, exist_ok=True)
    task_dirs = sorted([d for d in args.tasks_dir.iterdir() if d.is_dir() and d.name.startswith("t2v3-")])
    if args.limit:
        task_dirs = task_dirs[: args.limit]
    print(f"Processing {len(task_dirs)} task dirs...", file=sys.stderr)

    out: dict[str, dict] = {}
    failures: list[str] = []
    for i, td in enumerate(task_dirs, 1):
        print(f"[{i}/{len(task_dirs)}] {td.name}", file=sys.stderr)
        try:
            gold = extract_one(td, args.repos_cache)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR (git): {e.stderr.decode(errors='replace')[:300]}", file=sys.stderr)
            failures.append(td.name)
            continue
        except Exception as e:  # noqa: BLE001 — keep going through the corpus
            print(f"  ERROR: {e}", file=sys.stderr)
            failures.append(td.name)
            continue
        if gold is None:
            failures.append(td.name)
            continue
        out[td.name] = {
            "files": gold.files,
            "functions": gold.functions,
            "alt_files": gold.alt_files,
            "alt_functions": gold.alt_functions,
            "n_hunks": gold.n_hunks,
            "repo_url": gold.repo_url,
            "repo_commit": gold.repo_commit,
            "skipped_files": gold.skipped_files,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        f"\nWrote {len(out)} task entries → {args.output}",
        file=sys.stderr,
    )
    if failures:
        print(
            f"Failed on {len(failures)} tasks: {failures[:10]}{'...' if len(failures) > 10 else ''}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
