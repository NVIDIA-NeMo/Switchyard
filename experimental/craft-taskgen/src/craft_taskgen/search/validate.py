# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate search-from-T2 gold answers against repo ground truth.

Checks that gold files exist in the repo and gold functions resolve to
real definitions. Reuses validation logic from validate_gold_answers.py.

Also auto-populates alt_functions from class methods for method-level flexibility.

Ported from craft-bench scripts/search/validate_t2_gold.py.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

from craft_taskgen.mining.repo_indexer import RepoIndexer
from craft_taskgen.mining.schemas import RepoGroundTruth

# ---------------------------------------------------------------------------
# Ground truth helpers (simplified from validate_gold_answers.py)
# ---------------------------------------------------------------------------


def build_file_set(gt: RepoGroundTruth) -> set[str]:
    """All files in the repo, normalized."""
    return {f.lstrip("./") for f in gt.file_structure.files}


def build_class_methods(gt: RepoGroundTruth) -> dict[str, list[str]]:
    """Build {qualified_class_name: [method_name, ...]} from ground truth.

    Keys are like 'uvicorn.config.Config', values are like
    ['get_loop_factory', 'setup_event_loop', 'load', '__init__'].
    """
    result: dict[str, list[str]] = defaultdict(list)
    for cls in gt.type_defs.classes:
        # cls.name is just 'Config', need to qualify with module
        module = cls.file_path.replace("/", ".").removesuffix(".py")
        qualified = f"{module}.{cls.name}"
        for method in cls.methods:
            method_name = method.name if hasattr(method, "name") else str(method)
            result[qualified].append(method_name)
    return dict(result)


def build_all_callees(gt: RepoGroundTruth) -> set[str]:
    """All callee names from call graph edges."""
    return {edge.callee for edge in gt.call_graph.edges}


def check_file(filepath: str, file_set: set[str]) -> str | None:
    """Return error message if file doesn't exist, else None."""
    normalized = filepath.lstrip("./").rstrip("/")
    if normalized in file_set:
        return None
    # Try without leading directory
    for f in file_set:
        if f.endswith("/" + normalized) or f == normalized:
            return None
    return f"file not in repo: {filepath}"


def check_function(func_name: str, class_methods: dict[str, list[str]], callees: set[str]) -> str | None:
    """Return error message if function can't be resolved, else None."""
    # Direct callee match
    if func_name in callees:
        return None

    # Parse into module.Class.method
    parts = func_name.split(".")
    if len(parts) >= 3:
        # Try class.method match
        method = parts[-1]
        class_name = parts[-2]
        # Check if any qualified class has this method
        for qual_class, methods in class_methods.items():
            if qual_class.endswith(f".{class_name}") and method in methods:
                return None

    # Try leaf name in callees
    leaf = parts[-1] if parts else func_name
    if leaf in callees:
        return None

    # Try tail match (Class.method)
    if len(parts) >= 2:
        tail = f"{parts[-2]}.{parts[-1]}"
        for callee in callees:
            if callee.endswith(tail):
                return None

    return f"function not resolvable: {func_name}"


# ---------------------------------------------------------------------------
# Alt-function expansion
# ---------------------------------------------------------------------------


def expand_alt_functions(
    gold_functions: list[str],
    class_methods: dict[str, list[str]],
) -> list[str]:
    """For each gold function module.Class.method, add other methods on the same class."""
    alt = []
    seen = set(gold_functions)

    for func in gold_functions:
        parts = func.split(".")
        if len(parts) < 3:
            continue

        class_name = parts[-2]

        # Find the class in ground truth and add its other methods
        for qual_class, methods in class_methods.items():
            if qual_class.endswith(f".{class_name}"):
                module = qual_class.rsplit(".", 1)[0]
                for m in methods:
                    qualified = f"{module}.{class_name}.{m}"
                    if qualified not in seen:
                        alt.append(qualified)
                        seen.add(qualified)
                break

    return alt


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def load_ground_truths(repos_dir: str, contexts_path: str) -> dict[str, RepoGroundTruth]:
    """Load ground truth for each unique repo referenced by contexts."""
    with open(contexts_path) as f:
        contexts = json.load(f)

    gts: dict[str, RepoGroundTruth] = {}
    for ctx in contexts:
        url = ctx["solve_info"]["upstream_url"]
        repo_name = url.rstrip("/").removesuffix(".git").split("/")[-1] if url else ""
        if repo_name in gts:
            continue
        repo_dir = os.path.join(repos_dir, repo_name)
        if not os.path.isdir(repo_dir):
            print(f"  WARNING: repo not found: {repo_dir}", file=sys.stderr)
            continue
        print(f"  Indexing {repo_name}...", file=sys.stderr)
        indexer = RepoIndexer(repo_dir)
        gts[repo_name] = indexer.index()

    return gts


# ---------------------------------------------------------------------------
# Top-level entry point (called by step_validate)
# ---------------------------------------------------------------------------


def run_validate(
    contexts_dir: str,
    repos_dir: str = "repos",
    fix_alt_funcs: bool = True,
) -> None:
    """Validate gold answers against repo AST and optionally expand alt_functions.

    Parameters
    ----------
    contexts_dir:
        Directory containing _all_contexts.json and approach-{a,b,c}/search_tasks.json.
    repos_dir:
        Directory containing cloned repos.
    fix_alt_funcs:
        If True, auto-populate alt_functions from class methods.
    """
    contexts_path = os.path.join(contexts_dir, "_all_contexts.json")
    with open(contexts_path) as f:
        contexts = {c["task_id"]: c for c in json.load(f)}

    # Build ground truth for each repo
    print("Loading ground truth...", file=sys.stderr)
    gts = load_ground_truths(repos_dir, contexts_path)

    # Build indexes per repo
    repo_indexes: dict[str, tuple[set[str], dict[str, list[str]], set[str]]] = {}
    for repo_name, gt in gts.items():
        file_set = build_file_set(gt)
        class_methods = build_class_methods(gt)
        callees = build_all_callees(gt)
        repo_indexes[repo_name] = (file_set, class_methods, callees)
        print(f"  {repo_name}: {len(file_set)} files, {len(class_methods)} classes, {len(callees)} callees")

    # Validate all tasks
    print("\nValidating tasks...\n")

    all_tasks = []
    for approach in ["a", "b", "c"]:
        path = os.path.join(contexts_dir, f"approach-{approach}", "search_tasks.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            tasks = json.load(f)
        all_tasks.extend([(approach, t, path) for t in tasks])

    total = len(all_tasks)
    good = 0
    file_errors = 0
    func_errors = 0
    alt_added = 0

    for approach, task, tasks_path in sorted(all_tasks, key=lambda x: x[1]["id"]):
        gold = task["gold_answer"]
        parent = task.get("parent_t2_task", "")
        ctx = contexts.get(parent, {})
        url = ctx.get("solve_info", {}).get("upstream_url", "")
        repo_name = url.rstrip("/").removesuffix(".git").split("/")[-1] if url else ""

        if repo_name not in repo_indexes:
            print(f"  [{approach.upper()}] {task['id']}: SKIP (no ground truth for {repo_name})")
            continue

        file_set, class_methods, callees = repo_indexes[repo_name]
        issues = []

        # Check files
        for f in gold.get("files", []):
            err = check_file(f, file_set)
            if err:
                issues.append(f"FILE: {err}")
                file_errors += 1

        # Check functions
        for fn in gold.get("functions", []):
            err = check_function(fn, class_methods, callees)
            if err:
                issues.append(f"FUNC: {err}")
                func_errors += 1

        # Expand alt_functions if requested
        if fix_alt_funcs:
            new_alts = expand_alt_functions(gold.get("functions", []), class_methods)
            existing_alts = set(gold.get("alt_functions", []))
            added = [a for a in new_alts if a not in existing_alts]
            if added:
                gold["alt_functions"] = list(existing_alts | set(new_alts))
                alt_added += len(added)

        if issues:
            print(f"  [{approach.upper()}] {task['id']} ({parent}):")
            for issue in issues:
                print(f"    {issue}")
        else:
            good += 1

    print(f"\n{'=' * 60}")
    print(f"Total: {total}  Good: {good}  File errors: {file_errors}  Func errors: {func_errors}")
    if fix_alt_funcs:
        print(f"Alt functions added: {alt_added}")

    # Write back if we modified tasks
    if fix_alt_funcs and alt_added > 0:
        # Group by approach and rewrite
        by_path: dict[str, list] = defaultdict(list)
        for approach, task, tasks_path in all_tasks:
            by_path[tasks_path].append(task)
        for path, tasks in by_path.items():
            with open(path, "w") as f:
                json.dump(tasks, f, indent=2)
            print(f"  Updated {path}")
