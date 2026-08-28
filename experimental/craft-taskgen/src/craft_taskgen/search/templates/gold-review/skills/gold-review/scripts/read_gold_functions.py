#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and display the source code of all gold functions for a task.

For each gold function, finds it in the repo source and prints the body
with a classification hint (abstract stub, one-liner, substantive, etc.).

Usage:
    uv run python .claude/skills/gold-review/scripts/read_gold_functions.py craft-httpx-9e2db24f
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def dotted_to_file_candidates(repo: str, func_name: str) -> list[Path]:
    """Convert dotted function name to candidate file paths."""
    parts = func_name.split(".")
    candidates = []
    for i in range(len(parts) - 1, 0, -1):
        candidate = "/".join(parts[:i]) + ".py"
        candidates.append(ROOT / "repos" / repo / candidate)
    for i in range(len(parts) - 1, 0, -1):
        candidate = "/".join(parts[:i]) + "/__init__.py"
        candidates.append(ROOT / "repos" / repo / candidate)
    return candidates


def find_function_source(repo: str, func_name: str) -> tuple[str | None, str | None, str]:
    """Find the function source. Returns (file_path, source_text, classification)."""
    parts = func_name.split(".")
    # The function/method name is the last part; class name (if any) is second to last
    target_name = parts[-1]

    for candidate_path in dotted_to_file_candidates(repo, func_name):
        if not candidate_path.exists():
            continue

        try:
            source = candidate_path.read_text()
        except Exception:
            continue

        # Try AST parsing for accurate extraction
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        rel_path = str(candidate_path.relative_to(ROOT / "repos" / repo))

        # Look for the function/method
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
                # Check if it's inside the right class
                if len(parts) >= 3:
                    # Need to verify class context — walk class defs
                    class_name = parts[-2]
                    found_in_class = False
                    for cls_node in ast.walk(tree):
                        if isinstance(cls_node, ast.ClassDef) and cls_node.name == class_name:
                            for item in ast.walk(cls_node):
                                if item is node:
                                    found_in_class = True
                                    break
                    if not found_in_class:
                        continue

                lines = source.splitlines()
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
                func_source = "\n".join(lines[start:end])
                body_lines = end - start

                # Classify
                classification = classify_function(node, func_source, body_lines)
                return rel_path, func_source, classification

    return None, None, "NOT FOUND"


def classify_function(node: ast.AST, source: str, body_lines: int) -> str:
    """Classify a function by its pattern."""
    body = node.body if hasattr(node, "body") else []

    # Abstract stub
    decorators = [d for d in (node.decorator_list if hasattr(node, "decorator_list") else [])]
    for d in decorators:
        name = ""
        if isinstance(d, ast.Name):
            name = d.id
        elif isinstance(d, ast.Attribute):
            name = d.attr
        if name == "abstractmethod":
            return "ABSTRACT STUB"

    # Count non-docstring, non-pass statements
    real_stmts = []
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.Str)):
            continue  # docstring
        if isinstance(stmt, ast.Pass):
            continue
        real_stmts.append(stmt)

    if len(real_stmts) == 0:
        return "EMPTY/DOCSTRING-ONLY"

    if len(real_stmts) == 1:
        stmt = real_stmts[0]
        # One-liner return
        if isinstance(stmt, ast.Return) and stmt.value:
            # Check if it's a simple delegation
            if isinstance(stmt.value, ast.Call):
                return "ONE-LINER DELEGATION"
            if isinstance(stmt.value, ast.Compare) or isinstance(stmt.value, ast.BoolOp):
                return "TRIVIAL COMPARISON"
            return "ONE-LINER RETURN"
        # Expression statement (await call)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Call, ast.Await)):
            return "ONE-LINER DELEGATION"

    if body_lines <= 5 and len(real_stmts) <= 2:
        return "TRIVIAL (2-5 lines)"

    if any(isinstance(s, (ast.If, ast.For, ast.While, ast.With, ast.Try)) for s in real_stmts):
        return "SUBSTANTIVE (has branching/loops)"

    return f"MODERATE ({body_lines} lines, {len(real_stmts)} stmts)"


def main():
    if len(sys.argv) < 2:
        print("Usage: read_gold_functions.py <craft-task-id>", file=sys.stderr)
        sys.exit(1)

    task_id = sys.argv[1]

    # Load task from review_data.json
    with open(ROOT / "tools" / "search" / "review_data.json") as f:
        data = json.load(f)
    task = next((t for t in data["tasks"] if t["task_id"] == task_id), None)
    if not task:
        print(f"ERROR: Task {task_id} not found", file=sys.stderr)
        sys.exit(1)

    repo = task["repo"]
    gold = task["gold"]
    all_funcs = gold["functions"] + gold.get("alt_functions", [])

    for func_name in all_funcs:
        is_alt = func_name in gold.get("alt_functions", [])
        print(f"\n{'=' * 80}")
        print(f"{'[ALT] ' if is_alt else ''}{func_name}")
        print(f"{'=' * 80}")

        file_path, source, classification = find_function_source(repo, func_name)
        if file_path:
            print(f"File: {file_path}")
            print(f"Classification: {classification}")
            print(f"{'─' * 60}")
            print(source)
        else:
            print(f"Classification: {classification}")


if __name__ == "__main__":
    main()
