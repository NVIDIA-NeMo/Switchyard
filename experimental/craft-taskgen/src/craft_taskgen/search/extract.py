# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse Tools-track task artifacts (solve.sh, gold_reference_tests.py, instruction.md).

Ported from craft-bench scripts/search/extract_t2_gold.py.
Includes repo-map mining via native craft_taskgen.mining (no craft-bench dependency).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from craft_taskgen.mining.repo_indexer import RepoIndexer
from craft_taskgen.mining.repo_map import build_repo_map
from craft_taskgen.search.config import REPO_MAP_MAX_CHARS

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SolveShInfo:
    """Parsed solve.sh metadata."""

    commit_hash: str = ""
    upstream_url: str = ""
    checkout_paths: list[str] = field(default_factory=list)
    removed_paths: list[str] = field(default_factory=list)


@dataclass
class TestFunctionInfo:
    """Metadata about a single test function in gold_reference_tests.py."""

    name: str
    lineno: int
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    repo_imports: list[str] = field(default_factory=list)
    assertions: list[str] = field(default_factory=list)
    repo_calls: list[str] = field(default_factory=list)


@dataclass
class GoldTestMetadata:
    """Parsed metadata from gold_reference_tests.py."""

    docstring: str | None = None
    repo_imports: dict[str, list[str]] = field(default_factory=dict)
    test_functions: list[TestFunctionInfo] = field(default_factory=list)
    test_classes: list[str] = field(default_factory=list)
    is_documentation_only: bool = False


@dataclass
class T2GoldContext:
    """Complete extracted context for a Tools-track task."""

    task_id: str
    task_dir: str
    instruction: str
    difficulty: str
    solve_info: SolveShInfo
    gold_test_metadata: GoldTestMetadata
    source_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    enriched_functions: list[str] = field(default_factory=list)
    enriched_alt_functions: list[str] = field(default_factory=list)
    diff_functions: list[str] = field(default_factory=list)
    repo_map: str = ""


# ---------------------------------------------------------------------------
# solve.sh parser
# ---------------------------------------------------------------------------

_COMMIT_RE = re.compile(r"^COMMIT=([a-f0-9]+)", re.MULTILINE)
_UPSTREAM_RE = re.compile(r"git remote add upstream\s+(https?://\S+)", re.MULTILINE)
_CHECKOUT_RE = re.compile(
    r"git checkout FETCH_HEAD\s+--\s+(.*?)(?:\n\s*\n|\necho|\nrm|\n#|\Z)",
    re.DOTALL,
)
_RM_RE = re.compile(r"^rm\s+-f\s+(.+)$", re.MULTILINE)


def parse_solve_sh(path: str) -> SolveShInfo:
    """Parse a solve.sh file for commit, upstream, and checkout paths."""
    text = Path(path).read_text()
    info = SolveShInfo()

    m = _COMMIT_RE.search(text)
    if m:
        info.commit_hash = m.group(1)

    m = _UPSTREAM_RE.search(text)
    if m:
        info.upstream_url = m.group(1)

    m = _CHECKOUT_RE.search(text)
    if m:
        raw = m.group(1)
        raw = raw.replace("\\\n", " ")
        for var_match in re.finditer(r'(\w+)=(["\']?)(.+?)\2\s*$', text, re.MULTILINE):
            var_name, _, var_value = var_match.groups()
            raw = raw.replace(f'"${{{var_name}}}"', var_value)
            raw = raw.replace(f"${{{var_name}}}", var_value)
            raw = raw.replace(f"${var_name}", var_value)
        paths = []
        for token in raw.split():
            token = token.strip('"').strip("'").rstrip("\\")
            if token and not token.startswith("#"):
                paths.append(token)
        info.checkout_paths = paths

    for m in _RM_RE.finditer(text):
        info.removed_paths.append(m.group(1).strip())

    return info


# ---------------------------------------------------------------------------
# Dockerfile / patch fallbacks for v3 patch-apply format
# ---------------------------------------------------------------------------

# solve.sh in the v3 format just runs `git apply /solution/changes.patch`, so
# upstream URL + target commit live in environment/Dockerfile, and the file
# list lives in solution/changes.patch.

_DOCKERFILE_CLONE_RE = re.compile(r"git\s+clone\s+([^\n&]+)")
_DOCKERFILE_CHECKOUT_RE = re.compile(r"git\s+checkout\s+([a-f0-9]{7,40})\b")
_PATCH_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def parse_environment_dockerfile(path: str) -> tuple[str, str]:
    """Return (upstream_url, commit_hash) from an environment/Dockerfile."""
    if not os.path.exists(path):
        return "", ""
    text = Path(path).read_text(errors="replace")

    clone_match = _DOCKERFILE_CLONE_RE.search(text)
    if not clone_match:
        return "", ""
    # The regex captures everything after `git clone`; peel off flag tokens
    # like `--filter=blob:none` to find the actual URL.
    url = ""
    for tok in clone_match.group(1).split():
        tok = tok.strip().strip('"').strip("'")
        if tok and not tok.startswith("-"):
            url = tok
            break
    if not url:
        return "", ""

    checkout_match = _DOCKERFILE_CHECKOUT_RE.search(text, clone_match.end())
    commit = checkout_match.group(1) if checkout_match else ""
    return url, commit


def parse_patch_files(path: str) -> list[str]:
    """Return ordered list of file paths touched by a git-format unified patch."""
    if not os.path.exists(path):
        return []
    text = Path(path).read_text(errors="replace")
    paths: list[str] = []
    seen: set[str] = set()
    for m in _PATCH_DIFF_HEADER_RE.finditer(text):
        p = m.group(2)
        if p and p not in seen:
            seen.add(p)
            paths.append(p)
    return paths


def classify_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    """Split paths into source files and test files."""
    source = []
    test = []
    for p in paths:
        stripped = p.rstrip("/")
        basename = os.path.basename(stripped)
        is_test = (
            basename.startswith("test_")
            or basename.endswith("_test.py")
            or "/test/" in p
            or "/tests/" in p
            or stripped == "test"
            or stripped == "tests"
            or stripped.startswith("test/")
            or stripped.startswith("tests/")
            or stripped.startswith("egs3/")
        )
        if is_test:
            test.append(p)
        else:
            source.append(p)
    return source, test


# ---------------------------------------------------------------------------
# gold_reference_tests.py parser
# ---------------------------------------------------------------------------


def parse_gold_tests(path: str) -> GoldTestMetadata:
    """Parse gold_reference_tests.py using AST to extract test metadata."""
    text = Path(path).read_text()
    meta = GoldTestMetadata()

    try:
        tree = ast.parse(text)
    except SyntaxError:
        meta.is_documentation_only = True
        meta.docstring = text
        return meta

    if len(text.strip().splitlines()) < 25:
        has_test_func = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
            for node in ast.walk(tree)
        )
        if not has_test_func:
            meta.is_documentation_only = True
            meta.docstring = ast.get_docstring(tree)
            return meta

    meta.docstring = ast.get_docstring(tree)

    stdlib_and_test = {
        "pytest",
        "asyncio",
        "contextlib",
        "typing",
        "unittest",
        "os",
        "sys",
        "io",
        "json",
        "re",
        "pathlib",
        "functools",
        "collections",
        "copy",
        "time",
        "datetime",
        "tempfile",
        "textwrap",
        "inspect",
        "importlib",
        "abc",
        "enum",
        "dataclasses",
        "warnings",
        "logging",
        "threading",
        "multiprocessing",
        "socket",
        "struct",
        "hashlib",
        "itertools",
        "operator",
        "math",
        "decimal",
        "fractions",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            top_module = node.module.split(".")[0]
            if top_module not in stdlib_and_test and top_module != "__future__":
                names = [alias.name for alias in node.names if alias.name != "*"]
                if names:
                    meta.repo_imports[node.module] = names

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            meta.test_classes.append(node.name)

    source_lines = text.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("test_"):
                continue

            info = TestFunctionInfo(
                name=node.name,
                lineno=node.lineno,
                docstring=ast.get_docstring(node),
            )

            for dec in node.decorator_list:
                info.decorators.append(ast.dump(dec))

            func_source = ast.get_source_segment(text, node) or ""
            for module, names in meta.repo_imports.items():
                for name in names:
                    if name in func_source:
                        qualified = f"{module}.{name}"
                        if qualified not in info.repo_imports:
                            info.repo_imports.append(qualified)

            for child in ast.walk(node):
                if isinstance(child, ast.Assert):
                    try:
                        start = child.lineno - 1
                        end = child.end_lineno or child.lineno
                        assertion_text = "\n".join(source_lines[start:end]).strip()
                        info.assertions.append(assertion_text)
                    except (IndexError, AttributeError):
                        pass
                elif isinstance(child, ast.Call):
                    if _is_pytest_raises(child):
                        info.assertions.append(f"pytest.raises({_get_raises_arg(child)})")

            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    call_name = _get_call_name(child)
                    if call_name and _looks_like_repo_call(call_name, meta.repo_imports):
                        if call_name not in info.repo_calls:
                            info.repo_calls.append(call_name)

            meta.test_functions.append(info)

    return meta


def _is_pytest_raises(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Attribute):
        if node.func.attr == "raises" and isinstance(node.func.value, ast.Name):
            return node.func.value.id == "pytest"
    return False


def _get_raises_arg(node: ast.Call) -> str:
    if node.args:
        arg = node.args[0]
        if isinstance(arg, ast.Name):
            return arg.id
        if isinstance(arg, ast.Attribute):
            return _attr_chain(arg)
    return "..."


def _get_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return _attr_chain(node.func)
    return None


def _attr_chain(node: ast.Attribute) -> str:
    parts = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _looks_like_repo_call(name: str, repo_imports: dict[str, list[str]]) -> bool:
    first = name.split(".")[0]
    for module, names in repo_imports.items():
        if first in names:
            return True
        if first in module.split("."):
            return True
    return False


# ---------------------------------------------------------------------------
# postmerge_tests/ aggregator (v3 patch-apply format)
# ---------------------------------------------------------------------------


def _parse_f2p_entries(f2p_path: str) -> dict[str, set[str]]:
    """Map relative test file path -> set of test function base names."""
    wanted: dict[str, set[str]] = {}
    for line in Path(f2p_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "::" not in line:
            continue
        rel_path, _, tail = line.partition("::")
        # Strip parametrize suffix `[...]` and any `ClassName::` prefix so we
        # end up with the bare function name to match against AST nodes.
        base = tail.split("[", 1)[0]
        if "::" in base:
            base = base.rsplit("::", 1)[-1]
        wanted.setdefault(rel_path, set()).add(base)
    return wanted


def parse_postmerge_tests(task_dir: str) -> GoldTestMetadata:
    """Aggregate gold metadata from tests/postmerge_tests/ using fail_to_pass.txt.

    Used for v3 task format where gold tests are spread across multiple files
    under tests/postmerge_tests/ rather than in a single gold_reference_tests.py.
    """
    tests_dir = os.path.join(task_dir, "tests")
    postmerge_dir = os.path.join(tests_dir, "postmerge_tests")
    f2p_path = os.path.join(tests_dir, "fail_to_pass.txt")
    if not os.path.isdir(postmerge_dir) or not os.path.exists(f2p_path):
        return GoldTestMetadata()

    wanted = _parse_f2p_entries(f2p_path)
    merged = GoldTestMetadata()

    for rel_path, names in wanted.items():
        file_path = os.path.join(postmerge_dir, rel_path)
        if not os.path.exists(file_path):
            continue
        file_meta = parse_gold_tests(file_path)
        if file_meta.is_documentation_only:
            continue

        for module, imports in file_meta.repo_imports.items():
            dest = merged.repo_imports.setdefault(module, [])
            for imp in imports:
                if imp not in dest:
                    dest.append(imp)

        for cls in file_meta.test_classes:
            if cls not in merged.test_classes:
                merged.test_classes.append(cls)

        for tf in file_meta.test_functions:
            if tf.name in names:
                merged.test_functions.append(tf)

    return merged


# ---------------------------------------------------------------------------
# Git diff resolution
# ---------------------------------------------------------------------------


def resolve_changed_files(
    repo_dir: str, commit_hash: str, checkout_paths: list[str]
) -> tuple[list[str], list[str]]:
    """Resolve directory-level checkout paths to actual changed files via git diff."""
    try:
        subprocess.run(
            ["git", "-C", repo_dir, "cat-file", "-t", commit_hash],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return classify_paths(checkout_paths)

    try:
        parent = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", f"{commit_hash}^"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return classify_paths(checkout_paths)

    try:
        diff_output = subprocess.run(
            ["git", "-C", repo_dir, "diff", "--name-only", parent, commit_hash],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return classify_paths(checkout_paths)

    changed_files = [f for f in diff_output.splitlines() if f.strip()]

    filtered = []
    for f in changed_files:
        for cp in checkout_paths:
            cp_stripped = cp.rstrip("/")
            if f == cp_stripped or f.startswith(cp_stripped + "/") or f.startswith(cp_stripped + "."):
                filtered.append(f)
                break

    if not filtered:
        filtered = changed_files

    return classify_paths(filtered)


def resolve_changed_functions(repo_dir: str, commit_hash: str) -> list[str]:
    """Extract function/method definitions changed in a commit via git diff."""
    try:
        parent = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", f"{commit_hash}^"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return []

    try:
        diff = subprocess.run(
            ["git", "-C", repo_dir, "diff", "-U0", parent, commit_hash],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return []

    functions = []
    current_file = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            stripped = line[1:].strip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                func_name = stripped.split("(")[0].replace("def ", "").replace("async ", "").strip()
                if current_file and func_name:
                    module = current_file.replace("/", ".").replace(".py", "")
                    qualified = f"{module}.{func_name}"
                    if qualified not in functions:
                        functions.append(qualified)

    return functions


def enrich_gold_functions(gold_meta: dict) -> tuple[list[str], list[str]]:
    """Build method-level gold functions from test repo_calls + diff functions."""
    name_to_module: dict[str, str] = {}
    for module, names in gold_meta.get("repo_imports", {}).items():
        if module.startswith("tests."):
            continue
        for name in names:
            name_to_module[name] = module

    primary = []
    seen: set[str] = set()
    for tf in gold_meta.get("test_functions", []):
        for call in tf.get("repo_calls", []):
            if call.startswith("tests."):
                continue
            parts = call.split(".")
            root = parts[0]
            if root in name_to_module:
                module = name_to_module[root]
                qualified = f"{module}.{call}"
            elif "." in call:
                qualified = call
            else:
                continue
            if qualified not in seen:
                primary.append(qualified)
                seen.add(qualified)

    alt = []
    for module, names in gold_meta.get("repo_imports", {}).items():
        if module.startswith("tests."):
            continue
        for name in names:
            qualified = f"{module}.{name}"
            if qualified not in seen:
                alt.append(qualified)
                seen.add(qualified)

    return primary, alt


# ---------------------------------------------------------------------------
# instruction.md parser
# ---------------------------------------------------------------------------


def parse_instruction(path: str) -> str:
    """Read instruction.md and return the description text only."""
    from craft_taskgen.config import PipelineContext

    text = Path(path).read_text()
    preamble = PipelineContext().instruction_preamble
    if text.startswith(preamble):
        text = text[len(preamble) :].strip()
    else:
        # Old format: strip "#..." header and ## Environment section
        if not text.startswith("# "):
            print(
                f"    [parse_instruction] WARNING: {path} does not start with expected preamble "
                f"or '#' header — parsed text may be incomplete"
            )
        text = re.sub(r"^#\s+\S.*\n", "", text).strip()
        text = re.sub(r"\n##\s+Environment\n.*", "", text, flags=re.DOTALL).strip()
    return text


# ---------------------------------------------------------------------------
# task.toml parser
# ---------------------------------------------------------------------------


def parse_task_toml(path: str) -> str:
    """Extract difficulty from task.toml."""
    text = Path(path).read_text()
    m = re.search(r'difficulty\s*=\s*"(\w+)"', text)
    return m.group(1) if m else "hard"


# ---------------------------------------------------------------------------
# Main extraction (parsing only -- no repo map mining)
# ---------------------------------------------------------------------------


def extract_task(task_dir: str, repos_dir: str | None = None) -> T2GoldContext:
    """Extract gold context from a single Tools-track task directory.

    This performs all parsing and git-diff resolution but does NOT mine repo maps
    (that requires tree-sitter / RepoIndexer from craft-bench).
    """
    task_id = os.path.basename(task_dir)

    solve_path = os.path.join(task_dir, "solution", "solve.sh")
    gold_tests_path = os.path.join(task_dir, "tests", "gold_reference_tests.py")
    instruction_path = os.path.join(task_dir, "instruction.md")
    toml_path = os.path.join(task_dir, "task.toml")
    dockerfile_path = os.path.join(task_dir, "environment", "Dockerfile")
    patch_path = os.path.join(task_dir, "solution", "changes.patch")

    if not os.path.exists(solve_path):
        raise FileNotFoundError(f"Missing required file: {solve_path}")
    if not os.path.exists(instruction_path):
        raise FileNotFoundError(f"Missing required file: {instruction_path}")

    solve_info = parse_solve_sh(solve_path)
    # v3 patch-apply format: solve.sh only invokes `git apply changes.patch`,
    # so upstream URL + target commit come from the Dockerfile and the file
    # list comes from the patch itself.
    if not solve_info.upstream_url or not solve_info.commit_hash:
        docker_url, docker_commit = parse_environment_dockerfile(dockerfile_path)
        if not solve_info.upstream_url:
            solve_info.upstream_url = docker_url
        if not solve_info.commit_hash:
            solve_info.commit_hash = docker_commit
    if not solve_info.checkout_paths:
        solve_info.checkout_paths = parse_patch_files(patch_path)

    if os.path.exists(gold_tests_path):
        gold_meta = parse_gold_tests(gold_tests_path)
    else:
        gold_meta = parse_postmerge_tests(task_dir)
    instruction = parse_instruction(instruction_path)
    difficulty = parse_task_toml(toml_path) if os.path.exists(toml_path) else "hard"

    url = solve_info.upstream_url
    repo_name = url.rstrip("/").removesuffix(".git").split("/")[-1] if url else ""
    repo_dir = os.path.join(repos_dir, repo_name) if repos_dir and repo_name else None

    if repo_dir and os.path.isdir(repo_dir) and solve_info.commit_hash:
        source_files, test_files = resolve_changed_files(
            repo_dir, solve_info.commit_hash, solve_info.checkout_paths
        )
        diff_functions = resolve_changed_functions(repo_dir, solve_info.commit_hash)
    else:
        source_files, test_files = classify_paths(solve_info.checkout_paths)
        diff_functions = []

    gold_meta_dict = {
        "repo_imports": gold_meta.repo_imports,
        "test_functions": [
            {"repo_calls": tf.repo_calls, "repo_imports": tf.repo_imports} for tf in gold_meta.test_functions
        ],
    }
    enriched_funcs, enriched_alt = enrich_gold_functions(gold_meta_dict)

    # Build repo map at the PRE-CHANGE commit (parent of the T2 reference commit).
    # This ensures the LLM only sees code that exists in the agent's Docker environment.
    repo_map = ""
    if repo_dir and os.path.isdir(repo_dir) and solve_info.commit_hash:
        try:
            parent_commit = subprocess.run(
                ["git", "-C", repo_dir, "rev-parse", f"{solve_info.commit_hash}^"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

            # Checkout pre-change commit
            subprocess.run(["git", "-C", repo_dir, "stash", "-q"], capture_output=True)
            subprocess.run(
                ["git", "-C", repo_dir, "checkout", parent_commit, "-q"],
                capture_output=True,
                check=True,
            )
            print(f"  checked out pre-change commit {parent_commit[:12]}")

            indexer = RepoIndexer(repo_dir)
            gt = indexer.index()
            # Pass patch-touched files as priority so critical context is
            # preserved when the char budget is tight (e.g. large repos like
            # spack where the relevant file was being truncated out).
            priority = [p for p in solve_info.checkout_paths if p.endswith(".py")]
            repo_map = build_repo_map(gt, max_chars=REPO_MAP_MAX_CHARS, priority_files=priority)
            print(f"  repo_map: {len(repo_map):,} chars ({len(priority)} priority files)")

        except Exception as e:
            print(f"  WARNING: repo map failed: {e}", file=sys.stderr)
        finally:
            # Always restore HEAD
            subprocess.run(["git", "-C", repo_dir, "checkout", "-", "-q"], capture_output=True)
            subprocess.run(["git", "-C", repo_dir, "stash", "pop", "-q"], capture_output=True)

    return T2GoldContext(
        task_id=task_id,
        task_dir=task_dir,
        instruction=instruction,
        difficulty=difficulty,
        solve_info=solve_info,
        gold_test_metadata=gold_meta,
        source_files=source_files,
        test_files=test_files,
        enriched_functions=enriched_funcs,
        enriched_alt_functions=enriched_alt,
        diff_functions=diff_functions,
        repo_map=repo_map,
    )


def context_to_dict(ctx: T2GoldContext) -> dict:
    """Convert a T2GoldContext to a JSON-serializable dict."""
    return {
        "task_id": ctx.task_id,
        "task_dir": ctx.task_dir,
        "instruction": ctx.instruction,
        "difficulty": ctx.difficulty,
        "solve_info": {
            "commit_hash": ctx.solve_info.commit_hash,
            "upstream_url": ctx.solve_info.upstream_url,
            "checkout_paths": ctx.solve_info.checkout_paths,
            "removed_paths": ctx.solve_info.removed_paths,
        },
        "source_files": ctx.source_files,
        "test_files": ctx.test_files,
        "enriched_functions": ctx.enriched_functions,
        "enriched_alt_functions": ctx.enriched_alt_functions,
        "diff_functions": ctx.diff_functions,
        "repo_map": ctx.repo_map,
        "gold_test_metadata": {
            "docstring": ctx.gold_test_metadata.docstring,
            "repo_imports": ctx.gold_test_metadata.repo_imports,
            "test_classes": ctx.gold_test_metadata.test_classes,
            "is_documentation_only": ctx.gold_test_metadata.is_documentation_only,
            "test_functions": [
                {
                    "name": tf.name,
                    "lineno": tf.lineno,
                    "docstring": tf.docstring,
                    "decorators": tf.decorators,
                    "repo_imports": tf.repo_imports,
                    "assertions": tf.assertions,
                    "repo_calls": tf.repo_calls,
                }
                for tf in ctx.gold_test_metadata.test_functions
            ],
        },
    }


# ---------------------------------------------------------------------------
# Top-level entry point (called by step_extract)
# ---------------------------------------------------------------------------


def run_extract(
    tasks_dir: str,
    output_dir: str,
    repos_dir: str = "repos",
    task_filter: str | None = None,
) -> None:
    """Process all tasks in tasks_dir, write per-task context JSONs and _all_contexts.json.

    Parameters
    ----------
    tasks_dir:
        Directory containing Tools-track task subdirectories.
    output_dir:
        Output directory for extracted context files.
    repos_dir:
        Directory containing cloned repos (for git diff resolution + repo maps).
    task_filter:
        If set, only extract this single task ID.
    """
    os.makedirs(output_dir, exist_ok=True)

    if task_filter:
        task_dirs = [os.path.join(tasks_dir, task_filter)]
    else:
        task_dirs = sorted(
            os.path.join(tasks_dir, d)
            for d in os.listdir(tasks_dir)
            if os.path.isdir(os.path.join(tasks_dir, d)) and d.startswith("t2v3-")
        )

    if not task_dirs:
        print(f"WARNING: No task directories found in {tasks_dir}", file=sys.stderr)

    results = []
    for task_dir in task_dirs:
        task_id = os.path.basename(task_dir)
        print(f"Extracting: {task_id}")
        try:
            ctx = extract_task(task_dir, repos_dir=repos_dir)
            data = context_to_dict(ctx)
            results.append(data)

            # Write per-task file
            out_path = os.path.join(output_dir, f"{task_id}.json")
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  -> {out_path}")

            # Summary
            n_src = len(ctx.source_files)
            n_test = len(ctx.test_files)
            n_funcs = len(ctx.gold_test_metadata.test_functions)
            doc_only = ctx.gold_test_metadata.is_documentation_only
            print(
                f"  source_files={n_src}, test_files={n_test}, "
                f"gold_test_functions={n_funcs}, doc_only={doc_only}"
            )
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    # Write combined output
    combined_path = os.path.join(output_dir, "_all_contexts.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} contexts to {combined_path}")
