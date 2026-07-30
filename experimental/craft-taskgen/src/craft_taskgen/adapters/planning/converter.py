# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning track Harbor task converter.

Reads planning TaskCandidate JSON files produced by the upstream bootstrap
step (one JSON per task, flat under ``candidates_dir``) and writes Harbor
task directories with a binary F2P + P2P reward gate.

Each candidate JSON must contain:

    task_name, repo, parent_sha, merge_sha, spec,
    src_files, test_files, test_command,
    fail_to_pass, pass_to_pass,
    docker = { python, install, pre_install?, ... }

Optional fields: abbrev, pr, category, gold_symbols, removed_files.

Optional per-task resource overrides (with defaults):

    memory_mb          default 4096
    storage_mb         default 10240
    build_timeout_sec  default 900.0

Unlike the search-native adapter, this one clones each task's repo from
GitHub to extract post-merge test files — there is no manifest step, since
each candidate already names its own repo + SHAs.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from craft_taskgen.adapters._docker import (
    build_dockerfile,
    produce_manifest,
    spec_from_candidate,
    write_environment,
)

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_SUITE_NAME = "craft-planning"

_REQUIRED_CANDIDATE_FIELDS = (
    "task_name",
    "repo",
    "parent_sha",
    "merge_sha",
    "spec",
    "src_files",
    "test_files",
    "test_command",
    "fail_to_pass",
    "pass_to_pass",
    "docker",
)

_DEFAULT_MEMORY_MB = 4096
_DEFAULT_STORAGE_MB = 10240
_DEFAULT_BUILD_TIMEOUT_SEC = 900.0


# ---------------------------------------------------------------------------
# Inline templates (short enough to keep in Python)
# ---------------------------------------------------------------------------

_TASK_TOML_TEMPLATE = """\
version = "1.0"

[metadata]
name = "{task_name}"
difficulty = "hard"

[verifier]
timeout_sec = 600

[agent]
timeout_sec = 3600

[environment]
build_timeout_sec = {build_timeout}
cpus = 2
memory_mb = {memory_mb}
storage_mb = {storage_mb}
gpus = 0
allow_internet = true
mcp_servers = []

[environment.env]
ANTHROPIC_API_KEY = "${{ANTHROPIC_API_KEY}}"
ANTHROPIC_BASE_URL = "${{ANTHROPIC_BASE_URL}}"
OPENAI_API_KEY = "${{OPENAI_API_KEY}}"
OPENAI_BASE_URL = "${{OPENAI_BASE_URL}}"

[solution.env]
"""

_INSTRUCTION_TEMPLATE = """\
{spec}

## Environment

The project is at `/repo/`. Write any output files to the `/repo/output/` directory.
"""

_SOLVE_SH_TEMPLATE = """\
#!/bin/bash
set -euo pipefail
COMMIT={merge_sha}

cd /repo
git remote add upstream https://github.com/{repo}.git 2>/dev/null || true
git fetch --depth 2 upstream "$COMMIT"
{checkout_lines}
echo "Oracle solution applied. Run verifier to confirm."
"""

_P2P_BLOCK_TEMPLATE = """\
echo "Running P2P regression tests ({n_p2p} tests)..."
run_check "P2P regression tests pass" python3 -m pytest \\
    $(cat /tests/pass_to_pass.txt | tr '\\n' ' ') \\
    {p2p_extra_flags}-v -p no:cacheprovider -o 'addopts=' --continue-on-collection-errors"""


# ---------------------------------------------------------------------------
# Template loaders
# ---------------------------------------------------------------------------


def _read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text()


def _load_all_templates() -> dict[str, str]:
    return {
        "score_py": _read_template("score.py"),
        "test_sh": _read_template("test.sh"),
        "test_runner_py": _read_template("test_runner.py.template"),
        "verify_sh": _read_template("verify.sh.template"),
    }


# ---------------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------------


def _validate_candidate(candidate: dict[str, Any]) -> None:
    missing = [f for f in _REQUIRED_CANDIDATE_FIELDS if f not in candidate]
    if missing:
        raise ValueError(
            f"Candidate missing required fields: {missing} "
            f"(task_name={candidate.get('task_name', '<unknown>')})"
        )
    docker = candidate["docker"]
    if "install" not in docker:
        raise ValueError(
            f"Candidate docker config missing 'install' key (task_name={candidate['task_name']})"
        )


# ---------------------------------------------------------------------------
# Repo cache (for extracting post-merge test files)
# ---------------------------------------------------------------------------


def _clone_or_reuse(repo: str, cache_dir: Path) -> Path:
    owner, name = repo.split("/", 1)
    clone_path = cache_dir / f"{owner}__{name}"
    if not clone_path.exists():
        logger.info("Cloning %s ...", repo)
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", str(clone_path)],
            check=True,
            capture_output=True,
        )
    return clone_path


def _git_fetch(clone_path: Path, sha: str) -> None:
    subprocess.run(["git", "fetch", "origin", sha], cwd=clone_path, capture_output=True)


def _git_show(clone_path: Path, sha: str, filepath: str) -> bytes | None:
    result = subprocess.run(["git", "show", f"{sha}:{filepath}"], capture_output=True, cwd=clone_path)
    return result.stdout if result.returncode == 0 else None


# ---------------------------------------------------------------------------
# Verify-script construction
# ---------------------------------------------------------------------------


def _build_f2p_command(candidate: dict[str, Any]) -> str:
    test_cmd = candidate["test_command"].replace("/code/", "/repo/")
    if test_cmd.startswith("python3 -m pytest "):
        pytest_args = test_cmd[len("python3 -m pytest ") :]
    else:
        pytest_args = test_cmd

    if "-o 'addopts='" not in pytest_args:
        pytest_args += " -o 'addopts='"
    pytest_args = pytest_args.replace("--tb=short", "").replace("-v", "")
    pytest_args = " ".join(pytest_args.split())
    return f"python3 -m pytest {pytest_args} -v -p no:cacheprovider"


def _build_verify_sh(candidate: dict[str, Any], verify_template: str) -> str:
    f2p_command = _build_f2p_command(candidate)
    n_f2p = len(candidate["fail_to_pass"])
    n_p2p = len(candidate["pass_to_pass"])
    if n_p2p > 0:
        pytest_args = candidate["docker"].get("pytest_args", "")
        p2p_extra = f"{pytest_args} " if pytest_args else ""
        p2p_block = _P2P_BLOCK_TEMPLATE.format(n_p2p=n_p2p, p2p_extra_flags=p2p_extra) + "\n"
    else:
        p2p_block = ""
    return verify_template.format(n_f2p=n_f2p, f2p_command=f2p_command, p2p_block=p2p_block)


# ---------------------------------------------------------------------------
# Single task conversion
# ---------------------------------------------------------------------------


def _convert_single(
    candidate: dict[str, Any],
    output_dir: Path,
    repo_cache: Path,
    templates: dict[str, str],
) -> Path:
    _validate_candidate(candidate)

    task_name = candidate["task_name"]
    abbrev = candidate.get("abbrev") or task_name
    repo = candidate["repo"]

    logger.info("Generating %s ...", task_name)

    clone_path = _clone_or_reuse(repo, repo_cache)
    _git_fetch(clone_path, candidate["merge_sha"])
    _git_fetch(clone_path, candidate["parent_sha"])

    task_dir = output_dir / task_name
    if task_dir.exists():
        shutil.rmtree(task_dir)

    env_dir = task_dir / "environment"
    sol_dir = task_dir / "solution"
    test_dir = task_dir / "tests"
    postmerge_dir = test_dir / "postmerge"
    env_dir.mkdir(parents=True)
    sol_dir.mkdir(parents=True)
    postmerge_dir.mkdir(parents=True)

    # Optional per-task resource overrides; candidate declares what it needs,
    # adapter falls back to conservative defaults.
    memory_mb = candidate.get("memory_mb", _DEFAULT_MEMORY_MB)
    storage_mb = candidate.get("storage_mb", _DEFAULT_STORAGE_MB)
    build_timeout = candidate.get("build_timeout_sec", _DEFAULT_BUILD_TIMEOUT_SEC)

    (task_dir / "task.toml").write_text(
        _TASK_TOML_TEMPLATE.format(
            task_name=task_name,
            build_timeout=build_timeout,
            memory_mb=memory_mb,
            storage_mb=storage_mb,
        )
    )
    (task_dir / "instruction.md").write_text(_INSTRUCTION_TEMPLATE.format(spec=candidate["spec"]))
    docker_spec = spec_from_candidate(candidate)
    # Planning trials run the agent inside the task container. Bake the agent
    # CLIs at pinned versions so published benchmark numbers are reproducible
    # against a specific agent build, not whatever harbor installs at run time.
    docker_spec.bake_agents = True
    dockerfile = build_dockerfile(docker_spec)
    manifest = produce_manifest(
        docker_spec,
        adapter="planning",
        dockerfile=dockerfile,
        extra={"task_name": task_name, "merge_sha": candidate["merge_sha"]},
    )
    write_environment(env_dir, dockerfile, docker_spec.pinned_requirements, manifest=manifest)

    # shlex.quote every candidate-derived value before it lands in the generated
    # bash script; these fields come from untrusted PR/candidate JSON and would
    # otherwise allow shell command injection (CWE-78).
    checkout_lines: list[str] = []
    if candidate["src_files"]:
        files = " ".join(shlex.quote(f) for f in candidate["src_files"])
        checkout_lines.append(f"git checkout FETCH_HEAD -- {files}")
    for removed in candidate.get("removed_files", []) or []:
        checkout_lines.append(f"rm -f {shlex.quote(removed)}")
    (sol_dir / "solve.sh").write_text(
        _SOLVE_SH_TEMPLATE.format(
            merge_sha=shlex.quote(candidate["merge_sha"]),
            repo=shlex.quote(repo),
            checkout_lines="\n".join(checkout_lines),
        )
    )
    (sol_dir / "solve.sh").chmod(0o755)

    for test_file in candidate["test_files"]:
        content = _git_show(clone_path, candidate["merge_sha"], test_file)
        if content is None:
            logger.warning(
                "  postmerge: %s NOT FOUND at merge commit %s",
                test_file,
                candidate["merge_sha"][:12],
            )
            continue
        dest = postmerge_dir / test_file
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

    (test_dir / "verify_{}.sh".format(abbrev)).write_text(_build_verify_sh(candidate, templates["verify_sh"]))
    (test_dir / "verify_{}.sh".format(abbrev)).chmod(0o755)

    (test_dir / "test.sh").write_text(templates["test_sh"])
    (test_dir / "test.sh").chmod(0o755)

    (test_dir / "test_runner.py").write_text(templates["test_runner_py"].format(abbrev=abbrev))
    (test_dir / "score.py").write_text(templates["score_py"])

    f2p_lines = "\n".join(candidate["fail_to_pass"]) + "\n" if candidate["fail_to_pass"] else ""
    (test_dir / "fail_to_pass.txt").write_text(f2p_lines)
    p2p_lines = "\n".join(candidate["pass_to_pass"]) + "\n" if candidate["pass_to_pass"] else ""
    (test_dir / "pass_to_pass.txt").write_text(p2p_lines)

    logger.info(
        "  %s: %d F2P, %d P2P, %d src, %d tests",
        task_name,
        len(candidate["fail_to_pass"]),
        len(candidate["pass_to_pass"]),
        len(candidate["src_files"]),
        len(candidate["test_files"]),
    )
    return task_dir


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _write_registry(output_dir: Path, task_ids: list[str]) -> None:
    registry = [
        {
            "name": _SUITE_NAME,
            "version": "1.0",
            "description": f"Planning track ({len(task_ids)} tasks)",
            "metrics": [{"type": "mean"}],
            "tasks": [{"name": t, "path": t} for t in task_ids],
        }
    ]
    (output_dir / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_convert(
    candidates_dir: str,
    output_dir: str,
    *,
    repo_cache: str | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    """Convert all planning candidates to Harbor task directories.

    Args:
        candidates_dir: Directory containing one `{task_name}.json` per candidate.
        output_dir: Destination for Harbor task directories.
        repo_cache: Local dir for git clones (default: /tmp/craft-taskgen-repos).
        limit: Stop after this many tasks (0 = all).

    Returns:
        {"converted": int, "task_ids": list[str], "skipped": list[str]}
    """
    candidates_path = Path(candidates_dir)
    if not candidates_path.is_dir():
        raise FileNotFoundError(f"candidates_dir not found: {candidates_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_path = Path(repo_cache) if repo_cache else Path("/tmp/craft-taskgen-repos")
    cache_path.mkdir(parents=True, exist_ok=True)

    templates = _load_all_templates()

    task_ids: list[str] = []
    skipped: list[str] = []
    candidate_files = sorted(candidates_path.glob("*.json"))

    for path in candidate_files:
        if limit and len(task_ids) >= limit:
            break
        candidate = json.loads(path.read_text())
        try:
            _convert_single(candidate, output_path, cache_path, templates)
        except Exception as exc:
            logger.error("Skipping %s: %s", path.name, exc)
            skipped.append(path.stem)
            continue
        task_ids.append(candidate["task_name"])

    _write_registry(output_path, task_ids)

    logger.info("Converted %d/%d tasks to %s", len(task_ids), len(candidate_files), output_path)
    return {"converted": len(task_ids), "task_ids": task_ids, "skipped": skipped}
