# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Search Harbor converter.

Reads TaskCandidate JSON files produced by the search synthesis pipeline and
writes Harbor task directories. Unlike the from-T2 converter (which reuses a
parent Dockerfile), this builds a fresh Dockerfile per task that clones the
repo at a pinned commit and installs the agent runtimes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from craft_taskgen.adapters._docker import (
    CLAUDE_CODE_VERSION,
    CODEX_VERSION,
    HARBOR_COMMIT,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    OPENCODE_VERSION,
    _extract_base_image_digest,
)
from craft_taskgen.search._harbor_utils import (
    TIER_TIMEOUTS,
    load_instruction_template,
    load_verifier_files,
    task_id,
    write_gold_answer,
    write_instruction,
    write_registry,
    write_search_verifier,
    write_solve_sh,
    write_task_toml,
)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_SUITE_NAME = "craft-search"


# ---------------------------------------------------------------------------
# Native-specific helpers
# ---------------------------------------------------------------------------


def load_dockerfile_template() -> str:
    """Load the native Dockerfile template."""
    with open(_TEMPLATES_DIR / "Dockerfile.template") as f:
        return f.read()


def write_dockerfile(env_dir: str, repo_url: str, repo_commit: str, template: str) -> str:
    """Render and write environment/Dockerfile from the template. Returns the rendered text."""
    # The gateway base URL is environment-specific — read it at render time so no
    # endpoint is hardcoded in the template.
    gateway_base_url = os.environ.get("OPENAI_BASE_URL", "")
    if not gateway_base_url:
        raise RuntimeError(
            "OPENAI_BASE_URL must be set — it is written into the task image's "
            "codex config as the inference gateway base_url."
        )
    content = (
        template.replace("{repo_url}", repo_url)
        .replace("{repo_commit}", repo_commit)
        .replace("{claude_code_version}", CLAUDE_CODE_VERSION)
        .replace("{codex_version}", CODEX_VERSION)
        .replace("{opencode_version}", OPENCODE_VERSION)
        .replace("{gateway_base_url}", gateway_base_url)
    )
    with open(os.path.join(env_dir, "Dockerfile"), "w") as f:
        f.write(content)
    return content


def _write_manifest(
    env_dir: str, repo: str, repo_commit: str, dockerfile_text: str, task_id_str: str
) -> None:
    """Emit the per-task reproducibility manifest alongside the Dockerfile."""
    base = _extract_base_image_digest(dockerfile_text)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "adapter": "search-native",
        "produced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_image": base,
        "python": "3.10",  # ubuntu:22.04 default
        "agents": {
            "claude_code": CLAUDE_CODE_VERSION,
            "codex": CODEX_VERSION,
            "opencode": OPENCODE_VERSION,
        },
        "harbor_commit": HARBOR_COMMIT,
        "repo": repo,
        "parent_sha": repo_commit,
        "task_id": task_id_str,
    }
    with open(os.path.join(env_dir, MANIFEST_FILENAME), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def _validate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the inner task dict, validating required fields."""
    if "task" not in candidate:
        raise ValueError("Candidate missing 'task' key")
    task = candidate["task"]
    for required in ("id", "repo", "instruction", "tier", "gold_answer"):
        if required not in task:
            raise ValueError(f"task missing required field '{required}'")
    gold = task["gold_answer"]
    for required in ("files", "functions"):
        if required not in gold:
            raise ValueError(f"gold_answer missing required field '{required}'")
    return task


# ---------------------------------------------------------------------------
# Single task conversion
# ---------------------------------------------------------------------------


def _convert_task_loaded(
    candidate: dict[str, Any],
    manifest: dict[str, dict[str, str]],
    output_dir: str,
    *,
    instruction_template: str,
    dockerfile_template: str,
    verifier_files: dict[str, bytes],
) -> str:
    """Inner converter used by run_convert; assumes templates are already loaded."""
    task = _validate_candidate(candidate)
    repo = task["repo"]
    repo_info = manifest.get(repo)
    if repo_info is None:
        raise ValueError(f"Repo '{repo}' not found in manifest")
    if "url" not in repo_info or "commit" not in repo_info:
        raise ValueError(f"Manifest entry for '{repo}' missing url or commit")

    tid = task_id(repo, task["id"])
    difficulty = task["tier"]
    agent_timeout = TIER_TIMEOUTS.get(difficulty, 1200)

    task_dir = os.path.join(output_dir, tid)
    env_dir = os.path.join(task_dir, "environment")
    tests_dir = os.path.join(task_dir, "tests")
    sol_dir = os.path.join(task_dir, "solution")
    for d in (task_dir, env_dir, tests_dir, sol_dir):
        os.makedirs(d, exist_ok=True)

    write_task_toml(task_dir, tid, difficulty, agent_timeout)
    write_instruction(task_dir, task["instruction"], template=instruction_template)
    dockerfile_text = write_dockerfile(env_dir, repo_info["url"], repo_info["commit"], dockerfile_template)
    _write_manifest(env_dir, repo, repo_info["commit"], dockerfile_text, tid)
    write_search_verifier(tests_dir, verifier_files)

    gold = task["gold_answer"]
    write_gold_answer(tests_dir, gold)
    write_solve_sh(sol_dir, gold)

    return task_dir


def convert_task(
    candidate: dict[str, Any],
    manifest: dict[str, dict[str, str]],
    output_dir: str,
) -> str:
    """Convert a single native search candidate to a Harbor task directory.

    Loads templates from disk each call. For batch conversion, use `run_convert`
    which pre-loads templates once.
    """
    return _convert_task_loaded(
        candidate,
        manifest,
        output_dir,
        instruction_template=load_instruction_template(),
        dockerfile_template=load_dockerfile_template(),
        verifier_files=load_verifier_files(),
    )


# ---------------------------------------------------------------------------
# Top-level conversion
# ---------------------------------------------------------------------------


def _archive_stale_tasks(output_dir: str, expected_ids: set[str]) -> list[str]:
    """Move task dirs not in expected_ids to a timestamped `_stale_<ts>/` subdir.

    Returns the list of archived task IDs. Empty list if nothing archived.
    Caller is responsible for ensuring output_dir exists.
    """
    stale: list[str] = []
    for entry in sorted(os.listdir(output_dir)):
        if not entry.startswith("craft-") or entry in expected_ids:
            continue
        if os.path.isdir(os.path.join(output_dir, entry)):
            stale.append(entry)

    if not stale:
        return []

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = os.path.join(output_dir, f"_stale_{stamp}")
    os.makedirs(archive_dir, exist_ok=True)
    for entry in stale:
        shutil.move(os.path.join(output_dir, entry), os.path.join(archive_dir, entry))
    return stale


def _iter_candidate_files(candidates_dir: str) -> Iterator[tuple[str, str]]:
    """Yield (repo_name, filepath) for every candidate JSON under candidates_dir."""
    for repo_name in sorted(os.listdir(candidates_dir)):
        repo_dir = os.path.join(candidates_dir, repo_name)
        if not os.path.isdir(repo_dir):
            continue
        for filename in sorted(os.listdir(repo_dir)):
            if not filename.endswith(".json"):
                continue
            yield repo_name, os.path.join(repo_dir, filename)


def run_convert(
    candidates_dir: str,
    manifest_path: str,
    output_dir: str,
    *,
    limit: int = 0,
) -> dict[str, Any]:
    """Convert all candidates under candidates_dir to Harbor task directories.

    Args:
        candidates_dir: Root dir with per-repo candidate subdirs, each containing
            `{uuid}.json` TaskCandidate files.
        manifest_path: Path to repos/manifest.json (repo_name -> {url, commit}).
        output_dir: Output directory for Harbor search task directories.
        limit: Stop after this many tasks (0 = all).

    Returns:
        Dict with `converted` (count), `task_ids`, `archived` (stale task IDs).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    instruction_template = load_instruction_template()
    dockerfile_template = load_dockerfile_template()
    verifier_files = load_verifier_files()

    task_ids: list[str] = []
    errors = 0

    for _repo, filepath in _iter_candidate_files(candidates_dir):
        if limit and len(task_ids) >= limit:
            break
        try:
            with open(filepath) as f:
                candidate = json.load(f)
            task_dir = _convert_task_loaded(
                candidate,
                manifest,
                output_dir,
                instruction_template=instruction_template,
                dockerfile_template=dockerfile_template,
                verifier_files=verifier_files,
            )
            tid = os.path.basename(task_dir)
            task_ids.append(tid)
            task_tier = candidate.get("task", {}).get("tier", "?")
            print(f"  [{len(task_ids)}] {tid} ({task_tier})")
        except Exception as e:
            errors += 1
            print(f"  ERROR: {filepath}: {e}", file=sys.stderr)

    # Archive stale first so registry reflects the final clean state
    archived = _archive_stale_tasks(output_dir, set(task_ids))
    registry_path = write_registry(output_dir, task_ids, suite_name=_SUITE_NAME)

    if archived:
        print(f"\nArchived {len(archived)} stale task dirs")
    print(f"\nConverted {len(task_ids)} tasks ({errors} errors)")
    print(f"Output: {output_dir}")
    print(f"Registry: {registry_path}")

    return {
        "converted": len(task_ids),
        "errors": errors,
        "task_ids": task_ids,
        "archived": archived,
    }
