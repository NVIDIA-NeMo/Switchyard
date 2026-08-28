# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert synthesized search tasks (from Tools problems) to Harbor task directories.

Bridges Tools-track environments with Search scoring:
- Reuses Tools-track Dockerfiles (repo + deps already set up)
- Uses search instruction template (write answer.json, not code)
- Uses search test_runner.py (scores answer.json against gold_answer.json)
- Generates solve.sh that writes the gold answer
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any

from craft_taskgen.adapters._docker import CLAUDE_CODE_VERSION, CODEX_VERSION
from craft_taskgen.search._harbor_utils import (
    TIER_TIMEOUTS,
    load_verifier_files,
    write_gold_answer,
    write_instruction,
    write_registry,
    write_search_verifier,
    write_solve_sh,
    write_task_toml,
)

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


# Agent CLIs are pinned so reruns against a given task set always use the
# same versions. Unversioned installs (plain `curl | bash` / `npm install -g`)
# would silently drift to whatever was "latest" at image-build time, then
# freeze under Docker layer caching so the "latest" at first-build lock-in
# propagates to every subsequent search-track trial. Explicit pin makes the
# CLI version an MR-level change, not a filesystem-mtime artifact.
def _agent_install_snippet() -> str:
    """Dockerfile tail installing the agent CLIs + codex gateway config.

    The gateway base URL comes from `OPENAI_BASE_URL` at render time — no
    endpoint is baked into the source.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if not base_url:
        raise RuntimeError(
            "OPENAI_BASE_URL must be set — it is written into the task image's "
            "codex config as the inference gateway base_url."
        )
    return f"""
# --- Agent runtimes + NVIDIA gateway config (appended by harbor.py) ---
RUN apt-get update && apt-get install -y --no-install-recommends curl && \\
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \\
    apt-get install -y --no-install-recommends nodejs && \\
    curl -fsSL https://claude.ai/install.sh | bash -s -- {CLAUDE_CODE_VERSION} && \\
    npm install -g @openai/codex@{CODEX_VERSION} && \\
    apt-get remove -y curl && apt-get autoremove -y && \\
    rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"
# Codex system-level config (NVIDIA LiteLLM gateway)
RUN mkdir -p /etc/codex && cat > /etc/codex/config.toml << 'EOF'
model_provider = "nvidia_gateway"
[model_providers.nvidia_gateway]
name = "NVIDIA LiteLLM Gateway"
base_url = "{base_url}"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
EOF
RUN mkdir -p /app
"""


def _copy_parent_dockerfile(env_dir: str, t2_env_dir: str) -> None:
    """Copy the T2 Dockerfile plus any sidecar files (referenced by COPY/ADD) and
    append agent runtime installs + NVIDIA gateway config to the Dockerfile.

    Some T2 tasks ship sidecar files next to the Dockerfile (e.g. conftest
    overlays, entrypoint scripts, fixture generators). These are referenced via
    COPY/ADD and the build fails if they're missing — so we mirror every
    non-Dockerfile file from the parent environment/ directory.
    """
    src = os.path.join(t2_env_dir, "Dockerfile")
    if not os.path.exists(src):
        raise FileNotFoundError(f"Tools-track Dockerfile not found: {src}")
    with open(src) as f:
        content = f.read()
    content += _agent_install_snippet()
    dst = os.path.join(env_dir, "Dockerfile")
    with open(dst, "w") as f:
        f.write(content)

    for name in os.listdir(t2_env_dir):
        if name == "Dockerfile":
            continue
        src_side = os.path.join(t2_env_dir, name)
        dst_side = os.path.join(env_dir, name)
        if os.path.isdir(src_side):
            shutil.copytree(src_side, dst_side, dirs_exist_ok=True)
        else:
            shutil.copy2(src_side, dst_side)


# ---------------------------------------------------------------------------
# Single task conversion
# ---------------------------------------------------------------------------


def convert_task(
    task: dict[str, Any],
    tasks_dir: str,
    output_dir: str,
    verifier_files: dict[str, bytes] | None = None,
) -> str:
    """Convert a single synthesized search task to a Harbor task directory."""
    tid = task["id"]
    parent_id = task.get("parent_t2_task", "")
    difficulty = task.get("tier", "hard")
    agent_timeout = TIER_TIMEOUTS.get(difficulty, 1200)

    task_dir = os.path.join(output_dir, tid)
    env_dir = os.path.join(task_dir, "environment")
    tests_dir = os.path.join(task_dir, "tests")
    sol_dir = os.path.join(task_dir, "solution")

    for d in (task_dir, env_dir, tests_dir, sol_dir):
        os.makedirs(d, exist_ok=True)

    # 1. Task metadata
    write_task_toml(task_dir, tid, difficulty, agent_timeout)

    # 2. Search instruction (not Tools-track implementation instruction)
    write_instruction(task_dir, task["instruction"])

    # 3. Dockerfile from parent task (full repo + deps)
    parent_env_dir = os.path.join(tasks_dir, parent_id, "environment")
    _copy_parent_dockerfile(env_dir, parent_env_dir)

    # 4. Search verifier (test.sh + test_runner.py)
    if verifier_files is None:
        verifier_files = load_verifier_files()
    write_search_verifier(tests_dir, verifier_files)

    # 5. Gold answer for scoring
    gold = task.get("gold_answer", {})
    write_gold_answer(tests_dir, gold)

    # 6. Oracle solution
    write_solve_sh(sol_dir, gold)

    # 7. Write provenance metadata
    provenance = {
        "parent_t2_task": parent_id,
        "approach": task.get("approach", ""),
        "repo": task.get("repo", ""),
        "generated_from": "craft-taskgen search pipeline",
    }
    with open(os.path.join(task_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    return task_dir


# ---------------------------------------------------------------------------
# Top-level conversion
# ---------------------------------------------------------------------------


def run_harbor_convert(
    output_dir: str,
    tasks_dir: str,
    harbor_dir: str,
) -> str:
    """Convert all search tasks from approach dirs to Harbor task directories.

    Args:
        output_dir: Directory containing approach-{a,b,c}/search_tasks.json.
        tasks_dir: Directory with input task directories (for Dockerfiles).
        harbor_dir: Output directory for Harbor search task directories.

    Returns:
        Path to the harbor output directory.
    """
    import sys

    os.makedirs(harbor_dir, exist_ok=True)

    all_tasks: list[dict] = []
    for approach in ["a", "b", "c"]:
        path = os.path.join(output_dir, f"approach-{approach}", "search_tasks.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            tasks = json.load(f)
        print(f"Loaded {len(tasks)} tasks from {path}")
        all_tasks.extend(tasks)

    verifier_files = load_verifier_files()
    task_ids: list[str] = []
    errors = 0

    for task in all_tasks:
        tid = task.get("id", "unknown")
        try:
            convert_task(task, tasks_dir, harbor_dir, verifier_files=verifier_files)
            task_ids.append(tid)
            approach = task.get("approach", "?")
            parent = task.get("parent_t2_task", "?")
            print(f"  [{len(task_ids)}] {tid} (approach={approach}, parent={parent})")
        except Exception as e:
            errors += 1
            print(f"  ERROR: {tid}: {e}", file=sys.stderr)

    registry_path = write_registry(harbor_dir, task_ids, suite_name="craft-search-from-t2")

    print(f"\nConverted {len(task_ids)} tasks ({errors} errors)")
    print(f"Output: {harbor_dir}")
    print(f"Registry: {registry_path}")

    return harbor_dir
