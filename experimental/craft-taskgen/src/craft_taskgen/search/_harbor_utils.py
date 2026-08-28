# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for writing search-dimension Harbor task directories.

Used by both the from-T2 converter (`search/harbor.py`) and the native converter
(`adapters/search_native/converter.py`). Keeps layout, template loading, and
file writing consistent across adapters.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

TIER_TIMEOUTS: dict[str, int] = {
    "easy": 600,
    "medium": 900,
    "hard": 1200,
}


def load_instruction_template() -> str:
    """Load the shared instruction template."""
    with open(TEMPLATES_DIR / "instruction.md.template") as f:
        return f.read()


def load_verifier_files() -> dict[str, bytes]:
    """Read the shared search verifier files once for reuse across tasks."""
    result: dict[str, bytes] = {}
    for filename in ("test.sh", "test_runner.py"):
        with open(TEMPLATES_DIR / filename, "rb") as f:
            result[filename] = f.read()
    return result


def write_task_toml(task_dir: str, tid: str, difficulty: str, agent_timeout: int) -> None:
    """Write task.toml with search-appropriate timeouts and verifier env."""
    verifier_timeout = 120  # Search verifier is fast (LLM judge call at most)
    content = textwrap.dedent(f"""\
        version = "1.0"

        [metadata]
        name = "{tid}"
        difficulty = "{difficulty}"

        [verifier]
        timeout_sec = {verifier_timeout}

        [agent]
        timeout_sec = {agent_timeout}

        [environment]
        build_timeout_sec = 600.0
        cpus = 2
        memory_mb = 4096
        storage_mb = 10240
        gpus = 0
        allow_internet = true
        mcp_servers = []

        [environment.env]
        ANTHROPIC_API_KEY = "${{ANTHROPIC_API_KEY}}"
        ANTHROPIC_BASE_URL = "${{ANTHROPIC_BASE_URL}}"
        OPENAI_API_KEY = "${{OPENAI_API_KEY}}"
        OPENAI_BASE_URL = "${{OPENAI_BASE_URL}}"

        [verifier.env]
        OPENAI_API_KEY = "${{OPENAI_API_KEY}}"
        OPENAI_BASE_URL = "${{OPENAI_BASE_URL}}"
        # Pin the LLM judge to the NVIDIA gateway independent of the agent's
        # endpoint. The launcher captures the gateway URL+key from .env into
        # JUDGE_* before any vllm-mode override rewrites OPENAI_*. If unset,
        # test_runner.py falls back to OPENAI_BASE_URL (fine in gateway mode).
        JUDGE_API_KEY = "${{JUDGE_API_KEY}}"
        JUDGE_BASE_URL = "${{JUDGE_BASE_URL}}"
        JUDGE_MODEL = "aws/anthropic/bedrock-claude-sonnet-4-6"

        [solution.env]
    """)
    with open(os.path.join(task_dir, "task.toml"), "w") as f:
        f.write(content)


def write_instruction(task_dir: str, instruction: str, template: str | None = None) -> None:
    """Write instruction.md for the search task."""
    if template is None:
        template = load_instruction_template()
    content = template.replace("{instruction}", instruction)
    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write(content)


def write_solve_sh(sol_dir: str, gold_answer: dict[str, Any]) -> None:
    """Write solve.sh that outputs the gold answer to /app/answer.json."""
    answer_obj = {
        "files": gold_answer.get("files", []),
        "functions": gold_answer.get("functions", []),
        "explanation": gold_answer.get("explanation", ""),
    }
    answer_json = json.dumps(answer_obj, indent=2)
    lines = [
        "#!/usr/bin/env bash",
        "# Oracle solution -- writes full gold answer.",
        "mkdir -p /app",
        "cat > /app/answer.json << 'ORACLE_EOF'",
        answer_json,
        "ORACLE_EOF",
        "",
    ]
    path = os.path.join(sol_dir, "solve.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)


def write_search_verifier(tests_dir: str, verifier_files: dict[str, bytes]) -> None:
    """Write the shared search verifier files (test.sh + test_runner.py)."""
    for filename, content in verifier_files.items():
        with open(os.path.join(tests_dir, filename), "wb") as f:
            f.write(content)


def write_gold_answer(tests_dir: str, gold_answer: dict[str, Any]) -> None:
    """Write gold_answer.json for the search verifier."""
    with open(os.path.join(tests_dir, "gold_answer.json"), "w") as f:
        json.dump(gold_answer, f, indent=2)


def task_id(repo: str, uuid: str) -> str:
    """Canonical Harbor task ID: craft-{repo}-{first 8 chars of uuid}."""
    return f"craft-{repo}-{uuid[:8]}"


def write_registry(output_dir: str, task_ids: list[str], suite_name: str) -> str:
    """Write registry.json for a benchmark suite. Returns the registry path."""
    registry = [
        {
            "name": suite_name,
            "version": "1.0",
            "metrics": [{"type": "mean"}],
            "tasks": task_ids,
        }
    ]
    path = os.path.join(output_dir, "registry.json")
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
    return path
