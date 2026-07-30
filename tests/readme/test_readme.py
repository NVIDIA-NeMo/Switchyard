# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable coverage for the README.md."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README_PATH.read_text()


def _code_blocks(text: str, lang: str) -> list[str]:
    # markdown-it-py handles indented fences + trailing whitespace correctly,
    # which a naive triple-backtick regex does not.
    md = MarkdownIt()
    return [
        token.content
        for token in md.parse(text)
        if token.type == "fence" and token.info.strip() == lang
    ]


def test_rust_server_toml_blocks_validate_with_rust_schema(
    readme_text: str,
    tmp_path: Path,
) -> None:
    blocks = _code_blocks(readme_text, "toml")
    server_configs = [
        block
        for block in blocks
        if "schema_version" in block
        and "[llm_clients." in block
        and "[targets." in block
        and "[routes." in block
    ]

    assert server_configs, "no README TOML block found for the Rust server"

    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = "readme-test-key"
    for index, config in enumerate(server_configs):
        config_path = tmp_path / f"readme-server-{index}.toml"
        config_path.write_text(config)
        # Dry-run uses the production Rust loader for schema and reference validation.
        result = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--locked",
                "-p",
                "switchyard-server",
                "--",
                "--config",
                str(config_path),
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"README TOML block {index} failed Rust schema validation:\n"
            f"{result.stderr}{result.stdout}"
        )

# TODO: Add Python snippet tripwire test back in when we have a Python snippet to test.
# TODO: Add route block validation test back in when we document YAML route bundles.
# TODO: Add routing docs schema test back in when the README links those guides.
# TODO: Add CLI subcommand test back in when the README documents Python CLI commands.
# TODO: Add CLI flag test back in when the README documents Python CLI flags.
