# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server CLI reference drift tests."""

import subprocess
from pathlib import Path

CLI_REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "cli_reference.md"


def test_reference_lists_server_contract() -> None:
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "-p",
            "switchyard-server",
            "--",
            "--help",
        ],
        cwd=CLI_REFERENCE.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    text = CLI_REFERENCE.read_text()
    for flag in (
        "--config",
        "--host",
        "--port",
        "--backlog",
        "--dry-run",
        "--tls-cert",
        "--tls-key",
        "--help",
        "--version",
    ):
        assert flag in result.stdout
        assert flag in text
