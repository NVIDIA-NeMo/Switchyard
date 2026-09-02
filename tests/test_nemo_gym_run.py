# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

RUNNER = Path(__file__).parents[1] / "benchmark" / "nemo_gym" / "run.sh"


def test_help_documents_custom_switchyard_config() -> None:
    result = subprocess.run(
        ["bash", str(RUNNER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SWITCHYARD_CONFIG" in result.stdout
    assert "NVIDIA_API_KEY" not in result.stdout


def test_accepts_relative_custom_config_without_nvidia_key(tmp_path: Path) -> None:
    (tmp_path / "routes.toml").write_text("schema_version = 1\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=tmp_path,
        env={
            "PATH": os.environ["PATH"],
            "GYM_DIR": str(tmp_path / "gym"),
            "SWITCHYARD_CONFIG": "routes.toml",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "run 'uv sync --frozen --no-dev'" in result.stderr
    assert "NVIDIA_API_KEY" not in result.stderr
