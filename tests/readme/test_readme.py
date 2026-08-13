# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift checks for the repository README."""

import asyncio
import sys
import time
from pathlib import Path

import pytest

from ..onboarding_smoke import _read_listen_url, exercise_documented_server_flow


def test_readme_documents_current_paths() -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()

    assert readme.index("### Launcher Path") < readme.index("### Server Path")
    assert 'uv tool install --python 3.10 "nemo-switchyard[cli]"' in readme
    assert "switchyard launch claude --model switchyard" in readme
    assert "cargo install --locked switchyard-server" in readme
    assert "switchyard-server --config routes.toml --dry-run" in readme

    for legacy_command in (
        "switchyard configure",
        "switchyard serve",
        "switchyard verify",
    ):
        assert legacy_command not in readme


async def test_readme_server_path_is_executable(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    readme = (repository / "README.md").read_text()
    assert "[Getting Started](docs/getting_started.md)" in readme

    await exercise_documented_server_flow(
        repository / "docs" / "getting_started.md",
        tmp_path,
    )


async def test_startup_port_discovery_times_out_with_partial_output() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        (
            "import sys, time; "
            "sys.stdout.write('partial stdout'); sys.stdout.flush(); "
            "sys.stderr.write('partial stderr'); sys.stderr.flush(); "
            "time.sleep(10)"
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    started = time.monotonic()
    try:
        with pytest.raises(AssertionError) as error:
            await _read_listen_url(process, timeout=1)
        assert time.monotonic() - started < 3
        assert "partial stdout" in str(error.value)
        assert "partial stderr" in str(error.value)
    finally:
        process.kill()
        await process.communicate()


async def test_startup_port_discovery_accepts_url_without_newline() -> None:
    expected = "http://127.0.0.1:43123"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        (
            "import sys, time; "
            f"sys.stdout.write('listening: {expected}'); sys.stdout.flush(); "
            "time.sleep(10)"
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert await _read_listen_url(process, timeout=1) == expected
    finally:
        process.kill()
        await process.communicate()
