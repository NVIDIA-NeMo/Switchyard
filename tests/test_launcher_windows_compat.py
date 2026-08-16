# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-portability tests for the shared launcher runtime.

The launchers must degrade gracefully on Windows: no POSIX-only imports at
module load time, batch-shim subprocess routing, permission-bit-free
executable detection, and a platform-appropriate state directory. These tests
flip the module's ``_IS_WINDOWS``/``_IS_POSIX`` constants directly so the real
``os`` module (and pathlib's Windows/PosixPath selection) is never mutated.
"""

import os
from pathlib import Path

import pytest

from switchyard.cli.launchers import launcher_runtime


def test_is_windows_batch_shim_only_on_nt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher_runtime, "_IS_WINDOWS", True)
    assert launcher_runtime.is_windows_batch_shim(r"C:\npm\opencode.cmd") is True
    assert launcher_runtime.is_windows_batch_shim(r"C:\npm\opencode.bat") is True
    assert launcher_runtime.is_windows_batch_shim(r"C:\npm\opencode.exe") is False
    assert launcher_runtime.is_windows_batch_shim(r"C:\npm\opencode") is False


def test_is_windows_batch_shim_never_true_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher_runtime, "_IS_WINDOWS", False)
    assert launcher_runtime.is_windows_batch_shim("opencode.cmd") is False


def test_is_executable_file_ignores_permission_bit_on_nt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    candidate = tmp_path / "opencode"
    candidate.write_text("shim", encoding="utf-8")

    monkeypatch.setattr(launcher_runtime, "_IS_WINDOWS", True)
    assert launcher_runtime.is_executable_file(candidate) is True


def test_is_executable_file_requires_x_bit_on_posix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    candidate = tmp_path / "opencode"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(candidate, 0o644)

    monkeypatch.setattr(launcher_runtime, "_IS_WINDOWS", False)
    assert launcher_runtime.is_executable_file(candidate) is False

    os.chmod(candidate, 0o755)
    assert launcher_runtime.is_executable_file(candidate) is True


def test_is_executable_file_missing_file_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher_runtime, "_IS_WINDOWS", True)
    assert launcher_runtime.is_executable_file(Path("/nonexistent/opencode")) is False


def test_default_state_dir_uses_localappdata_on_nt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher_runtime, "_IS_WINDOWS", True)
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\dev\AppData\Local")

    assert launcher_runtime._default_state_dir() == Path(r"C:\Users\dev\AppData\Local")


def test_default_state_dir_is_home_dot_local_on_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher_runtime, "_IS_WINDOWS", False)
    assert launcher_runtime._default_state_dir() == Path.home() / ".local" / "state"


def test_stdin_is_tty_false_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher_runtime, "_IS_POSIX", False)
    assert launcher_runtime.stdin_is_tty() is False
