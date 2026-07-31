# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""README command drift checks."""

from pathlib import Path


def test_readme_uses_the_current_cli() -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    assert "switchyard launch claude --model switchyard" in readme
    assert "switchyard launch codex --model switchyard" in readme
    assert "switchyard serve" not in readme
    assert "switchyard configure" not in readme
    assert "switchyard verify" not in readme
