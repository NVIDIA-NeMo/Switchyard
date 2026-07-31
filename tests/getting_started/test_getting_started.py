# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift checks for the concise getting-started guide."""

from pathlib import Path


def test_getting_started_uses_the_current_cli() -> None:
    guide = (
        Path(__file__).resolve().parents[2] / "docs" / "getting_started.md"
    ).read_text()
    assert "switchyard launch claude --model switchyard" in guide
    assert "switchyard launch codex --model switchyard" in guide
    assert "switchyard serve" not in guide
    assert "switchyard configure" not in guide
    assert "switchyard verify" not in guide
