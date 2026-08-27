# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift checks for the Getting Started guide."""

from pathlib import Path


def test_getting_started_documents_current_paths() -> None:
    guide = (
        Path(__file__).resolve().parents[2] / "docs" / "getting_started.md"
    ).read_text()

    assert guide.index("## Server Path") < guide.index("## Library Path")
    assert "switchyard launch" not in guide
    assert "nemo-switchyard[cli]" not in guide
    assert "cargo install --locked switchyard-server" in guide
    assert "switchyard-server --config routes.toml --dry-run" in guide

    for legacy_command in (
        "switchyard configure",
        "switchyard serve",
        "switchyard verify",
        "switchyard launch",
        "--routing-profiles",
        "routes.yaml",
    ):
        assert legacy_command not in guide


# TODO: Add Python snippet coverage when the supported Rust-backed API is finalized.
# TODO: Add TOML schema coverage if the guide grows beyond one maintained example.
