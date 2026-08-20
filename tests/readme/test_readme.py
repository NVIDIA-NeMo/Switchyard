# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift checks for the repository README."""

from pathlib import Path


def test_readme_documents_current_paths() -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()

    assert readme.index("### Server Path") < readme.index("### Library Path")
    assert "switchyard launch" not in readme
    assert "nemo-switchyard[cli]" not in readme
    assert "cargo install --locked switchyard-server" in readme
    assert "switchyard-server --config routes.toml --dry-run" in readme

    for legacy_command in (
        "switchyard configure",
        "switchyard serve",
        "switchyard verify",
        "switchyard launch",
    ):
        assert legacy_command not in readme


# TODO: Add Python snippet coverage when the supported Rust-backed API is finalized.
# TODO: Add TOML schema coverage if the README includes a complete server config.
