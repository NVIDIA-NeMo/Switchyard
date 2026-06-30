# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the top-level ``switchyard --version`` flag."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from unittest.mock import patch

import pytest

from switchyard.cli.switchyard_cli import _build_parser, _switchyard_version


def test_version_flag_prints_version_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"switchyard {_switchyard_version()}"


def test_switchyard_version_matches_installed_metadata() -> None:
    assert _switchyard_version() == version("nemo-switchyard")


def test_switchyard_version_falls_back_to_dunder_when_uninstalled() -> None:
    from switchyard import __version__

    with patch(
        "switchyard.cli.switchyard_cli.version",
        side_effect=PackageNotFoundError("nemo-switchyard"),
    ):
        assert _switchyard_version() == __version__
