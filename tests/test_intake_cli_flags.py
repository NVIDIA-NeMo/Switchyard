# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Intake flags on ``switchyard serve``."""

import logging

import pytest

from switchyard.cli.switchyard_cli import _build_parser

_SERVE = ["serve", "--routing-profiles", "routes.yaml"]


def test_serve_accepts_canonical_intake_flag() -> None:
    args = _build_parser().parse_args([*_SERVE, "--intake-enabled"])
    assert args.intake_enabled is True


def test_serve_accepts_deprecated_intake_alias(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, "switchyard.cli.switchyard_cli")
    args = _build_parser().parse_args([*_SERVE, "--enable-intake"])
    assert args.intake_enabled is True
    assert "--enable-intake is deprecated; use --intake-enabled" in caplog.text


def test_serve_accepts_intake_connection_options() -> None:
    args = _build_parser().parse_args(
        [
            *_SERVE,
            "--intake-base-url",
            "https://nmp.example",
            "--intake-workspace",
            "workspace-a",
            "--intake-api-key",
            "ci-token",
            "--intake-target-url",
            "https://data-lake.example/posting",
        ]
    )
    assert args.intake_base_url == "https://nmp.example"
    assert args.intake_workspace == "workspace-a"
    assert args.intake_api_key == "ci-token"
    assert args.intake_target_url == "https://data-lake.example/posting"
