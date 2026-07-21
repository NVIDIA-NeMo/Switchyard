# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Intake enable flag normalization across CLI surfaces."""

from __future__ import annotations

import argparse
import logging

import pytest

from switchyard.cli.switchyard_cli import _build_parser


def test_serve_accepts_canonical_intake_flag() -> None:
    args = _build_parser().parse_args(["serve", "--intake-enabled"])

    assert args.intake_enabled is True
    assert not hasattr(args, "enable_intake")


def test_serve_accepts_deprecated_intake_alias(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, "switchyard.cli.switchyard_cli")

    args = _build_parser().parse_args(["serve", "--enable-intake"])

    assert args.intake_enabled is True
    assert not hasattr(args, "enable_intake")
    assert "--enable-intake is deprecated; use --intake-enabled" in caplog.text


@pytest.mark.parametrize("target", ["claude", "codex", "openclaw"])
def test_launchers_accept_canonical_intake_flag(target: str) -> None:
    args = _build_parser().parse_args([
        "launch",
        target,
        "--model",
        "nvidia/moonshotai/kimi-k2.5",
        "--intake-enabled",
    ])

    assert args.intake_enabled is True
    assert not hasattr(args, "enable_intake")


@pytest.mark.parametrize("target", ["claude", "codex", "openclaw"])
def test_launchers_accept_deprecated_intake_alias(
    target: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, "switchyard.cli.switchyard_cli")

    args = _build_parser().parse_args([
        "launch",
        target,
        "--model",
        "nvidia/moonshotai/kimi-k2.5",
        "--enable-intake",
    ])

    assert args.intake_enabled is True
    assert not hasattr(args, "enable_intake")
    assert "--enable-intake is deprecated; use --intake-enabled" in caplog.text


COMMON_INTAKE_ARGS = [
    "--intake-base-url", "https://nmp.example",
    "--intake-workspace", "workspace-a",
    "--intake-api-key", "ci-token",
    "--intake-target-url", "https://data-lake.example/posting",
]


def _assert_common_intake_args(args: argparse.Namespace) -> None:
    assert args.intake_base_url == "https://nmp.example"
    assert args.intake_workspace == "workspace-a"
    assert args.intake_api_key == "ci-token"
    assert args.intake_target_url == "https://data-lake.example/posting"


def test_serve_accepts_common_intake_args() -> None:
    args = _build_parser().parse_args(["serve", *COMMON_INTAKE_ARGS])

    _assert_common_intake_args(args)


@pytest.mark.parametrize("target", ["claude", "codex", "openclaw"])
def test_launchers_accept_common_intake_args(target: str) -> None:
    args = _build_parser().parse_args([
        "launch",
        target,
        "--model",
        "nvidia/moonshotai/kimi-k2.5",
        *COMMON_INTAKE_ARGS,
    ])

    _assert_common_intake_args(args)
