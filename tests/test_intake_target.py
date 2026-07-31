# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Intake target URL resolution on ``switchyard serve``."""

import argparse

from switchyard.cli.intake_cli_config import IntakeCliConfig


def test_server_target_flag_wins_over_environment() -> None:
    args = argparse.Namespace(
        intake_enabled=True,
        intake_target_url="https://from-flag.example/posting",
    )
    resolved = IntakeCliConfig.from_server_args(
        args,
        env={"SWITCHYARD_INTAKE_TARGET_URL": "https://from-env.example/posting"},
    )
    assert resolved.target_url == "https://from-flag.example/posting"


def test_server_target_resolves_from_environment() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_target_url=None)
    resolved = IntakeCliConfig.from_server_args(
        args,
        env={"SWITCHYARD_INTAKE_TARGET_URL": "https://from-env.example/posting"},
    )
    assert resolved.target_url == "https://from-env.example/posting"


def test_server_environment_can_enable_intake() -> None:
    args = argparse.Namespace(intake_enabled=False, intake_target_url=None)
    resolved = IntakeCliConfig.from_server_args(
        args,
        env={"SWITCHYARD_INTAKE_ENABLED": "1"},
    )
    assert resolved.enabled is True
