# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for intake target-URL CLI/env wiring into the intake sink config."""

import argparse

from switchyard.cli.intake_cli_config import IntakeCliConfig
from switchyard.cli.launchers.launch_intake_config import LaunchIntakeConfig

_URL = "https://data-lake.example/lake/example-project/posting"


def test_launch_args_resolve_target_from_flag() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_target_url=_URL)
    resolved = IntakeCliConfig.from_launch_args(args, env={})
    assert resolved.target_url == _URL


def test_launch_args_resolve_target_from_env() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_target_url=None)
    resolved = IntakeCliConfig.from_launch_args(
        args, env={"SWITCHYARD_INTAKE_TARGET_URL": "https://from-env.example/posting"}
    )
    assert resolved.target_url == "https://from-env.example/posting"


def test_flag_wins_over_env() -> None:
    args = argparse.Namespace(
        intake_enabled=True, intake_target_url="https://from-flag.example/posting"
    )
    resolved = IntakeCliConfig.from_launch_args(
        args, env={"SWITCHYARD_INTAKE_TARGET_URL": "https://from-env.example/posting"}
    )
    assert resolved.target_url == "https://from-flag.example/posting"


def test_server_args_resolve_target() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_target_url=_URL)
    resolved = IntakeCliConfig.from_server_args(args, env={})
    assert resolved.target_url == _URL


def test_target_absent_defaults_to_none() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_target_url=None)
    resolved = IntakeCliConfig.from_launch_args(args, env={})
    assert resolved.target_url is None


def test_to_sink_config_passes_target_through_binding() -> None:
    config = LaunchIntakeConfig(
        base_url=None,
        workspace=None,
        api_key=None,
        app="claude-code",
        task="developer-session",
        session_id="sess-1",
        user_id="0badf00d",
        target_url=_URL,
    )
    sink = config.to_sink_config()
    assert sink.target_url == _URL
    # A target URL alone defaults to the flat, unauthenticated data-lake shape.
    assert sink.target_format == "flat_document"
    assert sink.target_authenticated is False


def test_launch_env_var_enables_intake() -> None:
    # SWITCHYARD_INTAKE_ENABLED turns intake on for launch (parity with serve).
    args = argparse.Namespace(intake_enabled=False, intake_target_url=None)
    resolved = IntakeCliConfig.from_launch_args(args, env={"SWITCHYARD_INTAKE_ENABLED": "1"})
    assert resolved.enabled is True


def test_launch_disabled_without_flag_or_env() -> None:
    args = argparse.Namespace(intake_enabled=False, intake_target_url=None)
    resolved = IntakeCliConfig.from_launch_args(args, env={})
    assert resolved.enabled is False


def test_server_env_var_enables_intake() -> None:
    args = argparse.Namespace(intake_enabled=False, intake_target_url=None)
    resolved = IntakeCliConfig.from_server_args(args, env={"SWITCHYARD_INTAKE_ENABLED": "1"})
    assert resolved.enabled is True
