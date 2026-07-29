# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI/env resolution for intake options."""

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class IntakeCliConfig:
    """Resolved intake options shared by server and launcher CLI paths."""

    enabled: bool
    base_url: str | None = None
    workspace: str | None = None
    api_key: str | None = None
    app: str | None = None
    task: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    target_url: str | None = None

    @classmethod
    def from_server_args(
        cls,
        args: argparse.Namespace,
        *,
        env: Mapping[str, str] | None = None,
    ) -> "IntakeCliConfig":
        resolved_env = os.environ if env is None else env
        base_url, workspace, api_key = _resolve_sink_connection(args, resolved_env)
        return cls(
            enabled=bool(getattr(args, "intake_enabled", False))
            or _env_bool("SWITCHYARD_INTAKE_ENABLED", resolved_env),
            base_url=base_url,
            workspace=workspace,
            api_key=api_key,
            target_url=_arg_or_env(
                args, "intake_target_url", resolved_env, "SWITCHYARD_INTAKE_TARGET_URL",
            ),
        )

    @classmethod
    def from_launch_args(
        cls,
        args: argparse.Namespace,
        *,
        env: Mapping[str, str] | None = None,
    ) -> "IntakeCliConfig":
        resolved_env = os.environ if env is None else env
        base_url, workspace, api_key = _resolve_sink_connection(args, resolved_env)
        return cls(
            enabled=bool(getattr(args, "intake_enabled", False))
            or _env_bool("SWITCHYARD_INTAKE_ENABLED", resolved_env),
            base_url=base_url,
            workspace=workspace,
            api_key=api_key,
            app=_arg_or_env(args, "intake_app", resolved_env, "SWITCHYARD_INTAKE_APP"),
            task=_arg_or_env(args, "intake_task", resolved_env, "SWITCHYARD_INTAKE_TASK"),
            session_id=_arg_or_env(
                args, "intake_session_id", resolved_env, "SWITCHYARD_SESSION_ID",
            ),
            user_id=_arg_or_env(
                args, "intake_user_id", resolved_env, "SWITCHYARD_USER_ID",
            ),
            target_url=_arg_or_env(
                args, "intake_target_url", resolved_env, "SWITCHYARD_INTAKE_TARGET_URL",
            ),
        )


def _resolve_sink_connection(
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> tuple[str | None, str | None, str | None]:
    return (
        _arg_or_env(args, "intake_base_url", env, "SWITCHYARD_INTAKE_BASE_URL"),
        _arg_or_env(args, "intake_workspace", env, "SWITCHYARD_INTAKE_WORKSPACE"),
        _arg_or_env(
            args,
            "intake_api_key",
            env,
            "SWITCHYARD_INTAKE_API_KEY",
            "NMP_ACCESS_TOKEN",
        ),
    )


def _arg_or_env(
    args: argparse.Namespace,
    attr: str,
    env: Mapping[str, str],
    *env_names: str,
) -> str | None:
    arg_value = getattr(args, attr, None)
    if arg_value:
        return str(arg_value)
    for env_name in env_names:
        env_value = env.get(env_name)
        if env_value:
            return env_value
    return None


def _env_bool(name: str, env: Mapping[str, str]) -> bool:
    raw = env.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["IntakeCliConfig"]
