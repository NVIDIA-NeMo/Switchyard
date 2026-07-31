# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch coding agents against an explicit native Switchyard deployment."""

import argparse
from pathlib import Path

from switchyard.cli.command_utils import strip_forwarded_args

_DEFAULT_CONFIG = Path(__file__).with_name("defaults") / "openrouter.toml"


def _config_path(value: str | None) -> Path:
    """Resolve the selected deployment, defaulting to the packaged OpenRouter config."""
    path = Path(value).expanduser() if value else _DEFAULT_CONFIG
    if not path.is_file():
        raise SystemExit(f"launch: config file not found: {path}")
    return path


def cmd_launch_claude(args: argparse.Namespace) -> None:
    """Run Claude Code against a native TOML deployment."""
    from switchyard.cli.launchers.claude_code_launcher import launch_claude_config

    raise SystemExit(
        launch_claude_config(
            config=_config_path(args.config),
            model=args.model,
            claude_args=strip_forwarded_args(args.claude_args),
        )
    )


def cmd_launch_codex(args: argparse.Namespace) -> None:
    """Run Codex against a native TOML deployment."""
    from switchyard.cli.launchers.codex_cli_launcher import launch_codex_config

    raise SystemExit(
        launch_codex_config(
            config=_config_path(args.config),
            model=args.model,
            codex_args=strip_forwarded_args(args.codex_args),
        )
    )


def cmd_launch_openclaw(args: argparse.Namespace) -> None:
    """Run OpenClaw against a native TOML deployment."""
    from switchyard.cli.launchers.openclaw_launcher import launch_openclaw_config

    raise SystemExit(
        launch_openclaw_config(
            config=_config_path(args.config),
            model=args.model,
            openclaw_args=strip_forwarded_args(args.openclaw_args),
        )
    )


__all__ = [
    "cmd_launch_claude",
    "cmd_launch_codex",
    "cmd_launch_openclaw",
]
