#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Switchyard command-line entry point."""

import argparse

from switchyard import __version__
from switchyard.cli.launch_command import (
    cmd_launch_claude,
    cmd_launch_codex,
    cmd_launch_openclaw,
)


def _add_launch_parser(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    launch = subparsers.add_parser(
        "launch",
        help="Launch a coding agent through the native server",
    )
    launch_sub = launch.add_subparsers(dest="launch_target", help="Agent to launch")
    launcher_parsers = (
        ("claude", "Launch Claude Code", "claude_args", cmd_launch_claude),
        ("codex", "Launch Codex CLI", "codex_args", cmd_launch_codex),
        ("openclaw", "Launch OpenClaw", "openclaw_args", cmd_launch_openclaw),
    )
    for name, help_text, args_dest, command in launcher_parsers:
        agent = launch_sub.add_parser(name, help=help_text)
        agent.add_argument("--model", required=True, help="Route ID from the deployment.")
        agent.add_argument(
            "--config",
            metavar="PATH",
            help="TOML deployment (default: packaged OpenRouter deployment).",
        )
        agent.add_argument(
            args_dest,
            nargs=argparse.REMAINDER,
            help="Arguments forwarded to the coding agent after --.",
        )
        agent.set_defaults(func=command)

    def _launch_help(args: argparse.Namespace) -> None:  # noqa: ARG001
        launch.print_help()
        raise SystemExit(1)

    launch.set_defaults(func=_launch_help)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="switchyard",
        description="Switchyard LLM proxy",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_launch_parser(subparsers)
    return parser


def main() -> None:
    """Run the Switchyard CLI."""

    parser = _build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(1)
    args.func(args)


if __name__ == "__main__":
    main()
