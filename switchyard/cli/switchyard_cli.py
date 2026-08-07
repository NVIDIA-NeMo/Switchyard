#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Switchyard command-line entry point."""

import argparse
import logging
import os

from switchyard import __version__
from switchyard.cli.command_utils import (
    quiet_dependency_loggers as _quiet_dependency_loggers,
)
from switchyard.cli.launch_command import (
    cmd_launch_claude,
    cmd_launch_codex,
    cmd_launch_openclaw,
)
from switchyard.cli.route_bundle import RouteBundleConfigError, load_route_bundle_table
from switchyard.lib.processors.rl_logging_response_processor import build_rl_logging_processors
from switchyard.server.server_util import (
    add_transport_args,
    build_and_serve,
    resolve_rl_log_dir,
)

logger = logging.getLogger(__name__)

def _cmd_serve(args: argparse.Namespace) -> None:
    """Serve an explicit Python route bundle."""

    request_processors, response_processors = build_rl_logging_processors(
        resolve_rl_log_dir(args)
    )
    if args.routing_log_file:
        from switchyard.lib.processors.routing_log_response_processor import (
            RoutingLogResponseProcessor,
        )

        response_processors.append(RoutingLogResponseProcessor(args.routing_log_file))

    table = load_route_bundle_table(
        args.routes,
        pre_routing_request_processors=request_processors,
        extra_response_processors=response_processors,
    )
    logger.info(
        "Switchyard route bundle loaded %d route(s) from %s",
        len(table.registered_models()),
        args.routes,
    )
    strategy_summary: str | None = None
    default_model = table.default_model()
    if default_model:
        from switchyard.cli.launchers.launcher_runtime import (
            route_bundle_strategy_summary,
        )

        strategy_summary = route_bundle_strategy_summary(
            args.routes,
            default_model,
        )
    build_and_serve(
        args,
        table,
        inbound_default="both",
        strategy_summary=strategy_summary,
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

    serve = subparsers.add_parser("serve", help="Serve a Python route bundle")
    serve.add_argument(
        "--routes",
        "-c",
        required=True,
        metavar="PATH",
        help="YAML bundle containing noop and passthrough routes.",
    )
    serve.add_argument("--enable-rl-logging", action="store_true")
    serve.add_argument("--rl-log-dir", default="./rl_data", metavar="DIR")
    add_transport_args(serve)
    serve.add_argument("--routing-log-file", default=None, metavar="PATH")
    serve.add_argument(
        "--workers",
        "-w",
        type=int,
        default=int(os.environ.get("SWITCHYARD_WORKERS", "1")),
    )
    serve.set_defaults(func=_cmd_serve)

    _add_launch_parser(subparsers)
    return parser


def main() -> None:
    """Run the Switchyard CLI."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _quiet_dependency_loggers()
    parser = _build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(1)
    try:
        args.func(args)
    except RouteBundleConfigError as exc:
        raise SystemExit(f"error: invalid route bundle: {exc}") from exc


if __name__ == "__main__":
    main()
