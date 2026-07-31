#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Switchyard command-line entry point."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from typing import Any

from switchyard.cli.command_utils import (
    quiet_dependency_loggers as _quiet_dependency_loggers,
)
from switchyard.cli.intake_cli_config import IntakeCliConfig
from switchyard.cli.launch_command import (
    cmd_launch_claude,
    cmd_launch_codex,
    cmd_launch_openclaw,
)
from switchyard.cli.route_bundle import RouteBundleConfigError, load_route_bundle_table
from switchyard.lib.config import IntakeSinkConfig
from switchyard.lib.processors.intake_request_processor import IntakeRequestProcessor
from switchyard.lib.processors.intake_response_processor import IntakeResponseProcessor
from switchyard.lib.processors.rl_logging_response_processor import build_rl_logging_processors
from switchyard.server.server_util import (
    add_transport_args,
    build_and_serve,
    resolve_rl_log_dir,
)

logger = logging.getLogger(__name__)

_CANONICAL_INTAKE_ENABLE_FLAG = "--intake-enabled"
_DEPRECATED_INTAKE_ENABLE_FLAG = "--enable-intake"
_ARGPARSE_ACTION_SUPPORTS_DEPRECATED = (
    "deprecated" in signature(argparse.Action.__init__).parameters
)


class _IntakeEnabledAction(argparse.Action):
    """Normalize the Intake flag and warn on its deprecated alias."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        nargs: int | str | None = None,
        const: Any = None,
        default: Any = None,
        type: Any = None,
        choices: Any = None,
        required: bool = False,
        help: str | None = None,
        metavar: str | tuple[str, ...] | None = None,
        deprecated: bool = False,
    ) -> None:
        del nargs
        action_kwargs: dict[str, Any] = {
            "option_strings": option_strings,
            "dest": dest,
            "nargs": 0,
            "const": const,
            "default": default,
            "type": type,
            "choices": choices,
            "required": required,
            "help": help,
            "metavar": metavar,
        }
        if _ARGPARSE_ACTION_SUPPORTS_DEPRECATED:
            action_kwargs["deprecated"] = deprecated
        super().__init__(**action_kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        if option_string == _DEPRECATED_INTAKE_ENABLE_FLAG:
            logger.warning(
                "%s is deprecated; use %s",
                _DEPRECATED_INTAKE_ENABLE_FLAG,
                _CANONICAL_INTAKE_ENABLE_FLAG,
            )
        setattr(namespace, self.dest, True)


def _add_intake_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        _CANONICAL_INTAKE_ENABLE_FLAG,
        _DEPRECATED_INTAKE_ENABLE_FLAG,
        dest="intake_enabled",
        default=False,
        action=_IntakeEnabledAction,
        help=(
            "Enable Intake for requests that opt in with store=true or "
            "x-switchyard-intake-enabled=true."
        ),
    )
    parser.add_argument("--intake-base-url", default=None)
    parser.add_argument("--intake-workspace", default=None)
    parser.add_argument("--intake-api-key", default=None)
    parser.add_argument("--intake-target-url", default=None)


def _resolve_intake_config(args: argparse.Namespace) -> IntakeSinkConfig | None:
    intake = IntakeCliConfig.from_server_args(args)
    if not intake.enabled:
        return None
    return IntakeSinkConfig(
        intake_base_url=intake.base_url,
        workspace=intake.workspace,
        api_key=intake.api_key,
        target_url=intake.target_url,
    )


def _resolve_intake_processors(
    args: argparse.Namespace,
) -> tuple[list[Any], list[Any]]:
    intake = _resolve_intake_config(args)
    if intake is None:
        return [], []
    return [IntakeRequestProcessor()], [IntakeResponseProcessor(intake)]


def _cmd_serve(args: argparse.Namespace) -> None:
    """Serve an explicit routing-profile bundle."""

    intake_request, intake_response = _resolve_intake_processors(args)
    rl_request, rl_response = build_rl_logging_processors(resolve_rl_log_dir(args))
    request_processors = [*intake_request, *rl_request]
    response_processors = [*intake_response, *rl_response]
    if args.routing_log_file:
        from switchyard.lib.processors.routing_log_response_processor import (
            RoutingLogResponseProcessor,
        )

        response_processors.append(RoutingLogResponseProcessor(args.routing_log_file))

    table = load_route_bundle_table(
        args.routing_profiles,
        pre_routing_request_processors=request_processors,
        extra_response_processors=response_processors,
    )
    logger.info(
        "Switchyard route bundle loaded %d route(s) from %s",
        len(table.registered_models()),
        args.routing_profiles,
    )
    strategy_summary: str | None = None
    default_model = table.default_model()
    if default_model:
        from switchyard.cli.launchers.launcher_runtime import (
            routing_profiles_strategy_summary,
        )

        strategy_summary = routing_profiles_strategy_summary(
            args.routing_profiles,
            default_model,
        )
    build_and_serve(
        args,
        table,
        inbound_default="both",
        strategy_summary=strategy_summary,
    )


def _switchyard_version() -> str:
    """Resolve the installed distribution version."""

    try:
        return version("nemo-switchyard")
    except PackageNotFoundError:
        from switchyard import __version__

        return __version__


def _add_launch_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
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
        version=f"%(prog)s {_switchyard_version()}",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Serve a routing-profile bundle")
    serve.add_argument(
        "--routing-profiles",
        "-c",
        required=True,
        metavar="PATH",
        help="Routing-profile YAML bundle.",
    )
    serve.add_argument("--enable-rl-logging", action="store_true")
    serve.add_argument("--rl-log-dir", default="./rl_data", metavar="DIR")
    add_transport_args(serve)
    _add_intake_args(serve)
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
