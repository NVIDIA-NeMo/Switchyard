# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Claude Code through an in-process native Switchyard server."""

import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from switchyard.cli.launchers.launch_intake_config import (
    LaunchIntakeConfig,
    print_intake_warning,
)
from switchyard.cli.launchers.launcher_runtime import (
    banner_pause,
    configure_debug_file_logging,
    deterministic_strategy_summary,
    find_free_port,
    passthrough_strategy_summary,
    print_ready_banner,
    print_startup_failure,
    silence_launch_loggers,
    stdin_is_tty,
    wait_for_proxy_ready,
)
from switchyard.cli.launchers.live_stats_footer import LiveStatsFooter
from switchyard.cli.launchers.native_server import (
    NativeDeployment,
    NativeServer,
    deterministic_deployment,
    passthrough_deployment,
)
from switchyard.cli.launchers.proxy_health_monitor import ProxyHealthMonitor
from switchyard.cli.launchers.session_summary import print_session_summary
from switchyard.cli.launchers.stats_source import StatsSource
from switchyard.lib import startup_timing
from switchyard.lib.profiles import (
    DeterministicRoutingConfig,
)
from switchyard.server.shell_tui import ShellTUI

logger = logging.getLogger(__name__)

_READY_TIMEOUT_S = 10.0
_EXIT_BINARY_NOT_FOUND = 127
_EXIT_SIGINT = 130


def _quiet_launch_loggers() -> None:
    """Keep dependency chatter out of Claude Code's terminal UI."""
    silence_launch_loggers(local_logger=logger)


_find_free_port = find_free_port


def _find_claude_binary() -> str | None:
    """Locate the ``claude`` executable.

    Checks ``$PATH`` first, then falls back to the two paths Claude
    Code's installer writes to (``~/.claude/local/claude`` for the
    official installer, ``~/.local/bin/claude`` for alternative layouts).
    """
    path_hit = shutil.which("claude")
    if path_hit:
        return path_hit
    for candidate in (
        Path.home() / ".claude" / "local" / "claude",
        Path.home() / ".local" / "bin" / "claude",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _wait_ready(port: int, timeout_s: float = _READY_TIMEOUT_S) -> bool:
    """Probe ``GET /health`` until HTTP 200 or timeout."""
    return wait_for_proxy_ready(port, timeout_s=timeout_s)


def _format_anthropic_custom_headers(headers: dict[str, str]) -> str:
    """Encode header dict as Claude Code's ``ANTHROPIC_CUSTOM_HEADERS`` value."""
    return "\n".join(f"{name}: {value}" for name, value in headers.items())


def _claude_env(
    port: int,
    model: str,
    intake: LaunchIntakeConfig | None = None,
) -> dict[str, str]:
    """Build the env-var overrides that route Claude Code through our proxy.

    * ``ANTHROPIC_BASE_URL`` — our proxy URL.
    * ``ANTHROPIC_AUTH_TOKEN`` — opaque token; skips Console OAuth.
    * ``ANTHROPIC_API_KEY=""`` — silences the auth-conflict warning.
    * ``ANTHROPIC_MODEL`` / ``ANTHROPIC_SMALL_FAST_MODEL`` — initial
      active model for the session.
    * ``ANTHROPIC_CUSTOM_MODEL_OPTION`` — registers ``model`` as a custom
      slot in ``/model`` so the user can come back to it after toggling
      to a builtin.
    * ``CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`` — tells Claude Code
      to populate the picker from ``GET /v1/models``.
    """
    env = {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_AUTH_TOKEN": "switchyard",
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "ANTHROPIC_CUSTOM_MODEL_OPTION": model,
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
    }
    if intake is not None:
        env["ANTHROPIC_CUSTOM_HEADERS"] = _format_anthropic_custom_headers(
            intake.opt_in_headers(),
        )
        env["SWITCHYARD_SESSION_ID"] = intake.session_id
    return env


def _supervise_claude_plain(
    claude_bin: str,
    claude_args: list[str],
    port: int,
    model: str,
    intake: LaunchIntakeConfig | None = None,
) -> int:
    """Run ``claude`` via plain subprocess (non-TTY / headless fallback).

    ``subprocess.run`` inherits stdin/stdout/stderr so piped use works.
    ``KeyboardInterrupt`` is translated to exit code 130.
    """
    env = os.environ.copy()
    env.update(_claude_env(port, model, intake=intake))
    try:
        result = subprocess.run([claude_bin, *claude_args], env=env, check=False)
        return result.returncode
    except KeyboardInterrupt:
        return _EXIT_SIGINT


def _print_ready_banner(port: int, display_model: str) -> None:
    """Write the ready banner to stderr, bypassing the logger silencer above."""
    print_ready_banner(port=port, display_model=display_model)


def _make_footer_fn(
    stats: StatsSource,
    model: str,
    health: ProxyHealthMonitor,
) -> Callable[[int], list[tuple[str, int]]]:
    """Return the unified live-stats footer renderer."""
    return LiveStatsFooter(stats, model, health).as_footer_fn()


def _start_native_server(
    deployment: NativeDeployment,
    port: int | None,
) -> NativeServer:
    """Start the native server; kept separate for launcher supervision tests."""
    return NativeServer(deployment, port)


def _run_claude_with_switchyard(
    deployment: NativeDeployment,
    display_model: str,
    port: int | None,
    claude_args: list[str],
    intake: LaunchIntakeConfig | None = None,
    strategy_summary: str | None = None,
) -> int:
    """Host a native deployment and run Claude Code against it."""
    claude_bin = _find_claude_binary()
    if claude_bin is None:
        logger.error(
            "claude binary not found. Install it with "
            "`curl -fsSL https://claude.ai/install.sh | bash`, "
            "or place it on your PATH."
        )
        return _EXIT_BINARY_NOT_FOUND

    log_path = configure_debug_file_logging(display_model=display_model)
    logger.debug("claude launcher module=%s", __file__)
    server = _start_native_server(deployment, port)
    resolved_port = server.port
    stats = server.stats
    startup_timing.mark("native server started")

    try:
        if not _wait_ready(resolved_port):
            print_startup_failure(
                port=resolved_port,
                timeout_s=_READY_TIMEOUT_S,
                log_path=log_path,
            )
            return 1

        startup_timing.mark("proxy health-ready")
        logger.info("proxy ready on port %d", resolved_port)
        if intake is not None:
            print_intake_warning()
        print_ready_banner(
            port=resolved_port,
            display_model=display_model,
            log_path=log_path,
            strategy_summary=strategy_summary,
            profile_routes=list(deployment.models),
            default_route=display_model,
        )
        if stdin_is_tty():
            banner_pause()

        health = ProxyHealthMonitor(resolved_port)
        env_overrides = _claude_env(
            resolved_port,
            display_model,
            intake=intake,
        )
        logger.debug(
            "claude env ANTHROPIC_BASE_URL=%s ANTHROPIC_MODEL=%s "
            "ANTHROPIC_CUSTOM_MODEL_OPTION=%s GATEWAY_DISCOVERY=%s",
            env_overrides.get("ANTHROPIC_BASE_URL"),
            env_overrides.get("ANTHROPIC_MODEL"),
            env_overrides.get("ANTHROPIC_CUSTOM_MODEL_OPTION"),
            env_overrides.get("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"),
        )
        startup_timing.mark("child agent spawned")
        startup_timing.dump()
        if stdin_is_tty():
            footer = LiveStatsFooter(
                stats, display_model, health,
                strategy_label=strategy_summary.split(":")[0].strip() if strategy_summary else None,
            )
            tui = ShellTUI(
                command=[claude_bin, *claude_args],
                footer_fn=footer.as_footer_fn(),
                footer_height=lambda: footer.height,
                env=env_overrides,
            )
            return tui.run()
        return _supervise_claude_plain(
            claude_bin, claude_args, resolved_port, display_model,
            intake=intake,
        )
    finally:
        print_session_summary(stats)
        server.close()


def launch_claude(
    model: str,
    base_url: str,
    api_key: str,
    port: int | None,
    timeout: float | None,
    claude_args: list[str],
    intake: LaunchIntakeConfig | None = None,
    rl_log_dir: Path | None = None,
) -> int:
    """Start a native passthrough deployment and run Claude Code."""
    _quiet_launch_loggers()
    deployment = passthrough_deployment(
        model=model,
        api_key=api_key,
        base_url=base_url,
        claude_alias=True,
    )
    return _run_claude_with_switchyard(
        deployment,
        display_model=model,
        port=port,
        claude_args=claude_args,
        intake=intake,
        strategy_summary=passthrough_strategy_summary(model),
    )




def launch_claude_deterministic_routing(
    config: DeterministicRoutingConfig,
    port: int | None,
    claude_args: list[str],
    intake: LaunchIntakeConfig | None = None,
    discovery_disabled: bool = False,
    rl_log_dir: Path | None = None,
) -> int:
    """Run Claude Code with a native LLM-classifier deployment.

    Strong and weak targets remain directly selectable; the classifier target
    is internal to the routing algorithm.
    """
    from switchyard.cli.model_catalog.model_discovery import fetch_model_ids
    from switchyard.lib.route_table_builders import deterministic_routing_virtual_model_id

    def _discovery_fn(base_url: str, api_key: str) -> list[str]:
        return fetch_model_ids(base_url, api_key)

    _quiet_launch_loggers()
    routing_model = deterministic_routing_virtual_model_id(config)
    discovered_models = (
        []
        if discovery_disabled
        else _discovery_fn(config.strong.base_url or "", config.strong.api_key or "")
    )
    deployment = deterministic_deployment(
        config,
        additional_models=discovered_models,
        claude_aliases=True,
    )
    return _run_claude_with_switchyard(
        deployment,
        display_model=routing_model,
        port=port,
        claude_args=claude_args,
        intake=intake,
        strategy_summary=deterministic_strategy_summary(config),
    )
