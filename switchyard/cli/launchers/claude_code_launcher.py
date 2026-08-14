# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Claude Code through an in-process native Switchyard server."""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from switchyard.cli.launchers.launcher_runtime import (
    banner_pause,
    configure_debug_file_logging,
    print_ready_banner,
    print_startup_failure,
    silence_launch_loggers,
    stdin_is_tty,
    wait_for_proxy_ready,
)
from switchyard.cli.launchers.live_stats_footer import LiveStatsFooter
from switchyard.cli.launchers.native_server import NativeServer
from switchyard.cli.launchers.proxy_health_monitor import ProxyHealthMonitor
from switchyard.cli.launchers.session_summary import print_session_summary
from switchyard.cli.launchers.shell_tui import ShellTUI

logger = logging.getLogger(__name__)

_READY_TIMEOUT_S = 10.0
_EXIT_BINARY_NOT_FOUND = 127
_EXIT_SIGINT = 130


def _quiet_launch_loggers() -> None:
    """Keep dependency chatter out of Claude Code's terminal UI."""
    silence_launch_loggers(local_logger=logger)


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


def _claude_env(
    port: int,
    model: str,
    use_anthropic_auth: bool = False,
) -> dict[str, str]:
    """Build the env-var overrides that route Claude Code through our proxy.

    * ``ANTHROPIC_BASE_URL`` — our proxy URL.
    * ``ANTHROPIC_AUTH_TOKEN`` — an opaque local token unless the deployment
      forwards the caller's Anthropic login.
    * ``ANTHROPIC_API_KEY=""`` — silences the auth-conflict warning.
    * ``ANTHROPIC_MODEL`` — initial active model for the session.
    * ``ANTHROPIC_SMALL_FAST_MODEL`` — existing background-model override,
      falling back to the active model.
    * ``ANTHROPIC_CUSTOM_MODEL_OPTION`` — registers ``model`` as a custom
      slot in ``/model`` so the user can come back to it after toggling
      to a builtin.
    * ``CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`` — tells Claude Code
      to populate the picker from ``GET /v1/models``.
    """
    env = {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": os.environ.get("ANTHROPIC_SMALL_FAST_MODEL", model),
        "ANTHROPIC_CUSTOM_MODEL_OPTION": model,
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
    }
    if not use_anthropic_auth:
        env["ANTHROPIC_AUTH_TOKEN"] = "switchyard"
    return env


def _supervise_claude_plain(
    claude_bin: str,
    claude_args: list[str],
    env_overrides: dict[str, str],
) -> int:
    """Run ``claude`` via plain subprocess (non-TTY / headless fallback).

    ``subprocess.run`` inherits stdin/stdout/stderr so piped use works.
    ``KeyboardInterrupt`` is translated to exit code 130.
    """
    env = os.environ.copy()
    env.update(env_overrides)
    try:
        result = subprocess.run([claude_bin, *claude_args], env=env, check=False)
        return result.returncode
    except KeyboardInterrupt:
        return _EXIT_SIGINT


def _start_native_server(config: Path) -> NativeServer:
    """Start the native server; kept separate for launcher supervision tests."""
    return NativeServer(config)


def _run_claude_with_switchyard(
    config: Path,
    display_model: str,
    claude_args: list[str],
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
    server = _start_native_server(config)
    resolved_port = server.port
    stats = server.stats
    strategy_summary = f"config → {config.name}"

    try:
        try:
            caller_auth = server.caller_auth_kind(display_model)
        except ValueError as error:
            logger.error("%s", error)
            return 1
        if caller_auth == "openai":
            logger.error(
                "this route forwards an OpenAI login; run it with `switchyard launch codex`"
            )
            return 1
        env_overrides = _claude_env(
            resolved_port,
            display_model,
            use_anthropic_auth=caller_auth == "anthropic",
        )
        if not _wait_ready(resolved_port):
            print_startup_failure(
                port=resolved_port,
                timeout_s=_READY_TIMEOUT_S,
                log_path=log_path,
            )
            return 1

        logger.info("proxy ready on port %d", resolved_port)
        print_ready_banner(
            port=resolved_port,
            display_model=display_model,
            log_path=log_path,
            strategy_summary=strategy_summary,
            routes=[display_model],
            default_route=display_model,
        )
        if stdin_is_tty():
            banner_pause()

        health = ProxyHealthMonitor(resolved_port)
        logger.debug(
            "claude env ANTHROPIC_BASE_URL=%s ANTHROPIC_MODEL=%s "
            "ANTHROPIC_CUSTOM_MODEL_OPTION=%s GATEWAY_DISCOVERY=%s",
            env_overrides.get("ANTHROPIC_BASE_URL"),
            env_overrides.get("ANTHROPIC_MODEL"),
            env_overrides.get("ANTHROPIC_CUSTOM_MODEL_OPTION"),
            env_overrides.get("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"),
        )
        if stdin_is_tty():
            footer = LiveStatsFooter(
                stats, display_model, health,
                strategy_label="config",
            )
            tui = ShellTUI(
                command=[claude_bin, *claude_args],
                footer_fn=footer.as_footer_fn(),
                footer_height=lambda: footer.height,
                env=env_overrides,
            )
            return tui.run()
        return _supervise_claude_plain(
            claude_bin, claude_args, env_overrides,
        )
    finally:
        print_session_summary(stats)
        server.close()


def launch_claude_config(
    config: Path,
    model: str,
    claude_args: list[str],
) -> int:
    """Run Claude Code against an explicit native server TOML deployment."""
    _quiet_launch_loggers()
    return _run_claude_with_switchyard(
        config,
        display_model=model,
        claude_args=claude_args,
    )
