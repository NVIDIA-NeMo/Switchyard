# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Hermes through an in-process native Switchyard server.

Hermes resolves its endpoint through the same environment overrides other
providers honour: ``OPENROUTER_BASE_URL`` overrides the upstream base URL and
``OPENROUTER_API_KEY`` provides the bearer token. The launcher points both at
the local Switchyard proxy, selects the route through ``--model`` and
``--provider custom``, and launches ``hermes`` against it. No change is made to
the user's Hermes config (~/.hermes/config.yaml).
"""

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
_API_KEY_PLACEHOLDER = "switchyard"


def _find_hermes_binary() -> str | None:
    """Locate the ``hermes`` executable."""
    path_hit = shutil.which("hermes")
    if path_hit:
        return path_hit
    for candidate in (
        Path.home() / ".local" / "bin" / "hermes",
        Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "hermes",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _wait_ready(port: int, timeout_s: float = _READY_TIMEOUT_S) -> bool:
    """Probe ``GET /health`` until HTTP 200 or timeout."""
    return wait_for_proxy_ready(port, timeout_s=timeout_s)


def _hermes_env(port: int) -> dict[str, str]:
    """Build the env-var overrides that route Hermes through our proxy.

    * ``OPENROUTER_BASE_URL`` — our proxy URL. Hermes consults this override
      when resolving the endpoint for ``--provider custom``.
    * ``OPENROUTER_API_KEY`` — opaque token for the local proxy.
    """
    env = os.environ.copy()
    env["OPENROUTER_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    env["OPENROUTER_API_KEY"] = _API_KEY_PLACEHOLDER
    # Never let Hermes phone home to update channels during a proxied run.
    env.setdefault("HERMES_DISABLE_AUTOUPDATE", "1")
    return env


def _hermes_command(hermes_bin: str, hermes_args: list[str], model: str) -> list[str]:
    """Build the Hermes command for the local proxy.

    ``--provider custom -m <model>`` are global Hermes flags, so they always
    lead. Forwarded arguments are Hermes' own command — an explicit subcommand
    (``chat -q ...``, one-shot ``-z ...``, ``resume``, ...) plus any flags —
    passed through verbatim. With nothing forwarded, default to the interactive
    ``chat`` surface.
    """
    routing = ["--provider", "custom", "-m", model]
    if not hermes_args:
        return [hermes_bin, "chat", *routing]
    return [hermes_bin, *routing, *hermes_args]


def _supervise_hermes(
    hermes_bin: str,
    hermes_args: list[str],
    model: str,
    port: int,
) -> int:
    """Run Hermes and return its exit code."""
    try:
        result = subprocess.run(
            _hermes_command(hermes_bin, hermes_args, model),
            env=_hermes_env(port),
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        return _EXIT_SIGINT


def _start_native_server(config: Path) -> NativeServer:
    """Start the native server; kept separate for supervision tests."""
    return NativeServer(config)


def _run_hermes_with_switchyard(
    config: Path,
    display_model: str,
    hermes_args: list[str],
) -> int:
    """Host a native deployment and run Hermes against it."""
    hermes_bin = _find_hermes_binary()
    if hermes_bin is None:
        logger.error(
            "hermes binary not found. Install it with "
            "`curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash`, "
            "or place it on your PATH."
        )
        return _EXIT_BINARY_NOT_FOUND

    silence_launch_loggers(local_logger=logger)
    log_path = configure_debug_file_logging(display_model=display_model)
    server = _start_native_server(config)
    resolved_port = server.port
    stats = server.stats
    strategy_summary = f"config → {config.name}"

    try:
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

        if stdin_is_tty():
            footer = LiveStatsFooter(
                stats,
                display_model,
                ProxyHealthMonitor(resolved_port),
                strategy_label="config",
            )
            return ShellTUI(
                command=_hermes_command(hermes_bin, hermes_args, display_model),
                footer_fn=footer.as_footer_fn(),
                footer_height=lambda: footer.height,
                env=_hermes_env(resolved_port),
            ).run()

        return _supervise_hermes(
            hermes_bin,
            hermes_args,
            display_model,
            resolved_port,
        )
    finally:
        print_session_summary(stats)
        server.close()


def launch_hermes_config(
    config: Path,
    model: str,
    hermes_args: list[str],
) -> int:
    """Run Hermes against a native server TOML deployment."""
    _quiet_launch_loggers()
    return _run_hermes_with_switchyard(
        config,
        display_model=model,
        hermes_args=hermes_args,
    )


def _quiet_launch_loggers() -> None:
    """Keep dependency chatter out of Hermes' terminal UI."""
    silence_launch_loggers(local_logger=logger)
