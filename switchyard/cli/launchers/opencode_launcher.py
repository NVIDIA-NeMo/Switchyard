# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run OpenCode through an in-process native Switchyard server.

OpenCode reads an ``opencode.json`` config from ``OPENCODE_CONFIG_DIR``. The
launcher writes a transient config that declares a custom provider
(``@ai-sdk/openai-compatible``) whose base URL points at the local Switchyard
proxy, exposes the selected route as that provider's ``switchyard/<route>``
model, then launches OpenCode against it. The workspace is cleaned up on
success, error, and interruption.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

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
_PROVIDER_ID = "switchyard"
_API_KEY_PLACEHOLDER = "switchyard"

OpenCodeModelCatalogEntry: TypeAlias = tuple[str, str, str]


def _find_opencode_binary() -> str | None:
    """Locate the ``opencode`` executable."""
    path_hit = shutil.which("opencode")
    if path_hit:
        return path_hit
    for candidate in (
        Path.home() / ".opencode" / "bin" / "opencode",
        Path.home() / ".npm-global" / "bin" / "opencode",
        Path.home() / ".local" / "bin" / "opencode",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _wait_ready(port: int, timeout_s: float = _READY_TIMEOUT_S) -> bool:
    """Probe ``GET /health`` until HTTP 200 or timeout."""
    return wait_for_proxy_ready(port, timeout_s=timeout_s)


def _qualified_model_id(model_id: str) -> str:
    """Return the provider-qualified model ID used by OpenCode."""
    return f"{_PROVIDER_ID}/{model_id.lstrip('/')}"


def _opencode_model_display_name(model_id: str) -> str:
    """Return a short display name for a model ID."""
    return model_id.rsplit("/", maxsplit=1)[-1]


def _build_opencode_config(
    port: int,
    entries: Sequence[OpenCodeModelCatalogEntry],
    primary_model_id: str,
) -> dict[str, object]:
    """Build the transient OpenCode configuration pointing at the proxy."""
    model_defs: dict[str, dict[str, str]] = {}
    for model_id, display_name, _description in entries:
        model_defs[model_id] = {"name": display_name, "id": model_id}
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            _PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Switchyard",
                "options": {
                    "baseURL": f"http://127.0.0.1:{port}/v1",
                    "apiKey": _API_KEY_PLACEHOLDER,
                },
                "models": model_defs,
            }
        },
        "model": primary_model_id,
    }


def _write_opencode_workspace(
    port: int,
    entries: Sequence[OpenCodeModelCatalogEntry],
    primary_model_id: str,
) -> tuple[str, Path]:
    """Write a transient OpenCode workspace and return ``(dir, config_path)``."""
    workspace = tempfile.mkdtemp(prefix="switchyard-opencode-")
    config_path = Path(workspace) / "opencode.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _build_opencode_config(port, entries, primary_model_id),
            handle,
            indent=2,
        )
        handle.write("\n")
    return workspace, config_path


def _remove_opencode_workspace(workspace: str | None) -> None:
    """Remove a transient OpenCode workspace."""
    if workspace is not None:
        shutil.rmtree(workspace, ignore_errors=True)


def _opencode_env(workspace: str) -> dict[str, str]:
    """Return the environment that selects the transient config directory."""
    env = os.environ.copy()
    env["OPENCODE_CONFIG_DIR"] = workspace
    env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
    return env


def _opencode_command(
    opencode_bin: str,
    opencode_args: list[str],
    model_id: str,
) -> list[str]:
    """Build the OpenCode command for the local proxy."""
    # ``run`` is non-interactive; without it OpenCode starts the TUI.
    if opencode_args and opencode_args[0] == "run":
        return [opencode_bin, "run", "-m", model_id, *opencode_args[1:]]
    return [opencode_bin, "-m", model_id, *opencode_args]


def _supervise_opencode(
    opencode_bin: str,
    opencode_args: list[str],
    model_id: str,
    workspace: str,
) -> int:
    """Run OpenCode and return its exit code."""
    try:
        result = subprocess.run(
            _opencode_command(opencode_bin, opencode_args, model_id),
            env=_opencode_env(workspace),
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        return _EXIT_SIGINT


def _start_native_server(config: Path) -> NativeServer:
    """Start the native server; kept separate for supervision tests."""
    return NativeServer(config)


def _run_opencode_with_switchyard(
    config: Path,
    display_model: str,
    opencode_args: list[str],
    catalog_entries: Sequence[OpenCodeModelCatalogEntry],
) -> int:
    """Host a native deployment and run OpenCode against it."""
    opencode_bin = _find_opencode_binary()
    if opencode_bin is None:
        logger.error(
            "opencode binary not found. Install it with "
            "`npm install -g opencode-ai@latest`, or place it on your PATH."
        )
        return _EXIT_BINARY_NOT_FOUND

    silence_launch_loggers(local_logger=logger)
    log_path = configure_debug_file_logging(display_model=display_model)
    server = _start_native_server(config)
    resolved_port = server.port
    stats = server.stats
    workspace_dir: str | None = None
    strategy_summary = f"config → {config.name}"

    try:
        workspace_dir, _workspace_config = _write_opencode_workspace(
            port=resolved_port,
            entries=catalog_entries,
            primary_model_id=_qualified_model_id(display_model),
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

        if stdin_is_tty():
            footer = LiveStatsFooter(
                stats,
                display_model,
                ProxyHealthMonitor(resolved_port),
                strategy_label="config",
            )
            model_id = _qualified_model_id(display_model)
            return ShellTUI(
                command=_opencode_command(opencode_bin, opencode_args, model_id),
                footer_fn=footer.as_footer_fn(),
                footer_height=lambda: footer.height,
                env=_opencode_env(workspace_dir),
            ).run()

        return _supervise_opencode(
            opencode_bin,
            opencode_args,
            _qualified_model_id(display_model),
            workspace_dir,
        )
    finally:
        print_session_summary(stats)
        server.close()
        _remove_opencode_workspace(workspace_dir)


def launch_opencode_config(
    config: Path,
    model: str,
    opencode_args: list[str],
) -> int:
    """Run OpenCode against a native server TOML deployment."""
    catalog = [
        (
            model,
            f"{_opencode_model_display_name(model)} (Switchyard)",
            f"Route from {config.name}.",
        )
    ]
    return _run_opencode_with_switchyard(
        config,
        display_model=model,
        opencode_args=opencode_args,
        catalog_entries=catalog,
    )
