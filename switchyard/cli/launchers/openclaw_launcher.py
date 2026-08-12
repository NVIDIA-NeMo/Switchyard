# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run OpenClaw through an in-process native Switchyard server."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

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
_API_KEY_ENV = "SWITCHYARD_API_KEY"
_API_KEY_PLACEHOLDER = "switchyard"

OpenClawModelCatalogEntry: TypeAlias = tuple[str, str, str]


def _find_openclaw_binary() -> str | None:
    """Locate the ``openclaw`` executable."""
    path_hit = shutil.which("openclaw")
    if path_hit:
        return path_hit
    for candidate in (
        Path.home() / ".npm-global" / "bin" / "openclaw",
        Path.home() / ".local" / "bin" / "openclaw",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    nvm_root = Path.home() / ".nvm" / "versions" / "node"
    if nvm_root.is_dir():
        for node_version in sorted(nvm_root.iterdir(), reverse=True):
            candidate = node_version / "bin" / "openclaw"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _wait_ready(port: int, timeout_s: float = _READY_TIMEOUT_S) -> bool:
    """Probe ``GET /health`` until HTTP 200 or timeout."""
    return wait_for_proxy_ready(port, timeout_s=timeout_s)


def _qualified_model_id(model_id: str) -> str:
    """Return the provider-qualified model ID used by OpenClaw."""
    return f"{_PROVIDER_ID}/{model_id.lstrip('/')}"


def _openclaw_model_display_name(model_id: str) -> str:
    """Return a short display name for a model ID."""
    return model_id.rsplit("/", maxsplit=1)[-1]


def _build_openclaw_config(
    port: int,
    entries: Sequence[OpenClawModelCatalogEntry],
    primary_model_id: str,
) -> dict[str, Any]:
    """Build the transient OpenClaw configuration."""
    models = [
        {
            "id": model_id,
            "name": display_name,
            "input": ["text"],
            "contextWindow": 128000,
            "maxTokens": 32000,
        }
        for model_id, display_name, _description in entries
    ]
    aliases = {
        _qualified_model_id(model_id): {"alias": display_name}
        for model_id, display_name, _description in entries
    }
    return {
        "models": {
            "mode": "merge",
            "providers": {
                _PROVIDER_ID: {
                    "baseUrl": f"http://127.0.0.1:{port}/v1",
                    "apiKey": "${" + _API_KEY_ENV + "}",
                    "api": "openai-completions",
                    "models": models,
                }
            },
        },
        "agents": {
            "defaults": {
                "model": {"primary": primary_model_id},
                "models": aliases,
            }
        },
    }


def _write_openclaw_workspace(
    port: int,
    entries: Sequence[OpenClawModelCatalogEntry],
    primary_model_id: str,
) -> str:
    """Write a transient OpenClaw workspace and return its path."""
    workspace = tempfile.mkdtemp(prefix="switchyard-openclaw-")
    config_path = Path(workspace) / "openclaw.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _build_openclaw_config(port, entries, primary_model_id),
            handle,
            indent=2,
        )
        handle.write("\n")
    return workspace


def _remove_openclaw_workspace(workspace: str | None) -> None:
    """Remove a transient OpenClaw workspace."""
    if workspace is not None:
        shutil.rmtree(workspace, ignore_errors=True)


def _openclaw_env(workspace: str) -> dict[str, str]:
    """Return the environment that selects the transient workspace."""
    env = os.environ.copy()
    env["OPENCLAW_STATE_DIR"] = workspace
    env["OPENCLAW_HOME"] = workspace
    env["OPENCLAW_CONFIG_PATH"] = str(Path(workspace) / "openclaw.json")
    env["OPENCLAW_HIDE_BANNER"] = "1"
    env[_API_KEY_ENV] = _API_KEY_PLACEHOLDER
    return env


def _openclaw_command(openclaw_bin: str, openclaw_args: list[str]) -> list[str]:
    """Build the OpenClaw interactive command."""
    return [openclaw_bin, "chat", *openclaw_args]


def _supervise_openclaw(
    openclaw_bin: str,
    openclaw_args: list[str],
    workspace: str,
) -> int:
    """Run OpenClaw and return its exit code."""
    try:
        result = subprocess.run(
            _openclaw_command(openclaw_bin, openclaw_args),
            env=_openclaw_env(workspace),
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        return _EXIT_SIGINT


def _start_native_server(config: Path) -> NativeServer:
    """Start the native server; kept separate for supervision tests."""
    return NativeServer(config)


def _run_openclaw_with_switchyard(
    config: Path,
    display_model: str,
    openclaw_args: list[str],
    catalog_entries: Sequence[OpenClawModelCatalogEntry],
) -> int:
    """Host a native deployment and run OpenClaw against it."""
    openclaw_bin = _find_openclaw_binary()
    if openclaw_bin is None:
        logger.error(
            "openclaw binary not found. Install it with "
            "`npm install -g openclaw@latest`, or place it on your PATH.",
        )
        return _EXIT_BINARY_NOT_FOUND

    silence_launch_loggers(local_logger=logger)
    log_path = configure_debug_file_logging(display_model=display_model)
    server = _start_native_server(config)
    resolved_port = server.port
    stats = server.stats
    workspace: str | None = None
    strategy_summary = f"config → {config.name}"

    try:
        workspace = _write_openclaw_workspace(
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
            return ShellTUI(
                command=_openclaw_command(openclaw_bin, openclaw_args),
                footer_fn=footer.as_footer_fn(),
                footer_height=lambda: footer.height,
                env=_openclaw_env(workspace),
            ).run()

        return _supervise_openclaw(openclaw_bin, openclaw_args, workspace)
    finally:
        print_session_summary(stats)
        server.close()
        _remove_openclaw_workspace(workspace)


def launch_openclaw_config(
    config: Path,
    model: str,
    openclaw_args: list[str],
) -> int:
    """Run OpenClaw against a native server TOML deployment."""
    catalog = [
        (
            model,
            f"{_openclaw_model_display_name(model)} (Switchyard)",
            f"Route from {config.name}.",
        )
    ]
    return _run_openclaw_with_switchyard(
        config,
        display_model=model,
        openclaw_args=openclaw_args,
        catalog_entries=catalog,
    )
