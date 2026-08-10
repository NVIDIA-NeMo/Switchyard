# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Codex through an in-process native Switchyard server."""

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from switchyard.cli.launchers.codex_model_catalog import (
    CodexModelCatalogEntry,
    _codex_model_display_name,
    _remove_codex_model_catalog,
    _write_codex_model_catalog,
)
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


def _find_codex_binary() -> str | None:
    """Locate the ``codex`` executable."""
    path_hit = shutil.which("codex")
    if path_hit:
        return path_hit
    for candidate in (
        Path.home() / ".npm-global" / "bin" / "codex",
        Path.home() / ".local" / "bin" / "codex",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _wait_ready(port: int, timeout_s: float = _READY_TIMEOUT_S) -> bool:
    """Probe ``GET /health`` until HTTP 200 or timeout."""
    return wait_for_proxy_ready(port, timeout_s=timeout_s)


def _provider_overrides(
    port: int,
    *,
    model_catalog_json: str | None = None,
) -> list[str]:
    """Build transient Codex provider overrides for the local proxy."""
    base_url = f"http://127.0.0.1:{port}/v1"
    overrides = [
        "-c",
        f'model_provider="{_PROVIDER_ID}"',
        "-c",
        f'model_providers.{_PROVIDER_ID}.name="{_PROVIDER_ID}"',
        "-c",
        f'model_providers.{_PROVIDER_ID}.base_url="{base_url}"',
        "-c",
        f'model_providers.{_PROVIDER_ID}.wire_api="responses"',
        "-c",
        f'model_providers.{_PROVIDER_ID}.env_key="OPENAI_API_KEY"',
        "-c",
        f"model_providers.{_PROVIDER_ID}.requires_openai_auth=false",
    ]
    if model_catalog_json is not None:
        overrides.extend(["-c", f"model_catalog_json={json.dumps(model_catalog_json)}"])
    return overrides


def _codex_env() -> dict[str, str]:
    """Return the environment required by the transient provider."""
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "switchyard"
    return env


def _codex_command(
    codex_bin: str,
    codex_args: list[str],
    port: int,
    model: str,
    model_catalog_json: str | None = None,
) -> list[str]:
    """Build the exact Codex command for the local proxy."""
    return [
        codex_bin,
        *_provider_overrides(port, model_catalog_json=model_catalog_json),
        "-m",
        model,
        *codex_args,
    ]


def _supervise_codex(
    codex_bin: str,
    codex_args: list[str],
    port: int,
    model: str,
    model_catalog_json: str | None = None,
) -> int:
    """Run Codex and return its exit code."""
    try:
        result = subprocess.run(
            _codex_command(
                codex_bin,
                codex_args,
                port,
                model,
                model_catalog_json=model_catalog_json,
            ),
            env=_codex_env(),
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        return _EXIT_SIGINT


def _start_native_server(config: Path) -> NativeServer:
    """Start the native server; kept separate for supervision tests."""
    return NativeServer(config)


def _run_codex_with_switchyard(
    config: Path,
    display_model: str,
    codex_args: list[str],
    codex_model_catalog: Sequence[CodexModelCatalogEntry],
) -> int:
    """Host a native deployment and run Codex against it."""
    codex_bin = _find_codex_binary()
    if codex_bin is None:
        logger.error(
            "codex binary not found. Install it with "
            "`npm install -g @openai/codex`, or place it on your PATH.",
        )
        return _EXIT_BINARY_NOT_FOUND

    silence_launch_loggers(local_logger=logger)
    log_path = configure_debug_file_logging(display_model=display_model)
    server = _start_native_server(config)
    resolved_port = server.port
    stats = server.stats
    model_catalog_json: str | None = None
    strategy_summary = f"config → {config.name}"

    try:
        model_catalog_json = _write_codex_model_catalog(codex_bin, codex_model_catalog)
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
                command=_codex_command(
                    codex_bin,
                    codex_args,
                    resolved_port,
                    display_model,
                    model_catalog_json=model_catalog_json,
                ),
                footer_fn=footer.as_footer_fn(),
                footer_height=lambda: footer.height,
                env=_codex_env(),
            ).run()

        return _supervise_codex(
            codex_bin,
            codex_args,
            resolved_port,
            display_model,
            model_catalog_json=model_catalog_json,
        )
    finally:
        print_session_summary(stats)
        server.close()
        _remove_codex_model_catalog(model_catalog_json)


def launch_codex_config(
    config: Path,
    model: str,
    codex_args: list[str],
) -> int:
    """Run Codex against a native server TOML deployment."""
    catalog = [
        (
            model,
            f"{_codex_model_display_name(model)} (Switchyard)",
            f"Route from {config.name}.",
        )
    ]
    return _run_codex_with_switchyard(
        config,
        display_model=model,
        codex_args=codex_args,
        codex_model_catalog=catalog,
    )
