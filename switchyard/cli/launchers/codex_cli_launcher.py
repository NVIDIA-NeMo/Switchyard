# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Codex through an in-process native Switchyard server.

Codex is OpenAI-Responses-API-only (the inbound request hits
``POST /v1/responses`` on the proxy), and its built-in ``openai``
provider does **not** honor ``OPENAI_BASE_URL``.  Pointing it at a
custom endpoint requires defining a ``[model_providers.<id>]`` block in
``~/.codex/config.toml`` *or* injecting one transiently via repeated
``-c`` flags.  We use the second path so the user's existing
``config.toml`` is untouched and the proxy is fully self-contained.

The launcher generates an internal native deployment and closes the
server when Codex exits.
"""

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
from switchyard.lib.profiles import (
    DeterministicRoutingConfig,
)
from switchyard.lib.profiles.random_routing import (
    RandomRoutingConfig,
)
from switchyard.lib.route_table_builders import (
    random_routing_virtual_model_id,
)
from switchyard.server.shell_tui import ShellTUI

logger = logging.getLogger(__name__)

_READY_TIMEOUT_S = 10.0
_EXIT_BINARY_NOT_FOUND = 127
_EXIT_SIGINT = 130

# Identifier we register the transient provider under via ``-c`` overrides.
# Arbitrary — codex only cares that ``model_provider`` matches a key in
# ``model_providers``.  Kept short so the ``codex`` argv stays readable.
_PROVIDER_ID = "switchyard"


_find_free_port = find_free_port


def _find_codex_binary() -> str | None:
    """Locate the ``codex`` executable.

    Checks ``$PATH`` first, then falls back to the two paths Codex's
    installers commonly write to: ``~/.npm-global/bin/codex`` (the
    ``npm install -g @openai/codex`` default on machines that pin npm's
    global prefix) and ``~/.local/bin/codex`` (alternative layouts).
    """
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


def _format_codex_http_headers(headers: dict[str, str]) -> str:
    """Encode header dict as a TOML inline table for codex's ``-c`` flag."""
    parts = ", ".join(f'"{name}"="{value}"' for name, value in headers.items())
    return "{" + parts + "}"


def _codex_catalog_entry_for_registered_model(
    model_id: str,
    config: RandomRoutingConfig,
) -> CodexModelCatalogEntry:
    """Build a Codex picker entry for *model_id*, matching its role in *config*.

    The routing virtual id, the configured strong/weak models, and any
    discovered model each get a tailored display name and description so the
    Codex ``model`` picker explains what it would route to.
    """
    routing_model = random_routing_virtual_model_id(config)
    if model_id == routing_model:
        return (
            model_id,
            "Switchyard random routing",
            (
                "Random routes "
                f"{config.strong.model} (strong) and {config.weak.model} (weak), "
                f"p_strong={config.strong_probability:.2f}."
            ),
        )
    if model_id == config.strong.model:
        return (
            model_id,
            f"{_codex_model_display_name(model_id)} (Switchyard strong)",
            f"Direct Switchyard route to {model_id}.",
        )
    if model_id == config.weak.model:
        return (
            model_id,
            f"{_codex_model_display_name(model_id)} (Switchyard weak)",
            f"Direct Switchyard route to {model_id}.",
        )
    return (
        model_id,
        f"{_codex_model_display_name(model_id)} (Switchyard)",
        f"Direct Switchyard passthrough to discovered model {model_id}.",
    )


def _provider_overrides(
    port: int, *,
    intake: LaunchIntakeConfig | None = None,
    model_catalog_json: str | None = None,
) -> list[str]:
    """Build the ``-c key=value`` argv pairs that point codex at the proxy.

    Codex's ``-c`` flag takes a dotted ``key=value`` pair and parses the
    value as TOML, so string values must be wrapped in literal double
    quotes inside the argv string.

    Base overrides:

    * ``model_provider="switchyard"`` — switch the active provider.
    * ``model_providers.switchyard.name="switchyard"`` — display name.
    * ``model_providers.switchyard.base_url="http://127.0.0.1:<port>/v1"``
      — point at the local proxy.
    * ``model_providers.switchyard.wire_api="responses"`` — codex's
      Responses-API wire format, which our
      :class:`ResponsesEndpoint` accepts at ``/v1/responses``.
    * ``model_providers.switchyard.env_key="OPENAI_API_KEY"`` — name of
      the env var codex reads the bearer token from.
    * ``model_providers.switchyard.requires_openai_auth=false`` — opt
      out of the OAuth/login flow that the built-in ``openai`` provider
      uses; we want the env-key path, full stop.
    * ``model_catalog_json="..."`` — optional Switchyard-only catalog so
      Codex's ``/model`` picker can switch back to routed models.
    """
    base_url = f"http://127.0.0.1:{port}/v1"
    overrides = [
        "-c", f'model_provider="{_PROVIDER_ID}"',
        "-c", f'model_providers.{_PROVIDER_ID}.name="{_PROVIDER_ID}"',
        "-c", f'model_providers.{_PROVIDER_ID}.base_url="{base_url}"',
        "-c", f'model_providers.{_PROVIDER_ID}.wire_api="responses"',
        "-c", f'model_providers.{_PROVIDER_ID}.env_key="OPENAI_API_KEY"',
        "-c", f"model_providers.{_PROVIDER_ID}.requires_openai_auth=false",
    ]
    if model_catalog_json is not None:
        overrides.extend([
            "-c", f"model_catalog_json={json.dumps(model_catalog_json)}",
        ])
    if intake is not None:
        headers_toml = _format_codex_http_headers(intake.opt_in_headers())
        overrides.extend([
            "-c", f"model_providers.{_PROVIDER_ID}.http_headers={headers_toml}",
        ])
    return overrides


def _codex_env(intake: LaunchIntakeConfig | None = None) -> dict[str, str]:
    """Environment that makes Codex accept the transient Switchyard provider."""
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "switchyard"
    if intake is not None:
        env["SWITCHYARD_SESSION_ID"] = intake.session_id
    return env


def _codex_command(
    codex_bin: str,
    codex_args: list[str],
    port: int,
    model: str,
    intake: LaunchIntakeConfig | None = None,
    model_catalog_json: str | None = None,
) -> list[str]:
    """Build the exact Codex argv for the transient Switchyard provider."""
    return [
        codex_bin,
        *_provider_overrides(port, intake=intake, model_catalog_json=model_catalog_json),
        "-m",
        model,
        *codex_args,
    ]


def _supervise_codex(
    codex_bin: str,
    codex_args: list[str],
    port: int,
    model: str,
    intake: LaunchIntakeConfig | None = None,
    model_catalog_json: str | None = None,
) -> int:
    """Run ``codex`` with proxy provider injected; return its exit code.

    ``subprocess.run`` inherits stdin/stdout/stderr so the interactive
    TUI works.  ``KeyboardInterrupt`` during the child is translated to
    130 so callers can surface a meaningful exit code.

    Argv layout:

    * ``-c`` overrides from :func:`_provider_overrides` register
      the transient ``switchyard`` provider and switch to it (no edits
      to ``~/.codex/config.toml``).
    * ``-m <model>`` pins the initial model on the codex side so its
      session header / status line shows the right name. The proxy
      preserves Codex's request model so client-side model selection can
      route through the same process.
    * Caller-supplied ``codex_args`` (anything after the ``--``
      sentinel) are forwarded last so they can override our flags.

    Env tweak: ``OPENAI_API_KEY="switchyard"`` — opaque placeholder
    that satisfies codex's "no env_key set, refusing to start"
    precondition.  The proxy ignores the inbound ``Authorization``
    header; the real upstream credential is injected by
    :class:`OpenAiNativeBackend` at call time.
    """
    try:
        result = subprocess.run(
            _codex_command(
                codex_bin,
                codex_args,
                port,
                model,
                intake=intake,
                model_catalog_json=model_catalog_json,
            ),
            env=_codex_env(intake=intake),
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        return _EXIT_SIGINT


def _start_native_server(
    deployment: NativeDeployment,
    port: int | None,
) -> NativeServer:
    """Start the native server; kept separate for launcher supervision tests."""
    return NativeServer(deployment, port)


def _run_codex_with_switchyard(
    deployment: NativeDeployment,
    display_model: str,
    port: int | None,
    codex_args: list[str],
    intake: LaunchIntakeConfig | None = None,
    codex_model_catalog: Sequence[CodexModelCatalogEntry] = (),
    strategy_summary: str | None = None,
) -> int:
    """Host a native deployment and run Codex against it."""
    codex_bin = _find_codex_binary()
    if codex_bin is None:
        logger.error(
            "codex binary not found. Install it with "
            "`npm install -g @openai/codex`, or place it on your PATH.",
        )
        return _EXIT_BINARY_NOT_FOUND

    model_catalog_json = _write_codex_model_catalog(codex_bin, codex_model_catalog)
    silence_launch_loggers(local_logger=logger)
    log_path = configure_debug_file_logging(display_model=display_model)
    server = _start_native_server(deployment, port)
    resolved_port = server.port
    stats = server.stats

    try:
        if not _wait_ready(resolved_port):
            print_startup_failure(
                port=resolved_port,
                timeout_s=_READY_TIMEOUT_S,
                log_path=log_path,
            )
            return 1

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

        if stdin_is_tty():
            footer = LiveStatsFooter(
                stats,
                display_model,
                ProxyHealthMonitor(resolved_port),
                strategy_label=strategy_summary.split(":")[0].strip() if strategy_summary else None,
            )
            return ShellTUI(
                command=_codex_command(
                    codex_bin,
                    codex_args,
                    resolved_port,
                    display_model,
                    intake=intake,
                    model_catalog_json=model_catalog_json,
                ),
                footer_fn=footer.as_footer_fn(),
                footer_height=lambda: footer.height,
                env=_codex_env(intake=intake),
            ).run()

        return _supervise_codex(
            codex_bin, codex_args, resolved_port, display_model,
            intake=intake, model_catalog_json=model_catalog_json,
        )
    finally:
        print_session_summary(stats)
        server.close()
        _remove_codex_model_catalog(model_catalog_json)


def launch_codex(
    model: str,
    base_url: str,
    api_key: str,
    port: int | None,
    timeout: float | None,
    codex_args: list[str],
    intake: LaunchIntakeConfig | None = None,
    rl_log_dir: Path | None = None,
) -> int:
    """Start a native passthrough deployment and run Codex."""
    deployment = passthrough_deployment(model=model, api_key=api_key, base_url=base_url)
    codex_model_catalog: list[CodexModelCatalogEntry] = [
        (
            model_id,
            f"{_codex_model_display_name(model_id)} (Switchyard)",
            f"Routed through Switchyard to {model_id}.",
        )
        for model_id in deployment.models
    ]
    return _run_codex_with_switchyard(
        deployment,
        display_model=model,
        port=port,
        codex_args=codex_args,
        intake=intake,
        codex_model_catalog=codex_model_catalog,
        strategy_summary=passthrough_strategy_summary(model),
    )




def _codex_catalog_entry_for_deterministic_model(
    model_id: str,
    config: DeterministicRoutingConfig,
) -> CodexModelCatalogEntry:
    """Build a Codex picker entry tailored to a deterministic-routing config."""
    from switchyard.lib.route_table_builders import (
        deterministic_routing_virtual_model_id,
    )

    routing_model = deterministic_routing_virtual_model_id(config)
    if model_id == routing_model:
        return (
            model_id,
            "Switchyard deterministic routing",
            (
                "LLM-classifier routes between "
                f"{config.strong.model} (strong) and {config.weak.model} (weak) "
                f"using {config.classifier.model} (classifier, "
                f"profile={config.profile_name})."
            ),
        )
    if model_id == config.strong.model:
        return (
            model_id,
            f"{_codex_model_display_name(model_id)} (Switchyard strong)",
            f"Direct Switchyard route to {model_id}.",
        )
    if model_id == config.weak.model:
        return (
            model_id,
            f"{_codex_model_display_name(model_id)} (Switchyard weak)",
            f"Direct Switchyard route to {model_id}.",
        )
    return (
        model_id,
        f"{_codex_model_display_name(model_id)} (Switchyard)",
        f"Direct Switchyard passthrough to discovered model {model_id}.",
    )


def launch_codex_deterministic_routing(
    config: DeterministicRoutingConfig,
    port: int | None,
    codex_args: list[str],
    intake: LaunchIntakeConfig | None = None,
    discovery_disabled: bool = False,
    rl_log_dir: Path | None = None,
) -> int:
    """Start a deterministic-routing proxy and run ``codex`` against it."""
    from switchyard.cli.model_catalog.model_discovery import fetch_model_ids
    from switchyard.lib.route_table_builders import deterministic_routing_virtual_model_id

    def _discovery_fn(base_url: str, api_key: str) -> list[str]:
        return fetch_model_ids(base_url, api_key)

    routing_model = deterministic_routing_virtual_model_id(config)
    discovered_models = (
        []
        if discovery_disabled
        else _discovery_fn(config.strong.base_url or "", config.strong.api_key or "")
    )
    deployment = deterministic_deployment(
        config,
        additional_models=discovered_models,
    )
    codex_model_catalog: list[CodexModelCatalogEntry] = [
        _codex_catalog_entry_for_deterministic_model(
            model_id=routing_model,
            config=config,
        ),
        _codex_catalog_entry_for_deterministic_model(
            model_id=config.strong.model,
            config=config,
        ),
    ]
    catalog_models = {entry[0] for entry in codex_model_catalog}
    for model_id in deployment.models:
        if model_id in catalog_models:
            continue
        codex_model_catalog.append(_codex_catalog_entry_for_deterministic_model(
            model_id=model_id,
            config=config,
        ))
        catalog_models.add(model_id)
    return _run_codex_with_switchyard(
        deployment,
        # Boot codex on the virtual routing model so the LLM classifier runs by
        # default — matches launch_claude_deterministic_routing. Pinning the
        # strong model id here would hit its direct passthrough and silently
        # bypass routing.
        display_model=routing_model,
        port=port,
        codex_args=codex_args,
        intake=intake,
        codex_model_catalog=codex_model_catalog,
        strategy_summary=deterministic_strategy_summary(config),
    )
