# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-command ``codex`` + V2 proxy supervisor.

Sibling of :mod:`switchyard.cli.launchers.claude_code_launcher`, but
spawns the OpenAI Codex CLI instead of Claude Code:

* ``switchyard launch codex --model <name>`` — single-model
  passthrough.  Spin up an in-process V2 passthrough proxy on a free
  local port, wire up a Responses-native OpenAI backend through the generic
  ``LlmTarget`` recipe, then spawn ``codex`` against the proxy.

Codex is OpenAI-Responses-API-only (the inbound request hits
``POST /v1/responses`` on the proxy), and its built-in ``openai``
provider does **not** honor ``OPENAI_BASE_URL``.  Pointing it at a
custom endpoint requires defining a ``[model_providers.<id>]`` block in
``~/.codex/config.toml`` *or* injecting one transiently via repeated
``-c`` flags.  We use the second path so the user's existing
``config.toml`` is untouched and the proxy is fully self-contained.

Shared launcher helpers handle the proxy process, debug logging, and live
Switchyard stats footer so Codex and Claude Code expose the same operator
surface.

When ``codex`` exits (or Ctrl-C), the proxy thread is torn down cleanly.
"""

import json
import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import uvicorn

from switchyard.cli.launchers.codex_model_catalog import (
    CodexModelCatalogEntry,
    _codex_model_display_name,
    _remove_codex_model_catalog,
    _write_codex_model_catalog,
)
from switchyard.cli.launchers.launch_intake_config import (
    LaunchIntakeConfig,
    build_launch_capture_processors,
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
    routing_profiles_strategy_summary,
    silence_launch_loggers,
    spawn_proxy_thread,
    stdin_is_tty,
    suppress_uvicorn_stream_handlers,
    wait_for_proxy_ready,
)
from switchyard.cli.launchers.live_stats_footer import LiveStatsFooter
from switchyard.cli.launchers.proxy_health_monitor import ProxyHealthMonitor
from switchyard.cli.launchers.session_summary import print_session_summary
from switchyard.cli.launchers.skill_distillation import (
    launch_skill_distillation_session,
)
from switchyard.cli.route_bundle import (
    load_route_bundle_table,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.processors.model_rewrite_request_processor import (
    ModelRewriteRequestProcessor,
)
from switchyard.lib.profiles import (
    DeterministicRoutingConfig,
)
from switchyard.lib.profiles.random_routing import (
    RandomRoutingConfig,
)
from switchyard.lib.route_table import ChainRuntime, SwitchyardApp
from switchyard.lib.route_table_builders import (
    build_single_model_table,
    build_tier_passthrough_switchyard,
    random_routing_virtual_model_id,
)
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard.server.shell_tui import ShellTUI

logger = logging.getLogger(__name__)

_READY_TIMEOUT_S = 10.0
_SHUTDOWN_JOIN_S = 3.0
_EXIT_BINARY_NOT_FOUND = 127
_EXIT_SIGINT = 130
_CODEX_BIN_ENV = "SWITCHYARD_CODEX_BIN"

# Identifier we register the transient provider under via ``-c`` overrides.
# Arbitrary — codex only cares that ``model_provider`` matches a key in
# ``model_providers``.  Kept short so the ``codex`` argv stays readable.
_PROVIDER_ID = "switchyard"


_ModelRewriteRequestProcessor = ModelRewriteRequestProcessor
_find_free_port = find_free_port


def _find_codex_binary() -> str | None:
    """Locate the ``codex`` executable.

    Checks ``$PATH`` first, then falls back to the two paths Codex's
    installers commonly write to: ``~/.npm-global/bin/codex`` (the
    ``npm install -g @openai/codex`` default on machines that pin npm's
    global prefix) and ``~/.local/bin/codex`` (alternative layouts).
    """
    explicit = os.environ.get(_CODEX_BIN_ENV)
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_absolute() and candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return None
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


def _build_switchyard(
    model: str,
    api_key: str,
    base_url: str,
    timeout: float | None,
    stats: StatsAccumulator,
    extra_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> ChainRuntime:
    """OpenAI Responses-native chain with live stats.

    Codex always speaks ``POST /v1/responses``.  Uses :class:`BackendFormat.AUTO`
    so the resolver probes ``/v1/responses`` at startup: native pass-through for
    OpenAI upstreams, Chat Completions translation for upstreams like NVIDIA NIM
    that don't expose the Responses endpoint.
    """
    return build_tier_passthrough_switchyard(
        LlmTarget(
            id="default",
            model=model,
            format=BackendFormat.AUTO,
            api_key=api_key,
            base_url=base_url,
            timeout_secs=timeout,
        ),
        stats=stats,
        enable_stats=True,
        extra_request_processors=extra_request_processors,
        extra_response_processors=extra_response_processors,
    )


def _spawn_proxy_thread(
    switchyard: SwitchyardApp, port: int,
) -> tuple[uvicorn.Server, threading.Thread]:
    """Run uvicorn in a background daemon thread, return (server, thread)."""
    return spawn_proxy_thread(
        switchyard, port, thread_name="launch-codex-proxy",
    )


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


def _run_codex_with_switchyard(
    switchyard: SwitchyardApp,
    display_model: str,
    port: int | None,
    codex_args: list[str],
    stats: StatsAccumulator,
    intake: LaunchIntakeConfig | None = None,
    codex_model_catalog: Sequence[CodexModelCatalogEntry] = (),
    strategy_summary: str | None = None,
) -> int:
    """Chain-agnostic supervisor: host ``switchyard`` then spawn codex."""
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
    resolved_port = port if port is not None else _find_free_port()
    server: uvicorn.Server | None = None
    thread: threading.Thread | None = None

    try:
        server, thread = _spawn_proxy_thread(switchyard, resolved_port)
        if not _wait_ready(resolved_port):
            print_startup_failure(
                port=resolved_port,
                timeout_s=_READY_TIMEOUT_S,
                log_path=log_path,
            )
            return 1

        suppress_uvicorn_stream_handlers()
        logger.info("proxy ready on port %d", resolved_port)
        if intake is not None:
            print_intake_warning()
        from switchyard.lib.route_table import RouteTable
        table = switchyard if isinstance(switchyard, RouteTable) else None
        print_ready_banner(
            port=resolved_port,
            display_model=display_model,
            log_path=log_path,
            strategy_summary=strategy_summary,
            profile_routes=table.registered_models() if table is not None else None,
            default_route=table.default_model() if table is not None else None,
        )
        if stdin_is_tty():
            banner_pause()

        if stdin_is_tty():
            footer = LiveStatsFooter(
                stats,
                display_model,
                ProxyHealthMonitor(resolved_port),
                table=table,
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
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=_SHUTDOWN_JOIN_S)
        _remove_codex_model_catalog(model_catalog_json)


def launch_codex(
    model: str,
    base_url: str,
    api_key: str,
    port: int | None,
    timeout: float | None,
    codex_args: list[str],
    intake: LaunchIntakeConfig | None = None,
    routing_profiles: str | None = None,
    rl_log_dir: Path | None = None,
) -> int:
    """Start a passthrough proxy and run ``codex`` against it.

    Single-model UX — ``model`` seeds the Codex session, while the proxy
    preserves any model Codex sends later so client-side model selection
    remains effective.

    When ``routing_profiles`` is given, the launcher builds a
    :class:`RouteTable` instead of a single chain: ``model`` is
    registered as a tier passthrough, then every entry from the YAML file
    is merged on top (including each tier's ``GET /v1/models`` catalog
    hydration). Codex's ``/model`` picker is populated from the merged
    table, so YAML-declared models appear alongside the launcher-
    configured one.

    Returns the ``codex`` process's exit code (or ``127`` if the
    binary wasn't found, ``130`` on Ctrl-C).
    """
    stats = StatsAccumulator()
    strategy_summary = (
        routing_profiles_strategy_summary(routing_profiles, model)
        if routing_profiles is not None
        else passthrough_strategy_summary(model)
    )
    with launch_skill_distillation_session(
        target="codex",
        display_model=model,
        strategy_summary=strategy_summary,
        stats=stats,
    ) as skill_session:
        intake_request, intake_response = build_launch_capture_processors(
            intake, rl_log_dir, skill_session.capture,
        )
        switchyard = _build_switchyard(
            model,
            api_key,
            base_url,
            timeout,
            stats,
            extra_request_processors=intake_request,
            extra_response_processors=intake_response,
        )
        codex_model_catalog: list[CodexModelCatalogEntry] = [
            (
                model,
                f"{_codex_model_display_name(model)} (Switchyard)",
                f"Routed through Switchyard to {model}.",
            ),
        ]
        app: SwitchyardApp = build_single_model_table(model, switchyard)
        if routing_profiles is not None:
            # Wrap the single chain in a RouteTable so YAML routes can
            # merge on top. The launcher's `model` registers as a tier
            # passthrough; YAML entries land alongside (override on id conflict).
            # Codex's /model picker iterates the table, so YAML-declared
            # models surface in the picker automatically.
            from switchyard.lib.route_table import RouteTable
            table = app
            assert isinstance(table, RouteTable)
            yaml_table = load_route_bundle_table(
                routing_profiles,
                stats_accumulator=stats,
                pre_routing_request_processors=intake_request,
                extra_response_processors=intake_response,
            )
            for sub_model, sub_chain, sub_metadata in yaml_table.items():
                table.register(sub_model, sub_chain, metadata=sub_metadata)
            for warning in yaml_table.model_listing_warnings():
                table.add_model_listing_warning(warning)
            app = table
            # Extend the codex catalog to include every YAML-registered model.
            catalog_models = {entry[0] for entry in codex_model_catalog}
            for model_id in table.registered_models():
                if model_id in catalog_models:
                    continue
                codex_model_catalog.append((
                    model_id,
                    f"{_codex_model_display_name(model_id)} (Switchyard)",
                    f"Routed through Switchyard to {model_id}.",
                ))
                catalog_models.add(model_id)
        exit_code = _run_codex_with_switchyard(
            app,
            display_model=model,
            port=port,
            codex_args=codex_args,
            stats=stats,
            intake=intake,
            codex_model_catalog=codex_model_catalog,
            strategy_summary=strategy_summary,
        )
        skill_session.exit_code = exit_code
        return exit_code




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
    from switchyard.lib.route_table_builders import (
        build_deterministic_routing_switchyard,
        build_deterministic_routing_table,
        deterministic_routing_virtual_model_id,
    )

    def _discovery_fn(base_url: str, api_key: str) -> list[str]:
        return fetch_model_ids(base_url, api_key)

    stats = StatsAccumulator()
    routing_model = deterministic_routing_virtual_model_id(config)
    strategy_summary = deterministic_strategy_summary(config)
    with launch_skill_distillation_session(
        target="codex",
        display_model=routing_model,
        strategy_summary=strategy_summary,
        stats=stats,
    ) as skill_session:
        intake_request, intake_response = build_launch_capture_processors(
            intake, rl_log_dir, skill_session.capture,
        )
        switchyard = build_deterministic_routing_switchyard(
            config,
            stats,
            pre_routing_request_processors=intake_request,
            extra_response_processors=intake_response,
        )
        discovery_fn = None if discovery_disabled else _discovery_fn
        model_table = build_deterministic_routing_table(
            config,
            stats,
            deterministic_routing_switchyard=switchyard,
            routing_model=routing_model,
            discovery_fn=discovery_fn,
            pre_routing_request_processors=intake_request,
            extra_response_processors=intake_response,
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
        for model_id in model_table.registered_models():
            if model_id in catalog_models:
                continue
            codex_model_catalog.append(_codex_catalog_entry_for_deterministic_model(
                model_id=model_id,
                config=config,
            ))
            catalog_models.add(model_id)
        exit_code = _run_codex_with_switchyard(
            model_table,
            # Boot codex on the virtual routing model so the LLM classifier runs by
            # default — matches launch_claude_deterministic_routing. Pinning the
            # strong model id here would hit its direct passthrough and silently
            # bypass routing.
            display_model=routing_model,
            port=port,
            codex_args=codex_args,
            stats=stats,
            intake=intake,
            codex_model_catalog=codex_model_catalog,
            strategy_summary=strategy_summary,
        )
        skill_session.exit_code = exit_code
        return exit_code
