# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-command `claude` + V2 proxy supervisor.

Implements two UXes on top of a single chain-agnostic runner:

* ``switchyard launch claude --model <name>`` — single-model
  passthrough.  Spin up an in-process V2 passthrough proxy on a free
  local port, probe the backend for native ``POST /v1/messages``
  support, build the matching chain
  (:class:`AnthropicNativeBackend` or :class:`OpenAiNativeBackend`
  with translation), then spawn ``claude`` against the proxy.
* ``switchyard launch claude --preset <id>`` — weighted-coin
  random routing across the preset's strong / weak tiers.  Same
  supervisor loop, different chain: a
  :class:`RandomRoutingRequestProcessor` picks a tier per request and
  the :class:`MultiLlmBackend` dispatches to that tier, while
  ``ANTHROPIC_MODEL`` is pinned to a Switchyard virtual model that
  represents the strong / weak pair.

In both modes ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` /
``ANTHROPIC_MODEL`` are preset (ollama convention) so Claude Code
talks to the proxy without going through its own auth setup wizard.
When ``claude`` exits (or Ctrl-C), the proxy is torn down cleanly.

Routing mode is auto-selected at startup by the generic
:class:`LlmTarget` recipe: Anthropic-looking models probe for native
``POST /v1/messages`` support, otherwise the chain falls back to
:class:`OpenAiNativeBackend` with the existing Anthropic ↔ OpenAI Chat
translation.
"""

import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import uvicorn

from switchyard.cli.launchers.claude_alias import claude_alias_for, de_claude_alias
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
from switchyard.cli.route_bundle import (
    load_route_bundle_table,
)
from switchyard.lib import startup_timing
from switchyard.lib.backends.llm_target import (
    BackendFormat,
    LlmTarget,
)
from switchyard.lib.processors.model_rewrite_request_processor import (
    ModelRewriteRequestProcessor,
)
from switchyard.lib.profiles import (
    DeterministicRoutingConfig,
)
from switchyard.lib.route_table import ChainRuntime, SwitchyardApp
from switchyard.lib.route_table_builders import (
    build_single_model_table,
    build_tier_passthrough_switchyard,
)
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard.server.shell_tui import ShellTUI

logger = logging.getLogger(__name__)

_READY_TIMEOUT_S = 10.0
_SHUTDOWN_JOIN_S = 3.0
_EXIT_BINARY_NOT_FOUND = 127
_EXIT_SIGINT = 130


def _quiet_launch_loggers() -> None:
    """Keep dependency chatter out of Claude Code's terminal UI."""
    silence_launch_loggers(local_logger=logger)


_ModelRewriteRequestProcessor = ModelRewriteRequestProcessor
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


def _build_claude_switchyard(
    model: str,
    api_key: str,
    base_url: str,
    timeout: float | None,
    stats: StatsAccumulator,
    extra_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> ChainRuntime:
    """Build Claude's single-tier profile-backed chain."""
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
        switchyard, port, thread_name="launch-claude-proxy",
    )


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
    stats: StatsAccumulator,
    model: str,
    health: ProxyHealthMonitor,
) -> Callable[[int], list[tuple[str, int]]]:
    """Return the unified live-stats footer renderer."""
    return LiveStatsFooter(stats, model, health).as_footer_fn()


def _with_claude_aliases(src: SwitchyardApp) -> SwitchyardApp:
    """Expose every registered id under its ``claude-`` aliased AND raw form.

    Same chain object on every name — alias only, no duplicate Switchyard
    instance. Two-way aliasing so users can spell the model either with or
    without the prefix regardless of which form the YAML declared:

      - `foo` → also reachable as `claude-foo` (alias inserted first so
        `registered_models()[0]` is the prefixed view, which becomes
        `ANTHROPIC_CUSTOM_MODEL_OPTION`).
      - `claude-foo` → also reachable as `foo`.

    Existing entries win on collision (a YAML that declares both `foo` and
    `claude-foo` keeps both untouched). Single-chain inputs pass through
    unchanged — there's no table to alias into.
    """
    from switchyard.lib.route_table import RouteTable
    if not isinstance(src, RouteTable):
        return src
    out = RouteTable()
    seen: set[str] = set()

    def _put(name: str, chain: object, metadata: Mapping[str, object]) -> None:
        if name in seen:
            return
        out.register(name, chain, metadata=metadata)  # type: ignore[arg-type]
        seen.add(name)

    for model_id, chain, metadata in src.items():
        # Insert the claude-prefixed alias BEFORE the original so the
        # table's first id (which becomes ANTHROPIC_CUSTOM_MODEL_OPTION)
        # is always the prefixed view.
        alias = claude_alias_for(model_id)
        if alias is not None:
            _put(alias, chain, {**dict(metadata), "display_name": alias})
        _put(model_id, chain, metadata)
        bare = de_claude_alias(model_id)
        if bare is not None:
            _put(bare, chain, {"display_name": bare})
    default_model = src.default_model()
    if default_model is not None and default_model in out.registered_models():
        out.set_default_model(default_model)
    return out


def _run_claude_with_switchyard(
    switchyard: SwitchyardApp,
    display_model: str,
    port: int | None,
    claude_args: list[str],
    stats: StatsAccumulator,
    intake: LaunchIntakeConfig | None = None,
    strategy_summary: str | None = None,
) -> int:
    """Chain-agnostic supervisor: host ``switchyard`` then spawn claude.

    Takes a pre-built :class:`Switchyard` and a display-only model
    name.  The caller decides what's in the chain (single-model
    passthrough, random routing across a preset, latency-service
    pool, …) — this function owns only the uvicorn-in-a-thread +
    ``claude`` supervision + env-var injection boilerplate that is
    identical across those modes.

    ``display_model`` is forwarded to ``ANTHROPIC_MODEL`` /
    ``ANTHROPIC_SMALL_FAST_MODEL`` so Claude Code's ``/model`` picker
    and status line show something sensible. For random-routing chains
    this is a virtual model registered in Switchyard's local model
    table.

    When stdin is a TTY, ``claude`` runs inside a ``ShellTUI`` that
    draws a live token-usage footer at the bottom of the terminal.
    When stdin is not a TTY (CI, piped input), falls back to a plain
    ``subprocess.run`` so headless use is unaffected.

    Returns the ``claude`` process's exit code (or ``127`` if the
    binary wasn't found, ``130`` on Ctrl-C).
    """
    startup_timing.mark("chain assembled")
    # Expose every table entry under a `claude-` prefixed alias so Claude
    # Code's gateway-discovery filter accepts the full listing. Originals stay
    # registered for direct-id callers; aliases share the same chain object.
    switchyard = _with_claude_aliases(switchyard)
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
    resolved_port = port if port is not None else _find_free_port()
    server, thread = _spawn_proxy_thread(switchyard, resolved_port)
    logger.debug(
        "proxy thread started port=%d alive=%s server_started=%s",
        resolved_port,
        thread.is_alive(),
        server.started,
    )
    startup_timing.mark("proxy thread started")

    try:
        if not _wait_ready(resolved_port):
            logger.debug(
                "proxy readiness timed out port=%d thread_alive=%s server_started=%s",
                resolved_port,
                thread.is_alive(),
                server.started,
            )
            print_startup_failure(
                port=resolved_port,
                timeout_s=_READY_TIMEOUT_S,
                log_path=log_path,
            )
            return 1

        startup_timing.mark("proxy health-ready")
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
                stats, display_model, health, table=table,
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
        server.should_exit = True
        thread.join(timeout=_SHUTDOWN_JOIN_S)


def launch_claude(
    model: str,
    base_url: str,
    api_key: str,
    port: int | None,
    timeout: float | None,
    claude_args: list[str],
    intake: LaunchIntakeConfig | None = None,
    routing_profiles: str | None = None,
    rl_log_dir: Path | None = None,
) -> int:
    """Start a passthrough proxy and run ``claude`` against it.

    Single-model UX — ``model`` seeds Claude Code's session, while the
    proxy preserves any model Claude Code sends later so ``/model``
    selections remain effective.

    When ``routing_profiles`` is given, the launcher builds a
    :class:`RouteTable` instead of a single chain: ``model`` is
    registered as a tier passthrough, then every entry from the YAML file
    is merged on top via :meth:`RouteTable.register`. YAML routes
    win on id conflict. The launcher's stats accumulator is threaded into
    both the launcher's chain and the YAML loader so all traffic records
    into the same accumulator.

    Returns the ``claude`` process's exit code (or ``127`` if the
    binary wasn't found, ``130`` on Ctrl-C).
    """
    _quiet_launch_loggers()
    stats = StatsAccumulator()
    intake_request, intake_response = build_launch_capture_processors(intake, rl_log_dir)
    switchyard = _build_claude_switchyard(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        stats=stats,
        extra_request_processors=intake_request,
        extra_response_processors=intake_response,
    )
    app: SwitchyardApp = build_single_model_table(model, switchyard)
    if routing_profiles is not None:
        # Wrap the single chain in a RouteTable so YAML routes can
        # merge on top. The launcher's chain registers under `model`; YAML
        # entries land alongside (and override on id conflict).
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
    strategy_summary = (
        routing_profiles_strategy_summary(routing_profiles, model)
        if routing_profiles is not None
        else passthrough_strategy_summary(model)
    )
    return _run_claude_with_switchyard(
        app,
        display_model=model,
        port=port,
        claude_args=claude_args,
        stats=stats,
        intake=intake,
        strategy_summary=strategy_summary,
    )




def launch_claude_deterministic_routing(
    config: DeterministicRoutingConfig,
    port: int | None,
    claude_args: list[str],
    intake: LaunchIntakeConfig | None = None,
    discovery_disabled: bool = False,
    rl_log_dir: Path | None = None,
) -> int:
    """Start a deterministic-routing proxy and run ``claude`` against it.

    The LLM-classifier chain (classifier → tier selector → per-tier dispatch)
    is wrapped in a :class:`RouteTable` whose virtual model is the
    deterministic routing target. Configured strong + weak models register
    as direct passthrough chains so the Claude Code ``/model`` picker can
    override the routing policy. The classifier is not user-selectable.
    """
    from switchyard.cli.model_catalog.model_discovery import fetch_model_ids
    from switchyard.lib.route_table_builders import (
        build_deterministic_routing_switchyard,
        build_deterministic_routing_table,
        deterministic_routing_virtual_model_id,
    )

    def _discovery_fn(base_url: str, api_key: str) -> list[str]:
        return fetch_model_ids(base_url, api_key)

    _quiet_launch_loggers()
    stats = StatsAccumulator()
    intake_request, intake_response = build_launch_capture_processors(intake, rl_log_dir)
    switchyard = build_deterministic_routing_switchyard(
        config,
        stats,
        pre_routing_request_processors=intake_request,
        extra_response_processors=intake_response,
    )
    routing_model = deterministic_routing_virtual_model_id(config)
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
    return _run_claude_with_switchyard(
        model_table,
        display_model=routing_model,
        port=port,
        claude_args=claude_args,
        stats=stats,
        intake=intake,
        strategy_summary=deterministic_strategy_summary(config),
    )
