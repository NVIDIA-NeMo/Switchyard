# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-command ``openclaw`` + V2 proxy supervisor.

Sibling of :mod:`switchyard.cli.launchers.claude_code_launcher` and
:mod:`switchyard.cli.launchers.codex_cli_launcher`, but spawns the
`OpenClaw <https://github.com/openclaw/openclaw>`_ personal-agent CLI
instead of Claude Code / Codex:

* ``switchyard launch openclaw --model <name>`` — single-model
  passthrough.  Spin up an in-process V2 passthrough proxy on a free
  local port, wire up an OpenAI Chat backend through the generic
  ``LlmTarget`` recipe, then spawn ``openclaw chat`` against the proxy.

Unlike Claude Code (env-var overrides) and Codex (transient ``-c``
provider overrides), OpenClaw is configured exclusively through its
JSON5 config file at ``~/.openclaw/openclaw.json`` — there is no
per-invocation ``--model`` / ``--base-url`` / ``-c key=value`` override
on the CLI.  OpenClaw does recognise ``OPENCLAW_CONFIG_PATH`` /
``OPENCLAW_STATE_DIR`` / ``OPENCLAW_HOME`` env vars to relocate config
and state, so we use a transient workspace:

1. ``tempfile.mkdtemp(prefix="switchyard-openclaw-")`` creates an
   ephemeral state dir that survives only for this launch.
2. A minimal ``openclaw.json`` lands in that dir declaring a
   ``models.providers.switchyard`` block (baseUrl → proxy, ``api:
   "openai-completions"``) and pinning ``agents.defaults.model.primary``
   to the switchyard-prefixed model id.
3. ``OPENCLAW_STATE_DIR`` / ``OPENCLAW_HOME`` / ``OPENCLAW_CONFIG_PATH``
   point openclaw at the transient workspace, leaving the user's real
   ``~/.openclaw/`` (sessions, channels, plugins) untouched.

OpenClaw's ``openclaw chat`` subcommand (alias for ``openclaw tui
--local``) opens an interactive local terminal UI bound to the embedded
agent runtime — the right shape for a launcher that wants the user
dropped into a chat session.  ``openclaw agent`` exists too but is a
non-interactive one-shot turn (verify uses that one).  When the user
exits the chat (or hits Ctrl-C), the proxy thread is torn down and the
transient workspace is removed.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import uvicorn

from switchyard.cli.launchers.launch_intake_config import (
    LaunchIntakeConfig,
    build_launch_capture_processors,
    capture_session_id_for_api_key,
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
from switchyard.cli.route_bundle import (
    load_route_bundle_table,
    route_bundle_declares_token_capture,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.processors.model_rewrite_request_processor import (
    ModelRewriteRequestProcessor,
)
from switchyard.lib.processors.token_capture_request_processor import SESSION_API_KEY_MARKER
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

# Identifier we register the transient provider under in openclaw.json.
# OpenClaw references models as ``<provider>/<model>``, so picking
# ``switchyard`` keeps the user-visible model ids self-explanatory.
_PROVIDER_ID = "switchyard"

# Opaque placeholder substituted into the openclaw.json template's
# ``apiKey: "${SWITCHYARD_API_KEY}"`` field. The proxy ignores inbound
# Authorization headers; the real upstream credential is injected by
# ``OpenAiNativeBackend`` at call time.
_API_KEY_ENV = "SWITCHYARD_API_KEY"
_API_KEY_PLACEHOLDER = "switchyard"

# (model_id, display_name, description) — feeds the openclaw.json
# ``models[]`` catalog entries + the per-model ``agents.defaults.models``
# aliases. Same shape as :data:`CodexModelCatalogEntry`.
OpenClawModelCatalogEntry: TypeAlias = tuple[str, str, str]


_ModelRewriteRequestProcessor = ModelRewriteRequestProcessor
_find_free_port = find_free_port


def _find_openclaw_binary() -> str | None:
    """Locate the ``openclaw`` executable.

    Checks ``$PATH`` first, then falls back to the paths npm-installed
    Node binstubs commonly land on: ``~/.npm-global/bin/openclaw``
    (machines that pin npm's global prefix), ``~/.local/bin/openclaw``
    (alternative layouts), and ``~/.nvm/versions/node/*/bin/openclaw``
    (nvm-managed Node installs — the default on this dev box).
    """
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


def _build_switchyard(
    model: str,
    api_key: str,
    base_url: str,
    timeout: float | None,
    stats: StatsAccumulator,
    extra_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> ChainRuntime:
    """OpenAI Chat Completions translation chain with live stats.

    OpenClaw's ``models.providers.switchyard`` block declares
    ``api: "openai-completions"``, so the inbound request lands on
    Switchyard's ``/v1/chat/completions`` endpoint. We pin
    :class:`BackendFormat.OPENAI` so the chain always translates to
    OpenAI Chat Completions on the way out — same shape Codex uses (no
    Anthropic ``/v1/messages`` probe is meaningful here).
    """
    return build_tier_passthrough_switchyard(
        LlmTarget(
            id="default",
            model=model,
            format=BackendFormat.OPENAI,
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
        switchyard, port, thread_name="launch-openclaw-proxy",
    )


def _qualified_model_id(model_id: str) -> str:
    """Return the ``<provider>/<model>`` id OpenClaw uses to reference a model.

    OpenClaw expects every model reference to be qualified with the
    provider id (e.g. ``switchyard/openai/gpt-5.2``). Stripping any
    accidental leading slash so callers can pass either form.
    """
    return f"{_PROVIDER_ID}/{model_id.lstrip('/')}"


def _openclaw_model_display_name(model_id: str) -> str:
    """Pretty-printable short label for a model id (last path segment)."""
    return model_id.rsplit("/", maxsplit=1)[-1]


def _build_openclaw_config(
    port: int,
    entries: Sequence[OpenClawModelCatalogEntry],
    primary_model_id: str,
) -> dict[str, Any]:
    """Build the JSON5-compatible ``openclaw.json`` body.

    The schema lives at
    https://docs.openclaw.ai/gateway/config-tools — see the
    ``models.providers.<id>`` table.

    ``primary_model_id`` is the qualified ``switchyard/<model>`` id that
    seeds ``agents.defaults.model.primary``. Each entry in *entries*
    becomes both a ``models[]`` catalog row (so OpenClaw knows the
    model's context window) and an ``agents.defaults.models`` alias (so
    the ``/model`` picker shows a friendly display name).
    """
    models_array: list[dict[str, Any]] = []
    agent_models: dict[str, Any] = {}
    for model_id, display_name, _description in entries:
        models_array.append({
            "id": model_id,
            "name": display_name,
            # OpenClaw accepts modalities as a list; "text" is the
            # universal capability that every Switchyard-routed model
            # honors. Vision / audio aren't declared because the proxy's
            # translation layer doesn't materially differ for them.
            "input": ["text"],
            "contextWindow": 128000,
            "maxTokens": 32000,
        })
        # Qualified id keys (``switchyard/<model>``) — OpenClaw's
        # ``agents.defaults.models`` uses the same form as model
        # references elsewhere in the config.
        agent_models[_qualified_model_id(model_id)] = {
            "alias": display_name,
        }
    body: dict[str, Any] = {
        "models": {
            # ``merge`` makes the switchyard provider layer on top of
            # any defaults OpenClaw ships internally rather than
            # replacing the whole providers map — future-proofing
            # against an OpenClaw release that bundles a default
            # provider list we don't want to wipe out.
            "mode": "merge",
            "providers": {
                _PROVIDER_ID: {
                    "baseUrl": f"http://127.0.0.1:{port}/v1",
                    # ``${ENV_VAR}`` interpolation is documented for
                    # apiKey; we keep the placeholder env-driven so the
                    # actual string never lands in the JSON on disk.
                    "apiKey": "${" + _API_KEY_ENV + "}",
                    "api": "openai-completions",
                    "models": models_array,
                },
            },
        },
        "agents": {
            "defaults": {
                "model": {"primary": primary_model_id},
                "models": agent_models,
            },
        },
    }
    return body


def _write_openclaw_workspace(
    port: int,
    entries: Sequence[OpenClawModelCatalogEntry],
    primary_model_id: str,
) -> str:
    """Materialise the transient OpenClaw state dir; return its path.

    Caller is responsible for removing the directory via
    :func:`_remove_openclaw_workspace` once the child exits.
    """
    workspace = tempfile.mkdtemp(prefix="switchyard-openclaw-")
    config_path = Path(workspace) / "openclaw.json"
    body = _build_openclaw_config(
        port=port,
        entries=entries,
        primary_model_id=primary_model_id,
    )
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return workspace


def _remove_openclaw_workspace(workspace: str | None) -> None:
    """Best-effort removal of the transient workspace directory."""
    if workspace is None:
        return
    try:
        shutil.rmtree(workspace, ignore_errors=True)
    except OSError:
        logger.debug(
            "failed to remove temporary OpenClaw workspace %s",
            workspace, exc_info=True,
        )


def _openclaw_env(
    workspace: str,
    intake: LaunchIntakeConfig | None = None,
    capture_session_id: str | None = None,
) -> dict[str, str]:
    """Environment that points OpenClaw at the transient workspace.

    * ``OPENCLAW_STATE_DIR`` — relocates the openclaw state root to the
      tempdir so user sessions / channels / plugins in ``~/.openclaw/``
      stay untouched for the duration of the launch.
    * ``OPENCLAW_HOME`` — pinned alongside; OpenClaw's
      path-resolution precedence is ``OPENCLAW_HOME`` > ``$HOME``.
    * ``OPENCLAW_CONFIG_PATH`` — explicit pointer at the
      ``openclaw.json`` we wrote, in case OpenClaw doesn't auto-discover
      it from ``OPENCLAW_STATE_DIR`` in the version on the user's box.
    * ``OPENCLAW_HIDE_BANNER`` — Switchyard renders its own ready
      banner; OpenClaw's would double-print.
    * ``SWITCHYARD_API_KEY`` — opaque placeholder substituted into
      ``apiKey: "${SWITCHYARD_API_KEY}"``.  The proxy ignores inbound
      Authorization; the real upstream credential is injected by the
      ``OpenAiNativeBackend`` at call time.
    """
    env = os.environ.copy()
    env["OPENCLAW_STATE_DIR"] = workspace
    env["OPENCLAW_HOME"] = workspace
    env["OPENCLAW_CONFIG_PATH"] = str(Path(workspace) / "openclaw.json")
    env["OPENCLAW_HIDE_BANNER"] = "1"
    # OpenClaw has no custom-header surface; when token capture needs a
    # session id, ride the apiKey field instead — OpenClaw echoes it as the
    # Authorization bearer, and the capture processor recognizes the marker
    # prefix to recover the session id (any id, not only a fixed shape).
    if capture_session_id:
        env[_API_KEY_ENV] = SESSION_API_KEY_MARKER + capture_session_id
        env["SWITCHYARD_SESSION_ID"] = capture_session_id
    else:
        env[_API_KEY_ENV] = _API_KEY_PLACEHOLDER
    if intake is not None:
        env["SWITCHYARD_SESSION_ID"] = intake.session_id
    return env


def _openclaw_command(
    openclaw_bin: str,
    openclaw_args: list[str],
) -> list[str]:
    """Build the OpenClaw argv: ``openclaw chat <user args>``.

    ``openclaw chat`` is OpenClaw's interactive local terminal UI (alias
    for ``openclaw tui --local``).  It binds to the embedded agent
    runtime, which is what reads ``models.providers.switchyard`` from
    our transient ``openclaw.json``.  ``openclaw agent`` is *not* an
    interactive subcommand — it runs a single non-interactive turn —
    so it's reserved for the one-shot verify path.

    Any forwarded args land after ``chat`` so the user can pass
    OpenClaw-native flags (``--message`` for an opening prompt,
    ``--session`` for a non-default session key, ``--thinking``, etc.)
    without colliding with our env-var injection.
    """
    return [openclaw_bin, "chat", *openclaw_args]


def _supervise_openclaw(
    openclaw_bin: str,
    openclaw_args: list[str],
    workspace: str,
    intake: LaunchIntakeConfig | None = None,
    capture_session_id: str | None = None,
) -> int:
    """Run ``openclaw chat`` with the transient workspace; return exit code.

    ``subprocess.run`` inherits stdin/stdout/stderr so the chat session
    works on a real terminal.  ``KeyboardInterrupt`` during the child
    becomes 130 so callers can surface a meaningful exit code.
    """
    try:
        result = subprocess.run(
            _openclaw_command(openclaw_bin, openclaw_args),
            env=_openclaw_env(
                workspace=workspace, intake=intake,
                capture_session_id=capture_session_id,
            ),
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        return _EXIT_SIGINT


def _run_openclaw_with_switchyard(
    switchyard: SwitchyardApp,
    display_model: str,
    port: int | None,
    openclaw_args: list[str],
    stats: StatsAccumulator,
    catalog_entries: Sequence[OpenClawModelCatalogEntry],
    intake: LaunchIntakeConfig | None = None,
    strategy_summary: str | None = None,
    capture_session_id: str | None = None,
) -> int:
    """Chain-agnostic supervisor: host ``switchyard`` then spawn openclaw."""
    openclaw_bin = _find_openclaw_binary()
    if openclaw_bin is None:
        logger.error(
            "openclaw binary not found. Install it with "
            "`npm install -g openclaw@latest`, or place it on your PATH.",
        )
        return _EXIT_BINARY_NOT_FOUND

    silence_launch_loggers(local_logger=logger)
    log_path = configure_debug_file_logging(display_model=display_model)
    resolved_port = port if port is not None else _find_free_port()
    primary_model_id = _qualified_model_id(display_model)
    workspace = _write_openclaw_workspace(
        port=resolved_port,
        entries=catalog_entries,
        primary_model_id=primary_model_id,
    )
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
                command=_openclaw_command(openclaw_bin, openclaw_args),
                footer_fn=footer.as_footer_fn(),
                footer_height=lambda: footer.height,
                env=_openclaw_env(
                    workspace=workspace, intake=intake,
                    capture_session_id=capture_session_id,
                ),
            ).run()

        return _supervise_openclaw(
            openclaw_bin, openclaw_args,
            workspace=workspace, intake=intake,
            capture_session_id=capture_session_id,
        )
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=_SHUTDOWN_JOIN_S)
        _remove_openclaw_workspace(workspace)


def _openclaw_catalog_entry_for_registered_model(
    model_id: str,
    primary_model_id: str,
) -> OpenClawModelCatalogEntry:
    """Build an OpenClaw catalog entry for a YAML-registered model id."""
    display = _openclaw_model_display_name(model_id)
    if model_id == primary_model_id:
        return (
            model_id,
            f"{display} (Switchyard)",
            f"Routed through Switchyard to {model_id}.",
        )
    return (
        model_id,
        f"{display} (Switchyard)",
        f"Routed through Switchyard to {model_id}.",
    )


def launch_openclaw(
    model: str,
    base_url: str,
    api_key: str,
    port: int | None,
    timeout: float | None,
    openclaw_args: list[str],
    intake: LaunchIntakeConfig | None = None,
    routing_profiles: str | None = None,
    rl_log_dir: Path | None = None,
) -> int:
    """Start a passthrough proxy and run ``openclaw chat`` against it.

    Single-model UX — ``model`` seeds the OpenClaw session, while the
    proxy preserves any model OpenClaw sends later so a client-side
    ``/model`` selection remains effective.

    When ``routing_profiles`` is given, the launcher builds a
    :class:`RouteTable` instead of a single chain: ``model`` is
    registered as a tier passthrough, then every entry from the YAML
    file is merged on top (including each tier's ``GET /v1/models``
    catalog hydration). OpenClaw's ``/model`` picker is populated from
    the merged table via ``agents.defaults.models``, so YAML-declared
    models appear alongside the launcher-configured one.

    Returns the ``openclaw`` process's exit code (or ``127`` if the
    binary wasn't found, ``130`` on Ctrl-C).
    """
    stats = StatsAccumulator()
    # Token capture activates when RL logging is on AND the bundle declares
    # token_capture_engine on a route; the capture pair replaces the rl-logging pair.
    capture = rl_log_dir is not None and route_bundle_declares_token_capture(routing_profiles)
    intake_request, intake_response = build_launch_capture_processors(
        intake, rl_log_dir, token_capture=capture,
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
    catalog_entries: list[OpenClawModelCatalogEntry] = [
        (
            model,
            f"{_openclaw_model_display_name(model)} (Switchyard)",
            f"Routed through Switchyard to {model}.",
        ),
    ]
    app: SwitchyardApp = build_single_model_table(model, switchyard)
    if routing_profiles is not None:
        # Wrap the single chain in a RouteTable so YAML routes
        # can merge on top. The launcher's `model` registers as a tier
        # passthrough; YAML entries land alongside (override on id
        # conflict). OpenClaw's /model picker iterates the catalog, so
        # YAML-declared models surface there automatically.
        from switchyard.lib.route_table import RouteTable
        table = app
        assert isinstance(table, RouteTable)
        yaml_table = load_route_bundle_table(
            routing_profiles,
            stats_accumulator=stats,
            pre_routing_request_processors=intake_request,
            extra_response_processors=intake_response,
            token_capture_enabled=rl_log_dir is not None,
        )
        for sub_model, sub_chain, sub_metadata in yaml_table.items():
            table.register(sub_model, sub_chain, metadata=sub_metadata)
        for warning in yaml_table.model_listing_warnings():
            table.add_model_listing_warning(warning)
        app = table
        catalog_seen = {entry[0] for entry in catalog_entries}
        for model_id in table.registered_models():
            if model_id in catalog_seen:
                continue
            catalog_entries.append(_openclaw_catalog_entry_for_registered_model(
                model_id=model_id,
                primary_model_id=model,
            ))
            catalog_seen.add(model_id)
    strategy_summary = (
        routing_profiles_strategy_summary(routing_profiles, model)
        if routing_profiles is not None
        else passthrough_strategy_summary(model)
    )
    return _run_openclaw_with_switchyard(
        app,
        display_model=model,
        port=port,
        openclaw_args=openclaw_args,
        stats=stats,
        catalog_entries=catalog_entries,
        intake=intake,
        strategy_summary=strategy_summary,
        capture_session_id=capture_session_id_for_api_key(intake, capture, "openclaw"),
    )


def _openclaw_catalog_entry_for_deterministic_model(
    model_id: str,
    config: DeterministicRoutingConfig,
) -> OpenClawModelCatalogEntry:
    """Build an OpenClaw catalog entry tailored to a deterministic config."""
    from switchyard.lib.route_table_builders import (
        deterministic_routing_virtual_model_id,
    )

    routing_model = deterministic_routing_virtual_model_id(config)
    display = _openclaw_model_display_name(model_id)
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
            f"{display} (Switchyard strong)",
            f"Direct Switchyard route to {model_id}.",
        )
    if model_id == config.weak.model:
        return (
            model_id,
            f"{display} (Switchyard weak)",
            f"Direct Switchyard route to {model_id}.",
        )
    return (
        model_id,
        f"{display} (Switchyard)",
        f"Direct Switchyard passthrough to discovered model {model_id}.",
    )


def launch_openclaw_deterministic_routing(
    config: DeterministicRoutingConfig,
    port: int | None,
    openclaw_args: list[str],
    intake: LaunchIntakeConfig | None = None,
    discovery_disabled: bool = False,
    rl_log_dir: Path | None = None,
) -> int:
    """Start a deterministic-routing proxy and run ``openclaw`` against it."""
    from switchyard.cli.model_catalog.model_discovery import fetch_model_ids
    from switchyard.lib.route_table_builders import (
        build_deterministic_routing_switchyard,
        build_deterministic_routing_table,
        deterministic_routing_virtual_model_id,
    )

    def _discovery_fn(base_url: str, api_key: str) -> list[str]:
        return fetch_model_ids(base_url, api_key)

    stats = StatsAccumulator()
    intake_request, intake_response = build_launch_capture_processors(
        intake, rl_log_dir,
    )
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
    catalog_entries: list[OpenClawModelCatalogEntry] = [
        _openclaw_catalog_entry_for_deterministic_model(
            model_id=routing_model,
            config=config,
        ),
        _openclaw_catalog_entry_for_deterministic_model(
            model_id=config.strong.model,
            config=config,
        ),
    ]
    catalog_seen = {entry[0] for entry in catalog_entries}
    for model_id in model_table.registered_models():
        if model_id in catalog_seen:
            continue
        catalog_entries.append(_openclaw_catalog_entry_for_deterministic_model(
            model_id=model_id,
            config=config,
        ))
        catalog_seen.add(model_id)
    # OpenClaw's primary model needs to be the virtual routing id so a
    # /model selection of "strong" or "weak" rolls back to the
    # classifier-driven pick. The catalog still exposes the tier models
    # for manual overrides, mirroring the codex deterministic UX.
    return _run_openclaw_with_switchyard(
        model_table,
        display_model=routing_model,
        port=port,
        openclaw_args=openclaw_args,
        stats=stats,
        catalog_entries=catalog_entries,
        intake=intake,
        strategy_summary=deterministic_strategy_summary(config),
        capture_session_id=capture_session_id_for_api_key(intake, False, "openclaw"),
    )
