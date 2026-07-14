#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified CLI entry point for the ``Switchyard`` server.

Exposed as the ``switchyard`` console script.

Subcommands:
    serve           Serve a v2 profile config (--config) as model-keyed
                    profiles.
    launch claude   Spawn Claude Code against a proxy. Single model
                    (``--model``) or full routing via ``--routing-profiles``.
                    Pair with ``--smoke`` to run a one-shot harness round-trip
                    and exit (replaces the old ``verify claude``).
    launch codex    Spawn OpenAI Codex CLI against a proxy. Same shape as
                    ``launch claude``, including ``--smoke``.
    launch openclaw Spawn OpenClaw against a proxy via a transient
                    ``openclaw.json`` workspace. Same shape as
                    ``launch claude``, including ``--smoke``.
    verify          Run the proxy + backend e2e checklist (no harness binary).
    configure       Interactive setup wizard for saved defaults. ``--show``
                    (optionally with ``--check`` for a live ``GET /models``
                    probe), ``--reset``, and ``--list-models`` are mutually-
                    exclusive introspection modes.

Examples::

    # v2 profile config (primary serve path): each profile id + target id
    # becomes a model on GET /v1/models, selectable with --model <id>.
    switchyard serve --config profiles.yaml --port 4000

    # Deprecated: --routing-profiles is a global flag
    # (use -- to separate from the subcommand)
    switchyard --routing-profiles routes.yaml -- serve --port 4000
    switchyard --routing-profiles dev.yaml -- serve --port 4001
    switchyard --routing-profiles profiles.yaml -- launch claude
    switchyard --routing-profiles profiles.yaml -- launch codex

    # Single-model passthrough (--model stays on the launcher subcommand)
    switchyard launch claude --model openai/gpt-5.2
    switchyard launch claude --smoke --model openai/gpt-5.2
    switchyard launch codex  --model openai/gpt-5.2
    switchyard launch codex  --smoke --model openai/gpt-5.2
    switchyard launch openclaw --model openai/gpt-5.2
    switchyard launch openclaw --smoke --model openai/gpt-5.2
    switchyard verify --model openai/gpt-5.2

    # Forwarding args to the launched tool (second -- after the subcommand)
    switchyard --routing-profiles profiles.yaml -- launch claude -- --no-auto-approve

Legacy routing policies that used to be top-level CLI verbs (``passthrough``,
``random-routing``, ``routellm``, ``latency-service``) and launcher flags
(``--routing``, ``--weak-model``, ``--strong-probability``, ``--preset``)
are expressed in deprecated routing-profile YAML files. ``serve`` and
launchers still parse the YAML into profile-backed runtimes.
"""

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from typing import Any, cast

from switchyard.cli.command_utils import (
    quiet_dependency_loggers as _quiet_dependency_loggers,
)
from switchyard.cli.config.user_config import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_PROVIDER,
    DEFAULT_SECRETS_SECTION_PRIORITY,
    UserConfigError,
    resolve_provider_connectivity,
)
from switchyard.cli.configure_command import (
    cmd_configure,
)
from switchyard.cli.intake_cli_config import IntakeCliConfig
from switchyard.cli.launch_command import (
    cmd_launch_claude,
    cmd_launch_codex,
    cmd_launch_openclaw,
)
from switchyard.cli.route_bundle import (
    RouteBundleConfigError,
    load_route_bundle_table,
    route_bundle_declares_token_capture,
)
from switchyard.lib.config import IntakeSinkConfig
from switchyard.lib.processors.intake_request_processor import IntakeRequestProcessor
from switchyard.lib.processors.intake_response_processor import IntakeResponseProcessor
from switchyard.lib.processors.rl_logging_response_processor import build_rl_logging_processors
from switchyard.lib.processors.token_capture_response_processor import (
    build_token_capture_processors,
)
from switchyard.server.server_util import (
    DEFAULT_SECRETS_FILE,
    add_transport_args,
    build_and_serve,
    load_secrets,
    resolve_port,
    resolve_rl_log_dir,
)

logger = logging.getLogger(__name__)

_DEFAULT_OPENROUTER_BASE_URL = DEFAULT_OPENROUTER_BASE_URL
_CANONICAL_INTAKE_ENABLE_FLAG = "--intake-enabled"
_DEPRECATED_INTAKE_ENABLE_FLAG = "--enable-intake"
_DEPRECATED_ROUTING_PROFILES_FLAG = "--routing-profiles"
_ARGPARSE_ACTION_SUPPORTS_DEPRECATED = (
    "deprecated" in signature(argparse.Action.__init__).parameters
)


def _print_deprecation_warning(
    subject: str,
    details: tuple[str, ...] = (),
) -> None:
    """Print a concise, readable CLI deprecation warning to stderr."""
    print(f"warning: {subject} is deprecated.", file=sys.stderr)
    for detail in details:
        print(f"  {detail}", file=sys.stderr)


def _warn_deprecated_routing_profiles() -> None:
    _print_deprecation_warning(
        _DEPRECATED_ROUTING_PROFILES_FLAG,
        details=(
            "Prefer v2 profile configs with `switchyard serve --config PATH`.",
            "Legacy route bundles still run for now, but this flag will be removed in a future release.",
        ),
    )


def _warn_deprecated_saved_route_bundle() -> None:
    _print_deprecation_warning(
        "saved routing-profile bundle",
        details=(
            "Prefer v2 profile configs with `switchyard serve --config PATH`.",
            "Clear the saved bundle with `switchyard --routing-profiles '' -- configure`.",
        ),
    )


def _warn_deprecated_python_profile_server(python_profiles: Sequence[str]) -> None:
    _print_deprecation_warning(
        "Python-defined profile serving",
        details=(
            "This config is using the Python FastAPI adapter.",
            f"Python profile(s): {', '.join(python_profiles)}.",
            "Prefer Rust-defined components-v2 profiles for new serve configs.",
        ),
    )


class _IntakeEnabledAction(argparse.Action):
    """Store the normalized Intake enable flag and warn on the deprecated alias."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        nargs: int | str | None = None,
        const: Any = None,
        default: Any = None,
        type: Any = None,
        choices: Any = None,
        required: bool = False,
        help: str | None = None,
        metavar: str | tuple[str, ...] | None = None,
        deprecated: bool = False,
    ) -> None:
        del nargs
        action_kwargs: dict[str, Any] = {
            "option_strings": option_strings,
            "dest": dest,
            "nargs": 0,
            "const": const,
            "default": default,
            "type": type,
            "choices": choices,
            "required": required,
            "help": help,
            "metavar": metavar,
        }
        if _ARGPARSE_ACTION_SUPPORTS_DEPRECATED:
            action_kwargs["deprecated"] = deprecated
        super().__init__(**action_kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        if option_string == _DEPRECATED_INTAKE_ENABLE_FLAG:
            logger.warning(
                "%s is deprecated; use %s",
                _DEPRECATED_INTAKE_ENABLE_FLAG,
                _CANONICAL_INTAKE_ENABLE_FLAG,
            )
        setattr(namespace, self.dest, True)


def _add_intake_enabled_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        _CANONICAL_INTAKE_ENABLE_FLAG,
        _DEPRECATED_INTAKE_ENABLE_FLAG,
        dest="intake_enabled",
        default=False,
        action=_IntakeEnabledAction,
        help=f"{help_text} Deprecated alias: {_DEPRECATED_INTAKE_ENABLE_FLAG}.",
    )


def _resolve_intake_config(args: argparse.Namespace) -> IntakeSinkConfig | None:
    intake = IntakeCliConfig.from_server_args(args)
    if not intake.enabled:
        return None
    return IntakeSinkConfig(
        intake_base_url=intake.base_url,
        workspace=intake.workspace,
        api_key=intake.api_key,
        nvdataflow_project=intake.nvdataflow_project,
    )


def _resolve_intake_processors(
    args: argparse.Namespace,
) -> tuple[list[Any], list[Any]]:
    intake = _resolve_intake_config(args)
    if intake is None:
        return [], []
    return [IntakeRequestProcessor()], [IntakeResponseProcessor(intake)]


def _add_intake_args(parser: argparse.ArgumentParser) -> None:
    _add_intake_enabled_arg(
        parser,
        help_text=(
            "Enable the Intake sink; requests still opt in with store=true "
            "or x-switchyard-intake-enabled=true"
        ),
    )
    _add_common_intake_args(parser)


def _add_common_intake_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--intake-base-url", type=str, default=None,
        help=(
            "Override the intake base URL (default: read from nmp config "
            "or $SWITCHYARD_INTAKE_BASE_URL)."
        ),
    )
    parser.add_argument(
        "--intake-workspace", type=str, default=None,
        help=(
            "Override the workspace for intake entries (default: "
            "$SWITCHYARD_INTAKE_WORKSPACE, then the NMP SDK config)."
        ),
    )
    parser.add_argument(
        "--intake-api-key", type=str, default=None,
        help=(
            "Override the bearer token (default: read from nmp config — "
            "with refresh — or fall back to $SWITCHYARD_INTAKE_API_KEY / "
            "$NMP_ACCESS_TOKEN). Setting this opts out of the SDK's "
            "config-based bootstrap and disables transparent refresh."
        ),
    )
    parser.add_argument(
        "--intake-nvdataflow-project", type=str, default=None,
        help=(
            "Post flat per-request telemetry to this NVDataflow project's "
            "posting endpoint instead of nemo-platform chat-completions "
            "ingest. Defaults to $SWITCHYARD_NVDATAFLOW_PROJECT."
        ),
    )


_DETERMINISTIC_PROFILE_CHOICES = ("general", "coding_agent", "openclaw")


def _add_launch_deterministic_override_args(parser: argparse.ArgumentParser) -> None:
    """Attach the deterministic-trio override knobs to a ``launch`` subparser.

    Used by ``launch claude``, ``launch codex``, and ``launch openclaw``,
    where LLM-classifier routing is the implicit default — these flags let
    users tune the validated TB-Lite trio without writing a full
    routing-profiles YAML. They have no effect when ``--model X`` or
    ``--routing-profiles FILE`` opts the launcher out of LLM-classifier
    routing.
    """
    parser.add_argument(
        "--weak-model",
        type=str,
        default=None,
        help=(
            "Override the weak-tier model id of the default deterministic "
            "trio. Defaults to moonshotai/kimi-k2.6."
        ),
    )
    parser.add_argument(
        "--classifier-model",
        type=str,
        default=None,
        help=(
            "Override the classifier LLM. Defaults to "
            "google/gemini-3.5-flash (cheap + fast)."
        ),
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=_DETERMINISTIC_PROFILE_CHOICES,
        help=(
            "Classifier profile. Defaults to coding_agent "
            "(SIMPLE+MEDIUM→weak, COMPLEX/REASONING→strong, "
            "tool-planning escalation, high-confidence LLM alignment)."
        ),
    )
    parser.add_argument(
        "--classifier-min-confidence",
        type=float,
        default=None,
        help=(
            "Tier-selector confidence floor in [0.0, 1.0]. Below this "
            "the classifier's decision falls back to the strong tier "
            "(fail safer, not cheaper). Defaults to 0.0 — honor every "
            "non-abstain classification."
        ),
    )


def _add_launch_intake_args(parser: argparse.ArgumentParser) -> None:
    """
    Attach ``--intake-enabled`` + companion args to a ``launch`` subparser.
    """
    _add_intake_enabled_arg(
        parser,
        help_text=(
            "Wire intake into both the in-process proxy and the spawned "
            "client. Uses the NMP SDK's configured credentials by default "
            "(run `nmp auth login --base-url {INTAKE_BASE_URL}` "
            "once). Override with --intake-base-url / --intake-api-key for "
            "headless or CI use."
        ),
    )
    _add_common_intake_args(parser)
    parser.add_argument(
        "--intake-app", type=str, default=None,
        help=(
            "App name for entry context (default: $SWITCHYARD_INTAKE_APP, "
            "then a launcher-specific default)."
        ),
    )
    parser.add_argument(
        "--intake-task", type=str, default=None,
        help=(
            "Task name for entry context (default: $SWITCHYARD_INTAKE_TASK, "
            "then 'developer-session')."
        ),
    )
    parser.add_argument(
        "--intake-session-id", type=str, default=None,
        help=(
            "Session id stamped on every entry. Defaults to "
            "$SWITCHYARD_SESSION_ID, then '<target>-<unix-ms>-<rand>'."
        ),
    )
    parser.add_argument(
        "--intake-user-id", type=str, default=None,
        help=(
            "Anonymous user id stamped on every entry. Defaults to "
            "$SWITCHYARD_USER_ID, then the stable per-machine id at "
            "~/.switchyard/user_id (created on first use)."
        ),
    )


def _cmd_configure(args: argparse.Namespace) -> None:
    cmd_configure(args)


def _cmd_launch_claude(args: argparse.Namespace) -> None:
    cmd_launch_claude(args)


def _cmd_launch_codex(args: argparse.Namespace) -> None:
    cmd_launch_codex(args)


def _cmd_launch_openclaw(args: argparse.Namespace) -> None:
    cmd_launch_openclaw(args)


# ---------------------------------------------------------------------------
# Subcommand: serve
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> None:
    """Serve a v2 profile config or a legacy route bundle."""
    from switchyard.cli.config.user_config import load_user_config
    from switchyard.cli.route_bundle import build_route_bundle_table

    routing_profiles = args.routing_profiles
    if args.config:
        _cmd_serve_profile_config(args)
        return

    # Intake sink and local RL trace logging attach as chain processors.
    # Token capture activates when RL logging is on AND the bundle declares
    # an token_capture_engine target; the capture pair then replaces the
    # rl-logging pair (the unified record carries the trace fields — running
    # both would double-write).
    saved = None
    if not routing_profiles:
        saved = load_user_config().routing_profiles
        if not saved:
            raise SystemExit(
                "serve: no routing-profiles given. Pass --routing-profiles "
                "PATH or run `switchyard configure --routing-profiles PATH` "
                "to save one."
            )
    intake_request, intake_response = _resolve_intake_processors(args)
    rl_log_dir = resolve_rl_log_dir(args)
    capture = rl_log_dir is not None and route_bundle_declares_token_capture(
        routing_profiles or saved
    )
    if capture:
        log_request, log_response = build_token_capture_processors(rl_log_dir)
    else:
        log_request, log_response = build_rl_logging_processors(rl_log_dir)
    request_processors = [*intake_request, *log_request]
    response_processors = [*intake_response, *log_response]
    if routing_profiles:
        table = load_route_bundle_table(
            routing_profiles,
            pre_routing_request_processors=request_processors,
            extra_response_processors=response_processors,
            token_capture_enabled=rl_log_dir is not None,
        )
        source = routing_profiles
    else:
        # Saved bundles are stored as parsed dicts, so we can skip the
        # YAML parse step and feed them straight into the dict-driven
        # entrypoint. Env-var references inside the dict expand inside
        # build_route_bundle_table on each run.
        _warn_deprecated_saved_route_bundle()
        table = build_route_bundle_table(
            saved,
            pre_routing_request_processors=request_processors,
            extra_response_processors=response_processors,
            token_capture_enabled=rl_log_dir is not None,
        )
        source = f"<saved bundle, {len(table.registered_models())} route(s)>"
    logger.info(
        "Switchyard route bundle loaded %d route(s) from %s",
        len(table.registered_models()),
        source,
    )
    strategy_summary: str | None = None
    default_model = table.default_model()
    if routing_profiles and default_model:
        from switchyard.cli.launchers.launcher_runtime import routing_profiles_strategy_summary
        try:
            strategy_summary = routing_profiles_strategy_summary(
                routing_profiles[0], default_model,
            )
        except Exception:
            pass
    build_and_serve(args, table, inbound_default="both", strategy_summary=strategy_summary)


def _cmd_serve_profile_config(args: argparse.Namespace) -> None:
    """Serve a components-v2 profile config."""
    if args.routing_profiles:
        raise SystemExit(
            "serve --config cannot be combined with --routing-profiles; "
            "use exactly one config surface."
        )
    if args.inbound is not None:
        raise SystemExit(
            "serve --config always exposes "
            "OpenAI, Anthropic, and Responses endpoints; omit --inbound."
        )
    if args.reload:
        raise SystemExit("serve --config does not support --reload.")
    if args.workers != 1:
        raise SystemExit("serve --config does not support --workers.")
    unsupported_intake = any((
        args.intake_enabled,
        args.intake_base_url,
        args.intake_workspace,
        args.intake_api_key,
        args.intake_nvdataflow_project,
    ))
    if unsupported_intake:
        raise SystemExit("serve --config does not support Intake options yet.")
    if getattr(args, "enable_rl_logging", False):
        raise SystemExit(
            "serve --config does not support --enable-rl-logging: the Rust "
            "profile server has no Python processor chain to attach the trace "
            "logger to. Use serve --routing-profiles for local RL trace logging."
        )

    # Inspect first so files containing only Rust-defined profiles can use the
    # Rust server path, while files with Python-defined profiles use FastAPI.
    from switchyard.lib.profiles.loader import python_profile_ids

    try:
        python_profiles = python_profile_ids(args.config)
    except Exception as exc:
        raise SystemExit(
            "serve --config: failed to inspect profile config for "
            f"Python-defined profiles: {exc}"
        ) from exc
    if python_profiles:
        _cmd_serve_mixed_profile_config(args, python_profiles)
        return

    from switchyard_rust.server import run_profile_server

    port = args.port if isinstance(args.port, int) else resolve_port()
    logger.info(
        "Switchyard components-v2 profile config loaded from %s",
        args.config,
    )
    run_profile_server(args.config, args.host, port)


def _cmd_serve_mixed_profile_config(
    args: argparse.Namespace,
    python_profiles: list[str],
) -> None:
    """Serve a config containing Python-defined profiles through FastAPI."""
    _warn_deprecated_python_profile_server(python_profiles)
    table = _profile_config_route_table(args.config)
    logger.info(
        "Switchyard profile config loaded from %s with Python-defined profile(s): %s",
        args.config,
        ", ".join(python_profiles),
    )
    build_and_serve(args, table, inbound_default="both")


def _profile_config_route_table(config_path: str) -> Any:
    from switchyard.lib.profiles import PassthroughProfileConfig, ProfileSwitchyard
    from switchyard.lib.profiles.loader import load_profiles_and_targets
    from switchyard.lib.route_table import RouteTable

    profiles, targets = load_profiles_and_targets(config_path)
    table = RouteTable()
    for profile_id, profile in profiles.items():
        _register_profile_config_model(
            table,
            profile_id,
            ProfileSwitchyard(cast(Any, profile)),
            kind="profile",
        )
    for target_id, target in targets.items():
        target_switchyard = ProfileSwitchyard(PassthroughProfileConfig(target=target).build())
        _register_profile_config_model(
            table,
            target_id,
            target_switchyard,
            kind="target",
        )
        if target.model != target_id:
            _register_profile_config_model(
                table,
                target.model,
                target_switchyard,
                kind="target-model",
            )
    return table


def _register_profile_config_model(
    table: Any,
    model_id: str,
    switchyard: Any,
    kind: str,
) -> None:
    if model_id in table.registered_models():
        raise SystemExit(
            f"serve --config: duplicate public model id {model_id!r} while "
            "registering profile config routes."
        )
    table.register(
        model_id,
        switchyard,
        metadata={"switchyard": {"source": "profile-config", "kind": kind}},
    )


# ---------------------------------------------------------------------------
# Subcommand: verify {proxy,claude,codex}
# ---------------------------------------------------------------------------


# Default verify model.  Matches ``tests/offline_production_tests/conftest.py``
# (NVIDIA inference-api) so a user with secrets/secrets.json wired up can
# run ``switchyard verify proxy`` with no args at all and get a
# meaningful pass/fail.
_DEFAULT_VERIFY_MODEL = "openai/gpt-5.2"

# Env vars and secrets-section priority verified by ``verify``.  Same
# resolution order as :func:`_resolve_verify_credentials` uses through
# ``resolve_provider_connectivity`` — kept as module constants so the
# diagnostic helper can iterate them in the same order without importing
# private helper internals.
_VERIFY_API_KEY_ENV_VARS: tuple[str, ...] = (
    "OPENROUTER_API_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)
_VERIFY_BASE_URL_ENV_VARS: tuple[str, ...] = (
    "OPENROUTER_BASE_URL",
    "NVIDIA_BASE_URL",
    "OPENAI_BASE_URL",
)
_VERIFY_SECRETS_PRIORITY: tuple[str, ...] = DEFAULT_SECRETS_SECTION_PRIORITY


@dataclass
class _CredentialAttempt:
    """One row of the credential-resolution diagnostic.

    ``label`` is a stable short token (``"--api-key"`` /
    ``"$OPENROUTER_API_KEY"`` / ``"secrets.json[openrouter]"``) suitable for
    both the user-facing error table and the ``key_source`` label
    forwarded into :func:`verify_proxy` (which surfaces it in
    step 1's success line and step 2's 401 hint).

    ``status`` is the human-readable verdict for that source:
    ``"not provided"``, ``"set (32 chars)"``, ``"file not found"``,
    ``"missing 'api_key' field"``, etc.

    ``has_value`` is True iff the source produced a non-empty key.
    The first :class:`_CredentialAttempt` with ``has_value=True`` is
    the resolved key — same waterfall as
    :func:`resolve_provider_connectivity`.
    """

    label: str
    status: str
    has_value: bool


def _redact_status(value: str | None) -> str:
    """Render an env-var / secrets-field value as a status string.

    ``None`` ⇒ ``"not set"``.  Empty string ⇒ ``"empty (0 chars)"`` so
    the user notices an env var that was set but exported as empty
    (a common subtle bug).  Otherwise ``"set (N chars)"`` — the char
    count helps the user spot truncation without exposing the secret.
    """
    if value is None:
        return "not set"
    if not value:
        return "empty (0 chars)"
    return f"set ({len(value)} chars)"


def _diagnose_credential_resolution(
    args: argparse.Namespace,
) -> tuple[str | None, str | None, str | None, list[_CredentialAttempt]]:
    """Walk the same waterfall as ``_resolve_verify_credentials`` but
    record every source we touched along the way.

    Returns ``(api_key, base_url, key_source, attempts)``:

    * ``api_key`` / ``base_url`` — same values
      :func:`resolve_provider_connectivity` would return.
    * ``key_source`` — the ``label`` of the first attempt whose
      ``has_value`` is True, or ``None`` when nothing produced a key.
    * ``attempts`` — every source we considered, in resolution order.
      Always non-empty.  Used to build the tabulated error message
      when no source produced a key.

    The walk *parallels* :func:`resolve_provider_connectivity` rather
    than wrapping it because the underlying helper has no notion of
    "which source won".  Duplication here is bounded (one waterfall, six sources)
    and the diagnostic UX is the whole point of the wrapper.
    """
    attempts: list[_CredentialAttempt] = []
    resolved_key: str | None = None
    key_source: str | None = None

    def _record(label: str, value: str | None, status: str) -> None:
        """Append a row to ``attempts`` and pin the resolved key on
        the first non-empty value.  ``status`` is overridable so
        secrets-file rows can carry per-section context (file
        missing vs. field missing vs. set).
        """
        nonlocal resolved_key, key_source
        has_value = bool(value)
        attempts.append(_CredentialAttempt(label, status, has_value))
        if has_value and resolved_key is None:
            resolved_key = value
            key_source = label

    # 1. --api-key
    cli_key = args.api_key
    _record(
        "--api-key",
        cli_key,
        "provided" if cli_key else "not provided",
    )

    # 2. env vars (in order)
    for env_var in _VERIFY_API_KEY_ENV_VARS:
        env_value = os.environ.get(env_var)
        _record(f"${env_var}", env_value, _redact_status(env_value))

    # 3. secrets.json — break down failure modes so the user knows
    # whether the file is missing, the section is missing, or the
    # field is missing.  Each is a different fix.
    secrets_path = DEFAULT_SECRETS_FILE
    if not secrets_path.exists():
        for section_name in _VERIFY_SECRETS_PRIORITY:
            _record(
                f"secrets.json[{section_name}.api_key]",
                None,
                f"file not found ({secrets_path})",
            )
    else:
        secrets = load_secrets()
        for section_name in _VERIFY_SECRETS_PRIORITY:
            section_body = secrets.get(section_name)
            label = f"secrets.json[{section_name}.api_key]"
            # ``SecretsFile`` types every section as a Mapping, so the
            # only structural failures we surface are "section missing"
            # and "section present but no api_key field".  A malformed
            # JSON file would have crashed in :func:`load_secrets`
            # already, so we don't double-check the section's type here.
            if section_body is None:
                _record(label, None, f"section '{section_name}' missing")
                continue
            value = section_body.get("api_key")
            if not isinstance(value, str):
                _record(label, None, "field missing")
                continue
            _record(label, value, _redact_status(value))

    # base_url has its own waterfall, but verify's error story doesn't need
    # a per-source breakdown. Resolve it with the same provider-aware pairing
    # as launch/configure so mixed provider env vars do not get mismatched.
    connectivity = resolve_provider_connectivity(
        cli_api_key=args.api_key,
        cli_base_url=args.base_url,
        api_key_env_vars=_VERIFY_API_KEY_ENV_VARS,
        base_url_env_vars=_VERIFY_BASE_URL_ENV_VARS,
        secrets=load_secrets() if DEFAULT_SECRETS_FILE.exists() else None,
        secrets_section_priority=_VERIFY_SECRETS_PRIORITY,
    )

    return resolved_key, connectivity.base_url, key_source, attempts


def _format_credential_failure(attempts: list[_CredentialAttempt]) -> str:
    """Render the multi-line tabulated diagnostic for the no-key case.

    Goal: the user sees, at a glance, *every* source verify checked
    and *what was wrong with each one*, plus a concrete "to fix:"
    pointer for the most common case.  Aligning the labels makes the
    table scannable in dark-mode terminals where the eye is doing
    column scanning, not word reading.
    """
    width = max(len(a.label) for a in attempts) + 2
    lines = [
        "verify: no API key resolved.  Tried:",
    ]
    for attempt in attempts:
        lines.append(f"  {attempt.label.ljust(width)}{attempt.status}")
    lines.append("")
    lines.append(
        "To fix, set ONE of the above sources.  Most common:",
    )
    lines.append(
        "  export OPENROUTER_API_KEY=sk-...   "
        "# (or any other env var listed above)",
    )
    lines.append(
        "  cp secrets/secrets.template.json secrets/secrets.json   "
        "# then edit it",
    )
    return "\n".join(lines)


def _resolve_verify_credentials(
    args: argparse.Namespace,
) -> tuple[str, str, str | None]:
    """Common credential resolution for all three ``verify`` modes.

    Mirrors :func:`_cmd_launch_claude` / :func:`_cmd_launch_codex`:
    ``--api-key`` / ``OPENROUTER_API_KEY`` / ``NVIDIA_API_KEY`` /
    ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / secrets.json waterfall;
    ``--base-url`` / ``OPENROUTER_BASE_URL`` / ``NVIDIA_BASE_URL`` /
    ``OPENAI_BASE_URL`` waterfall; defaults to OpenRouter when the user
    didn't supply one.

    Unlike the launchers, this helper raises ``SystemExit`` with a
    *tabulated* error rather than warning + injecting a ``"dummy"``
    placeholder.  Verify is *the* command that confirms credential
    resolution works — silently masking a missing key would defeat
    the whole purpose, and a generic "set one of these" hint makes
    the user check every source instead of pointing them at the one
    they actually broke.

    Returns ``(api_key, base_url, key_source)`` where ``key_source``
    is the human label of the source that produced the key (e.g.
    ``"$OPENROUTER_API_KEY"``).  Forwarded to the verify orchestrators
    so the credential-step's success line and the backend-reach
    step's 401 hint can name the source by hand.
    """
    api_key, base_url, key_source, attempts = (
        _diagnose_credential_resolution(args)
    )
    resolved_base_url = base_url or _DEFAULT_OPENROUTER_BASE_URL
    if not api_key:
        raise SystemExit(_format_credential_failure(attempts))
    return api_key, resolved_base_url, key_source


def _cmd_verify(args: argparse.Namespace) -> None:
    """Run the proxy-only verify checklist.

    Harness-driven verify modes (``verify claude`` / ``verify codex``)
    moved to ``launch claude --smoke`` / ``launch codex --smoke``;
    ``verify`` is now the leaf for the proxy-only checklist that K8s
    readiness probes and CI install gates rely on.
    """
    from switchyard.server.verify import verify_proxy

    api_key, base_url, key_source = _resolve_verify_credentials(args)
    raise SystemExit(verify_proxy(
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        port=args.port,
        timeout=args.timeout,
        key_source=key_source,
    ))


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------


def _switchyard_version() -> str:
    """Resolve the installed Switchyard version for ``--version``.

    Reads the version from the installed package metadata (ultimately the
    ``version`` field in ``pyproject.toml``), matching the runtime lookup in
    ``intake_payload_builder``. Falls back to the static ``__version__``
    constant when the package is not installed (e.g. running from a source
    checkout that was never built), which the release tooling keeps in sync
    with ``pyproject.toml``.
    """
    try:
        return version("nemo-switchyard")
    except PackageNotFoundError:
        from switchyard import __version__

        return __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="switchyard",
        description="Switchyard — chain-based LLM proxy server CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_switchyard_version()}",
        help="Print the installed Switchyard version and exit.",
    )
    parser.add_argument(
        "--routing-profiles", "-c",
        dest="routing_profiles",
        default=None,
        metavar="PATH",
        help=(
            "Deprecated path to a routing-profiles YAML file. Applies to "
            "serve, launch, and configure (saves it as the default). Separate "
            "from the subcommand with -- for clarity: "
            "switchyard --routing-profiles dev.yaml -- launch claude"
        ),
    )
    parser.add_argument(
        "--enable-rl-logging",
        dest="enable_rl_logging",
        action="store_true",
        help=(
            "Write per-turn RL training traces (message_history JSON, one "
            "file per request/response pair) for `launch` and `serve` sessions. "
            "When the route bundle declares a `token_capture_engine` target, "
            "traces are token-level capture records grouped by session and "
            "served via GET /v1/sessions. Global flag — place it before the "
            "subcommand, e.g. switchyard --enable-rl-logging launch claude."
        ),
    )
    parser.add_argument(
        "--rl-log-dir",
        dest="rl_log_dir",
        default="./rl_data",
        metavar="DIR",
        help=(
            "Directory for RL trace logs (default: ./rl_data). Only takes "
            "effect with --enable-rl-logging."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Server mode")

    # -- serve --
    serve = subparsers.add_parser(
        "serve",
        help="Serve a v2 profile config (--config)",
        description=(
            "Serve a Switchyard v2 profile config via serve --config: one "
            "YAML/JSON/TOML file of endpoints, targets, and profiles; each "
            "profile id and target id is exposed on GET /v1/models."
        ),
    )
    add_transport_args(serve)
    _add_intake_args(serve)
    serve.add_argument(
        "--config",
        dest="config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a Switchyard v2 profile config (YAML, JSON, or TOML). "
            "Files containing only Rust-defined profiles use the Rust profile "
            "server; files with Python-defined profiles use the Python FastAPI "
            "adapter."
        ),
    )
    serve.add_argument(
        "--workers", "-w", type=int,
        default=int(os.environ.get("SWITCHYARD_WORKERS", "1")),
        help="Number of uvicorn worker processes (default: 1, or SWITCHYARD_WORKERS env var)",
    )
    serve.set_defaults(func=_cmd_serve)

    # -- configure --
    cfg = subparsers.add_parser(
        "configure",
        help="Store default provider credentials and launch models",
        description=(
            "Persist user-level Switchyard defaults under "
            "~/.config/switchyard. Credentials are stored separately "
            "from non-secret config with owner-only file permissions."
        ),
    )
    cfg_mode = cfg.add_mutually_exclusive_group()
    cfg_mode.add_argument(
        "--show", action="store_true",
        help=(
            "Print the redacted user config plus resolved provider, API-key "
            "source, and harness binary paths. Pair with --check for a live "
            "GET /models probe."
        ),
    )
    cfg_mode.add_argument(
        "--reset", action="store_true",
        help="Delete persisted user config and credentials",
    )
    cfg_mode.add_argument(
        "--list-models", action="store_true", dest="list_models",
        help=(
            "Fetch GET /models from the resolved provider and print a "
            "ranked, searchable list. Combine with --target / --query / "
            "--limit."
        ),
    )
    cfg.add_argument(
        "--json", action="store_true",
        help="With --show, print the raw redacted JSON snapshot",
    )
    cfg.add_argument(
        "--target", type=str, default="all",
        choices=("all", "provider", "claude", "codex", "openclaw"),
        help=(
            "Which defaults to configure: provider credentials only, "
            "Claude Code, Codex, OpenClaw, or all (default: all)"
        ),
    )
    cfg.add_argument(
        "--provider", type=str, default=DEFAULT_PROVIDER,
        help=f"Provider id to configure (default: {DEFAULT_PROVIDER})",
    )
    cfg.add_argument(
        "--base-url", type=str, default=None,
        help=(
            "Backend base URL to save (default: "
            f"{_DEFAULT_OPENROUTER_BASE_URL})"
        ),
    )
    cfg.add_argument(
        "--api-key", type=str, default=None,
        help="API key to save. Omit in an interactive terminal to be prompted.",
    )
    cfg.add_argument(
        "--claude-model", type=str, default=None,
        help="Claude Code default model. Omit to select from GET /models.",
    )
    cfg.add_argument(
        "--claude-base-url", type=str, default=None,
        help="Claude Code model endpoint override. Omit to use the default base URL.",
    )
    cfg.add_argument(
        "--claude-api-key", type=str, default=None,
        help="Claude Code model API-key override. Omit to use the default API key.",
    )
    cfg.add_argument(
        "--codex-model", type=str, default=None,
        help="Codex default model. Omit to select from GET /models.",
    )
    cfg.add_argument(
        "--codex-base-url", type=str, default=None,
        help="Codex model endpoint override. Omit to use the default base URL.",
    )
    cfg.add_argument(
        "--codex-api-key", type=str, default=None,
        help="Codex model API-key override. Omit to use the default API key.",
    )
    cfg.add_argument(
        "--openclaw-model", type=str, default=None,
        help="OpenClaw default model. Omit to select from GET /models.",
    )
    cfg.add_argument(
        "--openclaw-base-url", type=str, default=None,
        help="OpenClaw model endpoint override. Omit to use the default base URL.",
    )
    cfg.add_argument(
        "--openclaw-api-key", type=str, default=None,
        help="OpenClaw model API-key override. Omit to use the default API key.",
    )
    cfg_mode.add_argument(
        "--skill-distillation",
        metavar="NAMESPACE",
        default=None,
        help=(
            "Save a namespace for one skill that improves over time. Many "
            "sessions or trajectories can contribute to it; the namespace is "
            "not a session ID. Use a safe local path component (for example: "
            "tooluniverse-trialqa)."
        ),
    )
    cfg_mode.add_argument(
        "--disable-skill-distillation",
        action="store_true",
        help="Remove saved skill distillation config.",
    )
    cfg.add_argument(
        "--no-model-discovery", action="store_true",
        help="Skip GET /models and rely on explicit or existing model values",
    )
    cfg.add_argument(
        "--no-tui", action="store_true",
        help="Use plain text prompts instead of the interactive terminal selector",
    )
    cfg.add_argument(
        "--query", "-q", type=str, default=None,
        help=(
            "With --list-models, case-insensitive substring filter applied "
            "to the result set."
        ),
    )
    cfg.add_argument(
        "--limit", type=int, default=50,
        help=(
            "With --list-models, cap on the number of models printed "
            "(default: 50; pass 0 for unlimited)."
        ),
    )
    cfg.add_argument(
        "--check", action="store_true",
        help=(
            "With --show, also call GET /models on the resolved provider "
            "and report pass/fail."
        ),
    )
    cfg.set_defaults(func=_cmd_configure)

    # -- launch --
    launch = subparsers.add_parser(
        "launch",
        help="Start proxy + spawn a CLI tool in one command (ollama-style UX)",
        description=(
            "Start a proxy and spawn a CLI tool pointed at it in the same "
            "terminal, tear it down when the tool exits."
        ),
    )
    launch_sub = launch.add_subparsers(dest="launch_target", help="Tool to launch")

    # -- launch claude --
    lc = launch_sub.add_parser(
        "claude",
        help="Launch Claude Code with a proxy (one terminal, one command)",
        description=(
            "Starts a proxy on an auto-picked free local port, then "
            "spawns `claude` with ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, "
            "and ANTHROPIC_MODEL preset.  Proxy shuts down when claude exits.\n\n"
            "Route selection (mutually exclusive — pick one or neither):\n"
            "  --model X               single-model passthrough — every request "
            "is rewritten to model=X. Falls back to the saved configure default.\n"
            "  --routing-profiles PATH serve a YAML bundle of routes (random, "
            "stage_router, plan_execute, passthrough, …); the first declared route "
            "is the initial model.\n\n"
            "With neither flag, the saved routing bundle from "
            "`switchyard configure` is used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    lc.add_argument(
        "--model", type=str, default=None,
        help=(
            "Single-model passthrough: model id from GET /v1/models "
            "(e.g. openai/gpt-5.2). Falls back to the saved "
            "`configure` default when omitted. Mutually exclusive with "
            "--routing-profiles (pass that as a global switchyard flag before "
            "the subcommand)."
        ),
    )
    lc.add_argument(
        "--base-url", type=str, default=None,
        help=(
            "Backend base URL (default: "
            f"{_DEFAULT_OPENROUTER_BASE_URL}, or OPENROUTER_BASE_URL / "
            "NVIDIA_BASE_URL / "
            "OPENAI_BASE_URL env var)."
        ),
    )
    lc.add_argument(
        "--api-key", type=str, default=None,
        help=(
            "API key (falls back to OPENROUTER_API_KEY, NVIDIA_API_KEY, OPENAI_API_KEY, "
            "user config, secrets.json)"
        ),
    )
    lc.add_argument(
        "--no-model-discovery", action="store_true",
        help="First-run setup: skip GET /models and type the model manually",
    )
    lc.add_argument(
        "--no-tui", action="store_true",
        help="First-run setup: use plain text prompts instead of the TUI selector",
    )
    lc.add_argument(
        "--reconfigure", action="store_true",
        help="Run Claude setup before launching, even if defaults already exist",
    )
    lc.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved launch settings without starting the proxy or Claude",
    )
    lc.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke-test mode: start the proxy, run one "
            "`claude -p \"<smoke>\" --max-turns 1` round-trip, assert exit 0, "
            "and exit. Replaces the old `switchyard verify claude` subcommand. "
            "Requires --model; cannot be combined with --routing-profiles or --dry-run "
            "(use --model directly to pick the model to smoke-test)."
        ),
    )
    lc.add_argument(
        "--port", type=int, default=None,
        help="Proxy port (default: auto-pick free port)",
    )
    lc.add_argument(
        "--timeout", type=float, default=None,
        help="Request timeout in seconds for the backend LLM client",
    )
    _add_launch_deterministic_override_args(lc)
    _add_launch_intake_args(lc)
    lc.add_argument(
        "claude_args", nargs=argparse.REMAINDER,
        help="Args forwarded to claude (prefix with `--`, e.g. `-- --version`)",
    )
    lc.set_defaults(func=_cmd_launch_claude)

    # -- launch codex --
    cx = launch_sub.add_parser(
        "codex",
        help="Launch OpenAI Codex CLI with a proxy (one terminal, one command)",
        description=(
            "Starts a proxy on an auto-picked free local port, then "
            "spawns `codex` with a transient `switchyard` provider injected "
            "via repeated `-c` flags (no edits to ~/.codex/config.toml).\n\n"
            "Codex talks to the proxy via OpenAI Responses API "
            "(/v1/responses); the chain translates that to OpenAI Chat "
            "Completions for the upstream backend.  Proxy shuts down when "
            "codex exits.\n\n"
            "Route selection (mutually exclusive — pick one or neither):\n"
            "  --model X               single-model passthrough — every request "
            "is rewritten to model=X. Falls back to the saved configure default.\n"
            "  --routing-profiles PATH serve a YAML bundle of routes (random, "
            "stage_router, plan_execute, passthrough, …); the first declared route "
            "is the initial model.\n\n"
            "With neither flag, the saved routing bundle from "
            "`switchyard configure` is used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cx.add_argument(
        "--model", type=str, default=None,
        help=(
            "Single-model passthrough: model id from GET /v1/models "
            "(e.g. openai/gpt-5.2). Falls back to the saved "
            "`configure` default when omitted. Mutually exclusive with "
            "--routing-profiles (pass that as a global switchyard flag before "
            "the subcommand)."
        ),
    )
    cx.add_argument(
        "--base-url", type=str, default=None,
        help=(
            "Backend base URL (default: "
            f"{_DEFAULT_OPENROUTER_BASE_URL}, or OPENROUTER_BASE_URL / "
            "NVIDIA_BASE_URL / "
            "OPENAI_BASE_URL env var)."
        ),
    )
    cx.add_argument(
        "--api-key", type=str, default=None,
        help=(
            "API key (falls back to OPENROUTER_API_KEY, NVIDIA_API_KEY, OPENAI_API_KEY, "
            "user config, secrets.json)"
        ),
    )
    cx.add_argument(
        "--no-model-discovery", action="store_true",
        help="First-run setup: skip GET /models and type the model manually",
    )
    cx.add_argument(
        "--no-tui", action="store_true",
        help="First-run setup: use plain text prompts instead of the TUI selector",
    )
    cx.add_argument(
        "--reconfigure", action="store_true",
        help="Run Codex setup before launching, even if defaults already exist",
    )
    cx.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved launch settings without starting the proxy or Codex",
    )
    cx.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke-test mode: start the proxy, run one "
            "`codex exec \"<smoke>\"` round-trip, assert exit 0, and exit. "
            "Replaces the old `switchyard verify codex` subcommand. "
            "Requires --model; cannot be combined with --routing-profiles or --dry-run "
            "(use --model directly to pick the model to smoke-test)."
        ),
    )
    cx.add_argument(
        "--port", type=int, default=None,
        help="Proxy port (default: auto-pick free port)",
    )
    cx.add_argument(
        "--timeout", type=float, default=None,
        help="Request timeout in seconds for the backend LLM client",
    )
    _add_launch_deterministic_override_args(cx)
    _add_launch_intake_args(cx)
    cx.add_argument(
        "codex_args", nargs=argparse.REMAINDER,
        help="Args forwarded to codex (prefix with `--`, e.g. `-- exec \"hi\"`)",
    )
    cx.set_defaults(func=_cmd_launch_codex)

    # -- launch openclaw --
    ow = launch_sub.add_parser(
        "openclaw",
        help="Launch OpenClaw with a proxy (one terminal, one command)",
        description=(
            "Starts a proxy on an auto-picked free local port, then "
            "spawns `openclaw chat` (alias for `openclaw tui --local`) "
            "against a transient OpenClaw "
            "workspace (OPENCLAW_STATE_DIR / OPENCLAW_HOME / "
            "OPENCLAW_CONFIG_PATH point at a tempdir) — the user's "
            "real ~/.openclaw/ (sessions, channels, plugins) is "
            "untouched.\n\n"
            "OpenClaw talks to the proxy via OpenAI Chat Completions "
            "(/v1/chat/completions); the proxy translates as needed for "
            "the upstream backend. Proxy and the transient workspace "
            "are torn down when openclaw exits.\n\n"
            "Route selection (mutually exclusive — pick one or neither):\n"
            "  --model X               single-model passthrough — every request "
            "is rewritten to model=X. Falls back to the saved configure default.\n"
            "  --routing-profiles PATH serve a YAML bundle of routes (random, "
            "stage_router, plan_execute, passthrough, …); the first declared route "
            "is the initial model.\n\n"
            "With neither flag, the saved routing bundle from "
            "`switchyard configure` is used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ow.add_argument(
        "--model", type=str, default=None,
        help=(
            "Single-model passthrough: model id from GET /v1/models "
            "(e.g. openai/gpt-5.2). Falls back to the saved "
            "`configure` default when omitted. Mutually exclusive with "
            "--routing-profiles (pass that as a global switchyard flag before "
            "the subcommand)."
        ),
    )
    ow.add_argument(
        "--base-url", type=str, default=None,
        help=(
            "Backend base URL (default: "
            f"{_DEFAULT_OPENROUTER_BASE_URL}, or OPENROUTER_BASE_URL / "
            "NVIDIA_BASE_URL / "
            "OPENAI_BASE_URL env var)."
        ),
    )
    ow.add_argument(
        "--api-key", type=str, default=None,
        help=(
            "API key (falls back to OPENROUTER_API_KEY, NVIDIA_API_KEY, OPENAI_API_KEY, "
            "user config, secrets.json)"
        ),
    )
    ow.add_argument(
        "--no-model-discovery", action="store_true",
        help="First-run setup: skip GET /models and type the model manually",
    )
    ow.add_argument(
        "--no-tui", action="store_true",
        help="First-run setup: use plain text prompts instead of the TUI selector",
    )
    ow.add_argument(
        "--reconfigure", action="store_true",
        help="Run OpenClaw setup before launching, even if defaults already exist",
    )
    ow.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved launch settings without starting the proxy or OpenClaw",
    )
    ow.add_argument(
        "--port", type=int, default=None,
        help="Proxy port (default: auto-pick free port)",
    )
    ow.add_argument(
        "--timeout", type=float, default=None,
        help="Request timeout in seconds for the backend LLM client",
    )
    ow.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke-test mode: start the proxy, run one openclaw round-trip "
            "(non-interactive, JSON envelope), assert exit 0, and exit. "
            "Replaces the old `switchyard verify openclaw` subcommand. "
            "Requires --model; cannot be combined with --routing-profiles or --dry-run "
            "(use --model directly to pick the model to smoke-test)."
        ),
    )
    _add_launch_deterministic_override_args(ow)
    _add_launch_intake_args(ow)
    ow.add_argument(
        "openclaw_args", nargs=argparse.REMAINDER,
        help=(
            "Args forwarded to `openclaw chat` "
            "(prefix with `--`, e.g. `-- --message 'hi' --thinking high`)"
        ),
    )
    ow.set_defaults(func=_cmd_launch_openclaw)

    def _launch_help(args: argparse.Namespace) -> None:  # noqa: ARG001
        launch.print_help()
        raise SystemExit(1)
    launch.set_defaults(func=_launch_help)

    # -- verify --
    verify = subparsers.add_parser(
        "verify",
        help="Run the proxy + backend e2e checklist (smoke test for users)",
        description=(
            "Run the proxy-only checklist:\n"
            "  1. Resolve credentials\n"
            "  2. Reach backend (GET /models)\n"
            "  3. Probe /v1/messages support (informational)\n"
            "  4. Start proxy on a free port\n"
            "  5. Round-trip a chat completion through the chain\n"
            "  6. Tear down proxy\n\n"
            "Fast (~1-3s on a healthy stack), no extra dependencies. "
            "Suitable for K8s readiness probes and pre-deployment "
            "smoke tests.  For harness-driven smoke tests (spawn "
            "claude/codex/openclaw against the proxy), use "
            "`launch {claude,codex,openclaw} --smoke`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    verify.add_argument(
        "--model", type=str, default=_DEFAULT_VERIFY_MODEL,
        help=(
            "Model to verify against the backend (default: "
            f"{_DEFAULT_VERIFY_MODEL}, matches the e2e test default)."
        ),
    )
    verify.add_argument(
        "--base-url", type=str, default=None,
        help=(
            "Backend base URL (default: "
            f"{_DEFAULT_OPENROUTER_BASE_URL}, or OPENROUTER_BASE_URL / "
            "NVIDIA_BASE_URL / OPENAI_BASE_URL env var)."
        ),
    )
    verify.add_argument(
        "--api-key", type=str, default=None,
        help=(
            "API key (falls back to OPENROUTER_API_KEY, NVIDIA_API_KEY, "
            "OPENAI_API_KEY, secrets.json). No 'dummy' placeholder fallback "
            "— verify fails fast on a missing key by design."
        ),
    )
    verify.add_argument(
        "--port", type=int, default=None,
        help="Proxy port (default: auto-pick free port)",
    )
    verify.add_argument(
        "--timeout", type=float, default=None,
        help="Request timeout in seconds for the backend LLM client",
    )
    verify.set_defaults(func=_cmd_verify)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Switchyard server CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _quiet_dependency_loggers()

    parser = _build_parser()
    # Strip the first '--' separator, allowing the canonical form:
    #   switchyard --routing-profiles dev.yaml -- launch claude
    # The '--' is purely visual — argparse doesn't need it.
    argv = list(sys.argv[1:])
    try:
        argv.pop(argv.index("--"))
    except ValueError:
        pass
    args = parser.parse_args(argv)

    if args.routing_profiles is not None:
        _warn_deprecated_routing_profiles()

    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(1)

    try:
        args.func(args)
    except RouteBundleConfigError as exc:
        # Route-bundle misconfiguration is normal user error, not a crash:
        # surface the dedicated message as a one-line CLI diagnostic with a
        # non-zero exit instead of letting it propagate as a raw traceback.
        raise SystemExit(f"error: invalid route bundle: {exc}") from exc
    except UserConfigError as exc:
        raise SystemExit(f"error: invalid user config: {exc}") from exc


if __name__ == "__main__":
    main()
