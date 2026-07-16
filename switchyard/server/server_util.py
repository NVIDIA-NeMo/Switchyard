# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI helpers for ``switchyard.cli.switchyard_cli``.

Provides:

- secrets-file loading
- common argparse argument surface
- credential resolution from CLI / env
- :func:`build_and_serve`, which wraps a :class:`~switchyard.lib.switchyard.Switchyard`
  in a FastAPI app via :func:`~switchyard.server.switchyard_app.build_switchyard_app`
  and starts uvicorn
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from switchyard.lib.endpoints.base import Endpoint as NemoSwitchyardEndpoint
from switchyard.lib.switchyard import Switchyard

if TYPE_CHECKING:
    from switchyard.lib.route_table import RouteTable


class InboundFormat(Enum):
    """Inbound wire format accepted by the proxy."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BOTH = "both"


logger = logging.getLogger(__name__)

# Four parents up: foundation/server/server_util.py → foundation/server →
# foundation → switchyard → repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SECRETS_FILE = REPO_ROOT / "secrets" / "secrets.json"


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


# secrets.json is a flat map of section name → section body.  Each section
# body is itself a dict (provider creds or server config).  Values are
# typed ``object`` so the narrow ``isinstance(section, dict)`` checks below
# stay honest — the alternative ``Any`` would silently absorb type errors.
SecretsFile = Mapping[str, Mapping[str, object]]


def load_secrets(secrets_file: Path | None = None) -> SecretsFile:
    """Load secrets from ``secrets.json``, returning ``{}`` if not found."""
    path = secrets_file or DEFAULT_SECRETS_FILE
    if path.exists():
        with open(path) as f:
            loaded: SecretsFile = json.load(f)
            return loaded
    return {}


# ---------------------------------------------------------------------------
# Argparse surfaces
# ---------------------------------------------------------------------------


def add_transport_args(parser: argparse.ArgumentParser) -> None:
    """Register transport-layer arguments: ``--host``, ``--port``, ``--inbound``, ``--reload``.

    Subcommands that define their own credential args with non-standard
    names (e.g. ``--api-base`` instead of ``--base-url``) use this
    helper to avoid the full :func:`add_common_args` surface.
    """
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--port", "-p", type=int, default=None,
        help="Port to bind to (default: 4000, or server.port from secrets.json)",
    )
    parser.add_argument(
        "--inbound",
        type=str,
        default=None,
        choices=[f.value for f in InboundFormat],
        help="Inbound API format: openai, anthropic, or both",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_model: bool = True,
) -> None:
    """Register arguments shared by every subcommand.

    Adds the transport args (:func:`add_transport_args`) plus backend
    credentials (``--api-key`` / ``--base-url``), the optional model
    override (``--model``), and uvicorn worker count (``--workers``).

    Args:
        parser: Target parser (subcommand parser, typically).
        include_model: Whether to expose ``--model`` for backend model
            override.  Subcommands that don't yet support a model-override
            request processor should pass ``False``.
    """
    add_transport_args(parser)
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for the backend LLM (falls back to env vars / secrets.json)",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="Base URL for the backend LLM API",
    )
    if include_model:
        parser.add_argument(
            "--model", type=str, default=None,
            help="Model name override (replaces model from incoming requests)",
        )
    parser.add_argument(
        "--workers", "-w", type=int,
        default=int(os.environ.get("SWITCHYARD_WORKERS", "1")),
        help="Number of uvicorn worker processes (default: 1, or SWITCHYARD_WORKERS env var)",
    )


def resolve_rl_log_dir(args: argparse.Namespace) -> Path | None:
    """Resolve the RL trace-log directory from the global rl-logging flags.

    Returns ``None`` unless the global ``--enable-rl-logging`` flag is set, in
    which case the directory is ``--rl-log-dir`` (default ``./rl_data``). Shared
    by the ``launch`` and ``serve`` entry points.
    """
    if not getattr(args, "enable_rl_logging", False):
        return None
    return Path(getattr(args, "rl_log_dir", None) or "./rl_data").expanduser()


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def resolve_credentials_from_env(
    args: argparse.Namespace,
    *,
    check_anthropic: bool = False,
) -> tuple[str | None, str | None]:
    """Return ``(api_key, base_url)`` resolved from ``args`` + env vars.

    Resolution order for *api_key*:
        1. ``args.api_key``
        2. ``ANTHROPIC_API_KEY`` (only when ``check_anthropic=True``)
        3. ``OPENAI_API_KEY``

    Resolution order for *base_url*:
        1. ``args.base_url``
        2. ``OPENAI_BASE_URL``
        3. ``OPENAI_API_BASE``

    Does not read ``secrets.json`` — use :func:`resolve_config_with_secrets`
    when secrets-file fallback is also needed.
    """
    if check_anthropic:
        api_key = (
            args.api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = (
        args.base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
    )
    return api_key, base_url


def resolve_config_with_secrets(
    args: argparse.Namespace,
    *,
    api_key_env_vars: tuple[str, ...] = ("OPENAI_API_KEY",),
    base_url_env_vars: tuple[str, ...] = (),
    base_url_arg: str = "api_base",
    secrets_section_priority: tuple[str, ...] = (),
) -> tuple[str | None, str | None]:
    """Resolve ``(api_key, base_url)`` from CLI → env → ``secrets.json``.

    Also mutates ``args.port`` in place when the CLI didn't set it and
    ``secrets.json`` has a ``server.port`` entry.

    Resolution order:
        * api_key:  ``args.api_key`` → *api_key_env_vars* (in order) →
          secrets sections (``secrets_section_priority`` in order, then
          first provider as a final fallback)
        * base_url: ``getattr(args, base_url_arg)`` → *base_url_env_vars*
          (in order) → same secrets-section traversal as api_key
        * port:     ``args.port`` (if set) else ``secrets["server"]["port"]``

    The secrets-section traversal mirrors the pattern used by the
    legacy subcommands: a priority-list of sections (e.g.
    ``("nvidia",)``) is checked first, and if none of them yielded an
    api_key we fall all the way back to the first provider entry —
    whichever section happens to come first in ``secrets.json``.

    Args:
        args: Parsed CLI namespace.  Must have ``api_key`` and
            ``port`` attributes; ``base_url_arg`` must also resolve.
        api_key_env_vars: Env var names checked in order for the
            api_key fallback.
        base_url_env_vars: Env var names checked in order for the
            base_url fallback.  Empty means "CLI + secrets only".
        base_url_arg: Attribute name on ``args`` that holds the CLI
            base URL — typically ``"api_base"`` (for ``--api-base``)
            or ``"base_url"`` (for ``--base-url``).
        secrets_section_priority: Top-level section names in
            ``secrets.json`` to consult first (e.g. ``("nvidia",)``).
            Empty defaults to "first provider only".

    Returns:
        ``(api_key, base_url)`` — either element may be ``None``.
    """
    secrets = load_secrets()
    secrets_api_key: str | None = None
    secrets_base_url: str | None = None
    secrets_port: int | None = None

    if secrets:
        for section_name in secrets_section_priority:
            section = secrets.get(section_name, {})
            if isinstance(section, dict):
                api_key_value = section.get("api_key")
                if secrets_api_key is None and isinstance(api_key_value, str):
                    secrets_api_key = api_key_value
                base_url_value = section.get("base_url")
                if secrets_base_url is None and isinstance(base_url_value, str):
                    secrets_base_url = base_url_value

        # Mirror the legacy behavior: fall back to the first provider
        # section only when the priority sections didn't yield an api_key.
        if not secrets_api_key:
            first_provider: Mapping[str, object] = next(iter(secrets.values()), {})
            if isinstance(first_provider, dict):
                api_key_value = first_provider.get("api_key")
                if secrets_api_key is None and isinstance(api_key_value, str):
                    secrets_api_key = api_key_value
                base_url_value = first_provider.get("base_url")
                if secrets_base_url is None and isinstance(base_url_value, str):
                    secrets_base_url = base_url_value

        server_section = secrets.get("server", {})
        if isinstance(server_section, dict):
            port_value = server_section.get("port")
            if isinstance(port_value, (int, str)):
                secrets_port = int(port_value)

    # api_key: CLI > env vars (in order) > secrets
    api_key: str | None = args.api_key
    for env_var in api_key_env_vars:
        if api_key:
            break
        api_key = os.environ.get(env_var)
    api_key = api_key or secrets_api_key

    # base_url: CLI > env vars (in order) > secrets
    base_url: str | None = getattr(args, base_url_arg, None)
    for env_var in base_url_env_vars:
        if base_url:
            break
        base_url = os.environ.get(env_var)
    base_url = base_url or secrets_base_url

    if args.port is None and secrets_port is not None:
        args.port = secrets_port

    return api_key, base_url


def ensure_openai_api_key_env(api_key: str | None) -> None:
    """Set ``OPENAI_API_KEY`` env var iff unset and *api_key* is provided.

    Some downstream libraries read
    ``OPENAI_API_KEY`` directly from the environment, so subcommands
    that resolve credentials from CLI / secrets.json pin them into the
    environment for those libraries to see.
    """
    if api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key


def resolve_port(default: int = 4000) -> int:
    """Resolve the server port from ``secrets.json`` with a fixed fallback."""
    server_section = load_secrets().get("server", {})
    if isinstance(server_section, Mapping):
        value = server_section.get("port")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                logger.warning("Ignoring non-integer secrets.json server.port=%r", value)
    return default


# ---------------------------------------------------------------------------
# Build + serve
# ---------------------------------------------------------------------------


def build_and_serve(
    args: argparse.Namespace,
    switchyard: Switchyard | RouteTable,
    *,
    inbound_default: str = "openai",
    disable_backend_streaming: bool = False,
    extra_endpoints: list[NemoSwitchyardEndpoint] | None = None,
    strategy_summary: str | None = None,
) -> None:
    """Wire a Switchyard runtime object into a FastAPI app and serve it.

    Builds the app via :func:`~switchyard.server.switchyard_app.build_switchyard_app`
    (which registers all three inbound formats — OpenAI Chat, Anthropic Messages, and
    OpenAI Responses API), optionally appends *extra_endpoints*, then starts uvicorn.

    Expected attributes on *args*:
        * ``host`` (str), ``port`` (int | None), ``inbound`` (str | None),
          ``reload`` (bool), ``workers`` (int, optional; defaults to 1).

    Args:
        args: Parsed CLI namespace.
        switchyard: Already-built chain or table to serve.
        inbound_default: Unused — always registers all inbound formats.
        disable_backend_streaming: Unused — kept for signature compatibility.
        extra_endpoints: Additional endpoint modules to register after the defaults.
    """
    del inbound_default, disable_backend_streaming

    import threading

    import uvicorn

    from switchyard.server.switchyard_app import build_switchyard_app

    app = build_switchyard_app(switchyard)
    if extra_endpoints:
        for endpoint in extra_endpoints:
            endpoint.register(app)

    port = args.port if isinstance(args.port, int) else resolve_port()

    def _print_banner() -> None:
        from switchyard.cli.launchers.launcher_runtime import (
            print_ready_banner,
            wait_for_proxy_ready,
        )
        from switchyard.lib.route_table import RouteTable as _RT
        if not wait_for_proxy_ready(port, timeout_s=15.0):
            return
        table = switchyard if isinstance(switchyard, _RT) else None
        default_model = table.default_model() if table else None
        print_ready_banner(
            port=port,
            display_model=default_model or "switchyard",
            strategy_summary=strategy_summary,
            profile_routes=table.registered_models() if table else None,
            default_route=default_model,
        )

    threading.Thread(target=_print_banner, daemon=True).start()

    workers = getattr(args, "workers", 1)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        reload=args.reload,
        workers=workers,
    )
