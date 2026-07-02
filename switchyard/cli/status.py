# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Status view for launcher configuration and local tool readiness."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from switchyard.cli.config.user_config import (
    DEFAULT_PROVIDER,
    DEFAULT_SECRETS_SECTION_PRIORITY,
    LaunchTarget,
    SkillDistillationConfig,
    load_user_config,
    load_user_credentials,
    resolve_provider_connectivity,
)
from switchyard.cli.launchers.claude_code_launcher import (
    _find_claude_binary,
)
from switchyard.cli.launchers.codex_cli_launcher import (
    _find_codex_binary,
)
from switchyard.cli.model_catalog.model_discovery import (
    ModelDiscoveryError,
    fetch_model_ids,
)
from switchyard.cli.output import format_route_config
from switchyard.lib.skill_distillation_store import summarize_skill_distillation_store

_API_KEY_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)
_BASE_URL_ENV_VARS = ("OPENROUTER_BASE_URL", "NVIDIA_BASE_URL", "OPENAI_BASE_URL")


@dataclass(frozen=True)
class StatusRequest:
    """Inputs for rendering ``switchyard status``."""

    cli_api_key: str | None = None
    cli_base_url: str | None = None
    provider: str | None = None
    check: bool = False
    secrets: Mapping[str, Mapping[str, object]] | None = None


def _secret_value(section: Mapping[str, object], key: str) -> str | None:
    value = section.get(key)
    return value if isinstance(value, str) and value else None


def _secrets_source(
    secrets: Mapping[str, Mapping[str, object]] | None,
    key: str,
) -> str | None:
    if not secrets:
        return None
    priority = DEFAULT_SECRETS_SECTION_PRIORITY
    for section_name in priority:
        section = secrets.get(section_name, {})
        if _secret_value(section, key):
            return f"secrets.json[{section_name}.{key}]"
    for section_name, section in secrets.items():
        if _secret_value(section, key):
            return f"secrets.json[{section_name}.{key}]"
    return None


def _api_key_source(
    *,
    cli_api_key: str | None,
    provider: str,
    secrets: Mapping[str, Mapping[str, object]] | None,
) -> str:
    if cli_api_key:
        return "--api-key"
    for env_var in _API_KEY_ENV_VARS:
        if os.environ.get(env_var):
            return f"${env_var}"
    credentials = load_user_credentials()
    if credentials.api_key(provider):
        return f"{provider} credentials"
    return _secrets_source(secrets, "api_key") or "<missing>"


def _first_set_env_var(env_vars: tuple[str, ...]) -> str | None:
    for env_var in env_vars:
        if os.environ.get(env_var):
            return env_var
    return None


def _base_url_env_var_for(prefix: str | None) -> str | None:
    if not prefix:
        return None
    env_var = f"{prefix}_BASE_URL"
    if env_var in _BASE_URL_ENV_VARS and os.environ.get(env_var):
        return env_var
    return None


def _base_url_source(
    *,
    cli_base_url: str | None,
    provider: str,
    secrets: Mapping[str, Mapping[str, object]] | None,
) -> str:
    if cli_base_url:
        return "--base-url"
    api_key_env_var = _first_set_env_var(_API_KEY_ENV_VARS)
    api_key_prefix = (
        api_key_env_var.removesuffix("_API_KEY")
        if api_key_env_var and api_key_env_var.endswith("_API_KEY")
        else None
    )
    env_var = _base_url_env_var_for(api_key_prefix)
    if env_var:
        return f"${env_var}"
    if not api_key_env_var:
        env_var = _base_url_env_var_for(provider.upper().replace("-", "_"))
        if env_var:
            return f"${env_var}"
    user_config = load_user_config()
    if user_config.provider(provider).base_url:
        return f"{provider} config"
    return _secrets_source(secrets, "base_url") or "built-in default"


def _format_binary(name: str, path: str | None) -> str:
    return f"{name}: {path or '<not found>'}"


def _format_launch_target(target: LaunchTarget) -> list[str]:
    launch_config = load_user_config().launch_target(target)
    route = launch_config.effective_route()
    lines = [f"{target}:"]
    if route.model:
        for route_line in format_route_config(route):
            lines.append(f"  {route_line}")
    else:
        lines.append("  <not configured>")
    return lines


def _format_skill_distillation(config: SkillDistillationConfig) -> str:
    if not config.configured:
        return "skill distillation: not configured"
    assert config.namespace is not None
    summary = summarize_skill_distillation_store(config.namespace)
    active = "present" if summary.active_skill_exists else "missing"
    return (
        f"skill distillation: configured; namespace: {config.namespace}; "
        "session capture: on; "
        f"store: {summary.path}; "
        f"sessions: {summary.session_count}; "
        f"active skill: {active}"
    )


def render_status(request: StatusRequest) -> str:
    """Return a human-readable status report."""

    user_config = load_user_config()
    provider = request.provider or user_config.default_provider or DEFAULT_PROVIDER
    connectivity = resolve_provider_connectivity(
        cli_api_key=request.cli_api_key,
        cli_base_url=request.cli_base_url,
        api_key_env_vars=_API_KEY_ENV_VARS,
        base_url_env_vars=_BASE_URL_ENV_VARS,
        secrets=request.secrets,
        secrets_section_priority=DEFAULT_SECRETS_SECTION_PRIORITY,
        default_provider=provider,
    )
    lines = [
        "Switchyard status",
        f"provider: {connectivity.provider}",
        f"base URL: {connectivity.base_url}",
        f"base URL source: {_base_url_source(cli_base_url=request.cli_base_url, provider=connectivity.provider, secrets=request.secrets)}",
        f"API key source: {_api_key_source(cli_api_key=request.cli_api_key, provider=connectivity.provider, secrets=request.secrets)}",
    ]
    saved_bundle = user_config.routing_profiles
    if isinstance(saved_bundle, dict):
        routes = saved_bundle.get("routes")
        n = len(routes) if isinstance(routes, dict) else 0
        summary = f"<{n} route(s) saved>"
    else:
        summary = "<not configured>"
    lines.append(f"routing profiles: {summary}")
    # Surface built-in routing strategies that don't require a saved
    # bundle on `switchyard launch {claude,codex,openclaw}`.
    if not isinstance(saved_bundle, dict):
        lines.append(
            "built-in strategies: LLM-as-classifier routing (default, "
            "strong/weak); plan-execute (strong-planner + weak-executor) via a "
            "type: plan_execute route in a --routing-profiles bundle",
        )
    lines.append(_format_skill_distillation(user_config.skill_distillation))
    lines += [
        "",
        "Launch defaults",
        *_format_launch_target("claude"),
        *_format_launch_target("codex"),
        "",
        "Harness binaries",
        _format_binary("claude", _find_claude_binary()),
        _format_binary("codex", _find_codex_binary()),
    ]

    if request.check:
        lines.append("")
        lines.append("Provider check")
        if not connectivity.api_key:
            lines.append("models: skipped (missing API key)")
        else:
            try:
                model_ids = fetch_model_ids(connectivity.base_url, connectivity.api_key)
            except ModelDiscoveryError as exc:
                lines.append(f"models: failed ({exc})")
            else:
                lines.append(f"models: ok ({len(model_ids)} found)")
    return "\n".join(lines)


__all__ = ["StatusRequest", "render_status"]
