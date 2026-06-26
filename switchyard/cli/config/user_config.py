# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-level configuration for the launcher UX.

This module owns the persistent, cross-repo defaults used by
``switchyard configure`` and the one-command launchers.  It is
intentionally independent of argparse so tests and future non-CLI entry
points can reuse the same read/write and resolution behavior.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

DEFAULT_PROVIDER = "openrouter"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SECRETS_SECTION_PRIORITY = (DEFAULT_PROVIDER, "nvidia")
_PROVIDER_BASE_URL_DEFAULTS = {
    DEFAULT_PROVIDER: DEFAULT_OPENROUTER_BASE_URL,
    "nvidia": "https://inference-api.nvidia.com/v1",
    "openai": "https://api.openai.com/v1",
}
CONFIG_DIR_ENV_VAR = "SWITCHYARD_CONFIG_DIR"
CONFIG_FILENAME = "config.json"
CREDENTIALS_FILENAME = "credentials.json"

LaunchTarget = Literal["claude", "codex", "openclaw"]
LaunchRouteType = Literal["single", "random", "deterministic", "plan_execute"]
PRIMARY_TIER = "primary"
WEAK_TIER = "weak"

_SKILL_DISTILLATION_KEYS = frozenset({"namespace"})


class UserConfigError(RuntimeError):
    """Raised when a persisted user config file is malformed."""


@dataclass(frozen=True)
class ProviderConfig:
    """Non-secret provider defaults."""

    base_url: str | None = None


@dataclass(frozen=True)
class LaunchTierEndpointConfig:
    """Non-secret endpoint override for one launch route tier."""

    base_url: str | None = None


@dataclass(frozen=True)
class LaunchRouteConfig:
    """Per-harness route defaults.

    ``single`` is the no-routing path: every request is rewritten to
    ``model``.  ``random`` routes between ``model`` (strong/primary) and
    ``weak_model`` with ``strong_probability``.
    """

    type: LaunchRouteType = "single"
    model: str | None = None
    weak_model: str | None = None
    strong_probability: float = 0.5
    endpoints: dict[str, LaunchTierEndpointConfig] | None = None

    def endpoint(self, tier: str) -> LaunchTierEndpointConfig:
        """Return the endpoint override for *tier*, if configured."""

        return (self.endpoints or {}).get(tier, LaunchTierEndpointConfig())


@dataclass(frozen=True)
class LaunchConfig:
    """Per-launcher defaults."""

    model: str | None = None
    route: LaunchRouteConfig | None = None

    def effective_route(self) -> LaunchRouteConfig:
        """Return the saved route, or a single-model route from legacy fields."""

        if self.route:
            return LaunchRouteConfig(
                type=self.route.type,
                model=self.route.model or self.model,
                weak_model=self.route.weak_model,
                strong_probability=self.route.strong_probability,
                endpoints=self.route.endpoints,
            )
        return LaunchRouteConfig(
            type="single",
            model=self.model,
        )


@dataclass(frozen=True)
class SkillDistillationConfig:
    """User-level skill distillation defaults."""

    namespace: str | None = None

    def __post_init__(self) -> None:
        if self.namespace is not None:
            _validate_skill_namespace(self.namespace)

    @property
    def configured(self) -> bool:
        """Return whether skill distillation has a namespace to operate on."""

        return self.namespace is not None

    def is_default(self) -> bool:
        """Return whether this matches an unconfigured skill distillation config."""

        return self == SkillDistillationConfig()


@dataclass(frozen=True)
class UserConfig:
    """Non-secret Switchyard defaults stored under the user config dir.

    ``routing_profiles`` is a parsed routing-profile YAML bundle saved by
    ``switchyard configure --routing-profiles PATH``. Stored as a parsed
    JSON object inline (not as a path — paths rot when files move); env
    var references inside the bundle are preserved verbatim and
    re-expanded on each load. Consumed by ``switchyard serve`` (and as a
    fallback for ``switchyard launch claude/codex``) when no CLI
    ``--routing-profiles`` is passed.
    """

    default_provider: str = DEFAULT_PROVIDER
    providers: dict[str, ProviderConfig] | None = None
    launch: dict[LaunchTarget, LaunchConfig] | None = None
    routing_profiles: dict[str, object] | None = None
    skill_distillation: SkillDistillationConfig = field(
        default_factory=SkillDistillationConfig,
    )

    def provider(self, name: str | None = None) -> ProviderConfig:
        providers = self.providers or {}
        return providers.get(name or self.default_provider, ProviderConfig())

    def launch_target(self, target: LaunchTarget) -> LaunchConfig:
        launch = self.launch or {}
        return launch.get(target, LaunchConfig())


@dataclass(frozen=True)
class LaunchCredentials:
    """Secret per-launcher tier credentials."""

    api_keys: dict[str, str] | None = None

    def api_key(self, tier: str = PRIMARY_TIER) -> str | None:
        return (self.api_keys or {}).get(tier)


@dataclass(frozen=True)
class UserCredentials:
    """Secret provider credentials stored separately from config."""

    api_keys: dict[str, str] | None = None
    launch: dict[LaunchTarget, LaunchCredentials] | None = None

    def api_key(self, provider: str) -> str | None:
        return (self.api_keys or {}).get(provider)

    def launch_target(self, target: LaunchTarget) -> LaunchCredentials:
        launch = self.launch or {}
        return launch.get(target, LaunchCredentials())


@dataclass(frozen=True)
class ProviderConnectivity:
    """Resolved provider connectivity after applying precedence rules."""

    provider: str
    api_key: str | None
    base_url: str


def get_user_config_dir() -> Path:
    """Return the directory where Switchyard user config is stored."""

    override = os.environ.get(CONFIG_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home).expanduser() / "switchyard"

    return Path.home() / ".config" / "switchyard"


def get_config_path(config_dir: Path | None = None) -> Path:
    return (config_dir or get_user_config_dir()) / CONFIG_FILENAME


def get_credentials_path(config_dir: Path | None = None) -> Path:
    return (config_dir or get_user_config_dir()) / CREDENTIALS_FILENAME


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise UserConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise UserConfigError(f"{path} must contain a JSON object")
    return cast(dict[str, object], raw)


def _mapping_value(data: Mapping[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}


def _str_value(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) and value else None


def _validate_skill_namespace(namespace: str) -> None:
    if namespace != namespace.strip():
        raise UserConfigError(
            "skill_distillation.namespace must not have leading or trailing "
            "whitespace"
        )
    if namespace in {".", ".."}:
        raise UserConfigError(
            "skill_distillation.namespace must be a safe local path component"
        )
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not namespace or any(ch not in allowed for ch in namespace):
        raise UserConfigError(
            "skill_distillation.namespace may contain only letters, numbers, "
            "dot, underscore, and hyphen"
        )


def _float_value(
    data: Mapping[str, object], key: str, default: float,
) -> float:
    value = data.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _route_type_value(data: Mapping[str, object], key: str) -> LaunchRouteType | None:
    value = data.get(key)
    if value == "single":
        return "single"
    if value == "random":
        return "random"
    if value == "deterministic":
        return "deterministic"
    if value == "plan_execute":
        return "plan_execute"
    return None


def _parse_launch_route(raw_route: object) -> LaunchRouteConfig | None:
    if not isinstance(raw_route, dict):
        return None
    route_data = cast(dict[str, object], raw_route)
    route_type = _route_type_value(route_data, "type") or "single"
    return LaunchRouteConfig(
        type=route_type,
        model=_str_value(route_data, "model"),
        weak_model=_str_value(route_data, "weak_model"),
        strong_probability=_float_value(route_data, "strong_probability", 0.5),
        endpoints=_parse_launch_endpoints(route_data),
    )


def _parse_launch_endpoint(raw_endpoint: object) -> LaunchTierEndpointConfig | None:
    if not isinstance(raw_endpoint, dict):
        return None
    endpoint_data = cast(dict[str, object], raw_endpoint)
    base_url = _str_value(endpoint_data, "base_url")
    if not base_url:
        return None
    return LaunchTierEndpointConfig(base_url=base_url)


def _parse_launch_endpoints(
    route_data: Mapping[str, object],
) -> dict[str, LaunchTierEndpointConfig] | None:
    raw_endpoints = route_data.get("endpoints")
    endpoints_data = raw_endpoints if isinstance(raw_endpoints, dict) else {}
    endpoints: dict[str, LaunchTierEndpointConfig] = {}
    for tier, raw_endpoint in cast(dict[str, object], endpoints_data).items():
        endpoint = _parse_launch_endpoint(raw_endpoint)
        if endpoint:
            endpoints[tier] = endpoint

    # Transitional support for early local configs that used flat keys.
    primary_base_url = _str_value(route_data, "base_url")
    if primary_base_url:
        endpoints[PRIMARY_TIER] = LaunchTierEndpointConfig(base_url=primary_base_url)
    weak_base_url = _str_value(route_data, "weak_base_url")
    if weak_base_url:
        endpoints[WEAK_TIER] = LaunchTierEndpointConfig(base_url=weak_base_url)

    return endpoints or None


def _endpoint_to_json(endpoint: LaunchTierEndpointConfig) -> dict[str, object]:
    body: dict[str, object] = {}
    if endpoint.base_url:
        body["base_url"] = endpoint.base_url
    return body


def _route_to_json(route: LaunchRouteConfig) -> dict[str, object]:
    body: dict[str, object] = {"type": route.type}
    if route.model:
        body["model"] = route.model
    if route.weak_model:
        body["weak_model"] = route.weak_model
    if route.type == "random":
        body["strong_probability"] = route.strong_probability
    endpoints: dict[str, object] = {}
    for tier, endpoint in (route.endpoints or {}).items():
        endpoint_body = _endpoint_to_json(endpoint)
        if endpoint_body:
            endpoints[tier] = endpoint_body
    if endpoints:
        body["endpoints"] = endpoints
    return body


def _launch_config_to_json(target_config: LaunchConfig) -> dict[str, object]:
    body: dict[str, object] = {}
    route = target_config.effective_route()
    if route.model:
        body["model"] = route.model
    body["route"] = _route_to_json(route)
    return body


def _parse_skill_distillation_config(raw_config: object) -> SkillDistillationConfig:
    if raw_config is None:
        return SkillDistillationConfig()
    if not isinstance(raw_config, dict):
        raise UserConfigError("skill_distillation must contain a JSON object")
    config_data = cast(dict[str, object], raw_config)
    unsupported_keys = sorted(set(config_data) - _SKILL_DISTILLATION_KEYS)
    if unsupported_keys:
        raise UserConfigError(
            "skill_distillation supports only namespace; unsupported key(s): "
            f"{', '.join(unsupported_keys)}"
        )
    raw_namespace = config_data.get("namespace")
    if raw_namespace is None or raw_namespace == "":
        raise UserConfigError(
            "skill_distillation.namespace is required when skill distillation "
            "is configured"
        )
    if not isinstance(raw_namespace, str):
        raise UserConfigError(
            "skill_distillation.namespace must be a non-empty string"
        )
    return SkillDistillationConfig(namespace=raw_namespace)


def _skill_distillation_to_json(
    config: SkillDistillationConfig,
) -> dict[str, object]:
    body: dict[str, object] = {}
    if config.namespace:
        body["namespace"] = config.namespace
    return body


def load_user_config(config_dir: Path | None = None) -> UserConfig:
    """Load non-secret user config, returning defaults when absent."""

    data = _read_json_object(get_config_path(config_dir))
    providers_data = _mapping_value(data, "providers")
    providers: dict[str, ProviderConfig] = {}
    for name, raw_provider in providers_data.items():
        if not isinstance(raw_provider, dict):
            continue
        provider_data = cast(dict[str, object], raw_provider)
        providers[name] = ProviderConfig(base_url=_str_value(provider_data, "base_url"))

    launch_data = _mapping_value(data, "launch")
    launch: dict[LaunchTarget, LaunchConfig] = {}
    for target in ("claude", "codex", "openclaw"):
        raw_launch = launch_data.get(target)
        if not isinstance(raw_launch, dict):
            continue
        target_data = cast(dict[str, object], raw_launch)
        launch[target] = LaunchConfig(
            model=_str_value(target_data, "model"),
            route=_parse_launch_route(target_data.get("route")),
        )

    default_provider = _str_value(data, "default_provider") or DEFAULT_PROVIDER
    routing_profiles_raw = data.get("routing_profiles")
    routing_profiles = (
        cast(dict[str, object], routing_profiles_raw)
        if isinstance(routing_profiles_raw, dict) else None
    )
    return UserConfig(
        default_provider=default_provider,
        providers=providers,
        launch=launch,
        routing_profiles=routing_profiles,
        skill_distillation=_parse_skill_distillation_config(
            data.get("skill_distillation"),
        ),
    )


def load_user_credentials(config_dir: Path | None = None) -> UserCredentials:
    """Load secret user credentials, returning empty credentials when absent."""

    data = _read_json_object(get_credentials_path(config_dir))
    providers_data = _mapping_value(data, "providers")
    api_keys: dict[str, str] = {}
    for name, raw_provider in providers_data.items():
        if not isinstance(raw_provider, dict):
            continue
        provider_data = cast(dict[str, object], raw_provider)
        api_key = _str_value(provider_data, "api_key")
        if api_key:
            api_keys[name] = api_key

    launch_data = _mapping_value(data, "launch")
    launch: dict[LaunchTarget, LaunchCredentials] = {}
    for target in ("claude", "codex", "openclaw"):
        raw_launch = launch_data.get(target)
        if not isinstance(raw_launch, dict):
            continue
        target_data = cast(dict[str, object], raw_launch)
        tier_api_keys_data = _mapping_value(target_data, "api_keys")
        tier_api_keys: dict[str, str] = {}
        for tier, raw_api_key in tier_api_keys_data.items():
            if isinstance(raw_api_key, str) and raw_api_key:
                tier_api_keys[tier] = raw_api_key
        if tier_api_keys:
            launch[target] = LaunchCredentials(api_keys=tier_api_keys)

    return UserCredentials(api_keys=api_keys, launch=launch)


def _config_to_json(config: UserConfig) -> dict[str, object]:
    providers: dict[str, object] = {}
    for name, provider in (config.providers or {}).items():
        body: dict[str, object] = {}
        if provider.base_url:
            body["base_url"] = provider.base_url
        providers[name] = body

    launch: dict[str, object] = {}
    for target, target_config in (config.launch or {}).items():
        launch[target] = _launch_config_to_json(target_config)

    top: dict[str, object] = {
        "default_provider": config.default_provider,
        "providers": providers,
        "launch": launch,
    }
    if config.routing_profiles:
        top["routing_profiles"] = config.routing_profiles
    if not config.skill_distillation.is_default():
        top["skill_distillation"] = _skill_distillation_to_json(
            config.skill_distillation,
        )
    return top


def _credentials_to_json(credentials: UserCredentials) -> dict[str, object]:
    providers: dict[str, object] = {}
    for name, api_key in (credentials.api_keys or {}).items():
        providers[name] = {"api_key": api_key}

    launch: dict[str, object] = {}
    for target, target_credentials in (credentials.launch or {}).items():
        api_keys = {
            tier: api_key
            for tier, api_key in (target_credentials.api_keys or {}).items()
            if api_key
        }
        if api_keys:
            launch[target] = {"api_keys": api_keys}

    body: dict[str, object] = {"providers": providers}
    if launch:
        body["launch"] = launch
    return body


def _write_json(path: Path, data: dict[str, object], *, private: bool) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    if private:
        os.chmod(tmp_path, 0o600)
    tmp_path.replace(path)
    if private:
        os.chmod(path, 0o600)


def save_user_config(config: UserConfig, config_dir: Path | None = None) -> None:
    """Persist non-secret user config."""

    _write_json(get_config_path(config_dir), _config_to_json(config), private=False)


def save_user_credentials(
    credentials: UserCredentials, config_dir: Path | None = None,
) -> None:
    """Persist secret user credentials with owner-only permissions."""

    _write_json(
        get_credentials_path(config_dir),
        _credentials_to_json(credentials),
        private=True,
    )


def reset_user_config(config_dir: Path | None = None) -> list[Path]:
    """Delete persisted user config and credentials, returning removed paths."""

    removed: list[Path] = []
    for path in (get_config_path(config_dir), get_credentials_path(config_dir)):
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def redact_secret(secret: str | None) -> str | None:
    """Return a display-safe representation of *secret*."""

    if not secret:
        return None
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]}"


def build_redacted_snapshot(config_dir: Path | None = None) -> dict[str, object]:
    """Return config and credentials with secret values redacted."""

    config = load_user_config(config_dir)
    credentials = load_user_credentials(config_dir)
    providers: dict[str, object] = {}
    provider_names = set((config.providers or {}).keys()) | set((credentials.api_keys or {}).keys())
    for provider in sorted(provider_names):
        provider_config = config.provider(provider)
        body: dict[str, object] = {}
        if provider_config.base_url:
            body["base_url"] = provider_config.base_url
        redacted = redact_secret(credentials.api_key(provider))
        if redacted:
            body["api_key"] = redacted
        providers[provider] = body

    launch: dict[str, object] = {}
    for target, target_config in sorted((config.launch or {}).items()):
        body = _launch_config_to_json(target_config)
        route = _mapping_value(body, "route")
        launch_credentials = credentials.launch_target(target)
        for tier, api_key in (launch_credentials.api_keys or {}).items():
            redacted = redact_secret(api_key)
            if not redacted:
                continue
            endpoints = route.setdefault("endpoints", {})
            if not isinstance(endpoints, dict):
                endpoints = {}
                route["endpoints"] = endpoints
            endpoint = endpoints.setdefault(tier, {})
            if isinstance(endpoint, dict):
                endpoint["api_key"] = redacted
        launch[target] = body

    snapshot: dict[str, object] = {
        "config_path": str(get_config_path(config_dir)),
        "credentials_path": str(get_credentials_path(config_dir)),
        "default_provider": config.default_provider,
        "providers": providers,
        "launch": launch,
        "skill_distillation": _skill_distillation_to_json(
            config.skill_distillation,
        ),
    }
    if config.routing_profiles:
        # Surface only route ids — the saved bundle can carry env-var
        # references to secrets that we don't want printing to stdout.
        routes = config.routing_profiles.get("routes")
        if isinstance(routes, dict):
            snapshot["routing_profiles"] = {"route_ids": sorted(routes.keys())}
        else:
            snapshot["routing_profiles"] = {"route_ids": []}
    return snapshot


def get_configured_launch_model(
    target: LaunchTarget, config_dir: Path | None = None,
) -> str | None:
    """Return the configured default model for a launcher, if any."""

    return load_user_config(config_dir).launch_target(target).effective_route().model


def _first_env_setting(env_vars: tuple[str, ...]) -> tuple[str | None, str | None]:
    for env_var in env_vars:
        value = os.environ.get(env_var)
        if value:
            return value, env_var
    return None, None


def _env_prefix_from_api_key_var(env_var: str) -> str | None:
    suffix = "_API_KEY"
    if not env_var.endswith(suffix):
        return None
    return env_var.removesuffix(suffix)


def _env_prefix_from_provider(provider: str) -> str:
    return provider.upper().replace("-", "_")


def _provider_from_api_key_env_var(env_var: str | None) -> str | None:
    prefix = _env_prefix_from_api_key_var(env_var) if env_var is not None else None
    provider = prefix.lower().replace("_", "-") if prefix else None
    return provider if provider in _PROVIDER_BASE_URL_DEFAULTS else None


def _default_base_url_for_provider(provider: str) -> str:
    return _PROVIDER_BASE_URL_DEFAULTS.get(provider, DEFAULT_OPENROUTER_BASE_URL)


def _base_url_env_for_prefix(
    prefix: str | None,
    base_url_env_vars: tuple[str, ...],
) -> str | None:
    if not prefix:
        return None
    env_var = f"{prefix}_BASE_URL"
    if env_var not in base_url_env_vars:
        return None
    value = os.environ.get(env_var)
    return value if value else None


def _matching_base_url_env_value(
    *,
    api_key_env_var: str | None,
    provider: str,
    base_url_env_vars: tuple[str, ...],
) -> str | None:
    """Return a base URL env var that belongs to the selected credential source.

    Launcher shells often contain unrelated provider variables, e.g.
    ``OPENAI_BASE_URL`` from another CLI while Switchyard's selected key
    belongs to NVIDIA. Treating the first configured base-url env var as
    global can pair a key/model with the wrong endpoint. If an API-key env var
    won, only accept the matching base-url env var; otherwise fall back to the
    configured provider.
    """

    from_api_key = _base_url_env_for_prefix(
        _env_prefix_from_api_key_var(api_key_env_var)
        if api_key_env_var is not None else None,
        base_url_env_vars,
    )
    if api_key_env_var is not None:
        return from_api_key
    return _base_url_env_for_prefix(
        _env_prefix_from_provider(provider),
        base_url_env_vars,
    )


def _secrets_provider_values(
    secrets: Mapping[str, Mapping[str, object]] | None,
    *,
    secrets_section_priority: tuple[str, ...],
) -> tuple[str | None, str | None]:
    if not secrets:
        return None, None

    api_key: str | None = None
    base_url: str | None = None
    for section_name in secrets_section_priority:
        section = secrets.get(section_name, {})
        if api_key is None:
            api_key = _str_value(section, "api_key")
        if base_url is None:
            base_url = _str_value(section, "base_url")

    if api_key is None:
        first_provider = next(iter(secrets.values()), {})
        api_key = _str_value(first_provider, "api_key")
        if base_url is None:
            base_url = _str_value(first_provider, "base_url")

    return api_key, base_url


def resolve_provider_connectivity(
    *,
    cli_api_key: str | None,
    cli_base_url: str | None,
    api_key_env_vars: tuple[str, ...],
    base_url_env_vars: tuple[str, ...],
    secrets: Mapping[str, Mapping[str, object]] | None = None,
    secrets_section_priority: tuple[str, ...] = DEFAULT_SECRETS_SECTION_PRIORITY,
    default_provider: str | None = None,
    default_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    config_dir: Path | None = None,
) -> ProviderConnectivity:
    """Resolve provider connectivity using the launcher precedence rules.

    Precedence:
        API key: CLI flag > environment variable > user credentials >
        repo-local secrets.json.

        Base URL: CLI flag > environment variable matching the selected
        API-key/provider source > user config > repo-local secrets.json >
        built-in base-url default.
    """

    user_config = load_user_config(config_dir)
    provider = default_provider or user_config.default_provider or DEFAULT_PROVIDER
    user_credentials = load_user_credentials(config_dir)
    secrets_api_key, secrets_base_url = _secrets_provider_values(
        secrets,
        secrets_section_priority=secrets_section_priority,
    )

    api_key_env_value, api_key_env_var = _first_env_setting(api_key_env_vars)
    selected_provider = _provider_from_api_key_env_var(api_key_env_var) or provider
    selected_user_provider = user_config.provider(selected_provider)

    api_key = (
        cli_api_key
        or api_key_env_value
        or user_credentials.api_key(provider)
        or secrets_api_key
    )
    base_url = (
        cli_base_url
        or _matching_base_url_env_value(
            api_key_env_var=api_key_env_var,
            provider=provider,
            base_url_env_vars=base_url_env_vars,
        )
        or selected_user_provider.base_url
        or secrets_base_url
        or _PROVIDER_BASE_URL_DEFAULTS.get(selected_provider, default_base_url)
    )

    return ProviderConnectivity(
        provider=selected_provider,
        api_key=api_key,
        base_url=base_url,
    )
