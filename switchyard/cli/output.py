# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Human-readable CLI output helpers."""

from collections.abc import Mapping
from typing import cast

from switchyard.cli.config.user_config import (
    PRIMARY_TIER,
    WEAK_TIER,
    LaunchRouteConfig,
)


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    return {}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


_TIERED_ROUTE_TYPES = frozenset({"random", "deterministic"})


def _format_route(route: Mapping[str, object]) -> list[str]:
    route_type = _string(route.get("type")) or "single"
    lines = [f"route: {route_type}"]
    model = _string(route.get("model"))
    if model:
        lines.append(f"model: {model}")
    weak_model = _string(route.get("weak_model"))
    if route_type in _TIERED_ROUTE_TYPES and weak_model:
        lines.append(f"weak model: {weak_model}")
    probability = route.get("strong_probability")
    if route_type == "random" and isinstance(probability, (int, float)):
        lines.append(f"strong probability: {float(probability):.2f}")
    endpoints = _mapping(route.get("endpoints"))
    primary_endpoint = _mapping(endpoints.get(PRIMARY_TIER))
    primary_base_url = _string(primary_endpoint.get("base_url"))
    if primary_base_url:
        lines.append(f"base URL override: {primary_base_url}")
    primary_api_key = _string(primary_endpoint.get("api_key"))
    if primary_api_key:
        lines.append(f"API key override: {primary_api_key}")
    weak_endpoint = _mapping(endpoints.get(WEAK_TIER))
    weak_base_url = _string(weak_endpoint.get("base_url"))
    if route_type in _TIERED_ROUTE_TYPES and weak_base_url:
        lines.append(f"weak base URL override: {weak_base_url}")
    weak_api_key = _string(weak_endpoint.get("api_key"))
    if route_type in _TIERED_ROUTE_TYPES and weak_api_key:
        lines.append(f"weak API key override: {weak_api_key}")
    return lines


def format_route_config(route: LaunchRouteConfig) -> list[str]:
    body: dict[str, object] = {"type": route.type}
    if route.model:
        body["model"] = route.model
    if route.weak_model:
        body["weak_model"] = route.weak_model
    if route.type == "random":
        body["strong_probability"] = route.strong_probability
    endpoints: dict[str, object] = {}
    for tier, endpoint in (route.endpoints or {}).items():
        endpoint_body: dict[str, object] = {}
        if endpoint.base_url:
            endpoint_body["base_url"] = endpoint.base_url
        if endpoint_body:
            endpoints[tier] = endpoint_body
    if endpoints:
        body["endpoints"] = endpoints
    return _format_route(body)


def format_config_snapshot(snapshot: Mapping[str, object]) -> str:
    """Render ``build_redacted_snapshot`` output for humans."""

    lines = [
        "Switchyard config",
        f"config: {snapshot.get('config_path')}",
        f"credentials: {snapshot.get('credentials_path')}",
        f"default provider: {snapshot.get('default_provider')}",
    ]

    providers = _mapping(snapshot.get("providers"))
    if providers:
        lines.append("")
        lines.append("Providers")
        for name, raw_provider in providers.items():
            provider = _mapping(raw_provider)
            lines.append(f"  {name}")
            base_url = _string(provider.get("base_url"))
            if base_url:
                lines.append(f"    base URL: {base_url}")
            api_key = _string(provider.get("api_key"))
            lines.append(f"    API key: {api_key or '<missing>'}")

    saved_bundle = snapshot.get("routing_profiles")
    if isinstance(saved_bundle, dict):
        route_ids = saved_bundle.get("route_ids")
        if isinstance(route_ids, list) and route_ids:
            lines.append("")
            lines.append("Routing profiles (saved bundle)")
            for route_id in route_ids:
                lines.append(f"  {route_id}")

    skill_distillation = _mapping(snapshot.get("skill_distillation"))
    if skill_distillation:
        lines.append("")
        lines.append("Skill distillation")
        namespace = _string(skill_distillation.get("namespace"))
        lines.append(f"  configured: {bool(namespace)}")
        lines.append(f"  namespace: {namespace or '<not configured>'}")
        if namespace:
            lines.append("  session learning: namespace saved")

    launch = _mapping(snapshot.get("launch"))
    if launch:
        lines.append("")
        lines.append("Launch defaults")
        for target, raw_launch in launch.items():
            target_config = _mapping(raw_launch)
            route = _mapping(target_config.get("route"))
            lines.append(f"  {target}")
            for route_line in _format_route(route):
                lines.append(f"    {route_line}")
    return "\n".join(lines)


def format_dry_run(
    *,
    target: str,
    route: LaunchRouteConfig,
    base_url: str,
    api_key_set: bool,
    port: int | None,
    timeout: float | None,
    forwarded_args: list[str],
    classifier_model: str | None = None,
    profile: str | None = None,
    classifier_min_confidence: float | None = None,
    routing_profiles: str | None = None,
) -> str:
    """Render resolved launch settings without exposing secrets.

    The optional ``classifier_*`` / ``profile`` kwargs are populated by the
    deterministic-routing dispatch path — the routing-policy fields that
    aren't carried on :class:`LaunchRouteConfig`.

    ``routing_profiles`` is the resolved routing-profiles YAML path for a
    multi-chain bundle launch. When set, the summary reports ``route: bundle``
    plus the bundle path and its route ids instead of the single-tier
    :func:`format_route_config` view (``route`` is a placeholder single-tier
    config in that case).
    """

    lines = [
        f"Switchyard launch dry run: {target}",
        f"base URL: {base_url}",
        f"API key: {'set' if api_key_set else '<missing>'}",
        f"port: {port if port is not None else 'auto'}",
        f"timeout: {timeout if timeout is not None else 'default'}",
    ]
    if routing_profiles:
        from switchyard.cli.route_bundle import (
            parse_routing_profiles_file,
            routing_profile_model_ids,
        )

        lines.append("route: bundle")
        lines.append(f"routing profiles: {routing_profiles}")
        for route_id in routing_profile_model_ids(
            parse_routing_profiles_file(routing_profiles)
        ):
            lines.append(f"  {route_id}")
    else:
        for route_line in format_route_config(route):
            lines.append(route_line)
        if route.type == "deterministic":
            if classifier_model:
                lines.append(f"classifier model: {classifier_model}")
            if profile:
                lines.append(f"profile: {profile}")
            if classifier_min_confidence is not None:
                lines.append(f"min confidence: {classifier_min_confidence:.2f}")
    if forwarded_args:
        lines.append(f"forwarded args: {' '.join(forwarded_args)}")
    return "\n".join(lines)


__all__ = [
    "format_config_snapshot",
    "format_dry_run",
    "format_route_config",
]
