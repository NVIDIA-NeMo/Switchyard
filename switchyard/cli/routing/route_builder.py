# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build routing configs from persisted launcher routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from switchyard.cli.config.user_config import LaunchRouteConfig
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.profiles import (
    DeterministicRoutingConfig,
    DeterministicRoutingPresets,
)
from switchyard.lib.profiles.deterministic_routing_config import ProfileName
from switchyard.lib.profiles.random_routing import (
    RandomRoutingConfig,
)

_SUPPORTED_PROFILES: tuple[str, ...] = ("general", "coding_agent", "openclaw")


@dataclass(frozen=True)
class LaunchTierConnectivity:
    """Resolved endpoint credentials for one launch route tier."""

    api_key: str | None
    base_url: str


def require_route_model(route: LaunchRouteConfig, *, target: str) -> str:
    """Return the primary model or fail with a launch-specific message."""

    if route.model:
        return route.model
    raise SystemExit(
        f"launch {target}: requires a configured model. Run "
        f"`switchyard configure --target {target}` or pass --model."
    )


def build_random_routing_config(
    route: LaunchRouteConfig,
    *,
    primary: LaunchTierConnectivity,
    weak: LaunchTierConnectivity,
    timeout: float | None,
) -> RandomRoutingConfig:
    """Build the first-pass random route: primary model vs weaker model."""

    if not route.model:
        raise SystemExit("random routing requires a primary model.")
    if not route.weak_model:
        raise SystemExit(
            "random routing requires a weak model. Pass --weak-model or "
            "reconfigure this launcher."
        )

    return RandomRoutingConfig(
        strong=LlmTarget(
            id="strong",
            model=route.model,
            format=BackendFormat.AUTO,
            api_key=primary.api_key,
            base_url=primary.base_url,
            timeout_secs=timeout,
        ),
        weak=LlmTarget(
            id="weak",
            model=route.weak_model,
            format=BackendFormat.AUTO,
            api_key=weak.api_key,
            base_url=weak.base_url,
            timeout_secs=timeout,
        ),
        strong_probability=route.strong_probability,
        fallback_target_on_evict="strong",
    )


def build_deterministic_routing_config(
    route: LaunchRouteConfig,
    *,
    primary: LaunchTierConnectivity,
    weak: LaunchTierConnectivity,
    classifier_model: str | None,
    profile_name: str | None,
    classifier_min_confidence: float | None,
    backend_format: BackendFormat,
    timeout: float | None,
    strong_backend_format: BackendFormat | None = None,
) -> DeterministicRoutingConfig:
    """Layer LLM-as-classifier CLI overrides on top of the shipping preset.

    The shipping preset (``DeterministicRoutingPresets.coding_agent_default``)
    is the validated TB-Lite trio. We start from it and override only the
    fields the user explicitly supplied so ``switchyard launch claude``
    works with zero flags, while ``--model`` / ``--weak-model`` /
    ``--classifier-model`` / ``--profile`` / ``--classifier-min-confidence``
    each replace one piece.

    Args:
        route: Resolved :class:`LaunchRouteConfig`. ``route.model`` and
            ``route.weak_model`` carry the user's explicit overrides
            (``None`` when not supplied — the preset wins).
        primary: Connectivity for the strong tier (also reused for the
            classifier endpoint since the validated trio runs all three
            against the same gateway.
        weak: Connectivity for the weak tier.
        classifier_model: User override for the classifier LLM model id.
        profile_name: User override for the classifier profile (``general``
            / ``coding_agent`` / ``openclaw``).
        classifier_min_confidence: User override for the tier selector's
            confidence floor.
        backend_format: Wire format for the weak + classifier tiers (and the
            strong tier when ``strong_backend_format`` is ``None``).
        strong_backend_format: Strong-tier wire format override; ``AUTO``
            probes the backend for Anthropic Messages support at runtime.
        timeout: Per-request HTTP timeout for the tier backends
            (seconds).
    """
    if profile_name is not None and profile_name not in _SUPPORTED_PROFILES:
        raise SystemExit(
            f"deterministic routing: unknown profile {profile_name!r}; "
            f"choose one of {list(_SUPPORTED_PROFILES)}."
        )
    if classifier_min_confidence is not None and not (
        0.0 <= classifier_min_confidence <= 1.0
    ):
        raise SystemExit(
            "deterministic routing: --classifier-min-confidence must be in "
            "[0.0, 1.0].",
        )

    # Start from the validated shipping bundle so zero-flag launches work.
    preset = DeterministicRoutingPresets.coding_agent_default(
        api_key=primary.api_key or "",
        base_url=primary.base_url,
        timeout_secs=timeout,
    )

    strong_model = route.model or preset.strong.model
    weak_model = route.weak_model or preset.weak.model
    classifier_model_resolved = classifier_model or preset.classifier.model
    profile_resolved: ProfileName = cast(
        ProfileName, profile_name or preset.profile_name,
    )
    min_confidence = (
        classifier_min_confidence
        if classifier_min_confidence is not None
        else preset.classifier_min_confidence
    )

    strong_target = LlmTarget(
        id="strong",
        model=strong_model,
        format=strong_backend_format or backend_format,
        api_key=primary.api_key,
        base_url=primary.base_url,
        timeout_secs=timeout,
    )
    weak_target = LlmTarget(
        id="weak",
        model=weak_model,
        format=backend_format,
        api_key=weak.api_key,
        base_url=weak.base_url,
        timeout_secs=timeout,
    )
    classifier_target = LlmTarget(
        id="classifier",
        model=classifier_model_resolved,
        format=backend_format,
        api_key=primary.api_key,
        base_url=primary.base_url,
        timeout_secs=preset.classifier_timeout_s,
    )

    return DeterministicRoutingConfig(
        strong=strong_target,
        weak=weak_target,
        classifier=classifier_target,
        profile_name=profile_resolved,
        classifier_min_confidence=min_confidence,
        classifier_fail_open=preset.classifier_fail_open,
        classifier_recent_turn_window=preset.classifier_recent_turn_window,
        classifier_timeout_s=preset.classifier_timeout_s,
        enable_stats=preset.enable_stats,
        fallback_target_on_evict=preset.fallback_target_on_evict,
        preset=(
            preset.preset
            if (
                route.model is None
                and route.weak_model is None
                and classifier_model is None
                and profile_name is None
            )
            else None
        ),
    )


__all__ = [
    "LaunchTierConnectivity",
    "build_deterministic_routing_config",
    "build_random_routing_config",
    "require_route_model",
]
