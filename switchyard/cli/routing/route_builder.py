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
    PlanExecuteConfig,
    PlanExecutePresets,
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
    timeout: float | None,
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
        format=preset.strong.format,
        api_key=primary.api_key,
        base_url=primary.base_url,
        timeout_secs=timeout,
    )
    weak_target = LlmTarget(
        id="weak",
        model=weak_model,
        format=preset.weak.format,
        api_key=weak.api_key,
        base_url=weak.base_url,
        timeout_secs=timeout,
    )
    classifier_target = LlmTarget(
        id="classifier",
        model=classifier_model_resolved,
        format=preset.classifier.format,
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


def build_plan_execute_config(
    route: LaunchRouteConfig,
    *,
    primary: LaunchTierConnectivity,
    backend_format: BackendFormat,
    timeout: float | None,
) -> PlanExecuteConfig:
    """Layer ``--plan-execute`` CLI overrides on top of the shipping preset.

    The shipping preset (:meth:`PlanExecutePresets.coding_agent_default`)
    is the strong-planner + weak-executor pairing carried forward from
    commit ``ca5fcd8a``. We start from it and override only the executor
    model when the user supplied ``--model``. Planner overrides aren't
    exposed on the CLI in v1 — users with a benchmark reason to vary the
    planner write a YAML route bundle and pass ``--routing-profiles``.

    Args:
        route: Resolved :class:`LaunchRouteConfig`. ``route.model``
            carries the user's explicit executor override (``None`` when
            not supplied — the preset's Kimi executor wins).
        primary: Connectivity for both planner and executor — the
            shipping trio runs both against the same NVIDIA Inference
            Hub tenancy. The launcher's ``--base-url`` / ``--api-key``
            therefore apply uniformly.
        backend_format: Wire format pinned by the calling launcher
            (codex / openclaw: :class:`BackendFormat.OPENAI`; claude
            defaults the same for the shipping pairing).
        timeout: Per-request HTTP timeout for the executor backend
            (seconds). The planner uses its own timeout from the
            preset.
    """
    preset = PlanExecutePresets.coding_agent_default(
        api_key=primary.api_key or "",
        base_url=primary.base_url,
        timeout_secs=timeout,
    )

    # Executor override: --model wins when supplied; otherwise the
    # preset's validated weak tier stays.
    executor_model = route.model or preset.executor.model
    executor_target = LlmTarget(
        id="executor",
        model=executor_model,
        format=backend_format,
        api_key=primary.api_key,
        base_url=primary.base_url,
        timeout_secs=timeout,
    )

    # Planner stays on the preset's Opus tier — credentials come
    # from the same connectivity bundle so a single --api-key / --base-url
    # covers both.
    planner_target = LlmTarget(
        id="planner",
        model=preset.planner.model,
        format=preset.planner.format,
        api_key=primary.api_key,
        base_url=primary.base_url,
        timeout_secs=preset.planner.endpoint.timeout_secs,
    )

    return PlanExecuteConfig(
        planner=planner_target,
        executor=executor_target,
        cadence_n=preset.cadence_n,
        disable_reasoning=preset.disable_reasoning,
        fail_open=preset.fail_open,
        enable_stats=preset.enable_stats,
        fallback_target_on_evict=preset.fallback_target_on_evict,
        # Preserve the preset label only when the user did not override
        # the executor — once they pin a custom executor the bundle is
        # no longer the shipping default.
        preset=preset.preset if route.model is None else None,
    )


__all__ = [
    "LaunchTierConnectivity",
    "build_deterministic_routing_config",
    "build_plan_execute_config",
    "build_random_routing_config",
    "require_route_model",
]
