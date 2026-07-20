# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from switchyard.cli.config.user_config import LaunchRouteConfig
from switchyard.cli.routing.route_builder import (
    LaunchTierConnectivity,
    build_deterministic_routing_config,
    build_plan_execute_config,
    build_random_routing_config,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.profiles.random_routing import (
    RandomRoutingConfig,
)
from switchyard.lib.route_table_builders import build_passthrough_table
from switchyard.lib.stats_accumulator import StatsAccumulator


def _connectivity() -> LaunchTierConnectivity:
    return LaunchTierConnectivity(
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
    )


def test_random_launch_routes_use_auto_backend_format_for_both_tiers() -> None:
    route = LaunchRouteConfig(
        type="random",
        model="strong-model",
        weak_model="weak-model",
    )

    config = build_random_routing_config(
        route,
        primary=_connectivity(),
        weak=_connectivity(),
        timeout=None,
    )

    assert config.strong.format is BackendFormat.AUTO
    assert config.weak.format is BackendFormat.AUTO


def test_passthrough_table_dedupes_endpoints_and_registers_discovered_models() -> None:
    calls: list[tuple[str, str]] = []

    def fake_discover_models(base_url: str, api_key: str) -> list[str]:
        calls.append((base_url, api_key))
        if base_url == "https://primary.example/v1":
            return ["primary/extra", "shared/model"]
        return ["weak/extra", "shared/model"]

    config = RandomRoutingConfig(
        strong=LlmTarget(
            model="strong/model",
            format=BackendFormat.AUTO,
            api_key="primary-key",
            base_url="https://primary.example/v1",
        ),
        weak=LlmTarget(
            model="weak/model",
            format=BackendFormat.AUTO,
            api_key="weak-key",
            base_url="https://weak.example/v1",
        ),
    fallback_target_on_evict="strong")

    table = build_passthrough_table(
        (config.strong, config.weak),
        StatsAccumulator(),
        discovery_fn=fake_discover_models,
    )

    # Each (base_url, api_key) pair is fetched exactly once.
    assert calls == [
        ("https://primary.example/v1", "primary-key"),
        ("https://weak.example/v1", "weak-key"),
    ]
    # Configured tier models register before discovered models; identical
    # discovered ids across tiers register against the first tier that saw them.
    assert table.registered_models() == [
        "strong/model",
        "weak/model",
        "primary/extra",
        "shared/model",
        "weak/extra",
    ]


def test_passthrough_table_skips_discovery_when_fn_is_none() -> None:
    config = RandomRoutingConfig(
        strong=LlmTarget(
            model="strong/model",
            format=BackendFormat.AUTO,
            api_key="primary-key",
            base_url="https://primary.example/v1",
        ),
        weak=LlmTarget(
            model="weak/model",
            format=BackendFormat.AUTO,
            api_key="weak-key",
            base_url="https://weak.example/v1",
        ),
    fallback_target_on_evict="strong")

    table = build_passthrough_table(
        (config.strong, config.weak),
        StatsAccumulator(),
        # discovery_fn=None (default) — no discovery runs.
    )

    # Only the configured tier models register; discovery never runs.
    assert table.registered_models() == ["strong/model", "weak/model"]


def test_deterministic_launch_routes_use_preset_defaults() -> None:
    """Zero-override --deterministic launches use the validated TB-Lite trio."""
    route = LaunchRouteConfig(type="deterministic")
    config = build_deterministic_routing_config(
        route,
        primary=_connectivity(),
        weak=_connectivity(),
        classifier_model=None,
        profile_name=None,
        classifier_min_confidence=None,
        timeout=600.0,
    )
    assert config.strong.model == "anthropic/claude-opus-4.7"
    assert config.weak.model == "moonshotai/kimi-k2.6"
    assert config.classifier.model == "google/gemini-3.5-flash"
    assert config.profile_name == "coding_agent"
    assert config.classifier_min_confidence == 0.0
    assert config.preset == "coding_agent_default"
    # Tier formats are preset-owned.
    assert config.strong.format == BackendFormat.AUTO
    assert config.weak.format == BackendFormat.OPENAI
    assert config.classifier.format == BackendFormat.OPENAI


def test_deterministic_launch_routes_apply_user_overrides() -> None:
    """User-supplied --model / --weak-model / --classifier-model / --profile win."""
    route = LaunchRouteConfig(
        type="deterministic",
        model="openai/openai/gpt-5.2",
        weak_model="nvidia/moonshotai/kimi-k2.5",
    )
    config = build_deterministic_routing_config(
        route,
        primary=_connectivity(),
        weak=_connectivity(),
        classifier_model="nvidia/nvidia/nemotron-3-super-v3",
        profile_name="general",
        classifier_min_confidence=0.55,
        timeout=600.0,
    )
    assert config.strong.model == "openai/openai/gpt-5.2"
    assert config.weak.model == "nvidia/moonshotai/kimi-k2.5"
    assert config.classifier.model == "nvidia/nvidia/nemotron-3-super-v3"
    assert config.profile_name == "general"
    assert config.classifier_min_confidence == 0.55
    assert config.preset is None
    assert config.strong.format == BackendFormat.AUTO


def test_deterministic_launch_routes_reject_invalid_profile() -> None:
    route = LaunchRouteConfig(type="deterministic")
    with pytest.raises(SystemExit, match="unknown profile"):
        build_deterministic_routing_config(
            route,
            primary=_connectivity(),
            weak=_connectivity(),
            classifier_model=None,
            profile_name="invented_profile",
            classifier_min_confidence=None,
            timeout=600.0,
        )


def test_deterministic_launch_routes_reject_out_of_range_confidence() -> None:
    route = LaunchRouteConfig(type="deterministic")
    with pytest.raises(SystemExit, match=r"\[0\.0, 1\.0\]"):
        build_deterministic_routing_config(
            route,
            primary=_connectivity(),
            weak=_connectivity(),
            classifier_model=None,
            profile_name=None,
            classifier_min_confidence=1.5,
            timeout=600.0,
        )


def test_plan_execute_launch_routes_use_preset_defaults() -> None:
    """Zero-override --plan-execute launches use the strong-planner pairing."""
    route = LaunchRouteConfig(type="plan_execute")
    config = build_plan_execute_config(
        route,
        primary=_connectivity(),
        backend_format=BackendFormat.OPENAI,
        timeout=600.0,
    )
    assert config.planner.model == "anthropic/claude-opus-4.6"
    assert config.executor.model == "moonshotai/kimi-k2.6"
    assert config.cadence_n == 2
    assert config.preset == "coding_agent_default"


def test_plan_execute_launch_routes_apply_executor_override() -> None:
    """--model X swaps the executor but leaves the strong planner."""
    route = LaunchRouteConfig(
        type="plan_execute",
        model="my/custom-executor",
    )
    config = build_plan_execute_config(
        route,
        primary=_connectivity(),
        backend_format=BackendFormat.OPENAI,
        timeout=600.0,
    )
    # Planner stays on the preset; executor adopts the user's override.
    assert config.planner.model == "anthropic/claude-opus-4.6"
    assert config.executor.model == "my/custom-executor"
    # Preset label drops once any tier deviates from the shipping bundle.
    assert config.preset is None


def test_plan_execute_launch_routes_pin_backend_format() -> None:
    """codex/openclaw pin OPENAI; the executor tier honours the request."""
    route = LaunchRouteConfig(type="plan_execute")
    config = build_plan_execute_config(
        route,
        primary=_connectivity(),
        backend_format=BackendFormat.OPENAI,
        timeout=None,
    )
    assert config.executor.format is BackendFormat.OPENAI


def test_plan_execute_launch_routes_propagate_connectivity() -> None:
    route = LaunchRouteConfig(type="plan_execute")
    primary = LaunchTierConnectivity(
        api_key="sk-launch",
        base_url="https://example.invalid/v1",
    )
    config = build_plan_execute_config(
        route,
        primary=primary,
        backend_format=BackendFormat.OPENAI,
        timeout=600.0,
    )
    assert config.executor.endpoint.api_key == "sk-launch"
    assert config.executor.endpoint.base_url == "https://example.invalid/v1"
    assert config.planner.endpoint.api_key == "sk-launch"
    assert config.planner.endpoint.base_url == "https://example.invalid/v1"
