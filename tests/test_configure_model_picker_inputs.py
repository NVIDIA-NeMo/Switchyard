# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the `configure` wizard's model-picker input plumbing.

Pins the two helpers that feed `wizard.select_model(...)`:

* `_routing_profile_model_ids(...)` — extracts user-callable model ids from a
  parsed routing-profiles bundle.
* `_merge_candidate_ids(...)` — unions routing-profile ids with the upstream
  catalog in first-seen order.

Without these in the picker pipeline the wizard offers only the upstream
catalog and the user can't pick a route key like `opus-ds-stage_router` even when
their YAML declares it.
"""

from __future__ import annotations

from switchyard.cli.configure_command import _merge_candidate_ids
from switchyard.cli.route_bundle import routing_profile_model_ids as _routing_profile_model_ids


def test_empty_or_none_bundle_returns_empty() -> None:
    assert _routing_profile_model_ids(None) == []
    assert _routing_profile_model_ids({}) == []
    assert _routing_profile_model_ids({"routes": {}}) == []


def test_extracts_route_keys_and_tier_models() -> None:
    """StageRouter + deterministic routes contribute the route key plus
    strong/weak tier models. Classifier is intentionally skipped (internal)."""
    bundle = {
        "routes": {
            "opus-ds-stage_router": {
                "type": "stage_router",
                "picker": "capable_first",
                "fallback_target_on_evict": "strong",
                "strong": {"model": "aws/anthropic/bedrock-claude-opus-4-7"},
                "weak": {"model": "nvidia/deepseek-ai/evals-deepseek-v4-pro"},
                "classifier": {"model": "nvidia/deepseek-ai/deepseek-v4-flash"},
            },
            "opus-ds-classifier": {
                "type": "deterministic",
                "profile": "coding_agent",
                "fallback_target_on_evict": "strong",
                "strong": {"model": "aws/anthropic/bedrock-claude-opus-4-7"},
                "weak": {"model": "nvidia/deepseek-ai/evals-deepseek-v4-pro"},
                "classifier": {"model": "nvidia/deepseek-ai/deepseek-v4-flash"},
            },
        },
    }
    assert _routing_profile_model_ids(bundle) == [
        "opus-ds-stage_router",
        "aws/anthropic/bedrock-claude-opus-4-7",
        "nvidia/deepseek-ai/evals-deepseek-v4-pro",
        "opus-ds-classifier",
    ]


def test_single_model_passthrough_target_extracted() -> None:
    bundle = {
        "routes": {
            "opus-direct": {
                "type": "model",
                "target": {"model": "aws/anthropic/bedrock-claude-opus-4-7"},
            },
        },
    }
    assert _routing_profile_model_ids(bundle) == [
        "opus-direct",
        "aws/anthropic/bedrock-claude-opus-4-7",
    ]


def test_string_shorthand_tier_resolves_to_model_id() -> None:
    """``strong: "model/id"`` shorthand should land as a candidate too."""
    bundle = {
        "routes": {
            "rr": {
                "type": "random_routing",
                "strong": "gpt-4o",
                "weak": "gpt-4o-mini",
            },
        },
    }
    assert _routing_profile_model_ids(bundle) == ["rr", "gpt-4o", "gpt-4o-mini"]


def test_plan_execute_planner_executor_extracted() -> None:
    bundle = {
        "routes": {
            "plan-route": {
                "type": "plan_execute",
                "planner": {"model": "aws/anthropic/bedrock-claude-opus-4-7"},
                "executor": {"model": "nvidia/nvidia/nemotron-3-super-120b-long-ctx"},
            },
        },
    }
    assert _routing_profile_model_ids(bundle) == [
        "plan-route",
        "aws/anthropic/bedrock-claude-opus-4-7",
        "nvidia/nvidia/nemotron-3-super-120b-long-ctx",
    ]


def test_prefill_probe_exposes_completion_tiers_but_not_probe() -> None:
    bundle = {
        "routes": {
            "complexity-router": {
                "type": "prefill_probe",
                "probe": {"model": "Qwen/Qwen3.6-35B-A3B"},
                "strong": {"model": "anthropic/claude-opus-4.7"},
                "weak": {"model": "nvidia/nemotron-3-super"},
            },
        },
    }

    assert _routing_profile_model_ids(bundle) == [
        "complexity-router",
        "anthropic/claude-opus-4.7",
        "nvidia/nemotron-3-super",
    ]


def test_merge_candidate_ids_dedupes_preserves_first_seen_order() -> None:
    """Routing-profile entries first, then upstream catalog. Duplicates from
    later sources drop."""
    routing = ["opus-ds-stage_router", "aws/anthropic/bedrock-claude-opus-4-7"]
    upstream = [
        "openai/gpt-5.2",
        "aws/anthropic/bedrock-claude-opus-4-7",  # dup with routing[1]
        "nvidia/nvidia/nemotron-3-super-120b-long-ctx",
    ]
    assert _merge_candidate_ids(routing, upstream) == [
        "opus-ds-stage_router",
        "aws/anthropic/bedrock-claude-opus-4-7",
        "openai/gpt-5.2",
        "nvidia/nvidia/nemotron-3-super-120b-long-ctx",
    ]


def test_merge_candidate_ids_skips_empty_strings() -> None:
    assert _merge_candidate_ids(["a", "", "b"], ["", "c"]) == ["a", "b", "c"]
