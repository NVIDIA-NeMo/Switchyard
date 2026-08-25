# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import switchyard_litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import RoutingContext


def routing_context(
    messages: list[dict[str, Any]],
    candidates: list[str],
) -> RoutingContext:
    """Build the LiteLLM boundary object passed to candidate-bound plugins."""
    return RoutingContext(
        raw_messages=messages,
        structured_messages=messages,
        candidate_models=candidates,
        metadata={},
        signals={},
    )


async def test_stage_uses_deployer_candidates_in_capable_then_efficient_order() -> None:
    plugin_type = getattr(switchyard_litellm, "StageRoutingPlugin", None)
    assert plugin_type is not None, "StageRoutingPlugin must be part of the public integration API"
    plugin = plugin_type(
        picker="efficient_first",
        confidence_threshold=0.5,
        recent_window=3,
    )
    assert isinstance(plugin, CustomLogger)
    context = routing_context(
        [{"role": "user", "content": "Say hello."}],
        ["provider/deployer-capable", "provider/deployer-efficient"],
    )

    await plugin.run(context)

    assert context.candidate_models == ["provider/deployer-efficient"]
    assert context.signals["switchyard"] == {
        "selected_model_id": "provider/deployer-efficient",
        "fallback_models": ["provider/deployer-capable"],
    }


@pytest.mark.parametrize(
    "candidates",
    [
        ["provider/only-one"],
        ["provider/one", "provider/two", "provider/three"],
    ],
)
async def test_stage_rejects_candidate_pools_without_two_unique_models(
    candidates: list[str],
) -> None:
    plugin = switchyard_litellm.StageRoutingPlugin(
        picker="efficient_first",
        confidence_threshold=0.5,
    )
    context = routing_context([{"role": "user", "content": "Hello."}], candidates)

    with pytest.raises(ValueError, match="exactly two unique LiteLLM candidates"):
        await plugin.run(context)


async def test_stage_treats_duplicate_deployments_as_one_candidate_model() -> None:
    plugin = switchyard_litellm.StageRoutingPlugin(
        picker="efficient_first",
        confidence_threshold=0.5,
    )
    context = routing_context(
        [{"role": "user", "content": "Hello."}],
        ["provider/capable", "provider/capable", "provider/efficient"],
    )

    await plugin.run(context)

    assert context.candidate_models == ["provider/efficient"]


async def test_seeded_random_uses_and_retains_the_full_deployer_candidate_pool() -> None:
    plugin_type = getattr(switchyard_litellm, "RandomRoutingPlugin", None)
    assert plugin_type is not None, "RandomRoutingPlugin must be part of the public integration API"
    plugin = plugin_type(seed=6)
    assert isinstance(plugin, CustomLogger)
    candidates = ["provider/alpha", "provider/beta", "provider/gamma"]
    selected: list[str] = []

    for index in range(5):
        context = routing_context(
            [{"role": "user", "content": f"Request {index}."}],
            candidates.copy(),
        )
        await plugin.run(context)
        selected.extend(context.candidate_models)

    assert selected == [
        "provider/beta",
        "provider/gamma",
        "provider/gamma",
        "provider/gamma",
        "provider/alpha",
    ]


async def test_random_rejects_an_empty_candidate_pool() -> None:
    plugin = switchyard_litellm.RandomRoutingPlugin(seed=6)
    context = routing_context([{"role": "user", "content": "Hello."}], [])

    with pytest.raises(ValueError, match="requires at least one LiteLLM candidate"):
        await plugin.run(context)


async def test_random_weights_apply_to_unique_candidate_models() -> None:
    plugin = switchyard_litellm.RandomRoutingPlugin(weights=[0.0, 1.0], seed=6)
    context = routing_context(
        [{"role": "user", "content": "Hello."}],
        ["provider/alpha", "provider/alpha", "provider/beta"],
    )

    await plugin.run(context)

    assert context.candidate_models == ["provider/beta"]
