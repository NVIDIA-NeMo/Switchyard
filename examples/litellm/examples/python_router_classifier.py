#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send requests through LiteLLM's Router with the Switchyard capability classifier.

The judge is its own model group. The plugin serves the judge call back through
the same Router, so it is spend-tracked and uses the gateway's credentials, and
the request's `switchyard_judge` metadata marker keeps it out of routing.
"""

import asyncio

import litellm
from litellm.router import Router
from switchyard_litellm import ClassifierRoutingPlugin, LiteLLMJudge

MODEL_GROUP = "switchyard"
JUDGE_GROUP = "switchyard-judge"
MODEL_LIST: list[dict[str, object]] = [
    {  # capable first
        "model_name": MODEL_GROUP,
        "litellm_params": {
            "model": "openrouter/openai/gpt-5.6-sol",
            "api_key": "os.environ/OPENROUTER_API_KEY",
        },
    },
    {  # efficient second
        "model_name": MODEL_GROUP,
        "litellm_params": {
            "model": "openrouter/openai/gpt-5.6-terra",
            "api_key": "os.environ/OPENROUTER_API_KEY",
        },
    },
    {  # the judge: never a routing candidate
        "model_name": JUDGE_GROUP,
        "litellm_params": {
            "model": "openrouter/openai/gpt-5.6-terra",
            "api_key": "os.environ/OPENROUTER_API_KEY",
        },
    },
]
JUDGE = LiteLLMJudge(JUDGE_GROUP, timeout=60)
CLASSIFIER_ROUTING_PLUGIN = ClassifierRoutingPlugin(
    judge=JUDGE,
    judge_model=JUDGE_GROUP,
    base_threshold=0.5,
    threshold_step=0.1,
    max_output_tokens=512,
)


async def main() -> None:
    """Route two requests of different difficulty and show where each went."""
    router = Router(model_list=MODEL_LIST, plugins=[CLASSIFIER_ROUTING_PLUGIN])
    JUDGE.router = router  # the judge call goes back through this Router
    litellm.callbacks.append(CLASSIFIER_ROUTING_PLUGIN)
    try:
        for task in (
            "Rewrite in a friendlier tone: Your request was denied.",
            "Prove that the harmonic series diverges.",
        ):
            metadata: dict[str, object] = {}
            response = await router.acompletion(
                model=MODEL_GROUP,
                messages=[{"role": "user", "content": task}],
                max_tokens=200,
                metadata=metadata,
            )
            signals = metadata.get("routing_plugin_signals")
            decision = signals.get("switchyard") if isinstance(signals, dict) else None
            print("Task:", task)
            print("  Selected model:", response.model)
            print("  Evidence:", decision.get("evidence") if isinstance(decision, dict) else None)
    finally:
        litellm.callbacks.remove(CLASSIFIER_ROUTING_PLUGIN)


if __name__ == "__main__":
    asyncio.run(main())
