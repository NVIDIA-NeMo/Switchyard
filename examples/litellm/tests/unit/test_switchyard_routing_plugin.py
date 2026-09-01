# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from typing import Any

import pytest
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import RoutingContext
from switchyard_litellm import SwitchyardRoutingPlugin
from switchyard_litellm.plugins.request_rewrite import build_request_patch
from switchyard_litellm.plugins.switchyard_routing_plugin import _request

from switchyard.libsy import Step, TaskClassifierConfig, algorithms

SOL = "openrouter/openai/gpt-5.6-sol"
TERRA = "openrouter/openai/gpt-5.6-terra"


def routing_context(
    messages: list[dict[str, Any]],
    *,
    candidates: list[str] | None = None,
) -> RoutingContext:
    """Build the exact LiteLLM boundary object passed to a routing plugin."""
    return RoutingContext(
        raw_messages=messages,
        structured_messages=messages,
        candidate_models=candidates or [SOL, TERRA],
        metadata={"tenant": "test"},
        signals={"earlier-plugin": {"allowed": True}},
    )


def stage_plugin(**kwargs: object) -> SwitchyardRoutingPlugin:
    """Build the supported signal-only Stage configuration."""
    return SwitchyardRoutingPlugin(
        algorithms.stage_router(
            SOL,
            TERRA,
            picker="efficient_first",
            confidence_threshold=0.5,
            recent_window=3,
            **kwargs,
        )
    )


async def test_stage_converts_tool_history_and_narrows_to_capable_candidate() -> None:
    context = routing_context(
        [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "developer", "content": [{"type": "text", "text": "Be concise."}]},
            {"role": "user", "content": "Fix the failing tests."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": '{"command":"pytest"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "fatal runtime error: out of memory",
            },
        ],
        candidates=[SOL, TERRA, SOL],
    )

    result = await stage_plugin().run(context)

    assert result is context
    assert result.candidate_models == [SOL, SOL]
    assert result.signals == {
        "earlier-plugin": {"allowed": True},
        "switchyard": {
            "selected_model_id": SOL,
            "fallback_models": [TERRA],
        },
    }


async def test_litellm_conversion_preserves_stage_tool_signal_input() -> None:
    """Match direct libsy and LiteLLM-plugin routing over the same tool history."""
    litellm_messages = [
        {"role": "user", "content": "Fix the failing tests."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": '{"command":"pytest"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "fatal runtime error: out of memory",
        },
    ]
    original_messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Fix the failing tests."}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "Bash",
                    "arguments": {"command": "pytest"},
                }
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "text",
                            "text": "fatal runtime error: out of memory",
                        }
                    ],
                    "is_error": None,
                }
            ],
        },
    ]
    original_request: dict[str, object] = {
        "model": "auto",
        "messages": original_messages,
    }

    direct_outcome = None
    direct_algorithm = algorithms.stage_router(
        SOL,
        TERRA,
        picker="efficient_first",
        confidence_threshold=0.5,
        recent_window=3,
    )
    async for step in direct_algorithm.run_stream(original_request):
        match step:
            case Step.Done(outcome):
                direct_outcome = outcome

    # ToolSignals reads only normalized messages, so exact equality protects every signal input.
    assert _request(litellm_messages)["messages"] == original_messages
    assert direct_outcome is not None
    assert direct_outcome.selected_model_ids == [SOL, TERRA]

    context = routing_context(litellm_messages)
    await stage_plugin().run(context)

    assert context.signals["switchyard"] == {
        "selected_model_id": direct_outcome.selected_model_ids[0],
        "fallback_models": direct_outcome.selected_model_ids[1:],
    }


async def test_stage_selects_efficient_candidate_without_tool_signals() -> None:
    context = routing_context([{"role": "user", "content": "Say hello."}])

    await stage_plugin().run(context)

    assert context.candidate_models == [TERRA]
    assert context.signals["switchyard"] == {
        "selected_model_id": TERRA,
        "fallback_models": [SOL],
    }


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        (
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Bash", "arguments": "not-json"},
                        }
                    ],
                }
            ],
            "valid JSON object",
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.test/x"}}
                    ],
                }
            ],
            "content.*not supported",
        ),
        ([{"role": "function", "content": "legacy"}], "role.*not supported"),
    ],
)
async def test_unsupported_structured_messages_fail_closed(
    messages: list[dict[str, Any]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        await stage_plugin().run(routing_context(messages))


async def test_selection_outside_current_litellm_pool_fails_closed() -> None:
    plugin = SwitchyardRoutingPlugin(algorithms.random(["openrouter/openai/not-allowed"]))

    with pytest.raises(ValueError, match="not in LiteLLM's candidate pool"):
        await plugin.run(routing_context([{"role": "user", "content": "hello"}]))


async def test_classifier_backed_algorithm_fails_on_intermediate_model_call() -> None:
    plugin = SwitchyardRoutingPlugin(
        algorithms.llm_task_classifier(
            "openrouter/openai/gpt-5.6-judge",
            TERRA,
            SOL,
            config=TaskClassifierConfig(0.5),
        )
    )

    with pytest.raises(ValueError, match="intermediate model call"):
        await plugin.run(routing_context([{"role": "user", "content": "hello"}]))


async def test_algorithm_that_produces_a_response_fails_closed() -> None:
    with pytest.raises(ValueError, match="produced a response while routing"):
        await SwitchyardRoutingPlugin(algorithms.noop()).run(
            routing_context([{"role": "user", "content": "hello"}])
        )


async def test_stage_request_rewrite_is_applied_after_litellm_selects_a_deployment() -> None:
    messages = [
        {"role": "user", "content": "Fix the failing tests."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": '{"command":"pytest"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "out of memory"},
    ]
    context = routing_context(messages)
    plugin = stage_plugin(
        capable_system_prompt="Use the capable tier.",
        escalation_note="The efficient tier failed.",
    )

    await plugin.run(context)
    rewritten = await plugin.async_pre_call_deployment_hook(
        {
            "messages": messages,
            "temperature": 0.2,
            "tools": [{"type": "function", "function": {"name": "Bash"}}],
            "metadata": {"routing_plugin_signals": context.signals},
        },
        None,
    )

    assert isinstance(plugin, CustomLogger)
    assert rewritten is not None
    assert rewritten["messages"] == [
        {"role": "system", "content": "Use the capable tier."},
        {"role": "user", "content": "Fix the failing tests."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": '{"command":"pytest"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "out of memory"},
        {"role": "user", "content": "The efficient tier failed."},
    ]
    assert rewritten["temperature"] == 0.2
    assert rewritten["tools"] == [{"type": "function", "function": {"name": "Bash"}}]
    assert "request_patch" not in rewritten["metadata"]["routing_plugin_signals"]["switchyard"]
    assert "request_patch" in context.signals["switchyard"]


def test_request_patch_maps_supported_switchyard_request_overrides() -> None:
    original = {
        "model": "auto",
        "instructions": [],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "tools": [],
        "tool_choice": None,
        "sampling": {"temperature": None, "top_p": None, "top_k": None},
        "output": {"max_output_tokens": None, "response_format": None},
        "reasoning": {"effort": None, "raw": None},
        "stream": False,
        "extensions": {"fields": {}},
        "preservation": {"requests": {}, "responses": {}},
    }
    rewritten = {
        **original,
        "model": SOL,
        "instructions": [
            {
                "role": "developer",
                "content": [{"type": "text", "text": "Use the selected tier."}],
            }
        ],
        "tools": [
            {
                "name": "search",
                "description": "Search documents",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            }
        ],
        "tool_choice": {"type": "tool", "data": {"name": "search"}},
        "sampling": {"temperature": 0.3, "top_p": 0.8, "top_k": 20},
        "output": {
            "max_output_tokens": 512,
            "response_format": {"type": "json_object"},
        },
        "reasoning": {"effort": "high", "raw": None},
        "stream": True,
        "extensions": {
            "fields": {
                "parallel_tool_calls": False,
                "stop_sequences": ["END"],
            }
        },
    }

    assert build_request_patch(original, rewritten, selected_model_id=SOL) == {
        "set": {
            "messages": [
                {"role": "developer", "content": "Use the selected tier."},
                {"role": "user", "content": "Hello"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search documents",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": True,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "search"}},
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 20,
            "max_completion_tokens": 512,
            "response_format": {"type": "json_object"},
            "reasoning_effort": "high",
            "stream": True,
            "parallel_tool_calls": False,
            "stop": ["END"],
        },
        "remove": ["max_tokens"],
    }


@pytest.mark.parametrize(
    "update",
    [
        {"reasoning": {"effort": None, "raw": {"budget_tokens": 512}}},
        {"extensions": {"fields": {"api_key": "must-not-be-forwarded"}}},
        {"preservation": {"requests": {"openai_chat": {}}, "responses": {}}},
    ],
)
def test_request_patch_rejects_unsafe_or_unrepresentable_overrides(
    update: dict[str, object],
) -> None:
    original = {
        "model": "auto",
        "instructions": [],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "tools": [],
        "tool_choice": None,
        "sampling": {"temperature": None, "top_p": None, "top_k": None},
        "output": {"max_output_tokens": None, "response_format": None},
        "reasoning": {"effort": None, "raw": None},
        "stream": False,
        "extensions": {"fields": {}},
        "preservation": {"requests": {}, "responses": {}},
    }
    rewritten = {**original, "model": SOL, **update}

    with pytest.raises(ValueError, match="cannot safely apply"):
        build_request_patch(original, rewritten, selected_model_id=SOL)


class EmptyAlgorithm:
    """Algorithm-shaped test double whose stream violates the terminal-step contract."""

    async def run_stream(self, request: dict[str, object]) -> AsyncIterator[object]:
        if request:
            return
        yield object()


async def test_algorithm_stream_without_terminal_outcome_fails_closed() -> None:
    plugin = SwitchyardRoutingPlugin(EmptyAlgorithm())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="without a routing outcome"):
        await plugin.run(routing_context([{"role": "user", "content": "hello"}]))
