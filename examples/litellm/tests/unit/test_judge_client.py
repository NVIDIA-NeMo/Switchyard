# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest
from switchyard_litellm.plugins.judge import (
    LiteLLMJudge,
    is_judge_call,
    judge_completion_kwargs,
    judge_messages,
    judge_response,
)

from switchyard.libsy import LlmResponse

JUDGE_REQUEST: dict[str, Any] = {
    "model": "auto",
    "instructions": [
        {"role": "system", "content": [{"type": "text", "text": "Forecast success."}]}
    ],
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Prove it."}]}],
    "output": {
        "max_output_tokens": 256,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": {}},
        },
    },
    "sampling": {"temperature": 0, "top_p": None, "top_k": None},
}


def completion(
    content: str | None, finish_reason: str = "stop", model: str = "judge-deployment"
) -> Any:
    """Mimic the attributes of a LiteLLM ModelResponse the client reads."""
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
        ],
    )


def test_judge_messages_flattens_instructions_then_messages() -> None:
    assert judge_messages(JUDGE_REQUEST) == [
        {"role": "system", "content": "Forecast success."},
        {"role": "user", "content": "Prove it."},
    ]


def test_judge_messages_rejects_non_text_content() -> None:
    request = {"messages": [{"role": "user", "content": [{"type": "image", "url": "x"}]}]}
    with pytest.raises(ValueError, match="text only"):
        judge_messages(request)


def test_completion_kwargs_carry_limit_schema_and_sampling() -> None:
    kwargs = judge_completion_kwargs(JUDGE_REQUEST, "switchyard-judge")

    assert kwargs["model"] == "switchyard-judge"
    assert "metadata" not in kwargs
    assert kwargs["max_completion_tokens"] == 256
    assert "max_tokens" not in kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["temperature"] == 0
    assert "top_p" not in kwargs


def test_completion_kwargs_can_use_max_tokens_for_servers_without_the_newer_field() -> None:
    kwargs = judge_completion_kwargs(JUDGE_REQUEST, "judge", max_tokens_field="max_tokens")
    assert kwargs["max_tokens"] == 256
    assert "max_completion_tokens" not in kwargs


def test_completion_kwargs_reject_unknown_limit_field() -> None:
    with pytest.raises(ValueError, match="max_tokens_field"):
        judge_completion_kwargs(JUDGE_REQUEST, "judge", max_tokens_field="max_len")


def test_judge_response_wraps_content_and_maps_finish_reason() -> None:
    response = judge_response(completion('{"p_solve": 0.9}', "length"), "judge")

    assert isinstance(response, LlmResponse.Agg)


def test_judge_response_passes_empty_content_through_for_conservative_handling() -> None:
    assert isinstance(judge_response(completion(None), "judge"), LlmResponse.Agg)


def test_judge_response_requires_a_choice() -> None:
    with pytest.raises(ValueError, match="no choices"):
        judge_response(SimpleNamespace(model="m", choices=[]), "judge")


async def test_litellm_judge_calls_through_the_given_router() -> None:
    seen: dict[str, Any] = {}

    class FakeRouter:
        async def acompletion(self, **kwargs: Any) -> Any:
            seen.update(kwargs, judge_call=is_judge_call())
            return completion('{"p_solve": 0.2}')

    judge = LiteLLMJudge("switchyard-judge", router=FakeRouter(), timeout=20)
    assert not is_judge_call()
    response = await judge(JUDGE_REQUEST, "switchyard-judge")

    assert isinstance(response, LlmResponse.Agg)
    assert seen["model"] == "switchyard-judge"
    assert seen["judge_call"] is True
    assert not is_judge_call()
    assert seen["timeout"] == 20
    assert seen["max_completion_tokens"] == 256


async def test_litellm_judge_falls_back_to_litellm_acompletion_without_a_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from switchyard_litellm.plugins import judge as judge_module

    seen: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        seen.update(kwargs, judge_call=is_judge_call())
        return completion('{"p_solve": 0.8}')

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(judge_module, "_proxy_router", lambda: None)
    judge = LiteLLMJudge("openai/gpt-5.4-mini", api_base="https://example.invalid/v1", api_key="k")

    await judge(JUDGE_REQUEST, "openai/gpt-5.4-mini")

    assert seen["api_base"] == "https://example.invalid/v1"
    assert seen["judge_call"] is True
    assert "metadata" not in seen


def test_litellm_judge_requires_a_model() -> None:
    with pytest.raises(ValueError, match="judge model"):
        LiteLLMJudge("")
