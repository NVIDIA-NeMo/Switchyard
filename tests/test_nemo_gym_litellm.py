# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

litellm = pytest.importorskip("litellm")

from litellm import ResponsesAPIResponse  # noqa: E402

BIG = "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b"
SMALL = "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
ROOT = Path(__file__).resolve().parents[1]
CALLBACK = ROOT / "benchmark/nemo_gym/gym_routing_plugin.py"


@pytest.fixture
def callback(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.delenv("NEMO_GYM_LITELLM_RESULTS", raising=False)
    spec = importlib.util.spec_from_file_location("switchyard_gym_routing_plugin_test", CALLBACK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response(
    *,
    response_id: str = "response-1",
    status: str = "incomplete",
    usage: dict[str, Any] | None = None,
) -> ResponsesAPIResponse:
    response = ResponsesAPIResponse(
        id=response_id,
        created_at=1700000000,
        object="response",
        model=SMALL,
        output=[],
        status=status,
        incomplete_details={"reason": "max_output_tokens"} if status == "incomplete" else None,
        usage=usage,
    )
    response._hidden_params = {"model_id": "deployment-1", "custom_llm_provider": "nvidia_nim"}
    return response


def test_response_payload_preserves_wire_identity_and_unknown_usage_details(
    callback: ModuleType,
) -> None:
    response = _response(
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": None,
            "output_tokens_details": None,
        }
    )

    payload = callback.response_payload(response)

    assert payload["id"] == "response-1"
    assert payload["status"] == "incomplete"
    assert payload["incomplete_details"] == {"reason": "max_output_tokens"}
    assert payload["usage"]["input_tokens"] == 10
    assert payload["usage"]["output_tokens"] == 5
    assert payload["usage"]["total_tokens"] == 15
    assert payload["usage"]["input_tokens_details"] == {"cached_tokens": None}
    assert payload["usage"]["output_tokens_details"] == {"reasoning_tokens": None}
    assert payload["_hidden_params"] == response._hidden_params

    reported = _response(
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 4},
        }
    )
    reported_payload = callback.response_payload(reported)
    assert reported_payload["usage"]["input_tokens_details"]["cached_tokens"] == 3
    assert reported_payload["usage"]["output_tokens_details"]["reasoning_tokens"] == 4
    assert callback.response_payload(_response(usage=None))["usage"] is None


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_response_payload_normalizes_litellm_reasoning(
    callback: ModuleType, finish_reason: str
) -> None:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )
    from openai.types.responses.response_reasoning_item import ResponseReasoningItem

    response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="Synthetic schema test",
        responses_api_request={},
        chat_completion_response={
            "id": "chatcmpl-reasoning-test",
            "created": 1700000000,
            "object": "chat.completion",
            "model": SMALL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "Synthetic reasoning for schema compatibility.",
                        "content": "\\boxed{B}",
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )
    before = response.model_dump()
    payload = callback.response_payload(response)
    reasoning, answer = payload["output"]
    validated = ResponseReasoningItem.model_validate(reasoning)

    assert validated.summary == []
    assert validated.content is not None
    assert validated.content[0].type == "reasoning_text"
    assert validated.content[0].text == "Synthetic reasoning for schema compatibility."
    assert reasoning["id"] == before["output"][0]["id"]
    assert reasoning["status"] == before["output"][0]["status"]
    assert answer == before["output"][1]
    assert answer["content"][0]["type"] == "output_text"
    assert answer["content"][0]["text"] == "\\boxed{B}"
    assert payload["id"] == before["id"]
    assert payload["status"] == ("completed" if finish_reason == "stop" else "incomplete")
    assert payload.get("incomplete_details") == before.get("incomplete_details")
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        assert payload["usage"][key] == before["usage"][key]
    assert response.model_dump() == before


async def test_callback_records_serving_model_without_request_content(
    callback: ModuleType, tmp_path: Path
) -> None:
    plugin = callback.GymRoutingPlugin(tmp_path)
    response = _response(
        status="completed", usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    data = {
        "model": "routed",
        "messages": [{"role": "user", "content": "secret-prompt-marker"}],
        "headers": {"Authorization": "secret-header-marker"},
        "litellm_metadata": {
            "deployment": SMALL,
            "routing_plugin_signals": {"switchyard": {"selected_model_id": BIG}},
        },
    }
    payload = await plugin.async_post_call_success_hook(data, None, response)
    records = [
        json.loads(line) for line in (tmp_path / "routed/models.jsonl").read_text().splitlines()
    ]
    assert records == [{"response_id": "response-1", "model": SMALL}]
    assert payload == callback.response_payload(response)
    assert sorted(path.name for path in (tmp_path / "routed").iterdir()) == ["models.jsonl"]


async def test_callback_rejects_missing_serving_model(callback: ModuleType, tmp_path: Path) -> None:
    plugin = callback.GymRoutingPlugin(tmp_path)
    with pytest.raises(ValueError, match="serving deployment"):
        await plugin.async_post_call_success_hook(
            {"model": "routed"}, None, _response(status="completed")
        )
    assert not list(tmp_path.iterdir())
