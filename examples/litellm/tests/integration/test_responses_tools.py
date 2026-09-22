# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


@pytest.mark.parametrize(
    "optional_params", [{}, {"tools": []}, {"tools": [], "tool_choice": "auto"}]
)
def test_tool_free_responses_do_not_send_empty_chat_tools(
    optional_params: ResponsesAPIOptionalRequestParams,
) -> None:
    """Omit empty tools and tool_choice while preserving the user message."""
    request = (
        LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="nvidia_nim/example/model",
            input="Reply hello.",
            responses_api_request=optional_params,
            custom_llm_provider="nvidia_nim",
        )
    )

    assert "tools" not in request
    assert "tool_choice" not in request
    assert request["messages"] == [{"role": "user", "content": "Reply hello."}]


def test_responses_bridge_preserves_nonempty_tools_and_named_choice() -> None:
    """Preserve nonempty function tools and named choices in Chat format."""
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    request = (
        LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="nvidia_nim/example/model",
            input="Check the weather.",
            responses_api_request={
                "tools": [{"type": "function", "name": "weather", "parameters": parameters}],
                "tool_choice": {"type": "function", "name": "weather"},
            },
            custom_llm_provider="nvidia_nim",
        )
    )

    assert len(request["tools"]) == 1
    assert request["tools"][0]["type"] == "function"
    assert request["tools"][0]["function"]["name"] == "weather"
    assert request["tools"][0]["function"]["parameters"] == parameters
    assert request["tool_choice"] == {"type": "function", "function": {"name": "weather"}}
