# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust-backed response values and stream adapters."""

from switchyard.lib.chat_response.anthropic import (
    AnthropicChatResponse,
    AnthropicResponseStream,
    AnthropicStreamingChatResponse,
)
from switchyard.lib.chat_response.gemini import (
    GeminiChatResponse,
    GeminiResponseStream,
    GeminiStreamingChatResponse,
)
from switchyard.lib.chat_response.openai_chat import (
    CompletionChatResponse,
    ResponseStream,
    StreamingChatResponse,
)
from switchyard.lib.chat_response.openai_responses import (
    ResponsesApiChatResponse,
    ResponsesApiStream,
    ResponsesApiStreamingChatResponse,
)
from switchyard_rust.core import ChatResponse, ChatResponseStream, ChatResponseType

AnyResponseStream = ChatResponseStream

__all__ = [
    "AnyResponseStream",
    "AnthropicChatResponse",
    "AnthropicResponseStream",
    "AnthropicStreamingChatResponse",
    "ChatResponse",
    "ChatResponseStream",
    "ChatResponseType",
    "CompletionChatResponse",
    "GeminiChatResponse",
    "GeminiResponseStream",
    "GeminiStreamingChatResponse",
    "ResponsesApiChatResponse",
    "ResponsesApiStream",
    "ResponsesApiStreamingChatResponse",
    "ResponseStream",
    "StreamingChatResponse",
]
