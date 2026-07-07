# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility aliases for Rust-backed request values."""

from switchyard.lib.chat_request.anthropic import AnthropicChatRequest
from switchyard.lib.chat_request.base import ChatRequest, ChatRequestType
from switchyard.lib.chat_request.gemini import GeminiChatRequest
from switchyard.lib.chat_request.openai_chat import OpenAIChatRequest
from switchyard.lib.chat_request.openai_responses import ResponsesChatRequest

__all__ = [
    "AnthropicChatRequest",
    "ChatRequest",
    "ChatRequestType",
    "GeminiChatRequest",
    "OpenAIChatRequest",
    "ResponsesChatRequest",
]
