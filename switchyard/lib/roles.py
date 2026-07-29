# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python backend role class and translated-response aliases.

Request-side and response-side components are plain objects with async
``process(...)`` methods. The backend remains nominal because it owns upstream
transport behavior and request-format support; native implementations register
with the same Python role class.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Mapping
from typing import TypeAlias

from anthropic.types import Message as AnthropicMessage
from anthropic.types import RawMessageStreamEvent
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseStreamEvent

from switchyard_rust.core import LLMBackend

# The final translated response surfaced by TranslationEngine.translate().
# Union covers all three formats x (non-streaming | streaming) plus the
# dict-returning converters.
TranslatedStream: TypeAlias = (
    AsyncIterable[ChatCompletionChunk]
    | AsyncIterable[RawMessageStreamEvent]
    | AsyncIterable[ResponseStreamEvent]
    | AsyncIterable[Mapping[str, object]]
    | AsyncIterable[str]
)

TranslatedResponse: TypeAlias = (
    ChatCompletion
    | OpenAIResponse
    | AnthropicMessage
    | Mapping[str, object]
    | TranslatedStream
)

__all__ = [
    "LLMBackend",
    "TranslatedResponse",
    "TranslatedStream",
]
