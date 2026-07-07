# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemini generateContent stream adapter re-export."""

from typing import TypeAlias

from switchyard_rust.core import ChatResponse as _ChatResponse
from switchyard_rust.core import ChatResponseStream as _ChatResponseStream

GeminiChatResponse: TypeAlias = _ChatResponse
GeminiStreamingChatResponse: TypeAlias = _ChatResponse
GeminiResponseStream: TypeAlias = _ChatResponseStream

__all__ = [
    "GeminiChatResponse",
    "GeminiResponseStream",
    "GeminiStreamingChatResponse",
]
