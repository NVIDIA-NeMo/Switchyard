# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemini generateContent request compatibility alias."""

from typing import TypeAlias

from switchyard_rust.core import ChatRequest as _ChatRequest

GeminiChatRequest: TypeAlias = _ChatRequest

__all__ = ["GeminiChatRequest"]
