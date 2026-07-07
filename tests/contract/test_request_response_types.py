# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction + dispatch contract for Rust-backed chat request aliases.

The Python concrete request subclasses were collapsed into one Rust-backed
``ChatRequest`` type. The legacy provider-specific names remain importable as
aliases only; provider dispatch is keyed by ``request_type``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import pytest

from switchyard.lib.chat_request.anthropic import AnthropicChatRequest
from switchyard.lib.chat_request.base import ChatRequest
from switchyard.lib.chat_request.gemini import GeminiChatRequest
from switchyard.lib.chat_request.openai_chat import OpenAIChatRequest
from switchyard.lib.chat_request.openai_responses import ResponsesChatRequest
from switchyard_rust.core import ChatRequestType, request_type_matches

RequestAlias: TypeAlias = type[ChatRequest]
RequestFactory: TypeAlias = Callable[[dict], ChatRequest]

OPENAI_CHAT_BODY = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "ping"}],
}

OPENAI_RESPONSES_BODY = {
    "model": "gpt-4",
    "input": "ping",
}

ANTHROPIC_BODY = {
    "model": "claude-3-haiku",
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 16,
}

GEMINI_BODY = {
    "model": "gemini-2.5-flash",
    "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
}


REQUEST_CASES: list[tuple[str, RequestAlias, RequestFactory, object, dict]] = [
    (
        "OpenAIChatRequest",
        OpenAIChatRequest,
        ChatRequest.openai_chat,
        ChatRequestType.OPENAI_CHAT,
        OPENAI_CHAT_BODY,
    ),
    (
        "ResponsesChatRequest",
        ResponsesChatRequest,
        ChatRequest.openai_responses,
        ChatRequestType.OPENAI_RESPONSES,
        OPENAI_RESPONSES_BODY,
    ),
    (
        "AnthropicChatRequest",
        AnthropicChatRequest,
        ChatRequest.anthropic,
        ChatRequestType.ANTHROPIC,
        ANTHROPIC_BODY,
    ),
    (
        "GeminiChatRequest",
        GeminiChatRequest,
        ChatRequest.gemini,
        ChatRequestType.GEMINI,
        GEMINI_BODY,
    ),
]


@pytest.mark.parametrize(
    ("alias_name", "alias", "_factory", "_request_type", "_body"),
    REQUEST_CASES,
)
def test_request_legacy_names_are_aliases(
    alias_name: str,
    alias: RequestAlias,
    _factory: RequestFactory,
    _request_type: object,
    _body: dict,
) -> None:
    """Provider-specific request names remain aliases for the Rust type."""
    assert alias is ChatRequest, f"{alias_name} must alias ChatRequest"


@pytest.mark.parametrize(
    ("_alias_name", "_alias", "factory", "_request_type", "body"),
    REQUEST_CASES,
)
def test_request_constructible_with_factory(
    _alias_name: str,
    _alias: RequestAlias,
    factory: RequestFactory,
    _request_type: object,
    body: dict,
) -> None:
    """Each request factory must round-trip the raw body.

    Platform's bridge layer constructs requests from raw dict payloads coming
    off the IGW wire. The Rust-backed type exposes provider factories instead
    of the deleted ``cls(body=...)`` Python subclass constructors.
    """
    req = factory(body)
    assert req.body == body, "ChatRequest did not round-trip body"


@pytest.mark.parametrize(
    ("alias_name", "alias", "factory", "request_type", "body"),
    REQUEST_CASES,
)
def test_request_type_dispatch_works(
    alias_name: str,
    alias: RequestAlias,
    factory: RequestFactory,
    request_type: object,
    body: dict,
) -> None:
    """The alias-backed request must expose the Rust request type for dispatch."""
    req = factory(body)

    assert isinstance(req, alias), f"isinstance({alias_name}, ChatRequest alias) is false"
    assert isinstance(req, ChatRequest), f"{alias_name} is not a ChatRequest alias"
    assert request_type_matches(req, request_type), f"{alias_name} request_type is wrong"
