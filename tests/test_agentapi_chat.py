# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agentapi chat data types."""

from __future__ import annotations

from switchyard.lib.agentapi.chat import ChatRequest, ChatResponse, EnrichmentData


def test_chat_request_holds_prompt_and_model():
    req = ChatRequest(prompt="hi", model="client/model")
    assert req.prompt == "hi"
    assert req.model == "client/model"


def test_chat_response_holds_completion():
    assert ChatResponse(completion="done").completion == "done"


def test_enrichment_data_defaults_to_none():
    e = EnrichmentData()
    assert e.session_id is None
    assert e.extra_metadata is None
