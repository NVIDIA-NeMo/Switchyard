# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Gemini generateContent FastAPI endpoint."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.core import ChatRequest


class _RecordingSwitchyard:
    """Fake chain that records requests and replays a canned Gemini result."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.requests: list[ChatRequest] = []

    async def call(self, request: ChatRequest, *, ctx: object | None = None) -> Any:
        self.requests.append(request)
        return self.result


def _gemini_result() -> dict[str, Any]:
    return {
        "candidates": [{
            "content": {"parts": [{"text": "OK"}], "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        "modelVersion": "gemini-2.5-flash",
        "responseId": "resp-1",
    }


def test_generate_content_route_injects_model_and_dispatches() -> None:
    switchyard = _RecordingSwitchyard(_gemini_result())
    app = build_switchyard_app(switchyard)  # type: ignore[arg-type]

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1beta/models/gemini-2.5-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )

    assert response.status_code == 200
    assert response.json()["candidates"][0]["content"]["parts"][0]["text"] == "OK"
    request = switchyard.requests[0]
    assert request.request_type.value == "gemini"
    assert request.model == "gemini-2.5-flash"
    assert request.body["stream"] is False


def test_stream_generate_content_frames_chunks_as_sse() -> None:
    async def chunks() -> AsyncIterator[dict[str, Any]]:
        yield {"candidates": [{"content": {"parts": [{"text": "Hel"}], "role": "model"}}]}
        yield {
            "candidates": [{
                "content": {"parts": [{"text": "lo"}], "role": "model"},
                "finishReason": "STOP",
            }],
        }

    switchyard = _RecordingSwitchyard(chunks())
    app = build_switchyard_app(switchyard)  # type: ignore[arg-type]

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    text = "".join(
        part.get("text", "")
        for frame in frames
        for part in frame["candidates"][0].get("content", {}).get("parts", [])
    )
    assert text == "Hello"
    # Gemini streams end without a [DONE] sentinel.
    assert "[DONE]" not in response.text
    assert switchyard.requests[0].body["stream"] is True


def test_empty_contents_is_rejected_with_400() -> None:
    switchyard = _RecordingSwitchyard(_gemini_result())
    app = build_switchyard_app(switchyard)  # type: ignore[arg-type]

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1beta/models/gemini-2.5-flash:generateContent",
            json={"contents": []},
        )

    assert response.status_code == 400
    assert switchyard.requests == []


def test_unknown_action_verb_is_not_routed() -> None:
    switchyard = _RecordingSwitchyard(_gemini_result())
    app = build_switchyard_app(switchyard)  # type: ignore[arg-type]

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1beta/models/gemini-2.5-flash:countTokens",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )

    assert response.status_code == 404
    assert switchyard.requests == []


def test_malformed_json_body_maps_to_invalid_body_envelope() -> None:
    switchyard = _RecordingSwitchyard(_gemini_result())
    app = build_switchyard_app(switchyard)  # type: ignore[arg-type]

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1beta/models/gemini-2.5-flash:generateContent",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_body"
