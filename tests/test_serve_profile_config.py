# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for serving a v2 profile config via `serve --config`."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from switchyard.cli.switchyard_cli import _cmd_serve_profile_config, _profile_config_route_table
from switchyard.lib.processors.routing_log_response_processor import (
    RoutingLogResponseProcessor,
)
from switchyard.server.switchyard_app import build_switchyard_app

_PYTHON_CONFIG = """
targets:
  weak:
    model: provider/weak
    format: openai
    base_url: http://127.0.0.1:9/weak/v1
    api_key: test-key

profiles:
  smart:
    type: header-routing
    strong: weak
    weak: weak
"""


def _mixed_config(base_url: str) -> str:
    return f"""
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: "{base_url}/strong/v1"
    api_key: test-key
  weak:
    model: provider/weak
    format: openai
    base_url: "{base_url}/weak/v1"
    api_key: test-key

profiles:
  direct:
    type: passthrough
    target: weak
  smart:
    type: header-routing
    strong: strong
    weak: weak
"""


class _MockOpenAIServer(ThreadingHTTPServer):
    calls: list[dict[str, Any]]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _MockOpenAIHandler)
        self.calls = []

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    server: _MockOpenAIServer

    def do_POST(self) -> None:
        content_length = int(self.headers.get("content-length", "0"))
        raw_body = self.rfile.read(content_length)
        body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        self.server.calls.append({"path": self.path, "body": body})
        response = {
            "id": "chatcmpl-serve-profile-config",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 3,
                "total_tokens": 11,
                "prompt_tokens_details": {
                    "cached_tokens": 2,
                    "cache_creation_tokens": 1,
                },
            },
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def mock_openai_server() -> _MockOpenAIServer:
    try:
        server = _MockOpenAIServer()
    except PermissionError as exc:
        pytest.skip(f"loopback socket binding is unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _write(tmp_path: Path, text: str, name: str = "profiles.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_serve_config_uses_fastapi_profile_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _PYTHON_CONFIG)
    args = _serve_args(path, routing_log_file=tmp_path / "routing_requests.jsonl")
    captured: dict[str, Any] = {}

    def capture_build_and_serve(
        _args: argparse.Namespace,
        table: Any,
        inbound_default: str = "openai",
        disable_backend_streaming: bool = False,
        extra_endpoints: list[Any] | None = None,
    ) -> None:
        captured["models"] = table.registered_models()
        captured["inbound_default"] = inbound_default
        captured["disable_backend_streaming"] = disable_backend_streaming
        captured["extra_endpoints"] = extra_endpoints

    monkeypatch.setattr(
        "switchyard.cli.switchyard_cli.build_and_serve",
        capture_build_and_serve,
    )

    _cmd_serve_profile_config(args)

    assert captured["models"] == ["smart", "weak", "provider/weak"]
    assert captured["inbound_default"] == "both"


def test_profile_config_route_table_serves_mixed_profiles(
    mock_openai_server: _MockOpenAIServer,
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _mixed_config(mock_openai_server.base_url))
    routing_log = RoutingLogResponseProcessor(tmp_path / "routing_requests.jsonl")
    table = _profile_config_route_table(
        str(path), extra_response_processors=[routing_log],
    )

    assert table.registered_models() == [
        "direct",
        "smart",
        "strong",
        "provider/strong",
        "weak",
        "provider/weak",
    ]

    with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
        models_response = client.get("/v1/models")
        assert models_response.status_code == 200
        assert models_response.json()["model_pool"] == [
            "direct",
            "smart",
            "strong",
            "provider/strong",
            "weak",
            "provider/weak",
        ]

        responses = {
            path: client.post(
                path,
                headers={
                    "x-switchyard-tier": "strong",
                    "proxy_x_session_id": "profile-trial-1",
                },
                json=body,
            )
            for path, body in (
                (
                    "/v1/chat/completions",
                    {
                        "model": "smart",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                ),
                (
                    "/v1/messages",
                    {
                        "model": "smart",
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                ),
                ("/v1/responses", {"model": "smart", "input": "hi"}),
            )
        }
        stats_response = client.get(
            "/v1/routing/session-stats", params={"session_id": "profile-trial-1"},
        )

    assert [response.status_code for response in responses.values()] == [200, 200, 200]
    assert stats_response.status_code == 200
    assert stats_response.json() == {
        "session_id": "profile-trial-1",
        "total_calls": 3,
        "total_prompt_tokens": 24,
        "total_cached_tokens": 6,
        "total_cache_creation_tokens": 3,
        "total_completion_tokens": 9,
        "models": {
            "provider/strong": {
                "calls": 3, "prompt_tokens": 24, "cached_tokens": 6,
                "cache_creation_tokens": 3, "completion_tokens": 9,
            }
        },
    }
    assert [call["path"] for call in mock_openai_server.calls] == [
        "/strong/v1/chat/completions",
        "/strong/v1/chat/completions",
        "/strong/v1/chat/completions",
    ]
    assert responses["/v1/chat/completions"].json()["model"] == "provider/strong"
    assert responses["/v1/messages"].json()["content"][0]["text"] == "ok"
    assert (
        responses["/v1/responses"].json()["output"][0]["content"][0]["text"]
        == "ok"
    )


def _serve_args(path: Path, *, routing_log_file: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(path),
        routing_profiles=None,
        inbound=None,
        reload=False,
        workers=1,
        intake_enabled=False,
        intake_base_url=None,
        intake_workspace=None,
        intake_api_key=None,
        intake_target_url=None,
        routing_log_file=str(routing_log_file) if routing_log_file is not None else None,
        host="127.0.0.1",
        port=4000,
    )
