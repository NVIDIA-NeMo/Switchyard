# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for serving a v2 profile config via `serve --config`.

`dry_run` exercises the full load -> resolve -> build -> registry path of the
Rust profile server without binding a socket. Files with Python-defined profiles
use the Python FastAPI adapter so both profile implementations are routable on
the same paths.
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from switchyard.cli.switchyard_cli import _cmd_serve_profile_config, _profile_config_route_table
from switchyard.lib.profiles.loader import python_profile_ids
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.core import SwitchyardConfigError
from switchyard_rust.server import run_profile_server

_RUST_CONFIG = """
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: http://127.0.0.1:9/strong/v1
    api_key: test-key
  weak:
    model: provider/weak
    format: openai
    base_url: http://127.0.0.1:9/weak/v1
    api_key: test-key

profiles:
  fast:
    type: passthrough
    target: weak
  smart-stage-router:
    type: stage_router
    capable: strong
    efficient: weak
    fallback_target_on_evict: strong
    picker: capable_first
    confidence_threshold: 0.7
"""

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


def test_dry_run_validates_rust_profiles_and_direct_targets(tmp_path: Path) -> None:
    # passthrough + stage_router + the two targets (directly addressable).
    path = _write(tmp_path, _RUST_CONFIG)
    # dry_run loads, resolves, and builds the registry, then returns without
    # binding a socket. No exception == a servable config.
    run_profile_server(str(path), dry_run=True)


def test_dry_run_rejects_invalid_config(tmp_path: Path) -> None:
    path = _write(tmp_path, "profiles:\n  bad:\n    type: passthrough\n    target: ghost\n")
    with pytest.raises(SwitchyardConfigError):
        run_profile_server(str(path), dry_run=True)


def test_shipped_example_config_is_servable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Guards examples/profiles.yaml from rotting; dry_run never connects upstream.
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    example = Path(__file__).resolve().parents[1] / "examples" / "profiles.yaml"
    run_profile_server(str(example), dry_run=True)


def test_python_profile_ids_classifies_config(tmp_path: Path) -> None:
    assert python_profile_ids(_write(tmp_path, _PYTHON_CONFIG)) == ["smart"]
    assert python_profile_ids(_write(tmp_path, _RUST_CONFIG, name="rust.yaml")) == []


def test_serve_config_uses_fastapi_for_python_profiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write(tmp_path, _PYTHON_CONFIG)
    args = _serve_args(path)
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
    stderr = capsys.readouterr().err
    assert "warning: Python-defined profile serving is deprecated." in stderr
    assert "Python FastAPI adapter" in stderr
    assert "Python profile(s): smart." in stderr


def test_profile_config_route_table_serves_mixed_profiles(
    mock_openai_server: _MockOpenAIServer,
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _mixed_config(mock_openai_server.base_url))
    table = _profile_config_route_table(str(path))

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
                headers={"x-switchyard-tier": "strong"},
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

    assert [response.status_code for response in responses.values()] == [200, 200, 200]
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


def test_serve_config_fails_closed_when_python_profile_inspection_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _RUST_CONFIG)

    def fail_inspection(_path: str) -> list[str]:
        raise RuntimeError("inspection exploded")

    monkeypatch.setattr(
        "switchyard.lib.profiles.loader.python_profile_ids",
        fail_inspection,
    )

    with pytest.raises(SystemExit, match="failed to inspect.*inspection exploded"):
        _cmd_serve_profile_config(_serve_args(path))


def _serve_args(path: Path) -> argparse.Namespace:
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
        intake_nvdataflow_project=None,
        host="127.0.0.1",
        port=4000,
    )
