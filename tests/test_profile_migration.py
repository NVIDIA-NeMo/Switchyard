# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Migration tests for replacing legacy serving paths with components-v2 profiles."""

import argparse
import errno
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from switchyard.cli.route_bundle import build_route_bundle_table
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.core import ChatRequest, SwitchyardUpstreamError
from switchyard_rust.profiles import Profile, parse_profile_config_str


class _OpenAICompatStub:
    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._requests: list[dict[str, Any]] = []
        self._responses: list[tuple[int, dict[str, object]]] = []

    def __enter__(self) -> "_OpenAICompatStub":
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8")) if raw else {}
                with owner._lock:
                    owner._requests.append({"path": self.path, "body": body})
                    if owner._responses:
                        status, payload = owner._responses.pop(0)
                    else:
                        status = 200
                        payload = _completion_payload(
                            str(body.get("model", "missing-model")), self.path, "ok"
                        )

                content = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(content)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, _format: str, _line=None, _status=None, _size=None) -> None:
                return None

        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EPERM, errno.EADDRNOTAVAIL}:
                pytest.skip(f"loopback socket binding is unavailable in this sandbox: {exc}")
            raise
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("stub server is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    @property
    def requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._requests)

    def respond_json(self, status: int, payload: dict[str, object]) -> None:
        with self._lock:
            self._responses.append((status, payload))


@pytest.fixture
def openai_stub() -> Iterator[_OpenAICompatStub]:
    with _OpenAICompatStub() as stub:
        yield stub


@pytest.fixture(autouse=True)
def _disable_catalog_discovery(mocker: MockerFixture) -> None:
    """Keep legacy route-bundle tests offline and focused on configured routes."""
    mocker.patch(
        "switchyard.cli.route_bundle.fetch_model_ids",
        return_value=[],
    )


def _completion_payload(model: str, path: str, content: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-profile-migration",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "mock_path": path,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }


def _tool_call_payload(
    model: str,
    path: str,
    tool_name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    return {
        "id": "chatcmpl-profile-migration",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "mock_path": path,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_route",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }


def _chat_payload(model: str) -> dict[str, object]:
    return {"model": model, "messages": [{"role": "user", "content": "hello"}]}


def _legacy_app(route_bundle: dict[str, object]) -> TestClient:
    table = build_route_bundle_table(route_bundle)
    return TestClient(build_switchyard_app(table), raise_server_exceptions=False)


def _profile_runner(config: str, profile_id: str) -> Profile:
    return parse_profile_config_str(config).resolve().build_profile(profile_id)


def _passthrough_profile_config(base_url: str) -> str:
    return f"""
targets:
  direct:
    model: provider/direct
    format: openai
    base_url: "{base_url}/v2-direct/v1"
    api_key: test-key
profiles:
  direct-profile:
    type: passthrough
    target: direct
"""


def _random_profile_config(base_url: str, strong_probability: float) -> str:
    return f"""
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: "{base_url}/v2-random/v1"
    api_key: test-key
  weak:
    model: provider/weak
    format: openai
    base_url: "{base_url}/v2-random/v1"
    api_key: test-key
profiles:
  random-profile:
    type: random-routing
    strong: strong
    weak: weak
    strong_probability: {strong_probability}
    rng_seed: 7
"""


def _latency_profile_config(base_url: str, upstream_path: str = "v2-latency") -> str:
    return f"""
targets:
  latency:
    model: provider/latency
    format: openai
    base_url: "{base_url}/{upstream_path}/v1"
    api_key: test-key
profiles:
  latency-profile:
    type: latency-service
    latency_service_url: "http://latency.invalid"
    targets: [latency]
    max_retries: 0
"""


def _llm_routing_profile_config(base_url: str) -> str:
    return f"""
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: "{base_url}/v2-llm-routing/v1"
    api_key: test-key
  weak:
    model: provider/weak
    format: openai
    base_url: "{base_url}/v2-llm-routing/v1"
    api_key: test-key
  classifier:
    model: provider/classifier
    format: openai
    base_url: "{base_url}/v2-llm-routing-classifier/v1"
    api_key: test-key
profiles:
  llm-profile:
    type: llm-routing
    strong: strong
    weak: weak
    classifier: classifier
    profile_name: coding_agent
"""


def _stage_router_profile_config(base_url: str) -> str:
    return f"""
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: "{base_url}/v2-stage_router/v1"
    api_key: test-key
  weak:
    model: provider/weak
    format: openai
    base_url: "{base_url}/v2-stage_router/v1"
    api_key: test-key
  classifier:
    model: provider/classifier
    format: openai
    base_url: "{base_url}/v2-stage_router-classifier/v1"
    api_key: test-key
profiles:
  stage_router-profile:
    type: stage_router
    strong: strong
    weak: weak
    fallback_target_on_evict: strong
    picker: stage_router_strong_default
    confidence_threshold: 0.7
    classifier:
      model: provider/classifier
      api_key: test-key
      base_url: "{base_url}/v2-stage_router-classifier/v1"
"""


async def test_passthrough_profile_matches_legacy_single_target_serving(
    openai_stub: _OpenAICompatStub,
) -> None:
    legacy = _legacy_app(
        {
            "routes": {
                "provider/direct": {
                    "type": "model",
                    "target": {
                        "model": "provider/direct",
                        "format": "openai",
                        "base_url": f"{openai_stub.base_url}/legacy-direct/v1",
                        "api_key": "test-key",
                    },
                }
            }
        }
    )
    profile = _profile_runner(
        _passthrough_profile_config(openai_stub.base_url),
        "direct-profile",
    )

    legacy_response = legacy.post(
        "/v1/chat/completions",
        json=_chat_payload("provider/direct"),
    )
    profile_response = await profile.run(
        ChatRequest.openai_chat(_chat_payload("direct-profile")),
    )

    assert legacy_response.status_code == 200, legacy_response.text
    assert legacy_response.json()["choices"][0]["message"]["content"] == "ok"
    assert profile_response.body["choices"][0]["message"]["content"] == "ok"
    assert [call["body"]["model"] for call in openai_stub.requests] == [
        "provider/direct",
        "provider/direct",
    ]
    assert [call["path"] for call in openai_stub.requests] == [
        "/legacy-direct/v1/chat/completions",
        "/v2-direct/v1/chat/completions",
    ]


@pytest.mark.parametrize(
    ("strong_probability", "expected_model"),
    [(1.0, "provider/strong"), (0.0, "provider/weak")],
)
async def test_random_routing_profile_matches_legacy_strong_and_weak_selection(
    openai_stub: _OpenAICompatStub,
    strong_probability: float,
    expected_model: str,
) -> None:
    legacy = _legacy_app(
        {
            "routes": {
                "legacy-random": {
                    "type": "random-routing",
                    "strong": {
                        "model": "provider/strong",
                        "format": "openai",
                        "base_url": f"{openai_stub.base_url}/legacy-random/v1",
                        "api_key": "test-key",
                    },
                    "weak": {
                        "model": "provider/weak",
                        "format": "openai",
                        "base_url": f"{openai_stub.base_url}/legacy-random/v1",
                        "api_key": "test-key",
                    },
                    "fallback_target_on_evict": "strong",
                    "strong_probability": strong_probability,
                    "rng_seed": 7,
                }
            }
        }
    )
    profile = _profile_runner(
        _random_profile_config(openai_stub.base_url, strong_probability),
        "random-profile",
    )

    legacy_response = legacy.post(
        "/v1/chat/completions",
        json=_chat_payload("legacy-random"),
    )
    profile_response = await profile.run(
        ChatRequest.openai_chat(_chat_payload("random-profile")),
    )

    assert legacy_response.status_code == 200, legacy_response.text
    assert profile_response.body["choices"][0]["message"]["content"] == "ok"
    assert [call["body"]["model"] for call in openai_stub.requests] == [
        expected_model,
        expected_model,
    ]
    assert [call["path"] for call in openai_stub.requests] == [
        "/legacy-random/v1/chat/completions",
        "/v2-random/v1/chat/completions",
    ]


async def test_latency_service_profile_routes_to_configured_target(
    openai_stub: _OpenAICompatStub,
) -> None:
    profile = _profile_runner(
        _latency_profile_config(openai_stub.base_url),
        "latency-profile",
    )

    response = await profile.run(
        ChatRequest.openai_chat(_chat_payload("latency-profile")),
    )

    assert response.body["model"] == "provider/latency"
    assert [call["path"] for call in openai_stub.requests] == ["/v2-latency/v1/chat/completions"]
    assert [call["body"]["model"] for call in openai_stub.requests] == ["provider/latency"]


async def test_latency_service_profile_preserves_upstream_error(
    openai_stub: _OpenAICompatStub,
) -> None:
    openai_stub.respond_json(
        503,
        {"error": {"message": "upstream unavailable", "code": "unavailable"}},
    )
    runner = (
        parse_profile_config_str(_latency_profile_config(openai_stub.base_url))
        .resolve()
        .build_profile("latency-profile")
    )

    with pytest.raises(SwitchyardUpstreamError, match="upstream unavailable"):
        await runner.run(ChatRequest.openai_chat(_chat_payload("latency-profile")))

    assert [call["body"]["model"] for call in openai_stub.requests] == ["provider/latency"]


async def test_llm_routing_profile_routes_with_strict_classifier(
    openai_stub: _OpenAICompatStub,
) -> None:
    openai_stub.respond_json(
        200,
        _tool_call_payload(
            "provider/classifier",
            "/v2-llm-routing-classifier/v1/chat/completions",
            "select_route",
            {
                "recommended_tier": "medium",
                "confidence": 0.9,
                "abstain": False,
                "turn_type": "exploration",
                "code_modification_scope": "none",
                "tool_call_count_estimate": 0,
                "requires_codebase_context": False,
            },
        ),
    )
    profile = _profile_runner(
        _llm_routing_profile_config(openai_stub.base_url),
        "llm-profile",
    )

    response = await profile.run(
        ChatRequest.openai_chat(_chat_payload("llm-profile")),
    )

    assert response.body["model"] == "provider/weak"
    assert [call["path"] for call in openai_stub.requests] == [
        "/v2-llm-routing-classifier/v1/chat/completions",
        "/v2-llm-routing/v1/chat/completions",
    ]
    assert [call["body"]["model"] for call in openai_stub.requests] == [
        "provider/classifier",
        "provider/weak",
    ]
    classifier_body = openai_stub.requests[0]["body"]
    assert classifier_body["tools"][0]["function"]["name"] == "select_route"
    assert classifier_body["tools"][0]["function"]["strict"] is True
    assert classifier_body["tool_choice"]["function"]["name"] == "select_route"
    assert "response_format" not in classifier_body


async def test_stage_router_profile_routes_with_classifier(
    openai_stub: _OpenAICompatStub,
) -> None:
    openai_stub.respond_json(
        200,
        _completion_payload(
            "provider/classifier",
            "/v2-stage_router-classifier/v1/chat/completions",
            json.dumps({"tier": "weak"}),
        ),
    )
    profile = _profile_runner(
        _stage_router_profile_config(openai_stub.base_url),
        "stage_router-profile",
    )

    response = await profile.run(
        ChatRequest.openai_chat(_chat_payload("stage_router-profile")),
    )

    assert response.body["model"] == "provider/weak"
    assert [call["path"] for call in openai_stub.requests] == [
        "/v2-stage_router-classifier/v1/chat/completions",
        "/v2-stage_router/v1/chat/completions",
    ]
    assert [call["body"]["model"] for call in openai_stub.requests] == [
        "provider/classifier",
        "provider/weak",
    ]
    response_format = openai_stub.requests[0]["body"]["response_format"]
    assert response_format["type"] == "json_object"


def test_python_profile_plan_builds_expected_runner_and_target(
    openai_stub: _OpenAICompatStub,
) -> None:
    plan = parse_profile_config_str(_passthrough_profile_config(openai_stub.base_url)).resolve()
    profiles = plan.build_profiles()
    target = plan.target("direct")

    assert plan.profile_ids() == ["direct-profile"]
    assert plan.target_ids() == ["direct"]
    assert profiles["direct-profile"].profile_id == "direct-profile"
    assert target is not None
    assert target.model == "provider/direct"


async def test_profile_runner_uses_same_profile_config(
    openai_stub: _OpenAICompatStub,
) -> None:
    runner = _profile_runner(
        _passthrough_profile_config(openai_stub.base_url),
        "direct-profile",
    )

    response = await runner.run(ChatRequest.openai_chat(_chat_payload("direct-profile")))

    assert response.body["model"] == "provider/direct"
    assert response.body["mock_path"] == "/v2-direct/v1/chat/completions"
    assert [call["body"]["model"] for call in openai_stub.requests] == ["provider/direct"]


def test_cli_serve_config_delegates_to_rust_profile_server(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    import switchyard.cli.switchyard_cli as cli
    import switchyard_rust.server as rust_server

    config_path = tmp_path / "profiles.yaml"
    config_path.write_text("profiles:\n  bench:\n    type: noop\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_profile_server(
        config_path: str,
        host: str = "127.0.0.1",
        port: int = 4000,
        backlog: int = 65_535,
        dry_run: bool = False,
    ) -> None:
        captured["config_path"] = config_path
        captured["host"] = host
        captured["port"] = port
        captured["backlog"] = backlog
        captured["dry_run"] = dry_run

    mocker.patch.object(
        rust_server,
        "run_profile_server",
        side_effect=_fake_run_profile_server,
    )

    def _fail_build_and_serve(
        args: argparse.Namespace,
        switchyard: object,
        inbound_default: str = "openai",
        disable_backend_streaming: bool = False,
        extra_endpoints: list[object] | None = None,
    ) -> None:
        _ = (
            args,
            switchyard,
            inbound_default,
            disable_backend_streaming,
            extra_endpoints,
        )
        pytest.fail("route-bundle server should not run")

    mocker.patch.object(cli, "build_and_serve", side_effect=_fail_build_and_serve)

    args = cli._build_parser().parse_args(
        [
            "serve",
            "--config",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            "4555",
        ]
    )
    args.func(args)

    assert captured == {
        "config_path": str(config_path),
        "host": "127.0.0.1",
        "port": 4555,
        "backlog": 65_535,
        "dry_run": False,
    }
