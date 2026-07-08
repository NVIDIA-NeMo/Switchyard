# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for components-v2 profile config and runtime bindings."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from switchyard_rust.components import LlmTarget
from switchyard_rust.core import ChatRequest, ChatRequestType, SwitchyardConfigError
from switchyard_rust.profiles import (
    Profile,
    ProfileConfigDocument,
    ProfileConfigPlan,
    ProfileInput,
    ProfileRequestMetadata,
    load_profile_config,
    parse_profile_config_path,
    parse_profile_config_str,
)


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

        message: dict[str, Any] = {"role": "assistant", "content": "ok"}
        finish_reason = "stop"
        if body.get("tools"):
            tool_name = body["tool_choice"]["function"]["name"]
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_route",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(
                                {
                                    "recommended_tier": "medium",
                                    "confidence": 0.9,
                                    "abstain": False,
                                    "turn_type": "exploration",
                                    "code_modification_scope": "none",
                                    "tool_call_count_estimate": 0,
                                    "requires_codebase_context": False,
                                }
                            ),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        elif body.get("response_format", {}).get("type") == "json_schema":
            schema_name = body["response_format"]["json_schema"]["name"]
            if schema_name == "StageRouterTierDecision":
                message = {"role": "assistant", "content": json.dumps({"tier": "efficient"})}
        elif (
            body.get("response_format", {}).get("type") == "json_object"
            and body.get("model") == "provider/classifier"
        ):
            message = {"role": "assistant", "content": json.dumps({"tier": "efficient"})}

        response = {
            "id": "chatcmpl-profile-bindings",
            "object": "chat.completion",
            "model": body.get("model"),
            "mock_path": self.path,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(
        self,
        _format: str,
        _requestline: str | None = None,
        _code: str | None = None,
        _size: str | None = None,
    ) -> None:
        return


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


def _profile_config(base_url: str) -> str:
    return f"""
targets:
  direct:
    model: provider/direct
    format: openai
    base_url: "{base_url}/direct/v1"
    api_key: test-key
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
  latency:
    model: provider/latency
    format: openai
    base_url: "{base_url}/latency/v1"
    api_key: test-key
  classifier:
    model: provider/classifier
    format: openai
    base_url: "{base_url}/classifier/v1"
    api_key: test-key

profiles:
  direct:
    type: passthrough
    target: direct
  random:
    type: random-routing
    strong: strong
    weak: weak
    strong_probability: 1.0
    rng_seed: 7
  latency:
    type: latency-service
    latency_service_url: "http://latency.invalid"
    targets: [latency]
    max_retries: 0
  llm:
    type: llm-routing
    strong: strong
    weak: weak
    classifier: classifier
    profile_name: coding_agent
  stage_router:
    type: stage_router
    capable: strong
    efficient: weak
    fallback_target_on_evict: strong
    picker: capable_first
    confidence_threshold: 0.7
    classifier:
      model: provider/classifier
      api_key: test-key
      base_url: "{base_url}/classifier/v1"
  bench:
    type: noop
"""


def _plan(base_url: str) -> ProfileConfigPlan:
    document = parse_profile_config_str(_profile_config(base_url))
    assert isinstance(document, ProfileConfigDocument)
    return document.resolve()


def _offline_plan() -> ProfileConfigPlan:
    return _plan("http://127.0.0.1:9")


def _request(model: str) -> ChatRequest:
    return ChatRequest.openai_chat(
        {
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )


def test_profile_config_plan_is_inspectable() -> None:
    document = parse_profile_config_str(_profile_config("http://127.0.0.1:9"))

    assert document.profile_ids() == [
        "bench",
        "direct",
        "latency",
        "llm",
        "random",
        "stage_router",
    ]
    assert document.profile_type("direct") == "passthrough"
    assert document.profile_body("random") == {
        "strong": "strong",
        "weak": "weak",
        "strong_probability": 1.0,
        "rng_seed": 7,
    }
    rust_document = document.without_profiles(["random"])
    assert rust_document.profile_ids() == ["bench", "direct", "latency", "llm", "stage_router"]

    plan = document.resolve()

    assert isinstance(plan, ProfileConfigPlan)
    assert plan.profile_ids() == [
        "bench",
        "direct",
        "latency",
        "llm",
        "random",
        "stage_router",
    ]
    assert plan.target_ids() == ["classifier", "direct", "latency", "strong", "weak"]
    assert plan.profile_type("direct") == "passthrough"
    assert plan.profile_type("random") == "random-routing"
    assert plan.profile_type("latency") == "latency-service"
    assert plan.profile_type("llm") == "llm-routing"
    assert plan.profile_type("stage_router") == "stage_router"
    assert plan.profile_type("bench") == "noop"
    assert plan.profile_type("missing") is None

    target = plan.target("direct")
    assert isinstance(target, LlmTarget)
    assert target.id == "direct"
    assert target.model == "provider/direct"
    assert target.base_url == "http://127.0.0.1:9/direct/v1"
    assert plan.target("missing") is None

    profiles = plan.build_profiles()
    assert sorted(profiles) == [
        "bench",
        "direct",
        "latency",
        "llm",
        "random",
        "stage_router",
    ]
    assert all(isinstance(profile, Profile) for profile in profiles.values())
    assert profiles["direct"].profile_id == "direct"


def test_profile_config_can_be_loaded_from_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text(_profile_config("http://127.0.0.1:9"), encoding="utf-8")

    document = parse_profile_config_path(path)
    assert isinstance(document.resolve(), ProfileConfigPlan)

    plan = load_profile_config(path)
    assert plan.profile_ids() == [
        "bench",
        "direct",
        "latency",
        "llm",
        "random",
        "stage_router",
    ]


async def test_noop_profile_returns_local_response() -> None:
    profile = _offline_plan().build_profile("bench")

    assert isinstance(profile, Profile)
    assert profile.profile_id == "bench"
    response = await profile.run(_request("client/noop"))

    assert response.body["id"] == "switchyard-noop"
    assert response.body["model"] == "client/noop"
    assert response.body["choices"][0]["message"]["content"] == "ok"


async def test_native_profiles_run_against_local_openai_mock(
    mock_openai_server: _MockOpenAIServer,
) -> None:
    plan = _plan(mock_openai_server.base_url)

    direct_response = await plan.build_profile("direct").run(_request("client/direct"))
    random_response = await plan.build_profile("random").run(_request("client/random"))
    latency_response = await plan.build_profile("latency").run(_request("client/latency"))
    llm_response = await plan.build_profile("llm").run(_request("client/llm-routing"))
    stage_router_response = await plan.build_profile("stage_router").run(_request("client/stage_router"))

    assert direct_response.body["model"] == "provider/direct"
    assert direct_response.body["mock_path"] == "/direct/v1/chat/completions"
    assert random_response.body["model"] == "provider/strong"
    assert random_response.body["mock_path"] == "/strong/v1/chat/completions"
    assert latency_response.body["model"] == "provider/latency"
    assert latency_response.body["mock_path"] == "/latency/v1/chat/completions"
    assert llm_response.body["model"] == "provider/weak"
    assert llm_response.body["mock_path"] == "/weak/v1/chat/completions"
    assert stage_router_response.body["model"] == "provider/weak"
    assert stage_router_response.body["mock_path"] == "/weak/v1/chat/completions"
    assert [call["body"]["model"] for call in mock_openai_server.calls] == [
        "provider/direct",
        "provider/strong",
        "provider/latency",
        "provider/classifier",
        "provider/weak",
        "provider/classifier",
        "provider/weak",
    ]
    llm_classifier_body = mock_openai_server.calls[3]["body"]
    assert llm_classifier_body["tools"][0]["function"]["strict"] is True
    assert llm_classifier_body["tool_choice"]["function"]["name"] == "select_route"
    assert "response_format" not in llm_classifier_body
    stage_router_classifier_body = mock_openai_server.calls[5]["body"]
    assert stage_router_classifier_body["response_format"]["type"] == "json_object"


def test_profile_binding_errors_map_to_switchyard_exceptions() -> None:
    with pytest.raises(SwitchyardConfigError, match="unknown field.*routes"):
        parse_profile_config_str("routes: {}\n")

    unknown_target = """
targets: {}
profiles:
  bad:
    type: passthrough
    target: missing
"""
    with pytest.raises(SwitchyardConfigError, match="profile bad:.*unknown target missing"):
        parse_profile_config_str(unknown_target).resolve()

    with pytest.raises(SwitchyardConfigError, match="unknown profile missing"):
        parse_profile_config_str("profiles:\n  bench:\n    type: noop\n").resolve().build_profile(
            "missing"
        )


def test_profile_request_metadata_normalizes_headers() -> None:
    metadata = ProfileRequestMetadata.from_headers(
        {
            "X-Request-ID": "req-123",
            "X-Switchyard-Trace": ["trace-a", "trace-b"],
        },
        inbound_format=ChatRequestType.OPENAI_CHAT,
    )
    assert metadata.request_id == "req-123"
    assert metadata.inbound_format == ChatRequestType.OPENAI_CHAT
    assert metadata.headers == {
        "x-request-id": ["req-123"],
        "x-switchyard-trace": ["trace-a", "trace-b"],
    }
    assert metadata.session_id is None


def test_profile_request_metadata_reconciles_explicit_and_header_session_id() -> None:
    metadata = ProfileRequestMetadata(
        request_id="req-session",
        headers={
            "Proxy_X_Session_Id": [" session-1 ", "session-1"],
            "X-Nemo-Relay-Session-Id": "session-1",
        },
        session_id=" session-1 ",
    )

    assert metadata.session_id == "session-1"
    assert metadata.to_dict() == {
        "request_id": "req-session",
        "inbound_format": None,
        "headers": {
            "proxy_x_session_id": [" session-1 ", "session-1"],
            "x-nemo-relay-session-id": ["session-1"],
        },
        "session_id": "session-1",
    }
    assert repr(metadata) == (
        'ProfileRequestMetadata(request_id=Some("req-session"), session_id=Some("session-1"))'
    )


def test_profile_request_metadata_from_headers_accepts_equal_session_aliases() -> None:
    metadata = ProfileRequestMetadata.from_headers(
        {
            "Proxy_X_Session_Id": [" session-1 ", "session-1"],
            "X-Nemo-Relay-Session-Id": "session-1",
        }
    )

    assert metadata.session_id == "session-1"


@pytest.mark.parametrize(
    "headers",
    [
        {"proxy_x_session_id": ["session-1", "session-2"]},
        {
            "proxy_x_session_id": "session-1",
            "x-nemo-relay-session-id": "session-2",
        },
    ],
)
def test_profile_request_metadata_rejects_conflicting_session_headers(
    headers: dict[str, str | list[str]],
) -> None:
    with pytest.raises(ValueError, match="conflicting session IDs"):
        ProfileRequestMetadata.from_headers(headers)


@pytest.mark.parametrize(
    "headers",
    [
        {"proxy_x_session_id": ""},
        {"x-nemo-relay-session-id": "   "},
        {"proxy_x_session_id": ["session-1", ""]},
        {"x-nemo-relay-session-id": []},
    ],
)
def test_profile_request_metadata_rejects_empty_session_headers(
    headers: dict[str, str | list[str]],
) -> None:
    with pytest.raises(ValueError, match="non-empty session ID"):
        ProfileRequestMetadata.from_headers(headers)


def test_profile_request_metadata_rejects_explicit_header_session_conflict() -> None:
    with pytest.raises(ValueError, match="conflicting session IDs"):
        ProfileRequestMetadata(
            headers={"x-nemo-relay-session-id": "session-from-header"},
            session_id="session-explicit",
        )


def test_profile_request_metadata_rejects_empty_explicit_session_id() -> None:
    with pytest.raises(ValueError, match="non-empty session ID"):
        ProfileRequestMetadata(session_id="   ")


def test_profile_input_binding_wraps_request_and_metadata() -> None:
    metadata = ProfileRequestMetadata(
        request_id="req-profile-input",
        inbound_format=ChatRequestType.OPENAI_CHAT,
        headers={"X-Switchyard-Trace": "trace-profile-input"},
        session_id="session-profile-input",
    )

    input = ProfileInput(_request("client/profile-input"), metadata=metadata)

    assert input.request.model == "client/profile-input"
    assert input.metadata.request_id == "req-profile-input"
    assert input.metadata.session_id == "session-profile-input"
    assert input.metadata.inbound_format == ChatRequestType.OPENAI_CHAT
    assert input.metadata.headers == {"x-switchyard-trace": ["trace-profile-input"]}


def test_concrete_profile_trios_are_not_reexported_from_rust_profiles() -> None:
    import switchyard_rust.profiles as profiles

    for name in (
        "PassthroughProfileConfig",
        "PassthroughProfile",
        "PassthroughProcessedRequest",
        "RandomRoutingProfileConfig",
        "RandomRoutingProfile",
        "RandomRoutingProcessedRequest",
        "LatencyServiceProfileConfig",
        "LatencyServiceProfile",
        "LatencyServiceProcessedRequest",
        "LlmRoutingProfileConfig",
        "LlmRoutingProfile",
        "LlmRoutingProcessedRequest",
        "StageRouterProfileConfig",
        "StageRouterProfile",
        "StageRouterProcessedRequest",
        "NoopProfileConfig",
        "NoopProfile",
        "NoopProcessedRequest",
    ):
        assert not hasattr(profiles, name)
