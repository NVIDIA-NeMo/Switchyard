# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for envelope-level sub-agent routing (``subagent_target``).

A profile config may name a ``subagent_target`` in its common envelope; the
loader then wraps the built profile so recognized sub-agent requests run
through a passthrough to that target while all other traffic keeps the
profile's own routing.
"""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from switchyard import load_profiles
from switchyard.lib.profiles import ProfileConfigError
from switchyard.lib.profiles.subagent_override import SubagentOverrideProfile
from switchyard_rust.core import ChatRequest, SwitchyardConfigError
from switchyard_rust.profiles import (
    ProfileInput,
    ProfileRequestMetadata,
    is_subagent_request,
)

SUBAGENT_HEADERS = {
    "x-claude-code-session-id": "root-session",
    "x-claude-code-agent-id": "worker-1",
}


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
            "id": "chatcmpl-subagent",
            "object": "chat.completion",
            "model": body.get("model"),
            "mock_path": self.path,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@pytest.fixture
def mock_openai_server() -> Iterator[_MockOpenAIServer]:
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


def _config(base_url: str) -> str:
    """Rust and Python profiles that both name a ``subagent_target``."""
    return f"""
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: "{base_url}/strong/v1"
    api_key: test-key
  worker:
    model: provider/worker
    format: openai
    base_url: "{base_url}/worker/v1"
    api_key: test-key

profiles:
  direct:
    type: passthrough
    target: strong
    subagent_target: worker
  smart:
    type: header-routing
    strong: strong
    weak: strong
    subagent_target: worker
  plain:
    type: passthrough
    target: strong
"""


def _write(tmp_path: Path, text: str, name: str = "profiles.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _request(model: str = "client/x") -> ChatRequest:
    return ChatRequest.openai_chat(
        {"model": model, "messages": [{"role": "user", "content": "hello"}]}
    )


def _subagent_input() -> ProfileInput:
    return ProfileInput(
        _request(),
        ProfileRequestMetadata(headers=dict(SUBAGENT_HEADERS)),
    )


# --- Detection ------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        # No signal at all.
        ({}, False),
        # Claude Code lineage: any non-empty agent id marks a child agent.
        (SUBAGENT_HEADERS, True),
        # Agent id alone (no session header) is still a child agent.
        ({"x-claude-code-agent-id": "child-1"}, True),
        # Session alone — root agent, no agent-id sent.
        ({"x-claude-code-session-id": "s"}, False),
        # Codex delegated-work kinds route as sub-agent work.
        ({"x-openai-subagent": "review"}, True),
        ({"x-openai-subagent": "collab_spawn"}, True),
        # Codex harness maintenance and unknown kinds stay on normal routing.
        ({"x-openai-subagent": "compact"}, False),
        ({"x-openai-subagent": "memory_consolidation"}, False),
        ({"x-openai-subagent": "brand_new_kind"}, False),
        # The explicit Switchyard header decides the lineage fact, in both
        # directions; the work-kind policy still applies on top of it.
        ({"x-switchyard-is-subagent": "true"}, True),
        ({"x-switchyard-is-subagent": "false", "x-openai-subagent": "review"}, False),
        ({"x-switchyard-is-subagent": "true", "x-openai-subagent": "compact"}, False),
        # Header names are case-insensitive and values are trimmed.
        ({"X-OpenAI-Subagent": " review "}, True),
    ],
)
def test_is_subagent_request_detection(headers: dict[str, str], expected: bool) -> None:
    assert is_subagent_request(headers) is expected


# --- Loader wiring ----------------------------------------------------------


def test_loader_wraps_profiles_with_subagent_target(tmp_path: Path) -> None:
    path = _write(tmp_path, _config("http://127.0.0.1:9"))
    profiles = load_profiles(path)
    assert isinstance(profiles["direct"], SubagentOverrideProfile)
    assert isinstance(profiles["smart"], SubagentOverrideProfile)
    assert not isinstance(profiles["plain"], SubagentOverrideProfile)


async def test_rust_profile_routes_subagent_requests_to_target(
    mock_openai_server: _MockOpenAIServer,
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _config(mock_openai_server.base_url))
    profiles = load_profiles(path)

    normal = await profiles["direct"].run(ProfileInput(_request()))
    assert normal.body["model"] == "provider/strong"
    assert normal.body["mock_path"] == "/strong/v1/chat/completions"

    subagent = await profiles["direct"].run(_subagent_input())
    assert subagent.body["model"] == "provider/worker"
    assert subagent.body["mock_path"] == "/worker/v1/chat/completions"


async def test_python_profile_routes_subagent_requests_to_target(
    mock_openai_server: _MockOpenAIServer,
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _config(mock_openai_server.base_url))
    profiles = load_profiles(path)

    normal = await profiles["smart"].run(ProfileInput(_request()))
    assert normal.body["model"] == "provider/strong"

    subagent = await profiles["smart"].run(_subagent_input())
    assert subagent.body["model"] == "provider/worker"
    assert subagent.body["mock_path"] == "/worker/v1/chat/completions"


async def test_maintenance_kinds_keep_normal_routing(
    mock_openai_server: _MockOpenAIServer,
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _config(mock_openai_server.base_url))
    profiles = load_profiles(path)

    metadata = ProfileRequestMetadata(headers={"x-openai-subagent": "compact"})
    response = await profiles["direct"].run(ProfileInput(_request(), metadata))
    assert response.body["model"] == "provider/strong"


def test_rust_profile_unknown_subagent_target_is_rejected(tmp_path: Path) -> None:
    config = """
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: http://127.0.0.1:9/v1
    api_key: test-key

profiles:
  direct:
    type: passthrough
    target: strong
    subagent_target: ghost
"""
    path = _write(tmp_path, config)
    with pytest.raises(SwitchyardConfigError, match="unknown target ghost"):
        load_profiles(path)


def test_python_profile_unknown_subagent_target_is_rejected(tmp_path: Path) -> None:
    config = """
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: http://127.0.0.1:9/v1
    api_key: test-key

profiles:
  smart:
    type: header-routing
    strong: strong
    weak: strong
    subagent_target: ghost
"""
    path = _write(tmp_path, config)
    with pytest.raises(ProfileConfigError, match="unknown target 'ghost'"):
        load_profiles(path)


# --- Wrapper behavior -------------------------------------------------------


class _StubRunner:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    async def run(self, input: ProfileInput) -> Any:
        self.calls += 1
        return self.label


def test_iter_components_spans_both_branches() -> None:
    inner = _StubRunner("inner")
    override = _StubRunner("override")
    wrapper = SubagentOverrideProfile(inner, override)
    assert wrapper.iter_components() == [inner, override]


async def test_override_failure_is_not_rerouted_to_the_wrapped_profile() -> None:
    class _FailingRunner:
        async def run(self, input: ProfileInput) -> Any:
            raise RuntimeError("worker target unavailable")

    inner = _StubRunner("inner")
    wrapper = SubagentOverrideProfile(inner, _FailingRunner())
    with pytest.raises(RuntimeError, match="worker target unavailable"):
        await wrapper.run(_subagent_input())
    assert inner.calls == 0
