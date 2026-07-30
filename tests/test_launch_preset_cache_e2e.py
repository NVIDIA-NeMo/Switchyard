# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-flag launcher configs keep Anthropic prompt caching on the wire.

The preset's strong tier is a Claude model with ``format=auto``. Built through
the launcher route builder and the deterministic profile, its outbound
``/v1/messages`` body must carry ``cache_control`` markers even for an
OpenAI-shaped harness; the weak tier stays OpenAI format with no markers.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from switchyard.cli.config.user_config import LaunchRouteConfig
from switchyard.cli.routing.route_builder import (
    LaunchTierConnectivity,
    build_deterministic_routing_config,
)
from switchyard.lib.backends.anthropic_cache_breakpoint_backend import (
    AnthropicCacheBreakpointBackend,
)
from switchyard.lib.backends.deterministic_routing_llm_backend import (
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.profiles.deterministic_routing_profile_config import (
    DeterministicRoutingProfileConfig,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest

CAPTURED = []


class Upstream(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        CAPTURED.append({"path": self.path, "body": body})
        if self.path.endswith("/v1/messages"):
            resp = {"id": "m1", "type": "message", "role": "assistant", "model": body.get("model"),
                    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1}}
        else:
            resp = {"id": "c1", "object": "chat.completion", "model": body.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()




async def test_launcher_preset_path_emits_cache_control_on_strong(upstream) -> None:
    config = build_deterministic_routing_config(
        LaunchRouteConfig(type="deterministic"),
        primary=LaunchTierConnectivity(base_url=upstream, api_key="sk-test"),
        weak=LaunchTierConnectivity(base_url=upstream, api_key="sk-test"),
        classifier_model=None,
        profile_name=None,
        classifier_min_confidence=None,
        timeout=30.0,
    )
    assert config.strong.model == "anthropic/claude-opus-4.7"
    assert str(config.strong.format).lower().endswith("auto")

    profile = (
        DeterministicRoutingProfileConfig.from_config(config)
        .build()
        .with_runtime_components(enable_stats=False)
    )
    backend = next(
        c for c in profile.iter_components()
        if isinstance(c, DeterministicRoutingLLMBackend)
    )
    assert isinstance(backend._backends["strong"], AnthropicCacheBreakpointBackend)
    assert not isinstance(backend._backends["weak"], AnthropicCacheBreakpointBackend)

    CAPTURED.clear()
    request = ChatRequest.openai_chat({
        "model": "inbound",
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "hello"},
        ],
    })
    ctx = ProxyContext()
    response = await backend._backends["strong"].call(ctx, request)
    assert response is not None

    strong_calls = [c for c in CAPTURED if c["path"].endswith("/v1/messages")]
    assert strong_calls, f"no anthropic call captured; saw {[c['path'] for c in CAPTURED]}"
    body = strong_calls[-1]["body"]
    flat = json.dumps(body)
    assert '"cache_control"' in flat and '"ephemeral"' in flat, (
        f"cache_control missing from wire body: {flat[:400]}"
    )

    CAPTURED.clear()
    await backend._backends["weak"].call(ctx, request)
    weak_calls = [c for c in CAPTURED if c["path"].endswith("/v1/chat/completions")]
    assert weak_calls, f"no chat call captured; saw {[c['path'] for c in CAPTURED]}"
    assert "cache_control" not in json.dumps(weak_calls[-1]["body"])
