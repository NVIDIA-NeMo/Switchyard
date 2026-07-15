# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test infrastructure for classifier and chain-composition tests.

Private (underscore-prefixed) module so pytest doesn't try to collect
it as a test file.  Symbols inside also keep the underscore-private
convention used by the test files that import them.

Owners are the offline e2e tests for:

* :mod:`tests.test_llm_classifier_e2e` — classifier-only routing chain.
* :mod:`tests.test_classifier_planner_chain` — classifier + planner
  composition.  The planner-specific stub bodies and URLs live in that
  file (not here) so this module stays orthogonal to plan_execute.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import httpx

# ---------------------------------------------------------------------------
# URL + model constants shared between classifier and chain harnesses.
# ---------------------------------------------------------------------------

#: respx-intercepted classifier endpoint.  Distinct from any planner /
#: backend URL so an unexpected hit is caught by missing-mock errors.
_CLASSIFIER_BASE = "https://classifier.test/v1"
_CLASSIFIER_URL = f"{_CLASSIFIER_BASE}/chat/completions"

_CLASSIFIER_MODEL = "router-classifier-llm"

#: Tier model IDs the deterministic backend exposes to its OpenAI-compatible
#: stubs.  The two-tier (simple / complex) setup is enough to validate
#: classifier-driven dispatch without proliferating tiers.
_SIMPLE_MODEL = "tier/simple-model"
_COMPLEX_MODEL = "tier/complex-model"


# ---------------------------------------------------------------------------
# Classifier payloads (shared between classifier-only and chain tests).
# ---------------------------------------------------------------------------


def _classifier_payload(content: str) -> dict[str, object]:
    """An OpenAI Chat Completion JSON body whose content is the classifier output."""
    return {
        "id": "chatcmpl-classifier",
        "object": "chat.completion",
        "created": 1700000000,
        "model": _CLASSIFIER_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }


def _signals_json(**overrides: object) -> str:
    payload: dict[str, object] = {
        "task_type": "debugging",
        "complexity": "complex",
        "reasoning_depth": "multi_step",
        "tool_planning_required": False,
        "precision_requirement": "high",
        "context_dependency": "conversation",
        "structured_output_risk": "low",
        "recommended_tier": "complex",
        "confidence": 0.88,
        "reason_code": "debugging",
        "abstain": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Generic backend stub bodies (tier-agnostic).
# ---------------------------------------------------------------------------


def _backend_payload(*, content: str, model: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-backend",
        "object": "chat.completion",
        "created": 1700000001,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }


def _stream_chunk(*, content: str = "", finish: str | None = None) -> dict[str, object]:
    delta: dict[str, object] = {}
    if content:
        delta["content"] = content
    return {
        "id": "chatcmpl-backend-stream",
        "object": "chat.completion.chunk",
        "created": 1700000002,
        "model": _COMPLEX_MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse_body(chunks: list[dict[str, object]]) -> bytes:
    return (
        "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    ).encode("utf-8")


def _last_body(stub: _OpenAICompatStub) -> dict[str, object]:
    return cast(dict[str, object], stub.requests[-1]["body"])


# ---------------------------------------------------------------------------
# OpenAI-compatible loopback stub.  Used for tier backends because Rust
# reqwest bypasses respx; a real local HTTP server lets us assert on the
# exact body the backend emitted.
# ---------------------------------------------------------------------------


class _OpenAICompatStub:
    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._requests: list[dict[str, object]] = []
        self._responses: list[tuple[int, bytes, str]] = []

    def __enter__(self) -> _OpenAICompatStub:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                """Record the request (path, body, headers) and pop the queued reply."""
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8"))
                with owner._lock:
                    # Header names lower-cased so tests can look them up
                    # without caring how the client cased them on the wire.
                    owner._requests.append({
                        "path": self.path,
                        "body": body,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                    })
                    if owner._responses:
                        status, content, content_type = owner._responses.pop(0)
                    else:
                        status = 500
                        content = b'{"error":{"message":"no stub response queued"}}'
                        content_type = "application/json"

                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(content)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, _format: str, *args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # ``shutdown()`` blocks up to serve_forever's poll interval; the 0.5s
        # default adds half a second of idle teardown to every test using the
        # stub, so poll frequently.
        server = self._server
        self._thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.05), daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("stub server is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> list[dict[str, object]]:
        with self._lock:
            return list(self._requests)

    @property
    def called(self) -> bool:
        return bool(self.requests)

    def respond_json(self, body: dict[str, object], *, status: int = 200) -> None:
        content = json.dumps(body).encode("utf-8")
        with self._lock:
            self._responses.append((status, content, "application/json"))

    def respond_sse(self, body: bytes) -> None:
        with self._lock:
            self._responses.append((200, body, "text/event-stream"))


# ---------------------------------------------------------------------------
# Harness dataclass returned by chain-test fixtures.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ClassifierHarness:
    """ASGI-driven client + per-tier loopback stubs for chain composition tests."""

    client: httpx.AsyncClient
    simple: _OpenAICompatStub
    complex: _OpenAICompatStub
