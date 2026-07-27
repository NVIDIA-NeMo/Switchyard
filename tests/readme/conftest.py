# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown-docs fixture: run the README snippet against a local mock upstream.

Mirrors ``tests/getting_started/conftest.py``: the README's "Use as a Python
library" snippet builds a passthrough profile and calls ``switchyard.call``,
which would otherwise hit a real backend. The fixture points that passthrough
profile at a local in-process mock OpenAI server (loopback, canned
``chat.completion``) so it runs offline. Gated on the ``--markdown-docs`` flag
so regular runs are untouched.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

#: Canned OpenAI Chat Completion the mock upstream returns for every request.
_CANNED_COMPLETION: dict[str, object] = {
    "id": "chatcmpl-hermetic",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "hermetic-mock",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "4"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


class _MockOpenAIServer:
    """In-process OpenAI-compatible upstream returning a canned completion.

    Loopback-only ``ThreadingHTTPServer`` so the README snippet exercises the
    real passthrough backend + openai-SDK + httpx path without any network.
    """

    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _MockOpenAIServer:
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                self.rfile.read(length)
                content = json.dumps(_CANNED_COMPLETION).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(content)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, _format: str, *_args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("mock server is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"


def _markdown_docs_active(config: pytest.Config) -> bool:
    try:
        return bool(config.getoption("markdowndocs", default=False))
    except (KeyError, ValueError):
        return False


@pytest.fixture(autouse=True, scope="session")
def _markdown_docs_hermetic_upstream(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    if not _markdown_docs_active(request.config):
        yield
        return

    from switchyard import PassthroughProfileConfig

    real_build = PassthroughProfileConfig.build
    monkeypatch = pytest.MonkeyPatch()
    with _MockOpenAIServer() as upstream:
        # Redirect the snippet's passthrough profile at the local mock upstream
        # so ``switchyard.call`` runs the real backend path fully offline.
        def _hermetic_build(_self: PassthroughProfileConfig) -> object:
            return real_build(
                PassthroughProfileConfig(api_key="test-key", base_url=upstream.base_url)
            )

        monkeypatch.setattr(PassthroughProfileConfig, "build", _hermetic_build)
        try:
            yield
        finally:
            monkeypatch.undo()
