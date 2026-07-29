# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local in-process mock OpenAI upstream shared by the markdown-docs fixtures.

The getting-started and README doc tests both redirect a passthrough profile at
a loopback server returning a fixed ``chat.completion`` so the guide snippets
exercise the real backend, openai SDK, and httpx code without touching the
network. This module holds the single copy both fixtures use: the mock server
and the ``redirect_passthrough_to_local_mock`` helper that points
``PassthroughProfileConfig.build`` at it.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

#: Fixed OpenAI Chat Completion the mock upstream returns for every request.
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
    """In-process OpenAI-compatible upstream returning a fixed completion.

    Loopback-only ``ThreadingHTTPServer`` so the doc snippets exercise the real
    passthrough backend, openai SDK, and httpx code without touching the network.
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


def markdown_docs_active(config: pytest.Config) -> bool:
    """Return whether the run enabled the ``--markdown-docs`` flag."""
    try:
        return bool(config.getoption("markdowndocs", default=False))
    except (KeyError, ValueError):
        return False


@contextmanager
def redirect_passthrough_to_local_mock(*, stub_uvicorn: bool = False) -> Iterator[None]:
    """Point ``PassthroughProfileConfig.build`` at a local mock OpenAI upstream.

    Starts a loopback :class:`_MockOpenAIServer` and patches the passthrough
    profile's ``build`` so a guide snippet's ``switchyard.call`` runs the real
    backend, openai SDK, and httpx code fully offline, keeping any other fields
    the snippet set on its config. With ``stub_uvicorn=True`` it also replaces
    ``uvicorn.run`` with a no-op so a "host as HTTP server" snippet returns
    instead of blocking the test session forever.
    """
    from switchyard import PassthroughProfileConfig

    real_build = PassthroughProfileConfig.build
    monkeypatch = pytest.MonkeyPatch()
    with _MockOpenAIServer() as upstream:

        def _hermetic_build(_self: PassthroughProfileConfig) -> object:
            return real_build(
                dataclasses.replace(_self, api_key="test-key", base_url=upstream.base_url)
            )

        monkeypatch.setattr(PassthroughProfileConfig, "build", _hermetic_build)
        if stub_uvicorn:
            import uvicorn

            monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)
        try:
            yield
        finally:
            monkeypatch.undo()


__all__ = [
    "_CANNED_COMPLETION",
    "_MockOpenAIServer",
    "markdown_docs_active",
    "redirect_passthrough_to_local_mock",
]
