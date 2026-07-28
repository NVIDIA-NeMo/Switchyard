# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local in-process mock OpenAI upstream shared by the markdown-docs fixtures.

The getting-started and README doc tests both redirect a passthrough profile at
a loopback server returning a canned ``chat.completion`` so the guide snippets
exercise the real backend + openai-SDK + httpx path without touching the
network. This module holds the single copy both fixtures import.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

    Loopback-only ``ThreadingHTTPServer`` so the doc snippets exercise the real
    passthrough backend + openai-SDK + httpx path without touching the network.
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


__all__ = ["_CANNED_COMPLETION", "_MockOpenAIServer"]
