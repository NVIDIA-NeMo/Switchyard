# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture opencode's outbound HTTP requests.

Throwaway-but-kept diagnostic. Used during the codex-review P1b
investigation to determine whether opencode forwards
`reasoning_effort` when `reasoningEffort` is set at
`provider.<name>.models.<id>.reasoningEffort` vs
`provider.<name>.models.<id>.options.reasoningEffort`.

Usage:
    python scripts/opencode_request_capture.py &
    # point opencode at http://localhost:8765/v1 and run
    jq . /tmp/opencode-request-capture.log

The server logs every incoming POST body to the log file, then
returns a minimal OpenAI-compatible streaming response so opencode
doesn't hang waiting for tool calls.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

LOG = Path("/tmp/opencode-request-capture.log")
PORT = 8765


def _ndjson_stream_chunk(content: str, finish: bool = False) -> bytes:
    """One SSE frame in the OpenAI chat-completions streaming shape."""
    delta = {"role": "assistant", "content": content} if content else {}
    choice = {"index": 0, "delta": delta}
    if finish:
        choice["finish_reason"] = "stop"
        choice["delta"] = {}
    payload = {
        "id": "probe",
        "object": "chat.completion.chunk",
        "choices": [choice],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        # Suppress default stderr logging; we write our own.
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length) if length else b""
        try:
            parsed_body = json.loads(body.decode()) if body else None
        except json.JSONDecodeError:
            parsed_body = body.decode(errors="replace")

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "path": self.path,
            "headers": dict(self.headers.items()),
            "body": parsed_body,
        }
        with LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        # Return an SSE stream; opencode treats the gateway as streaming.
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(_ndjson_stream_chunk("ok"))
        self.wfile.write(_ndjson_stream_chunk("", finish=True))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> int:
    LOG.write_text("")  # truncate per run
    print(f"echo server listening on http://localhost:{PORT}, log={LOG}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
