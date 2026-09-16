# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inject stable per-task Switchyard session IDs into hold-out model traffic."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

SESSION_HEADER = "x-switchyard-session-id"
TASK_HEADER = "x-switchyard-intake-task"
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _leading_conversation(payload: dict[str, Any]) -> list[Any]:
    """Return the stable request prefix that identifies one agent task."""
    conversation = payload.get("messages")
    if not isinstance(conversation, list):
        conversation = payload.get("input")
    if not isinstance(conversation, list):
        return []

    leading: list[Any] = []
    for item in conversation:
        if isinstance(item, dict) and item.get("role") == "assistant":
            break
        leading.append(item)
    return leading


def session_id_for_payload(namespace: str, body: bytes) -> str:
    """Derive a stable session ID from the task's initial conversation prefix."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("model request body must be UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("model request body must be a JSON object")

    leading = _leading_conversation(payload)
    if not leading:
        raise ValueError("model request has no stable leading conversation")
    canonical = json.dumps(leading, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(f"{namespace}\0{canonical}".encode()).hexdigest()
    return f"{namespace}-{digest[:24]}"


class SessionProxyHandler(BaseHTTPRequestHandler):
    """Forward HTTP requests while attaching a deterministic task session."""

    upstream_host = "127.0.0.1"
    upstream_port = 4000
    namespace = "holdout"

    def _forward(self, body: bytes = b"") -> None:
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        if self.command == "POST" and self.path.startswith(
            ("/v1/chat/completions", "/v1/messages", "/v1/responses")
        ):
            session_id = session_id_for_payload(self.namespace, body)
            headers[SESSION_HEADER] = session_id
            headers[TASK_HEADER] = session_id
        if body:
            headers["content-length"] = str(len(body))

        connection = http.client.HTTPConnection(
            self.upstream_host,
            self.upstream_port,
            timeout=900,
        )
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("content-length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        finally:
            connection.close()

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def _handle(self) -> None:
        try:
            content_length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(content_length) if content_length else b""
            self._forward(body)
        except (OSError, ValueError, http.client.HTTPException) as error:
            payload = json.dumps({"error": str(error)}).encode()
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def _validated_upstream(value: str) -> tuple[str, int]:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise argparse.ArgumentTypeError("upstream must be an HTTP localhost URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("upstream must not include a path, query, or fragment")
    return parsed.hostname, parsed.port or 80


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=4001)
    parser.add_argument("--upstream", type=_validated_upstream, default=("127.0.0.1", 4000))
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()

    if not 1 <= args.listen_port <= 65535:
        parser.error("--listen-port must be between 1 and 65535")
    if not args.namespace.replace("-", "").isalnum():
        parser.error("--namespace must contain only letters, digits, and hyphens")

    SessionProxyHandler.upstream_host, SessionProxyHandler.upstream_port = args.upstream
    SessionProxyHandler.namespace = args.namespace
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), SessionProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
