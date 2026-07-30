# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-protocol fake provider for the external-plugin process E2E."""

from __future__ import annotations

import argparse
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CALLS: Counter[str] = Counter()
CALLS_LOCK = threading.Lock()


def call_number(model: str) -> int:
    with CALLS_LOCK:
        CALLS[model] += 1
        return CALLS[model]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        print(format % args, flush=True)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._json(200, {"ok": True})
        elif self.path == "/calls":
            with CALLS_LOCK:
                self._json(200, dict(CALLS))
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        size = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(size) or b"{}")
        model = request.get("model", "unknown")
        attempt = call_number(model)
        if model == "fake/retry-once" and attempt == 1:
            self._json(503, {"error": {"message": "retry this request"}})
            return
        if model == "fake/always-fail":
            self._json(400, {"error": {"message": "invalid routed request"}})
            return
        if self.path == "/v1/chat/completions":
            self._chat(request)
        elif self.path == "/v1/responses":
            self._responses(request)
        elif self.path == "/v1/messages":
            self._anthropic(request)
        else:
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})

    def _chat(self, request: dict[str, object]) -> None:
        model = str(request.get("model", "unknown"))
        classifier = model == "fake/classifier"
        answer = (
            '{"recommended_route":"efficient","p_solve":0.9,'
            '"confidence":0.95,"abstain":false,'
            '"capability_boundary":"supported","primary_rule":"SUP-1",'
            '"crux":"bounded task"}'
            if classifier
            else f"chat from {model}"
        )
        if request.get("stream"):
            events: list[dict[str, object]] = [
                {
                    "id": "chatcmpl-dynamic",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "system_fingerprint": "fp_dynamic_plugin",
                    "provider_extension": {"preserved": True},
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": answer},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-dynamic",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "system_fingerprint": "fp_dynamic_plugin",
                    "provider_extension": {"preserved": True},
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                },
            ]
            self._sse(events)
            return
        self._json(
            200,
            {
                "id": "chatcmpl-dynamic",
                "object": "chat.completion",
                "model": model,
                "system_fingerprint": "fp_dynamic_plugin",
                "provider_extension": {"preserved": True},
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            },
        )

    def _responses(self, request: dict[str, object]) -> None:
        model = str(request.get("model", "unknown"))
        text = f"responses from {model}"
        if request.get("stream"):
            self._sse(
                [
                    {
                        "type": "response.created",
                        "response": {"id": "resp-dynamic", "model": model},
                        "provider_extension": {"preserved": True},
                    },
                    {
                        "type": "response.output_text.delta",
                        "item_id": "item-dynamic",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                        "provider_extension": {"preserved": True},
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp-dynamic",
                            "model": model,
                            "usage": {
                                "input_tokens": 4,
                                "output_tokens": 3,
                                "total_tokens": 7,
                            },
                        },
                        "provider_extension": {"preserved": True},
                    },
                ],
                named=True,
            )
            return
        self._json(
            200,
            {
                "id": "resp-dynamic",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": model,
                "output": [
                    {
                        "id": "msg-dynamic",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": text, "annotations": []}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 3,
                    "total_tokens": 7,
                },
                "provider_extension": {"preserved": True},
            },
        )

    def _anthropic(self, request: dict[str, object]) -> None:
        model = str(request.get("model", "unknown"))
        text = f"anthropic from {model}"
        if request.get("stream"):
            self._sse(
                [
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg-dynamic",
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [],
                            "usage": {"input_tokens": 4, "output_tokens": 0},
                        },
                        "provider_extension": {"preserved": True},
                    },
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                        "provider_extension": {"preserved": True},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                        "provider_extension": {"preserved": True},
                    },
                    {
                        "type": "content_block_stop",
                        "index": 0,
                        "provider_extension": {"preserved": True},
                    },
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 3},
                        "provider_extension": {"preserved": True},
                    },
                    {
                        "type": "message_stop",
                        "provider_extension": {"preserved": True},
                    },
                ],
                named=True,
            )
            return
        self._json(
            200,
            {
                "id": "msg-dynamic",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
                "provider_extension": {"preserved": True},
            },
        )

    def _sse(self, events: list[dict[str, object]], named: bool = False) -> None:
        parts = []
        for event in events:
            if named:
                parts.append(f"event: {event['type']}\n")
            parts.append(f"data: {json.dumps(event)}\n\n")
        if not named:
            parts.append("data: [DONE]\n\n")
        data = "".join(parts).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def _json(self, status: int, value: object) -> None:
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
