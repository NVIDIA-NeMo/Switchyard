#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible GLiNER classifier sidecar for Switchyard custom routing."""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Route:
    target: str
    description: str


@dataclass(frozen=True)
class RouterConfig:
    routes: Mapping[str, Route]
    fallback_target: str
    minimum_confidence: float


def load_config(path: Path) -> RouterConfig:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw_routes = document.get("routes")
    if not isinstance(raw_routes, dict) or not raw_routes:
        raise ValueError("routes must be a non-empty object")

    routes: dict[str, Route] = {}
    targets: set[str] = set()
    for alias, raw_route in raw_routes.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("each route alias must be a non-empty string")
        if not isinstance(raw_route, dict):
            raise ValueError(f"route {alias!r} must be an object")
        target = raw_route.get("target")
        description = raw_route.get("description")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"route {alias!r} needs a non-empty target")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"route {alias!r} needs a non-empty description")
        if target in targets:
            raise ValueError(f"target {target!r} is mapped by more than one alias")
        routes[alias] = Route(target=target, description=description)
        targets.add(target)

    fallback = document.get("fallback_target")
    if fallback not in targets:
        raise ValueError("fallback_target must match a configured route target")
    minimum_confidence = document.get("minimum_confidence", 0.0)
    if not isinstance(minimum_confidence, (int, float)) or not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between 0 and 1")
    return RouterConfig(routes, fallback, float(minimum_confidence))


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and block.get("type") in {None, "text", "input_text"}:
            parts.append(text)
    return "\n".join(parts)


def routing_text(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    user_messages = [
        message_text(message.get("content"))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    user_messages = [text for text in user_messages if text.strip()]
    if not user_messages:
        raise ValueError("at least one non-empty user message is required")
    return "\n\n".join(user_messages)


def allowed_targets(response_format: Any) -> set[str]:
    try:
        values = response_format["json_schema"]["schema"]["properties"]["decision"]["properties"][
            "target"
        ]["enum"]
    except (KeyError, TypeError):
        raise ValueError("response_format must contain the Switchyard target enum") from None
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(item, str) for item in values)
    ):
        raise ValueError("the Switchyard target enum must be a non-empty string array")
    return set(values)


class RoutingEngine:
    def __init__(
        self,
        config: RouterConfig,
        classify: Callable[[str], tuple[str, float]],
    ) -> None:
        self.config = config
        self._classify = classify
        self._lock = threading.Lock()

    def route(self, text: str, allowed: set[str]) -> tuple[str, float]:
        configured_targets = {route.target for route in self.config.routes.values()}
        missing = configured_targets - allowed
        if missing:
            raise ValueError(
                f"response schema does not allow configured targets: {sorted(missing)}"
            )
        with self._lock:
            alias, confidence = self._classify(text)
        route = self.config.routes.get(alias)
        if route is None:
            return self.config.fallback_target, 0.0
        if confidence < self.config.minimum_confidence:
            return self.config.fallback_target, confidence
        return route.target, confidence


def gliner_classifier(
    model_name: str, config: RouterConfig, device: str
) -> Callable[[str], tuple[str, float]]:
    import torch
    from gliner2.classification import ClassificationSchema, Classifier

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = Classifier.from_pretrained(model_name).to(device).eval()
    labels = {alias: route.description for alias, route in config.routes.items()}
    schema = ClassificationSchema().single("route", labels)

    def classify(text: str) -> tuple[str, float]:
        result = model.classify(text, schema)
        route = result["route"]
        return route.label or "", float(route.confidence or 0.0)

    return classify


def completion(target: str, confidence: float, model: str) -> dict[str, Any]:
    verdict = json.dumps(
        {"decision": {"target": target, "confidence": confidence}},
        separators=(",", ":"),
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": verdict},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def handler_for(engine: RoutingEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "GLiNERRouter/1"

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_048_576:
                    raise ValueError("request body must be between 1 byte and 1 MiB")
                request = json.loads(self.rfile.read(length))
                text = routing_text(request.get("messages"))
                allowed = allowed_targets(request.get("response_format"))
                target, confidence = engine.route(text, allowed)
                response = completion(
                    target, confidence, str(request.get("model", "gliner-router"))
                )
            except (json.JSONDecodeError, ValueError) as error:
                self._send(HTTPStatus.BAD_REQUEST, {"error": {"message": str(error)}})
                return
            except Exception:
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": {"message": "classification failed"}},
                )
                return
            self._send(HTTPStatus.OK, response)

        def _send(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            # Deliberately avoid logging request bodies, prompts, or credentials.
            return

    return Handler


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--model", default="fastino/gliner2.5-base-v1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.routes)
    classify = gliner_classifier(args.model, config, args.device)
    server = ThreadingHTTPServer(
        (args.host, args.port), handler_for(RoutingEngine(config, classify))
    )
    print(f"GLiNER router listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
