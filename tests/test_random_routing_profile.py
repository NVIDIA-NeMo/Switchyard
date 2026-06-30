# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profile-backed random-routing construction."""

from __future__ import annotations

import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from pydantic import ValidationError

from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.profiles import (
    PassthroughProfileConfig,
    ProfileSwitchyard,
    RandomRoutingConfig,
    RandomRoutingProfileConfig,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.components import MultiLlmBackend
from switchyard_rust.core import ChatRequest, ChatResponse
from switchyard_rust.translation import TranslationEngine


class _OpenAICompatStub:
    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._requests: list[dict[str, Any]] = []

    def __enter__(self) -> _OpenAICompatStub:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8"))
                with owner._lock:
                    owner._requests.append({
                        "path": self.path,
                        "authorization": self.headers.get("authorization"),
                        "body": body,
                    })

                response = json.dumps({
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": body.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, _format: str, *args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
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
    def requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._requests)


def _openai_config(
    *,
    strong_model: str = "strong-m",
    weak_model: str = "weak-m",
    strong_probability: float = 0.5,
    api_key: str = "sk-test",
    base_url: str = "https://example.invalid/v1",
    enable_stats: bool = True,
) -> RandomRoutingConfig:
    return RandomRoutingConfig(
        strong=LlmTarget(
            id="strong",
            model=strong_model,
            format=BackendFormat.OPENAI,
            api_key=api_key,
            base_url=base_url,
        ),
        weak=LlmTarget(
            id="weak",
            model=weak_model,
            format=BackendFormat.OPENAI,
            api_key=api_key,
            base_url=base_url,
        ),
        strong_probability=strong_probability,
        enable_stats=enable_stats,
        fallback_target_on_evict="strong",
    )


def _random_routing_switchyard(
    config: RandomRoutingConfig,
    *,
    pre_routing_request_processors: list[Any] | None = None,
    extra_request_processors: list[Any] | None = None,
    extra_response_processors: list[Any] | None = None,
) -> ProfileSwitchyard:
    """Build the profile-backed runtime used by these tests."""
    return ProfileSwitchyard(
        RandomRoutingProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            pre_request_processors=pre_routing_request_processors or (),
            post_request_processors=extra_request_processors or (),
            response_processors=extra_response_processors or (),
        )
    )


def _llm_target_switchyard(
    target: LlmTarget,
) -> ProfileSwitchyard:
    """Build the single-target passthrough profile used by parity tests."""
    return ProfileSwitchyard(
        PassthroughProfileConfig(
            target=target,
        )
        .build()
        .with_runtime_components()
    )


class _NoopRequestProcessor:
    async def process(self, _ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        return request


class _NoopResponseProcessor:
    async def process(self, _ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        return response


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestProfileStructure:
    def test_returns_profile_backed_switchyard_adapter(self):
        switchyard = _random_routing_switchyard(
            _openai_config(),
        )
        assert isinstance(switchyard, ProfileSwitchyard)

    def test_backend_is_random_routing(self):
        config = _openai_config(enable_stats=False)
        switchyard = _random_routing_switchyard(config)
        backends = [
            c for c in switchyard.iter_components()
            if isinstance(c, MultiLlmBackend)
        ]
        assert len(backends) == 1

    def test_default_response_translator(self):
        switchyard = _random_routing_switchyard(
            _openai_config(),
        )
        translators = [
            c for c in switchyard.iter_components()
            if isinstance(c, TranslationEngine)
        ]
        assert len(translators) == 1

    def test_config_propagated_to_backend(self):
        config = _openai_config(
            strong_model="s-123", weak_model="w-456", strong_probability=0.7,
            enable_stats=False,
        )
        switchyard = _random_routing_switchyard(config)
        backend = next(
            c for c in switchyard.iter_components()
            if isinstance(c, MultiLlmBackend)
        )
        assert backend.target_ids() == ["strong", "weak"]

        from switchyard.lib.processors.random_routing_request_processor import (
            RandomRoutingRequestProcessor,
        )

        router = next(
            c for c in switchyard.iter_components()
            if isinstance(c, RandomRoutingRequestProcessor)
        )
        assert router.config.strong_probability == 0.7
        assert router.config.strong.model == "s-123"
        assert router.config.weak.model == "w-456"

    def test_rng_passed_through(self):
        """An explicit rng_seed still produces a runnable profile chain."""
        switchyard = _random_routing_switchyard(
            _openai_config(enable_stats=False).model_copy(update={"rng_seed": 123}),
        )
        backends = [
            c for c in switchyard.iter_components()
            if isinstance(c, MultiLlmBackend)
        ]
        assert len(backends) == 1

    def test_preset_remains_on_config(self):
        """Preset provenance stays on the profile config."""
        config = _openai_config(enable_stats=False).model_copy(update={"preset": "opus_47_minimax"})
        switchyard = _random_routing_switchyard(config)
        backend = next(
            c for c in switchyard.iter_components()
            if isinstance(c, MultiLlmBackend)
        )
        assert backend.target_ids() == ["strong", "weak"]
        assert config.preset == "opus_47_minimax"

    def test_preset_defaults_to_none(self):
        """No ``preset`` on config keeps profile construction unannotated."""
        config = _openai_config(enable_stats=False)
        switchyard = _random_routing_switchyard(_openai_config(enable_stats=False))
        backend = next(
            c for c in switchyard.iter_components()
            if isinstance(c, MultiLlmBackend)
        )
        assert backend.target_ids() == ["strong", "weak"]
        assert config.preset is None

    def test_extra_processors_are_wired(self):
        request_processor = _NoopRequestProcessor()
        response_processor = _NoopResponseProcessor()
        switchyard = _random_routing_switchyard(
            _openai_config(),
            extra_request_processors=[request_processor],
            extra_response_processors=[response_processor],
        )

        components = list(switchyard.iter_components())
        assert request_processor in components
        assert response_processor in components

    async def test_llm_target_profile_builds_single_target_chain(self):
        switchyard = _llm_target_switchyard(
            LlmTarget(
                model="single-m",
                format=BackendFormat.OPENAI,
                api_key="sk-test",
                base_url="https://example.invalid/v1",
            ),
        )

        backend = next(
            c for c in switchyard.iter_components()
            if hasattr(c, "supported_request_types")
        )
        assert [item.value for item in backend.supported_request_types] == ["openai_chat"]


# ---------------------------------------------------------------------------
# Validation passthrough (all validation lives on the config dataclass)
# ---------------------------------------------------------------------------


class TestProfileValidation:
    def test_invalid_probability_rejected_at_config_construction(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            _openai_config(strong_probability=1.5)

    def test_empty_model_rejected_at_config_construction(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            RandomRoutingConfig(
                strong={"model": ""},
                weak={"model": "w"},
            fallback_target_on_evict="strong")


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------


class TestProfileEndToEnd:
    async def test_fifty_requests_hit_both_tiers_through_full_chain(self):
        """Fire 50 requests through the full profile chain; both tiers should call upstream."""
        config = _openai_config(
            strong_model="strong-m", weak_model="weak-m",
            strong_probability=0.5,
            api_key="sk-local",
            enable_stats=False,
        )
        config = config.model_copy(update={"rng_seed": 42})
        with _OpenAICompatStub() as upstream:
            switchyard = _random_routing_switchyard(
                config.model_copy(
                    update={
                        "strong": LlmTarget(
                            id="strong",
                            model="strong-m",
                            format=BackendFormat.OPENAI,
                            api_key="sk-local",
                            base_url=upstream.base_url,
                        ),
                        "weak": LlmTarget(
                            id="weak",
                            model="weak-m",
                            format=BackendFormat.OPENAI,
                            api_key="sk-local",
                            base_url=upstream.base_url,
                        ),
                    },
                ),
            )
            for _ in range(50):
                response = await switchyard.call(
                    ChatRequest.openai_chat({
                        "model": "client-sent",
                        "messages": [{"role": "user", "content": "hi"}],
                    }),
                )
                assert response["choices"][0]["message"]["content"] == "hi"

        bodies = [entry["body"] for entry in upstream.requests]
        paths = [entry["path"] for entry in upstream.requests]
        auth_headers = [entry["authorization"] for entry in upstream.requests]
        selected_models = [body["model"] for body in bodies]

        assert set(paths) == {"/v1/chat/completions"}
        assert set(auth_headers) == {"Bearer sk-local"}
        assert set(selected_models) <= {"strong-m", "weak-m"}
        assert "strong-m" in selected_models
        assert "weak-m" in selected_models
        assert all(body["messages"][0]["content"] == "hi" for body in bodies)
        counts = Counter(selected_models)
        assert sum(counts.values()) == 50
