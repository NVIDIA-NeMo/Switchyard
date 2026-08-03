# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a Python server route table from a minimal YAML bundle."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from switchyard.lib.backends.llm_target import LlmTarget, coerce_llm_target
from switchyard.lib.backends.multi_llm_backend import build_native_backend
from switchyard.lib.backends.stats_llm_backend import StatsLlmBackend
from switchyard.lib.processors.stats_request_processor import StatsRequestProcessor
from switchyard.lib.processors.stats_response_processor_accumulator import (
    StatsResponseProcessor,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard.lib.route_table import ChainRuntime, RouteTable
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard.lib.switchyard import Switchyard
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse
from switchyard_rust.translation import TranslationEngine

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_TOP_LEVEL_KEYS = frozenset({"defaults", "routes"})
_ROUTE_METADATA_KEYS = frozenset({"display_name", "description"})
_ROUTE_KEYS = {
    "noop": frozenset({"type"}) | _ROUTE_METADATA_KEYS,
    "passthrough": frozenset({"type", "target"}) | _ROUTE_METADATA_KEYS,
}


class RouteBundleConfigError(ValueError):
    """Raised when a Python server route bundle is invalid."""


class _YamlModule(Protocol):
    def safe_load(self, stream: str) -> object: ...


class _NoopBackend(LLMBackend):
    """Return a fixed response without making an upstream request."""

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return list(ChatRequestType)

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        model = request.model or "switchyard/noop"
        ctx.selected_model = model
        ctx.selected_target = model
        return ChatResponse.openai_completion({
            "id": "switchyard-noop",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        })


def parse_route_bundle_file(path: str | Path) -> dict[str, object]:
    """Read a YAML route bundle and return its top-level mapping."""
    resolved = Path(path)
    try:
        contents = resolved.read_text()
    except FileNotFoundError as error:
        raise RouteBundleConfigError(f"{resolved}: file not found") from error
    except (OSError, UnicodeError) as error:
        raise RouteBundleConfigError(f"{resolved}: cannot read: {error}") from error

    try:
        yaml = cast(_YamlModule, import_module("yaml"))
        raw = yaml.safe_load(contents)
    except Exception as error:
        message = " ".join(str(error).splitlines())
        raise RouteBundleConfigError(f"{resolved}: invalid YAML: {message}") from error
    return _mapping(raw, "route bundle")


def load_route_bundle_table(
    path: str | Path,
    *,
    stats_accumulator: StatsAccumulator | None = None,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> RouteTable:
    """Load a YAML route bundle into a server dispatch table."""
    return build_route_bundle_table(
        parse_route_bundle_file(path),
        stats_accumulator=stats_accumulator,
        pre_routing_request_processors=pre_routing_request_processors,
        extra_response_processors=extra_response_processors,
    )


def build_route_bundle_table(
    raw: object,
    *,
    stats_accumulator: StatsAccumulator | None = None,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> RouteTable:
    """Build a table containing only noop and passthrough routes."""
    bundle = _mapping(_expand_env(raw), "route bundle")
    _reject_unknown_keys(bundle, _TOP_LEVEL_KEYS, "route bundle")
    defaults = _mapping(bundle.get("defaults", {}), "defaults")
    routes = _mapping(bundle.get("routes"), "routes")
    if not routes:
        raise RouteBundleConfigError("routes must contain at least one route")

    stats = stats_accumulator or StatsAccumulator()
    table = RouteTable()
    for route_id, raw_route in routes.items():
        if not route_id:
            raise RouteBundleConfigError("route ids must be non-empty strings")
        route = _mapping(raw_route, f"route {route_id!r}")
        route_type = route.get("type")
        if not isinstance(route_type, str):
            raise RouteBundleConfigError(f"route {route_id!r}: missing string 'type'")
        if route_type not in _ROUTE_KEYS:
            raise RouteBundleConfigError(
                f"route {route_id!r}: unsupported route type {route_type!r}; "
                "expected 'noop' or 'passthrough'"
            )
        _reject_unknown_keys(route, _ROUTE_KEYS[route_type], f"route {route_id!r}")

        if route_type == "noop":
            runtime = _build_runtime(
                _NoopBackend(),
                stats,
                pre_routing_request_processors,
                extra_response_processors,
            )
        else:
            target = _target(route_id, route.get("target"), defaults)
            runtime = _build_runtime(
                StatsLlmBackend(build_native_backend(target), stats),
                stats,
                pre_routing_request_processors,
                extra_response_processors,
            )

        metadata = {
            key: value
            for key in _ROUTE_METADATA_KEYS
            if (value := route.get(key)) is not None
        }
        table.register(route_id, runtime, metadata=metadata, default=table.default_model() is None)
    return table


def _build_runtime(
    backend: LLMBackend,
    stats: StatsAccumulator,
    request_processors: Sequence[Any],
    response_processors: Sequence[Any],
) -> ChainRuntime:
    return Switchyard(
        request_processors=[StatsRequestProcessor(), *request_processors],
        backend=backend,
        response_processors=[StatsResponseProcessor(stats), *response_processors],
        translator=TranslationEngine(),
    )


def _target(route_id: str, value: object, defaults: Mapping[str, object]) -> LlmTarget:
    if isinstance(value, str):
        target: dict[str, object] = {"model": value}
    else:
        target = _mapping(value, f"route {route_id!r} target")
    try:
        return coerce_llm_target({**defaults, **target}, default_id=route_id)
    except (TypeError, ValueError) as error:
        raise RouteBundleConfigError(f"route {route_id!r}: invalid target: {error}") from error


def _expand_env(value: object) -> object:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise RouteBundleConfigError(f"environment variable {name} is not set")
        return os.environ[name]

    return _ENV_REF_RE.sub(replace, value)


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RouteBundleConfigError(f"{where} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise RouteBundleConfigError(f"{where} keys must be strings")
    return {str(key): item for key, item in value.items()}


def _reject_unknown_keys(
    value: Mapping[str, object], allowed: frozenset[str], where: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RouteBundleConfigError(f"unknown key(s) for {where}: {', '.join(unknown)}")


__all__ = [
    "RouteBundleConfigError",
    "build_route_bundle_table",
    "load_route_bundle_table",
    "parse_route_bundle_file",
]
