# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Rust server lifecycle and configuration generation for launchers."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from switchyard.cli.launchers.stats_source import StatsSource
from switchyard.lib.backends.llm_target import LlmTarget
from switchyard.lib.profiles import DeterministicRoutingConfig
from switchyard.lib.route_table_builders import deterministic_routing_virtual_model_id

log = logging.getLogger(__name__)

_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_LAUNCH_API_KEY_PREFIX = "SWITCHYARD_LAUNCH_API_KEY"


@dataclass(frozen=True)
class NativeDeployment:
    """Generated Rust server configuration and its resolved credentials."""

    config: str
    credentials: Mapping[str, str]
    models: tuple[str, ...]


class HttpStatsSource(StatsSource):
    """Reads launcher statistics from the native server's JSON endpoint."""

    def __init__(self, base_url: str) -> None:
        self._url = f"{base_url}/v1/stats"
        self._last_snapshot: Mapping[str, object] = {}

    def snapshot_sync(self) -> Mapping[str, object]:
        """Return fresh stats, retaining the last snapshot on transient failure."""
        try:
            with _LOCAL_OPENER.open(self._url, timeout=0.5) as response:
                payload = json.load(response)
            if isinstance(payload, dict):
                self._last_snapshot = payload
        except Exception:
            log.debug("failed to fetch native launcher stats", exc_info=True)
        return self._last_snapshot


class NativeServer:
    """Hosts a deployment through the PyO3 Rust server binding."""

    def __init__(self, deployment: NativeDeployment, port: int | None) -> None:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="switchyard-launch-",
            suffix=".toml",
            delete=False,
        )
        with handle:
            handle.write(deployment.config)
        config_path = Path(handle.name)

        try:
            with _temporary_environment(deployment.credentials):
                from switchyard_rust.server import Server

                self._server = Server(config_path, port=port or 0)
        finally:
            config_path.unlink(missing_ok=True)

        self.port: int = self._server.port
        self.base_url: str = self._server.base_url
        self.stats: StatsSource = HttpStatsSource(self.base_url)

    def close(self) -> None:
        """Gracefully stop the native server."""
        self._server.close()


def passthrough_deployment(
    *,
    model: str,
    base_url: str,
    api_key: str,
    claude_alias: bool = False,
) -> NativeDeployment:
    """Build a single-target OpenAI Chat deployment."""
    targets = [_Target(model=model, base_url=base_url, api_key=api_key)]
    routes = [_Route(id=model, type="passthrough", target=0)]
    if claude_alias and not model.startswith("claude-"):
        routes.append(_Route(id=f"claude-{model}", type="passthrough", target=0))
    return _render_deployment(targets, routes)


def deterministic_deployment(
    config: DeterministicRoutingConfig,
    *,
    additional_models: Sequence[str] = (),
    claude_aliases: bool = False,
) -> NativeDeployment:
    """Build the Rust LLM-classifier route and direct model overrides."""
    targets = [
        _target_from_llm_target(config.strong),
        _target_from_llm_target(config.weak),
        _target_from_llm_target(config.classifier),
    ]
    routing_model = deterministic_routing_virtual_model_id(config)
    routes = [
        _Route(
            id=routing_model,
            type="llm_classifier",
            classifier_target=2,
            strong_target=0,
            weak_target=1,
            base_threshold=0.5,
            min_confidence=config.classifier_min_confidence,
            recent_turn_window=config.classifier_recent_turn_window,
            session_affinity=config.session_affinity,
        ),
        _Route(id=config.strong.model, type="passthrough", target=0),
        _Route(id=config.weak.model, type="passthrough", target=1),
    ]

    seen = {config.strong.model, config.weak.model, config.classifier.model}
    for model in additional_models:
        if model in seen:
            continue
        seen.add(model)
        targets.append(
            _Target(
                model=model,
                base_url=config.strong.base_url or "",
                api_key=config.strong.api_key or "",
            )
        )
        routes.append(_Route(id=model, type="passthrough", target=len(targets) - 1))

    if claude_aliases:
        routes.extend(
            replace(route, id=f"claude-{route.id}")
            for route in tuple(routes)
            if not route.id.startswith("claude-")
        )
    return _render_deployment(targets, routes)


@dataclass(frozen=True)
class _Target:
    model: str
    base_url: str
    api_key: str
    extra_body: Mapping[str, Any] | None = None
    extra_headers: Mapping[str, str] | None = None


@dataclass(frozen=True)
class _Route:
    id: str
    type: str
    target: int | None = None
    classifier_target: int | None = None
    strong_target: int | None = None
    weak_target: int | None = None
    base_threshold: float | None = None
    min_confidence: float | None = None
    recent_turn_window: int | None = None
    session_affinity: bool | None = None


def _target_from_llm_target(target: LlmTarget) -> _Target:
    """Copy the Python-facing target fields needed by Rust server TOML."""
    extra_body = target.extra_body
    return _Target(
        model=target.model,
        base_url=target.base_url or "",
        api_key=target.api_key or "",
        extra_body=extra_body if isinstance(extra_body, dict) else None,
        extra_headers=target.extra_headers,
    )


def _render_deployment(
    targets: Sequence[_Target],
    routes: Sequence[_Route],
) -> NativeDeployment:
    """Render typed launcher inputs into the explicit Rust server schema."""
    lines = ["schema_version = 1", ""]
    credentials: dict[str, str] = {}
    for index, target in enumerate(targets):
        client = f"client_{index}"
        target_name = f"target_{index}"
        api_key_env = f"{_LAUNCH_API_KEY_PREFIX}_{index}"
        credentials[api_key_env] = target.api_key
        lines.extend([
            f"[llm_clients.{client}]",
            'format = "openai_chat"',
            f"base_url = {_toml_value(target.base_url)}",
            f"api_key_env = {_toml_value(api_key_env)}",
        ])
        if target.extra_headers:
            lines.append(f"extra_headers = {_toml_value(target.extra_headers)}")
        lines.extend([
            "",
            f"[targets.{target_name}]",
            f"id = {_toml_value(target.model)}",
            f'llm_client = "{client}"',
        ])
        if target.extra_body:
            lines.append(f"extra_body = {_toml_value(target.extra_body)}")
        lines.append("")

    for index, route in enumerate(routes):
        lines.extend([
            f"[routes.route_{index}]",
            f"id = {_toml_value(route.id)}",
            f"type = {_toml_value(route.type)}",
        ])
        if route.target is not None:
            lines.append(f'target = "target_{route.target}"')
        if route.classifier_target is not None:
            lines.extend([
                f'classifier_target = "target_{route.classifier_target}"',
                f'strong_target = "target_{route.strong_target}"',
                f'weak_target = "target_{route.weak_target}"',
                f"base_threshold = {route.base_threshold}",
                f"min_confidence = {route.min_confidence}",
                f"recent_turn_window = {route.recent_turn_window}",
                f"session_affinity = {_toml_value(route.session_affinity)}",
            ])
        lines.append("")

    return NativeDeployment(
        config="\n".join(lines),
        credentials=credentials,
        models=tuple(route.id for route in routes),
    )


def _toml_value(value: object) -> str:
    """Render the JSON-compatible subset used by launcher deployments."""
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        entries = (
            f"{json.dumps(str(key))} = {_toml_value(item)}"
            for key, item in value.items()
        )
        return "{ " + ", ".join(entries) + " }"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


@contextlib.contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Expose resolved credentials only while Rust constructs its clients."""
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, prior in previous.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


__all__ = [
    "NativeDeployment",
    "NativeServer",
    "deterministic_deployment",
    "passthrough_deployment",
]
