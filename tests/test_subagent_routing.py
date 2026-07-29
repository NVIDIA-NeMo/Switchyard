# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve-path sub-agent routing: the ``subagent_target`` route envelope key."""

from __future__ import annotations

from typing import Any

import pytest

from switchyard.cli.route_bundle import (
    RouteBundle,
    RouteBundleConfigError,
    build_table_from_bundle,
)
from switchyard.lib.profiles.subagent_override import SubagentOverrideRuntime
from switchyard.lib.request_metadata import CTX_PROFILE_REQUEST_HEADERS
from switchyard_rust.core import is_subagent_request


class _RecordingRuntime:
    """A ChainRuntime stand-in that records the calls routed to it."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def call(self, request: Any, *, ctx: Any = None) -> str:
        self.calls += 1
        return self.name

    def iter_components(self) -> list[Any]:
        return [self.name]


class _Ctx:
    """Minimal ProxyContext stand-in exposing only the retained header map."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.metadata: dict[str, Any] = {}
        if headers is not None:
            self.metadata[CTX_PROFILE_REQUEST_HEADERS] = headers


# --- detection policy -----------------------------------------------------------------


def test_detection_matches_the_protocol_policy() -> None:
    # Claude Code child-agent lineage is delegated work.
    assert is_subagent_request(
        {"x-claude-code-session-id": "root", "x-claude-code-agent-id": "child-1"}
    )
    # Codex delegated-work kinds.
    assert is_subagent_request({"x-openai-subagent": "review"})
    # Harness maintenance is sub-agent lineage but NOT delegated work.
    assert not is_subagent_request({"x-openai-subagent": "compact"})
    # Explicit opt-out wins.
    assert not is_subagent_request({"x-switchyard-is-subagent": "false"})
    # Ordinary traffic.
    assert not is_subagent_request({})


# --- runtime wrapper ------------------------------------------------------------------


async def test_delegated_work_goes_to_the_worker() -> None:
    inner, worker = _RecordingRuntime("inner"), _RecordingRuntime("worker")
    runtime = SubagentOverrideRuntime(inner, worker)

    served = await runtime.call(
        object(),
        ctx=_Ctx({"x-claude-code-session-id": "s", "x-claude-code-agent-id": "child-1"}),
    )
    assert served == "worker"
    assert (inner.calls, worker.calls) == (0, 1)


async def test_normal_traffic_runs_the_wrapped_runtime() -> None:
    inner, worker = _RecordingRuntime("inner"), _RecordingRuntime("worker")
    runtime = SubagentOverrideRuntime(inner, worker)

    assert await runtime.call(object(), ctx=_Ctx({})) == "inner"
    # A maintenance turn carries lineage but is not delegated work.
    assert await runtime.call(object(), ctx=_Ctx({"x-openai-subagent": "compact"})) == "inner"
    assert (inner.calls, worker.calls) == (2, 0)


async def test_missing_context_routes_normally() -> None:
    inner, worker = _RecordingRuntime("inner"), _RecordingRuntime("worker")
    runtime = SubagentOverrideRuntime(inner, worker)

    # No context means no retained headers, so nothing identifies the request.
    assert await runtime.call(object(), ctx=None) == "inner"
    assert worker.calls == 0


def test_components_span_both_branches() -> None:
    runtime = SubagentOverrideRuntime(_RecordingRuntime("inner"), _RecordingRuntime("worker"))
    assert runtime.iter_components() == ["inner", "worker"]


# --- route bundle wiring --------------------------------------------------------------


def _bundle(route: dict[str, Any]) -> RouteBundle:
    return RouteBundle(
        routes={"my-route": route},
        defaults={"api_key": "sk-test", "base_url": "https://example.invalid/v1"},
    )


def test_subagent_target_wraps_any_route_type() -> None:
    table = build_table_from_bundle(
        _bundle({
            "type": "model",
            "target": {"model": "strong-model"},
            "subagent_target": {"model": "worker-model"},
        })
    )
    assert isinstance(table.lookup_switchyard("my-route"), SubagentOverrideRuntime)


def test_routes_without_the_key_are_unwrapped() -> None:
    table = build_table_from_bundle(_bundle({"type": "model", "target": {"model": "strong-model"}}))
    assert not isinstance(table.lookup_switchyard("my-route"), SubagentOverrideRuntime)


def test_unknown_route_keys_are_still_rejected() -> None:
    with pytest.raises(RouteBundleConfigError, match="subagent_targets"):
        build_table_from_bundle(
            _bundle({
                "type": "model",
                "target": {"model": "strong-model"},
                "subagent_targets": {"model": "typo"},
            })
        )
