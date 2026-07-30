# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for launcher-generated native server deployments."""

from __future__ import annotations

import os
import tomllib

from switchyard.cli.launchers.launcher_runtime import wait_for_proxy_ready
from switchyard.cli.launchers.native_server import (
    NativeDeployment,
    NativeServer,
    deterministic_deployment,
    passthrough_deployment,
)
from switchyard.lib.backends.llm_target import LlmTarget
from switchyard.lib.profiles import DeterministicRoutingConfig


def test_passthrough_deployment_runs_and_restores_credentials(monkeypatch) -> None:
    monkeypatch.delenv("SWITCHYARD_LAUNCH_API_KEY_0", raising=False)
    deployment = passthrough_deployment(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
    )

    server = NativeServer(deployment, port=None)
    try:
        assert wait_for_proxy_ready(server.port, timeout_s=2.0)
        snapshot = server.stats.snapshot_sync()
        assert snapshot["total_requests"] == 0
        assert snapshot["models"] == {}
    finally:
        server.close()

    assert "SWITCHYARD_LAUNCH_API_KEY_0" not in os.environ


def test_native_server_uses_explicit_config_path(tmp_path) -> None:
    generated = passthrough_deployment(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
    )
    assert isinstance(generated.config, str)
    config_path = tmp_path / "deployment.toml"
    config_path.write_text(generated.config)
    deployment = NativeDeployment(
        config=config_path,
        credentials=generated.credentials,
        models=generated.models,
    )

    server = NativeServer(deployment, port=None)
    try:
        assert wait_for_proxy_ready(server.port, timeout_s=2.0)
        assert config_path.exists()
    finally:
        server.close()


def test_deterministic_deployment_maps_routes_to_native_toml() -> None:
    config = DeterministicRoutingConfig(
        strong=LlmTarget(id="strong", model="provider/strong"),
        weak=LlmTarget(id="weak", model="provider/weak"),
        classifier=LlmTarget(id="classifier", model="provider/classifier"),
        fallback_target_on_evict="strong",
        classifier_min_confidence=0.4,
        classifier_recent_turn_window=6,
        session_affinity=True,
    )

    deployment = deterministic_deployment(
        config,
        additional_models=["provider/extra"],
        claude_aliases=True,
    )
    parsed = tomllib.loads(str(deployment.config))
    classifier = parsed["routes"]["route_0"]

    assert classifier["type"] == "llm_classifier"
    assert classifier["min_confidence"] == 0.4
    assert classifier["recent_turn_window"] == 6
    assert classifier["session_affinity"] is True
    assert "provider/extra" in deployment.models
    assert f"claude-{classifier['id']}" in deployment.models
