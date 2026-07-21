# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the routing-policy banner passed to print_ready_banner at launch.

Verifies the strategy summary helpers directly and that each launcher
(claude, codex, openclaw) calls the right helper for passthrough,
routing-profiles, and deterministic modes.
"""

import textwrap
from unittest.mock import MagicMock, patch

import pytest

from switchyard.cli.launchers.claude_code_launcher import launch_claude
from switchyard.cli.launchers.codex_cli_launcher import launch_codex
from switchyard.cli.launchers.launcher_runtime import (
    deterministic_strategy_summary,
    passthrough_strategy_summary,
    routing_profiles_strategy_summary,
)
from switchyard.cli.launchers.openclaw_launcher import launch_openclaw

_STAGE_ROUTER_YAML = textwrap.dedent("""\
    routes:
      my_route:
        type: stage_router
        confidence_threshold: 0.7
        strong:
          model: strong-model/v1
        weak:
          model: weak-model/v1
        classifier:
          model: clf-model/v1
""")

_PASSTHROUGH_YAML = textwrap.dedent("""\
    routes:
      my_route:
        type: passthrough
        model: some/model
""")


class TestStrategyHelpers:
    def test_passthrough_summary(self):
        """passthrough_strategy_summary returns 'passthrough → <model>'."""
        assert passthrough_strategy_summary("my/model") == "passthrough → my/model"

    def test_routing_profiles_stage_router(self, tmp_path):
        """routing_profiles_strategy_summary describes the default stage_router route."""
        p = tmp_path / "profiles.yaml"
        p.write_text(_STAGE_ROUTER_YAML)
        result = routing_profiles_strategy_summary(str(p), "my_route")
        assert result == "stage_router: strong=strong-model/v1, weak=weak-model/v1, llm-classifier=clf-model/v1, confidence_threshold=0.7"

    def test_routing_profiles_passthrough_type(self, tmp_path):
        """routing_profiles_strategy_summary describes a passthrough-type route."""
        p = tmp_path / "profiles.yaml"
        p.write_text(_PASSTHROUGH_YAML)
        result = routing_profiles_strategy_summary(str(p), "my_route")
        assert result == "passthrough → some/model"

    def test_routing_profiles_fallback_on_missing_file(self):
        """Falls back to default_model when the profiles file cannot be read."""
        result = routing_profiles_strategy_summary("/nonexistent/path.yaml", "fallback-model")
        assert result == "routing-profiles: fallback-model"

    def test_deterministic_summary(self):
        """deterministic_strategy_summary formats all config fields."""
        config = MagicMock()
        config.classifier.model = "clf-model"
        config.strong.model = "strong-model"
        config.weak.model = "weak-model"
        config.profile_name = "default"
        config.classifier_fail_open = True
        result = deterministic_strategy_summary(config)
        assert result == "llm-classifier: classifier=clf-model, strong=strong-model, weak=weak-model, profile=default, alert=switchyard_classifier_fail_open_triggered_total"


def _captured_strategy(captured: dict) -> str | None:
    return captured.get("strategy_summary")


def _patch_codex_runner(captured: dict):
    """Patch _run_codex_with_switchyard to capture kwargs and return 0."""
    def _fake_run(*args, **kwargs):
        captured.update(kwargs)
        return 0
    return patch(
        "switchyard.cli.launchers.codex_cli_launcher._run_codex_with_switchyard",
        side_effect=_fake_run,
    )


def _patch_claude_runner(captured: dict):
    def _fake_run(*args, **kwargs):
        captured.update(kwargs)
        return 0
    return patch(
        "switchyard.cli.launchers.claude_code_launcher._run_claude_with_switchyard",
        side_effect=_fake_run,
    )


def _patch_openclaw_runner(captured: dict):
    """Patch _run_openclaw_with_switchyard to capture kwargs and return 0."""
    def _fake_run(*args, **kwargs):
        captured.update(kwargs)
        return 0
    return patch(
        "switchyard.cli.launchers.openclaw_launcher._run_openclaw_with_switchyard",
        side_effect=_fake_run,
    )


@pytest.fixture(autouse=True)
def _patch_build_deps(monkeypatch):
    """Stub out chain-building so tests don't need real API keys or Rust init."""
    monkeypatch.setattr(
        "switchyard.cli.launchers.codex_cli_launcher._build_switchyard",
        lambda *a, **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "switchyard.cli.launchers.claude_code_launcher._build_claude_switchyard",
        lambda *a, **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "switchyard.cli.launchers.openclaw_launcher._build_switchyard",
        lambda *a, **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "switchyard.lib.route_table_builders.build_single_model_table",
        lambda *a, **kw: MagicMock(),
    )


class TestCodexRoutingBanner:
    def test_passthrough_banner(self):
        """launch_codex produces passthrough → <model> for single-model launch."""
        captured: dict = {}
        with _patch_codex_runner(captured):
            launch_codex(
                model="nvidia/moonshotai/kimi-k2.6",
                base_url="https://example.com/v1",
                api_key="key",
                port=4000,
                timeout=None,
                codex_args=[],
            )
        assert _captured_strategy(captured) == "passthrough → nvidia/moonshotai/kimi-k2.6"

    def test_routing_profiles_banner(self, tmp_path):
        """launch_codex describes the default route type for routing-profiles launch."""
        profiles_path = tmp_path / "profiles.yaml"
        profiles_path.write_text(_STAGE_ROUTER_YAML)
        captured: dict = {}
        with (
            _patch_codex_runner(captured),
            patch(
                "switchyard.cli.launchers.codex_cli_launcher.load_route_bundle_table",
                return_value=MagicMock(items=lambda: [], model_listing_warnings=lambda: []),
            ),
        ):
            launch_codex(
                model="my_route",
                base_url="https://example.com/v1",
                api_key="key",
                port=4000,
                timeout=None,
                codex_args=[],
                routing_profiles=str(profiles_path),
            )
        assert _captured_strategy(captured) == "stage_router: strong=strong-model/v1, weak=weak-model/v1, llm-classifier=clf-model/v1, confidence_threshold=0.7"


class TestOpenclawRoutingBanner:
    def test_passthrough_banner(self):
        """launch_openclaw produces passthrough → <model> for single-model launch."""
        captured: dict = {}
        with _patch_openclaw_runner(captured):
            launch_openclaw(
                model="nvidia/moonshotai/kimi-k2.6",
                base_url="https://example.com/v1",
                api_key="key",
                port=4000,
                timeout=None,
                openclaw_args=[],
            )
        assert _captured_strategy(captured) == "passthrough → nvidia/moonshotai/kimi-k2.6"

    def test_routing_profiles_banner(self, tmp_path):
        """launch_openclaw describes the default route type for routing-profiles launch."""
        profiles_path = tmp_path / "profiles.yaml"
        profiles_path.write_text(_STAGE_ROUTER_YAML)
        captured: dict = {}
        with (
            _patch_openclaw_runner(captured),
            patch(
                "switchyard.cli.launchers.openclaw_launcher.load_route_bundle_table",
                return_value=MagicMock(items=lambda: [], model_listing_warnings=lambda: []),
            ),
        ):
            launch_openclaw(
                model="my_route",
                base_url="https://example.com/v1",
                api_key="key",
                port=4000,
                timeout=None,
                openclaw_args=[],
                routing_profiles=str(profiles_path),
            )
        assert _captured_strategy(captured) == "stage_router: strong=strong-model/v1, weak=weak-model/v1, llm-classifier=clf-model/v1, confidence_threshold=0.7"


class TestClaudeRoutingBanner:
    def test_passthrough_banner(self):
        """launch_claude produces passthrough → <model> for single-model launch."""
        captured: dict = {}
        with _patch_claude_runner(captured):
            launch_claude(
                model="nvidia/moonshotai/kimi-k2.6",
                base_url="https://example.com/v1",
                api_key="key",
                port=4000,
                timeout=None,
                claude_args=[],
            )
        assert _captured_strategy(captured) == "passthrough → nvidia/moonshotai/kimi-k2.6"

    def test_routing_profiles_banner(self, tmp_path):
        """launch_claude describes the default route type for routing-profiles launch."""
        profiles_path = tmp_path / "profiles.yaml"
        profiles_path.write_text(_STAGE_ROUTER_YAML)
        captured: dict = {}
        with (
            _patch_claude_runner(captured),
            patch(
                "switchyard.cli.launchers.claude_code_launcher.load_route_bundle_table",
                return_value=MagicMock(items=lambda: [], model_listing_warnings=lambda: []),
            ),
        ):
            launch_claude(
                model="my_route",
                base_url="https://example.com/v1",
                api_key="key",
                port=4000,
                timeout=None,
                claude_args=[],
                routing_profiles=str(profiles_path),
            )
        assert _captured_strategy(captured) == "stage_router: strong=strong-model/v1, weak=weak-model/v1, llm-classifier=clf-model/v1, confidence_threshold=0.7"
