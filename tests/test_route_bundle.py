# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the minimal Python server route bundle."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

import switchyard.cli.switchyard_cli as cli
from switchyard.cli.launchers.launcher_runtime import route_bundle_strategy_summary
from switchyard.cli.route_bundle import RouteBundleConfigError, build_route_bundle_table
from switchyard.lib.backends.advisor_loop_backend import AdvisorLoopBackend
from switchyard.lib.processors.reasoning_effort_normalizer import ReasoningEffortNormalizer
from switchyard.lib.route_table import RouteTable
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.components import StatsLlmBackend
from switchyard_rust.core import ChatRequestType


async def test_noop_route_returns_ok_without_an_upstream() -> None:
    table = build_route_bundle_table({
        "routes": {"test/noop": {"type": "noop"}},
    })

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_switchyard_app(table)),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test/noop",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "OK"


@pytest.mark.parametrize(
    ("path", "body", "expected"),
    [
        (
            "/v1/messages",
            {
                "model": "test/noop",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
            },
            ("content", 0, "text"),
        ),
        (
            "/v1/responses",
            {"model": "test/noop", "input": "hello"},
            ("output", 0, "content"),
        ),
    ],
)
async def test_noop_route_translates_to_inbound_format(
    path: str,
    body: dict[str, object],
    expected: tuple[str, int, str],
) -> None:
    table = build_route_bundle_table({"routes": {"test/noop": {"type": "noop"}}})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_switchyard_app(table)),
        base_url="http://test",
    ) as client:
        response = await client.post(path, json=body)

    assert response.status_code == 200
    value: object = response.json()
    for part in expected:
        value = value[part]  # type: ignore[index]
    assert value


def test_passthrough_route_builds_one_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTREAM_KEY", "secret")
    table = build_route_bundle_table({
        "defaults": {
            "api_key": "${UPSTREAM_KEY}",
            "base_url": "https://example.invalid/v1",
            "format": "openai",
        },
        "routes": {
            "direct": {
                "type": "passthrough",
                "target": {"model": "upstream/model"},
                "display_name": "Direct model",
            }
        },
    })

    assert table.registered_models() == ["direct"]
    assert table.default_model() == "direct"
    assert table.registered_model_entries()[0]["display_name"] == "Direct model"
    components = table.lookup_switchyard("direct").iter_components()
    stats_backend = next(component for component in components if isinstance(component, StatsLlmBackend))
    assert isinstance(stats_backend, StatsLlmBackend)


def test_passthrough_summary_labels_the_model(tmp_path: Path) -> None:
    path = tmp_path / "routes.yaml"
    path.write_text("routes:\n  direct:\n    type: passthrough\n    target: upstream/model\n")

    assert route_bundle_strategy_summary(str(path), "direct") == (
        "passthrough: model=upstream/model"
    )


@pytest.mark.parametrize(
    "bundle, match",
    [
        ({}, "routes must be a mapping"),
        ({"routes": {}}, "at least one route"),
        ({"routes": {"r": {}}}, "missing string 'type'"),
        (
            {"routes": {"r": {"type": "random"}}},
            "expected 'noop', 'passthrough', or 'advisor'",
        ),
        (
            {"routes": {"r": {"type": "noop", "target": "unused"}}},
            "unknown key",
        ),
        (
            {"routes": {"r": {"type": "passthrough"}}},
            "target must be a mapping",
        ),
    ],
)
def test_invalid_bundles_fail_closed(bundle: object, match: str) -> None:
    with pytest.raises(RouteBundleConfigError, match=match):
        build_route_bundle_table(bundle)


def test_missing_environment_variable_is_rejected() -> None:
    with pytest.raises(RouteBundleConfigError, match="MISSING_ROUTE_KEY"):
        build_route_bundle_table({
            "defaults": {"api_key": "${MISSING_ROUTE_KEY}"},
            "routes": {"direct": {"type": "passthrough", "target": "model"}},
        })


def test_main_reports_missing_bundle_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(sys, "argv", ["switchyard", "serve", "--routes", str(missing)])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == f"error: invalid route bundle: {missing}: file not found"


def test_serve_loads_noop_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "routes.yaml"
    path.write_text("routes:\n  test/noop:\n    type: noop\n")
    captured: dict[str, object] = {}

    def fake_serve(args: object, switchyard: object, **kwargs: object) -> None:
        captured.update(args=args, switchyard=switchyard, **kwargs)

    monkeypatch.setattr(cli, "build_and_serve", fake_serve)
    parser = cli._build_parser()
    args = parser.parse_args(["serve", "--routes", str(path)])
    args.func(args)

    assert isinstance(captured["switchyard"], RouteTable)
    assert captured["switchyard"].registered_models() == ["test/noop"]


def _advisor_bundle() -> dict[str, object]:
    return {
        "routes": {
            "myrouter/advisor": {
                "type": "advisor",
                "display_name": "Advisor gate",
                "executor": {
                    "model": "aws/anthropic/bedrock-claude-opus-4-7",
                    "api_key": "",
                    "base_url": "https://exec.invalid/v1",
                    "format": "anthropic",
                    "extra_headers": {"Authorization": "Bearer sk-exec"},
                },
                "advisor": {
                    "model": "aws/anthropic/bedrock-claude-opus-4-8",
                    "api_key": "sk-adv",
                    "base_url": "https://adv.invalid/v1",
                    "format": "anthropic",
                },
            },
        },
    }


class TestAdvisorRouteType:
    """``type: advisor`` wires the executor + review-gate advisor chain via YAML."""

    def test_registers_and_builds_loop_backend(self) -> None:
        table = build_route_bundle_table(_advisor_bundle())
        assert table.registered_models() == ["myrouter/advisor"]
        assert table.registered_model_entries()[0]["display_name"] == "Advisor gate"
        components = table.lookup_switchyard("myrouter/advisor").iter_components()
        backend = next(c for c in components if isinstance(c, AdvisorLoopBackend))
        # Anthropic executor tier advertises the Anthropic wire.
        assert backend.supported_request_types == [ChatRequestType.ANTHROPIC]
        # Claude Code's ``/effort xhigh`` is normalized before the executor.
        assert any(isinstance(c, ReasoningEffortNormalizer) for c in components)

    def test_accumulator_injected_into_backend(self) -> None:
        # The advisor backend is Python-only, so it cannot be wrapped in
        # StatsLlmBackend; the bundle builder hands it the accumulator instead.
        stats = StatsAccumulator()
        table = build_route_bundle_table(_advisor_bundle(), stats_accumulator=stats)
        backend = next(
            c for c in table.iter_components() if isinstance(c, AdvisorLoopBackend)
        )
        assert backend._stats is stats
        assert not any(isinstance(c, StatsLlmBackend) for c in table.iter_components())

    def test_openai_wire_with_custom_redo_prefix(self) -> None:
        bundle = {
            "routes": {
                "myrouter/advisor-gate-oss": {
                    "type": "advisor",
                    "redo_feedback_prefix": "REVIEWER SAYS: ",
                    "executor": {
                        "model": "qwen/qwen3-max",
                        "api_key": "sk-exec",
                        "base_url": "https://exec.invalid/v1",
                        "format": "openai",
                    },
                    "advisor": {
                        "model": "deepseek/deepseek-r2",
                        "api_key": "sk-adv",
                        "base_url": "https://adv.invalid/v1",
                        "format": "openai",
                    },
                },
            },
        }
        table = build_route_bundle_table(bundle)
        backend = next(
            c for c in table.iter_components() if isinstance(c, AdvisorLoopBackend)
        )
        assert backend.supported_request_types == [ChatRequestType.OPENAI_CHAT]
        assert backend._config.redo_feedback_prefix == "REVIEWER SAYS: "

    def test_defaults_apply_to_tiers(self) -> None:
        table = build_route_bundle_table({
            "defaults": {
                "api_key": "sk-shared",
                "base_url": "https://shared.invalid/v1",
                "format": "openai",
            },
            "routes": {
                "adv": {
                    "type": "advisor",
                    "executor": "qwen/qwen3-max",
                    "advisor": "deepseek/deepseek-r2",
                },
            },
        })
        backend = next(
            c for c in table.iter_components() if isinstance(c, AdvisorLoopBackend)
        )
        assert backend._config.executor.model == "qwen/qwen3-max"
        assert backend._config.executor.endpoint.base_url == "https://shared.invalid/v1"
        assert backend._config.advisor.endpoint.api_key == "sk-shared"

    def test_rejects_unknown_route_key(self) -> None:
        bundle = _advisor_bundle()
        bundle["routes"]["myrouter/advisor"]["bogus_field"] = 1  # type: ignore[index]
        with pytest.raises(RouteBundleConfigError, match="bogus_field"):
            build_route_bundle_table(bundle)

    def test_missing_tier_rejected(self) -> None:
        bundle = _advisor_bundle()
        del bundle["routes"]["myrouter/advisor"]["advisor"]  # type: ignore[union-attr]
        with pytest.raises(RouteBundleConfigError, match="advisor target is required"):
            build_route_bundle_table(bundle)

    def test_responses_format_rejected(self) -> None:
        bundle = _advisor_bundle()
        bundle["routes"]["myrouter/advisor"]["executor"]["format"] = "responses"  # type: ignore[index]
        with pytest.raises(RouteBundleConfigError, match="responses"):
            build_route_bundle_table(bundle)

    def test_summary_labels_the_tiers(self, tmp_path: Path) -> None:
        path = tmp_path / "routes.yaml"
        path.write_text(
            "routes:\n"
            "  adv:\n"
            "    type: advisor\n"
            "    executor:\n"
            "      model: exec/model\n"
            "    advisor:\n"
            "      model: adv/model\n"
        )
        assert route_bundle_strategy_summary(str(path), "adv") == (
            "advisor: executor=exec/model, advisor=adv/model"
        )
