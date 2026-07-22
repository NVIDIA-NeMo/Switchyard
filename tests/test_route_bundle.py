# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for YAML route-bundle table construction."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import switchyard.cli.switchyard_cli as cli
from switchyard.cli.route_bundle import (
    RouteBundleConfigError,
    build_route_bundle_table,
)
from switchyard.lib.processors.llm_classifier import (
    CODING_AGENT_CLASSIFIER_SYSTEM_PROMPT,
    LLMClassifierRequestProcessor,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.route_table import RouteTable
from switchyard_rust.core import ChatRequest, ChatResponse


@pytest.fixture(autouse=True)
def _stub_model_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catalog discovery is always-on for route bundles; default to an empty
    catalog so unit tests stay hermetic. Tests that assert on hydrated catalog
    entries override this with their own ``monkeypatch.setattr``.
    """
    monkeypatch.setattr(
        "switchyard.cli.route_bundle.fetch_model_ids",
        lambda base_url, api_key: [],
    )


def _random_processor(chain: object) -> Any:
    from switchyard.lib.processors.random_routing_request_processor import (
        RandomRoutingRequestProcessor,
    )

    return next(
        component
        for component in chain.iter_components()
        if isinstance(component, RandomRoutingRequestProcessor)
    )


def _latency_backend(chain: object) -> Any:
    from switchyard.lib.backends.latency_service_llm_backend import (
        LatencyServiceLLMBackend,
    )

    return next(
        component
        for component in chain.iter_components()
        if isinstance(component, LatencyServiceLLMBackend)
    )


class _NoopRequestProcessor:
    async def process(self, _ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        return request


class _NoopResponseProcessor:
    async def process(self, _ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        return response


def test_random_route_bundle_registers_model_keys_and_applies_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTE_BUNDLE_KEY", "sk-default")

    table = build_route_bundle_table({
        "defaults": {
            "api_key": "${ROUTE_BUNDLE_KEY}",
            "base_url": "https://example.invalid/v1",
            "format": "openai",
            "timeout": 30,
        },
        "routes": {
            "A": {
                "type": "random-routing",
                "fallback_target_on_evict": "strong",
                "strong": "model-1",
                "weak": {"model": "model-2", "api_key": "sk-weak"},
                "strong_probability": 0.7,
            },
            "B": {
                "type": "random_routing",
                "fallback_target_on_evict": "strong",
                "strong": "model-3",
                "weak": "model-4",
                "strong_probability": 0.2,
            },
        },
    })

    # Unified ordering: YAML route key first, then tier passthroughs.
    # Catalog hydration is always attempted; the stubbed catalog is empty here,
    # so only configured tier models register.
    assert table.registered_models() == [
        "A",         # A's virtual routing-policy id (YAML key)
        "model-1",   # A's strong tier
        "model-2",   # A's weak tier
        "B",         # B's virtual routing-policy id (YAML key)
        "model-3",   # B's strong tier
        "model-4",   # B's weak tier
    ]
    assert table.default_model() == "A"

    route_a = _random_processor(table.lookup_switchyard("A"))
    assert route_a.config.strong.model == "model-1"
    assert route_a.config.strong.api_key == "sk-default"
    assert route_a.config.strong.endpoint.base_url == "https://example.invalid/v1"
    assert route_a.config.weak.model == "model-2"
    assert route_a.config.weak.api_key == "sk-weak"
    assert route_a.config.strong_probability == 0.7

    route_b = _random_processor(table.lookup_switchyard("B"))
    assert route_b.config.strong.model == "model-3"
    assert route_b.config.weak.model == "model-4"
    assert route_b.config.strong_probability == 0.2


def test_route_bundle_rejects_missing_environment_variable() -> None:
    with pytest.raises(RouteBundleConfigError, match="MISSING_ROUTE_KEY"):
        build_route_bundle_table({
            "defaults": {"api_key": "${MISSING_ROUTE_KEY}"},
            "routes": {"A": "model-a"},
        })


@pytest.mark.parametrize(
    ("route", "match"),
    [
        ({"modle": "gpt-4o"}, "recognizable route shape"),
        ({"type": "model", "modle": "gpt-4o"}, "modle"),
        (
            {
                "type": "random-routing",
                "fallback_target_on_evict": "strong",
                "strong": "model-1",
                "weak": "model-2",
                "strong_probablity": 0.9,
            },
            "strong_probablity",
        ),
        ({"type": "passthrough", "target": "gpt-4o"}, "unsupported route type"),
        ({"type": "noop"}, "unsupported route type"),
        ({"type": "model", "target": {"modle": "gpt-4o"}}, "modle"),
    ],
)
def test_route_bundle_rejects_unknown_route_keys(
    route: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(RouteBundleConfigError, match=match):
        build_route_bundle_table({"routes": {"bad": route}})


def test_empty_route_mapping_is_rejected() -> None:
    with pytest.raises(RouteBundleConfigError, match="recognizable route shape"):
        build_route_bundle_table({"routes": {"empty": {}}})


def test_random_routing_hydrates_tier_and_catalog_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A random_routing route always mirrors launcher catalog hydration.

    Registers each configured tier model + each tier's catalog model + the
    virtual routing-policy id (the route's YAML key). One YAML route expands
    into the same N+1 shape ``build_random_routing_table`` produces for the
    Claude/Codex launchers.
    """
    monkeypatch.setattr(
        "switchyard.cli.route_bundle.fetch_model_ids",
        lambda base_url, api_key: (
            ["catalog/extra"]
            if base_url == "https://primary.example/v1"
            else ["catalog/weak-extra"]
        ),
    )

    table = build_route_bundle_table({
        "routes": {
            "switchyard-route": {
                "type": "random_routing",
                "fallback_target_on_evict": "strong",
                "strong": {
                    "model": "strong/model",
                    "api_key": "k-strong",
                    "base_url": "https://primary.example/v1",
                },
                "weak": {
                    "model": "weak/model",
                    "api_key": "k-weak",
                    "base_url": "https://weak.example/v1",
                },
                "strong_probability": 0.4,
            },
        },
    })

    # Unified ordering: YAML route key first, tier passthroughs + catalog after.
    assert table.registered_models() == [
        "switchyard-route",
        "strong/model",
        "weak/model",
        "catalog/extra",
        "catalog/weak-extra",
    ]
    assert table.default_model() == "switchyard-route"


def test_model_route_aliases_under_route_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``model`` route is a pure alias — no catalog discovery.

    It registers only under the route's YAML key (the friendly alias); the
    upstream ``target.model`` is what the backend calls, not a separate
    table entry, and no catalog is hydrated even if the upstream has one.
    """
    monkeypatch.setattr(
        "switchyard.cli.route_bundle.fetch_model_ids",
        lambda base_url, api_key: ["catalog/a", "catalog/b"],
    )

    table = build_route_bundle_table({
        "routes": {
            "configured-route": {
                "type": "model",
                "target": {
                    "model": "primary/model",
                    "api_key": "k",
                    "base_url": "https://primary.example/v1",
                },
            },
        },
    })

    # Only the alias key registers — no `primary/model`, no catalog entries.
    assert table.registered_models() == ["configured-route"]
    assert table.default_model() == "configured-route"


def test_random_routing_with_empty_catalog_registers_only_tier_passthroughs() -> None:
    """When discovery yields an empty catalog, only tier passthroughs register.

    Matches the launcher's contract: client model pickers always see strong/
    weak as direct overrides for the random-routing default, even when the
    upstream ``/v1/models`` catalog is empty or unreachable.
    """
    table = build_route_bundle_table({
        "routes": {
            "switchyard-route": {
                "type": "random_routing",
                "fallback_target_on_evict": "strong",
                "strong": {
                    "model": "strong/model",
                    "api_key": "k-strong",
                    "base_url": "https://primary.example/v1",
                },
                "weak": {
                    "model": "weak/model",
                    "api_key": "k-weak",
                    "base_url": "https://weak.example/v1",
                },
            },
        },
    })

    # Unified ordering: YAML route key first, then tier passthroughs.
    # No catalog entries (empty catalog).
    assert table.registered_models() == [
        "switchyard-route",
        "strong/model",
        "weak/model",
    ]
    assert table.default_model() == "switchyard-route"


def test_route_bundle_threads_extra_processors_through_routes() -> None:
    request_processor = _NoopRequestProcessor()
    response_processor = _NoopResponseProcessor()

    table = build_route_bundle_table(
        {"routes": {"single-model": {"type": "model", "target": "upstream/model"}}},
        pre_routing_request_processors=[request_processor],
        extra_response_processors=[response_processor],
    )

    components = table.lookup_switchyard("single-model").iter_components()
    assert request_processor in components
    assert response_processor in components


def test_serve_subcommand_hands_table_to_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import switchyard.cli.switchyard_cli as cli

    yaml_path = tmp_path / "routes.yaml"
    yaml_path.write_text("routes:\n  single-model:\n    type: model\n    target: upstream/model\n")
    captured: dict[str, Any] = {}

    def _fake_serve(
        args: Any,
        switchyard: object,
        inbound_default: str,
        **_kwargs: object,
    ) -> None:
        captured["args"] = args
        captured["switchyard"] = switchyard
        captured["inbound_default"] = inbound_default

    monkeypatch.setattr(cli, "build_and_serve", _fake_serve)

    args = cli._build_parser().parse_args([
        "--routing-profiles",
        str(yaml_path),
        "serve",
        "--port",
        "4555",
    ])
    args.func(args)

    assert isinstance(captured["switchyard"], RouteTable)
    assert captured["switchyard"].registered_models() == ["single-model"]
    assert captured["args"].port == 4555
    assert captured["inbound_default"] == "both"


def test_serve_config_and_routing_profiles_are_mutually_exclusive(
    tmp_path,
) -> None:
    config_path = tmp_path / "profiles.yaml"
    routes_path = tmp_path / "routes.yaml"
    config_path.write_text("profiles:\n  bench:\n    type: noop\n")
    routes_path.write_text("routes:\n  bench:\n    type: model\n    target: upstream/model\n")

    args = cli._build_parser().parse_args([
        "--routing-profiles",
        str(routes_path),
        "serve",
        "--config",
        str(config_path),
    ])

    with pytest.raises(SystemExit, match="cannot be combined"):
        args.func(args)


@pytest.mark.parametrize(
    "serve_args, match",
    [
        (["--reload"], "--reload"),
        (["--workers", "2"], "--workers"),
        (["--inbound", "openai"], "--inbound"),
        (["--inbound", "both"], "--inbound"),
        (["--intake-enabled"], "Intake"),
    ],
)
def test_serve_config_rejects_python_only_options(
    tmp_path,
    serve_args: list[str],
    match: str,
) -> None:
    config_path = tmp_path / "profiles.yaml"
    config_path.write_text("profiles:\n  bench:\n    type: noop\n")
    args = cli._build_parser().parse_args([
        "serve",
        "--config",
        str(config_path),
        *serve_args,
    ])

    with pytest.raises(SystemExit, match=match):
        args.func(args)


def test_main_reports_route_bundle_config_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "unknown-key.yaml"
    yaml_path.write_text(
        "routes:\n"
        "  r:\n"
        "    type: model\n"
        "    target: x\n"
        "    bogus_key: 1\n"
    )
    monkeypatch.setattr(
        sys, "argv", ["switchyard", "--routing-profiles", str(yaml_path), "serve"]
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    # A clean one-line diagnostic, not a raw traceback, and a non-zero exit.
    assert excinfo.value.code == (
        "error: invalid route bundle: unknown key(s) for route 'r': bogus_key"
    )


def test_main_reports_missing_route_bundle_file_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "does-not-exist.yaml"
    monkeypatch.setattr(
        sys, "argv", ["switchyard", "--routing-profiles", str(missing_path), "serve"]
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == (
        f"error: invalid route bundle: {missing_path}: file not found"
    )


def test_main_reports_bad_route_bundle_yaml_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "bad_yaml.yaml"
    yaml_path.write_text(
        "routes:\n"
        "  r:\n"
        "    type: model\n"
        "   target: x\n"
    )
    monkeypatch.setattr(
        sys, "argv", ["switchyard", "--routing-profiles", str(yaml_path), "serve"]
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert str(excinfo.value.code).startswith(
        f"error: invalid route bundle: {yaml_path}: invalid YAML: "
    )
    assert "\n" not in str(excinfo.value.code)


def test_main_reports_non_utf8_route_bundle_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "non_utf8.yaml"
    yaml_path.write_bytes(b"\xff")
    monkeypatch.setattr(
        sys, "argv", ["switchyard", "--routing-profiles", str(yaml_path), "serve"]
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert str(excinfo.value.code).startswith(
        f"error: invalid route bundle: {yaml_path}: cannot read: "
    )
    assert "\n" not in str(excinfo.value.code)


def test_main_warns_when_routing_profiles_flag_is_used(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    yaml_path = tmp_path / "routes.yaml"
    yaml_path.write_text("routes:\n  single-model:\n    type: model\n    target: upstream/model\n")

    monkeypatch.setattr(
        sys, "argv", ["switchyard", "--routing-profiles", str(yaml_path), "serve"]
    )

    def _fake_serve(
        args: argparse.Namespace,
        switchyard: object,
        inbound_default: str,
        **_kwargs: object,
    ) -> None:
        return

    monkeypatch.setattr(cli, "build_and_serve", _fake_serve)

    cli.main()

    stderr = capsys.readouterr().err
    assert "warning: --routing-profiles is deprecated." in stderr
    assert "switchyard serve --config PATH" in stderr
    assert "removed in a future release" in stderr


def test_serve_warns_when_saved_route_bundle_is_used(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from switchyard.cli.config.user_config import UserConfig, save_user_config

    monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
    save_user_config(UserConfig(routing_profiles={"routes": {"single-model": {"type": "model", "target": "upstream/model"}}}))

    def _fake_serve(
        args: argparse.Namespace,
        switchyard: object,
        inbound_default: str,
        **_kwargs: object,
    ) -> None:
        return

    monkeypatch.setattr(cli, "build_and_serve", _fake_serve)
    args = cli._build_parser().parse_args(["serve", "--port", "4000"])

    cli._cmd_serve(args)

    stderr = capsys.readouterr().err
    assert "warning: saved routing-profile bundle is deprecated." in stderr
    assert "switchyard serve --config PATH" in stderr
    assert "Clear the saved bundle" in stderr


def test_serve_subcommand_enables_intake_from_cli_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import switchyard.cli.switchyard_cli as cli
    from switchyard.lib.processors import (
        IntakeRequestProcessor,
        IntakeResponseProcessor,
    )

    yaml_path = tmp_path / "routes.yaml"
    yaml_path.write_text(
        "routes:\n"
        "  single-model:\n"
        "    type: model\n"
        "    target: upstream/model\n"
    )
    captured: dict[str, Any] = {}

    def _fake_serve(
        args: Any,
        switchyard: object,
        inbound_default: str,
        **_kwargs: object,
    ) -> None:
        captured["switchyard"] = switchyard

    monkeypatch.setattr(cli, "build_and_serve", _fake_serve)

    args = cli._build_parser().parse_args([
        "--routing-profiles",
        str(yaml_path),
        "serve",
        "--intake-enabled",
        "--intake-base-url",
        "https://intake.example.test",
        "--intake-api-key",
        "sk-intake",
        "--port",
        "4555",
    ])
    args.func(args)

    components = captured["switchyard"].lookup_switchyard("single-model").iter_components()
    assert any(isinstance(component, IntakeRequestProcessor) for component in components)
    assert any(isinstance(component, IntakeResponseProcessor) for component in components)


class TestEscalationRouterRouteType:
    """`type: escalation_router` wires the judge-latched chain via YAML."""

    def _bundle(self) -> dict:
        return {
            "routes": {
                "myrouter/escalation": {
                    "type": "escalation_router",
                    "fallback_target_on_evict": "strong",
                    "judge": {
                        "model": "google/gemini-3.5-flash",
                        "api_key": "sk-judge",
                        "base_url": "https://judge.invalid/v1",
                        "timeout_secs": 5.0,
                        "min_turn": 4,
                        "recent_turn_window": 10,
                    },
                    "strong": {
                        "model": "openai/gpt-5.2",
                        "api_key": "sk-strong",
                        "base_url": "https://strong.invalid/v1",
                    },
                    "weak": {
                        "model": "deepseek/deepseek-v4-pro",
                        "api_key": "sk-weak",
                        "base_url": "https://weak.invalid/v1",
                    },
                    "session_key_depth": 2,
                },
            },
        }

    def test_registers_route_key_and_tier_passthroughs(self):
        from switchyard.cli.route_bundle import build_route_bundle_table
        table = build_route_bundle_table(self._bundle())
        # Route key first, tiers as direct passthroughs; the judge tier is
        # internal-only and never registered.
        assert table.registered_models() == [
            "myrouter/escalation",
            "openai/gpt-5.2",
            "deepseek/deepseek-v4-pro",
        ]

    def test_judge_settings_thread_into_processor(self):
        from switchyard.cli.route_bundle import build_route_bundle_table
        from switchyard.lib.processors.escalation_judge_request_processor import (
            EscalationJudgeRequestProcessor,
        )
        table = build_route_bundle_table(self._bundle())
        switchyard = table.lookup_switchyard("myrouter/escalation")
        judge = next(
            c for c in switchyard.iter_components()
            if isinstance(c, EscalationJudgeRequestProcessor)
        )
        assert judge._config.model == "google/gemini-3.5-flash"
        assert judge._config.min_judge_turn == 4
        assert judge._config.recent_turn_window == 10
        assert judge._config.timeout_s == 5.0
        assert judge._session_key_depth == 2

    def test_defaults_come_from_the_config_model(self):
        """Omitted keys inherit EscalationRouterConfig defaults, owned once."""
        from switchyard.cli.route_bundle import build_route_bundle_table
        from switchyard.lib.processors.escalation_judge_request_processor import (
            EscalationJudgeRequestProcessor,
        )
        bundle = self._bundle()
        del bundle["routes"]["myrouter/escalation"]["judge"]["timeout_secs"]
        table = build_route_bundle_table(bundle)
        switchyard = table.lookup_switchyard("myrouter/escalation")
        judge = next(
            c for c in switchyard.iter_components()
            if isinstance(c, EscalationJudgeRequestProcessor)
        )
        assert judge._config.escalate_confirmations == 1
        assert judge._config.window_message_chars == 300
        assert judge._config.dump_verdicts_to_stderr is False
        assert judge._config.max_completion_tokens == 128
        assert judge._config.timeout_s == 5.0
        assert judge._affinity._l2 is None

    def test_invalid_value_is_a_one_line_config_error(self):
        """Pydantic rejections surface as RouteBundleConfigError, not a traceback."""
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        bundle["routes"]["myrouter/escalation"]["session_key_depth"] = "two"
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "session_key_depth" in str(exc.value)

    def test_benchmark_knobs_and_redis_latch_thread_through(self):
        from switchyard.cli.route_bundle import build_route_bundle_table
        from switchyard.lib.processors.escalation_judge_request_processor import (
            EscalationJudgeRequestProcessor,
        )
        from switchyard.lib.redis_pin_store import RedisPinStore
        bundle = self._bundle()
        route = bundle["routes"]["myrouter/escalation"]
        route["judge"]["dump_verdicts"] = True
        route["judge"]["max_completion_tokens"] = 2048
        route["affinity_store"] = "redis"
        route["affinity_store_url"] = "redis://cache:6379/0"
        table = build_route_bundle_table(bundle)
        switchyard = table.lookup_switchyard("myrouter/escalation")
        judge = next(
            c for c in switchyard.iter_components()
            if isinstance(c, EscalationJudgeRequestProcessor)
        )
        assert judge._config.dump_verdicts_to_stderr is True
        assert judge._config.max_completion_tokens == 2048
        assert isinstance(judge._affinity._l2, RedisPinStore)

    def test_judge_prompt_path_reads_file(self, tmp_path):
        from switchyard.cli.route_bundle import build_route_bundle_table
        from switchyard.lib.processors.escalation_judge_request_processor import (
            EscalationJudgeRequestProcessor,
        )
        prompt_file = tmp_path / "judge_prompt.md"
        prompt_file.write_text("file-based judge prompt\n", encoding="utf-8")
        bundle = self._bundle()
        bundle["routes"]["myrouter/escalation"]["judge"]["prompt_path"] = (
            str(prompt_file)
        )
        table = build_route_bundle_table(bundle)
        switchyard = table.lookup_switchyard("myrouter/escalation")
        judge = next(
            c for c in switchyard.iter_components()
            if isinstance(c, EscalationJudgeRequestProcessor)
        )
        assert judge._config.system_prompt == "file-based judge prompt\n"

    def test_judge_prompt_and_prompt_path_are_mutually_exclusive(self, tmp_path):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        prompt_file = tmp_path / "judge_prompt.md"
        prompt_file.write_text("file prompt", encoding="utf-8")
        bundle = self._bundle()
        judge = bundle["routes"]["myrouter/escalation"]["judge"]
        judge["prompt"] = "inline prompt"
        judge["prompt_path"] = str(prompt_file)
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "mutually exclusive" in str(exc.value)

    def test_judge_prompt_path_missing_file_is_a_config_error(self, tmp_path):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        bundle["routes"]["myrouter/escalation"]["judge"]["prompt_path"] = (
            str(tmp_path / "nope.md")
        )
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "prompt_path" in str(exc.value)

    def test_rejects_missing_judge_block(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        del bundle["routes"]["myrouter/escalation"]["judge"]
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "judge" in str(exc.value)

    def test_rejects_fallback_not_matching_a_tier(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        bundle["routes"]["myrouter/escalation"]["fallback_target_on_evict"] = "judge"
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "fallback_target_on_evict" in str(exc.value)


class TestDeterministicRouteType:
    """`type: deterministic` wires the LLM-classifier chain via YAML."""

    def _bundle(self) -> dict:
        return {
            "routes": {
                "myrouter/llm-classifier": {
                    "type": "deterministic",
                "fallback_target_on_evict": "strong",
                    "profile": "general",
                    "classifier": {
                        "model": "nvidia/nv-classifier",
                        "api_key": "sk-classifier",
                        "base_url": "https://classifier.invalid/v1",
                        "timeout_secs": 30.0,
                        "min_confidence": 0.6,
                        "fail_open": True,
                        "recent_turn_window": 4,
                    },
                    "strong": {
                        "model": "openai/gpt-5.2",
                        "api_key": "sk-strong",
                        "base_url": "https://strong.invalid/v1",
                    },
                    "weak": {
                        "model": "nvidia/nemotron-3-super",
                        "api_key": "sk-weak",
                        "base_url": "https://weak.invalid/v1",
                    },
                },
            },
        }

    def test_registers_route_key_and_tier_passthroughs(self):
        from switchyard.cli.route_bundle import build_route_bundle_table
        table = build_route_bundle_table(self._bundle())
        # Unified ordering: route key first as the deterministic virtual id,
        # tier models registered as direct passthroughs after. The classifier
        # tier is internal-only and never registered.
        assert table.registered_models() == [
            "myrouter/llm-classifier",
            "openai/gpt-5.2",
            "nvidia/nemotron-3-super",
        ]

    def test_anthropic_tier_is_cache_wrapped(self):
        """An Anthropic-format tier is wrapped for prompt caching via the YAML
        serve path, so an OpenAI-origin harness (Codex) routed onto a Claude
        tier still gets ``cache_control`` breakpoints injected. OpenAI tiers
        stay bare. Distinct from the DeterministicRoutingFactory path.
        """
        from switchyard.cli.route_bundle import build_route_bundle_table
        from switchyard.lib.backends.anthropic_cache_breakpoint_backend import (
            AnthropicCacheBreakpointBackend,
        )
        from switchyard.lib.backends.deterministic_routing_llm_backend import (
            DeterministicRoutingLLMBackend,
        )
        bundle = self._bundle()
        # Weak tier = Claude on Anthropic format (explicit -> no AUTO probe).
        bundle["routes"]["myrouter/llm-classifier"]["weak"] = {
            "model": "aws/anthropic/bedrock-claude-opus-4-7",
            "api_key": "sk-weak",
            "base_url": "https://weak.invalid/v1",
            "format": "anthropic",
        }
        table = build_route_bundle_table(bundle)
        switchyard = table.lookup_switchyard("myrouter/llm-classifier")
        backend = next(
            c for c in switchyard.iter_components()
            if isinstance(c, DeterministicRoutingLLMBackend)
        )
        assert isinstance(
            backend._backends["weak"], AnthropicCacheBreakpointBackend,
        )
        # Strong tier is OpenAI-format -> not wrapped.
        assert not isinstance(
            backend._backends["strong"], AnthropicCacheBreakpointBackend,
        )

    def test_rejects_missing_classifier_block(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        del bundle["routes"]["myrouter/llm-classifier"]["classifier"]
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "classifier" in str(exc.value)

    def test_rejects_unknown_profile(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        bundle["routes"]["myrouter/llm-classifier"]["profile"] = "no_such_profile"
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "profile" in str(exc.value).lower()

    def test_accepts_coding_agent_and_openclaw_profiles(self):
        from switchyard.cli.route_bundle import build_route_bundle_table
        for profile in ("coding_agent", "openclaw"):
            bundle = self._bundle()
            bundle["routes"]["myrouter/llm-classifier"]["profile"] = profile
            table = build_route_bundle_table(bundle)
            assert "myrouter/llm-classifier" in table.registered_models()

    def test_classifier_prompt_and_context_thread_into_processor(self):
        bundle = self._bundle()
        classifier = bundle["routes"]["myrouter/llm-classifier"]["classifier"]
        classifier["prompt"] = "custom yaml prompt"
        classifier["max_request_chars"] = 1024
        classifier["recent_turn_window"] = 2

        table = build_route_bundle_table(bundle)
        switchyard = table.lookup_switchyard("myrouter/llm-classifier")
        processor = next(
            c for c in switchyard.iter_components()
            if isinstance(c, LLMClassifierRequestProcessor)
        )

        assert processor._config.system_prompt == "custom yaml prompt"
        assert processor._config.max_request_chars == 1024
        assert processor._config.recent_turn_window == 2

    def test_blank_classifier_prompt_falls_back_to_profile_default(self):
        bundle = self._bundle()
        bundle["routes"]["myrouter/llm-classifier"]["profile"] = "coding_agent"
        bundle["routes"]["myrouter/llm-classifier"]["classifier"]["prompt"] = "  "

        table = build_route_bundle_table(bundle)
        switchyard = table.lookup_switchyard("myrouter/llm-classifier")
        processor = next(
            c for c in switchyard.iter_components()
            if isinstance(c, LLMClassifierRequestProcessor)
        )

        assert processor._config.system_prompt == CODING_AGENT_CLASSIFIER_SYSTEM_PROMPT

    def test_rejects_invalid_classifier_context_type(self):
        bundle = self._bundle()
        bundle["routes"]["myrouter/llm-classifier"]["classifier"][
            "max_request_chars"
        ] = "1024"

        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "integer" in str(exc.value)

    def test_tier_timeout_null_disables_default_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        captured_timeouts = []

        def fake_apply_default_tier_timeout(target, timeout_s):
            captured_timeouts.append((target.id, timeout_s))
            return target

        monkeypatch.setattr(
            "switchyard.lib.profiles.tier_target_builders.apply_default_tier_timeout",
            fake_apply_default_tier_timeout,
        )
        bundle = self._bundle()
        bundle["routes"]["myrouter/llm-classifier"]["tier_timeout_s"] = None

        build_route_bundle_table(bundle)

        assert captured_timeouts == [("strong", None), ("weak", None)]

    def test_weak_tier_on_own_endpoint_registers(self):
        """Weak tier can point at a self-hosted endpoint (e.g. local vLLM).

        Guards the GETTING_STARTED "serving the weak tier from your own
        endpoint" example: a per-tier ``base_url`` overrides ``defaults`` so
        weak-classified turns hit a model the user hosts, while strong and
        classifier stay on the shared endpoint. The weak model id registers as
        a direct passthrough.
        """
        from switchyard.cli.route_bundle import build_route_bundle_table
        table = build_route_bundle_table({
            "defaults": {
                "api_key": "sk-default",
                "base_url": "https://inference-api.nvidia.com/v1",
                "format": "openai",
            },
            "routes": {
                "local-weak": {
                    "type": "deterministic",
                    "profile": "coding_agent",
                    "fallback_target_on_evict": "strong",
                    "strong": {"model": "aws/anthropic/bedrock-claude-opus-4-7"},
                    "weak": {
                        "model": "my-rl-qwen",
                        "base_url": "http://localhost:8000/v1",
                        "api_key": "dummy",
                    },
                    "classifier": {
                        "model": "nvidia/deepseek-ai/deepseek-v4-flash",
                        "api_key": "sk-default",
                        "base_url": "https://inference-api.nvidia.com/v1",
                    },
                },
            },
        })
        assert table.registered_models() == [
            "local-weak",
            "aws/anthropic/bedrock-claude-opus-4-7",
            "my-rl-qwen",
        ]

    def test_classifier_inherits_route_defaults(self):
        from switchyard.cli.route_bundle import build_route_bundle_table
        from switchyard.lib.processors.llm_classifier import (
            LLMClassifierRequestProcessor,
        )

        table = build_route_bundle_table({
            "defaults": {
                "api_key": "sk-default",
                "base_url": "https://default.invalid/v1",
                "format": "openai",
                "timeout": 12.0,
            },
            "routes": {
                "defaults/classifier": {
                    "type": "deterministic",
                    "profile": "coding_agent",
                    "fallback_target_on_evict": "strong",
                    "strong": {"id": "strong", "model": "strong/model"},
                    "weak": {"id": "weak", "model": "weak/model"},
                    "classifier": {"model": "classifier/model"},
                },
            },
        })
        switchyard = table.lookup_switchyard("defaults/classifier")
        classifier = next(
            c for c in switchyard.iter_components()
            if isinstance(c, LLMClassifierRequestProcessor)
        )

        assert classifier._config.api_key == "sk-default"
        assert classifier._config.base_url == "https://default.invalid/v1"
        assert classifier._config.timeout_s == 12.0

    def test_rejects_unknown_classifier_key(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        bundle["routes"]["myrouter/llm-classifier"]["classifier"]["bogus"] = 1
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "bogus" in str(exc.value)

    def _classifier_disable_reasoning(self, classifier_model: str) -> bool:
        from switchyard.cli.route_bundle import build_route_bundle_table
        from switchyard.lib.processors.llm_classifier import (
            LLMClassifierRequestProcessor,
        )
        bundle = self._bundle()
        bundle["routes"]["myrouter/llm-classifier"]["classifier"]["model"] = classifier_model
        table = build_route_bundle_table(bundle)
        switchyard = table.lookup_switchyard("myrouter/llm-classifier")
        classifier = next(
            c for c in switchyard.iter_components()
            if isinstance(c, LLMClassifierRequestProcessor)
        )
        return classifier._config.disable_reasoning

    def test_bedrock_claude_classifier_disables_reasoning(self):
        assert self._classifier_disable_reasoning(
            "aws/anthropic/bedrock-claude-sonnet-4-6",
        ) is False

    def test_deepseek_classifier_keeps_reasoning_disabled(self):
        assert self._classifier_disable_reasoning(
            "nvidia/deepseek-ai/deepseek-v4-flash",
        ) is True


class TestStageRouterRouteType:
    """`type: stage_router` wires the stage-router-routing chain via YAML."""

    def _bundle(self) -> dict:
        return {
            "routes": {
                "myrouter/stage_router": {
                    "type": "stage_router",
                    "picker": "capable_first",
                    "strong": {
                        "id": "strong",
                        "model": "anthropic/claude-opus-4-7",
                        "api_key": "sk-strong",
                        "base_url": "https://strong.invalid/v1",
                        "format": "anthropic",
                    },
                    "weak": {
                        "id": "weak",
                        "model": "nvidia/nemotron-3-super",
                        "api_key": "sk-weak",
                        "base_url": "https://weak.invalid/v1",
                        "format": "openai",
                    },
                    "confidence_threshold": 0.7,
                    "fallback_target_on_evict": "strong",
                },
            },
        }

    def test_registers_route_key_and_tier_passthroughs(self):
        from switchyard.cli.route_bundle import build_route_bundle_table
        table = build_route_bundle_table(self._bundle())
        # Unified ordering: route key first as the stage-router virtual id,
        # tier models registered as direct passthroughs after.
        assert table.registered_models() == [
            "myrouter/stage_router",
            "anthropic/claude-opus-4-7",
            "nvidia/nemotron-3-super",
        ]

    def test_rejects_missing_strong(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        del bundle["routes"]["myrouter/stage_router"]["strong"]
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "strong" in str(exc.value)

    def test_rejects_missing_weak(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        del bundle["routes"]["myrouter/stage_router"]["weak"]
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "weak" in str(exc.value)

    def test_rejects_unknown_picker(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        bundle["routes"]["myrouter/stage_router"]["picker"] = "no-such-picker"
        with pytest.raises((RouteBundleConfigError, ValueError)) as exc:
            build_route_bundle_table(bundle)
        assert "picker" in str(exc.value).lower()

    def test_rejects_missing_fallback_target_on_evict(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        del bundle["routes"]["myrouter/stage_router"]["fallback_target_on_evict"]
        # Pydantic ValidationError extends ValueError; route_bundle does not
        # additionally wrap it for missing-field cases.
        with pytest.raises((RouteBundleConfigError, ValueError)) as exc:
            build_route_bundle_table(bundle)
        assert "fallback_target_on_evict" in str(exc.value)

    def test_rejects_fallback_target_id_not_matching(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        bundle["routes"]["myrouter/stage_router"]["fallback_target_on_evict"] = "nope"
        with pytest.raises((RouteBundleConfigError, ValueError)) as exc:
            build_route_bundle_table(bundle)
        assert "fallback_target_on_evict" in str(exc.value)

    def test_rejects_unknown_route_key(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        bundle["routes"]["myrouter/stage_router"]["bogus_field"] = 1
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "bogus_field" in str(exc.value)

    def test_accepts_both_pickers(self):
        from switchyard.cli.route_bundle import build_route_bundle_table
        for picker in ("capable_first", "efficient_first"):
            bundle = self._bundle()
            bundle["routes"]["myrouter/stage_router"]["picker"] = picker
            table = build_route_bundle_table(bundle)
            assert "myrouter/stage_router" in table.registered_models()

    def test_rejects_removed_legacy_knobs(self):
        from switchyard.cli.route_bundle import (
            RouteBundleConfigError,
            build_route_bundle_table,
        )
        bundle = self._bundle()
        bundle["routes"]["myrouter/stage_router"]["escalate_at"] = 0.5
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "escalate_at" in str(exc.value)

    def test_classifier_block_is_optional(self):
        from switchyard.cli.route_bundle import build_route_bundle_table
        bundle = self._bundle()
        bundle["routes"]["myrouter/stage_router"]["classifier"] = {
            "model": "nvidia/deepseek-ai/deepseek-v4-flash",
            "api_key": "sk-classifier",
            "base_url": "https://classifier.invalid/v1",
        }
        table = build_route_bundle_table(bundle)
        assert "myrouter/stage_router" in table.registered_models()

    def test_signal_recent_window_accepted(self):
        from switchyard.cli.route_bundle import build_route_bundle_table
        bundle = self._bundle()
        bundle["routes"]["myrouter/stage_router"]["signal_recent_window"] = 5
        table = build_route_bundle_table(bundle)
        assert "myrouter/stage_router" in table.registered_models()

    def test_classifier_inherits_route_defaults(self):
        from switchyard.cli.route_bundle import build_route_bundle_table
        from switchyard.lib.processors.stage_router import TierClassifier
        from switchyard.lib.processors.stage_router_request_processor import (
            StageRouterRequestProcessor,
        )

        table = build_route_bundle_table({
            "defaults": {
                "api_key": "sk-default",
                "base_url": "https://default.invalid/v1",
                "format": "openai",
                "timeout": 12.0,
            },
            "routes": {
                "defaults/stage_router": {
                    "type": "stage_router",
                    "picker": "capable_first",
                    "fallback_target_on_evict": "strong",
                    "strong": {"id": "strong", "model": "strong/model"},
                    "weak": {"id": "weak", "model": "weak/model"},
                    "classifier": {"model": "classifier/model"},
                },
            },
        })
        switchyard = table.lookup_switchyard("defaults/stage_router")
        processor = next(
            c for c in switchyard.iter_components()
            if isinstance(c, StageRouterRequestProcessor)
        )
        classifier = processor._picker.keywords["classifier"]

        assert isinstance(classifier, TierClassifier)
        assert classifier._api_key == "sk-default"
        assert classifier._recent_turn_window == 3

    async def test_classifier_usage_records_into_route_bundle_stats(self) -> None:
        """Route-bundle stage_router classifiers write usage into shared stats."""
        from switchyard.cli.route_bundle import build_route_bundle_table
        from switchyard.lib.processors.stage_router_request_processor import (
            StageRouterRequestProcessor,
        )
        from switchyard.lib.profiles.chain import ComponentChainProfile
        from switchyard.lib.stats_accumulator import StatsAccumulator
        from switchyard_rust.profiles import ProfileInput

        class _ClassifierClient:
            """Fake classifier client used after route-bundle construction."""

            calls = 0

            async def acompletion(self, **_kwargs: object) -> object:
                """Return a deterministic classifier response with usage data."""
                self.calls += 1
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=json.dumps({"tier": "capable"}),
                            ),
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=13,
                        completion_tokens=5,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                    ),
                )

        stats = StatsAccumulator()
        bundle = self._bundle()
        bundle["routes"]["myrouter/stage_router"]["confidence_threshold"] = 1.0
        bundle["routes"]["myrouter/stage_router"]["classifier"] = {
            "model": "classifier/model",
            "api_key": "sk-classifier",
            "base_url": "https://classifier.invalid/v1",
        }
        table = build_route_bundle_table(bundle, stats_accumulator=stats)
        switchyard = table.lookup_switchyard("myrouter/stage_router")
        assert isinstance(switchyard._profile, ComponentChainProfile)
        processor = next(
            component
            for component in switchyard.iter_components()
            if isinstance(component, StageRouterRequestProcessor)
        )
        assert processor._classifier is not None
        client = _ClassifierClient()
        processor._classifier._client = client  # type: ignore[assignment]

        processed = await switchyard._profile.process(ProfileInput(ChatRequest.openai_chat({
            "model": "myrouter/stage_router",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        })))

        snapshot = await stats.snapshot()
        assert client.calls == 1
        assert processed.selected_target == "strong"
        assert snapshot["classifier"]["total_requests"] == 1
        assert snapshot["classifier"]["models"]["classifier/model"]["calls"] == 1
        assert snapshot["classifier"]["models"]["classifier/model"]["prompt_tokens"] == 13
        assert snapshot["classifier"]["models"]["classifier/model"]["completion_tokens"] == 5
        assert snapshot["classifier"]["models"]["classifier/model"]["cached_tokens"] == 2
        assert snapshot["routing_decisions"]["stage_router"]["llm-classifier"] == 1


class TestPlanExecuteRouteType:
    """`type: plan_execute` wires the strong-planner / weak-executor chain via YAML."""

    def _bundle(self) -> dict:
        return {
            "routes": {
                "myrouter/plan-execute": {
                    "type": "plan_execute",
                    "cadence_n": 3,
                    "planner": {
                        "model": "azure/anthropic/claude-opus-4-6",
                        "api_key": "sk-planner",
                        "base_url": "https://planner.invalid/v1",
                    },
                    "executor": {
                        "model": "nvidia/nvidia/nemotron-3-super-v3",
                        "api_key": "sk-executor",
                        "base_url": "https://executor.invalid/v1",
                    },
                },
            },
        }

    def test_registers_under_route_key(self):
        table = build_route_bundle_table(self._bundle())
        assert table.registered_models() == ["myrouter/plan-execute"]

    def test_metadata_records_plan_execute_profile(self):
        table = build_route_bundle_table(self._bundle())
        _, _, metadata = next(iter(table.items()))
        assert metadata["switchyard"]["profile"] == "plan_execute"

    def test_plan_alias_registers_plan_execute_route(self):
        bundle = self._bundle()
        bundle["routes"]["myrouter/plan-execute"]["type"] = "plan"
        table = build_route_bundle_table(bundle)
        _, _, metadata = next(iter(table.items()))
        assert metadata["switchyard"]["profile"] == "plan_execute"

    def test_tiers_default_to_shipping_preset(self):
        # A minimal route (no planner/executor) reproduces the retired
        # --plan-execute flag: tiers fall back to the coding-agent preset.
        bundle = {
            "defaults": {"api_key": "sk-x", "base_url": "https://x.invalid/v1"},
            "routes": {"router/pe": {"type": "plan_execute"}},
        }
        table = build_route_bundle_table(bundle)
        assert table.registered_models() == ["router/pe"]

    def test_rejects_unknown_route_key(self):
        bundle = self._bundle()
        bundle["routes"]["myrouter/plan-execute"]["bogus_field"] = 1
        with pytest.raises(RouteBundleConfigError) as exc:
            build_route_bundle_table(bundle)
        assert "bogus_field" in str(exc.value)

    def test_rejects_invalid_cadence(self):
        bundle = self._bundle()
        bundle["routes"]["myrouter/plan-execute"]["cadence_n"] = 0
        with pytest.raises((RouteBundleConfigError, ValueError)):
            build_route_bundle_table(bundle)

    def test_string_tier_shorthand_accepted(self):
        bundle = self._bundle()
        bundle["routes"]["myrouter/plan-execute"]["executor"] = "nvidia/nvidia/nemotron-3-super-v3"
        table = build_route_bundle_table(bundle)
        assert "myrouter/plan-execute" in table.registered_models()

    def test_enable_stats_false_disables_stats_processors(self):
        from switchyard.lib.processors.stats_request_processor import (
            StatsRequestProcessor,
        )
        from switchyard.lib.processors.stats_response_processor_accumulator import (
            StatsResponseProcessor,
        )

        bundle = self._bundle()
        bundle["routes"]["myrouter/plan-execute"]["enable_stats"] = False
        table = build_route_bundle_table(bundle)
        components = list(table.iter_components())
        assert not any(isinstance(c, StatsRequestProcessor) for c in components)
        assert not any(isinstance(c, StatsResponseProcessor) for c in components)


def test_stage_router_route_hydrates_tier_catalogs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``type: stage_router`` registers each tier's catalog alongside the
    routing-policy chain — strong/weak as direct passthroughs, plus the
    route's YAML key as the stage-router virtual id.

    The classifier tier (when present) is intentionally NOT discovered — it's
    an internal-only LLM call, not a user-facing target.
    """
    monkeypatch.setattr(
        "switchyard.cli.route_bundle.fetch_model_ids",
        lambda base_url, api_key: (
            ["catalog/strong-extra"]
            if base_url == "https://primary.example/v1"
            else ["catalog/weak-extra"]
        ),
    )

    table = build_route_bundle_table({
        "routes": {
            "opus-ds-stage_router": {
                "type": "stage_router",
                "picker": "capable_first",
                "fallback_target_on_evict": "strong",
                "strong": {
                    "id": "strong",
                    "model": "strong/model",
                    "api_key": "k-strong",
                    "base_url": "https://primary.example/v1",
                },
                "weak": {
                    "id": "weak",
                    "model": "weak/model",
                    "api_key": "k-weak",
                    "base_url": "https://weak.example/v1",
                },
            },
        },
    })

    # Unified ordering: stage_router virtual id first, then tier models + catalog.
    # The classifier tier is absent here AND would be skipped even if present.
    assert table.registered_models() == [
        "opus-ds-stage_router",
        "strong/model",
        "weak/model",
        "catalog/strong-extra",
        "catalog/weak-extra",
    ]


def test_deterministic_route_hydrates_tier_catalogs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``type: deterministic`` — same catalog-hydration shape as stage_router,
    with the classifier tier intentionally skipped."""
    monkeypatch.setattr(
        "switchyard.cli.route_bundle.fetch_model_ids",
        lambda base_url, api_key: (
            ["catalog/strong-extra"]
            if base_url == "https://primary.example/v1"
            else ["catalog/weak-extra"]
        ),
    )

    table = build_route_bundle_table({
        "routes": {
            "opus-ds-classifier": {
                "type": "deterministic",
                "profile": "coding_agent",
                "fallback_target_on_evict": "strong",
                "classifier": {
                    "model": "classifier/model",
                    "api_key": "k-classifier",
                    "base_url": "https://classifier.example/v1",
                },
                "strong": {
                    "id": "strong",
                    "model": "strong/model",
                    "api_key": "k-strong",
                    "base_url": "https://primary.example/v1",
                },
                "weak": {
                    "id": "weak",
                    "model": "weak/model",
                    "api_key": "k-weak",
                    "base_url": "https://weak.example/v1",
                },
            },
        },
    })

    # Unified ordering: virtual id first, then tier models + catalog.
    # Classifier's `classifier/model` and its catalog are not in the list:
    # discovery skips internal-only tiers.
    assert table.registered_models() == [
        "opus-ds-classifier",
        "strong/model",
        "weak/model",
        "catalog/strong-extra",
        "catalog/weak-extra",
    ]


def test_route_bundle_keeps_first_multi_target_route_as_default() -> None:
    """StageRouter/deterministic discovery merges preserve the first YAML route."""
    table = build_route_bundle_table({
        "routes": {
            "first-stage_router": {
                "type": "stage_router",
                "picker": "capable_first",
                "fallback_target_on_evict": "strong",
                "strong": {
                    "id": "strong",
                    "model": "first/strong",
                    "api_key": "k-strong",
                    "base_url": "https://first-strong.example/v1",
                },
                "weak": {
                    "id": "weak",
                    "model": "first/weak",
                    "api_key": "k-weak",
                    "base_url": "https://first-weak.example/v1",
                },
            },
            "second-stage_router": {
                "type": "stage_router",
                "picker": "capable_first",
                "fallback_target_on_evict": "strong",
                "strong": {
                    "id": "strong",
                    "model": "second/strong",
                    "api_key": "k-strong",
                    "base_url": "https://second-strong.example/v1",
                },
                "weak": {
                    "id": "weak",
                    "model": "second/weak",
                    "api_key": "k-weak",
                    "base_url": "https://second-weak.example/v1",
                },
            },
        },
    })

    assert table.default_model() == "first-stage_router"


_AFFINITY_DET_ROUTE = {
    "type": "deterministic",
    "profile": "coding_agent",
    "fallback_target_on_evict": "strong",
    "session_affinity": True,
    "affinity_max_sessions": 50,
    "classifier": {"model": "m", "api_key": "k", "base_url": "https://ls.test/v1"},
    "strong": {"model": "s", "api_key": "k", "base_url": "https://ls.test/v1"},
    "weak": {"model": "w", "api_key": "k", "base_url": "https://ls.test/v1"},
}
_AFFINITY_LAT_ROUTE = {
    "type": "latency_service",
    "latency_service_url": "http://ls.test:8080",
    "session_affinity": True,
    "affinity_max_sessions": 50,
    "endpoints": [
        {"model": "s", "api_key": "k", "base_url": "https://ls.test/v1"},
        {"model": "w", "api_key": "k", "base_url": "https://ls.test/v1"},
    ],
}


@pytest.mark.parametrize("route", [_AFFINITY_DET_ROUTE, _AFFINITY_LAT_ROUTE])
def test_session_affinity_keys_accepted_by_route_bundle(route: dict[str, Any]) -> None:
    """session_affinity / affinity_max_sessions are valid keys on both route types."""
    table = build_route_bundle_table({"routes": {"r": route}})
    assert isinstance(table, RouteTable)


def test_latency_service_credential_policy_defaults_to_endpoint_keys() -> None:
    """Omitting credential_policy keeps the server-configured endpoint keys authoritative."""
    table = build_route_bundle_table({"routes": {"r": _AFFINITY_LAT_ROUTE}})

    backend = _latency_backend(table.lookup_switchyard("r"))

    assert backend._config.credential_policy == "configured_endpoint"


def test_latency_service_credential_policy_reaches_backend_config() -> None:
    """YAML credential_policy opts into BYO-key caller overrides."""
    table = build_route_bundle_table({
        "routes": {
            "r": {
                **_AFFINITY_LAT_ROUTE,
                "credential_policy": "caller_override",
            },
        },
    })

    backend = _latency_backend(table.lookup_switchyard("r"))

    assert backend._config.credential_policy == "caller_override"


def test_latency_service_invalid_credential_policy_rejected_via_bundle() -> None:
    """Invalid latency-service credential policy values fail closed."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        build_route_bundle_table({
            "routes": {
                "r": {
                    **_AFFINITY_LAT_ROUTE,
                    "credential_policy": "caller-overrides",
                },
            },
        })


def test_latency_endpoint_request_type_reaches_backend_config() -> None:
    """Endpoint-level request_type in YAML selects the upstream API surface."""
    table = build_route_bundle_table({
        "routes": {
            "r": {
                "type": "latency_service",
                "latency_service_url": "http://ls.test:8080",
                "endpoints": [
                    {
                        "model": "codex-mini",
                        "api_key": "k",
                        "base_url": "https://ls.test/v1",
                        "request_type": "openai_responses",
                    },
                    {"model": "w", "api_key": "k", "base_url": "https://ls.test/v1"},
                ],
            },
        },
    })

    backend = _latency_backend(table.lookup_switchyard("r"))

    by_model = {endpoint.model: endpoint for endpoint in backend._config.endpoints}
    assert by_model["codex-mini"].request_type == "openai_responses"
    assert by_model["w"].request_type == "openai_chat"


def test_latency_route_key_reaches_backend_as_route_model() -> None:
    """The YAML route key becomes the backend's metrics route_model id."""
    table = build_route_bundle_table({
        "routes": {
            "nvidia/switchyard/gpt-5.4": {
                "type": "latency_service",
                "latency_service_url": "http://ls.test:8080",
                "endpoints": [
                    {
                        "model": "azure/openai/gpt-5.4",
                        "api_key": "k",
                        "base_url": "https://ls.test/v1",
                    },
                ],
            },
        },
    })

    backend = _latency_backend(table.lookup_switchyard("nvidia/switchyard/gpt-5.4"))

    assert backend._config.route_model == "nvidia/switchyard/gpt-5.4"


def test_deterministic_affinity_warmup_turns_accepted_by_route_bundle() -> None:
    """affinity_warmup_turns is a deterministic-route knob."""
    table = build_route_bundle_table({
        "routes": {"r": {**_AFFINITY_DET_ROUTE, "affinity_warmup_turns": 2}},
    })
    assert isinstance(table, RouteTable)


@pytest.mark.parametrize("route", [_AFFINITY_DET_ROUTE, _AFFINITY_LAT_ROUTE])
def test_zero_capacity_affinity_rejected_via_bundle(route: dict[str, Any]) -> None:
    """A zero cap with affinity on is rejected — proving the keys reach the config."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        build_route_bundle_table({"routes": {"r": {**route, "affinity_max_sessions": 0}}})


def test_latency_affinity_redis_reaches_backend_via_bundle() -> None:
    """Redis L2 keys parse and construct a RedisPinStore on the latency backend."""
    from switchyard.lib.redis_pin_store import RedisPinStore

    table = build_route_bundle_table({
        "routes": {
            "r": {
                **_AFFINITY_LAT_ROUTE,
                "affinity_store": "redis",
                "affinity_store_url": "redis://cache:6379/0",
                "affinity_store_ttl_seconds": 120,
                "affinity_key_prefix": "k:",
            },
        },
    })
    backend = _latency_backend(table.lookup_switchyard("r"))
    assert backend._config.affinity_store == "redis"
    assert backend._config.affinity_store_url == "redis://cache:6379/0"
    assert isinstance(backend._affinity._l2, RedisPinStore)


def test_latency_affinity_redis_requires_url_via_bundle() -> None:
    """affinity_store=redis without a URL fails closed at config load."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        build_route_bundle_table({
            "routes": {"r": {**_AFFINITY_LAT_ROUTE, "affinity_store": "redis"}},
        })


def test_negative_affinity_warmup_turns_rejected_via_bundle() -> None:
    """The deterministic config validates affinity_warmup_turns as non-negative."""
    from pydantic import ValidationError

    route = {**_AFFINITY_DET_ROUTE, "affinity_warmup_turns": -1}
    with pytest.raises(ValidationError):
        build_route_bundle_table({"routes": {"r": route}})
