# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic default of ``switchyard launch codex``.

``launch codex`` defaults to LLM-classifier deterministic routing when
no ``--model`` is given. The legacy
``--deterministic`` flag has been removed from this subparser; tier
overrides (``--weak-model``, ``--classifier-model``, ``--profile``,
``--classifier-min-confidence``) still tune the default trio.
"""

from __future__ import annotations

import pytest


class TestArgparse:
    def test_deterministic_flag_removed(self) -> None:
        from switchyard.cli.switchyard_cli import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["launch", "codex", "--deterministic"])

    def test_default_no_flags_parses(self) -> None:
        from switchyard.cli.switchyard_cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["launch", "codex"])
        assert args.model is None
        assert args.routing_profiles is None
        assert args.weak_model is None
        assert args.classifier_model is None
        assert args.profile is None
        assert args.classifier_min_confidence is None

    def test_overrides_parse(self) -> None:
        from switchyard.cli.switchyard_cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "codex",
            "--classifier-model", "nvidia/nvidia/nemotron-3-super-v3",
            "--profile", "general",
            "--classifier-min-confidence", "0.55",
        ])
        assert args.classifier_model == "nvidia/nvidia/nemotron-3-super-v3"
        assert args.profile == "general"
        assert args.classifier_min_confidence == 0.55


class TestDispatch:
    def test_default_dispatches_to_deterministic_launcher(
        self, monkeypatch, tmp_path,
    ) -> None:
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_codex

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "codex", "--api-key", "sk-test",
        ])

        captured: dict = {}

        def fake_launch(**kwargs):
            captured.update(kwargs)
            raise SystemExit(0)

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://openrouter.ai/api/v1"),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher."
            "launch_codex_deterministic_routing",
            fake_launch,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_codex(args)

        config = captured["config"]
        assert config.strong.model == "anthropic/claude-opus-4.7"
        assert config.weak.model == "moonshotai/kimi-k2.6"
        assert config.classifier.model == "google/gemini-3.5-flash"
        assert config.profile_name == "coding_agent"
        assert config.preset == "coding_agent_default"

    def test_model_flag_opts_out_of_deterministic(
        self, monkeypatch, tmp_path,
    ) -> None:
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_codex

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "codex",
            "--api-key", "sk-test",
            "--model", "nvidia/moonshotai/kimi-k2.5",
        ])

        captured_passthrough: dict = {}

        def fake_passthrough(**kwargs):
            captured_passthrough.update(kwargs)
            raise SystemExit(0)

        def fake_deterministic(**_kwargs):
            raise AssertionError(
                "--model X should not dispatch to deterministic launcher",
            )

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://openrouter.ai/api/v1"),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher.launch_codex",
            fake_passthrough,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher."
            "launch_codex_deterministic_routing",
            fake_deterministic,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_codex(args)

        assert captured_passthrough["model"] == "nvidia/moonshotai/kimi-k2.5"

    def test_dispatch_honors_weak_model_override(
        self, monkeypatch, tmp_path,
    ) -> None:
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_codex

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "codex", "--api-key", "sk-test",
            "--weak-model", "nvidia/moonshotai/kimi-k2.5",
        ])

        captured: dict = {}

        def fake_launch(**kwargs):
            captured.update(kwargs)
            raise SystemExit(0)

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://openrouter.ai/api/v1"),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher."
            "launch_codex_deterministic_routing",
            fake_launch,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_codex(args)

        config = captured["config"]
        assert config.weak.model == "nvidia/moonshotai/kimi-k2.5"
        assert config.preset is None

    def test_dry_run_does_not_invoke_launcher(self, monkeypatch, tmp_path) -> None:
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_codex

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "codex", "--api-key", "sk-test",
            "--dry-run",
        ])

        def fake_launch(**_kwargs):
            raise AssertionError("dry-run must not invoke the launcher")

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://openrouter.ai/api/v1"),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher."
            "launch_codex_deterministic_routing",
            fake_launch,
        )

        _cmd_launch_codex(args)


class TestRoutesByDefault:
    """The deterministic launch must boot codex on the *router*, not strong.

    Regression guard for the bug where ``launch_codex_deterministic_routing``
    pinned codex's ``-m`` to ``config.strong.model`` — a direct passthrough that
    silently bypassed the LLM classifier and sent every request straight to the
    strong model. It must use the virtual routing model id, matching claude.
    """

    def test_codex_boots_on_routing_virtual_model(self, monkeypatch) -> None:
        from switchyard.cli.launchers.codex_cli_launcher import (
            launch_codex_deterministic_routing,
        )
        from switchyard.lib.profiles import (
            DeterministicRoutingPresets,
        )
        from switchyard.lib.route_table_builders import (
            deterministic_routing_virtual_model_id,
        )

        config = DeterministicRoutingPresets.coding_agent_default(api_key="sk-test")
        captured: dict = {}

        def fake_run(_table, **kwargs):
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._run_codex_with_switchyard",
            fake_run,
        )

        rc = launch_codex_deterministic_routing(
            config=config,
            port=None,
            codex_args=[],
            discovery_disabled=True,  # no GET /models network call
        )

        assert rc == 0
        # codex is launched against the router, so the classifier runs by default.
        assert captured["display_model"] == deterministic_routing_virtual_model_id(
            config,
        )
        assert captured["display_model"] != config.strong.model
