# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic default of ``switchyard launch claude``.

``launch claude`` defaults to LLM-classifier deterministic routing when
no ``--model`` or ``--routing-profiles`` is given. The legacy
``--deterministic`` flag has been removed from this subparser; tier
overrides (``--weak-model``, ``--classifier-model``, ``--profile``,
``--classifier-min-confidence``) still tune the default trio.
"""

import pytest


class TestArgparse:
    def test_deterministic_flag_removed(self) -> None:
        from switchyard.cli.switchyard_cli import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["launch", "claude", "--deterministic"])

    def test_default_no_flags_parses(self) -> None:
        from switchyard.cli.switchyard_cli import _build_parser

        parser = _build_parser()
        # No --deterministic, no --model: the launcher will dispatch to
        # deterministic routing as the implicit default.
        args = parser.parse_args(["launch", "claude"])
        assert args.model is None
        assert args.routing_profiles is None
        # Override knobs default to None — preset values fill them in.
        assert args.weak_model is None
        assert args.classifier_model is None
        assert args.profile is None
        assert args.classifier_min_confidence is None

    def test_overrides_parse(self) -> None:
        from switchyard.cli.switchyard_cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "claude",
            "--classifier-model", "nvidia/nvidia/nemotron-3-super-v3",
            "--profile", "general",
            "--classifier-min-confidence", "0.55",
        ])
        assert args.classifier_model == "nvidia/nvidia/nemotron-3-super-v3"
        assert args.profile == "general"
        assert args.classifier_min_confidence == 0.55

    def test_profile_choices_enforced(self) -> None:
        from switchyard.cli.switchyard_cli import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "launch", "claude",
                "--profile", "invented",
            ])


class TestDispatch:
    def test_default_dispatches_to_deterministic_launcher(
        self, monkeypatch, tmp_path,
    ) -> None:
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_claude

        parser = _build_parser()
        # Zero flags beyond credentials — deterministic should fire.
        args = parser.parse_args([
            "launch", "claude", "--api-key", "sk-test",
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
            "switchyard.cli.launchers.claude_code_launcher."
            "launch_claude_deterministic_routing",
            fake_launch,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_claude(args)

        config = captured["config"]
        assert config.strong.model == "anthropic/claude-opus-4.7"
        assert config.weak.model == "moonshotai/kimi-k2.6"
        assert config.classifier.model == "google/gemini-3.5-flash"
        assert config.profile_name == "coding_agent"
        assert config.preset == "coding_agent_default"

    def test_launch_uses_auto_format_for_strong_tier(
        self, monkeypatch, tmp_path,
    ) -> None:
        """Claude can probe Anthropic support on compatible gateways while
        OpenRouter defaults fall back to OpenAI-compatible chat completions."""
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_claude
        from switchyard.lib.backends.llm_target import BackendFormat

        parser = _build_parser()
        args = parser.parse_args(["launch", "claude", "--api-key", "sk-test"])

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
            "switchyard.cli.launchers.claude_code_launcher."
            "launch_claude_deterministic_routing",
            fake_launch,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_claude(args)

        config = captured["config"]
        assert config.strong.format == BackendFormat.AUTO
        assert config.weak.format == BackendFormat.OPENAI
        assert config.classifier.format == BackendFormat.OPENAI

    def test_model_flag_opts_out_of_deterministic(
        self, monkeypatch, tmp_path,
    ) -> None:
        """Passing --model X falls through to single-model passthrough."""
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_claude

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "claude",
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
            "switchyard.cli.launchers.claude_code_launcher.launch_claude",
            fake_passthrough,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher."
            "launch_claude_deterministic_routing",
            fake_deterministic,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_claude(args)

        assert captured_passthrough["model"] == "nvidia/moonshotai/kimi-k2.5"

    def test_dispatch_honors_user_overrides(self, monkeypatch, tmp_path) -> None:
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_claude

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "claude", "--api-key", "sk-test",
            "--weak-model", "nvidia/moonshotai/kimi-k2.5",
            "--profile", "general",
            "--classifier-min-confidence", "0.55",
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
            "switchyard.cli.launchers.claude_code_launcher."
            "launch_claude_deterministic_routing",
            fake_launch,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_claude(args)

        config = captured["config"]
        assert config.weak.model == "nvidia/moonshotai/kimi-k2.5"
        assert config.profile_name == "general"
        assert config.classifier_min_confidence == 0.55
        # Strong + classifier still come from the preset
        assert config.strong.model == "anthropic/claude-opus-4.7"
        assert config.classifier.model == "google/gemini-3.5-flash"
        # Preset blanked because user overrode at least one model
        assert config.preset is None

    def test_dry_run_does_not_invoke_launcher(self, monkeypatch, tmp_path) -> None:
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_claude

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "claude", "--api-key", "sk-test",
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
            "switchyard.cli.launchers.claude_code_launcher."
            "launch_claude_deterministic_routing",
            fake_launch,
        )

        # Dry-run prints + returns without SystemExit.
        _cmd_launch_claude(args)

    def test_dry_run_routing_profiles_reports_bundle(
        self, monkeypatch, tmp_path, capsys,
    ) -> None:
        """A ``--routing-profiles`` dry run reports ``route: bundle``, not single."""
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_claude

        yaml_path = tmp_path / "routes.yaml"
        yaml_path.write_text(
            "routes:\n"
            "  my-route:\n"
            "    type: model\n"
            "    model: openai/gpt-4o-mini\n",
            encoding="utf-8",
        )

        parser = _build_parser()
        args = parser.parse_args([
            "--routing-profiles", str(yaml_path),
            "--", "launch", "claude", "--dry-run",
        ])

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://openrouter.ai/api/v1"),
        )

        _cmd_launch_claude(args)

        out = capsys.readouterr().out
        assert "route: bundle" in out
        assert "route: single" not in out

    def test_zero_config_non_openrouter_rejects_default_trio(
        self, monkeypatch, tmp_path,
    ) -> None:
        """Zero-flag launch against a non-OpenRouter provider fails fast."""
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_claude

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("nvapi-test", "https://inference-api.nvidia.com/v1"),
        )

        parser = _build_parser()
        args = parser.parse_args(["launch", "claude"])

        with pytest.raises(SystemExit) as exc_info:
            _cmd_launch_claude(args)

        message = str(exc_info.value)
        assert "OpenRouter-only" in message
        assert "anthropic/claude-opus-4.7" in message
        assert "moonshotai/kimi-k2.6" in message
        assert "google/gemini-3.5-flash" in message
        assert "'nvidia'" in message

    def test_model_flag_bypasses_non_openrouter_guard(
        self, monkeypatch, tmp_path,
    ) -> None:
        """Passing --model opts out of the default trio, bypassing the guard."""
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_claude

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("nvapi-test", "https://inference-api.nvidia.com/v1"),
        )

        captured: dict = {}

        def fake_passthrough(**kwargs):
            captured.update(kwargs)
            raise SystemExit(0)

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher.launch_claude",
            fake_passthrough,
        )

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "claude", "--model", "nvidia/moonshotai/kimi-k2.5",
        ])

        with pytest.raises(SystemExit) as exc_info:
            _cmd_launch_claude(args)

        assert "OpenRouter-only" not in str(exc_info.value)
        assert captured["model"] == "nvidia/moonshotai/kimi-k2.5"


class TestRoutesByDefault:
    """The deterministic launch must boot claude on the *router*, not strong.

    Mirror of the codex guard — both launchers pin the agent to the virtual
    routing model id so the LLM classifier runs by default.
    """

    def test_claude_boots_on_routing_virtual_model(self, monkeypatch) -> None:
        from switchyard.cli.launchers.claude_code_launcher import (
            launch_claude_deterministic_routing,
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
            "switchyard.cli.launchers.claude_code_launcher._run_claude_with_switchyard",
            fake_run,
        )

        rc = launch_claude_deterministic_routing(
            config=config,
            port=None,
            claude_args=[],
            discovery_disabled=True,
        )

        assert rc == 0
        assert captured["display_model"] == deterministic_routing_virtual_model_id(
            config,
        )
        assert captured["display_model"] != config.strong.model
