# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LLM-classifier default of ``switchyard launch openclaw``.

``launch openclaw`` defaults to LLM-classifier routing when no ``--model``
or ``--routing-profiles`` is given — same shape as ``launch claude`` /
``launch codex``. The legacy ``--deterministic`` opt-in flag has been
removed; tier overrides (``--weak-model``, ``--classifier-model``,
``--profile``, ``--classifier-min-confidence``) still tune the default trio.
"""

from __future__ import annotations

import pytest


class TestArgparse:
    def test_deterministic_flag_removed(self) -> None:
        from switchyard.cli.switchyard_cli import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["launch", "openclaw", "--deterministic"])

    def test_default_no_flags_parses(self) -> None:
        from switchyard.cli.switchyard_cli import _build_parser

        parser = _build_parser()
        # No --model: the launcher dispatches to LLM-classifier routing as
        # the implicit default.
        args = parser.parse_args(["launch", "openclaw"])
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
            "launch", "openclaw",
            "--classifier-model", "nvidia/nvidia/nemotron-3-super-v3",
            "--profile", "openclaw",
            "--classifier-min-confidence", "0.55",
        ])
        assert args.classifier_model == "nvidia/nvidia/nemotron-3-super-v3"
        assert args.profile == "openclaw"
        assert args.classifier_min_confidence == 0.55


class TestMutualExclusion:
    def test_routing_profiles_opts_out_of_classifier(
        self, monkeypatch, tmp_path,
    ) -> None:
        """``--routing-profiles`` opts out of the LLM-classifier default."""
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_openclaw

        yaml_path = tmp_path / "routes.yaml"
        yaml_path.write_text(
            "defaults:\n"
            "  api_key: sk-test\n"
            "  base_url: https://upstream.invalid/v1\n"
            "  format: openai\n"
            "routes:\n"
            "  single-model:\n"
            "    type: model\n"
            "    target: upstream/model\n"
        )
        parser = _build_parser()
        args = parser.parse_args([
            "--routing-profiles", str(yaml_path),
            "launch", "openclaw",
            "--dry-run",
        ])

        def fake_deterministic(**_kwargs):
            raise AssertionError(
                "--routing-profiles should opt out of LLM-classifier routing",
            )

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://openrouter.ai/api/v1"),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher."
            "launch_openclaw_deterministic_routing",
            fake_deterministic,
        )
        # Dry-run prints + returns without invoking the classifier launcher.
        _cmd_launch_openclaw(args)


class TestDispatch:
    def test_default_dispatches_to_classifier_launcher(
        self, monkeypatch, tmp_path,
    ) -> None:
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_openclaw

        parser = _build_parser()
        # Zero flags beyond credentials — LLM-classifier routing should fire.
        args = parser.parse_args([
            "launch", "openclaw", "--api-key", "sk-test",
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
            "switchyard.cli.launchers.openclaw_launcher."
            "launch_openclaw_deterministic_routing",
            fake_launch,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_openclaw(args)

        config = captured["config"]
        # The launcher inherits the coding_agent_default trio by default
        # (same defaults as claude/codex implicit LLM-classifier mode).
        assert config.strong.model == "anthropic/claude-opus-4.7"
        assert config.weak.model == "moonshotai/kimi-k2.6"
        assert config.classifier.model == "google/gemini-3.5-flash"
        assert config.profile_name == "coding_agent"
        assert config.preset == "coding_agent_default"

    def test_model_flag_opts_out_of_classifier(
        self, monkeypatch, tmp_path,
    ) -> None:
        """Passing --model X falls through to single-model route."""
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_openclaw

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "openclaw",
            "--api-key", "sk-test",
            "--model", "nvidia/moonshotai/kimi-k2.5",
        ])

        captured_passthrough: dict = {}

        def fake_passthrough(**kwargs):
            captured_passthrough.update(kwargs)
            raise SystemExit(0)

        def fake_deterministic(**_kwargs):
            raise AssertionError(
                "--model X should not dispatch to the classifier launcher",
            )

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://openrouter.ai/api/v1"),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.launch_openclaw",
            fake_passthrough,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher."
            "launch_openclaw_deterministic_routing",
            fake_deterministic,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_openclaw(args)

        assert captured_passthrough["model"] == "nvidia/moonshotai/kimi-k2.5"

    def test_dispatch_honors_profile_override(
        self, monkeypatch, tmp_path,
    ) -> None:
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_openclaw

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "openclaw", "--api-key", "sk-test",
            "--profile", "openclaw",
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
            "switchyard.cli.launchers.openclaw_launcher."
            "launch_openclaw_deterministic_routing",
            fake_launch,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_openclaw(args)

        config = captured["config"]
        assert config.profile_name == "openclaw"

    def test_dry_run_does_not_invoke_launcher(self, monkeypatch, tmp_path) -> None:
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_openclaw

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "openclaw", "--api-key", "sk-test",
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
            "switchyard.cli.launchers.openclaw_launcher."
            "launch_openclaw_deterministic_routing",
            fake_launch,
        )

        _cmd_launch_openclaw(args)
