# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``switchyard.cli.launchers.claude_code_launcher``.

Exercises the public ``launch_claude`` entry, the private model-rewrite
request processor, and the ``claude`` binary lookup. Real uvicorn and
``subprocess.run`` are mocked — these tests don't start a server or
spawn a child process.
"""

import argparse
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from switchyard.cli.launchers.claude_code_launcher import (
    _EXIT_BINARY_NOT_FOUND,
    _EXIT_SIGINT,
    ProxyHealthMonitor,
    _find_claude_binary,
    _find_free_port,
    _make_footer_fn,
    _ModelRewriteRequestProcessor,
    _print_ready_banner,
    launch_claude,
)
from switchyard.cli.launchers.launch_intake_config import LaunchIntakeConfig
from switchyard.lib.backends.llm_target import LlmTarget
from switchyard.lib.processors.model_rewrite_request_processor import (
    ModelRewriteRequestProcessor,
)
from switchyard.lib.profiles.random_routing import RandomRoutingConfig
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.route_table import RouteTable
from switchyard.lib.route_table_builders import (
    random_routing_virtual_model_id,
)
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard_rust.core import ChatRequest


def test_random_routing_virtual_model_id_is_client_neutral() -> None:
    config = RandomRoutingConfig(
        strong=LlmTarget(model="azure/anthropic/claude-opus-4-7"),
        weak=LlmTarget(model="nvidia/nvidia/nemotron-3-super-120b-long-ctx"),
        strong_probability=0.25,
        fallback_target_on_evict="strong",
    )

    model = random_routing_virtual_model_id(config)

    assert model.startswith("switchyard-default-random-")
    assert "/" not in model
    assert "claude-opus-4-7" not in model
    assert "nemotron-3-super-120b-long-ctx" not in model


def test_bootstrap_persists_selected_env_provider_base_url(monkeypatch, tmp_path) -> None:
    from switchyard.cli.config.user_config import (
        load_user_config,
        load_user_credentials,
    )
    from switchyard.cli.launch_command import maybe_bootstrap_launch_config

    monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
    for env_var in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://nvidia.test/v1")
    monkeypatch.setattr("switchyard.cli.launch_command.is_interactive_terminal", lambda: True)
    monkeypatch.setattr("switchyard.cli.launch_command.load_secrets", lambda: {})
    monkeypatch.setattr(
        "switchyard.cli.configure_command.is_interactive_terminal", lambda: False
    )

    args = argparse.Namespace(
        reconfigure=True,
        api_key=None,
        base_url=None,
        model="nvidia/model",
        routing_profiles=None,
        no_model_discovery=True,
        no_tui=True,
    )

    maybe_bootstrap_launch_config(
        args,
        target="claude",
        api_key_env_vars=(
            "OPENROUTER_API_KEY",
            "NVIDIA_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ),
    )

    config = load_user_config()
    credentials = load_user_credentials()
    assert config.default_provider == "nvidia"
    assert config.provider("nvidia").base_url == "https://nvidia.test/v1"
    assert config.launch_target("claude").effective_route().model == "nvidia/model"
    assert credentials.api_key("nvidia") == "nvidia-key"  # pragma: allowlist secret
    assert args.api_key == "nvidia-key"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# _ModelRewriteRequestProcessor
# ---------------------------------------------------------------------------


class TestModelRewriteRequestProcessor:
    """Processor must rewrite ``body['model']`` for every ChatRequest subclass."""

    async def test_rewrites_anthropic_request(self):
        proc = _ModelRewriteRequestProcessor("nvidia/moonshotai/kimi-k2.5")
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        out = await proc.process(ProxyContext(), req)
        assert out is req
        assert req.body["model"] == "nvidia/moonshotai/kimi-k2.5"

    async def test_rewrites_openai_chat_request(self):
        proc = _ModelRewriteRequestProcessor("target-model")
        req = ChatRequest.openai_chat({"model": "gpt-4o", "messages": []})
        await proc.process(ProxyContext(), req)
        assert req.body["model"] == "target-model"

    async def test_rewrites_responses_request(self):
        proc = _ModelRewriteRequestProcessor("target-model")
        req = ChatRequest.openai_responses({"model": "gpt-4o", "input": "hi"})
        await proc.process(ProxyContext(), req)
        assert req.body["model"] == "target-model"


# ---------------------------------------------------------------------------
# _find_claude_binary
# ---------------------------------------------------------------------------


class TestFindClaudeBinary:
    def test_returns_path_hit_when_on_path(self):
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            assert _find_claude_binary() == "/usr/local/bin/claude"

    def test_falls_back_to_claude_local(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        fake_home = tmp_path
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        claude = fake_home / ".claude" / "local" / "claude"
        claude.parent.mkdir(parents=True)
        claude.write_text("#!/bin/sh\necho claude\n")
        claude.chmod(0o755)
        assert _find_claude_binary() == str(claude)

    def test_falls_back_to_local_bin(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        fake_home = tmp_path
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        claude = fake_home / ".local" / "bin" / "claude"
        claude.parent.mkdir(parents=True)
        claude.write_text("#!/bin/sh\necho claude\n")
        claude.chmod(0o755)
        assert _find_claude_binary() == str(claude)

    def test_returns_none_when_nowhere(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _find_claude_binary() is None


# ---------------------------------------------------------------------------
# _find_free_port
# ---------------------------------------------------------------------------


class TestFindFreePort:
    def test_returns_usable_port(self):
        port = _find_free_port()
        assert 1024 <= port <= 65535


# ---------------------------------------------------------------------------
# launch_claude — integration (with mocked externals)
# ---------------------------------------------------------------------------


def _make_fake_server(started: bool = True) -> MagicMock:
    """uvicorn.Server stand-in that reports started immediately."""
    server = MagicMock()
    server.started = started
    server.should_exit = False
    return server


def _stub_spawn_proxy(server: MagicMock):
    """Return a function that mimics _spawn_proxy_thread's (server, thread) tuple."""
    def _inner(switchyard, port):
        thread = MagicMock()
        return server, thread
    return _inner


@pytest.fixture(autouse=True)
def _mock_probe(monkeypatch, tmp_path):
    """Mock external launcher side effects in all tests."""
    monkeypatch.setattr(
        "switchyard.lib.backends.backend_format_resolver.probe_openai_chat_completions_support_sync",
        lambda **_: False,
    )
    monkeypatch.setattr(
        "switchyard.lib.backends.backend_format_resolver.probe_anthropic_messages_support_sync",
        lambda **_: False,
    )
    monkeypatch.setattr(
        "switchyard.lib.backends.backend_format_resolver.probe_openai_responses_support_sync",
        lambda **_: False,
    )
    monkeypatch.setattr(
        "switchyard.cli.launchers.claude_code_launcher._wait_ready",
        lambda port, timeout_s=10.0: True,
    )
    monkeypatch.setattr(
        "switchyard.cli.launchers.claude_code_launcher.configure_debug_file_logging",
        lambda display_model: tmp_path / "switchyard.log",
    )
    # pytest redirects stdin to a pseudofile with no real fd;
    # isatty would raise UnsupportedOperation — force non-TTY mode.
    monkeypatch.setattr("os.isatty", lambda fd: False)


class TestLaunchClaude:
    def test_happy_path(self, monkeypatch):
        fake_server = _make_fake_server(started=True)

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_claude_binary",
            lambda: "/fake/bin/claude",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(fake_server),
        )

        captured: dict = {}

        def fake_run(cmd, env, check):
            captured["cmd"] = cmd
            captured["env"] = env
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        exit_code = launch_claude(
            model="nvidia/moonshotai/kimi-k2.5",
            base_url="https://inference-api.nvidia.com/v1",
            api_key="test-key",
            port=None,
            timeout=None,
            claude_args=["--version"],
        )

        assert exit_code == 0
        assert captured["cmd"] == ["/fake/bin/claude", "--version"]
        assert captured["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:54321"
        # ANTHROPIC_AUTH_TOKEN tells Claude Code auth is external, skipping
        # the Console / 3rd-party provider setup wizard. ANTHROPIC_API_KEY
        # is emptied to suppress the "Auth conflict" warning.
        assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "switchyard"
        assert captured["env"]["ANTHROPIC_API_KEY"] == ""
        # ANTHROPIC_MODEL / ANTHROPIC_SMALL_FAST_MODEL set Claude Code's
        # initial active model.  Selecting a builtin via /model overrides
        # this at runtime.
        assert captured["env"]["ANTHROPIC_MODEL"] == "nvidia/moonshotai/kimi-k2.5"
        assert captured["env"]["ANTHROPIC_SMALL_FAST_MODEL"] == "nvidia/moonshotai/kimi-k2.5"
        assert captured["env"]["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
        # ANTHROPIC_CUSTOM_MODEL_OPTION registers the Switchyard-routed model
        # as a *persistent* /model picker entry — the picker reads this on
        # every render, so the entry survives toggling to a builtin.
        assert captured["env"]["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "nvidia/moonshotai/kimi-k2.5"
        assert "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME" not in captured["env"]
        assert "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION" not in captured["env"]
        # Proxy torn down on return
        assert fake_server.should_exit is True

    def test_intake_injects_custom_headers(self, monkeypatch):
        fake_server = _make_fake_server(started=True)
        fake_sdk = MagicMock()
        fake_sdk.workspace = "sdk-workspace"

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_claude_binary",
            lambda: "/fake/bin/claude",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(fake_server),
        )
        monkeypatch.setattr(
            "switchyard.lib.processors.intake_client._build_sdk_client",
            lambda config: fake_sdk,
        )

        captured: dict = {}

        def fake_run(cmd, env, check):
            captured["env"] = env
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        intake = LaunchIntakeConfig.from_resolved(
            base_url="https://intake.example",
            workspace=None,
            api_key=None,
            app="claude-code",
            task="developer-session",
            session_id="sess-xyz",
            target="claude",
        )
        exit_code = launch_claude(
            model="nvidia/moonshotai/kimi-k2.5",
            base_url="https://inference-api.nvidia.com/v1",
            api_key="test-key",
            port=None,
            timeout=None,
            claude_args=[],
            intake=intake,
        )

        assert exit_code == 0
        assert captured["env"]["SWITCHYARD_SESSION_ID"] == "sess-xyz"
        custom = captured["env"]["ANTHROPIC_CUSTOM_HEADERS"]
        assert "x-switchyard-intake-enabled: true" in custom
        assert "x-switchyard-intake-app: claude-code" in custom
        assert "proxy_x_session_id: sess-xyz" in custom

    def test_port_override(self, monkeypatch):
        fake_server = _make_fake_server(started=True)

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_claude_binary",
            lambda: "/fake/bin/claude",
        )

        # If --port is set, _find_free_port should NOT be called
        def _should_not_be_called():
            raise AssertionError("_find_free_port called despite --port override")
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_free_port",
            _should_not_be_called,
        )

        captured: dict = {}

        def stub_spawn(switchyard, port):
            captured["port"] = port
            thread = MagicMock()
            return fake_server, thread

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._spawn_proxy_thread",
            stub_spawn,
        )
        monkeypatch.setattr(
            subprocess, "run",
            lambda cmd, env, check: subprocess.CompletedProcess(cmd, returncode=0),
        )

        exit_code = launch_claude(
            model="m", base_url="u", api_key="k",
            port=4000, timeout=None, claude_args=[],
        )

        assert exit_code == 0
        assert captured["port"] == 4000

    def test_missing_binary_returns_127(self, monkeypatch):
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_claude_binary",
            lambda: None,
        )

        # If we reach _spawn_proxy_thread, we failed to short-circuit
        def _should_not_spawn(*args, **kwargs):
            raise AssertionError("proxy spawned despite missing binary")
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._spawn_proxy_thread",
            _should_not_spawn,
        )

        exit_code = launch_claude(
            model="m", base_url="u", api_key="k",
            port=None, timeout=None, claude_args=[],
        )
        assert exit_code == _EXIT_BINARY_NOT_FOUND

    def test_ctrl_c_returns_130_and_tears_down(self, monkeypatch):
        fake_server = _make_fake_server(started=True)

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_claude_binary",
            lambda: "/fake/bin/claude",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(fake_server),
        )

        def raise_sigint(cmd, env, check):
            raise KeyboardInterrupt()
        monkeypatch.setattr(subprocess, "run", raise_sigint)

        exit_code = launch_claude(
            model="m", base_url="u", api_key="k",
            port=None, timeout=None, claude_args=[],
        )

        assert exit_code == _EXIT_SIGINT
        assert fake_server.should_exit is True

    def test_strips_leading_double_dash_from_claude_args(self, monkeypatch, tmp_path):
        """``argparse.REMAINDER`` keeps the ``--`` sentinel in the captured
        list, so ``launch claude ... -- --version`` produces
        ``['--', '--version']``. The handler must strip the leading ``--``
        before forwarding so ``subprocess.run`` doesn't receive a bare
        ``--`` as an arg.
        """
        from switchyard.cli.switchyard_cli import (
            _build_parser,
            _cmd_launch_claude,
        )

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "claude",
            "--model", "nvidia/moonshotai/kimi-k2.5",
            "--api-key", "sk-test",
            "--", "--version",
        ])
        assert args.claude_args == ["--", "--version"]  # argparse kept '--'

        captured: dict = {}

        def fake_launch(**kwargs):
            captured.update(kwargs)
            raise SystemExit(0)

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://inference-api.nvidia.com/v1"),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher.launch_claude",
            fake_launch,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_claude(args)

        # Handler stripped the '--' before forwarding.
        assert captured["claude_args"] == ["--version"]

    def test_cmd_launch_claude_resolves_intake_args(self, monkeypatch, tmp_path):
        from switchyard.cli.switchyard_cli import (
            _build_parser,
            _cmd_launch_claude,
        )

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "claude",
            "--model", "nvidia/moonshotai/kimi-k2.5",
            "--api-key", "sk-test",
            "--intake-enabled",
            "--intake-base-url", "https://nmp.example",
            "--intake-api-key", "ci-token",
            "--intake-app", "cli-app",
            "--intake-task", "custom-task",
            "--intake-session-id", "sess-cli",
        ])

        captured: dict = {}

        def fake_launch(**kwargs):
            captured.update(kwargs)
            raise SystemExit(0)

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://inference-api.nvidia.com/v1"),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher.launch_claude",
            fake_launch,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_claude(args)

        intake = captured["intake"]
        assert intake.base_url == "https://nmp.example"
        assert intake.api_key == "ci-token"
        assert intake.app == "cli-app"
        assert intake.task == "custom-task"
        assert intake.session_id == "sess-cli"

    def test_selects_native_backend_when_probe_true(self, monkeypatch):
        """Probe returning True -> chain stats-wraps an Anthropic-native backend."""
        from switchyard.lib.backends.stats_llm_backend import StatsLlmBackend

        monkeypatch.setattr(
            "switchyard.lib.backends.backend_format_resolver."
            "probe_anthropic_messages_support_sync",
            lambda **_: True,
        )

        model = "aws/anthropic/bedrock-claude-opus-4-6"
        captured_switchyard: dict = {}

        def stub_spawn(app, port):
            assert isinstance(app, RouteTable)
            chain = app.lookup_switchyard(model)
            captured_switchyard["app"] = app
            captured_switchyard["switchyard"] = chain
            captured_switchyard["backend"] = next(
                component
                for component in chain.iter_components()
                if isinstance(component, StatsLlmBackend)
            )
            fake_server = _make_fake_server(started=True)
            return fake_server, MagicMock()

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_claude_binary",
            lambda: "/fake/bin/claude",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._spawn_proxy_thread",
            stub_spawn,
        )
        monkeypatch.setattr(
            subprocess, "run",
            lambda cmd, env, check: subprocess.CompletedProcess(cmd, returncode=0),
        )

        launch_claude(
            model=model,
            base_url="https://inference-api.nvidia.com/v1",
            api_key="sk-test",
            port=None, timeout=None, claude_args=[],
        )
        table = captured_switchyard["app"]
        assert table.registered_models() == [f"claude-{model}", model]
        assert table.default_model() == model
        assert table.lookup_switchyard(f"claude-{model}") is table.lookup_switchyard(model)
        backend = captured_switchyard["backend"]
        assert isinstance(backend, StatsLlmBackend)
        assert [item.value for item in backend.supported_request_types] == ["anthropic"]
        assert not any(
            isinstance(component, ModelRewriteRequestProcessor)
            for component in captured_switchyard["switchyard"].iter_components()
        )

    def test_selects_translated_backend_when_probe_false(self, monkeypatch):
        """Probe returning False -> chain stats-wraps an OpenAI-native backend."""
        from switchyard.lib.backends.stats_llm_backend import StatsLlmBackend
        # Default autouse fixture already sets probe to False; no override needed.

        model = "nvidia/moonshotai/kimi-k2.5"
        captured_switchyard: dict = {}

        def stub_spawn(app, port):
            assert isinstance(app, RouteTable)
            chain = app.lookup_switchyard(model)
            captured_switchyard["app"] = app
            captured_switchyard["switchyard"] = chain
            captured_switchyard["backend"] = next(
                component
                for component in chain.iter_components()
                if isinstance(component, StatsLlmBackend)
            )
            fake_server = _make_fake_server(started=True)
            return fake_server, MagicMock()

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_claude_binary",
            lambda: "/fake/bin/claude",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._spawn_proxy_thread",
            stub_spawn,
        )
        monkeypatch.setattr(
            subprocess, "run",
            lambda cmd, env, check: subprocess.CompletedProcess(cmd, returncode=0),
        )

        launch_claude(
            model=model,
            base_url="https://inference-api.nvidia.com/v1",
            api_key="sk-test",
            port=None, timeout=None, claude_args=[],
        )
        table = captured_switchyard["app"]
        assert table.registered_models() == [f"claude-{model}", model]
        assert table.default_model() == model
        assert table.lookup_switchyard(f"claude-{model}") is table.lookup_switchyard(model)
        backend = captured_switchyard["backend"]
        assert isinstance(backend, StatsLlmBackend)
        assert [item.value for item in backend.supported_request_types] == ["openai_chat"]
        assert not any(
            isinstance(component, ModelRewriteRequestProcessor)
            for component in captured_switchyard["switchyard"].iter_components()
        )

    def test_smoke_with_routing_profiles_errors_clearly(
        self, monkeypatch, tmp_path,
    ):
        """``--smoke --routing-profiles FILE`` is rejected at the CLI level."""
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_claude

        yaml_path = tmp_path / "bundle.yaml"
        yaml_path.write_text(
            "routes:\n"
            "  fast-nemotron:\n"
            "    type: model\n"
            "    target: nvidia/nvidia/nemotron-nano-9b-v2\n"
        )
        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://inference-api.nvidia.com/v1"),
        )
        parser = _build_parser()
        args = parser.parse_args([
            "--routing-profiles", str(yaml_path),
            "launch", "claude", "--smoke", "--api-key", "sk-test",
        ])
        with pytest.raises(SystemExit) as exc_info:
            _cmd_launch_claude(args)
        assert "--smoke and --routing-profiles cannot be combined" in str(exc_info.value)

    def test_smoke_without_model_errors_with_helpful_message(
        self, monkeypatch, tmp_path,
    ):
        """``--smoke`` with no model gives a clear error directing to ``--model``."""
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_claude

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://inference-api.nvidia.com/v1"),
        )
        parser = _build_parser()
        args = parser.parse_args(["launch", "claude", "--smoke", "--api-key", "sk-test"])
        with pytest.raises(SystemExit) as exc_info:
            _cmd_launch_claude(args)
        assert "--smoke requires --model" in str(exc_info.value)

    def test_routing_profiles_merges_extras_on_top_of_launcher_table(
        self, monkeypatch, tmp_path,
    ):
        """`--routing-profiles` adds YAML routes on top of the launcher's chain.

        The launcher registers its own ``--model`` chain under ``model``;
        every YAML entry is merged on top via
        :meth:`RouteTable.register`. YAML wins on id conflict.
        """
        captured_switchyard: dict = {}

        def stub_spawn(switchyard, port):
            captured_switchyard["app"] = switchyard
            fake_server = _make_fake_server(started=True)
            return fake_server, MagicMock()

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_claude_binary",
            lambda: "/fake/bin/claude",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._spawn_proxy_thread",
            stub_spawn,
        )
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, env, check: subprocess.CompletedProcess(cmd, returncode=0),
        )

        yaml_path = tmp_path / "extras.yaml"
        yaml_path.write_text(
            "routes:\n"
            "  extras/yaml-only:\n"
            "    type: model\n"
            "    target:\n"
            "      model: extras/yaml-only\n"
            "      api_key: sk-yaml\n"
            "      base_url: https://yaml.example/v1\n"
            "  primary/model:\n"
            "    type: model\n"
            "    target:\n"
            "      model: primary/model\n"
            "      api_key: sk-yaml-override\n"
            "      base_url: https://yaml-override.example/v1\n"
        )

        launch_claude(
            model="primary/model",
            base_url="https://example.invalid/v1",
            api_key="sk-test",
            port=None,
            timeout=None,
            claude_args=[],
            routing_profiles=str(yaml_path),
        )

        table = captured_switchyard["app"]
        assert isinstance(table, RouteTable)

        # Launcher registers `primary/model`; YAML overrides it in place and
        # adds `extras/yaml-only`. The claude launcher also exposes each
        # non-prefixed id under a `claude-` alias (same chain object) so
        # Claude Code's gateway-discovery filter accepts the full listing.
        # Aliases come right before their originals in iteration order.
        assert table.registered_models() == [
            "claude-primary/model",
            "primary/model",
            "claude-extras/yaml-only",
            "extras/yaml-only",
        ]
        # Alias and original resolve to the same chain object.
        assert table.default_model() == "primary/model"
        assert table.lookup_switchyard("claude-extras/yaml-only") is table.lookup_switchyard(
            "extras/yaml-only"
        )
        assert table.lookup_switchyard("claude-primary/model") is table.lookup_switchyard(
            "primary/model"
        )

    def test_proxy_never_ready_returns_error(self, monkeypatch):
        fake_server = _make_fake_server(started=False)  # never flips to True

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_claude_binary",
            lambda: "/fake/bin/claude",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(fake_server),
        )
        # Override the autouse _wait_ready mock to simulate a timeout.
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._wait_ready",
            lambda port, timeout_s=10.0: False,
        )

        def _should_not_run(*args, **kwargs):
            raise AssertionError("claude spawned despite proxy not ready")
        monkeypatch.setattr(subprocess, "run", _should_not_run)

        exit_code = launch_claude(
            model="m", base_url="u", api_key="k",
            port=None, timeout=None, claude_args=[],
        )
        assert exit_code == 1
        assert fake_server.should_exit is True


# ---------------------------------------------------------------------------
# _print_ready_banner
# ---------------------------------------------------------------------------


class TestPrintReadyBanner:
    """Banner must surface the proxy URL + stats curl on stderr unconditionally.

    Critical because Claude Code's TUI takeover plus the silencer that drops
    ``switchyard`` to WARNING was hiding the previous logger-based
    status line entirely.
    """

    def test_includes_proxy_url_and_stats_curl(self, capsys):
        _print_ready_banner(46385, "azure/anthropic/claude-opus-4-6")
        err = capsys.readouterr().err
        assert "http://127.0.0.1:46385" in err
        assert "curl -s http://127.0.0.1:46385/v1/routing/stats" in err
        assert "azure/anthropic/claude-opus-4-6" in err

    def test_writes_to_stderr_not_stdout(self, capsys):
        _print_ready_banner(4000, "m")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "switchyard" in captured.err and "ready" in captured.err

    def test_survives_logger_silencing(self, capsys):
        logging.getLogger("switchyard").setLevel(logging.WARNING)
        try:
            _print_ready_banner(4000, "m")
        finally:
            logging.getLogger("switchyard").setLevel(logging.NOTSET)
        assert "http://127.0.0.1:4000" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _make_footer_fn — passthrough vs random-routing layouts
# ---------------------------------------------------------------------------


class _StubHealth:
    """Minimal stand-in for ``ProxyHealthMonitor`` — the renderer only
    calls ``poll()`` (no-op here) and reads the ``indicator`` tuple.
    Avoids opening a real socket from a unit test.
    """

    def __init__(self, indicator: tuple[str, int] = ("●", 1)) -> None:
        self._indicator = indicator

    def poll(self) -> None:
        return None

    @property
    def indicator(self) -> tuple[str, int]:
        return self._indicator


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _record_stats(
    stats: StatsAccumulator,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    tier: str | None = None,
) -> None:
    async def _record() -> None:
        await stats.record_success(model=model, tier=tier)
        await stats.record_usage(
            model=model,
            tier=tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    asyncio.run(_record())


class TestMakeFooterFn:
    """Footer: aggregate row + one row per active model tier."""

    def test_renders_two_rows_with_active_model(self):
        stats = StatsAccumulator()
        _record_stats(
            stats,
            model="kimi-k2.5", tier="strong",
            prompt_tokens=800, completion_tokens=400,
        )
        fn = _make_footer_fn(
            stats, "openai/openai/kimi-k2.5", cast(ProxyHealthMonitor, _StubHealth()),
        )
        rows = fn(120)
        assert len(rows) == 2
        agg, tier = (_strip_ansi(r[0]) for r in rows)
        # Aggregate row carries totals but no model name.
        assert "1 req" in agg and "800 in" in agg and "400 out" in agg
        assert "kimi-k2.5" not in agg
        # Tier row carries the model the backend actually saw.
        assert "kimi-k2.5" in tier
        assert "1 req" in tier

    def test_renders_two_rows_at_zero_traffic_with_default_label(self):
        stats = StatsAccumulator()
        fn = _make_footer_fn(
            stats, "openai/openai/kimi-k2.5",
            cast(ProxyHealthMonitor, _StubHealth()),
        )
        rows = fn(120)
        assert len(rows) == 2
        # Aggregate is non-empty even at zero traffic; tier falls back to
        # the launch default label.
        assert _strip_ansi(rows[0][0]).strip() != ""
        assert "kimi-k2.5" in _strip_ansi(rows[1][0])

    def test_shows_one_row_per_active_tier(self):
        """Two-tier routing → 3 rows total: aggregate + one per model."""
        stats = StatsAccumulator()
        _record_stats(
            stats,
            model="aws/anthropic/bedrock-claude-opus-4-7", tier="strong",
            prompt_tokens=120, completion_tokens=80,
        )
        _record_stats(
            stats,
            model="nvidia/deepseek-ai/evals-deepseek-v4-pro", tier="weak",
            prompt_tokens=200, completion_tokens=150,
        )
        fn = _make_footer_fn(
            stats, "switchyard-deterministic-abc12345",
            cast(ProxyHealthMonitor, _StubHealth()),
        )
        rows = fn(120)
        assert len(rows) == 3
        agg = _strip_ansi(rows[0][0])
        tier_texts = [_strip_ansi(r[0]) for r in rows[1:]]
        # Aggregate sees both calls.
        assert "2 req" in agg
        # Both models appear, one per row.
        all_text = " ".join(tier_texts)
        assert "bedrock-claude-opus-4-7" in all_text
        assert "evals-deepseek-v4-pro" in all_text

_MINIMAL_YAML_BUNDLE = (
    "routes:\n"
    "  example/model:\n"
    "    type: model\n"
    "    target:\n"
    "      model: example/model\n"
    "      api_key: sk-test\n"
    "      base_url: https://example.invalid/v1\n"
)


class TestResolveRoutingProfiles:
    """Precedence rules for the saved-bundle + CLI routing-profile resolution."""

    _BUNDLE = {"routes": {"example/model": {"type": "model"}}}

    def test_cli_value_wins_over_saved(self, monkeypatch, tmp_path):
        from switchyard.cli.config.user_config import UserConfig, save_user_config
        from switchyard.cli.launch_command import _resolve_routing_profiles
        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        save_user_config(UserConfig(routing_profiles=self._BUNDLE))
        args = argparse.Namespace(routing_profiles="/cli/path.yaml", model=None)
        assert _resolve_routing_profiles(args) == "/cli/path.yaml"

    def test_saved_bundle_materialized_to_yaml_tempfile(
        self, monkeypatch, tmp_path,
    ):
        """The saved dict is re-serialized to a tempfile path the launcher reads."""
        import yaml

        from switchyard.cli.config.user_config import UserConfig, save_user_config
        from switchyard.cli.launch_command import _resolve_routing_profiles
        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        save_user_config(UserConfig(routing_profiles=self._BUNDLE))
        args = argparse.Namespace(routing_profiles=None, model=None)
        resolved = _resolve_routing_profiles(args)
        assert resolved is not None
        materialized = Path(resolved)
        assert materialized.exists()
        assert yaml.safe_load(materialized.read_text()) == self._BUNDLE

    def test_model_only_does_not_inject_saved(self, monkeypatch, tmp_path):
        """`--model X` alone is an explicit opt-in to single-model; saved bundle stays out."""
        from switchyard.cli.config.user_config import UserConfig, save_user_config
        from switchyard.cli.launch_command import _resolve_routing_profiles
        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        save_user_config(UserConfig(routing_profiles=self._BUNDLE))
        args = argparse.Namespace(routing_profiles=None, model="some/model")
        assert _resolve_routing_profiles(args) is None

    def test_cli_value_wins_even_with_model_flag(self, monkeypatch, tmp_path):
        """`--model X --routing-profiles Y` composes (CLI path wins, saved ignored)."""
        from switchyard.cli.config.user_config import UserConfig, save_user_config
        from switchyard.cli.launch_command import _resolve_routing_profiles
        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        save_user_config(UserConfig(routing_profiles=self._BUNDLE))
        args = argparse.Namespace(
            routing_profiles="/cli/path.yaml", model="some/model",
        )
        assert _resolve_routing_profiles(args) == "/cli/path.yaml"

    def test_returns_none_when_nothing_saved_or_passed(self, monkeypatch, tmp_path):
        from switchyard.cli.launch_command import _resolve_routing_profiles
        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        args = argparse.Namespace(routing_profiles=None, model=None)
        assert _resolve_routing_profiles(args) is None


class TestResolveInitialFromProfiles:
    """First declared route is always returned; --model + --routing-profiles is an error."""

    def _write_yaml(self, tmp_path):
        yaml_path = tmp_path / "profiles.yaml"
        yaml_path.write_text(
            "routes:\n"
            "  some/route:\n"
            "    type: model\n"
            "    target:\n"
            "      model: some/upstream\n"
            "      api_key: sk-test\n"
            "      base_url: https://example.invalid/v1\n"
        )
        return str(yaml_path)

    def test_returns_first_yaml_route(self, tmp_path):
        from switchyard.cli.launch_command import _resolve_initial_from_profiles
        yaml_path = self._write_yaml(tmp_path)
        assert (
            _resolve_initial_from_profiles(target="codex", routing_profiles=yaml_path)
            == "some/route"
        )

    def test_empty_bundle_raises(self, tmp_path):
        from switchyard.cli.launch_command import _resolve_initial_from_profiles
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("routes:\n  noop:\n    type: noop\n")
        assert (
            _resolve_initial_from_profiles(target="codex", routing_profiles=str(yaml_path))
            == "noop"
        )

    def test_model_and_profiles_mutually_exclusive(self):
        """Runtime check rejects --model + --routing-profiles together for all launchers."""
        from switchyard.cli.launch_command import (
            cmd_launch_claude,
            cmd_launch_codex,
            cmd_launch_openclaw,
        )
        from switchyard.cli.switchyard_cli import _build_parser
        parser = _build_parser()
        handlers = {
            "claude": cmd_launch_claude,
            "codex": cmd_launch_codex,
            "openclaw": cmd_launch_openclaw,
        }
        for cmd, handler in handlers.items():
            # --routing-profiles is now global; --model stays on the launcher
            args = parser.parse_args(
                ["--routing-profiles", "p.yaml", "launch", cmd, "--model", "some/model"]
            )
            with pytest.raises(SystemExit) as exc:
                handler(args)
            assert exc.value.code != 0 or "mutually exclusive" in str(exc.value)


class TestServeRoutingProfilesFallback:
    """`switchyard serve` falls back to the saved parsed bundle (no tempfile)."""

    _BUNDLE = {
        "routes": {
            "example/model": {
                "type": "model",
                "target": {
                    "model": "example/model",
                    "api_key": "sk-test",
                    "base_url": "https://example.invalid/v1",
                },
            },
        },
    }

    def test_falls_back_to_saved_bundle_when_cli_omitted(
        self, monkeypatch, tmp_path,
    ):
        """Saved dict feeds straight into build_route_bundle_table."""
        from switchyard.cli.config.user_config import UserConfig, save_user_config
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_serve

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        save_user_config(UserConfig(routing_profiles=self._BUNDLE))

        captured: dict = {}

        def fake_build_and_serve(args, table, inbound_default, **_kwargs):
            captured["registered_models"] = table.registered_models()
            captured["inbound_default"] = inbound_default

        monkeypatch.setattr(
            "switchyard.cli.switchyard_cli.build_and_serve",
            fake_build_and_serve,
        )

        parser = _build_parser()
        args = parser.parse_args(["serve", "--port", "4000"])
        _cmd_serve(args)
        assert captured["registered_models"] == ["example/model"]
        assert captured["inbound_default"] == "both"

    def test_cli_path_overrides_saved(self, monkeypatch, tmp_path):
        from switchyard.cli.config.user_config import UserConfig, save_user_config
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_serve

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        cli_yaml = tmp_path / "cli.yaml"
        cli_yaml.write_text(_MINIMAL_YAML_BUNDLE)
        save_user_config(UserConfig(routing_profiles=self._BUNDLE))

        captured: dict = {}

        def fake_build_and_serve(args, table, inbound_default, **_kwargs):
            captured["routing_profiles"] = args.routing_profiles

        monkeypatch.setattr(
            "switchyard.cli.switchyard_cli.build_and_serve",
            fake_build_and_serve,
        )

        parser = _build_parser()
        args = parser.parse_args([
            "--routing-profiles", str(cli_yaml), "serve", "--port", "4000",
        ])
        _cmd_serve(args)
        # CLI path wins as-is.
        assert captured["routing_profiles"] == str(cli_yaml)

    def test_errors_when_neither_cli_nor_saved(self, monkeypatch, tmp_path):
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_serve

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        parser = _build_parser()
        args = parser.parse_args(["serve", "--port", "4000"])
        with pytest.raises(SystemExit) as excinfo:
            _cmd_serve(args)
        assert "routing-profiles" in str(excinfo.value)


class TestConfigurePersistsRoutingProfiles:
    """`switchyard configure --routing-profiles PATH` parses + snapshots the bundle."""

    def test_cli_path_persists_parsed_bundle(self, monkeypatch, tmp_path):
        from switchyard.cli.config.user_config import load_user_config
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_configure

        cwd = tmp_path / "cwd"
        cwd.mkdir()
        yaml_content = (
            "routes:\n"
            "  example/model:\n"
            "    type: model\n"
            "    target:\n"
            "      model: example/model\n"
            "      api_key: ${TEST_API_KEY}\n"
            "      base_url: https://example.invalid/v1\n"
        )
        rel_yaml = cwd / "route.yaml"
        rel_yaml.write_text(yaml_content)
        monkeypatch.chdir(cwd)

        config_dir = tmp_path / "config"
        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(config_dir))
        monkeypatch.setattr(
            "switchyard.cli.command_utils.is_interactive_terminal",
            lambda: False,
        )
        monkeypatch.setattr(
            "switchyard.cli.configure_command.is_interactive_terminal",
            lambda: False,
        )
        monkeypatch.setattr(
            "switchyard.cli.configure_command.discover_models",
            lambda base_url, api_key, disabled: ["model-a"],
        )

        parser = _build_parser()
        args = parser.parse_args([
            "--routing-profiles", "route.yaml",
            "configure",
            "--target", "claude",
            "--api-key", "sk-test",
            "--claude-model", "model-a",
            "--no-model-discovery",
        ])
        _cmd_configure(args)

        saved = load_user_config(config_dir).routing_profiles
        assert saved is not None
        # Env-var references should be preserved verbatim (re-expanded at load).
        target = saved["routes"]["example/model"]["target"]
        assert target["api_key"] == "${TEST_API_KEY}"

    def test_first_route_becomes_model_default(self, monkeypatch, tmp_path):
        """With a routing profile and no --claude-model, the first route key is
        the saved Claude model default (matching what the launcher seeds)."""
        from switchyard.cli.config.user_config import load_user_config
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_configure

        cwd = tmp_path / "cwd"
        cwd.mkdir()
        # Two routes; `coding-agent` is declared first, so it wins as the default.
        yaml_content = (
            "routes:\n"
            "  coding-agent:\n"
            "    type: model\n"
            "    target:\n"
            "      model: aws/anthropic/bedrock-claude-opus-4-7\n"
            "  other-route:\n"
            "    type: model\n"
            "    target:\n"
            "      model: nvidia/nemotron-3-super\n"
        )
        (cwd / "route.yaml").write_text(yaml_content)
        monkeypatch.chdir(cwd)

        config_dir = tmp_path / "config"
        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(config_dir))
        monkeypatch.setattr(
            "switchyard.cli.command_utils.is_interactive_terminal",
            lambda: False,
        )
        monkeypatch.setattr(
            "switchyard.cli.configure_command.is_interactive_terminal",
            lambda: False,
        )
        monkeypatch.setattr(
            "switchyard.cli.configure_command.discover_models",
            lambda base_url, api_key, *, disabled: [],
        )

        parser = _build_parser()
        # No --claude-model: the routing profile's first route supplies the default.
        args = parser.parse_args([
            "--routing-profiles", "route.yaml",
            "configure",
            "--target", "claude",
            "--api-key", "sk-test",
            "--no-model-discovery",
        ])
        _cmd_configure(args)

        saved = load_user_config(config_dir)
        # The saved claude model uses the `claude-` aliased form (the launcher
        # exposes every non-prefixed route under that alias so Claude Code's
        # gateway-discovery picker picks it up).
        assert saved.launch_target("claude").effective_route().model == "claude-coding-agent"

    def test_explicit_model_flag_overrides_routing_profile_default(
        self, monkeypatch, tmp_path,
    ):
        """An explicit --claude-model wins over the routing profile's first route."""
        from switchyard.cli.config.user_config import load_user_config
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_configure

        cwd = tmp_path / "cwd"
        cwd.mkdir()
        (cwd / "route.yaml").write_text(
            "routes:\n"
            "  coding-agent:\n"
            "    type: model\n"
            "    target:\n"
            "      model: aws/anthropic/bedrock-claude-opus-4-7\n"
        )
        monkeypatch.chdir(cwd)

        config_dir = tmp_path / "config"
        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(config_dir))
        monkeypatch.setattr(
            "switchyard.cli.command_utils.is_interactive_terminal",
            lambda: False,
        )
        monkeypatch.setattr(
            "switchyard.cli.configure_command.is_interactive_terminal",
            lambda: False,
        )
        monkeypatch.setattr(
            "switchyard.cli.configure_command.discover_models",
            lambda base_url, api_key, *, disabled: [],
        )

        parser = _build_parser()
        args = parser.parse_args([
            "--routing-profiles", "route.yaml",
            "configure",
            "--target", "claude",
            "--api-key", "sk-test",
            "--claude-model", "my/explicit-model",
            "--no-model-discovery",
        ])
        _cmd_configure(args)

        saved = load_user_config(config_dir)
        assert saved.launch_target("claude").effective_route().model == "my/explicit-model"

    def test_empty_routing_profiles_clears_saved_content(
        self, monkeypatch, tmp_path,
    ):
        """Passing --routing-profiles '' wipes the saved snapshot."""
        from switchyard.cli.config.user_config import (
            UserConfig,
            load_user_config,
            save_user_config,
        )
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_configure

        config_dir = tmp_path / "config"
        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(config_dir))
        save_user_config(
            UserConfig(routing_profiles={"routes": {"x": {"type": "model"}}}),
            config_dir=config_dir,
        )
        monkeypatch.setattr(
            "switchyard.cli.command_utils.is_interactive_terminal",
            lambda: False,
        )
        monkeypatch.setattr(
            "switchyard.cli.configure_command.is_interactive_terminal",
            lambda: False,
        )
        monkeypatch.setattr(
            "switchyard.cli.configure_command.discover_models",
            lambda base_url, api_key, disabled: ["model-a"],
        )

        parser = _build_parser()
        args = parser.parse_args([
            "--routing-profiles", "",
            "configure",
            "--target", "claude",
            "--api-key", "sk-test",
            "--claude-model", "model-a",
            "--no-model-discovery",
        ])
        _cmd_configure(args)

        assert load_user_config(config_dir).routing_profiles is None

    def test_missing_path_errors_clearly(self, monkeypatch, tmp_path):
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_configure

        config_dir = tmp_path / "config"
        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(config_dir))
        monkeypatch.setattr(
            "switchyard.cli.command_utils.is_interactive_terminal",
            lambda: False,
        )
        monkeypatch.setattr(
            "switchyard.cli.configure_command.is_interactive_terminal",
            lambda: False,
        )
        monkeypatch.setattr(
            "switchyard.cli.configure_command.discover_models",
            lambda base_url, api_key, disabled: ["model-a"],
        )

        parser = _build_parser()
        args = parser.parse_args([
            "--routing-profiles", "/this/does/not/exist.yaml",
            "configure",
            "--target", "claude",
            "--api-key", "sk-test",
            "--claude-model", "model-a",
            "--no-model-discovery",
        ])
        with pytest.raises(SystemExit) as excinfo:
            _cmd_configure(args)
        assert "file not found" in str(excinfo.value)
