# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``switchyard.cli.launchers.claude_code_launcher``.

Exercises the public ``launch_claude`` entry and ``claude`` binary lookup.
The native server and ``subprocess.run`` are mocked, so these tests do not
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

from switchyard.cli.configure_request import ConfigureRequest
from switchyard.cli.launchers.claude_code_launcher import (
    _EXIT_BINARY_NOT_FOUND,
    _EXIT_SIGINT,
    ProxyHealthMonitor,
    _find_claude_binary,
    _find_free_port,
    _make_footer_fn,
    _print_ready_banner,
    launch_claude,
)
from switchyard.cli.launchers.launch_intake_config import LaunchIntakeConfig
from switchyard.lib.backends.llm_target import LlmTarget
from switchyard.lib.profiles.random_routing import RandomRoutingConfig
from switchyard.lib.route_table_builders import (
    random_routing_virtual_model_id,
)
from switchyard.lib.stats_accumulator import StatsAccumulator


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
    captured: dict[str, ConfigureRequest] = {}
    monkeypatch.setattr(
        "switchyard.cli.launch_command.cmd_configure",
        lambda configure_request: captured.setdefault("request", configure_request),
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

    configure_request = captured["request"]
    assert configure_request.provider == "nvidia"
    assert configure_request.base_url == "https://nvidia.test/v1"
    assert configure_request.prompt_default_api_key == "nvidia-key"  # pragma: allowlist secret
    assert configure_request.prompt_default_api_key_source == "$NVIDIA_API_KEY"

    # Setup leaves these three at their ConfigureRequest defaults. Assert them
    # directly so a regression in any one field fails here, not mid-launch: a
    # missing default is exactly what crashed the first run before this fix.
    assert configure_request.routing_profiles is None
    assert configure_request.skill_distillation is None
    assert configure_request.disable_skill_distillation is False

    # The request setup builds must work with the real configure helpers. Before
    # the typed ConfigureRequest, setup passed a hand-built namespace missing the
    # skill-distillation fields, and these helpers crashed with AttributeError.
    from switchyard.cli.config.user_config import SkillDistillationConfig
    from switchyard.cli.configure_command import (
        _apply_skill_distillation_args,
        _skill_only_config_update,
    )

    assert (
        _apply_skill_distillation_args(SkillDistillationConfig(), configure_request)
        == SkillDistillationConfig()
    )
    assert _skill_only_config_update(configure_request) is False


def test_configure_parser_builds_a_complete_request() -> None:
    # A real `switchyard configure` parse must fill every ConfigureRequest field
    # through the one from_namespace boundary. This is the check the original bug
    # lacked: disable_skill_distillation is the field that crashed first-run.
    from switchyard.cli.switchyard_cli import _build_parser

    namespace = _build_parser().parse_args(["configure"])
    request = ConfigureRequest.from_namespace(namespace)

    # --provider defaults to None so it can't shadow a saved default_provider;
    # cmd_configure resolves the effective provider from the saved config.
    assert request.provider is None
    assert request.disable_skill_distillation is False
    assert request.routing_profiles is None  # the global flag is merged in


def test_main_forwards_harness_args_after_separator(monkeypatch, tmp_path) -> None:
    """``launch claude ... -- --version`` forwards harness args past the ``--``.

    main() must only strip a ``--`` that occurs before the subcommand token,
    so the launcher-side ``--`` separating claude args survives argparse and
    the forwarded args reach ``launch_claude`` intact. Before the fix the first
    ``--`` was popped unconditionally, so ``--version`` hit argparse as an
    unrecognized argument (``SystemExit(2)``).
    """
    import sys

    from switchyard.cli import switchyard_cli as cli

    monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        sys, "argv",
        [
            "switchyard", "launch", "claude",
            "--model", "nvidia/x", "--api-key", "sk-test",
            "--", "--version",
        ],
    )
    monkeypatch.setattr(
        "switchyard.cli.launch_command.resolve_launch_connectivity",
        lambda args, **_kw: ("sk-test", "https://inference-api.nvidia.com/v1"),
    )

    captured: dict = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(
        "switchyard.cli.launchers.claude_code_launcher.launch_claude",
        fake_launch,
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0
    assert captured["claude_args"] == ["--version"]


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
    """Native server stand-in with empty HTTP-backed stats."""
    server = MagicMock()
    server.started = started
    server.should_exit = False
    server.port = 54321
    server.base_url = "http://127.0.0.1:54321"
    server.stats.snapshot_sync.return_value = {}
    server.close.side_effect = lambda: setattr(server, "should_exit", True)
    return server


def _stub_native_server(server: MagicMock):
    """Return a function that starts the supplied native server stand-in."""
    def _inner(deployment, port):
        if port is not None:
            server.port = port
            server.base_url = f"http://127.0.0.1:{port}"
        return server
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
            "switchyard.cli.launchers.claude_code_launcher._start_native_server",
            _stub_native_server(fake_server),
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
            "switchyard.cli.launchers.claude_code_launcher._start_native_server",
            _stub_native_server(fake_server),
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

        def stub_spawn(deployment, port):
            captured["port"] = port
            fake_server.port = port
            return fake_server

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._start_native_server",
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

        # If we reach _start_native_server, we failed to short-circuit
        def _should_not_spawn(*args, **kwargs):
            raise AssertionError("proxy spawned despite missing binary")
        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher._start_native_server",
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
            "switchyard.cli.launchers.claude_code_launcher._start_native_server",
            _stub_native_server(fake_server),
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

    def test_cmd_launch_claude_uses_explicit_native_config(self, monkeypatch, tmp_path):
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_claude

        config_path = tmp_path / "deployment.toml"
        captured: dict = {}

        def fake_launch(**kwargs):
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(
            "switchyard.cli.launchers.claude_code_launcher.launch_claude_config",
            fake_launch,
        )
        args = _build_parser().parse_args([
            "launch", "claude",
            "--config", str(config_path),
            "--model", "route-id",
            "--", "--version",
        ])

        with pytest.raises(SystemExit) as exc_info:
            _cmd_launch_claude(args)

        assert exc_info.value.code == 0
        assert captured == {
            "config": config_path,
            "model": "route-id",
            "port": None,
            "claude_args": ["--version"],
            "intake": None,
        }

    def test_routing_profiles_errors_clearly(
        self, monkeypatch, tmp_path,
    ):
        """Launchers do not consume the serve/configure routing profile."""
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
            "launch", "claude", "--api-key", "sk-test",
        ])
        with pytest.raises(SystemExit) as exc_info:
            _cmd_launch_claude(args)
        assert "--routing-profiles is only supported by switchyard serve/configure" in str(
            exc_info.value
        )

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
            "switchyard.cli.launchers.claude_code_launcher._start_native_server",
            _stub_native_server(fake_server),
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
        assert "curl -s http://127.0.0.1:46385/v1/stats" in err
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


class TestLaunchRoutingProfiles:
    def test_rejected_for_all_launchers(self):
        """Routing profiles belong to serve/configure, not native launchers."""
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
            args = parser.parse_args(
                ["--routing-profiles", "p.yaml", "launch", cmd]
            )
            with pytest.raises(SystemExit) as exc:
                handler(args)
            assert "only supported by switchyard serve/configure" in str(exc.value)


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
        assert "switchyard --routing-profiles PATH configure" in str(excinfo.value)


class TestConfigurePersistsRoutingProfiles:
    """`switchyard --routing-profiles PATH configure` snapshots the bundle."""

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
