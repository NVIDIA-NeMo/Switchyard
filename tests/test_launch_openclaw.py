# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``switchyard.cli.launchers.openclaw_launcher``.

Exercises the public ``launch_openclaw`` entry, the transient JSON5
config builder, the env-var injection contract, and the ``openclaw``
binary lookup. Real uvicorn and ``subprocess.run`` are mocked —
these tests don't start a server or spawn a child process.

Mirrors :mod:`tests.test_launch_codex`; the per-launch transient
workspace replaces codex's ``-c`` provider overrides.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from switchyard.cli.launchers.launch_intake_config import LaunchIntakeConfig
from switchyard.cli.launchers.openclaw_launcher import (
    _API_KEY_ENV,
    _API_KEY_PLACEHOLDER,
    _EXIT_BINARY_NOT_FOUND,
    _EXIT_SIGINT,
    _PROVIDER_ID,
    _build_openclaw_config,
    _find_free_port,
    _find_openclaw_binary,
    _ModelRewriteRequestProcessor,
    _openclaw_command,
    _openclaw_env,
    _qualified_model_id,
    _write_openclaw_workspace,
    launch_openclaw,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.route_table import RouteTable
from switchyard_rust.core import ChatRequest

# ---------------------------------------------------------------------------
# _ModelRewriteRequestProcessor
# ---------------------------------------------------------------------------


class TestModelRewriteRequestProcessor:
    """OpenClaw clients only send Chat-Completions requests, but the
    processor is type-agnostic — exercise all three inbound shapes for
    parity with the claude/codex tests.
    """

    async def test_rewrites_openai_chat_request(self):
        proc = _ModelRewriteRequestProcessor("nvidia/moonshotai/kimi-k2.5")
        req = ChatRequest.openai_chat({"model": "gpt-4o", "messages": []})
        out = await proc.process(ProxyContext(), req)
        assert out is req
        assert req.body["model"] == "nvidia/moonshotai/kimi-k2.5"

    async def test_rewrites_responses_request(self):
        proc = _ModelRewriteRequestProcessor("target-model")
        req = ChatRequest.openai_responses({"model": "gpt-5.2", "input": "hi"})
        await proc.process(ProxyContext(), req)
        assert req.body["model"] == "target-model"

    async def test_rewrites_anthropic_request(self):
        proc = _ModelRewriteRequestProcessor("target-model")
        req = ChatRequest.anthropic({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        await proc.process(ProxyContext(), req)
        assert req.body["model"] == "target-model"


# ---------------------------------------------------------------------------
# _find_openclaw_binary
# ---------------------------------------------------------------------------


class TestFindOpenclawBinary:
    def test_returns_path_hit_when_on_path(self):
        with patch("shutil.which", return_value="/usr/local/bin/openclaw"):
            assert _find_openclaw_binary() == "/usr/local/bin/openclaw"

    def test_falls_back_to_npm_global(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        openclaw = tmp_path / ".npm-global" / "bin" / "openclaw"
        openclaw.parent.mkdir(parents=True)
        openclaw.write_text("#!/bin/sh\necho openclaw\n")
        openclaw.chmod(0o755)
        assert _find_openclaw_binary() == str(openclaw)

    def test_falls_back_to_local_bin(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        openclaw = tmp_path / ".local" / "bin" / "openclaw"
        openclaw.parent.mkdir(parents=True)
        openclaw.write_text("#!/bin/sh\necho openclaw\n")
        openclaw.chmod(0o755)
        assert _find_openclaw_binary() == str(openclaw)

    def test_falls_back_to_nvm(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        openclaw = tmp_path / ".nvm" / "versions" / "node" / "v22.0.0" / "bin" / "openclaw"
        openclaw.parent.mkdir(parents=True)
        openclaw.write_text("#!/bin/sh\necho openclaw\n")
        openclaw.chmod(0o755)
        assert _find_openclaw_binary() == str(openclaw)

    def test_nvm_picks_highest_version(self, tmp_path, monkeypatch):
        """When multiple Node versions exist under nvm, we prefer the
        lexicographically-highest one (newest by default for v22.x.x)."""
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        for version in ("v20.0.0", "v22.5.0"):
            bin_dir = tmp_path / ".nvm" / "versions" / "node" / version / "bin"
            bin_dir.mkdir(parents=True)
            openclaw = bin_dir / "openclaw"
            openclaw.write_text("#!/bin/sh\necho openclaw\n")
            openclaw.chmod(0o755)
        assert _find_openclaw_binary().endswith("/v22.5.0/bin/openclaw")

    def test_returns_none_when_nowhere(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _find_openclaw_binary() is None


# ---------------------------------------------------------------------------
# _find_free_port
# ---------------------------------------------------------------------------


class TestFindFreePort:
    def test_returns_usable_port(self):
        port = _find_free_port()
        assert 1024 <= port <= 65535


# ---------------------------------------------------------------------------
# _qualified_model_id
# ---------------------------------------------------------------------------


class TestQualifiedModelId:
    def test_prefixes_provider(self):
        assert _qualified_model_id("openai/gpt-5.2") == "switchyard/openai/gpt-5.2"

    def test_strips_leading_slash(self):
        assert _qualified_model_id("/openai/gpt-5.2") == "switchyard/openai/gpt-5.2"

    def test_bare_model_name(self):
        assert _qualified_model_id("kimi-k2.5") == "switchyard/kimi-k2.5"


# ---------------------------------------------------------------------------
# _build_openclaw_config
# ---------------------------------------------------------------------------


class TestBuildOpenclawConfig:
    """Pin the JSON5 schema — typos here silently leave openclaw unable
    to resolve the switchyard provider and the user falls back to
    openclaw's built-in defaults (or fails to start).
    """

    def test_declares_switchyard_provider_with_correct_base_url(self):
        body = _build_openclaw_config(
            port=54321,
            entries=[("openai/gpt-5.2", "GPT 5.2", "desc")],
            primary_model_id="switchyard/openai/gpt-5.2",
        )
        provider = body["models"]["providers"][_PROVIDER_ID]
        assert provider["baseUrl"] == "http://127.0.0.1:54321/v1"
        # The api field must match the openclaw docs' allowed values.
        assert provider["api"] == "openai-completions"
        # apiKey uses ${ENV_VAR} interpolation so the real placeholder
        # never lands in the JSON on disk.
        assert provider["apiKey"] == "${" + _API_KEY_ENV + "}"

    def test_merge_mode_does_not_clobber_default_providers(self):
        body = _build_openclaw_config(
            port=54321,
            entries=[("openai/gpt-5.2", "GPT 5.2", "desc")],
            primary_model_id="switchyard/openai/gpt-5.2",
        )
        assert body["models"]["mode"] == "merge"

    def test_primary_model_is_qualified(self):
        body = _build_openclaw_config(
            port=54321,
            entries=[("openai/gpt-5.2", "GPT 5.2", "desc")],
            primary_model_id="switchyard/openai/gpt-5.2",
        )
        assert body["agents"]["defaults"]["model"]["primary"] == "switchyard/openai/gpt-5.2"

    def test_emits_one_models_entry_per_catalog_row(self):
        body = _build_openclaw_config(
            port=54321,
            entries=[
                ("openai/gpt-5.2", "GPT 5.2", "primary"),
                ("nvidia/moonshotai/kimi-k2.5", "Kimi", "extra"),
            ],
            primary_model_id="switchyard/openai/gpt-5.2",
        )
        models = body["models"]["providers"][_PROVIDER_ID]["models"]
        assert [m["id"] for m in models] == [
            "openai/gpt-5.2",
            "nvidia/moonshotai/kimi-k2.5",
        ]
        assert [m["name"] for m in models] == ["GPT 5.2", "Kimi"]
        # Every entry has the minimal capability set openclaw expects.
        for entry in models:
            assert entry["input"] == ["text"]
            assert entry["contextWindow"] > 0
            assert entry["maxTokens"] > 0

    def test_aliases_use_qualified_model_id_keys(self):
        body = _build_openclaw_config(
            port=54321,
            entries=[("openai/gpt-5.2", "GPT 5.2", "primary")],
            primary_model_id="switchyard/openai/gpt-5.2",
        )
        aliases = body["agents"]["defaults"]["models"]
        # OpenClaw references models by ``provider/model`` everywhere;
        # the alias keys must use the same form.
        assert "switchyard/openai/gpt-5.2" in aliases
        assert aliases["switchyard/openai/gpt-5.2"]["alias"] == "GPT 5.2"


# ---------------------------------------------------------------------------
# _write_openclaw_workspace
# ---------------------------------------------------------------------------


class TestWriteOpenclawWorkspace:
    def test_writes_valid_json5_compatible_json(self, tmp_path, monkeypatch):
        # mkdtemp lands the workspace in tmp_path so the test stays
        # isolated from the system temp dir.
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.tempfile.mkdtemp",
            lambda prefix: str(tmp_path / "ws"),
        )
        (tmp_path / "ws").mkdir()
        workspace = _write_openclaw_workspace(
            port=54321,
            entries=[("openai/gpt-5.2", "GPT 5.2", "desc")],
            primary_model_id="switchyard/openai/gpt-5.2",
        )
        config_path = Path(workspace) / "openclaw.json"
        assert config_path.is_file()
        body = json.loads(config_path.read_text())
        assert body["models"]["providers"]["switchyard"]["api"] == "openai-completions"
        assert (
            body["agents"]["defaults"]["model"]["primary"]
            == "switchyard/openai/gpt-5.2"
        )


# ---------------------------------------------------------------------------
# _openclaw_env
# ---------------------------------------------------------------------------


class TestOpenclawEnv:
    """The env-var contract is the only thing pointing openclaw at our
    transient workspace — drift here silently sends openclaw at the
    user's real ~/.openclaw/ and bypasses the proxy entirely.
    """

    def test_relocates_state_dir(self):
        env = _openclaw_env(workspace="/tmp/switchyard-openclaw-xyz")
        assert env["OPENCLAW_STATE_DIR"] == "/tmp/switchyard-openclaw-xyz"
        assert env["OPENCLAW_HOME"] == "/tmp/switchyard-openclaw-xyz"
        assert (
            env["OPENCLAW_CONFIG_PATH"]
            == "/tmp/switchyard-openclaw-xyz/openclaw.json"
        )

    def test_hides_banner(self):
        env = _openclaw_env(workspace="/tmp/x")
        assert env["OPENCLAW_HIDE_BANNER"] == "1"

    def test_sets_api_key_placeholder(self):
        env = _openclaw_env(workspace="/tmp/x")
        assert env[_API_KEY_ENV] == _API_KEY_PLACEHOLDER

    def test_intake_session_id_propagates(self):
        intake = LaunchIntakeConfig.from_resolved(
            base_url=None, workspace=None, api_key=None,
            app="openclaw", task="developer-session",
            session_id="sess-xyz", target="openclaw",
        )
        env = _openclaw_env(workspace="/tmp/x", intake=intake)
        assert env["SWITCHYARD_SESSION_ID"] == "sess-xyz"

    def test_no_intake_omits_session_id(self):
        env = _openclaw_env(workspace="/tmp/x")
        assert "SWITCHYARD_SESSION_ID" not in env


# ---------------------------------------------------------------------------
# _openclaw_command
# ---------------------------------------------------------------------------


class TestOpenclawCommand:
    def test_prepends_chat_subcommand(self):
        # `openclaw chat` (alias for `openclaw tui --local`) is the
        # interactive subcommand. `openclaw agent` is the one-shot
        # non-interactive form and is reserved for verify.
        cmd = _openclaw_command("/fake/bin/openclaw", [])
        assert cmd == ["/fake/bin/openclaw", "chat"]

    def test_forwards_user_args_after_chat(self):
        cmd = _openclaw_command("/fake/bin/openclaw", ["--thinking", "high"])
        assert cmd == ["/fake/bin/openclaw", "chat", "--thinking", "high"]


# ---------------------------------------------------------------------------
# launch_openclaw — integration (with mocked externals)
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
def _mock_ready_and_tty(monkeypatch, tmp_path):
    """Launcher tests use fake servers, so bypass external runtime side effects."""
    monkeypatch.setattr(
        "switchyard.cli.launchers.openclaw_launcher._wait_ready",
        lambda port, timeout_s=10.0: True,
    )
    monkeypatch.setattr(
        "switchyard.cli.launchers.openclaw_launcher.stdin_is_tty",
        lambda: False,
    )
    monkeypatch.setattr(
        "switchyard.cli.launchers.openclaw_launcher.configure_debug_file_logging",
        lambda display_model: tmp_path / "switchyard.log",
    )


class TestLaunchOpenclaw:
    def test_happy_path(self, monkeypatch, tmp_path):
        fake_server = _make_fake_server(started=True)

        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_openclaw_binary",
            lambda: "/fake/bin/openclaw",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(fake_server),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.tempfile.mkdtemp",
            lambda prefix: str(tmp_path / "ws"),
        )
        (tmp_path / "ws").mkdir()

        captured: dict = {}

        def fake_run(cmd, env, check):
            # Snapshot the config file at the moment openclaw would
            # see it — the launcher's finally block tears down the
            # tempdir before the call returns.
            captured["cmd"] = cmd
            captured["env"] = env
            config_path = Path(env["OPENCLAW_CONFIG_PATH"])
            captured["config"] = json.loads(config_path.read_text())
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        exit_code = launch_openclaw(
            model="nvidia/moonshotai/kimi-k2.5",
            base_url="https://inference-api.nvidia.com/v1",
            api_key="test-key",
            port=None,
            timeout=None,
            openclaw_args=["--verbose"],
        )

        assert exit_code == 0
        cmd = captured["cmd"]
        # argv layout: [openclaw_bin, "chat", *openclaw_args] — `chat`
        # is the interactive local TUI; `agent` is the one-shot form
        # verify uses, not what launch wants.
        assert cmd == ["/fake/bin/openclaw", "chat", "--verbose"]
        # Env-var contract — these are the only knobs pointing openclaw
        # at our proxy. Verify every required one was set.
        env = captured["env"]
        assert env["OPENCLAW_STATE_DIR"] == str(tmp_path / "ws")
        assert env["OPENCLAW_HOME"] == str(tmp_path / "ws")
        assert env["OPENCLAW_CONFIG_PATH"] == str(tmp_path / "ws" / "openclaw.json")
        assert env[_API_KEY_ENV] == _API_KEY_PLACEHOLDER
        # Config file is written before the spawn.
        provider = captured["config"]["models"]["providers"]["switchyard"]
        assert provider["baseUrl"] == "http://127.0.0.1:54321/v1"
        # Tempdir is cleaned up after openclaw exits.
        assert not (tmp_path / "ws" / "openclaw.json").exists()
        # Proxy torn down on return.
        assert fake_server.should_exit is True

    def test_port_override(self, monkeypatch, tmp_path):
        fake_server = _make_fake_server(started=True)

        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_openclaw_binary",
            lambda: "/fake/bin/openclaw",
        )

        # If --port is set, _find_free_port should NOT be called.
        def _should_not_be_called():
            raise AssertionError("_find_free_port called despite --port override")
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_free_port",
            _should_not_be_called,
        )

        captured: dict = {}

        def stub_spawn(switchyard, port):
            captured["port"] = port
            thread = MagicMock()
            return fake_server, thread

        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._spawn_proxy_thread",
            stub_spawn,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.tempfile.mkdtemp",
            lambda prefix: str(tmp_path / "ws2"),
        )
        (tmp_path / "ws2").mkdir()

        def capture_run(cmd, env, check):
            config_path = Path(env["OPENCLAW_CONFIG_PATH"])
            captured["config"] = json.loads(config_path.read_text())
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr(subprocess, "run", capture_run)

        exit_code = launch_openclaw(
            model="m", base_url="u", api_key="k",
            port=4000, timeout=None, openclaw_args=[],
        )

        assert exit_code == 0
        assert captured["port"] == 4000
        assert (
            captured["config"]["models"]["providers"]["switchyard"]["baseUrl"]
            == "http://127.0.0.1:4000/v1"
        )

    def test_missing_binary_returns_127(self, monkeypatch):
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_openclaw_binary",
            lambda: None,
        )

        def _should_not_spawn(*args, **kwargs):
            raise AssertionError("proxy spawned despite missing binary")
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._spawn_proxy_thread",
            _should_not_spawn,
        )

        exit_code = launch_openclaw(
            model="m", base_url="u", api_key="k",
            port=None, timeout=None, openclaw_args=[],
        )
        assert exit_code == _EXIT_BINARY_NOT_FOUND

    def test_ctrl_c_returns_130_and_tears_down(self, monkeypatch, tmp_path):
        fake_server = _make_fake_server(started=True)

        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_openclaw_binary",
            lambda: "/fake/bin/openclaw",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(fake_server),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.tempfile.mkdtemp",
            lambda prefix: str(tmp_path / "ws3"),
        )
        (tmp_path / "ws3").mkdir()

        def raise_sigint(cmd, env, check):
            raise KeyboardInterrupt()
        monkeypatch.setattr(subprocess, "run", raise_sigint)

        exit_code = launch_openclaw(
            model="m", base_url="u", api_key="k",
            port=None, timeout=None, openclaw_args=[],
        )

        assert exit_code == _EXIT_SIGINT
        assert fake_server.should_exit is True

    def test_strips_leading_double_dash_from_openclaw_args(
        self, monkeypatch, tmp_path,
    ):
        """``argparse.REMAINDER`` keeps the ``--`` sentinel in the captured
        list, so ``launch openclaw ... -- --verbose`` produces
        ``['--', '--verbose']``. The handler must strip the leading ``--``
        before forwarding so openclaw doesn't receive a bare ``--`` arg.
        """
        from switchyard.cli.switchyard_cli import (
            _build_parser,
            _cmd_launch_openclaw,
        )

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "openclaw",
            "--model", "nvidia/moonshotai/kimi-k2.5",
            "--api-key", "sk-test",
            "--", "--verbose",
        ])
        assert args.openclaw_args == ["--", "--verbose"]

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
            "switchyard.cli.launchers.openclaw_launcher.launch_openclaw",
            fake_launch,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_openclaw(args)

        assert captured["openclaw_args"] == ["--verbose"]

    def test_no_model_defaults_to_classifier(self, monkeypatch, tmp_path):
        """No ``--model`` no longer errors — it defaults to LLM-classifier
        routing, the same implicit default as ``launch claude`` / ``codex``.
        """
        from switchyard.cli.switchyard_cli import (
            _build_parser,
            _cmd_launch_openclaw,
        )

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))

        def fake_passthrough(**_kwargs):
            raise AssertionError(
                "no --model should default to the classifier launcher, "
                "not single-model passthrough",
            )

        def fake_classifier(**_kwargs):
            raise SystemExit(0)

        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://inference-api.nvidia.com/v1"),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.launch_openclaw",
            fake_passthrough,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher."
            "launch_openclaw_deterministic_routing",
            fake_classifier,
        )
        parser = _build_parser()
        args = parser.parse_args([
            "launch", "openclaw", "--api-key", "sk-test",
        ])
        with pytest.raises(SystemExit):
            _cmd_launch_openclaw(args)

    def test_proxy_never_ready_returns_error(self, monkeypatch, tmp_path):
        fake_server = _make_fake_server(started=False)

        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_openclaw_binary",
            lambda: "/fake/bin/openclaw",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(fake_server),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.tempfile.mkdtemp",
            lambda prefix: str(tmp_path / "ws-fail"),
        )
        (tmp_path / "ws-fail").mkdir()
        # Override the autouse readiness mock to simulate a timeout.
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._wait_ready",
            lambda port, timeout_s=10.0: False,
        )

        def _should_not_run(*args, **kwargs):
            raise AssertionError("openclaw spawned despite proxy not ready")
        monkeypatch.setattr(subprocess, "run", _should_not_run)

        exit_code = launch_openclaw(
            model="m", base_url="u", api_key="k",
            port=None, timeout=None, openclaw_args=[],
        )
        assert exit_code == 1
        assert fake_server.should_exit is True

    def test_uses_openai_translation_chain(self, monkeypatch, tmp_path):
        """OpenClaw speaks OpenAI Chat Completions → backend is an OpenAI-native backend."""
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
                if hasattr(component, "supported_request_types")
            )
            return _make_fake_server(started=True), MagicMock()

        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_openclaw_binary",
            lambda: "/fake/bin/openclaw",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._spawn_proxy_thread",
            stub_spawn,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.tempfile.mkdtemp",
            lambda prefix: str(tmp_path / "ws-chain"),
        )
        (tmp_path / "ws-chain").mkdir()
        monkeypatch.setattr(
            subprocess, "run",
            lambda cmd, env, check: subprocess.CompletedProcess(cmd, returncode=0),
        )

        launch_openclaw(
            model=model,
            base_url="https://inference-api.nvidia.com/v1",
            api_key="sk-test",
            port=None, timeout=None, openclaw_args=[],
        )
        table = captured_switchyard["app"]
        assert table.registered_models() == [model]
        assert table.default_model() == model
        assert [
            item.value
            for item in captured_switchyard["backend"].supported_request_types
        ] == ["openai_chat"]

    def test_tty_mode_runs_openclaw_in_shell_tui(self, monkeypatch, tmp_path):
        """Interactive OpenClaw launches run the child inside a ``ShellTUI``."""
        captured: dict = {}

        class FakeShellTUI:
            def __init__(self, command, env):
                captured["command"] = command
                captured["env"] = env

            def run(self):
                return 0

        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_openclaw_binary",
            lambda: "/fake/bin/openclaw",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(_make_fake_server(started=True)),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.stdin_is_tty",
            lambda: True,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.ShellTUI",
            FakeShellTUI,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.tempfile.mkdtemp",
            lambda prefix: str(tmp_path / "ws-tty"),
        )
        (tmp_path / "ws-tty").mkdir()

        exit_code = launch_openclaw(
            model="nvidia/moonshotai/kimi-k2.5",
            base_url="https://inference-api.nvidia.com/v1",
            api_key="sk-test",
            port=None,
            timeout=None,
            openclaw_args=["--verbose"],
        )

        assert exit_code == 0
        assert captured["command"] == ["/fake/bin/openclaw", "chat", "--verbose"]
        assert captured["env"]["OPENCLAW_STATE_DIR"] == str(tmp_path / "ws-tty")

    def test_smoke_with_routing_profiles_errors_clearly(
        self, monkeypatch, tmp_path,
    ):
        """``--smoke --routing-profiles FILE`` is rejected at the CLI level."""
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_openclaw

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
            "launch", "openclaw", "--smoke", "--api-key", "sk-test",
        ])
        with pytest.raises(SystemExit) as exc_info:
            _cmd_launch_openclaw(args)
        assert "--smoke and --routing-profiles cannot be combined" in str(exc_info.value)

    def test_smoke_without_model_errors_with_helpful_message(
        self, monkeypatch, tmp_path,
    ):
        """``--smoke`` with no model gives a clear error directing to ``--model``."""
        from switchyard.cli.switchyard_cli import _build_parser, _cmd_launch_openclaw

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://inference-api.nvidia.com/v1"),
        )
        parser = _build_parser()
        args = parser.parse_args(["launch", "openclaw", "--smoke", "--api-key", "sk-test"])
        with pytest.raises(SystemExit) as exc_info:
            _cmd_launch_openclaw(args)
        assert "--smoke requires --model" in str(exc_info.value)

    def test_routing_profiles_merges_extras_on_top_of_launcher_table(
        self, monkeypatch, tmp_path,
    ):
        """`--routing-profiles` adds YAML routes on top of the launcher's chain.

        The launcher registers its ``--model`` chain under ``model``; every
        YAML entry is merged on top via
        :meth:`RouteTable.register`. YAML wins on id conflict.
        """
        captured_switchyard: dict = {}

        def stub_spawn(switchyard, port):
            captured_switchyard["app"] = switchyard
            return _make_fake_server(started=True), MagicMock()

        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_openclaw_binary",
            lambda: "/fake/bin/openclaw",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher._spawn_proxy_thread",
            stub_spawn,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.openclaw_launcher.tempfile.mkdtemp",
            lambda prefix: str(tmp_path / "ws-routes"),
        )
        (tmp_path / "ws-routes").mkdir()
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

        def capture_run(cmd, env, check):
            config_path = Path(env["OPENCLAW_CONFIG_PATH"])
            captured_switchyard["config"] = json.loads(config_path.read_text())
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr(subprocess, "run", capture_run)

        launch_openclaw(
            model="primary/model",
            base_url="https://example.invalid/v1",
            api_key="sk-test",
            port=None,
            timeout=None,
            openclaw_args=[],
            routing_profiles=str(yaml_path),
        )

        table = captured_switchyard["app"]
        assert isinstance(table, RouteTable)
        assert table.registered_models() == [
            "primary/model",
            "extras/yaml-only",
        ]
        assert table.default_model() == "primary/model"
        # The YAML-registered model lands in the openclaw.json catalog too.
        config = captured_switchyard["config"]
        catalog_ids = [
            entry["id"]
            for entry in config["models"]["providers"]["switchyard"]["models"]
        ]
        assert "primary/model" in catalog_ids
        assert "extras/yaml-only" in catalog_ids
