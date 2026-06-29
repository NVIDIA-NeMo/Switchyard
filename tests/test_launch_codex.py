# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``switchyard.cli.launchers.codex_cli_launcher``.

Exercises the public ``launch_codex`` entry, the private model-rewrite
request processor, the codex provider override builder, and the
``codex`` binary lookup. Real uvicorn and ``subprocess.run`` are mocked
— these tests don't start a server or spawn a child process.

Mirrors :mod:`tests.test_launch_claude_v2`; shared launcher helpers cover
the proxy process and live stats footer, while harness-specific tests stay
split by child process.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from switchyard.cli.launchers.codex_cli_launcher import (
    _EXIT_BINARY_NOT_FOUND,
    _EXIT_SIGINT,
    _PROVIDER_ID,
    _find_codex_binary,
    _find_free_port,
    _ModelRewriteRequestProcessor,
    _provider_overrides,
    launch_codex,
)
from switchyard.cli.launchers.codex_model_catalog import (
    _build_codex_model_catalog,
    _fallback_codex_model_template,
)
from switchyard.cli.launchers.launch_intake_config import LaunchIntakeConfig
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.route_table import RouteTable
from switchyard_rust.core import ChatRequest

# ---------------------------------------------------------------------------
# _ModelRewriteRequestProcessor
# ---------------------------------------------------------------------------


class TestModelRewriteRequestProcessor:
    """Processor must rewrite ``body['model']`` for every ChatRequest subclass.

    Codex itself only ever sends Responses-shaped requests, but the
    processor is type-agnostic, so we exercise all three subclasses for
    parity with the launch_claude tests.
    """

    async def test_rewrites_responses_request(self):
        proc = _ModelRewriteRequestProcessor("nvidia/moonshotai/kimi-k2.5")
        req = ChatRequest.openai_responses({
            "model": "gpt-5.3-codex",
            "input": "hi",
        })
        out = await proc.process(ProxyContext(), req)
        assert out is req
        assert req.body["model"] == "nvidia/moonshotai/kimi-k2.5"

    async def test_rewrites_openai_chat_request(self):
        proc = _ModelRewriteRequestProcessor("target-model")
        req = ChatRequest.openai_chat({"model": "gpt-4o", "messages": []})
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
# _find_codex_binary
# ---------------------------------------------------------------------------


class TestFindCodexBinary:
    def test_returns_path_hit_when_on_path(self):
        with patch("shutil.which", return_value="/usr/local/bin/codex"):
            assert _find_codex_binary() == "/usr/local/bin/codex"

    def test_falls_back_to_npm_global(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        codex = tmp_path / ".npm-global" / "bin" / "codex"
        codex.parent.mkdir(parents=True)
        codex.write_text("#!/bin/sh\necho codex\n")
        codex.chmod(0o755)
        assert _find_codex_binary() == str(codex)

    def test_falls_back_to_local_bin(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        codex = tmp_path / ".local" / "bin" / "codex"
        codex.parent.mkdir(parents=True)
        codex.write_text("#!/bin/sh\necho codex\n")
        codex.chmod(0o755)
        assert _find_codex_binary() == str(codex)

    def test_returns_none_when_nowhere(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _find_codex_binary() is None


# ---------------------------------------------------------------------------
# _find_free_port
# ---------------------------------------------------------------------------


class TestFindFreePort:
    def test_returns_usable_port(self):
        port = _find_free_port()
        assert 1024 <= port <= 65535


# ---------------------------------------------------------------------------
# _provider_overrides
# ---------------------------------------------------------------------------


class TestProviderOverrides:
    """The codex ``-c`` argv pairs are the contract with codex's TOML
    parser — typos here silently route the user to the built-in OpenAI
    provider instead of our proxy.  These assertions pin the wire shape.
    """

    def test_emits_six_overrides(self):
        out = _provider_overrides(54321)
        # Six ``-c key=value`` pairs = 12 argv entries (alternating).
        assert len(out) == 12
        # Every even index is the literal ``-c`` flag.
        assert all(out[i] == "-c" for i in range(0, len(out), 2))

    def test_activates_switchyard_provider(self):
        out = _provider_overrides(54321)
        assert f'model_provider="{_PROVIDER_ID}"' in out

    def test_base_url_includes_port_and_v1(self):
        out = _provider_overrides(54321)
        expected = f'model_providers.{_PROVIDER_ID}.base_url="http://127.0.0.1:54321/v1"'
        assert expected in out

    def test_wire_api_is_responses(self):
        # Codex speaks /v1/responses; ResponsesEndpoint
        # is what the inbound side mounts.  ``chat`` would route codex
        # to /v1/chat/completions which the proxy also serves but
        # bypasses codex's Responses-specific request shape.
        out = _provider_overrides(54321)
        assert f'model_providers.{_PROVIDER_ID}.wire_api="responses"' in out

    def test_env_key_is_openai_api_key(self):
        # Has to match the env var that ``_supervise_codex`` sets.
        out = _provider_overrides(54321)
        assert f'model_providers.{_PROVIDER_ID}.env_key="OPENAI_API_KEY"' in out

    def test_disables_openai_oauth(self):
        # Without this, codex still attempts ChatGPT OAuth refresh
        # against the built-in openai provider's token store.  The
        # 401 chatter is harmless but spammy; opting out is cleaner.
        out = _provider_overrides(54321)
        assert (
            f"model_providers.{_PROVIDER_ID}.requires_openai_auth=false" in out
        )

    def test_model_catalog_override_is_appended(self):
        out = _provider_overrides(54321, model_catalog_json="/tmp/switchyard models.json")
        assert len(out) == 14
        assert 'model_catalog_json="/tmp/switchyard models.json"' in out

    def test_intake_appends_http_headers(self):
        intake = LaunchIntakeConfig.from_resolved(
            base_url=None,
            workspace=None,
            api_key=None,
            app="codex",
            task="developer-session",
            session_id="sess-xyz",
            target="codex",
        )
        out = _provider_overrides(54321, intake=intake)
        http_overrides = [arg for arg in out if "http_headers" in arg]

        assert len(http_overrides) == 1
        value = http_overrides[0].split("=", 1)[1]
        assert '"x-switchyard-intake-enabled"="true"' in value
        assert '"x-switchyard-intake-app"="codex"' in value
        assert '"proxy_x_session_id"="sess-xyz"' in value


class TestCodexModelCatalog:
    def test_builds_switchyard_only_catalog_from_template(self, monkeypatch):
        template = {
            "slug": "gpt-5.5",
            "display_name": "GPT-5.5",
            "description": "template",
            "default_reasoning_level": "xhigh",
            "supported_reasoning_levels": [
                {"effort": "xhigh", "description": "Extra high reasoning"},
            ],
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": True,
            "priority": 99,
            "availability_nux": {"message": "template nux"},
            "upgrade": {"message": "template upgrade"},
            "base_instructions": "template instructions",
        }
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_model_catalog._load_codex_model_template",
            lambda codex_bin: template,
        )

        catalog = _build_codex_model_catalog(
            "/fake/bin/codex",
            [
                (
                    "switchyard-default-random-12345678",
                    "Switchyard random routing",
                    "Random route.",
                ),
                ("strong/model", "Strong model", "Direct strong route."),
            ],
        )

        models = catalog["models"]
        assert [model["slug"] for model in models] == [
            "switchyard-default-random-12345678",
            "strong/model",
        ]
        assert models[0]["display_name"] == "Switchyard random routing"
        assert models[0]["description"] == "Random route."
        assert models[0]["priority"] == 0
        assert models[0]["availability_nux"] is None
        assert models[0]["upgrade"] is None
        assert models[0]["base_instructions"] == "template instructions"
        assert models[1]["priority"] == 1

# ---------------------------------------------------------------------------
# launch_codex — integration (with mocked externals)
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
        "switchyard.cli.launchers.codex_cli_launcher._wait_ready",
        lambda port, timeout_s=10.0: True,
    )
    monkeypatch.setattr(
        "switchyard.cli.launchers.codex_cli_launcher.stdin_is_tty",
        lambda: False,
    )
    monkeypatch.setattr(
        "switchyard.cli.launchers.codex_cli_launcher.configure_debug_file_logging",
        lambda display_model: tmp_path / "switchyard.log",
    )
    monkeypatch.setattr(
        "switchyard.cli.launchers.codex_model_catalog._load_codex_model_template",
        lambda codex_bin: _fallback_codex_model_template(),
    )
    # Prevent real network calls; simulate OpenAI upstream: Chat Completions probe fails so
    # Responses wins (OpenAI gpt models expose /v1/responses but not /v1/chat/completions
    # as the primary surface in these tests).
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
        lambda **_: True,
    )


class TestLaunchCodex:
    def test_happy_path(self, monkeypatch):
        fake_server = _make_fake_server(started=True)

        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_codex_binary",
            lambda: "/fake/bin/codex",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(fake_server),
        )

        captured: dict = {}

        def fake_run(cmd, env, check):
            captured["cmd"] = cmd
            captured["env"] = env
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        exit_code = launch_codex(
            model="nvidia/moonshotai/kimi-k2.5",
            base_url="https://inference-api.nvidia.com/v1",
            api_key="test-key",
            port=None,
            timeout=None,
            codex_args=["exec", "say hi"],
        )

        assert exit_code == 0
        cmd = captured["cmd"]
        # argv layout: [codex_bin, *provider -c overrides, -m, model, *codex_args]
        assert cmd[0] == "/fake/bin/codex"
        assert cmd[-4:] == ["-m", "nvidia/moonshotai/kimi-k2.5", "exec", "say hi"]
        # The first provider overrides occupy indices 1..13 (12 entries).
        assert cmd[1] == "-c"
        assert any(
            v == 'model_provider="switchyard"' for v in cmd[1:13]
        )
        assert any(
            'base_url="http://127.0.0.1:54321/v1"' in v for v in cmd[1:13]
        )
        assert any(v.startswith("model_catalog_json=") for v in cmd)
        # OPENAI_API_KEY must be set (codex's provider config refuses to
        # start without it). Real upstream key is injected by the proxy
        # at call time — this is just a placeholder.
        assert captured["env"]["OPENAI_API_KEY"] == "switchyard"
        # Proxy torn down on return
        assert fake_server.should_exit is True

    def test_port_override(self, monkeypatch):
        fake_server = _make_fake_server(started=True)

        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_codex_binary",
            lambda: "/fake/bin/codex",
        )

        # If --port is set, _find_free_port should NOT be called
        def _should_not_be_called():
            raise AssertionError("_find_free_port called despite --port override")
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_free_port",
            _should_not_be_called,
        )

        captured: dict = {}

        def stub_spawn(switchyard, port):
            captured["port"] = port
            thread = MagicMock()
            return fake_server, thread

        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._spawn_proxy_thread",
            stub_spawn,
        )

        def fake_run(cmd, env, check):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, returncode=0)
        monkeypatch.setattr(subprocess, "run", fake_run)

        exit_code = launch_codex(
            model="m", base_url="u", api_key="k",
            port=4000, timeout=None, codex_args=[],
        )

        assert exit_code == 0
        assert captured["port"] == 4000
        # Provider override base_url must reflect the chosen port.
        assert any(
            'base_url="http://127.0.0.1:4000/v1"' in v for v in captured["cmd"]
        )

    def test_missing_binary_returns_127(self, monkeypatch):
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_codex_binary",
            lambda: None,
        )

        # If we reach _spawn_proxy_thread, we failed to short-circuit
        def _should_not_spawn(*args, **kwargs):
            raise AssertionError("proxy spawned despite missing binary")
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._spawn_proxy_thread",
            _should_not_spawn,
        )

        exit_code = launch_codex(
            model="m", base_url="u", api_key="k",
            port=None, timeout=None, codex_args=[],
        )
        assert exit_code == _EXIT_BINARY_NOT_FOUND

    def test_ctrl_c_returns_130_and_tears_down(self, monkeypatch):
        fake_server = _make_fake_server(started=True)

        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_codex_binary",
            lambda: "/fake/bin/codex",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(fake_server),
        )

        def raise_sigint(cmd, env, check):
            raise KeyboardInterrupt()
        monkeypatch.setattr(subprocess, "run", raise_sigint)

        exit_code = launch_codex(
            model="m", base_url="u", api_key="k",
            port=None, timeout=None, codex_args=[],
        )

        assert exit_code == _EXIT_SIGINT
        assert fake_server.should_exit is True

    def test_strips_leading_double_dash_from_codex_args(self, monkeypatch, tmp_path):
        """``argparse.REMAINDER`` keeps the ``--`` sentinel in the captured
        list, so ``launch codex ... -- exec hi`` produces
        ``['--', 'exec', 'hi']``. The handler must strip the leading ``--``
        before forwarding so codex doesn't receive a bare ``--`` arg.
        """
        from switchyard.cli.switchyard_cli import (
            _build_parser,
            _cmd_launch_codex,
        )

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "codex",
            "--model", "nvidia/moonshotai/kimi-k2.5",
            "--api-key", "sk-test",
            "--", "exec", "hi",
        ])
        assert args.codex_args == ["--", "exec", "hi"]  # argparse kept '--'

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
            "switchyard.cli.launchers.codex_cli_launcher.launch_codex",
            fake_launch,
        )

        with pytest.raises(SystemExit):
            _cmd_launch_codex(args)

        # Handler stripped the '--' before forwarding.
        assert captured["codex_args"] == ["exec", "hi"]

    def test_no_flags_dispatches_to_deterministic(self, monkeypatch, tmp_path):
        """No ``--model``, no ``--routing-profiles`` → deterministic default.

        Previously ``launch codex`` errored asking for ``--model``; the
        zero-flag default now resolves to the LLM-classifier
        deterministic chain.
        """
        from switchyard.cli.switchyard_cli import (
            _build_parser,
            _cmd_launch_codex,
        )

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://inference-api.nvidia.com/v1"),
        )

        captured: dict = {}

        def fake_launch(**kwargs):
            captured.update(kwargs)
            raise SystemExit(0)

        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher."
            "launch_codex_deterministic_routing",
            fake_launch,
        )

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "codex", "--api-key", "sk-test",
        ])
        with pytest.raises(SystemExit):
            _cmd_launch_codex(args)
        assert captured["config"].preset == "coding_agent_default"

    def test_proxy_never_ready_returns_error(self, monkeypatch):
        fake_server = _make_fake_server(started=False)  # never flips to True

        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_codex_binary",
            lambda: "/fake/bin/codex",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(fake_server),
        )
        # Override the autouse readiness mock to simulate a timeout.
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._wait_ready",
            lambda port, timeout_s=10.0: False,
        )

        def _should_not_run(*args, **kwargs):
            raise AssertionError("codex spawned despite proxy not ready")
        monkeypatch.setattr(subprocess, "run", _should_not_run)

        exit_code = launch_codex(
            model="m", base_url="u", api_key="k",
            port=None, timeout=None, codex_args=[],
        )
        assert exit_code == 1
        assert fake_server.should_exit is True

    def test_uses_openai_translation_chain(self, monkeypatch):
        """Codex always speaks Responses API → backend is an OpenAI-native backend."""
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
            "switchyard.cli.launchers.codex_cli_launcher._find_codex_binary",
            lambda: "/fake/bin/codex",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._spawn_proxy_thread",
            stub_spawn,
        )
        monkeypatch.setattr(
            subprocess, "run",
            lambda cmd, env, check: subprocess.CompletedProcess(cmd, returncode=0),
        )

        launch_codex(
            model=model,
            base_url="https://inference-api.nvidia.com/v1",
            api_key="sk-test",
            port=None, timeout=None, codex_args=[],
        )
        table = captured_switchyard["app"]
        assert table.registered_models() == [model]
        assert table.default_model() == model
        assert [
            item.value
            for item in captured_switchyard["backend"].supported_request_types
        ] == ["openai_responses"]
        assert not any(
            isinstance(component, _ModelRewriteRequestProcessor)
            for component in captured_switchyard["switchyard"].iter_components()
        )

    def test_tty_mode_runs_codex_in_shell_tui(self, monkeypatch):
        """Interactive Codex launches run the child inside a ``ShellTUI``."""
        captured: dict = {}

        class FakeShellTUI:
            def __init__(self, command, env):
                captured["command"] = command
                captured["env"] = env

            def run(self):
                return 0

        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_codex_binary",
            lambda: "/fake/bin/codex",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._spawn_proxy_thread",
            _stub_spawn_proxy(_make_fake_server(started=True)),
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher.stdin_is_tty",
            lambda: True,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher.ShellTUI",
            FakeShellTUI,
        )

        exit_code = launch_codex(
            model="nvidia/moonshotai/kimi-k2.5",
            base_url="https://inference-api.nvidia.com/v1",
            api_key="sk-test",
            port=None,
            timeout=None,
            codex_args=["exec", "hi"],
        )

        assert exit_code == 0
        assert captured["command"][0] == "/fake/bin/codex"
        assert captured["command"][-4:] == [
            "-m",
            "nvidia/moonshotai/kimi-k2.5",
            "exec",
            "hi",
        ]
        assert any(v.startswith("model_catalog_json=") for v in captured["command"])
        assert captured["env"]["OPENAI_API_KEY"] == "switchyard"

    def test_smoke_with_routing_profiles_errors_clearly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """``--smoke --routing-profiles FILE`` is rejected at the CLI level with a
        clear error directing the user to pass ``--model`` instead.
        """
        from switchyard.cli.switchyard_cli import (
            _build_parser,
            _cmd_launch_codex,
        )

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
            "launch", "codex",
            "--smoke", "--api-key", "sk-test",
        ])
        with pytest.raises(SystemExit) as exc_info:
            _cmd_launch_codex(args)
        assert "--smoke and --routing-profiles cannot be combined" in str(exc_info.value)

    def test_smoke_without_model_errors_with_helpful_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """``--smoke`` without ``--model`` gives a clear error directing the
        user to pass ``--model``, not ``--routing-profiles``.
        """
        from switchyard.cli.switchyard_cli import (
            _build_parser,
            _cmd_launch_codex,
        )

        monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "switchyard.cli.launch_command.resolve_launch_connectivity",
            lambda args, **_kw: ("sk-test", "https://inference-api.nvidia.com/v1"),
        )

        parser = _build_parser()
        args = parser.parse_args([
            "launch", "codex", "--smoke", "--api-key", "sk-test",
        ])
        with pytest.raises(SystemExit) as exc_info:
            _cmd_launch_codex(args)
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
            "switchyard.cli.launchers.codex_cli_launcher._find_codex_binary",
            lambda: "/fake/bin/codex",
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._find_free_port",
            lambda: 54321,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._spawn_proxy_thread",
            stub_spawn,
        )
        monkeypatch.setattr(
            "switchyard.cli.launchers.codex_cli_launcher._write_codex_model_catalog",
            lambda _codex_bin, _entries: "/tmp/switchyard-codex-models.json",
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

        launch_codex(
            model="primary/model",
            base_url="https://example.invalid/v1",
            api_key="sk-test",
            port=None,
            timeout=None,
            codex_args=[],
            routing_profiles=str(yaml_path),
        )

        table = captured_switchyard["app"]
        assert isinstance(table, RouteTable)
        assert table.registered_models() == [
            "primary/model",
            "extras/yaml-only",
        ]
        assert table.default_model() == "primary/model"
        table.lookup_switchyard("extras/yaml-only")
        table.lookup_switchyard("primary/model")
