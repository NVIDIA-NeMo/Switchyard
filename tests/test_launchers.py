# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the minimal coding-agent launcher surface."""

import argparse
from pathlib import Path

import pytest

from switchyard.cli.launch_command import _config_path
from switchyard.cli.launchers.claude_code_launcher import _claude_env
from switchyard.cli.launchers.codex_cli_launcher import _codex_env, _provider_overrides
from switchyard.cli.launchers.codex_model_catalog import _build_codex_model_catalog
from switchyard.cli.launchers.native_server import NativeServer
from switchyard.cli.switchyard_cli import _build_parser


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return action.choices  # type: ignore[return-value]


def test_cli_exposes_only_launch() -> None:
    assert set(_subparsers(_build_parser())) == {"launch"}


@pytest.mark.parametrize("agent", ["claude", "codex", "openclaw"])
def test_launcher_surface_is_model_config_and_forwarded_args(agent: str) -> None:
    launch = _subparsers(_build_parser())["launch"]
    parser = _subparsers(launch)[agent]
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option != "--help"
    }
    assert options == {"-h", "--model", "--config"}

    args = parser.parse_args(["--model", "switchyard", "--", "--version"])
    assert args.model == "switchyard"
    assert args.config is None
    assert vars(args)[f"{agent}_args"] == ["--", "--version"]


def test_default_config_is_packaged_openrouter_deployment() -> None:
    config = _config_path(None)
    assert config.name == "openrouter.toml"
    assert "OPENROUTER_API_KEY" in config.read_text()


def test_claude_env_preserves_small_fast_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_SMALL_FAST_MODEL", "background-route")

    env = _claude_env(4321, "agent-route")

    assert env["ANTHROPIC_MODEL"] == "agent-route"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "background-route"


def test_forward_auth_does_not_replace_the_agent_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "inherited-claude-token")
    monkeypatch.setenv("OPENAI_API_KEY", "inherited-openai-key")

    claude_auth = _claude_env(4321, "agent-route", use_anthropic_auth=True)
    claude_local = _claude_env(4321, "agent-route")
    codex_auth = _codex_env(use_openai_auth=True)
    codex_local = _codex_env()
    codex_auth_config = " ".join(_provider_overrides(4321, use_openai_auth=True))

    assert "ANTHROPIC_AUTH_TOKEN" not in claude_auth
    assert claude_local["ANTHROPIC_AUTH_TOKEN"] == "switchyard"
    assert codex_auth["OPENAI_API_KEY"] == "inherited-openai-key"
    assert codex_local["OPENAI_API_KEY"] == "switchyard"
    assert "requires_openai_auth=true" in codex_auth_config
    assert "env_key" not in codex_auth_config


def test_native_server_passes_config_directly_to_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "routes.toml"
    config.write_text("schema_version = 1\n")
    captured: dict[str, object] = {}

    class FakeServer:
        port = 4321
        base_url = "http://127.0.0.1:4321"

        def __init__(self, path: Path, port: int) -> None:
            captured["path"] = path
            captured["port"] = port

        def close(self) -> None:
            captured["closed"] = True

        def caller_auth_kind(self, model: str) -> str | None:
            captured["model"] = model
            return "anthropic"

        def input_modalities(self, model: str) -> list[str]:
            captured["modalities_model"] = model
            return ["text", "image"]

    import switchyard_rust.server

    monkeypatch.setattr(switchyard_rust.server, "Server", FakeServer)
    server = NativeServer(config)
    assert server.caller_auth_kind("switchyard/route") == "anthropic"
    assert server.input_modalities("switchyard/route") == ["text", "image"]
    server.close()

    assert captured == {
        "path": config,
        "port": 0,
        "model": "switchyard/route",
        "modalities_model": "switchyard/route",
        "closed": True,
    }
    assert server.port == 4321
    assert config.exists()


def test_codex_catalog_uses_route_modalities_and_defaults_undeclared_to_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import switchyard.cli.launchers.codex_model_catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "_load_codex_model_template",
        lambda _codex_bin: {"input_modalities": ["text", "image", "audio"]},
    )
    entries = [("switchyard/route", "Route", "Test route")]

    discovered = _build_codex_model_catalog(
        "codex",
        entries,
        input_modalities_by_model={"switchyard/route": ["text", "image"]},
    )
    undeclared = _build_codex_model_catalog("codex", entries)

    assert discovered["models"][0]["input_modalities"] == ["text", "image"]
    assert undeclared["models"][0]["input_modalities"] == ["text"]


def test_codex_launcher_discovers_modalities_for_every_catalog_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import switchyard.cli.launchers.codex_cli_launcher as launcher

    class FakeServer:
        port = 4321
        stats = object()

        def __init__(self) -> None:
            self.queried_models: list[str] = []
            self.closed = False

        def caller_auth_kind(self, model: str) -> str | None:
            assert model == "switchyard/text"
            return None

        def input_modalities(self, model: str) -> list[str]:
            self.queried_models.append(model)
            return {
                "switchyard/text": ["text"],
                "switchyard/vision": ["text", "image"],
            }[model]

        def close(self) -> None:
            self.closed = True

    server = FakeServer()
    captured_modalities: dict[str, list[str]] = {}

    def write_catalog(
        _codex_bin: str,
        _entries: object,
        input_modalities_by_model: dict[str, list[str]] | None = None,
    ) -> None:
        captured_modalities.update(input_modalities_by_model or {})

    monkeypatch.setattr(launcher, "_find_codex_binary", lambda: "codex")
    monkeypatch.setattr(launcher, "silence_launch_loggers", lambda **_kwargs: None)
    monkeypatch.setattr(
        launcher,
        "configure_debug_file_logging",
        lambda **_kwargs: tmp_path / "switchyard.log",
    )
    monkeypatch.setattr(launcher, "_start_native_server", lambda _config: server)
    monkeypatch.setattr(launcher, "_write_codex_model_catalog", write_catalog)
    monkeypatch.setattr(launcher, "_wait_ready", lambda _port: True)
    monkeypatch.setattr(launcher, "print_ready_banner", lambda **_kwargs: None)
    monkeypatch.setattr(launcher, "stdin_is_tty", lambda: False)
    monkeypatch.setattr(launcher, "_supervise_codex", lambda _command, _env: 0)
    monkeypatch.setattr(launcher, "print_session_summary", lambda _stats: None)
    monkeypatch.setattr(launcher, "_remove_codex_model_catalog", lambda _path: None)

    result = launcher._run_codex_with_switchyard(
        tmp_path / "routes.toml",
        display_model="switchyard/text",
        codex_args=[],
        codex_model_catalog=[
            ("switchyard/text", "Text", "Text route"),
            ("switchyard/vision", "Vision", "Vision route"),
        ],
    )

    assert result == 0
    assert server.queried_models == ["switchyard/text", "switchyard/vision"]
    assert captured_modalities == {
        "switchyard/text": ["text"],
        "switchyard/vision": ["text", "image"],
    }
    assert server.closed


def test_native_server_exposes_route_derived_modalities(tmp_path: Path) -> None:
    config = tmp_path / "modalities.toml"
    config.write_text(
        """
schema_version = 1

[llm_clients.local]
format = "openai_chat"
base_url = "http://127.0.0.1:9/v1"

[targets.text]
id = "model/text"
llm_client = "local"
input_modalities = ["text"]

[targets.vision]
id = "model/vision"
llm_client = "local"
input_modalities = ["image", "text"]

[targets.legacy]
id = "model/legacy"
llm_client = "local"

[routes.multimodal]
id = "switchyard/multimodal"
type = "random"
targets = ["text", "vision"]

[routes.legacy]
id = "switchyard/legacy"
type = "passthrough"
target = "legacy"
"""
    )
    server = NativeServer(config)
    try:
        assert server.input_modalities("switchyard/multimodal") == ["text", "image"]
        assert server.input_modalities("switchyard/legacy") == ["text"]
    finally:
        server.close()


def test_missing_explicit_config_is_a_cli_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(SystemExit, match="config file not found"):
        _config_path(str(missing))
