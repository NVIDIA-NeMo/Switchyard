# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the minimal coding-agent launcher surface."""

import argparse
from pathlib import Path

import pytest

from switchyard.cli.launch_command import _config_path
from switchyard.cli.launchers.claude_code_launcher import _claude_env
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

        def __init__(self, path: Path, *, port: int) -> None:
            captured["path"] = path
            captured["port"] = port

        def close(self) -> None:
            captured["closed"] = True

    import switchyard_rust.server

    monkeypatch.setattr(switchyard_rust.server, "Server", FakeServer)
    server = NativeServer(config)
    server.close()

    assert captured == {"path": config, "port": 0, "closed": True}
    assert server.port == 4321
    assert config.exists()


def test_missing_explicit_config_is_a_cli_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(SystemExit, match="config file not found"):
        _config_path(str(missing))
