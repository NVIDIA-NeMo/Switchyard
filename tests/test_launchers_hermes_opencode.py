# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Hermes and OpenCode launchers."""

import json

from switchyard.cli.launchers.hermes_launcher import (
    _hermes_command,
    _hermes_env,
)
from switchyard.cli.launchers.opencode_launcher import (
    _build_opencode_config,
    _opencode_command,
    _opencode_env,
    _qualified_model_id,
)


def test_hermes_env_points_proxy_and_placeholder_key() -> None:
    env = _hermes_env(4321)
    assert env["OPENROUTER_BASE_URL"] == "http://127.0.0.1:4321/v1"
    assert env["OPENROUTER_API_KEY"] == "switchyard"


def test_hermes_command_defaults_to_interactive_chat(monkeypatch) -> None:
    cmd = _hermes_command("/usr/bin/hermes", [], "my-route")
    assert cmd == [
        "/usr/bin/hermes",
        "chat",
        "--provider",
        "custom",
        "-m",
        "my-route",
    ]


def test_hermes_command_keeps_forwarded_args(monkeypatch) -> None:
    cmd = _hermes_command(
        "/usr/bin/hermes",
        ["--reasoning", "high", "-p", "build"],
        "my-route",
    )
    # Model selector leads; forwarded flags are preserved verbatim afterward.
    assert cmd[:6] == [
        "/usr/bin/hermes",
        "chat",
        "--provider",
        "custom",
        "-m",
        "my-route",
    ]
    assert cmd[6:] == ["--reasoning", "high", "-p", "build"]


def test_opencode_qualified_model_id() -> None:
    assert _qualified_model_id("switchyard") == "switchyard/switchyard"
    assert _qualified_model_id("/route") == "switchyard/route"


def test_opencode_config_points_proxy_and_primary_model() -> None:
    entries = [("switchyard", "Switchyard (Switchyard)", "desc")]
    cfg = json.loads(json.dumps(_build_opencode_config(4321, entries, "switchyard/switchyard")))
    provider = cfg["provider"]["switchyard"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:4321/v1"
    assert provider["options"]["apiKey"] == "switchyard"
    assert "switchyard" in provider["models"]
    assert cfg["model"] == "switchyard/switchyard"


def test_opencode_config_serializes_to_valid_json(tmp_path, monkeypatch) -> None:
    entries = [("switchyard", "Switchyard", "desc")]
    cfg = _build_opencode_config(1234, entries, "switchyard/switchyard")
    payload = json.dumps(cfg)
    parsed = json.loads(payload)
    assert parsed["provider"]["switchyard"]["options"]["apiKey"] == "switchyard"


def test_opencode_env_selects_config_dir() -> None:
    env = _opencode_env("/tmp/switchyard-opencode-abc")
    assert env["OPENCODE_CONFIG_DIR"] == "/tmp/switchyard-opencode-abc"


def test_opencode_command_run_injects_model(monkeypatch) -> None:
    # ``run`` is non-interactive; the model goes after ``-m``.
    cmd = _opencode_command("/usr/bin/opencode", ["run", "--auto", "fix"], "switchyard/r")
    assert cmd[:3] == ["/usr/bin/opencode", "run", "-m"]
    assert "switchyard/r" in cmd
    # ``--auto`` and the message are preserved in order.
    assert cmd[cmd.index("switchyard/r") + 1 :] == ["--auto", "fix"]


def test_opencode_command_defaults_to_tui(monkeypatch) -> None:
    cmd = _opencode_command("/usr/bin/opencode", [], "switchyard/r")
    assert cmd == ["/usr/bin/opencode", "-m", "switchyard/r"]
