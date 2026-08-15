# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Windows compatibility tests for the Codex launcher."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from switchyard.cli.launchers import codex_cli_launcher, codex_model_catalog


def test_windows_finder_prefers_a_launchable_codex_shim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    codex_shim = tmp_path / "codex"
    cmd_shim = tmp_path / "codex.cmd"
    codex_shim.write_text("#!/bin/sh\n")
    cmd_shim.write_text("@node codex.js\n")

    def fake_which(executable: str) -> str | None:
        calls.append(executable)
        return str(codex_shim)

    monkeypatch.setattr(codex_cli_launcher, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(codex_cli_launcher.shutil, "which", fake_which)

    assert codex_cli_launcher._find_codex_binary() == str(cmd_shim)
    assert calls == ["codex"]


def test_posix_finder_keeps_the_existing_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_which(executable: str) -> str | None:
        calls.append(executable)
        return "/usr/local/bin/codex"

    monkeypatch.setattr(codex_cli_launcher, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(codex_cli_launcher.shutil, "which", fake_which)

    assert codex_cli_launcher._find_codex_binary() == "/usr/local/bin/codex"
    assert calls == ["codex"]


def test_windows_finder_prefers_cmd_for_fallback_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    codex_shim = tmp_path / ".npm-global" / "bin" / "codex"
    cmd_shim = Path(f"{codex_shim}.cmd")
    codex_shim.parent.mkdir(parents=True)
    codex_shim.write_text("#!/bin/sh\n")
    cmd_shim.write_text("@node codex.js\n")

    monkeypatch.setattr(codex_cli_launcher.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        codex_cli_launcher,
        "os",
        SimpleNamespace(name="nt", access=lambda path, mode: True, X_OK=1),
    )
    monkeypatch.setattr(codex_cli_launcher.shutil, "which", lambda executable: None)

    assert codex_cli_launcher._find_codex_binary() == str(cmd_shim)


def test_windows_tty_uses_the_plain_supervisor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeServer:
        port = 4321
        stats = object()

        def caller_auth_kind(self, model: str) -> None:
            captured["model"] = model

        def close(self) -> None:
            captured["closed"] = True

    server = FakeServer()
    monkeypatch.setattr(
        codex_cli_launcher,
        "os",
        SimpleNamespace(name="nt", environ={}),
    )
    monkeypatch.setattr(codex_cli_launcher, "_find_codex_binary", lambda: "codex.cmd")
    monkeypatch.setattr(codex_cli_launcher, "_start_native_server", lambda config: server)
    monkeypatch.setattr(codex_cli_launcher, "_wait_ready", lambda port: True)
    monkeypatch.setattr(codex_cli_launcher, "stdin_is_tty", lambda: True)
    monkeypatch.setattr(codex_cli_launcher, "silence_launch_loggers", lambda **kwargs: None)
    monkeypatch.setattr(
        codex_cli_launcher,
        "configure_debug_file_logging",
        lambda **kwargs: tmp_path / "switchyard.log",
    )
    monkeypatch.setattr(codex_cli_launcher, "print_ready_banner", lambda **kwargs: None)
    monkeypatch.setattr(codex_cli_launcher, "print_session_summary", lambda stats: None)
    monkeypatch.setattr(
        codex_cli_launcher,
        "_write_codex_model_catalog",
        lambda codex_bin, catalog: None,
    )
    monkeypatch.setattr(
        codex_cli_launcher,
        "_remove_codex_model_catalog",
        lambda path: captured.setdefault("removed", path),
    )

    def fake_supervise(command: list[str], env: dict[str, str]) -> int:
        captured["command"] = command
        captured["env"] = env
        return 0

    monkeypatch.setattr(codex_cli_launcher, "_supervise_codex", fake_supervise)

    result = codex_cli_launcher._run_codex_with_switchyard(
        tmp_path / "routes.toml",
        "windows-route",
        [],
        [],
    )

    assert result == 0
    assert captured["command"][0] == "codex.cmd"  # type: ignore[index]
    assert captured["model"] == "windows-route"
    assert captured["closed"] is True
    assert captured["removed"] is None


def test_codex_env_bypasses_proxies_for_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        codex_cli_launcher,
        "os",
        SimpleNamespace(
            environ={
                "NO_PROXY": "uppercase.example,shared.example",
                "no_proxy": "lowercase.example,shared.example",
            }
        ),
    )

    env = codex_cli_launcher._codex_env()

    assert env["NO_PROXY"] == (
        "uppercase.example,shared.example,lowercase.example,127.0.0.1,localhost"
    )
    assert env["no_proxy"] == env["NO_PROXY"]


def test_codex_catalog_is_decoded_as_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_check_output(command: list[str], **kwargs: object) -> str:
        captured["command"] = command
        captured.update(kwargs)
        return json.dumps({"models": [{"slug": "gpt-5.4"}]})

    monkeypatch.setattr(
        codex_model_catalog.subprocess,
        "check_output",
        fake_check_output,
    )

    template = codex_model_catalog._load_codex_model_template("codex")

    assert template["slug"] == "gpt-5.4"
    assert captured["encoding"] == "utf-8"
    assert "text" not in captured
