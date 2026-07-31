# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI reference drift tests."""

import argparse
import subprocess
from pathlib import Path

from switchyard.cli.switchyard_cli import _build_parser

CLI_REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "cli_reference.md"


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return action.choices  # type: ignore[return-value]


def _long_options(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }


def test_reference_documents_launcher_before_server() -> None:
    text = CLI_REFERENCE.read_text()
    launcher = text.index("## Launcher Path: `switchyard launch`")
    server = text.index("## Server Path: `switchyard-server`")
    related = text.index("## Related Documentation")

    assert launcher < server < related
    assert "switchyard serve" not in text
    assert "switchyard configure" not in text
    assert "switchyard verify" not in text


def test_reference_lists_launcher_contract() -> None:
    launch = _subparsers(_build_parser())["launch"]
    text = CLI_REFERENCE.read_text()
    for parser in _subparsers(launch).values():
        assert _long_options(parser) == {"--model", "--config"}
    assert "--model" in text
    assert "--config" in text


def test_reference_lists_server_contract() -> None:
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "-p",
            "switchyard-server",
            "--",
            "--help",
        ],
        cwd=CLI_REFERENCE.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    text = CLI_REFERENCE.read_text()
    for flag in (
        "--config",
        "--host",
        "--port",
        "--backlog",
        "--dry-run",
        "--tls-cert",
        "--tls-key",
        "--help",
        "--version",
    ):
        assert flag in result.stdout
        assert flag in text
