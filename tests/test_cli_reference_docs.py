# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLI reference documentation drift."""

import argparse
from collections.abc import Callable
from pathlib import Path

from markdown_it import MarkdownIt

from switchyard.cli.switchyard_cli import _build_parser

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_REFERENCE = REPO_ROOT / "docs" / "cli_reference.md"
_MARKDOWN = MarkdownIt()
_AGENT_SPECIFIC_CONFIGURE_OPTIONS = {
    "--claude-api-key",
    "--claude-base-url",
    "--claude-model",
    "--codex-api-key",
    "--codex-base-url",
    "--codex-model",
    "--openclaw-api-key",
    "--openclaw-base-url",
    "--openclaw-model",
}


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return action.choices  # type: ignore[return-value]


def _markdown_section(text: str, heading: str) -> str:
    tokens = _MARKDOWN.parse(text)
    lines = text.splitlines()
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.tag != "h2":
            continue
        title = tokens[index + 1].content
        if title.strip("`") != heading:
            continue
        start = token.map[0] if token.map else 0
        end = len(lines)
        for next_token in tokens[index + 1:]:
            if next_token.type == "heading_open" and next_token.tag == "h2" and next_token.map:
                end = next_token.map[0]
                break
        return "\n".join(lines[start:end])
    raise AssertionError(f"docs/cli_reference.md missing section {heading!r}")


def _long_options(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }


def test_cli_reference_documents_every_serve_flag() -> None:
    parser = _build_parser()
    serve = _subparsers(parser)["serve"]
    serve_section = _markdown_section(CLI_REFERENCE.read_text(), "switchyard serve")

    expected = _long_options(serve) | {"--routing-profiles"}
    missing = sorted(flag for flag in expected if flag not in serve_section)

    assert not missing, (
        "docs/cli_reference.md serve section missing flags: " + ", ".join(missing)
    )


def test_cli_reference_documents_supported_configure_flags() -> None:
    parser = _build_parser()
    configure = _subparsers(parser)["configure"]
    configure_section = _markdown_section(
        CLI_REFERENCE.read_text(),
        "switchyard configure",
    )

    expected = (
        _long_options(configure) - _AGENT_SPECIFIC_CONFIGURE_OPTIONS
    ) | {"--routing-profiles"}
    missing = sorted(flag for flag in expected if flag not in configure_section)

    assert not missing, (
        "docs/cli_reference.md configure section missing flags: "
        + ", ".join(missing)
    )


def test_cli_reference_omits_unsupported_skill_distillation_flags() -> None:
    configure_section = _markdown_section(
        CLI_REFERENCE.read_text(),
        "switchyard configure",
    )
    unsupported_flags = {
        "--skill-session-store",
        "--skill-trigger",
        "--skill-mount",
        "--skill-stage-only",
        "--skill-lookback-sessions",
    }

    stale = sorted(flag for flag in unsupported_flags if flag in configure_section)

    assert not stale, (
        "docs/cli_reference.md configure section documents unsupported flags: "
        + ", ".join(stale)
    )


def test_cli_reference_serve_overview_points_to_route_bundle() -> None:
    serve_row = next(
        line
        for line in CLI_REFERENCE.read_text().splitlines()
        if line.startswith("| [`serve`](#switchyard-serve)")
    )

    assert "--routing-profiles" in serve_row
    assert "`serve --config`" not in serve_row
    assert "inbound format" not in serve_row.lower()


def test_cli_reference_marks_serve_inbound_as_noop() -> None:
    serve_section = _markdown_section(CLI_REFERENCE.read_text(), "switchyard serve").lower()

    assert "`--inbound format`" in serve_section
    assert "no-op" in serve_section
    assert "compat" in serve_section


def test_cli_reference_documents_serve_intake_flags_once() -> None:
    doc = CLI_REFERENCE.read_text()

    assert "### Intake sink (serve)" in doc
    assert "`--intake-enabled`" in doc
    assert "`--enable-intake`" in doc
    assert doc.count("`--intake-base-url URL`") == 1
    assert doc.count("`--intake-workspace NAME`") == 1
    assert doc.count("`--intake-api-key VALUE`") == 1
    assert doc.count("`--intake-target-url URL`") == 1


def _public_cli_doc_paths() -> list[Path]:
    return [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]


def _markdown_command_lines(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    matches: list[tuple[int, str]] = []
    for token in _MARKDOWN.parse(text):
        if token.type == "inline" and token.children and token.map:
            for child in token.children:
                if child.type == "code_inline" and child.content.strip():
                    matches.append((token.map[0] + 1, child.content.strip()))
            continue
        if token.type not in {"fence", "code_block"} or token.map is None:
            continue
        line_base = token.map[0] + (2 if token.type == "fence" else 1)
        for offset, line in enumerate(token.content.splitlines()):
            stripped = line.strip()
            if stripped:
                matches.append((line_base + offset, stripped))
    return matches


def _command_words(line: str) -> list[str]:
    return line.replace("`", "").replace("\\", " ").split()


def _has_stale_routing_profiles(line: str) -> bool:
    words = _command_words(line)
    for index, word in enumerate(words):
        if word != "switchyard" or index + 2 >= len(words):
            continue
        if words[index + 1] in {"serve", "configure"}:
            if "--routing-profiles" in words[index + 2:]:
                return True
    return False


def _has_serve_api_key(line: str) -> bool:
    words = _command_words(line)
    for index, word in enumerate(words):
        if word == "switchyard" and words[index + 1:index + 2] == ["serve"]:
            if "--api-key" in words[index + 2:]:
                return True
    return False


def _matching_markdown_lines(predicate: Callable[[str], bool]) -> list[str]:
    matches: list[str] = []
    for path in _public_cli_doc_paths():
        for lineno, line in _markdown_command_lines(path):
            if predicate(line):
                matches.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line}")
    return matches


def test_cli_docs_do_not_use_stale_flag_placements() -> None:
    stale_checks = {
        "--routing-profiles must be a global switchyard flag": _has_stale_routing_profiles,
        "serve does not accept --api-key": _has_serve_api_key,
    }

    failures = []
    for reason, predicate in stale_checks.items():
        matches = _matching_markdown_lines(predicate)
        if matches:
            failures.append(reason + ":\n" + "\n".join(matches))

    assert not failures, "\n\n".join(failures)


# TODO: Add agent-specific command-reference coverage when that documentation returns.
