# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable coverage for ``docs/getting_started.mdx``."""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import yaml as pyyaml
from markdown_it import MarkdownIt

from switchyard.cli.launchers.launcher_runtime import (
    find_free_port,
    wait_for_proxy_ready,
)
from switchyard.cli.route_bundle import RouteBundleConfigError, build_route_bundle_table
from switchyard.cli.switchyard_cli import _build_parser

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE_PATH = REPO_ROOT / "docs" / "getting_started.mdx"

#: How long to wait for ``switchyard serve`` to become ready on /health.
STARTUP_TIMEOUT_S: float = 30.0

#: HTTP read timeout for in-test requests to the local proxy.
REQUEST_TIMEOUT_S: float = 10.0

#: Grace period given to ``switchyard serve`` after SIGTERM before SIGKILL.
TEARDOWN_GRACE_S: float = 10.0

#: Final wait after SIGKILL so zombies are reaped before the test ends.
KILL_REAP_S: float = 5.0


@pytest.fixture(scope="module")
def guide_text() -> str:
    return GUIDE_PATH.read_text()


def _code_blocks(text: str, lang: str) -> list[str]:
    # markdown-it-py handles indented fences + trailing whitespace correctly,
    # which a naive ```lang ... ``` regex does not.
    md = MarkdownIt()
    return [
        token.content
        for token in md.parse(text)
        if token.type == "fence" and token.info.strip() == lang
    ]


def _route_bundle_blocks(text: str) -> list[str]:
    blocks = _code_blocks(text, "yaml")
    for block in _code_blocks(text, "bash"):
        lines = block.splitlines()
        for start, line in enumerate(lines):
            if line.startswith("cat > ") and "<<'EOF'" in line:
                for end in range(start + 1, len(lines)):
                    if lines[end] == "EOF":
                        blocks.append("\n".join(lines[start + 1:end]))
                        break
    return blocks


@pytest.mark.parametrize(
    "needle",
    [
        # `switchyard launch claude` is not exercised live — needs the Claude
        # Code binary, which CI doesn't install — so the only check that the
        # guide still names these invocations is this string-match.
        "switchyard launch claude",
        "switchyard --routing-profiles dev.yaml -- launch claude",
        "switchyard launch claude --model openai/gpt-4o",
    ],
)
def test_guide_still_references_launch_claude_invocations(needle: str, guide_text: str) -> None:
    assert needle in guide_text, (
        f"Guide no longer mentions {needle!r}; if intentional, update this test."
    )


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return action.choices  # type: ignore[return-value]


def test_cli_parser_exposes_every_subcommand_the_guide_names() -> None:
    parser = _build_parser()
    subs = _subparsers(parser)
    # Only check subcommands the guide actually invokes — `configure` is not in
    # the guide, so excluding it keeps the test honest about what it covers.
    for cmd in ("serve", "launch"):
        assert cmd in subs, f"top-level `switchyard {cmd}` is documented but missing"
        assert subs[cmd].format_help().strip()

    launch_subs = _subparsers(subs["launch"])
    assert "claude" in launch_subs, "`switchyard launch claude` is documented but missing"
    assert launch_subs["claude"].format_help().strip()


def test_cli_parser_advertises_documented_flags() -> None:
    parser = _build_parser()
    subs = _subparsers(parser)
    # --routing-profiles is a global switchyard flag, not a subcommand flag
    assert "--routing-profiles" in parser.format_help()
    claude_help = _subparsers(subs["launch"])["claude"].format_help()
    assert "--base-url" in claude_help, "`launch claude --base-url` documented but missing"


def test_switchyard_cli_entrypoint_script_is_wired_up() -> None:
    # The in-process parser tests miss broken console-scripts wiring; one
    # subprocess via `python -m` is hermetic (no PATH dependency) and catches
    # that.
    completed = subprocess.run(
        [sys.executable, "-m", "switchyard.cli.switchyard_cli", "--help"],
        capture_output=True, text=True, timeout=STARTUP_TIMEOUT_S, check=False,
    )
    assert completed.returncode == 0, (
        f"switchyard CLI failed to run:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "Switchyard" in completed.stdout


def test_all_yaml_blocks_in_guide_validate_as_route_bundles(
    guide_text: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub the env vars the YAML examples reference so ``${VAR}`` interpolation
    # works on a clean CI runner.
    for key, value in {
        "OPENROUTER_API_KEY": "sk-or-test",  # pragma: allowlist secret
        "OPENAI_API_KEY": "sk-test",
        "ANTHROPIC_API_KEY": "sk-ant-test",
    }.items():
        monkeypatch.setenv(key, value)

    blocks = _route_bundle_blocks(guide_text)
    assert blocks, "Guide unexpectedly has no route-bundle examples"

    for idx, block in enumerate(blocks):
        payload = pyyaml.safe_load(block)
        if not isinstance(payload, dict) or "routes" not in payload:
            continue
        try:
            build_route_bundle_table(payload)
        except RouteBundleConfigError as exc:
            raise AssertionError(
                f"YAML block {idx} in getting_started.mdx failed to parse "
                f"as a route bundle: {exc}\n\nBlock:\n{block}"
            ) from exc


@pytest.fixture
def noop_routes_yaml(tmp_path: Path) -> Path:
    # Same shape as the guide's Step 3 YAML (defaults + a single named route),
    # but `type: noop` so the lifecycle test runs without an upstream.
    path = tmp_path / "routes.yaml"
    path.write_text(
        textwrap.dedent("""\
        defaults:
          api_key: dummy
          base_url: https://upstream.invalid/v1
          format: openai

        routes:
          gpt-4o:
            type: noop
        """)
    )
    return path


def test_step3_and_step4_serve_lifecycle_with_noop(noop_routes_yaml: Path) -> None:
    port = find_free_port()
    with _serve_in_background(noop_routes_yaml, port):
        health_status, health_body = _http_get(f"http://127.0.0.1:{port}/health")
        assert health_status == 200, f"GET /health → {health_status}: {health_body!r}"

        status, body = _http_post_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
            },
        )
        assert status == 200, (
            f"POST /v1/chat/completions → {status}: {body!r}"
        )
        assert b'"choices"' in body, f"Response missing 'choices': {body!r}"


# Snippet execution lives in pytest-markdown-docs; this tripwire guards the
# shape the conftest's fixture depends on.


def test_python_snippet_tripwire(guide_text: str) -> None:
    assert (
        "from switchyard import ChatRequest, PassthroughProfileConfig, ProfileSwitchyard"
        in guide_text
    ), (
        "Python snippet's imports moved — update conftest.py."
    )
    assert "ProfileSwitchyard(PassthroughProfileConfig(" in guide_text, (
        "Python snippet no longer builds a passthrough profile — update the "
        "markdown-docs fixture in conftest.py."
    )


@contextmanager
def _serve_in_background(routes_yaml: Path, port: int) -> Iterator[subprocess.Popen[bytes]]:
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "switchyard.cli.switchyard_cli",
            "--routing-profiles", str(routes_yaml), "--", "serve",
            "--port", str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        if not wait_for_proxy_ready(port, timeout_s=STARTUP_TIMEOUT_S):
            # Distinguish "exited early" from "still starting" so failures
            # point at the right thing.
            if proc.poll() is not None:
                stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                raise RuntimeError(
                    f"`switchyard serve` exited early (code={proc.returncode}):\n{stdout}"
                )
            raise TimeoutError(
                f"`switchyard serve` not ready on port {port} within {STARTUP_TIMEOUT_S}s"
            )
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=TEARDOWN_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=KILL_REAP_S)


def _http_get(url: str) -> tuple[int, bytes]:
    try:
        response = httpx.get(url, timeout=REQUEST_TIMEOUT_S)
    except httpx.HTTPError as exc:
        return 0, str(exc).encode()
    return response.status_code, response.content


def _http_post_json(url: str, payload: dict[str, object]) -> tuple[int, bytes]:
    try:
        response = httpx.post(url, json=payload, timeout=STARTUP_TIMEOUT_S)
    except httpx.HTTPError as exc:
        return 0, str(exc).encode()
    return response.status_code, response.content
