# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable coverage for ``docs/getting_started.md``."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from markdown_it import MarkdownIt

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE_PATH = REPO_ROOT / "docs" / "getting_started.md"
STARTUP_TIMEOUT_S = 30.0
REQUEST_TIMEOUT_S = 1.0
TEARDOWN_GRACE_S = 10.0


@pytest.fixture(scope="module")
def guide_text() -> str:
    """Return the published Getting Started source."""
    return GUIDE_PATH.read_text()


@pytest.fixture(scope="session")
def rust_server_binary() -> Path:
    """Build and return the Rust server binary exercised by the guide."""
    completed = subprocess.run(
        ["cargo", "build", "--quiet", "--locked", "-p", "switchyard-server"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"failed to build switchyard-server:\n{completed.stderr}{completed.stdout}"
    )
    binary = REPO_ROOT / "target" / "debug" / "switchyard-server"
    assert binary.is_file(), f"cargo did not create {binary}"
    return binary


def _code_blocks(text: str, lang: str) -> list[str]:
    """Extract fenced Markdown blocks for one language."""
    md = MarkdownIt()
    return [
        token.content
        for token in md.parse(text)
        if token.type == "fence" and token.info.strip() == lang
    ]


def test_rust_server_toml_blocks_validate_with_rust_schema(
    guide_text: str,
    rust_server_binary: Path,
    tmp_path: Path,
) -> None:
    """Validate every documented server config with the production Rust loader."""
    configs = [
        block
        for block in _code_blocks(guide_text, "toml")
        if "schema_version" in block
        and "[llm_clients." in block
        and "[targets." in block
        and "[routes." in block
    ]
    assert configs, "no Rust server TOML config found in the guide"

    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = "getting-started-test-key"
    for index, config in enumerate(configs):
        config_path = tmp_path / f"getting-started-{index}.toml"
        config_path.write_text(config)
        completed = subprocess.run(
            [rust_server_binary, "--config", config_path, "--dry-run"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            f"TOML block {index} failed Rust schema validation:\n"
            f"{completed.stderr}{completed.stdout}"
        )
        assert "server OK: switchyard" in completed.stdout


def test_rust_server_help_advertises_documented_flags(rust_server_binary: Path) -> None:
    """Keep the guide's server flags aligned with the Rust CLI."""
    completed = subprocess.run(
        [rust_server_binary, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for flag in ("--config", "--dry-run", "--host", "--port"):
        assert flag in completed.stdout, f"documented flag {flag} is missing from --help"


def test_guide_uses_the_rust_server_flow(guide_text: str) -> None:
    """Prevent legacy Python install, configure, and serve commands from returning."""
    assert "cargo build --locked --release -p switchyard-server" in guide_text
    assert "./target/release/switchyard-server --config routes.toml" in guide_text
    for legacy_command in (
        'pip install "nemo-switchyard',
        "switchyard configure",
        "switchyard serve",
        "--routing-profiles",
        "routes.yaml",
    ):
        assert legacy_command not in guide_text


@pytest.fixture
def noop_config(tmp_path: Path) -> Path:
    """Create a network-free Rust route for exercising the documented endpoints."""
    config_path = tmp_path / "routes.toml"
    config_path.write_text(
        """\
schema_version = 1

[llm_clients]

[targets]

[routes.health_check]
id = "switchyard/test"
type = "noop"
"""
    )
    return config_path


def test_rust_server_health_and_models(
    rust_server_binary: Path,
    noop_config: Path,
) -> None:
    """Start the Rust server and exercise the guide's operational endpoints."""
    port = _find_free_port()
    with _serve_in_background(rust_server_binary, noop_config, port):
        health = httpx.get(
            f"http://127.0.0.1:{port}/health",
            timeout=REQUEST_TIMEOUT_S,
        )
        assert health.status_code == 200

        models = httpx.get(
            f"http://127.0.0.1:{port}/v1/models",
            timeout=REQUEST_TIMEOUT_S,
        )
        assert models.status_code == 200
        assert any(model["id"] == "switchyard/test" for model in models.json()["data"])


def _find_free_port() -> int:
    """Reserve an available loopback port for the next server process."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@contextmanager
def _serve_in_background(
    binary: Path,
    config_path: Path,
    port: int,
) -> Iterator[subprocess.Popen[bytes]]:
    """Run the Rust server until the lifecycle assertion completes."""
    process = subprocess.Popen(
        [
            binary,
            "--config",
            config_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read().decode(errors="replace") if process.stdout else ""
                raise RuntimeError(
                    f"switchyard-server exited early with {process.returncode}:\n{output}"
                )
            try:
                response = httpx.get(
                    f"http://127.0.0.1:{port}/health",
                    timeout=REQUEST_TIMEOUT_S,
                )
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            raise TimeoutError(
                f"switchyard-server did not become ready within {STARTUP_TIMEOUT_S}s"
            )
        yield process
    finally:
        process.terminate()
        try:
            process.wait(timeout=TEARDOWN_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=TEARDOWN_GRACE_S)


# TODO: Add launcher coverage back when launchers are wired to the Rust server.
# TODO: Add Python snippet coverage back with the supported Rust-backed API.
# TODO: Add YAML route-bundle coverage back only if YAML becomes a supported Rust format.
