# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable coverage for ``docs/getting_started.md``."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
async def rust_server_binary() -> Path:
    """Build and return the Rust server binary exercised by the guide."""
    returncode, stdout, stderr = await _run_command(
        ["cargo", "build", "--quiet", "--locked", "-p", "switchyard-server"],
    )
    assert returncode == 0, f"failed to build switchyard-server:\n{stderr}{stdout}"
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


async def test_rust_server_toml_blocks_validate_with_rust_schema(
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
        returncode, stdout, stderr = await _run_command(
            [rust_server_binary, "--config", config_path, "--dry-run"],
            env=env,
        )
        assert returncode == 0, (
            f"TOML block {index} failed Rust schema validation:\n{stderr}{stdout}"
        )
        assert "server OK: switchyard" in stdout


async def test_rust_server_help_advertises_documented_flags(
    rust_server_binary: Path,
) -> None:
    """Keep the guide's server flags aligned with the Rust CLI."""
    returncode, stdout, stderr = await _run_command(
        [rust_server_binary, "--help"],
    )
    assert returncode == 0, stderr
    for flag in ("--config", "--dry-run", "--host", "--port"):
        assert flag in stdout, f"documented flag {flag} is missing from --help"


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


async def test_rust_server_health_and_models(
    rust_server_binary: Path,
    noop_config: Path,
) -> None:
    """Start the Rust server and exercise the guide's operational endpoints."""
    async with _serve_in_background(rust_server_binary, noop_config) as client:
        health = await client.get("/health")
        assert health.status_code == 200

        models = await client.get("/v1/models")
        assert models.status_code == 200
        assert any(model["id"] == "switchyard/test" for model in models.json()["data"])


async def _run_command(
    command: list[str | os.PathLike[str]],
    *,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a command without blocking the active test event loop."""
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=REPO_ROOT,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode is None:
        raise RuntimeError(f"command did not exit: {command}")
    return (
        process.returncode,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def _read_server_base_url(process: asyncio.subprocess.Process) -> str:
    """Read the ephemeral address from the Rust server startup banner."""
    if process.stdout is None:
        raise RuntimeError("switchyard-server stdout was not captured")

    deadline = asyncio.get_running_loop().time() + STARTUP_TIMEOUT_S
    output: list[str] = []
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(
                "switchyard-server did not report its bound address:\n" + "".join(output)
            )
        try:
            raw_line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
        except TimeoutError as error:
            raise TimeoutError(
                "switchyard-server did not report its bound address:\n" + "".join(output)
            ) from error
        if not raw_line:
            await process.wait()
            raise RuntimeError(
                f"switchyard-server exited early with {process.returncode}:\n"
                + "".join(output)
            )

        line = raw_line.decode(errors="replace")
        output.append(line)
        if line.startswith("  listening: "):
            return line.removeprefix("  listening: ").strip()


async def _wait_for_proxy_ready(
    process: asyncio.subprocess.Process,
    client: httpx.AsyncClient,
) -> None:
    """Wait until the server accepts requests or exits."""
    deadline = asyncio.get_running_loop().time() + STARTUP_TIMEOUT_S
    while asyncio.get_running_loop().time() < deadline:
        if process.returncode is not None:
            raise RuntimeError(
                f"switchyard-server exited early with {process.returncode}"
            )
        try:
            response = await client.get("/health")
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.1)
    raise TimeoutError(
        f"switchyard-server did not become ready within {STARTUP_TIMEOUT_S}s"
    )


@asynccontextmanager
async def _serve_in_background(
    binary: Path,
    config_path: Path,
) -> AsyncIterator[httpx.AsyncClient]:
    """Run the Rust server until the lifecycle assertion completes."""
    process = await asyncio.create_subprocess_exec(
        binary,
        "--config",
        config_path,
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        cwd=REPO_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        base_url = await _read_server_base_url(process)
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=REQUEST_TIMEOUT_S,
        ) as client:
            await _wait_for_proxy_ready(process, client)
            yield client
    finally:
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=TEARDOWN_GRACE_S)
        except TimeoutError:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=TEARDOWN_GRACE_S)


# TODO: Add launcher coverage back when launchers are wired to the Rust server.
# TODO: Add Python snippet coverage back with the supported Rust-backed API.
# TODO: Add YAML route-bundle coverage back only if YAML becomes a supported Rust format.
