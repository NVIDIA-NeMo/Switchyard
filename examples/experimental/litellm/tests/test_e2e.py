# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from switchyard_litellm import LiteLLMSyClient

from switchyard.libsy import LlmTarget, algorithms

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def litellm_base_url() -> Iterator[str]:
    if os.environ.get("SWITCHYARD_LITELLM_E2E") != "1":
        pytest.skip("SWITCHYARD_LITELLM_E2E=1 is required for paid E2E tests")
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is required for paid E2E tests")
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for paid E2E tests")
    subprocess.run(
        ["docker", "compose", "version"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    port = _free_port()
    project = f"switchyard-litellm-e2e-{os.getpid()}"
    network = f"{project}-network"
    env = {
        **os.environ,
        "LITELLM_PORT": str(port),
        "LITELLM_NETWORK": network,
    }
    compose = ["docker", "compose", "--project-name", project]
    try:
        subprocess.run(
            [*compose, "up", "-d", "--wait"],
            cwd=PACKAGE_ROOT,
            env=env,
            check=True,
        )
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        subprocess.run(
            [*compose, "down", "--volumes", "--remove-orphans"],
            cwd=PACKAGE_ROOT,
            env=env,
            check=True,
        )


def test_paid_e2e_requires_explicit_spend_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("SWITCHYARD_LITELLM_E2E", raising=False)

    def fail_if_docker_is_checked(_: str) -> str | None:
        pytest.fail("Docker must not be checked without the paid E2E opt-in")

    monkeypatch.setattr(shutil, "which", fail_if_docker_is_checked)
    fixture = litellm_base_url.__wrapped__()
    with pytest.raises(pytest.skip.Exception, match="SWITCHYARD_LITELLM_E2E"):
        next(fixture)


def test_paid_e2e_requires_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SWITCHYARD_LITELLM_E2E", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def fail_if_docker_is_checked(_: str) -> str | None:
        pytest.fail("Docker must not be checked without OPENROUTER_API_KEY")

    monkeypatch.setattr(shutil, "which", fail_if_docker_is_checked)
    fixture = litellm_base_url.__wrapped__()
    with pytest.raises(pytest.skip.Exception, match="OPENROUTER_API_KEY"):
        next(fixture)


def _request() -> dict[str, object]:
    return {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Reply with a short greeting for a Switchyard E2E test.",
                    }
                ],
            }
        ],
        "reasoning": {"effort": "low"},
        "output": {"max_output_tokens": 128},
    }


@pytest.mark.e2e
async def test_random_router_calls_both_real_openrouter_models(
    litellm_base_url: str,
) -> None:
    strong_client = LiteLLMSyClient("strong", base_url=litellm_base_url)
    fast_client = LiteLLMSyClient("fast", base_url=litellm_base_url)
    targets = [
        LlmTarget("strong", strong_client),
        LlmTarget("fast", fast_client),
    ]
    try:
        strong_trace, strong_response = await algorithms.random(
            targets, weights=[1, 0], seed=42
        ).run(_request())
        fast_trace, fast_response = await algorithms.random(
            targets, weights=[0, 1], seed=42
        ).run(_request())
    finally:
        await strong_client.aclose()
        await fast_client.aclose()

    assert [item["selected_model"] for item in strong_trace] == ["strong"]
    assert [item["selected_model"] for item in fast_trace] == ["fast"]
    for response in (strong_response, fast_response):
        text = response["outputs"][0]["content"][0]["text"]
        assert isinstance(text, str)
        assert text.strip()
