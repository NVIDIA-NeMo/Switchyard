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

from switchyard.libsy import Algorithm, Decision, Step, algorithms

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


def _request(*, critical_error: bool = False) -> dict[str, object]:
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Reply with a short greeting for a Switchyard E2E test.",
                }
            ],
        }
    ]
    if critical_error:
        messages.extend([
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "Bash",
                        "arguments": {"command": "pytest"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_1",
                        "content": [
                            {
                                "type": "text",
                                "text": "fatal runtime error: out of memory",
                            }
                        ],
                        "is_error": True,
                    }
                ],
            },
        ])
    return {
        "model": "auto",
        "messages": messages,
        "reasoning": {"effort": "low"},
        "output": {"max_output_tokens": 128},
    }


async def _run_router(
    router: Algorithm,
    request: dict[str, object],
    client: LiteLLMSyClient,
) -> tuple[list[Decision], dict[str, object]]:
    decisions: list[Decision] = []
    async for step in router.run_stream(request):
        match step:
            case Step.Decision(decision):
                decisions.append(decision)
            case Step.CallModel(call):
                call.respond(await client.call(call.request))
            case Step.Done(response):
                return decisions, response
    raise AssertionError("algorithm stream ended without a response")


@pytest.mark.e2e
async def test_stage_router_calls_both_real_openrouter_models(
    litellm_base_url: str,
) -> None:
    client = LiteLLMSyClient(base_url=litellm_base_url)
    router = algorithms.stage_router(
        "strong",
        "fast",
        picker="efficient_first",
        confidence_threshold=0.5,
        recent_window=3,
    )
    try:
        fast_trace, fast_response = await _run_router(router, _request(), client)
        strong_trace, strong_response = await _run_router(
            router, _request(critical_error=True), client
        )
    finally:
        await client.aclose()

    assert [item.selected_model_id for item in strong_trace] == ["strong"]
    assert [item.selected_model_id for item in fast_trace] == ["fast"]
    for response in (strong_response, fast_response):
        text = response["outputs"][0]["content"][0]["text"]
        assert isinstance(text, str)
        assert text.strip()
