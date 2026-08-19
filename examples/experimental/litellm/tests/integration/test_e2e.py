# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_ROOT = PACKAGE_ROOT / "deployment"
MODEL_GROUP = "switchyard"
SOL_MODEL = "openrouter/openai/gpt-5.6-sol"
TERRA_MODEL = "openrouter/openai/gpt-5.6-terra"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _require_live_environment() -> None:
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


@pytest.fixture
def litellm_base_url(request: pytest.FixtureRequest) -> Iterator[str]:
    _require_live_environment()

    profile = str(request.param)
    port = _free_port()
    project = f"switchyard-litellm-{profile}-{os.getpid()}"
    env = {
        **os.environ,
        "LITELLM_NETWORK": f"{project}-network",
        "LITELLM_PORT": str(port),
        "SWITCHYARD_LITELLM_PROFILE": profile,
    }
    compose = ["docker", "compose", "--project-name", project]
    try:
        subprocess.run(
            [*compose, "up", "-d", "--build", "--wait"],
            cwd=DEPLOYMENT_ROOT,
            env=env,
            check=True,
        )
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        subprocess.run(
            [*compose, "down", "--volumes", "--remove-orphans"],
            cwd=DEPLOYMENT_ROOT,
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
    with pytest.raises(pytest.skip.Exception, match="SWITCHYARD_LITELLM_E2E"):
        _require_live_environment()


def test_paid_e2e_requires_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SWITCHYARD_LITELLM_E2E", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def fail_if_docker_is_checked(_: str) -> str | None:
        pytest.fail("Docker must not be checked without OPENROUTER_API_KEY")

    monkeypatch.setattr(shutil, "which", fail_if_docker_is_checked)
    with pytest.raises(pytest.skip.Exception, match="OPENROUTER_API_KEY"):
        _require_live_environment()


def _critical_tool_history() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "Fix the failing tests, then report the result."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": '{"command":"pytest"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "fatal runtime error: out of memory",
        },
    ]


async def _complete(
    base_url: str,
    messages: list[dict[str, object]],
) -> tuple[dict[str, object], str]:
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            json={"model": MODEL_GROUP, "messages": messages, "max_tokens": 64},
        )
    response.raise_for_status()
    return response.json(), response.headers["x-litellm-model-name"]


def _assert_selected(
    response: dict[str, object],
    selected_model: str,
    expected: str,
) -> None:
    assert response.get("model") == MODEL_GROUP
    assert selected_model == expected
    choices = response.get("choices")
    assert isinstance(choices, list) and choices


@pytest.mark.e2e
@pytest.mark.parametrize("litellm_base_url", ["stage"], indirect=True)
async def test_stage_plugin_routes_live_requests_to_both_gpt_5_6_models(
    litellm_base_url: str,
) -> None:
    efficient, efficient_model = await _complete(
        litellm_base_url,
        [{"role": "user", "content": "Reply with the word hello."}],
    )
    capable, capable_model = await _complete(litellm_base_url, _critical_tool_history())

    _assert_selected(efficient, efficient_model, TERRA_MODEL)
    _assert_selected(capable, capable_model, SOL_MODEL)


@pytest.mark.e2e
@pytest.mark.parametrize("litellm_base_url", ["random"], indirect=True)
async def test_random_plugin_routes_live_requests_to_both_gpt_5_6_models(
    litellm_base_url: str,
) -> None:
    selected = []
    for index in range(2):
        _, selected_model = await _complete(
            litellm_base_url,
            [{"role": "user", "content": f"Reply with the number {index}."}],
        )
        selected.append(selected_model)

    assert selected == [SOL_MODEL, TERRA_MODEL]
