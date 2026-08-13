# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic smoke coverage for the documented standalone server flow."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_DOCUMENTED_BASE_URL = 'base_url = "https://openrouter.ai/api/v1"'
_CLASSIFIER_RESULT = {
    "crux": "bounded task",
    "primary_rule": "SUP-1",
    "capability_boundary": "supported",
    "p_solve": 0.9,
}


class _OpenAIStub(ThreadingHTTPServer):
    requests: list[dict[str, Any]]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _OpenAIStubHandler)
        self.requests = []
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> _OpenAIStub:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
        self.server_close()
        self.thread.join(timeout=5)


class _OpenAIStubHandler(BaseHTTPRequestHandler):
    server: _OpenAIStub

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.server.requests.append(body)

        if "response_format" in body:
            content = json.dumps(_CLASSIFIER_RESULT, separators=(",", ":"))
        else:
            content = "hello from the local upstream"

        payload = json.dumps(
            {
                "id": "chatcmpl-onboarding-smoke",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_: object) -> None:
        pass


def _extract_documented_config(guide: str) -> str:
    configure_section = guide.split("### Configure", maxsplit=1)[1]
    match = re.search(r"```toml\n(?P<config>.*?)\n```", configure_section, re.DOTALL)
    assert match is not None, "Getting Started must contain a TOML server config"
    return match.group("config") + "\n"


def _extract_documented_completion(guide: str) -> dict[str, Any]:
    run_section = guide.split("### Run the server", maxsplit=1)[1]
    match = re.search(
        r"/v1/chat/completions.*?-d '(?P<body>\{.*?\})'",
        run_section,
        re.DOTALL,
    )
    assert match is not None, "Getting Started must contain a completion request"
    return json.loads(match.group("body"))


def _server_binary(repository: Path) -> Path:
    configured = os.environ.get("SWITCHYARD_SERVER_BIN")
    binary = Path(configured) if configured else repository / "target/debug/switchyard-server"
    assert binary.is_file(), (
        f"server binary not found at {binary}; run "
        "`cargo build --locked -p switchyard-server` first"
    )
    return binary


def _request_json(
    url: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = 2,
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        assert response.status == 200
        return json.load(response)


async def _wait_until_healthy(base_url: str, process: asyncio.subprocess.Process) -> None:
    deadline = asyncio.get_running_loop().time() + 10
    while asyncio.get_running_loop().time() < deadline:
        if process.returncode is not None:
            stdout, stderr = await process.communicate()
            raise AssertionError(
                "server exited before becoming healthy\n"
                f"stdout:\n{stdout.decode(errors='replace')}\n"
                f"stderr:\n{stderr.decode(errors='replace')}"
            )
        try:
            health = await asyncio.to_thread(_request_json, f"{base_url}/health")
            if health.get("status") == "ok":
                return
        except (OSError, urllib.error.URLError):
            await asyncio.sleep(0.05)
    raise AssertionError("server did not become healthy within 10 seconds")


async def _read_listen_url(
    process: asyncio.subprocess.Process,
    *,
    timeout: float = 10,
) -> str:
    assert process.stdout is not None
    assert process.stderr is not None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    readers: dict[
        asyncio.Task[bytes],
        tuple[str, asyncio.StreamReader],
    ] = {
        asyncio.create_task(process.stdout.read(4096)): ("stdout", process.stdout),
        asyncio.create_task(process.stderr.read(4096)): ("stderr", process.stderr),
    }
    deadline = asyncio.get_running_loop().time() + timeout
    try:
        while readers:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            done, _ = await asyncio.wait(
                readers,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                break
            for task in done:
                label, reader = readers.pop(task)
                chunk = task.result()
                if not chunk:
                    continue
                buffers[label].extend(chunk)
                match = re.search(
                    rb"listening: (http://127\.0\.0\.1:\d+)",
                    buffers["stdout"],
                )
                if match is not None:
                    return match.group(1).decode()
                readers[asyncio.create_task(reader.read(4096))] = (label, reader)
    finally:
        for task in readers:
            task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)

    outcome = "exited" if process.returncode is not None else "timed out"
    raise AssertionError(
        f"server {outcome} without reporting its ephemeral port\n"
        f"stdout:\n{buffers['stdout'].decode(errors='replace')}\n"
        f"stderr:\n{buffers['stderr'].decode(errors='replace')}"
    )


async def exercise_documented_server_flow(guide_path: Path, tmp_path: Path) -> None:
    """Run the documented dry-run, server, endpoints, and completion request."""

    repository = guide_path.parents[1]
    guide = await asyncio.to_thread(guide_path.read_text)
    documented_config = _extract_documented_config(guide)
    documented_request = _extract_documented_completion(guide)

    upstream = await asyncio.to_thread(_OpenAIStub)
    await asyncio.to_thread(upstream.__enter__)
    try:
        assert documented_config.count(_DOCUMENTED_BASE_URL) == 1
        config = documented_config.replace(
            _DOCUMENTED_BASE_URL,
            f'base_url = "{upstream.base_url}"',
        )
        config_path = tmp_path / "routes.toml"
        await asyncio.to_thread(config_path.write_text, config)

        environment = os.environ.copy()
        environment.update(
            {
                "OPENROUTER_API_KEY": "onboarding-smoke-test-key",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
        )
        binary = await asyncio.to_thread(_server_binary, repository)
        dry_run = await asyncio.create_subprocess_exec(
            str(binary),
            "--config",
            str(config_path),
            "--dry-run",
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            dry_stdout, dry_stderr = await asyncio.wait_for(dry_run.communicate(), timeout=10)
        except TimeoutError:
            dry_run.kill()
            await dry_run.communicate()
            raise AssertionError("dry-run did not finish within 10 seconds") from None
        assert dry_run.returncode == 0, dry_stderr.decode(errors="replace")
        assert b"server OK: switchyard" in dry_stdout

        server = await asyncio.create_subprocess_exec(
            str(binary),
            "--config",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            base_url = await _read_listen_url(server)
            await _wait_until_healthy(base_url, server)
            health = await asyncio.to_thread(_request_json, f"{base_url}/health")
            assert health["status"] == "ok"

            models = await asyncio.to_thread(_request_json, f"{base_url}/v1/models")
            assert "switchyard" in models["model_pool"]

            completion = await asyncio.to_thread(
                _request_json,
                f"{base_url}/v1/chat/completions",
                documented_request,
                timeout=10,
            )
            assert completion["choices"][0]["message"]["content"] == "hello from the local upstream"
            assert len(upstream.requests) == 2
            classifier_call, routed_call = upstream.requests
            assert classifier_call["model"] == "openai/gpt-4o-mini"
            assert classifier_call["response_format"]["type"] == "json_schema"
            assert routed_call["model"] in {
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
            }
            assert routed_call["messages"] == documented_request["messages"]
        finally:
            if server.returncode is None:
                server.terminate()
            try:
                await asyncio.wait_for(server.communicate(), timeout=5)
            except TimeoutError:
                server.kill()
                await server.communicate()
    finally:
        await asyncio.to_thread(upstream.__exit__, None, None, None)
