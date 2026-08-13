# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic smoke coverage for the documented standalone server flow."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
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


def _wait_until_healthy(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"server exited before becoming healthy\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            if _request_json(f"{base_url}/health").get("status") == "ok":
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise AssertionError("server did not become healthy within 10 seconds")


def _read_listen_url(process: subprocess.Popen[str]) -> str:
    assert process.stdout is not None
    startup_lines = []
    for line in process.stdout:
        startup_lines.append(line)
        match = re.search(r"listening: (http://127\.0\.0\.1:\d+)", line)
        if match is not None:
            return match.group(1)
        if process.poll() is not None:
            break
    stderr = "" if process.stderr is None else process.stderr.read()
    raise AssertionError(
        "server exited without reporting its ephemeral port\n"
        f"stdout:\n{''.join(startup_lines)}\nstderr:\n{stderr}"
    )


def exercise_documented_server_flow(guide_path: Path, tmp_path: Path) -> None:
    """Run the documented dry-run, server, endpoints, and completion request."""

    repository = guide_path.parents[1]
    guide = guide_path.read_text()
    documented_config = _extract_documented_config(guide)
    documented_request = _extract_documented_completion(guide)

    with _OpenAIStub() as upstream:
        assert documented_config.count(_DOCUMENTED_BASE_URL) == 1
        config = documented_config.replace(
            _DOCUMENTED_BASE_URL,
            f'base_url = "{upstream.base_url}"',
        )
        config_path = tmp_path / "routes.toml"
        config_path.write_text(config)

        environment = os.environ.copy()
        environment.update(
            {
                "OPENROUTER_API_KEY": "onboarding-smoke-test-key",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
        )
        binary = _server_binary(repository)
        dry_run = subprocess.run(
            [binary, "--config", config_path, "--dry-run"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert dry_run.returncode == 0, dry_run.stderr
        assert "server OK: switchyard" in dry_run.stdout

        server = subprocess.Popen(
            [
                binary,
                "--config",
                config_path,
                "--host",
                "127.0.0.1",
                "--port",
                "0",
            ],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            base_url = _read_listen_url(server)
            _wait_until_healthy(base_url, server)
            health = _request_json(f"{base_url}/health")
            assert health["status"] == "ok"

            models = _request_json(f"{base_url}/v1/models")
            assert "switchyard" in models["model_pool"]

            completion = _request_json(
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
            server.terminate()
            try:
                server.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.communicate(timeout=5)
