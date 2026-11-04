# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight checks the configured model without relying on provider prefixes."""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest

from craft_taskgen import config
from craft_taskgen.preflight import check_baseline_readiness, check_claude_auth


@pytest.fixture
def claude_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A local executable records its environment and prints a fixture response."""
    response = tmp_path / "response.json"
    captured = tmp_path / "captured.json"
    command = tmp_path / "claude"
    command.write_text(
        f"#!{sys.executable}\n"
        "import json, os\n"
        f"open({str(captured)!r}, 'w').write(json.dumps(dict(os.environ)))\n"
        f"print(open({str(response)!r}).read())\n"
    )
    command.chmod(0o700)
    monkeypatch.setattr(config, "CLAUDE_CMD", str(command))
    return response, captured


@pytest.mark.parametrize("model", ["claude-opus-4-6", "provider/custom-model"])
def test_configured_model_with_or_without_prefix(claude_response, model, capsys):
    response, captured = claude_response
    response.write_text(json.dumps({"modelUsage": {model: {"inputTokens": 1}}}))

    assert check_claude_auth(model)
    env = json.loads(captured.read_text())
    assert env["ANTHROPIC_MODEL"] == model
    assert env["ANTHROPIC_BASE_URL"] == "https://gateway.example.com"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert "[PASS]" in capsys.readouterr().out


@pytest.mark.parametrize(
    "body",
    [
        "not JSON",
        "[]",
        "{}",
        '{"modelUsage": []}',
        '{"modelUsage": {}}',
        '{"modelUsage": {"claude-opus-4-6": []}}',
        '{"modelUsage": {"other/model": {}}}',
        '{"modelUsage": {"claude-opus-4-6": {}, "other/model": {}}}',
        '{"is_error": true, "modelUsage": {"claude-opus-4-6": {}}}',
    ],
)
def test_invalid_or_unexpected_models_fail(claude_response, body, capsys):
    response, _ = claude_response
    response.write_text(body)

    assert not check_claude_auth("claude-opus-4-6")
    assert "[FAIL]" in capsys.readouterr().out


@pytest.mark.parametrize("missing", ["model", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"])
def test_missing_configuration_does_not_start_claude(claude_response, missing, monkeypatch):
    _, captured = claude_response
    model = "claude-opus-4-6"
    if missing == "model":
        model = None
    else:
        monkeypatch.delenv(missing)

    assert not check_claude_auth(model)
    assert not captured.exists()


@pytest.mark.parametrize("model", ["model", "vendor/model"])
def test_vllm_preflight_preserves_model_id(tmp_path, monkeypatch, model):
    """Preflight must preserve namespaces because the launcher already removes dispatch prefixes."""
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body["model"]))
            self.send_response(200 if body["model"] == model else 404)
            self.end_headers()
            self.wfile.write(b"{}")

    # The empty directory and PATH isolate the endpoint check from Docker and local setup.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    with HTTPServer(("127.0.0.1", 0), Endpoint) as server:
        monkeypatch.setenv("VLLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            results = check_baseline_readiness("vllm", model, ["opencode"])
        finally:
            server.shutdown()
            thread.join(timeout=2)
    assert not thread.is_alive()
    assert results[-1]
    assert requests == [("/v1/chat/completions", model)]
