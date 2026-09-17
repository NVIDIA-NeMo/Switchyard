# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual native library against a Relay binary and a local upstream.

Build both artifacts first, then set NEMO_RELAY_TEST_BIN and
SWITCHYARD_TEST_PLUGIN_LIBRARY. No provider account or credentials are needed.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGER = REPOSITORY_ROOT / "crates/switchyard-nemo-relay-plugin/scripts/package_bundle.py"
RELAY = os.environ.get("NEMO_RELAY_TEST_BIN")
LIBRARY = os.environ.get("SWITCHYARD_TEST_PLUGIN_LIBRARY")


class MockUpstream(BaseHTTPRequestHandler):
    """Return deterministic errors and a minimal valid Chat response."""

    def log_message(self, *_args: object) -> None:
        """Keep the test output free of HTTP request logs."""

    def do_POST(self) -> None:
        """Distinguish denials from the one valid target model."""
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = request["model"]
        status = {"denied-401": 401, "denied-403": 403}.get(
            model, 200 if model == "healthy" else 404
        )
        content_type = "application/json"
        if status != 200:
            body = json.dumps(
                {"error": {"message": f"Denied {model}", "type": "mock_denial"}}
            ).encode()
        elif request.get("stream"):
            content_type = "text/event-stream"
            chunks = [
                {"delta": {"role": "assistant", "content": "ready"}, "finish_reason": None},
                {"delta": {}, "finish_reason": "stop"},
            ]
            body = (
                b"".join(
                    b"data: "
                    + json.dumps(
                        {
                            "id": "mock-chat",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": "healthy",
                            "choices": [{"index": 0, **chunk}],
                        }
                    ).encode()
                    + b"\n\n"
                    for chunk in chunks
                )
                + b"data: [DONE]\n\n"
            )
        else:
            body = json.dumps(
                {
                    "id": "mock-chat",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "healthy",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ready"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-Id", "mock-upstream-request")
        if status != 200:
            self.send_header("WWW-Authenticate", 'Bearer realm="mock"')
            self.send_header("X-Denial-Detail", "first")
            self.send_header("X-Denial-Detail", "second")
        self.end_headers()
        self.wfile.write(body)


@unittest.skipUnless(
    RELAY and LIBRARY, "set NEMO_RELAY_TEST_BIN and SWITCHYARD_TEST_PLUGIN_LIBRARY"
)
class UpstreamErrorsTest(unittest.TestCase):
    """Require native plugin OFF/ON parity for unmanaged upstream denials."""

    @classmethod
    def setUpClass(cls) -> None:
        """Package the supplied library and start isolated OFF/ON gateways."""
        cls.temp = tempfile.TemporaryDirectory(prefix="switchyard-relay-errors-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), MockUpstream)
        cls.addClassCleanup(cls.upstream.server_close)
        thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        thread.start()
        cls.addClassCleanup(thread.join)
        cls.addClassCleanup(cls.upstream.shutdown)
        cls.upstream_port = cls.upstream.server_port
        bundle = cls.root / "bundle"
        subprocess.run(
            [
                sys.executable,
                str(PACKAGER),
                "--library",
                str(Path(LIBRARY).resolve()),
                "--output",
                str(bundle),
            ],
            check=True,
            capture_output=True,
        )
        cls.ports = {"direct": cls.upstream_port}
        for enabled in (False, True):
            label = "on" if enabled else "off"
            env = {
                key: value for key, value in os.environ.items() if not key.startswith("NEMO_RELAY_")
            }
            for key in (
                "XDG_CONFIG_HOME",
                "XDG_DATA_HOME",
                "XDG_STATE_HOME",
                "XDG_CACHE_HOME",
                "XDG_CONFIG_DIRS",
            ):
                directory = cls.root / label / key
                directory.mkdir(parents=True)
                env[key] = str(directory)
            env["SWITCHYARD_TEST_PROVIDER_KEY"] = "mock-configured-key"
            if enabled:
                registry = Path(env["XDG_CONFIG_HOME"]) / "nemo-relay/plugins.toml"
                registry.parent.mkdir()
                registry.write_text(f"""version = 1
[[plugins.dynamic]]
manifest = {json.dumps(str(bundle / "relay-plugin.toml"))}
[plugins.dynamic.config]
priority = 0
[plugins.dynamic.config.switchyard_config]
schema_version = 1
[plugins.dynamic.config.switchyard_config.llm_clients.mock]
format = "openai_chat"
base_url = "http://127.0.0.1:{cls.upstream_port}/v1"
api_key_env = "SWITCHYARD_TEST_PROVIDER_KEY"
max_retries = 0
[plugins.dynamic.config.switchyard_config.targets.healthy]
id = "healthy"
llm_client = "mock"
[plugins.dynamic.config.switchyard_config.routes.core]
id = "switchyard/core"
type = "passthrough"
target = "healthy"
[plugins.policy.overrides."nvidia.switchyard"]
attestation = "integrity_only"
""")
                subprocess.run(
                    [RELAY, "plugins", "enable", "nvidia.switchyard"],
                    env=env,
                    cwd=cls.root,
                    check=True,
                    capture_output=True,
                )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            log_path = cls.root / f"{label}.log"
            log = log_path.open("w")
            cls.addClassCleanup(log.close)
            proc = subprocess.Popen(
                [
                    RELAY,
                    "--bind",
                    f"127.0.0.1:{port}",
                    "--openai-base-url",
                    f"http://127.0.0.1:{cls.upstream_port}/v1",
                    "--anthropic-base-url",
                    f"http://127.0.0.1:{cls.upstream_port}",
                ],
                env=env,
                cwd=cls.root,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            cls.addClassCleanup(cls.stop, proc)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Relay {label} exited: {log_path.read_text()}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise RuntimeError(f"Relay {label} startup timed out: {log_path.read_text()}")
            cls.ports[label] = port

    @staticmethod
    def stop(proc: subprocess.Popen) -> None:
        """Reap every gateway even if a test fails."""
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    def request(self, arm: str, path: str, model: str, stream: bool) -> tuple:
        """Read the entire response so a truncated stream fails the test."""
        payload = {"model": model, "stream": stream}
        if path == "/v1/responses":
            payload["input"] = "Reply ready."
        else:
            payload["messages"] = [{"role": "user", "content": "Reply ready."}]
            payload["max_tokens"] = 16
        connection = http.client.HTTPConnection("127.0.0.1", self.ports[arm], timeout=15)
        try:
            connection.request(
                "POST",
                path,
                json.dumps(payload),
                {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer mock-request-key",
                    "x-api-key": "mock-request-key",
                    "anthropic-version": "2023-06-01",
                },
            )
            response = connection.getresponse()
            return response.status, response.getheaders(), response.read()
        finally:
            connection.close()

    def test_unmanaged_denials_preserve_status_headers_and_body(self) -> None:
        """Exercise the 36 direct/OFF/ON denial combinations across all inbound formats."""
        for path in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            for streaming in (False, True):
                for status in (401, 403):
                    expected = self.request("direct", path, f"denied-{status}", streaming)
                    for arm in ("off", "on"):
                        with self.subTest(path=path, streaming=streaming, status=status, arm=arm):
                            actual = self.request(arm, path, f"denied-{status}", streaming)
                            self.assertEqual(actual[0], status)
                            self.assertEqual(actual[2], expected[2])
                            for header in (
                                "content-type",
                                "www-authenticate",
                                "x-request-id",
                                "x-denial-detail",
                            ):
                                actual_values = [v for k, v in actual[1] if k.lower() == header]
                                expected_values = [v for k, v in expected[1] if k.lower() == header]
                                self.assertEqual(actual_values, expected_values)

    def test_healthy_streams_and_actual_route_activation(self) -> None:
        """Healthy requests finish, and only an enabled real plugin can serve its route."""
        self.assertEqual(
            self.request("off", "/v1/chat/completions", "switchyard/core", False)[0], 404
        )
        for arm, model in (("off", "healthy"), ("on", "healthy"), ("on", "switchyard/core")):
            for streaming in (False, True):
                with self.subTest(arm=arm, model=model, streaming=streaming):
                    status, _, body = self.request(arm, "/v1/chat/completions", model, streaming)
                    self.assertEqual(status, 200)
                    if streaming:
                        self.assertEqual(body.count(b"[DONE]"), 1)
                        self.assertEqual(body.count(b"ready"), 1)
                    else:
                        self.assertEqual(
                            json.loads(body)["choices"][0]["message"]["content"], "ready"
                        )


if __name__ == "__main__":
    unittest.main()
