# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loopback readiness probes must ignore env proxies (HTTP_PROXY/NO_PROXY)."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from switchyard.cli.launchers.launcher_runtime import wait_for_proxy_ready


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def test_wait_for_proxy_ready_bypasses_env_proxy(monkeypatch):
    """A configured HTTP_PROXY must not intercept the 127.0.0.1 health probe."""
    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Point env proxy at a dead port and remove any NO_PROXY exemption; before
    # the fix the loopback probe would route here and fail.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    try:
        assert wait_for_proxy_ready(port, timeout_s=2) is True
    finally:
        server.shutdown()
        server.server_close()
