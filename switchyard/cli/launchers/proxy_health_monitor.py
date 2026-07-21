# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight health indicator for launcher footers."""

import time

from switchyard.cli.launchers.launcher_runtime import _LOCAL_OPENER


class ProxyHealthMonitor:
    """Periodic liveness probe for the in-process proxy.

    ``poll()`` is designed to run from a footer-render thread. It performs a
    quick ``GET /health`` only when the check interval has elapsed and otherwise
    returns immediately.
    """

    _CHECK_INTERVAL_S = 2.0

    def __init__(self, port: int) -> None:
        self._url = f"http://127.0.0.1:{port}/health"
        self._healthy: bool | None = None
        self._last_check: float = 0.0

    def poll(self) -> None:
        """Run a health check if the check interval has elapsed."""
        now = time.monotonic()
        if now - self._last_check < self._CHECK_INTERVAL_S:
            return
        self._last_check = now
        try:
            # Loopback: bypass env proxies so a configured HTTP_PROXY can't
            # intercept the 127.0.0.1 health probe.
            with _LOCAL_OPENER.open(self._url, timeout=0.5):
                self._healthy = True
        except Exception:
            self._healthy = False

    @property
    def indicator(self) -> tuple[str, int]:
        """Return ``(styled_str, visible_width)`` for the footer."""
        if self._healthy is None:
            return "\x1b[33m●\x1b[0m", 1
        return ("\x1b[92m●\x1b[0m", 1) if self._healthy else ("\x1b[91m●\x1b[0m", 1)
