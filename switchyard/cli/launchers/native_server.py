# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Rust server lifecycle for coding-agent launchers."""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from switchyard.cli.launchers.stats_source import StatsSource

log = logging.getLogger(__name__)

_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class HttpStatsSource(StatsSource):
    """Read launcher statistics from the native server."""

    def __init__(self, base_url: str) -> None:
        self._url = f"{base_url}/v1/stats"
        self._last_snapshot: Mapping[str, object] = {}

    def snapshot_sync(self) -> Mapping[str, object]:
        """Return fresh stats, retaining the last snapshot on transient failure."""
        try:
            with _LOCAL_OPENER.open(self._url, timeout=0.5) as response:
                payload = json.load(response)
            if isinstance(payload, dict):
                self._last_snapshot = payload
        except Exception:
            log.debug("failed to fetch native launcher stats", exc_info=True)
        return self._last_snapshot


class NativeServer:
    """Host one TOML deployment through the PyO3 Rust server binding."""

    def __init__(self, config: Path) -> None:
        from switchyard_rust.server import Server

        self._server = Server(config, port=0)
        self.port: int = self._server.port
        self.base_url: str = self._server.base_url
        self.stats: StatsSource = HttpStatsSource(self.base_url)

    def close(self) -> None:
        """Gracefully stop the native server."""
        self._server.close()


__all__ = ["HttpStatsSource", "NativeServer"]
