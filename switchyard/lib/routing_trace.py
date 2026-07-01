# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in recording and JSONL export for routing audit events."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from switchyard.lib.proxy_context import ProxyContext

log = logging.getLogger(__name__)

ROUTING_TRACE_JSONL_ENV = "SWITCHYARD_ROUTING_TRACE_JSONL"
ROUTING_TRACE_CAPTURE_CONTENT_ENV = "SWITCHYARD_ROUTING_TRACE_CAPTURE_CONTENT"

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
_WRITE_LOCK = threading.Lock()


def routing_trace_enabled() -> bool:
    """Return whether this process is configured to export routing events."""
    return bool(os.environ.get(ROUTING_TRACE_JSONL_ENV, "").strip())


def routing_trace_content_enabled() -> bool:
    """Return whether producers may include raw request or model content."""
    if not routing_trace_enabled():
        return False
    value = os.environ.get(ROUTING_TRACE_CAPTURE_CONTENT_ENV, "")
    return value.strip().lower() in _TRUTHY_VALUES


def capture_routing_text(value: str) -> dict[str, Any]:
    """Describe text safely, including its content only with explicit opt-in."""
    encoded = value.encode(errors="surrogatepass")
    captured: dict[str, Any] = {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "chars": len(value),
    }
    if routing_trace_content_enabled():
        captured["content"] = value.encode(errors="backslashreplace").decode()
    return captured


def record_routing_event(
    ctx: ProxyContext,
    name: str,
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Append and export an algorithm-owned routing event.

    Routing remains independent of observability: when capture is disabled the
    function is a no-op, and a trace failure never changes the selected route.
    The framework assigns request correlation, sequence, and timestamp only;
    the producer owns the payload and must gate sensitive content explicitly.
    """
    path_value = os.environ.get(ROUTING_TRACE_JSONL_ENV, "").strip()
    if not path_value:
        return None

    try:
        recorded = ctx.record_routing_event(name, payload)
    except Exception:
        log.exception("Failed to record routing trace event %s", name)
        return None
    row = {"request_id": ctx.request_id, **recorded}
    try:
        _append_jsonl(Path(path_value).expanduser(), row)
    except Exception:
        log.exception("Failed to append routing trace event to %s", path_value)
    return recorded


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one complete JSON line without interleaving process threads."""
    encoded = (json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "ab") as target:
            target.write(encoded)
            target.flush()


__all__ = [
    "ROUTING_TRACE_CAPTURE_CONTENT_ENV",
    "ROUTING_TRACE_JSONL_ENV",
    "capture_routing_text",
    "record_routing_event",
    "routing_trace_content_enabled",
    "routing_trace_enabled",
]
