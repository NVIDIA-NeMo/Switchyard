# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response-side processor that appends one JSONL routing record per request."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchyard.lib.chat_response.streaming_response_accumulator import (
    attach_final_response_callback,
)
from switchyard.lib.proxy_context import CTX_PROXY_ACTUAL_MODEL, ProxyContext
from switchyard.lib.request_metadata import CTX_REQUEST_METADATA
from switchyard_rust.core import ChatResponse

logger = logging.getLogger(__name__)


class RoutingLogResponseProcessor:
    """Append one JSON line per completed request to ``log_file``.

    Each record carries the routing decision (selected model and tier), the
    caller-supplied task and session identity headers, and token usage, so a
    benchmark harness can attribute router traffic to individual tasks.
    Streaming responses log once the stream drains; write failures are logged
    and never break the proxied response.
    """

    def __init__(self, log_file: Path | str) -> None:
        self._log_file = Path(log_file)
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    async def process(self, ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        served_model: str = ctx.selected_model or ctx.metadata.get(
            CTX_PROXY_ACTUAL_MODEL, "unknown",
        )

        async def _emit(final: ChatResponse) -> None:
            self._write_record(ctx, served_model, final)

        attached = attach_final_response_callback(
            response, served_model=served_model, callback=_emit,
        )
        if not attached:
            await _emit(response)
        return response

    def _write_record(self, ctx: ProxyContext, served_model: str, response: ChatResponse) -> None:
        metadata = ctx.metadata.get(CTX_REQUEST_METADATA)
        record = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "task": getattr(getattr(metadata, "intake", None), "task", None),
            "session_id": getattr(metadata, "session_id", None),
            "model": served_model,
            "tier": ctx.metadata.get("_random_routing_tier", "") or (ctx.selected_target or ""),
            **_usage_tokens(response.body),
        }
        try:
            line = json.dumps(record, separators=(",", ":"))
            with self._lock, self._log_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            logger.warning("Routing log: failed to append to %s: %s", self._log_file, exc)


def _usage_tokens(body: object) -> dict[str, int]:
    """Token counts from any native usage shape (OpenAI Chat/Responses, Anthropic)."""
    usage = _field(body, "usage")
    prompt = _int_field(usage, "prompt_tokens")
    completion = _int_field(usage, "completion_tokens")
    if not prompt and not completion:
        prompt = (
            _int_field(usage, "input_tokens")
            + _int_field(usage, "cache_creation_input_tokens")
            + _int_field(usage, "cache_read_input_tokens")
        )
        completion = _int_field(usage, "output_tokens")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _field(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _int_field(value: object, name: str) -> int:
    field = _field(value, name)
    return field if isinstance(field, int) else 0
