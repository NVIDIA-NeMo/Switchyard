# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response-side processor that appends one JSONL routing record per request."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from switchyard.lib.chat_response.streaming_response_accumulator import (
    attach_final_response_callback,
)
from switchyard.lib.proxy_context import CTX_PROXY_ACTUAL_MODEL, ProxyContext
from switchyard.lib.request_metadata import CTX_REQUEST_METADATA
from switchyard_rust.core import ChatResponse

if TYPE_CHECKING:
    from switchyard.lib.endpoints.base import Endpoint

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
        """Log one routing record for the completed request; return the response unchanged."""
        served_model: str = (
            ctx.metadata.get(CTX_PROXY_ACTUAL_MODEL) or ctx.selected_model or "unknown"
        )

        async def _emit(final: ChatResponse) -> None:
            await asyncio.to_thread(self._write_record, ctx, served_model, final)

        attached = attach_final_response_callback(
            response, served_model=served_model, callback=_emit,
        )
        if not attached:
            await _emit(response)
        return response

    def _write_record(self, ctx: ProxyContext, served_model: str, response: ChatResponse) -> None:
        metadata = ctx.metadata.get(CTX_REQUEST_METADATA)
        actual_model = _field(response.body, "model")
        record = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "task": getattr(getattr(metadata, "intake", None), "task", None),
            "session_id": getattr(metadata, "session_id", None),
            "model": actual_model if isinstance(actual_model, str) and actual_model else served_model,
            "tier": ctx.metadata.get("_random_routing_tier", "") or (ctx.selected_target or ""),
            **_usage_tokens(response.body),
        }
        try:
            line = json.dumps(record, separators=(",", ":"))
            with self._lock, self._log_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            logger.warning("Routing log: failed to append to %s: %s", self._log_file, exc)

    def snapshot_session(self, session_id: str) -> dict[str, object] | None:
        """Aggregate the durable request log for one exact trial session.

        This re-reads and re-parses the whole log on every call. It is sized for
        benchmark runs (thousands of records), not a long-lived production server
        with millions of sessions, where it would be O(records) per query.
        """
        models: dict[str, dict[str, int]] = {}
        totals = {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_cached_tokens": 0,
            "total_cache_creation_tokens": 0,
            "total_completion_tokens": 0,
        }
        try:
            if not self._log_file.is_file():
                return None
            lines = self._log_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("Routing log: failed to read %s: %s", self._log_file, exc)
            return None

        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, Mapping) or record.get("session_id") != session_id:
                continue
            model_value = record.get("model")
            model = model_value if isinstance(model_value, str) and model_value else "unknown"
            bucket = models.setdefault(
                model,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "cached_tokens": 0,
                    "cache_creation_tokens": 0,
                    "completion_tokens": 0,
                },
            )
            bucket["calls"] += 1
            totals["total_calls"] += 1
            for record_key, bucket_key, total_key in (
                ("prompt_tokens", "prompt_tokens", "total_prompt_tokens"),
                ("cached_tokens", "cached_tokens", "total_cached_tokens"),
                (
                    "cache_creation_tokens",
                    "cache_creation_tokens",
                    "total_cache_creation_tokens",
                ),
                ("completion_tokens", "completion_tokens", "total_completion_tokens"),
            ):
                value = record.get(record_key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    bucket[bucket_key] += value
                    totals[total_key] += value

        if not totals["total_calls"]:
            return None
        return {"session_id": session_id, **totals, "models": models}

    def get_endpoint(self) -> Endpoint:
        """Contribute the session-scoped routing-stat snapshot endpoint."""
        from switchyard.lib.endpoints.routing_log_stats_endpoint import (
            RoutingLogStatsEndpoint,
        )

        return RoutingLogStatsEndpoint(self)


def _usage_tokens(body: object) -> dict[str, int]:
    """Token counts from any native usage shape (OpenAI Chat/Responses, Anthropic)."""
    usage = _field(body, "usage")
    prompt = _int_field(usage, "prompt_tokens")
    completion = _int_field(usage, "completion_tokens")
    prompt_details = _field(usage, "prompt_tokens_details")
    cached = _int_field(prompt_details, "cached_tokens")
    cache_creation = _int_field(prompt_details, "cache_creation_tokens")
    if not prompt and not completion:
        input_tokens = _int_field(usage, "input_tokens")
        input_details = _field(usage, "input_tokens_details")
        if input_details is not None:
            cached = cached or _int_field(input_details, "cached_tokens")
            cache_creation = cache_creation or _int_field(
                input_details, "cache_creation_tokens"
            )
            prompt = input_tokens
        else:
            cached = cached or _int_field(usage, "cache_read_input_tokens")
            cache_creation = cache_creation or _int_field(
                usage, "cache_creation_input_tokens"
            )
            prompt = input_tokens + cached + cache_creation
        completion = _int_field(usage, "output_tokens")
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_creation_tokens": cache_creation,
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
