# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build intake payloads from request/response primitives."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

from switchyard.lib.config.intake_sink_config import (
    IntakeSinkConfig,
)
from switchyard.lib.cost_estimator import estimate_model_cost
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.request_metadata import (
    CTX_REQUEST_METADATA,
    RequestMetadata,
)
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
    request_type_matches,
    response_type_matches,
)
from switchyard_rust.translation import TranslationEngine

JsonObject = dict[str, object]

#: Context metadata key for random routing tier selection (mirrored from core.backends.random_routing_llm_backend)
_CTX_RANDOM_ROUTING_TIER = "_random_routing_tier"

INTAKE_STARTED_AT_MS_KEY = "_intake_started_at_ms"
INTAKE_ENDED_AT_MS_KEY = "_intake_ended_at_ms"
INTAKE_SESSION_ID_KEY = "_intake_session_id"
INTAKE_INBOUND_FORMAT_KEY = "_intake_inbound_format"
INTAKE_REQUEST_SNAPSHOT_KEY = "_intake_request_snapshot"
INTAKE_SKIP_KEY = "_intake_skip"
SYNTHETIC_STREAM_RESPONSE_IDS = frozenset(
    {
        "chatcmpl-intake-stream",
        "chatcmpl-switchyard-stream",
        "msg_switchyard_stream",
        "resp_switchyard_stream",
    }
)


class IntakePayloadBuilder:
    """Pure builder that produces one intake payload per completed turn."""

    def __init__(self, config: IntakeSinkConfig) -> None:
        self._config = config
        self._translation = TranslationEngine()

    def build(
        self,
        *,
        ctx: ProxyContext,
        request_snapshot: ChatRequest,
        response: ChatResponse,
        stream: bool,
    ) -> JsonObject:
        """Build a single intake payload."""
        openai_request = self._translation.request_to(
            ChatRequestType.OPENAI_CHAT,
            request_snapshot,
        )
        if not request_type_matches(openai_request, ChatRequestType.OPENAI_CHAT):
            raise NotImplementedError(
                f"Intake payloads require an OpenAI Chat-shaped request, got "
                f"{type(request_snapshot).__name__}",
            )
        openai_response = self._build_openai_response_dict(response)
        session_id_raw = ctx.metadata.get(INTAKE_SESSION_ID_KEY)
        session_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else None
        request_entry = self._build_request_entry(
            ctx=ctx,
            openai_request=dict(openai_request.body),
            stream=stream,
        )
        response_entry = _sanitize_response_entry(openai_response)
        # Metadata-only unless content capture is explicitly enabled.
        if not self._config.capture_content:
            _redact_content(request_entry, response_entry)
        payload: JsonObject = {
            "request": request_entry,
            "response": response_entry,
            "provider": "switchyard",
        }
        # Cost reads usage from the unredacted response, so metrics are unaffected.
        payload.update(_cost_fields(openai_response))
        evaluation_context = self._evaluation_context(ctx)
        if evaluation_context:
            payload["evaluation_context"] = evaluation_context
        if session_id is not None:
            payload["session_id"] = session_id
        return payload

    def request_from_snapshot(self, ctx: ProxyContext) -> ChatRequest:
        """Read the copied original request wrapper from context metadata."""
        request = ctx.metadata.get(INTAKE_REQUEST_SNAPSHOT_KEY)
        if not isinstance(request, ChatRequest):
            raise ValueError("Missing intake request snapshot in context")
        return request

    def _build_openai_response_dict(self, response: ChatResponse) -> JsonObject:
        translated = self._translation.response_to(ChatRequestType.OPENAI_CHAT, response)
        if response_type_matches(translated, ChatResponseType.OPENAI_COMPLETION):
            return cast(JsonObject, dict(translated.body))
        raise NotImplementedError(
            f"Intake payloads require an OpenAI Chat-shaped response, got "
            f"{type(response).__name__}",
        )

    def _build_routing(self, ctx: ProxyContext) -> dict[str, str]:
        random_tier = ctx.metadata.get(_CTX_RANDOM_ROUTING_TIER)
        if not isinstance(random_tier, str) or not random_tier:
            return {}

        return {
            "router_type": "random",
            "routed_to": random_tier,
        }

    def _build_request_entry(
        self,
        *,
        ctx: ProxyContext,
        openai_request: JsonObject,
        stream: bool,
    ) -> JsonObject:
        request_entry = dict(openai_request)
        if stream or request_entry.get("stream") is True:
            request_entry["stream"] = False
        started_at_ms = _coerce_int(ctx.metadata.get(INTAKE_STARTED_AT_MS_KEY))
        ended_at_ms = _coerce_int(ctx.metadata.get(INTAKE_ENDED_AT_MS_KEY))
        switchyard_metadata: JsonObject = {
            "version": _switchyard_version(),
            "inbound_format": ctx.metadata.get(INTAKE_INBOUND_FORMAT_KEY),
            "stream": stream,
            "user_id": self._config.user_id,
            "created_at": _created_at_iso(started_at_ms, ended_at_ms),
        }
        session_id_raw = ctx.metadata.get(INTAKE_SESSION_ID_KEY)
        if isinstance(session_id_raw, str) and session_id_raw:
            switchyard_metadata["session_id"] = session_id_raw
        if started_at_ms is not None and ended_at_ms is not None:
            switchyard_metadata["latency_ms"] = ended_at_ms - started_at_ms
        routing = self._build_routing(ctx)
        if routing:
            switchyard_metadata["routing"] = routing
        request_entry["switchyard"] = switchyard_metadata
        return request_entry

    def _task_name(self, ctx: ProxyContext) -> str:
        return _request_metadata(ctx).intake.task or "chat"

    def _evaluation_context(self, ctx: ProxyContext) -> JsonObject | None:
        session_id_raw = ctx.metadata.get(INTAKE_SESSION_ID_KEY)
        evaluation_run_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else None
        if not evaluation_run_id:
            return None
        return {
            "evaluation_run_id": evaluation_run_id,
            "test_case_id": self._task_name(ctx),
        }


def _request_metadata(ctx: ProxyContext) -> RequestMetadata:
    """Read :class:`RequestMetadata` off ``ctx.metadata`` or return a blank one."""
    value = ctx.metadata.get(CTX_REQUEST_METADATA)
    return value if isinstance(value, RequestMetadata) else RequestMetadata()


def _redact_content(request_entry: JsonObject, response_entry: JsonObject) -> None:
    """Strip prompt/response text so the payload is metadata-only; model/usage stay."""
    for key in (
        "messages",
        "system",
        "prompt",
        "input",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
    ):
        request_entry.pop(key, None)
    choices = response_entry.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice.pop("logprobs", None)
            for shape in ("message", "delta"):
                part = choice.get(shape)
                if isinstance(part, dict):
                    for key in [k for k in part if k != "role"]:
                        part.pop(key, None)


def _sanitize_response_entry(openai_response: JsonObject) -> JsonObject:
    response = dict(openai_response)
    response_id = response.get("id")
    if response_id in SYNTHETIC_STREAM_RESPONSE_IDS:
        response.pop("id", None)
    return response


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _cost_fields(openai_response: JsonObject) -> JsonObject:
    model = _str_or_none(openai_response.get("model"))
    if model is None:
        return {}
    usage = _usage_from_response(openai_response)
    if usage is None:
        return {}
    breakdown = estimate_model_cost(
        model=model,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        cached_tokens=usage["cached_tokens"],
        cache_creation_tokens=usage["cache_creation_tokens"],
    )
    total_cost = breakdown["total_cost"]
    if total_cost <= 0:
        return {}
    input_cost = breakdown["input_cost"]
    output_cost = breakdown["output_cost"]
    return {
        "cost_usd": total_cost,
        "cost_input_usd": input_cost,
        "cost_output_usd": output_cost,
        "cost_details": {
            "base_input": breakdown["base_input_cost"],
            "cached_input": breakdown["cached_input_cost"],
            "cache_write": breakdown["cache_write_cost"],
        },
    }


def _usage_from_response(openai_response: JsonObject) -> dict[str, int] | None:
    usage = openai_response.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_details = _dict_or_empty(usage.get("prompt_tokens_details"))
    prompt_tokens = _clean_int(usage.get("prompt_tokens"))
    completion_tokens = _clean_int(usage.get("completion_tokens"))
    if prompt_tokens is None or completion_tokens is None:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": _clean_int(prompt_details.get("cached_tokens")) or 0,
        "cache_creation_tokens": _clean_int(prompt_details.get("cache_creation_tokens")) or 0,
    }


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clean_int(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value if value >= 0 else None


def _str_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


@cache
def _switchyard_version() -> str:
    try:
        return version("nemo-switchyard")
    except PackageNotFoundError:
        return "unknown"


def _created_at_iso(started_at_ms: int | None, ended_at_ms: int | None) -> str:
    source_ms = started_at_ms if started_at_ms is not None else ended_at_ms
    if source_ms is None:
        return datetime.now(UTC).isoformat()
    return datetime.fromtimestamp(source_ms / 1000, tz=UTC).isoformat()
