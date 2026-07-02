# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/response processors that save skill-distillation session turns."""

import logging
from copy import deepcopy
from typing import Any

from switchyard.lib.chat_response.streaming_response_accumulator import (
    attach_final_response_callback,
)
from switchyard.lib.proxy_context import CTX_PROXY_ACTUAL_MODEL, CTX_ROUTING, ProxyContext
from switchyard.lib.skill_distillation_store import SkillDistillationSessionCapture
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
    response_type_matches,
)
from switchyard_rust.translation import TranslationEngine

logger = logging.getLogger(__name__)

CTX_SKILL_DISTILLATION_REQUEST = "_skill_distillation_request"

JsonObject = dict[str, Any]


class SkillDistillationRequestProcessor:
    """Snapshot the inbound request for a saved skill-distillation turn."""

    def __init__(self) -> None:
        self._translation = TranslationEngine()

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        try:
            openai_request = self._translation.request_to(
                ChatRequestType.OPENAI_CHAT, request,
            )
        except Exception as exc:
            logger.warning(
                "Skill distillation: failed to snapshot request: %s",
                exc,
            )
            if CTX_SKILL_DISTILLATION_REQUEST in ctx.metadata:
                del ctx.metadata[CTX_SKILL_DISTILLATION_REQUEST]
            return request
        ctx.metadata[CTX_SKILL_DISTILLATION_REQUEST] = deepcopy(dict(openai_request.body))
        return request


class SkillDistillationResponseProcessor:
    """Append completed turns to a :class:`SkillDistillationSessionCapture`."""

    def __init__(self, session: SkillDistillationSessionCapture) -> None:
        self._session = session
        self._translation = TranslationEngine()

    async def process(self, ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        served_model: str = ctx.selected_model or ctx.metadata.get(
            CTX_PROXY_ACTUAL_MODEL, "unknown",
        )

        async def _emit(final: ChatResponse) -> None:
            self._write_turn(ctx, final, served_model=served_model)

        try:
            attached = attach_final_response_callback(
                response, served_model=served_model, callback=_emit,
            )
        except Exception as exc:
            logger.warning(
                "Skill distillation: failed to attach streaming capture: %s",
                exc,
            )
            return response
        if not attached:
            await _emit(response)
        return response

    def _write_turn(
        self,
        ctx: ProxyContext,
        response: ChatResponse,
        *,
        served_model: str,
    ) -> None:
        request = ctx.metadata.get(CTX_SKILL_DISTILLATION_REQUEST)
        if not isinstance(request, dict):
            return
        try:
            entry = self._build_entry(ctx, request, response, served_model=served_model)
        except Exception as exc:
            logger.warning(
                "Skill distillation: failed to build turn record: %s",
                exc,
            )
            return
        if entry is None:
            return
        try:
            self._session.record_turn(entry)
        except Exception as exc:  # pragma: no cover - record_turn is fail-open.
            logger.warning(
                "Skill distillation: failed to save turn record: %s",
                exc,
            )

    def _build_entry(
        self,
        ctx: ProxyContext,
        request: JsonObject,
        response: ChatResponse,
        *,
        served_model: str,
    ) -> JsonObject | None:
        translated = self._translation.response_to(ChatRequestType.OPENAI_CHAT, response)
        if not response_type_matches(translated, ChatResponseType.OPENAI_COMPLETION):
            logger.debug("Skill distillation: skipping non-completion response")
            return None
        response_body = dict(translated.body)
        usage = response_body.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        entry: JsonObject = {
            "served_model": served_model,
            "request": request,
            "response": response_body,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "messages": _messages_with_assistant(request, response_body),
        }
        routing = ctx.metadata.get(CTX_ROUTING)
        if isinstance(routing, dict):
            entry["routing"] = routing
        return entry


def build_skill_distillation_processors(
    session: SkillDistillationSessionCapture | None,
) -> tuple[list[Any], list[Any]]:
    """Return request/response processors for *session*, if configured."""

    if session is None:
        return [], []
    return [SkillDistillationRequestProcessor()], [
        SkillDistillationResponseProcessor(session),
    ]


def _messages_with_assistant(
    request: JsonObject,
    response: JsonObject,
) -> list[JsonObject]:
    messages = [dict(m) for m in request.get("messages", []) if isinstance(m, dict)]
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return messages
    message = choices[0].get("message")
    if isinstance(message, dict):
        messages.append(dict(message))
    return messages


__all__ = [
    "CTX_SKILL_DISTILLATION_REQUEST",
    "SkillDistillationRequestProcessor",
    "SkillDistillationResponseProcessor",
    "build_skill_distillation_processors",
]
