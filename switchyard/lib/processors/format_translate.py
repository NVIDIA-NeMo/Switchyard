# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router-agnostic format-translation processors for the IGW path.

When Switchyard owns the LLM call (standalone server), inbound→target
format translation happens *inside* the LLM backend (e.g.
:class:`OpenAiNativeBackend.call` runs
:meth:`TranslationEngine.request_to_any_of` at the top), and
outbound translation happens in :class:`TranslationEngine`.
The host's chain has dedicated slots for both.

When the host (NMP IGW) owns the LLM call, neither slot is available
— IGW supplies its own backend and there's no
:class:`TranslationEngine` slot. Format translation has to live in
processors instead. The three processors here are the IGW equivalent
of the standalone path, and reusable by *any* router (random routing
today, RouteLLM later) that needs cross-format dispatch:

* :class:`StampOriginalFormatProcessor` — runs at the head of the
  chain and captures the inbound :class:`ChatRequestType` into
  ``ctx.metadata[CTX_ORIGINAL_FORMAT]`` so the response processor can
  translate back later.
* :class:`FormatTranslateRequestProcessor` — runs after a router
  picks a tier and reads ``ctx.metadata[CTX_TARGET_FORMAT]``.
  No-ops when the target is unset or matches
  ``request.request_type``; otherwise delegates to
  :meth:`TranslationEngine.request_to_any_of`.
* :class:`FormatTranslateResponseProcessor` — runs after the host
  backend returns. Reads ``ctx.metadata[CTX_ORIGINAL_FORMAT]`` and
  converts the response back to the client's inbound format via
  :class:`TranslationEngine`.

Composability: any router that wants cross-format support stamps a
:class:`ChatRequestType` into ``CTX_TARGET_FORMAT``; these processors
handle the actual conversion. Router and converter are fully
decoupled — neither imports the other.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict

from switchyard.lib.backends.llm_target import BackendFormat
from switchyard.lib.proxy_context import (
    CTX_ORIGINAL_FORMAT,
    CTX_ORIGINAL_REQUEST,
    CTX_PROXY_ACTUAL_MODEL,
    CTX_TARGET_FORMAT,
)
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    request_type_matches,
    request_type_value,
    request_with_type,
    response_is_streaming,
    response_matches_request_type,
)
from switchyard_rust.translation import TranslationEngine

if TYPE_CHECKING:
    from anthropic.types.message_create_params import MessageCreateParamsBase
    from openai.types.chat.completion_create_params import CompletionCreateParamsBase
    from openai.types.responses.response_create_params import ResponseCreateParamsBase

    from switchyard.lib.proxy_context import ProxyContext

log = logging.getLogger(__name__)


class TranslateConfig(BaseModel):
    """Model-to-format map used by IGW-owned translation processors."""

    model_config = ConfigDict(frozen=True)

    models: list[dict[str, str]] = []


class ModelFormatLookupProcessor:
    """Look up the selected model's target format from the translate config.

    After a router picks a tier and stamps its model name into
    ``ctx.selected_model``, this processor looks up the model in the
    translate config's models list and stamps the corresponding backend
    format into ``ctx.metadata[CTX_TARGET_FORMAT]``. The legacy
    ``ctx.metadata[CTX_PROXY_ACTUAL_MODEL]`` key remains a fallback for
    Python-only routers that have not migrated yet.

    ``BackendFormat.AUTO`` is resolved locally because the IGW path has
    no Switchyard-owned backend slot where backend construction can probe
    endpoint capabilities. The resolver keeps Anthropic-native inbound
    traffic native, and otherwise normalizes to OpenAI Chat as the broad
    fallback wire format.

    No-op when the model is not found in the config (passthrough).
    """

    def __init__(self, config: TranslateConfig) -> None:
        self._config = config
        # Build a lookup dict: model name → backend_format
        self._model_to_format: dict[str, BackendFormat] = {}
        for entry in config.models:
            model_name = entry.get("model")
            format_str = entry.get("backend_format")
            if model_name and format_str:
                try:
                    self._model_to_format[model_name] = BackendFormat(format_str)
                except ValueError:
                    log.warning(
                        "Unknown backend_format in translate config: %s", format_str,
                    )

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        selected_model = ctx.selected_model or ctx.metadata.get(CTX_PROXY_ACTUAL_MODEL)
        if not isinstance(selected_model, str) or not selected_model:
            return request
        target_format = self._model_to_format.get(selected_model)
        if target_format is not None:
            target_request_type = _target_request_type_for_backend_format(
                target_format, request,
            )
            ctx.metadata[CTX_TARGET_FORMAT] = target_request_type
            log.info(
                "ModelFormatLookupProcessor: model=%s format=%s "
                "target_request_type=%s",
                selected_model,
                target_format.value,
                target_request_type.value,
            )
        return request


class StampOriginalFormatProcessor:
    """Capture the inbound request type into ``ctx.metadata``.

    Place at the head of any IGW chain that may rewrite the request
    format before the host backend sees it. The matching response
    processor reads ``CTX_ORIGINAL_FORMAT`` to know what format to
    translate the response back to.
    """

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        ctx.metadata[CTX_ORIGINAL_FORMAT] = request.request_type
        if CTX_ORIGINAL_REQUEST not in ctx.metadata:
            body = getattr(request, "body", None)
            if isinstance(body, Mapping):
                ctx.metadata[CTX_ORIGINAL_REQUEST] = deepcopy(dict(body))
        return request


class FormatTranslateRequestProcessor:
    """Translate the request to the target wire format chosen by a router.

    Reads ``ctx.metadata[CTX_TARGET_FORMAT]`` (a
    :class:`ChatRequestType`) and routes through
    :meth:`TranslationEngine.request_to_any_of` when the target
    differs from ``request.request_type``. Pure passthrough when:

    * No router stamped a target (``CTX_TARGET_FORMAT`` absent).
    * Target matches the inbound type (already in the right format).

    Router-agnostic — no notion of *why* a target was chosen, only
    what to do with it. Reusable by every routing factory that needs
    cross-format dispatch.
    """

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        target = ctx.metadata.get(CTX_TARGET_FORMAT)
        if target is None:
            return request
        if not _is_request_type(target):
            raise TypeError(
                f"ctx.metadata[{CTX_TARGET_FORMAT!r}] must be a ChatRequestType, "
                f"got {type(target).__name__}",
            )
        if request_type_matches(request, target):
            return request
        return TranslationEngine().request_to_any_of(request, [target])


class FormatTranslateResponseProcessor:
    """Translate the response back to the inbound format the client expects.

    Reads ``ctx.metadata[CTX_ORIGINAL_FORMAT]`` (stamped by
    :class:`StampOriginalFormatProcessor` at the head of the chain)
    and converts the response via
    :class:`TranslationEngine`. No-ops when the response
    is already in the original format or when ``CTX_ORIGINAL_FORMAT``
    is absent.

    Streaming responses use the original request body captured in
    context to preserve target-format stream metadata such as model
    names while translating lazily.
    """

    async def process(self, ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        original = ctx.metadata.get(CTX_ORIGINAL_FORMAT)
        if original is None:
            return response
        if not _is_request_type(original):
            raise TypeError(
                f"ctx.metadata[{CTX_ORIGINAL_FORMAT!r}] must be a ChatRequestType, "
                f"got {type(original).__name__}",
            )
        if response_is_streaming(response):
            return _translate_streaming_response(ctx, response, original)
        if _matches_format(response, original):
            return response
        if request_type_value(original) == request_type_value(ChatRequestType.OPENAI_CHAT):
            return TranslationEngine().response_to(ChatRequestType.OPENAI_CHAT, response)
        if request_type_value(original) == request_type_value(ChatRequestType.ANTHROPIC):
            return TranslationEngine().response_to(ChatRequestType.ANTHROPIC, response)
        if request_type_value(original) == request_type_value(ChatRequestType.OPENAI_RESPONSES):
            return TranslationEngine().response_to(
                ChatRequestType.OPENAI_RESPONSES,
                response,
                original_body=_original_body(ctx),
            )
        if request_type_value(original) == request_type_value(ChatRequestType.GEMINI):
            return TranslationEngine().response_to(ChatRequestType.GEMINI, response)
        raise NotImplementedError(
            f"Unsupported original format for response translation: {original!r}",
        )


def _translate_streaming_response(
    ctx: ProxyContext,
    response: ChatResponse,
    original: ChatRequestType | str,
) -> ChatResponse:
    if _matches_streaming_format(response, original):
        return response
    return TranslationEngine().response_to(
        original,
        response,
        original_body=_original_body(ctx),
    )


def _request_for_original_format(
    original: ChatRequestType | str,
    body: dict[str, object],
) -> ChatRequest:
    if request_type_value(original) == request_type_value(ChatRequestType.OPENAI_CHAT):
        return request_with_type(original, cast("CompletionCreateParamsBase", body))
    if request_type_value(original) == request_type_value(ChatRequestType.ANTHROPIC):
        return request_with_type(original, cast("MessageCreateParamsBase", body))
    if request_type_value(original) == request_type_value(ChatRequestType.OPENAI_RESPONSES):
        return request_with_type(original, cast("ResponseCreateParamsBase", body))
    if request_type_value(original) == request_type_value(ChatRequestType.GEMINI):
        return request_with_type(original, body)
    raise NotImplementedError(f"Unsupported original format: {original!r}")


def _original_body(ctx: ProxyContext) -> dict[str, object]:
    body = ctx.metadata.get(CTX_ORIGINAL_REQUEST)
    if isinstance(body, Mapping):
        return dict(body)
    return {}


def _matches_format(response: ChatResponse, target: ChatRequestType | str) -> bool:
    """Return ``True`` when *response* is already in *target*'s wire format."""
    return bool(response_matches_request_type(response, target)) and not bool(
        response_is_streaming(response)
    )


def _matches_streaming_format(response: ChatResponse, target: ChatRequestType | str) -> bool:
    return bool(response_matches_request_type(response, target)) and bool(
        response_is_streaming(response)
    )


def _is_request_type(value: object) -> bool:
    try:
        request_type_value(cast("ChatRequestType | str", value))
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _target_request_type_for_backend_format(
    backend_format: BackendFormat,
    request: ChatRequest,
) -> ChatRequestType:
    """Resolve a configured backend wire format to a request format.

    ``BackendFormat.AUTO`` on the processor-only IGW path cannot do the
    backend-owned capability probe used by backend factories. It instead
    chooses the least surprising concrete request shape:

    * Anthropic inbound remains Anthropic, preserving native Claude
      fields instead of translating them away.
    * OpenAI Chat inbound remains OpenAI Chat.
    * Responses-capable targets preserve OpenAI Responses instead of
      normalizing through Chat.
    * Gemini inbound remains Gemini, preserving native generateContent
      fields instead of translating them away.
    """
    if backend_format == BackendFormat.OPENAI:
        return ChatRequestType.OPENAI_CHAT
    if backend_format == BackendFormat.RESPONSES:
        return ChatRequestType.OPENAI_RESPONSES
    if backend_format == BackendFormat.ANTHROPIC:
        return ChatRequestType.ANTHROPIC
    if backend_format == BackendFormat.GEMINI:
        return ChatRequestType.GEMINI
    if backend_format == BackendFormat.AUTO:
        if request_type_matches(request, ChatRequestType.ANTHROPIC):
            return ChatRequestType.ANTHROPIC
        if request_type_matches(request, ChatRequestType.GEMINI):
            return ChatRequestType.GEMINI
        return ChatRequestType.OPENAI_CHAT
    raise ValueError(f"Unsupported backend_format: {backend_format!r}")
