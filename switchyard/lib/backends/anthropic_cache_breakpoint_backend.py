# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic ephemeral prompt-caching wrapper backend.

When wired into a tier slot of
:class:`switchyard.lib.backends.deterministic_routing_llm_backend.DeterministicRoutingLLMBackend` (or any
multi-LLM-backend), this wrapper prepares each outbound call to its
inner :class:`AnthropicNativeBackend` so that Anthropic / Bedrock
honors prompt caching:

1. Translates the inbound :class:`ChatRequest` body to Anthropic
   Messages format (no-op if already Anthropic).
2. Injects ``cache_control: {"type": "ephemeral"}`` markers on (a)
   the system field's last text block and (b) the last message's
   last block — two breakpoints out of Anthropic's four-per-request
   budget. Injection is skipped when the inbound body already carries
   its own markers (an Anthropic-native client like Claude Code), so
   wrapping that path can never exceed the four-breakpoint cap.
3. Builds a **local** Anthropic-typed :class:`ChatRequest` and
   delegates to the inner backend, which sees an already-Anthropic
   request and passes the body through verbatim (preserving the
   markers all the way to the wire).

Why this lives in a wrapper instead of in
:class:`switchyard.lib.processors.turn_based_router_request_processor.TurnBasedRouterRequestProcessor`:
mutating the chain-level :attr:`ChatRequest.request_type` at the
request-processor layer propagates through the rest of the chain.
The terminal :class:`TranslationEngine` step then reads the modified
type, decides "the client wanted Anthropic," and skips translation
of the Anthropic response back to the client's original wire format.
OpenAI Chat / OpenAI Responses clients (Harbor's LiteLLM, OpenAI
SDK) receive an Anthropic-shape body and crash on missing
``choices``.

Keeping the translation **inside the backend** means the chain-level
request object stays in its inbound shape; the terminal translator
correctly converts the Anthropic response back to whatever the
client expected.  No translator or backend-Rust changes required —
PR #82's removal of global cache_control injection stands.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from switchyard.lib.roles import LLMBackend

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard_rust.core import ChatRequest, ChatResponse
    from switchyard_rust.translation import TranslationEngine


def maybe_wrap_anthropic_cache(backend: LLMBackend, target: Any) -> LLMBackend:
    """Wrap ``backend`` for prompt caching iff ``target`` is Anthropic-format.

    Shared by every deterministic-routing construction path (the
    ``DeterministicRoutingFactory`` and the ``serve --routing-profiles`` YAML
    bundle builder) so an OpenAI-shaped harness (Codex, OpenAI SDK) routed onto
    a Claude tier still gets ``cache_control`` breakpoints injected.

    ``target`` must already be format-resolved (``BackendFormat.AUTO`` is gone
    by this point). :class:`AnthropicCacheBreakpointBackend` is a Python-only
    backend.
    """
    from switchyard.lib.backends.llm_target import BackendFormat

    wrap = getattr(target, "format", None) == BackendFormat.ANTHROPIC
    log.info(
        "anthropic cache-wrap decision: model=%s resolved_format=%s wrapped=%s",
        getattr(target, "model", "?"), getattr(target, "format", "?"), wrap,
    )
    return AnthropicCacheBreakpointBackend(backend) if wrap else backend


class AnthropicCacheBreakpointBackend(LLMBackend):
    """Wrap an Anthropic-format LLMBackend to inject ephemeral cache breakpoints.

    Drop-in around the inner backend.  ``startup`` / ``shutdown`` /
    ``supported_request_types`` are forwarded; ``call`` is intercepted
    to prepare the body before delegation.
    """

    def __init__(self, inner: LLMBackend) -> None:
        self._inner = inner
        # Lazy: build the engine on first call, not at construction
        # time, so unit tests can construct the wrapper without
        # importing the Rust translation extension.
        self._engine: TranslationEngine | None = None

    async def startup(self) -> None:
        await self._inner.startup()

    async def shutdown(self) -> None:
        await self._inner.shutdown()

    async def call(
        self,
        ctx: ProxyContext,
        request: ChatRequest,
    ) -> ChatResponse:
        from switchyard_rust.core import ChatRequestType, request_with_type

        if request.request_type is ChatRequestType.ANTHROPIC:
            body = dict(request.body)
            # An Anthropic-native client (e.g. Claude Code) chooses its own
            # cache breakpoints. Respect them verbatim — re-marking could push
            # the request past Anthropic's 4-breakpoint cap — so only inject
            # when the client supplied none (e.g. a bare Anthropic SDK call).
            if _has_cache_control(body):
                return await self._inner.call(ctx, request)
        else:
            engine = self._engine
            if engine is None:
                from switchyard_rust.translation import TranslationEngine
                engine = TranslationEngine()
                self._engine = engine
            anthropic_request = engine.request_to("anthropic_messages", request)
            body = dict(anthropic_request.body)

        body = _inject_anthropic_cache_breakpoints(body)
        marked_request = request_with_type("anthropic", body)
        return await self._inner.call(ctx, marked_request)


def _inject_anthropic_cache_breakpoints(body: dict[str, Any]) -> dict[str, Any]:
    """Add ``cache_control: {"type": "ephemeral"}`` markers to an Anthropic body.

    Places at most two breakpoints (Anthropic supports four):

    * One **stable** anchor on the ``system`` field — caches the
      terminus-2 / Harbor system prompt across every turn of the
      task, since it doesn't change.
    * One **rolling** anchor on the last message — extends the
      cached prefix as the conversation grows, so each subsequent
      turn re-uses the prior turn's full history.

    Both markers are placed on the *last text block* of their
    container, which is what Anthropic's API uses as the cache
    boundary.  If the container is a plain string, it's converted to
    a single-block list with the marker — Anthropic accepts both
    string and list shapes for these fields, and the list shape is
    the only one that admits ``cache_control``.

    Returns a fresh dict; the input is not mutated.
    """
    body = dict(body)
    _mark_system_field(body)
    _mark_last_message(body)
    return body


def _mark_system_field(body: dict[str, Any]) -> None:
    """In-place: annotate ``body["system"]`` with a cache breakpoint."""
    system = body.get("system")
    if isinstance(system, str) and system:
        body["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            },
        ]
    elif isinstance(system, list) and system:
        new_system = [
            dict(blk) if isinstance(blk, dict) else blk for blk in system
        ]
        last_text = _last_text_block_index(new_system)
        if last_text is not None:
            new_system[last_text]["cache_control"] = {"type": "ephemeral"}
            body["system"] = new_system


def _mark_last_message(body: dict[str, Any]) -> None:
    """In-place: annotate the last text block in the last message.

    Falls back to the last block of any type if the last message has
    no text block (e.g., a pure tool_result message) — Anthropic
    allows ``cache_control`` on non-text blocks too.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return

    last_idx = len(messages) - 1
    last_msg = messages[last_idx]
    if not isinstance(last_msg, dict):
        return

    new_msg = dict(last_msg)
    content = new_msg.get("content")

    if isinstance(content, str) and content:
        new_msg["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            },
        ]
    elif isinstance(content, list) and content:
        new_content = [
            dict(blk) if isinstance(blk, dict) else blk for blk in content
        ]
        last_text = _last_text_block_index(new_content)
        if last_text is None:
            for i in range(len(new_content) - 1, -1, -1):
                blk = new_content[i]
                if isinstance(blk, dict):
                    blk["cache_control"] = {"type": "ephemeral"}
                    last_text = i
                    break
        else:
            new_content[last_text]["cache_control"] = {"type": "ephemeral"}
        if last_text is not None:
            new_msg["content"] = new_content
    else:
        return

    new_messages = list(messages)
    new_messages[last_idx] = new_msg
    body["messages"] = new_messages


def _has_cache_control(body: dict[str, Any]) -> bool:
    """True if an Anthropic body already carries a ``cache_control`` marker.

    Only list-shaped ``system`` / message ``content`` blocks can hold one;
    the plain-string forms never do. Used to leave a client's own
    breakpoints untouched instead of stacking ours on top.
    """
    if _blocks_have_cache_control(body.get("system")):
        return True
    messages = body.get("messages")
    if isinstance(messages, list):
        return any(
            isinstance(msg, dict) and _blocks_have_cache_control(msg.get("content"))
            for msg in messages
        )
    return False


def _blocks_have_cache_control(field: Any) -> bool:
    """True if ``field`` is a block list with any ``cache_control`` marker."""
    return isinstance(field, list) and any(
        isinstance(blk, dict) and "cache_control" in blk for blk in field
    )


def _last_text_block_index(blocks: list[Any]) -> int | None:
    """Return the index of the last ``{"type": "text", ...}`` block, or None."""
    for i in range(len(blocks) - 1, -1, -1):
        blk = blocks[i]
        if isinstance(blk, dict) and blk.get("type") == "text":
            return i
    return None


__all__ = ["AnthropicCacheBreakpointBackend", "maybe_wrap_anthropic_cache"]
