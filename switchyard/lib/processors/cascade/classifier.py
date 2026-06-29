# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM tier classifier — invoked on low-confidence turns, fails open to ``None``."""

from __future__ import annotations

import json
import logging
from importlib.resources import files
from typing import TYPE_CHECKING, Any, Protocol, cast

from switchyard.lib import metrics
from switchyard.lib.cost_estimator import estimate_model_cost
from switchyard.lib.llm_client import OpenAILLMClient

if TYPE_CHECKING:
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard_rust.components import ToolResultSignal

log = logging.getLogger(__name__)

#: Tier values returned by :meth:`TierClassifier.classify`. Mirrors picker constants.
STRONG_TIER: str = "strong"
WEAK_TIER: str = "weak"

#: ``ctx.metadata`` key for inbound conversation messages — read by the classifier.
RECENT_MESSAGES_KEY: str = "cascade_recent_messages"

#: Per-message char cap when rendering recent turns into the classifier prompt.
_MAX_MESSAGE_CHARS: int = 400


def _load_system_prompt() -> str:
    """Read the classifier system prompt from the prompts/ package-data file."""
    return (
        files("switchyard.lib.processors.cascade.prompts")
        .joinpath("tier_classifier.md")
        .read_text(encoding="utf-8")
        .strip()
    )


_SYSTEM_PROMPT: str = _load_system_prompt()


def _summarise(
    signal: ToolResultSignal,
    *,
    recent_messages: list[Any] | None = None,
) -> str:
    """Compact snapshot of agent state; appends trailing messages when present."""
    parts = [
        f"turn_depth={signal.turn_depth}",
        f"severity={signal.severity:.1f}",
        f"writes={signal.write_count}",
        f"edits={signal.edit_count}",
        f"reads={signal.read_count}",
        f"todowrites={signal.todowrite_count}",
        f"recent_writes={signal.recent_write_count}",
        f"recent_edits={signal.recent_edit_count}",
        f"recent_reads={signal.recent_read_count}",
        f"pure_bash_streak={signal.pure_bash_streak}",
        f"no_error_streak={signal.no_error_streak}",
        f"tests_passed={signal.tests_passed}",
    ]
    state_line = "State: " + ", ".join(parts)
    if not recent_messages:
        return "Decide STRONG or WEAK for the next call. " + state_line

    transcript_lines = ["Recent turns (most recent last):"]
    for msg in recent_messages:
        transcript_lines.append(_format_message(msg))
    return (
        "Decide STRONG or WEAK for the next call.\n"
        + state_line
        + "\n"
        + "\n".join(transcript_lines)
    )


def _format_message(msg: Any) -> str:
    """Render one message as ``[role] <one-line summary>``; truncated."""
    if not isinstance(msg, dict):
        return f"[?] {str(msg)[:_MAX_MESSAGE_CHARS]}"
    role = msg.get("role", "?")
    content = msg.get("content")
    if isinstance(content, str):
        body = content
    elif isinstance(content, list):
        snippets: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
                snippets.append(f"<tool_use:{block.get('name', '?')}>")
            elif block_type == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                    snippets.append(f"<tool_result:{str(inner[0].get('text', ''))[:120]}>")
                elif isinstance(inner, str):
                    snippets.append(f"<tool_result:{inner[:120]}>")
                else:
                    snippets.append("<tool_result>")
            elif block_type == "text":
                snippets.append(str(block.get("text", ""))[:_MAX_MESSAGE_CHARS])
        body = " ".join(snippets) if snippets else "(empty)"
    else:
        body = str(content)
    if len(body) > _MAX_MESSAGE_CHARS:
        body = body[:_MAX_MESSAGE_CHARS] + "..."
    return f"[{role}] {body}"


def _parse_tier(response: object) -> str | None:
    """Extract ``"strong"`` / ``"weak"`` from a chat-completion response, or ``None``.

    Mirrors ``_completion_content`` in
    :mod:`switchyard.lib.processors.llm_classifier.request_processor` — only reads
    ``message.content``. Empty content is treated as a failure (fall-open);
    the ``enable_thinking=False`` extra_body in :meth:`TierClassifier.classify`
    is what keeps reasoning-model output in ``content``.
    """
    try:
        content = response.choices[0].message.content  # type: ignore[attr-defined]
    except (AttributeError, IndexError):
        log.warning("classifier response missing choices/message/content")
        return None
    if not isinstance(content, str) or not content.strip():
        log.warning("classifier response had empty content (reasoning model misroute?)")
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        log.warning("classifier response not valid JSON: %r", content[:120])
        return None
    tier = payload.get("tier") if isinstance(payload, dict) else None
    if tier == STRONG_TIER or tier == WEAK_TIER:
        return tier
    log.warning("classifier returned unexpected tier %r", tier)
    return None


class _LLMClient(Protocol):
    async def acompletion(self, **kwargs: object) -> object: ...


class TierClassifier:
    """Async LLM classifier — one call per invocation; fails open to ``None``."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_secs: float = 30.0,
        recent_turn_window: int = 3,
        disable_reasoning: bool = True,
        client: _LLMClient | None = None,
    ) -> None:
        if recent_turn_window < 0:
            raise ValueError(f"recent_turn_window must be >= 0, got {recent_turn_window}")
        self._model = model
        self._api_key = api_key
        self._recent_turn_window = recent_turn_window
        # Reasoning models on the NVIDIA Inference Hub (DeepSeek V4 Flash /
        # V4 Pro, R1-style) misroute strict-`response_format` JSON into
        # `reasoning_content` while leaving `content` empty. Passing
        # `enable_thinking=False` via vLLM's `chat_template_kwargs` extra_body
        # restores `content`. Mirrors `LLMClassifierConfig.disable_reasoning`
        # from `switchyard/lib/processors/llm_classifier/`. Non-reasoning models
        # ignore the hint as a no-op, so the default is safe.
        self._disable_reasoning = disable_reasoning
        if client is None:
            client = cast(_LLMClient, OpenAILLMClient(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_secs,
                max_retries=0,
            ))
        self._client = client

    async def classify(self, ctx: ProxyContext, signal: ToolResultSignal) -> str | None:
        """Return ``"strong"``, ``"weak"``, or ``None`` (fall-open)."""
        recent_messages: list[Any] = []
        if self._recent_turn_window > 0:
            stashed = ctx.metadata.get(RECENT_MESSAGES_KEY) if hasattr(ctx, "metadata") else None
            if isinstance(stashed, list):
                recent_messages = stashed[-self._recent_turn_window:]
        user_prompt = _summarise(signal, recent_messages=recent_messages)
        extra_body: dict[str, Any] | None = None
        if self._disable_reasoning:
            # Mirror `OpenAIChatLLMClassifierClient` in
            # `switchyard/lib/processors/llm_classifier/`: vLLM-side hint that
            # forces reasoning models (DeepSeek V4 Flash / V4 Pro,
            # R1-style) to emit the JSON tier in `content` rather than
            # `reasoning_content`. Without this, `content` stays None on
            # the NVIDIA Inference Hub and every call falls open.
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        # 4096 matches the canonical LLM-classifier default — wide
        # enough to fit reasoning_content + JSON when the
        # `enable_thinking=False` hint is ignored upstream, but never
        # binds on non-reasoning models that emit ~10 tokens of JSON.
        try:
            response = await self._client.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=4096,
                extra_body=extra_body,
            )
        except Exception:
            log.warning("classifier call failed; falling open", exc_info=True)
            return None
        self._record_usage(response)
        return _parse_tier(response)

    def _record_usage(self, response: object) -> None:
        """Record classifier token + cost spend via OTel (``role=classifier``)."""
        try:
            usage = getattr(response, "usage", None)
            if usage is None:
                return
            details = getattr(usage, "prompt_tokens_details", None)
            prompt = getattr(usage, "prompt_tokens", 0) or 0
            completion = getattr(usage, "completion_tokens", 0) or 0
            cached = getattr(details, "cached_tokens", None) or 0
            metrics.record_tokens(
                model=self._model, tier="classifier",
                prompt=prompt, completion=completion, cached=cached,
            )
            costs = estimate_model_cost(self._model, prompt, completion, cached)
            for kind, key in (
                ("input", "base_input_cost"),
                ("cached", "cached_input_cost"),
                ("output", "output_cost"),
            ):
                value = costs.get(key, 0.0)
                if value:
                    metrics.record_cost(
                        model=self._model, tier="classifier", role="classifier",
                        kind=kind, cost_usd=value,
                    )
        except Exception:
            log.debug("failed to record cascade classifier usage", exc_info=True)


__all__ = ["STRONG_TIER", "TierClassifier", "WEAK_TIER"]
