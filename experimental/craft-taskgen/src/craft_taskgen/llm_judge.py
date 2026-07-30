# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct-API judge wrapper over `litellm.acompletion` through the NVIDIA gateway.

Pattern-matches `search/synthesize.py::_llm_call` (proven in production):
completion → strip markdown fences → `json.loads` → `jsonschema.validate`. On
parse or schema error, one retry with the error appended as a system message.
On transient gateway errors (timeout / 5xx / rate-limit), exponential-backoff
retry up to 5 times (mirrors `claude_cli.run_claude_async`).

All models route through the OpenAI-compatible gateway endpoint — CRAFT's
NVIDIA gateway accepts Opus, GPT, Gemini, Codex via a single route with the
`openai/` prefix. `_normalize_model` prepends the prefix when a caller passes
a bare name so configs can stay concise.

Phase A note: this module does not use structured-output forcing (tool-use with
`strict: true`, OpenAI `response_format`). The manual-parse-plus-validate path
is what `search/synthesize.py` runs in production; we match it. Switching to
structured-output enforcement is a Phase B follow-up gated on parse-error rate
in bulk runs.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import jsonschema
import litellm

from craft_taskgen.gateway import get_openai_gateway_creds

litellm.suppress_debug_info = True

_MAX_TRANSIENT_RETRIES = 5
_BACKOFF_BASE = 5.0
_BACKOFF_CAP = 30.0
_DEFAULT_TIMEOUT_S = 120
_DEFAULT_MAX_TOKENS = 4096

# Transient errors are worth retrying. Non-transient (auth, bad request,
# context-window) bubble up on the first miss.
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    litellm.Timeout,
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)


@dataclass
class JudgeResult:
    """Outcome of a successful `judge()` call."""

    result: dict[str, Any]
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    latency_s: float = 0.0


async def judge(
    *,
    prompt: str,
    schema: dict,
    model: str,
    system_prompt: str | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> JudgeResult:
    """Dispatch a judge call through the NVIDIA gateway with schema validation.

    Parameters
    ----------
    prompt
        User-turn content, pre-assembled by the caller (no tool use inside
        the call; all context must be in the string).
    schema
        JSON Schema dict; response `result` is validated against it.
    model
        Gateway model identifier (e.g. `aws/anthropic/bedrock-claude-opus-4-6`
        or `us/azure/openai/gpt-5.4`). `openai/` prefix added automatically
        if missing — all gateway routes go through litellm's OpenAI transport.
    system_prompt
        Optional system-role content placed ahead of the user turn.
    max_tokens, timeout_s
        Model defaults otherwise. No explicit temperature or thinking-mode
        parameters — those are left to the provider for portability across
        Opus / GPT / Codex.

    Returns
    -------
    `JudgeResult` with schema-validated `result`, token usage, dispatched
    model string, and wall-clock latency for the successful call.

    Raises
    ------
    RuntimeError
        If (a) all transient retries exhaust, (b) the response fails to parse
        or validate even after one feedback-retry, or (c) a non-transient
        litellm error surfaces.
    """
    api_key, base_url = get_openai_gateway_creds()
    normalized_model = _normalize_model(model)
    messages = _build_messages(prompt, system_prompt)

    parse_retry_used = False

    for transient_attempt in range(_MAX_TRANSIENT_RETRIES + 1):
        try:
            start = time.monotonic()
            response = await litellm.acompletion(
                model=normalized_model,
                messages=messages,
                max_tokens=max_tokens,
                api_key=api_key,
                api_base=base_url,
                num_retries=0,  # this wrapper owns retry bookkeeping
                timeout=timeout_s,
            )
            latency = time.monotonic() - start
        except _TRANSIENT_EXCEPTIONS:
            if transient_attempt >= _MAX_TRANSIENT_RETRIES:
                raise
            await asyncio.sleep(_jittered_backoff(transient_attempt))
            continue

        raw = response.choices[0].message.content or ""
        text = _strip_markdown_fences(raw)

        try:
            data = json.loads(text)
            jsonschema.validate(data, schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as err:
            if parse_retry_used:
                raise RuntimeError(
                    f"Judge response failed parse/schema validation after retry: {err}"
                ) from err
            parse_retry_used = True
            messages = _build_messages(
                prompt,
                system_prompt,
                extra_system=(
                    "The previous response did not satisfy the required JSON schema: "
                    f"{err}. Return only a single JSON object that matches the schema. "
                    "Do not wrap the JSON in markdown fences."
                ),
            )
            # Fall through to retry loop; this counts as a fresh transient
            # attempt since we're rebuilding the request.
            continue

        usage = _extract_usage(response)
        return JudgeResult(
            result=data,
            usage=usage,
            model=normalized_model,
            latency_s=latency,
        )

    # Exhausted the retry loop without either succeeding or raising — defensive.
    raise RuntimeError(f"llm_judge.judge exited retry loop without a result (model={normalized_model})")


def _normalize_model(model: str) -> str:
    """Ensure the model string routes through the OpenAI-compatible litellm transport."""
    return model if model.startswith("openai/") else f"openai/{model}"


def _build_messages(
    prompt: str,
    system_prompt: str | None,
    *,
    extra_system: str | None = None,
) -> list[dict[str, str]]:
    # Merge system_prompt and extra_system into a single system message.
    # Two back-to-back role="system" entries are handled inconsistently by
    # the OpenAI-compatible gateway (ignored, concatenated, or 400), which
    # would defeat the parse-retry for the build step — its only caller
    # that passes system_prompt=.
    messages: list[dict[str, str]] = []
    system_parts = [part for part in (system_prompt, extra_system) if part]
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    messages.append({"role": "user", "content": prompt})
    return messages


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` fences if the model wrapped its output.

    Matches the pattern in `search/synthesize.py:165`; kept inline so `llm_judge`
    has no import dependency on the search package.
    """
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _extract_usage(response: Any) -> dict[str, int]:
    """Pull input/output/cached token counts out of a litellm response."""
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    if hasattr(usage_obj, "model_dump"):
        usage = usage_obj.model_dump()
    elif isinstance(usage_obj, dict):
        usage = usage_obj
    else:
        usage = {k: getattr(usage_obj, k, 0) for k in ("prompt_tokens", "completion_tokens")}
    cached = (
        usage.get("cache_read_input_tokens")
        or usage.get("cached_tokens")
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or 0
    )
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "cached_tokens": int(cached),
    }


def _jittered_backoff(attempt: int) -> float:
    """Exponential backoff with jitter — base 5s, cap 30s, matches claude_cli."""
    base = min(_BACKOFF_BASE * (2**attempt), _BACKOFF_CAP)
    return base * (0.5 + random.random())
