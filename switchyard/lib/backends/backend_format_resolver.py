# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve generic backend tier formats into concrete backend wire formats.

``BackendFormat.AUTO`` probes the upstream in priority order:
  1. OpenAI Chat Completions (``/v1/chat/completions``) → ``OPENAI``
  2. Anthropic Messages (``/v1/messages``) → ``ANTHROPIC``
  3. OpenAI Responses (``/v1/responses``) → ``RESPONSES``
  4. Fallback → ``OPENAI`` (Chat Completions, assumed universal)

Chat Completions is probed first because it is the most widely supported format.
Endpoints that bridge multiple API surfaces (e.g. NVIDIA Inference Hub via
LiteLLM) will satisfy all three probes; preferring Chat Completions avoids
silently upgrading NIM models to Anthropic Messages format. Anthropic-native
endpoints (api.anthropic.com) return 404 for Chat Completions, so they correctly
fall through to the Anthropic probe.

The TranslationEngine converts any inbound format to any backend format through
a neutral IR, so all (inbound, backend) combinations are valid regardless of
which format the client uses.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from switchyard.lib import startup_timing
from switchyard.lib.backends.llm_target import (
    BackendFormat,
    LlmTarget,
)

log = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class BackendFormatResolution:
    """Concrete backend format selected for a generic tier."""

    format: BackendFormat
    reason: str


class BackendFormatResolver:
    """Resolve ``BackendFormat.AUTO`` through reusable capability probes."""

    @staticmethod
    def resolve(tier: LlmTarget) -> BackendFormatResolution:
        """Return the concrete backend format for ``tier``.

        Explicit formats are already resolved. ``AUTO`` needs a real endpoint
        probe, so missing probe inputs fail fast instead of silently picking
        a backend that may only work by accident.
        """
        if tier.format != BackendFormat.AUTO:
            return BackendFormatResolution(
                format=tier.format,
                reason="backend format is explicitly configured",
            )

        return BackendFormatResolver._resolve_auto(tier)

    @staticmethod
    def _resolve_auto(tier: LlmTarget) -> BackendFormatResolution:
        if not tier.endpoint.base_url:
            raise ValueError(
                "format='auto' requires base_url so Switchyard can probe upstream capabilities.",
            )
        if not tier.endpoint.api_key:
            raise ValueError(
                "format='auto' requires api_key so Switchyard can probe upstream capabilities.",
            )

        if _model_is_anthropic(tier.model):
            return BackendFormatResolution(
                format=BackendFormat.ANTHROPIC,
                reason="model prefix indicates native Anthropic; skipping probes",
            )

        timeout_s = tier.endpoint.timeout_secs or _DEFAULT_TIMEOUT_S

        # startup_timing marks let `switchyard launch --startup-timing` show each
        # probe as its own line, so a slow AUTO detection names which route stalled.
        startup_timing.mark("chain init")

        # Chat Completions is probed first: it is the most widely supported
        # format and the universal fallback. A transport timeout (the endpoint
        # is reachable but slow, e.g. a cold NVCF/Azure deployment) is a
        # different signal from a fast "not wired" 404. On a timeout, assume
        # Chat Completions rather than serially probing the rarer /v1/messages
        # and /v1/responses routes — each of those would stack another timeout
        # and turn one slow startup into several seconds. A 404 (even a slow
        # one) still falls through to the probes below, so Anthropic-native and
        # Responses-only endpoints are detected as before.
        try:
            chat_supported = probe_openai_chat_completions_support_sync(
                base_url=tier.endpoint.base_url,
                api_key=tier.endpoint.api_key,
                model=tier.model,
                timeout_s=timeout_s,
            )
        except httpx.TimeoutException:
            startup_timing.mark("probe: /v1/chat/completions (timed out)")
            return BackendFormatResolution(
                format=BackendFormat.OPENAI,
                reason="chat-completions probe timed out; assuming Chat Completions",
            )
        startup_timing.mark("probe: /v1/chat/completions")
        if chat_supported:
            return BackendFormatResolution(
                format=BackendFormat.OPENAI,
                reason="upstream /v1/chat/completions probe succeeded",
            )

        anthropic_supported = probe_anthropic_messages_support_sync(
            base_url=tier.endpoint.base_url,
            api_key=tier.endpoint.api_key,
            model=tier.model,
            timeout_s=timeout_s,
        )
        startup_timing.mark("probe: /v1/messages")
        if anthropic_supported:
            return BackendFormatResolution(
                format=BackendFormat.ANTHROPIC,
                reason="upstream /v1/messages probe succeeded; Chat Completions not available",
            )

        responses_supported = probe_openai_responses_support_sync(
            base_url=tier.endpoint.base_url,
            api_key=tier.endpoint.api_key,
            model=tier.model,
            timeout_s=timeout_s,
        )
        startup_timing.mark("probe: /v1/responses")
        if responses_supported:
            return BackendFormatResolution(
                format=BackendFormat.RESPONSES,
                reason="upstream /v1/responses probe succeeded; Chat Completions not available",
            )

        return BackendFormatResolution(
            format=BackendFormat.OPENAI,
            reason="all probes failed; assuming Chat Completions",
        )


def _model_is_anthropic(model: str | None) -> bool:
    """Return True if the model ID prefix unambiguously identifies a native Anthropic model.

    Matches ``anthropic/<name>`` and ``claude<…>`` prefixes only — these map
    exclusively to the Anthropic API or OpenRouter's direct Anthropic passthrough,
    both of which require ``/v1/messages``.

    Gateway-namespaced paths like ``aws/anthropic/bedrock-…`` and
    ``openrouter/anthropic/…`` are intentionally NOT matched: those gateways
    also expose Chat Completions, so probing is preferred over assuming.
    """
    if not model:
        return False
    m = model.lower()
    return m.startswith("anthropic/") or m.startswith("claude")


def probe_openai_chat_completions_support_sync(
    *,
    base_url: str,
    api_key: str,
    model: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> bool:
    """Return True iff ``{base_url}/chat/completions`` is a functional route.

    Sends a minimal-body probe POST with Bearer auth.  A 404 means the
    Chat Completions endpoint is not wired — the caller should probe
    Anthropic Messages or Responses next.

    Raises ``httpx.TimeoutException`` on a transport timeout. A timeout means
    the endpoint is reachable but slow (e.g. a cold NVCF/Azure deployment),
    which is a different signal from a fast "not wired" 404: the resolver
    stops probing and assumes Chat Completions instead of stacking the slower
    ``/v1/messages`` and ``/v1/responses`` probes.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}
    req_body: dict[str, Any] = {
        "messages": [{"role": "user", "content": " "}],
        "max_tokens": 1,
        **({"model": model} if model else {}),
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, headers=headers, json=req_body)
    except httpx.TimeoutException:
        raise
    except httpx.RequestError as e:
        log.warning(
            "OpenAI /v1/chat/completions probe failed (%s); "
            "falling back to Anthropic/Responses probes.",
            type(e).__name__,
        )
        return False
    if resp.status_code == 404:
        return False
    if resp.status_code == 401:
        log.warning(
            "OpenAI /v1/chat/completions probe got HTTP 401 — check --api-key. "
            "Falling back to Anthropic/Responses probes.",
        )
        return False
    if 200 <= resp.status_code < 500:
        return True
    log.warning(
        "OpenAI /v1/chat/completions probe got HTTP %d; "
        "falling back to Anthropic/Responses probes.",
        resp.status_code,
    )
    return False


def _interpret_status(status: int, body: bytes = b"") -> bool:
    """Return True iff the status code indicates the route is wired.

    When ``body`` is provided, a 400/422 that names the model as not found or
    unsupported is treated as a probe failure — the route exists but this model
    is not valid for it.  Without a body (legacy call-sites) the old behaviour
    is preserved.
    """
    if status == 404:
        return False
    if status == 401:
        log.warning(
            "Anthropic /v1/messages probe got HTTP 401 — check --api-key. "
            "Falling back to translation mode.",
        )
        return False
    if 200 <= status < 300:
        return True
    if status in (400, 422) and body:
        if _body_signals_model_error(body):
            log.debug(
                "Anthropic /v1/messages probe got HTTP %d with model error; "
                "model is unsupported on this endpoint.",
                status,
            )
            return False
        return True  # validation error about other fields — route and model exist
    if 200 <= status < 500:
        return True
    log.warning(
        "Anthropic /v1/messages probe got HTTP %d; falling back to translation mode.",
        status,
    )
    return False


def _body_signals_model_error(body: bytes) -> bool:
    """Return True if the JSON error body indicates the model is not available."""
    try:
        data = json.loads(body)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    error = data.get("error")
    if not isinstance(error, dict):
        return False
    if error.get("type") in ("not_found_error",):
        return True
    msg = (error.get("message") or "").lower()
    return "model" in msg and any(
        kw in msg for kw in ("not found", "not supported", "unsupported", "unknown", "invalid")
    )


def _probe_headers(api_key: str) -> dict[str, str]:
    # Include both auth styles: native Anthropic uses x-api-key, but
    # OpenAI-compatible gateways (e.g. NVIDIA Inference Hub) that also expose
    # /v1/messages expect Authorization: Bearer.
    return {
        "x-api-key": api_key,
        "authorization": f"Bearer {api_key}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def strip_v1_suffix(base_url: str) -> str:
    """Return *base_url* with a trailing ``/v1`` path component removed.

    Switchyard's ``--base-url`` convention follows OpenAI's (e.g.
    ``https://openrouter.ai/api/v1``), but the Anthropic SDK and
    raw ``/v1/messages`` probing both treat the base URL as the API
    root and append ``/v1/messages`` themselves. Without this trim the
    two conventions collide — ``https://host/v1`` + ``/v1/messages``
    becomes ``https://host/v1/v1/messages``.
    """
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        return stripped[:-3]
    return stripped


def probe_openai_responses_support_sync(
    *,
    base_url: str,
    api_key: str,
    model: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> bool:
    """Return True iff ``{base_url}/responses`` is a functional route.

    Sends a minimal-body probe POST with Bearer auth.  A 404 means the
    Responses endpoint is not wired upstream; callers should fall back to
    ``BackendFormat.OPENAI`` (Chat Completions).  Non-OpenAI upstreams
    (e.g. NVIDIA NIM) commonly 404 here even when they support Chat
    Completions.
    """
    url = f"{base_url.rstrip('/')}/responses"
    headers = {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}
    req_body: dict[str, Any] = {
        "input": "",
        "stream": False,
        **({"model": model} if model else {}),
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, headers=headers, json=req_body)
    except httpx.RequestError as e:
        log.warning(
            "OpenAI /v1/responses probe failed (%s); "
            "falling back to Chat Completions format.",
            type(e).__name__,
        )
        return False
    if resp.status_code == 404:
        return False
    if resp.status_code == 401:
        log.warning(
            "OpenAI /v1/responses probe got HTTP 401; "
            "falling back to Chat Completions format.",
        )
        return False
    if 200 <= resp.status_code < 500:
        return True
    log.warning(
        "OpenAI /v1/responses probe got HTTP %d; falling back to Chat Completions format.",
        resp.status_code,
    )
    return False


def probe_anthropic_messages_support_sync(
    *,
    base_url: str,
    api_key: str,
    model: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> bool:
    """Synchronous version of :func:`probe_anthropic_messages_support`.

    Preferred at startup (no running event loop required). Uses
    ``httpx.Client`` so no asyncio event loop is created; async clients
    built afterward bind their connection pools to uvicorn's event loop
    on first use rather than to a now-closed startup loop.
    """
    url = f"{strip_v1_suffix(base_url)}/v1/messages"
    req_body: dict[str, Any] = {
        "messages": [{"role": "user", "content": " "}],
        "max_tokens": 1,
        **({"model": model} if model else {}),
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, headers=_probe_headers(api_key), json=req_body)
    except httpx.RequestError as e:
        log.warning(
            "Anthropic /v1/messages probe failed (%s); "
            "falling back to translation mode.",
            type(e).__name__,
        )
        return False
    return _interpret_status(resp.status_code, resp.content)


async def probe_anthropic_messages_support(
    *,
    base_url: str,
    api_key: str,
    model: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> bool:
    """Return True iff ``{base_url}/v1/messages`` is a functional route.

    Sends a model-scoped probe POST with real auth. Response interpretation:

    * 404 → route not wired → return False (use translation)
    * 401 → return False (credential validation happens on the real request)
    * 400 / 422 with model error body → model unsupported → return False
    * 400 / 422 with field error body → route exists → return True
    * 200 → route exists → return True
    * 5xx / timeout / network error → return False
    """
    url = f"{strip_v1_suffix(base_url)}/v1/messages"
    req_body: dict[str, Any] = {
        "messages": [{"role": "user", "content": " "}],
        "max_tokens": 1,
        **({"model": model} if model else {}),
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, headers=_probe_headers(api_key), json=req_body)
    except httpx.RequestError as e:
        log.debug(
            "Anthropic /v1/messages probe unavailable (%s); "
            "using OpenAI translation mode.",
            type(e).__name__,
        )
        return False

    return _interpret_status(resp.status_code, resp.content)
