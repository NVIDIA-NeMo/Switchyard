# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust-owned LLM target configuration helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from switchyard_rust.components import BackendFormat, EndpointConfig, LlmTarget

_DISABLE_THINKING_MODEL_FRAGMENTS = ("nemotron-3-super",)


def coerce_llm_target(value: object, *, default_id: str) -> LlmTarget:
    """Build a Rust ``LlmTarget`` from a target object or legacy mapping."""
    if isinstance(value, LlmTarget):
        if value.id == "default" and default_id != "default":
            return LlmTarget(
                id=default_id,
                model=value.model,
                format=value.format,
                endpoint=value.endpoint,
                extra_body=value.extra_body,
                extra_headers=value.extra_headers,
            )
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, dict):
        raise TypeError(f"expected LlmTarget or dict, got {type(value).__name__}")

    data: dict[str, Any] = dict(value)
    target_id = str(data.pop("id", default_id))
    model = data.pop("model", None)
    if not isinstance(model, str):
        raise TypeError("LlmTarget.model must be a string")

    target_format = data.pop("format", data.pop("backend_format", BackendFormat.OPENAI))
    endpoint = data.pop("endpoint", None)
    base_url = data.pop("base_url", None)
    api_key = data.pop("api_key", None)
    timeout_secs = data.pop("timeout_secs", data.pop("timeout", None))
    extra_body = data.pop("extra_body", None)
    extra_headers = data.pop("extra_headers", None)
    token_capture_engine = data.pop("token_capture_engine", None)
    data.pop("tuning", None)
    if data:
        unknown = ", ".join(sorted(data))
        raise ValueError(f"unknown LlmTarget fields: {unknown}")

    target = LlmTarget(
        id=target_id,
        model=model,
        format=target_format,
        endpoint=endpoint,
        base_url=base_url,
        api_key=api_key,
        timeout_secs=timeout_secs,
        extra_body=extra_body,
        extra_headers=extra_headers,
    )
    if token_capture_engine is not None:
        # Declaring the serving engine opts the target into token-capture
        # request params (pairs with `--enable-rl-logging` on the server).
        target = llm_target_with_token_capture(target, str(token_capture_engine))
    return target


def llm_target_with_format(target: LlmTarget, target_format: BackendFormat) -> LlmTarget:
    """Return ``target`` with a resolved backend format."""
    return LlmTarget(
        id=target.id,
        model=target.model,
        format=target_format,
        endpoint=target.endpoint,
        extra_body=target.extra_body,
        extra_headers=target.extra_headers,
    )


def llm_target_with_runtime_defaults(target: LlmTarget) -> LlmTarget:
    """Return ``target`` with Switchyard's safe per-model runtime defaults."""
    if target.extra_body:
        return target
    if not _should_disable_thinking(target.model):
        return target
    return LlmTarget(
        id=target.id,
        model=target.model,
        format=target.format,
        endpoint=target.endpoint,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        extra_headers=target.extra_headers,
    )


def _should_disable_thinking(model: str) -> bool:
    normalized = model.lower()
    return any(fragment in normalized for fragment in _DISABLE_THINKING_MODEL_FRAGMENTS)


_TOKEN_CAPTURE_ENGINE_PARAMS: dict[str, dict[str, Any]] = {
    # vLLM: `return_token_ids` emits `response.prompt_token_ids` + `choice.token_ids`;
    # `top_logprobs` must be non-None for `logprobs.content[]` to populate given
    # `logprobs=true` (0 returns just the sampled token's logprob).
    "vllm": {"logprobs": True, "return_token_ids": True, "top_logprobs": 0},
}


def llm_target_with_token_capture(target: LlmTarget, token_capture_engine: str) -> LlmTarget:
    """Return ``target`` with the engine's token-capture request params in ``extra_body``.

    These params must ride ``extra_body`` (merged after request translation in the
    backend) — request-side injection is dropped by the translation allowlist for
    cross-format traffic. Explicit ``extra_body`` keys on the target win over derived
    params; harness-side conflicts are resolved caller-wins at request time.
    """
    params = _TOKEN_CAPTURE_ENGINE_PARAMS.get(token_capture_engine)
    if params is None:
        supported = ", ".join(sorted(_TOKEN_CAPTURE_ENGINE_PARAMS))
        raise ValueError(
            f"unknown token_capture_engine {token_capture_engine!r} for token capture; supported: {supported}"
        )
    return LlmTarget(
        id=target.id,
        model=target.model,
        format=target.format,
        endpoint=target.endpoint,
        extra_body={**params, **(target.extra_body or {})},
        extra_headers=target.extra_headers,
    )


__all__ = [
    "BackendFormat",
    "EndpointConfig",
    "LlmTarget",
    "coerce_llm_target",
    "llm_target_with_format",
    "llm_target_with_runtime_defaults",
    "llm_target_with_token_capture",
]
