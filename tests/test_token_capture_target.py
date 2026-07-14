# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-capture engine params derived into ``LlmTarget.extra_body``."""

from __future__ import annotations

import pytest

from switchyard.lib.backends.llm_target import (
    BackendFormat,
    LlmTarget,
    coerce_llm_target,
    llm_target_with_token_capture,
)


def _target(**overrides: object) -> LlmTarget:
    params: dict[str, object] = {
        "id": "policy",
        "model": "Qwen/Qwen3-0.6B",
        "format": BackendFormat.OPENAI,
        "base_url": "https://example.invalid/v1",
        "api_key": "sk-test",
    }
    params.update(overrides)
    return LlmTarget(**params)


class TestLlmTargetWithTokenCapture:
    def test_vllm_params_derived_into_extra_body(self) -> None:
        target = llm_target_with_token_capture(_target(), "vllm")

        assert target.extra_body == {
            "logprobs": True,
            "return_token_ids": True,
            "top_logprobs": 0,
        }

    def test_explicit_extra_body_key_wins_over_derived(self) -> None:
        target = llm_target_with_token_capture(_target(extra_body={"top_logprobs": 5}), "vllm")

        assert target.extra_body == {
            "logprobs": True,
            "return_token_ids": True,
            "top_logprobs": 5,
        }

    def test_unrelated_extra_body_keys_preserved(self) -> None:
        target = llm_target_with_token_capture(
            _target(extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
            "vllm",
        )

        assert target.extra_body["chat_template_kwargs"] == {"enable_thinking": False}
        assert target.extra_body["return_token_ids"] is True

    def test_target_identity_fields_preserved(self) -> None:
        source = _target()
        target = llm_target_with_token_capture(source, "vllm")

        assert target.id == source.id
        assert target.model == source.model
        assert target.format == source.format

    def test_unknown_engine_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown token_capture_engine 'tgi'"):
            llm_target_with_token_capture(_target(), "tgi")


class TestCoerceLlmTargetTokenCaptureEngine:
    def test_token_capture_engine_field_derives_params(self) -> None:
        target = coerce_llm_target(
            {
                "model": "Qwen/Qwen3-0.6B",
                "base_url": "https://example.invalid/v1",
                "api_key": "sk-test",
                "token_capture_engine": "vllm",
            },
            default_id="policy",
        )

        assert target.extra_body == {
            "logprobs": True,
            "return_token_ids": True,
            "top_logprobs": 0,
        }

    def test_without_token_capture_engine_no_params(self) -> None:
        target = coerce_llm_target(
            {"model": "m", "base_url": "https://example.invalid/v1", "api_key": "k"},
            default_id="policy",
        )

        assert not target.extra_body
