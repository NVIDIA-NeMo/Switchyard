# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python binding tests for the Rust-owned Gemini native backend."""

from __future__ import annotations

import pytest

from switchyard.lib.backends.multi_llm_backend import build_native_backend
from switchyard_rust.components import BackendFormat, GeminiNativeBackend, LlmTarget
from switchyard_rust.core import ChatRequestType


def _gemini_target(**overrides: object) -> LlmTarget:
    data: dict[str, object] = {
        "id": "gemini",
        "model": "gemini-2.5-flash",
        "format": BackendFormat.GEMINI,
        "base_url": "https://generativelanguage.googleapis.com",
        "api_key": "AIza-test",
        "timeout_secs": 12.5,
    }
    data.update(overrides)
    return LlmTarget(**data)


def test_constructs_from_resolved_gemini_target() -> None:
    target = _gemini_target()

    backend = GeminiNativeBackend(target)

    assert backend.target == target
    assert backend.target.endpoint.base_url == "https://generativelanguage.googleapis.com"
    assert backend.target.endpoint.api_key == "AIza-test"
    assert backend.target.endpoint.timeout_secs == 12.5


def test_supported_request_types_is_gemini_only() -> None:
    backend = GeminiNativeBackend(_gemini_target())

    assert backend.supported_request_types == [ChatRequestType.GEMINI]


@pytest.mark.parametrize(
    "target_format",
    [BackendFormat.OPENAI, BackendFormat.ANTHROPIC, BackendFormat.AUTO, "openai", "auto"],
)
def test_rejects_unresolved_or_non_gemini_target_format(target_format: object) -> None:
    with pytest.raises(RuntimeError, match="resolved Gemini format"):
        GeminiNativeBackend(_gemini_target(format=target_format))


def test_build_native_backend_selects_gemini_binding() -> None:
    backend = build_native_backend(_gemini_target())

    assert isinstance(backend, GeminiNativeBackend)


def test_target_is_immutable_after_binding_construction() -> None:
    target = _gemini_target()
    backend = GeminiNativeBackend(target)

    with pytest.raises(AttributeError):
        backend.target.model = "mutated"
    assert backend.target.model == "gemini-2.5-flash"


def test_request_types_are_value_objects_not_strings() -> None:
    request_type = GeminiNativeBackend(_gemini_target()).supported_request_types[0]

    assert request_type is ChatRequestType.GEMINI
    assert request_type.value == "gemini"
    assert not isinstance(request_type, str)
