# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for outbound telemetry version headers."""

from __future__ import annotations

import importlib.metadata
from typing import Any
from unittest.mock import patch

import pytest

from switchyard.telemetry import (
    HEADER_NAME,
    LEGACY_OPT_OUT_ENVVAR,
    OPT_OUT_ENVVAR,
    _get_version,
    get_telemetry_headers,
)


@pytest.fixture(autouse=True)
def _reset_telemetry_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPT_OUT_ENVVAR, raising=False)
    monkeypatch.delenv(LEGACY_OPT_OUT_ENVVAR, raising=False)
    _get_version.cache_clear()
    yield
    _get_version.cache_clear()


def test_get_telemetry_headers_uses_switchyard_package_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_names: list[str] = []

    def fake_version(package_name: str) -> str:
        package_names.append(package_name)
        return "1.2.3"

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    assert get_telemetry_headers() == {HEADER_NAME: "1.2.3"}
    assert package_names == ["nemo-switchyard"]


@pytest.mark.parametrize("envvar", [OPT_OUT_ENVVAR, LEGACY_OPT_OUT_ENVVAR])
@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", "on"])
def test_get_telemetry_headers_respects_opt_out_envvars(
    monkeypatch: pytest.MonkeyPatch,
    envvar: str,
    value: str,
) -> None:
    monkeypatch.setenv(envvar, value)

    assert get_telemetry_headers() == {}


@pytest.mark.parametrize("value", ["", "0", "false", "no", " FALSE "])
def test_get_telemetry_headers_ignores_falsey_opt_out_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "9.8.7")
    monkeypatch.setenv(OPT_OUT_ENVVAR, value)

    assert get_telemetry_headers() == {HEADER_NAME: "9.8.7"}


def test_get_telemetry_headers_falls_back_to_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_package_not_found(_package_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", raise_package_not_found)

    assert get_telemetry_headers() == {HEADER_NAME: "unknown"}


def test_openai_llm_client_passes_default_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "1.2.3")
    captured: list[dict[str, Any]] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    with patch("switchyard.lib.llm_client.AsyncOpenAI", FakeAsyncOpenAI):
        from switchyard.lib.llm_client import OpenAILLMClient

        OpenAILLMClient(api_key="sk-test", base_url="https://llm.test/v1", timeout=3.0)

    assert captured == [
        {
            "api_key": "sk-test",
            "base_url": "https://llm.test/v1",
            "timeout": 3.0,
            "default_headers": {HEADER_NAME: "1.2.3"},
        },
    ]


def test_openai_llm_client_passes_empty_default_headers_when_opted_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPT_OUT_ENVVAR, "1")
    captured: list[dict[str, Any]] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    with patch("switchyard.lib.llm_client.AsyncOpenAI", FakeAsyncOpenAI):
        from switchyard.lib.llm_client import OpenAILLMClient

        OpenAILLMClient(api_key="sk-test")

    assert captured == [
        {"api_key": "sk-test", "default_headers": {}},
    ]
