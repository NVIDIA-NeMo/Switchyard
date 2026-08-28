# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures.

Sets fake gateway credentials for every test so `_build_gateway_env()` doesn't
raise. Tests that specifically exercise the guardrail unset these via
`monkeypatch.delenv` to assert the RuntimeError path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _fake_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate ANTHROPIC_* + OPENAI_* gateway env vars so claude_cli and
    llm_judge helpers don't trip the gateway-only guardrail during unit tests.

    Tests that specifically exercise the guardrail unset these via
    `monkeypatch.delenv` to assert the RuntimeError path.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://inference-api.test.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://inference-api.test.invalid/v1")
