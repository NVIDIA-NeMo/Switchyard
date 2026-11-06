# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for baselines.reasoning_defaults lookup table."""

import pytest

from craft_taskgen.baselines.reasoning_defaults import REASONING_EFFORT, effort_for


def test_unknown_agent_returns_none():
    assert effort_for("gemini-cli", "claude-opus-4-6") is None


def test_unknown_model_returns_none():
    assert effort_for("claude-code", "gpt-5") is None


def test_lookup_is_case_sensitive():
    """Agent names are lowercase-hyphenated; uppercase shouldn't match."""
    assert effort_for("Codex", "gpt-5.3-codex") is None
    assert effort_for("CLAUDE-CODE", "claude-opus-4-6") is None


def test_qwen_not_listed():
    """Qwen3.5 uses enable_thinking, not effort — the table returns None and the
    launcher handles it via a separate branch."""
    assert effort_for("opencode", "Qwen3.5-397B-A17B-FP8") is None
    assert effort_for("opencode", "some-other-model") is None


def test_opencode_keys_use_canonical_gateway_slug():
    """Lookup uses the endpoint's model ID before the launcher adds gateway/."""
    for (agent, model), _ in REASONING_EFFORT.items():
        if agent == "opencode":
            assert not model.startswith("gateway/"), (
                f"opencode key {model!r} still carries the gateway/ dispatch prefix; "
                "strip it so the key matches the string the gateway actually receives"
            )


def test_haiku_not_listed_for_claude_code():
    """Claude-code's CLI doesn't support effort on haiku; listing a row would be
    a silent no-op."""
    assert effort_for("claude-code", "claude-haiku-4-5") is None


@pytest.mark.parametrize(
    ("agent", "model", "expected"),
    [
        ("codex", "gpt-5.3-codex", "high"),
        ("claude-code", "claude-opus-4-6", "high"),
        ("opencode", "claude-opus-4-6", "high"),
        ("opencode", "claude-haiku-4-5", "medium"),
    ],
)
def test_known_pairs_explicit(agent: str, model: str, expected: str):
    assert effort_for(agent, model) == expected


def test_effort_values_are_valid():
    """Every value in the table is a level harbor's CliFlag validator
    accepts for claude-code. The harbor patch in
    patches/harbor-agent-patches.diff extends the original
    {low, medium, high} set with `xhigh` so opus-4-7 can use Anthropic's
    recommended setting.
    """
    valid = {"low", "medium", "high", "xhigh"}
    for key, val in REASONING_EFFORT.items():
        assert val in valid, f"{key} has invalid effort {val!r}"
