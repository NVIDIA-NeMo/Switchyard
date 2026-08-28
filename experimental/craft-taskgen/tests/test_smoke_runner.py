# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the configurable smoke-step harbor invocation builder."""

from __future__ import annotations

import json
from pathlib import Path

from craft_taskgen.adapters._docker import CLAUDE_CODE_VERSION, CODEX_VERSION
from craft_taskgen.runner import _build_smoke_cmd


def _kwargs(cmd: list[str]) -> list[str]:
    """Extract the values that follow each --agent-kwarg flag."""
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--agent-kwarg"]


def test_codex_invocation_wires_catalog_and_reasoning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd, env = _build_smoke_cmd("tasks/foo", "codex", "openai/openai/gpt-5.5", "high", "smoke-probe-foo")

    assert cmd[:9] == [
        ".venv/bin/harbor",
        "run",
        "--yes",
        "-p",
        "tasks/foo",
        "-a",
        "codex",
        "-m",
        "openai/openai/gpt-5.5",
    ]
    assert "--job-name" in cmd and "smoke-probe-foo" in cmd
    kwargs = _kwargs(cmd)
    assert f"version={CODEX_VERSION}" in kwargs
    assert "reasoning_effort=high" in kwargs
    # No claude-code-only plan-mode restriction for codex.
    assert "disallowed_tools=EnterPlanMode,ExitPlanMode" not in kwargs

    # Catalog is filtered to the single requested slug and pointed at by env.
    catalog_path = env["CODEX_MODEL_CATALOG_JSON"]
    data = json.loads(Path(catalog_path).read_text())
    slugs = [m["slug"] for m in data["models"]]
    assert slugs == ["openai/openai/gpt-5.5"]


def test_claude_code_invocation_keeps_plan_mode_restriction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd, env = _build_smoke_cmd(
        "tasks/bar", "claude-code", "azure/anthropic/claude-opus-4-6", "high", "smoke-probe-bar"
    )
    kwargs = _kwargs(cmd)
    assert f"version={CLAUDE_CODE_VERSION}" in kwargs
    assert "disallowed_tools=EnterPlanMode,ExitPlanMode" in kwargs
    assert "reasoning_effort=high" in kwargs
    # claude-code reads ANTHROPIC_* from os.environ — no extra env overrides.
    assert env == {}


def test_empty_effort_falls_back_to_reasoning_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # ("codex", "openai/openai/gpt-5.5") is "high" in reasoning_defaults.
    cmd, _ = _build_smoke_cmd("tasks/foo", "codex", "openai/openai/gpt-5.5", "", "job")
    assert "reasoning_effort=high" in _kwargs(cmd)


def test_unknown_codex_slug_warns_and_skips_catalog(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cmd, env = _build_smoke_cmd("tasks/foo", "codex", "openai/openai/not-a-real-model", "high", "job")
    assert "CODEX_MODEL_CATALOG_JSON" not in env
    assert "no codex catalog row" in capsys.readouterr().out
    # Reasoning kwarg still passed even without a catalog row.
    assert "reasoning_effort=high" in _kwargs(cmd)
