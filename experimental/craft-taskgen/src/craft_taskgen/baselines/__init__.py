# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline-launcher helpers (reasoning defaults, etc)."""

from __future__ import annotations

from typing import Final

from craft_taskgen.baselines.output_cap import OUTPUT_TOKEN_CAP
from craft_taskgen.baselines.reasoning_defaults import REASONING_EFFORT, effort_for
from craft_taskgen.baselines.run_manifest import (
    SCHEMA_VERSION,
    probe_vllm_models,
    task_dir_digest,
    write_manifest,
)

# Canonical names for the LLM-driven agents the launcher supports.
# Imported by preflight.py, reasoning_defaults.py, and the run_manifest
# CLI so a typo in one place can't silently diverge from another.
AGENT_NAMES: Final[tuple[str, ...]] = ("claude-code", "codex", "opencode", "qwen-coder", "pi")

# Names of harbor's built-in non-LLM agents the launcher supports for
# dataset sanity checks (oracle applies the reference solution; nop
# leaves the tree unchanged). They share the launcher infrastructure
# but bypass model/effort/cap machinery.
SANITY_AGENT_NAMES: Final[tuple[str, ...]] = ("oracle", "nop")

# Every effort level harbor's CliFlag validators currently accept for
# claude-code (low/medium/high) and codex (low/medium/high/xhigh). `max`
# exists in Anthropic/OpenAI docs but is not accepted by harbor today —
# adding it here would require a harbor patch.
EFFORT_LEVELS: Final[tuple[str, ...]] = ("low", "medium", "high", "xhigh")

__all__ = [
    "AGENT_NAMES",
    "EFFORT_LEVELS",
    "OUTPUT_TOKEN_CAP",
    "REASONING_EFFORT",
    "SANITY_AGENT_NAMES",
    "SCHEMA_VERSION",
    "effort_for",
    "probe_vllm_models",
    "task_dir_digest",
    "write_manifest",
]
