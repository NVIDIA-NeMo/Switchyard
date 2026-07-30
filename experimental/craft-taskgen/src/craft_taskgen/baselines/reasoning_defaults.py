# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default reasoning_effort for baseline smoke-test (agent, model) combos.

One row per pair, aligned columns. Grep for the agent name (e.g. `codex`) to
see every model it covers.

Pairs NOT listed here: the launcher passes no reasoning_effort and whatever
default the agent ships with fires (harbor-default "high" for codex; nothing
for claude-code+haiku-4-5 since haiku doesn't support effort at all). The
Qwen3.5 family uses a different knob entirely (enable_thinking boolean) and
is handled by a separate launcher branch, not by this table. The qwen-coder
agent (Qwen Code CLI fork of Gemini CLI) similarly has no reasoning_effort
kwarg — Qwen3.x thinking is server-side and is enabled by default on the
gateway-hosted models.

The `pi` agent (earendil-works/pi-coding-agent) uses its own
`--thinking {off,minimal,low,medium,high,xhigh}` CLI flag, exposed by
harbor as the `thinking` agent-kwarg. Driven by the launcher's per-agent
case, not by this table.

See docs/runbooks/baseline-reproducibility.md for rationale + verification recipes.
"""

from __future__ import annotations

REASONING_EFFORT: dict[tuple[str, str], str] = {
    # ---------- codex ----------
    # OpenAI recommends `medium` as daily driver for coding; we use `high` since
    # our smoke tasks are harder than average.
    ("codex", "azure/openai/gpt-5.3-codex"): "high",
    ("codex", "openai/openai/gpt-5.3-codex"): "high",
    ("codex", "azure/openai/gpt-5.4"): "high",
    ("codex", "openai/openai/gpt-5.4"): "high",
    ("codex", "openai/openai/gpt-5.5"): "high",
    # ---------- claude-code (per Anthropic's effort docs) ----------
    # Opus 4.7: Anthropic recommends xhigh for coding. Harbor's CliFlag
    # validator originally accepted only {low, medium, high}; the patch in
    # patches/harbor-agent-patches.diff now adds xhigh to the choices list
    # so the launcher can pass --effort xhigh to the in-container `claude`
    # binary. Empirical justification: under effort=high, opus-4-7 emits
    # only ~32K cache_creation_input_tokens per trial (vs opus-4-6 at
    # ~166K under the same task set), suggesting reasoning is being
    # under-applied; bumping to xhigh raised cache_creation by ~2.7×
    # in a 1-task probe.
    ("claude-code", "aws/anthropic/bedrock-claude-opus-4-7"): "xhigh",
    # Opus 4.6: API default is `high`
    ("claude-code", "aws/anthropic/bedrock-claude-opus-4-6"): "high",
    ("claude-code", "azure/anthropic/claude-opus-4-6"): "high",
    # Opus 4.5: effort supported, no specific recommendation — use API default
    ("claude-code", "aws/anthropic/claude-opus-4-5"): "high",
    ("claude-code", "azure/anthropic/claude-opus-4-5"): "high",
    # Sonnet 4.6: "Medium effort (recommended default)" for agentic coding
    ("claude-code", "aws/anthropic/bedrock-claude-sonnet-4-6"): "medium",
    ("claude-code", "azure/anthropic/claude-sonnet-4-6"): "medium",
    # ---------- opencode ----------
    # Same effort intent as claude-code, but opencode routes through the NVIDIA
    # gateway's /v1/chat/completions which rejects `xhigh` per CLIProxyAPI#2185.
    # Opus-4-7 via opencode caps at `high` until the gateway accepts xhigh on
    # that endpoint; everything else matches claude-code above.
    #
    # Keys use the canonical gateway slug (the string the gateway actually
    # sees on the wire). The opencode-internal `nvidia/` provider prefix is
    # an opencode dispatch token, not part of the model identity, and is
    # prepended by the launcher when constructing harbor's --model argument.
    ("opencode", "aws/anthropic/bedrock-claude-opus-4-7"): "high",
    ("opencode", "aws/anthropic/bedrock-claude-opus-4-6"): "high",
    ("opencode", "azure/anthropic/claude-opus-4-6"): "high",
    ("opencode", "aws/anthropic/claude-opus-4-5"): "high",
    ("opencode", "azure/anthropic/claude-opus-4-5"): "high",
    ("opencode", "aws/anthropic/bedrock-claude-sonnet-4-6"): "medium",
    ("opencode", "azure/anthropic/claude-sonnet-4-6"): "medium",
    ("opencode", "aws/anthropic/claude-haiku-4-5-v1"): "medium",
    ("opencode", "azure/anthropic/claude-haiku-4-5"): "medium",
    # ---------- pi (renders as --thinking via launcher's pi branch) ----------
    # Pi's `--thinking` flag levels: off, minimal, low, medium, high, xhigh.
    # For reasoning-capable open models we use `high`; haiku-4-5 is not
    # reasoning-capable so we leave it OFF (no row → no --thinking flag).
    ("pi", "nvidia/qwen/qwen3.6-35b-a3b"): "high",
    ("pi", "nvidia/zai-org/glm-5.1"): "high",
    ("pi", "nvidia/nvidia/nemotron-3-ultra-preview"): "high",
}


def effort_for(agent: str, model: str) -> str | None:
    """Return the default reasoning_effort for (agent, model), or None if not defined.

    Exact-match lookup by (agent, full_model_id). Add a row to REASONING_EFFORT
    when you start smoke-testing a new combo.
    """
    return REASONING_EFFORT.get((agent, model))
