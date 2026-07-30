# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single global per-call output-token cap for baseline smoke trials.

One constant, applied to every agent that can honor it. This is a **cap**
(a safety ceiling, disclosed in the paper), not a per-model tuning knob —
we deliberately don't expose an override env var.

Why 64000:

- Anthropic's effort docs say: "When running Claude Opus 4.7 at xhigh or
  max effort, set a large max_tokens so the model has room to think and
  act across subagents and tool calls. Starting at 64k tokens and tuning
  from there is a reasonable default." (Anthropic's own suggestion for
  their highest-effort model.)
  https://platform.claude.com/docs/en/build-with-claude/effort
- Qwen3 HF model card recommends 32768 standard / 38912 competitions;
  64k is comfortably above competition headroom.
  https://huggingface.co/Qwen/Qwen3-32B
- Any single-call trial emitting >64k tokens is either runaway reasoning
  or a prompt error; capping there catches both without biting healthy
  runs.

Agents and how the cap plumbs through:

- claude-code: via `CLAUDE_CODE_MAX_OUTPUT_TOKENS` env (documented at
  https://code.claude.com/docs/en/env-vars). Harbor forwards it into the
  container at claude_code.py:870-871.
- opencode: via `OPENCODE_BUILD_MAX_TOKENS` + `OPENCODE_PLAN_MAX_TOKENS`
  on the host. Harbor's opencode agent reads these at config-generation
  time (opencode.py:374-391) and writes them into opencode.json under
  `agent.{build,plan}.max_tokens`.
- codex: no working cap knob today. The `model_max_output_tokens` config
  key is parsed but never applied upstream —
  https://github.com/openai/codex/issues/4138. Codex runs uncapped for
  now; the paper should footnote this gap.
- qwen-coder: no documented per-call cap. The Qwen Code CLI (a Gemini CLI
  fork) honors a `maxOutputTokens` setting in ~/.qwen/settings.json but
  it's not currently plumbed by harbor's install template. Qwen-coder runs
  uncapped; same paper footnote as codex.
- pi (earendil-works/pi-coding-agent): no documented per-call cap exposed
  via env var. The pi models.json provider entry has a `maxTokensField`
  knob naming which request field carries max_tokens, but no global
  ceiling is set today. Pi runs uncapped; same paper footnote as codex.
"""

from __future__ import annotations

OUTPUT_TOKEN_CAP: int = 64000
