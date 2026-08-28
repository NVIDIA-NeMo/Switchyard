# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for llm_judge against the real NVIDIA gateway.

Exercises each model we plan to use, with a simple schema, and reports
latency + token counts. Also tries to trigger the parse-retry path by
crafting a schema that an ambiguous prompt will likely miss on first try.

Run: `uv run python scripts/smoke-llm-judge.py` after `uv sync`.
"""

from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv

from craft_taskgen import llm_judge

load_dotenv()

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "color": {"type": "string", "enum": ["red", "green", "blue"]},
        "justification": {"type": "string"},
    },
    "required": ["color", "justification"],
}

RUBRIC_LIKE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "vague", "narrow_tests", "leaked", "misaligned"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}

MODELS: list[tuple[str, str]] = [
    ("Opus 4.6", "aws/anthropic/bedrock-claude-opus-4-6"),
    ("GPT-5.4", "us/azure/openai/gpt-5.4"),
    ("Gemini-3.1-Pro", "gcp/google/gemini-3.1-pro-preview"),
]

SIMPLE_PROMPT = (
    "Pick one color from red/green/blue and briefly justify the choice. "
    "Respond as JSON matching the schema — no markdown fences, no extra text."
)

RUBRIC_PROMPT = (
    "Consider this (fabricated) scenario: an instruction says 'Add SSL support' "
    "and the reference test asserts a specific private method `_ssl_mode_flag` is "
    "called with value `verify-full`. Classify the instruction-tests alignment. "
    "Return JSON matching the schema — no markdown fences."
)


async def smoke_one(label: str, model: str, prompt: str, schema: dict) -> None:
    print(f"\n--- {label} @ {model} ---")
    try:
        t0 = time.monotonic()
        out = await llm_judge.judge(prompt=prompt, schema=schema, model=model)
        wall = time.monotonic() - t0
        first_key = next(iter(out.result))
        print(
            f"OK  verdict={out.result[first_key]!r}  "
            f"in={out.usage['input_tokens']}  out={out.usage['output_tokens']}  "
            f"cached={out.usage['cached_tokens']}  latency={out.latency_s:.2f}s  "
            f"wall={wall:.2f}s"
        )
        print(f"    full: {out.result}")
    except Exception as err:
        print(f"FAIL {type(err).__name__}: {err}")


async def main() -> None:
    # Confirm creds actually loaded from .env (litellm won't error cleanly if missing).
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not key or not base:
        raise RuntimeError("OPENAI_API_KEY / OPENAI_BASE_URL not set — check .env loading")
    print(f"Gateway: {base}  (key len={len(key)})")

    print("\n======== Simple-schema smoke (pick a color) ========")
    for label, model in MODELS:
        await smoke_one(label, model, SIMPLE_PROMPT, SIMPLE_SCHEMA)

    print("\n======== Rubric-like smoke (alignment-style verdict) ========")
    for label, model in MODELS:
        await smoke_one(label, model, RUBRIC_PROMPT, RUBRIC_LIKE_SCHEMA)


if __name__ == "__main__":
    asyncio.run(main())
