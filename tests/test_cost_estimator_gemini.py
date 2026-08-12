# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pricing for the default LLM-classifier model (Gemini 3.5 Flash).

Guards the ``cost_estimator`` entry so a typo or accidental removal —
which silently zeroes the default classifier's cost in routing-stats —
fails loudly instead.
"""

from __future__ import annotations

from switchyard.cli.launchers.cost_estimator import MODEL_PRICING, estimate_cost

_GEMINI_KEYS = (
    "gcp/google/gemini-3.5-flash",
    "openai/gcp/google/gemini-3.5-flash",
    "gemini-3.5-flash",
)


def test_gemini_flash_keys_registered_with_global_list_price() -> None:
    for key in _GEMINI_KEYS:
        price = MODEL_PRICING[key]
        # Google Vertex global list price (ai.google.dev), USD / 1M tokens.
        assert price.input == 1.50
        assert price.output == 9.00
        assert price.cached == 0.15  # 90% off input
        assert price.cache_write == price.input  # no per-token write premium


def test_gemini_flash_estimate_includes_cache_discount() -> None:
    # prompt_tokens is the TOTAL input (cached is a subset): 2M total with
    # 1M cached => 1M fresh + 1M cached, plus 1M output.
    result = estimate_cost({
        "gcp/google/gemini-3.5-flash": {
            "prompt_tokens": 2_000_000,
            "cached_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
        },
    })
    model = result["models"]["gcp/google/gemini-3.5-flash"]
    # 1.50 (fresh in) + 0.15 (cached in) + 9.00 (out) = 10.65
    assert model["total_cost"] == 10.65
