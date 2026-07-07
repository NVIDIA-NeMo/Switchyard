# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe per-model LLM usage statistics collector.

Tracks token usage bucketed by model name, with an optional tier label
(e.g. ``"strong"`` / ``"weak"`` from random routing).  Used by:

* :class:`StatsResponseProcessor` — records after every LLM response.
* ``launch_claude`` TUI footer — reads the aggregate via ``snapshot()``.
* ``GET /v1/routing/stats`` HTTP endpoint — reads ``to_dict()``.

Cost estimation is best-effort: ``to_dict()`` includes an
``estimated_cost_usd`` field per model when the model name appears in
the built-in price table; ``None`` otherwise.  Prices are per 1 M
tokens (input / output) as of 2025 Q3 and do **not** account for cache
read / write premiums.
"""

from __future__ import annotations

import dataclasses
import re
import threading
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Price table — USD per 1 M tokens (input, output).
# Add entries as needed; unknown models get estimated_cost_usd = None.
# Keys use the base name without date suffixes (see _normalize_model_name).
# ---------------------------------------------------------------------------

_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-7":           (15.00, 75.00),
    "claude-sonnet-4-6":         ( 3.00, 15.00),
    "claude-sonnet-4-5":         ( 3.00, 15.00),
    "claude-haiku-4-5":          ( 0.80,  4.00),
    # OpenAI
    "gpt-4o":                    ( 2.50, 10.00),
    "gpt-4o-mini":               ( 0.15,  0.60),
    "o1":                        (15.00, 60.00),
    "o3":                        (10.00, 40.00),
    "o4-mini":                   ( 1.10,  4.40),
}

_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def _normalize_model_name(model: str) -> str:
    """Strip trailing ISO-8601 date suffixes used by some providers.

    ``"claude-sonnet-4-6-20251022"`` → ``"claude-sonnet-4-6"``
    """
    return _DATE_SUFFIX_RE.sub("", model)


def _estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    prices = _PRICE_PER_1M.get(model) or _PRICE_PER_1M.get(_normalize_model_name(model))
    if prices is None:
        return None
    input_rate, output_rate = prices
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestStats:
    """Aggregate snapshot for the TUI footer."""

    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class ModelStats:
    """Accumulated token usage for a single model across all requests.

    Field semantics:

    * ``prompt_tokens`` — total input billed.  For Anthropic this is the
      sum ``input_tokens + cache_creation_input_tokens +
      cache_read_input_tokens`` (Anthropic reports them as siblings, not
      parent/child).  For OpenAI it is ``usage.prompt_tokens``.
    * ``max_prompt_tokens`` — largest input-token count observed in one
      completed response.
    * ``completion_tokens`` — total output tokens.
    * ``total_tokens`` — ``prompt_tokens + completion_tokens``.
    * ``reasoning_tokens`` — subset of ``completion_tokens`` for chain-of-
      thought / thinking output (OpenAI o-series / Responses API).  0 for
      Anthropic (not reported on the Messages API).
    * ``cache_read_tokens`` — tokens served from the provider cache
      (``cache_read_input_tokens`` on Anthropic, ``cached_tokens`` on
      OpenAI).  Priced at 0.1× on Anthropic.
    * ``cache_creation_tokens`` — tokens written to the provider cache
      (Anthropic only, priced at 1.25×).  Always 0 for OpenAI.
    """

    tier: str = ""
    calls: int = 0
    prompt_tokens: int = 0
    max_prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class LiveStatsCollector:
    """Thread-safe per-model token usage collector.

    ``record()`` is called from the async request path (response
    processors / stream taps); ``snapshot()`` is called from the
    footer-repaint thread; ``to_dict()`` is called by the HTTP stats
    endpoint.  All methods acquire a single :class:`threading.Lock`
    so they are safe across both threads and async tasks.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, ModelStats] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record(
        self,
        model: str = "unknown",
        tier: str = "",
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        """Accumulate one response's token counts into *model*'s bucket."""
        with self._lock:
            bucket = self._models.get(model)
            if bucket is None:
                bucket = ModelStats(tier=tier)
                self._models[model] = bucket
            bucket.calls += 1
            bucket.prompt_tokens += prompt_tokens
            bucket.max_prompt_tokens = max(bucket.max_prompt_tokens, prompt_tokens)
            bucket.completion_tokens += completion_tokens
            bucket.total_tokens += prompt_tokens + completion_tokens
            bucket.reasoning_tokens += reasoning_tokens
            bucket.cache_read_tokens += cache_read_tokens
            bucket.cache_creation_tokens += cache_creation_tokens

    def reset(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self._models.clear()

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def snapshot(self) -> RequestStats:
        """Aggregate totals for the TUI footer."""
        with self._lock:
            buckets = list(self._models.values())
        return RequestStats(
            request_count=sum(b.calls for b in buckets),
            prompt_tokens=sum(b.prompt_tokens for b in buckets),
            completion_tokens=sum(b.completion_tokens for b in buckets),
            cache_read_tokens=sum(b.cache_read_tokens for b in buckets),
            cache_creation_tokens=sum(b.cache_creation_tokens for b in buckets),
        )

    def tier_breakdown(self) -> list[tuple[str, ModelStats]]:
        """Per-model rows ordered ``strong → weak → others``.

        Returned for the random-routing footer so the user can see, at
        a glance, how requests / tokens have split across tiers.  The
        ordering is stable across paints (sort key = ``(tier_rank,
        model_name)``) so the footer doesn't flicker between rows.
        """
        _ORDER = {"strong": 0, "weak": 1}
        with self._lock:
            items = [(name, dataclasses.replace(b)) for name, b in self._models.items()]
        items.sort(key=lambda kv: (_ORDER.get(kv[1].tier, 99), kv[0]))
        return items

    def to_dict(self) -> dict[str, object]:
        """Full per-model breakdown for the HTTP stats endpoint.

        Wire shape::

            {
                "total_requests": int,
                "total_tokens": {
                    "prompt": int, "completion": int, "total": int,
                    "reasoning": int, "cache_read": int, "cache_creation": int,
                },
                "models": {
                    "<model-name>": {
                        "tier": str,
                        "calls": int,
                        "prompt_tokens": int,
                        "max_prompt_tokens": int,
                        "completion_tokens": int,
                        "total_tokens": int,
                        "reasoning_tokens": int,
                        "cache_read_tokens": int,
                        "cache_creation_tokens": int,
                        "token_pct": float,
                        "estimated_cost_usd": float | null,
                    },
                    ...
                },
            }
        """
        with self._lock:
            snapshot = {k: dataclasses.replace(v) for k, v in self._models.items()}

        total_prompt = sum(b.prompt_tokens for b in snapshot.values())
        total_completion = sum(b.completion_tokens for b in snapshot.values())
        total_tokens = total_prompt + total_completion
        total_requests = sum(b.calls for b in snapshot.values())

        def _pct(part: int, whole: int) -> float:
            return round(part / whole * 100, 2) if whole else 0.0

        models_out: dict[str, object] = {}
        for name, b in snapshot.items():
            models_out[name] = {
                "tier": b.tier,
                "calls": b.calls,
                "prompt_tokens": b.prompt_tokens,
                "max_prompt_tokens": b.max_prompt_tokens,
                "completion_tokens": b.completion_tokens,
                "total_tokens": b.total_tokens,
                "reasoning_tokens": b.reasoning_tokens,
                "cache_read_tokens": b.cache_read_tokens,
                "cache_creation_tokens": b.cache_creation_tokens,
                "token_pct": _pct(b.total_tokens, total_tokens),
                "estimated_cost_usd": _estimate_cost(
                    name, b.prompt_tokens, b.completion_tokens,
                ),
            }

        return {
            "total_requests": total_requests,
            "total_tokens": {
                "prompt": total_prompt,
                "completion": total_completion,
                "total": total_tokens,
                "reasoning": sum(b.reasoning_tokens for b in snapshot.values()),
                "cache_read": sum(b.cache_read_tokens for b in snapshot.values()),
                "cache_creation": sum(b.cache_creation_tokens for b in snapshot.values()),
            },
            "models": models_out,
        }
