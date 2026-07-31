# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-of-session summary printed when a launcher exits."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from typing import cast

from switchyard.cli.launchers.stats_source import StatsSource
from switchyard.lib.cost_estimator import estimate_model_cost

_RULE = "─" * 51
_LOG = logging.getLogger(__name__)


def print_session_summary(stats: StatsSource) -> None:
    """Print a session summary to stdout after the launcher exits."""
    try:
        snapshot = stats.snapshot_sync()
        summary = _format_summary(snapshot)
        if summary:
            with contextlib.suppress(BrokenPipeError):
                print(summary)
    except Exception:
        _LOG.debug("Failed to print session summary", exc_info=True)


def _format_summary(snapshot: Mapping[str, object]) -> str:
    """Return a formatted summary string, or empty string if no requests were made."""
    total_requests = _int(snapshot, "total_requests")
    if total_requests == 0:
        return ""

    total_errors = _int(snapshot, "total_errors")
    totals = _mapping(snapshot, "total_tokens")
    prompt = _int(totals, "prompt")
    completion = _int(totals, "completion")
    cached = _int(totals, "cached")

    req_str = f"{total_requests:,}" + (f"  ({total_errors} errors)" if total_errors else "")
    tok_str = f"{prompt:,} in  {completion:,} out" + (f"  {cached:,} cached" if cached else "")

    lines = [
        "",
        _RULE,
        f"  requests : {req_str}",
        f"  tokens   : {tok_str}",
    ]

    models = _mapping(snapshot, "models")
    if models:
        lines.append("  models   :")
        for model_name, model_data in models.items():
            md = cast(Mapping[str, object], model_data) if isinstance(model_data, Mapping) else {}
            calls = _int(md, "calls")
            pt = _int(md, "prompt_tokens")
            ct = _int(md, "completion_tokens")
            cached_t = _int(md, "cached_tokens")
            cache_creation_t = _int(md, "cache_creation_tokens")
            cost = estimate_model_cost(
                model_name, pt, ct, cached_tokens=cached_t, cache_creation_tokens=cache_creation_t
            )
            short = model_name.rsplit("/", 1)[-1]
            cost_str = f"  ${cost['total_cost']:.4f}" if cost["total_cost"] else ""
            lines.append(f"    {short}  {calls:,} req · {pt:,} in  {ct:,} out{cost_str}")

    lines.append(_RULE)
    return "\n".join(lines)


def _mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Safely extract a nested mapping from snapshot data, returning {} on missing/wrong type."""
    value = data.get(key)
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _int(data: Mapping[str, object], key: str) -> int:
    """Safely extract an int from snapshot data, returning 0 on missing/wrong type."""
    value = data.get(key)
    return value if isinstance(value, int) else 0
