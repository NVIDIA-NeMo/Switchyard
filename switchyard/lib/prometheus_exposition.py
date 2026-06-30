# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a :class:`StatsAccumulator` snapshot as Prometheus exposition text.

Hand-rolled so the metrics surface ships without adding the ``prometheus-client``
dependency. The exposition follows Prometheus text format 0.0.4: ``# HELP`` /
``# TYPE`` headers, one sample per line, label values escaped per the spec.

Metric design (all names prefixed ``switchyard_``):

- ``requests_total{model,tier}`` — counter; selected model / tier traffic.
- ``errors_total{model,tier}`` — counter; backend-call errors.
- ``prompt_tokens_total`` / ``completion_tokens_total`` /
  ``cached_tokens_total`` / ``cache_creation_tokens_total`` /
  ``reasoning_tokens_total`` — counters per model/tier.
- ``model_call_latency_ms`` — summary; backend-only call latency,
  per model/tier, with ``quantile="0.5"`` and ``quantile="0.99"``.
- ``total_latency_ms`` — summary; end-to-end request latency, per model/tier.
- ``routing_overhead_ms`` — summary; global router decision overhead
  (``total_latency - backend_latency``).
- ``total_requests`` / ``total_errors`` — gauges; running totals across all
  models, mirroring the ``/v1/stats`` JSON top-level fields.

Label cardinality is intentionally bounded: model names and tier ids are
both small fixed sets in any real deployment; nothing per-request (no
request ids, no prompts) is ever emitted.
"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version
from typing import Any

try:
    _SWITCHYARD_VERSION = _pkg_version("nemo-switchyard")
except Exception:
    _SWITCHYARD_VERSION = "unknown"


def render_prometheus(snapshot: dict[str, Any]) -> str:
    """Render a stats snapshot as Prometheus text-format exposition.

    Args:
        snapshot: The dict returned by :meth:`StatsAccumulator.snapshot_sync`.

    Returns:
        A string containing the full exposition payload (UTF-8, LF-terminated
        lines, trailing newline).
    """
    lines: list[str] = []

    _emit_gauge(
        lines,
        "switchyard_build_info",
        "Switchyard build information.",
        [({"version": _SWITCHYARD_VERSION}, 1)],
    )

    total_requests = snapshot.get("total_requests", 0)
    total_errors = snapshot.get("total_errors", 0)
    _emit_gauge(
        lines,
        "switchyard_total_requests",
        "Total chain-level requests recorded.",
        [({}, total_requests)],
    )
    _emit_gauge(
        lines,
        "switchyard_total_errors",
        "Total chain-level backend errors recorded.",
        [({}, total_errors)],
    )

    models: dict[str, dict[str, Any]] = snapshot.get("models", {}) or {}

    # Counter families keyed by (description, attribute on per-model dict).
    counters: list[tuple[str, str, str]] = [
        ("switchyard_requests_total", "Calls per selected model/tier.", "calls"),
        ("switchyard_errors_total", "Backend errors per selected model/tier.", "errors"),
        ("switchyard_prompt_tokens_total", "Prompt tokens billed per model/tier.", "prompt_tokens"),
        (
            "switchyard_completion_tokens_total",
            "Completion tokens generated per model/tier.",
            "completion_tokens",
        ),
        ("switchyard_cached_tokens_total", "Cached prompt tokens per model/tier.", "cached_tokens"),
        (
            "switchyard_cache_creation_tokens_total",
            "Cache-creation tokens per model/tier.",
            "cache_creation_tokens",
        ),
        (
            "switchyard_reasoning_tokens_total",
            "Reasoning tokens per model/tier.",
            "reasoning_tokens",
        ),
    ]
    for metric, help_text, attr in counters:
        samples = [
            (_labels_for(model, m), m.get(attr, 0)) for model, m in sorted(models.items())
        ]
        _emit_counter(lines, metric, help_text, samples)

    # Summary families: backend-call latency and total-latency per model/tier.
    summaries: list[tuple[str, str, str]] = [
        (
            "switchyard_model_call_latency_ms",
            "Backend-call latency per model/tier (ms).",
            "model_call_latency",
        ),
        (
            "switchyard_total_latency_ms",
            "End-to-end request latency per model/tier (ms).",
            "total_latency",
        ),
    ]
    for metric, help_text, attr in summaries:
        per_model = [
            (_labels_for(model, m), m.get(attr) or _empty_histogram())
            for model, m in sorted(models.items())
        ]
        _emit_summary(lines, metric, help_text, per_model)

    # Global routing-overhead summary (router decision time minus backend time).
    overhead = snapshot.get("routing_overhead") or _empty_histogram()
    _emit_summary(
        lines,
        "switchyard_routing_overhead_ms",
        "Router decision latency overhead across all requests (ms).",
        [({}, overhead)],
    )

    return "\n".join(lines) + "\n"


def _labels_for(model: str, m: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {"model": model}
    tier = m.get("tier")
    if tier:
        labels["tier"] = str(tier)
    return labels


def _empty_histogram() -> dict[str, float | int]:
    return {"count": 0, "total_ms": 0.0, "p50_ms": 0.0, "p99_ms": 0.0}


def _emit_gauge(
    lines: list[str],
    name: str,
    help_text: str,
    samples: list[tuple[dict[str, str], float | int]],
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} gauge")
    for labels, value in samples:
        lines.append(f"{name}{render_labels(labels)} {format_number(value)}")


def _emit_counter(
    lines: list[str],
    name: str,
    help_text: str,
    samples: list[tuple[dict[str, str], float | int]],
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} counter")
    for labels, value in samples:
        lines.append(f"{name}{render_labels(labels)} {format_number(value)}")


def _emit_summary(
    lines: list[str],
    name: str,
    help_text: str,
    samples: list[tuple[dict[str, str], dict[str, float | int]]],
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} summary")
    for labels, hist in samples:
        for quantile in ("0.5", "0.99"):
            q_labels = dict(labels)
            q_labels["quantile"] = quantile
            field = "p50_ms" if quantile == "0.5" else "p99_ms"
            lines.append(
                f"{name}{render_labels(q_labels)} {format_number(hist.get(field, 0.0))}"
            )
        lines.append(
            f"{name}_sum{render_labels(labels)} {format_number(hist.get('total_ms', 0.0))}"
        )
        lines.append(
            f"{name}_count{render_labels(labels)} {format_number(hist.get('count', 0))}"
        )


def render_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{_escape_label_value(v)}"' for k, v in labels.items()]
    return "{" + ",".join(parts) + "}"


def _escape_label_value(value: str) -> str:
    # Per Prometheus exposition spec: escape backslash, double-quote, newline.
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return "0"
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"
