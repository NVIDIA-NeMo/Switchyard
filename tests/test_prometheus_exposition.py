# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`switchyard.lib.prometheus_exposition`.

Covers metric names, label sets, summary quantile rendering, and the
empty-accumulator edge case. Exposition is parsed line-by-line back into
samples for assertion rather than string-matched, so reorder-safe.
"""

from __future__ import annotations

import re

from switchyard.lib.prometheus_exposition import render_prometheus
from switchyard.lib.stats_accumulator import StatsAccumulator

_SAMPLE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})? (?P<value>.+)$")


def _parse(exposition: str) -> tuple[dict[str, str], dict[str, str], dict[tuple[str, frozenset[tuple[str, str]]], str]]:
    """Return ``(help_map, type_map, samples)`` from exposition text.

    samples keys are ``(metric_name, frozenset(labels.items()))``.
    """
    help_map: dict[str, str] = {}
    type_map: dict[str, str] = {}
    samples: dict[tuple[str, frozenset[tuple[str, str]]], str] = {}
    for raw in exposition.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# HELP "):
            name, _, text = line[len("# HELP ") :].partition(" ")
            help_map[name] = text
            continue
        if line.startswith("# TYPE "):
            name, _, kind = line[len("# TYPE ") :].partition(" ")
            type_map[name] = kind
            continue
        match = _SAMPLE_RE.match(line)
        assert match, f"unparseable exposition line: {line!r}"
        name = match.group("name")
        labels_blob = match.group("labels") or ""
        labels: dict[str, str] = {}
        if labels_blob:
            inner = labels_blob[1:-1]
            # Cheap parser — values never contain "," or "\"" in these tests.
            for part in inner.split(","):
                k, _, v = part.partition("=")
                labels[k] = v.strip('"')
        samples[(name, frozenset(labels.items()))] = match.group("value")
    return help_map, type_map, samples


async def test_empty_accumulator_renders_zero_totals():
    snapshot = await StatsAccumulator().snapshot()
    text = render_prometheus(snapshot)

    help_map, type_map, samples = _parse(text)

    assert type_map["switchyard_total_requests"] == "gauge"
    assert type_map["switchyard_total_errors"] == "gauge"
    assert samples[("switchyard_total_requests", frozenset())] == "0"
    assert samples[("switchyard_total_errors", frozenset())] == "0"
    # Routing overhead summary should still emit a header even with no data.
    assert type_map["switchyard_routing_overhead_ms"] == "summary"
    assert samples[("switchyard_routing_overhead_ms_count", frozenset())] == "0"
    # Exposition must end in a single trailing newline (scraper requirement).
    assert text.endswith("\n")


async def test_two_tier_snapshot_emits_expected_metric_names_and_labels():
    stats = StatsAccumulator()
    await stats.record_success(model="strong/m", backend_latency_ms=42.5, tier="strong")
    await stats.record_usage(
        model="strong/m",
        prompt_tokens=120,
        completion_tokens=30,
        cached_tokens=10,
        total_latency_ms=88.0,
        routing_overhead_ms=8.0,
        tier="strong",
    )
    await stats.record_error(model="weak/m", tier="weak")
    await stats.record_success(model="weak/m", backend_latency_ms=5.0, tier="weak")
    await stats.record_usage(
        model="weak/m",
        prompt_tokens=40,
        completion_tokens=5,
        total_latency_ms=15.0,
        routing_overhead_ms=3.0,
        tier="weak",
    )

    text = render_prometheus(await stats.snapshot())
    _, type_map, samples = _parse(text)

    # Every metric the ticket lists must be present with the expected type.
    expected_types = {
        "switchyard_total_requests": "gauge",
        "switchyard_total_errors": "gauge",
        "switchyard_requests_total": "counter",
        "switchyard_errors_total": "counter",
        "switchyard_prompt_tokens_total": "counter",
        "switchyard_completion_tokens_total": "counter",
        "switchyard_cached_tokens_total": "counter",
        "switchyard_model_call_latency_ms": "summary",
        "switchyard_total_latency_ms": "summary",
        "switchyard_routing_overhead_ms": "summary",
    }
    for metric, kind in expected_types.items():
        assert type_map.get(metric) == kind, f"{metric} missing or wrong type"

    strong_labels = frozenset({("model", "strong/m"), ("tier", "strong")})
    weak_labels = frozenset({("model", "weak/m"), ("tier", "weak")})

    # weak/m had one record_error then one record_success → calls=1, errors=1.
    assert samples[("switchyard_requests_total", strong_labels)] == "1"
    assert samples[("switchyard_requests_total", weak_labels)] == "1"
    assert samples[("switchyard_errors_total", weak_labels)] == "1"
    assert samples[("switchyard_prompt_tokens_total", strong_labels)] == "120"
    assert samples[("switchyard_completion_tokens_total", weak_labels)] == "5"
    assert samples[("switchyard_cached_tokens_total", strong_labels)] == "10"

    # Summaries must emit quantile=0.5 and quantile=0.99 plus _sum / _count.
    strong_p50_key = ("switchyard_model_call_latency_ms", strong_labels | {("quantile", "0.5")})
    strong_p99_key = ("switchyard_model_call_latency_ms", strong_labels | {("quantile", "0.99")})
    assert strong_p50_key in samples and strong_p99_key in samples
    assert samples[("switchyard_model_call_latency_ms_count", strong_labels)] == "1"
    assert samples[("switchyard_total_latency_ms_count", weak_labels)] == "1"

    # Global routing-overhead summary has no labels.
    assert samples[("switchyard_routing_overhead_ms_count", frozenset())] == "2"


def test_label_value_escapes_backslash_quote_and_newline():
    snapshot = {
        "total_requests": 0,
        "total_errors": 0,
        "models": {
            'weird"name\\with\nnewline': {
                "calls": 1,
                "errors": 0,
                "tier": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_creation_tokens": 0,
                "reasoning_tokens": 0,
                "model_call_latency": {"count": 0, "total_ms": 0, "p50_ms": 0, "p99_ms": 0},
                "total_latency": {"count": 0, "total_ms": 0, "p50_ms": 0, "p99_ms": 0},
            },
        },
        "routing_overhead": {"count": 0, "total_ms": 0, "p50_ms": 0, "p99_ms": 0},
    }
    text = render_prometheus(snapshot)
    # Escaped form per Prometheus exposition spec.
    assert 'model="weird\\"name\\\\with\\nnewline"' in text


def test_build_info_gauge_present() -> None:
    from importlib.metadata import version as pkg_version
    ver = pkg_version("nemo-switchyard")
    text = render_prometheus({"total_requests": 0, "total_errors": 0, "models": {}})
    _, type_map, samples = _parse(text)
    assert type_map.get("switchyard_build_info") == "gauge"
    key = ("switchyard_build_info", frozenset({("version", ver)}))
    assert samples[key] == "1"


def test_build_info_gauge_carries_version_label() -> None:
    from importlib.metadata import version as pkg_version
    expected = pkg_version("nemo-switchyard")
    text = render_prometheus({"total_requests": 0, "total_errors": 0, "models": {}})
    assert f'version="{expected}"' in text
