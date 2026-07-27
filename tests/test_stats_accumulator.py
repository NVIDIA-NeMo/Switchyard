# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from switchyard.lib.stats_accumulator import StatsAccumulator


async def test_snapshot_sync_matches_async_snapshot_and_includes_tier_tokens():
    stats = StatsAccumulator()
    await stats.record_success(
        model="strong/model",
        backend_latency_ms=12.0,
        tier="strong",
    )
    await stats.record_usage(
        model="strong/model",
        prompt_tokens=100,
        completion_tokens=25,
        cached_tokens=10,
        total_latency_ms=20.0,
        routing_overhead_ms=8.0,
        tier="strong",
    )
    await stats.record_success(model="weak/model", tier="weak")
    await stats.record_usage(
        model="weak/model",
        prompt_tokens=40,
        completion_tokens=5,
        tier="weak",
    )

    sync_snapshot = stats.snapshot_sync()

    assert sync_snapshot == await stats.snapshot()
    assert sync_snapshot["total_requests"] == 2
    assert sync_snapshot["total_tokens"]["prompt"] == 140
    assert sync_snapshot["total_tokens"]["completion"] == 30
    assert sync_snapshot["models"]["strong/model"]["tier"] == "strong"
    assert sync_snapshot["tiers"]["strong"]["prompt_tokens"] == 100
    assert sync_snapshot["tiers"]["strong"]["completion_tokens"] == 25
    assert sync_snapshot["tiers"]["weak"]["prompt_tokens"] == 40
    assert sync_snapshot["tiers"]["weak"]["completion_tokens"] == 5


async def test_snapshot_includes_generic_tier_rollup():
    stats = StatsAccumulator()
    await stats.record_success(model="plugin/model-a", tier="plugin")
    await stats.record_usage(
        model="plugin/model-a",
        prompt_tokens=2,
        completion_tokens=3,
        tier="plugin",
    )
    await stats.record_success(model="plugin/model-b", tier="plugin")
    await stats.record_usage(
        model="plugin/model-b",
        prompt_tokens=5,
        completion_tokens=7,
        tier="plugin",
    )

    snapshot = await stats.snapshot()

    assert snapshot["models"]["plugin/model-a"]["tier"] == "plugin"
    assert snapshot["models"]["plugin/model-b"]["tier"] == "plugin"
    assert snapshot["tiers"]["plugin"]["model"] == "plugin/model-a"
    assert snapshot["tiers"]["plugin"]["calls"] == 2
    assert snapshot["tiers"]["plugin"]["prompt_tokens"] == 7
    assert snapshot["tiers"]["plugin"]["completion_tokens"] == 10


async def test_same_model_can_contribute_to_distinct_tier_rollups():
    stats = StatsAccumulator()
    await stats.record_success(model="shared/model", tier="weak")
    await stats.record_usage(
        model="shared/model",
        prompt_tokens=2,
        completion_tokens=3,
        tier="weak",
    )
    await stats.record_success(model="shared/model", tier="primary")
    await stats.record_usage(
        model="shared/model",
        prompt_tokens=5,
        completion_tokens=7,
        tier="primary",
    )

    snapshot = await stats.snapshot()

    assert snapshot["models"]["shared/model"]["calls"] == 2
    assert snapshot["models"]["shared/model"]["prompt_tokens"] == 7
    assert snapshot["models"]["shared/model"]["completion_tokens"] == 10
    assert snapshot["tiers"]["weak"]["calls"] == 1
    assert snapshot["tiers"]["weak"]["prompt_tokens"] == 2
    assert snapshot["tiers"]["weak"]["completion_tokens"] == 3
    assert snapshot["tiers"]["primary"]["calls"] == 1
    assert snapshot["tiers"]["primary"]["prompt_tokens"] == 5
    assert snapshot["tiers"]["primary"]["completion_tokens"] == 7


async def test_usage_can_attach_explicit_untiered_success_to_tier():
    stats = StatsAccumulator()
    await stats.record_success(model="shared/model")
    await stats.record_usage(
        model="shared/model",
        prompt_tokens=2,
        completion_tokens=3,
        tier="weak",
        success_was_untiered=True,
    )

    snapshot = await stats.snapshot()

    assert snapshot["models"]["shared/model"]["calls"] == 1
    assert snapshot["tiers"]["weak"]["calls"] == 1
    assert snapshot["tiers"]["weak"]["prompt_tokens"] == 2
    assert snapshot["tiers"]["weak"]["completion_tokens"] == 3


async def test_legacy_untiered_success_then_tiered_usage_counts_tier_call():
    stats = StatsAccumulator()
    await stats.record_success(model="shared/model")
    await stats.record_usage(
        model="shared/model",
        prompt_tokens=2,
        completion_tokens=3,
        tier="weak",
    )

    snapshot = await stats.snapshot()

    assert snapshot["models"]["shared/model"]["calls"] == 1
    assert snapshot["tiers"]["weak"]["calls"] == 1
    assert snapshot["tiers"]["weak"]["prompt_tokens"] == 2
    assert snapshot["tiers"]["weak"]["completion_tokens"] == 3


async def test_reset_sync_clears_async_recorded_stats():
    stats = StatsAccumulator()
    await stats.record_success(model="model")
    await stats.record_usage(model="model", prompt_tokens=10, completion_tokens=5)

    stats.reset_sync()

    snapshot = await stats.snapshot()
    assert snapshot["total_requests"] == 0
    assert snapshot["total_tokens"]["total"] == 0
    assert snapshot["models"] == {}


async def test_classifier_usage_recorded_into_separate_bucket():
    """Classifier calls don't leak into the routed-backend ``models`` block.

    Default TB-lite config has classifier-model == weak-model
    (Nemotron-3-Super-v3). Without the separate bucket the two would
    accumulate into the same entry and the per-classifier breakdown
    would be lost.
    """
    stats = StatsAccumulator()
    # Same model name on both sides — exactly the collision case.
    await stats.record_success(model="nvidia/nemotron-3-super-v3", tier="weak")
    await stats.record_usage(
        model="nvidia/nemotron-3-super-v3",
        prompt_tokens=1_000,
        completion_tokens=200,
        tier="weak",
    )
    await stats.record_classifier_usage(
        model="nvidia/nemotron-3-super-v3",
        prompt_tokens=300,
        completion_tokens=50,
        latency_ms=42.0,
    )

    snapshot = await stats.snapshot()

    # Backend bucket counts the routed call only.
    backend = snapshot["models"]["nvidia/nemotron-3-super-v3"]
    assert backend["prompt_tokens"] == 1_000
    assert backend["completion_tokens"] == 200
    assert backend["calls"] == 1
    # Classifier bucket counts the classifier call only.
    classifier = snapshot["classifier"]["models"]["nvidia/nemotron-3-super-v3"]
    assert classifier["prompt_tokens"] == 300
    assert classifier["completion_tokens"] == 50
    assert classifier["calls"] == 1
    assert classifier["model_call_latency"]["count"] == 1


async def test_cost_estimate_total_includes_classifier_overhead():
    """Headline ``cost_estimate.total_cost`` is backend + classifier.

    Existing consumers (baseline manifests, dashboards) read
    ``total_cost`` and don't know about the new classifier bucket; the
    accumulator must roll the two together so those readers reflect
    true spend.
    """
    stats = StatsAccumulator()
    # Use a model with known pricing so we get non-zero numbers.
    await stats.record_success(model="nvidia/nvidia/nemotron-3-super-v3", tier="weak")
    await stats.record_usage(
        model="nvidia/nvidia/nemotron-3-super-v3",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        tier="weak",
    )
    await stats.record_classifier_usage(
        model="nvidia/nvidia/nemotron-3-super-v3",
        prompt_tokens=1_000_000,
        completion_tokens=0,
    )

    snapshot = await stats.snapshot()
    cost = snapshot["cost_estimate"]
    # Nemotron-3 Super input is $0.10/Mtok — backend 1M tokens = $0.10,
    # classifier 1M tokens = $0.10, total = $0.20.
    assert cost["backend_cost"] == pytest.approx(0.10, rel=0.01)
    assert cost["classifier_cost"] == pytest.approx(0.10, rel=0.01)
    assert cost["total_cost"] == pytest.approx(0.20, rel=0.01)


async def test_reset_clears_classifier_bucket():
    stats = StatsAccumulator()
    await stats.record_classifier_usage(
        model="router/clf",
        prompt_tokens=100,
        completion_tokens=20,
    )

    stats.reset_sync()

    snapshot = await stats.snapshot()
    assert snapshot["classifier"]["total_requests"] == 0
    assert snapshot["classifier"]["models"] == {}
    assert snapshot["cost_estimate"]["classifier_cost"] == 0.0


# Per-target attribution after evict-and-reroute is covered by
# `evict_and_reroute_attributes_error_to_weak_and_success_to_strong` in
# crates/switchyard-components/tests/stats_processors.rs — that integration
# test drives the chain executor end-to-end. Re-asserting the same shape here
# by hand-recording into the accumulator would only verify the accumulator's
# incrementers (already covered above), not the eviction code path.
