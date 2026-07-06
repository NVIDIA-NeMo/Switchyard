// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::thread;

use serde_json::json;
use switchyard_components::{
    prefix_probe, ModelStatsSnapshot, StatsAccumulator, StatsSnapshot, TokenUsage,
};
use switchyard_core::{Result, SwitchyardError};

#[test]
fn accumulator_snapshot_starts_zero_and_reset_clears_all_state() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let initial = accumulator.snapshot()?;
    assert_eq!(initial.total_requests, 0);
    assert_eq!(initial.total_tokens.total, 0);
    assert!(initial.models.is_empty());

    accumulator.record_success("model-a", Some(12.5), Some("strong"))?;
    accumulator.record_usage(
        "model-a",
        TokenUsage {
            prompt_tokens: 5,
            completion_tokens: 7,
            cached_tokens: 2,
            cache_creation_tokens: 1,
            reasoning_tokens: 3,
            cacheable_prompt_tokens: 0,
        },
        Some(20.0),
        Some(7.5),
        Some("strong"),
    )?;
    assert_eq!(accumulator.snapshot()?.total_requests, 1);

    accumulator.reset()?;
    let reset = accumulator.snapshot()?;
    assert_eq!(reset.total_requests, 0);
    assert_eq!(reset.total_tokens.total, 0);
    assert!(reset.models.is_empty());
    Ok(())
}

#[test]
fn accumulator_matches_python_two_tier_snapshot_contract() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_success("strong/model", Some(12.0), Some("strong"))?;
    accumulator.record_usage(
        "strong/model",
        TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 25,
            cached_tokens: 10,
            cache_creation_tokens: 5,
            reasoning_tokens: 3,
            cacheable_prompt_tokens: 90,
        },
        Some(20.0),
        Some(8.0),
        Some("strong"),
    )?;
    accumulator.record_success("weak/model", None, Some("weak"))?;
    accumulator.record_usage(
        "weak/model",
        TokenUsage {
            prompt_tokens: 40,
            completion_tokens: 5,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("weak"),
    )?;

    let snapshot = accumulator.snapshot()?;

    assert_eq!(snapshot.total_requests, 2);
    assert_eq!(snapshot.total_tokens.prompt, 140);
    assert_eq!(snapshot.total_tokens.completion, 30);
    assert_eq!(snapshot.total_tokens.cached, 10);
    assert_eq!(snapshot.total_tokens.cache_creation, 5);
    assert_eq!(snapshot.total_tokens.reasoning, 3);
    assert_eq!(snapshot.total_tokens.total, 170);

    let strong = model_stats(&snapshot, "strong/model")?;
    assert_eq!(strong.tier.as_deref(), Some("strong"));
    assert_eq!(strong.request_pct, 50.0);
    assert_eq!(strong.token_pct, 73.53);
    assert_eq!(strong.avg_prompt_tokens, 100.0);
    assert_eq!(strong.avg_completion_tokens, 25.0);
    assert_eq!(strong.cache_hit_rate, 0.1);
    assert_eq!(strong.theoretical_cache_hit_rate, 0.9);

    let strong_tier = snapshot
        .tiers
        .get("strong")
        .ok_or_else(|| SwitchyardError::Other("strong tier should exist".to_string()))?;
    assert_eq!(strong_tier.model, "strong/model");
    assert_eq!(strong_tier.prompt_tokens, 100);
    assert_eq!(strong_tier.completion_tokens, 25);

    let weak_tier = snapshot
        .tiers
        .get("weak")
        .ok_or_else(|| SwitchyardError::Other("weak tier should exist".to_string()))?;
    assert_eq!(weak_tier.model, "weak/model");
    assert_eq!(weak_tier.prompt_tokens, 40);
    assert_eq!(weak_tier.completion_tokens, 5);
    Ok(())
}

// Switch-aware theoretical: cold on switch-in, partial credit on switching back.
#[test]
fn theoretical_is_switch_aware_across_a_model_switch_and_back() -> Result<()> {
    let acc = StatsAccumulator::new();
    // One nested conversation; equal-length turns so fractions are exact.
    let turn = |content: &[&str]| {
        let messages: Vec<_> = content
            .iter()
            .map(|c| json!({"role": "user", "content": c}))
            .collect();
        prefix_probe(&json!({ "messages": messages }))
    };
    let p1 = turn(&["aaaa"]);
    let p2 = turn(&["aaaa", "bbbb"]);
    let p3 = turn(&["aaaa", "bbbb", "cccc"]);
    let p4 = turn(&["aaaa", "bbbb", "cccc", "dddd"]);

    // strong serves turns 1-2: cold first sight, then the turn-1 prefix is eligible.
    assert_eq!(acc.prefix_eligibility("strong", &p1), 0.0);
    assert_eq!(acc.prefix_eligibility("strong", &p2), 0.5);

    // switch to weak at turn 3: weak's cache is empty.
    assert_eq!(acc.prefix_eligibility("weak", &p3), 0.0);

    // back to strong at turn 4: it saw turns 1-2 but not turn 3 (weak's) -> 2/4.
    assert_eq!(acc.prefix_eligibility("strong", &p4), 0.5);
    Ok(())
}

// Cache stats are attributed per model across an N-model rotation.
#[test]
fn cache_stats_are_per_model_across_multi_model_rotation() -> Result<()> {
    let accumulator = StatsAccumulator::new();

    // Three models each entered cold (full re-warm).
    for (model, tier) in [
        ("model/a", "strong"),
        ("model/b", "weak"),
        ("model/c", "third"),
    ] {
        accumulator.record_usage(
            model,
            TokenUsage {
                prompt_tokens: 100,
                cache_creation_tokens: 100,
                ..TokenUsage::default()
            },
            None,
            None,
            Some(tier),
        )?;
    }
    // a and b take a second, warm turn served from cache; c stays cold.
    for model in ["model/a", "model/b"] {
        accumulator.record_usage(
            model,
            TokenUsage {
                prompt_tokens: 100,
                cached_tokens: 100,
                ..TokenUsage::default()
            },
            None,
            None,
            None,
        )?;
    }

    let snapshot = accumulator.snapshot()?;
    let a = model_stats(&snapshot, "model/a")?;
    assert_eq!(a.cache_hit_rate, 0.5);
    assert_eq!(a.cache_creation_tokens, 100);
    let c = model_stats(&snapshot, "model/c")?;
    assert_eq!(c.cache_hit_rate, 0.0);
    assert_eq!(c.cache_creation_tokens, 100);
    Ok(())
}

#[test]
fn accumulator_is_thread_safe_under_concurrent_recording() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let n_threads = 16_u64;
    let calls_per_thread = 250_u64;
    let mut handles = Vec::new();

    for _ in 0..n_threads {
        let accumulator = accumulator.clone();
        handles.push(thread::spawn(move || -> Result<()> {
            for _ in 0..calls_per_thread {
                accumulator.record_success("threaded/model", None, Some("strong"))?;
                accumulator.record_usage(
                    "threaded/model",
                    TokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        ..TokenUsage::default()
                    },
                    None,
                    None,
                    Some("strong"),
                )?;
            }
            Ok(())
        }));
    }

    for handle in handles {
        match handle.join() {
            Ok(result) => result?,
            Err(_) => {
                return Err(SwitchyardError::Other(
                    "stats worker thread panicked".to_string(),
                ));
            }
        }
    }

    let expected = n_threads * calls_per_thread;
    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "threaded/model")?;
    assert_eq!(snapshot.total_requests, expected);
    assert_eq!(model.calls, expected);
    assert_eq!(model.prompt_tokens, expected);
    assert_eq!(model.completion_tokens, expected);
    Ok(())
}

#[test]
fn latency_reservoir_uses_replacement_path_after_saturation() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    for latency_ms in 0..10_050 {
        accumulator.record_success("reservoir/model", Some(latency_ms as f64), Some("strong"))?;
    }

    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "reservoir/model")?;

    assert_eq!(model.model_call_latency.count, 10_050);
    assert_eq!(model.model_call_latency.min_ms, 0.0);
    assert_eq!(model.model_call_latency.max_ms, 10_049.0);
    assert_eq!(model.model_call_latency.p50_ms, 5_050.0);
    assert_eq!(model.model_call_latency.p99_ms, 9_950.0);
    Ok(())
}

#[test]
fn cost_estimate_matches_python_known_model_and_unknown_model_behavior() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_usage(
        "claude-sonnet-4-6",
        TokenUsage {
            prompt_tokens: 1_000_000,
            ..TokenUsage::default()
        },
        None,
        None,
        None,
    )?;
    accumulator.record_usage(
        "claude-sonnet-4-6-20251022",
        TokenUsage {
            prompt_tokens: 1_000_000,
            ..TokenUsage::default()
        },
        None,
        None,
        None,
    )?;
    accumulator.record_usage(
        "unknown/model",
        TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            ..TokenUsage::default()
        },
        None,
        None,
        None,
    )?;

    let snapshot = accumulator.snapshot()?;
    let sonnet = snapshot
        .cost_estimate
        .models
        .get("claude-sonnet-4-6")
        .ok_or_else(|| SwitchyardError::Other("sonnet cost should exist".to_string()))?;
    let dated_sonnet = snapshot
        .cost_estimate
        .models
        .get("claude-sonnet-4-6-20251022")
        .ok_or_else(|| SwitchyardError::Other("dated sonnet cost should exist".to_string()))?;
    let unknown = snapshot
        .cost_estimate
        .models
        .get("unknown/model")
        .ok_or_else(|| SwitchyardError::Other("unknown cost should exist".to_string()))?;

    assert_eq!(sonnet.total_cost, 3.0);
    assert_eq!(dated_sonnet.total_cost, 0.0);
    assert_eq!(unknown.total_cost, 0.0);
    assert_eq!(snapshot.cost_estimate.total_cost, 3.0);
    Ok(())
}

#[test]
fn tier_rollup_aggregates_shared_tier_under_first_model() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_success("strong/first", None, Some("strong"))?;
    accumulator.record_usage(
        "strong/first",
        TokenUsage {
            prompt_tokens: 2,
            completion_tokens: 3,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("strong"),
    )?;
    accumulator.record_success("strong/second", None, Some("strong"))?;
    accumulator.record_usage(
        "strong/second",
        TokenUsage {
            prompt_tokens: 5,
            completion_tokens: 7,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("strong"),
    )?;

    let snapshot = accumulator.snapshot()?;
    let tier = snapshot
        .tiers
        .get("strong")
        .ok_or_else(|| SwitchyardError::Other("strong tier should exist".to_string()))?;

    assert_eq!(tier.model, "strong/first");
    assert_eq!(tier.calls, 2);
    assert_eq!(tier.prompt_tokens, 7);
    assert_eq!(tier.completion_tokens, 10);
    assert_eq!(tier.total_tokens, 17);
    Ok(())
}

#[test]
fn tier_rollup_keeps_distinct_tiers_for_shared_model() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_success("shared/model", None, Some("weak"))?;
    accumulator.record_usage(
        "shared/model",
        TokenUsage {
            prompt_tokens: 2,
            completion_tokens: 3,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("weak"),
    )?;
    accumulator.record_success("shared/model", None, Some("executor"))?;
    accumulator.record_usage(
        "shared/model",
        TokenUsage {
            prompt_tokens: 5,
            completion_tokens: 7,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("executor"),
    )?;

    let snapshot = accumulator.snapshot()?;
    let shared = model_stats(&snapshot, "shared/model")?;
    assert_eq!(shared.calls, 2);
    assert_eq!(shared.prompt_tokens, 7);
    assert_eq!(shared.completion_tokens, 10);

    let weak = snapshot
        .tiers
        .get("weak")
        .ok_or_else(|| SwitchyardError::Other("weak tier should exist".to_string()))?;
    assert_eq!(weak.model, "shared/model");
    assert_eq!(weak.calls, 1);
    assert_eq!(weak.prompt_tokens, 2);
    assert_eq!(weak.completion_tokens, 3);

    let executor = snapshot
        .tiers
        .get("executor")
        .ok_or_else(|| SwitchyardError::Other("executor tier should exist".to_string()))?;
    assert_eq!(executor.model, "shared/model");
    assert_eq!(executor.calls, 1);
    assert_eq!(executor.prompt_tokens, 5);
    assert_eq!(executor.completion_tokens, 7);
    Ok(())
}

#[test]
fn tier_usage_can_attach_explicit_untiered_success() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_success("shared/model", None, None)?;
    accumulator.record_success("shared/model", None, None)?;
    accumulator.record_usage_with_success_was_untiered(
        "shared/model",
        TokenUsage {
            prompt_tokens: 2,
            completion_tokens: 3,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("weak"),
    )?;
    accumulator.record_usage_with_success_was_untiered(
        "shared/model",
        TokenUsage {
            prompt_tokens: 5,
            completion_tokens: 7,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("executor"),
    )?;

    let snapshot = accumulator.snapshot()?;
    let shared = model_stats(&snapshot, "shared/model")?;
    assert_eq!(shared.calls, 2);
    assert_eq!(shared.prompt_tokens, 7);
    assert_eq!(shared.completion_tokens, 10);

    let weak = snapshot
        .tiers
        .get("weak")
        .ok_or_else(|| SwitchyardError::Other("weak tier should exist".to_string()))?;
    assert_eq!(weak.calls, 1);
    assert_eq!(weak.prompt_tokens, 2);
    assert_eq!(weak.completion_tokens, 3);

    let executor = snapshot
        .tiers
        .get("executor")
        .ok_or_else(|| SwitchyardError::Other("executor tier should exist".to_string()))?;
    assert_eq!(executor.calls, 1);
    assert_eq!(executor.prompt_tokens, 5);
    assert_eq!(executor.completion_tokens, 7);
    Ok(())
}

#[test]
fn tier_usage_preserves_legacy_untiered_success_sequence() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_success("shared/model", None, None)?;
    accumulator.record_usage(
        "shared/model",
        TokenUsage {
            prompt_tokens: 2,
            completion_tokens: 3,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("weak"),
    )?;

    let snapshot = accumulator.snapshot()?;
    let shared = model_stats(&snapshot, "shared/model")?;
    assert_eq!(shared.calls, 1);
    assert_eq!(shared.prompt_tokens, 2);
    assert_eq!(shared.completion_tokens, 3);

    let weak = snapshot
        .tiers
        .get("weak")
        .ok_or_else(|| SwitchyardError::Other("weak tier should exist".to_string()))?;
    assert_eq!(weak.calls, 1);
    assert_eq!(weak.prompt_tokens, 2);
    assert_eq!(weak.completion_tokens, 3);
    Ok(())
}

#[test]
fn already_attributed_usage_does_not_consume_legacy_pending_success() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_success("shared/model", None, None)?;
    accumulator.record_success("shared/model", None, Some("weak"))?;
    accumulator.record_usage_after_success_attribution(
        "shared/model",
        TokenUsage {
            prompt_tokens: 2,
            completion_tokens: 3,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("weak"),
    )?;
    accumulator.record_usage(
        "shared/model",
        TokenUsage {
            prompt_tokens: 5,
            completion_tokens: 7,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("executor"),
    )?;

    let snapshot = accumulator.snapshot()?;
    let shared = model_stats(&snapshot, "shared/model")?;
    assert_eq!(shared.calls, 2);
    assert_eq!(shared.prompt_tokens, 7);
    assert_eq!(shared.completion_tokens, 10);

    let weak = snapshot
        .tiers
        .get("weak")
        .ok_or_else(|| SwitchyardError::Other("weak tier should exist".to_string()))?;
    assert_eq!(weak.calls, 1);
    assert_eq!(weak.prompt_tokens, 2);
    assert_eq!(weak.completion_tokens, 3);

    let executor = snapshot
        .tiers
        .get("executor")
        .ok_or_else(|| SwitchyardError::Other("executor tier should exist".to_string()))?;
    assert_eq!(executor.calls, 1);
    assert_eq!(executor.prompt_tokens, 5);
    assert_eq!(executor.completion_tokens, 7);
    Ok(())
}

#[test]
fn generic_tier_labels_are_included_in_tier_rollups() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_success("plugin-a", None, Some("plugin"))?;
    accumulator.record_usage(
        "plugin-a",
        TokenUsage {
            prompt_tokens: 2,
            completion_tokens: 3,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("plugin"),
    )?;
    accumulator.record_success("plugin-b", None, Some("plugin"))?;
    accumulator.record_usage(
        "plugin-b",
        TokenUsage {
            prompt_tokens: 5,
            completion_tokens: 7,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("plugin"),
    )?;

    let snapshot = accumulator.snapshot()?;
    let plugin_a = model_stats(&snapshot, "plugin-a")?;
    let plugin_b = model_stats(&snapshot, "plugin-b")?;

    assert_eq!(plugin_a.tier.as_deref(), Some("plugin"));
    assert_eq!(plugin_b.tier.as_deref(), Some("plugin"));
    let plugin_tier = snapshot
        .tiers
        .get("plugin")
        .ok_or_else(|| SwitchyardError::Other("plugin tier should exist".to_string()))?;
    assert_eq!(plugin_tier.model, "plugin-a");
    assert_eq!(plugin_tier.calls, 2);
    assert_eq!(plugin_tier.prompt_tokens, 7);
    assert_eq!(plugin_tier.completion_tokens, 10);
    assert_eq!(plugin_tier.total_tokens, 17);
    assert_eq!(snapshot.total_tokens.prompt, 7);
    assert_eq!(snapshot.total_tokens.completion, 10);
    Ok(())
}

#[test]
fn classifier_bucket_is_empty_by_default() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_usage(
        "claude-sonnet-4-6",
        TokenUsage {
            prompt_tokens: 1_000_000,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("strong"),
    )?;
    let snapshot = accumulator.snapshot()?;
    assert!(snapshot.classifier.models.is_empty());
    assert_eq!(snapshot.classifier.total_requests, 0);
    assert_eq!(snapshot.cost_estimate.classifier_cost, 0.0);
    // backend_cost == total_cost when classifier is empty.
    assert_eq!(
        snapshot.cost_estimate.backend_cost,
        snapshot.cost_estimate.total_cost
    );
    Ok(())
}

#[test]
fn classifier_bucket_keeps_same_model_separate_from_routed_traffic() -> Result<()> {
    // Default TB-lite config: classifier_model == weak_model. Both record
    // against the same model id but must land in distinct buckets so spend
    // is plainly attributable.
    let accumulator = StatsAccumulator::new();
    let model = "claude-sonnet-4-6";
    accumulator.record_success(model, None, Some("weak"))?;
    accumulator.record_usage(
        model,
        TokenUsage {
            prompt_tokens: 1_000_000,
            ..TokenUsage::default()
        },
        None,
        None,
        Some("weak"),
    )?;
    accumulator.record_classifier_usage(
        model,
        TokenUsage {
            prompt_tokens: 500_000,
            ..TokenUsage::default()
        },
        Some(42.0),
    )?;

    let snapshot = accumulator.snapshot()?;

    // Backend bucket: one row, original 1M prompt tokens.
    let backend = snapshot
        .models
        .get(model)
        .ok_or_else(|| SwitchyardError::Other("backend row missing".to_string()))?;
    assert_eq!(backend.prompt_tokens, 1_000_000);
    assert_eq!(backend.calls, 1);

    // Classifier bucket: same model id, separate row, separate counts.
    let classifier = snapshot
        .classifier
        .models
        .get(model)
        .ok_or_else(|| SwitchyardError::Other("classifier row missing".to_string()))?;
    assert_eq!(classifier.prompt_tokens, 500_000);
    assert_eq!(classifier.calls, 1);
    assert_eq!(snapshot.classifier.total_requests, 1);

    // Cost split: backend = 3.0 (1M @ $3/M); classifier = 1.5 (500k @ $3/M).
    // total_cost rolls both in.
    assert_eq!(snapshot.cost_estimate.backend_cost, 3.0);
    assert_eq!(snapshot.cost_estimate.classifier_cost, 1.5);
    assert_eq!(snapshot.cost_estimate.total_cost, 4.5);
    Ok(())
}

#[test]
fn classifier_latency_lands_on_classifier_model_call_latency() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_classifier_usage("claude-sonnet-4-6", TokenUsage::default(), Some(15.0))?;
    let snapshot = accumulator.snapshot()?;
    let row = snapshot
        .classifier
        .models
        .get("claude-sonnet-4-6")
        .ok_or_else(|| SwitchyardError::Other("classifier row missing".to_string()))?;
    assert_eq!(row.model_call_latency.count, 1);
    assert_eq!(row.model_call_latency.total_ms, 15.0);
    Ok(())
}

#[test]
fn reset_clears_classifier_bucket() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_classifier_usage(
        "claude-sonnet-4-6",
        TokenUsage {
            prompt_tokens: 100,
            ..TokenUsage::default()
        },
        None,
    )?;
    accumulator.reset()?;
    let snapshot = accumulator.snapshot()?;
    assert!(snapshot.classifier.models.is_empty());
    assert_eq!(snapshot.classifier.total_requests, 0);
    assert_eq!(snapshot.cost_estimate.classifier_cost, 0.0);
    Ok(())
}

#[test]
fn routing_decision_counts_are_grouped_by_profile_type() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    accumulator.record_routing_decision("stage_router", "dimensions")?;
    accumulator.record_routing_decision("stage_router", "dimensions")?;
    accumulator.record_routing_decision("stage_router", "llm-classifier")?;
    accumulator.record_routing_decision("latency-service", "health")?;

    let snapshot = accumulator.snapshot()?;
    assert_eq!(
        snapshot
            .routing_decisions
            .get("stage_router")
            .and_then(|sources| sources.get("dimensions")),
        Some(&2)
    );
    assert_eq!(
        snapshot
            .routing_decisions
            .get("stage_router")
            .and_then(|sources| sources.get("llm-classifier")),
        Some(&1)
    );
    assert_eq!(
        snapshot
            .routing_decisions
            .get("latency-service")
            .and_then(|sources| sources.get("health")),
        Some(&1)
    );

    accumulator.reset()?;
    assert!(accumulator.snapshot()?.routing_decisions.is_empty());
    Ok(())
}

fn model_stats<'a>(snapshot: &'a StatsSnapshot, model: &str) -> Result<&'a ModelStatsSnapshot> {
    snapshot
        .models
        .get(model)
        .ok_or_else(|| SwitchyardError::Other(format!("model stats missing for {model}")))
}
