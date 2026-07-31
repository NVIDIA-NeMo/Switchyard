// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parity tests for the OpenAI vs Anthropic usage shapes.
//!
//! OpenAI exposes `prompt_tokens` as the *inclusive* total with
//! `prompt_tokens_details.{cached_tokens, cache_creation_tokens}` as nested
//! subsets. Anthropic exposes `input_tokens` (non-cached base),
//! `cache_read_input_tokens`, and `cache_creation_input_tokens` as
//! *sibling* fields — each disjoint from the others. The extractor in
//! `switchyard_components::stats::usage_from_body` must normalise both
//! into the OpenAI-style inclusive convention so the downstream cost
//! estimator (`base = prompt - cached - cache_creation`) never goes
//! negative and the cache buckets sum into the prompt total. The tests
//! below pin that contract.
//!
//! Replaces the Python `tests/test_cost_estimator_cache_wiring.py` Layer 1
//! cases, which targeted a `_record()` helper that the Rust migration
//! eliminated.

use serde_json::json;
use switchyard_components::TokenUsage;
use switchyard_components::stats::usage_from_body;

#[test]
fn openai_inclusive_shape_extracts_prompt_as_inclusive_total() {
    // OpenAI: `prompt_tokens` already counts the cached + cache_creation tokens.
    let body = json!({
        "usage": {
            "prompt_tokens": 550,
            "completion_tokens": 100,
            "prompt_tokens_details": {
                "cached_tokens": 100,
                "cache_creation_tokens": 50,
            },
        }
    });
    assert_eq!(
        usage_from_body(&body),
        TokenUsage {
            prompt_tokens: 550,
            completion_tokens: 100,
            cached_tokens: 100,
            cache_creation_tokens: 50,
            reasoning_tokens: 0,
            cacheable_prompt_tokens: 0,
        }
    );
}

#[test]
fn anthropic_sibling_shape_sums_into_inclusive_prompt_total() {
    // Anthropic: `input_tokens` is the BASE (non-cached, non-creation) —
    // `cache_read_input_tokens` and `cache_creation_input_tokens` are siblings.
    // The extractor must sum the three to produce the inclusive total.
    let body = json!({
        "usage": {
            "input_tokens": 400,
            "output_tokens": 100,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
        }
    });
    assert_eq!(
        usage_from_body(&body),
        TokenUsage {
            prompt_tokens: 550,
            completion_tokens: 100,
            cached_tokens: 100,
            cache_creation_tokens: 50,
            reasoning_tokens: 0,
            cacheable_prompt_tokens: 0,
        }
    );
}

#[test]
fn openai_and_anthropic_shapes_produce_identical_canonical_output() {
    // Same logical request expressed in both shapes — extractor output
    // must be byte-for-byte identical so cost math doesn't drift between
    // providers.
    let openai_body = json!({
        "usage": {
            "prompt_tokens": 550,
            "completion_tokens": 100,
            "prompt_tokens_details": {
                "cached_tokens": 100,
                "cache_creation_tokens": 50,
            },
        }
    });
    let anthropic_body = json!({
        "usage": {
            "input_tokens": 400,
            "output_tokens": 100,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
        }
    });
    assert_eq!(
        usage_from_body(&openai_body),
        usage_from_body(&anthropic_body)
    );
}

#[test]
fn anthropic_without_cache_fields_keeps_prompt_equal_to_input() {
    // No cache fields: prompt_tokens degenerates to input_tokens.
    let body = json!({
        "usage": {
            "input_tokens": 200,
            "output_tokens": 50,
        }
    });
    assert_eq!(
        usage_from_body(&body),
        TokenUsage {
            prompt_tokens: 200,
            completion_tokens: 50,
            cached_tokens: 0,
            cache_creation_tokens: 0,
            reasoning_tokens: 0,
            cacheable_prompt_tokens: 0,
        }
    );
}

#[test]
fn openai_without_prompt_tokens_details_keeps_cache_counts_zero() {
    let body = json!({
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 50,
        }
    });
    assert_eq!(
        usage_from_body(&body),
        TokenUsage {
            prompt_tokens: 200,
            completion_tokens: 50,
            cached_tokens: 0,
            cache_creation_tokens: 0,
            reasoning_tokens: 0,
            cacheable_prompt_tokens: 0,
        }
    );
}

#[test]
fn openai_reasoning_tokens_extracted_from_completion_tokens_details() {
    let body = json!({
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "completion_tokens_details": {
                "reasoning_tokens": 150,
            },
        }
    });
    let usage = usage_from_body(&body);
    assert_eq!(usage.reasoning_tokens, 150);
    assert_eq!(usage.completion_tokens, 200);
}

#[test]
fn anthropic_reasoning_tokens_extracted_from_output_tokens_details() {
    let body = json!({
        "usage": {
            "input_tokens": 100,
            "output_tokens": 200,
            "output_tokens_details": {
                "reasoning_tokens": 150,
            },
        }
    });
    let usage = usage_from_body(&body);
    assert_eq!(usage.reasoning_tokens, 150);
    assert_eq!(usage.completion_tokens, 200);
}

#[test]
fn missing_usage_block_yields_zero_usage() {
    let body = json!({"id": "msg-1", "content": "hi"});
    assert!(usage_from_body(&body).is_zero());
}

#[test]
fn non_object_usage_block_is_ignored() {
    // Defensive: a malformed `usage: "garbage"` must not panic; extractor
    // returns default rather than half-populating fields.
    let body = json!({"usage": "garbage"});
    assert!(usage_from_body(&body).is_zero());
}

#[test]
fn anthropic_with_input_tokens_details_cached_takes_precedence_over_cache_read() {
    // Defensive corner: when both `input_tokens_details.cached_tokens` and
    // `cache_read_input_tokens` are present, the details object wins so
    // providers that surface both naming conventions don't double-count.
    let body = json!({
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 30,
            "input_tokens_details": {"cached_tokens": 25},
        }
    });
    let usage = usage_from_body(&body);
    assert_eq!(usage.cached_tokens, 25);
}

#[test]
fn cost_estimator_invariant_no_negative_base_input() {
    // The cost estimator computes
    //   base = prompt_tokens - cached_tokens - cache_creation_tokens
    // and relies on this being non-negative. Both shapes must respect that
    // — otherwise the base-input cost line silently clamps to 0 and we
    // under-bill the cache-write cost.
    let cases = [
        json!({"usage": {
            "prompt_tokens": 550,
            "prompt_tokens_details": {"cached_tokens": 100, "cache_creation_tokens": 50},
        }}),
        json!({"usage": {
            "input_tokens": 400,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
        }}),
    ];
    for body in cases {
        let usage = usage_from_body(&body);
        let base = usage
            .prompt_tokens
            .saturating_sub(usage.cached_tokens)
            .saturating_sub(usage.cache_creation_tokens);
        assert_eq!(
            base, 400,
            "{body:?} should leave 400 non-cached prompt tokens"
        );
    }
}
