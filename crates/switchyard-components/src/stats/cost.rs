// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Best-effort token cost estimation for stats snapshots.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::accumulator::ModelStatsSnapshot;

/// Cost estimate for all recorded models.
///
/// `total_cost` is the grand total spend including overhead calls;
/// `backend_cost` is the portion attributed to routed-backend traffic;
/// `classifier_cost` is the LLM-classifier-overhead portion (zero unless
/// `record_classifier_usage` has been called);
/// `planner_cost` is the planner-overhead portion (zero unless
/// `record_planner_usage` has been called). The split exists because the
/// default TB-lite configs can use the same model id for multiple
/// buckets and a single-row aggregation cannot distinguish them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostEstimate {
    pub models: BTreeMap<String, CostBreakdown>,
    pub total_cost: f64,
    pub backend_cost: f64,
    pub classifier_cost: f64,
    pub planner_cost: f64,
}

/// Per-model cost breakdown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub base_input_cost: f64,
    pub cached_input_cost: f64,
    pub cache_write_cost: f64,
    pub input_cost: f64,
    pub output_cost: f64,
    pub total_cost: f64,
}

#[derive(Clone, Copy, Debug)]
struct ModelPrice {
    input: f64,
    output: f64,
    cached: f64,
    cache_write: f64,
}

/// Estimates token cost from per-model stats.
pub fn estimate_cost(models: &BTreeMap<String, ModelStatsSnapshot>) -> CostEstimate {
    let mut estimated = BTreeMap::new();
    let mut total_cost = 0.0;

    for (model, stats) in models {
        let breakdown = estimate_model_cost(
            model,
            stats.prompt_tokens,
            stats.completion_tokens,
            stats.cached_tokens,
            stats.cache_creation_tokens,
        );
        total_cost += breakdown.total_cost;
        estimated.insert(model.clone(), breakdown);
    }

    let total = round6(total_cost);
    CostEstimate {
        models: estimated,
        total_cost: total,
        backend_cost: total,
        classifier_cost: 0.0,
        planner_cost: 0.0,
    }
}

pub fn estimate_model_cost(
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
) -> CostBreakdown {
    let prices = raw_model_price(model).unwrap_or(ModelPrice {
        input: 0.0,
        output: 0.0,
        cached: 0.0,
        cache_write: 0.0,
    });
    let base_input = prompt_tokens
        .saturating_sub(cached_tokens)
        .saturating_sub(cache_creation_tokens);
    let base_input_cost = base_input as f64 / 1e6 * prices.input;
    let cached_input_cost = cached_tokens as f64 / 1e6 * prices.cached;
    let cache_write_cost = cache_creation_tokens as f64 / 1e6 * prices.cache_write;
    let input_cost = base_input_cost + cached_input_cost + cache_write_cost;
    let output_cost = completion_tokens as f64 / 1e6 * prices.output;
    CostBreakdown {
        base_input_cost: round6(base_input_cost),
        cached_input_cost: round6(cached_input_cost),
        cache_write_cost: round6(cache_write_cost),
        input_cost: round6(input_cost),
        output_cost: round6(output_cost),
        total_cost: round6(input_cost + output_cost),
    }
}

pub fn has_model_price(model: &str) -> bool {
    raw_model_price(model).is_some()
}

fn raw_model_price(model: &str) -> Option<ModelPrice> {
    let price = match model {
        "openai/openai/gpt-5.2" | "openai/openai/openai/gpt-5.2" => ModelPrice {
            input: 1.75,
            output: 14.00,
            cached: 0.175,
            cache_write: 1.75,
        },
        "nvidia/nvidia/nemotron-3-super-v3" | "openai/nvidia/nvidia/nemotron-3-super-v3" => {
            ModelPrice {
                input: 0.10,
                output: 0.50,
                cached: 0.01,
                cache_write: 0.10,
            }
        }
        // Moonshot Kimi K2.6 — official platform.kimi.ai pricing (May 2026).
        // OpenAI wire format on NVIDIA Inference Hub; no cache_write
        // premium (cache_write equals input).
        "nvidia/moonshotai/kimi-k2.6" | "openai/nvidia/moonshotai/kimi-k2.6" => ModelPrice {
            input: 0.95,
            output: 4.00,
            cached: 0.16,
            cache_write: 0.95,
        },
        // Moonshot Kimi K2 / K2.5 — platform.kimi.ai (k2-thinking variant
        // pricing, the closest standard tier; NVIDIA hub serves this as
        // ``kimi-k2.5``).  Same no-cache-write-premium posture.
        "nvidia/moonshotai/kimi-k2.5" | "openai/nvidia/moonshotai/kimi-k2.5" => ModelPrice {
            input: 0.60,
            output: 2.50,
            cached: 0.15,
            cache_write: 0.60,
        },
        // DeepSeek V4 Flash — official api-docs.deepseek.com standard list
        // price (post-promo).  284B total / 13B active, 1M-token context
        // window.  Aggressive cache discount (98% off on hits).  OpenAI
        // wire format on NVIDIA hub; no cache_write premium.
        "nvidia/deepseek-ai/deepseek-v4-flash"
        | "openai/nvidia/deepseek-ai/deepseek-v4-flash"
        | "deepseek-v4-flash" => ModelPrice {
            input: 0.14,
            output: 0.28,
            cached: 0.0028,
            cache_write: 0.14,
        },
        // DeepSeek V4 Pro — official api-docs.deepseek.com standard list
        // price (post-promo).  1.6T total / 49B active, 1M-token context
        // window.  Pro tier is currently under a 75% promotional discount
        // through 2026-05-31 UTC ($0.435 / $0.87 effective); we price at
        // the standard rate for stable cost-model comparisons that
        // outlive the promo window.  ``evals-`` prefix variant is the
        // same model exposed on NVIDIA Inference Hub's benchmarking
        // gateway (paired with the ``X-Inference-Priority: batch``
        // header) — it bypasses the regular gateway's 6-min timeout
        // that under high concurrency manifests as cascading 504s on
        // V4-class models.  Same per-token pricing.
        "nvidia/deepseek-ai/deepseek-v4-pro"
        | "openai/nvidia/deepseek-ai/deepseek-v4-pro"
        | "nvidia/deepseek-ai/evals-deepseek-v4-pro"
        | "openai/nvidia/deepseek-ai/evals-deepseek-v4-pro"
        | "deepseek-v4-pro" => ModelPrice {
            input: 1.74,
            output: 3.48,
            cached: 0.0145,
            cache_write: 1.74,
        },
        // Gemini 3.5 Flash — Google Vertex global list price (ai.google.dev):
        // input $1.50, output $9.00, cache-read $0.15 (90% off input). The
        // default LLM-classifier model (gcp wire). No per-token cache-write
        // premium (cache storage is billed per-hour, not per-token), so
        // cache_write = input.
        "gcp/google/gemini-3.5-flash"
        | "openai/gcp/google/gemini-3.5-flash"
        | "gemini-3.5-flash" => ModelPrice {
            input: 1.50,
            output: 9.00,
            cached: 0.15,
            cache_write: 1.50,
        },
        // Nemotron 3 Ultra (550B/55B MoE) — OpenRouter reference list
        // price, June 2026. evals-N ids are parallel benchmarking-gateway
        // deployments of the same model.
        "nvidia/nvidia/nemotron-3-ultra"
        | "openai/nvidia/nvidia/nemotron-3-ultra"
        | "nvidia/nvidia/evals-nemotron-ultra"
        | "nvidia/nvidia/evals-nemotron-ultra-2"
        | "nvidia/nvidia/evals-nemotron-ultra-3"
        | "nvidia/nvidia/evals-nemotron-ultra-4" => ModelPrice {
            input: 0.50,
            output: 2.20,
            cached: 0.05,
            cache_write: 0.50,
        },
        "aws/anthropic/bedrock-claude-opus-4-8"
        | "aws/anthropic/bedrock-claude-opus-4-7"
        | "aws/anthropic/bedrock-claude-opus-4-6"
        | "aws/anthropic/bedrock-claude-opus-4-5"
        | "azure/anthropic/claude-opus-4-8"
        | "azure/anthropic/claude-opus-4-7"
        | "azure/anthropic/claude-opus-4-6"
        | "claude-opus-4-8"
        | "claude-opus-4-7"
        | "claude-opus-4-6"
        | "claude-opus-4-5" => ModelPrice {
            input: 5.00,
            output: 25.00,
            cached: 0.50,
            cache_write: 6.25,
        },
        "aws/anthropic/bedrock-claude-sonnet-4-6"
        | "aws/anthropic/bedrock-claude-sonnet-4-5"
        | "azure/anthropic/claude-sonnet-4-5"
        | "claude-sonnet-4-6"
        | "claude-sonnet-4-5" => ModelPrice {
            input: 3.00,
            output: 15.00,
            cached: 0.30,
            cache_write: 3.75,
        },
        "aws/anthropic/bedrock-claude-haiku-4-5" | "claude-haiku-4-5" => ModelPrice {
            input: 1.00,
            output: 5.00,
            cached: 0.10,
            cache_write: 1.25,
        },
        _ => return None,
    };
    Some(price)
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_3_5_flash_is_priced() {
        // The default LLM-classifier model must not silently cost $0.
        for model in [
            "gcp/google/gemini-3.5-flash",
            "openai/gcp/google/gemini-3.5-flash",
            "gemini-3.5-flash",
        ] {
            assert!(has_model_price(model), "{model} should be priced");
            let cost = estimate_model_cost(model, 1_000_000, 1_000_000, 0, 0);
            assert_eq!(cost.base_input_cost, 1.50);
            assert_eq!(cost.output_cost, 9.00);
            assert_eq!(cost.total_cost, 10.50);
        }
    }

    #[test]
    fn unknown_model_defaults_to_zero() {
        assert!(!has_model_price("nvidia/qwen/qwen3.6-35b-a3b"));
        let cost = estimate_model_cost("nvidia/qwen/qwen3.6-35b-a3b", 1_000_000, 1_000_000, 0, 0);
        assert_eq!(cost.total_cost, 0.0);
    }
}
