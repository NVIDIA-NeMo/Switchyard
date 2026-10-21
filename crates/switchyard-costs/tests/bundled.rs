// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use switchyard_costs::{CostSource, Pricing};
use switchyard_protocol::Usage;

#[test]
fn bundled_prices_cover_direct_gateway_and_free_models() {
    let pricing = Pricing::bundled().expect("bundled prices");
    // Cache tokens are counted apart from input tokens, so each total also checks the
    // cache-read and cache-write rates. The prompt stays below Haiku 5.5's 100,000-token limit.
    let usage = Usage {
        input_tokens: Some(10_000),
        output_tokens: Some(1_000),
        cache: Usage::cache_details(Some(1_000), Some(1_000)),
        ..Usage::default()
    };
    // These totals are computed by hand from prices.toml.
    for (model, expected) in [
        ("claude-fable-5-1", 0.16275),
        ("anthropic/claude-haiku-5.5", 0.001635),
        ("gemini-4-argon", 0.0321),
        ("gemini-3.8-flash", 0.012075),
        ("google/gemini-3.8-flash", 0.0132825),
        ("deepseek-flash", 0.004506),
        ("x-ai/grok-4.7", 0.0285),
        ("qwen/qwen3.7-plus", 0.004944),
        ("moonshotai/kimi-k2.6", 0.01071),
        ("nvidia/nemotron-3.5-lightning", 0.00101),
        ("meta/muse-wonder", 0.0642),
        ("stepfun/step-5-preview", 0.0138),
        ("minimax/minimax-m3.1-flash:free", 0.0),
        ("z-ai/glm-5.3", 0.002347),
        ("tencent/hy4-preview", 0.0105453),
    ] {
        let cost = pricing.estimate(model, &usage).expect(model);
        assert_eq!(cost.source, CostSource::Estimated);
        assert!((cost.usd - expected).abs() < 1e-10, "{model}: {cost:?}");
    }
    for model in ["gemini-4-argon-typo", "unknown/gemini-4-argon"] {
        assert!(
            pricing.estimate(model, &usage).is_none(),
            "{model} should have no rate"
        );
    }
    // A provider-reported charge of zero takes precedence over the bundled rates.
    let reported = Usage {
        provider_cost: Some(0.0),
        ..usage
    };
    let cost = pricing.cost("gemini-4-argon", &reported).expect("charge");
    assert_eq!(cost.usd, 0.0);
    assert_eq!(cost.source, CostSource::ProviderReported);
}
