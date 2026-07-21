// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public API tests for the components-v2 random-routing profile config.

use serde_json::json;
use switchyard_components_v2::{ProfileConfig, RandomRoutingProfileConfig};
use switchyard_core::{BackendFormat, LlmTarget, LlmTargetId, ModelId, Result, SwitchyardError};

fn target(id: &str, model: &str) -> Result<LlmTarget> {
    let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
    target.format = BackendFormat::OpenAi;
    Ok(target)
}

fn config(strong: LlmTarget, weak: LlmTarget, probability: f64) -> RandomRoutingProfileConfig {
    RandomRoutingProfileConfig {
        strong,
        weak,
        strong_probability: probability,
        rng_seed: Some(7),
        fallback_target_on_evict: None,
    }
}

#[test]
fn profile_config_build_uses_existing_native_backend_stack() -> Result<()> {
    let config = config(
        target("strong", "frontier/model")?,
        target("weak", "cheap/model")?,
        0.5,
    );

    let _profile = config.build()?;
    Ok(())
}

#[test]
fn profile_config_macro_adds_type_metadata_and_strict_serde() -> Result<()> {
    let config = config(
        target("strong", "frontier/model")?,
        target("weak", "cheap/model")?,
        0.5,
    );

    assert_eq!(RandomRoutingProfileConfig::PROFILE_TYPE, "random-routing");
    assert_eq!(config.profile_type(), "random-routing");

    let old_stats_toggle = json!({
        "strong": config.strong,
        "weak": config.weak,
        "strong_probability": 0.5,
        "rng_seed": 7,
        "fallback_target_on_evict": "strong",
        "stats": false,
    });
    let error = serde_json::from_value::<RandomRoutingProfileConfig>(old_stats_toggle)
        .err()
        .ok_or_else(|| SwitchyardError::Other("unknown profile field should fail".into()))?;
    assert!(error.to_string().contains("unknown field"));
    Ok(())
}

#[test]
fn invalid_probability_is_rejected_by_profile_config_build() -> Result<()> {
    let config = config(
        target("strong", "frontier/model")?,
        target("weak", "cheap/model")?,
        1.5,
    );

    match config.build() {
        Err(SwitchyardError::InvalidConfig(message)) => {
            assert!(message.contains("strong_probability must be finite and in [0.0, 1.0]"));
        }
        Ok(_) => {
            return Err(SwitchyardError::Other(
                "invalid probability should reject profile construction".into(),
            ));
        }
        Err(other) => {
            return Err(SwitchyardError::Other(format!(
                "expected InvalidConfig, got {other}"
            )));
        }
    }
    Ok(())
}
