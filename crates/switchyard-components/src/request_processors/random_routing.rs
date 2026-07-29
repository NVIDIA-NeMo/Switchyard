// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime support for random routing.

use std::fmt;

use crate::{LlmTarget, LlmTargetId, ModelId, Result, SwitchyardError};
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

const DEFAULT_STRONG_PROBABILITY: f64 = 0.5;

/// Named side of a strong/weak random-routing decision.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomRoutingTier {
    Strong,
    Weak,
}

impl RandomRoutingTier {
    /// Returns the stable lowercase tier label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strong => "strong",
            Self::Weak => "weak",
        }
    }
}

/// Runtime config for weighted random routing between two LLM targets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomRoutingProcessorConfig {
    pub strong: LlmTarget,
    pub weak: LlmTarget,
    pub strong_probability: f64,
    pub rng_seed: Option<u64>,
}

impl RandomRoutingProcessorConfig {
    /// Creates a config with a 50/50 split and entropy-backed randomness.
    pub fn new(strong: LlmTarget, weak: LlmTarget) -> Self {
        Self {
            strong,
            weak,
            strong_probability: DEFAULT_STRONG_PROBABILITY,
            rng_seed: None,
        }
    }

    /// Creates a config from just strong and weak model names.
    pub fn from_models(
        strong_model: impl Into<String>,
        weak_model: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self::new(
            LlmTarget::new(
                LlmTargetId::from_static("strong"),
                ModelId::new(strong_model)?,
            ),
            LlmTarget::new(LlmTargetId::from_static("weak"), ModelId::new(weak_model)?),
        ))
    }

    /// Sets the probability of selecting the strong tier.
    pub fn with_strong_probability(mut self, strong_probability: f64) -> Result<Self> {
        validate_probability(strong_probability)?;
        self.strong_probability = strong_probability;
        Ok(self)
    }

    /// Sets an optional seed for deterministic routing sequences.
    pub fn with_rng_seed(mut self, rng_seed: impl Into<Option<u64>>) -> Self {
        self.rng_seed = rng_seed.into();
        self
    }

    /// Validates the random-routing configuration.
    pub fn validate(&self) -> Result<()> {
        validate_probability(self.strong_probability)
    }
}

/// Captures the chosen strong/weak side and target for downstream components.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomRoutingDecision {
    pub tier: RandomRoutingTier,
    pub selected_target: LlmTargetId,
    pub selected_model: ModelId,
    pub original_model: Option<String>,
    pub strong_probability: f64,
    pub draw: f64,
}

/// Pure random-routing engine decoupled from request mutation.
pub struct RandomRoutingEngine {
    config: RandomRoutingProcessorConfig,
    rng: Mutex<StdRng>,
}

impl fmt::Debug for RandomRoutingEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RandomRoutingEngine")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl RandomRoutingEngine {
    /// Creates a routing engine and initializes its random number generator.
    pub fn new(config: RandomRoutingProcessorConfig) -> Result<Self> {
        config.validate()?;
        let rng = match config.rng_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Ok(Self {
            config,
            rng: Mutex::new(rng),
        })
    }

    /// Returns the immutable routing configuration.
    pub fn config(&self) -> &RandomRoutingProcessorConfig {
        &self.config
    }

    /// Selects a target without mutating a request.
    pub fn select(&self, original_model: Option<String>) -> Result<RandomRoutingDecision> {
        let draw = self.next_draw();
        let tier = if draw < self.config.strong_probability {
            RandomRoutingTier::Strong
        } else {
            RandomRoutingTier::Weak
        };
        let selected = self.tier_config(tier);
        Ok(RandomRoutingDecision {
            tier,
            selected_target: selected.id.clone(),
            selected_model: selected.model.clone(),
            original_model,
            strong_probability: self.config.strong_probability,
            draw,
        })
    }

    // Returns the configured target for the selected tier.
    fn tier_config(&self, tier: RandomRoutingTier) -> &LlmTarget {
        match tier {
            RandomRoutingTier::Strong => &self.config.strong,
            RandomRoutingTier::Weak => &self.config.weak,
        }
    }

    // Draws the next probability sample.
    fn next_draw(&self) -> f64 {
        self.rng.lock().gen()
    }
}

// Validates the weighted random-routing probability.
fn validate_probability(strong_probability: f64) -> Result<()> {
    if strong_probability.is_finite() && (0.0..=1.0).contains(&strong_probability) {
        return Ok(());
    }
    Err(SwitchyardError::InvalidConfig(format!(
        "strong_probability must be finite and in [0.0, 1.0], got {strong_probability:?}"
    )))
}
