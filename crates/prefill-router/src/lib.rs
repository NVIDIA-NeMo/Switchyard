// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill feature extraction behind a backend-neutral Rust contract.
//!
//! [`TransformersForward`] embeds Python and reproduces the Hugging Face
//! Transformers forward path used by the NVIDIA LLM Router blueprint. Router
//! policy and model selection intentionally live outside this crate boundary.

mod error;
mod transformers;

use std::collections::BTreeMap;

pub use error::{PrefillRouterError, Result};
pub use transformers::{TransformersForward, TransformersForwardConfig};

/// A prefill implementation that extracts pooled hidden states from prompts.
///
/// The contract contains no Python-specific types so another implementation,
/// such as Candle, can replace [`TransformersForward`] without changing users.
pub trait PrefillForward: Send {
    /// Runs one batched prefill and returns the requested hidden-state features.
    fn forward(&mut self, request: &ForwardRequest) -> Result<ForwardOutput>;

    /// Releases model resources held by the implementation.
    fn unload(&mut self) -> Result<()>;
}

/// Inputs for one batched prefill forward pass.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardRequest {
    /// User prompts to format and tokenize.
    pub prompts: Vec<String>,
    /// Extra keyword arguments passed to the tokenizer's chat template.
    pub chat_template_kwargs: serde_json::Map<String, serde_json::Value>,
    /// Hidden-state layers to extract.
    pub layers: LayerSelection,
    /// Pooling modes to calculate for every requested layer.
    pub pooling: Vec<Pooling>,
    /// Maximum number of prompts in each model forward pass.
    pub batch_size: usize,
    /// Maximum tokenized length of each prompt.
    pub max_length: usize,
}

impl ForwardRequest {
    /// Creates a request using the reference Transformers defaults.
    pub fn new(prompts: Vec<String>) -> Self {
        Self {
            prompts,
            chat_template_kwargs: serde_json::Map::new(),
            layers: LayerSelection::UpperHalf,
            pooling: vec![Pooling::Last, Pooling::Mean],
            batch_size: 4,
            max_length: 2_048,
        }
    }
}

/// Hidden-state layers requested from the encoder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum LayerSelection {
    /// Extract the upper half of the encoder layers.
    #[default]
    UpperHalf,
    /// Extract every encoder layer.
    All,
    /// Extract the listed direct hidden-state indexes in order.
    Selected(Vec<usize>),
}

/// Pooling applied across each prompt's non-padding tokens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Pooling {
    /// Select the final non-padding token.
    Last,
    /// Average all non-padding tokens.
    Mean,
}

impl Pooling {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Last => "last",
            Self::Mean => "mean",
        }
    }
}

/// Pooled hidden states returned by one forward pass.
///
/// Each layer maps to one hidden vector per input prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardOutput {
    /// Last-token vectors keyed by direct hidden-state index.
    pub hidden_last: BTreeMap<usize, Vec<Vec<f32>>>,
    /// Mean-pooled vectors keyed by direct hidden-state index.
    pub hidden_mean: BTreeMap<usize, Vec<Vec<f32>>>,
    /// Number of transformer layers reported by the model.
    pub n_layers: usize,
    /// Width of each hidden-state vector.
    pub hidden_dim: usize,
}

impl ForwardOutput {
    pub(crate) fn validate(self, prompt_count: usize) -> Result<Self> {
        if self.n_layers == 0 || self.hidden_dim == 0 {
            return Err(PrefillRouterError::InvalidResult(
                "model dimensions must be non-zero".to_string(),
            ));
        }
        if self.hidden_last.is_empty() && self.hidden_mean.is_empty() {
            return Err(PrefillRouterError::InvalidResult(
                "no hidden states were returned".to_string(),
            ));
        }

        for (pooling, layers) in [("last", &self.hidden_last), ("mean", &self.hidden_mean)] {
            for (layer, rows) in layers {
                if *layer >= self.n_layers {
                    return Err(PrefillRouterError::InvalidResult(format!(
                        "{pooling} layer {layer} is outside encoder range 0..{}",
                        self.n_layers - 1
                    )));
                }
                if rows.len() != prompt_count {
                    return Err(PrefillRouterError::InvalidResult(format!(
                        "{pooling} layer {layer} returned {} rows for {prompt_count} prompts",
                        rows.len()
                    )));
                }
                if rows.iter().any(|row| row.len() != self.hidden_dim) {
                    return Err(PrefillRouterError::InvalidResult(format!(
                        "{pooling} layer {layer} contains a vector with the wrong hidden width"
                    )));
                }
            }
        }

        Ok(self)
    }
}
