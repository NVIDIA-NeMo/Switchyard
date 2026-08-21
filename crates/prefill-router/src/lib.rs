// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill feature extraction behind a backend-neutral Rust contract.
//!
//! [`TransformersForward`] embeds Python and reproduces the Hugging Face
//! Transformers forward path used by the NVIDIA LLM Router blueprint. Router
//! policy and model selection intentionally live outside this crate boundary.

mod error;
mod transformers;

use std::collections::{BTreeMap, BTreeSet};

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
    hidden_last: BTreeMap<usize, Vec<Vec<f32>>>,
    /// Mean-pooled vectors keyed by direct hidden-state index.
    hidden_mean: BTreeMap<usize, Vec<Vec<f32>>>,
    /// Number of transformer layers reported by the model.
    n_layers: usize,
    /// Width of each hidden-state vector.
    hidden_dim: usize,
}

impl ForwardOutput {
    pub(crate) fn parse(
        request: &ForwardRequest,
        hidden_last: BTreeMap<usize, Vec<Vec<f32>>>,
        hidden_mean: BTreeMap<usize, Vec<Vec<f32>>>,
        n_layers: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        if n_layers == 0 || hidden_dim == 0 {
            return Err(PrefillRouterError::InvalidResult(
                "model dimensions must be non-zero".to_string(),
            ));
        }

        let expected_layers: BTreeSet<usize> = match &request.layers {
            LayerSelection::UpperHalf => (n_layers / 2..n_layers).collect(),
            LayerSelection::All => (0..n_layers).collect(),
            LayerSelection::Selected(layers) => layers.iter().copied().collect(),
        };
        if expected_layers.is_empty() || expected_layers.iter().any(|layer| *layer >= n_layers) {
            return Err(PrefillRouterError::InvalidResult(
                "requested layers are empty or outside the encoder range".to_string(),
            ));
        }

        let wants_last = request.pooling.contains(&Pooling::Last);
        let wants_mean = request.pooling.contains(&Pooling::Mean);
        if !wants_last && !wants_mean {
            return Err(PrefillRouterError::InvalidResult(
                "at least one pooling mode is required".to_string(),
            ));
        }

        parse_pooling(
            "last",
            wants_last,
            &hidden_last,
            &expected_layers,
            request.prompts.len(),
            hidden_dim,
        )?;
        parse_pooling(
            "mean",
            wants_mean,
            &hidden_mean,
            &expected_layers,
            request.prompts.len(),
            hidden_dim,
        )?;

        Ok(Self {
            hidden_last,
            hidden_mean,
            n_layers,
            hidden_dim,
        })
    }

    /// Returns last-token vectors keyed by hidden-state layer.
    pub fn hidden_last(&self) -> &BTreeMap<usize, Vec<Vec<f32>>> {
        &self.hidden_last
    }

    /// Returns mean-pooled vectors keyed by hidden-state layer.
    pub fn hidden_mean(&self) -> &BTreeMap<usize, Vec<Vec<f32>>> {
        &self.hidden_mean
    }

    /// Returns the number of transformer layers reported by the model.
    pub const fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Returns the width of each hidden-state vector.
    pub const fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

fn parse_pooling(
    name: &str,
    requested: bool,
    layers: &BTreeMap<usize, Vec<Vec<f32>>>,
    expected_layers: &BTreeSet<usize>,
    prompt_count: usize,
    hidden_dim: usize,
) -> Result<()> {
    let actual_layers = layers.keys().copied().collect::<BTreeSet<_>>();
    let expected = if requested {
        expected_layers.clone()
    } else {
        BTreeSet::new()
    };
    if actual_layers != expected {
        return Err(PrefillRouterError::InvalidResult(format!(
            "{name} pooling returned layers {actual_layers:?}, expected {expected:?}"
        )));
    }

    for (layer, rows) in layers {
        if rows.len() != prompt_count {
            return Err(PrefillRouterError::InvalidResult(format!(
                "{name} layer {layer} returned {} rows for {prompt_count} prompts",
                rows.len()
            )));
        }
        if rows.iter().any(|row| row.len() != hidden_dim) {
            return Err(PrefillRouterError::InvalidResult(format!(
                "{name} layer {layer} contains a vector with the wrong hidden width"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_parser_requires_every_requested_pooling_and_layer() {
        let mut request = ForwardRequest::new(vec!["test prompt".to_string()]);
        request.layers = LayerSelection::Selected(vec![0, 1]);

        let pooled = || BTreeMap::from([(0, vec![vec![0.0, 1.0]]), (1, vec![vec![2.0, 3.0]])]);
        let output = ForwardOutput::parse(&request, pooled(), pooled(), 2, 2)
            .expect("complete output should parse");
        assert_eq!(output.hidden_last().len(), 2);
        assert_eq!(output.hidden_mean().len(), 2);

        assert!(matches!(
            ForwardOutput::parse(&request, pooled(), BTreeMap::new(), 2, 2),
            Err(PrefillRouterError::InvalidResult(_))
        ));
        assert!(matches!(
            ForwardOutput::parse(
                &request,
                BTreeMap::from([(0, vec![vec![0.0, 1.0]])]),
                pooled(),
                2,
                2,
            ),
            Err(PrefillRouterError::InvalidResult(_))
        ));
    }
}
