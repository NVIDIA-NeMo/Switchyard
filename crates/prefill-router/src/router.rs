// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Target binding for checkpoint-backed prefill inference.

use std::collections::BTreeSet;

use switchyard_protocol::ModelId;

use crate::{PrefillForward, PrefillRouterError, Result};

/// Maps ordered prefill probabilities to semantic model identifiers.
pub struct PrefillRouter<F> {
    forward: F,
    targets: Vec<ModelId>,
}

impl<F: PrefillForward> PrefillRouter<F> {
    /// Positionally binds checkpoint heads to unique targets.
    pub fn new(targets: Vec<ModelId>, forward: F) -> Result<Self> {
        let output_count = forward.output_count();
        if targets.is_empty() {
            return Err(configuration_error("at least one target is required"));
        }
        if targets.len() != output_count {
            return Err(configuration_error(format!(
                "received {} targets for {} checkpoint heads",
                targets.len(),
                output_count
            )));
        }
        if targets.iter().collect::<BTreeSet<_>>().len() != targets.len() {
            return Err(configuration_error("targets must be unique"));
        }
        Ok(Self { forward, targets })
    }

    /// Predicts correctness for each target in checkpoint-head order.
    pub fn predict(&mut self, prompt: &str) -> Result<Vec<(ModelId, f32)>> {
        let mut predictions = self.predict_batch(&[prompt.to_string()])?;
        predictions.pop().ok_or_else(|| {
            PrefillRouterError::InvalidResult("forward returned no prediction".to_string())
        })
    }

    /// Predicts correctness for a batch while preserving prompt order.
    pub fn predict_batch(&mut self, prompts: &[String]) -> Result<Vec<Vec<(ModelId, f32)>>> {
        if prompts.is_empty() || prompts.iter().any(String::is_empty) {
            return Err(PrefillRouterError::InvalidRequest(
                "prompts must be non-empty".to_string(),
            ));
        }
        let probabilities = self.forward.forward(prompts)?;
        if probabilities.len() != prompts.len()
            || probabilities.iter().any(|row| {
                row.len() != self.targets.len()
                    || row
                        .iter()
                        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            })
        {
            return Err(PrefillRouterError::InvalidResult(
                "forward returned invalid probabilities".to_string(),
            ));
        }
        Ok(probabilities
            .into_iter()
            .map(|row| self.targets.iter().cloned().zip(row).collect())
            .collect())
    }

    /// Releases resources held by the configured implementation.
    pub fn unload(&mut self) -> Result<()> {
        self.forward.unload()
    }
}

fn configuration_error(message: impl Into<String>) -> PrefillRouterError {
    PrefillRouterError::InvalidConfiguration(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedForward;

    impl PrefillForward for FixedForward {
        fn output_count(&self) -> usize {
            2
        }

        fn forward(&mut self, prompts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![0.25, 0.75]; prompts.len()])
        }

        fn unload(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn maps_probabilities_to_targets() {
        let targets = vec![ModelId::from("small"), ModelId::from("large")];
        let mut router = PrefillRouter::new(targets.clone(), FixedForward).expect("valid router");

        assert_eq!(
            router.predict("route me").expect("valid prediction"),
            vec![(targets[0].clone(), 0.25), (targets[1].clone(), 0.75)]
        );
        assert_eq!(
            router
                .predict_batch(&["first".to_string(), "second".to_string()])
                .expect("valid batch")
                .len(),
            2
        );
    }
}
