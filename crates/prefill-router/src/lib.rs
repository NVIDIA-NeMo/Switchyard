// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Learned prefill routing behind a backend-neutral forward contract.

mod algorithm;
mod error;
mod router;
mod transformers;

pub use algorithm::{PrefillRouterAlgo, PrefillRouterConfig};
pub use error::{PrefillRouterError, Result};
pub use router::PrefillRouter;
pub use transformers::{TransformersForward, TransformersForwardConfig};

/// A complete prefill forward pass from prompt to per-model probabilities.
///
/// Implementations own feature extraction and learned checkpoint inference, so
/// callers do not depend on intermediate tensor layouts or a specific runtime.
pub trait PrefillForward: Send {
    /// Returns the number of ordered model probabilities produced per prompt.
    fn output_count(&self) -> usize;

    /// Predicts one ordered probability row per prompt.
    fn forward(&mut self, prompts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Releases resources held by the implementation.
    fn unload(&mut self) -> Result<()>;
}
