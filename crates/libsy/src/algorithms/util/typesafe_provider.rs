// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Port for an externally-judged, non-generative routing decision.
//!
//! This is the seam [`crate::TypeSafeTaskClassifier`] uses to reach a "System One
//! Model" style classifier (for example TypeSafe's Jev) without `libsy` performing
//! any I/O itself. `libsy` depends only on the [`TypeSafeProvider`] trait defined
//! here; the concrete implementation that actually calls out over HTTP lives in a
//! separate, runner-owned crate (see `switchyard-typesafe-client`) and is injected
//! at deployment-load time.
//!
//! This shape directly answers
//! [Switchyard issue #723](https://github.com/NVIDIA-NeMo/Switchyard/issues/723),
//! which asks for such a classifier to be implemented "via an external HTTP
//! judgment step, or a runner-owned classification provider... avoid direct HTTP
//! calls from libsy": the trait is the external judgment step, and the runner owns
//! whatever calls it.

use async_trait::async_trait;
use std::collections::BTreeMap;
use std::fmt;

/// One label a [`TypeSafeProvider`] may return, together with the natural-language
/// criteria describing when it applies.
///
/// Mirrors the `criteria` map a TypeSafe `Choice` question is asked to pick from:
/// each option's `label` is ordinarily one of the deployment's configured routing
/// categories (`capable`, `efficient`, or a deployment-defined name), and
/// `description` is shown to the provider verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeSafeOption {
    /// The label returned verbatim in [`TypeSafeVerdict::label`] when chosen.
    pub label: String,
    /// Natural-language criteria shown to the provider for this label.
    pub description: String,
}

impl TypeSafeOption {
    /// Builds one labeled option.
    pub fn new(label: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: description.into(),
        }
    }
}

/// Conversation material handed to a [`TypeSafeProvider`] for one classification call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeSafeClassifierInput {
    /// The instruction asked of the provider, such as "which tier does this need?".
    pub question: String,
    /// Flattened conversation text the provider classifies. Free text, not a chat
    /// transcript — a "System One Model" scores state, it does not converse.
    pub context: String,
}

/// A provider's resolved decision: the chosen label and its confidence.
#[derive(Clone, Debug, PartialEq)]
pub struct TypeSafeVerdict {
    /// The label the provider chose.
    ///
    /// Not guaranteed to be one of the labels it was offered, or to parse as a
    /// configured routing category — [`crate::TypeSafeTaskClassifier`] treats an
    /// unresolved label the same as a missing verdict (fail-open), rather than
    /// erroring.
    pub label: String,
    /// Confidence in `label`, expected in `[0.0, 1.0]`.
    ///
    /// Not validated at this layer, so a caller comparing it against a threshold
    /// should still treat an out-of-range value defensively.
    pub confidence: f64,
    /// Averaged probability for every configured candidate label.
    pub probabilities: BTreeMap<String, f64>,
    /// Wall-clock time spent waiting for the TypeSafe decision.
    pub decision_latency_ms: u64,
}

/// Opaque failure from a [`TypeSafeProvider`] call.
///
/// [`crate::TypeSafeTaskClassifier`] folds every variant of failure to fail-open —
/// the same request is simply handed to the next classifier in the cascade — so
/// this carries only a display message suitable for logs and telemetry, never
/// structured detail a caller might branch on.
#[derive(Clone, Debug)]
pub struct TypeSafeProviderError(pub String);

impl fmt::Display for TypeSafeProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TypeSafeProviderError {}

/// External, non-generative judgment step a `TypeSafeTaskClassifier` delegates to.
///
/// Implementations perform the actual network call; `libsy` never does. The host
/// (typically `switchyard-runner`, reading credentials from an environment
/// variable — never from TOML) constructs the concrete implementation once per
/// deployment and injects it as `Arc<dyn TypeSafeProvider>`, so this trait is the
/// entire seam between libsy's routing logic and a provider's API.
#[async_trait]
pub trait TypeSafeProvider: Send + Sync {
    /// Classifies `input` against `options`, returning the provider's choice.
    ///
    /// Implementations should never log or otherwise surface request credentials
    /// in the returned error.
    async fn classify(
        &self,
        input: TypeSafeClassifierInput,
        options: &[TypeSafeOption],
    ) -> Result<TypeSafeVerdict, TypeSafeProviderError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_builder_converts_into_string() {
        let option = TypeSafeOption::new("capable", "complex, multi-step work");
        assert_eq!(option.label, "capable");
        assert_eq!(option.description, "complex, multi-step work");
    }

    #[test]
    fn provider_error_displays_its_message() {
        let error = TypeSafeProviderError("boom".to_string());
        assert_eq!(error.to_string(), "boom");
    }
}
