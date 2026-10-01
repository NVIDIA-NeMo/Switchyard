// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral decision questions and answers, separate from LLM messages.
//!
//! Enums use snake-case `type` tags and a `data` payload in serialized form.
//! Fields are public; providers and callers are responsible for valid values.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ModelId, Usage};

/// Answer probability on a `[0, 1]` scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Probability(pub f64);

/// Position in the request's rubric, including fractional positions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScoreValue(pub f64);

/// Provider confidence; its scale and meaning are provider-specific.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProviderConfidence(pub f64);

/// Shared context evaluated against independent, named questions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionRequest {
    /// Optional until a target is selected.
    pub model: Option<ModelId>,
    /// Conversation, application state, or other material to evaluate.
    pub context: Value,
    /// Independent questions keyed by their IDs.
    pub questions: BTreeMap<String, DecisionQuestion>,
}

/// Instructions and the expected answer shape for one question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionQuestion {
    /// Structured or textual instructions shared with the provider.
    pub instructions: Value,
    /// Expected answer shape.
    pub kind: DecisionKind,
}

/// The answer shape and any options or ordered rubric levels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum DecisionKind {
    /// A Boolean judgment or probability of true.
    Boolean {
        /// Meaning of a true answer, when needed.
        true_description: Option<Value>,
        /// Meaning of a false answer, when needed.
        false_description: Option<Value>,
    },
    /// Select one of the declared options.
    Choice {
        /// Nonempty options with unique IDs, preserving caller order.
        options: Vec<ChoiceOption>,
    },
    /// A position on an ordered rubric, not an arbitrary numeric measurement.
    Score {
        /// At least two levels, ordered low to high and indexed from zero.
        levels: Vec<Value>,
    },
}

/// An identified choice with an optional structured description.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChoiceOption {
    /// Stable identifier used by choice answers and distributions.
    pub id: String,
    /// Meaning of this option, when its ID alone is insufficient.
    pub description: Option<Value>,
}

/// Answers keyed by the matching request's question IDs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionResponse {
    /// Provider-reported response identifier.
    pub id: Option<String>,
    /// Provider-reported model identifier.
    pub model: Option<ModelId>,
    /// Typed answers corresponding to the request's questions.
    pub answers: BTreeMap<String, DecisionAnswer>,
    /// Available token counts; absent counts remain unknown.
    #[serde(default)]
    pub usage: Usage,
}

/// An answer and separate, optional provider confidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionAnswer {
    /// The estimate for the matching question.
    pub value: DecisionValue,
    /// Provider-specific confidence, distinct from answer probabilities.
    pub provider_confidence: Option<ProviderConfidence>,
}

/// A typed estimate; missing distributions remain unknown.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum DecisionValue {
    /// A Boolean judgment or probability, without an implicit threshold.
    Boolean(BooleanEstimate),
    /// One selected option with an optional complete distribution.
    Choice {
        /// Must name an option in the matching question.
        selected: String,
        /// Maps every declared option ID to its probability when available.
        probabilities: Option<BTreeMap<String, Probability>>,
    },
    /// A fractional position in the matching request's rubric.
    Score {
        /// Must lie in `0..=N-1` for the request's N levels.
        value: ScoreValue,
        /// Follows the request's level order. Retain that request to interpret it.
        probabilities: Option<Vec<Probability>>,
    },
}

/// Preserves Boolean-only answers without inventing probability or certainty.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum BooleanEstimate {
    /// A Boolean judgment with no probability supplied.
    Value(bool),
    /// Probability of true; algorithms choose their own thresholds.
    ProbabilityTrue(Probability),
}
