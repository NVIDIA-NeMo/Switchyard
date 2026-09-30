// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral decision questions and answers, separate from LLM messages.
//!
//! Requests use checked construction and deserialization. Responses must be
//! constructed or decoded with their request to check answer kinds and rubrics.
//! Enums use snake-case `type` tags and a `data` payload in serialized form.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Value;

use crate::{ModelId, Usage};

/// Maximum absolute error in a distribution's sum. Values are never renormalized.
pub const DISTRIBUTION_SUM_TOLERANCE: f64 = 1e-6;

/// A decision value or message violates the protocol contract.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct DecisionError(String);

/// A finite probability in `[0, 1]`, serialized as a number.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Probability(f64);

impl Probability {
    /// Rejects non-finite numbers and values outside `[0, 1]`.
    pub fn new(value: f64) -> Result<Self, DecisionError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(DecisionError(
                "probability must be finite and in [0, 1]".into(),
            ));
        }
        Ok(Self(value))
    }

    /// Returns the probability as a number.
    pub fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Probability {
    type Error = DecisionError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Probability> for f64 {
    fn from(value: Probability) -> Self {
        value.get()
    }
}

/// A finite, nonnegative rubric position, including fractional positions.
/// The response constructor also checks the request's upper bound.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct ScoreValue(f64);

impl ScoreValue {
    /// Rejects non-finite and negative positions.
    pub fn new(value: f64) -> Result<Self, DecisionError> {
        if !value.is_finite() || value < 0.0 {
            return Err(DecisionError("score must be finite and nonnegative".into()));
        }
        Ok(Self(value))
    }

    /// Returns the rubric position.
    pub fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for ScoreValue {
    type Error = DecisionError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ScoreValue> for f64 {
    fn from(value: ScoreValue) -> Self {
        value.get()
    }
}

/// Finite provider confidence; its scale and meaning are provider-specific.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct ProviderConfidence(f64);

impl ProviderConfidence {
    /// Rejects non-finite confidence values without imposing a probability scale.
    pub fn new(value: f64) -> Result<Self, DecisionError> {
        if !value.is_finite() {
            return Err(DecisionError("provider confidence must be finite".into()));
        }
        Ok(Self(value))
    }

    /// Returns the provider's confidence value.
    pub fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for ProviderConfidence {
    type Error = DecisionError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ProviderConfidence> for f64 {
    fn from(value: ProviderConfidence) -> Self {
        value.get()
    }
}

/// Shared context evaluated against independent, named questions.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecisionRequest {
    /// Optional until a target is selected.
    pub model: Option<ModelId>,
    /// Conversation, application state, or other material to evaluate.
    pub context: Value,
    questions: BTreeMap<String, DecisionQuestion>,
}

impl DecisionRequest {
    /// Requires questions, nonempty unique choice options, and at least two score levels.
    pub fn new(
        model: Option<ModelId>,
        context: Value,
        questions: BTreeMap<String, DecisionQuestion>,
    ) -> Result<Self, DecisionError> {
        if questions.is_empty() {
            return Err(DecisionError(
                "request must contain at least one question".into(),
            ));
        }
        for (id, question) in &questions {
            match &question.kind {
                DecisionKind::Choice { options } => {
                    let ids: BTreeSet<_> = options.iter().map(|option| &option.id).collect();
                    if options.is_empty() || ids.len() != options.len() {
                        return Err(DecisionError(format!(
                            "question {id:?}: choice options must be nonempty with unique IDs"
                        )));
                    }
                }
                DecisionKind::Score { levels } if levels.len() < 2 => {
                    return Err(DecisionError(format!(
                        "question {id:?}: score requires at least two levels"
                    )));
                }
                _ => {}
            }
        }
        Ok(Self {
            model,
            context,
            questions,
        })
    }

    /// Questions are immutable so their checked rubrics remain valid.
    pub fn questions(&self) -> &BTreeMap<String, DecisionQuestion> {
        &self.questions
    }
}

impl<'de> Deserialize<'de> for DecisionRequest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct RawRequest {
            model: Option<ModelId>,
            context: Value,
            #[serde(deserialize_with = "unique_map")]
            questions: BTreeMap<String, DecisionQuestion>,
        }
        let raw = RawRequest::deserialize(deserializer)?;
        Self::new(raw.model, raw.context, raw.questions).map_err(de::Error::custom)
    }
}

/// Instructions and the expected answer shape for one question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionQuestion {
    /// Structured or textual instructions shared with the provider.
    pub instructions: Value,
    /// Checked against the answer when constructing a response.
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

/// Complete answers checked against a request.
///
/// Use [`Self::deserialize`] with the matching request to decode a response.
/// There is no request-free `Deserialize` implementation.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecisionResponse {
    /// Provider-reported response identifier.
    pub id: Option<String>,
    /// Provider-reported model identifier.
    pub model: Option<ModelId>,
    answers: BTreeMap<String, DecisionAnswer>,
    /// Available token counts; absent counts remain unknown.
    pub usage: Usage,
}

impl DecisionResponse {
    /// Checks answer coverage, kinds, selected options, score bounds, and distributions.
    /// Score/distribution consistency remains the provider's responsibility.
    pub fn new(
        request: &DecisionRequest,
        id: Option<String>,
        model: Option<ModelId>,
        answers: BTreeMap<String, DecisionAnswer>,
        usage: Usage,
    ) -> Result<Self, DecisionError> {
        if !answers.keys().eq(request.questions.keys()) {
            return Err(DecisionError(
                "answer IDs must exactly match question IDs".into(),
            ));
        }
        for (id, question) in &request.questions {
            check_answer(&question.kind, &answers[id].value)
                .map_err(|error| DecisionError(format!("question {id:?}: {error}")))?;
        }
        Ok(Self {
            id,
            model,
            answers,
            usage,
        })
    }

    /// Decodes the canonical representation and checks it against its request.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        request: &DecisionRequest,
        deserializer: D,
    ) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct RawResponse {
            id: Option<String>,
            model: Option<ModelId>,
            #[serde(deserialize_with = "unique_map")]
            answers: BTreeMap<String, DecisionAnswer>,
            #[serde(default)]
            usage: Usage,
        }
        let raw = RawResponse::deserialize(deserializer)?;
        Self::new(request, raw.id, raw.model, raw.answers, raw.usage).map_err(de::Error::custom)
    }

    /// Answers are immutable after their request-dependent checks.
    pub fn answers(&self) -> &BTreeMap<String, DecisionAnswer> {
        &self.answers
    }
}

/// An answer and separate, optional provider confidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionAnswer {
    /// The estimate, checked against its question when building a response.
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
        #[serde(default, deserialize_with = "optional_unique_map")]
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

fn check_answer(kind: &DecisionKind, answer: &DecisionValue) -> Result<(), DecisionError> {
    match (kind, answer) {
        (DecisionKind::Boolean { .. }, DecisionValue::Boolean(_)) => Ok(()),
        (
            DecisionKind::Choice { options },
            DecisionValue::Choice {
                selected,
                probabilities,
            },
        ) => {
            if !options.iter().any(|option| option.id == *selected) {
                return Err(DecisionError(
                    "selected choice is not a declared option".into(),
                ));
            }
            if let Some(probabilities) = probabilities {
                if probabilities.len() != options.len()
                    || options
                        .iter()
                        .any(|option| !probabilities.contains_key(&option.id))
                {
                    return Err(DecisionError(
                        "distribution must cover every option exactly".into(),
                    ));
                }
                check_sum(probabilities.values())?;
            }
            Ok(())
        }
        (
            DecisionKind::Score { levels },
            DecisionValue::Score {
                value,
                probabilities,
            },
        ) => {
            if value.get() > (levels.len() - 1) as f64 {
                return Err(DecisionError("score exceeds the request's rubric".into()));
            }
            if let Some(probabilities) = probabilities {
                if probabilities.len() != levels.len() {
                    return Err(DecisionError(
                        "distribution must cover every level exactly".into(),
                    ));
                }
                check_sum(probabilities.iter())?;
            }
            Ok(())
        }
        _ => Err(DecisionError(
            "answer kind does not match question kind".into(),
        )),
    }
}

fn check_sum<'a>(
    probabilities: impl Iterator<Item = &'a Probability>,
) -> Result<(), DecisionError> {
    let sum: f64 = probabilities.map(|probability| probability.get()).sum();
    if (sum - 1.0).abs() > DISTRIBUTION_SUM_TOLERANCE {
        return Err(DecisionError(format!(
            "distribution must sum to one within {DISTRIBUTION_SUM_TOLERANCE}"
        )));
    }
    Ok(())
}

// Reject duplicate JSON keys before a map could silently discard an answer or option.
fn unique_map<'de, D, T>(deserializer: D) -> Result<BTreeMap<String, T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Visitor<T>(std::marker::PhantomData<T>);

    impl<'de, T: Deserialize<'de>> de::Visitor<'de> for Visitor<T> {
        type Value = BTreeMap<String, T>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a map with unique IDs")
        }

        fn visit_map<A: de::MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
            let mut map = BTreeMap::new();
            while let Some((key, value)) = access.next_entry::<String, T>()? {
                match map.entry(key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(value);
                    }
                    std::collections::btree_map::Entry::Occupied(entry) => {
                        return Err(de::Error::custom(format!("duplicate ID {:?}", entry.key())));
                    }
                }
            }
            Ok(map)
        }
    }
    deserializer.deserialize_map(Visitor(std::marker::PhantomData))
}

fn optional_unique_map<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<BTreeMap<String, Probability>>, D::Error> {
    #[derive(Deserialize)]
    struct UniqueMap(#[serde(deserialize_with = "unique_map")] BTreeMap<String, Probability>);

    Ok(Option::<UniqueMap>::deserialize(deserializer)?.map(|map| map.0))
}
