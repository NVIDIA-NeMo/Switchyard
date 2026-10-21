// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use switchyard_protocol::{
    ChoiceOption, DecisionKind, DecisionQuestion, DecisionRequest, DecisionResponse, DecisionValue,
    ModelId,
};

/// Stable question ID used in task-budget requests and answers.
pub const QUESTION: &str = "minimum_likely_budget";

/// Original task-budget band upper bounds, before five equal subdivisions.
pub const BUDGET_UPPER_BOUNDS: [u64; 16] = [
    10_000, 50_000, 100_000, 149_000, 189_000, 212_000, 232_000, 250_000, 269_000, 280_000,
    295_000, 345_000, 415_000, 602_000, 1_000_000, 1_500_000,
];

const INSTRUCTION: &str = "Evaluate whether the agent will complete the task successfully within the token budgets specified in the questions. Token counts are uncached input plus output; reasoning tokens are included in output. Use the task_history and token_usage fields as evidence. The first task prompt is included in task_history when available.\nHow far along is the task? Distinguish verified progress from remaining work. Plans are not completed work; allow for unfinished implementation, tests, debugging, and a plausible revision.";
const TASK: &str = "In which total uncached-input-plus-output token budget band is the smallest budget likely to allow successful completion? Choose impossible if none is likely to be sufficient. Use a realistic completion budget rather than an optimistic minimum. Allow for remaining implementation, tests, debugging, and plausible revisions, especially when the prefix is short.";

fn bands() -> impl Iterator<Item = (String, u64, u64)> {
    BUDGET_UPPER_BOUNDS
        .into_iter()
        .scan(0, |lower, upper| {
            let start = *lower;
            *lower = upper;
            Some((start, upper))
        })
        .flat_map(|(lower, upper)| {
            (0..5).map(move |i| {
                (
                    lower + (upper - lower) * i / 5,
                    lower + (upper - lower) * (i + 1) / 5,
                )
            })
        })
        .enumerate()
        .map(|(i, (lower, upper))| (format!("b{:03}", i + 1), lower, upper))
}

/// Observed usage of one completed model invocation, in chronological order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseUsage {
    /// Input tokens excluding cache reads. Include cache writes when uncached.
    pub uncached_input_tokens: u64,
    /// Total output, including reasoning and tool-call generation.
    pub output_tokens: u64,
    /// Reasoning tokens, already included in `output_tokens`.
    pub reasoning_tokens: u64,
}

impl ResponseUsage {
    /// Uncached input plus output, counting reasoning once.
    pub fn tokens_used(&self) -> u64 {
        self.uncached_input_tokens + self.output_tokens
    }
}

/// Builds the experiment's buffered, fine-band task-completion question.
///
/// `history` contains caller-prepared JSON steps from the observed task prefix,
/// including the initial task prompt. The caller controls history size;
/// `prior_usage` covers all completed responses, even if older history is omitted.
/// The forecast is a total task budget, including tokens already used.
pub fn request(
    judge_model: ModelId,
    history: &[Value],
    prior_usage: &[ResponseUsage],
) -> DecisionRequest {
    let mut options: Vec<_> = bands()
        .map(|(id, lower, upper)| ChoiceOption {
            id,
            description: Some(json!(format!("({lower}, {upper}] tokens"))),
        })
        .collect();
    options.push(ChoiceOption {
        id: "impossible".into(),
        description: Some(json!("No listed budget, including the largest (1500000 total uncached-input-plus-output tokens), is likely to permit successful completion.")),
    });
    DecisionRequest {
        model: Some(judge_model),
        context: json!({
            "instructions": INSTRUCTION,
            "task_history": history,
            "token_usage": {
                "definition": "Uncached input plus output tokens; reasoning is a subset of output.",
                "tokens_used_so_far": prior_usage.iter().map(ResponseUsage::tokens_used).sum::<u64>(),
                "reasoning_output_tokens_so_far": prior_usage.iter().map(|u| u.reasoning_tokens).sum::<u64>(),
                "per_step": prior_usage.iter().enumerate().map(|(i,u)| json!({
                    "step": i + 1, "tokens_used": u.tokens_used(), "reasoning_output_tokens": u.reasoning_tokens,
                })).collect::<Vec<_>>(),
            },
            "prefix": { "start_response": 1, "end_response": prior_usage.len() },
        }),
        questions: [(
            QUESTION.into(),
            DecisionQuestion {
                instructions: json!({
                    "task": TASK,
                    "use_state_fields": ["instructions", "task_history", "token_usage"],
                }),
                kind: DecisionKind::Choice { options },
            },
        )]
        .into(),
    }
}

/// Forecast derived from the complete Jev probability distribution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    /// Probability-weighted band midpoint, conditional on a finite task budget.
    pub predicted_tokens: Option<f64>,
    /// Probability-weighted band upper bound, conditional on a finite task budget.
    pub expected_upper_tokens: Option<f64>,
    /// Selected total-budget interval; absent when `impossible` is selected.
    pub selected_interval: Option<[u64; 2]>,
    /// Jev's selected budget band ID or `impossible`.
    pub selected_choice: String,
    /// Probability that no listed task budget is likely to permit success.
    pub impossible_probability: f64,
    /// Normalized probability covered by the finite bands.
    pub finite_budget_probability: f64,
}

/// A missing or malformed Jev probability distribution.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    /// The expected Choice answer or its probabilities are absent.
    #[error("Expected a Choice answer with probabilities for {QUESTION}")]
    MissingAnswer,
    /// The distribution has unknown/missing bands, an invalid selection, or invalid values.
    #[error(
        "Expected all 81 task-budget options, a declared selection, and finite probabilities in [0, 1] summing to approximately one"
    )]
    InvalidDistribution,
}

/// Decodes the forecast using the experiment's midpoint and upper-bound estimates.
///
/// All 81 probabilities are required. Sums in `[0.9, 1.1]` are normalized to one,
/// matching the Python predictor. `impossible` has no finite token value.
pub fn prediction(response: &DecisionResponse) -> Result<Prediction, Error> {
    let Some(DecisionValue::Choice {
        selected,
        probabilities: Some(probabilities),
    }) = response.answers.get(QUESTION).map(|answer| &answer.value)
    else {
        return Err(Error::MissingAnswer);
    };
    if probabilities.len() != BUDGET_UPPER_BOUNDS.len() * 5 + 1
        || !probabilities.contains_key("impossible")
        || !probabilities.contains_key(selected)
        || probabilities
            .values()
            .any(|p| !p.0.is_finite() || !(0.0..=1.0).contains(&p.0))
    {
        return Err(Error::InvalidDistribution);
    }
    let total: f64 = probabilities.values().map(|p| p.0).sum();
    if !(0.9..=1.1).contains(&total) {
        return Err(Error::InvalidDistribution);
    }
    let mut finite = 0.0;
    let mut midpoint = 0.0;
    let mut upper_mean = 0.0;
    let mut selected_interval = None;
    for (key, lower, upper) in bands() {
        let probability = probabilities.get(&key).ok_or(Error::InvalidDistribution)?.0 / total;
        finite += probability;
        midpoint += probability * (lower + upper) as f64 / 2.0;
        upper_mean += probability * upper as f64;
        if selected == &key {
            selected_interval = Some([lower, upper]);
        }
    }
    Ok(Prediction {
        predicted_tokens: (finite > 0.0).then(|| midpoint / finite),
        expected_upper_tokens: (finite > 0.0).then(|| upper_mean / finite),
        selected_interval,
        selected_choice: selected.clone(),
        impossible_probability: probabilities["impossible"].0 / total,
        finite_budget_probability: finite,
    })
}

impl Prediction {
    /// Expected additional tokens after subtracting observed usage from the total.
    ///
    /// Uses the same uncached-input-plus-output definition as the request. A total
    /// below observed usage yields zero remaining tokens. `impossible` alone yields
    /// no finite estimate.
    pub fn remaining_tokens(&self, tokens_used_so_far: u64) -> Option<f64> {
        self.predicted_tokens
            .map(|total| (total - tokens_used_so_far as f64).max(0.0))
    }
}
