// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Answer a known list-selection prompt using the choices present in that request.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::json;
use switchyard_protocol::{
    BooleanEstimate, Category, ContentBlock, DecisionKind, DecisionQuestion, DecisionRequest,
    DecisionResponse, DecisionValue, LlmRequest, LlmResponse, Request, Response, Role, StopReason,
    WireFormat, text_response,
};

use crate::{Algorithm, Driver, LibsyError, OutcomeMetadata, Result, RoutingOutcome};

/// Wire contract for a prompt that asks for a newline-separated list of names.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionRule {
    /// Enable substitution. When false, the LLM serves the original request.
    pub enabled: bool,
    /// Exact trusted system instruction identifying the call site.
    pub system_instruction: String,
    /// Prefix of the first user message, followed by a YAML map of names to descriptions.
    /// One level of groups is supported. Group and option names are joined with a dot.
    pub catalog_prefix: String,
    /// Text immediately before the task in the final user message.
    pub task_prefix: String,
    /// Text immediately after the task in the final user message.
    pub task_suffix: String,
    /// Instruction for assessing one option against the task.
    pub question: String,
    /// Omit these lowercase words from both the query and documents during shortlisting.
    #[serde(default)]
    pub stop_words: BTreeSet<String>,
    /// Prefix for numbered question IDs. Defaults to `q`.
    #[serde(default = "default_question_prefix")]
    pub question_prefix: String,
    /// Maximum number of options to score in one decision call.
    pub shortlist: usize,
    /// Maximum number of names returned to the caller.
    pub top_k: usize,
}

fn default_question_prefix() -> String {
    "q".into()
}

/// Replaces matching LLM requests with one batched decision call.
pub struct DecisionSelect {
    rule: SelectionRule,
}

struct Candidate {
    name: String,
    description: String,
}

impl DecisionSelect {
    /// Check the prompt contract and the size of the batch before serving traffic.
    pub fn new(rule: SelectionRule) -> Result<Self> {
        if [
            &rule.system_instruction,
            &rule.catalog_prefix,
            &rule.task_prefix,
            &rule.task_suffix,
            &rule.question,
        ]
        .iter()
        .any(|value| value.trim().is_empty())
            || !(1..=512).contains(&rule.shortlist)
            || !(1..=rule.shortlist).contains(&rule.top_k)
        {
            return Err(LibsyError::AlgorithmError {
                message:
                    "selection requires nonempty prompt fields and 1 <= top_k <= shortlist <= 512"
                        .into(),
            });
        }
        Ok(Self { rule })
    }

    fn extract<'a>(&self, request: &'a LlmRequest) -> Option<(&'a str, Vec<Candidate>)> {
        // A chat target must own Responses state so it can serve later turns.
        let fields = &request.extensions.fields;
        if fields.contains_key("previous_response_id")
            || fields.contains_key("conversation")
            || (request
                .preservation
                .requests
                .contains_key(&WireFormat::OpenAiResponses.into())
                && fields.get("store").and_then(serde_json::Value::as_bool) != Some(false))
        {
            return None;
        }
        let [system] = request.instructions.as_slice() else {
            return None;
        };
        if system.role != Role::System
            || plain_text(&system.content)? != self.rule.system_instruction
            || !request.tools.is_empty()
            || request.tool_choice.is_some()
            || request.output.response_format.is_some()
            || request.output.max_output_tokens.is_some()
            || fields
                .get("n")
                .and_then(|n| n.as_u64())
                .is_some_and(|n| n != 1)
            || fields.get("logprobs").and_then(|v| v.as_bool()) == Some(true)
            || ["stop", "stop_sequences", "modalities", "audio"]
                .iter()
                .any(|key| fields.get(*key).is_some_and(|v| !v.is_null()))
            || request.is_compaction
        {
            return None;
        }
        let [first, .., last] = request.messages.as_slice() else {
            return None;
        };
        if first.role != Role::User || last.role != Role::User {
            return None;
        }
        let task = plain_text(&last.content)?
            .strip_prefix(&self.rule.task_prefix)?
            .strip_suffix(&self.rule.task_suffix)?;
        let catalog = plain_text(&first.content)?.strip_prefix(&self.rule.catalog_prefix)?;
        if task.trim().is_empty() || catalog.len() > 256 * 1024 {
            return None;
        }
        let groups: yaml_serde::Mapping = yaml_serde::from_str(catalog).ok()?;
        let mut options = Vec::new();
        let mut names = BTreeSet::new();
        let mut push = |name: String, description: &str| -> Option<()> {
            if name.trim().is_empty() || name.contains(['\n', '\r']) || !names.insert(name.clone())
            {
                return None;
            }
            options.push(Candidate {
                name,
                description: description.into(),
            });
            Some(())
        };
        for (key, value) in &groups {
            let key = key.as_str()?;
            if let Some(description) = value.as_str() {
                push(key.to_owned(), description)?;
            } else {
                for (name, description) in value.as_mapping()? {
                    push(format!("{key}.{}", name.as_str()?), description.as_str()?)?;
                }
            }
        }
        (!options.is_empty()).then_some((task, options))
    }
}

fn plain_text(content: &[ContentBlock]) -> Option<&str> {
    match content {
        [ContentBlock::Text { text }] => Some(text),
        _ => None,
    }
}

fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
}

// Rank by BM25. Stable ties preserve the order in the caller's catalog.
fn shortlist(
    task: &str,
    options: &mut Vec<Candidate>,
    limit: usize,
    stop_words: &BTreeSet<String>,
) {
    if options.len() <= limit {
        return;
    }
    // Sum in query order so rounding does not reorder nearly tied documents.
    let mut seen = BTreeSet::new();
    let query: Vec<_> = words(task)
        .filter(|w| !stop_words.contains(w) && seen.insert(w.clone()))
        .collect();
    let mut document_counts = BTreeMap::new();
    let documents: Vec<_> = options
        .drain(..)
        .map(|option| {
            let mut counts = BTreeMap::new();
            let mut length = 0;
            for word in words(&option.name)
                .chain(words(&option.description))
                .filter(|w| !stop_words.contains(w))
            {
                length += 1;
                if seen.contains(&word) {
                    *counts.entry(word).or_insert(0u32) += 1;
                }
            }
            for word in counts.keys() {
                *document_counts.entry(word.clone()).or_insert(0u32) += 1;
            }
            (option, length, counts)
        })
        .collect();
    let n = documents.len() as f64;
    let average = documents.iter().map(|(_, length, _)| length).sum::<usize>() as f64 / n;
    let mut ranked: Vec<_> = documents
        .into_iter()
        .map(|(option, length, counts)| {
            let score: f64 = query
                .iter()
                .filter_map(|word| {
                    let count = f64::from(*counts.get(word)?);
                    let norm = 1.5 * (0.25 + 0.75 * length as f64 / average);
                    let df = f64::from(document_counts[word]);
                    let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                    Some(idf * count * 2.5 / (count + norm))
                })
                .sum();
            (option, score)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    options.extend(ranked.into_iter().take(limit).map(|(option, _)| option));
}

fn select_names(
    response: &DecisionResponse,
    prefix: &str,
    options: &[Candidate],
    limit: usize,
) -> Option<String> {
    if options.len() != response.answers.len() {
        return None;
    }
    let mut scores = Vec::with_capacity(options.len());
    for (i, option) in options.iter().enumerate() {
        let answer = response.answers.get(&format!("{prefix}{i}"))?;
        let DecisionValue::Boolean(BooleanEstimate::ProbabilityTrue(p)) = answer.value else {
            return None;
        };
        if !(0.0..=1.0).contains(&p.0) {
            return None;
        }
        scores.push((option.name.as_str(), p.0));
    }
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    Some(
        scores
            .into_iter()
            .take(limit)
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

#[async_trait::async_trait]
impl Algorithm for DecisionSelect {
    fn name(&self) -> &str {
        "decision_select"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let target = driver
            .models_for(&Category::Any)
            .first()
            .ok_or(LibsyError::NoTargets)?;
        let fallback = |request, reason| {
            let mut outcome = RoutingOutcome::route_to(target.clone(), Vec::new(), request);
            outcome.metadata = Some(OutcomeMetadata::new(
                self.name().into(),
                Some(json!({"reason":reason})),
            ));
            Ok(outcome)
        };
        if !self.rule.enabled {
            return fallback(request, "disabled");
        }
        let Some((task, mut options)) = self.extract(&request.llm_request) else {
            return fallback(request, "unmatched");
        };
        let catalog_size = options.len();
        shortlist(
            task,
            &mut options,
            self.rule.shortlist,
            &self.rule.stop_words,
        );
        let prefix = &self.rule.question_prefix;
        let questions = options.iter().enumerate().map(|(i, option)| (
            format!("{prefix}{i}"),
            DecisionQuestion {
                instructions: json!(format!("{}\nConsider one option only: {} ({})\nDoes this option belong to the answer?", self.rule.question, option.name, option.description)),
                kind: DecisionKind::Boolean { true_description: None, false_description: None },
            },
        )).collect();
        let model = driver
            .models_for(&Category::Judge)
            .first()
            .ok_or(LibsyError::NoTargets)?;
        let decision = DecisionRequest {
            model: None,
            input: json!(task),
            questions,
        };
        let response = match driver.call_decision(decision, model.clone()).await {
            Ok(response) => response,
            Err(LibsyError::ClientCall { .. } | LibsyError::InvalidDecision { .. }) => {
                return fallback(request, "decision_failed");
            }
            Err(error) => return Err(error),
        };
        let Some(names) = select_names(&response, prefix, &options, self.rule.top_k) else {
            return fallback(request, "invalid_selection");
        };
        let mut answer = text_response(Some(driver.provider_model(model).to_string()), names);
        answer.id = response.id;
        answer.usage = response.usage;
        answer.outputs[0].stop_reason = Some(StopReason::EndTurn);
        let llm_response = if request.llm_request.stream {
            LlmResponse::Stream(answer.into_stream())
        } else {
            LlmResponse::Agg(answer)
        };
        let mut response = Response {
            llm_response,
            metadata: request.metadata.clone(),
            upstream_headers: Default::default(),
        };
        response.set_served_model(model);
        let mut outcome = RoutingOutcome::answered(model.clone(), request, response);
        outcome.metadata = Some(OutcomeMetadata::new(
            self.name().into(),
            Some(json!({
                "reason": "decision_answer", "catalog_size": catalog_size, "scored": options.len(),
                "returned": self.rule.top_k.min(options.len())
            })),
        ));
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_words_apply_to_query_and_document_lengths() {
        let mut options = [
            ("second", "the the the song"),
            ("first", "song"),
            ("other", "the the the"),
        ]
        .into_iter()
        .map(|(name, description)| Candidate {
            name: name.into(),
            description: description.into(),
        })
        .collect();
        shortlist("the song", &mut options, 2, &BTreeSet::from(["the".into()]));
        assert_eq!(options[0].name, "second");
        assert_eq!(options[1].name, "first");
    }

    #[test]
    fn shortlist_scores_task_terms_before_applying_the_limit() {
        let mut options = [
            ("music.play", "play a track"),
            ("wallet.approve", "approve a payment request"),
            ("wallet.reject", "reject a payment request"),
        ]
        .into_iter()
        .map(|(name, description)| Candidate {
            name: name.into(),
            description: description.into(),
        })
        .collect();
        shortlist(
            "approve a payment request",
            &mut options,
            2,
            &BTreeSet::new(),
        );
        assert_eq!(
            options
                .iter()
                .map(|option| option.name.as_str())
                .collect::<Vec<_>>(),
            ["wallet.approve", "wallet.reject"]
        );
        shortlist("unrelated", &mut options, 1, &BTreeSet::new());
        assert_eq!(options[0].name, "wallet.approve");
    }
}
