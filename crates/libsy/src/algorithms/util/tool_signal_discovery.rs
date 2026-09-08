// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Judge-derived analogue of [`super::tool_signals`]: an LLM discovers its own
//! severity/category buckets from raw tool activity instead of regex tables, and
//! writes the same [`ToolSignals`] shape into [`State::tool_signals`]. A
//! `stage_router` can therefore be configured to source its signal from either
//! extractor — [`crate::algorithms::stage::StageRouterConfig`] picks one, and
//! [`crate::algorithms::stage::StageClassifier`] is unaffected by the choice.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, InstructionBlock, LlmRequest, Message, ModelId, OutputParams,
    Request, Role, completion_text,
};

use crate::core::algorithm::Driver;
use crate::core::processor::{Event, Processor};
use crate::core::state::State;
use crate::{LibsyError, Result};

use super::tool_signals::ToolSignals;

const SYSTEM_PROMPT: &str = include_str!("../../prompts/tool_signal_discovery/prompt.md");
const MAX_OUTPUT_TOKENS: u64 = 1024;
/// Same marker `tool_signals.rs` latches on; duplicated rather than made
/// cross-module-visible since it is the only piece of that module this needs.
const COMPACTION_MARKER: &str = "session is being continued";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum DiscoveredCategory {
    Write,
    Edit,
    Read,
    Plan,
    Other,
}

impl DiscoveredCategory {
    fn as_str(self) -> &'static str {
        match self {
            Self::Write => "write",
            Self::Edit => "edit",
            Self::Read => "read",
            Self::Plan => "plan",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Deserialize)]
struct DiscoveredToolCall {
    category: DiscoveredCategory,
    pattern_label: String,
}

#[derive(Debug, Deserialize)]
struct DiscoveredToolResult {
    severity: f32,
    pattern_label: Option<String>,
    tests_passed: bool,
}

#[derive(Debug, Default, Deserialize)]
struct DiscoveryVerdict {
    #[serde(default)]
    tool_calls: Vec<DiscoveredToolCall>,
    #[serde(default)]
    tool_results: Vec<DiscoveredToolResult>,
}

/// Discovered vocabulary shared across every session this route serves — the
/// calibration a judge-driven signal source is meant to accumulate over many
/// tasks, not just one. Owned by [`LlmToolSignalProcessor`] itself (a singleton
/// for the route's lifetime), never by per-session [`State`].
#[derive(Clone, Debug, Default)]
pub struct SharedVocabulary {
    /// Discovered `pattern_label` -> cumulative occurrences, across every
    /// session, for both calls and results.
    pub pattern_counts: HashMap<String, u32>,
    /// Tool-call `pattern_label` -> the category it was first (and should stay)
    /// associated with, so vocabulary feedback can catch cross-category reuse.
    label_categories: HashMap<String, DiscoveredCategory>,
}

impl SharedVocabulary {
    /// Vocabulary fed back into the prompt: call labels annotated with the
    /// category they were first seen under, so the judge can catch itself
    /// reusing a label across a category boundary; result labels stay bare.
    fn vocabulary(&self) -> Vec<String> {
        self.pattern_counts
            .keys()
            .map(|label| match self.label_categories.get(label) {
                Some(category) => format!("{label} ({})", category.as_str()),
                None => label.clone(),
            })
            .collect()
    }

    fn record_call_label(&mut self, label: &str, category: DiscoveredCategory) {
        self.label_categories
            .entry(label.to_string())
            .or_insert(category);
        *self.pattern_counts.entry(label.to_string()).or_default() += 1;
    }

    fn record_result_label(&mut self, label: &str) {
        *self.pattern_counts.entry(label.to_string()).or_default() += 1;
    }
}

/// Judge-discovered analogue of [`ToolSignals`], accumulated incrementally across
/// this session's own turns — reset per session, unlike [`SharedVocabulary`].
#[derive(Clone, Debug, Default)]
pub struct LlmToolSignals {
    last_processed_message_index: usize,
    recent_categories: VecDeque<DiscoveredCategory>,
    recent_severities: VecDeque<f32>,
    recent_tests_passed: VecDeque<bool>,
    no_error_streak: u32,
    pure_bash_streak: u32,
    edit_count: u32,
    write_count: u32,
    read_count: u32,
    todowrite_count: u32,
}

impl LlmToolSignals {
    fn record_call(&mut self, category: DiscoveredCategory, window: usize) {
        match category {
            DiscoveredCategory::Edit => self.edit_count += 1,
            DiscoveredCategory::Write => self.write_count += 1,
            DiscoveredCategory::Read => self.read_count += 1,
            DiscoveredCategory::Plan => self.todowrite_count += 1,
            DiscoveredCategory::Other => {}
        }
        self.pure_bash_streak = if category == DiscoveredCategory::Other {
            self.pure_bash_streak + 1
        } else {
            0
        };
        push_bounded(&mut self.recent_categories, category, window);
    }

    fn record_result(&mut self, severity: f32, tests_passed: bool, window: usize) {
        self.no_error_streak = if severity > 0.0 { 0 } else { self.no_error_streak + 1 };
        push_bounded(&mut self.recent_severities, severity, window);
        push_bounded(&mut self.recent_tests_passed, tests_passed, window);
    }

    fn recent_count(&self, category: DiscoveredCategory) -> u32 {
        self.recent_categories.iter().filter(|c| **c == category).count() as u32
    }

    /// Synthesizes the same [`ToolSignals`] shape `tool_signals.rs` produces, from
    /// this accumulator's running windows plus this request's structural counts.
    fn to_tool_signals(
        &self,
        turn_depth: u32,
        assistant_turn_count: u32,
        tool_result_count: u32,
        compacted: bool,
    ) -> ToolSignals {
        ToolSignals {
            severity: self.recent_severities.iter().cloned().fold(0.0, f32::max),
            no_error_streak: self.no_error_streak,
            edit_count: self.edit_count,
            write_count: self.write_count,
            read_count: self.read_count,
            todowrite_count: self.todowrite_count,
            recent_edit_count: self.recent_count(DiscoveredCategory::Edit),
            recent_write_count: self.recent_count(DiscoveredCategory::Write),
            recent_read_count: self.recent_count(DiscoveredCategory::Read),
            recent_todowrite_count: self.recent_count(DiscoveredCategory::Plan),
            pure_bash_streak: self.pure_bash_streak,
            tests_passed: self.recent_tests_passed.iter().any(|passed| *passed),
            tool_result_count,
            assistant_turn_count,
            turn_depth,
            compacted,
        }
    }
}

fn push_bounded<T>(buffer: &mut VecDeque<T>, value: T, window: usize) {
    buffer.push_back(value);
    while buffer.len() > window.max(1) {
        buffer.pop_front();
    }
}

/// Runs the discovery judge on each request's new tool activity, accumulates its
/// output into [`State::llm_tool_signals`], and writes the synthesized
/// [`ToolSignals`] into [`State::tool_signals`] — the same field
/// `super::tool_signals::ToolSignalProcessor` writes, so the picker cannot tell
/// which extractor produced it.
pub struct LlmToolSignalProcessor {
    pub judge_target: ModelId,
    /// Window the accumulator's `recent_*` counts and windowed severity are
    /// computed over. Mirrors `ToolSignalProcessor::recent_window`.
    pub recent_window: usize,
    /// Discovered vocabulary, shared and cumulative across every session this
    /// route serves — this processor is one singleton for the route's whole
    /// lifetime, so this is where calibration actually accumulates.
    pub vocabulary: Arc<Mutex<SharedVocabulary>>,
}

impl LlmToolSignalProcessor {
    /// A fresh processor with its own empty, unshared vocabulary.
    pub fn new(judge_target: ModelId, recent_window: usize) -> Self {
        Self {
            judge_target,
            recent_window,
            vocabulary: Arc::new(Mutex::new(SharedVocabulary::default())),
        }
    }
}

#[async_trait]
impl Processor<State> for LlmToolSignalProcessor {
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<()> {
        let Event::Request { request, driver } = event else {
            return Ok(());
        };
        let messages = &request.llm_request.messages;
        let (turn_depth, assistant_turn_count, tool_result_count, compacted) =
            structural_signals(messages);

        let verdict = match driver {
            Some(driver) => {
                let signals = state.llm_tool_signals.get_or_insert_with(LlmToolSignals::default);
                let start = signals.last_processed_message_index;
                signals.last_processed_message_index = messages.len();
                if start < messages.len() {
                    let (calls, results) = new_tool_activity(&messages[start..]);
                    if calls.is_empty() && results.is_empty() {
                        None
                    } else {
                        let vocab = self.vocabulary.lock().vocabulary();
                        consult_judge(driver, &self.judge_target, &calls, &results, &vocab).await
                    }
                } else {
                    None
                }
            }
            None => None,
        };

        let signals = state.llm_tool_signals.get_or_insert_with(LlmToolSignals::default);
        if let Some(verdict) = verdict {
            let mut vocabulary = self.vocabulary.lock();
            for call in &verdict.tool_calls {
                tracing::info!(target: "libsy", category = ?call.category, label = call.pattern_label, "discovered tool-call pattern");
                signals.record_call(call.category, self.recent_window);
                vocabulary.record_call_label(&call.pattern_label, call.category);
            }
            for result in &verdict.tool_results {
                tracing::info!(
                    target: "libsy",
                    severity = result.severity,
                    tests_passed = result.tests_passed,
                    label = result.pattern_label.as_deref(),
                    "discovered tool-result pattern"
                );
                signals.record_result(result.severity, result.tests_passed, self.recent_window);
                if let Some(label) = &result.pattern_label {
                    vocabulary.record_result_label(label);
                }
            }
        }
        let tool_signal = signals.to_tool_signals(
            turn_depth,
            assistant_turn_count,
            tool_result_count,
            compacted,
        );
        tracing::info!(target: "libsy", ?tool_signal, "synthesized llm tool signals");
        state.tool_signals = Some(tool_signal);
        Ok(())
    }
}

/// Calls the judge and parses its verdict. Fails open (`None`) and logs a warning
/// on any transport, stream, or parse failure — a judge outage should not stall
/// routing, and the caller keeps the accumulator's last-known-good signal.
async fn consult_judge(
    driver: &Driver,
    judge_target: &ModelId,
    calls: &[(String, Value)],
    results: &[String],
    vocab: &[String],
) -> Option<DiscoveryVerdict> {
    let vocab_refs: Vec<&str> = vocab.iter().map(String::as_str).collect();
    let (judge_request, activity) = build_request(judge_target, calls, results, &vocab_refs);
    tracing::info!(
        target: "libsy",
        judge_target = %judge_target,
        new_tool_calls = calls.len(),
        new_tool_results = results.len(),
        prior_vocabulary_size = vocab.len(),
        prior_vocabulary = %vocab.join(", "),
        request = %activity,
        "signal-discovery judge request"
    );

    let response = match driver
        .call_model(judge_request, vec![judge_target.clone()])
        .await
    {
        Ok(response) => response,
        Err(error) => {
            tracing::warn!(target: "libsy", %error, "signal-discovery judge call failed");
            return None;
        }
    };
    let aggregate = match response.llm_response.into_agg().await {
        Ok(aggregate) => aggregate,
        Err(error) => {
            tracing::warn!(target: "libsy", %error, "signal-discovery judge stream failed");
            return None;
        }
    };
    tracing::info!(target: "libsy", response = %completion_text(&aggregate), "signal-discovery judge response");
    match parse_verdict(&aggregate) {
        Ok(verdict) => Some(verdict),
        Err(error) => {
            tracing::warn!(target: "libsy", %error, "signal-discovery judge verdict did not parse");
            None
        }
    }
}

/// Raw tool calls and tool-result text found in `messages`, in document order.
fn new_tool_activity(messages: &[Message]) -> (Vec<(String, Value)>, Vec<String>) {
    let mut calls = Vec::new();
    let mut results = Vec::new();
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::ToolCall(call) => calls.push((call.name.clone(), call.arguments.clone())),
                ContentBlock::ToolResult(result) => {
                    let text = result
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !text.is_empty() {
                        results.push(text);
                    }
                }
                _ => {}
            }
        }
    }
    (calls, results)
}

/// Counts that need no judge call: message-count depth, assistant-turn count,
/// total tool-result blocks, and compaction — computed fresh every request, the
/// same way `tool_signals.rs` computes its equivalents.
fn structural_signals(messages: &[Message]) -> (u32, u32, u32, bool) {
    let mut assistant_turn_count = 0u32;
    let mut tool_result_count = 0u32;
    let mut compacted = false;
    for message in messages {
        if message.role == Role::Assistant {
            assistant_turn_count += 1;
        }
        for block in &message.content {
            match block {
                ContentBlock::ToolResult(_) => tool_result_count += 1,
                ContentBlock::Text { text } => {
                    compacted |= text.to_lowercase().contains(COMPACTION_MARKER);
                }
                _ => {}
            }
        }
    }
    (
        messages.len() as u32,
        assistant_turn_count,
        tool_result_count,
        compacted,
    )
}

fn build_request(
    target: &ModelId,
    calls: &[(String, Value)],
    results: &[String],
    vocab: &[&str],
) -> (Request, String) {
    let activity = json!({
        "tool_calls": calls.iter().map(|(name, args)| json!({"name": name, "arguments": args})).collect::<Vec<_>>(),
        "tool_results": results,
    })
    .to_string();
    let prior_vocabulary = if vocab.is_empty() {
        "(none yet)".to_string()
    } else {
        vocab.join(", ")
    };
    let prompt = SYSTEM_PROMPT.replace("{prior_vocabulary}", &prior_vocabulary);

    let request = Request {
        llm_request: LlmRequest {
            model: Some(target.to_string()),
            instructions: vec![InstructionBlock {
                role: Role::System,
                content: Message::text(Role::System, prompt).content,
            }],
            messages: vec![Message::text(Role::User, activity.clone())],
            output: OutputParams {
                max_output_tokens: Some(MAX_OUTPUT_TOKENS),
                response_format: Some(json!({"type": "json_object"})),
            },
            ..LlmRequest::default()
        },
        raw_request: None,
        metadata: None,
    };
    (request, activity)
}

fn parse_verdict(response: &AggLlmResponse) -> Result<DiscoveryVerdict> {
    let text = completion_text(response);
    let trimmed = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    serde_json::from_str(trimmed).map_err(|err| LibsyError::AlgorithmError {
        message: format!("discovery verdict did not parse: {err}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_protocol::{ContentBlock, Role, ToolCall, ToolResult};

    fn tc(name: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: String::new(),
                name: name.to_string(),
                arguments: json!({}),
            })],
        }
    }

    fn tr(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: String::new(),
                content: vec![ContentBlock::Text {
                    text: text.to_string(),
                }],
                is_error: None,
            })],
        }
    }

    #[test]
    fn extracts_calls_and_results_in_order() {
        let messages = vec![tc("write_file"), tr("wrote ok"), tc("terminal")];
        let (calls, results) = new_tool_activity(&messages);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "write_file");
        assert_eq!(results, vec!["wrote ok".to_string()]);
    }

    #[test]
    fn parses_fenced_verdict() {
        let raw = "```json\n{\"tool_calls\":[{\"category\":\"write\",\"pattern_label\":\"write_file_tool\"}],\"tool_results\":[]}\n```";
        let trimmed = raw
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        let verdict: DiscoveryVerdict = serde_json::from_str(trimmed).unwrap();
        assert_eq!(verdict.tool_calls.len(), 1);
        assert_eq!(verdict.tool_calls[0].pattern_label, "write_file_tool");
    }

    #[test]
    fn accumulates_pattern_counts_in_shared_vocabulary() {
        let mut vocabulary = SharedVocabulary::default();
        let verdict = DiscoveryVerdict {
            tool_calls: vec![DiscoveredToolCall {
                category: DiscoveredCategory::Write,
                pattern_label: "write_file_tool".to_string(),
            }],
            tool_results: vec![DiscoveredToolResult {
                severity: 0.7,
                pattern_label: Some("missing_python_module".to_string()),
                tests_passed: false,
            }],
        };
        for call in &verdict.tool_calls {
            vocabulary.record_call_label(&call.pattern_label, call.category);
        }
        for result in &verdict.tool_results {
            if let Some(label) = &result.pattern_label {
                vocabulary.record_result_label(label);
            }
        }
        assert_eq!(vocabulary.pattern_counts["write_file_tool"], 1);
        assert_eq!(vocabulary.pattern_counts["missing_python_module"], 1);
    }

    #[test]
    fn synthesizes_tool_signals_shape_from_accumulator() {
        let mut signals = LlmToolSignals::default();
        signals.record_call(DiscoveredCategory::Write, 3);
        signals.record_result(0.7, false, 3);
        let synthesized = signals.to_tool_signals(4, 2, 1, false);
        assert_eq!(synthesized.severity, 0.7);
        assert_eq!(synthesized.write_count, 1);
        assert_eq!(synthesized.recent_write_count, 1);
        assert_eq!(synthesized.no_error_streak, 0);
        assert_eq!(synthesized.turn_depth, 4);
        assert_eq!(synthesized.assistant_turn_count, 2);
        assert_eq!(synthesized.tool_result_count, 1);
    }

    #[test]
    fn recent_window_bounds_accumulator_history() {
        let mut signals = LlmToolSignals::default();
        for _ in 0..5 {
            signals.record_call(DiscoveredCategory::Edit, 3);
        }
        assert_eq!(signals.edit_count, 5);
        assert_eq!(signals.recent_count(DiscoveredCategory::Edit), 3);
    }

    #[test]
    fn vocabulary_annotates_call_labels_with_their_first_category() {
        let mut vocabulary = SharedVocabulary::default();
        vocabulary.record_call_label("heredoc_write", DiscoveredCategory::Write);
        vocabulary.record_result_label("import_error");
        let vocab = vocabulary.vocabulary();
        assert!(vocab.contains(&"heredoc_write (write)".to_string()));
        assert!(vocab.contains(&"import_error".to_string()));
    }

    #[test]
    fn vocabulary_persists_across_sessions_via_shared_processor() {
        let processor = LlmToolSignalProcessor::new(ModelId::from("judge"), 3);
        processor
            .vocabulary
            .lock()
            .record_call_label("heredoc_write", DiscoveredCategory::Write);
        // A second, independent session's LlmToolSignals starts empty...
        let session_two_signals = LlmToolSignals::default();
        assert_eq!(session_two_signals.write_count, 0);
        // ...but the processor's shared vocabulary — not session state — still has it.
        assert_eq!(
            processor.vocabulary.lock().pattern_counts["heredoc_write"],
            1
        );
    }
}
