// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Staged (cascade) routing as an SDK component.
//!
//! [`StagedRouter`] recreates the reference cascade router's first, heuristic stage over
//! libsy's normalized conversation model. It plays both SDK roles over shared [`State`]:
//!
//! - As a [`Processor`] it extracts a [`ToolResultSignal`] from the request's conversation
//!   history — the tool calls and tool results already accumulated over the session — and
//!   stashes it in [`State`]. This is the port of the cascade router's request-side
//!   dimension collection; it reads the conversation, not correlation metadata.
//! - As a [`Classifier`] it performs the per-request heuristic scoring: apply hard
//!   overrides, then a weighted linear score over normalized dimensions. The signed
//!   `[-1, 1]` score is reranged to a `[0, 1]` capability confidence
//!   (`confidence = (score + 1) / 2`; `0.0` = strongly efficient, `1.0` = strongly
//!   capable). A confident score yields one [`Score`] for the winning tier's model; a
//!   score in the ambiguous band yields no scores, deferring to the next cascade stage.

use async_trait::async_trait;
use switchyard_protocol::{ContentBlock, Request, Role};

use super::core::{Classifier, Event, Processor, Score, State};

/// Boxed, thread-safe error type used across the SDK.
type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Default confidence threshold, in reranged `[0, 1]` confidence units: the heuristic
/// routes capable at `confidence >= threshold`, efficient at `confidence <= 1 - threshold`,
/// and defers to the next cascade stage inside the `[1 - threshold, threshold]` band.
/// `0.75` is the rerange of the reference's `0.5` magnitude threshold (`(0.5 + 1) / 2`).
pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.75;

/// Number of most-recent tool calls the `recent_*` counters window over.
pub const DEFAULT_RECENT_WINDOW: usize = 3;

/// Minimum turn depth for the "clean tests, run settled" efficient override.
const CLEAN_TESTS_MIN_TURN_DEPTH: u32 = 10;
/// Maximum writes for the "clean tests, run settled" efficient override.
const CLEAN_TESTS_MAX_WRITES: u32 = 1;
/// Half-saturation point for the pure-bash streak intensity.
const PURE_BASH_NORM: f64 = 8.0;

// --- Signal extraction (port of dimension_collector::tool_signals) --------------------

/// Soft/hard/critical severities on the `[0, 1]` scale.
const SOFT: f32 = 0.3;
const HARD: f32 = 0.7;
const CRITICAL: f32 = 1.0;

/// Error-pattern table: `(name, severity, substrings)`. A tool result trips a pattern
/// when it contains any of the substrings (case-insensitive); its severity is the max.
static ERROR_PATTERNS: &[(&str, f32, &[&str])] = &[
    (
        "oom",
        CRITICAL,
        &["out of memory", "memoryerror", "cannot allocate memory"],
    ),
    (
        "connection_refused",
        CRITICAL,
        &[
            "connection refused",
            "connectionrefusederror",
            "econnrefused",
        ],
    ),
    ("traceback", HARD, &["traceback (most recent call last)"]),
    (
        "import_error",
        HARD,
        &["modulenotfounderror:", "importerror:", "no module named "],
    ),
    (
        "cmd_not_found",
        HARD,
        &["command not found", "not found\n", "/usr/bin/env: "],
    ),
    ("assertion", HARD, &["assertionerror"]),
    ("value_error", HARD, &["valueerror:"]),
    ("syntax_error", HARD, &["syntaxerror:"]),
    (
        "timeout",
        HARD,
        &[
            "timed out",
            "timeouterror",
            "timeout expired",
            "deadline exceeded",
        ],
    ),
    (
        "no_such_file",
        HARD,
        &["filenotfounderror:", "no such file or directory"],
    ),
    (
        "exit_nonzero",
        SOFT,
        &[
            "exit code 1",
            "exit code 2",
            "exit status 1",
            "returned non-zero",
            "exited with code",
        ],
    ),
];

static EDIT_TOOL_NAMES: &[&str] = &[
    "edit",
    "multiedit",
    "notebookedit",
    "str_replace",
    "str_replace_based_edit_tool",
    "text_editor",
];
static WRITE_TOOL_NAMES: &[&str] = &["write", "create_file", "new_file"];
static READ_TOOL_NAMES: &[&str] = &["read", "view"];
static PLAN_TOOL_NAMES: &[&str] = &["todowrite", "todo_write", "todo", "update_plan"];
static BASH_TOOL_NAMES: &[&str] = &["bash", "shell_command", "shell", "local_shell_call"];

static BASH_WRITE_PATTERNS: &[&str] = &[
    "cat >",
    "cat >>",
    "echo >",
    "echo >>",
    "tee ",
    "printf >",
    "printf >>",
    "> /",
    ">> /",
    "<< 'eof'",
    "<<eof",
    "<<'eof'",
    "<< eof",
];
static BASH_EDIT_PATTERNS: &[&str] = &[
    "sed -i",
    "sed --in-place",
    "awk -i inplace",
    "awk 'inplace=1'",
    "patch ",
    "patch -p",
    "perl -i",
    "perl -p -i",
    "perl -pi",
];
static BASH_READ_PATTERNS: &[&str] = &[
    "cat /", "cat ./", "cat ../", "grep ", "ls ", "ls -", "find ", "head ", "tail ", "wc ",
    "diff ", "which ", "ps ", "df ", "du ", "stat ", "file ", "less ", "more ",
];

static TEST_PASS_PHRASES: &[&str] = &[
    " passed",
    "passed in",
    "tests passed",
    "all tests passed",
    "test ok",
    "test result: ok",
    "passed.\n",
    "tests pass",
    "\nok ",
    "✓ ",
];
static TEST_FAILURE_LITERAL: &[&str] = &["✗ ", "fatal:", "assertionerror", "error:"];
static NUMERIC_FAILURE_KEYWORDS: &[&str] = &["failed", "failure", "failures", "errors", "error"];

/// A normalized snapshot of coding-agent state extracted from a conversation, the input to
/// the heuristic score. Mirrors the reference `ToolResultSignal`.
#[derive(Clone, Debug, Default)]
pub struct ToolResultSignal {
    /// Severity of the latest tool result, in `[0, 1]`; `1.0` is critical.
    pub severity: f32,
    /// Names of the error patterns the latest tool result tripped.
    pub patterns: Vec<String>,
    /// Consecutive clean tool results, counted back from the most recent.
    pub no_error_streak: u32,
    /// Cumulative edit/write/read/plan tool-call counts.
    pub edit_count: u32,
    pub write_count: u32,
    pub read_count: u32,
    pub todowrite_count: u32,
    /// The same counts restricted to the most recent [`DEFAULT_RECENT_WINDOW`] calls.
    pub recent_edit_count: u32,
    pub recent_write_count: u32,
    pub recent_read_count: u32,
    pub recent_todowrite_count: u32,
    /// Trailing run of uncategorized (bash-ish) tool calls.
    pub pure_bash_streak: u32,
    /// Whether a recent tool result reads as a passing test run.
    pub tests_passed: bool,
    /// Number of messages in the conversation.
    pub turn_depth: u32,
    /// Character length of the last user message's text.
    pub prompt_char_count: u32,
}

/// A tool call observed in the conversation: its name and (for bash) its command string.
struct ObservedToolCall {
    name: String,
    command: Option<String>,
}

/// The write/edit/read/plan class a tool call falls into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolCategory {
    Write,
    Edit,
    Read,
    Plan,
    Other,
}

/// Classifies a tool result's text: the max severity over the patterns it trips, plus
/// their names in table order.
fn classify_text(text: &str) -> (f32, Vec<String>) {
    let lower = text.to_lowercase();
    let mut patterns = Vec::new();
    let mut severity: f32 = 0.0;
    for (name, sev, substrings) in ERROR_PATTERNS {
        if substrings.iter().any(|sub| lower.contains(sub)) {
            patterns.push((*name).to_string());
            severity = severity.max(*sev);
        }
    }
    (severity, patterns)
}

/// Classifies a tool call by name, consulting the command for bash-family tools.
fn classify_tool_call(name: &str, command: Option<&str>) -> ToolCategory {
    let lower = name.to_lowercase();
    if WRITE_TOOL_NAMES.contains(&lower.as_str()) {
        return ToolCategory::Write;
    }
    if EDIT_TOOL_NAMES.contains(&lower.as_str()) {
        return ToolCategory::Edit;
    }
    if READ_TOOL_NAMES.contains(&lower.as_str()) {
        return ToolCategory::Read;
    }
    if PLAN_TOOL_NAMES.contains(&lower.as_str()) {
        return ToolCategory::Plan;
    }
    if BASH_TOOL_NAMES.contains(&lower.as_str()) {
        if let Some(cmd) = command {
            // Redirection (write) trumps in-place edit trumps read inspection.
            if BASH_WRITE_PATTERNS.iter().any(|p| cmd.contains(p)) {
                return ToolCategory::Write;
            }
            if BASH_EDIT_PATTERNS.iter().any(|p| cmd.contains(p)) {
                return ToolCategory::Edit;
            }
            if BASH_READ_PATTERNS.iter().any(|p| cmd.contains(p)) {
                return ToolCategory::Read;
            }
        }
    }
    ToolCategory::Other
}

/// Consecutive clean (severity-zero) tool results, counted back from the most recent.
fn compute_no_error_streak(tool_texts: &[String]) -> u32 {
    let mut streak = 0u32;
    for text in tool_texts.iter().rev() {
        let (sev, _) = classify_text(text);
        if sev > 0.0 {
            break;
        }
        streak += 1;
    }
    streak
}

/// Whether any of the last up-to-three tool results reads as a passing test run.
fn detect_tests_passed(tool_texts: &[String]) -> bool {
    let recent = if tool_texts.len() > 3 {
        &tool_texts[tool_texts.len() - 3..]
    } else {
        tool_texts
    };
    recent.iter().any(|text| {
        let lower = text.to_lowercase();
        TEST_PASS_PHRASES.iter().any(|p| lower.contains(p))
            && !TEST_FAILURE_LITERAL.iter().any(|p| lower.contains(p))
            && !has_nonzero_failure_count(&lower)
    })
}

/// Whether the text contains a nonzero failure/error count like "1 failed" (but not
/// "0 failed" or "errored").
fn has_nonzero_failure_count(lower: &str) -> bool {
    for kw in NUMERIC_FAILURE_KEYWORDS {
        let mut cursor = 0usize;
        while let Some(rel) = lower[cursor..].find(kw) {
            let kw_start = cursor + rel;
            let kw_end = kw_start + kw.len();
            // Require a word boundary after the keyword so "errored" doesn't match "error".
            let boundary_after = lower[kw_end..]
                .chars()
                .next()
                .is_none_or(|c| !c.is_ascii_alphanumeric());
            if boundary_after {
                let prefix = &lower[..kw_start];
                let trimmed = prefix.trim_end_matches(char::is_whitespace);
                let digits_rev: String = trimmed
                    .chars()
                    .rev()
                    .take_while(char::is_ascii_digit)
                    .collect();
                if !digits_rev.is_empty() && digits_rev.chars().any(|d| d != '0') {
                    return true;
                }
            }
            cursor = kw_start + kw.len();
        }
    }
    false
}

/// Joins the text blocks of a tool result's content, or `None` when it has no text.
fn content_to_text(content: &[ContentBlock]) -> Option<String> {
    let parts: Vec<&str> = content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

/// Extracts a [`ToolResultSignal`] from a request's conversation history, windowing the
/// `recent_*` counters over the last `recent_window` tool calls.
fn extract_tool_signals_with_window(request: &Request, recent_window: usize) -> ToolResultSignal {
    let messages = &request.llm_request.messages;
    let mut tool_texts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ObservedToolCall> = Vec::new();
    let mut prompt_char_count = 0u32;

    for message in messages {
        let is_user = matches!(message.role, Role::User);
        let mut user_text_len = 0u32;
        for block in &message.content {
            match block {
                ContentBlock::ToolCall(call) => {
                    let command = call
                        .arguments
                        .get("command")
                        .and_then(|value| value.as_str())
                        .map(str::to_lowercase);
                    tool_calls.push(ObservedToolCall {
                        name: call.name.clone(),
                        command,
                    });
                }
                ContentBlock::ToolResult(result) => {
                    if let Some(text) = content_to_text(&result.content) {
                        tool_texts.push(text);
                    }
                }
                ContentBlock::Text { text } if is_user => {
                    user_text_len += text.len() as u32;
                }
                _ => {}
            }
        }
        if is_user && user_text_len > 0 {
            prompt_char_count = user_text_len;
        }
    }

    let turn_depth = messages.len() as u32;
    build_signal(
        &tool_texts,
        &tool_calls,
        turn_depth,
        prompt_char_count,
        recent_window,
    )
}

/// Extracts a [`ToolResultSignal`] using the default recent window.
fn extract_tool_signals(request: &Request) -> ToolResultSignal {
    extract_tool_signals_with_window(request, DEFAULT_RECENT_WINDOW)
}

/// Assembles a [`ToolResultSignal`] from collected tool texts and calls.
fn build_signal(
    tool_texts: &[String],
    tool_calls: &[ObservedToolCall],
    turn_depth: u32,
    prompt_char_count: u32,
    recent_window: usize,
) -> ToolResultSignal {
    let (severity, patterns) = tool_texts
        .last()
        .map(|text| classify_text(text))
        .unwrap_or((0.0, Vec::new()));
    let no_error_streak = compute_no_error_streak(tool_texts);

    // Single reverse pass: cumulative + windowed counts, and the trailing bash streak.
    let recent_start = tool_calls.len().saturating_sub(recent_window);
    let mut signal = ToolResultSignal {
        severity,
        patterns,
        no_error_streak,
        turn_depth,
        prompt_char_count,
        tests_passed: detect_tests_passed(tool_texts),
        ..ToolResultSignal::default()
    };
    let mut streak_open = true;
    for (index, call) in tool_calls.iter().enumerate().rev() {
        let category = classify_tool_call(&call.name, call.command.as_deref());
        if streak_open {
            if matches!(category, ToolCategory::Other) {
                signal.pure_bash_streak += 1;
            } else {
                streak_open = false;
            }
        }
        let recent = index >= recent_start;
        match category {
            ToolCategory::Write => {
                signal.write_count += 1;
                signal.recent_write_count += u32::from(recent);
            }
            ToolCategory::Edit => {
                signal.edit_count += 1;
                signal.recent_edit_count += u32::from(recent);
            }
            ToolCategory::Read => {
                signal.read_count += 1;
                signal.recent_read_count += u32::from(recent);
            }
            ToolCategory::Plan => {
                signal.todowrite_count += 1;
                signal.recent_todowrite_count += u32::from(recent);
            }
            ToolCategory::Other => {}
        }
    }
    signal
}

// --- Heuristic scoring (port of stage_router scorer + overrides) ----------------------

/// The two model tiers the cascade routes between.
#[derive(Clone, Copy)]
enum Tier {
    /// The stronger, more expensive model.
    Capable,
    /// The cheaper, faster model.
    Efficient,
}

/// Weighted linear coefficients over [`Dimensions`]. Positive pushes toward capable,
/// negative toward efficient; magnitudes are calibrated so one high-impact axis clears
/// the default confidence threshold.
struct Weights {
    severity: f64,
    no_error_streak_intensity: f64,
    write_intensity: f64,
    edit_intensity: f64,
    recent_write_intensity: f64,
    planning_active: f64,
    pure_bash_intensity: f64,
    stuck_exploring: f64,
    no_progress: f64,
    tests_passed: f64,
}

/// Default scorer weights, mirroring the reference stage_router calibration.
const DEFAULT_WEIGHTS: Weights = Weights {
    severity: 0.80,
    stuck_exploring: 0.70,
    no_progress: 0.60,
    tests_passed: -0.80,
    planning_active: -0.70,
    write_intensity: -0.40,
    edit_intensity: -0.30,
    recent_write_intensity: -0.30,
    pure_bash_intensity: -0.30,
    no_error_streak_intensity: -0.20,
};

/// The scorer-ready projection of a [`ToolResultSignal`], every field in `[0, 1]`.
struct Dimensions {
    severity: f64,
    no_error_streak_intensity: f64,
    write_intensity: f64,
    edit_intensity: f64,
    recent_write_intensity: f64,
    planning_active: f64,
    pure_bash_intensity: f64,
    stuck_exploring: f64,
    no_progress: f64,
    tests_passed: f64,
}

/// Maps a nonnegative count into `[0, 1)`; `scale` is the half-saturation point.
fn saturating(x: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0 - (-x / scale).exp()
    }
}

/// Fraction of `numerator` in `denominator`, or `0.0` when there is nothing to divide.
fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Projects a signal onto the normalized dimension space the scorer reads.
fn dimensions(signal: &ToolResultSignal) -> Dimensions {
    let total_tool_ops = f64::from(signal.write_count + signal.edit_count + signal.read_count);
    let recent_tool_ops =
        f64::from(signal.recent_write_count + signal.recent_edit_count + signal.recent_read_count);
    // A deep turn with many reads but almost no writes reads as stuck exploration.
    let stuck = signal.turn_depth >= 8 && signal.write_count <= 1 && signal.read_count >= 5;
    let no_progress = signal.turn_depth > 60 && signal.write_count == 0;
    Dimensions {
        severity: f64::from(signal.severity),
        no_error_streak_intensity: saturating(f64::from(signal.no_error_streak), 3.0),
        write_intensity: ratio(f64::from(signal.write_count), total_tool_ops),
        edit_intensity: ratio(f64::from(signal.edit_count), total_tool_ops),
        recent_write_intensity: ratio(f64::from(signal.recent_write_count), recent_tool_ops),
        planning_active: f64::from(signal.recent_todowrite_count > 0),
        pure_bash_intensity: saturating(f64::from(signal.pure_bash_streak), PURE_BASH_NORM),
        stuck_exploring: f64::from(stuck),
        no_progress: f64::from(no_progress),
        // Treat passing tests as confirmatory only once real changes have been made;
        // early runs against an unmodified tree are exploratory, not settled.
        tests_passed: f64::from(signal.tests_passed && signal.write_count >= 3),
    }
}

/// Weighted linear score in `[-1, +1]`: positive favors capable, negative favors
/// efficient. Callers rerange this onto the `[0, 1]` capability confidence via
/// `(score + 1) / 2`.
fn weighted_score(dimensions: &Dimensions) -> f64 {
    let w = &DEFAULT_WEIGHTS;
    let raw = dimensions.severity * w.severity
        + dimensions.no_error_streak_intensity * w.no_error_streak_intensity
        + dimensions.write_intensity * w.write_intensity
        + dimensions.edit_intensity * w.edit_intensity
        + dimensions.recent_write_intensity * w.recent_write_intensity
        + dimensions.planning_active * w.planning_active
        + dimensions.pure_bash_intensity * w.pure_bash_intensity
        + dimensions.stuck_exploring * w.stuck_exploring
        + dimensions.no_progress * w.no_progress
        + dimensions.tests_passed * w.tests_passed;
    raw.clamp(-1.0, 1.0)
}

/// Non-negotiable, signal-derived shortcuts that bypass the scorer entirely.
fn apply_overrides(signal: &ToolResultSignal) -> Option<Tier> {
    if signal.severity >= CRITICAL {
        return Some(Tier::Capable);
    }
    // Passing tests after a settled run (deep enough, few writes) is a strong efficient cue.
    if signal.tests_passed
        && signal.turn_depth >= CLEAN_TESTS_MIN_TURN_DEPTH
        && signal.write_count <= CLEAN_TESTS_MAX_WRITES
    {
        return Some(Tier::Efficient);
    }
    None
}

/// Heuristic first stage of a two-tier cascade router.
///
/// Register the same instance as both a processor and a classifier; the processor extracts
/// the signal and the classifier scores it, sharing [`State`].
pub struct StagedRouter {
    capable_model: String,
    efficient_model: String,
    confidence_threshold: f64,
}

impl StagedRouter {
    /// Creates a router between `capable_model` and `efficient_model` at the default
    /// confidence threshold.
    pub fn new(capable_model: impl Into<String>, efficient_model: impl Into<String>) -> Self {
        Self {
            capable_model: capable_model.into(),
            efficient_model: efficient_model.into(),
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
        }
    }

    /// Sets the confidence bound at which the heuristic decides rather than defers. The
    /// abstain band is symmetric around the `0.5` neutral point: `[1 - threshold, threshold]`.
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// The model name for a tier.
    fn model_for(&self, tier: Tier) -> &str {
        match tier {
            Tier::Capable => &self.capable_model,
            Tier::Efficient => &self.efficient_model,
        }
    }

    /// One score for the given tier's model at `confidence`.
    fn tier_score(&self, tier: Tier, confidence: f64) -> Vec<Score> {
        vec![Score {
            confidence,
            target: self.model_for(tier).to_string(),
        }]
    }
}

#[async_trait(?Send)]
impl Processor for StagedRouter {
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<(), BoxErr> {
        // Extract the tool-result signal from the request's history and stash it for the
        // classifier. The metadata processor's Metadata fact is left untouched.
        if let Event::Request(request) = event {
            state.insert(extract_tool_signals(request));
        }
        Ok(())
    }
}

#[async_trait]
impl Classifier for StagedRouter {
    async fn score(&self, state: &mut State, _request: &Request) -> Result<Vec<Score>, BoxErr> {
        let Some(signal) = state.get::<ToolResultSignal>() else {
            // No signal extracted yet: abstain.
            return Ok(Vec::new());
        };

        // Hard overrides jump to the extremes of the capability axis:
        // capable → 1.0, efficient → 0.0.
        if let Some(tier) = apply_overrides(signal) {
            let confidence = match tier {
                Tier::Capable => 1.0,
                Tier::Efficient => 0.0,
            };
            return Ok(self.tier_score(tier, confidence));
        }

        // Rerange the signed [-1, 1] weighted score onto the [0, 1] capability
        // confidence, then compare it directly against the threshold band. Capable
        // above `threshold`, efficient below `1 - threshold`, abstain in between.
        let confidence = (weighted_score(&dimensions(signal)) + 1.0) / 2.0;
        if confidence >= self.confidence_threshold {
            Ok(self.tier_score(Tier::Capable, confidence))
        } else if confidence <= 1.0 - self.confidence_threshold {
            Ok(self.tier_score(Tier::Efficient, confidence))
        } else {
            // Ambiguous band: defer to the next cascade stage.
            Ok(Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use switchyard_protocol::{
        ContentBlock, LlmRequest, Message, Request, Role, ToolCall, ToolResult,
    };

    fn tool_call(name: &str, command: Option<&str>) -> ContentBlock {
        let arguments = match command {
            Some(command) => serde_json::json!({ "command": command }),
            None => serde_json::Value::Null,
        };
        ContentBlock::ToolCall(ToolCall {
            id: "call".to_string(),
            name: name.to_string(),
            arguments,
        })
    }

    fn tool_result(text: &str) -> ContentBlock {
        ContentBlock::ToolResult(ToolResult {
            tool_call_id: "call".to_string(),
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            is_error: None,
        })
    }

    fn assistant(content: Vec<ContentBlock>) -> Message {
        Message {
            role: Role::Assistant,
            content,
        }
    }

    fn user(content: Vec<ContentBlock>) -> Message {
        Message {
            role: Role::User,
            content,
        }
    }

    fn request_from(messages: Vec<Message>) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("auto".to_string()),
                messages,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    fn router() -> StagedRouter {
        StagedRouter::new("capable-model", "efficient-model")
    }

    // --- extraction --------------------------------------------------------------------

    #[test]
    fn categorizes_tool_calls_including_bash_commands() {
        let signal = extract_tool_signals(&request_from(vec![assistant(vec![
            tool_call("Write", None),
            tool_call("Edit", None),
            tool_call("Read", None),
            tool_call("TodoWrite", None),
            tool_call("Bash", Some("sed -i 's/a/b/' file")),
            tool_call("Bash", Some("grep foo file")),
        ])]));
        assert_eq!(signal.write_count, 1);
        assert_eq!(signal.edit_count, 2); // Edit tool + `sed -i` bash edit
        assert_eq!(signal.read_count, 2); // Read tool + `grep` bash read
        assert_eq!(signal.todowrite_count, 1);
    }

    #[test]
    fn severity_and_patterns_come_from_the_last_tool_result() {
        let signal = extract_tool_signals(&request_from(vec![
            user(vec![tool_result("all good here")]),
            user(vec![tool_result("Traceback (most recent call last):")]),
        ]));
        assert_eq!(signal.severity, HARD);
        assert!(signal.patterns.iter().any(|p| p == "traceback"));
    }

    #[test]
    fn no_error_streak_counts_trailing_clean_results() {
        let signal = extract_tool_signals(&request_from(vec![
            user(vec![tool_result("out of memory")]),
            user(vec![tool_result("ok")]),
            user(vec![tool_result("done")]),
        ]));
        assert_eq!(signal.no_error_streak, 2);
    }

    #[test]
    fn detects_passing_and_failing_test_runs() {
        let passed = extract_tool_signals(&request_from(vec![user(vec![tool_result(
            "test result: ok. 5 passed",
        )])]));
        assert!(passed.tests_passed);

        let failed = extract_tool_signals(&request_from(vec![user(vec![tool_result(
            "test result: FAILED. 1 failed, 4 passed",
        )])]));
        assert!(!failed.tests_passed);
    }

    #[test]
    fn recent_window_bounds_the_recent_counts() {
        // Five writes, default window of three: only the last three count as recent.
        let calls: Vec<ContentBlock> = (0..5).map(|_| tool_call("Write", None)).collect();
        let signal = extract_tool_signals(&request_from(vec![assistant(calls)]));
        assert_eq!(signal.write_count, 5);
        assert_eq!(signal.recent_write_count, DEFAULT_RECENT_WINDOW as u32);
    }

    // --- routing (processor + classifier) ----------------------------------------------

    async fn route(router: &StagedRouter, request: &Request) -> Result<Vec<Score>, BoxErr> {
        let mut state = State::default();
        router.process(&mut state, Event::Request(request)).await?;
        router.score(&mut state, request).await
    }

    #[tokio::test]
    async fn no_signal_abstains() -> Result<(), BoxErr> {
        let mut state = State::default();
        let scores = router().score(&mut state, &request_from(vec![])).await?;
        assert!(scores.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn critical_error_routes_capable() -> Result<(), BoxErr> {
        let request = request_from(vec![user(vec![tool_result("fatal: out of memory")])]);
        let scores = route(&router(), &request).await?;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].target, "capable-model");
        assert_eq!(scores[0].confidence, 1.0); // capable override sits at the capable extreme
        Ok(())
    }

    #[tokio::test]
    async fn planning_activity_routes_efficient() -> Result<(), BoxErr> {
        let request = request_from(vec![assistant(vec![tool_call("TodoWrite", None)])]);
        let scores = route(&router(), &request).await?;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].target, "efficient-model");
        // planning_active=1.0 · -0.70 → raw -0.70 → confidence (−0.70 + 1)/2 = 0.15.
        assert!((scores[0].confidence - 0.15).abs() < 1e-9);
        Ok(())
    }

    #[tokio::test]
    async fn stuck_exploration_routes_capable() -> Result<(), BoxErr> {
        // Eight read-only turns: deep, many reads, no writes → stuck exploration.
        let messages: Vec<Message> = (0..8)
            .map(|_| assistant(vec![tool_call("Read", None)]))
            .collect();
        let scores = route(&router(), &request_from(messages)).await?;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].target, "capable-model");
        // stuck_exploring=1.0 · 0.70 → raw 0.70 → confidence (0.70 + 1)/2 = 0.85.
        assert!((scores[0].confidence - 0.85).abs() < 1e-9);
        Ok(())
    }

    #[tokio::test]
    async fn read_only_activity_is_ambiguous() -> Result<(), BoxErr> {
        // A single read carries no weighted signal: defer to the next stage.
        let request = request_from(vec![assistant(vec![tool_call("Read", None)])]);
        assert!(route(&router(), &request).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn one_router_serves_both_roles() -> Result<(), BoxErr> {
        let router = Arc::new(router());
        let processor: Arc<dyn Processor> = router.clone();
        let classifier: Arc<dyn Classifier> = router;
        let mut state = State::default();

        let request = request_from(vec![user(vec![tool_result("out of memory")])]);
        processor
            .process(&mut state, Event::Request(&request))
            .await?;
        let scores = classifier.score(&mut state, &request).await?;
        assert_eq!(
            scores.first().map(|score| score.target.as_str()),
            Some("capable-model")
        );
        Ok(())
    }

    // --- classify_text (port of tool_signals.rs) ---------------------------------------
    // Reference: crates/switchyard-components/src/dimension_collector/tool_signals.rs

    #[test]
    fn clean_text_has_zero_severity() {
        let (sev, patterns) = classify_text("everything went fine");
        assert_eq!(sev, 0.0);
        assert!(patterns.is_empty());
    }

    #[test]
    fn traceback_is_hard() {
        let (sev, patterns) = classify_text("Traceback (most recent call last):\n  ValueError");
        assert_eq!(sev, HARD);
        assert!(patterns.contains(&"traceback".to_string()));
    }

    #[test]
    fn oom_is_critical() {
        let (sev, _) = classify_text("Out of memory: kill process 1234");
        assert_eq!(sev, CRITICAL);
    }

    #[test]
    fn severity_is_max_across_patterns() {
        // exit_nonzero (SOFT) + traceback (HARD) → HARD.
        let (sev, _) = classify_text("exit code 1\nTraceback (most recent call last):");
        assert_eq!(sev, HARD);
    }

    #[test]
    fn no_error_streak_all_clean() {
        let texts = vec!["ok".to_string(), "all good".to_string()];
        assert_eq!(compute_no_error_streak(&texts), 2);
    }

    #[test]
    fn no_error_streak_stops_at_error() {
        let texts = vec![
            "Traceback (most recent call last):".to_string(),
            "ok".to_string(),
            "ok".to_string(),
        ];
        assert_eq!(compute_no_error_streak(&texts), 2);
    }

    // --- classify_tool_call (port of tool_signals.rs) ----------------------------------
    // Reference: crates/switchyard-components/src/dimension_collector/tool_signals.rs

    #[test]
    fn todowrite_and_update_plan_classify_as_plan() {
        assert_eq!(classify_tool_call("TodoWrite", None), ToolCategory::Plan);
        assert_eq!(classify_tool_call("todo_write", None), ToolCategory::Plan);
        // `update_plan` is codex's equivalent of `todowrite`.
        assert_eq!(classify_tool_call("update_plan", None), ToolCategory::Plan);
    }

    #[test]
    fn codex_shell_command_runs_bash_pattern_match() {
        // shell_command + heredoc -> Write.
        assert_eq!(
            classify_tool_call("shell_command", Some("cat > /app/foo.py <<'eof'\nx=1\neof")),
            ToolCategory::Write,
        );
        // shell_command + read-like inspection -> Read.
        assert_eq!(
            classify_tool_call("shell_command", Some("ls /app")),
            ToolCategory::Read
        );
        // shell_command without matching patterns -> Other.
        assert_eq!(
            classify_tool_call("shell_command", Some("./run_tests.sh")),
            ToolCategory::Other
        );
    }

    #[test]
    fn read_tool_and_bash_reads_classify_as_read() {
        assert_eq!(classify_tool_call("Read", None), ToolCategory::Read);
        assert_eq!(classify_tool_call("View", None), ToolCategory::Read);
        for cmd in [
            "cat /etc/passwd",
            "grep foo bar.txt",
            "ls /app",
            "find . -name '*.py'",
        ] {
            assert_eq!(
                classify_tool_call("Bash", Some(cmd)),
                ToolCategory::Read,
                "expected Read for {cmd}"
            );
        }
    }

    #[test]
    fn bash_write_precedence_over_read() {
        // `cat /file > /out` contains both `cat /` (read) and `> /` (write);
        // write redirection must win.
        assert_eq!(
            classify_tool_call("Bash", Some("cat /etc/hosts > /tmp/out")),
            ToolCategory::Write,
        );
    }

    // --- bash-command bucketing (port of tool_signals.rs) ------------------------------
    // Reference: crates/switchyard-components/src/dimension_collector/tool_signals.rs

    #[test]
    fn bash_heredoc_counts_as_write() {
        let signal = extract_tool_signals(&request_from(vec![assistant(vec![tool_call(
            "Bash",
            Some("cat > /tmp/test.py <<'EOF'\nprint(1)\nEOF"),
        )])]));
        assert_eq!(
            signal.write_count, 1,
            "Bash heredoc should bucket into write_count"
        );
        assert_eq!(signal.edit_count, 0);
    }

    #[test]
    fn bash_sed_inplace_counts_as_edit() {
        let signal = extract_tool_signals(&request_from(vec![assistant(vec![tool_call(
            "Bash",
            Some("sed -i 's/foo/bar/g' /app/file.py"),
        )])]));
        assert_eq!(
            signal.edit_count, 1,
            "Bash sed -i should bucket into edit_count"
        );
        assert_eq!(signal.write_count, 0);
    }

    #[test]
    fn bash_non_mutating_does_not_count() {
        let signal = extract_tool_signals(&request_from(vec![assistant(vec![
            tool_call("Bash", Some("ls -la /app")),
            tool_call("Bash", Some("cat /app/main.py")),
        ])]));
        assert_eq!(signal.write_count, 0);
        assert_eq!(signal.edit_count, 0);
    }

    #[test]
    fn pure_bash_streak_counts_trailing_other() {
        // 5 trailing non-classified Bash calls → streak == 5.
        let signal = extract_tool_signals(&request_from(vec![assistant(vec![
            tool_call("Bash", Some("make")),
            tool_call("Bash", Some("./configure")),
            tool_call("Bash", Some("make install")),
            tool_call("Bash", Some("./run.sh")),
            tool_call("Bash", Some("./test")),
        ])]));
        assert_eq!(signal.pure_bash_streak, 5);
        assert_eq!(signal.write_count, 0);
        assert_eq!(signal.read_count, 0);
    }

    #[test]
    fn pure_bash_streak_resets_on_write() {
        // A trailing Write closes the streak, even after a bash `make`.
        let signal = extract_tool_signals(&request_from(vec![assistant(vec![
            tool_call("Bash", Some("make")),
            tool_call("Write", None),
        ])]));
        assert_eq!(signal.pure_bash_streak, 0);
        assert_eq!(signal.write_count, 1);
    }

    #[test]
    fn recent_window_tracks_todowrite_and_read() {
        // Final 3 tool calls: TodoWrite, Read, TodoWrite (preceded by a bash `make`).
        let signal = extract_tool_signals(&request_from(vec![assistant(vec![
            tool_call("Bash", Some("make")),
            tool_call("TodoWrite", None),
            tool_call("Read", None),
            tool_call("TodoWrite", None),
        ])]));
        assert_eq!(signal.todowrite_count, 2);
        assert_eq!(signal.recent_todowrite_count, 2);
        assert_eq!(signal.read_count, 1);
        assert_eq!(signal.recent_read_count, 1);
    }

    #[test]
    fn recent_window_size_is_caller_overridable() {
        // 5 writes then 1 edit. window=3 → recent_writes=2; window=6 → recent_writes=5.
        let mut calls: Vec<ContentBlock> = (0..5).map(|_| tool_call("Write", None)).collect();
        calls.push(tool_call("Edit", None));
        let request = request_from(vec![assistant(calls)]);

        let narrow = extract_tool_signals_with_window(&request, 3);
        assert_eq!(narrow.recent_write_count, 2);
        assert_eq!(narrow.recent_edit_count, 1);

        let wide = extract_tool_signals_with_window(&request, 6);
        assert_eq!(wide.recent_write_count, 5);
        assert_eq!(wide.recent_edit_count, 1);
    }

    // --- tests_passed detection (port of tool_signals.rs) ------------------------------
    // Reference: crates/switchyard-components/src/dimension_collector/tool_signals.rs
    // Guards `has_nonzero_failure_count`: "0 failed" / "0 errors" clean summaries must
    // not be misread as failures, while a nonzero count before the keyword must.

    #[test]
    fn tests_passed_accepts_cargo_clean_summary() {
        assert!(detect_tests_passed(&[
            "running 3 tests\ntest result: ok. 3 passed; 0 failed; 0 ignored".to_string()
        ]));
    }

    #[test]
    fn tests_passed_rejects_cargo_real_failure() {
        assert!(!detect_tests_passed(&[
            "running 3 tests\ntest result: FAILED. 2 passed; 1 failed; 0 ignored".to_string()
        ]));
    }

    #[test]
    fn tests_passed_accepts_go_clean_summary() {
        assert!(detect_tests_passed(&[
            "ok  github.com/foo/bar\t0.012s (5 passed, 0 errors)".to_string()
        ]));
    }

    #[test]
    fn tests_passed_accepts_pytest_zero_errors() {
        assert!(detect_tests_passed(&[
            "5 passed, 0 errors in 0.30s".to_string()
        ]));
    }

    #[test]
    fn tests_passed_detects_diy_checkmark() {
        assert!(detect_tests_passed(&["✓ all checks passed".to_string()]));
    }

    #[test]
    fn tests_passed_ignores_partial_failures() {
        assert!(!detect_tests_passed(&[
            "2 failed, 5 passed in 0.56s".to_string()
        ]));
    }

    // --- scorer (port of scorer.py / test_stage_router_scorer.py) ----------------------
    // Reference: switchyard/lib/processors/stage_router/scorer.py
    //            tests/test_stage_router_scorer.py

    fn zero_dimensions() -> Dimensions {
        Dimensions {
            severity: 0.0,
            no_error_streak_intensity: 0.0,
            write_intensity: 0.0,
            edit_intensity: 0.0,
            recent_write_intensity: 0.0,
            planning_active: 0.0,
            pure_bash_intensity: 0.0,
            stuck_exploring: 0.0,
            no_progress: 0.0,
            tests_passed: 0.0,
        }
    }

    #[test]
    fn zero_dimensions_score_to_zero() {
        assert_eq!(weighted_score(&zero_dimensions()), 0.0);
    }

    #[test]
    fn critical_severity_pushes_toward_capable() {
        let dims = Dimensions {
            severity: 1.0,
            ..zero_dimensions()
        };
        assert!(weighted_score(&dims) > 0.0);
    }

    #[test]
    fn tests_passed_pushes_toward_efficient() {
        let dims = Dimensions {
            tests_passed: 1.0,
            ..zero_dimensions()
        };
        assert!(weighted_score(&dims) < 0.0);
    }

    #[test]
    fn score_is_clamped_to_unit_interval() {
        // Positive axes sum past +1 (0.80 + 0.70 + 0.60 = 2.10) → clamp to +1.
        let capable = Dimensions {
            severity: 1.0,
            stuck_exploring: 1.0,
            no_progress: 1.0,
            ..zero_dimensions()
        };
        assert_eq!(weighted_score(&capable), 1.0);
        // Negative axes sum past -1 → clamp to -1.
        let efficient = Dimensions {
            tests_passed: 1.0,
            planning_active: 1.0,
            write_intensity: 1.0,
            edit_intensity: 1.0,
            recent_write_intensity: 1.0,
            pure_bash_intensity: 1.0,
            no_error_streak_intensity: 1.0,
            ..zero_dimensions()
        };
        assert_eq!(weighted_score(&efficient), -1.0);
    }

    #[test]
    fn dimensions_normalise_a_real_extracted_signal() {
        // End-to-end: extraction → ToolResultSignal → Dimensions, all in [0, 1].
        // Port of test_stage_router_scorer.py::test_from_signal_normalises_real_extracted_signal.
        let signal = extract_tool_signals(&request_from(vec![
            assistant(vec![tool_call("Write", None)]),
            user(vec![tool_result("ok")]),
            user(vec![ContentBlock::Text {
                text: "next".to_string(),
            }]),
        ]));
        let dims = dimensions(&signal);
        assert!((0.0..=1.0).contains(&dims.severity));
        assert!((0.0..=1.0).contains(&dims.write_intensity));
        assert!(dims.write_intensity > 0.0); // we issued one Write call
    }

    // --- routing overrides (port of test_stage_router_pickers.py) ----------------------
    // Reference: tests/test_stage_router_pickers.py
    // NOTE: the reference pickers fall open to a default tier (capable-first /
    // efficient-first) or an LLM classifier on low-confidence turns; StagedRouter
    // instead abstains (empty scores) to defer to the next cascade stage, so only the
    // override / confident-score paths are portable here.

    #[tokio::test]
    async fn tests_passed_after_settled_run_routes_efficient() -> Result<(), BoxErr> {
        // 3 edits + 3 "all tests passed" results + 12 user turns: tests_passed, deep,
        // no writes → the settled-tests override forces EFFICIENT.
        let mut messages: Vec<Message> = (0..3)
            .map(|_| assistant(vec![tool_call("Edit", None)]))
            .collect();
        messages.extend((0..3).map(|_| user(vec![tool_result("all tests passed")])));
        messages.extend((0..12).map(|_| {
            user(vec![ContentBlock::Text {
                text: "ok continue".to_string(),
            }])
        }));
        let scores = route(&router(), &request_from(messages)).await?;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].target, "efficient-model");
        assert_eq!(scores[0].confidence, 0.0); // efficient override sits at the efficient extreme
        Ok(())
    }
}
