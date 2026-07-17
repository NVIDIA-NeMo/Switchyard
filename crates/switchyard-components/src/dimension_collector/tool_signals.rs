// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-result context signals — thin adapter over libsy's extractor.
//!
//! The extraction logic lives in [`switchyard_libsy::ToolSignals`]. This module
//! bridges the crate's [`ChatRequest`] (a format-tagged JSON body) to libsy's
//! [`switchyard_protocol::Request`] (raw body + wire-format metadata) so the two
//! request models share one implementation.

use switchyard_core::{ChatRequest, ChatRequestType};
use switchyard_protocol::{Metadata, Request, WireFormat};

/// The tool-signal output type. Re-exported from libsy so downstream consumers
/// (the `ToolResultSignal` stamped on `ProxyContext`) see a single type.
pub use switchyard_libsy::{ToolSignals as ToolResultSignal, DEFAULT_RECENT_WINDOW};

const SOFT: f32 = 0.3;
const HARD: f32 = 0.7;
const CRITICAL: f32 = 1.0;

// ─── pattern table ────────────────────────────────────────────────────────────

/// (name, severity, lower-cased substrings — any hit fires the pattern)
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
        &[
            "filenotfounderror:",
            "no such file or directory",
            // Claude Code Read-tool miss. Anchored as "file does not exist" (not a
            // bare "does not exist", which fires on `ls` output and prose) — trace-
            // mined across 1006 local trajectories at 22 true / 2 false positives.
            "file does not exist",
        ],
    ),
    // SOFT: plain non-zero exit without a recognisable exception traceback.
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

// Bash subcommand patterns. Lowercased; callers must lowercase the command
// before matching. Bucketed into write_count / edit_count alongside the
// dedicated `Write` / `Edit` tools.
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

// Read-like Bash inspections. Match only when none of the write/edit patterns
// fire (redirection / in-place edit trumps the read intent of the command).
static BASH_READ_PATTERNS: &[&str] = &[
    "cat /", "cat ./", "cat ../", "grep ", "ls ", "ls -", "find ", "head ", "tail ", "wc ",
    "diff ", "which ", "ps ", "df ", "du ", "stat ", "file ", "less ", "more ",
];

static READ_TOOL_NAMES: &[&str] = &["read", "view"];

// Planning / scratchpad tool calls. Used by Opus as a struggle indicator
// in the capable-first picker direction.
// `update_plan` is codex's equivalent of `todowrite`.
static PLAN_TOOL_NAMES: &[&str] = &["todowrite", "todo_write", "todo", "update_plan"];

// Tool names that route through Bash-command pattern matching. `bash` is
// claude-code's name; `shell_command` is codex's; `shell` / `local_shell_call`
// are seen on some OpenAI-derived harnesses.
static BASH_TOOL_NAMES: &[&str] = &["bash", "shell_command", "shell", "local_shell_call"];

// Prefer false negatives: tests_passed routes the picker to EFFICIENT, so a false
// positive would drop tier on an unfinished task.
static TEST_PASS_PHRASES: &[&str] = &[
    " passed",
    "passed in",
    "tests passed",
    "all tests passed",
    "test ok",
    "test result: ok",
    "passed.\n",
    "tests pass",
    "\nok ", // go test; newline-anchored to avoid "...lookup..." mid-text
    "✓ ",
];

// Literal failure phrases that cannot appear inside a clean run. Substring
// matched as-is. Patterns that pair with a count (e.g. "failed", "errors")
// are handled separately by `has_nonzero_failure_count` so "0 failed" /
// "0 errors" do not trigger a false negative.
static TEST_FAILURE_LITERAL: &[&str] = &["✗ ", "fatal:", "assertionerror", "error:"];

// Count-prefixed failure keywords. Trip only when a nonzero integer precedes
// the keyword (modulo whitespace), so cargo's "0 failed" and go's
// "0 errors" summaries on a clean run are not misread as failures.
static NUMERIC_FAILURE_KEYWORDS: &[&str] = &["failed", "failure", "failures", "errors", "error"];

/// Default sliding-window size for `recent_*` counts.
///
/// Calibrated against agent trajectories where a 3-call horizon is enough
/// to capture the "what is the agent doing right now" signal without
/// over-smoothing short tasks. Override per-extractor by calling
/// [`extract_tool_signals_with_window`] directly.
pub const DEFAULT_RECENT_WINDOW: usize = 3;

// ─── output type ─────────────────────────────────────────────────────────────

/// Tool-execution signals stamped on `ProxyContext`. Read by stage_router pickers
/// via [`crate::get_tool_result_signal`].
#[derive(Clone, Debug, Default)]
pub struct ToolResultSignal {
    /// Max severity across the recent window (last `recent_window` tool results):
    /// `0.0` clean · `0.3` soft (exit_nonzero) · `0.7` hard · `1.0` critical.
    /// Windowed so an error persists through the recovery turns instead of clearing
    /// the instant the next result is clean.
    pub severity: f32,
    /// Error pattern names from the highest-severity result in the recent window.
    pub patterns: Vec<String>,
    /// Consecutive clean tool results back from the most recent. `0` if the last failed.
    pub no_error_streak: u32,
    pub edit_count: u32,
    pub write_count: u32,
    /// Read-type calls (Read tool + read-like Bash). Used by the build-pit gate.
    pub read_count: u32,
    /// TodoWrite tool calls. Strong fail predictor for Opus; used by the
    /// `capable_first` drop-to-weak gate.
    pub todowrite_count: u32,
    /// Edit-type calls within the last [`RECENT_WINDOW`] tool calls.
    pub recent_edit_count: u32,
    /// Write-type calls within the last [`RECENT_WINDOW`] tool calls.
    pub recent_write_count: u32,
    /// Read-type calls within the last [`RECENT_WINDOW`] tool calls.
    pub recent_read_count: u32,
    /// TodoWrite calls within the last [`RECENT_WINDOW`] tool calls.
    pub recent_todowrite_count: u32,
    /// Consecutive trailing tool calls that hit no Write/Edit/Read/Plan
    /// patterns — proxy for the "stuck in non-Read Bash" build-pit loop.
    pub pure_bash_streak: u32,
    /// At least one of the last three tool results matched a test-pass pattern.
    pub tests_passed: bool,
    /// Message-count proxy for turn depth.
    pub turn_depth: u32,
    /// Char count of the last user message.
    pub prompt_char_count: u32,
}

/// Extract tool-execution signals using the default `recent_*` window.
pub fn extract_tool_signals(request: &ChatRequest) -> ToolResultSignal {
    ToolResultSignal::from_request(&to_protocol_request(request), None)
}

/// Extract tool-execution signals with a caller-supplied `recent_*` window.
pub fn extract_tool_signals_with_window(
    request: &ChatRequest,
    recent_window: usize,
) -> ToolResultSignal {
    let body = request.body();
    let Some(obj) = body.as_object() else {
        return ToolResultSignal::default();
    };

    match request.request_type() {
        ChatRequestType::Anthropic => {
            let messages = obj
                .get("messages")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            extract_from_messages_anthropic(messages, recent_window)
        }
        ChatRequestType::OpenAiResponses => {
            let items = obj
                .get("input")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            extract_from_input_responses(items, recent_window)
        }
        ChatRequestType::OpenAiChat => {
            let messages = obj
                .get("messages")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            extract_from_messages_openai_chat(messages, recent_window)
        }
    }
}

// ─── format-specific extractors ──────────────────────────────────────────────

fn extract_from_messages_openai_chat(messages: &[Value], recent_window: usize) -> ToolResultSignal {
    let mut tool_texts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ObservedToolCall> = Vec::new();
    let mut prompt_char_count: u32 = 0;

    for msg in messages {
        let Some(obj) = msg.as_object() else { continue };
        let role = obj.get("role").and_then(Value::as_str).unwrap_or("");
        match role {
            "tool" => {
                if let Some(text) = content_to_text(obj.get("content")) {
                    tool_texts.push(text);
                }
            }
            "assistant" => {
                if let Some(tc_list) = obj.get("tool_calls").and_then(Value::as_array) {
                    for tc in tc_list {
                        let Some(fn_obj) = tc
                            .as_object()
                            .and_then(|t| t.get("function"))
                            .and_then(|f| f.as_object())
                        else {
                            continue;
                        };
                        let Some(name) = fn_obj.get("name").and_then(Value::as_str) else {
                            continue;
                        };
                        // OpenAI Chat encodes `arguments` as a JSON string.
                        let command = fn_obj
                            .get("arguments")
                            .and_then(Value::as_str)
                            .and_then(|s| serde_json::from_str::<Value>(s).ok())
                            .and_then(|v| {
                                v.get("command")
                                    .and_then(Value::as_str)
                                    .map(|s| s.to_lowercase())
                            });
                        tool_calls.push(ObservedToolCall {
                            name: name.to_string(),
                            command,
                        });
                    }
                }
            }
            "user" => {
                prompt_char_count = user_message_char_count(obj.get("content"));
            }
            _ => {}
        }
    }

    build_signal(
        tool_texts,
        tool_calls,
        messages.len() as u32,
        prompt_char_count,
        recent_window,
    )
}

fn extract_from_messages_anthropic(messages: &[Value], recent_window: usize) -> ToolResultSignal {
    let mut tool_texts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ObservedToolCall> = Vec::new();
    let mut prompt_char_count: u32 = 0;

    for msg in messages {
        let Some(obj) = msg.as_object() else { continue };
        let role = obj.get("role").and_then(Value::as_str).unwrap_or("");
        let content = obj.get("content");

        match role {
            "user" => {
                if let Some(Value::Array(blocks)) = content {
                    let mut saw_text = false;
                    for block in blocks {
                        let Some(b) = block.as_object() else { continue };
                        match b.get("type").and_then(Value::as_str) {
                            Some("tool_result") => {
                                if let Some(text) = content_to_text(b.get("content")) {
                                    tool_texts.push(text);
                                }
                            }
                            Some("text") => {
                                if let Some(t) = b.get("text").and_then(Value::as_str) {
                                    prompt_char_count = t.len() as u32;
                                    saw_text = true;
                                }
                            }
                            _ => {}
                        }
                    }
                    if !saw_text {
                        if let Some(text_val) = content {
                            prompt_char_count = user_message_char_count(Some(text_val));
                        }
                    }
                } else {
                    prompt_char_count = user_message_char_count(content);
                }
            }
            "assistant" => {
                if let Some(Value::Array(blocks)) = content {
                    for block in blocks {
                        let Some(b) = block.as_object() else { continue };
                        if b.get("type").and_then(Value::as_str) == Some("tool_use") {
                            let Some(name) = b.get("name").and_then(Value::as_str) else {
                                continue;
                            };
                            // Anthropic delivers `input` as a parsed object.
                            let command = b
                                .get("input")
                                .and_then(Value::as_object)
                                .and_then(|i| i.get("command"))
                                .and_then(Value::as_str)
                                .map(|s| s.to_lowercase());
                            tool_calls.push(ObservedToolCall {
                                name: name.to_string(),
                                command,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    build_signal(
        tool_texts,
        tool_calls,
        messages.len() as u32,
        prompt_char_count,
        recent_window,
    )
}

fn extract_from_input_responses(items: &[Value], recent_window: usize) -> ToolResultSignal {
    let mut tool_texts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ObservedToolCall> = Vec::new();
    let mut prompt_char_count: u32 = 0;

    for item in items {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let item_type = obj.get("type").and_then(Value::as_str).unwrap_or("");
        let role = obj.get("role").and_then(Value::as_str).unwrap_or("");

        match item_type {
            "function_call_output" => {
                if let Some(output) = obj.get("output").and_then(Value::as_str) {
                    tool_texts.push(output.to_string());
                }
            }
            "function_call" => {
                let Some(name) = obj.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let command = obj
                    .get("arguments")
                    .and_then(Value::as_str)
                    .and_then(|s| serde_json::from_str::<Value>(s).ok())
                    .and_then(|v| {
                        v.get("command")
                            .and_then(Value::as_str)
                            .map(|s| s.to_lowercase())
                    });
                tool_calls.push(ObservedToolCall {
                    name: name.to_string(),
                    command,
                });
            }
            _ => {
                if role == "user" {
                    prompt_char_count = user_message_char_count(obj.get("content"));
                }
            }
        }
    }

    build_signal(
        tool_texts,
        tool_calls,
        items.len() as u32,
        prompt_char_count,
        recent_window,
    )
}

// ─── aggregation ─────────────────────────────────────────────────────────────

fn build_signal(
    tool_texts: Vec<String>,
    tool_calls: Vec<ObservedToolCall>,
    turn_depth: u32,
    prompt_char_count: u32,
    recent_window: usize,
) -> ToolResultSignal {
    // Windowed severity: take the MAX severity across the last `recent_window` tool
    // results rather than only the last one. An error's severity then persists for
    // the recent window and decays out of it — parallel to the windowed `recent_*`
    // counts and `stuck_exploring` — so a fix written a couple of turns after an error
    // still routes on the error signal instead of the router flapping straight back to
    // the weak tier. `patterns` carries the labels of the highest-severity result.
    let sev_start = tool_texts.len().saturating_sub(recent_window.max(1));
    let mut severity = 0.0f32;
    let mut patterns: Vec<String> = Vec::new();
    for text in &tool_texts[sev_start..] {
        let (sev, pats) = classify_text(text);
        if sev > severity {
            severity = sev;
            patterns = pats;
        }
    }

    let no_error_streak = compute_no_error_streak(&tool_texts);

    // Single pass: cumulative + sliding-window counters together. Also tracks
    // the trailing pure-bash streak (consecutive `Other`-category calls back
    // from the end) — the build-pit proxy.
    let recent_start = tool_calls.len().saturating_sub(recent_window);
    let mut write_count = 0u32;
    let mut edit_count = 0u32;
    let mut read_count = 0u32;
    let mut todowrite_count = 0u32;
    let mut recent_write_count = 0u32;
    let mut recent_edit_count = 0u32;
    let mut recent_read_count = 0u32;
    let mut recent_todowrite_count = 0u32;
    let mut pure_bash_streak = 0u32;
    let mut streak_open = true;
    for (i, tc) in tool_calls.iter().enumerate().rev() {
        let cat = classify_tool_call(&tc.name, tc.command.as_deref());
        if streak_open {
            if matches!(cat, ToolCategory::Other) {
                pure_bash_streak += 1;
            } else {
                streak_open = false;
            }
        }
        match cat {
            ToolCategory::Write => {
                write_count += 1;
                if i >= recent_start {
                    recent_write_count += 1;
                }
            }
            ToolCategory::Edit => {
                edit_count += 1;
                if i >= recent_start {
                    recent_edit_count += 1;
                }
            }
            ToolCategory::Read => {
                read_count += 1;
                if i >= recent_start {
                    recent_read_count += 1;
                }
            }
            ToolCategory::Plan => {
                todowrite_count += 1;
                if i >= recent_start {
                    recent_todowrite_count += 1;
                }
            }
            ToolCategory::Other => {}
        }
    }

    let tests_passed = detect_tests_passed(&tool_texts);

    ToolResultSignal {
        severity,
        patterns,
        no_error_streak,
        edit_count,
        write_count,
        read_count,
        todowrite_count,
        recent_edit_count,
        recent_write_count,
        recent_read_count,
        recent_todowrite_count,
        pure_bash_streak,
        tests_passed,
        turn_depth,
        prompt_char_count,
    }
}

// ─── pure helpers ─────────────────────────────────────────────────────────────

/// Normalise a JSON tool-result content value to a plain string.
fn content_to_text(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(s) => Some(s.clone()),
        Value::Array(blocks) => {
            let parts: Vec<&str> = blocks
                .iter()
                .filter_map(|b| {
                    b.as_object()
                        .filter(|o| o.get("type").and_then(Value::as_str) == Some("text"))
                        .and_then(|o| o.get("text"))
                        .and_then(Value::as_str)
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
        }
        _ => None,
    }
}

/// Character count of a user-message content value.
fn user_message_char_count(content: Option<&Value>) -> u32 {
    match content {
        Some(Value::String(s)) => s.len() as u32,
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| {
                b.as_object()
                    .filter(|o| o.get("type").and_then(Value::as_str) == Some("text"))
                    .and_then(|o| o.get("text"))
                    .and_then(Value::as_str)
            })
            .map(|s| s.len() as u32)
            .sum(),
        _ => 0,
    }
}

/// Match `text` against the error pattern table.
///
/// Returns `(max_severity, matched_pattern_names)`.
pub(crate) fn classify_text(text: &str) -> (f32, Vec<String>) {
    let lower = text.to_lowercase();
    let mut patterns = Vec::new();
    let mut severity: f32 = 0.0;
    for (name, sev, substrings) in ERROR_PATTERNS {
        if substrings.iter().any(|sub| lower.contains(sub)) {
            patterns.push(name.to_string());
            severity = severity.max(*sev);
        }
    }
    (severity, patterns)
}

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

// True iff `lower` contains a `NUMERIC_FAILURE_KEYWORDS` token preceded
// (modulo whitespace) by a nonzero integer. The "modulo whitespace" lets
// "1 failed", "1\nfailed", and "1  failed" all trip; the nonzero guard
// keeps cargo's "0 failed" / go's "0 errors" / pytest's "0 errors in"
// summaries from being misread as failures on a clean run.
fn has_nonzero_failure_count(lower: &str) -> bool {
    for kw in NUMERIC_FAILURE_KEYWORDS {
        let mut cursor = 0usize;
        while let Some(rel) = lower[cursor..].find(kw) {
            let kw_start = cursor + rel;
            let kw_end = kw_start + kw.len();
            // Word boundary AFTER the keyword — "errors" mid-word (e.g.
            // "errored") shouldn't count as a failure-count site.
            let boundary_after = lower[kw_end..]
                .chars()
                .next()
                .is_none_or(|c| !c.is_ascii_alphanumeric());
            if boundary_after {
                let prefix = &lower[..kw_start];
                let trimmed = prefix.trim_end_matches(|c: char| c.is_whitespace());
                let digits_rev: String = trimmed
                    .chars()
                    .rev()
                    .take_while(|c| c.is_ascii_digit())
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

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
    fn file_does_not_exist_is_hard() {
        // Claude Code Read-tool miss. Trace-mined addition (22 true / 2 false positives).
        let (sev, patterns) =
            classify_text("Error: File does not exist. Note: current working directory is /app.");
        assert_eq!(sev, HARD);
        assert!(patterns.contains(&"no_such_file".to_string()));
    }

    #[test]
    fn bare_does_not_exist_stays_clean() {
        // Precision guard: only the anchored "file does not exist" fires, so a bare
        // "does not exist" in prose or directory output must not trip a false error.
        let (sev, _) = classify_text("The directory does not exist yet, creating it now.");
        assert_eq!(sev, 0.0);
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

    #[test]
    fn tests_passed_detects_pytest_output() {
        assert!(detect_tests_passed(&[
            "====== 5 passed in 0.12s ======".to_string()
        ]));
    }

    #[test]
    fn tests_passed_ignores_partial_failures() {
        assert!(!detect_tests_passed(&[
            "2 failed, 5 passed in 0.56s".to_string()
        ]));
    }

    #[test]
    fn severity_is_windowed_over_recent_results() {
        // An error two results back, then two clean results.
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "tool", "tool_call_id": "1",
                 "content": "Traceback (most recent call last):\n  ValueError"},
                {"role": "tool", "tool_call_id": "2", "content": "ok"},
                {"role": "tool", "tool_call_id": "3", "content": "ok"},
            ]
        }));
        // window covers the error → severity persists (max over the window)
        assert_eq!(extract_tool_signals_with_window(&request, 3).severity, HARD);
        // window of 1 sees only the last (clean) result → severity has decayed out
        assert_eq!(extract_tool_signals_with_window(&request, 1).severity, 0.0);
    }

    #[test]
    fn extract_openai_chat_tool_results() {
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "user", "content": "do something"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Edit"}}]},
                {"role": "tool", "tool_call_id": "1", "content": "Traceback (most recent call last):\n  ValueError"},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(sig.severity, HARD);
        assert!(sig.patterns.contains(&"traceback".to_string()));
        assert_eq!(sig.edit_count, 1);
        assert_eq!(sig.turn_depth, 3);
    }

    #[test]
    fn extract_anthropic_tool_results() {
        let request = ChatRequest::anthropic(json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "1",
                     "content": "Traceback (most recent call last):\n  ValueError"}
                ]},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(sig.severity, HARD);
    }

    #[test]
    fn extract_responses_api_tool_results() {
        let request = ChatRequest::openai_responses(json!({
            "input": [
                {"type": "function_call", "name": "Write"},
                {"type": "function_call_output", "call_id": "1",
                 "output": "file written successfully"},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(sig.severity, 0.0);
        assert_eq!(sig.write_count, 1);
    }

    #[test]
    fn recent_window_counts_only_last_three_tool_calls() {
        // 5 writes + 1 edit at the end → recent window of 3 should see
        // 1 edit + 2 writes (not all 5 writes).
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Edit"}}]},
                {"role": "tool", "content": "ok"},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(sig.write_count, 5);
        assert_eq!(sig.edit_count, 1);
        assert_eq!(sig.recent_write_count, 2);
        assert_eq!(sig.recent_edit_count, 1);
    }

    #[test]
    fn recent_window_size_is_caller_overridable() {
        // Same six tool calls (1 edit at the end, 5 writes before).
        // With recent_window=3 → recent_writes=2, recent_edits=1.
        // With recent_window=6 → recent_writes=5, recent_edits=1 (all calls).
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Edit"}}]},
                {"role": "tool", "content": "ok"},
            ]
        }));
        let narrow = extract_tool_signals_with_window(&request, 3);
        assert_eq!(narrow.recent_write_count, 2);
        assert_eq!(narrow.recent_edit_count, 1);

        let wide = extract_tool_signals_with_window(&request, 6);
        assert_eq!(wide.recent_write_count, 5);
        assert_eq!(wide.recent_edit_count, 1);
    }

    #[test]
    fn bash_heredoc_counts_as_write() {
        // Claude Code's pattern on TB 2.0 — write a scratch file via heredoc.
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "assistant", "tool_calls": [{
                    "function": {
                        "name": "Bash",
                        "arguments": "{\"command\": \"cat > /tmp/test.py <<'EOF'\\nprint(1)\\nEOF\"}"
                    }
                }]},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(
            sig.write_count, 1,
            "Bash heredoc should bucket into write_count"
        );
        assert_eq!(sig.edit_count, 0);
    }

    #[test]
    fn bash_sed_inplace_counts_as_edit() {
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "assistant", "tool_calls": [{
                    "function": {
                        "name": "Bash",
                        "arguments": "{\"command\": \"sed -i 's/foo/bar/g' /app/file.py\"}"
                    }
                }]},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(
            sig.edit_count, 1,
            "Bash sed -i should bucket into edit_count"
        );
        assert_eq!(sig.write_count, 0);
    }

    #[test]
    fn bash_non_mutating_does_not_count() {
        // ls, cat, grep — should not increment either counter.
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "assistant", "tool_calls": [{
                    "function": {"name": "Bash", "arguments": "{\"command\": \"ls -la /app\"}"}
                }]},
                {"role": "assistant", "tool_calls": [{
                    "function": {"name": "Bash", "arguments": "{\"command\": \"cat /app/main.py\"}"}
                }]},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(sig.write_count, 0);
        assert_eq!(sig.edit_count, 0);
    }

    #[test]
    fn tests_passed_detects_pytest_with_failure_block() {
        // Mixed pytest run: 2 failed + 5 passed → NOT considered tests_passed.
        assert!(!detect_tests_passed(&[
            "2 failed, 5 passed in 0.56s".to_string()
        ]));
    }

    #[test]
    fn tests_passed_accepts_cargo_clean_summary() {
        // Cargo's clean-run summary contains "0 failed" — must not trip the
        // failure list (regression: previously substring-matched "failed").
        assert!(detect_tests_passed(&[
            "running 3 tests\ntest result: ok. 3 passed; 0 failed; 0 ignored".to_string()
        ]));
    }

    #[test]
    fn tests_passed_rejects_cargo_real_failure() {
        // Cargo's actual-failure summary: nonzero count before "failed".
        assert!(!detect_tests_passed(&[
            "running 3 tests\ntest result: FAILED. 2 passed; 1 failed; 0 ignored".to_string()
        ]));
    }

    #[test]
    fn tests_passed_accepts_go_clean_summary() {
        // Go test's clean-run "0 errors" must not trip (regression).
        assert!(detect_tests_passed(&[
            "ok  github.com/foo/bar\t0.012s (5 passed, 0 errors)".to_string()
        ]));
    }

    #[test]
    fn tests_passed_accepts_pytest_zero_errors() {
        // Pytest long-form: "0 errors in 0.3s" on a clean run.
        assert!(detect_tests_passed(&[
            "5 passed, 0 errors in 0.30s".to_string()
        ]));
    }

    #[test]
    fn tests_passed_detects_diy_checkmark() {
        assert!(detect_tests_passed(&["✓ all checks passed".to_string()]));
    }

    #[test]
    fn anthropic_bash_heredoc_extracts_command() {
        // Anthropic format: tool_use.input is an object, not a JSON string.
        let request = ChatRequest::anthropic(json!({
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "name": "Bash",
                     "input": {"command": "cat > /tmp/foo.txt << 'EOF'\nhi\nEOF"}}
                ]},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(
            sig.write_count, 1,
            "Anthropic Bash heredoc must also be detected"
        );
    }

    #[test]
    fn recent_window_falls_back_to_full_history_when_short() {
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(sig.recent_write_count, 1);
        assert_eq!(sig.recent_edit_count, 0);
    }

    #[test]
    fn clean_tool_result_has_zero_severity_and_non_empty_streak() {
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "tool", "tool_call_id": "1", "content": "output ok"},
                {"role": "tool", "tool_call_id": "2", "content": "another ok"},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(sig.severity, 0.0);
        assert_eq!(sig.no_error_streak, 2);
    }

    // ─── asymmetric-signal extensions ────────────────────────────────────

    #[test]
    fn todowrite_classifies_as_plan() {
        assert_eq!(classify_tool_call("TodoWrite", None), ToolCategory::Plan);
        assert_eq!(classify_tool_call("todo_write", None), ToolCategory::Plan);
    }

    #[test]
    fn codex_update_plan_classifies_as_plan() {
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
            ToolCategory::Read,
        );
        // shell_command without matching patterns -> Other.
        assert_eq!(
            classify_tool_call("shell_command", Some("./run_tests.sh")),
            ToolCategory::Other,
        );
    }

    #[test]
    fn read_tool_classifies_as_read() {
        assert_eq!(classify_tool_call("Read", None), ToolCategory::Read);
        assert_eq!(classify_tool_call("View", None), ToolCategory::Read);
    }

    #[test]
    fn bash_read_patterns_classify_as_read() {
        let cases = [
            "cat /etc/passwd",
            "grep foo bar.txt",
            "ls /app",
            "find . -name '*.py'",
        ];
        for cmd in cases {
            assert_eq!(
                classify_tool_call("Bash", Some(cmd)),
                ToolCategory::Read,
                "expected Read for {cmd}"
            );
        }
    }

    #[test]
    fn bash_write_precedence_over_read() {
        // `cat /file > out` contains both `cat /` (read) and ` > ` (write);
        // write redirection must win.
        assert_eq!(
            classify_tool_call("Bash", Some("cat /etc/hosts > /tmp/out")),
            ToolCategory::Write,
        );
    }

    #[test]
    fn pure_bash_streak_counts_trailing_other() {
        // 5 trailing non-classified Bash calls → streak == 5.
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "Bash",
                    "arguments": "{\"command\": \"make\"}"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Bash",
                    "arguments": "{\"command\": \"./configure\"}"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Bash",
                    "arguments": "{\"command\": \"make install\"}"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Bash",
                    "arguments": "{\"command\": \"./run.sh\"}"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Bash",
                    "arguments": "{\"command\": \"./test\"}"}}]},
                {"role": "tool", "content": "ok"},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(sig.pure_bash_streak, 5);
        assert_eq!(sig.write_count, 0);
        assert_eq!(sig.read_count, 0);
    }

    #[test]
    fn pure_bash_streak_resets_on_write() {
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "Bash",
                    "arguments": "{\"command\": \"make\"}"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Write"}}]},
                {"role": "tool", "content": "ok"},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(sig.pure_bash_streak, 0);
        assert_eq!(sig.write_count, 1);
    }

    #[test]
    fn recent_window_tracks_todowrite_and_read() {
        // Final 3 tool calls: TodoWrite, Read, TodoWrite.
        let request = ChatRequest::openai_chat(json!({
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "Bash",
                    "arguments": "{\"command\": \"make\"}"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "TodoWrite"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "Read"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "TodoWrite"}}]},
                {"role": "tool", "content": "ok"},
            ]
        }));
        let sig = extract_tool_signals(&request);
        assert_eq!(sig.todowrite_count, 2);
        assert_eq!(sig.recent_todowrite_count, 2);
        assert_eq!(sig.read_count, 1);
        assert_eq!(sig.recent_read_count, 1);
    }
}
