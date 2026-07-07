// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider usage extraction for buffered and streaming responses.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use switchyard_core::StreamEvent;

/// Normalized token usage counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub reasoning_tokens: u64,
    /// Set by the stats response processor, not by provider usage extraction.
    #[serde(default)]
    pub cacheable_prompt_tokens: u64,
}

impl TokenUsage {
    /// Returns whether all counters are zero.
    pub fn is_zero(self) -> bool {
        self.prompt_tokens == 0
            && self.completion_tokens == 0
            && self.cached_tokens == 0
            && self.cache_creation_tokens == 0
            && self.reasoning_tokens == 0
    }
}

/// Extracts usage from a buffered response body.
pub fn usage_from_body(body: &Value) -> TokenUsage {
    body.get("usage")
        .and_then(usage_from_candidate)
        .unwrap_or_default()
}

/// Extracts usage from a buffered Gemini generateContent response body.
pub fn gemini_usage_from_body(body: &Value) -> TokenUsage {
    body.get("usageMetadata")
        .map(token_usage_from_gemini_metadata)
        .unwrap_or_default()
}

/// Extracts OpenAI Chat streaming usage from an event.
pub fn openai_chat_usage_from_stream_event(event: &StreamEvent) -> Option<TokenUsage> {
    let StreamEvent::Json(value) = event else {
        return None;
    };
    value.get("usage").and_then(usage_from_candidate)
}

/// Extracts OpenAI Responses streaming usage from an event.
///
/// Fidelity-preserving backends yield raw SSE frame *strings*
/// (`StreamEvent::Text`) instead of decoded JSON events; those frames are
/// parsed here so usage accounting survives verbatim passthrough.
pub fn openai_responses_usage_from_stream_event(event: &StreamEvent) -> Option<TokenUsage> {
    match event {
        StreamEvent::Json(value) => usage_from_responses_value(value),
        StreamEvent::Text(text) => sse_data_payloads(text)
            .iter()
            .find_map(usage_from_responses_value),
    }
}

/// Reads `response.usage` from one decoded Responses stream event.
fn usage_from_responses_value(value: &Value) -> Option<TokenUsage> {
    value
        .get("response")
        .and_then(|response| response.get("usage"))
        .and_then(usage_from_candidate)
}

/// Parses the JSON `data:` payload(s) out of a raw SSE frame string.
///
/// Per the SSE contract, a frame's `data:` lines join with newlines to form
/// one payload; a single leading space after the colon is stripped. Comment
/// frames, `[DONE]` sentinels, and non-JSON payloads yield nothing.
fn sse_data_payloads(text: &str) -> Vec<Value> {
    let mut payloads = Vec::new();
    for block in text.split("\n\n") {
        let data_lines: Vec<&str> = block
            .split('\n')
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|value| value.strip_prefix(' ').unwrap_or(value))
            .collect();
        if data_lines.is_empty() {
            continue;
        }
        let data = data_lines.join("\n");
        if data.trim() == "[DONE]" {
            continue;
        }
        if let Ok(parsed) = serde_json::from_str::<Value>(&data) {
            payloads.push(parsed);
        }
    }
    payloads
}

/// Accumulates Anthropic streaming usage and commits once at `message_stop`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnthropicStreamUsage {
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: u64,
    cache_creation_input_tokens: u64,
    saw_usage: bool,
    committed: bool,
}

impl AnthropicStreamUsage {
    /// Observes one stream event and returns usage exactly once at `message_stop`.
    /// A stop event before any usage frame is a known no-op, matching Python stream taps.
    pub fn observe(&mut self, event: &StreamEvent) -> Option<TokenUsage> {
        let StreamEvent::Json(value) = event else {
            return None;
        };
        match value.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(usage) = value
                    .get("message")
                    .and_then(|message| message.get("usage"))
                {
                    self.merge(usage);
                }
                None
            }
            Some("message_delta") => {
                if let Some(usage) = value
                    .get("usage")
                    .or_else(|| value.get("delta").and_then(|delta| delta.get("usage")))
                {
                    self.merge(usage);
                }
                None
            }
            Some("message_stop") if self.saw_usage && !self.committed => {
                self.committed = true;
                Some(TokenUsage {
                    prompt_tokens: self
                        .input_tokens
                        .saturating_add(self.cache_read_input_tokens)
                        .saturating_add(self.cache_creation_input_tokens),
                    completion_tokens: self.output_tokens,
                    cached_tokens: self.cache_read_input_tokens,
                    cache_creation_tokens: self.cache_creation_input_tokens,
                    reasoning_tokens: 0,
                    cacheable_prompt_tokens: 0,
                })
            }
            _ => None,
        }
    }

    fn merge(&mut self, usage: &Value) {
        if !usage.is_object() {
            return;
        }
        self.saw_usage = true;
        if let Some(value) = int_field(usage, "input_tokens") {
            self.input_tokens = value;
        }
        if let Some(value) = int_field(usage, "output_tokens") {
            self.output_tokens = value;
        }
        if let Some(value) = int_field(usage, "cache_read_input_tokens") {
            self.cache_read_input_tokens = value;
        }
        if let Some(value) = int_field(usage, "cache_creation_input_tokens") {
            self.cache_creation_input_tokens = value;
        }
    }
}

/// Accumulates Gemini streaming usage and commits once on the terminal chunk.
///
/// Gemini streams have no explicit stop event; the final chunk carries both
/// `candidates[].finishReason` and the authoritative `usageMetadata`. A
/// finish-reason chunk before any usage frame is a known no-op, matching the
/// Anthropic stream tap behavior.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeminiStreamUsage {
    prompt_tokens: u64,
    candidates_tokens: u64,
    thoughts_tokens: u64,
    cached_tokens: u64,
    saw_usage: bool,
    committed: bool,
}

impl GeminiStreamUsage {
    /// Observes one stream chunk and returns usage exactly once at the finish chunk.
    pub fn observe(&mut self, event: &StreamEvent) -> Option<TokenUsage> {
        let StreamEvent::Json(value) = event else {
            return None;
        };
        if let Some(usage) = value.get("usageMetadata") {
            self.merge(usage);
        }
        let finished = value
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
            .and_then(|candidate| candidate.get("finishReason"))
            .and_then(Value::as_str)
            .is_some();
        if finished && self.saw_usage && !self.committed {
            self.committed = true;
            return Some(TokenUsage {
                prompt_tokens: self.prompt_tokens,
                // Gemini bills thinking tokens as output but reports them
                // outside candidatesTokenCount; fold them back in.
                completion_tokens: self.candidates_tokens.saturating_add(self.thoughts_tokens),
                cached_tokens: self.cached_tokens,
                cache_creation_tokens: 0,
                reasoning_tokens: self.thoughts_tokens,
                cacheable_prompt_tokens: 0,
            });
        }
        None
    }

    fn merge(&mut self, usage: &Value) {
        if !usage.is_object() {
            return;
        }
        self.saw_usage = true;
        if let Some(value) = int_field(usage, "promptTokenCount") {
            self.prompt_tokens = value;
        }
        if let Some(value) = int_field(usage, "candidatesTokenCount") {
            self.candidates_tokens = value;
        }
        if let Some(value) = int_field(usage, "thoughtsTokenCount") {
            self.thoughts_tokens = value;
        }
        if let Some(value) = int_field(usage, "cachedContentTokenCount") {
            self.cached_tokens = value;
        }
    }
}

// Converts Gemini `usageMetadata` counters into normalized token usage.
fn token_usage_from_gemini_metadata(usage: &Value) -> TokenUsage {
    let candidates = int_field(usage, "candidatesTokenCount").unwrap_or(0);
    let thoughts = int_field(usage, "thoughtsTokenCount").unwrap_or(0);
    TokenUsage {
        prompt_tokens: int_field(usage, "promptTokenCount").unwrap_or(0),
        completion_tokens: candidates.saturating_add(thoughts),
        cached_tokens: int_field(usage, "cachedContentTokenCount").unwrap_or(0),
        cache_creation_tokens: 0,
        reasoning_tokens: thoughts,
        cacheable_prompt_tokens: 0,
    }
}

fn usage_from_candidate(usage: &Value) -> Option<TokenUsage> {
    usage.is_object().then(|| usage_from_value(usage))
}

fn usage_from_value(usage: &Value) -> TokenUsage {
    let completion_tokens = int_field(usage, "completion_tokens")
        .unwrap_or_else(|| int_field(usage, "output_tokens").unwrap_or(0));
    let mut output = TokenUsage {
        completion_tokens,
        ..TokenUsage::default()
    };

    if let Some(prompt_tokens) = int_field(usage, "prompt_tokens") {
        output.prompt_tokens = prompt_tokens;
        if let Some(details) = usage.get("prompt_tokens_details") {
            output.cached_tokens = int_field(details, "cached_tokens").unwrap_or(0);
            output.cache_creation_tokens = int_field(details, "cache_creation_tokens").unwrap_or(0);
        }
    } else {
        let base = int_field(usage, "input_tokens").unwrap_or(0);
        if let Some(details) = usage.get("input_tokens_details") {
            output.cached_tokens = int_field(details, "cached_tokens").unwrap_or(0);
        }
        let cache_read = int_field(usage, "cache_read_input_tokens").unwrap_or(0);
        let cache_creation = int_field(usage, "cache_creation_input_tokens").unwrap_or(0);
        if output.cached_tokens == 0 {
            output.cached_tokens = cache_read;
        }
        output.cache_creation_tokens = cache_creation;
        output.prompt_tokens = base
            .saturating_add(cache_read)
            .saturating_add(cache_creation);
    }

    output.reasoning_tokens = usage
        .get("completion_tokens_details")
        .and_then(|details| int_field(details, "reasoning_tokens"))
        .or_else(|| {
            usage
                .get("output_tokens_details")
                .and_then(|details| int_field(details, "reasoning_tokens"))
        })
        .unwrap_or(0);
    output
}

fn int_field(value: &Value, name: &str) -> Option<u64> {
    value.get(name).and_then(Value::as_u64)
}
