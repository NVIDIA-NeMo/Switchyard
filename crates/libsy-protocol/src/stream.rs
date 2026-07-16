// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming half of the neutral IR: incremental response chunks ([`LlmResponseChunk`])
//! and the streamed response ([`LlmResponse`]) that carries either a live stream of them
//! or the terminal [`AggLlmResponse`].

use std::collections::BTreeMap;
use std::error::Error;
use std::pin::Pin;

use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm::{AggLlmResponse, ContentBlock, ResponseOutput, Role, StopReason, ToolCall, Usage};

/// Boxed, thread-safe error carried by a stream item.
type BoxError = Box<dyn Error + Send + Sync>;

/// A boxed, `Send` stream of [`LlmResponseChunk`]s — the token-by-token output of a
/// streaming backend. Each item may fail independently mid-stream.
pub type LlmResponseStream = Pin<Box<dyn Stream<Item = Result<LlmResponseChunk, BoxError>> + Send>>;

/// A model response: either a live [`Stream`](LlmResponse::Stream) of chunks or the
/// terminal buffered [`Agg`](LlmResponse::Agg)regate.
///
/// Not `Clone` — the `Stream` variant owns a single-consumption stream. A buffered
/// backend returns `Agg` directly; a streaming one returns `Stream` and the consumer
/// drives it, folding to an [`AggLlmResponse`] when it needs the whole response.
pub enum LlmResponse {
    Stream(LlmResponseStream),
    Agg(AggLlmResponse),
}

impl LlmResponse {
    /// Borrow the aggregate; `None` while this is still a stream.
    pub fn as_agg(&self) -> Option<&AggLlmResponse> {
        match self {
            LlmResponse::Agg(agg) => Some(agg),
            LlmResponse::Stream(_) => None,
        }
    }

    /// Reduce to the buffered aggregate: return an `Agg` unchanged, or drive a `Stream`
    /// to completion, folding its chunks into an [`AggLlmResponse`] via
    /// [`ResponseAccumulator`]. A stream item error, or an in-band
    /// [`LlmResponseChunk::Error`], aborts with `Err`.
    pub async fn into_agg(self) -> Result<AggLlmResponse, BoxError> {
        match self {
            LlmResponse::Agg(agg) => Ok(agg),
            LlmResponse::Stream(mut stream) => {
                let mut accumulator = ResponseAccumulator::new();
                while let Some(item) = stream.next().await {
                    match item? {
                        LlmResponseChunk::Error { message } => return Err(message.into()),
                        chunk => accumulator.push(chunk),
                    }
                }
                Ok(accumulator.finish())
            }
        }
    }
}

/// One provider-neutral streaming event — the normalized counterpart to
/// [`AggLlmResponse`](crate::AggLlmResponse), sitting between stream decoders and
/// encoders. `switchyard-translation` re-exports it as `ConversationStreamEvent`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LlmResponseChunk {
    MessageStart {
        id: Option<String>,
        model: Option<String>,
    },
    TextDelta {
        index: usize,
        text: String,
    },
    ReasoningDelta {
        index: usize,
        text: String,
    },
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: Option<String>,
    },
    Usage(Usage),
    MessageStop {
        reason: Option<String>,
    },
    Error {
        message: String,
    },
}

/// Folds a sequence of [`LlmResponseChunk`]s into the terminal [`AggLlmResponse`].
///
/// Text and reasoning deltas concatenate; tool-call deltas assemble by index (name,
/// id, and a growing arguments string parsed as JSON at the end); `MessageStart`,
/// `Usage`, and `MessageStop` set the corresponding fields. `Error` chunks are
/// ignored here — a driver consuming the stream is expected to surface them.
///
/// Drive it by `push`-ing each chunk in order, then call [`finish`](Self::finish).
#[derive(Default)]
pub struct ResponseAccumulator {
    id: Option<String>,
    model: Option<String>,
    text: String,
    reasoning: Option<String>,
    tool_calls: BTreeMap<usize, PartialToolCall>,
    usage: Usage,
    stop_reason: Option<StopReason>,
}

/// A tool call being assembled from streamed [`LlmResponseChunk::ToolCallDelta`]s.
#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl ResponseAccumulator {
    /// A fresh accumulator with no chunks applied.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one chunk. Later `MessageStart`/`Usage`/`MessageStop` fields overwrite
    /// earlier ones; text, reasoning, and tool-call arguments append.
    pub fn push(&mut self, chunk: LlmResponseChunk) {
        match chunk {
            LlmResponseChunk::MessageStart { id, model } => {
                if id.is_some() {
                    self.id = id;
                }
                if model.is_some() {
                    self.model = model;
                }
            }
            LlmResponseChunk::TextDelta { text, .. } => self.text.push_str(&text),
            LlmResponseChunk::ReasoningDelta { text, .. } => {
                self.reasoning
                    .get_or_insert_with(String::new)
                    .push_str(&text);
            }
            LlmResponseChunk::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                let call = self.tool_calls.entry(index).or_default();
                if id.is_some() {
                    call.id = id;
                }
                if name.is_some() {
                    call.name = name;
                }
                if let Some(delta) = arguments_delta {
                    call.arguments.push_str(&delta);
                }
            }
            LlmResponseChunk::Usage(usage) => self.usage = usage,
            LlmResponseChunk::MessageStop { reason } => {
                self.stop_reason = Some(stop_reason_from_str(reason.as_deref()));
            }
            LlmResponseChunk::Error { .. } => {}
        }
    }

    /// Build the buffered response. Content is ordered reasoning, then text, then
    /// tool calls (by ascending delta index) — a single assistant output.
    pub fn finish(self) -> AggLlmResponse {
        let mut content = Vec::new();
        if let Some(reasoning) = self.reasoning {
            content.push(ContentBlock::Reasoning {
                text: reasoning,
                signature: None,
            });
        }
        if !self.text.is_empty() {
            content.push(ContentBlock::Text { text: self.text });
        }
        for call in self.tool_calls.into_values() {
            content.push(ContentBlock::ToolCall(ToolCall {
                id: call.id.unwrap_or_default(),
                name: call.name.unwrap_or_default(),
                arguments: parse_tool_arguments(&call.arguments),
            }));
        }
        AggLlmResponse {
            id: self.id,
            model: self.model,
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content,
                stop_reason: self.stop_reason,
            }],
            usage: self.usage,
            ..AggLlmResponse::default()
        }
    }
}

/// Parse an assembled tool-call arguments string as JSON, falling back to a JSON
/// string when it is not valid JSON and to an empty object when it is empty.
fn parse_tool_arguments(arguments: &str) -> Value {
    if arguments.is_empty() {
        return Value::Object(serde_json::Map::new());
    }
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

/// Map a provider stop-reason string (as carried by [`LlmResponseChunk::MessageStop`])
/// to a normalized [`StopReason`], covering the common OpenAI and Anthropic spellings.
fn stop_reason_from_str(reason: Option<&str>) -> StopReason {
    match reason {
        Some("length" | "max_tokens") => StopReason::MaxTokens,
        Some("tool_calls" | "function_call" | "tool_use") => StopReason::ToolUse,
        Some("content_filter") => StopReason::ContentFilter,
        Some("stop" | "end_turn" | "stop_sequence") | None => StopReason::EndTurn,
        Some(_) => StopReason::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fold(chunks: Vec<LlmResponseChunk>) -> AggLlmResponse {
        let mut accumulator = ResponseAccumulator::new();
        for chunk in chunks {
            accumulator.push(chunk);
        }
        accumulator.finish()
    }

    #[test]
    fn folds_text_usage_and_stop_reason() {
        let agg = fold(vec![
            LlmResponseChunk::MessageStart {
                id: Some("id1".to_string()),
                model: Some("m".to_string()),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "Hel".to_string(),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "lo".to_string(),
            },
            LlmResponseChunk::Usage(Usage {
                output_tokens: Some(2),
                ..Usage::default()
            }),
            LlmResponseChunk::MessageStop {
                reason: Some("length".to_string()),
            },
        ]);
        assert_eq!(agg.id.as_deref(), Some("id1"));
        assert_eq!(agg.model.as_deref(), Some("m"));
        assert_eq!(agg.usage.output_tokens, Some(2));
        assert_eq!(agg.outputs[0].stop_reason, Some(StopReason::MaxTokens));
        assert_eq!(
            agg.outputs[0].content,
            vec![ContentBlock::Text {
                text: "Hello".to_string()
            }]
        );
    }

    #[test]
    fn assembles_tool_calls_by_index() {
        // id/name arrive once, arguments stream across deltas and parse as JSON.
        let agg = fold(vec![
            LlmResponseChunk::ToolCallDelta {
                index: 0,
                id: Some("call_1".to_string()),
                name: Some("lookup".to_string()),
                arguments_delta: Some("{\"q\":".to_string()),
            },
            LlmResponseChunk::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: Some("\"rust\"}".to_string()),
            },
            LlmResponseChunk::MessageStop {
                reason: Some("tool_calls".to_string()),
            },
        ]);
        assert_eq!(agg.outputs[0].stop_reason, Some(StopReason::ToolUse));
        assert_eq!(
            agg.outputs[0].content,
            vec![ContentBlock::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "lookup".to_string(),
                arguments: json!({"q": "rust"}),
            })]
        );
    }

    #[test]
    fn reasoning_precedes_text_in_content() {
        let agg = fold(vec![
            LlmResponseChunk::ReasoningDelta {
                index: 0,
                text: "think".to_string(),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "answer".to_string(),
            },
        ]);
        assert_eq!(
            agg.outputs[0].content,
            vec![
                ContentBlock::Reasoning {
                    text: "think".to_string(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "answer".to_string(),
                },
            ]
        );
    }
}
