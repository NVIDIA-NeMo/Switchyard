// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming half of the neutral IR: incremental response chunks ([`LlmResponseChunk`])
//! and the streamed response ([`LlmResponse`]) that carries either a live stream of them
//! or the terminal [`AggLlmResponse`].

use std::collections::BTreeMap;
use std::pin::Pin;

use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    LlmClientError,
    llm::{AggLlmResponse, ContentBlock, ResponseOutput, Role, StopReason, ToolCall, Usage},
};

/// Status reported for an upstream error delivered inside a streaming body. The
/// upstream already sent a success status line before failing, so there is no real
/// code to propagate; 502 matches how a failed upstream call surfaces elsewhere.
const MID_STREAM_UPSTREAM_STATUS: u16 = 502;

/// A boxed, `Send` stream of [`LlmResponseChunk`]s — the token-by-token output of a
/// streaming backend. Each item may fail independently mid-stream.
pub type LlmResponseStream =
    Pin<Box<dyn Stream<Item = Result<LlmResponseChunk, LlmClientError>> + Send>>;

/// A model response: either a live [`Stream`](LlmResponse::Stream) of chunks or a
/// terminal buffered [`LlmResponse::Agg`] response.
///
/// Not `Clone` — the `Stream` variant owns a single-consumption stream. A buffered
/// backend returns `Agg` directly; a streaming one returns `Stream` and the consumer
/// drives it, folding to an [`AggLlmResponse`] when it needs the whole response.
pub enum LlmResponse {
    /// Live, single-consumption response stream.
    Stream(LlmResponseStream),
    /// Fully buffered terminal response.
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
    /// [`ResponseAccumulator`]. A stream item error aborts with `Err`, as does an
    /// in-band [`LlmResponseChunk::DecodeError`] (as `ResponseTranslation`) or
    /// [`LlmResponseChunk::StreamError`] (as `UpstreamHttp`).
    pub async fn into_agg(self) -> Result<AggLlmResponse, LlmClientError> {
        match self {
            LlmResponse::Agg(agg) => Ok(agg),
            LlmResponse::Stream(mut stream) => {
                let mut accumulator = ResponseAccumulator::new();
                while let Some(item) = stream.next().await {
                    match item? {
                        LlmResponseChunk::DecodeError { message } => {
                            return Err(LlmClientError::ResponseTranslation(message));
                        }
                        LlmResponseChunk::StreamError { message } => {
                            // The upstream reported the failure inside the response body, so
                            // there is no real status line to carry; 502 stands in for "the
                            // upstream failed" the same way a failed non-streaming call would.
                            return Err(LlmClientError::UpstreamHttp {
                                status: MID_STREAM_UPSTREAM_STATUS,
                                body: message,
                            });
                        }
                        chunk => accumulator.push(chunk),
                    }
                }
                Ok(accumulator.finish())
            }
        }
    }

    /// Returns the model recorded by a buffered response.
    ///
    /// Returns `None` for a live stream because reading its model-bearing
    /// [`LlmResponseChunk::MessageStart`] event would consume the stream.
    pub fn selected_model(&self) -> Option<&str> {
        match self {
            LlmResponse::Agg(agg) => agg.model.as_deref(),
            // TODO: How do we get the model name on a stream?
            LlmResponse::Stream(_) => None,
        }
    }
}

impl AggLlmResponse {
    /// Converts a fully-buffered response into a synthetic chunk stream.
    ///
    /// Useful when a caller had to buffer the response (e.g. to judge it) but the
    /// downstream expects `LlmResponse::Stream` — for instance when `stream: true` was
    /// requested and the algorithm had to aggregate before it could return.
    ///
    /// This conversion is lossy: only text, reasoning, and tool-call content has
    /// a synthetic chunk representation. Refusals, tool results, media, files,
    /// unknown blocks, response extensions, and preservation metadata are omitted.
    pub fn into_stream(self) -> LlmResponseStream {
        let mut chunks: Vec<LlmResponseChunk> = Vec::new();
        chunks.push(LlmResponseChunk::MessageStart {
            id: self.id,
            model: self.model,
        });
        let mut tool_call_index = 0usize;
        for (output_index, output) in self.outputs.into_iter().enumerate() {
            for block in output.content {
                match block {
                    ContentBlock::Text { text } => {
                        chunks.push(LlmResponseChunk::TextDelta {
                            index: output_index,
                            text,
                        });
                    }
                    ContentBlock::Reasoning { text, .. } => {
                        chunks.push(LlmResponseChunk::ReasoningDelta {
                            index: output_index,
                            text,
                        });
                    }
                    ContentBlock::ToolCall(tool) => {
                        let args = serde_json::to_string(&tool.arguments).unwrap_or_default();
                        chunks.push(LlmResponseChunk::ToolCallDelta {
                            index: tool_call_index,
                            id: Some(tool.id),
                            name: Some(tool.name),
                            arguments_delta: Some(args),
                        });
                        tool_call_index += 1;
                    }
                    // Other variants (Image, ToolResult, Refusal, etc.) don't have
                    // a streaming chunk representation and don't appear in assistant outputs.
                    _ => {}
                }
            }
            chunks.push(LlmResponseChunk::MessageStop {
                reason: output.stop_reason.and_then(|r| {
                    serde_json::to_value(r)
                        .ok()
                        .and_then(|v| v.as_str().map(String::from))
                }),
            });
        }
        chunks.push(LlmResponseChunk::Usage(self.usage));
        Box::pin(futures::stream::iter(chunks.into_iter().map(Ok)))
    }
}

/// One provider-neutral streaming event — the normalized counterpart to
/// [`AggLlmResponse`], sitting between stream decoders and encoders.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LlmResponseChunk {
    /// Starts a response message.
    MessageStart {
        /// Provider response identifier.
        id: Option<String>,
        /// Model reported by the provider.
        model: Option<String>,
    },
    /// Adds text to one output index.
    TextDelta {
        /// Provider output index.
        index: usize,
        /// Text fragment.
        text: String,
    },
    /// Adds reasoning text to one output index.
    ReasoningDelta {
        /// Provider output index.
        index: usize,
        /// Reasoning fragment.
        text: String,
    },
    /// Adds or updates a tool call at one index.
    ToolCallDelta {
        /// Tool-call index within the response.
        index: usize,
        /// Tool-call identifier, normally supplied by the first delta.
        id: Option<String>,
        /// Tool name, normally supplied by the first delta.
        name: Option<String>,
        /// Fragment of the serialized tool arguments.
        arguments_delta: Option<String>,
    },
    /// Reports token usage, normally near the end of the stream.
    Usage(Usage),
    /// Ends a response message.
    MessageStop {
        /// Provider stop-reason string before normalization.
        reason: Option<String>,
    },
    /// Reports that an inbound stream event could not be decoded.
    DecodeError {
        /// Human-readable decoding failure.
        message: String,
    },
    /// Reports an upstream failure delivered inside an otherwise successful stream.
    StreamError {
        /// Human-readable upstream failure.
        message: String,
    },
}

/// Folds a sequence of [`LlmResponseChunk`]s into the terminal [`AggLlmResponse`].
///
/// Text and reasoning deltas concatenate; tool-call deltas assemble by index (name,
/// id, and a growing arguments string parsed as JSON at the end); `MessageStart`,
/// `Usage`, and `MessageStop` set the corresponding fields.
/// [`DecodeError`](LlmResponseChunk::DecodeError) and
/// [`StreamError`](LlmResponseChunk::StreamError) chunks are ignored here; use
/// [`LlmResponse::into_agg`] when they must become errors.
///
/// Folding is lossy for multiple outputs: text and reasoning indices are ignored,
/// and [`finish`](Self::finish) produces one assistant output. Prefer
/// [`LlmResponse::into_agg`] when stream errors must be surfaced.
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
            LlmResponseChunk::DecodeError { .. } | LlmResponseChunk::StreamError { .. } => {}
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
    use futures::executor::block_on;
    use futures::stream;

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

    #[test]
    fn into_stream_round_trips_through_into_agg() {
        let original = AggLlmResponse {
            id: Some("id1".to_string()),
            model: Some("m".to_string()),
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "hello".to_string(),
                }],
                stop_reason: Some(StopReason::EndTurn),
            }],
            usage: Usage {
                output_tokens: Some(3),
                ..Usage::default()
            },
            ..AggLlmResponse::default()
        };
        let stream = LlmResponse::Stream(original.clone().into_stream());
        let recovered = block_on(stream.into_agg()).expect("into_agg failed");
        assert_eq!(recovered.id, original.id);
        assert_eq!(recovered.model, original.model);
        assert_eq!(recovered.usage.output_tokens, original.usage.output_tokens);
        assert_eq!(
            recovered.outputs[0].stop_reason,
            original.outputs[0].stop_reason
        );
        assert_eq!(recovered.outputs[0].content, original.outputs[0].content);
    }

    #[test]
    fn into_agg_preserves_stream_item_error() {
        let response = LlmResponse::Stream(Box::pin(stream::once(async {
            Err(LlmClientError::Timeout {
                source: Box::new(std::io::Error::other("timed out")),
            })
        })));

        let Err(error) = block_on(response.into_agg()) else {
            panic!("expected stream aggregation to fail");
        };
        assert!(matches!(error, LlmClientError::Timeout { .. }));
    }
}
