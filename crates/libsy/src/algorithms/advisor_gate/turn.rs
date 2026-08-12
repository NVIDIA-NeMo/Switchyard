// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One buffered executor turn: consuming it to completion, inspecting it,
//! and replaying it to the client verbatim.

use futures::StreamExt;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmClientError, LlmResponse, LlmResponseChunk,
    LlmResponseStreamEvent, Message, Metadata, Response, ResponseAccumulator, Role, StopReason,
};

use crate::{LibsyError, Result};

// ── Turn buffering and replay ───────────────────────────────────────────────

/// One fully generated executor turn held while the gate decides.
pub(super) struct GatedTurn {
    /// Buffered provider events for streamed turns, preservation included, so
    /// replay re-emits them verbatim (signed thinking and provider extensions
    /// survive; folding to an aggregate and re-synthesizing would drop them).
    pub(super) events: Option<Vec<LlmResponseStreamEvent>>,
    /// Folded view for detection, the review tail, the REDO echo, and
    /// discarded-turn usage. For buffered turns this is the original
    /// response, its own preservation intact.
    pub(super) agg: AggLlmResponse,
    pub(super) metadata: Option<Metadata>,
}

impl GatedTurn {
    /// Releases the turn to the client: streamed turns replay their buffered
    /// events verbatim, buffered turns return the original aggregate.
    pub(super) fn into_response(self) -> Response {
        let llm_response = match self.events {
            Some(events) => {
                LlmResponse::Stream(Box::pin(futures::stream::iter(events.into_iter().map(Ok))))
            }
            None => LlmResponse::Agg(self.agg),
        };
        Response {
            llm_response,
            metadata: self.metadata,
        }
    }
}

/// Consumes the executor response to completion. Mid-stream failures — item
/// errors and in-band error chunks — become typed client-call errors exactly
/// as [`LlmResponse::into_agg`] maps them; the client saw nothing yet, so the
/// turn fails whole.
pub(super) async fn buffer_turn(executor: &str, response: Response) -> Result<GatedTurn> {
    let metadata = response.metadata;
    match response.llm_response {
        LlmResponse::Agg(agg) => Ok(GatedTurn {
            events: None,
            agg,
            metadata,
        }),
        LlmResponse::Stream(mut stream) => {
            let mut events = Vec::new();
            let mut accumulator = ResponseAccumulator::new();
            while let Some(item) = stream.next().await {
                let event =
                    item.map_err(|source| LibsyError::client_call(executor.to_string(), source))?;
                for chunk in event.normalized() {
                    let failure = match chunk {
                        LlmResponseChunk::DecodeError { message } => {
                            Some(LlmClientError::ResponseTranslation(message.clone()))
                        }
                        LlmResponseChunk::StreamError { message } => {
                            Some(LlmClientError::UpstreamHttp {
                                status: 502,
                                body: message.clone(),
                            })
                        }
                        chunk => {
                            accumulator.push(chunk.clone());
                            None
                        }
                    };
                    if let Some(source) = failure {
                        return Err(LibsyError::client_call(executor.to_string(), source));
                    }
                }
                events.push(event);
            }
            Ok(GatedTurn {
                events: Some(events),
                agg: accumulator.finish(),
                metadata,
            })
        }
    }
}

// ── Detection over the folded turn ──────────────────────────────────────────

/// Whether the turn carries tool use on either signal: a `ToolUse` stop
/// reason, or any tool-call block (some OSS servers mislabel tool-call turns
/// as an ordinary stop, so block presence wins).
pub(super) fn has_tool_use(agg: &AggLlmResponse) -> bool {
    agg.outputs.iter().any(|output| {
        output.stop_reason == Some(StopReason::ToolUse)
            || output
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall(_)))
    })
}

/// The turn's visible text: all text blocks joined; empty means none.
pub(super) fn visible_text(agg: &AggLlmResponse) -> Option<String> {
    let text: Vec<&str> = agg
        .outputs
        .iter()
        .flat_map(|output| output.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if text.is_empty() {
        return None;
    }
    let joined = text.join("\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// The turn's internal reasoning, the review evidence of last resort.
pub(super) fn reasoning_text(agg: &AggLlmResponse) -> Option<String> {
    let text: Vec<&str> = agg
        .outputs
        .iter()
        .flat_map(|output| output.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if text.is_empty() {
        return None;
    }
    let joined = text.join("\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// Tool results carried by the conversation so far (both wires normalize
/// tool results into `ContentBlock::ToolResult`).
pub(super) fn count_tool_results(messages: &[Message]) -> u32 {
    let count = messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter(|block| matches!(block, ContentBlock::ToolResult(_)))
        .count();
    u32::try_from(count).unwrap_or(u32::MAX)
}

/// Assistant turns already in the request — the stall checkpoint's clock.
pub(super) fn assistant_turns(messages: &[Message]) -> u32 {
    let count = messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .count();
    u32::try_from(count).unwrap_or(u32::MAX)
}
