// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared fixtures for the tier-routing algorithms.
//!
//! [`Recorder`] answers [`JUDGE`] with a settable verdict, so a cascade that
//! consults a capability judge runs without a model.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;
use switchyard_protocol::{
    ContentBlock, LlmRequest, Message, Metadata, ModelId, Request, Role, ToolCall, ToolResult,
    WireFormat,
};

use crate::core::testing::{Serve, reply};

/// Target the fake judge answers on.
pub(crate) const JUDGE: &str = "judge";

#[derive(Clone, Debug)]
pub(crate) struct Call {
    pub target: String,
    pub messages: Vec<String>,
}

/// Records what each target receives.
#[derive(Default)]
pub(crate) struct Recorder {
    pub calls: Mutex<Vec<Call>>,
    pub judge_p_solve: Mutex<f64>,
}

impl Recorder {
    pub fn routed(&self) -> Vec<Call> {
        self.filter_calls(|target| target != JUDGE)
    }

    pub fn judge_calls(&self) -> usize {
        self.filter_calls(|target| target == JUDGE).len()
    }

    fn filter_calls(&self, keep: impl Fn(&str) -> bool) -> Vec<Call> {
        self.calls
            .lock()
            .iter()
            .filter(|call| keep(&call.target))
            .cloned()
            .collect()
    }

    /// Serves every call, recording it. The judge target gets a structured verdict
    /// back so the fallback classifier has an answer without a real model.
    pub fn serve(self: &Arc<Self>) -> impl Serve {
        let recorder = Arc::clone(self);
        move |target: ModelId, request: Request| {
            let recorder = Arc::clone(&recorder);
            async move {
                let target = target.to_string();
                recorder.calls.lock().push(Call {
                    target: target.clone(),
                    messages: request
                        .llm_request
                        .messages
                        .iter()
                        .filter_map(|message| message.text_content("|"))
                        .collect(),
                });
                let completion = if target == JUDGE {
                    let p_solve = *recorder.judge_p_solve.lock();
                    format!(
                        r#"{{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":{p_solve}}}"#
                    )
                } else {
                    target
                };
                Ok(reply(completion))
            }
        }
    }
}

/// One tool step: a task, a `Bash` call, and its result.
pub(crate) fn turn_request(failed: bool) -> Request {
    let content = if failed {
        "fatal runtime error: out of memory"
    } else {
        "ok"
    };
    Request {
        llm_request: LlmRequest {
            model: Some("auto".to_string()),
            messages: vec![
                Message::text(Role::User, "fix the build"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall(ToolCall {
                        id: "call_1".to_string(),
                        name: "Bash".to_string(),
                        arguments: json!({"command": "cargo test"}),
                    })],
                },
                Message {
                    role: Role::Tool,
                    content: vec![ContentBlock::ToolResult(ToolResult {
                        tool_call_id: "call_1".to_string(),
                        content: vec![ContentBlock::Text {
                            text: content.to_string(),
                        }],
                        is_error: Some(failed),
                    })],
                },
            ],
            ..LlmRequest::default()
        },
        raw_request: Some(json!({
            "model": "auto",
            "messages": [
                {"role": "user", "content": "fix the build"},
                {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "Bash", "arguments": "{\"command\": \"cargo test\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": content},
            ],
        })),
        metadata: Some(Metadata {
            wire_format: Some(WireFormat::OpenAiChat),
            session_id: Some("session-1".to_string()),
            ..Default::default()
        }),
    }
}
