// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded input checks before semantic privacy assessment.

use switchyard_protocol::{ContentBlock, LlmRequest, Request, Tool, ToolChoice, WireFormat};

const MAX_SCAN_BYTES: usize = 1024 * 1024;
const MAX_SCAN_DEPTH: usize = 64;
const MAX_SCAN_STEPS: usize = 4096;

pub(super) fn validate_input(request: &Request) -> Result<(), &'static str> {
    Scan::default().request(&request.llm_request)
}

#[derive(Default)]
struct Scan {
    bytes: usize,
    steps: usize,
}

impl Scan {
    fn request(&mut self, request: &LlmRequest) -> Result<(), &'static str> {
        // Keep this exhaustive so each new protocol field gets an explicit privacy decision.
        let LlmRequest {
            model: _,
            instructions,
            messages,
            tools,
            tool_choice,
            sampling: _,
            output,
            reasoning,
            stream: _,
            extensions,
            preservation,
        } = request;
        for instruction in instructions {
            self.blocks(&instruction.content, 0)?;
        }
        for message in messages {
            self.blocks(&message.content, 0)?;
        }
        for tool in tools {
            let Tool::Function(tool) = tool else {
                return Err("opaque_content");
            };
            self.text(&tool.name)?;
            if let Some(description) = &tool.description {
                self.text(description)?;
            }
            self.json(&tool.parameters, 0)?;
        }
        if let Some(choice) = tool_choice {
            match choice {
                ToolChoice::Tool { name } => self.text(name)?,
                ToolChoice::Raw(value) => self.json(value, 0)?,
                ToolChoice::Auto | ToolChoice::Required | ToolChoice::None => {}
            }
        }
        if let Some(format) = &output.response_format {
            self.json(format, 0)?;
        }
        if let Some(effort) = &reasoning.effort {
            self.text(effort)?;
        }
        if let Some(raw) = &reasoning.raw {
            self.json(raw, 0)?;
        }
        for (key, value) in &extensions.fields {
            self.text(key)?;
            self.json(value, 0)?;
        }
        // Exact replay may retain fields omitted by normalization; bound both representations.
        for (format, body) in &preservation.requests {
            if ![
                WireFormat::OpenAiChat,
                WireFormat::OpenAiResponses,
                WireFormat::AnthropicMessages,
            ]
            .into_iter()
            .any(|known| format.as_str() == known.as_str())
            {
                return Err("opaque_content");
            }
            self.json(body, 0)?;
        }
        Ok(())
    }

    fn blocks(&mut self, blocks: &[ContentBlock], depth: usize) -> Result<(), &'static str> {
        for block in blocks {
            self.step(depth)?;
            match block {
                ContentBlock::Text { text } | ContentBlock::Refusal { text } => self.text(text)?,
                ContentBlock::Reasoning {
                    text,
                    signature,
                    details,
                } => {
                    self.text(text)?;
                    if let Some(signature) = signature {
                        self.text(signature)?;
                    }
                    for detail in details {
                        self.json(detail, depth + 1)?;
                    }
                }
                ContentBlock::ToolCall(call) => {
                    self.text(&call.id)?;
                    self.text(&call.name)?;
                    self.json(&call.arguments, depth + 1)?;
                }
                ContentBlock::ToolResult(result) => {
                    self.text(&result.tool_call_id)?;
                    self.blocks(&result.content, depth + 1)?;
                }
                ContentBlock::Image { .. }
                | ContentBlock::Audio { .. }
                | ContentBlock::Video { .. }
                | ContentBlock::File { .. }
                | ContentBlock::Mcp(_)
                | ContentBlock::Unknown { .. } => return Err("opaque_content"),
            }
        }
        Ok(())
    }

    fn json(&mut self, value: &serde_json::Value, depth: usize) -> Result<(), &'static str> {
        self.step(depth)?;
        match value {
            serde_json::Value::String(value) => self.text(value),
            serde_json::Value::Array(values) => {
                for value in values {
                    self.json(value, depth + 1)?;
                }
                Ok(())
            }
            serde_json::Value::Object(values) => {
                let kind = values.get("type").and_then(serde_json::Value::as_str);
                let encrypted_reasoning = kind == Some("reasoning.encrypted")
                    || kind == Some("reasoning")
                        && values
                            .get("encrypted_content")
                            .is_some_and(|value| !value.is_null());
                if encrypted_reasoning {
                    return Err("opaque_content");
                }
                for (key, value) in values {
                    self.text(key)?;
                    self.json(value, depth + 1)?;
                }
                Ok(())
            }
            serde_json::Value::Number(value) => self.text(&value.to_string()),
            serde_json::Value::Null | serde_json::Value::Bool(_) => Ok(()),
        }
    }

    fn text(&mut self, text: &str) -> Result<(), &'static str> {
        self.step(0)?;
        self.bytes = self.bytes.saturating_add(text.len());
        if self.bytes > MAX_SCAN_BYTES {
            return Err("scan_limit");
        }
        Ok(())
    }

    fn step(&mut self, depth: usize) -> Result<(), &'static str> {
        if depth > MAX_SCAN_DEPTH {
            return Err("scan_limit");
        }
        self.steps = self.steps.saturating_add(1);
        if self.steps > MAX_SCAN_STEPS {
            return Err("scan_limit");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use switchyard_protocol::{ImageSource, McpContent, Message, Role, ToolCall};

    use super::*;

    #[test]
    fn accepts_inspectable_content_without_classifying_it() {
        let mut request = Request::default();
        request.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call-1".into(),
                name: "lookup".into(),
                arguments: json!({"email": "user@company.test", "card": 4111111111111111_u64}),
            })],
        });
        request.llm_request.preservation.requests.insert(
            "openai_chat".into(),
            json!({"token": "Bearer abcdefghijklmnop"}),
        );
        assert_eq!(validate_input(&request), Ok(()));
    }

    #[test]
    fn rejects_opaque_content_and_unrecognized_preservation() {
        for block in [
            ContentBlock::Unknown {
                provider: "custom".into(),
                raw: json!({}),
            },
            ContentBlock::Image {
                source: ImageSource::Url {
                    url: "https://example.test/image".into(),
                    detail: None,
                },
            },
            ContentBlock::Mcp(McpContent::Listing {
                server_name: "inventory".into(),
                id: None,
                tools: vec![],
                error: None,
            }),
        ] {
            let mut request = Request::default();
            request.llm_request.messages.push(Message {
                role: Role::User,
                content: vec![block],
            });
            assert_eq!(validate_input(&request), Err("opaque_content"));
        }
        for value in [
            json!({"type": "reasoning", "encrypted_content": "opaque"}),
            json!({"type": "reasoning.encrypted", "data": "opaque"}),
        ] {
            let mut request = Request::default();
            request.llm_request.reasoning.raw = Some(value);
            assert_eq!(validate_input(&request), Err("opaque_content"));
        }
        let mut request = Request::default();
        request
            .llm_request
            .preservation
            .requests
            .insert("custom".into(), json!({}));
        assert_eq!(validate_input(&request), Err("opaque_content"));
        for tool in [
            json!({"type": "mcp", "name": "inventory", "url": "https://example.invalid/mcp"}),
            json!({"type": "provider", "provider": "openai_responses", "raw": {}}),
        ] {
            let mut request = Request::default();
            request
                .llm_request
                .tools
                .push(serde_json::from_value(tool).expect("tool"));
            assert_eq!(validate_input(&request), Err("opaque_content"));
        }
    }

    #[test]
    fn bounds_bytes_depth_and_steps() {
        let oversized = Request {
            llm_request: LlmRequest {
                messages: vec![Message::text(Role::User, "x".repeat(MAX_SCAN_BYTES + 1))],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut deep_value = serde_json::Value::Null;
        for _ in 0..=MAX_SCAN_DEPTH {
            deep_value = json!([deep_value]);
        }
        let mut too_deep = Request::default();
        too_deep
            .llm_request
            .extensions
            .fields
            .insert("deep".into(), deep_value);
        let mut too_many = Request::default();
        too_many.llm_request.extensions.fields.insert(
            "many".into(),
            serde_json::Value::Array(vec![serde_json::Value::Null; MAX_SCAN_STEPS]),
        );
        for request in [&oversized, &too_deep, &too_many] {
            assert_eq!(validate_input(request), Err("scan_limit"));
        }
    }
}
