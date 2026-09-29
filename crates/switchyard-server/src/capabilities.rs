// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reject inputs that the selected route explicitly disables.

use serde_json::Value;
use switchyard_protocol::{ContentBlock, LlmRequest};
use switchyard_runner::ModelCapabilities;

/// Return the first explicitly disabled capability used by decoded or preserved input.
/// Forwarding can retain provider JSON fields that decoding omits.
pub(crate) fn unsupported_capability(
    capabilities: ModelCapabilities,
    request: &LlmRequest,
    body: &Value,
) -> Option<&'static str> {
    if capabilities.reasoning == Some(false)
        && (request.reasoning.effort.is_some()
            || request
                .reasoning
                .raw
                .as_ref()
                .is_some_and(|value| !value.is_null())
            || ["reasoning", "reasoning_effort", "thinking"]
                .iter()
                .any(|key| body.get(key).is_some_and(|value| !value.is_null()))
            || body
                .pointer("/output_config/effort")
                .is_some_and(|value| !value.is_null()))
    {
        return Some("reasoning");
    }
    if capabilities.tool_calling == Some(false)
        && (!request.tools.is_empty()
            || request.tool_choice.is_some()
            || ["tools", "functions"].iter().any(|key| {
                body.get(key)
                    .and_then(Value::as_array)
                    .is_some_and(|tools| !tools.is_empty())
            })
            || ["tool_choice", "parallel_tool_calls", "function_call"]
                .iter()
                .any(|key| body.get(key).is_some_and(|value| !value.is_null())))
    {
        return Some("tool_calling");
    }
    if capabilities.tool_calling != Some(false) && capabilities.vision != Some(false) {
        return None;
    }
    request
        .instructions
        .iter()
        .map(|instruction| instruction.content.as_slice())
        .chain(
            request
                .messages
                .iter()
                .map(|message| message.content.as_slice()),
        )
        .find_map(|content| unsupported_content(capabilities, content))
        .or_else(|| unsupported_preserved_content(capabilities, body))
}

// Some provider fields survive forwarding without becoming decoded content blocks.
// Inspect protocol content and tool outputs, leaving tool arguments and text untouched.
fn unsupported_preserved_content(
    capabilities: ModelCapabilities,
    body: &Value,
) -> Option<&'static str> {
    let mut pending: Vec<_> = ["messages", "input", "system"]
        .iter()
        .filter_map(|key| body.get(key))
        .collect();
    while let Some(value) = pending.pop() {
        if let Some(items) = value.as_array() {
            pending.extend(items);
            continue;
        }
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if capabilities.vision == Some(false)
            && matches!(
                kind,
                "image" | "image_url" | "input_image" | "computer_screenshot"
            )
        {
            return Some("vision");
        }
        if capabilities.tool_calling == Some(false)
            && (matches!(
                kind,
                "additional_tools"
                    | "tool_use"
                    | "tool_result"
                    | "mcp_list_tools"
                    | "mcp_approval_request"
                    | "mcp_approval_response"
            ) || kind.ends_with("_call")
                || kind.ends_with("_call_output")
                || kind.ends_with("_tool_use")
                || kind.ends_with("_tool_result"))
        {
            return Some("tool_calling");
        }
        pending.extend(value.get("content"));
        if kind.ends_with("_call_output") {
            pending.extend(value.get("output"));
        }
    }
    None
}

/// Check decoded blocks, including images nested inside tool-result content.
fn unsupported_content(
    capabilities: ModelCapabilities,
    content: &[ContentBlock],
) -> Option<&'static str> {
    content.iter().find_map(|block| match block {
        ContentBlock::Image { .. } if capabilities.vision == Some(false) => Some("vision"),
        ContentBlock::ToolCall(_) | ContentBlock::ToolResult(_)
            if capabilities.tool_calling == Some(false) =>
        {
            Some("tool_calling")
        }
        ContentBlock::ToolResult(result) => unsupported_content(capabilities, &result.content),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_translation::WireFormat::{AnthropicMessages, OpenAiChat, OpenAiResponses};
    use switchyard_translation::{WireFormat, decode_request, encode_request};

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    // Decode each fixture so disabled, enabled, and unset cases use the same request.
    fn assert_requires_capability(
        format: WireFormat,
        body: &Value,
        disabled: ModelCapabilities,
        capability: &str,
    ) -> TestResult<LlmRequest> {
        let request = decode_request(format, body)?;
        for (capabilities, expected) in [
            (disabled, Some(capability)),
            (ModelCapabilities::default(), None),
            (
                ModelCapabilities {
                    tool_calling: Some(true),
                    reasoning: Some(true),
                    vision: Some(true),
                    ..Default::default()
                },
                None,
            ),
        ] {
            assert_eq!(
                unsupported_capability(capabilities, &request, body),
                expected,
                "{body}"
            );
        }
        Ok(request)
    }

    // Tool controls and history, including MCP approvals, can survive forwarding
    // without appearing in decoded tools.
    #[test]
    fn rejects_tool_controls_and_history_preserved_outside_normalized_tools() -> TestResult {
        let capabilities = ModelCapabilities {
            tool_calling: Some(false),
            ..Default::default()
        };
        for body in [
            json!({"input": "hello", "tools": [{"type": "web_search"}]}),
            json!({"input": "hello", "functions": [{"name": "lookup", "parameters": {"type": "object"}}]}),
            json!({"input": "hello", "tool_choice": "none"}),
            json!({"input": "hello", "tool_choice": {"type": "web_search"}}),
            json!({"input": "hello", "parallel_tool_calls": false}),
            json!({"input": [{"type": "additional_tools", "tools": []}]}),
            json!({"input": [{"type": "shell_call", "id": "shell_1", "call_id": "call_1", "action": {"commands": ["pwd"]}}]}),
            json!({"input": [{"type": "function_call_output", "call_id": "call_1", "output": "done"}]}),
            json!({"input": [{"type": "mcp_list_tools", "server_label": "server", "tools": []}]}),
            json!({"input": [{"type": "mcp_approval_request", "id": "approval_1", "name": "lookup", "arguments": "{}", "server_label": "server"}]}),
            json!({"input": [{"type": "mcp_approval_response", "approval_request_id": "approval_1", "approve": true}]}),
            json!({"input": [{"type": "mcp_approval_response", "approval_request_id": "approval_1", "approve": false}]}),
        ] {
            assert_requires_capability(OpenAiResponses, &body, capabilities, "tool_calling")?;
        }
        let body = json!({"messages": [{"role": "user", "content": "hello"}], "tools": [], "functions": []});
        let request = decode_request(OpenAiChat, &body)?;
        assert_eq!(unsupported_capability(capabilities, &request, &body), None);
        Ok(())
    }

    // Preserved reasoning controls still reach the provider when decoding omits them.
    #[test]
    fn rejects_preserved_reasoning_controls() -> TestResult {
        let capabilities = ModelCapabilities {
            reasoning: Some(false),
            ..Default::default()
        };
        for (format, body, pointer) in [
            (
                OpenAiChat,
                json!({"messages": [{"role": "user", "content": "hello"}], "reasoning": {"enabled": true}}),
                "/reasoning",
            ),
            (
                OpenAiChat,
                json!({"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": 123}),
                "/reasoning_effort",
            ),
            (
                AnthropicMessages,
                json!({"messages": [{"role": "user", "content": "hello"}], "output_config": {"effort": 123}}),
                "/output_config/effort",
            ),
        ] {
            let request = assert_requires_capability(format, &body, capabilities, "reasoning")?;
            assert_eq!(
                encode_request(&request, format)?.pointer(pointer),
                body.pointer(pointer)
            );
        }
        Ok(())
    }

    // Decoded file-ID images and preserved computer screenshots both require vision.
    #[test]
    fn rejects_file_id_images_and_computer_screenshots() -> TestResult {
        let capabilities = ModelCapabilities {
            vision: Some(false),
            ..Default::default()
        };
        for body in [
            json!({"input": [{"role": "user", "content": [{"type": "input_image", "file_id": "file_1"}]}]}),
            json!({"input": [{"type": "computer_call_output", "call_id": "call_1", "output": {"type": "computer_screenshot", "image_url": "https://example.test/image.png"}}]}),
        ] {
            let request =
                assert_requires_capability(OpenAiResponses, &body, capabilities, "vision")?;
            assert_eq!(
                encode_request(&request, OpenAiResponses)?["input"],
                body["input"]
            );
        }
        Ok(())
    }

    // Allowing tool results must not hide unsupported images nested in their content.
    #[test]
    fn rejects_images_inside_tool_results_when_tools_are_allowed() -> TestResult {
        let body = json!({"messages": [{"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "call_1", "content": [{
                "type": "image", "source": {"type": "url", "url": "https://example.test/image.png"}
            }]
        }]}]});
        assert_requires_capability(
            AnthropicMessages,
            &body,
            ModelCapabilities {
                tool_calling: Some(true),
                vision: Some(false),
                ..Default::default()
            },
            "vision",
        )?;
        Ok(())
    }
}
