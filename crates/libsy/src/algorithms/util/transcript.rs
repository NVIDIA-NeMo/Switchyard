// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plain-text message rendering shared by routing judges.

use switchyard_protocol::{ContentBlock, Message, Role};

/// Flattens a message to plain text, tool calls and tool results included.
///
/// [`Message::text_content`] is deliberately not used here: it keeps only text and refusal
/// blocks, which would erase exactly the repeated-command signal the judge's loop detection
/// relies on.
pub(crate) fn message_text(message: &Message) -> String {
    let mut parts = Vec::new();
    let terminus_commands = if message.role == Role::Assistant {
        message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolCall(call) if call.name == "bash_command" => call
                    .arguments
                    .get("keystrokes")
                    .and_then(|value| value.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    collect_text(&message.content, &mut parts, &terminus_commands);
    parts.join(" ")
}

/// Removes a Terminus command batch when a structured bash call carries the same action.
///
/// The model-facing request remains untouched. Only the judge's plain-text view is normalized,
/// so one action cannot look like two attempts while the agent still sees its native history.
fn without_duplicated_terminus_commands(text: &str, tool_commands: &[&str]) -> String {
    let mut normalized_text = String::with_capacity(text.len());
    let mut unmatched_tool_commands = tool_commands.to_vec();
    let mut copied_through = 0;
    let mut scan_from = 0;

    while let Some(relative_start) = text[scan_from..].find('{') {
        let start = scan_from + relative_start;

        let mut values =
            serde_json::Deserializer::from_str(&text[start..]).into_iter::<serde_json::Value>();
        let Some(Ok(mut value)) = values.next() else {
            scan_from = start + 1;
            continue;
        };
        let end = start + values.byte_offset();
        let Some(commands) = value
            .get("commands")
            .and_then(|commands| commands.as_array())
        else {
            scan_from = start + 1;
            continue;
        };
        let Some(command_batch) = commands
            .iter()
            .map(|command| command.get("keystrokes").and_then(|value| value.as_str()))
            .collect::<Option<Vec<_>>>()
        else {
            scan_from = start + 1;
            continue;
        };
        let mut remaining_tool_commands = unmatched_tool_commands.clone();
        let fully_encoded = command_batch.iter().all(|command| {
            let Some(index) = remaining_tool_commands
                .iter()
                .position(|candidate| candidate == command)
            else {
                return false;
            };
            remaining_tool_commands.swap_remove(index);
            true
        });
        if command_batch.is_empty() || !fully_encoded {
            scan_from = start + 1;
            continue;
        }

        value["commands"] = serde_json::Value::Array(Vec::new());
        let Ok(normalized) = serde_json::to_string(&value) else {
            scan_from = start + 1;
            continue;
        };
        normalized_text.push_str(&text[copied_through..start]);
        normalized_text.push_str(&normalized);
        copied_through = end;
        scan_from = end;
        unmatched_tool_commands = remaining_tool_commands;
    }
    normalized_text.push_str(&text[copied_through..]);
    normalized_text
}

/// Appends the judge-relevant text of each block, descending into tool results.
pub(super) fn collect_text(
    content: &[ContentBlock],
    parts: &mut Vec<String>,
    tool_commands: &[&str],
) {
    for block in content {
        match block {
            ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
                parts.push(without_duplicated_terminus_commands(text, tool_commands));
            }
            ContentBlock::ToolCall(call) => {
                parts.push(format!("tool_call {}({})", call.name, call.arguments));
            }
            ContentBlock::ToolResult(result) => collect_text(&result.content, parts, &[]),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::message_text;
    use serde_json::json;
    use switchyard_protocol::{ContentBlock, Message, Role, ToolCall, ToolResult};

    #[test]
    fn message_text_keeps_tool_calls_and_results() {
        let call = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "running it".to_string(),
                },
                ContentBlock::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    name: "bash".to_string(),
                    arguments: json!({"cmd": "ls"}),
                }),
            ],
        };
        let text = message_text(&call);
        assert!(text.contains("running it"), "{text}");
        assert!(text.contains(r#"tool_call bash({"cmd":"ls"})"#), "{text}");

        let result = Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "call-1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "no such file".to_string(),
                }],
                is_error: Some(true),
            })],
        };
        assert_eq!(message_text(&result), "no such file");
    }

    /// A raw command batch fully mirrored by structured tool calls is emptied in the judge view.
    #[test]
    fn message_text_deduplicates_terminus_commands_for_the_judge() {
        let first_command = "grep -n bug app.py\n";
        let second_command = "sed -n '1,80p' app.py\n";
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: format!(
                        "Before\n```json\n{}\n```\nAfter",
                        json!({
                            "analysis": "inspect the reported file",
                            "commands": [
                                {"keystrokes": first_command, "duration": 0.1},
                                {"keystrokes": second_command, "duration": 0.1},
                            ],
                        })
                    ),
                },
                ContentBlock::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    name: "bash_command".to_string(),
                    arguments: json!({"keystrokes": first_command, "duration": 0.1}),
                }),
                ContentBlock::ToolCall(ToolCall {
                    id: "call-2".to_string(),
                    name: "bash_command".to_string(),
                    arguments: json!({"keystrokes": second_command, "duration": 0.1}),
                }),
            ],
        };

        let text = message_text(&message);

        assert!(text.contains("inspect the reported file"), "{text}");
        assert!(text.contains("Before"), "{text}");
        assert!(text.contains("After"), "{text}");
        assert!(text.contains(r#""commands":[]"#), "{text}");
        assert_eq!(text.matches("grep -n bug app.py").count(), 1, "{text}");
        assert_eq!(text.matches("sed -n '1,80p' app.py").count(), 1, "{text}");
        assert_eq!(text.matches("tool_call bash_command(").count(), 2, "{text}");
    }

    /// Each structured tool call can absorb only one rendered batch, so a repeated batch stays.
    #[test]
    fn message_text_deduplicates_multiple_batches_once_per_tool_call() {
        let first_command = "grep -n bug app.py\n";
        let second_command = "sed -n '1,80p' app.py\n";
        let batch = |command| {
            json!({
                "analysis": "inspect",
                "commands": [{"keystrokes": command}],
            })
            .to_string()
        };
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: format!(
                        "First {} second {} repeated {}",
                        batch(first_command),
                        batch(second_command),
                        batch(first_command)
                    ),
                },
                ContentBlock::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    name: "bash_command".to_string(),
                    arguments: json!({"keystrokes": first_command}),
                }),
                ContentBlock::ToolCall(ToolCall {
                    id: "call-2".to_string(),
                    name: "bash_command".to_string(),
                    arguments: json!({"keystrokes": second_command}),
                }),
            ],
        };

        let text = message_text(&message);

        assert_eq!(text.matches(r#""commands":[]"#).count(), 2, "{text}");
        assert_eq!(text.matches("grep -n bug app.py").count(), 2, "{text}");
        assert_eq!(text.matches("sed -n '1,80p' app.py").count(), 1, "{text}");
    }

    /// A rendered batch stays when no structured tool call carries the same command.
    #[test]
    fn message_text_keeps_terminus_commands_without_matching_tool_call() {
        let command = "grep -n bug app.py\n";
        let text = json!({
            "analysis": "inspect the reported file",
            "commands": [{"keystrokes": command}],
        })
        .to_string();
        let without_tool_call = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: text.clone() }],
        };
        let mismatched_tool_call = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text { text },
                ContentBlock::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    name: "bash_command".to_string(),
                    arguments: json!({"keystrokes": "sed -n '1,20p' app.py\n", "duration": 0.1}),
                }),
            ],
        };

        assert_eq!(
            message_text(&without_tool_call)
                .matches("grep -n bug app.py")
                .count(),
            1
        );
        assert_eq!(
            message_text(&mismatched_tool_call)
                .matches("grep -n bug app.py")
                .count(),
            1
        );
    }

    /// A batch stays intact when only some of its commands have matching tool calls.
    #[test]
    fn message_text_keeps_a_partially_encoded_terminus_batch() {
        let first_command = "grep -n bug app.py\n";
        let second_command = "sed -n '1,80p' app.py\n";
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: json!({
                        "commands": [
                            {"keystrokes": first_command},
                            {"keystrokes": second_command},
                        ],
                    })
                    .to_string(),
                },
                ContentBlock::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    name: "bash_command".to_string(),
                    arguments: json!({"keystrokes": first_command}),
                }),
            ],
        };

        let text = message_text(&message);

        assert_eq!(text.matches("grep -n bug app.py").count(), 2, "{text}");
        assert_eq!(text.matches("sed -n '1,80p' app.py").count(), 1, "{text}");
        assert!(!text.contains(r#""commands":[]"#), "{text}");
    }

    /// Duplicate commands in one batch need one tool call each before the batch is removed.
    #[test]
    fn message_text_keeps_duplicate_commands_without_one_tool_call_each() {
        let command = "grep -n bug app.py\n";
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: json!({
                        "commands": [
                            {"keystrokes": command},
                            {"keystrokes": command},
                        ],
                    })
                    .to_string(),
                },
                ContentBlock::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    name: "bash_command".to_string(),
                    arguments: json!({"keystrokes": command}),
                }),
            ],
        };

        let text = message_text(&message);

        assert_eq!(text.matches("grep -n bug app.py").count(), 3, "{text}");
        assert!(!text.contains(r#""commands":[]"#), "{text}");
    }
}
