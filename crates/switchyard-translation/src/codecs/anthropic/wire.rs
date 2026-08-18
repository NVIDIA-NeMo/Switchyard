// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated Anthropic Messages wire types.

typify::import_types!(
    schema = "schemas/anthropic-create-message.schema.json",
    replace = {
        Model = String,
    },
);

#[cfg(test)]
mod tests {
    use super::AnthropicCreateMessageParams;

    // The generated root accepts the smallest request allowed by Anthropic's schema.
    #[test]
    fn deserializes_minimal_create_message_request() {
        let request: AnthropicCreateMessageParams = serde_json::from_value(serde_json::json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 64
        }))
        .expect("the official schema should accept a minimal Messages request");

        assert_eq!(request.max_tokens, 64);
        assert_eq!(request.messages.len(), 1);
    }
}
