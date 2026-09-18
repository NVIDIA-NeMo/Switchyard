// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Responses buffered and streaming codecs.

mod buffered;
mod stream;

pub use buffered::OpenAiResponsesCodec;
pub use stream::OpenAiResponsesStreamCodec;

// This is a transport envelope, not encryption. Keep it distinct from native OpenAI
// ciphertext, and retain the original text because the signature covers that text.
const ANTHROPIC_THINKING_PREFIX: &str = "switchyard:anthropic-thinking:v1:";

fn encode_anthropic_thinking(text: &str, signature: &str) -> String {
    format!(
        "{ANTHROPIC_THINKING_PREFIX}{}",
        serde_json::json!([text, signature])
    )
}

fn decode_anthropic_thinking(payload: &str) -> Option<(String, String)> {
    serde_json::from_str(payload.strip_prefix(ANTHROPIC_THINKING_PREFIX)?).ok()
}
