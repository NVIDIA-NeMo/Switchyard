// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Responses buffered and streaming codecs.

use serde_json::{Map, Value};

mod buffered;
mod stream;

pub use buffered::OpenAiResponsesCodec;
pub use stream::OpenAiResponsesStreamCodec;

pub(super) fn call_arguments(item: &Map<String, Value>) -> Option<&Value> {
    ["arguments", "input", "payload"]
        .into_iter()
        .filter_map(|field| item.get(field))
        .find(|value| !value.is_null() && !matches!(value, Value::String(text) if text.is_empty()))
}

pub(super) fn usage_u64(usage: &Map<String, Value>, fields: &[&str]) -> Option<u64> {
    fields
        .iter()
        .find_map(|field| usage.get(*field).and_then(Value::as_u64))
}

pub(super) fn usage_detail_u64(
    usage: &Map<String, Value>,
    detail_fields: &[&str],
    value_fields: &[&str],
) -> Option<u64> {
    detail_fields.iter().find_map(|detail_field| {
        let details = usage.get(*detail_field)?.as_object()?;
        usage_u64(details, value_fields)
    })
}

pub(crate) fn is_native_output(kind: &str) -> bool {
    matches!(
        kind,
        "apply_patch_call" | "shell_call" | "computer_call" | "image_generation_call"
    )
}

pub(crate) fn validate_response_output(
    response: &crate::AggLlmResponse,
    target: crate::WireFormat,
) -> crate::Result<()> {
    for block in response.outputs.iter().flat_map(|output| &output.content) {
        if let crate::ContentBlock::Unknown { provider, raw } = block
            && provider.as_str() == crate::WireFormat::OpenAiResponses.as_str()
            && raw["type"].as_str().is_some_and(is_native_output)
        {
            return Err(crate::TranslationError::UnsupportedTranslation {
                from: provider.clone(),
                to: target.into(),
            });
        }
    }
    Ok(())
}

// Native outputs have no neutral streaming representation; inspect them before discarding raw events.
pub(crate) fn validate_stream_output(
    source: &crate::FormatId,
    target: &crate::FormatId,
    event: &serde_json::Value,
) -> crate::Result<()> {
    if source.as_str() == crate::WireFormat::OpenAiResponses.as_str() && source != target {
        let mut items = event
            .get("item")
            .into_iter()
            .chain(event["response"]["output"].as_array().into_iter().flatten());
        if items.any(|item| item["type"].as_str().is_some_and(is_native_output)) {
            return Err(crate::TranslationError::UnsupportedTranslation {
                from: source.clone(),
                to: target.clone(),
            });
        }
    }
    Ok(())
}

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
