// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-agnostic helpers shared by buffered wire-format codecs.

use serde_json::{Map, Value};

use crate::ir::ContentBlock;

/// Extracts text-like blocks and joins them for text-only provider fields.
pub(crate) fn text_from_blocks(content: &[ContentBlock], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::Refusal { text } => Some(text.as_str()),
            ContentBlock::Unknown { raw, .. } => raw.as_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(separator)
}

/// Extracts private reasoning blocks without mixing them into visible text.
pub(crate) fn reasoning_text_from_blocks(content: &[ContentBlock], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(separator)
}

/// Splits a `data:<media type>;base64,<data>` URL into its components.
///
/// Provider APIs disagree on inline-image encoding: OpenAI embeds base64
/// payloads in data URLs while Anthropic and Gemini carry explicit media-type
/// plus data fields, so decoders normalize data URLs into base64 sources.
pub(crate) fn parse_data_url(url: &str) -> Option<(Option<String>, String)> {
    let rest = url.strip_prefix("data:")?;
    let (header, data) = rest.split_once(',')?;
    let header = header.strip_suffix(";base64")?;
    let media_type = (!header.is_empty()).then(|| header.to_string());
    Some((media_type, data.to_string()))
}

/// Copies unknown provider fields into the IR extension map.
pub(crate) fn provider_extensions(
    object: &Map<String, Value>,
    known: &[&str],
) -> Map<String, Value> {
    let mut extensions = Map::new();
    for (key, value) in object {
        if !known.contains(&key.as_str()) {
            extensions.insert(key.clone(), value.clone());
        }
    }
    extensions
}
