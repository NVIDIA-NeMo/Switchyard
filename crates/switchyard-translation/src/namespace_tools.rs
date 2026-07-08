// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reversible encoding for Responses namespace tools on flat tool APIs.

use crate::error::{Result, TranslationError};

const NAMESPACE_TOOL_PREFIX: &str = "__sy1n";
const CHAT_FUNCTION_NAME_MAX_BYTES: usize = 64;

/// A Responses function name split into its namespace and child name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NamespaceToolName {
    pub(crate) namespace: String,
    pub(crate) name: String,
}

/// Encodes one Responses namespace child as a valid OpenAI Chat function name.
///
/// The namespace length makes the representation unambiguous even when either
/// component contains repeated underscores. Namespace and child names are
/// restricted to the portable Chat function-name alphabet, and overlong names
/// fail instead of being truncated into a possible collision.
pub(crate) fn encode_namespace_tool_name(
    namespace: &str,
    name: &str,
    path: impl Into<String>,
) -> Result<String> {
    let path = path.into();
    validate_component(namespace, &path, "namespace")?;
    validate_component(name, &path, "tool name")?;

    let encoded = format!(
        "{NAMESPACE_TOOL_PREFIX}{}_{}{}",
        namespace.len(),
        namespace,
        name
    );
    if encoded.len() > CHAT_FUNCTION_NAME_MAX_BYTES {
        return Err(invalid_value(
            path,
            format!(
                "namespaced tool encoding is {} bytes; OpenAI Chat function names are limited to {CHAT_FUNCTION_NAME_MAX_BYTES}",
                encoded.len(),
            ),
        ));
    }
    Ok(encoded)
}

/// Parses an internally encoded namespace tool name.
///
/// Names outside the reserved prefix are ordinary function names. Once the
/// prefix is present, every field must be canonical; malformed marker-like
/// names are errors rather than ordinary tools.
pub(crate) fn decode_namespace_tool_name(
    encoded: &str,
    path: impl Into<String>,
) -> Result<Option<NamespaceToolName>> {
    let path = path.into();
    let Some(rest) = encoded.strip_prefix(NAMESPACE_TOOL_PREFIX) else {
        return Ok(None);
    };
    if encoded.len() > CHAT_FUNCTION_NAME_MAX_BYTES {
        return Err(invalid_value(
            path,
            format!(
                "encoded namespace tool is {} bytes; maximum is {CHAT_FUNCTION_NAME_MAX_BYTES}",
                encoded.len(),
            ),
        ));
    }

    let Some(length_end) = rest.find('_') else {
        return Err(invalid_value(
            path,
            "namespace marker has no length separator",
        ));
    };
    let length_text = &rest[..length_end];
    if length_text.is_empty()
        || (length_text.len() > 1 && length_text.starts_with('0'))
        || !length_text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(invalid_value(
            path,
            "namespace marker length must be a canonical positive decimal integer",
        ));
    }
    let namespace_len = length_text
        .parse::<usize>()
        .map_err(|_| invalid_value(path.clone(), "namespace marker length is out of range"))?;
    if namespace_len == 0 {
        return Err(invalid_value(
            path,
            "namespace marker length must be greater than zero",
        ));
    }

    let components = &rest[length_end + 1..];
    if components.len() <= namespace_len || !components.is_char_boundary(namespace_len) {
        return Err(invalid_value(
            path,
            "namespace marker length exceeds the encoded component",
        ));
    }
    let namespace = &components[..namespace_len];
    let name = &components[namespace_len..];
    validate_component(namespace, &path, "namespace")?;
    validate_component(name, &path, "tool name")?;

    let canonical = encode_namespace_tool_name(namespace, name, path.clone())?;
    if canonical != encoded {
        return Err(invalid_value(path, "namespace marker is not canonical"));
    }
    Ok(Some(NamespaceToolName {
        namespace: namespace.to_string(),
        name: name.to_string(),
    }))
}

/// Rejects a normal Responses function that collides with the reserved marker.
pub(crate) fn reject_reserved_namespace_tool_name(
    name: &str,
    path: impl Into<String>,
) -> Result<()> {
    let path = path.into();
    if name.starts_with(NAMESPACE_TOOL_PREFIX) {
        // Parse first so malformed marker-shaped names surface their precise
        // validation error. A valid marker is still reserved for translation.
        let _ = decode_namespace_tool_name(name, path.clone())?;
        return Err(invalid_value(
            path,
            "function name collides with Switchyard's reserved namespace-tool encoding",
        ));
    }
    Ok(())
}

fn validate_component(value: &str, path: &str, label: &str) -> Result<()> {
    if value.is_empty() {
        return Err(invalid_value(path, format!("{label} must not be empty")));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(invalid_value(
            path,
            format!("{label} must contain only ASCII letters, digits, '_' or '-'"),
        ));
    }
    Ok(())
}

fn invalid_value(path: impl Into<String>, message: impl Into<String>) -> TranslationError {
    TranslationError::InvalidValue {
        path: path.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_encoding_is_canonical_and_reversible() {
        let encoded = encode_namespace_tool_name(
            "mcp__tooluniverse",
            "trialqa_extract_adverse_events",
            "$.tools[0]",
        )
        .expect("TrialQA name should fit Chat constraints");
        assert_eq!(
            encoded,
            "__sy1n17_mcp__tooluniversetrialqa_extract_adverse_events",
        );
        assert_eq!(encoded.len(), 56);
        assert_eq!(
            decode_namespace_tool_name(&encoded, "$.tools[0]").expect("encoded name should parse"),
            Some(NamespaceToolName {
                namespace: "mcp__tooluniverse".to_string(),
                name: "trialqa_extract_adverse_events".to_string(),
            }),
        );
    }

    #[test]
    fn malformed_and_overlong_encodings_fail() {
        for malformed in [
            "__sy1n_ns_tool",
            "__sy1n0_tool",
            "__sy1n02_nstool",
            "__sy1n9_nstool",
            "__sy1n2_ns",
        ] {
            assert!(decode_namespace_tool_name(malformed, "$.name").is_err());
        }
        assert!(encode_namespace_tool_name(
            "namespace",
            "a_name_that_is_far_too_long_for_the_flat_openai_chat_function_limit",
            "$.name",
        )
        .is_err());
    }
}
