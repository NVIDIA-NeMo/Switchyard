// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Target-specific replay policy for OpenAI Responses reasoning items.

use serde::Deserialize;
use serde_json::{Map, Value};

/// Controls which Responses reasoning items are replayed to an upstream.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesReasoningPolicy {
    /// Choose from the target on each request.
    ///
    /// OpenAI-hosted targets use [`Self::PreserveEncrypted`]. Every other target uses
    /// [`Self::PreservePlaintext`].
    #[default]
    Auto,
    /// Keep encrypted reasoning and stored IDs. Remove plaintext reasoning.
    ///
    /// OpenAI rejects plaintext reasoning `content` and unresolvable stored IDs.
    PreserveEncrypted,
    /// Keep plaintext reasoning. Remove opaque encrypted payloads.
    ///
    /// Switchyard cannot tell which provider issued an encrypted payload, so it
    /// removes the payload and keeps any `summary` or `content` beside it.
    PreservePlaintext,
    /// Drop all reasoning items while preserving messages and tool-call history.
    ///
    /// Use this for local Responses-compatible servers that cannot consume
    /// another provider's reasoning representation.
    Drop,
}

impl ResponsesReasoningPolicy {
    /// Normalizes a Responses request body sent to `model` at `base_url`.
    pub(crate) fn normalize(self, body: &mut Value, base_url: &str, model: &str) {
        let policy = match self {
            Self::Auto if is_openai_target(base_url, model) => Self::PreserveEncrypted,
            Self::Auto => Self::PreservePlaintext,
            policy => policy,
        };
        let has_stored_reasoning = body.get("store").and_then(Value::as_bool) != Some(false);
        let Some(Value::Array(input)) = body.get_mut("input") else {
            return;
        };
        let mut stats = Stats::default();
        input.retain_mut(|item| match item.as_object_mut() {
            Some(item) if item.get("type").and_then(Value::as_str) == Some("reasoning") => {
                stats.items += 1;
                let keep = policy.keep(item, has_stored_reasoning, &mut stats.removed_payloads);
                stats.removed_items += usize::from(!keep);
                keep
            }
            _ => true,
        });
        if stats.items > 0 {
            tracing::trace!(
                policy = ?policy,
                reasoning_items = stats.items,
                removed_items = stats.removed_items,
                removed_payloads = stats.removed_payloads,
                "normalized Responses reasoning"
            );
        }
    }

    // Returns whether to keep a reasoning item, after removing parts the target cannot use.
    // Stored reasoning can be replayed by ID without encrypted content.
    fn keep(
        self,
        item: &mut Map<String, Value>,
        has_stored_reasoning: bool,
        removed_payloads: &mut usize,
    ) -> bool {
        let encrypted_content = item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty());
        let has_stored_id = has_stored_reasoning
            && item
                .get("id")
                .and_then(Value::as_str)
                .is_some_and(|id| !id.is_empty());
        match self {
            Self::PreserveEncrypted => {
                if encrypted_content
                    .is_some_and(switchyard_translation::is_anthropic_thinking_envelope)
                    || (encrypted_content.is_none() && !has_stored_id)
                {
                    return false;
                }
                item.insert("content".to_string(), Value::Array(Vec::new()));
                true
            }
            Self::PreservePlaintext => {
                let has_plaintext = ["summary", "content"].into_iter().any(|field| {
                    item.get(field)
                        .and_then(Value::as_array)
                        .is_some_and(|parts| !parts.is_empty())
                });
                if encrypted_content.is_none() {
                    return has_plaintext || has_stored_id;
                }
                // The item ID belongs to the payload's issuer, so it cannot stand alone.
                item.remove("encrypted_content");
                *removed_payloads += 1;
                has_plaintext
            }
            // `normalize` resolves `Auto` before calling this.
            Self::Auto | Self::Drop => false,
        }
    }
}

// Counts for one request. They hold no prompt or reasoning text.
#[derive(Default)]
struct Stats {
    items: usize,
    removed_items: usize,
    removed_payloads: usize,
}

// OpenAI and Azure OpenAI reject plaintext reasoning in Responses input. A shared
// gateway hostname says nothing about the model behind it, so other hosts are judged
// by the upstream model name alone.
fn is_openai_target(base_url: &str, model: &str) -> bool {
    let host = reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase));
    if host.is_some_and(|host| host == "api.openai.com" || host.ends_with(".openai.azure.com")) {
        return true;
    }
    let name = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    let is_o_series = name
        .strip_prefix('o')
        .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()));
    (name.starts_with("gpt-") && !name.starts_with("gpt-oss"))
        || name.starts_with("codex-")
        || is_o_series
}

#[cfg(test)]
mod tests {
    use super::is_openai_target;

    #[test]
    fn openai_targets_come_from_host_or_model_name() {
        let hub = "https://inference-api.nvidia.com/v1";
        let proxy = "http://inference-proxy.internal:8080/v1";
        for (base_url, model, expected) in [
            ("https://api.openai.com/v1", "my-alias", true),
            (
                "https://acme.openai.azure.com/openai/v1",
                "deployment",
                true,
            ),
            (hub, "openai/openai/gpt-5.5", true),
            (proxy, "azure/openai/gpt-5.6-sol", true),
            (hub, "openai/o4-mini", true),
            (proxy, "codex-mini-latest", true),
            (hub, "nvidia/zai-org/glm-5.3-flash", false),
            (proxy, "nvidia/zai-org/glm-5.3-flash", false),
            (hub, "openai/gpt-oss-120b", false),
            (proxy, "switchyard", false),
            ("not a url", "deepseek-v4-pro", false),
        ] {
            assert_eq!(
                is_openai_target(base_url, model),
                expected,
                "{base_url} {model}"
            );
        }
    }
}
