// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-provider backend configuration: wire format, upstream URL, and auth.

use std::{collections::BTreeMap, fmt};

use reqwest::RequestBuilder;
use reqwest::header::HeaderValue;
use serde_json::Value;
use switchyard_protocol::{Metadata, WireFormat};

use crate::error::{LlmClientError, Result, is_overflow_body};

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default number of retries for server-configured upstream calls.
pub const DEFAULT_MAX_RETRIES: u32 = 2;

// Canonical OpenAI phrase plus NVIDIA/LiteLLM wrap variants. Adding a new
// provider-wrap is a one-line entry here, not a fork of the parsing logic.
const OPENAI_OVERFLOW_PHRASES: &[&str] = &[
    "maximum context length",
    "context length exceeded",
    "context window",
    "context length is only",
    "please reduce the length of the input",
    "exceeds the maximum allowed input length",
];

// Anthropic has no structured `error.code`, so detection is phrase-based only.
const ANTHROPIC_OVERFLOW_PHRASES: &[&str] = &[
    "prompt is too long",
    "maximum number of tokens",
    "context window",
    "context length",
];

/// Shared HTTP configuration for one upstream backend.
#[derive(Clone)]
pub struct HttpBackendConfig {
    /// Base URL of the provider API (e.g. `https://api.openai.com/v1`).
    pub base_url: String,
    /// API key for the provider, loaded by the caller. `None` sends no configured auth.
    pub api_key: Option<String>,
    /// Whether this backend forwards the caller's provider credential instead.
    pub forward_auth: bool,
    /// Custom headers added to every outbound call to this backend.
    ///
    /// Provider-owned headers are rejected so a static value cannot replace
    /// configured or forwarded auth. Header names are case-insensitive.
    pub extra_headers: BTreeMap<String, String>,
    /// Default top-level request fields, applied only when the request omits the key.
    pub extra_body: BTreeMap<String, Value>,
    /// Additional attempts after the initial upstream request.
    pub max_retries: u32,
}

impl fmt::Debug for HttpBackendConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpBackendConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("forward_auth", &self.forward_auth)
            .field("extra_header_names", &self.extra_headers.keys())
            .field("extra_body_keys", &self.extra_body.keys())
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

/// A configured upstream backend, one variant per built-in wire format.
///
/// The variant fixes the wire format, URL path, and auth scheme together so no
/// invalid combination can be constructed.
#[derive(Clone, Debug)]
pub enum Backend {
    /// OpenAI-compatible Chat Completions API.
    OpenAiChat(HttpBackendConfig),
    /// OpenAI Responses API.
    OpenAiResponses(HttpBackendConfig),
    /// Anthropic Messages API.
    Anthropic(HttpBackendConfig),
}

impl Backend {
    // Checks custom headers before the client can send a request.
    pub(crate) fn validate_extra_headers(&self, model_name: &str) -> Result<()> {
        let invalid_name = self.config().extra_headers.keys().find(|name| match self {
            Backend::OpenAiChat(_) | Backend::OpenAiResponses(_) => {
                name.eq_ignore_ascii_case("authorization")
                    || (self.is_forwarding_auth()
                        && (name.eq_ignore_ascii_case("chatgpt-account-id")
                            || name.eq_ignore_ascii_case("x-openai-fedramp")))
            }
            Backend::Anthropic(_) => {
                name.eq_ignore_ascii_case("x-api-key")
                    || name.eq_ignore_ascii_case("anthropic-version")
                    || (self.is_forwarding_auth()
                        && (name.eq_ignore_ascii_case("authorization")
                            || name.eq_ignore_ascii_case("anthropic-beta")))
            }
        });
        if let Some(name) = invalid_name {
            return Err(LlmClientError::Configuration {
                message: format!(
                    "model {model_name:?} extra_headers cannot set {name:?}; extra_headers is only for additional headers"
                ),
            });
        }
        Ok(())
    }

    /// The wire format the request IR is encoded to for this backend.
    pub fn wire_format(&self) -> WireFormat {
        match self {
            Backend::OpenAiChat(_) => WireFormat::OpenAiChat,
            Backend::OpenAiResponses(_) => WireFormat::OpenAiResponses,
            Backend::Anthropic(_) => WireFormat::AnthropicMessages,
        }
    }

    // Shared HTTP config, regardless of variant.
    fn config(&self) -> &HttpBackendConfig {
        match self {
            Backend::OpenAiChat(config)
            | Backend::OpenAiResponses(config)
            | Backend::Anthropic(config) => config,
        }
    }

    /// The fully resolved upstream URL for this backend's endpoint.
    ///
    /// Tolerates base URLs that already include the provider path (or a bare
    /// `/v1`), matching the join rules of the existing native backends.
    pub fn url(&self) -> String {
        let base_url = self.config().base_url.trim_end_matches('/');
        match self {
            Backend::OpenAiChat(_) => openai_url(base_url, "/chat/completions"),
            Backend::OpenAiResponses(_) => openai_url(base_url, "/responses"),
            Backend::Anthropic(_) => anthropic_url(base_url),
        }
    }

    /// Applies this backend's configured auth and version headers to a request builder.
    ///
    /// OpenAI variants use `Authorization: Bearer <key>`; Anthropic uses
    /// `x-api-key: <key>` plus the required `anthropic-version` header. A backend
    /// with `forward_auth` uses the caller's provider credential instead.
    pub fn apply_auth(&self, mut builder: RequestBuilder) -> RequestBuilder {
        let api_key = if self.is_forwarding_auth() {
            None
        } else {
            self.config().api_key.as_deref()
        };
        match self {
            Backend::OpenAiChat(_) | Backend::OpenAiResponses(_) => {
                if let Some(api_key) = api_key {
                    builder = builder.bearer_auth(api_key);
                }
            }
            Backend::Anthropic(_) => {
                builder = builder.header("anthropic-version", ANTHROPIC_VERSION);
                if let Some(api_key) = api_key {
                    builder = builder.header("x-api-key", api_key);
                }
            }
        }
        builder
    }

    pub(crate) fn is_forwarding_auth(&self) -> bool {
        self.config().forward_auth
    }

    /// Applies only the caller credential accepted by this provider.
    pub(crate) fn apply_forwarded_auth(
        &self,
        mut builder: RequestBuilder,
        metadata: Option<&Metadata>,
    ) -> RequestBuilder {
        if !self.is_forwarding_auth() {
            return builder;
        }
        let Some(headers) = metadata.and_then(|metadata| metadata.http_headers.as_ref()) else {
            return builder;
        };
        match self {
            Backend::OpenAiChat(_) | Backend::OpenAiResponses(_) => {
                for name in ["authorization", "chatgpt-account-id", "x-openai-fedramp"] {
                    if let Some(value) = headers.get(name) {
                        builder = builder.header(name, sensitive_header(value));
                    }
                }
            }
            Backend::Anthropic(_) => {
                for name in ["authorization", "x-api-key"] {
                    if let Some(value) = headers.get(name) {
                        builder = builder.header(name, sensitive_header(value));
                    }
                }
                if let Some(value) = headers.get("anthropic-beta")
                    && let Some(value) = oauth_beta_header(value)
                {
                    builder = builder.header("anthropic-beta", value);
                }
            }
        }
        builder
    }

    /// Removes an echoed caller credential before an upstream error is returned or logged.
    pub(crate) fn redact_forwarded_auth(
        &self,
        mut body: String,
        metadata: Option<&Metadata>,
    ) -> String {
        if !self.is_forwarding_auth() {
            return body;
        }
        let Some(headers) = metadata.and_then(|metadata| metadata.http_headers.as_ref()) else {
            return body;
        };
        let secret_headers: &[&str] = match self {
            Backend::OpenAiChat(_) | Backend::OpenAiResponses(_) => {
                &["authorization", "chatgpt-account-id"]
            }
            Backend::Anthropic(_) => &["authorization", "x-api-key"],
        };
        for name in secret_headers {
            let Some(value) = headers.get(*name).and_then(|value| value.to_str().ok()) else {
                continue;
            };
            if !value.is_empty() {
                body = body.replace(value, "[REDACTED]");
            }
        }
        body
    }

    /// Custom per-backend headers to forward on every call.
    pub fn extra_headers(&self) -> &BTreeMap<String, String> {
        &self.config().extra_headers
    }

    /// Default top-level fields to merge into outbound request bodies.
    pub fn extra_body(&self) -> &BTreeMap<String, Value> {
        &self.config().extra_body
    }

    /// Additional attempts allowed after the initial request.
    pub fn max_retries(&self) -> u32 {
        self.config().max_retries
    }

    /// Whether this backend speaks the Anthropic Messages wire format — the only
    /// one with a `count_tokens` endpoint.
    pub fn is_anthropic(&self) -> bool {
        matches!(self, Backend::Anthropic(_))
    }

    /// The upstream `/v1/messages/count_tokens` URL, derived from the same base
    /// URL join as [`url`](Self::url).
    pub fn count_tokens_url(&self) -> String {
        let base_url = self.config().base_url.trim_end_matches('/');
        format!("{}/count_tokens", anthropic_url(base_url))
    }

    /// Whether an upstream 400 `body` looks like a context-window overflow for
    /// this backend's provider.
    pub(crate) fn is_context_overflow(&self, body: &str) -> bool {
        match self {
            Backend::OpenAiChat(_) | Backend::OpenAiResponses(_) => is_overflow_body(
                body,
                |value| {
                    value
                        .get("error")
                        .and_then(|err| err.get("code"))
                        .and_then(serde_json::Value::as_str)
                        == Some("context_length_exceeded")
                },
                OPENAI_OVERFLOW_PHRASES,
            ),
            Backend::Anthropic(_) => is_overflow_body(body, |_| false, ANTHROPIC_OVERFLOW_PHRASES),
        }
    }
}

// Retains OAuth markers while keeping provider feature betas backend-owned.
fn sensitive_header(value: &HeaderValue) -> HeaderValue {
    let mut value = value.clone();
    value.set_sensitive(true);
    value
}

fn oauth_beta_header(value: &HeaderValue) -> Option<HeaderValue> {
    let oauth_betas = value
        .to_str()
        .ok()?
        .split(',')
        .map(str::trim)
        .filter(|beta| {
            beta.get(..6)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("oauth-"))
        });
    let value = oauth_betas.collect::<Vec<_>>().join(",");
    if value.is_empty() {
        return None;
    }
    let mut value = HeaderValue::from_str(&value).ok()?;
    value.set_sensitive(true);
    Some(value)
}

// Accept either a root `/v1` URL or an already-specific OpenAI endpoint URL.
fn openai_url(base_url: &str, suffix: &str) -> String {
    let base_root = base_url
        .strip_suffix("/chat/completions")
        .or_else(|| base_url.strip_suffix("/responses"))
        .unwrap_or(base_url);
    format!("{base_root}{suffix}")
}

// Accept a bare host, a `/v1` root, or an already-specific `/v1/messages` URL.
fn anthropic_url(base_url: &str) -> String {
    if base_url.ends_with("/v1/messages") {
        base_url.to_string()
    } else if base_url.ends_with("/v1") {
        format!("{base_url}/messages")
    } else {
        format!("{base_url}/v1/messages")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(base_url: &str) -> HttpBackendConfig {
        HttpBackendConfig {
            base_url: base_url.to_string(),
            api_key: Some("secret".to_string()),
            forward_auth: false,
            extra_headers: BTreeMap::new(),
            extra_body: BTreeMap::new(),
            max_retries: 0,
        }
    }

    #[test]
    fn openai_chat_url_joins_bare_v1() {
        let backend = Backend::OpenAiChat(config("https://api.openai.com/v1"));
        assert_eq!(backend.url(), "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn openai_chat_url_tolerates_trailing_slash_and_existing_suffix() {
        assert_eq!(
            Backend::OpenAiChat(config("https://api.openai.com/v1/")).url(),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            Backend::OpenAiChat(config("https://api.openai.com/v1/chat/completions")).url(),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn openai_responses_url_uses_responses_path() {
        assert_eq!(
            Backend::OpenAiResponses(config("https://api.openai.com/v1")).url(),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn anthropic_url_join_cases() {
        assert_eq!(
            Backend::Anthropic(config("https://api.anthropic.com")).url(),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            Backend::Anthropic(config("https://api.anthropic.com/v1")).url(),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            Backend::Anthropic(config("https://api.anthropic.com/v1/messages")).url(),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn count_tokens_url_joins_every_base_url_shape() {
        assert_eq!(
            Backend::Anthropic(config("https://host")).count_tokens_url(),
            "https://host/v1/messages/count_tokens"
        );
        assert_eq!(
            Backend::Anthropic(config("https://host/v1")).count_tokens_url(),
            "https://host/v1/messages/count_tokens"
        );
        assert_eq!(
            Backend::Anthropic(config("https://host/v1/messages")).count_tokens_url(),
            "https://host/v1/messages/count_tokens"
        );
        // Trailing slash is trimmed before the join.
        assert_eq!(
            Backend::Anthropic(config("https://host/v1/")).count_tokens_url(),
            "https://host/v1/messages/count_tokens"
        );
    }

    #[test]
    fn only_anthropic_backend_is_anthropic() {
        assert!(Backend::Anthropic(config("x")).is_anthropic());
        assert!(!Backend::OpenAiChat(config("x")).is_anthropic());
        assert!(!Backend::OpenAiResponses(config("x")).is_anthropic());
    }

    #[test]
    fn wire_format_matches_variant() {
        assert_eq!(
            Backend::OpenAiChat(config("x")).wire_format(),
            WireFormat::OpenAiChat
        );
        assert_eq!(
            Backend::OpenAiResponses(config("x")).wire_format(),
            WireFormat::OpenAiResponses
        );
        assert_eq!(
            Backend::Anthropic(config("x")).wire_format(),
            WireFormat::AnthropicMessages
        );
    }

    #[test]
    fn openai_detects_canonical_and_wrapped_overflow() {
        let backend = Backend::OpenAiChat(config("x"));
        assert!(
            backend.is_context_overflow(
                r#"{"error":{"code":"context_length_exceeded","message":"x"}}"#
            )
        );
        // NVIDIA/LiteLLM message wrap with no structured code.
        assert!(backend.is_context_overflow(
            r#"{"error":{"message":"the model's context length is only 131072 tokens"}}"#
        ));
        assert!(!backend.is_context_overflow(r#"{"error":{"code":"invalid_api_key"}}"#));
        // Hub GLM (LiteLLM-wrapped): code is "400", detection relies on phrase match.
        assert!(backend.is_context_overflow(
            r#"{"error":{"message":"Input length 877338 exceeds the maximum allowed input length of 639968 tokens","code":"400"}}"#
        ));
    }

    #[test]
    fn anthropic_detects_prompt_too_long() {
        let backend = Backend::Anthropic(config("x"));
        assert!(
            backend.is_context_overflow(
                r#"{"error":{"message":"prompt is too long: 200000 tokens"}}"#
            )
        );
        assert!(!backend.is_context_overflow(r#"{"error":{"message":"overloaded"}}"#));
    }
}
