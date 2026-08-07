// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`TranslatingLlmClient`] — the crate's single public entry point: encode a neutral
//! request, call the configured backend over HTTP, decode the neutral response.

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::RequestBuilder;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde_json::{Map, Value};
use switchyard_protocol::{
    Context, Decision, LlmRequest, LlmResponse, Metadata, Request, Response, RoutedLlmClient,
};
use switchyard_translation::{
    WireFormat, decode_aggregated_response, decode_request, decode_stream,
    encode_aggregated_response, encode_request, encode_stream,
};
use tracing::Instrument;

use crate::backend::Backend;
use crate::error::{LlmClientError, Result};
use crate::metrics;
use crate::raw::RawResponse;

// TODO: Why is this here? What does it do?
// Headers this client owns or that are hop-by-hop; never forwarded from the
// caller's metadata. Auth/version/content-type are set by the backend or the
// JSON body, so a forwarded copy would either be ignored or conflict. Compared
// case-insensitively. Aligns with `_SENSITIVE_HEADERS` in the Python
// `switchyard/lib/request_metadata.py` forwarding logic.
const RESERVED_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "connection",
    "authorization",
    "proxy-authorization",
    "proxy-authenticate",
    "cookie",
    "set-cookie",
    "x-api-key",
    "anthropic-beta",
    "anthropic-version",
    "content-type",
    "accept-encoding",
];

const INITIAL_RETRY_DELAY: Duration = Duration::from_millis(250);
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(2);
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

/// How one model is served: the `default_backend` used when the request does not
/// pin a wire format, plus any `other_backends` reachable over additional formats.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    model_name: String,
    default_backend: Backend,
    other_backends: Option<Vec<Backend>>,
}

impl ModelConfig {
    /// A model named `model_name` served by `default_backend`, optionally reachable
    /// over additional wire formats via `other_backends`.
    pub fn new(
        model_name: impl Into<String>,
        default_backend: Backend,
        other_backends: Option<Vec<Backend>>,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            default_backend,
            other_backends,
        }
    }
}

/// A client that dispatches neutral-IR requests to per-model HTTP backends.
///
/// Construct it with a list of [`ModelConfig`]s — one per model, each naming a
/// default [`Backend`] and any additional per-format backends. Each call resolves
/// the model and wire format, encodes the request to that backend's wire format,
/// applies auth and forwarded headers, sends the HTTP request with a shared
/// [`reqwest::Client`], and decodes the response back to the neutral IR (buffered
/// or streamed).
pub struct TranslatingLlmClient {
    model_to_config: HashMap<String, ModelConfig>,
    client: reqwest::Client,
}

impl TranslatingLlmClient {
    /// Builds a client over the given [`ModelConfig`]s, with a fresh shared HTTP
    /// client and the built-in translation codecs.
    pub fn new(model_configs: &[ModelConfig]) -> Result<Self> {
        let client =
            reqwest::Client::builder()
                .build()
                .map_err(|error| LlmClientError::Transport {
                    source: Box::new(error),
                })?;
        let model_to_config = model_configs
            .iter()
            .map(|config| (config.model_name.clone(), config.clone()))
            .collect();

        Ok(Self {
            model_to_config,
            client,
        })
    }

    /// The backend serving `model` over `format` — the default backend when its
    /// format matches, otherwise a matching entry in `other_backends`; `None` when
    /// the model is unknown or has no backend for `format`.
    pub fn backend_for(&self, model: &str, format: WireFormat) -> Option<&Backend> {
        self.model_to_config.get(model).and_then(|config| {
            if config.default_backend.wire_format() == format {
                Some(&config.default_backend)
            } else {
                config
                    .other_backends
                    .as_ref()
                    .and_then(|backends| backends.iter().find(|b| b.wire_format() == format))
            }
        })
    }

    /// Whether `model` has an Anthropic backend that supports token counting.
    pub fn supports_count_tokens(&self, model: &str) -> bool {
        self.backend_for(model, WireFormat::AnthropicMessages)
            .is_some()
    }

    /// Counts input tokens with `model`'s Anthropic backend.
    ///
    /// Returns an error when the model has no Anthropic backend or the upstream
    /// request fails or returns invalid JSON.
    pub async fn count_tokens(&self, model: &str, request: Request) -> Result<Value> {
        let backend = self
            .backend_for(model, WireFormat::AnthropicMessages)
            .ok_or_else(|| LlmClientError::Configuration {
                message: format!("model {model} has no Anthropic backend for count_tokens"),
            })?;
        let Request {
            llm_request,
            metadata,
            ..
        } = request;
        let http_response = self
            .send_encoded(
                backend,
                WireFormat::AnthropicMessages,
                llm_request,
                metadata.as_ref(),
                model,
                UpstreamEndpoint::CountTokens,
            )
            .await?;
        let body = match http_response {
            EncodedResponse::Buffered { body, .. } => body,
            EncodedResponse::Streaming(_) => {
                return Err(LlmClientError::InvalidRequest {
                    message: "count_tokens does not support streaming requests".to_string(),
                });
            }
        };
        serde_json::from_slice(&body).map_err(|error| LlmClientError::InvalidResponse {
            source: Box::new(error),
        })
    }

    /// Encode `llm_request` (its model restamped to `model`) for `wire_format`,
    /// POST it to `url` with the request's forwarded headers plus the backend's
    /// static headers and auth, and return the successful upstream response. A
    /// buffered response is fully collected within the retry boundary; a streamed
    /// response is returned as soon as its successful headers arrive. A non-success
    /// status maps to a typed error — a 400 is classified as a context-window
    /// overflow via the backend's provider rules. Shared by
    /// [`call_rewrite_model`](Self::call_rewrite_model) (which POSTs to the
    /// backend's completion URL and decodes a response) and
    /// [`count_tokens`](Self::count_tokens) (which POSTs to the `count_tokens`
    /// URL and returns the raw JSON).
    async fn send_encoded(
        &self,
        backend: &Backend,
        wire_format: WireFormat,
        mut llm_request: LlmRequest,
        metadata: Option<&Metadata>,
        model: &str,
        endpoint: UpstreamEndpoint,
    ) -> Result<EncodedResponse> {
        // The resolved name is the upstream model id (per the crate contract).
        llm_request.model = Some(model.to_string());
        let mut body = encode_request(&llm_request, wire_format)
            .map_err(|error| LlmClientError::RequestEncoding(error.to_string()))?;
        // `encode_request` round-trips a preserved same-format body verbatim,
        // which keeps the caller's original `model`; force the resolved model so
        // the upstream always sees the target id.
        set_json_model(&mut body, model);
        // Strip before `merge_extra_body` so a target can reinstate either field
        // deliberately via `extra_body`.
        if matches!(backend, Backend::Anthropic(_)) {
            strip_anthropic_incompatible_fields(&mut body);
            strip_unsigned_thinking_blocks(&mut body);
        }
        merge_extra_body(&mut body, backend.extra_body());
        if matches!(backend, Backend::Anthropic(_)) {
            enable_anthropic_prompt_caching(&mut body);
        }
        if matches!(backend, Backend::OpenAiChat(_)) {
            ensure_openai_stream_usage(&mut body);
        }
        let streaming = endpoint.allows_streaming()
            && body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let url = endpoint.url(backend);
        record_gen_ai_request(&url, model, streaming);

        let max_retries = u64::from(backend.max_retries());
        let max_attempts = max_retries + 1;
        let mut attempt = 0_u64;
        loop {
            let span = tracing::debug_span!(
                target: "libsy",
                "libsy.upstream_attempt",
                model,
                wire_format = %wire_format,
                attempt = attempt + 1,
                max_attempts,
                retry = attempt > 0,
                openinference.span.kind = "CHAIN",
                outcome = tracing::field::Empty,
                status_code = tracing::field::Empty,
                will_retry = tracing::field::Empty,
                retry_delay_ms = tracing::field::Empty,
            );
            let result = self
                .send_once(&url, backend, &body, metadata, model, streaming)
                .instrument(span.clone())
                .await;
            // The retained handle updates this same attempt span with its outcome.
            match result {
                Ok(response) => {
                    span.record("outcome", "success");
                    span.record("status_code", response.status());
                    span.record("will_retry", false);
                    return Ok(response);
                }
                Err(failure) => {
                    let will_retry = attempt < max_retries && failure.is_retryable();
                    span.record("outcome", "error");
                    if let Some(status) = failure.status {
                        span.record("status_code", status);
                    }
                    span.record("will_retry", will_retry);
                    if !will_retry {
                        return Err(failure.error);
                    }

                    let delay = retry_delay(attempt, failure.retry_after);
                    span.record("retry_delay_ms", duration_millis(delay));
                    // Close the attempt span before sleeping so backoff is not attempt latency.
                    drop(span);
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }

    // Performs one HTTP attempt and retains the retry metadata alongside any error.
    async fn send_once(
        &self,
        url: &str,
        backend: &Backend,
        body: &Value,
        metadata: Option<&Metadata>,
        model: &str,
        streaming: bool,
    ) -> std::result::Result<EncodedResponse, AttemptFailure> {
        let builder = self.client.post(url).json(body);
        let builder = forward_metadata_headers(builder, metadata);
        let builder = apply_extra_headers(builder, backend);
        let builder = backend.apply_auth(builder);

        let response = match builder.send().await {
            Ok(response) => response,
            Err(error) => {
                metrics::record_upstream_attempt(None);
                return Err(AttemptFailure {
                    error: convert_reqwest_error(error),
                    status: None,
                    retry_after: None,
                });
            }
        };
        let status = response.status();
        if status.is_success() {
            if streaming {
                // Streaming body failures happen after the retry boundary.
                metrics::record_upstream_attempt(Some(status.as_u16()));
                return Ok(EncodedResponse::Streaming(response));
            }
            let body = match response.bytes().await {
                Ok(body) => body,
                Err(error) => {
                    metrics::record_upstream_attempt(None);
                    return Err(AttemptFailure {
                        error: convert_reqwest_error(error),
                        status: Some(status.as_u16()),
                        retry_after: None,
                    });
                }
            };
            metrics::record_upstream_attempt(Some(status.as_u16()));
            return Ok(EncodedResponse::Buffered {
                status: status.as_u16(),
                body: body.to_vec(),
            });
        }

        let retry_after = retry_after_delay(response.headers());
        let body = match response.text().await {
            Ok(body) => body,
            Err(error) => {
                metrics::record_upstream_attempt(None);
                return Err(AttemptFailure {
                    error: convert_reqwest_error(error),
                    status: Some(status.as_u16()),
                    retry_after,
                });
            }
        };
        metrics::record_upstream_attempt(Some(status.as_u16()));
        let error =
            if status == reqwest::StatusCode::BAD_REQUEST && backend.is_context_overflow(&body) {
                LlmClientError::ContextWindowExceeded {
                    model: model.to_string(),
                    message: body,
                }
            } else {
                LlmClientError::UpstreamHttp {
                    status: status.as_u16(),
                    body,
                }
            };
        Err(AttemptFailure {
            error,
            status: Some(status.as_u16()),
            retry_after,
        })
    }

    /// Calls the backend for `model_name` (or the request's own model), over the
    /// wire format the request pins in its metadata (else the model's default
    /// backend), and returns the neutral response.
    ///
    /// Resolution: `model_name` wins over `request.llm_request.model`; the
    /// resolved name is both the outer map key and the model id written into the
    /// request before translation. Missing models are invalid requests; unknown
    /// models or wire formats are configuration errors.
    pub async fn call_rewrite_model(
        &self,
        _ctx: Context,
        request: Request,
        model_name: Option<&str>,
    ) -> Result<Response> {
        // Own the request's parts so the model can be set without a `mut` param
        // and without cloning the messages. `raw_request` is unused here.
        let Request {
            llm_request,
            metadata,
            ..
        } = request;

        let model = model_name
            .map(str::to_string)
            .or_else(|| llm_request.model.clone())
            .ok_or_else(|| LlmClientError::InvalidRequest {
                message: "no model given".to_string(),
            })?;

        let orig_format = metadata.as_ref().and_then(|m| m.wire_format);
        let wire_format = orig_format.unwrap_or(
            self.model_to_config
                .get(&model)
                .map(|config| config.default_backend.wire_format())
                .ok_or_else(|| LlmClientError::Configuration {
                    message: format!("no backend configured for model {model:?}"),
                })?,
        );
        let backend =
            self.backend_for(&model, wire_format)
                .ok_or_else(|| LlmClientError::Configuration {
                    message: format!("model {model:?} has no backend for format {wire_format}"),
                })?;

        let http_response = self
            .send_encoded(
                backend,
                wire_format,
                llm_request,
                metadata.as_ref(),
                &model,
                UpstreamEndpoint::Completion,
            )
            .await?;

        let llm_response = match http_response {
            EncodedResponse::Streaming(http_response) => {
                // Adapt the reqwest body stream to plain bytes; the SSE-decode itself is
                // transport-agnostic and lives in `switchyard-translation`.
                let bytes = http_response.bytes_stream().map(|chunk| {
                    chunk.map(|bytes| bytes.to_vec()).map_err(|error| {
                        if error.is_timeout() {
                            LlmClientError::Timeout {
                                source: Box::new(error),
                            }
                        } else {
                            LlmClientError::Transport {
                                source: Box::new(error),
                            }
                        }
                    })
                });
                let chunks = decode_stream(bytes, wire_format)?;
                LlmResponse::Stream(chunks)
            }
            EncodedResponse::Buffered { body, .. } => {
                let body = serde_json::from_slice::<Value>(&body).map_err(|error| {
                    LlmClientError::ResponseTranslation(format!("invalid upstream JSON: {error}"))
                })?;
                let agg = decode_aggregated_response(&body, wire_format)
                    .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
                LlmResponse::Agg(agg)
            }
        };

        Ok(Response {
            llm_response,
            metadata,
        })
    }

    /// The whole decode → call → encode path a wire endpoint needs, in one call.
    ///
    /// Decodes `raw_http_request` from `wire_format` to the neutral IR, serves it via
    /// [`call_rewrite_model`](Self::call_rewrite_model) — the *upstream* wire format is
    /// resolved there from the model's backend, independently of `wire_format` — then
    /// encodes the neutral response back into `wire_format`. The result is a buffered
    /// [`RawResponse::Buffered`] JSON body or a streamed [`RawResponse::Stream`] of
    /// wire events (the caller frames the stream as SSE). The response's `model` is
    /// restamped with the model that actually served the call, so the body names the
    /// model that answered rather than the route the caller addressed.
    ///
    /// `http_headers` are carried through as the request's
    /// [`Metadata::http_headers`] and forwarded to the upstream (minus the reserved
    /// set); pass `None` to forward nothing.
    pub async fn call_rewrite_model_raw(
        &self,
        ctx: Context,
        raw_http_request: Value,
        http_headers: Option<http::HeaderMap>,
        model: Option<&str>,
        wire_format: WireFormat,
    ) -> Result<RawResponse> {
        let llm_request = decode_request(wire_format, &raw_http_request)
            .map_err(|error| LlmClientError::RequestTranslation(error.to_string()))?;
        // The model that serves the call — the rewrite target when the caller pinned
        // one, else the request's own model. Mirrors `call_rewrite_model`'s own
        // resolution so the response names whoever answered.
        let served_model = model
            .map(str::to_string)
            .or_else(|| llm_request.model.clone());

        let request = Request {
            llm_request,
            raw_request: None,
            metadata: Some(Metadata {
                session_id: None,
                agent_id: None,
                task_id: None,
                correlation_id: None,
                extra_metadata: None,
                http_headers,
                wire_format: None,
                ..Default::default()
            }),
        };
        let response = self.call_rewrite_model(ctx, request, model).await?;

        match response.llm_response {
            LlmResponse::Agg(agg) => {
                let body =
                    encode_aggregated_response(&agg, wire_format, served_model.as_deref())
                        .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
                Ok(RawResponse::Buffered(body))
            }
            LlmResponse::Stream(chunks) => {
                let events = encode_stream(chunks, wire_format, served_model)?;
                Ok(RawResponse::Stream(events))
            }
        }
    }
}

#[async_trait]
impl RoutedLlmClient for TranslatingLlmClient {
    async fn call(
        &self,
        ctx: Context,
        request: Request,
        decision: std::sync::Arc<dyn Decision>,
    ) -> Result<Response> {
        let model_name = Some(decision.selected_model());
        self.call_rewrite_model(ctx, request, model_name).await
    }
}

#[derive(Clone, Copy)]
enum UpstreamEndpoint {
    Completion,
    CountTokens,
}

impl UpstreamEndpoint {
    fn url(self, backend: &Backend) -> String {
        match self {
            UpstreamEndpoint::Completion => backend.url(),
            UpstreamEndpoint::CountTokens => backend.count_tokens_url(),
        }
    }

    fn allows_streaming(self) -> bool {
        matches!(self, UpstreamEndpoint::Completion)
    }
}

enum EncodedResponse {
    Buffered { status: u16, body: Vec<u8> },
    Streaming(reqwest::Response),
}

impl EncodedResponse {
    fn status(&self) -> u16 {
        match self {
            EncodedResponse::Buffered { status, .. } => *status,
            EncodedResponse::Streaming(response) => response.status().as_u16(),
        }
    }
}

// The typed error decides retry eligibility; status and Retry-After feed
// attempt telemetry and delay selection.
struct AttemptFailure {
    error: LlmClientError,
    status: Option<u16>,
    retry_after: Option<Duration>,
}

impl AttemptFailure {
    fn is_retryable(&self) -> bool {
        match &self.error {
            LlmClientError::Transport { .. } | LlmClientError::Timeout { .. } => true,
            LlmClientError::UpstreamHttp { status, .. } => {
                metrics::is_retryable_http_status(*status)
            }
            _ => false,
        }
    }
}

// Uses Retry-After when supplied, capped so an upstream cannot stall a request indefinitely.
fn retry_after_delay(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?;
    let delay = if let Ok(seconds) = value.parse::<u64>() {
        Duration::from_secs(seconds)
    } else {
        let retry_at = httpdate::parse_http_date(value).ok()?;
        retry_at
            .duration_since(SystemTime::now())
            .unwrap_or(Duration::ZERO)
    };
    Some(delay.min(MAX_RETRY_AFTER))
}

fn retry_delay(retry_number: u64, retry_after: Option<Duration>) -> Duration {
    // Retry-After wins; otherwise double 250 ms up to the two-second cap.
    retry_after.unwrap_or_else(|| {
        let multiplier = 1_u32 << retry_number.min(3);
        INITIAL_RETRY_DELAY
            .saturating_mul(multiplier)
            .min(MAX_RETRY_BACKOFF)
    })
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn record_gen_ai_request(url: &str, model: &str, streaming: bool) {
    let span = tracing::Span::current();
    span.record("gen_ai.request.model", model);
    if streaming {
        span.record("gen_ai.request.stream", true);
    }
    if let Ok(url) = reqwest::Url::parse(url) {
        if let Some(host) = url.host_str() {
            span.record("server.address", host);
        }
        if let Some(port) = url.port_or_known_default() {
            span.record("server.port", i64::from(port));
        }
    }
}

fn convert_reqwest_error(error: reqwest::Error) -> LlmClientError {
    // Reqwest labels truncated or otherwise unreadable response bodies as decode
    // errors, so distinguish them from serde JSON failures at the call site.
    if error.is_timeout() {
        LlmClientError::Timeout {
            source: Box::new(error),
        }
    } else if error.is_builder() {
        LlmClientError::Configuration {
            message: format!("failed to build upstream request: {error}"),
        }
    } else {
        LlmClientError::Transport {
            source: Box::new(error),
        }
    }
}

// Forwards caller-supplied metadata headers, skipping the reserved set.
fn forward_metadata_headers(
    mut builder: RequestBuilder,
    metadata: Option<&Metadata>,
) -> RequestBuilder {
    let Some(headers) = metadata.and_then(|metadata| metadata.http_headers.as_ref()) else {
        return builder;
    };
    for (name, value) in headers {
        if is_reserved_header(name.as_str()) {
            continue;
        }
        builder = builder.header(name, value);
    }
    builder
}

// Adds the backend's static per-call headers.
fn apply_extra_headers(mut builder: RequestBuilder, backend: &Backend) -> RequestBuilder {
    for (name, value) in backend.extra_headers() {
        builder = builder.header(name, value);
    }
    builder
}

// Overwrites the outbound body's `model` field with the resolved model id.
fn set_json_model(body: &mut Value, model: &str) {
    if let Value::Object(object) = body {
        object.insert("model".to_string(), Value::String(model.to_string()));
    }
}

// Drops fields accepted by OpenAI-like APIs but rejected by Anthropic Messages.
//
// A router can serve earlier turns of a session from an OpenAI-format target and
// later turns from an Anthropic one. Clients such as Claude Code send
// `context_management` on every turn, so the Anthropic leg must strip it or the
// upstream rejects the request (for example `clear_thinking_20251015` strategy
// requires `thinking` to be enabled or adaptive). Mirrors
// `switchyard-components`' `strip_anthropic_incompatible_fields`.
fn strip_anthropic_incompatible_fields(body: &mut Value) {
    if let Value::Object(object) = body {
        object.remove("reasoning_effort");
        object.remove("context_management");
    }
}

// Removes replayed `thinking` blocks that carry no signature.
//
// Anthropic requires signed thinking blocks on replay. A router can serve earlier
// turns of a session from an OpenAI-format target whose thinking blocks are
// unsigned, so the Anthropic leg must drop them or the upstream rejects the
// request. Bedrock enforces this (surfacing as a SigV4 signature mismatch) where
// Azure-hosted Anthropic currently does not. Mirrors `switchyard-components`'
// `strip_unsigned_thinking_blocks`.
fn strip_unsigned_thinking_blocks(body: &mut Value) {
    let Value::Object(object) = body else {
        return;
    };
    let Some(Value::Array(messages)) = object.get_mut("messages") else {
        return;
    };
    for message in messages {
        strip_unsigned_thinking_from_message(message);
    }
}

// Drops unsigned thinking blocks from one message, collapsing content that ends
// up empty to an empty string so the message stays valid.
fn strip_unsigned_thinking_from_message(message: &mut Value) {
    let Value::Object(message) = message else {
        return;
    };
    let Some(Value::Array(blocks)) = message.get("content") else {
        return;
    };
    if !blocks.iter().any(is_unsigned_thinking_block) {
        return;
    }
    let Some(Value::Array(blocks)) = message.get_mut("content") else {
        return;
    };
    blocks.retain(|block| !is_unsigned_thinking_block(block));
    if blocks.is_empty() {
        message.insert("content".to_string(), Value::String(String::new()));
    }
}

// A thinking block is unsigned when `signature` is absent or empty.
fn is_unsigned_thinking_block(block: &Value) -> bool {
    if block.get("type").and_then(Value::as_str) != Some("thinking") {
        return false;
    }
    !matches!(
        block.get("signature").and_then(Value::as_str),
        Some(signature) if !signature.is_empty()
    )
}

// Applies target defaults without overriding fields supplied by the caller.
fn merge_extra_body(body: &mut Value, extra_body: &BTreeMap<String, Value>) {
    let Value::Object(object) = body else {
        return;
    };
    for (key, value) in extra_body {
        object.entry(key.clone()).or_insert_with(|| value.clone());
    }
}

// Marks the final message content block as the Anthropic prompt-cache breakpoint.
fn enable_anthropic_prompt_caching(body: &mut Value) {
    let Some(content) = body
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|messages| messages.last_mut())
        .and_then(|message| message.get_mut("content"))
    else {
        return;
    };
    match content {
        Value::String(text) => {
            *content = serde_json::json!([{
                "type": "text",
                "text": std::mem::take(text),
                "cache_control": {"type": "ephemeral"}
            }]);
        }
        Value::Array(blocks) => {
            if let Some(block) = blocks.last_mut().and_then(Value::as_object_mut) {
                block
                    .entry("cache_control".to_string())
                    .or_insert_with(|| serde_json::json!({"type": "ephemeral"}));
            }
        }
        _ => {}
    }
}

// Requests streamed Chat usage by default while preserving an explicit caller choice.
fn ensure_openai_stream_usage(body: &mut Value) {
    let Value::Object(object) = body else {
        return;
    };
    if object.get("stream").and_then(Value::as_bool) != Some(true) {
        return;
    }

    match object.get_mut("stream_options") {
        Some(Value::Object(options)) => {
            options
                .entry("include_usage".to_string())
                .or_insert(Value::Bool(true));
        }
        _ => {
            let mut options = Map::new();
            options.insert("include_usage".to_string(), Value::Bool(true));
            object.insert("stream_options".to_string(), Value::Object(options));
        }
    }
}

// Case-insensitive membership test against RESERVED_HEADERS.
fn is_reserved_header(name: &str) -> bool {
    RESERVED_HEADERS
        .iter()
        .any(|reserved| name.eq_ignore_ascii_case(reserved))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::error::Error;
    use std::io::{Read, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread::JoinHandle;

    use serde_json::json;
    use switchyard_protocol::{LlmRequest, completion_text, text_request};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::backend::HttpBackendConfig;

    fn config(base_url: &str) -> HttpBackendConfig {
        HttpBackendConfig {
            base_url: base_url.to_string(),
            api_key: Some("secret".to_string()),
            extra_headers: BTreeMap::new(),
            extra_body: BTreeMap::new(),
            max_retries: 0,
        }
    }

    fn config_with_retries(base_url: &str, max_retries: u32) -> HttpBackendConfig {
        HttpBackendConfig {
            max_retries,
            ..config(base_url)
        }
    }

    // A one-model config list: "gpt" served over OpenAI Chat at base_url.
    fn chat_map(base_url: &str) -> Vec<ModelConfig> {
        vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(config(base_url)),
            None,
        )]
    }

    fn chat_map_with_extra_body(
        base_url: &str,
        extra_body: BTreeMap<String, Value>,
    ) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        backend.extra_body = extra_body;
        vec![ModelConfig::new("gpt", Backend::OpenAiChat(backend), None)]
    }

    fn anthropic_map(base_url: &str) -> Vec<ModelConfig> {
        vec![ModelConfig::new(
            "claude",
            Backend::Anthropic(config(base_url)),
            None,
        )]
    }

    fn chat_map_with_retries(base_url: &str, max_retries: u32) -> Vec<ModelConfig> {
        vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(config_with_retries(base_url, max_retries)),
            None,
        )]
    }

    fn chat_success_response() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "model": "gpt",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "recovered"},
                "finish_reason": "stop"
            }],
            "usage": {}
        }))
    }

    fn truncated_response_server(
        content_type: &str,
        body: &str,
    ) -> std::io::Result<(String, JoinHandle<std::io::Result<()>>)> {
        response_sequence_server(vec![format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len() + 100
        )])
    }

    fn response_sequence_server(
        responses: Vec<String>,
    ) -> std::io::Result<(String, JoinHandle<std::io::Result<()>>)> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let handle = std::thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept()?;
                let mut request = [0_u8; 1024];
                if stream.read(&mut request)? == 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "client closed before sending a request",
                    ));
                }
                stream.write_all(response.as_bytes())?;
            }
            Ok(())
        });
        Ok((format!("http://{address}/v1"), handle))
    }

    fn raw_chat_success_response() -> String {
        let body = r#"{"id":"chatcmpl-1","model":"gpt","choices":[{"index":0,"message":{"role":"assistant","content":"recovered"},"finish_reason":"stop"}],"usage":{}}"#;
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn request_for(model: Option<&str>, stream: bool) -> Request {
        let mut llm_request = text_request(model.map(str::to_string), "hi");
        llm_request.stream = stream;
        Request {
            llm_request,
            raw_request: None,
            metadata: None,
        }
    }

    #[test]
    fn anthropic_prompt_caching_marks_final_message() {
        let mut body = json!({
            "messages": [{"role": "user", "content": "hello"}]
        });

        enable_anthropic_prompt_caching(&mut body);

        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    // A request that pins `format` in its metadata, so the client resolves that
    // wire format instead of the model's default backend.
    fn request_with_wire_format(model: &str, format: WireFormat) -> Request {
        let mut request = request_for(Some(model), false);
        request.metadata = Some(Metadata {
            session_id: None,
            agent_id: None,
            task_id: None,
            correlation_id: None,
            extra_metadata: None,
            http_headers: None,
            wire_format: Some(format),
            ..Default::default()
        });
        request
    }

    #[tokio::test]
    async fn missing_model_errors()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let client = TranslatingLlmClient::new(&[])?;
        let Err(error) = client
            .call_rewrite_model(Context::default(), request_for(None, false), None)
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::InvalidRequest { message } if message == "no model given"
        ));
        Ok(())
    }

    #[tokio::test]
    async fn unknown_model_errors()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let client = TranslatingLlmClient::new(&[])?;
        let Err(error) = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::Configuration { message }
                if message.contains("gpt")
        ));
        Ok(())
    }

    #[tokio::test]
    async fn unknown_model_format_errors()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // "gpt" exists but only over OpenAI Chat; the request pins Anthropic.
        let client = TranslatingLlmClient::new(&chat_map("https://example.test/v1"))?;
        let Err(error) = client
            .call_rewrite_model(
                Context::default(),
                request_with_wire_format("gpt", WireFormat::AnthropicMessages),
                None,
            )
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::Configuration { message }
                if message.contains("gpt")
                    && message.contains(&WireFormat::AnthropicMessages.to_string())
        ));
        Ok(())
    }

    #[test]
    fn backend_for_resolves_configured_format()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let client = TranslatingLlmClient::new(&chat_map("https://example.test/v1"))?;
        // "gpt" is served over OpenAI Chat only; other formats and models miss.
        assert!(client.backend_for("gpt", WireFormat::OpenAiChat).is_some());
        assert!(
            client
                .backend_for("gpt", WireFormat::AnthropicMessages)
                .is_none()
        );
        assert!(
            client
                .backend_for("missing", WireFormat::OpenAiChat)
                .is_none()
        );
        Ok(())
    }

    #[tokio::test]
    async fn model_name_arg_wins_over_request_model()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let client = TranslatingLlmClient::new(&[])?;
        // Arg "b" is looked up (and reported), not the request's "a".
        let Err(error) = client
            .call_rewrite_model(Context::default(), request_for(Some("a"), false), Some("b"))
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::Configuration { message }
                if message.contains("\"b\"")
        ));
        Ok(())
    }

    #[tokio::test]
    async fn buffered_openai_chat_round_trips()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;

        let response = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), false), None)
            .await?;
        let agg = response.llm_response.into_agg().await?;
        assert_eq!(completion_text(&agg), "Hi there");

        Ok(())
    }

    #[tokio::test]
    async fn invalid_json_is_a_response_translation_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                observed_calls.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(200).set_body_raw("not json", "application/json")
            })
            .mount(&server)
            .await;

        let client =
            TranslatingLlmClient::new(&chat_map_with_retries(&format!("{}/v1", server.uri()), 2))?;
        let Err(error) = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected invalid JSON to fail");
        };

        assert!(matches!(
            error,
            LlmClientError::ResponseTranslation(message)
                if message.contains("invalid upstream JSON")
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn response_body_io_failure_is_a_transport_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let (base_url, server) = truncated_response_server("application/json", "{}")?;
        let client = TranslatingLlmClient::new(&chat_map(&base_url))?;
        let result = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), false), None)
            .await;
        server
            .join()
            .map_err(|_| std::io::Error::other("response server thread panicked"))??;

        let Err(error) = result else {
            panic!("expected the truncated response body to fail");
        };
        let LlmClientError::Transport { source } = error else {
            panic!("expected a transport error");
        };
        let Some(source) = source.downcast_ref::<reqwest::Error>() else {
            panic!("expected the reqwest transport source");
        };
        assert!(source.is_decode());
        assert!(
            !std::error::Error::source(&source)
                .is_some_and(|source| source.is::<serde_json::Error>())
        );
        Ok(())
    }

    #[tokio::test]
    async fn response_body_transport_failures_are_retried()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let truncated_responses = [
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
             Content-Length: 102\r\nConnection: close\r\n\r\n{}",
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: text/plain\r\n\
             Content-Length: 100\r\nConnection: close\r\n\r\nbad",
        ];

        for truncated in truncated_responses {
            let (base_url, server) =
                response_sequence_server(vec![truncated.to_string(), raw_chat_success_response()])?;
            let client = TranslatingLlmClient::new(&chat_map_with_retries(&base_url, 1))?;
            let response = client
                .call_rewrite_model(Context::default(), request_for(Some("gpt"), false), None)
                .await?;
            server
                .join()
                .map_err(|_| std::io::Error::other("response server thread panicked"))??;

            assert_eq!(
                completion_text(&response.llm_response.into_agg().await?),
                "recovered"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn streaming_body_io_failure_preserves_transport_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n";
        let (base_url, server) = truncated_response_server("text/event-stream", body)?;
        let client = TranslatingLlmClient::new(&chat_map_with_retries(&base_url, 2))?;
        let response = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), true), None)
            .await?;
        let result = response.llm_response.into_agg().await;
        server
            .join()
            .map_err(|_| std::io::Error::other("response server thread panicked"))??;

        let Err(error) = result else {
            panic!("expected the truncated stream body to fail");
        };

        assert!(matches!(error, LlmClientError::Transport { .. }));
        Ok(())
    }

    #[tokio::test]
    async fn rewrites_model_to_resolved_upstream_id()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // Inbound body says "switchyard"; the upstream must receive "gpt".
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({"model": "gpt"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1", "model": "gpt",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        // Inbound model differs from the map key / resolved model.
        client
            .call_rewrite_model(
                Context::default(),
                request_for(Some("switchyard"), false),
                Some("gpt"),
            )
            .await?;
        // The body_partial_json matcher asserts the upstream saw model "gpt".
        Ok(())
    }

    #[tokio::test]
    async fn extra_body_adds_defaults_without_overriding_the_request()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "model": "gpt",
                "max_tokens": 7,
                "service_tier": "priority"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let extra_body = BTreeMap::from([
            ("max_tokens".to_string(), json!(999)),
            ("service_tier".to_string(), json!("priority")),
        ]);
        let client = TranslatingLlmClient::new(&chat_map_with_extra_body(
            &format!("{}/v1", server.uri()),
            extra_body,
        ))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 7
        });

        client
            .call_rewrite_model_raw(
                Context::default(),
                raw,
                None,
                Some("gpt"),
                WireFormat::OpenAiChat,
            )
            .await?;
        Ok(())
    }

    // A weak OpenAI-format tier emits thinking blocks with no signature. Replaying
    // them to Anthropic is rejected (Bedrock reports it as a SigV4 mismatch), so
    // the Anthropic leg must drop them while keeping signed ones.
    #[tokio::test]
    async fn anthropic_requests_drop_unsigned_thinking_blocks()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(|request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                let messages = body.get("messages").and_then(Value::as_array).cloned();
                let Some(messages) = messages else {
                    return false;
                };
                // The unsigned block is gone, the signed one survives, and the
                // message whose only block was unsigned is not left with an empty
                // content array.
                let blocks: Vec<&Value> = messages
                    .iter()
                    .filter_map(|message| message.get("content"))
                    .filter_map(Value::as_array)
                    .flatten()
                    .collect();
                let thinking: Vec<&&Value> = blocks
                    .iter()
                    .filter(|block| block.get("type").and_then(Value::as_str) == Some("thinking"))
                    .collect();
                thinking.len() == 1
                    && thinking[0].get("signature").and_then(Value::as_str) == Some("sig-abc")
                    && messages
                        .iter()
                        .all(|message| message.get("content") != Some(&json!([])))
            })
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&anthropic_map(&server.uri()))?;
        let raw = json!({
            "model": "client-facing",
            "max_tokens": 7,
            "messages": [
                {"role": "user", "content": "fix the build"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "weak tier reasoning", "signature": ""}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "signed reasoning", "signature": "sig-abc"},
                    {"type": "text", "text": "here goes"}
                ]},
                {"role": "user", "content": "continue"}
            ]
        });

        client
            .call_rewrite_model_raw(
                Context::default(),
                raw,
                None,
                Some("claude"),
                WireFormat::AnthropicMessages,
            )
            .await?;
        Ok(())
    }

    // A router can serve earlier turns from an OpenAI target and later turns from
    // an Anthropic one, so the Anthropic leg must drop OpenAI-only fields the
    // caller keeps sending or the upstream rejects the whole request.
    #[tokio::test]
    async fn anthropic_requests_drop_openai_only_fields()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(|request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                body.get("context_management").is_none() && body.get("reasoning_effort").is_none()
            })
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&anthropic_map(&server.uri()))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 7,
            "reasoning_effort": "high",
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015"}]
            }
        });

        client
            .call_rewrite_model_raw(
                Context::default(),
                raw,
                None,
                Some("claude"),
                WireFormat::AnthropicMessages,
            )
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn streaming_openai_chat_aggregates()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
             data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n\
             data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "stream": true,
                "stream_options": {"include_usage": true}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;

        let response = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), true), None)
            .await?;
        assert!(matches!(response.llm_response, LlmResponse::Stream(_)));
        let agg = response.llm_response.into_agg().await?;
        assert_eq!(completion_text(&agg), "Hello world");
        assert_eq!(agg.usage.input_tokens, Some(1));
        assert_eq!(agg.usage.output_tokens, Some(2));
        assert_eq!(agg.usage.total_tokens, Some(3));
        Ok(())
    }

    #[tokio::test]
    async fn streaming_openai_chat_preserves_usage_opt_out()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "stream": true,
                "stream_options": {"include_usage": false}
            })))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw("data: [DONE]\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": false}
        });

        let response = client
            .call_rewrite_model_raw(
                Context::default(),
                raw,
                None,
                Some("gpt"),
                WireFormat::OpenAiChat,
            )
            .await?;
        assert!(matches!(response, RawResponse::Stream(_)));
        Ok(())
    }

    #[tokio::test]
    async fn upstream_500_is_upstream_http()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;

        let Err(error) = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::UpstreamHttp { status: 500, .. }
        ));
        Ok(())
    }

    #[tokio::test]
    async fn retryable_http_failure_recovers_within_budget()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                if observed_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    ResponseTemplate::new(503)
                        .insert_header("retry-after", "0")
                        .set_body_string("temporarily unavailable")
                } else {
                    chat_success_response()
                }
            })
            .mount(&server)
            .await;

        let client =
            TranslatingLlmClient::new(&chat_map_with_retries(&format!("{}/v1", server.uri()), 1))?;
        let response = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), false), None)
            .await?;
        let agg = response.llm_response.into_agg().await?;

        assert_eq!(completion_text(&agg), "recovered");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[tokio::test]
    async fn deterministic_http_failure_is_not_retried()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                observed_calls.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(401).set_body_string("invalid key")
            })
            .mount(&server)
            .await;

        let client =
            TranslatingLlmClient::new(&chat_map_with_retries(&format!("{}/v1", server.uri()), 2))?;
        let Err(error) = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected an upstream error");
        };

        assert!(matches!(
            error,
            LlmClientError::UpstreamHttp {
                status: 401,
                body
            } if body == "invalid key"
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn retry_exhaustion_returns_the_final_upstream_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                let attempt = observed_calls.fetch_add(1, Ordering::SeqCst) + 1;
                ResponseTemplate::new(500)
                    .insert_header("retry-after", "0")
                    .set_body_string(format!("attempt {attempt}"))
            })
            .mount(&server)
            .await;

        let client =
            TranslatingLlmClient::new(&chat_map_with_retries(&format!("{}/v1", server.uri()), 2))?;
        let Err(error) = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected retry exhaustion");
        };

        assert!(matches!(
            error,
            LlmClientError::UpstreamHttp {
                status: 500,
                body
            } if body == "attempt 3"
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        Ok(())
    }

    #[tokio::test]
    async fn timeout_is_retried_before_a_response_is_returned()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                if observed_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    ResponseTemplate::new(200).set_delay(Duration::from_millis(500))
                } else {
                    chat_success_response()
                }
            })
            .mount(&server)
            .await;

        let mut client =
            TranslatingLlmClient::new(&chat_map_with_retries(&format!("{}/v1", server.uri()), 1))?;
        client.client = reqwest::Client::builder()
            .timeout(Duration::from_millis(100))
            .build()?;
        let response = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), false), None)
            .await?;

        assert_eq!(
            completion_text(&response.llm_response.into_agg().await?),
            "recovered"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn retryable_error_classes_are_explicit() {
        let transport = AttemptFailure {
            error: LlmClientError::Transport {
                source: std::io::Error::other("disconnected").into(),
            },
            status: None,
            retry_after: None,
        };
        assert!(transport.is_retryable());

        for status in [408, 429, 500, 503, 599] {
            let failure = AttemptFailure {
                error: LlmClientError::UpstreamHttp {
                    status,
                    body: String::new(),
                },
                status: Some(status),
                retry_after: None,
            };
            assert!(failure.is_retryable(), "HTTP {status} should retry");
        }
        for status in [400, 401, 409, 600] {
            let failure = AttemptFailure {
                error: LlmClientError::UpstreamHttp {
                    status,
                    body: String::new(),
                },
                status: Some(status),
                retry_after: None,
            };
            assert!(!failure.is_retryable(), "HTTP {status} should fail fast");
        }

        let configuration = AttemptFailure {
            error: LlmClientError::Configuration {
                message: "invalid header".to_string(),
            },
            status: None,
            retry_after: None,
        };
        assert!(!configuration.is_retryable());

        let context_window = AttemptFailure {
            error: LlmClientError::ContextWindowExceeded {
                model: "gpt".to_string(),
                message: "too long".to_string(),
            },
            status: Some(400),
            retry_after: None,
        };
        assert!(!context_window.is_retryable());
    }

    #[test]
    fn retry_after_supports_seconds_and_http_dates() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, reqwest::header::HeaderValue::from_static("3"));
        assert_eq!(retry_after_delay(&headers), Some(Duration::from_secs(3)));

        let retry_at = SystemTime::now() + Duration::from_secs(2);
        let value = httpdate::fmt_http_date(retry_at);
        let Ok(value) = reqwest::header::HeaderValue::from_str(&value) else {
            panic!("formatted HTTP date should be a valid header");
        };
        headers.insert(RETRY_AFTER, value);
        let Some(delay) = retry_after_delay(&headers) else {
            panic!("HTTP date should produce a retry delay");
        };
        assert!(delay <= Duration::from_secs(2));
    }

    #[tokio::test]
    async fn routed_llm_client_exposes_timeout_variant()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_delay(std::time::Duration::from_millis(100)),
            )
            .mount(&server)
            .await;

        let mut client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        client.client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(10))
            .build()?;
        let decision: std::sync::Arc<dyn Decision> = std::sync::Arc::new(FixedDecision("gpt"));

        let Err(error) = client
            .call(Context::default(), request_for(None, false), decision)
            .await
        else {
            panic!("expected a timeout");
        };
        let LlmClientError::Timeout { source } = error else {
            panic!("expected the protocol timeout variant");
        };
        let Some(source) = source.downcast_ref::<reqwest::Error>() else {
            panic!("expected the reqwest timeout source");
        };
        assert!(source.is_timeout());
        Ok(())
    }

    #[tokio::test]
    async fn context_overflow_400_is_mapped()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"code": "context_length_exceeded", "message": "too big"}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;

        let Err(error) = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::ContextWindowExceeded { model, .. } if model == "gpt"
        ));
        Ok(())
    }

    #[tokio::test]
    async fn forwards_metadata_headers_except_reserved()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wiremock::matchers::header("x-request-id", "abc"))
            // A forwarded Authorization must NOT override the backend's bearer key.
            .and(wiremock::matchers::header("authorization", "Bearer secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1", "model": "gpt",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let mut headers = http::HeaderMap::new();
        headers.insert("x-request-id", http::HeaderValue::from_static("abc"));
        headers.insert(
            "authorization",
            http::HeaderValue::from_static("Bearer client-key"),
        );
        headers.insert(
            "accept-encoding",
            http::HeaderValue::from_static("gzip, br"),
        );
        let request = Request {
            llm_request: LlmRequest {
                model: Some("gpt".to_string()),
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: Some(Metadata {
                session_id: None,
                agent_id: None,
                task_id: None,
                correlation_id: None,
                extra_metadata: None,
                http_headers: Some(headers),
                wire_format: None,
                ..Default::default()
            }),
        };

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;

        // Matchers assert forwarded x-request-id survives and reserved
        // authorization is the backend's, not the client's.
        client
            .call_rewrite_model(Context::default(), request, None)
            .await?;
        let received = server
            .received_requests()
            .await
            .ok_or("request recording should be enabled")?;
        let received = received.first().ok_or("expected one upstream request")?;
        assert!(!received.headers.contains_key("accept-encoding"));
        Ok(())
    }

    // Minimal `Decision` for driving the client through the `RoutedLlmClient` trait.
    struct FixedDecision(&'static str);

    impl Decision for FixedDecision {
        fn selected_model(&self) -> &str {
            self.0
        }
        fn reasoning(&self) -> Option<&str> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    // Exercises the `RoutedLlmClient` impl: `call` resolves the upstream model from the
    // decision (the request carries none) and round-trips a buffered response.
    #[tokio::test]
    async fn routed_llm_client_serves_the_decision_model()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "routed hi"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let decision: std::sync::Arc<dyn Decision> = std::sync::Arc::new(FixedDecision("gpt"));
        // Called through the trait; the request has no model, so "gpt" comes from the decision.
        let response = client
            .call(Context::default(), request_for(None, false), decision)
            .await?;
        let agg = response.llm_response.into_agg().await?;
        assert_eq!(completion_text(&agg), "routed hi");
        Ok(())
    }

    #[tokio::test]
    async fn invalid_raw_request_is_a_request_translation_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let client = TranslatingLlmClient::new(&[])?;
        let Err(error) = client
            .call_rewrite_model_raw(
                Context::default(),
                json!("invalid"),
                None,
                Some("gpt"),
                WireFormat::OpenAiChat,
            )
            .await
        else {
            panic!("expected request translation to fail");
        };

        assert!(matches!(
            error,
            LlmClientError::RequestTranslation(message) if !message.is_empty()
        ));
        Ok(())
    }

    // Raw path, buffered: decode an OpenAI Chat body -> call -> encode back to OpenAI
    // Chat JSON, with the served `model` restamped over the id the caller addressed.
    #[tokio::test]
    async fn call_rewrite_model_raw_round_trips_buffered_json()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let RawResponse::Buffered(body) = client
            .call_rewrite_model_raw(
                Context::default(),
                raw,
                None,
                Some("gpt"),
                WireFormat::OpenAiChat,
            )
            .await?
        else {
            panic!("expected a buffered response");
        };

        assert_eq!(body["choices"][0]["message"]["content"], "Hi there");
        // The client sees the model that answered, not the "client-facing" route id.
        assert_eq!(body["model"], "gpt");
        Ok(())
    }

    // Raw path, streaming: an inbound `stream: true` request yields an unframed stream
    // of OpenAI Chat chunk objects whose deltas reassemble the completion.
    #[tokio::test]
    async fn call_rewrite_model_raw_streams_wire_events()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        use futures::TryStreamExt;

        let server = MockServer::start().await;
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
             data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let RawResponse::Stream(stream) = client
            .call_rewrite_model_raw(
                Context::default(),
                raw,
                None,
                Some("gpt"),
                WireFormat::OpenAiChat,
            )
            .await?
        else {
            panic!("expected a streamed response");
        };

        let events: Vec<Value> = stream.try_collect().await?;
        assert!(!events.is_empty(), "expected at least one wire event");
        let content: String = events
            .iter()
            .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(content, "Hello world");
        // The mock frames carry no `model`, so every chunk's model comes from the
        // served id rather than the "unknown" fallback or the caller's route id.
        assert!(events.iter().all(|event| event["model"] == "gpt"));
        Ok(())
    }

    // Raw path forwards caller headers (minus the reserved set) to the upstream.
    #[tokio::test]
    async fn call_rewrite_model_raw_forwards_headers()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wiremock::matchers::header("x-request-id", "abc"))
            // A forwarded authorization must NOT override the backend's bearer key.
            .and(wiremock::matchers::header("authorization", "Bearer secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1", "model": "gpt",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let mut headers = http::HeaderMap::new();
        headers.insert("x-request-id", http::HeaderValue::from_static("abc"));
        headers.insert(
            "authorization",
            http::HeaderValue::from_static("Bearer client-key"),
        );

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({"model": "gpt", "messages": [{"role": "user", "content": "hi"}]});
        // Matchers assert the forwarded x-request-id survives and reserved
        // authorization is the backend's, not the client's.
        client
            .call_rewrite_model_raw(
                Context::default(),
                raw,
                Some(headers),
                Some("gpt"),
                WireFormat::OpenAiChat,
            )
            .await?;
        Ok(())
    }
}
