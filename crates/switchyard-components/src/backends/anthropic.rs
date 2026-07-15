// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic-compatible Messages backend.

use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{json, Map, Value};
use switchyard_core::{
    merge_target_extra_body, BackendFormat, BoxResponseStream, ChatRequest, ChatRequestType,
    ChatResponse, LlmBackend, LlmTarget, LlmTargetId, ProxyContext, Result, StreamEvent,
    SwitchyardError,
};
use switchyard_translation::{
    normalize_anthropic_tool_use_ids, TranslationEngine, TranslationPolicy, WireFormat,
};

use super::common::{
    build_reqwest_client, decode_sse_frame, drain_next_sse_frame, has_non_whitespace_bytes,
    parse_json_sse_frame, request_wire_format, set_json_model, shared_translation_engine,
    ParsedSseFrame,
};
use super::BackendSelection;
use crate::telemetry::{telemetry_header_value, SWITCHYARD_VERSION_HEADER};

const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_VERSION: &str = "2023-06-01";
static ANTHROPIC_ONLY: [ChatRequestType; 1] = [ChatRequestType::Anthropic];

/// Backend that calls an Anthropic-compatible Messages API.
pub struct AnthropicNativeBackend {
    /// Resolved target used for endpoint credentials and model rewriting.
    target: LlmTarget,
    /// HTTP transport, injectable for deterministic tests.
    transport: Arc<dyn AnthropicTransport>,
    /// Shared request translator for non-Anthropic inbound payloads.
    translation: Arc<TranslationEngine>,
    /// Translation policy kept explicit so backend behavior is inspectable.
    translation_policy: TranslationPolicy,
}

impl fmt::Debug for AnthropicNativeBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicNativeBackend")
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}

impl AnthropicNativeBackend {
    /// Creates an Anthropic-compatible backend for one target.
    pub fn new(target: LlmTarget) -> Result<Self> {
        let transport = Arc::new(ReqwestAnthropicTransport::new(
            target.endpoint.timeout_secs,
        )?);
        Self::with_transport(target, transport)
    }

    /// Returns the configured upstream target.
    pub fn target(&self) -> &LlmTarget {
        &self.target
    }

    fn with_transport(target: LlmTarget, transport: Arc<dyn AnthropicTransport>) -> Result<Self> {
        validate_target_format(&target)?;
        Ok(Self {
            target,
            transport,
            translation: shared_translation_engine(),
            translation_policy: TranslationPolicy::default(),
        })
    }

    fn outbound_body(&self, request: &ChatRequest) -> Result<Value> {
        // Native Anthropic requests are a TRUE passthrough: forward the client's
        // body verbatim, rewriting only the model id (routing) and merging any
        // operator-configured `extra_body`. The strips/normalization in the
        // translated branch below exist for OpenAI/Responses -> Anthropic bodies,
        // which can carry fields/shapes the Anthropic API rejects. Applying them
        // to a real Anthropic client (e.g. Claude Code) silently drops valid
        // Anthropic features — `context_management` (context auto-compaction),
        // signed thinking blocks, mid-conversation system turns, client tool ids
        // — which a passthrough must never do.
        if matches!(request.request_type(), ChatRequestType::Anthropic) {
            let mut body = request.body().clone();
            set_json_model(&mut body, self.target.model.as_str());
            merge_target_extra_body(&mut body, self.target.extra_body.as_ref());
            return Ok(body);
        }
        let mut body = self
            .translation
            .translate_request(
                request_wire_format(request.request_type()),
                WireFormat::AnthropicMessages,
                request.body(),
                &self.translation_policy,
            )
            .map_err(|error| {
                SwitchyardError::Backend(format!(
                    "failed to translate {:?} request to Anthropic Messages: {error}",
                    request.request_type()
                ))
            })?
            .body;
        set_json_model(&mut body, self.target.model.as_str());
        strip_anthropic_incompatible_fields(&mut body);
        normalize_anthropic_body(&mut body);
        // Per-target ``extra_body`` merged last; caller wins on key
        // conflicts (see :func:`merge_target_extra_body`).
        merge_target_extra_body(&mut body, self.target.extra_body.as_ref());
        Ok(body)
    }

    /// Calls this target without requiring chain-local `ProxyContext` state.
    pub async fn call_without_context(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let http_request = self.http_request(request)?;
        self.send_http_request(http_request).await
    }

    // Builds the upstream HTTP request before any context observations are recorded.
    fn http_request(&self, request: &ChatRequest) -> Result<AnthropicHttpRequest> {
        let body = self.outbound_body(request)?;
        let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        Ok(AnthropicHttpRequest {
            target_id: self.target.id.clone(),
            url: messages_url(self.target.endpoint.base_url.as_deref()),
            api_key: anthropic_api_key(self.target.endpoint.api_key.as_deref()),
            body,
            stream,
            extra_headers: self.target.extra_headers.clone(),
        })
    }

    // Sends an already-normalized upstream request.
    async fn send_http_request(&self, request: AnthropicHttpRequest) -> Result<ChatResponse> {
        match self.transport.send(request).await? {
            AnthropicHttpResponse::Buffered(body) => Ok(ChatResponse::anthropic_completion(body)),
            AnthropicHttpResponse::Stream(stream) => Ok(ChatResponse::AnthropicStream(stream)),
        }
    }
}

#[async_trait]
impl LlmBackend for AnthropicNativeBackend {
    fn supported_request_types(&self) -> &[ChatRequestType] {
        &ANTHROPIC_ONLY
    }

    async fn call(&self, ctx: &mut ProxyContext, request: &ChatRequest) -> Result<ChatResponse> {
        let http_request = self.http_request(request)?;

        ctx.inbound_format = ctx.inbound_format.or(Some(request.request_type()));
        let previous_selection = ctx.get::<BackendSelection>().cloned();
        ctx.insert(BackendSelection::native_target_observation(
            previous_selection.as_ref(),
            self.target.id.clone(),
            self.target.model.clone(),
            request.model().map(str::to_string),
        ));

        self.send_http_request(http_request).await
    }
}

#[derive(Clone, Debug, PartialEq)]
struct AnthropicHttpRequest {
    /// Target ID used only for logging and diagnostics.
    target_id: LlmTargetId,
    /// Fully resolved Messages API URL.
    url: String,
    /// Per-target API key or process environment fallback.
    api_key: Option<String>,
    /// Already-normalized Anthropic Messages request body.
    body: Value,
    /// Whether the upstream call should be treated as SSE.
    stream: bool,
    /// Per-target headers merged onto the outbound request.
    extra_headers: BTreeMap<String, String>,
}

enum AnthropicHttpResponse {
    /// Complete JSON response from a non-streaming upstream call.
    Buffered(Value),
    /// Streamed SSE response converted into Switchyard stream events.
    Stream(BoxResponseStream),
}

#[async_trait]
trait AnthropicTransport: Send + Sync {
    /// Sends one already-normalized Anthropic Messages request.
    async fn send(&self, request: AnthropicHttpRequest) -> Result<AnthropicHttpResponse>;
}

struct ReqwestAnthropicTransport {
    /// Reused async HTTP client with configured timeout behavior.
    client: reqwest::Client,
}

impl ReqwestAnthropicTransport {
    fn new(timeout_secs: Option<f64>) -> Result<Self> {
        let client = build_reqwest_client("Anthropic", timeout_secs)?;
        Ok(Self { client })
    }
}

#[async_trait]
impl AnthropicTransport for ReqwestAnthropicTransport {
    async fn send(&self, request: AnthropicHttpRequest) -> Result<AnthropicHttpResponse> {
        let target_id = request.target_id.clone();
        let mut builder = self
            .client
            .post(&request.url)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&request.body);
        if let Some(api_key) = request.api_key {
            builder = builder.header("x-api-key", api_key);
        }
        if let Some(version) = telemetry_header_value() {
            builder = builder.header(SWITCHYARD_VERSION_HEADER, version);
        }
        for (name, value) in &request.extra_headers {
            builder = builder.header(name, value);
        }

        let response = builder.send().await.map_err(|error| {
            tracing::warn!(
                target_id = %target_id,
                error = %error,
                "Anthropic messages request failed"
            );
            SwitchyardError::Upstream(format!("Anthropic messages request failed: {error}"))
        })?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|error| format!("<failed to read error body: {error}>"));
            tracing::warn!(
                target_id = %target_id,
                status = %status,
                "Anthropic messages returned error status"
            );
            if status == reqwest::StatusCode::BAD_REQUEST && is_context_overflow(&body) {
                let model = request
                    .body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                return Err(SwitchyardError::ContextWindowExceeded {
                    target_id: target_id.to_string(),
                    model,
                    message: body,
                });
            }
            return Err(SwitchyardError::UpstreamHttp {
                provider: "Anthropic messages".to_string(),
                status_code: status.as_u16(),
                body,
            });
        }

        if request.stream {
            return Ok(AnthropicHttpResponse::Stream(anthropic_sse_stream(
                response,
            )));
        }

        let body = response.json::<Value>().await.map_err(|error| {
            SwitchyardError::Upstream(format!("Anthropic messages returned invalid JSON: {error}"))
        })?;
        Ok(AnthropicHttpResponse::Buffered(body))
    }
}

fn validate_target_format(target: &LlmTarget) -> Result<()> {
    match target.format {
        BackendFormat::Anthropic => Ok(()),
        BackendFormat::Auto | BackendFormat::OpenAi | BackendFormat::Responses => {
            Err(SwitchyardError::InvalidConfig(format!(
                "AnthropicNativeBackend requires a target with resolved Anthropic format, got {:?} for {}",
                target.format, target.id
            )))
        }
    }
}

// Drop fields accepted by OpenAI-like APIs but rejected by Anthropic Messages.
fn strip_anthropic_incompatible_fields(body: &mut Value) {
    if let Value::Object(object) = body {
        object.remove("reasoning_effort");
        object.remove("context_management");
    }
}

// Normalize translated Anthropic payloads before applying target overrides.
fn normalize_anthropic_body(body: &mut Value) {
    let Value::Object(object) = body else {
        return;
    };
    if let Some(messages) = object.remove("messages") {
        // AWS documents message-level `role: "system"` as an Opus 4.8-only
        // Anthropic dialect:
        // https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-mid-conversation-system.html
        //
        // Keep the default conservative for older Bedrock/LiteLLM targets by
        // lifting those turns into top-level `system`.
        // TODO: add target-level Anthropic dialect support so
        // Opus 4.8 can opt into mid-conversation system messages explicitly.
        let (messages, system_text) = lift_message_level_system(messages);
        append_lifted_system_text(object, system_text);
        let messages = normalize_anthropic_tool_use_ids(messages);
        object.insert(
            "messages".to_string(),
            strip_unsigned_thinking_blocks(messages),
        );
    }
}

// Moves Anthropic Opus-4.8-style message-level system turns out of the
// conversation so legacy Anthropic-compatible backends do not reject them.
fn lift_message_level_system(messages: Value) -> (Value, Vec<String>) {
    let Value::Array(messages) = messages else {
        return (messages, Vec::new());
    };

    let mut kept_messages = Vec::with_capacity(messages.len());
    let mut system_text = Vec::new();
    for message in messages {
        if is_message_level_system(&message) {
            if let Some(text) = system_text_from_message(&message) {
                system_text.push(text);
            }
        } else {
            kept_messages.push(message);
        }
    }

    (Value::Array(kept_messages), system_text)
}

// Treats `system` and OpenAI/Codex `developer` roles as instruction-like
// turns. `developer` is not an Anthropic Opus 4.8 role; lifting it matches the
// existing OpenAI-to-Anthropic translator behavior and prevents malformed
// Anthropic-bound traffic from leaking an invalid role upstream.
fn is_message_level_system(message: &Value) -> bool {
    matches!(
        message.get("role").and_then(Value::as_str),
        Some("system") | Some("developer")
    )
}

// Extracts text from an invalid message-level system/developer turn.
fn system_text_from_message(message: &Value) -> Option<String> {
    message.get("content").and_then(system_text_from_content)
}

// Converts Anthropic text-ish content into top-level system text.
fn system_text_from_content(content: &Value) -> Option<String> {
    match content {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        Value::String(_) | Value::Null => None,
        Value::Array(blocks) => {
            let parts = blocks
                .iter()
                .filter_map(system_text_from_content_block)
                .collect::<Vec<_>>();
            (!parts.is_empty()).then(|| parts.join("\n\n"))
        }
        other => Some(other.to_string()),
    }
}

// Extracts the supported text shape from one structured content block.
fn system_text_from_content_block(block: &Value) -> Option<String> {
    match block {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        Value::Object(object) => match object.get("type").and_then(Value::as_str) {
            Some("text") | Some("input_text") => object
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned),
            // Message-level system/developer turns are downgraded for legacy Anthropic
            // compatibility. Only text-like instruction content is replayed; images and
            // other non-text blocks are intentionally not promoted into top-level system.
            _ => None,
        },
        _ => None,
    }
}

// Appends lifted message-level system text onto any existing Anthropic system field.
fn append_lifted_system_text(object: &mut Map<String, Value>, system_text: Vec<String>) {
    if system_text.is_empty() {
        return;
    }

    let joined = system_text.join("\n\n");
    match object.remove("system") {
        None | Some(Value::Null) => {
            object.insert("system".to_string(), Value::String(joined));
        }
        Some(Value::String(existing)) if existing.is_empty() => {
            object.insert("system".to_string(), Value::String(joined));
        }
        Some(Value::String(existing)) => {
            object.insert(
                "system".to_string(),
                Value::String(format!("{existing}\n\n{joined}")),
            );
        }
        Some(Value::Array(mut blocks)) => {
            blocks.extend(
                system_text
                    .into_iter()
                    .map(|text| json!({"type": "text", "text": text})),
            );
            object.insert("system".to_string(), Value::Array(blocks));
        }
        Some(other) => {
            object.insert(
                "system".to_string(),
                Value::String(format!("{other}\n\n{joined}")),
            );
        }
    }
}

// Anthropic requires signed thinking blocks on replay; remove unsigned blocks
// so passthrough and translated requests remain accepted by the API.
fn strip_unsigned_thinking_blocks(messages: Value) -> Value {
    match messages {
        Value::Array(messages) => Value::Array(
            messages
                .into_iter()
                .map(strip_unsigned_thinking_from_message)
                .collect(),
        ),
        other => other,
    }
}

fn strip_unsigned_thinking_from_message(message: Value) -> Value {
    match message {
        Value::Object(mut message) => {
            let Some(content) = message.remove("content") else {
                return Value::Object(message);
            };
            let Value::Array(blocks) = content else {
                message.insert("content".to_string(), content);
                return Value::Object(message);
            };

            let kept = blocks
                .into_iter()
                .filter(|block| !is_unsigned_thinking_block(block))
                .collect::<Vec<_>>();
            let content = if kept.is_empty() {
                Value::String(String::new())
            } else {
                Value::Array(kept)
            };
            message.insert("content".to_string(), content);
            Value::Object(message)
        }
        other => other,
    }
}

fn is_unsigned_thinking_block(block: &Value) -> bool {
    if block.get("type").and_then(Value::as_str) != Some("thinking") {
        return false;
    }
    !matches!(
        block.get("signature").and_then(Value::as_str),
        Some(signature) if !signature.is_empty()
    )
}

fn messages_url(base_url: Option<&str>) -> String {
    let base_url = base_url
        .unwrap_or(DEFAULT_ANTHROPIC_BASE_URL)
        .trim_end_matches('/');
    if base_url.ends_with("/v1/messages") {
        base_url.to_string()
    } else if base_url.ends_with("/v1") {
        format!("{base_url}/messages")
    } else {
        format!("{base_url}/v1/messages")
    }
}

fn anthropic_api_key(configured: Option<&str>) -> Option<String> {
    // Resolve per call so long-lived backends can pick up rotated environment credentials.
    configured
        .map(str::to_string)
        .or_else(|| env::var(ANTHROPIC_API_KEY_ENV).ok())
        .filter(|value| !value.trim().is_empty())
}

// Canonical Anthropic 4xx looks like
// `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: ..."}}`.
// Anthropic has no structured `error.code` field, so detection is phrase-based only.
const ANTHROPIC_OVERFLOW_PHRASES: &[&str] = &[
    "prompt is too long",
    "maximum number of tokens",
    "context window",
    "context length",
];

fn is_context_overflow(body: &str) -> bool {
    super::context_overflow::is_overflow_body(body, |_| false, ANTHROPIC_OVERFLOW_PHRASES)
}

fn anthropic_sse_stream(response: reqwest::Response) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut chunks = response.bytes_stream();
        let mut buffer = Vec::new();

        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.map_err(|error| {
                SwitchyardError::Upstream(format!("Anthropic stream read failed: {error}"))
            })?;
            buffer.extend_from_slice(&chunk);

            // Anthropic SSE has named events, but the payload we care about is
            // still the JSON `data:` line.
            while let Some(frame) = drain_next_sse_frame(&mut buffer, "Anthropic")? {
                match parse_json_sse_frame(&frame, "Anthropic", None)? {
                    ParsedSseFrame::Json(value) => yield StreamEvent::Json(value),
                    ParsedSseFrame::Done | ParsedSseFrame::Empty => {}
                }
            }
        }

        // Preserve the last frame when an upstream closes without a final SSE
        // separator.
        if has_non_whitespace_bytes(&buffer) {
            let frame = decode_sse_frame(&buffer, "Anthropic")?;
            match parse_json_sse_frame(&frame, "Anthropic", None)? {
                ParsedSseFrame::Json(value) => yield StreamEvent::Json(value),
                ParsedSseFrame::Done | ParsedSseFrame::Empty => {}
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;
    use switchyard_core::{EndpointConfig, LlmTargetId, ModelId};

    use super::*;

    struct FakeAnthropicTransport {
        requests: Mutex<Vec<AnthropicHttpRequest>>,
        response: Mutex<Option<Result<AnthropicHttpResponse>>>,
    }

    impl FakeAnthropicTransport {
        fn with_error(message: &str) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: Mutex::new(Some(Err(SwitchyardError::Upstream(message.to_string())))),
            }
        }
    }

    #[async_trait]
    impl AnthropicTransport for FakeAnthropicTransport {
        async fn send(&self, request: AnthropicHttpRequest) -> Result<AnthropicHttpResponse> {
            self.requests
                .lock()
                .map_err(|_| {
                    SwitchyardError::Other("fake transport request mutex poisoned".to_string())
                })?
                .push(request);
            self.response
                .lock()
                .map_err(|_| {
                    SwitchyardError::Other("fake transport response mutex poisoned".to_string())
                })?
                .take()
                .ok_or_else(|| {
                    SwitchyardError::Other("fake transport response already consumed".to_string())
                })?
        }
    }

    fn anthropic_target() -> LlmTarget {
        LlmTarget {
            id: LlmTargetId::from_static("primary"),
            model: ModelId::from_static("target-claude"),
            format: BackendFormat::Anthropic,
            endpoint: EndpointConfig {
                base_url: Some("https://example.test/v1".to_string()),
                api_key: Some("secret".to_string()),
                timeout_secs: None,
            },
            extra_body: None,
            extra_headers: BTreeMap::new(),
        }
    }

    #[tokio::test]
    async fn transport_errors_are_backend_errors() -> Result<()> {
        let transport = Arc::new(FakeAnthropicTransport::with_error("upstream exploded"));
        let backend = AnthropicNativeBackend::with_transport(anthropic_target(), transport)?;
        let request = ChatRequest::anthropic(json!({
            "model": "client-model",
            "max_tokens": 128,
            "messages": [],
        }));
        let mut ctx = ProxyContext::new();

        let Err(error) = backend.call(&mut ctx, &request).await else {
            return Err(SwitchyardError::Other(
                "backend call should fail".to_string(),
            ));
        };

        assert!(matches!(error, SwitchyardError::Upstream(_)));
        assert!(error.to_string().contains("upstream exploded"));
        Ok(())
    }

    #[test]
    fn rejects_openai_targets() -> Result<()> {
        let mut target = anthropic_target();
        target.format = BackendFormat::OpenAi;

        let Err(error) = AnthropicNativeBackend::new(target) else {
            return Err(SwitchyardError::Other(
                "OpenAI target should be rejected".to_string(),
            ));
        };

        assert!(matches!(error, SwitchyardError::InvalidConfig(_)));
        Ok(())
    }

    #[test]
    fn formats_messages_urls_for_root_v1_and_explicit_paths() {
        assert_eq!(
            messages_url(Some("https://example.test")),
            "https://example.test/v1/messages"
        );
        assert_eq!(
            messages_url(Some("https://example.test/v1")),
            "https://example.test/v1/messages"
        );
        assert_eq!(
            messages_url(Some("https://example.test/v1/messages")),
            "https://example.test/v1/messages"
        );
    }

    #[test]
    fn parses_anthropic_sse_json_frames() -> Result<()> {
        let ParsedSseFrame::Json(value) = parse_json_sse_frame(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\"}}\n",
            "Anthropic",
            None,
        )?
        else {
            return Err(SwitchyardError::Other(
                "JSON frame should produce a JSON value".to_string(),
            ));
        };
        assert_eq!(value["type"], "message_start");
        assert!(matches!(
            parse_json_sse_frame("event: ping\n\n", "Anthropic", None)?,
            ParsedSseFrame::Empty
        ));
        Ok(())
    }

    #[test]
    fn anthropic_context_overflow_canonical_shape_matches() {
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 200001 tokens > 200000 maximum"}}"#;
        assert!(is_context_overflow(body));
    }

    #[test]
    fn anthropic_context_overflow_max_tokens_phrase_matches() {
        let body = r#"{"type":"error","error":{"message":"exceeds the maximum number of tokens for this model"}}"#;
        assert!(is_context_overflow(body));
    }

    #[test]
    fn anthropic_context_overflow_unrelated_400_does_not_match() {
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"missing system prompt"}}"#;
        assert!(!is_context_overflow(body));
    }
}
