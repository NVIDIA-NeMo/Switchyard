// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI-compatible backend for Chat Completions or Responses targets.

use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{Map, Value};
use switchyard_core::{
    merge_target_extra_body, BackendFormat, BoxResponseStream, ChatRequest, ChatRequestType,
    ChatResponse, EndpointConfig, LlmBackend, LlmTarget, LlmTargetId, ModelId, ProxyContext,
    Result, StreamEvent, SwitchyardError,
};
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

use super::common::{
    build_reqwest_client, decode_sse_frame, drain_next_sse_frame, has_non_whitespace_bytes,
    parse_json_sse_frame, request_wire_format, set_json_model, shared_translation_engine,
    ParsedSseFrame,
};
use super::{BackendSelection, BackendSelectionReason};
use crate::telemetry::{telemetry_header_value, SWITCHYARD_VERSION_HEADER};

const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
static OPENAI_CHAT_ONLY: [ChatRequestType; 1] = [ChatRequestType::OpenAiChat];
static OPENAI_RESPONSES_ONLY: [ChatRequestType; 1] = [ChatRequestType::OpenAiResponses];
static OPENAI_PASSTHROUGH_TARGET_ID: &str = "passthrough";

/// Backend that calls an OpenAI-compatible Chat Completions or Responses API.
pub struct OpenAiNativeBackend {
    /// Resolved target used for endpoint credentials and model rewriting.
    target: LlmTarget,
    /// HTTP transport, injectable for deterministic tests.
    transport: Arc<dyn OpenAiTransport>,
    /// Shared request translator for non-OpenAI inbound payloads.
    translation: Arc<TranslationEngine>,
    /// Translation policy kept explicit so future server policy remains visible.
    translation_policy: TranslationPolicy,
}

impl fmt::Debug for OpenAiNativeBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiNativeBackend")
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}

impl OpenAiNativeBackend {
    /// Creates an OpenAI-compatible backend for one target.
    pub fn new(target: LlmTarget) -> Result<Self> {
        let transport = Arc::new(ReqwestOpenAiTransport::new(target.endpoint.timeout_secs)?);
        Self::with_transport(target, transport)
    }

    /// Returns the configured upstream target.
    pub fn target(&self) -> &LlmTarget {
        &self.target
    }

    fn with_transport(target: LlmTarget, transport: Arc<dyn OpenAiTransport>) -> Result<Self> {
        validate_target_format(&target)?;
        Ok(Self {
            target,
            transport,
            translation: shared_translation_engine(),
            translation_policy: TranslationPolicy::default(),
        })
    }

    fn target_request_type(&self) -> ChatRequestType {
        match self.target.format {
            BackendFormat::OpenAi => ChatRequestType::OpenAiChat,
            BackendFormat::Responses => ChatRequestType::OpenAiResponses,
            BackendFormat::Auto | BackendFormat::Anthropic | BackendFormat::Gemini => {
                unreachable!("OpenAiNativeBackend target format is validated at construction")
            }
        }
    }

    fn target_wire_format(&self) -> WireFormat {
        match self.target_request_type() {
            ChatRequestType::OpenAiChat => WireFormat::OpenAiChat,
            ChatRequestType::OpenAiResponses => WireFormat::OpenAiResponses,
            ChatRequestType::Anthropic | ChatRequestType::Gemini => {
                unreachable!("OpenAiNativeBackend only targets OpenAI wire formats")
            }
        }
    }

    fn outbound_body(&self, request: &ChatRequest) -> Result<Value> {
        let target_request_type = self.target_request_type();
        let mut body = match request.request_type() {
            source if source == target_request_type => request.body().clone(),
            source => {
                self.translation
                    .translate_request(
                        request_wire_format(source),
                        self.target_wire_format(),
                        request.body(),
                        &self.translation_policy,
                    )
                    .map_err(|error| {
                        SwitchyardError::Backend(format!(
                            "failed to translate {source:?} request to {:?}: {error}",
                            self.target.format
                        ))
                    })?
                    .body
            }
        };
        set_json_model(&mut body, self.target.model.as_str());
        if self.target.format == BackendFormat::OpenAi {
            ensure_stream_usage(&mut body);
        }
        // Merge per-target ``extra_body`` last so e.g. DeepSeek V4's
        // ``chat_template_kwargs.enable_thinking=False`` reaches the
        // upstream.  Caller wins on key conflicts (see
        // :func:`merge_target_extra_body`).
        merge_target_extra_body(&mut body, self.target.extra_body.as_ref());
        Ok(body)
    }

    /// Calls this target without requiring chain-local `ProxyContext` state.
    pub async fn call_without_context(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let http_request = self.http_request(request)?;
        self.send_http_request(http_request).await
    }

    // Builds the upstream HTTP request before any context observations are recorded.
    fn http_request(&self, request: &ChatRequest) -> Result<OpenAiHttpRequest> {
        let body = self.outbound_body(request)?;
        let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let endpoint = endpoint_for_backend_format(self.target.format)?;
        Ok(OpenAiHttpRequest {
            target_id: self.target.id.clone(),
            url: openai_url(self.target.endpoint.base_url.as_deref(), endpoint),
            api_key: openai_api_key(self.target.endpoint.api_key.as_deref()),
            body,
            stream,
            extra_headers: self.target.extra_headers.clone(),
            endpoint,
        })
    }

    // Sends an already-normalized upstream request.
    async fn send_http_request(&self, request: OpenAiHttpRequest) -> Result<ChatResponse> {
        let endpoint = request.endpoint;
        match self.transport.send(request).await? {
            OpenAiHttpResponse::Buffered(body) => match endpoint {
                OpenAiEndpoint::ChatCompletions => Ok(ChatResponse::openai_completion(body)),
                OpenAiEndpoint::Responses => Ok(ChatResponse::openai_responses_completion(body)),
            },
            OpenAiHttpResponse::Stream(stream) => match endpoint {
                OpenAiEndpoint::ChatCompletions => Ok(ChatResponse::OpenAiStream(stream)),
                OpenAiEndpoint::Responses => Ok(ChatResponse::OpenAiResponsesStream(stream)),
            },
        }
    }
}

/// Backend that calls an OpenAI-compatible Chat Completions API without rewriting `model`.
pub struct OpenAiPassthroughBackend {
    /// Endpoint used without an owning LLM target or model rewrite.
    endpoint: EndpointConfig,
    /// HTTP transport, injectable for tests.
    transport: Arc<dyn OpenAiTransport>,
    /// Shared request translator for supported non-OpenAI inbound payloads.
    translation: Arc<TranslationEngine>,
    /// Translation policy kept local to the backend.
    translation_policy: TranslationPolicy,
}

impl fmt::Debug for OpenAiPassthroughBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiPassthroughBackend")
            .field("base_url", &self.endpoint.base_url)
            .field("timeout_secs", &self.endpoint.timeout_secs)
            .finish_non_exhaustive()
    }
}

impl OpenAiPassthroughBackend {
    /// Creates a passthrough OpenAI-compatible backend.
    pub fn new(endpoint: EndpointConfig) -> Result<Self> {
        let transport = Arc::new(ReqwestOpenAiTransport::new(endpoint.timeout_secs)?);
        Self::with_transport(endpoint, transport)
    }

    /// Returns the configured upstream endpoint.
    pub fn endpoint(&self) -> &EndpointConfig {
        &self.endpoint
    }

    fn with_transport(
        endpoint: EndpointConfig,
        transport: Arc<dyn OpenAiTransport>,
    ) -> Result<Self> {
        Ok(Self {
            endpoint,
            transport,
            translation: shared_translation_engine(),
            translation_policy: TranslationPolicy::default(),
        })
    }

    fn outbound_body(&self, request: &ChatRequest) -> Result<Value> {
        let mut body = match request.request_type() {
            ChatRequestType::OpenAiChat => request.body().clone(),
            source => {
                self.translation
                    .translate_request(
                        request_wire_format(source),
                        WireFormat::OpenAiChat,
                        request.body(),
                        &self.translation_policy,
                    )
                    .map_err(|error| {
                        SwitchyardError::Backend(format!(
                            "failed to translate {source:?} request to OpenAI Chat: {error}"
                        ))
                    })?
                    .body
            }
        };
        ensure_stream_usage(&mut body);
        Ok(body)
    }
}

#[async_trait]
impl LlmBackend for OpenAiNativeBackend {
    fn supported_request_types(&self) -> &[ChatRequestType] {
        match self.target.format {
            BackendFormat::OpenAi => &OPENAI_CHAT_ONLY,
            BackendFormat::Responses => &OPENAI_RESPONSES_ONLY,
            BackendFormat::Auto | BackendFormat::Anthropic | BackendFormat::Gemini => {
                unreachable!("OpenAiNativeBackend target format is validated at construction")
            }
        }
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

#[async_trait]
impl LlmBackend for OpenAiPassthroughBackend {
    fn supported_request_types(&self) -> &[ChatRequestType] {
        &OPENAI_CHAT_ONLY
    }

    async fn call(&self, ctx: &mut ProxyContext, request: &ChatRequest) -> Result<ChatResponse> {
        let body = self.outbound_body(request)?;
        let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let http_request = OpenAiHttpRequest {
            target_id: LlmTargetId::from_static(OPENAI_PASSTHROUGH_TARGET_ID),
            url: openai_url(
                self.endpoint.base_url.as_deref(),
                OpenAiEndpoint::ChatCompletions,
            ),
            api_key: openai_api_key(self.endpoint.api_key.as_deref()),
            body,
            stream,
            extra_headers: BTreeMap::new(),
            endpoint: OpenAiEndpoint::ChatCompletions,
        };

        ctx.inbound_format = ctx.inbound_format.or(Some(request.request_type()));
        if let Some(model) = http_request
            .body
            .get("model")
            .and_then(Value::as_str)
            .and_then(|model| ModelId::new(model.to_string()).ok())
        {
            ctx.insert(BackendSelection::for_model(
                model,
                request.model().map(str::to_string),
                BackendSelectionReason::PassthroughModel,
            ));
        }

        match self.transport.send(http_request).await? {
            OpenAiHttpResponse::Buffered(body) => Ok(ChatResponse::openai_completion(body)),
            OpenAiHttpResponse::Stream(stream) => Ok(ChatResponse::OpenAiStream(stream)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct OpenAiHttpRequest {
    /// Target ID used only for logging and diagnostics.
    target_id: LlmTargetId,
    /// Fully resolved OpenAI-compatible endpoint URL.
    url: String,
    /// Per-target API key or process environment fallback.
    api_key: Option<String>,
    /// Already-normalized OpenAI-compatible request body.
    body: Value,
    /// Whether the upstream call should be treated as SSE.
    stream: bool,
    /// Per-target extra headers (e.g. ``X-Inference-Priority: batch``
    /// for NIH evals gateway routing on DeepSeek V4).  Empty for the
    /// passthrough backend, which has no LlmTarget.
    extra_headers: BTreeMap<String, String>,
    /// Upstream endpoint family used for response wrapping and diagnostics.
    endpoint: OpenAiEndpoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpenAiEndpoint {
    /// OpenAI Chat Completions API.
    ChatCompletions,
    /// OpenAI Responses API.
    Responses,
}

impl OpenAiEndpoint {
    fn label(self) -> &'static str {
        match self {
            Self::ChatCompletions => "OpenAI chat completions",
            Self::Responses => "OpenAI responses",
        }
    }
}

enum OpenAiHttpResponse {
    /// Complete JSON response from a non-streaming upstream call.
    Buffered(Value),
    /// Streamed SSE response converted into Switchyard stream events.
    Stream(BoxResponseStream),
}

#[async_trait]
trait OpenAiTransport: Send + Sync {
    /// Sends one already-normalized OpenAI-compatible request.
    async fn send(&self, request: OpenAiHttpRequest) -> Result<OpenAiHttpResponse>;
}

struct ReqwestOpenAiTransport {
    /// Reused async HTTP client with configured timeout behavior.
    client: reqwest::Client,
}

impl ReqwestOpenAiTransport {
    fn new(timeout_secs: Option<f64>) -> Result<Self> {
        let client = build_reqwest_client("OpenAI", timeout_secs)?;
        Ok(Self { client })
    }
}

#[async_trait]
impl OpenAiTransport for ReqwestOpenAiTransport {
    async fn send(&self, request: OpenAiHttpRequest) -> Result<OpenAiHttpResponse> {
        let target_id = request.target_id.clone();
        let endpoint = request.endpoint;
        let mut builder = self.client.post(&request.url).json(&request.body);
        if let Some(api_key) = request.api_key {
            builder = builder.bearer_auth(api_key);
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
                endpoint = endpoint.label(),
                "OpenAI request failed"
            );
            SwitchyardError::Upstream(format!("{} request failed: {error}", endpoint.label()))
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
                endpoint = endpoint.label(),
                "OpenAI request returned error status"
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
                provider: endpoint.label().to_string(),
                status_code: status.as_u16(),
                body,
            });
        }

        if request.stream {
            return Ok(OpenAiHttpResponse::Stream(openai_sse_stream(response)));
        }

        let body = response.json::<Value>().await.map_err(|error| {
            SwitchyardError::Upstream(format!(
                "{} returned invalid JSON: {error}",
                endpoint.label()
            ))
        })?;
        Ok(OpenAiHttpResponse::Buffered(body))
    }
}

fn validate_target_format(target: &LlmTarget) -> Result<()> {
    match target.format {
        BackendFormat::OpenAi | BackendFormat::Responses => Ok(()),
        BackendFormat::Auto | BackendFormat::Anthropic | BackendFormat::Gemini => {
            Err(SwitchyardError::InvalidConfig(format!(
                "OpenAiNativeBackend requires a target with resolved OpenAI format, got {:?} for {}",
                target.format, target.id
            )))
        }
    }
}

fn endpoint_for_backend_format(format: BackendFormat) -> Result<OpenAiEndpoint> {
    match format {
        BackendFormat::OpenAi => Ok(OpenAiEndpoint::ChatCompletions),
        BackendFormat::Responses => Ok(OpenAiEndpoint::Responses),
        BackendFormat::Auto | BackendFormat::Anthropic | BackendFormat::Gemini => {
            Err(SwitchyardError::InvalidConfig(format!(
                "OpenAiNativeBackend cannot dispatch target format {format:?}"
            )))
        }
    }
}

// OpenAI-compatible streaming users need usage events for stats accounting.
fn ensure_stream_usage(body: &mut Value) {
    let Value::Object(object) = body else {
        return;
    };
    if !object
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
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

// Accept either a root `/v1` URL or an already-specific OpenAI endpoint URL.
fn openai_url(base_url: Option<&str>, endpoint: OpenAiEndpoint) -> String {
    let base_url = base_url
        .unwrap_or(DEFAULT_OPENAI_BASE_URL)
        .trim_end_matches('/');
    let base_root = base_url
        .strip_suffix("/chat/completions")
        .or_else(|| base_url.strip_suffix("/responses"))
        .unwrap_or(base_url);
    let suffix = match endpoint {
        OpenAiEndpoint::ChatCompletions => "/chat/completions",
        OpenAiEndpoint::Responses => "/responses",
    };
    format!("{base_root}{suffix}")
}

fn openai_api_key(configured: Option<&str>) -> Option<String> {
    // Resolve per call so long-lived backends can pick up rotated environment credentials.
    configured
        .map(str::to_string)
        .or_else(|| env::var(OPENAI_API_KEY_ENV).ok())
        .filter(|value| !value.trim().is_empty())
}

// Best-effort match for OpenAI-shape context-window-overflow error bodies.
// Canonical signal is `error.code == "context_length_exceeded"`; NVIDIA and
// other proxies sometimes only set the human-readable message, so we fall back
// to substring matching. Only reached for upstream 400s, and a false positive
// triggers a single bounded evict-and-retry (never an infinite loop), so erring
// toward matching the message is safe.
// OpenAI canonical phrase + NVIDIA/LiteLLM wrap variants. Adding a new
// provider-wrap is a one-line entry here, not a fork of the parsing logic.
const OPENAI_OVERFLOW_PHRASES: &[&str] = &[
    "maximum context length",
    "context length exceeded",
    "context window",
    "context length is only",
    "please reduce the length of the input",
];

fn is_context_overflow(body: &str) -> bool {
    super::context_overflow::is_overflow_body(
        body,
        |value| {
            value
                .get("error")
                .and_then(|err| err.get("code"))
                .and_then(Value::as_str)
                == Some("context_length_exceeded")
        },
        OPENAI_OVERFLOW_PHRASES,
    )
}

fn openai_sse_stream(response: reqwest::Response) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut chunks = response.bytes_stream();
        let mut buffer = Vec::new();

        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.map_err(|error| {
                SwitchyardError::Upstream(format!("OpenAI stream read failed: {error}"))
            })?;
            buffer.extend_from_slice(&chunk);

            // Drain complete frames immediately while preserving partial frames
            // across TCP chunks.
            while let Some(frame) = drain_next_sse_frame(&mut buffer, "OpenAI")? {
                match parse_json_sse_frame(&frame, "OpenAI", Some("[DONE]"))? {
                    ParsedSseFrame::Json(value) => yield StreamEvent::Json(value),
                    ParsedSseFrame::Done => return,
                    ParsedSseFrame::Empty => {}
                }
            }
        }

        // A non-standard upstream might omit the final double newline; parse a
        // trailing complete frame instead of losing its usage chunk.
        if has_non_whitespace_bytes(&buffer) {
            let frame = decode_sse_frame(&buffer, "OpenAI")?;
            match parse_json_sse_frame(&frame, "OpenAI", Some("[DONE]"))? {
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

    struct FakeOpenAiTransport {
        requests: Mutex<Vec<OpenAiHttpRequest>>,
        response: Mutex<Option<Result<OpenAiHttpResponse>>>,
    }

    impl FakeOpenAiTransport {
        fn with_error(message: &str) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: Mutex::new(Some(Err(SwitchyardError::Upstream(message.to_string())))),
            }
        }
    }

    #[async_trait]
    impl OpenAiTransport for FakeOpenAiTransport {
        async fn send(&self, request: OpenAiHttpRequest) -> Result<OpenAiHttpResponse> {
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

    fn openai_target() -> LlmTarget {
        LlmTarget {
            id: LlmTargetId::from_static("primary"),
            model: ModelId::from_static("target-model"),
            format: BackendFormat::OpenAi,
            endpoint: EndpointConfig {
                base_url: Some("https://example.test/v1".to_string()),
                api_key: Some("secret".to_string()),
                timeout_secs: None,
            },
            extra_body: None,
            extra_headers: BTreeMap::new(),
        }
    }

    #[test]
    fn outbound_body_merges_target_extra_body() -> Result<()> {
        // Use-case: DeepSeek V4 on NVIDIA Inference Hub.  The target sets
        // ``chat_template_kwargs.enable_thinking=False`` so V4 skips its
        // chain-of-thought pass.  Without this the model 504s at -n 8
        // concurrency from the Hub gateway timeout.
        let mut target = openai_target();
        target.extra_body = Some(json!({
            "chat_template_kwargs": {"enable_thinking": false}
        }));
        let transport = Arc::new(FakeOpenAiTransport::with_error("ignored"));
        let backend = OpenAiNativeBackend::with_transport(target, transport)?;
        let request = ChatRequest::openai_chat(json!({
            "model": "client-model",
            "messages": [{"role": "user", "content": "hi"}],
        }));
        let body = backend.outbound_body(&request)?;
        assert_eq!(
            body.get("chat_template_kwargs"),
            Some(&json!({"enable_thinking": false})),
            "target.extra_body should land at the top level of the outbound body",
        );
        // Caller fields preserved.
        assert_eq!(
            body.get("model").and_then(|v| v.as_str()),
            Some("target-model"),
            "outbound_body should rewrite model to the target's",
        );
        Ok(())
    }

    #[test]
    fn outbound_body_caller_wins_on_extra_body_key_conflict() -> Result<()> {
        let mut target = openai_target();
        target.extra_body = Some(json!({
            "chat_template_kwargs": {"enable_thinking": false},
            "logit_bias": {"50256": -100},
        }));
        let transport = Arc::new(FakeOpenAiTransport::with_error("ignored"));
        let backend = OpenAiNativeBackend::with_transport(target, transport)?;
        // Caller sets chat_template_kwargs explicitly with thinking=true.
        let request = ChatRequest::openai_chat(json!({
            "model": "client-model",
            "messages": [],
            "chat_template_kwargs": {"enable_thinking": true},
        }));
        let body = backend.outbound_body(&request)?;
        // Caller-supplied chat_template_kwargs wins on the top-level key
        // (the merge is shallow / caller-wins).  Target's logit_bias
        // still lands because caller didn't set it.
        assert_eq!(
            body.get("chat_template_kwargs"),
            Some(&json!({"enable_thinking": true})),
        );
        assert_eq!(body.get("logit_bias"), Some(&json!({"50256": -100})),);
        Ok(())
    }

    #[tokio::test]
    async fn transport_errors_are_backend_errors() -> Result<()> {
        let transport = Arc::new(FakeOpenAiTransport::with_error("upstream exploded"));
        let backend = OpenAiNativeBackend::with_transport(openai_target(), transport)?;
        let request = ChatRequest::openai_chat(json!({
            "model": "client-model",
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
    fn rejects_anthropic_targets() -> Result<()> {
        let mut target = openai_target();
        target.format = BackendFormat::Anthropic;

        let Err(error) = OpenAiNativeBackend::new(target) else {
            return Err(SwitchyardError::Other(
                "Anthropic target should be rejected".to_string(),
            ));
        };

        assert!(matches!(error, SwitchyardError::InvalidConfig(_)));
        Ok(())
    }

    #[test]
    fn parses_openai_sse_json_frames_and_done() -> Result<()> {
        let ParsedSseFrame::Json(value) = parse_json_sse_frame(
            "event: message\ndata: {\"choices\":[]}\n",
            "OpenAI",
            Some("[DONE]"),
        )?
        else {
            return Err(SwitchyardError::Other(
                "JSON frame should produce a JSON value".to_string(),
            ));
        };
        assert_eq!(value, json!({"choices": []}));

        let ParsedSseFrame::Done =
            parse_json_sse_frame("data: [DONE]\n", "OpenAI", Some("[DONE]"))?
        else {
            return Err(SwitchyardError::Other("DONE frame should stop".to_string()));
        };
        Ok(())
    }

    #[test]
    fn context_overflow_canonical_code_matches() {
        let body = r#"{"error":{"code":"context_length_exceeded","message":"x","type":"invalid_request_error"}}"#;
        assert!(is_context_overflow(body));
    }

    #[test]
    fn context_overflow_nvidia_message_matches() {
        let body = r#"{"error":{"message":"This model's maximum context length is 131072 tokens, however you requested ..."}}"#;
        assert!(is_context_overflow(body));
    }

    #[test]
    fn context_overflow_unrelated_400_does_not_match() {
        let body = r#"{"error":{"code":"invalid_api_key","message":"bad key"}}"#;
        assert!(!is_context_overflow(body));
    }

    #[test]
    fn context_overflow_nvidia_litellm_wrap_matches() {
        // Body shape observed from inference-api.nvidia.com's LiteLLM proxy
        // wrapping a Nemotron context-window overflow.
        let body = r#"{"error":{"message":"litellm.BadRequestError: OpenAIException - {\"error\":{\"message\":\"You passed 131041 input tokens and requested 32 output tokens. However, the model's context length is only 131072 tokens, resulting in a maximum input length of 131040 tokens. Please reduce the length of the input prompt. (parameter=input_tokens, value=131041)\",\"type\":\"BadRequestError\",\"param\":\"input_tokens\",\"code\":400}}","type":null,"param":null,"code":"400"}}"#;
        assert!(is_context_overflow(body));
    }
}
