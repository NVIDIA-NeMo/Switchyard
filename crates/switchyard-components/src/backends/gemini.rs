// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gemini-compatible generateContent backend.

use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::Value;
use switchyard_core::{
    merge_target_extra_body, BackendFormat, BoxResponseStream, ChatRequest, ChatRequestType,
    ChatResponse, LlmBackend, LlmTarget, LlmTargetId, ProxyContext, Result, StreamEvent,
    SwitchyardError,
};
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

use super::common::{
    build_reqwest_client, decode_sse_frame, drain_next_sse_frame, has_non_whitespace_bytes,
    parse_json_sse_frame, request_wire_format, set_json_model, shared_translation_engine,
    ParsedSseFrame,
};
use super::BackendSelection;
use crate::telemetry::{telemetry_header_value, SWITCHYARD_VERSION_HEADER};

const DEFAULT_GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const GEMINI_API_KEY_ENV: &str = "GEMINI_API_KEY";
static GEMINI_ONLY: [ChatRequestType; 1] = [ChatRequestType::Gemini];

/// Backend that calls a Gemini-compatible generateContent API.
pub struct GeminiNativeBackend {
    /// Resolved target used for endpoint credentials and model rewriting.
    target: LlmTarget,
    /// HTTP transport, injectable for deterministic tests.
    transport: Arc<dyn GeminiTransport>,
    /// Shared request translator for non-Gemini inbound payloads.
    translation: Arc<TranslationEngine>,
    /// Translation policy kept explicit so backend behavior is inspectable.
    translation_policy: TranslationPolicy,
}

impl fmt::Debug for GeminiNativeBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiNativeBackend")
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}

impl GeminiNativeBackend {
    /// Creates a Gemini-compatible backend for one target.
    pub fn new(target: LlmTarget) -> Result<Self> {
        let transport = Arc::new(ReqwestGeminiTransport::new(target.endpoint.timeout_secs)?);
        Self::with_transport(target, transport)
    }

    /// Returns the configured upstream target.
    pub fn target(&self) -> &LlmTarget {
        &self.target
    }

    fn with_transport(target: LlmTarget, transport: Arc<dyn GeminiTransport>) -> Result<Self> {
        validate_target_format(&target)?;
        Ok(Self {
            target,
            transport,
            translation: shared_translation_engine(),
            translation_policy: TranslationPolicy::default(),
        })
    }

    fn outbound_body(&self, request: &ChatRequest) -> Result<Value> {
        let mut body = match request.request_type() {
            ChatRequestType::Gemini => request.body().clone(),
            source => {
                self.translation
                    .translate_request(
                        request_wire_format(source),
                        WireFormat::GeminiGenerateContent,
                        request.body(),
                        &self.translation_policy,
                    )
                    .map_err(|error| {
                        SwitchyardError::Backend(format!(
                        "failed to translate {source:?} request to Gemini generateContent: {error}"
                    ))
                    })?
                    .body
            }
        };
        set_json_model(&mut body, self.target.model.as_str());
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
    //
    // The internal Gemini wire body carries synthetic `model` and `stream`
    // fields because the real API encodes both in the URL; they are popped
    // here so the upstream body only contains generateContent fields.
    fn http_request(&self, request: &ChatRequest) -> Result<GeminiHttpRequest> {
        let mut body = self.outbound_body(request)?;
        let stream = take_body_field(&mut body, "stream")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let model = take_body_field(&mut body, "model")
            .and_then(|value| value.as_str().map(str::to_string))
            .unwrap_or_else(|| self.target.model.as_str().to_string());
        Ok(GeminiHttpRequest {
            target_id: self.target.id.clone(),
            url: generate_content_url(self.target.endpoint.base_url.as_deref(), &model, stream),
            api_key: gemini_api_key(self.target.endpoint.api_key.as_deref()),
            body,
            model,
            stream,
            extra_headers: self.target.extra_headers.clone(),
        })
    }

    // Sends an already-normalized upstream request.
    async fn send_http_request(&self, request: GeminiHttpRequest) -> Result<ChatResponse> {
        match self.transport.send(request).await? {
            GeminiHttpResponse::Buffered(body) => Ok(ChatResponse::gemini_completion(body)),
            GeminiHttpResponse::Stream(stream) => Ok(ChatResponse::GeminiStream(stream)),
        }
    }
}

#[async_trait]
impl LlmBackend for GeminiNativeBackend {
    fn supported_request_types(&self) -> &[ChatRequestType] {
        &GEMINI_ONLY
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
struct GeminiHttpRequest {
    /// Target ID used only for logging and diagnostics.
    target_id: LlmTargetId,
    /// Fully resolved generateContent or streamGenerateContent URL.
    url: String,
    /// Per-target API key or process environment fallback.
    api_key: Option<String>,
    /// Already-normalized Gemini request body without synthetic fields.
    body: Value,
    /// Model resolved into the URL, kept for error reporting.
    model: String,
    /// Whether the upstream call should be treated as SSE.
    stream: bool,
    /// Per-target headers merged onto the outbound request.
    extra_headers: BTreeMap<String, String>,
}

enum GeminiHttpResponse {
    /// Complete JSON response from a non-streaming upstream call.
    Buffered(Value),
    /// Streamed SSE response converted into Switchyard stream events.
    Stream(BoxResponseStream),
}

#[async_trait]
trait GeminiTransport: Send + Sync {
    /// Sends one already-normalized Gemini generateContent request.
    async fn send(&self, request: GeminiHttpRequest) -> Result<GeminiHttpResponse>;
}

struct ReqwestGeminiTransport {
    /// Reused async HTTP client with configured timeout behavior.
    client: reqwest::Client,
}

impl ReqwestGeminiTransport {
    fn new(timeout_secs: Option<f64>) -> Result<Self> {
        let client = build_reqwest_client("Gemini", timeout_secs)?;
        Ok(Self { client })
    }
}

#[async_trait]
impl GeminiTransport for ReqwestGeminiTransport {
    async fn send(&self, request: GeminiHttpRequest) -> Result<GeminiHttpResponse> {
        let target_id = request.target_id.clone();
        let mut builder = self.client.post(&request.url).json(&request.body);
        if let Some(api_key) = request.api_key {
            builder = builder.header("x-goog-api-key", api_key);
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
                "Gemini generateContent request failed"
            );
            SwitchyardError::Upstream(format!("Gemini generateContent request failed: {error}"))
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
                "Gemini generateContent returned error status"
            );
            if status == reqwest::StatusCode::BAD_REQUEST && is_context_overflow(&body) {
                return Err(SwitchyardError::ContextWindowExceeded {
                    target_id: target_id.to_string(),
                    model: request.model,
                    message: body,
                });
            }
            return Err(SwitchyardError::UpstreamHttp {
                provider: "Gemini generateContent".to_string(),
                status_code: status.as_u16(),
                body,
            });
        }

        if request.stream {
            return Ok(GeminiHttpResponse::Stream(gemini_sse_stream(response)));
        }

        let body = response.json::<Value>().await.map_err(|error| {
            SwitchyardError::Upstream(format!(
                "Gemini generateContent returned invalid JSON: {error}"
            ))
        })?;
        Ok(GeminiHttpResponse::Buffered(body))
    }
}

fn validate_target_format(target: &LlmTarget) -> Result<()> {
    match target.format {
        BackendFormat::Gemini => Ok(()),
        BackendFormat::Auto
        | BackendFormat::OpenAi
        | BackendFormat::Responses
        | BackendFormat::Anthropic => Err(SwitchyardError::InvalidConfig(format!(
            "GeminiNativeBackend requires a target with resolved Gemini format, got {:?} for {}",
            target.format, target.id
        ))),
    }
}

// Removes a top-level field from an object body, returning its value.
fn take_body_field(body: &mut Value, key: &str) -> Option<Value> {
    body.as_object_mut().and_then(|object| object.remove(key))
}

// Builds the generateContent URL; the verb and `alt=sse` encode streaming.
fn generate_content_url(base_url: Option<&str>, model: &str, stream: bool) -> String {
    let base_url = base_url
        .unwrap_or(DEFAULT_GEMINI_BASE_URL)
        .trim_end_matches('/');
    let base_url = if base_url.ends_with("/v1beta") {
        base_url.to_string()
    } else {
        format!("{base_url}/v1beta")
    };
    if stream {
        format!("{base_url}/models/{model}:streamGenerateContent?alt=sse")
    } else {
        format!("{base_url}/models/{model}:generateContent")
    }
}

fn gemini_api_key(configured: Option<&str>) -> Option<String> {
    // Resolve per call so long-lived backends can pick up rotated environment credentials.
    configured
        .map(str::to_string)
        .or_else(|| env::var(GEMINI_API_KEY_ENV).ok())
        .filter(|value| !value.trim().is_empty())
}

// Canonical Gemini overflow 400 looks like
// `{"error":{"code":400,"message":"The input token count (N) exceeds the maximum
// number of tokens allowed (M).","status":"INVALID_ARGUMENT"}}`. The status is a
// generic INVALID_ARGUMENT, so detection is phrase-based only.
const GEMINI_OVERFLOW_PHRASES: &[&str] = &[
    "input token count",
    "exceeds the maximum number of tokens",
    "context window",
];

fn is_context_overflow(body: &str) -> bool {
    super::context_overflow::is_overflow_body(body, |_| false, GEMINI_OVERFLOW_PHRASES)
}

fn gemini_sse_stream(response: reqwest::Response) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut chunks = response.bytes_stream();
        let mut buffer = Vec::new();

        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.map_err(|error| {
                SwitchyardError::Upstream(format!("Gemini stream read failed: {error}"))
            })?;
            buffer.extend_from_slice(&chunk);

            // Gemini SSE frames are unnamed `data:` lines, each carrying one
            // complete GenerateContentResponse chunk; there is no done marker.
            while let Some(frame) = drain_next_sse_frame(&mut buffer, "Gemini")? {
                match parse_json_sse_frame(&frame, "Gemini", None)? {
                    ParsedSseFrame::Json(value) => yield StreamEvent::Json(value),
                    ParsedSseFrame::Done | ParsedSseFrame::Empty => {}
                }
            }
        }

        // Preserve the last frame when an upstream closes without a final SSE
        // separator.
        if has_non_whitespace_bytes(&buffer) {
            let frame = decode_sse_frame(&buffer, "Gemini")?;
            match parse_json_sse_frame(&frame, "Gemini", None)? {
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

    struct FakeGeminiTransport {
        requests: Mutex<Vec<GeminiHttpRequest>>,
        response: Mutex<Option<Result<GeminiHttpResponse>>>,
    }

    impl FakeGeminiTransport {
        fn with_response(response: GeminiHttpResponse) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: Mutex::new(Some(Ok(response))),
            }
        }

        fn with_error(message: &str) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: Mutex::new(Some(Err(SwitchyardError::Upstream(message.to_string())))),
            }
        }

        fn recorded_requests(&self) -> Result<Vec<GeminiHttpRequest>> {
            Ok(self
                .requests
                .lock()
                .map_err(|_| {
                    SwitchyardError::Other("fake transport request mutex poisoned".to_string())
                })?
                .clone())
        }
    }

    #[async_trait]
    impl GeminiTransport for FakeGeminiTransport {
        async fn send(&self, request: GeminiHttpRequest) -> Result<GeminiHttpResponse> {
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

    fn gemini_target() -> LlmTarget {
        LlmTarget {
            id: LlmTargetId::from_static("primary"),
            model: ModelId::from_static("gemini-2.5-flash"),
            format: BackendFormat::Gemini,
            endpoint: EndpointConfig {
                base_url: Some("https://example.test".to_string()),
                api_key: Some("secret".to_string()),
                timeout_secs: None,
            },
            extra_body: None,
            extra_headers: BTreeMap::new(),
        }
    }

    #[tokio::test]
    async fn transport_errors_are_backend_errors() -> Result<()> {
        let transport = Arc::new(FakeGeminiTransport::with_error("upstream exploded"));
        let backend = GeminiNativeBackend::with_transport(gemini_target(), transport)?;
        let request = ChatRequest::gemini(json!({
            "model": "client-model",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
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
        let mut target = gemini_target();
        target.format = BackendFormat::OpenAi;

        let Err(error) = GeminiNativeBackend::new(target) else {
            return Err(SwitchyardError::Other(
                "OpenAI target should be rejected".to_string(),
            ));
        };

        assert!(matches!(error, SwitchyardError::InvalidConfig(_)));
        Ok(())
    }

    // Synthetic model/stream fields must move into the URL, not the body.
    #[tokio::test]
    async fn moves_model_and_stream_into_the_url() -> Result<()> {
        let transport = Arc::new(FakeGeminiTransport::with_response(
            GeminiHttpResponse::Buffered(json!({"candidates": []})),
        ));
        let backend = GeminiNativeBackend::with_transport(gemini_target(), transport.clone())?;
        let request = ChatRequest::gemini(json!({
            "model": "client-model",
            "stream": true,
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }));

        backend.call_without_context(&request).await?;

        let requests = transport.recorded_requests()?;
        let sent = requests.first().ok_or_else(|| {
            SwitchyardError::Other("transport should record one request".to_string())
        })?;
        assert_eq!(
            sent.url,
            "https://example.test/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
        assert!(sent.stream);
        assert!(sent.body.get("model").is_none());
        assert!(sent.body.get("stream").is_none());
        assert!(sent.body.get("contents").is_some());
        Ok(())
    }

    #[test]
    fn formats_generate_content_urls_for_root_and_versioned_bases() {
        assert_eq!(
            generate_content_url(None, "gemini-2.5-flash", false),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
        assert_eq!(
            generate_content_url(Some("https://example.test/"), "m", false),
            "https://example.test/v1beta/models/m:generateContent"
        );
        assert_eq!(
            generate_content_url(Some("https://example.test/v1beta"), "m", true),
            "https://example.test/v1beta/models/m:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn gemini_context_overflow_canonical_shape_matches() {
        let body = r#"{"error":{"code":400,"message":"The input token count (1189926) exceeds the maximum number of tokens allowed (1048576).","status":"INVALID_ARGUMENT"}}"#;
        assert!(is_context_overflow(body));
    }

    #[test]
    fn gemini_context_overflow_unrelated_400_does_not_match() {
        let body = r#"{"error":{"code":400,"message":"Invalid JSON payload received.","status":"INVALID_ARGUMENT"}}"#;
        assert!(!is_context_overflow(body));
    }

    #[test]
    fn parses_gemini_sse_json_frames() -> Result<()> {
        let ParsedSseFrame::Json(value) = parse_json_sse_frame(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]}\n",
            "Gemini",
            None,
        )?
        else {
            return Err(SwitchyardError::Other(
                "JSON frame should produce a JSON value".to_string(),
            ));
        };
        assert!(value["candidates"].is_array());
        Ok(())
    }
}
