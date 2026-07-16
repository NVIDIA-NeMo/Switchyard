// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`LlmModelClient`] — the crate's single public entry point: encode a neutral
//! request, call the configured backend over HTTP, decode the neutral response.

use std::collections::HashMap;

use async_stream::try_stream;
use futures_util::StreamExt;
use switchyard_protocol::{LlmResponse, LlmResponseStream, Metadata, Request, Response};
use reqwest::RequestBuilder;
use serde_json::Value;
use switchyard_translation::{
    StreamCodecRegistry, StreamTranslationState, TranslationEngine, TranslationPolicy, WireFormat,
};

use crate::backend::Backend;
use crate::error::{LlmClientError, Result};
use crate::sse::{
    decode_sse_frame, drain_next_sse_frame, has_non_whitespace_bytes, parse_json_sse_frame,
    ParsedSseFrame,
};

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
    "x-api-key",
    "anthropic-version",
    "content-type",
];

/// A client that dispatches neutral-IR requests to per-model HTTP backends.
///
/// Construct it with a two-layer map — model name to wire format to
/// [`Backend`] — so one model can be served over several upstream formats. Each
/// call resolves the model and format, encodes the request to that backend's
/// wire format, applies auth and forwarded headers, sends the HTTP request with
/// a shared [`reqwest::Client`], and decodes the response back to the neutral IR
/// (buffered or streamed).
pub struct LlmModelClient {
    model_name_to_config: HashMap<String, HashMap<WireFormat, Backend>>,
    client: reqwest::Client,
    engine: TranslationEngine,
    policy: TranslationPolicy,
}

impl LlmModelClient {
    /// Builds a client over the given model→format→backend map, with a fresh
    /// shared HTTP client and the built-in translation codecs.
    pub fn new(
        model_name_to_config: HashMap<String, HashMap<WireFormat, Backend>>,
    ) -> Result<Self> {
        let client = reqwest::Client::builder().build().map_err(|error| {
            LlmClientError::Transport(format!("failed to build HTTP client: {error}"))
        })?;
        Ok(Self {
            model_name_to_config,
            client,
            engine: TranslationEngine::default(),
            policy: TranslationPolicy::default(),
        })
    }

    /// The wire formats configured for `model`, or `None` when the model is
    /// unknown. Lets a caller pick a supported format without duplicating the
    /// map.
    pub fn formats_for(&self, model: &str) -> Option<Vec<WireFormat>> {
        self.model_name_to_config
            .get(model)
            .map(|backends| backends.keys().copied().collect())
    }

    /// Calls the backend for `model_name` (or the request's own model) at the
    /// given wire `format` and returns the neutral response.
    ///
    /// Resolution: `model_name` wins over `request.llm_request.model`; the
    /// resolved name is both the outer map key and the model id written into the
    /// request before translation. Errors with [`LlmClientError::MissingModel`]
    /// when neither is set, [`LlmClientError::UnknownModel`] when the model has
    /// no backends, and [`LlmClientError::UnknownModelFormat`] when the model has
    /// no backend for `format`.
    pub async fn call(
        &self,
        request: Request,
        model_name: Option<&str>,
        format: WireFormat,
    ) -> Result<Response> {
        // Own the request's parts so the model can be set without a `mut` param
        // and without cloning the messages. `raw_request` is unused here.
        let Request {
            mut llm_request,
            metadata,
            ..
        } = request;

        let model = model_name
            .map(str::to_string)
            .or_else(|| llm_request.model.clone())
            .ok_or(LlmClientError::MissingModel)?;

        let backend = self
            .model_name_to_config
            .get(&model)
            .ok_or_else(|| LlmClientError::UnknownModel(model.clone()))?
            .get(&format)
            .ok_or_else(|| LlmClientError::UnknownModelFormat {
                model: model.clone(),
                format,
            })?;

        // The resolved name is the upstream model id (per the crate contract).
        llm_request.model = Some(model.clone());

        let wire_format = backend.wire_format();
        let mut body = self
            .engine
            .encode_request(wire_format, &llm_request, &self.policy)
            .map_err(|error| LlmClientError::Translation(error.to_string()))?
            .body;
        // `encode_request` round-trips a preserved same-format body verbatim,
        // which keeps the caller's original `model`; force the resolved model so
        // the upstream always sees the target id.
        set_json_model(&mut body, &model);
        let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

        let builder = self.client.post(backend.url()).json(&body);
        let builder = forward_metadata_headers(builder, metadata.as_ref());
        let builder = apply_extra_headers(builder, backend);
        let builder = backend.apply_auth(builder);

        let http_response = builder
            .send()
            .await
            .map_err(|error| LlmClientError::Transport(error.to_string()))?;
        let status = http_response.status();
        if !status.is_success() {
            let body = http_response
                .text()
                .await
                .unwrap_or_else(|error| format!("<failed to read error body: {error}>"));
            if status == reqwest::StatusCode::BAD_REQUEST && backend.is_context_overflow(&body) {
                return Err(LlmClientError::ContextWindowExceeded {
                    model,
                    message: body,
                });
            }
            return Err(LlmClientError::UpstreamHttp {
                status: status.as_u16(),
                body,
            });
        }

        let llm_response = if streaming {
            LlmResponse::Stream(decode_stream(http_response, wire_format))
        } else {
            let body = http_response.json::<Value>().await.map_err(|error| {
                LlmClientError::Transport(format!("invalid upstream JSON: {error}"))
            })?;
            let decoded = self
                .engine
                .decode_response(wire_format, &body, &self.policy)
                .map_err(|error| LlmClientError::Translation(error.to_string()))?;
            LlmResponse::Agg(decoded.response)
        };

        Ok(Response {
            llm_response,
            metadata,
        })
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
        if is_reserved_header(name) {
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

// Case-insensitive membership test against RESERVED_HEADERS.
fn is_reserved_header(name: &str) -> bool {
    RESERVED_HEADERS
        .iter()
        .any(|reserved| name.eq_ignore_ascii_case(reserved))
}

// The terminal SSE marker for the format, if any. OpenAI Chat and Responses use
// `[DONE]`; Anthropic ends on its `message_stop` event with no marker.
fn done_marker(format: WireFormat) -> Option<&'static str> {
    match format {
        WireFormat::OpenAiChat | WireFormat::OpenAiResponses => Some("[DONE]"),
        WireFormat::AnthropicMessages => None,
    }
}

// Reads the upstream SSE body, draining frames and decoding each into neutral IR
// chunks. Partial frames are preserved across TCP chunks.
//
// The stream codec is resolved once and reused for every frame: it is stateless
// (all per-response state lives in `StreamTranslationState`), so rebuilding the
// registry per frame — as the `decode_stream_event` free function does — would
// allocate a throwaway registry for each of a response's many chunks.
fn decode_stream(response: reqwest::Response, source: WireFormat) -> LlmResponseStream {
    let marker = done_marker(source);
    // Source is always a built-in wire format, so this lookup cannot fail; a
    // failure still surfaces as a single error item rather than a panic.
    let codec = match StreamCodecRegistry::with_builtins().codec(source) {
        Ok(codec) => codec,
        Err(error) => {
            let boxed: Box<dyn std::error::Error + Send + Sync> =
                Box::new(LlmClientError::Stream(error.to_string()));
            return Box::pin(futures::stream::once(async move { Err(boxed) }));
        }
    };
    Box::pin(try_stream! {
        let mut state = StreamTranslationState::default();
        let mut bytes = response.bytes_stream();
        let mut buffer = Vec::new();

        while let Some(chunk) = bytes.next().await {
            let chunk = chunk
                .map_err(|error| LlmClientError::Stream(format!("stream read failed: {error}")))?;
            buffer.extend_from_slice(&chunk);

            while let Some(frame) = drain_next_sse_frame(&mut buffer)? {
                match parse_json_sse_frame(&frame, marker)? {
                    ParsedSseFrame::Json(value) => {
                        for event in codec.decode_event(&mut state, &value) {
                            yield event;
                        }
                    }
                    ParsedSseFrame::Done => return,
                    ParsedSseFrame::Empty => {}
                }
            }
        }

        // A non-standard upstream might omit the final blank line; parse a
        // trailing complete frame instead of losing its last chunk.
        if has_non_whitespace_bytes(&buffer) {
            let frame = decode_sse_frame(&buffer)?;
            if let ParsedSseFrame::Json(value) = parse_json_sse_frame(&frame, marker)? {
                for event in codec.decode_event(&mut state, &value) {
                    yield event;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use switchyard_protocol::{completion_text, text_request, LlmRequest};
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::backend::HttpBackendConfig;

    fn config(base_url: &str) -> HttpBackendConfig {
        HttpBackendConfig {
            base_url: base_url.to_string(),
            api_key: Some("secret".to_string()),
            extra_headers: BTreeMap::new(),
        }
    }

    // A one-model, one-format map: "gpt" served over OpenAI Chat at base_url.
    fn chat_map(base_url: &str) -> HashMap<String, HashMap<WireFormat, Backend>> {
        HashMap::from([(
            "gpt".to_string(),
            HashMap::from([(
                WireFormat::OpenAiChat,
                Backend::OpenAiChat(config(base_url)),
            )]),
        )])
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
    fn reserved_headers_are_case_insensitive() {
        assert!(is_reserved_header("Authorization"));
        assert!(is_reserved_header("content-type"));
        assert!(is_reserved_header("X-Api-Key"));
        assert!(!is_reserved_header("x-request-id"));
    }

    #[tokio::test]
    async fn missing_model_errors() {
        let client = LlmModelClient::new(HashMap::new()).unwrap();
        let Err(error) = client
            .call(request_for(None, false), None, WireFormat::OpenAiChat)
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(error, LlmClientError::MissingModel));
    }

    #[tokio::test]
    async fn unknown_model_errors() {
        let client = LlmModelClient::new(HashMap::new()).unwrap();
        let Err(error) = client
            .call(
                request_for(Some("gpt"), false),
                None,
                WireFormat::OpenAiChat,
            )
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(error, LlmClientError::UnknownModel(model) if model == "gpt"));
    }

    #[tokio::test]
    async fn unknown_model_format_errors() {
        // "gpt" exists but only over OpenAI Chat; ask for Anthropic.
        let client = LlmModelClient::new(chat_map("https://example.test/v1")).unwrap();
        let Err(error) = client
            .call(
                request_for(Some("gpt"), false),
                None,
                WireFormat::AnthropicMessages,
            )
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::UnknownModelFormat { model, format }
                if model == "gpt" && format == WireFormat::AnthropicMessages
        ));
    }

    #[test]
    fn formats_for_lists_configured_formats() {
        let client = LlmModelClient::new(chat_map("https://example.test/v1")).unwrap();
        assert_eq!(
            client.formats_for("gpt"),
            Some(vec![WireFormat::OpenAiChat])
        );
        assert_eq!(client.formats_for("missing"), None);
    }

    #[tokio::test]
    async fn model_name_arg_wins_over_request_model() {
        let client = LlmModelClient::new(HashMap::new()).unwrap();
        // Arg "b" is looked up (and reported), not the request's "a".
        let Err(error) = client
            .call(
                request_for(Some("a"), false),
                Some("b"),
                WireFormat::OpenAiChat,
            )
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(error, LlmClientError::UnknownModel(model) if model == "b"));
    }

    #[tokio::test]
    async fn buffered_openai_chat_round_trips() {
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

        let client = LlmModelClient::new(chat_map(&format!("{}/v1", server.uri()))).unwrap();

        let response = client
            .call(
                request_for(Some("gpt"), false),
                None,
                WireFormat::OpenAiChat,
            )
            .await
            .unwrap();
        let agg = response.llm_response.into_agg().expect("buffered response");
        assert_eq!(completion_text(&agg), "Hi there");
    }

    #[tokio::test]
    async fn rewrites_model_to_resolved_upstream_id() {
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

        let client = LlmModelClient::new(chat_map(&format!("{}/v1", server.uri()))).unwrap();
        // Inbound model differs from the map key / resolved model.
        client
            .call(
                request_for(Some("switchyard"), false),
                Some("gpt"),
                WireFormat::OpenAiChat,
            )
            .await
            .unwrap();
        // The body_partial_json matcher asserts the upstream saw model "gpt".
    }

    #[tokio::test]
    async fn streaming_openai_chat_aggregates() {
        let server = MockServer::start().await;
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
             data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .mount(&server)
            .await;

        let client = LlmModelClient::new(chat_map(&format!("{}/v1", server.uri()))).unwrap();

        let response = client
            .call(request_for(Some("gpt"), true), None, WireFormat::OpenAiChat)
            .await
            .unwrap();
        assert!(matches!(response.llm_response, LlmResponse::Stream(_)));
        let agg = response.llm_response.aggregate().await.unwrap();
        assert_eq!(completion_text(&agg), "Hello world");
    }

    #[tokio::test]
    async fn upstream_500_is_upstream_http() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;

        let client = LlmModelClient::new(chat_map(&format!("{}/v1", server.uri()))).unwrap();

        let Err(error) = client
            .call(
                request_for(Some("gpt"), false),
                None,
                WireFormat::OpenAiChat,
            )
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::UpstreamHttp { status: 500, .. }
        ));
    }

    #[tokio::test]
    async fn context_overflow_400_is_mapped() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"code": "context_length_exceeded", "message": "too big"}
            })))
            .mount(&server)
            .await;

        let client = LlmModelClient::new(chat_map(&format!("{}/v1", server.uri()))).unwrap();

        let Err(error) = client
            .call(
                request_for(Some("gpt"), false),
                None,
                WireFormat::OpenAiChat,
            )
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::ContextWindowExceeded { model, .. } if model == "gpt"
        ));
    }

    #[tokio::test]
    async fn forwards_metadata_headers_except_reserved() {
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

        let mut headers = BTreeMap::new();
        headers.insert("x-request-id".to_string(), "abc".to_string());
        headers.insert("authorization".to_string(), "Bearer client-key".to_string());
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
            }),
        };

        let client = LlmModelClient::new(chat_map(&format!("{}/v1", server.uri()))).unwrap();

        // Matchers assert forwarded x-request-id survives and reserved
        // authorization is the backend's, not the client's.
        client
            .call(request, None, WireFormat::OpenAiChat)
            .await
            .unwrap();
    }
}
