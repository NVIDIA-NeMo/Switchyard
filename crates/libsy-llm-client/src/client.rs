// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`TranslatingLlmClient`] — the crate's single public entry point: encode a neutral
//! request, call the configured backend over HTTP, decode the neutral response.

use std::collections::{BTreeMap, HashMap};

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::RequestBuilder;
use serde_json::Value;
use switchyard_protocol::{
    Context, Decision, LlmResponse, Metadata, Request, Response, RoutedLlmClient,
};
use switchyard_translation::{
    decode_aggregated_response, decode_request, decode_stream, encode_aggregated_response,
    encode_request, encode_stream, WireFormat,
};

use crate::backend::Backend;
use crate::error::{LlmClientError, Result};
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
    "anthropic-version",
    "content-type",
];

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
            mut llm_request,
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

        // The resolved name is the upstream model id (per the crate contract).
        llm_request.model = Some(model.clone());

        let mut body = encode_request(&llm_request, wire_format)
            .map_err(|error| LlmClientError::RequestEncoding(error.to_string()))?;
        // `encode_request` round-trips a preserved same-format body verbatim,
        // which keeps the caller's original `model`; force the resolved model so
        // the upstream always sees the target id.
        set_json_model(&mut body, &model);
        let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

        let builder = self.client.post(backend.url()).json(&body);
        let builder = forward_metadata_headers(builder, metadata.as_ref());
        let builder = apply_extra_headers(builder, backend);
        let builder = backend.apply_auth(builder);

        let http_response = builder.send().await.map_err(classify_transport_error)?;
        let status = http_response.status();
        if !status.is_success() {
            let body = match http_response.text().await {
                Ok(body) => body,
                Err(error) if error.is_timeout() => {
                    return Err(LlmClientError::Timeout {
                        source: Box::new(error),
                    })
                }
                Err(error) => format!("<failed to read error body: {error}>"),
            };
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
            // Adapt the reqwest body stream to plain bytes; the SSE-decode itself is
            // transport-agnostic and lives in `switchyard-translation`.
            let bytes = http_response.bytes_stream().map(|chunk| {
                chunk.map(|bytes| bytes.to_vec()).map_err(|error| {
                    let error = if error.is_timeout() {
                        LlmClientError::Timeout {
                            source: Box::new(error),
                        }
                    } else {
                        LlmClientError::Transport {
                            source: Box::new(error),
                        }
                    };
                    Box::new(error) as Box<dyn std::error::Error + Send + Sync>
                })
            });
            let chunks = decode_stream(bytes, wire_format)
                .map_err(|source| LlmClientError::InvalidResponse { source })?;
            LlmResponse::Stream(chunks)
        } else {
            let body = http_response
                .json::<Value>()
                .await
                .map_err(classify_json_response_error)?;
            let agg = decode_aggregated_response(&body, wire_format)
                .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
            LlmResponse::Agg(agg)
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
    /// restamped with the model the request asked for, never the upstream id.
    ///
    /// `http_headers` are carried through as the request's
    /// [`Metadata::http_headers`] and forwarded to the upstream (minus the reserved
    /// set); pass `None` to forward nothing.
    pub async fn call_rewrite_model_raw(
        &self,
        ctx: Context,
        raw_http_request: Value,
        http_headers: Option<BTreeMap<String, String>>,
        model: Option<&str>,
        wire_format: WireFormat,
    ) -> Result<RawResponse> {
        let llm_request = decode_request(wire_format, &raw_http_request)
            .map_err(|error| LlmClientError::RequestTranslation(error.to_string()))?;
        // The model the client asked for; restamped onto the response so it never
        // leaks the upstream id.
        let requested_model = llm_request.model.clone();

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
                    encode_aggregated_response(&agg, wire_format, requested_model.as_deref())
                        .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
                Ok(RawResponse::Buffered(body))
            }
            LlmResponse::Stream(chunks) => {
                let events = encode_stream(chunks, wire_format, requested_model)
                    .map_err(|source| LlmClientError::InvalidResponse { source })?;
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

fn classify_transport_error(error: reqwest::Error) -> LlmClientError {
    if error.is_timeout() {
        LlmClientError::Timeout {
            source: Box::new(error),
        }
    } else {
        LlmClientError::Transport {
            source: Box::new(error),
        }
    }
}

fn classify_json_response_error(error: reqwest::Error) -> LlmClientError {
    // `Response::json` labels body collection and JSON parsing failures as
    // decode errors; only the latter is a translation failure.
    if error.is_timeout() {
        LlmClientError::Timeout {
            source: Box::new(error),
        }
    } else if error.is_body() {
        LlmClientError::Transport {
            source: Box::new(error),
        }
    } else if error.is_decode()
        && std::error::Error::source(&error).is_some_and(|source| source.is::<serde_json::Error>())
    {
        LlmClientError::ResponseTranslation(format!("invalid upstream JSON: {error}"))
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::error::Error;
    use std::io::{Read, Write};
    use std::thread::JoinHandle;

    use serde_json::json;
    use switchyard_protocol::{completion_text, text_request, LlmRequest};
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

    // A one-model config list: "gpt" served over OpenAI Chat at base_url.
    fn chat_map(base_url: &str) -> Vec<ModelConfig> {
        vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(config(base_url)),
            None,
        )]
    }

    fn truncated_response_server() -> std::io::Result<(String, JoinHandle<std::io::Result<()>>)> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 1024];
            if stream.read(&mut request)? == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "client closed before sending a request",
                ));
            }
            stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                  Content-Length: 100\r\nConnection: close\r\n\r\n{}",
            )
        });
        Ok((format!("http://{address}/v1"), handle))
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
    async fn missing_model_errors(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    async fn unknown_model_errors(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    async fn unknown_model_format_errors(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    fn backend_for_resolves_configured_format(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let client = TranslatingLlmClient::new(&chat_map("https://example.test/v1"))?;
        // "gpt" is served over OpenAI Chat only; other formats and models miss.
        assert!(client.backend_for("gpt", WireFormat::OpenAiChat).is_some());
        assert!(client
            .backend_for("gpt", WireFormat::AnthropicMessages)
            .is_none());
        assert!(client
            .backend_for("missing", WireFormat::OpenAiChat)
            .is_none());
        Ok(())
    }

    #[tokio::test]
    async fn model_name_arg_wins_over_request_model(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    async fn buffered_openai_chat_round_trips(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    async fn invalid_json_is_a_response_translation_error(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw("not json", "application/json"))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
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
        Ok(())
    }

    #[tokio::test]
    async fn response_body_io_failure_is_a_transport_error(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let (base_url, server) = truncated_response_server()?;
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
        assert!(!std::error::Error::source(&source)
            .is_some_and(|source| source.is::<serde_json::Error>()));
        Ok(())
    }

    #[tokio::test]
    async fn rewrites_model_to_resolved_upstream_id(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    async fn streaming_openai_chat_aggregates(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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

        let response = client
            .call_rewrite_model(Context::default(), request_for(Some("gpt"), true), None)
            .await?;
        assert!(matches!(response.llm_response, LlmResponse::Stream(_)));
        let agg = response.llm_response.into_agg().await?;
        assert_eq!(completion_text(&agg), "Hello world");
        Ok(())
    }

    #[tokio::test]
    async fn upstream_500_is_upstream_http(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    async fn routed_llm_client_exposes_timeout_variant(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    async fn context_overflow_400_is_mapped(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    async fn forwards_metadata_headers_except_reserved(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    async fn routed_llm_client_serves_the_decision_model(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    async fn invalid_raw_request_is_a_request_translation_error(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
    // Chat JSON, with the client-facing `model` restamped over the upstream id.
    #[tokio::test]
    async fn call_rewrite_model_raw_round_trips_buffered_json(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
        // The client sees the model it asked for, not the upstream "gpt".
        assert_eq!(body["model"], "client-facing");
        Ok(())
    }

    // Raw path, streaming: an inbound `stream: true` request yields an unframed stream
    // of OpenAI Chat chunk objects whose deltas reassemble the completion.
    #[tokio::test]
    async fn call_rewrite_model_raw_streams_wire_events(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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
        Ok(())
    }

    // Raw path forwards caller headers (minus the reserved set) to the upstream.
    #[tokio::test]
    async fn call_rewrite_model_raw_forwards_headers(
    ) -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
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

        let mut headers = BTreeMap::new();
        headers.insert("x-request-id".to_string(), "abc".to_string());
        headers.insert("authorization".to_string(), "Bearer client-key".to_string());

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
