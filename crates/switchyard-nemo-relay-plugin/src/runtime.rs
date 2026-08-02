// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::{Stream, StreamExt};
use nemo_relay_plugin::{
    Json, LlmContinuationFailureV2, LlmContinuationInvocationV2, LlmContinuationTargetV2,
    LlmContinuationV2, LlmJsonAsyncStreamV2, LlmNonHttpFailureKindV2, LlmRequest as RelayRequest,
    LlmStreamContinuationV2, LlmStreamExecutionOutcomeV2, PluginRuntime,
};
use serde_json::{json, Map};
use switchyard_libsy::{Algorithm, CallLlmRequest, LibsyError, Step};
use switchyard_protocol::{
    Context, Decision, LlmClientError, LlmRequest as SwitchyardLlmRequest, LlmResponse,
    LlmResponseChunk, LlmResponseStream, LlmResponseStreamEvent, Metadata, Request, Response,
    WireFormat,
};
use switchyard_translation::{StreamTranslationState, TranslationEngine};

use crate::config::{protocol_from_call, PreparedTargetBinding, SwitchyardConfig};
use crate::translation;

pub(crate) struct SwitchyardRuntime {
    max_retries: u32,
    algorithm: Arc<dyn Algorithm>,
    targets: BTreeMap<String, PreparedTargetBinding>,
    default_targets: BTreeMap<WireFormat, String>,
    translation: Arc<TranslationEngine>,
    relay: PluginRuntime,
}

impl SwitchyardRuntime {
    pub(crate) fn new(config: SwitchyardConfig, relay: PluginRuntime) -> Result<Self, String> {
        let prepared = config.prepare()?;
        Ok(Self {
            max_retries: prepared.max_retries,
            algorithm: prepared.algorithm,
            targets: prepared.targets,
            default_targets: prepared.default_targets,
            translation: Arc::new(TranslationEngine::default()),
            relay,
        })
    }

    pub(crate) async fn execute_buffered(
        &self,
        name: String,
        request: RelayRequest,
        continuation: LlmContinuationV2,
    ) -> Result<Json, String> {
        let Some(inbound) = self.managed_protocol(&name) else {
            return continuation.call_passthrough(request).await;
        };
        let libsy_request = self.libsy_request(inbound, &request, false)?;
        let metadata = identity_metadata(libsy_request.metadata.as_ref());
        let max_attempts = self.max_retries + 1;
        for attempt in 1..=max_attempts {
            self.mark(
                "switchyard.routing.requested",
                json!({"algorithm": self.algorithm.name(), "attempt": attempt}),
                &metadata,
            );
            match self
                .drive_buffered(libsy_request.clone(), &continuation, attempt, &metadata)
                .await
            {
                Ok(response) => {
                    return match response.llm_response {
                        LlmResponse::Agg(response) => {
                            translation::encode_response(&self.translation, inbound, &response)
                        }
                        LlmResponse::Stream(_) => {
                            Err("libsy returned a stream for a buffered request".into())
                        }
                    };
                }
                Err(failure) if libsy_error_retryable(&failure) && attempt < max_attempts => {
                    self.mark(
                        "switchyard.routing.retry",
                        failure_mark_data(attempt, &failure),
                        &metadata,
                    );
                }
                Err(failure) => {
                    self.mark(
                        "switchyard.routing.error",
                        failure_mark_data(attempt, &failure),
                        &metadata,
                    );
                    return self
                        .fallback_buffered(inbound, libsy_request, &continuation, &metadata)
                        .await;
                }
            }
        }
        Err("Switchyard retry loop ended without a result".into())
    }

    async fn drive_buffered(
        &self,
        request: Request,
        continuation: &LlmContinuationV2,
        attempt: u32,
        mark_metadata: &Json,
    ) -> Result<Response, LibsyError> {
        let mut steps = self
            .algorithm
            .clone()
            .run_stream(Context::default(), request, None);
        while let Some(step) = steps.next().await {
            match step {
                Ok(Step::Decision(decision)) => {
                    self.emit_decision(decision.as_ref(), attempt, mark_metadata);
                }
                Ok(Step::CallLlm(call)) => {
                    self.serve_buffered_call(*call, continuation).await?;
                }
                Ok(Step::ReturnToAgent(response)) => return Ok(*response),
                Err(error) => return Err(error),
            }
        }
        Err(LibsyError::MissingFinalResponse)
    }

    async fn serve_buffered_call(
        &self,
        call: CallLlmRequest,
        continuation: &LlmContinuationV2,
    ) -> switchyard_libsy::Result<()> {
        let target_name = call.get_decision().selected_model().to_string();
        let request = call.get_request().llm_request.clone();
        let result = async {
            let target = self.target(&target_name)?;
            let request = self.dispatch_request(target, request, false)?;
            let response = continuation.call(request).await.map_err(client_error)?;
            let response =
                translation::decode_response(&self.translation, target.protocol, &response)
                    .map_err(LlmClientError::ResponseTranslation)?;
            Ok(Response {
                llm_response: LlmResponse::Agg(response),
                metadata: Some(Metadata {
                    wire_format: Some(target.protocol),
                    ..Metadata::default()
                }),
            })
        }
        .await
        .map_err(|source| LibsyError::client_call(target_name, source));
        call.respond(result)
    }

    async fn fallback_buffered(
        &self,
        inbound: WireFormat,
        request: Request,
        continuation: &LlmContinuationV2,
        metadata: &Json,
    ) -> Result<Json, String> {
        let target_name = self.default_target(inbound);
        let target = self
            .target(target_name)
            .map_err(|error| error.to_string())?;
        self.mark(
            "switchyard.routing.fallback",
            json!({"selected_target": target_name}),
            metadata,
        );
        let dispatch = self
            .dispatch_request(target, request.llm_request, false)
            .map_err(|error| error.to_string())?;
        let response = continuation
            .call(dispatch)
            .await
            .map_err(|error| format!("trusted fallback failed: {error:?}"))?;
        let response = translation::decode_response(&self.translation, target.protocol, &response)?;
        translation::encode_response(&self.translation, inbound, &response)
    }

    fn libsy_request(
        &self,
        inbound: WireFormat,
        original: &RelayRequest,
        streaming: bool,
    ) -> Result<Request, String> {
        let mut request = translation::decode_request(&self.translation, inbound, original)?;
        request.stream = streaming;
        let headers = string_headers(&original.headers);
        let mut metadata = Metadata::from_headers(&headers);
        metadata.wire_format = Some(inbound);
        Ok(Request {
            llm_request: request,
            raw_request: None,
            metadata: Some(metadata),
        })
    }

    fn dispatch_request(
        &self,
        target: &PreparedTargetBinding,
        mut request: SwitchyardLlmRequest,
        streaming: bool,
    ) -> Result<LlmContinuationInvocationV2, LlmClientError> {
        request.stream = streaming;
        let headers = target.headers.clone();
        let mut request = translation::encode_request(&self.translation, target.protocol, &request)
            .map_err(LlmClientError::RequestEncoding)?;
        let body = request.content.as_object_mut().ok_or_else(|| {
            LlmClientError::RequestEncoding("translated provider request is not an object".into())
        })?;
        body.insert("model".into(), Json::String(target.model.clone()));
        body.insert("stream".into(), Json::Bool(streaming));
        Ok(LlmContinuationInvocationV2 {
            request,
            target: LlmContinuationTargetV2 {
                url: target.dispatch_url().to_string(),
                headers,
            },
        })
    }

    fn target(&self, name: &str) -> Result<&PreparedTargetBinding, LlmClientError> {
        self.targets
            .get(name)
            .ok_or_else(|| LlmClientError::Configuration {
                message: format!("libsy selected unknown target {name:?}"),
            })
    }

    fn managed_protocol(&self, name: &str) -> Option<WireFormat> {
        protocol_from_call(name).filter(|protocol| self.default_targets.contains_key(protocol))
    }

    fn default_target(&self, protocol: WireFormat) -> &str {
        self.default_targets
            .get(&protocol)
            .expect("managed protocol must have a default target")
    }

    fn mark(&self, name: &str, data: Json, metadata: &Json) {
        if let Err(error) = self.relay.emit_mark(name, Some(&data), Some(metadata)) {
            eprintln!("Switchyard could not emit routing mark {name:?}: {error}");
        }
    }

    fn emit_decision(&self, decision: &dyn Decision, attempt: u32, metadata: &Json) {
        self.mark(
            "switchyard.routing.decision",
            json!({
                "algorithm": self.algorithm.name(),
                "attempt": attempt,
                "selected_target": decision.selected_model(),
                "reasoning": decision.reasoning(),
                "routing_tier": decision.routing_tier(),
                "is_routed_call": decision.is_routed_call(),
            }),
            metadata,
        );
    }

    pub(crate) async fn execute_stream(
        self: Arc<Self>,
        name: String,
        request: RelayRequest,
        continuation: LlmStreamContinuationV2,
    ) -> Result<LlmStreamExecutionOutcomeV2, String> {
        let Some(inbound) = self.managed_protocol(&name) else {
            return Ok(LlmStreamExecutionOutcomeV2::Passthrough(request));
        };
        let libsy_request = self.libsy_request(inbound, &request, true)?;
        let metadata = identity_metadata(libsy_request.metadata.as_ref());
        let stream = self.routed_stream(inbound, libsy_request, continuation, metadata);
        Ok(LlmStreamExecutionOutcomeV2::Stream(stream))
    }

    fn routed_stream(
        self: Arc<Self>,
        inbound: WireFormat,
        request: Request,
        continuation: LlmStreamContinuationV2,
        metadata: Json,
    ) -> LlmJsonAsyncStreamV2 {
        Box::pin(async_stream::try_stream! {
        let max_attempts = self.max_retries + 1;
        for attempt in 1..=max_attempts {
            self.mark(
                "switchyard.routing.requested",
                json!({"algorithm": self.algorithm.name(), "attempt": attempt}),
                &metadata,
            );
            let failure = match self
                .drive_stream(
                    request.clone(),
                    &continuation,
                    attempt,
                    &metadata,
                )
                .await
            {
                Ok(response) => {
                    let mut returned = Self::returned_stream(
                        response,
                        inbound,
                        Arc::clone(&self.translation),
                    );
                    let mut committed = false;
                    loop {
                        match returned.next().await {
                            Some(Ok(event)) => {
                                committed = true;
                                yield event;
                            }
                            Some(Err(failure)) if committed => {
                                self.mark(
                                    "switchyard.routing.error",
                                    failure_mark_data(attempt, &failure),
                                    &metadata,
                                );
                                Err(format!(
                                    "Switchyard stream failed after response commitment: {failure}"
                                ))?;
                            }
                            Some(Err(failure)) => break failure,
                            None => return,
                        }
                    }
                }
                Err(failure) => failure,
            };
            if libsy_error_retryable(&failure) && attempt < max_attempts {
                self.mark(
                    "switchyard.routing.retry",
                    failure_mark_data(attempt, &failure),
                    &metadata,
                );
                continue;
            }
            self.mark(
                "switchyard.routing.error",
                failure_mark_data(attempt, &failure),
                &metadata,
            );
            let mut fallback = self
                .fallback_stream(inbound, request.clone(), &continuation, &metadata)
                .await?;
            while let Some(item) = fallback.next().await {
                yield item.map_err(|failure| {
                    format!("trusted fallback stream failed: {failure}")
                })?;
            }
            return;
        }
        Err("Switchyard stream retry loop ended without a result".to_string())?;
        })
    }

    async fn drive_stream(
        &self,
        request: Request,
        continuation: &LlmStreamContinuationV2,
        attempt: u32,
        mark_metadata: &Json,
    ) -> Result<Response, LibsyError> {
        let mut steps = self
            .algorithm
            .clone()
            .run_stream(Context::default(), request, None);
        while let Some(step) = steps.next().await {
            match step {
                Ok(Step::Decision(decision)) => {
                    self.emit_decision(decision.as_ref(), attempt, mark_metadata);
                }
                Ok(Step::CallLlm(call)) => {
                    self.serve_stream_call(*call, continuation).await?;
                }
                Ok(Step::ReturnToAgent(response)) => return Ok(*response),
                Err(error) => return Err(error),
            }
        }
        Err(LibsyError::MissingFinalResponse)
    }

    async fn serve_stream_call(
        &self,
        call: CallLlmRequest,
        continuation: &LlmStreamContinuationV2,
    ) -> switchyard_libsy::Result<()> {
        let target_name = call.get_decision().selected_model().to_string();
        let request = call.get_request();
        let llm_request = request.llm_request.clone();
        let metadata = request.metadata.clone();
        let result = self
            .provider_stream_response(&target_name, llm_request, metadata, continuation)
            .await
            .map_err(|source| LibsyError::client_call(target_name, source));
        call.respond(result)
    }

    async fn provider_stream_response(
        &self,
        target_name: &str,
        request: SwitchyardLlmRequest,
        metadata: Option<Metadata>,
        continuation: &LlmStreamContinuationV2,
    ) -> Result<Response, LlmClientError> {
        let target = self.target(target_name)?;
        let dispatch = self.dispatch_request(target, request, true)?;
        let mut upstream = continuation
            .open_stream(dispatch)
            .await
            .map_err(client_error)?;
        let first_raw = match upstream.next().await {
            Some(Ok(first)) => first,
            Some(Err(error)) => return Err(client_error(error)),
            None => {
                return Err(LlmClientError::InvalidResponse {
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "provider returned an empty stream",
                    )),
                });
            }
        };
        let mut state = StreamTranslationState::new(target.protocol, target.protocol);
        let first =
            decode_provider_event(&self.translation, &mut state, target.protocol, first_raw)?;
        let protocol = target.protocol;
        let translation = Arc::clone(&self.translation);
        let stream: LlmResponseStream = Box::pin(async_stream::try_stream! {
            yield first;
            while let Some(item) = upstream.next().await {
                let raw = item.map_err(client_error)?;
                yield decode_provider_event(&translation, &mut state, protocol, raw)?;
            }
        });
        Ok(Response {
            llm_response: LlmResponse::Stream(stream),
            metadata: Some(Metadata {
                wire_format: Some(target.protocol),
                ..metadata.unwrap_or_default()
            }),
        })
    }

    fn returned_stream(
        response: Response,
        inbound: WireFormat,
        translation: Arc<TranslationEngine>,
    ) -> ReturnedJsonStream {
        Box::pin(async_stream::stream! {
            let source = match response
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.wire_format)
            {
                Some(source) => source,
                None => {
                    yield Err(returned_stream_translation_error(
                        "libsy returned a stream without a supported source wire format",
                    ));
                    return;
                }
            };
            let LlmResponse::Stream(mut stream) = response.llm_response else {
                yield Err(returned_stream_translation_error(
                    "libsy returned a buffered response for a streaming request",
                ));
                return;
            };
            let mut state = StreamTranslationState::new(source, inbound);
            let mut emitted = false;
            while let Some(item) = stream.next().await {
                let event = match item {
                    Ok(event) => event,
                    Err(error) => {
                        yield Err(LibsyError::client_call("return_to_agent", error));
                        return;
                    }
                };
                let events = match translation::encode_stream_event(
                    &translation,
                    &mut state,
                    inbound,
                    event,
                ) {
                    Ok(events) => events,
                    Err(error) => {
                        yield Err(returned_stream_translation_error(error));
                        return;
                    }
                };
                for event in events {
                    emitted = true;
                    yield Ok(event);
                }
            }
            let events = match translation::finish_stream(&translation, &mut state, inbound) {
                Ok(events) => events,
                Err(error) => {
                    yield Err(returned_stream_translation_error(error));
                    return;
                }
            };
            for event in events {
                emitted = true;
                yield Ok(event);
            }
            if !emitted {
                yield Err(returned_stream_translation_error(
                    "Switchyard produced an empty output stream",
                ));
            }
        })
    }

    async fn fallback_stream(
        &self,
        inbound: WireFormat,
        request: Request,
        continuation: &LlmStreamContinuationV2,
        metadata: &Json,
    ) -> Result<ReturnedJsonStream, String> {
        let target_name = self.default_target(inbound);
        self.mark(
            "switchyard.routing.fallback",
            json!({"selected_target": target_name}),
            metadata,
        );
        let response = self
            .provider_stream_response(
                target_name,
                request.llm_request,
                request.metadata,
                continuation,
            )
            .await
            .map_err(|error| format!("trusted fallback stream failed: {error}"))?;
        Ok(Self::returned_stream(
            response,
            inbound,
            Arc::clone(&self.translation),
        ))
    }
}

type ReturnedJsonStream = Pin<Box<dyn Stream<Item = Result<Json, LibsyError>> + Send>>;

fn decode_provider_event(
    translation: &TranslationEngine,
    state: &mut StreamTranslationState,
    protocol: WireFormat,
    raw: Json,
) -> Result<LlmResponseStreamEvent, LlmClientError> {
    let event = translation::decode_stream_event(translation, state, protocol, raw)
        .map_err(LlmClientError::ResponseTranslation)?;
    for chunk in event.normalized() {
        match chunk {
            LlmResponseChunk::DecodeError { message } => {
                return Err(LlmClientError::ResponseTranslation(message.clone()));
            }
            LlmResponseChunk::StreamError { message } => {
                return Err(LlmClientError::UpstreamHttp {
                    status: 502,
                    body: message.clone(),
                });
            }
            _ => {}
        }
    }
    Ok(event)
}

fn returned_stream_translation_error(error: impl Into<String>) -> LibsyError {
    LibsyError::client_call(
        "return_to_agent",
        LlmClientError::ResponseTranslation(error.into()),
    )
}

fn libsy_error_retryable(error: &LibsyError) -> bool {
    let LibsyError::ClientCall { source, .. } = error else {
        return false;
    };
    match source {
        LlmClientError::UpstreamHttp { status, .. } => {
            matches!(*status, 408 | 425 | 429 | 500 | 502 | 503 | 504)
        }
        LlmClientError::Transport { .. } | LlmClientError::Timeout { .. } => true,
        _ => false,
    }
}

fn client_error(error: LlmContinuationFailureV2) -> LlmClientError {
    match error {
        LlmContinuationFailureV2::Http { status, body, .. } => {
            LlmClientError::UpstreamHttp { status, body }
        }
        LlmContinuationFailureV2::NonHttp { kind, message } => match kind {
            LlmNonHttpFailureKindV2::Transport => LlmClientError::Transport {
                source: Box::new(std::io::Error::other(message)),
            },
            LlmNonHttpFailureKindV2::Timeout => LlmClientError::Timeout {
                source: Box::new(std::io::Error::new(std::io::ErrorKind::TimedOut, message)),
            },
            LlmNonHttpFailureKindV2::InvalidRequest | LlmNonHttpFailureKindV2::Guardrail => {
                LlmClientError::InvalidRequest { message }
            }
            LlmNonHttpFailureKindV2::Cancelled | LlmNonHttpFailureKindV2::Internal => {
                LlmClientError::General(message)
            }
        },
    }
}

fn failure_mark_data(attempt: u32, failure: &LibsyError) -> Json {
    let mut data = Map::from_iter([
        ("attempt".into(), Json::from(attempt)),
        (
            "retryable".into(),
            Json::from(libsy_error_retryable(failure)),
        ),
    ]);
    match failure {
        LibsyError::ClientCall {
            source: LlmClientError::UpstreamHttp { status, .. },
            ..
        } => {
            data.insert("failure_kind".into(), Json::from("http"));
            data.insert("http_status".into(), Json::from(*status));
        }
        LibsyError::ClientCall { source, .. } => {
            data.insert("failure_kind".into(), Json::from("non_http"));
            data.insert(
                "non_http_kind".into(),
                Json::from(client_error_label(source)),
            );
        }
        _ => {
            data.insert("failure_kind".into(), Json::from("algorithm"));
        }
    }
    Json::Object(data)
}

fn client_error_label(error: &LlmClientError) -> &'static str {
    match error {
        LlmClientError::InvalidRequest { .. } => "invalid_request",
        LlmClientError::RequestTranslation(_) => "request_translation",
        LlmClientError::RequestEncoding(_) => "request_encoding",
        LlmClientError::ResponseTranslation(_) => "response_translation",
        LlmClientError::Configuration { .. } => "configuration",
        LlmClientError::Transport { .. } => "transport",
        LlmClientError::Timeout { .. } => "timeout",
        LlmClientError::ContextWindowExceeded { .. } => "context_window_exceeded",
        LlmClientError::UpstreamHttp { .. } => "http",
        LlmClientError::InvalidResponse { .. } => "invalid_response",
        LlmClientError::Ffi { .. } => "ffi",
        LlmClientError::General(_) => "general",
        _ => "unknown",
    }
}

fn string_headers(headers: &Map<String, Json>) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| value.as_str().map(|value| (name.clone(), value.into())))
        .collect()
}

fn identity_metadata(metadata: Option<&Metadata>) -> Json {
    json!({
        "session_id": metadata.and_then(|value| value.session_id.as_deref()),
        "agent_id": metadata.and_then(|value| value.agent_id.as_deref()),
        "turn_id": metadata.and_then(|value| value.turn_id.as_deref()),
        "request_id": metadata.and_then(|value| value.correlation_id.as_deref()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::FutureExt;

    #[test]
    fn http_failures_keep_status_semantics_without_provider_classification() {
        let error = client_error(LlmContinuationFailureV2::Http {
            status: 400,
            body: "context length exceeded".into(),
            headers: BTreeMap::new(),
        });
        assert!(matches!(
            error,
            LlmClientError::UpstreamHttp { status: 400, .. }
        ));

        let error = client_error(LlmContinuationFailureV2::NonHttp {
            kind: LlmNonHttpFailureKindV2::Guardrail,
            message: "blocked".into(),
        });
        assert!(matches!(error, LlmClientError::InvalidRequest { .. }));
    }

    #[test]
    fn switchyard_retry_policy_uses_libsy_client_error() {
        for status in [408, 425, 429, 500, 502, 503, 504] {
            let failure = LibsyError::client_call(
                "provider",
                LlmClientError::UpstreamHttp {
                    status,
                    body: String::new(),
                },
            );
            assert!(libsy_error_retryable(&failure), "status={status}");
        }
        for status in [400, 401, 404, 409, 422, 501] {
            let failure = LibsyError::client_call(
                "provider",
                LlmClientError::UpstreamHttp {
                    status,
                    body: String::new(),
                },
            );
            assert!(!libsy_error_retryable(&failure), "status={status}");
        }
        for source in [
            LlmClientError::Transport {
                source: Box::new(std::io::Error::other("transport")),
            },
            LlmClientError::Timeout {
                source: Box::new(std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout")),
            },
        ] {
            assert!(libsy_error_retryable(&LibsyError::client_call(
                "provider", source
            )));
        }
        assert!(!libsy_error_retryable(&LibsyError::client_call(
            "provider",
            LlmClientError::InvalidRequest {
                message: "invalid".into(),
            },
        )));
        assert!(!libsy_error_retryable(&LibsyError::MissingFinalResponse));
    }

    #[test]
    fn routing_failure_marks_exclude_provider_payloads() {
        let failure = LibsyError::client_call(
            "provider",
            LlmClientError::UpstreamHttp {
                status: 429,
                body: "provider body must not be recorded".into(),
            },
        );
        assert_eq!(
            failure_mark_data(2, &failure),
            json!({
                "attempt": 2,
                "retryable": true,
                "failure_kind": "http",
                "http_status": 429,
            })
        );

        let failure = LibsyError::client_call(
            "provider",
            LlmClientError::Timeout {
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "timeout detail must not be recorded",
                )),
            },
        );
        assert_eq!(
            failure_mark_data(3, &failure),
            json!({
                "attempt": 3,
                "retryable": true,
                "failure_kind": "non_http",
                "non_http_kind": "timeout",
            })
        );

        let failure = LibsyError::client_call(
            "provider",
            client_error(LlmContinuationFailureV2::NonHttp {
                kind: LlmNonHttpFailureKindV2::Guardrail,
                message: "guardrail detail must not be recorded".into(),
            }),
        );
        assert_eq!(
            failure_mark_data(4, &failure),
            json!({
                "attempt": 4,
                "retryable": false,
                "failure_kind": "non_http",
                "non_http_kind": "invalid_request",
            })
        );

        let failure = LibsyError::MissingFinalResponse;
        assert_eq!(
            failure_mark_data(5, &failure),
            json!({
                "attempt": 5,
                "retryable": false,
                "failure_kind": "algorithm",
            })
        );
    }

    #[test]
    fn returned_stream_preserves_late_provider_failure() {
        let raw = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "model": "provider/model",
            "choices": [{
                "index": 0,
                "delta": {"content": "committed"},
                "finish_reason": null
            }]
        });
        let response = Response {
            llm_response: LlmResponse::Stream(Box::pin(futures_util::stream::iter([
                Ok(LlmResponseStreamEvent::preserved(
                    WireFormat::OpenAiChat,
                    raw.clone(),
                    vec![LlmResponseChunk::TextDelta {
                        index: 0,
                        text: "committed".into(),
                    }],
                )),
                Err(LlmClientError::Timeout {
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "provider failed after its first event",
                    )),
                }),
            ]))),
            metadata: Some(Metadata {
                wire_format: Some(WireFormat::OpenAiChat),
                ..Metadata::default()
            }),
        };

        let mut stream = SwitchyardRuntime::returned_stream(
            response,
            WireFormat::OpenAiChat,
            Arc::new(TranslationEngine::default()),
        );
        let first = stream
            .next()
            .now_or_never()
            .expect("first event is ready")
            .expect("first event exists");
        assert_eq!(first.ok(), Some(raw));

        let failure = stream
            .next()
            .now_or_never()
            .expect("late failure is ready")
            .expect("late failure exists")
            .expect_err("late failure is propagated");
        assert!(libsy_error_retryable(&failure));
    }
}
