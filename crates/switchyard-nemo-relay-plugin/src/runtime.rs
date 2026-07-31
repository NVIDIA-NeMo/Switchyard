// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

use futures::Stream;
use futures_util::StreamExt;
use nemo_relay_plugin::{
    Json, LlmContinuationFailureV2, LlmContinuationInvocationV2, LlmContinuationTargetV2,
    LlmNonHttpFailureKindV2, LlmRequest as RelayRequest, NemoRelayNativeAsyncNext,
    NemoRelayNativeAsyncStream, NemoRelayNativeHostApiV4, PluginRuntime,
};
use serde::Deserialize;
use serde_json::{json, Map};
use switchyard_libsy::{
    Algorithm, CallLlmRequest, Context, LibsyError, LlmResponse, Request, Response, Step,
};
use switchyard_protocol::{
    LlmClientError, LlmResponseChunk, LlmResponseStream, LlmResponseStreamEvent, Metadata,
};
use switchyard_translation::{StreamTranslationState, TranslationEngine};

use crate::config::{SwitchyardConfig, TargetBinding, WireProtocol};
use crate::{ffi, translation};

#[derive(Deserialize)]
pub struct Invocation {
    pub name: String,
    pub request: RelayRequest,
}

pub struct SwitchyardRuntime {
    config: SwitchyardConfig,
    algorithm: Arc<dyn Algorithm>,
    target_headers: BTreeMap<String, BTreeMap<String, String>>,
    translation: TranslationEngine,
    relay: PluginRuntime,
}

impl SwitchyardRuntime {
    pub fn new(config: SwitchyardConfig, relay: PluginRuntime) -> Result<Self, String> {
        config.validate()?;
        let algorithm = config.build_algorithm()?;
        let target_headers = config
            .targets
            .iter()
            .map(|(name, target)| {
                target
                    .resolved_headers()
                    .map(|headers| (name.clone(), headers))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            config,
            algorithm,
            target_headers,
            translation: TranslationEngine::default(),
            relay,
        })
    }

    pub async fn execute_buffered(
        &self,
        invocation: Invocation,
        host: &NemoRelayNativeHostApiV4,
        next: *const NemoRelayNativeAsyncNext,
    ) -> Result<Json, String> {
        let Some(inbound) = WireProtocol::from_call(&invocation.name) else {
            return ffi::dispatch_passthrough_buffered(host, next, &invocation.request).await;
        };
        if !self.config.enabled_inbound_profiles.contains(&inbound) {
            return ffi::dispatch_passthrough_buffered(host, next, &invocation.request).await;
        }
        let request = self.libsy_request(inbound, &invocation.request, false)?;
        let max_attempts = self.config.max_retries.saturating_add(1);
        for attempt in 1..=max_attempts {
            self.mark(
                "switchyard.routing.requested",
                json!({"algorithm": self.algorithm.name(), "attempt": attempt}),
                identity_metadata(&invocation.request),
            );
            match self
                .drive_buffered(
                    request.clone(),
                    host,
                    next,
                    attempt,
                    identity_metadata(&invocation.request),
                )
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
                Err(failure) if failure.retryable() && attempt < max_attempts => {
                    self.mark(
                        "switchyard.routing.retry",
                        failure_mark_data(attempt, &failure),
                        identity_metadata(&invocation.request),
                    );
                }
                Err(failure) => {
                    self.mark(
                        "switchyard.routing.error",
                        failure_mark_data(attempt, &failure),
                        identity_metadata(&invocation.request),
                    );
                    return self
                        .fallback_buffered(
                            inbound,
                            request,
                            host,
                            next,
                            identity_metadata(&invocation.request),
                        )
                        .await;
                }
            }
        }
        Err("Switchyard retry loop ended without a result".into())
    }

    async fn drive_buffered(
        &self,
        request: Request,
        host: &NemoRelayNativeHostApiV4,
        next: *const NemoRelayNativeAsyncNext,
        attempt: u32,
        mark_metadata: Json,
    ) -> Result<Response, RunFailure> {
        let mut context = Context::default();
        context
            .values
            .insert("relay.routing_attempt".into(), attempt.to_string());
        let mut steps = self.algorithm.clone().run_stream(context, request);
        let provider_error = Arc::new(Mutex::new(None));
        while let Some(step) = steps.next().await {
            match step {
                Ok(Step::Decision(decision)) => {
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
                        mark_metadata.clone(),
                    );
                }
                Ok(Step::CallLlm(call)) => {
                    self.serve_buffered_call(*call, host, next, Arc::clone(&provider_error))
                        .await
                        .map_err(|error| RunFailure::new(error, &provider_error))?;
                }
                Ok(Step::ReturnToAgent(response)) => return Ok(*response),
                Err(error) => return Err(RunFailure::new(error, &provider_error)),
            }
        }
        Err(RunFailure::new(
            LibsyError::MissingFinalResponse,
            &provider_error,
        ))
    }

    async fn serve_buffered_call(
        &self,
        call: CallLlmRequest,
        host: &NemoRelayNativeHostApiV4,
        next: *const NemoRelayNativeAsyncNext,
        provider_error: Arc<Mutex<Option<LlmContinuationFailureV2>>>,
    ) -> switchyard_libsy::Result<()> {
        let routed = call.get_routed().clone();
        let target_name = routed.decision.selected_model().to_string();
        let result = async {
            let target = self.target(&target_name)?;
            let request = self.dispatch_request(&target_name, target, routed.request, false)?;
            match ffi::dispatch_buffered(host, next, &request).await {
                Ok(response) => {
                    let response =
                        translation::decode_response(&self.translation, target.protocol, &response)
                            .map_err(LlmClientError::ResponseTranslation)?;
                    Ok(Response {
                        llm_response: LlmResponse::Agg(response),
                        metadata: Some(Metadata {
                            wire_format: Some(target.protocol.wire_format()),
                            ..Metadata::default()
                        }),
                    })
                }
                Err(error) => {
                    if let Ok(mut stored) = provider_error.lock() {
                        *stored = Some(error.clone());
                    }
                    Err(client_error(error))
                }
            }
        }
        .await
        .map_err(|source| LibsyError::client_call(target_name, source));
        call.respond(result)
    }

    async fn fallback_buffered(
        &self,
        inbound: WireProtocol,
        request: Request,
        host: &NemoRelayNativeHostApiV4,
        next: *const NemoRelayNativeAsyncNext,
        metadata: Json,
    ) -> Result<Json, String> {
        let target_name = self.config.default_targets.target(inbound);
        let target = self
            .target(target_name)
            .map_err(|error| error.to_string())?;
        self.mark(
            "switchyard.routing.fallback",
            json!({"selected_target": target_name}),
            metadata,
        );
        let dispatch = self
            .dispatch_request(target_name, target, request, false)
            .map_err(|error| error.to_string())?;
        let response = ffi::dispatch_buffered(host, next, &dispatch)
            .await
            .map_err(|error| format!("trusted fallback failed: {error:?}"))?;
        let response = translation::decode_response(&self.translation, target.protocol, &response)?;
        translation::encode_response(&self.translation, inbound, &response)
    }

    fn libsy_request(
        &self,
        inbound: WireProtocol,
        original: &RelayRequest,
        streaming: bool,
    ) -> Result<Request, String> {
        let mut request = translation::decode_request(&self.translation, inbound, original)?;
        request.stream = streaming;
        let headers = string_headers(&original.headers);
        let mut metadata = Metadata::from_headers(&headers);
        metadata.wire_format = Some(inbound.wire_format());
        Ok(Request {
            llm_request: request,
            raw_request: Some(original.content.clone()),
            metadata: Some(metadata),
        })
    }

    fn dispatch_request(
        &self,
        target_name: &str,
        target: &TargetBinding,
        mut request: Request,
        streaming: bool,
    ) -> Result<LlmContinuationInvocationV2, LlmClientError> {
        request.llm_request.stream = streaming;
        let headers = self
            .target_headers
            .get(target_name)
            .cloned()
            .unwrap_or_default();
        let mut request =
            translation::encode_request(&self.translation, target.protocol, &request.llm_request)
                .map_err(LlmClientError::RequestEncoding)?;
        let body = request.content.as_object_mut().ok_or_else(|| {
            LlmClientError::RequestEncoding("translated provider request is not an object".into())
        })?;
        body.insert("model".into(), Json::String(target.model.clone()));
        body.insert("stream".into(), Json::Bool(streaming));
        Ok(LlmContinuationInvocationV2 {
            request,
            target: LlmContinuationTargetV2 {
                method: "POST".into(),
                url: target.dispatch_url(),
                route: target.protocol.relay_route(),
                headers,
            },
        })
    }

    fn target(&self, name: &str) -> Result<&TargetBinding, LlmClientError> {
        self.config
            .targets
            .get(name)
            .ok_or_else(|| LlmClientError::Configuration {
                message: format!("libsy selected unknown target {name:?}"),
            })
    }

    fn mark(&self, name: &str, data: Json, metadata: Json) {
        if let Err(error) = self.relay.emit_mark(name, Some(&data), Some(&metadata)) {
            eprintln!("Switchyard could not emit routing mark {name:?}: {error}");
        }
    }

    pub async fn execute_stream(
        &self,
        invocation: Invocation,
        host: &NemoRelayNativeHostApiV4,
        next: *const NemoRelayNativeAsyncNext,
        output: *const NemoRelayNativeAsyncStream,
    ) -> Result<(), String> {
        let Some(inbound) = WireProtocol::from_call(&invocation.name) else {
            return ffi::dispatch_passthrough_stream(host, next, output, &invocation.request).await;
        };
        if !self.config.enabled_inbound_profiles.contains(&inbound) {
            return ffi::dispatch_passthrough_stream(host, next, output, &invocation.request).await;
        }
        let request = self.libsy_request(inbound, &invocation.request, true)?;
        let metadata = identity_metadata(&invocation.request);
        let max_attempts = self.config.max_retries.saturating_add(1);
        for attempt in 1..=max_attempts {
            self.mark(
                "switchyard.routing.requested",
                json!({"algorithm": self.algorithm.name(), "attempt": attempt}),
                metadata.clone(),
            );
            let run = match self
                .drive_stream(
                    request.clone(),
                    host,
                    next,
                    output,
                    attempt,
                    metadata.clone(),
                )
                .await
            {
                Ok(response) => response,
                Err(failure) if failure.retryable() && attempt < max_attempts => {
                    self.emit_stream_retry(attempt, &failure, &metadata);
                    continue;
                }
                Err(failure) => {
                    self.emit_stream_error(attempt, &failure, &metadata);
                    return self
                        .fallback_stream(inbound, request, host, next, output, metadata.clone())
                        .await;
                }
            };
            match self
                .emit_returned_stream(run.response, inbound, host, output, &run.provider_error)
                .await
            {
                Ok(()) => {
                    let status = ffi::finish_stream(host, output);
                    return if status == nemo_relay_plugin::NemoRelayStatus::Ok {
                        Ok(())
                    } else {
                        Err(format!("Relay rejected output stream finish: {status:?}"))
                    };
                }
                Err(failure)
                    if !failure.committed
                        && failure.failure.retryable()
                        && attempt < max_attempts =>
                {
                    self.emit_stream_retry(attempt, &failure.failure, &metadata);
                }
                Err(failure) if !failure.committed => {
                    self.emit_stream_error(attempt, &failure.failure, &metadata);
                    return self
                        .fallback_stream(inbound, request, host, next, output, metadata.clone())
                        .await;
                }
                Err(failure) => {
                    self.emit_stream_error(attempt, &failure.failure, &metadata);
                    return Err(format!(
                        "Switchyard stream failed after response commitment: {}",
                        failure.failure.error
                    ));
                }
            }
        }
        Err("Switchyard stream retry loop ended without a result".into())
    }

    async fn drive_stream(
        &self,
        request: Request,
        host: &NemoRelayNativeHostApiV4,
        next: *const NemoRelayNativeAsyncNext,
        output: *const NemoRelayNativeAsyncStream,
        attempt: u32,
        mark_metadata: Json,
    ) -> Result<StreamRun, RunFailure> {
        let mut context = Context::default();
        context
            .values
            .insert("relay.routing_attempt".into(), attempt.to_string());
        let mut steps = self.algorithm.clone().run_stream(context, request);
        let provider_error = Arc::new(Mutex::new(None));
        while let Some(step) = steps.next().await {
            match step {
                Ok(Step::Decision(decision)) => {
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
                        mark_metadata.clone(),
                    );
                }
                Ok(Step::CallLlm(call)) => {
                    self.serve_stream_call(*call, host, next, output, Arc::clone(&provider_error))
                        .await
                        .map_err(|error| RunFailure::new(error, &provider_error))?;
                }
                Ok(Step::ReturnToAgent(response)) => {
                    return Ok(StreamRun {
                        response: *response,
                        provider_error,
                    });
                }
                Err(error) => return Err(RunFailure::new(error, &provider_error)),
            }
        }
        Err(RunFailure::new(
            LibsyError::MissingFinalResponse,
            &provider_error,
        ))
    }

    async fn serve_stream_call(
        &self,
        call: CallLlmRequest,
        host: &NemoRelayNativeHostApiV4,
        next: *const NemoRelayNativeAsyncNext,
        output: *const NemoRelayNativeAsyncStream,
        provider_error: Arc<Mutex<Option<LlmContinuationFailureV2>>>,
    ) -> switchyard_libsy::Result<()> {
        let routed = call.get_routed().clone();
        let target_name = routed.decision.selected_model().to_string();
        let result = self
            .provider_stream_response(
                &target_name,
                routed.request,
                host,
                next,
                output,
                Arc::clone(&provider_error),
            )
            .await
            .map_err(|source| LibsyError::client_call(target_name, source));
        call.respond(result)
    }

    async fn provider_stream_response(
        &self,
        target_name: &str,
        request: Request,
        host: &NemoRelayNativeHostApiV4,
        next: *const NemoRelayNativeAsyncNext,
        output: *const NemoRelayNativeAsyncStream,
        provider_error: Arc<Mutex<Option<LlmContinuationFailureV2>>>,
    ) -> Result<Response, LlmClientError> {
        let target = self.target(target_name)?;
        let metadata = request.metadata.clone();
        let dispatch = self.dispatch_request(target_name, target, request, true)?;
        let mut upstream = match ffi::dispatch_stream(host, next, output, &dispatch).await {
            Ok(upstream) => upstream,
            Err(error) => {
                remember_provider_error(&provider_error, &error);
                return Err(client_error(error));
            }
        };
        let first_raw = match upstream.next().await {
            Some(Ok(first)) => first,
            Some(Err(error)) => {
                remember_provider_error(&provider_error, &error);
                return Err(client_error(error));
            }
            None => {
                return Err(LlmClientError::InvalidResponse {
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "provider returned an empty stream",
                    )),
                });
            }
        };
        let mut state = StreamTranslationState::new(
            target.protocol.wire_format(),
            target.protocol.wire_format(),
        );
        let first =
            decode_provider_event(&self.translation, &mut state, target.protocol, first_raw)?;
        let stream: LlmResponseStream = Box::pin(TranslatedProviderStream {
            upstream,
            first: Some(first),
            protocol: target.protocol,
            state,
            provider_error,
        });
        Ok(Response {
            llm_response: LlmResponse::Stream(stream),
            metadata: Some(Metadata {
                wire_format: Some(target.protocol.wire_format()),
                ..metadata.unwrap_or_default()
            }),
        })
    }

    async fn emit_returned_stream(
        &self,
        response: Response,
        inbound: WireProtocol,
        host: &NemoRelayNativeHostApiV4,
        output: *const NemoRelayNativeAsyncStream,
        provider_error: &Arc<Mutex<Option<LlmContinuationFailureV2>>>,
    ) -> Result<(), StreamAttemptFailure> {
        let source = response
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.wire_format.as_ref())
            .and_then(WireProtocol::from_wire_format)
            .ok_or_else(|| {
                StreamAttemptFailure::translation(
                    "libsy returned a stream without a supported source wire format",
                    false,
                    provider_error,
                )
            })?;
        let LlmResponse::Stream(mut stream) = response.llm_response else {
            return Err(StreamAttemptFailure::translation(
                "libsy returned a buffered response for a streaming request",
                false,
                provider_error,
            ));
        };
        let mut state = StreamTranslationState::new(source.wire_format(), inbound.wire_format());
        let mut committed = false;
        while let Some(item) = stream.next().await {
            let event = item
                .map_err(|error| StreamAttemptFailure::client(error, committed, provider_error))?;
            let events =
                translation::encode_stream_event(&self.translation, &mut state, inbound, event)
                    .map_err(|error| {
                        StreamAttemptFailure::translation(&error, committed, provider_error)
                    })?;
            for event in events {
                ffi::push_stream(host, output, &event).map_err(|error| {
                    StreamAttemptFailure::translation(&error, committed, provider_error)
                })?;
                committed = true;
            }
        }
        if source != inbound {
            let events = translation::finish_stream(&self.translation, &mut state, inbound)
                .map_err(|error| {
                    StreamAttemptFailure::translation(&error, committed, provider_error)
                })?;
            for event in events {
                ffi::push_stream(host, output, &event).map_err(|error| {
                    StreamAttemptFailure::translation(&error, committed, provider_error)
                })?;
                committed = true;
            }
        }
        if !committed {
            return Err(StreamAttemptFailure::translation(
                "Switchyard produced an empty output stream",
                false,
                provider_error,
            ));
        }
        Ok(())
    }

    async fn fallback_stream(
        &self,
        inbound: WireProtocol,
        request: Request,
        host: &NemoRelayNativeHostApiV4,
        next: *const NemoRelayNativeAsyncNext,
        output: *const NemoRelayNativeAsyncStream,
        metadata: Json,
    ) -> Result<(), String> {
        let target_name = self.config.default_targets.target(inbound).to_string();
        self.mark(
            "switchyard.routing.fallback",
            json!({"selected_target": target_name}),
            metadata,
        );
        let provider_error = Arc::new(Mutex::new(None));
        let response = self
            .provider_stream_response(
                &target_name,
                request,
                host,
                next,
                output,
                Arc::clone(&provider_error),
            )
            .await
            .map_err(|error| format!("trusted fallback stream failed: {error}"))?;
        self.emit_returned_stream(response, inbound, host, output, &provider_error)
            .await
            .map_err(|failure| {
                format!("trusted fallback stream failed: {}", failure.failure.error)
            })?;
        let status = ffi::finish_stream(host, output);
        if status == nemo_relay_plugin::NemoRelayStatus::Ok {
            Ok(())
        } else {
            Err(format!("Relay rejected fallback stream finish: {status:?}"))
        }
    }

    fn emit_stream_retry(&self, attempt: u32, failure: &RunFailure, metadata: &Json) {
        self.mark(
            "switchyard.routing.retry",
            failure_mark_data(attempt, failure),
            metadata.clone(),
        );
    }

    fn emit_stream_error(&self, attempt: u32, failure: &RunFailure, metadata: &Json) {
        self.mark(
            "switchyard.routing.error",
            failure_mark_data(attempt, failure),
            metadata.clone(),
        );
    }
}

struct StreamRun {
    response: Response,
    provider_error: Arc<Mutex<Option<LlmContinuationFailureV2>>>,
}

struct TranslatedProviderStream {
    upstream: ffi::ProviderJsonStream,
    first: Option<LlmResponseStreamEvent>,
    protocol: WireProtocol,
    state: StreamTranslationState,
    provider_error: Arc<Mutex<Option<LlmContinuationFailureV2>>>,
}

impl Stream for TranslatedProviderStream {
    type Item = Result<LlmResponseStreamEvent, LlmClientError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        if let Some(first) = self.first.take() {
            return Poll::Ready(Some(Ok(first)));
        }
        match Pin::new(&mut self.upstream).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(raw))) => {
                let translation = TranslationEngine::default();
                let protocol = self.protocol;
                Poll::Ready(Some(decode_provider_event(
                    &translation,
                    &mut self.state,
                    protocol,
                    raw,
                )))
            }
            Poll::Ready(Some(Err(error))) => {
                remember_provider_error(&self.provider_error, &error);
                Poll::Ready(Some(Err(client_error(error))))
            }
            Poll::Ready(None) => Poll::Ready(None),
        }
    }
}

fn decode_provider_event(
    translation: &TranslationEngine,
    state: &mut StreamTranslationState,
    protocol: WireProtocol,
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

fn remember_provider_error(
    provider_error: &Arc<Mutex<Option<LlmContinuationFailureV2>>>,
    error: &LlmContinuationFailureV2,
) {
    if let Ok(mut stored) = provider_error.lock() {
        *stored = Some(error.clone());
    }
}

struct StreamAttemptFailure {
    failure: RunFailure,
    committed: bool,
}

impl StreamAttemptFailure {
    fn client(
        error: LlmClientError,
        committed: bool,
        provider_error: &Arc<Mutex<Option<LlmContinuationFailureV2>>>,
    ) -> Self {
        Self {
            failure: RunFailure::new(
                LibsyError::client_call("return_to_agent", error),
                provider_error,
            ),
            committed,
        }
    }

    fn translation(
        error: &str,
        committed: bool,
        provider_error: &Arc<Mutex<Option<LlmContinuationFailureV2>>>,
    ) -> Self {
        Self::client(
            LlmClientError::ResponseTranslation(error.to_string()),
            committed,
            provider_error,
        )
    }
}

struct RunFailure {
    error: LibsyError,
    provider_error: Option<LlmContinuationFailureV2>,
}

impl RunFailure {
    fn new(
        error: LibsyError,
        provider_error: &Arc<Mutex<Option<LlmContinuationFailureV2>>>,
    ) -> Self {
        Self {
            error,
            provider_error: provider_error.lock().ok().and_then(|error| error.clone()),
        }
    }

    fn retryable(&self) -> bool {
        self.provider_error
            .as_ref()
            .is_some_and(LlmContinuationFailureV2::is_retryable)
    }
}

fn client_error(error: LlmContinuationFailureV2) -> LlmClientError {
    match error {
        LlmContinuationFailureV2::Http { failure } => LlmClientError::UpstreamHttp {
            status: failure.status,
            body: failure.body,
        },
        LlmContinuationFailureV2::NonHttp { failure } => match failure.kind {
            LlmNonHttpFailureKindV2::Transport => LlmClientError::Transport {
                source: Box::new(std::io::Error::other(failure.message)),
            },
            LlmNonHttpFailureKindV2::Timeout => LlmClientError::Timeout {
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    failure.message,
                )),
            },
            LlmNonHttpFailureKindV2::InvalidRequest | LlmNonHttpFailureKindV2::Guardrail => {
                LlmClientError::InvalidRequest {
                    message: failure.message,
                }
            }
            LlmNonHttpFailureKindV2::Cancelled | LlmNonHttpFailureKindV2::Internal => {
                LlmClientError::General(failure.message)
            }
        },
    }
}

fn failure_mark_data(attempt: u32, failure: &RunFailure) -> Json {
    let mut data = Map::from_iter([
        ("attempt".into(), Json::from(attempt)),
        ("retryable".into(), Json::from(failure.retryable())),
    ]);
    match &failure.provider_error {
        Some(LlmContinuationFailureV2::Http { failure }) => {
            data.insert("failure_kind".into(), Json::from("http"));
            data.insert("http_status".into(), Json::from(failure.status));
        }
        Some(LlmContinuationFailureV2::NonHttp { failure }) => {
            data.insert("failure_kind".into(), Json::from("non_http"));
            data.insert(
                "non_http_kind".into(),
                Json::from(non_http_failure_label(failure.kind)),
            );
        }
        None => {
            data.insert("failure_kind".into(), Json::from("algorithm"));
        }
    }
    Json::Object(data)
}

const fn non_http_failure_label(kind: LlmNonHttpFailureKindV2) -> &'static str {
    match kind {
        LlmNonHttpFailureKindV2::Transport => "transport",
        LlmNonHttpFailureKindV2::Timeout => "timeout",
        LlmNonHttpFailureKindV2::Cancelled => "cancelled",
        LlmNonHttpFailureKindV2::InvalidRequest => "invalid_request",
        LlmNonHttpFailureKindV2::Guardrail => "guardrail",
        LlmNonHttpFailureKindV2::Internal => "internal",
    }
}

fn string_headers(headers: &Map<String, Json>) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| value.as_str().map(|value| (name.clone(), value.into())))
        .collect()
}

fn identity_metadata(request: &RelayRequest) -> Json {
    let metadata = Metadata::from_headers(&string_headers(&request.headers));
    json!({
        "session_id": metadata.session_id,
        "agent_id": metadata.agent_id,
        "turn_id": metadata.turn_id,
        "request_id": metadata.correlation_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nemo_relay_plugin::{LlmHttpFailureV2, LlmNonHttpFailureV2};

    #[test]
    fn http_failures_keep_status_semantics_without_provider_classification() {
        let error = client_error(LlmContinuationFailureV2::Http {
            failure: LlmHttpFailureV2 {
                status: 400,
                body: "context length exceeded".into(),
                headers: BTreeMap::new(),
            },
        });
        assert!(matches!(
            error,
            LlmClientError::UpstreamHttp { status: 400, .. }
        ));
    }

    #[test]
    fn routing_failure_marks_exclude_provider_payloads() {
        let failure = RunFailure {
            error: LibsyError::MissingFinalResponse,
            provider_error: Some(LlmContinuationFailureV2::Http {
                failure: LlmHttpFailureV2 {
                    status: 429,
                    body: "provider body must not be recorded".into(),
                    headers: BTreeMap::from([("retry-after".into(), "secret".into())]),
                },
            }),
        };
        assert_eq!(
            failure_mark_data(2, &failure),
            json!({
                "attempt": 2,
                "retryable": true,
                "failure_kind": "http",
                "http_status": 429,
            })
        );

        let failure = RunFailure {
            error: LibsyError::MissingFinalResponse,
            provider_error: Some(LlmContinuationFailureV2::NonHttp {
                failure: LlmNonHttpFailureV2 {
                    kind: LlmNonHttpFailureKindV2::Timeout,
                    message: "timeout detail must not be recorded".into(),
                },
            }),
        };
        assert_eq!(
            failure_mark_data(3, &failure),
            json!({
                "attempt": 3,
                "retryable": true,
                "failure_kind": "non_http",
                "non_http_kind": "timeout",
            })
        );
    }
}
