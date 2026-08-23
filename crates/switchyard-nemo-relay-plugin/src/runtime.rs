// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use futures_util::{StreamExt, stream};
use nemo_relay_plugin::{Json, LlmRequest as RelayRequest};
use serde_json::{Map, json};
use switchyard_libsy::{Algorithm, LibsyError};
use switchyard_llm_client::{ClientRouter, LlmCallObservation, RunObservation, RunObserver, run};
use switchyard_protocol::{
    LlmClientError, LlmResponse, Metadata, ModelId, Request, Response, WireFormat,
};
use switchyard_translation::{TranslationEngine, encode_stream};

use crate::config::{PreparedTargetBinding, SwitchyardConfig, protocol_from_call};
use crate::translation;

#[derive(Debug)]
pub(crate) struct RoutingMark {
    pub(crate) name: String,
    pub(crate) data: Json,
    pub(crate) metadata: Json,
}

#[derive(Debug)]
pub(crate) enum StreamMessage {
    Mark(RoutingMark),
    Event(Json),
}

pub(crate) struct SwitchyardRuntime {
    algorithm: Arc<dyn Algorithm>,
    targets: BTreeMap<String, PreparedTargetBinding>,
    default_targets: BTreeMap<WireFormat, String>,
    translation: TranslationEngine,
}

impl SwitchyardRuntime {
    pub(crate) fn new(config: SwitchyardConfig) -> Result<Self, String> {
        let prepared = config.prepare()?;
        Ok(Self {
            algorithm: prepared.algorithm,
            targets: prepared.targets,
            default_targets: prepared.default_targets,
            translation: TranslationEngine::default(),
        })
    }

    pub(crate) fn managed_protocol(&self, name: &str) -> Option<WireFormat> {
        protocol_from_call(name).filter(|protocol| self.default_targets.contains_key(protocol))
    }

    pub(crate) fn decode_request(
        &self,
        inbound: WireFormat,
        request: &RelayRequest,
        streaming: bool,
    ) -> Result<Request, String> {
        let mut llm_request = translation::decode_request(&self.translation, inbound, request)?;
        llm_request.stream = streaming;
        let headers = string_headers(&request.headers);
        let mut metadata = Metadata::from_headers(&headers);
        let relay_gateway_placeholder = !headers.contains_key("x-switchyard-session-id")
            && headers
                .get("x-nemo-relay-source")
                .and_then(|value| value.to_str().ok())
                == Some("gateway")
            && metadata.session_id.as_deref() == Some("gateway-gateway");
        if relay_gateway_placeholder {
            metadata.session_id = None;
        }
        // Keep identity/routing metadata, but target clients deliberately clear
        // these caller headers before HTTP dispatch.
        metadata.http_headers = Some(headers);
        metadata.wire_format = Some(inbound);
        Ok(Request {
            llm_request,
            raw_request: Some(request.content.clone()),
            metadata: Some(metadata),
        })
    }

    pub(crate) async fn execute_buffered(
        &self,
        inbound: WireFormat,
        request: Request,
        marks: &mut Vec<RoutingMark>,
    ) -> Result<Json, String> {
        let metadata = identity_metadata(request.metadata.as_ref());
        self.mark(
            marks,
            "switchyard.routing.requested",
            json!({"algorithm": self.algorithm.name(), "attempt": 1}),
            &metadata,
        );
        let result = self
            .drive(request.clone(), 1, marks, &metadata)
            .await
            .and_then(|response| {
                finalize_buffered_response(&self.translation, inbound, response)
                    .map_err(|source| LibsyError::client_call("return_to_agent", source))
            });
        match result {
            Ok(response) => Ok(response),
            Err(failure) => {
                self.mark(
                    marks,
                    "switchyard.routing.error",
                    failure_mark_data(1, &failure),
                    &metadata,
                );
                let response = self
                    .fallback_response(inbound, request, marks, &metadata)
                    .await?;
                finalize_buffered_response(&self.translation, inbound, response)
                    .map_err(|error| public_response_failure("trusted fallback response", &error))
            }
        }
    }

    pub(crate) async fn execute_stream(
        &self,
        inbound: WireFormat,
        request: Request,
        output: &async_channel::Sender<StreamMessage>,
    ) -> Result<(), String> {
        let metadata = identity_metadata(request.metadata.as_ref());
        let mut marks = Vec::new();
        self.mark(
            &mut marks,
            "switchyard.routing.requested",
            json!({"algorithm": self.algorithm.name(), "attempt": 1}),
            &metadata,
        );
        let (response, mut fallback_used) =
            match self.drive(request.clone(), 1, &mut marks, &metadata).await {
                Ok(response) => (response, false),
                Err(failure) => {
                    self.mark(
                        &mut marks,
                        "switchyard.routing.error",
                        failure_mark_data(1, &failure),
                        &metadata,
                    );
                    let fallback = self
                        .fallback_response(inbound, request.clone(), &mut marks, &metadata)
                        .await;
                    send_marks(output, &mut marks).await?;
                    (fallback?, true)
                }
            };
        send_marks(output, &mut marks).await?;

        let mut events = match returned_events(response, inbound).await {
            Ok(events) => events,
            Err(failure) if !fallback_used => {
                self.mark(
                    &mut marks,
                    "switchyard.routing.error",
                    failure_mark_data(1, &failure),
                    &metadata,
                );
                fallback_used = true;
                let fallback = self
                    .fallback_response(inbound, request.clone(), &mut marks, &metadata)
                    .await;
                send_marks(output, &mut marks).await?;
                let fallback = fallback?;
                returned_events(fallback, inbound)
                    .await
                    .map_err(|error| public_libsy_failure("trusted fallback stream", &error))?
            }
            Err(failure) => {
                return Err(public_libsy_failure("trusted fallback stream", &failure));
            }
        };

        let mut committed = false;
        while let Some(item) = events.next().await {
            match item {
                Ok(event) => {
                    send_event(output, event).await?;
                    committed = true;
                }
                Err(failure) if !fallback_used && !committed => {
                    self.mark(
                        &mut marks,
                        "switchyard.routing.error",
                        failure_mark_data(1, &failure),
                        &metadata,
                    );
                    let fallback = self
                        .fallback_response(inbound, request.clone(), &mut marks, &metadata)
                        .await;
                    send_marks(output, &mut marks).await?;
                    let fallback = fallback?;
                    let mut fallback = returned_events(fallback, inbound)
                        .await
                        .map_err(|error| public_libsy_failure("trusted fallback stream", &error))?;
                    while let Some(item) = fallback.next().await {
                        let event = item.map_err(|error| {
                            public_libsy_failure("trusted fallback stream", &error)
                        })?;
                        send_event(output, event).await?;
                    }
                    return Ok(());
                }
                Err(failure) if !committed => {
                    return Err(public_libsy_failure("trusted fallback stream", &failure));
                }
                Err(failure) => {
                    self.mark(
                        &mut marks,
                        "switchyard.routing.error",
                        failure_mark_data(1, &failure),
                        &metadata,
                    );
                    send_marks(output, &mut marks).await?;
                    return Err(public_libsy_failure(
                        "Switchyard stream failed after response commitment",
                        &failure,
                    ));
                }
            }
        }
        if committed {
            Ok(())
        } else {
            Err("Switchyard response stream produced no caller events".into())
        }
    }

    async fn drive(
        &self,
        request: Request,
        attempt: u32,
        marks: &mut Vec<RoutingMark>,
        mark_metadata: &Json,
    ) -> Result<Response, LibsyError> {
        let observations = Arc::new(Mutex::new(Vec::new()));
        let observed_calls = observations.clone();
        let observer: RunObserver = Arc::new(move |observation| {
            if let RunObservation::LlmCall(call) = observation {
                observed_calls
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(call);
            }
        });
        let clients = ClientRouter::new(
            self.targets
                .iter()
                .map(|(name, target)| (ModelId::from(name.as_str()), target.client.clone()))
                .collect::<HashMap<_, _>>(),
        );
        match run(self.algorithm.clone(), clients, request, Some(observer)).await {
            Ok((selected_model_id, response)) => {
                self.emit_decision(marks, &selected_model_id, attempt, mark_metadata);
                self.emit_routing_llm_calls(
                    marks,
                    take_observed_calls(&observations),
                    attempt,
                    mark_metadata,
                );
                Ok(response)
            }
            Err(error) => {
                self.emit_routing_llm_calls(
                    marks,
                    take_observed_calls(&observations),
                    attempt,
                    mark_metadata,
                );
                Err(error)
            }
        }
    }

    async fn fallback_response(
        &self,
        inbound: WireFormat,
        request: Request,
        marks: &mut Vec<RoutingMark>,
        metadata: &Json,
    ) -> Result<Response, String> {
        let target_name = self.default_target(inbound)?;
        let target = self.target(target_name)?;
        self.mark(
            marks,
            "switchyard.routing.fallback",
            json!({"selected_target": target_name}),
            metadata,
        );
        target
            .client
            .call(request)
            .await
            .map_err(|error| public_client_failure("trusted fallback", &error))
    }

    fn target(&self, name: &str) -> Result<&PreparedTargetBinding, String> {
        self.targets
            .get(name)
            .ok_or_else(|| format!("libsy selected unknown target {name:?}"))
    }

    fn default_target(&self, protocol: WireFormat) -> Result<&str, String> {
        self.default_targets
            .get(&protocol)
            .map(String::as_str)
            .ok_or_else(|| format!("managed protocol {protocol} has no default target"))
    }

    fn mark(&self, marks: &mut Vec<RoutingMark>, name: &str, data: Json, metadata: &Json) {
        marks.push(RoutingMark {
            name: name.to_string(),
            data,
            metadata: metadata.clone(),
        });
    }

    fn emit_decision(
        &self,
        marks: &mut Vec<RoutingMark>,
        selected_model_id: &ModelId,
        attempt: u32,
        metadata: &Json,
    ) {
        self.mark(
            marks,
            "switchyard.routing.decision",
            json!({
                "algorithm": self.algorithm.name(),
                "attempt": attempt,
                "selected_target": selected_model_id,
            }),
            metadata,
        );
    }

    fn emit_routing_llm_calls(
        &self,
        marks: &mut Vec<RoutingMark>,
        calls: Vec<LlmCallObservation>,
        attempt: u32,
        metadata: &Json,
    ) {
        for (index, call) in calls.into_iter().enumerate() {
            self.mark(
                marks,
                "switchyard.routing.llm_call",
                json!({
                    "algorithm": self.algorithm.name(),
                    "attempt": attempt,
                    "call_index": index + 1,
                    "selected_target": call.selected_model,
                    "call_role": "routing",
                    "outcome": if call.is_success { "ok" } else { "error" },
                    "latency_ms": call.duration.as_secs_f64() * 1_000.0,
                    "usage": call.usage,
                    "contributes_to_routing_overhead": true,
                }),
                metadata,
            );
        }
    }
}

fn take_observed_calls(observations: &Mutex<Vec<LlmCallObservation>>) -> Vec<LlmCallObservation> {
    std::mem::take(
        &mut *observations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    )
}

async fn send_marks(
    output: &async_channel::Sender<StreamMessage>,
    marks: &mut Vec<RoutingMark>,
) -> Result<(), String> {
    for mark in marks.drain(..) {
        output
            .send(StreamMessage::Mark(mark))
            .await
            .map_err(|_| "Relay cancelled the Switchyard response stream".to_string())?;
    }
    Ok(())
}

async fn send_event(
    output: &async_channel::Sender<StreamMessage>,
    event: Json,
) -> Result<(), String> {
    output
        .send(StreamMessage::Event(event))
        .await
        .map_err(|_| "Relay cancelled the Switchyard response stream".to_string())
}

type ReturnedEventStream =
    std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<Json, LibsyError>> + Send>>;

fn finalize_buffered_response(
    translation_engine: &TranslationEngine,
    inbound: WireFormat,
    response: Response,
) -> Result<Json, LlmClientError> {
    let LlmResponse::Agg(response) = response.llm_response else {
        return Err(LlmClientError::InvalidResponse {
            source: Box::new(std::io::Error::other(
                "libsy returned a stream for a buffered request",
            )),
        });
    };
    translation::encode_response(translation_engine, inbound, &response)
        .map_err(LlmClientError::ResponseTranslation)
}

async fn returned_events(
    response: Response,
    inbound: WireFormat,
) -> Result<ReturnedEventStream, LibsyError> {
    let chunks = match response.llm_response {
        LlmResponse::Agg(response) => response.into_stream(),
        LlmResponse::Stream(mut chunks) => {
            let Some(first) = chunks.next().await else {
                return Err(LibsyError::client_call(
                    "return_to_agent",
                    LlmClientError::InvalidResponse {
                        source: Box::new(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "provider returned an empty stream",
                        )),
                    },
                ));
            };
            Box::pin(stream::once(async move { first }).chain(chunks))
        }
    };
    let events = encode_stream(chunks, inbound, None)
        .map_err(|error| LibsyError::client_call("return_to_agent", error))?;
    Ok(Box::pin(events.map(|item| {
        item.map_err(|source| match source.downcast::<LlmClientError>() {
            Ok(source) => LibsyError::client_call("return_to_agent", *source),
            Err(source) => LibsyError::client_call(
                "return_to_agent",
                LlmClientError::ResponseTranslation(source.to_string()),
            ),
        })
    })))
}

fn failure_mark_data(attempt: u32, failure: &LibsyError) -> Json {
    let mut data = Map::from_iter([("attempt".into(), Json::from(attempt))]);
    match failure {
        LibsyError::ClientCall {
            source: LlmClientError::UpstreamHttp { status, .. },
            ..
        } => {
            data.insert("failure_kind".into(), Json::from("http"));
            data.insert("http_status".into(), Json::from(status.as_u16()));
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

fn public_libsy_failure(prefix: &str, error: &LibsyError) -> String {
    match error {
        LibsyError::ClientCall { source, .. } => public_client_failure(prefix, source),
        _ => format!("{prefix}: Switchyard algorithm failure"),
    }
}

fn public_response_failure(prefix: &str, error: &LlmClientError) -> String {
    match error {
        LlmClientError::InvalidResponse { .. } => format!("{prefix}: invalid response"),
        LlmClientError::ResponseTranslation(_) => {
            format!("{prefix}: response translation failure")
        }
        _ => format!("{prefix}: response finalization failure"),
    }
}

fn public_client_failure(prefix: &str, error: &LlmClientError) -> String {
    match error {
        LlmClientError::UpstreamHttp { status, .. } => {
            format!("{prefix}: provider returned HTTP {status}")
        }
        _ => format!("{prefix}: provider {} failure", client_error_label(error)),
    }
}

fn string_headers(headers: &Map<String, Json>) -> http::HeaderMap {
    let mut parsed = http::HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        let Some(value) = value.as_str() else {
            continue;
        };
        let (Ok(name), Ok(value)) = (
            http::HeaderName::from_bytes(name.as_bytes()),
            http::HeaderValue::from_str(value),
        ) else {
            continue;
        };
        parsed.insert(name, value);
    }
    parsed
}

fn identity_metadata(metadata: Option<&Metadata>) -> Json {
    json!({
        "session_id": metadata.and_then(|value| value.session_id.as_deref()),
        "agent_id": metadata.and_then(|value| value.agent_id.as_deref()),
        "parent_agent_id": metadata.and_then(|value| value.parent_agent_id.as_deref()),
        "task_id": metadata.and_then(|value| value.task_id.as_deref()),
        "turn_id": metadata.and_then(|value| value.turn_id.as_deref()),
        "correlation_id": metadata.and_then(|value| value.correlation_id.as_deref()),
    })
}

#[cfg(test)]
mod tests;
