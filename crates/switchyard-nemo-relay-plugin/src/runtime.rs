// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{StreamExt, stream};
use nemo_relay_plugin::{Json, LlmRequest as RelayRequest};
use serde_json::{Map, json};
use switchyard_libsy::{Algorithm, LibsyError, PickOutcome, ToolSignals, pick_tier};
use switchyard_llm_client::{ClientRouter, LlmCallObservation, RunObservation, RunObserver, run};
use switchyard_protocol::{
    Context, Decision, LlmClientError, LlmResponse, Metadata, Request, Response, WireFormat,
};
use switchyard_translation::{TranslationEngine, encode_stream};

use crate::config::{PreparedTargetBinding, StageMarkConfig, SwitchyardConfig, protocol_from_call};
use crate::translation;

const INITIAL_RETRY_BACKOFF: Duration = Duration::from_millis(250);
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(2);

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
    max_retries: u32,
    algorithm: Arc<dyn Algorithm>,
    targets: BTreeMap<String, PreparedTargetBinding>,
    default_targets: BTreeMap<WireFormat, String>,
    target_tiers: BTreeMap<String, &'static str>,
    stage_marks: Option<StageMarkConfig>,
    translation: TranslationEngine,
}

impl SwitchyardRuntime {
    pub(crate) fn new(config: SwitchyardConfig) -> Result<Self, String> {
        let prepared = config.prepare()?;
        Ok(Self {
            max_retries: prepared.max_retries,
            algorithm: prepared.algorithm,
            targets: prepared.targets,
            default_targets: prepared.default_targets,
            target_tiers: prepared.target_tiers,
            stage_marks: prepared.stage_marks,
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
        let max_attempts = self.max_retries + 1;
        let mut attempt = 1;
        loop {
            self.mark(
                marks,
                "switchyard.routing.requested",
                json!({"algorithm": self.algorithm.name(), "attempt": attempt}),
                &metadata,
            );
            let result = self
                .drive(request.clone(), attempt, marks, &metadata)
                .await
                .and_then(|response| {
                    finalize_buffered_response(&self.translation, inbound, response)
                        .map_err(|source| LibsyError::client_call("return_to_agent", source))
                });
            match result {
                Ok(response) => return Ok(response),
                Err(failure) if libsy_error_retryable(&failure) && attempt < max_attempts => {
                    self.mark(
                        marks,
                        "switchyard.routing.retry",
                        failure_mark_data(attempt, &failure),
                        &metadata,
                    );
                    sleep_before_retry(attempt).await;
                    attempt += 1;
                }
                Err(failure) => {
                    self.mark(
                        marks,
                        "switchyard.routing.error",
                        failure_mark_data(attempt, &failure),
                        &metadata,
                    );
                    let response = self
                        .fallback_response(inbound, request, marks, &metadata)
                        .await?;
                    return finalize_buffered_response(&self.translation, inbound, response)
                        .map_err(|error| {
                            public_response_failure("trusted fallback response", &error)
                        });
                }
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
        let max_attempts = self.max_retries + 1;
        let mut attempt = 1;
        let mut marks = Vec::new();
        'attempts: loop {
            self.mark(
                &mut marks,
                "switchyard.routing.requested",
                json!({"algorithm": self.algorithm.name(), "attempt": attempt}),
                &metadata,
            );
            let (response, mut fallback_used) = match self
                .drive(request.clone(), attempt, &mut marks, &metadata)
                .await
            {
                Ok(response) => (response, false),
                Err(failure) if libsy_error_retryable(&failure) && attempt < max_attempts => {
                    self.mark(
                        &mut marks,
                        "switchyard.routing.retry",
                        failure_mark_data(attempt, &failure),
                        &metadata,
                    );
                    send_marks(output, &mut marks).await?;
                    sleep_before_retry(attempt).await;
                    attempt += 1;
                    continue;
                }
                Err(failure) => {
                    self.mark(
                        &mut marks,
                        "switchyard.routing.error",
                        failure_mark_data(attempt, &failure),
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
                Err(failure)
                    if !fallback_used
                        && libsy_error_retryable(&failure)
                        && attempt < max_attempts =>
                {
                    self.mark(
                        &mut marks,
                        "switchyard.routing.retry",
                        failure_mark_data(attempt, &failure),
                        &metadata,
                    );
                    send_marks(output, &mut marks).await?;
                    sleep_before_retry(attempt).await;
                    attempt += 1;
                    continue;
                }
                Err(failure) if !fallback_used => {
                    self.mark(
                        &mut marks,
                        "switchyard.routing.error",
                        failure_mark_data(attempt, &failure),
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
                    Err(failure)
                        if !fallback_used
                            && !committed
                            && libsy_error_retryable(&failure)
                            && attempt < max_attempts =>
                    {
                        self.mark(
                            &mut marks,
                            "switchyard.routing.retry",
                            failure_mark_data(attempt, &failure),
                            &metadata,
                        );
                        send_marks(output, &mut marks).await?;
                        sleep_before_retry(attempt).await;
                        attempt += 1;
                        continue 'attempts;
                    }
                    Err(failure) if !fallback_used && !committed => {
                        self.mark(
                            &mut marks,
                            "switchyard.routing.error",
                            failure_mark_data(attempt, &failure),
                            &metadata,
                        );
                        let fallback = self
                            .fallback_response(inbound, request.clone(), &mut marks, &metadata)
                            .await;
                        send_marks(output, &mut marks).await?;
                        let fallback = fallback?;
                        let mut fallback =
                            returned_events(fallback, inbound).await.map_err(|error| {
                                public_libsy_failure("trusted fallback stream", &error)
                            })?;
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
                            failure_mark_data(attempt, &failure),
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
                return Ok(());
            }
            return Err("Switchyard response stream produced no caller events".into());
        }
    }

    async fn drive(
        &self,
        request: Request,
        attempt: u32,
        marks: &mut Vec<RoutingMark>,
        mark_metadata: &Json,
    ) -> Result<Response, LibsyError> {
        let context = context_from_metadata(request.metadata.as_ref());
        let stage_request = self.stage_marks.as_ref().map(|_| request.clone());
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
                .map(|(name, target)| (name.clone(), target.client.clone()))
                .collect::<HashMap<_, _>>(),
        );
        match run(
            self.algorithm.clone(),
            clients,
            context,
            request,
            Some(observer),
        )
        .await
        {
            Ok((decisions, response)) => {
                for decision in decisions {
                    self.emit_decision(
                        marks,
                        decision.as_ref(),
                        stage_request.as_ref(),
                        attempt,
                        mark_metadata,
                    );
                }
                self.emit_routing_llm_calls(
                    marks,
                    take_observed_calls(&observations),
                    attempt,
                    mark_metadata,
                    true,
                );
                Ok(response)
            }
            Err(error) => {
                self.emit_routing_llm_calls(
                    marks,
                    take_observed_calls(&observations),
                    attempt,
                    mark_metadata,
                    false,
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
        let decision = Arc::new(Decision::new(
            target_name,
            Some("trusted fallback target".into()),
            true,
        ));
        let context = context_from_metadata(request.metadata.as_ref());
        target
            .client
            .call(context, request, decision)
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
        decision: &Decision,
        request: Option<&Request>,
        attempt: u32,
        metadata: &Json,
    ) {
        let decision_source =
            request.and_then(|request| self.stage_decision_source(request, decision));
        let routing_tier = self.target_tiers.get(decision.selected_model_id()).copied();
        self.mark(
            marks,
            "switchyard.routing.decision",
            json!({
                "algorithm": self.algorithm.name(),
                "attempt": attempt,
                "selected_target": decision.selected_model_id(),
                "reasoning": decision.reasoning(),
                "routing_tier": routing_tier,
                "decision_source": decision_source,
                "is_routed_call": decision.is_answer_call(),
            }),
            metadata,
        );
    }

    fn stage_decision_source(
        &self,
        request: &Request,
        decision: &Decision,
    ) -> Option<&'static str> {
        let config = self.stage_marks.as_ref()?;
        let signals = ToolSignals::from_request(request, config.recent_turn_window);
        match pick_tier(&signals, config.picker, config.confidence_threshold) {
            PickOutcome::Resolved { source, .. } => Some(source.as_str()),
            PickOutcome::ConsultClassifier { .. } => {
                let classifier_decided = config.classifier_enabled
                    && decision
                        .reasoning()
                        .and_then(decision_confidence)
                        .is_some_and(|confidence| confidence > 0.0);
                Some(if classifier_decided {
                    "llm-classifier"
                } else {
                    "fall_open"
                })
            }
        }
    }

    fn emit_routing_llm_calls(
        &self,
        marks: &mut Vec<RoutingMark>,
        mut calls: Vec<LlmCallObservation>,
        attempt: u32,
        metadata: &Json,
        successful_run: bool,
    ) {
        // The last successful routed call produced the response represented by Relay's
        // outer LLM lifecycle event. Keep it out of these marks so consumers can add
        // routing overhead without counting the serving call twice. Earlier routed calls
        // are discarded candidates (for example, escalation's weak draft).
        if successful_run
            && let Some(position) = calls
                .iter()
                .rposition(|call| call.is_answer_call && call.is_success)
        {
            calls.remove(position);
        }

        for (index, call) in calls.into_iter().enumerate() {
            let routing_tier = self.target_tiers.get(&call.selected_model).copied();
            self.mark(
                marks,
                "switchyard.routing.llm_call",
                json!({
                    "algorithm": self.algorithm.name(),
                    "attempt": attempt,
                    "call_index": index + 1,
                    "selected_target": call.selected_model,
                    "routing_tier": routing_tier,
                    "call_role": if call.is_answer_call { "candidate" } else { "judge" },
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

fn decision_confidence(reasoning: &str) -> Option<f64> {
    let (_, suffix) = reasoning.rsplit_once("confidence ")?;
    let numeric = suffix
        .trim_start_matches(|character: char| {
            !character.is_ascii_digit() && !matches!(character, '.' | '-' | '+')
        })
        .chars()
        .take_while(|character| {
            character.is_ascii_digit() || matches!(character, '.' | '-' | '+' | 'e' | 'E')
        })
        .collect::<String>();
    numeric.parse().ok()
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

fn retry_backoff(attempt: u32) -> Duration {
    let exponent = attempt.saturating_sub(1).min(3);
    INITIAL_RETRY_BACKOFF
        .saturating_mul(1_u32 << exponent)
        .min(MAX_RETRY_BACKOFF)
}

async fn sleep_before_retry(attempt: u32) {
    tokio::time::sleep(retry_backoff(attempt)).await;
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

fn context_from_metadata(metadata: Option<&Metadata>) -> Context {
    let Some(metadata) = metadata else {
        return Context::default();
    };
    let mut values = std::collections::HashMap::new();
    for (name, value) in [
        ("session_id", metadata.session_id.as_deref()),
        ("agent_id", metadata.agent_id.as_deref()),
        ("parent_agent_id", metadata.parent_agent_id.as_deref()),
        ("agent_kind", metadata.agent_kind.as_deref()),
        ("agent_role", metadata.agent_role.as_deref()),
        ("task_id", metadata.task_id.as_deref()),
        ("task_kind", metadata.task_kind.as_deref()),
        ("turn_id", metadata.turn_id.as_deref()),
        ("correlation_id", metadata.correlation_id.as_deref()),
    ] {
        if let Some(value) = value {
            values.insert(name.to_string(), value.to_string());
        }
    }
    values.insert("is_subagent".into(), metadata.is_subagent.to_string());
    values.insert(
        "is_delegated_work".into(),
        metadata.is_delegated_work.to_string(),
    );
    if let Some(session_final) = metadata.session_final {
        values.insert("session_final".into(), session_final.to_string());
    }
    if let Some(extra) = &metadata.extra_metadata {
        for (name, value) in extra {
            values.entry(name.clone()).or_insert_with(|| value.clone());
        }
    }
    let mut context = Context::default();
    context.values = values;
    context
}

#[cfg(test)]
mod tests;
