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
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use switchyard_libsy::{
        ClassifierContractConfig, EscalationJudgeConfig, LlmClassifierConfig, LlmFallback,
        LlmTarget, LlmTaskClassifier, Passthrough, PickerMode, StageRouter, StageRouterConfig,
        TaskClassifierConfig,
    };
    use switchyard_protocol::{
        ContentBlock, LlmRequest, LlmResponseStream, Message, Role, RoutedLlmClient, ToolCall,
        ToolResult, Usage, text_request, text_response,
    };

    use super::*;

    #[derive(Clone, Copy)]
    enum StreamBehavior {
        Empty,
        Failing,
        CallFailure,
    }

    struct StreamClient {
        behavior: StreamBehavior,
        calls: AtomicUsize,
    }

    struct BufferedClient {
        calls: AtomicUsize,
    }

    enum FixedBehavior {
        Text(&'static str),
        TransportFailure,
    }

    struct FixedClient {
        behavior: FixedBehavior,
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl RoutedLlmClient for StreamClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            _decision: Arc<Decision>,
        ) -> Result<Response, LlmClientError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let stream: LlmResponseStream = match self.behavior {
                StreamBehavior::Empty => Box::pin(stream::empty()),
                StreamBehavior::Failing => Box::pin(stream::once(async {
                    Err(LlmClientError::Transport {
                        source: Box::new(std::io::Error::other("fallback stream failed")),
                    })
                })),
                StreamBehavior::CallFailure => {
                    return Err(LlmClientError::Transport {
                        source: Box::new(std::io::Error::other("fallback call failed")),
                    });
                }
            };
            Ok(Response {
                llm_response: LlmResponse::Stream(stream),
                metadata: None,
            })
        }
    }

    #[async_trait::async_trait]
    impl RoutedLlmClient for BufferedClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            _decision: Arc<Decision>,
        ) -> Result<Response, LlmClientError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Response {
                llm_response: LlmResponse::Agg(Default::default()),
                metadata: None,
            })
        }
    }

    #[async_trait::async_trait]
    impl RoutedLlmClient for FixedClient {
        async fn call(
            &self,
            _ctx: Context,
            request: Request,
            _decision: Arc<Decision>,
        ) -> Result<Response, LlmClientError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            match self.behavior {
                FixedBehavior::Text(text) => {
                    let mut response = text_response(None, text);
                    response.usage = Usage {
                        input_tokens: Some(11),
                        output_tokens: Some(7),
                        total_tokens: Some(18),
                        ..Usage::default()
                    };
                    Ok(Response {
                        llm_response: LlmResponse::Agg(response),
                        metadata: request.metadata,
                    })
                }
                FixedBehavior::TransportFailure => Err(LlmClientError::Transport {
                    source: Box::new(std::io::Error::other("scripted failure")),
                }),
            }
        }
    }

    fn fixed_target(name: &str, _client: Arc<FixedClient>) -> LlmTarget {
        LlmTarget {
            semantic_name: name.to_string(),
        }
    }

    fn runtime_with_algorithm(
        algorithm: Arc<dyn Algorithm>,
        fallback: Arc<FixedClient>,
        protocol: WireFormat,
    ) -> SwitchyardRuntime {
        runtime_with_algorithm_clients(algorithm, fallback, protocol, Vec::new())
    }

    fn runtime_with_algorithm_clients(
        algorithm: Arc<dyn Algorithm>,
        fallback: Arc<FixedClient>,
        protocol: WireFormat,
        clients: Vec<(&str, Arc<FixedClient>)>,
    ) -> SwitchyardRuntime {
        let is_stage = algorithm.name() == "stage_router";
        let mut targets = BTreeMap::from([(
            "fallback".into(),
            PreparedTargetBinding {
                client: fallback as Arc<dyn RoutedLlmClient>,
            },
        )]);
        for (name, client) in clients {
            targets.insert(
                name.to_string(),
                PreparedTargetBinding {
                    client: client as Arc<dyn RoutedLlmClient>,
                },
            );
        }
        SwitchyardRuntime {
            max_retries: 0,
            algorithm,
            targets,
            default_targets: BTreeMap::from([(protocol, "fallback".into())]),
            target_tiers: BTreeMap::from([("weak".into(), "weak"), ("strong".into(), "strong")]),
            stage_marks: is_stage.then_some(StageMarkConfig {
                picker: PickerMode::CapableFirst,
                confidence_threshold: 0.5,
                recent_turn_window: None,
                classifier_enabled: true,
            }),
            translation: TranslationEngine::default(),
        }
    }

    fn request_with_session(protocol: WireFormat, session: Option<&str>) -> Request {
        Request {
            llm_request: text_request(Some("auto".into()), "fix the build"),
            raw_request: None,
            metadata: Some(Metadata {
                wire_format: Some(protocol),
                session_id: session.map(str::to_string),
                ..Metadata::default()
            }),
        }
    }

    fn stage_signal_request(protocol: WireFormat) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("auto".into()),
                messages: vec![
                    Message::text(Role::User, "fix the build"),
                    Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::ToolCall(ToolCall {
                            id: "call-1".into(),
                            name: "bash".into(),
                            arguments: json!({"cmd": "cargo test"}),
                        })],
                    },
                    Message {
                        role: Role::Tool,
                        content: vec![ContentBlock::ToolResult(ToolResult {
                            tool_call_id: "call-1".into(),
                            content: vec![ContentBlock::Text {
                                text: "fatal runtime error: out of memory".into(),
                            }],
                            is_error: Some(true),
                        })],
                    },
                ],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: Some(Metadata {
                wire_format: Some(protocol),
                session_id: Some(format!("stage-{}", protocol.as_str())),
                ..Metadata::default()
            }),
        }
    }

    #[test]
    fn relay_gateway_placeholder_session_is_not_retained() {
        let fallback = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("fallback"),
            calls: AtomicUsize::new(0),
        });
        let runtime = runtime_with_algorithm(
            Arc::new(Passthrough::new(LlmTarget {
                semantic_name: "selected".into(),
            })),
            fallback,
            WireFormat::OpenAiChat,
        );
        let request = RelayRequest {
            headers: Map::from_iter([
                ("x-nemo-relay-source".into(), json!("gateway")),
                ("x-nemo-relay-session-id".into(), json!("gateway-gateway")),
                ("x-dynamo-session-id".into(), json!("gateway-gateway")),
            ]),
            content: json!({
                "model": "router",
                "messages": [{"role": "user", "content": "hello"}]
            }),
        };

        let decoded = runtime
            .decode_request(WireFormat::OpenAiChat, &request, false)
            .unwrap();

        assert_eq!(decoded.metadata.unwrap().session_id, None);
    }

    #[test]
    fn explicit_switchyard_session_overrides_relay_gateway_placeholder() {
        let fallback = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("fallback"),
            calls: AtomicUsize::new(0),
        });
        let runtime = runtime_with_algorithm(
            Arc::new(Passthrough::new(LlmTarget {
                semantic_name: "selected".into(),
            })),
            fallback,
            WireFormat::OpenAiChat,
        );
        let request = RelayRequest {
            headers: Map::from_iter([
                ("x-switchyard-session-id".into(), json!("caller-session")),
                ("x-nemo-relay-source".into(), json!("gateway")),
                ("x-nemo-relay-session-id".into(), json!("gateway-gateway")),
            ]),
            content: json!({
                "model": "router",
                "messages": [{"role": "user", "content": "hello"}]
            }),
        };

        let decoded = runtime
            .decode_request(WireFormat::OpenAiChat, &request, false)
            .unwrap();

        assert_eq!(
            decoded.metadata.unwrap().session_id.as_deref(),
            Some("caller-session")
        );
    }

    #[tokio::test]
    async fn buffered_finalization_failure_uses_fallback_once() {
        let selected = Arc::new(StreamClient {
            behavior: StreamBehavior::Empty,
            calls: AtomicUsize::new(0),
        });
        let fallback = Arc::new(BufferedClient {
            calls: AtomicUsize::new(0),
        });
        let runtime = SwitchyardRuntime {
            max_retries: 1,
            algorithm: Arc::new(Passthrough::new(LlmTarget {
                semantic_name: "selected".into(),
            })),
            targets: BTreeMap::from([
                (
                    "selected".into(),
                    PreparedTargetBinding {
                        client: selected.clone(),
                    },
                ),
                (
                    "fallback".into(),
                    PreparedTargetBinding {
                        client: fallback.clone(),
                    },
                ),
            ]),
            default_targets: BTreeMap::from([(WireFormat::OpenAiChat, "fallback".into())]),
            target_tiers: BTreeMap::new(),
            stage_marks: None,
            translation: TranslationEngine::default(),
        };
        let mut marks = Vec::new();

        let response = runtime
            .execute_buffered(WireFormat::OpenAiChat, Request::default(), &mut marks)
            .await
            .expect("the buffered fallback response should be encoded");

        assert!(response.is_object());
        assert_eq!(selected.calls.load(Ordering::Relaxed), 1);
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 1);
        assert!(
            !marks
                .iter()
                .any(|mark| mark.name == "switchyard.routing.retry")
        );
        let error = marks
            .iter()
            .find(|mark| mark.name == "switchyard.routing.error")
            .expect("finalization failure should emit an error mark");
        assert_eq!(error.data["retryable"], false);
        assert_eq!(error.data["non_http_kind"], "invalid_response");
        assert_eq!(
            marks
                .iter()
                .filter(|mark| mark.name == "switchyard.routing.fallback")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn returned_events_replays_preserved_openai_chat_without_duplicate_terminal() {
        let content = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "system_fingerprint": "fp_provider_specific",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hi"},
                "finish_reason": null
            }]
        });
        let terminal = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        });
        let body = format!("data: {content}\n\ndata: {terminal}\n\ndata: [DONE]\n\n").into_bytes();
        let stream = switchyard_translation::decode_stream(
            stream::once(async move { Ok::<_, LlmClientError>(body) }),
            WireFormat::OpenAiChat,
        )
        .expect("provider SSE should decode");
        let response = Response {
            llm_response: LlmResponse::Stream(stream),
            metadata: None,
        };

        let replayed = returned_events(response, WireFormat::OpenAiChat)
            .await
            .expect("return stream should encode")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("return stream should not fail");

        assert_eq!(replayed, vec![content, terminal]);
    }

    #[tokio::test]
    async fn invalid_selected_stream_does_not_invoke_failing_fallback_twice() {
        let selected = Arc::new(StreamClient {
            behavior: StreamBehavior::Empty,
            calls: AtomicUsize::new(0),
        });
        let fallback = Arc::new(StreamClient {
            behavior: StreamBehavior::Failing,
            calls: AtomicUsize::new(0),
        });
        let runtime = SwitchyardRuntime {
            max_retries: 0,
            algorithm: Arc::new(Passthrough::new(LlmTarget {
                semantic_name: "selected".into(),
            })),
            targets: BTreeMap::from([
                (
                    "selected".into(),
                    PreparedTargetBinding {
                        client: selected.clone(),
                    },
                ),
                (
                    "fallback".into(),
                    PreparedTargetBinding {
                        client: fallback.clone(),
                    },
                ),
            ]),
            default_targets: BTreeMap::from([(WireFormat::OpenAiChat, "fallback".into())]),
            target_tiers: BTreeMap::new(),
            stage_marks: None,
            translation: TranslationEngine::default(),
        };
        let (output, _messages) = async_channel::bounded(32);

        let error = runtime
            .execute_stream(WireFormat::OpenAiChat, Request::default(), &output)
            .await
            .expect_err("the failing fallback stream must fail the request");

        assert_eq!(error, "trusted fallback stream: provider transport failure");
        assert_eq!(selected.calls.load(Ordering::Relaxed), 1);
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn failing_fallback_call_flushes_error_and_fallback_marks() {
        let selected = Arc::new(StreamClient {
            behavior: StreamBehavior::Empty,
            calls: AtomicUsize::new(0),
        });
        let fallback = Arc::new(StreamClient {
            behavior: StreamBehavior::CallFailure,
            calls: AtomicUsize::new(0),
        });
        let runtime = SwitchyardRuntime {
            max_retries: 0,
            algorithm: Arc::new(Passthrough::new(LlmTarget {
                semantic_name: "selected".into(),
            })),
            targets: BTreeMap::from([
                (
                    "selected".into(),
                    PreparedTargetBinding {
                        client: selected.clone(),
                    },
                ),
                (
                    "fallback".into(),
                    PreparedTargetBinding {
                        client: fallback.clone(),
                    },
                ),
            ]),
            default_targets: BTreeMap::from([(WireFormat::OpenAiChat, "fallback".into())]),
            target_tiers: BTreeMap::new(),
            stage_marks: None,
            translation: TranslationEngine::default(),
        };
        let (output, messages) = async_channel::bounded(32);

        let error = runtime
            .execute_stream(WireFormat::OpenAiChat, Request::default(), &output)
            .await
            .expect_err("the failing fallback call must fail the request");

        assert_eq!(error, "trusted fallback: provider transport failure");
        assert_eq!(selected.calls.load(Ordering::Relaxed), 1);
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 1);
        let mut terminal_marks = Vec::new();
        while let Ok(message) = messages.try_recv() {
            if let StreamMessage::Mark(mark) = message
                && matches!(
                    mark.name.as_str(),
                    "switchyard.routing.error" | "switchyard.routing.fallback"
                )
            {
                terminal_marks.push(mark.name);
            }
        }
        assert_eq!(
            terminal_marks,
            ["switchyard.routing.error", "switchyard.routing.fallback"]
        );
    }

    #[test]
    fn retry_backoff_increases_exponentially_and_is_capped() {
        assert_eq!(retry_backoff(1), Duration::from_millis(250));
        assert_eq!(retry_backoff(2), Duration::from_millis(500));
        assert_eq!(retry_backoff(3), Duration::from_secs(1));
        assert_eq!(retry_backoff(4), Duration::from_secs(2));
        assert_eq!(retry_backoff(u32::MAX), Duration::from_secs(2));
    }

    #[tokio::test]
    async fn capability_classifier_emits_judge_usage_without_serving_usage() {
        let weak = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("weak answer"),
            calls: AtomicUsize::new(0),
        });
        let strong = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("strong answer"),
            calls: AtomicUsize::new(0),
        });
        let judge = Arc::new(FixedClient {
            behavior: FixedBehavior::Text(
                r#"{"crux":"bounded","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#,
            ),
            calls: AtomicUsize::new(0),
        });
        let fallback = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("fallback"),
            calls: AtomicUsize::new(0),
        });
        let algorithm = LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: fixed_target("judge", judge.clone()),
            efficient_target: fixed_target("weak", weak.clone()),
            capable_target: fixed_target("strong", strong.clone()),
            config: TaskClassifierConfig {
                base_threshold: 0.5,
                ..TaskClassifierConfig::default()
            },
        })
        .unwrap();
        let runtime = runtime_with_algorithm_clients(
            Arc::new(algorithm),
            fallback,
            WireFormat::OpenAiChat,
            vec![
                ("weak", weak.clone()),
                ("strong", strong.clone()),
                ("judge", judge.clone()),
            ],
        );
        let mut marks = Vec::new();

        runtime
            .execute_buffered(
                WireFormat::OpenAiChat,
                request_with_session(WireFormat::OpenAiChat, Some("capability")),
                &mut marks,
            )
            .await
            .unwrap();

        assert_eq!(judge.calls.load(Ordering::Relaxed), 1);
        assert_eq!(weak.calls.load(Ordering::Relaxed), 1);
        assert_eq!(strong.calls.load(Ordering::Relaxed), 0);
        let routing_calls = marks
            .iter()
            .filter(|mark| mark.name == "switchyard.routing.llm_call")
            .collect::<Vec<_>>();
        assert_eq!(routing_calls.len(), 1);
        assert_eq!(routing_calls[0].data["selected_target"], "judge");
        assert_eq!(routing_calls[0].data["usage"]["total_tokens"], 18);
    }

    #[tokio::test]
    async fn escalation_buffers_weak_stream_then_latches_the_session_to_strong() {
        let weak = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("weak draft"),
            calls: AtomicUsize::new(0),
        });
        let strong = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("strong answer"),
            calls: AtomicUsize::new(0),
        });
        let judge = Arc::new(FixedClient {
            behavior: FixedBehavior::Text(r#"{"escalate":true,"reason":"stuck"}"#),
            calls: AtomicUsize::new(0),
        });
        let fallback = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("fallback"),
            calls: AtomicUsize::new(0),
        });
        let algorithm = LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
            judge_target: fixed_target("judge", judge.clone()),
            efficient_target: fixed_target("weak", weak.clone()),
            capable_target: fixed_target("strong", strong.clone()),
            contract: ClassifierContractConfig::default(),
            config: EscalationJudgeConfig {
                confirmations: 1,
                ..EscalationJudgeConfig::default()
            },
            max_output_tokens: 128,
        })
        .unwrap();
        let runtime = runtime_with_algorithm_clients(
            Arc::new(algorithm),
            fallback.clone(),
            WireFormat::OpenAiChat,
            vec![
                ("weak", weak.clone()),
                ("strong", strong.clone()),
                ("judge", judge.clone()),
            ],
        );

        let mut first = request_with_session(WireFormat::OpenAiChat, Some("session-1"));
        first.llm_request.stream = true;
        let (output, messages) = async_channel::bounded(32);
        runtime
            .execute_stream(WireFormat::OpenAiChat, first, &output)
            .await
            .unwrap();
        let mut streamed = Vec::new();
        let mut routing_calls = Vec::new();
        while let Ok(message) = messages.try_recv() {
            match message {
                StreamMessage::Event(event) => streamed.push(event),
                StreamMessage::Mark(mark) if mark.name == "switchyard.routing.llm_call" => {
                    routing_calls.push(mark.data)
                }
                StreamMessage::Mark(_) => {}
            }
        }
        assert!(!streamed.is_empty());
        assert!(
            streamed
                .iter()
                .any(|event| event.to_string().contains("strong answer"))
        );
        assert_eq!(routing_calls.len(), 2);
        assert_eq!(routing_calls[0]["selected_target"], "weak");
        assert_eq!(routing_calls[0]["call_role"], "candidate");
        assert_eq!(routing_calls[0]["usage"]["total_tokens"], 18);
        assert_eq!(routing_calls[1]["selected_target"], "judge");
        assert_eq!(routing_calls[1]["call_role"], "judge");
        assert_eq!(routing_calls[1]["usage"]["total_tokens"], 18);
        assert!(
            routing_calls
                .iter()
                .all(|call| call["selected_target"] != "strong")
        );

        let mut marks = Vec::new();
        let response = runtime
            .execute_buffered(
                WireFormat::OpenAiChat,
                request_with_session(WireFormat::OpenAiChat, Some("session-1")),
                &mut marks,
            )
            .await
            .unwrap();
        assert!(response.to_string().contains("strong answer"));
        assert_eq!(weak.calls.load(Ordering::Relaxed), 1);
        assert_eq!(judge.calls.load(Ordering::Relaxed), 1);
        assert_eq!(strong.calls.load(Ordering::Relaxed), 2);
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
        assert!(
            !marks
                .iter()
                .any(|mark| mark.name == "switchyard.routing.llm_call")
        );
        assert!(marks.iter().any(|mark| {
            mark.name == "switchyard.routing.decision"
                && mark.data["selected_target"] == "strong"
                && mark.data["routing_tier"] == "strong"
                && mark.metadata["session_id"] == "session-1"
        }));
    }

    #[tokio::test]
    async fn escalation_judge_failure_falls_open_to_the_buffered_weak_response() {
        let weak = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("weak answer"),
            calls: AtomicUsize::new(0),
        });
        let strong = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("strong answer"),
            calls: AtomicUsize::new(0),
        });
        let judge = Arc::new(FixedClient {
            behavior: FixedBehavior::TransportFailure,
            calls: AtomicUsize::new(0),
        });
        let fallback = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("fallback"),
            calls: AtomicUsize::new(0),
        });
        let algorithm = LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
            judge_target: fixed_target("judge", judge.clone()),
            efficient_target: fixed_target("weak", weak.clone()),
            capable_target: fixed_target("strong", strong.clone()),
            contract: ClassifierContractConfig::default(),
            config: EscalationJudgeConfig::default(),
            max_output_tokens: 128,
        })
        .unwrap();
        let runtime = runtime_with_algorithm_clients(
            Arc::new(algorithm),
            fallback.clone(),
            WireFormat::OpenAiChat,
            vec![
                ("weak", weak.clone()),
                ("strong", strong.clone()),
                ("judge", judge.clone()),
            ],
        );
        let mut marks = Vec::new();

        let response = runtime
            .execute_buffered(
                WireFormat::OpenAiChat,
                request_with_session(WireFormat::OpenAiChat, Some("session-1")),
                &mut marks,
            )
            .await
            .unwrap();

        assert!(response.to_string().contains("weak answer"));
        assert_eq!(weak.calls.load(Ordering::Relaxed), 1);
        assert_eq!(judge.calls.load(Ordering::Relaxed), 1);
        assert_eq!(strong.calls.load(Ordering::Relaxed), 0);
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
        let routing_calls = marks
            .iter()
            .filter(|mark| mark.name == "switchyard.routing.llm_call")
            .collect::<Vec<_>>();
        assert_eq!(routing_calls.len(), 1);
        assert_eq!(routing_calls[0].data["selected_target"], "judge");
        assert_eq!(routing_calls[0].data["call_role"], "judge");
        assert_eq!(routing_calls[0].data["outcome"], "error");
        assert!(routing_calls[0].data["usage"].is_null());
    }

    #[tokio::test]
    async fn escalation_without_session_identity_cannot_accumulate_confirmations() {
        let weak = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("weak answer"),
            calls: AtomicUsize::new(0),
        });
        let strong = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("strong answer"),
            calls: AtomicUsize::new(0),
        });
        let judge = Arc::new(FixedClient {
            behavior: FixedBehavior::Text(r#"{"escalate":true,"reason":"stuck"}"#),
            calls: AtomicUsize::new(0),
        });
        let fallback = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("fallback"),
            calls: AtomicUsize::new(0),
        });
        let algorithm = LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
            judge_target: fixed_target("judge", judge.clone()),
            efficient_target: fixed_target("weak", weak.clone()),
            capable_target: fixed_target("strong", strong.clone()),
            contract: ClassifierContractConfig::default(),
            config: EscalationJudgeConfig {
                confirmations: 2,
                ..EscalationJudgeConfig::default()
            },
            max_output_tokens: 128,
        })
        .unwrap();
        let runtime = runtime_with_algorithm_clients(
            Arc::new(algorithm),
            fallback.clone(),
            WireFormat::OpenAiChat,
            vec![
                ("weak", weak.clone()),
                ("strong", strong.clone()),
                ("judge", judge.clone()),
            ],
        );

        for _ in 0..2 {
            let mut marks = Vec::new();
            let response = runtime
                .execute_buffered(
                    WireFormat::OpenAiChat,
                    request_with_session(WireFormat::OpenAiChat, None),
                    &mut marks,
                )
                .await
                .unwrap();
            assert!(response.to_string().contains("weak answer"));
        }
        assert_eq!(weak.calls.load(Ordering::Relaxed), 2);
        assert_eq!(judge.calls.load(Ordering::Relaxed), 2);
        assert_eq!(strong.calls.load(Ordering::Relaxed), 0);
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn stage_router_uses_tool_signals_for_every_managed_protocol() {
        for protocol in [
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            WireFormat::AnthropicMessages,
        ] {
            let capable = Arc::new(FixedClient {
                behavior: FixedBehavior::Text("capable answer"),
                calls: AtomicUsize::new(0),
            });
            let efficient = Arc::new(FixedClient {
                behavior: FixedBehavior::Text("efficient answer"),
                calls: AtomicUsize::new(0),
            });
            let fallback = Arc::new(FixedClient {
                behavior: FixedBehavior::Text("fallback"),
                calls: AtomicUsize::new(0),
            });
            let algorithm = StageRouter::new(
                fixed_target("strong", capable.clone()),
                fixed_target("weak", efficient.clone()),
                StageRouterConfig::new(PickerMode::EfficientFirst, 0.5),
            )
            .unwrap();
            let runtime = runtime_with_algorithm_clients(
                Arc::new(algorithm),
                fallback.clone(),
                protocol,
                vec![("strong", capable.clone()), ("weak", efficient.clone())],
            );
            let mut marks = Vec::new();

            let response = runtime
                .execute_buffered(protocol, stage_signal_request(protocol), &mut marks)
                .await
                .unwrap();

            assert!(response.to_string().contains("capable answer"));
            assert_eq!(capable.calls.load(Ordering::Relaxed), 1);
            assert_eq!(efficient.calls.load(Ordering::Relaxed), 0);
            assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
            assert!(marks.iter().any(|mark| {
                mark.name == "switchyard.routing.decision"
                    && mark.data["algorithm"] == "stage_router"
                    && mark.data["selected_target"] == "strong"
                    && mark.data["routing_tier"] == "strong"
                    && mark.data["decision_source"] == "override"
                    && mark.metadata["session_id"] == format!("stage-{}", protocol.as_str())
            }));
        }
    }

    #[tokio::test]
    async fn stage_router_falls_open_to_each_picker_default_without_tool_history() {
        for (picker, expected) in [
            (PickerMode::CapableFirst, "strong"),
            (PickerMode::EfficientFirst, "weak"),
        ] {
            let capable = Arc::new(FixedClient {
                behavior: FixedBehavior::Text("strong"),
                calls: AtomicUsize::new(0),
            });
            let efficient = Arc::new(FixedClient {
                behavior: FixedBehavior::Text("weak"),
                calls: AtomicUsize::new(0),
            });
            let fallback = Arc::new(FixedClient {
                behavior: FixedBehavior::Text("fallback"),
                calls: AtomicUsize::new(0),
            });
            let algorithm = StageRouter::new(
                fixed_target("strong", capable.clone()),
                fixed_target("weak", efficient.clone()),
                StageRouterConfig::new(picker, 0.5),
            )
            .unwrap();
            let runtime = runtime_with_algorithm_clients(
                Arc::new(algorithm),
                fallback,
                WireFormat::OpenAiChat,
                vec![("strong", capable), ("weak", efficient)],
            );
            let mut marks = Vec::new();

            runtime
                .execute_buffered(
                    WireFormat::OpenAiChat,
                    request_with_session(WireFormat::OpenAiChat, None),
                    &mut marks,
                )
                .await
                .unwrap();

            assert!(marks.iter().any(|mark| {
                mark.name == "switchyard.routing.decision"
                    && mark.data["selected_target"] == expected
                    && mark.data["routing_tier"] == expected
                    && mark.data["decision_source"] == "fall_open"
            }));
        }
    }

    #[tokio::test]
    async fn stage_router_classifier_resolves_an_ambiguous_turn() {
        let capable = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("strong"),
            calls: AtomicUsize::new(0),
        });
        let efficient = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("weak"),
            calls: AtomicUsize::new(0),
        });
        let judge = Arc::new(FixedClient {
            behavior: FixedBehavior::Text(
                r#"{"crux":"bounded","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#,
            ),
            calls: AtomicUsize::new(0),
        });
        let fallback = Arc::new(FixedClient {
            behavior: FixedBehavior::Text("fallback"),
            calls: AtomicUsize::new(0),
        });
        let mut config = StageRouterConfig::new(PickerMode::CapableFirst, 0.5);
        config.llm_fallback = Some(LlmFallback {
            judge_target: fixed_target("judge", judge.clone()),
            config: TaskClassifierConfig {
                base_threshold: 0.5,
                ..TaskClassifierConfig::default()
            },
        });
        let algorithm = StageRouter::new(
            fixed_target("strong", capable.clone()),
            fixed_target("weak", efficient.clone()),
            config,
        )
        .unwrap();
        let runtime = runtime_with_algorithm_clients(
            Arc::new(algorithm),
            fallback.clone(),
            WireFormat::OpenAiChat,
            vec![
                ("strong", capable.clone()),
                ("weak", efficient.clone()),
                ("judge", judge.clone()),
            ],
        );
        let mut marks = Vec::new();

        runtime
            .execute_buffered(
                WireFormat::OpenAiChat,
                request_with_session(WireFormat::OpenAiChat, Some("stage-classifier")),
                &mut marks,
            )
            .await
            .unwrap();

        assert_eq!(judge.calls.load(Ordering::Relaxed), 1);
        assert_eq!(efficient.calls.load(Ordering::Relaxed), 1);
        assert_eq!(capable.calls.load(Ordering::Relaxed), 0);
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
        let routing_calls = marks
            .iter()
            .filter(|mark| mark.name == "switchyard.routing.llm_call")
            .collect::<Vec<_>>();
        assert_eq!(routing_calls.len(), 1);
        assert_eq!(routing_calls[0].data["selected_target"], "judge");
        assert_eq!(routing_calls[0].data["call_role"], "judge");
        assert_eq!(routing_calls[0].data["outcome"], "ok");
        assert_eq!(routing_calls[0].data["usage"]["total_tokens"], 18);
        assert!(marks.iter().any(|mark| {
            mark.name == "switchyard.routing.decision"
                && mark.data["selected_target"] == "weak"
                && mark.data["routing_tier"] == "weak"
                && mark.data["decision_source"] == "llm-classifier"
        }));
    }

    #[test]
    fn context_carries_identity_without_http_headers() {
        let context = context_from_metadata(Some(&Metadata {
            session_id: Some("session-1".into()),
            agent_id: Some("agent-1".into()),
            is_subagent: true,
            extra_metadata: Some(BTreeMap::from([("tenant".into(), "blue".into())])),
            http_headers: Some(http::HeaderMap::from_iter([(
                http::HeaderName::from_static("authorization"),
                http::HeaderValue::from_static("Bearer caller-secret"),
            )])),
            ..Metadata::default()
        }));

        assert_eq!(
            context.values.get("session_id").map(String::as_str),
            Some("session-1")
        );
        assert_eq!(
            context.values.get("agent_id").map(String::as_str),
            Some("agent-1")
        );
        assert_eq!(
            context.values.get("is_subagent").map(String::as_str),
            Some("true")
        );
        assert_eq!(
            context.values.get("tenant").map(String::as_str),
            Some("blue")
        );
        assert!(!context.values.contains_key("authorization"));
    }
}
