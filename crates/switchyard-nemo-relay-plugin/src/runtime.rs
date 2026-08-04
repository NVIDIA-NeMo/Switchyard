// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::{StreamExt, stream};
use nemo_relay_plugin::{Json, LlmRequest as RelayRequest};
use serde_json::{Map, json};
use switchyard_libsy::{Algorithm, CallLlmRequest, LibsyError, Step};
use switchyard_protocol::{
    Context, Decision, LlmClientError, LlmResponse, Metadata, Request, Response, SimpleDecision,
    WireFormat,
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
    Error(String),
}

pub(crate) struct SwitchyardRuntime {
    max_retries: u32,
    algorithm: Arc<dyn Algorithm>,
    targets: BTreeMap<String, PreparedTargetBinding>,
    default_targets: BTreeMap<WireFormat, String>,
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
            match self.drive(request.clone(), attempt, marks, &metadata).await {
                Ok(response) => {
                    let LlmResponse::Agg(response) = response.llm_response else {
                        return Err("libsy returned a stream for a buffered request".into());
                    };
                    return translation::encode_response(&self.translation, inbound, &response);
                }
                Err(failure) if libsy_error_retryable(&failure) && attempt < max_attempts => {
                    self.mark(
                        marks,
                        "switchyard.routing.retry",
                        failure_mark_data(attempt, &failure),
                        &metadata,
                    );
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
                    let LlmResponse::Agg(response) = response.llm_response else {
                        return Err(
                            "trusted fallback returned a stream for a buffered request".into()
                        );
                    };
                    return translation::encode_response(&self.translation, inbound, &response);
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
        loop {
            self.mark(
                &mut marks,
                "switchyard.routing.requested",
                json!({"algorithm": self.algorithm.name(), "attempt": attempt}),
                &metadata,
            );
            let (response, fallback_used) = match self
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
                    attempt += 1;
                    send_marks(output, &mut marks).await?;
                    continue;
                }
                Err(failure) => {
                    self.mark(
                        &mut marks,
                        "switchyard.routing.error",
                        failure_mark_data(attempt, &failure),
                        &metadata,
                    );
                    (
                        self.fallback_response(inbound, request.clone(), &mut marks, &metadata)
                            .await?,
                        true,
                    )
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
                    attempt += 1;
                    send_marks(output, &mut marks).await?;
                    continue;
                }
                Err(failure) if !fallback_used => {
                    self.mark(
                        &mut marks,
                        "switchyard.routing.error",
                        failure_mark_data(attempt, &failure),
                        &metadata,
                    );
                    let fallback = self
                        .fallback_response(inbound, request.clone(), &mut marks, &metadata)
                        .await?;
                    send_marks(output, &mut marks).await?;
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
                        attempt += 1;
                        send_marks(output, &mut marks).await?;
                        break;
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
                            .await?;
                        send_marks(output, &mut marks).await?;
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
            // A retry path breaks the event loop before commitment and starts a
            // fresh libsy run. A successfully encoded stream always emits at
            // least one event because `returned_events` rejects empty inputs.
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
        let mut steps = self.algorithm.clone().run_stream(context, request, None);
        while let Some(step) = steps.next().await {
            match step {
                Ok(Step::Decision(decision)) => {
                    self.emit_decision(marks, decision.as_ref(), attempt, mark_metadata);
                }
                Ok(Step::CallLlm(call)) => self.serve_call(*call).await?,
                Ok(Step::ReturnToAgent(response)) => return Ok(*response),
                Err(error) => return Err(error),
            }
        }
        Err(LibsyError::MissingFinalResponse)
    }

    async fn serve_call(&self, call: CallLlmRequest) -> switchyard_libsy::Result<()> {
        let routed = call.get_routed().clone();
        let target_name = routed.decision.selected_model().to_string();
        let result = match routed.default_client {
            Some(client) => client
                .call(routed.ctx, routed.request, routed.decision)
                .await
                .map_err(|source| LibsyError::client_call(target_name, source)),
            None => Err(LibsyError::client_call(
                target_name,
                LlmClientError::Configuration {
                    message: "libsy CallLlm step has no Switchyard HTTP client".into(),
                },
            )),
        };
        call.respond(result)
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
        let decision: Arc<dyn Decision> = Arc::new(SimpleDecision {
            selected_model: target_name.to_string(),
            reasoning: Some("trusted fallback target".into()),
        });
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
        decision: &dyn Decision,
        attempt: u32,
        metadata: &Json,
    ) {
        self.mark(
            marks,
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

fn public_client_failure(prefix: &str, error: &LlmClientError) -> String {
    match error {
        LlmClientError::UpstreamHttp { status, .. } => {
            format!("{prefix}: provider returned HTTP {status}")
        }
        _ => format!("{prefix}: provider {} failure", client_error_label(error)),
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
    use super::*;

    #[test]
    fn context_carries_identity_without_http_headers() {
        let context = context_from_metadata(Some(&Metadata {
            session_id: Some("session-1".into()),
            agent_id: Some("agent-1".into()),
            is_subagent: true,
            extra_metadata: Some(BTreeMap::from([("tenant".into(), "blue".into())])),
            http_headers: Some(BTreeMap::from([(
                "authorization".into(),
                "Bearer caller-secret".into(),
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
