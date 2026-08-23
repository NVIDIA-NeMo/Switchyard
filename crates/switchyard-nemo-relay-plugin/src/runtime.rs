// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::{Stream, StreamExt};
use nemo_relay_plugin::{Json, LlmRequest as RelayRequest};
use serde_json::{Map, json};
use switchyard_llm_client::{LlmCallObservation, RunObservation, RunObserver};
use switchyard_protocol::{LlmResponse, Metadata, Request, Response, WireFormat};
use switchyard_runner::{Route, Runner};
use switchyard_translation::{TranslationEngine, encode_stream};

use crate::config::SwitchyardConfig;
use crate::translation;

#[derive(Debug)]
pub(crate) struct RoutingMark {
    pub(crate) name: String,
    pub(crate) data: Json,
    pub(crate) metadata: Json,
}

pub(crate) type ReturnedEventStream = Pin<Box<dyn Stream<Item = Result<Json, String>> + Send>>;

pub(crate) struct Execution<T> {
    pub(crate) result: Result<T, String>,
    pub(crate) marks: Vec<RoutingMark>,
}

pub(crate) struct SwitchyardRuntime {
    runner: Runner,
    translation: TranslationEngine,
}

impl SwitchyardRuntime {
    pub(crate) fn new(config: SwitchyardConfig) -> Result<Self, String> {
        Ok(Self {
            runner: config.load_runner()?,
            translation: TranslationEngine::default(),
        })
    }

    pub(crate) fn manages(&self, request: &Request) -> bool {
        request
            .llm_request
            .model
            .as_deref()
            .is_some_and(|model| self.runner.route(model).is_some())
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
    ) -> Execution<Json> {
        let mut execution = self.execute(request).await;
        if let Ok(response) = execution.result {
            execution.result = finalize_buffered_response(&self.translation, inbound, response);
            if execution.result.is_err() {
                self.error_mark(&mut execution.marks, "response_finalization", None);
            }
        }
        execution
    }

    pub(crate) async fn execute_stream(
        &self,
        inbound: WireFormat,
        request: Request,
    ) -> Execution<ReturnedEventStream> {
        let mut execution = self.execute(request).await;
        if let Ok(response) = execution.result {
            execution.result = returned_events(response, inbound);
            if execution.result.is_err() {
                self.error_mark(&mut execution.marks, "response_finalization", None);
            }
        }
        execution
    }

    async fn execute(&self, request: Request) -> Execution<Response> {
        let Some(route) = self.route(&request) else {
            return Execution {
                result: Err("Switchyard has no route for this request model".into()),
                marks: Vec::new(),
            };
        };
        let metadata = identity_metadata(request.metadata.as_ref());
        let mut marks = vec![RoutingMark {
            name: "switchyard.routing.requested".into(),
            data: json!({"algorithm": route.algorithm_name()}),
            metadata: metadata.clone(),
        }];
        if let Err(error) = route.check_caller_format(metadata_wire_format(&request)) {
            self.error_mark(&mut marks, "caller_format", None);
            return Execution {
                result: Err(format!("Switchyard caller format is incompatible: {error}")),
                marks,
            };
        }
        let observations = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&observations);
        let observer: RunObserver = Arc::new(move |observation| {
            observed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(observation);
        });
        match route.execute(request, Some(observer)).await {
            Ok(output) => {
                self.emit_observations(&mut marks, take_observations(&observations), &metadata);
                marks.push(RoutingMark {
                    name: "switchyard.routing.decision".into(),
                    data: json!({
                        "algorithm": route.algorithm_name(),
                        "selected_model": output.selected_model,
                    }),
                    metadata,
                });
                Execution {
                    result: Ok(output.response),
                    marks,
                }
            }
            Err(_) => {
                self.emit_observations(&mut marks, take_observations(&observations), &metadata);
                self.error_mark(&mut marks, "route_execution", None);
                Execution {
                    result: Err("Switchyard route execution failed".into()),
                    marks,
                }
            }
        }
    }

    fn route(&self, request: &Request) -> Option<&Route> {
        request
            .llm_request
            .model
            .as_deref()
            .and_then(|model| self.runner.route(model))
    }

    fn emit_observations(
        &self,
        marks: &mut Vec<RoutingMark>,
        observations: Vec<RunObservation>,
        metadata: &Json,
    ) {
        let mut call_index = 0;
        for observation in observations {
            match observation {
                RunObservation::LlmCall(call) => {
                    call_index += 1;
                    self.routing_call_mark(marks, call, call_index, metadata);
                }
                RunObservation::RoutingOverhead(duration) => marks.push(RoutingMark {
                    name: "switchyard.routing.overhead".into(),
                    data: json!({"latency_ms": duration.as_secs_f64() * 1_000.0}),
                    metadata: metadata.clone(),
                }),
                RunObservation::AnswerCall(_) => {}
            }
        }
    }

    fn routing_call_mark(
        &self,
        marks: &mut Vec<RoutingMark>,
        call: LlmCallObservation,
        call_index: usize,
        metadata: &Json,
    ) {
        marks.push(RoutingMark {
            name: "switchyard.routing.llm_call".into(),
            data: json!({
                "call_index": call_index,
                "selected_model": call.selected_model,
                "call_role": "routing",
                "outcome": if call.is_success { "ok" } else { "error" },
                "latency_ms": call.duration.as_secs_f64() * 1_000.0,
                "usage": call.usage,
            }),
            metadata: metadata.clone(),
        });
    }

    fn error_mark(&self, marks: &mut Vec<RoutingMark>, failure_kind: &str, metadata: Option<&Json>) {
        let metadata = metadata.cloned().unwrap_or_else(|| {
            marks
                .first()
                .map(|mark| mark.metadata.clone())
                .unwrap_or_else(|| Json::Object(Map::new()))
        });
        marks.push(RoutingMark {
            name: "switchyard.routing.error".into(),
            data: json!({"failure_kind": failure_kind}),
            metadata,
        });
    }
}

fn take_observations(observations: &Mutex<Vec<RunObservation>>) -> Vec<RunObservation> {
    std::mem::take(
        &mut *observations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    )
}

fn metadata_wire_format(request: &Request) -> WireFormat {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.wire_format)
        .expect("decoded Relay requests always carry a wire format")
}

fn finalize_buffered_response(
    translation_engine: &TranslationEngine,
    inbound: WireFormat,
    response: Response,
) -> Result<Json, String> {
    let LlmResponse::Agg(response) = response.llm_response else {
        return Err("Switchyard returned a stream for a buffered request".into());
    };
    translation::encode_response(translation_engine, inbound, &response)
}

fn returned_events(response: Response, inbound: WireFormat) -> Result<ReturnedEventStream, String> {
    let chunks = match response.llm_response {
        LlmResponse::Agg(response) => response.into_stream(),
        LlmResponse::Stream(chunks) => chunks,
    };
    let events = encode_stream(chunks, inbound, None)
        .map_err(|error| format!("Switchyard response stream setup failed: {error}"))?;
    Ok(Box::pin(events.map(|item| {
        item.map_err(|error| format!("Switchyard response stream failed: {error}"))
    })))
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
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use switchyard_llm_client::ClientRouter;
    use switchyard_protocol::{ModelId, text_request};
    use switchyard_runner::{AlgorithmSpec, ModelCapabilities};

    use super::*;

    fn runtime_for(model: &str) -> SwitchyardRuntime {
        let algorithm = AlgorithmSpec::Noop {}
            .build("relay", &BTreeMap::new())
            .expect("noop route should build");
        let route = Route::new(
            algorithm,
            ClientRouter::new(HashMap::new()),
            None,
            ModelCapabilities::default(),
            None,
            Vec::new(),
        );
        SwitchyardRuntime {
            runner: Runner::new(vec![(ModelId::from(model), route)]),
            translation: TranslationEngine::default(),
        }
    }

    #[test]
    fn only_configured_route_models_are_managed() {
        let runtime = runtime_for("switchyard");
        let configured = Request {
            llm_request: text_request(Some("switchyard".into()), "hello"),
            ..Request::default()
        };
        let other = Request {
            llm_request: text_request(Some("other".into()), "hello"),
            ..Request::default()
        };

        assert!(runtime.manages(&configured));
        assert!(!runtime.manages(&other));
    }
}
