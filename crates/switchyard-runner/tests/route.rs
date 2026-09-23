// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use libsy::RuntimeModels;
use switchyard_llm_client::{ClientRouter, RunObservation};
use switchyard_protocol::{
    Category, LlmClientError, LlmResponse, ModelId, Request, Response, RoutedLlmClient,
    text_request, text_response,
};
use switchyard_runner::{AlgorithmSpec, ModelCapabilities, Route};

struct StubClient;

#[async_trait]
impl RoutedLlmClient for StubClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(
                request.llm_request.model.clone(),
                "plugin response",
            )),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

fn plugin_route(client: Arc<dyn RoutedLlmClient>) -> Route {
    let spec = AlgorithmSpec::Passthrough {
        target: "semantic-target".to_string(),
        subagents: None,
    };
    let targets = BTreeMap::from([(
        "semantic-target".to_string(),
        ModelId::from("semantic-target"),
    )]);
    let algorithm = spec
        .build("switchyard", &targets)
        .expect("identity target map should build");
    let clients = ClientRouter::new(
        BTreeMap::from([(ModelId::from("semantic-target"), client)])
            .into_iter()
            .collect(),
    );
    Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::new([(Category::Any, vec![ModelId::from("semantic-target")])].into()),
    )
}

#[tokio::test]
async fn plugin_shaped_route_executes_without_runner_model_or_toml() {
    let route = plugin_route(Arc::new(StubClient));
    let observations = Arc::new(Mutex::new(Vec::new()));
    let observer = {
        let observations = Arc::clone(&observations);
        Arc::new(move |observation| observations.lock().unwrap().push(observation))
    };
    let request = Request {
        llm_request: text_request(Some("arbitrary-upstream-model".to_string()), "hello"),
        ..Request::default()
    };
    let output = route
        .execute(request, Some(observer))
        .await
        .expect("route should execute");

    assert_eq!(output.selected_model, "semantic-target");
    assert_eq!(
        output
            .response
            .llm_response
            .as_agg()
            .unwrap()
            .model
            .as_deref(),
        Some("semantic-target")
    );
    assert!(
        observations
            .lock()
            .unwrap()
            .iter()
            .any(|observation| { matches!(observation, RunObservation::AnswerCall(_)) })
    );
    assert!(
        observations
            .lock()
            .unwrap()
            .iter()
            .any(|observation| { matches!(observation, RunObservation::RoutingOverhead(_)) })
    );
}

struct LazyStreamClient {
    polls: Arc<AtomicUsize>,
}

#[async_trait]
impl RoutedLlmClient for LazyStreamClient {
    async fn call(&self, _request: Request) -> Result<Response, LlmClientError> {
        let polls = Arc::clone(&self.polls);
        let stream = futures_util::stream::poll_fn(move |_| {
            polls.fetch_add(1, Ordering::SeqCst);
            std::task::Poll::Ready(None)
        })
        .boxed();
        Ok(Response {
            llm_response: LlmResponse::Stream(stream),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

#[tokio::test]
async fn route_returns_stream_without_polling_it() {
    let polls = Arc::new(AtomicUsize::new(0));
    let route = plugin_route(Arc::new(LazyStreamClient {
        polls: Arc::clone(&polls),
    }));
    let request = Request {
        llm_request: text_request(None, "hello"),
        ..Request::default()
    };

    let output = route
        .execute(request, None)
        .await
        .expect("stream handle should be returned");

    assert!(matches!(
        output.response.llm_response,
        LlmResponse::Stream(_)
    ));
    assert_eq!(polls.load(Ordering::SeqCst), 0);
}

// Preserve checkpoint target order and explicit TOML overrides.
#[test]
fn prefill_router_config_preserves_target_order_and_overrides() {
    let spec: AlgorithmSpec = toml::from_str(
        r#"
type = "prefill_router"
targets = ["fast", "strong"]
checkpoint = "/models/router.pt"
device = "cuda:1"
max_length = 4096
batch_size = 8
"#,
    )
    .expect("prefill router config should parse");

    let AlgorithmSpec::PrefillRouter {
        targets,
        checkpoint,
        device,
        cache_dir,
        max_length,
        batch_size,
    } = spec
    else {
        panic!("expected prefill router config");
    };
    assert_eq!(targets, ["fast", "strong"]);
    assert_eq!(checkpoint.to_string_lossy(), "/models/router.pt");
    assert_eq!(device.as_deref(), Some("cuda:1"));
    assert_eq!(cache_dir, None);
    assert_eq!(max_length, Some(4096));
    assert_eq!(batch_size, Some(8));
}

// ---RLCD decision routing -------------------------------------------------------

fn request() -> Request {
    Request {
        llm_request: text_request(Some("auto".to_string()), "hello"),
        ..Request::default()
    }
}

/// Answers the decision model with a calibrated verdict that picks `strong`,
/// and echoes every other target's name.
struct RlcdStubClient;

#[async_trait]
impl RoutedLlmClient for RlcdStubClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let model = request.llm_request.model.clone().unwrap_or_default();
        let text = if model == "decision" {
            r#"{"target":"strong","probabilities":[{"option":"fast","probability":0.2},{"option":"strong","probability":0.8}]}"#.to_string()
        } else {
            model
        };
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(request.llm_request.model.clone(), text)),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

fn rlcd_targets() -> BTreeMap<String, ModelId> {
    BTreeMap::from([
        ("decision".to_string(), ModelId::from("decision")),
        ("fast".to_string(), ModelId::from("fast")),
        ("strong".to_string(), ModelId::from("strong")),
    ])
}

fn rlcd_spec(default_target: &str) -> AlgorithmSpec {
    AlgorithmSpec::Rlcd {
        classifier_target: "decision".to_string(),
        targets: vec!["fast".to_string(), "strong".to_string()],
        default_target: default_target.to_string(),
        max_output_tokens: 128,
    }
}

fn rlcd_route(client: Arc<dyn RoutedLlmClient>) -> Route {
    let spec = rlcd_spec("fast");
    let algorithm = spec
        .build("rlcd_test", &rlcd_targets())
        .expect("rlcd spec should build");
    let clients = ClientRouter::new(
        BTreeMap::from([
            (ModelId::from("decision"), client.clone()),
            (ModelId::from("fast"), client.clone()),
            (ModelId::from("strong"), client),
        ])
        .into_iter()
        .collect(),
    );
    Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::new(
            [
                (Category::Judge, vec![ModelId::from("decision")]),
                (
                    Category::Any,
                    vec![ModelId::from("fast"), ModelId::from("strong")],
                ),
            ]
            .into(),
        ),
    )
}

#[tokio::test]
async fn rlcd_route_routes_through_the_argmax_target() {
    let route = rlcd_route(Arc::new(RlcdStubClient));
    let output = route
        .execute(request(), None)
        .await
        .expect("rlcd route should execute");

    assert_eq!(output.selected_model, "strong");
    assert_eq!(
        output
            .response
            .llm_response
            .as_agg()
            .unwrap()
            .model
            .as_deref(),
        Some("strong")
    );
}

/// A decision model that is never available; other targets answer normally.
struct UnavailableDecisionClient;

#[async_trait]
impl RoutedLlmClient for UnavailableDecisionClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        if request.llm_request.model.as_deref() == Some("decision") {
            return Err(LlmClientError::Timeout {
                source: Box::new(std::io::Error::other("decision model unreachable")),
            });
        }
        let model = request.llm_request.model.clone().unwrap_or_default();
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(request.llm_request.model.clone(), model)),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

#[tokio::test]
async fn rlcd_route_fails_when_the_decision_model_is_unavailable() {
    // A failed routing-time client call aborts the request — the same behavior
    // as a stalled classifier judge (see the server's client-deadline tests).
    let route = rlcd_route(Arc::new(UnavailableDecisionClient));
    let error = route
        .execute(request(), None)
        .await
        .err()
        .expect("route should fail with the decision model down");
    assert!(error.to_string().contains("decision"), "{error}");
}

#[test]
fn rlcd_spec_lists_decision_and_routing_targets() {
    let spec = rlcd_spec("fast");
    assert_eq!(spec.routing_target_names(), ["fast", "strong"]);
    assert_eq!(spec.callable_target_names(), ["fast", "strong", "decision"]);
}

#[test]
fn rlcd_spec_rejects_an_invalid_configuration() {
    let targets = rlcd_targets();
    let too_few = AlgorithmSpec::Rlcd {
        classifier_target: "decision".to_string(),
        targets: vec!["fast".to_string()],
        default_target: "fast".to_string(),
        max_output_tokens: 128,
    };
    let error = too_few
        .build("test", &targets)
        .err()
        .expect("build should fail");
    assert!(
        error.to_string().contains("at least two targets"),
        "{error}"
    );

    let bad_default = AlgorithmSpec::Rlcd {
        classifier_target: "decision".to_string(),
        targets: vec!["fast".to_string(), "strong".to_string()],
        default_target: "missing".to_string(),
        max_output_tokens: 128,
    };
    let error = bad_default
        .build("test", &targets)
        .err()
        .expect("build should fail");
    assert!(
        error.to_string().contains("must be one of targets"),
        "{error}"
    );

    let duplicate = AlgorithmSpec::Rlcd {
        classifier_target: "decision".to_string(),
        targets: vec!["fast".to_string(), "fast".to_string()],
        default_target: "fast".to_string(),
        max_output_tokens: 128,
    };
    let error = duplicate
        .build("test", &targets)
        .err()
        .expect("build should fail");
    assert!(error.to_string().contains("must be unique"), "{error}");

    let judge_is_target = AlgorithmSpec::Rlcd {
        classifier_target: "strong".to_string(),
        targets: vec!["fast".to_string(), "strong".to_string()],
        default_target: "fast".to_string(),
        max_output_tokens: 128,
    };
    let error = judge_is_target
        .build("test", &targets)
        .err()
        .expect("build should fail");
    assert!(error.to_string().contains("classifier_target"), "{error}");
}

#[test]
fn rlcd_spec_rejects_aliases_that_collide_after_resolution() {
    // Two target names resolving to one model would make every verdict
    // invalid, so the route would permanently fall back.
    let mut aliased = rlcd_targets();
    aliased.insert("fast-alias".to_string(), ModelId::from("fast"));
    let spec = AlgorithmSpec::Rlcd {
        classifier_target: "decision".to_string(),
        targets: vec!["fast".to_string(), "fast-alias".to_string()],
        default_target: "fast".to_string(),
        max_output_tokens: 128,
    };
    let error = spec
        .build("test", &aliased)
        .err()
        .expect("build should fail");
    assert!(
        error
            .to_string()
            .contains("resolve to duplicate model fast"),
        "{error}"
    );

    // The classifier itself must not resolve to a candidate model.
    let mut shared = rlcd_targets();
    shared.insert("decision".to_string(), ModelId::from("strong"));
    let spec = AlgorithmSpec::Rlcd {
        classifier_target: "decision".to_string(),
        targets: vec!["fast".to_string(), "strong".to_string()],
        default_target: "fast".to_string(),
        max_output_tokens: 128,
    };
    let error = spec
        .build("test", &shared)
        .err()
        .expect("build should fail");
    assert!(
        error
            .to_string()
            .contains("classifier_target resolves to candidate model strong"),
        "{error}"
    );
}
