// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Drive a libsy algorithm to completion, making the model calls it offloads.
//!
//! [`switchyard_libsy::Algorithm::run_stream`] is the whole libsy API: it yields a stream of
//! steps and expects its consumer to serve every offloaded model call. [`run()`] is that
//! consumer — it drives the stream with [`switchyard_libsy::drive`], hands each call to a
//! [`RoutedLlmClient`], and returns the final response with the trace of decisions.
//!
//! libsy owns the stream mechanics; what this module adds is ordered candidate fallback and the
//! `libsy.client_call` span around each candidate. Each candidate exhausts its backend retry
//! budget before fallback advances, so the worst case is `candidates × (max_retries + 1)`
//! upstream attempts plus every candidate's backoff.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use switchyard_libsy::{Algorithm, CallModel, LibsyError, Result, drive};
use switchyard_protocol::{
    Decision, LlmClientError, ModelId, Request, Response, RoutedLlmClient, RoutingFallbackReason,
};

use crate::observation::{LlmCallObservation, RunObservation, RunObserver};
use crate::{metrics, observability};

/// Run one request to completion, serving every offloaded model call with `client`.
///
/// Returns the final [`Response`] and the trace of [`Decision`]s the algorithm published along
/// the way. `observer`, when present, receives each completed model call and, after a
/// successful routed run, its routing overhead.
///
/// `clients` resolves each offloaded call to the client for the target the algorithm
/// selected — an algorithm may route among targets served by different providers, so this is
/// a per-call lookup, not one client for the whole run. Use
/// [`ClientRouter::single`](ClientRouter::single) when one client serves every target.
///
/// A failed *model* call is forwarded back into the algorithm, which may route around it;
/// this returns `Err` only when the run itself cannot complete.
pub async fn run(
    algorithm: Arc<dyn Algorithm>,
    clients: ClientRouter,
    request: Request,
    observer: Option<RunObserver>,
) -> Result<(Vec<Decision>, Response)> {
    let algorithm_name = algorithm.name().to_string();
    // The output from `serve` goes in here: when each successful routed call was in
    // flight. Everything else the run spent time on is routing overhead.
    let routed_calls = Arc::new(Mutex::new(RoutedCallWindows::default()));
    let run_started = Instant::now();
    let result = drive(algorithm, request, {
        let observer = observer.clone();
        let routed_calls = Arc::clone(&routed_calls);
        move |call| {
            serve(
                clients.clone(),
                call,
                observer.clone(),
                Arc::clone(&routed_calls),
            )
        }
    })
    .await?;
    if let Some(served) = routed_calls.lock().served() {
        let overhead =
            metrics::record_routing_overhead(&algorithm_name, run_started.elapsed(), served);
        if let Some(observer) = observer {
            observer(RunObservation::RoutingOverhead(overhead));
        }
    }
    Ok(result)
}

/// The wall-clock windows during which a successful routed call was in flight.
/// An algorithm can make multiple overlapping calls. This is the union of all of them.
#[derive(Default)]
struct RoutedCallWindows(Vec<(Instant, Instant)>);

impl RoutedCallWindows {
    /// Record one completed routed call.
    fn record(&mut self, started: Instant, ended: Instant) {
        self.0.push((started, ended));
    }

    /// Total time at least one routed call was in flight, merging overlapping windows.
    fn served(&mut self) -> Option<Duration> {
        self.0.sort_unstable_by_key(|(started, _)| *started);
        // Sweep the windows in start order, advancing a cursor along the timeline and
        // counting only the time each window covers that the cursor has not reached.
        let mut covered = self.0.first()?.0;
        let mut total = Duration::ZERO;
        for &(started, ended) in &self.0 {
            covered = covered.max(started);
            if ended > covered {
                total += ended - covered;
                covered = ended;
            }
        }
        Some(total)
    }
}

/// Serve one offloaded call and fulfill its promise.
///
/// Errors only when the promise itself could not be fulfilled; a call that failed on every
/// candidate is forwarded to the algorithm as an `Err`.
async fn serve(
    clients: ClientRouter,
    call: CallModel,
    observer: Option<RunObserver>,
    // Output parameter because `drive` takes a function that returns a plain `Result<()>`.
    routed_calls: Arc<Mutex<RoutedCallWindows>>,
) -> Result<()> {
    let result = call_first_available(&clients, &call, &observer, &routed_calls).await;
    call.respond(result)
}

/// Try candidates in order until one succeeds or a failure stops fallback.
async fn call_first_available(
    clients: &ClientRouter,
    call: &CallModel,
    observer: &Option<RunObserver>,
    routed_calls: &Arc<Mutex<RoutedCallWindows>>,
) -> Result<Response> {
    for (index, target) in call.models.iter().enumerate() {
        let request = request_for(&call.request, target);
        match call_one(
            clients,
            target,
            request,
            call,
            observer,
            routed_calls,
            index,
            call.models.len(),
        )
        .await
        {
            Ok(response) => return Ok(response),
            Err(error) if index + 1 == call.models.len() => return Err(error),
            Err(error) => match fallback_reason(&error) {
                Some(reason) => tracing::info!(
                    from = %target,
                    to = %call.models[index + 1],
                    reason = reason.as_str(),
                    "model call failed; trying next candidate"
                ),
                None => return Err(error),
            },
        }
    }
    Err(LibsyError::NoTargets)
}

/// Call one candidate model and record its observation and span.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(
    target = "libsy",
    name = "libsy.client_call",
    skip_all,
    fields(
        algorithm = call.algorithm,
        switchyard.algorithm = call.algorithm,
        switchyard.candidate = index + 1,
        switchyard.candidate_count = count,
        selected_model = %model_id,
        otel.kind = "client",
        otel.name = %format_args!("chat {model_id}"),
        openinference.span.kind = "LLM",
        gen_ai.operation.name = "chat",
        gen_ai.request.model = %model_id,
        gen_ai.request.stream = tracing::field::Empty,
        gen_ai.request.temperature = tracing::field::Empty,
        gen_ai.request.top_p = tracing::field::Empty,
        gen_ai.request.top_k = tracing::field::Empty,
        gen_ai.request.max_tokens = tracing::field::Empty,
        gen_ai.request.reasoning.level = tracing::field::Empty,
        gen_ai.output.type = tracing::field::Empty,
        gen_ai.conversation.id = tracing::field::Empty,
        server.address = tracing::field::Empty,
        server.port = tracing::field::Empty,
        gen_ai.response.id = tracing::field::Empty,
        gen_ai.response.model = tracing::field::Empty,
        gen_ai.usage.input_tokens = tracing::field::Empty,
        gen_ai.usage.output_tokens = tracing::field::Empty,
        gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
        gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
        gen_ai.usage.reasoning.output_tokens = tracing::field::Empty,
        outcome = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
        error.type = tracing::field::Empty,
        error = tracing::field::Empty,
    )
)]
async fn call_one(
    clients: &ClientRouter,
    model_id: &ModelId,
    request: Request,
    call: &CallModel,
    observer: &Option<RunObserver>,
    routed_calls: &Arc<Mutex<RoutedCallWindows>>,
    // index is for span log
    index: usize,
    // count is for span log
    count: usize,
) -> Result<Response> {
    let span = tracing::Span::current();
    observability::record_gen_ai_request(&span, &request.llm_request);
    if let Some(session_id) = call
        .request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.session_id.as_deref())
    {
        span.record("gen_ai.conversation.id", session_id);
    }
    let is_answer_call = call.is_answer_call;
    // Resolved before the clock starts: picking the client is Switchyard's work, not
    // the provider's, so it belongs in the routing overhead.
    let client = clients.route(model_id);
    let started = Instant::now();
    let result = match client {
        Ok(client) => client.call(request).await,
        Err(error) => Err(error),
    }
    .map_err(|source| LibsyError::client_call(model_id.clone(), source));
    let ended = Instant::now();
    let duration = ended - started;

    let result = result.map(|mut response| {
        response.set_served_model(model_id);
        response
    });
    let result = observability::observe_client_call(result);
    if let Some(observer) = observer {
        observer(RunObservation::LlmCall(LlmCallObservation {
            selected_model: model_id.clone(),
            is_answer_call,
            is_success: result.is_ok(),
            duration,
            usage: result
                .as_ref()
                .ok()
                .and_then(|response| response.llm_response.as_agg())
                .map(|response| response.usage.clone()),
        }));
    }
    if is_answer_call && result.is_ok() {
        routed_calls.lock().record(started, ended);
    }

    result
}

/// Whether a failed candidate is worth routing around.
fn fallback_reason(error: &LibsyError) -> Option<RoutingFallbackReason> {
    let LibsyError::ClientCall { source, .. } = error else {
        return None;
    };
    match source {
        LlmClientError::ContextWindowExceeded { .. } => Some(RoutingFallbackReason::ContextWindow),
        LlmClientError::Transport { .. } | LlmClientError::Timeout { .. } => {
            Some(RoutingFallbackReason::Unavailable)
        }
        LlmClientError::UpstreamHttp { status, .. }
            if matches!(*status, 403 | 408 | 429) || (500..=599).contains(status) =>
        {
            Some(RoutingFallbackReason::Unavailable)
        }
        _ => None,
    }
}

/// Clone a request and stamp the candidate model that should receive it.
fn request_for(request: &Request, target: &ModelId) -> Request {
    let mut request = request.clone();
    request.llm_request.model = Some(target.to_string());
    request
}

/// Resolves a routed call's selected model to the client that serves it.
///
/// An algorithm routes among named targets; which provider each target lives on is the
/// host's concern, and two targets in one run may sit on different providers. A router owns
/// that mapping. It is *not* itself a client: it hands back a [`RoutedLlmClient`] and the
/// caller makes the call.
///
/// Cloning is cheap — the mapping is shared, so one router can serve every request.
#[derive(Clone)]
pub struct ClientRouter {
    routing: Arc<Routing>,
}

enum Routing {
    /// One client serves every model.
    Single(Arc<dyn RoutedLlmClient>),
    /// Each model is served by the client configured for it.
    ByModel(HashMap<ModelId, Arc<dyn RoutedLlmClient>>),
}

impl ClientRouter {
    /// Build a router over `model name -> client`, for targets spread across providers.
    pub fn new(by_model: HashMap<ModelId, Arc<dyn RoutedLlmClient>>) -> Self {
        Self {
            routing: Arc::new(Routing::ByModel(by_model)),
        }
    }

    /// A router that serves every model with one client — the single-provider case.
    ///
    /// [`TranslatingLlmClient`](crate::TranslatingLlmClient) already maps model names to
    /// backends internally and rejects ones it does not know, so enumerating them here would
    /// only duplicate that.
    pub fn single(client: Arc<dyn RoutedLlmClient>) -> Self {
        Self {
            routing: Arc::new(Routing::Single(client)),
        }
    }

    /// The client that serves `model`.
    ///
    /// Errors with [`LlmClientError::Configuration`] when the router maps models and has no
    /// entry for this one, rather than silently sending the call to another provider.
    pub fn route(
        &self,
        model: &ModelId,
    ) -> std::result::Result<&Arc<dyn RoutedLlmClient>, LlmClientError> {
        match self.routing.as_ref() {
            Routing::Single(client) => Ok(client),
            Routing::ByModel(by_model) => {
                by_model
                    .get(model)
                    .ok_or_else(|| LlmClientError::Configuration {
                        message: format!("no llm client is configured for model {model:?}"),
                    })
            }
        }
    }
}

impl FromIterator<(ModelId, Arc<dyn RoutedLlmClient>)> for ClientRouter {
    fn from_iter<I: IntoIterator<Item = (ModelId, Arc<dyn RoutedLlmClient>)>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use async_trait::async_trait;
    use futures::StreamExt;
    use switchyard_libsy::Driver;
    use switchyard_protocol::{
        LlmResponse, LlmResponseChunk, LlmResponseStreamEvent, completion_text, text_request,
        text_response,
    };
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};

    struct CandidateAlgorithm {
        models: Vec<ModelId>,
    }

    #[async_trait]
    impl Algorithm for CandidateAlgorithm {
        fn name(&self) -> &str {
            "candidate_test"
        }

        async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
            driver.call_model(request, self.models.clone(), true).await
        }
    }

    #[derive(Clone, Copy)]
    enum FirstOutcome {
        ContextWindow,
        Unauthorized,
        StreamSuccess,
        MidStreamError,
    }

    struct CandidateClient {
        calls: Mutex<Vec<ModelId>>,
        first: FirstOutcome,
    }

    #[async_trait]
    impl RoutedLlmClient for CandidateClient {
        async fn call(&self, request: Request) -> std::result::Result<Response, LlmClientError> {
            let model = request.model_id().unwrap_or_default();
            self.calls.lock().push(model.clone());
            if model == "weak" {
                return match self.first {
                    FirstOutcome::ContextWindow => Err(LlmClientError::ContextWindowExceeded {
                        model,
                        message: "too long".to_string(),
                    }),
                    FirstOutcome::Unauthorized => Err(LlmClientError::UpstreamHttp {
                        status: 401,
                        body: "unauthorized".to_string(),
                    }),
                    FirstOutcome::StreamSuccess => Ok(stream_response(vec![
                        LlmResponseChunk::TextDelta {
                            index: 0,
                            text: "streamed".to_string(),
                        },
                        LlmResponseChunk::MessageStop {
                            reason: Some("stop".to_string()),
                        },
                    ])),
                    FirstOutcome::MidStreamError => Ok(stream_response(vec![
                        LlmResponseChunk::TextDelta {
                            index: 0,
                            text: "partial".to_string(),
                        },
                        LlmResponseChunk::StreamError {
                            message: "stream failed".to_string(),
                        },
                    ])),
                };
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(Some(model.to_string()), model)),
                metadata: None,
            })
        }
    }

    fn stream_response(chunks: Vec<LlmResponseChunk>) -> Response {
        Response {
            llm_response: LlmResponse::Stream(
                futures::stream::iter(
                    chunks
                        .into_iter()
                        .map(|chunk| Ok(LlmResponseStreamEvent::from(chunk))),
                )
                .boxed(),
            ),
            metadata: None,
        }
    }

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hello".to_string()),
            raw_request: None,
            metadata: None,
        }
    }

    async fn run_candidates(
        first: FirstOutcome,
    ) -> (Arc<CandidateClient>, Result<(Vec<Decision>, Response)>) {
        let client = Arc::new(CandidateClient {
            calls: Mutex::new(Vec::new()),
            first,
        });
        let algorithm = Arc::new(CandidateAlgorithm {
            models: vec!["weak".into(), "strong".into()],
        });
        let result = run(
            algorithm,
            ClientRouter::single(client.clone()),
            request(),
            None,
        )
        .await;
        (client, result)
    }

    #[test]
    fn fallback_only_accepts_context_and_unavailable_failures() {
        let error = |source| LibsyError::client_call("target", source);
        assert_eq!(
            fallback_reason(&error(LlmClientError::ContextWindowExceeded {
                model: "target".into(),
                message: "too long".to_string(),
            })),
            Some(RoutingFallbackReason::ContextWindow)
        );
        for status in [403, 408, 429, 500, 599] {
            assert_eq!(
                fallback_reason(&error(LlmClientError::UpstreamHttp {
                    status,
                    body: "failed".to_string(),
                })),
                Some(RoutingFallbackReason::Unavailable)
            );
        }
        for status in [400, 401, 404, 409, 499, 600] {
            assert_eq!(
                fallback_reason(&error(LlmClientError::UpstreamHttp {
                    status,
                    body: "failed".to_string(),
                })),
                None
            );
        }
    }

    #[tokio::test]
    async fn candidate_failures_follow_the_fallback_policy() -> Result<()> {
        // Context overflow is retryable across candidates.
        let (client, result) = run_candidates(FirstOutcome::ContextWindow).await;
        let (_, response) = result?;
        assert_eq!(
            &*client.calls.lock(),
            &[ModelId::from("weak"), "strong".into()]
        );
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(|response| response.model.as_deref()),
            Some(Some("strong"))
        );
        assert_eq!(response.served_model().map(ModelId::as_str), Some("strong"));

        // Authentication failure is not retryable, so the second candidate is untouched.
        let (client, result) = run_candidates(FirstOutcome::Unauthorized).await;
        assert!(matches!(
            result,
            Err(LibsyError::ClientCall {
                source: LlmClientError::UpstreamHttp { status: 401, .. },
                ..
            })
        ));
        assert_eq!(&*client.calls.lock(), &[ModelId::from("weak")]);
        Ok(())
    }

    #[tokio::test]
    async fn retry_budget_is_exhausted_before_falling_through() -> Result<()> {
        let server = MockServer::start().await;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |request: &wiremock::Request| {
                let body: serde_json::Value =
                    serde_json::from_slice(&request.body).unwrap_or(serde_json::Value::Null);
                let model = body["model"].as_str().unwrap_or_default().to_string();
                observed_calls.lock().push(model.clone());
                if model == "weak" {
                    ResponseTemplate::new(503)
                        .insert_header("retry-after", "0")
                        .set_body_string("unavailable")
                } else {
                    ResponseTemplate::new(200).set_body_json(serde_json::json!({
                        "id": "answer",
                        "model": "strong",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"
                        }],
                        "usage": {}
                    }))
                }
            })
            .mount(&server)
            .await;

        let backend = || {
            Backend::OpenAiChat(HttpBackendConfig {
                base_url: format!("{}/v1", server.uri()),
                api_key: None,
                forward_auth: false,
                extra_headers: BTreeMap::new(),
                extra_body: BTreeMap::new(),
                max_retries: 2,
            })
        };
        let client = Arc::new(
            TranslatingLlmClient::new(&[
                ModelConfig::new("weak", backend(), None),
                ModelConfig::new("strong", backend(), None),
            ])
            .map_err(|error| LibsyError::external("building test client", error))?,
        );
        let algorithm = Arc::new(CandidateAlgorithm {
            models: vec!["weak".into(), "strong".into()],
        });
        run(algorithm, ClientRouter::single(client), request(), None).await?;

        assert_eq!(&*calls.lock(), &["weak", "weak", "weak", "strong"]);
        Ok(())
    }

    #[tokio::test]
    async fn streams_are_outside_the_candidate_fallback_boundary() -> Result<()> {
        // Receiving a stream handle is a successful call and ends candidate selection.
        let (client, result) = run_candidates(FirstOutcome::StreamSuccess).await;
        let (_, response) = result?;
        assert_eq!(response.served_model().map(ModelId::as_str), Some("weak"));
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .map_err(|source| LibsyError::client_call("weak", source))?;
        assert_eq!(&*client.calls.lock(), &[ModelId::from("weak")]);
        assert_eq!(completion_text(&aggregate), "streamed");

        // A later in-stream failure surfaces during aggregation without trying another model.
        let (client, result) = run_candidates(FirstOutcome::MidStreamError).await;
        let (_, response) = result?;
        let error = response
            .llm_response
            .into_agg()
            .await
            .expect_err("the stream should fail during aggregation");
        assert!(error.to_string().contains("stream failed"));
        assert_eq!(&*client.calls.lock(), &[ModelId::from("weak")]);
        Ok(())
    }
}
