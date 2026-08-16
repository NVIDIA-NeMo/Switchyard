// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The [`Algorithm`] trait and its [`Driver`] — the orchestration contract every
//! algorithm implements, and the offload channel it uses to make model calls and
//! publish [`Decision`]s.

use std::{future::Future, panic::AssertUnwindSafe, pin::Pin, sync::Arc, time::Instant};

use async_trait::async_trait;
use futures::{FutureExt, Stream, StreamExt};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tracing::Instrument;

/// The request/response protocol types come from [`switchyard_protocol`].
/// [`switchyard_protocol::LlmRequest`] is the normalized request;
/// [`switchyard_protocol::AggLlmResponse`] is the buffered response;
/// [`switchyard_protocol::LlmResponseChunk`] is normalized streaming content;
/// [`switchyard_protocol::LlmResponseStreamEvent`] is its host/algorithm envelope; and
/// [`switchyard_protocol::LlmResponse`] carries either a live
/// [`switchyard_protocol::LlmResponseStream`] or the terminal aggregate.
use switchyard_protocol::{Decision, ModelId, Request, Response};

use crate::{DriverError, LibsyError, Result, observability};

/// A boxed, `Send` stream of [`Step`]s — the output of
/// [`Algorithm::run_stream`]. Boxed so the trait method that produces it keeps
/// `Arc<dyn Algorithm>` object-safe.
pub type StepStream = Pin<Box<dyn Stream<Item = Result<Step>> + Send>>;

/// An offloaded model call, surfaced inside [`Step::CallModel`].
///
/// The host reads the public fields, performs (or delegates) the model call, and fulfills it
/// with [`respond`](Self::respond) — unblocking the algorithm's [`Driver::call_model`] on the
/// other side. `switchyard-llm-client`'s `run` is the ready-made consumer that does this for
/// you. A host that only wants the routing outcome can take the contents with
/// [`into_parts`](Self::into_parts) and never respond; dropping the stream ends the run.
///
/// [`Driver::call_model`] stamps the first candidate model onto the request before publishing
/// the call. A consumer that falls through to a later candidate must re-stamp it.
pub struct CallModel {
    /// The name of the algorithm that produced this call, so a host instrumenting the
    /// calls it serves can attribute its own spans to the algorithm behind them.
    pub algorithm: String,
    /// The request to serve; its `model` is stamped with the first candidate.
    pub request: Request,
    /// Candidate models, tried in order until one answers. Never empty.
    pub models: Vec<ModelId>,
    /// True for an answer-generating call, false for classifier and judge calls.
    pub is_answer_call: bool,
    // How to send the response back to the algorithm
    reply: oneshot::Sender<Result<Response>>,
}

impl CallModel {
    /// Fulfill the promise with the caller's model-call result. Pass `Err(..)` to
    /// propagate a failed model call back to the algorithm. Consumes the promise: it
    /// can only be fulfilled once.
    pub fn respond(self, result: Result<Response>) -> Result<()> {
        self.reply
            .send(result)
            .map_err(|_| DriverError::ResponseDropped.into())
    }

    /// Take the call's contents without answering it, dropping the promise — the routing
    /// outcome plus the request as the algorithm would have sent it, after any rewriting.
    ///
    /// Should only be called if `is_answer_call` is true as that is the final call.
    /// The algorithm's [`Driver::call_model`] will fail with [`DriverError::Abandoned`] and
    /// the run ends there. Taking a call the algorithm does not depend on (a judge or
    /// classifier call) may instead let it fail open and complete with degraded routing.
    ///
    /// An abandoned run is not recorded as a failed one. Dropping a [`CallModel`] without
    /// calling this still yields [`DriverError::ResponseDropped`], which is.
    pub fn into_parts(self) -> (Request, Vec<ModelId>) {
        let Self {
            request,
            models,
            reply,
            ..
        } = self;
        // Tell the algorithm the call was taken deliberately rather than lost, so its
        // telemetry can tell an abandoned run from a failed one. The receiver is already
        // gone if the algorithm stopped waiting, which is fine.
        let _ = reply.send(Err(DriverError::Abandoned.into()));
        (request, models)
    }
}

/// How an algorithm's [`route`](Algorithm::route) makes model calls.
#[derive(Clone)]
pub struct Driver {
    step_tx: mpsc::Sender<Result<Step>>,
    /// The owning algorithm's telemetry label, stamped onto every call and decision
    /// this driver publishes.
    algorithm: String,
}

impl Driver {
    /// Build an empty driver with its step channel ready. Created per call by
    /// [`run_stream`](Algorithm::run_stream). Also returns the Step receiver.
    pub(crate) fn new(algorithm: &str) -> (Self, mpsc::Receiver<Result<Step>>) {
        // Capacity one keeps the algorithm paced by the stream consumer. It limits queued steps,
        // not model calls already pulled from the stream, which can still run at the same time.
        // A larger buffer would use more memory and let the algorithm run farther ahead with
        // little benefit because reading a step is cheap compared with serving a model call.
        let (step_tx, step_rx) = mpsc::channel(1);
        (
            Self {
                step_tx,
                algorithm: algorithm.to_string(),
            },
            step_rx,
        )
    }

    /// Publish a model call and await the consumer's response.
    ///
    /// Errors if the stream is closed or the call failed.
    /// The await is wrapped in a `libsy.llm_call` span measuring *fulfillment* as
    /// the algorithm observes it (host queueing/serving included; a streamed
    /// response resolves when its stream handle arrives); latency, outcome, and
    /// token usage are recorded when it resolves. The provider call itself is the
    /// host's, and is instrumented by whoever makes it.
    #[tracing::instrument(
        target = "libsy",
        name = "libsy.llm_call",
        skip_all,
        fields(
            algorithm = self.algorithm,
            selected_model = %models.first().map(ModelId::as_str).unwrap_or("NoTargets"),
            openinference.span.kind = "CHAIN",
            outcome = tracing::field::Empty,
            error = tracing::field::Empty,
            input_tokens = tracing::field::Empty,
            output_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            reasoning_tokens = tracing::field::Empty,
        )
    )]
    pub async fn call_model(
        &self,
        mut request: Request,
        models: Vec<ModelId>,
        is_answer_call: bool,
    ) -> Result<Response> {
        let Some(selected_model_id) = models.first().cloned() else {
            return Err(LibsyError::NoTargets);
        };
        request.llm_request.model = Some(selected_model_id.to_string());
        let started = Instant::now();
        let (reply, response) = oneshot::channel::<Result<Response>>();
        let call = CallModel {
            algorithm: self.algorithm.clone(),
            request,
            models,
            is_answer_call,
            reply,
        };
        let result = async {
            self.step_tx
                .send(Ok(Step::CallModel(Box::new(call))))
                .await
                .map_err(|_| DriverError::StreamClosed)?;
            response
                .await
                .map_err(|_| LibsyError::from(DriverError::ResponseDropped))?
        }
        .await;
        let elapsed = started.elapsed();
        observability::record_llm_call(
            &self.algorithm,
            selected_model_id.as_str(),
            is_answer_call,
            elapsed,
            &result,
            &tracing::Span::current(),
        );
        result
    }

    /// Publish a routing [`Decision`] as a [`Step::Decision`] on the stream.
    /// Each successfully published decision is counted and logged; a decision
    /// the stream never accepted is not recorded.
    pub async fn decide(&self, decision: Decision) -> Result<()> {
        self.step_tx
            .send(Ok(Step::Decision(decision.clone())))
            .await
            .map_err(|_| DriverError::StreamClosed)?;
        observability::record_decision(&self.algorithm, &decision);
        Ok(())
    }

    /// Emit the terminal step: [`Step::Done`] on `Ok`, or an `Err` stream
    /// item on failure. Internal: called once by [`run_stream`](Algorithm::run_stream)
    /// when the algorithm finishes.
    pub(crate) async fn finish(&self, result: Result<Response>) -> Result<()> {
        let step = result.map(|response| Step::Done(Box::new(response)));
        self.step_tx
            .send(step)
            .await
            .map_err(|_| DriverError::StreamClosed.into())
    }
}

/// One item in the stream returned by [`Algorithm::run_stream`].
pub enum Step {
    /// The algorithm needs this model call performed. The host serves it and fulfills
    /// it with [`CallModel::respond`]. Boxed: it is by far the largest variant.
    CallModel(Box<CallModel>),
    /// A routing decision the algorithm made, published via [`Driver::decide`] as it
    /// happens (rather than collected into a trace returned at the end).
    Decision(Decision),
    /// The algorithm finished with its final response — the last step of a run.
    Done(Box<Response>),
}

/// Drive [`Algorithm::run_stream`] to completion, handing each offloaded call to `serve`.
///
/// Returns the final [`Response`] and the trace of [`Decision`]s the algorithm published.
/// `serve` owns the call: it performs it however the host likes and must fulfill the promise
/// with [`CallModel::respond`]. A failed *model* call belongs in `respond` — the
/// algorithm may route around it. Returning `Err` from `serve` aborts the whole run, so
/// reserve it for infrastructure failures. Calls are served concurrently, so an algorithm
/// that offloads several at once (hedging, fan-out) gets real parallelism.
///
/// libsy performs no I/O; this is only the mechanics of consuming its own step stream, kept
/// here so every host does not reimplement the same loop. `switchyard-llm-client`'s `run`
/// is this function plus an HTTP client.
pub async fn drive<F, Fut>(
    algorithm: Arc<dyn Algorithm>,
    request: Request,
    serve: F,
) -> Result<(Vec<Decision>, Response)>
where
    F: Fn(CallModel) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    let stream = algorithm.run_stream(request);
    tokio::pin!(stream);

    let mut trace: Vec<Decision> = Vec::new();
    let mut in_flight = futures::stream::FuturesUnordered::new();
    let mut final_response: Option<Response> = None;

    loop {
        tokio::select! {
            Some(result) = in_flight.next() => match result {
                Ok(()) => {}, // CallModel completed successfully
                Err(err) => return Err(err), // CallModel failed, propagate the error
            },
            step = stream.next() => {
                match step {
                    None => break, // stream has ended, no more steps
                    Some(item) => match item? {
                        Step::CallModel(call) => in_flight.push(serve(*call)),
                        Step::Decision(decision) => trace.push(decision),
                        Step::Done(response) => {
                            final_response = Some(*response);
                            break;
                        }
                    }
                }
            },
        }
    }
    final_response
        .map(|response| (trace, response))
        .ok_or(LibsyError::MissingFinalResponse)
}

/// Recover the message from an algorithm's panic.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&'static str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".to_string())
}

/// Abort guard
struct AbortOnDrop(tokio::task::AbortHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Errors unless `targets` contains `name`.
///
/// Config target names must be resolved before an algorithm is built. This list contains
/// model IDs, not target names.
pub(crate) fn ensure_model_is_target(targets: &[ModelId], name: &ModelId) -> Result<()> {
    targets
        .iter()
        .any(|target| target == name)
        .then_some(())
        .ok_or_else(|| LibsyError::TargetNotFound {
            target: name.clone(),
        })
}

/// Key for routing affinity: a root request by its session, a child request by its session
/// and agent.
#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) enum RoutingIdentity {
    /// Root request, keyed by session ID.
    Session(String),
    /// Child request, keyed by session and agent IDs.
    Subagent { session: String, agent: String },
}

impl RoutingIdentity {
    /// Builds a root or child identity from non-empty request metadata.
    ///
    /// A child request missing either ID returns `None`, so it keeps no routing history
    /// rather than sharing the parent's.
    pub(crate) fn from_request(request: &Request) -> Option<Self> {
        let metadata = request.metadata.as_ref()?;
        let session = metadata.session_id.as_deref().filter(|id| !id.is_empty())?;
        if metadata.is_subagent {
            let agent = metadata.agent_id.as_deref().filter(|id| !id.is_empty())?;
            Some(Self::Subagent {
                session: session.to_string(),
                agent: agent.to_string(),
            })
        } else {
            Some(Self::Session(session.to_string()))
        }
    }
}

/// An optimization strategy. Implement [`route`](Self::route);
/// callers drive it with [`run_stream`](Self::run_stream), serving each [`Step::CallModel`]
/// it emits. `switchyard-llm-client`'s `run` is the ready-made consumer that does this
/// over HTTP.
///
/// Methods take `self: Arc<Self>`: one algorithm (`Arc<dyn Algorithm>`) is shared across
/// requests and run concurrently, so it owns its thread-safety and any shared state.
///
/// # Concurrency
///
/// A host may run the same algorithm concurrently for many requests. Implementations
/// must synchronize their own mutable shared state. Each call to [`run_stream`](Self::run_stream)
/// creates an independent [`Driver`], so model-call promises and emitted [`Step`]s cannot
/// cross between runs.
///
/// # Observability
///
/// [`run_stream`](Self::run_stream) creates a `libsy.run` span, and each offloaded model
/// call creates a `libsy.llm_call` span. Decisions and failures are emitted through
/// `tracing`; metrics use the global OpenTelemetry meter provider. The provider call
/// itself belongs to the host, and is instrumented by whoever makes it.
#[async_trait]
pub trait Algorithm: Send + Sync + 'static {
    /// Stable, low-cardinality name identifying this algorithm — the
    /// `algorithm` attribute on every span, metric, and log line the crate
    /// emits for its runs.
    fn name(&self) -> &str;

    /// Run one request to completion: make model calls with [`Driver::call_model`],
    /// publish [`Decision`]s with [`Driver::decide`], and return the final [`Response`].
    /// The method an algorithm implements; [`run_stream`](Self::run_stream) drives it.
    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response>;

    /// Process a request to completion, returning a stream of [`Step`]s.
    ///
    /// The consumer must fulfill every [`Step::CallModel`] before the algorithm can
    /// continue. Every run ends with exactly one terminal item — [`Step::Done`] on
    /// success, an `Err` item on failure, including when the algorithm panics. Dropping
    /// the stream aborts the spawned algorithm task.
    ///
    /// Every invocation owns a separate [`Driver`].
    fn run_stream(self: Arc<Self>, request: Request) -> StepStream {
        let (driver, step_rx) = Driver::new(self.name());
        let span = observability::run_span(self.name(), &request);
        let handle = tokio::spawn(
            async move {
                let algorithm = self.name().to_string();
                // Catch a panicking algorithm so the run still publishes a terminal step.
                let route = AssertUnwindSafe(self.route(driver.clone(), request)).catch_unwind();
                let result = observability::observe_run(&algorithm, async move {
                    route.await.unwrap_or_else(|payload| {
                        Err(LibsyError::AlgorithmError {
                            message: format!(
                                "algorithm task panicked: {}",
                                panic_message(payload.as_ref())
                            ),
                        })
                    })
                })
                .await;

                let _ = driver.finish(result).await;
            }
            .instrument(span),
        );
        // Dropping the stream aborts the algorithm task when its consumer goes away.
        let abort_guard = AbortOnDrop(handle.abort_handle());
        Box::pin(ReceiverStream::new(step_rx).map(move |step| {
            // link abort guard to stream
            let _keep_alive = &abort_guard;
            step
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::testing::{Serve, ServeResult, echo, reply, test_drive};
    use futures::StreamExt;
    use switchyard_protocol::{
        LlmResponse, LlmResponseChunk, completion_text, text_request, text_response,
    };

    #[derive(Debug, thiserror::Error)]
    #[error("{0}")]
    struct TestError(&'static str);

    fn test_error(message: &'static str) -> LibsyError {
        LibsyError::external("test", TestError(message))
    }

    /// Build a routed decision for orchestration tests.
    fn test_decision(selected_model_id: ModelId) -> Decision {
        Decision::new(selected_model_id, true)
    }

    /// Trivial algo used only to exercise the orchestrator: calls the first target
    /// and returns its response with a one-item trace.
    struct TestAlgo {
        target_set: Vec<ModelId>,
    }

    #[async_trait]
    impl Algorithm for TestAlgo {
        fn name(&self) -> &str {
            "test"
        }

        async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
            let target = self
                .target_set
                .first()
                .ok_or(LibsyError::NoTargets)?
                .clone();
            let decision = test_decision(target.clone());
            driver.decide(decision.clone()).await?;
            driver.call_model(request, vec![target], true).await
        }
    }

    /// Build a shared `TestAlgo` over the given target set.
    fn orch(target_set: Vec<ModelId>) -> Arc<dyn Algorithm> {
        Arc::new(TestAlgo { target_set })
    }

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi".to_string()),
            raw_request: None,
            metadata: None,
        }
    }

    fn target_set(names: &[&str]) -> Vec<ModelId> {
        names.iter().map(|name| ModelId::from(*name)).collect()
    }

    #[tokio::test]
    async fn typed_driver_preserves_call_and_stream_boundaries() -> Result<()> {
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            // Distinct oneshots keep reverse-order replies paired with their producers, and a
            // retained call remains pending until the host responds.
            let (driver, mut step_rx) = Driver::new("test");
            let first_driver = driver.clone();
            let mut first = tokio::spawn(async move {
                first_driver
                    .call_model(request(), vec![ModelId::from("first")], true)
                    .await
            });
            let second = tokio::spawn(async move {
                driver
                    .call_model(request(), vec![ModelId::from("second")], true)
                    .await
            });

            let mut calls = HashMap::new();
            for _ in 0..2 {
                let step = step_rx.recv().await.ok_or(DriverError::StreamClosed)??;
                let Step::CallModel(call) = step else {
                    return Err(test_error("expected a CallModel step"));
                };
                let selected_model = call
                    .models
                    .first()
                    .ok_or_else(|| test_error("model call has no candidates"))?
                    .to_string();
                calls.insert(selected_model, call);
            }
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(20), &mut first)
                    .await
                    .is_err(),
                "call completed before the host responded"
            );
            calls
                .remove("second")
                .ok_or_else(|| test_error("missing second call"))?
                .respond(Ok(reply("second response")))?;
            calls
                .remove("first")
                .ok_or_else(|| test_error("missing first call"))?
                .respond(Ok(reply("first response")))?;

            let first_response = first
                .await
                .map_err(|source| LibsyError::external("joining a test task", source))??;
            let second_response = second
                .await
                .map_err(|source| LibsyError::external("joining a test task", source))??;
            assert_eq!(
                first_response.llm_response.as_agg().map(completion_text),
                Some("first response".to_string())
            );
            assert_eq!(
                second_response.llm_response.as_agg().map(completion_text),
                Some("second response".to_string())
            );

            // Dropping the host-facing promise closes only that call's reply channel.
            let (driver, mut step_rx) = Driver::new("test");
            let producer = tokio::spawn(async move {
                driver
                    .call_model(request(), vec![ModelId::from("dropped")], true)
                    .await
            });
            let step = step_rx.recv().await.ok_or(DriverError::StreamClosed)??;
            let Step::CallModel(call) = step else {
                return Err(test_error("expected a CallModel step"));
            };
            drop(call);
            let result = producer
                .await
                .map_err(|source| LibsyError::external("joining a test task", source))?;
            assert!(matches!(
                result,
                Err(LibsyError::Driver(DriverError::ResponseDropped))
            ));

            // A standalone driver reports the typed step receiver disappearing at its next send.
            let (driver, step_rx) = Driver::new("test");
            drop(step_rx);
            let decision = test_decision(ModelId::from("closed"));
            let result = driver.decide(decision).await;
            assert!(matches!(
                result,
                Err(LibsyError::Driver(DriverError::StreamClosed))
            ));
            Ok(())
        })
        .await
        .map_err(|error| LibsyError::external("waiting for typed driver boundaries", error))?
    }

    /// A consumer that only wants the routing outcome takes the call apart instead of
    /// answering it: it gets the request as the algorithm would have sent it, and the
    /// algorithm learns the call was abandoned rather than lost — the distinction that
    /// keeps a deliberate decision-only run out of the failure counters.
    #[tokio::test]
    async fn into_parts_yields_the_selected_model_without_answering_it() -> Result<()> {
        let (driver, mut step_rx) = Driver::new("test");
        let producer = tokio::spawn(async move {
            driver
                .call_model(request(), vec![ModelId::from("answer/model")], true)
                .await
        });

        let step = step_rx.recv().await.ok_or(DriverError::StreamClosed)??;
        let Step::CallModel(call) = step else {
            return Err(test_error("expected a CallModel step"));
        };
        let (taken_request, taken_models) = call.into_parts();
        assert_eq!(taken_models, vec![ModelId::from("answer/model")]);
        assert_eq!(
            taken_request.llm_request.model.as_deref(),
            Some("answer/model")
        );
        let mut expected = request().llm_request;
        expected.model = Some("answer/model".to_string());
        assert_eq!(taken_request.llm_request, expected);

        let result = producer
            .await
            .map_err(|source| LibsyError::external("joining a test task", source))?;
        assert!(matches!(
            result,
            Err(LibsyError::Driver(DriverError::Abandoned))
        ));
        Ok(())
    }

    /// The two ways a call goes unanswered are told apart: `into_parts` is the consumer's
    /// choice, a bare drop is the promise being lost.
    #[test]
    fn abandoning_and_dropping_a_call_report_different_outcomes() {
        let abandoned: Result<Response> = Err(DriverError::Abandoned.into());
        let dropped: Result<Response> = Err(DriverError::ResponseDropped.into());
        assert_eq!(observability::outcome_value(&abandoned), "abandoned");
        assert_eq!(observability::outcome_value(&dropped), "error");
        assert!(observability::is_abandoned(&abandoned));
        assert!(!observability::is_abandoned(&dropped));
    }

    #[test]
    fn target_lookup_returns_the_missing_target() {
        let error = ensure_model_is_target(&target_set(&[]), &ModelId::from("missing")).err();
        assert!(matches!(
            error,
            Some(LibsyError::TargetNotFound { target }) if target == "missing"
        ));
    }

    /// Build a single-target algo, plus a `serve` that answers it as a token stream
    /// replaying `chunks` in order (as `Ok` items).
    fn streaming_orch(chunks: Vec<LlmResponseChunk>) -> (Arc<dyn Algorithm>, impl Serve) {
        let algo = orch(target_set(&["stream/model"]));
        let serve = move |_target: ModelId, _request: Request| {
            let chunks = chunks.clone();
            async move {
                let stream =
                    futures::stream::iter(chunks.into_iter().map(|chunk| Ok(chunk.into()))).boxed();
                Ok(Response {
                    llm_response: LlmResponse::Stream(stream),
                    metadata: None,
                })
            }
        };
        (algo, serve)
    }

    #[tokio::test]
    async fn run_returns_a_streamed_response_the_caller_aggregates() -> Result<()> {
        // A streaming client -> its chunks flow through the promise and `Done`,
        // and `run` returns the live stream untouched for the caller to fold.
        let (orch, serve) = streaming_orch(vec![
            LlmResponseChunk::MessageStart {
                id: Some("m1".to_string()),
                model: Some("stream/model".to_string()),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "hel".to_string(),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "lo".to_string(),
            },
            LlmResponseChunk::MessageStop {
                reason: Some("stop".to_string()),
            },
        ]);
        let (trace, response) = test_drive(orch, request(), serve).await?;
        // The run handed back the live stream; the caller folds it to a buffered aggregate.
        let agg = response
            .llm_response
            .into_agg()
            .await
            .map_err(|error| LibsyError::external("aggregating response stream", error))?;
        assert_eq!(completion_text(&agg), "hello");
        assert_eq!(agg.model.as_deref(), Some("stream/model"));
        assert_eq!(trace.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn aggregating_a_streamed_response_propagates_a_mid_stream_error() -> Result<()> {
        // The run succeeds and returns the stream; the in-band `Error` chunk surfaces only
        // when the caller aggregates it.
        let (orch, serve) = streaming_orch(vec![
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "partial".to_string(),
            },
            LlmResponseChunk::StreamError {
                message: "upstream exploded".to_string(),
            },
        ]);
        let (_, response) = test_drive(orch, request(), serve).await?;
        match response.llm_response.into_agg().await {
            Ok(_) => panic!("expected a mid-stream error, got an aggregate"),
            Err(err) => {
                assert!(err.to_string().contains("upstream exploded"));
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn run_offloads_via_promise_then_finishes() -> Result<()> {
        // Every call is offloaded via a promise the orchestrator surfaces as a
        // `CallModel` step for us to fulfill.
        let stream = orch(target_set(&["offload/model"])).run_stream(request());
        tokio::pin!(stream);

        let mut saw_call = false;
        let mut final_completion = None;
        while let Some(step) = stream.next().await {
            match step? {
                Step::CallModel(call) => {
                    saw_call = true;
                    assert_eq!(call.models, vec![ModelId::from("offload/model")]);
                    // Fulfilling the promise is the "real" model call the caller makes.
                    call.respond(Ok(Response {
                        llm_response: LlmResponse::Agg(text_response(
                            None,
                            "fulfilled".to_string(),
                        )),
                        metadata: None,
                    }))?;
                }
                Step::Decision(decision) => {
                    assert_eq!(decision.selected_model_id(), "offload/model");
                }
                Step::Done(response) => {
                    final_completion = Some(
                        response
                            .llm_response
                            .as_agg()
                            .map(completion_text)
                            .unwrap_or_default(),
                    );
                }
            }
        }

        assert!(saw_call, "expected a CallModel step before Done");
        assert_eq!(
            final_completion.ok_or_else(|| test_error("no Done step"))?,
            "fulfilled"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_driven_run_returns_the_trace_and_the_final_response() -> Result<()> {
        let (trace, response) =
            test_drive(orch(target_set(&["direct/model"])), request(), echo()).await?;
        // TestAlgo calls the first target; `echo` answers with its name.
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "direct/model"
        );
        assert_eq!(trace[0].selected_model_id(), "direct/model");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 12)]
    async fn requests_are_processed_in_parallel() -> Result<()> {
        use std::time::Duration;
        use tokio::sync::Barrier;

        const N: usize = 12;

        // Serving blocks until all N concurrent calls have arrived. If requests were
        // serialized (one algorithm behind a `Mutex`), only one call could be in flight,
        // the barrier would never reach N, and the test would time out. It passes only
        // because the shared algorithm is driven concurrently across requests.
        let barrier = Arc::new(Barrier::new(N));
        // One shared algorithm driven by many concurrent requests.
        let algo = orch(target_set(&["m"]));

        let mut handles = Vec::new();
        for _ in 0..N {
            let algo = algo.clone();
            let barrier = barrier.clone();
            let serve = move |target: ModelId, _request: Request| {
                let barrier = barrier.clone();
                async move {
                    barrier.wait().await;
                    Ok(reply(target))
                }
            };
            handles.push(tokio::spawn(async move {
                test_drive(algo, request(), serve)
                    .await
                    .map(|(_, response)| {
                        response
                            .llm_response
                            .as_agg()
                            .map(completion_text)
                            .unwrap_or_default()
                    })
            }));
        }

        for handle in handles {
            // The timeout turns a serialization deadlock into a failure, not a hang.
            let completion = tokio::time::timeout(Duration::from_secs(5), handle)
                .await
                .map_err(|error| LibsyError::external("waiting for test task", error))?
                .map_err(|source| LibsyError::external("joining a test task", source))??;
            assert_eq!(completion, "m");
        }
        Ok(())
    }

    #[tokio::test]
    async fn offload_error_propagates_back_to_the_algorithm() -> Result<()> {
        // A client-less target offloads its call; we fulfill the promise with an
        // Err, which must flow back through `call_model_target` into the algorithm and
        // out as an error step — not a response.
        let stream = orch(target_set(&["offload/model"])).run_stream(request());
        tokio::pin!(stream);

        let mut saw_error = false;
        while let Some(step) = stream.next().await {
            match step {
                Ok(Step::CallModel(call)) => {
                    call.respond(Err(test_error("upstream model call failed")))?;
                }
                Ok(Step::Decision(_)) => {}
                Ok(Step::Done(..)) => {
                    return Err(test_error(
                        "expected the offload error to propagate, got a response",
                    ));
                }
                Err(err) => {
                    // The algorithm's `call_model_target` saw the error via the promise.
                    assert!(err.to_string().contains("upstream model call failed"));
                    saw_error = true;
                }
            }
        }

        assert!(saw_error, "expected an error step");
        Ok(())
    }

    #[tokio::test]
    async fn dropping_the_stream_cancels_the_algorithm_task() -> Result<()> {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::Duration;
        use tokio::sync::mpsc;

        // Sets a flag when dropped, so we can observe whether the algorithm task was
        // cancelled/dropped.
        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        struct StuckAlgo {
            started: mpsc::UnboundedSender<()>,
            dropped: Arc<AtomicBool>,
        }

        #[async_trait]
        impl Algorithm for StuckAlgo {
            fn name(&self) -> &str {
                "stuck"
            }

            async fn route(
                self: Arc<Self>,
                _driver: Driver,
                _request: Request,
            ) -> Result<Response> {
                let _guard = DropGuard(self.dropped.clone());
                let _ = self.started.send(());
                // Await forever without ever touching the driver.
                std::future::pending::<()>().await;
                unreachable!()
            }
        }

        let (started_tx, mut started_rx) = mpsc::unbounded_channel();
        let dropped = Arc::new(AtomicBool::new(false));
        let algo: Arc<dyn Algorithm> = Arc::new(StuckAlgo {
            started: started_tx,
            dropped: dropped.clone(),
        });

        let stream = algo.run_stream(request());
        started_rx
            .recv()
            .await
            .ok_or_else(|| test_error("task never started"))?;
        drop(stream);
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(
            dropped.load(Ordering::SeqCst),
            "algorithm task was NOT cancelled after dropping the stream"
        );
        Ok(())
    }

    #[tokio::test]
    async fn route_panic_surfaces_as_a_stream_error() -> Result<()> {
        // An algorithm whose task panics must surface an `Err` step carrying the panic
        // message, not abort the process from an unobserved detached task.
        struct Panicky;

        #[async_trait]
        impl Algorithm for Panicky {
            fn name(&self) -> &str {
                "panicky"
            }

            async fn route(
                self: Arc<Self>,
                _driver: Driver,
                _request: Request,
            ) -> Result<Response> {
                panic!("boom");
            }
        }

        let algo: Arc<dyn Algorithm> = Arc::new(Panicky);
        let stream = algo.run_stream(request());
        tokio::pin!(stream);

        let mut saw_error = false;
        while let Some(step) = stream.next().await {
            match step {
                Err(err) => {
                    // The panic message is preserved, not flattened into an opaque failure.
                    assert!(err.to_string().contains("algorithm task panicked: boom"));
                    saw_error = true;
                }
                Ok(_) => return Err(test_error("expected the panic to surface as an error step")),
            }
        }

        assert!(saw_error, "expected an error step from the panicked task");
        Ok(())
    }

    /// A panicking algorithm must publish its terminal step even when it left a `Driver`
    /// clone alive in another task. That clone holds the step channel open, so a run that
    /// merely unwound would never terminate and the consumer would wait forever.
    #[tokio::test]
    async fn a_panic_with_a_leaked_driver_clone_still_terminates_the_run() -> Result<()> {
        struct LeakyPanic;

        #[async_trait]
        impl Algorithm for LeakyPanic {
            fn name(&self) -> &str {
                "leaky_panic"
            }

            async fn route(self: Arc<Self>, driver: Driver, _request: Request) -> Result<Response> {
                tokio::spawn(async move {
                    // Outlives the panic below, keeping a sender clone alive.
                    let _keep_alive = driver;
                    std::future::pending::<()>().await;
                });
                tokio::task::yield_now().await;
                panic!("boom");
            }
        }

        let algo: Arc<dyn Algorithm> = Arc::new(LeakyPanic);
        // The timeout turns the hang this guards against into a failure rather than a hang.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            test_drive(algo, request(), echo()),
        )
        .await
        .map_err(|error| LibsyError::external("waiting for the panicked run to end", error))?;

        match result {
            Ok(_) => Err(test_error(
                "expected the panic to end the run with an error",
            )),
            Err(err) => {
                assert!(err.to_string().contains("algorithm task panicked: boom"));
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn cancelling_run_cancels_the_algorithm_task() -> Result<()> {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::Duration;
        use tokio::sync::mpsc;

        // Sets a flag when dropped, so we can observe whether the algorithm task was
        // cancelled once the `run` future driving it is dropped.
        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        struct StuckAlgo {
            started: mpsc::UnboundedSender<()>,
            dropped: Arc<AtomicBool>,
        }

        #[async_trait]
        impl Algorithm for StuckAlgo {
            fn name(&self) -> &str {
                "stuck"
            }

            async fn route(
                self: Arc<Self>,
                _driver: Driver,
                _request: Request,
            ) -> Result<Response> {
                let _guard = DropGuard(self.dropped.clone());
                let _ = self.started.send(());
                // Hang forever without ever touching the driver, so only cancellation
                // (not a dropped step channel) can stop this task.
                std::future::pending::<()>().await;
                unreachable!()
            }
        }

        let (started_tx, mut started_rx) = mpsc::unbounded_channel();
        let dropped = Arc::new(AtomicBool::new(false));
        let algo: Arc<dyn Algorithm> = Arc::new(StuckAlgo {
            started: started_tx,
            dropped: dropped.clone(),
        });

        // Drive the run on its own task, wait until the algorithm task is up, then cancel
        // it — dropping its future (and the `run_stream` stream it holds).
        let run_task = tokio::spawn(async move { test_drive(algo, request(), echo()).await });
        started_rx
            .recv()
            .await
            .ok_or_else(|| test_error("task never started"))?;
        run_task.abort();
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(
            dropped.load(Ordering::SeqCst),
            "algorithm task was NOT cancelled after cancelling run"
        );
        Ok(())
    }

    // --- first-wins hedging: `run` must not wait on losing speculative calls -------------

    /// Offloads two targets concurrently and returns the first to resolve, dropping the
    /// loser's call (first-wins hedging).
    struct Hedge {
        winner: String,
        loser: String,
    }

    #[async_trait]
    impl Algorithm for Hedge {
        fn name(&self) -> &str {
            "hedge"
        }

        async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
            let win = driver.call_model(request.clone(), vec![self.winner.clone().into()], true);
            let lose = driver.call_model(request, vec![self.loser.clone().into()], true);
            // First to resolve wins; `select!` drops the losing future (and its promise).
            tokio::select! {
                res = win => res,
                res = lose => res,
            }
        }
    }

    /// Builds a hedging algo and the `serve` that drives it: the winner is gated behind the
    /// loser starting (so the loser's serve is guaranteed in flight when the winner wins),
    /// and the loser finishes after `loser_delay` — or never, when `None`.
    fn hedge(loser_delay: Option<std::time::Duration>) -> (Arc<dyn Algorithm>, impl Serve) {
        let started = Arc::new(tokio::sync::Notify::new());
        let algo = Arc::new(Hedge {
            winner: "winner".to_string(),
            loser: "loser".to_string(),
        });
        let serve = move |target: ModelId, _request: Request| {
            let started = started.clone();
            async move {
                if target == "loser" {
                    started.notify_one();
                    match loser_delay {
                        Some(delay) => tokio::time::sleep(delay).await,
                        None => std::future::pending::<()>().await,
                    }
                } else {
                    started.notified().await;
                }
                Ok(reply(target))
            }
        };
        (algo, serve)
    }

    #[tokio::test]
    async fn run_returns_the_winner_without_a_late_loser_overwriting_it() -> Result<()> {
        // The loser responds 50ms after the winner has already won. `run` must return the
        // winner, not the loser's `respond`-to-a-dropped-receiver error.
        let (algo, serve) = hedge(Some(std::time::Duration::from_millis(50)));
        let (_trace, response) = test_drive(algo, request(), serve).await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "winner"
        );
        Ok(())
    }

    #[tokio::test]
    async fn run_returns_the_winner_without_hanging_on_a_pending_loser() -> Result<()> {
        // The loser never resolves. `run` must return the winner promptly, not hang
        // waiting for the in-flight loser.
        let (algo, serve) = hedge(None);
        let run = test_drive(algo, request(), serve);
        let (_trace, response) = tokio::time::timeout(std::time::Duration::from_secs(1), run)
            .await
            .map_err(|error| LibsyError::external("waiting for pending loser", error))??;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "winner"
        );
        Ok(())
    }

    #[tokio::test]
    async fn run_surfaces_a_terminal_error_with_many_calls_in_flight() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // A large fan-out (10 matched the old, now-removed concurrency cap). The terminal
        // error must still reach the caller with all of these calls pending.
        const N: usize = 10;

        // Fans out N calls, then errors as soon as all N are in flight — exercising a
        // terminal failure emitted while the offloaded calls are still pending.
        struct FanOutThenError {
            all_started: Arc<tokio::sync::Notify>,
            n: usize,
        }

        #[async_trait]
        impl Algorithm for FanOutThenError {
            fn name(&self) -> &str {
                "fan_out_then_error"
            }

            async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
                let offloads = futures::future::join_all((0..self.n).map(|i| {
                    driver.call_model(request.clone(), vec![format!("m{i}").into()], true)
                }));
                tokio::select! {
                    _ = offloads => Err(test_error("offloads unexpectedly completed")),
                    _ = self.all_started.notified() => {
                        Err(test_error("terminal error while calls pending"))
                    }
                }
            }
        }

        let all_started = Arc::new(tokio::sync::Notify::new());
        let algo: Arc<dyn Algorithm> = Arc::new(FanOutThenError {
            all_started: all_started.clone(),
            n: N,
        });

        // Serving enters each call; once all N are in flight it signals, then pends forever.
        let started = Arc::new(AtomicUsize::new(0));
        let serve = move |_target: ModelId, _request: Request| {
            let started = started.clone();
            let all_started = all_started.clone();
            async move {
                if started.fetch_add(1, Ordering::SeqCst) + 1 == N {
                    all_started.notify_one();
                }
                std::future::pending::<ServeResult>().await
            }
        };

        // With the cap gone, the driver keeps polling the stream even with N calls in
        // flight, so the terminal error surfaces promptly instead of hanging.
        let run = test_drive(algo, request(), serve);
        let result = tokio::time::timeout(std::time::Duration::from_millis(500), run)
            .await
            .map_err(|error| {
                LibsyError::external("waiting for terminal error with full call cap", error)
            })?;
        match result {
            Ok(_) => Err(test_error("expected the terminal error, got a response")),
            Err(err) => {
                assert!(
                    err.to_string()
                        .contains("terminal error while calls pending")
                );
                Ok(())
            }
        }
    }
}
