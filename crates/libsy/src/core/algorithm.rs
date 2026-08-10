// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The [`Algorithm`] trait and its [`Driver`] — the orchestration contract every
//! algorithm implements, and the offload channel it uses to make model calls and
//! publish [`Decision`]s.

use std::{
    collections::{HashMap, HashSet},
    future::Future,
    pin::Pin,
    sync::Arc,
    time::Instant,
};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
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
use switchyard_protocol::{
    Context, Decision, LlmClientError, Request, Response, RoutingFallbackReason, Signals,
};

use crate::{DriverError, LibsyError, Result, observability};

/// A boxed, `Send` stream of [`Step`]s — the output of
/// [`Algorithm::run_stream`]. Boxed so the trait method that produces it keeps
/// `Arc<dyn Algorithm>` object-safe.
pub type StepStream = Pin<Box<dyn Stream<Item = Result<Step>> + Send>>;

/// A request paired with the routing [`Decision`] that produced it — the offload
/// payload a host reads (via [`CallLlmRequest::get_routed`]) to serve the call.
///
/// The selected model and inbound route name live in separate, unambiguous places: the model
/// identifier is [`decision.selected_model_id()`](Decision::selected_model_id), while
/// `request.llm_request.model` is the *inbound* name the agent asked for (libsy
/// never overwrites it). A client maps `selected_model_id()` to the provider model
/// id it hits.
#[derive(Clone)]
pub struct RoutedRequest {
    /// The request to serve; its `model` is the agent's original name.
    pub request: Request,
    /// The routing decision behind this call; `selected_model_id()` identifies the model to hit.
    pub decision: Arc<Decision>,
    /// The request's cross-cutting context, carried through the offload so the host
    /// serving the call can pass it to its client.
    pub ctx: Context,
}

/// The host-facing half of an offloaded model call, surfaced inside [`Step::CallLlm`].
///
/// The host reads the routed request ([`get_routed`](Self::get_routed)) and the decision
/// behind it ([`get_decision`](Self::get_decision)), performs (or delegates) the model
/// call, and fulfills it with [`respond`](Self::respond) — unblocking the algorithm's
/// [`Driver::call_llm`] on the other side. `switchyard-llm-client`'s `run` is the ready-made
/// consumer that does this for you.
pub struct CallLlmRequest {
    routed: RoutedRequest,
    reply: oneshot::Sender<Result<Response>>,
}

impl CallLlmRequest {
    /// The routed request the host should serve; its `decision.selected_model_id()` names
    /// the model to hit.
    pub fn get_routed(&self) -> &RoutedRequest {
        &self.routed
    }

    /// The model request to perform (the [`Request`] inside the routed request).
    pub fn get_request(&self) -> &Request {
        &self.get_routed().request
    }

    /// The decision that led to this call; its `selected_model_id()` identifies the model to hit.
    pub fn get_decision(&self) -> &Decision {
        self.get_routed().decision.as_ref()
    }

    /// Fulfill the promise with the caller's model-call result. Pass `Err(..)` to
    /// propagate a failed model call back to the algorithm. Consumes the promise: it
    /// can only be fulfilled once.
    pub fn respond(self, result: Result<Response>) -> Result<()> {
        self.reply
            .send(result)
            .map_err(|_| DriverError::ResponseDropped.into())
    }
}

/// The offload channel handed to an algorithm's
/// [`create_run_task`](Algorithm::create_run_task). The algorithm makes model calls
/// with [`call_llm`](Self::call_llm) and publishes its [`Decision`]s with
/// [`info`](Self::info); each call is offloaded to the request's [`Step`] stream and
/// awaits the consumer's response. The step channel is bounded, so the consumer paces
/// the algorithm one step at a time.
#[derive(Clone)]
pub struct Driver {
    step_tx: mpsc::Sender<Result<Step>>,
}

impl Driver {
    /// Build an empty driver with its step channel ready. Created per call by
    /// [`run_stream`](Algorithm::run_stream). Also returns the Step receiver.
    pub(crate) fn new() -> (Self, mpsc::Receiver<Result<Step>>) {
        // Capacity one keeps the algorithm paced by the stream consumer. It limits queued steps,
        // not model calls already pulled from the stream, which can still run at the same time.
        // A larger buffer would use more memory and let the algorithm run farther ahead with
        // little benefit because reading a step is cheap compared with serving a model call.
        let (step_tx, step_rx) = mpsc::channel(1);
        (Self { step_tx }, step_rx)
    }

    /// Offload a model call: publish `routed` as a [`Step::CallLlm`] and await the
    /// consumer's [`Response`]. The call's context travels inside
    /// [`routed.ctx`](RoutedRequest::ctx). Errors if the stream is closed or the call failed.
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
            algorithm = observability::algorithm_label(&routed.ctx),
            selected_model = routed.decision.selected_model_id(),
            openinference.span.kind = "CHAIN",
            outcome = tracing::field::Empty,
            error = tracing::field::Empty,
            input_tokens = tracing::field::Empty,
            output_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            reasoning_tokens = tracing::field::Empty,
        )
    )]
    pub async fn call_llm(&self, routed: RoutedRequest) -> Result<Response> {
        let algorithm = observability::algorithm_label(&routed.ctx).to_string();
        let decision = Arc::clone(&routed.decision);
        let is_answer_call = decision.is_answer_call();
        let started = Instant::now();
        let (reply, response) = oneshot::channel::<Result<Response>>();
        let call = CallLlmRequest { routed, reply };
        let result = async {
            self.step_tx
                .send(Ok(Step::CallLlm(Box::new(call))))
                .await
                .map_err(|_| DriverError::StreamClosed)?;
            response
                .await
                .map_err(|_| LibsyError::from(DriverError::ResponseDropped))?
        }
        .await;
        let elapsed = started.elapsed();
        observability::record_llm_call(
            &algorithm,
            decision.selected_model_id(),
            is_answer_call,
            elapsed,
            &result,
            &tracing::Span::current(),
        );
        result
    }

    /// Publish a routing [`Decision`] as a [`Step::Decision`] on the stream.
    /// Each successfully published decision is counted and logged with its
    /// reasoning; a decision the stream never accepted is not recorded.
    pub async fn info(&self, ctx: Context, decision: Arc<Decision>) -> Result<()> {
        self.step_tx
            .send(Ok(Step::Decision(decision.clone())))
            .await
            .map_err(|_| DriverError::StreamClosed)?;
        observability::record_decision(&ctx, decision.as_ref());
        Ok(())
    }

    /// Emit the terminal step: [`Step::ReturnToAgent`] on `Ok`, or an `Err` stream
    /// item on failure. Internal: called once by [`run_stream`](Algorithm::run_stream)
    /// when the algorithm finishes.
    pub(crate) async fn finish(&self, result: Result<Response>) -> Result<()> {
        let step = result.map(|response| Step::ReturnToAgent(Box::new(response)));
        self.step_tx
            .send(step)
            .await
            .map_err(|_| DriverError::StreamClosed.into())
    }
}

/// One item in the stream returned by [`Algorithm::run_stream`].
pub enum Step {
    /// The algorithm needs this model call performed. The host serves it and fulfills
    /// it with [`CallLlmRequest::respond`]. Boxed: it is by far the largest variant.
    CallLlm(Box<CallLlmRequest>),
    /// A routing decision the algorithm made, published via [`Driver::info`] as it
    /// happens (rather than collected into a trace returned at the end).
    Decision(Arc<Decision>),
    /// The algorithm finished with its final response — the last step of a run.
    ReturnToAgent(Box<Response>),
}

/// Drive [`Algorithm::run_stream`] to completion, handing each offloaded call to `serve`.
///
/// Returns the final [`Response`] and the trace of [`Decision`]s the algorithm published.
/// `serve` owns the call: it performs it however the host likes and must fulfill the promise
/// with [`CallLlmRequest::respond`]. A failed *model* call belongs in `respond` — the
/// algorithm may route around it. Returning `Err` from `serve` aborts the whole run, so
/// reserve it for infrastructure failures. Calls are served concurrently, so an algorithm
/// that offloads several at once (hedging, fan-out) gets real parallelism.
///
/// libsy performs no I/O; this is only the mechanics of consuming its own step stream, kept
/// here so every host does not reimplement the same loop. `switchyard-llm-client`'s `run`
/// is this function plus an HTTP client.
pub async fn drive<F, Fut>(
    algorithm: Arc<dyn Algorithm>,
    ctx: Context,
    request: Request,
    serve: F,
) -> Result<(Vec<Arc<Decision>>, Response)>
where
    F: Fn(CallLlmRequest) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    let stream = algorithm.run_stream(ctx, request);
    tokio::pin!(stream);

    let mut trace: Vec<Arc<Decision>> = Vec::new();
    let mut in_flight = futures::stream::FuturesUnordered::new();
    let mut final_response: Option<Response> = None;

    loop {
        tokio::select! {
            Some(result) = in_flight.next() => match result {
                Ok(()) => {}, // CallLlm completed successfully
                Err(err) => return Err(err), // CallLlm failed, propagate the error
            },
            step = stream.next() => {
                match step {
                    None => break, // stream has ended, no more steps
                    Some(item) => match item? {
                        Step::CallLlm(call) => in_flight.push(serve(*call)),
                        Step::Decision(decision) => trace.push(decision),
                        Step::ReturnToAgent(response) => {
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

/// Abort guard
struct AbortOnDrop(tokio::task::AbortHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// A named routing target an algorithm routes by. Serving its calls is the stream
/// consumer's concern: the selected identifier reaches the consumer as
/// `decision.selected_model_id()` on the offloaded [`RoutedRequest`].
#[derive(Clone)]
pub struct LlmTarget {
    /// The routing label an algorithm selects this target by — a logical tier like
    /// `"strong"`, or the model id when they coincide. Mapping it to a provider model
    /// id is the consumer's concern, never the algorithm's.
    pub semantic_name: String,
}

/// The set of targets an algorithm may route among. An algorithm is constructed
/// with one and picks targets by position ([`targets`](Self::targets)) or by name
/// ([`get_target`](Self::get_target)).
#[derive(Clone)]
pub struct LlmTargetSet {
    targets: Vec<LlmTarget>,
}

impl LlmTargetSet {
    /// Build a target set from a list of targets.
    pub fn new(targets: Vec<LlmTarget>) -> Self {
        Self { targets }
    }

    /// All targets in the set — e.g. for an algorithm to select among.
    pub fn targets(&self) -> &[LlmTarget] {
        &self.targets
    }

    /// Look up a target by name; errors if no target has that name.
    pub fn get_target(&self, name: &str) -> Result<LlmTarget> {
        self.targets
            .iter()
            .find(|t| t.semantic_name == name)
            .cloned()
            .ok_or_else(|| LibsyError::TargetNotFound {
                target: name.to_string(),
            })
    }

    /// The named target, or the first one this request is not barred from when it has been
    /// excluded (see [`Context::exclude_target`]). Errors if every target is excluded.
    pub fn resolve_target(&self, name: &str, ctx: &Context) -> Result<LlmTarget> {
        let target = self.get_target(name)?;
        if !ctx.is_excluded(&target.semantic_name) {
            return Ok(target);
        }
        self.targets
            .iter()
            .find(|t| !ctx.is_excluded(&t.semantic_name))
            .cloned()
            .ok_or(LibsyError::AllTargetsExcluded)
    }
}

/// Key for overflow history: a root request by its session, a child request by its session
/// and agent. Keying a child finer than its session keeps one child's overflow from evicting
/// a target for the parent or a sibling sharing the session.
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

    /// The session this identity belongs to; shared by a session's root and its children.
    fn session(&self) -> &str {
        match self {
            Self::Session(session) | Self::Subagent { session, .. } => session,
        }
    }
}

/// Bounds process-local overflow history. Dropping a live entry costs one rediscovered
/// overflow, so the victim choice does not need to be exact.
const MAX_EVICTION_IDENTITIES: usize = 1_024;

/// Per-identity record of the targets that overflowed their context window.
///
/// A conversation only grows, so a target that could not fit one turn will not fit a
/// later one; remembering it lets the next turn skip a call certain to fail. Requests
/// without a routing identity are not tracked — there is nothing to remember them by.
#[derive(Default)]
pub(crate) struct SessionEvictions {
    by_identity: Mutex<HashMap<RoutingIdentity, HashSet<String>>>,
}

impl SessionEvictions {
    /// Forgets overflow history for a completed session, including every child of it.
    pub(crate) fn remove_session(&self, session: &str) {
        self.by_identity
            .lock()
            .retain(|identity, _| identity.session() != session);
    }

    /// The targets `identity` has already overflowed; empty for an untracked request.
    fn evicted_for(&self, identity: Option<&RoutingIdentity>) -> Vec<String> {
        let Some(identity) = identity else {
            return Vec::new();
        };
        self.by_identity
            .lock()
            .get(identity)
            .map(|targets| targets.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Remembers that `target` overflowed for `identity`, tracking at most
    /// [`MAX_EVICTION_IDENTITIES`] identities.
    fn record(&self, identity: Option<&RoutingIdentity>, target: &str) {
        let Some(identity) = identity else { return };
        let mut histories = self.by_identity.lock();
        if histories.len() >= MAX_EVICTION_IDENTITIES
            && !histories.contains_key(identity)
            && let Some(oldest) = histories.keys().next().cloned()
        {
            histories.remove(&oldest);
        }
        histories
            .entry(identity.clone())
            .or_default()
            .insert(target.to_string());
    }
}

/// How many of `targets` this request is still allowed to reach.
fn eligible_targets(targets: &LlmTargetSet, ctx: &Context) -> usize {
    targets
        .targets()
        .iter()
        .filter(|t| !ctx.is_excluded(&t.semantic_name))
        .count()
}

/// Bars the targets `identity` has already overflowed from this request, so routing does
/// not select one that is certain to fail again.
pub(crate) fn exclude_evicted(
    ctx: &mut Context,
    targets: &LlmTargetSet,
    evictions: &SessionEvictions,
    identity: Option<&RoutingIdentity>,
) {
    for target in evictions.evicted_for(identity) {
        // Never seed the pool empty: a later turn may be small enough to serve, and the
        // caller should get the upstream's answer rather than a routing error.
        if eligible_targets(targets, ctx) <= 1 {
            break;
        }
        ctx.exclude_target(target);
    }
}

/// Returns the failed target and routing fallback policy for a terminal client error.
fn classify_fallback(error: &LibsyError) -> Option<(&str, RoutingFallbackReason)> {
    let LibsyError::ClientCall { target, source } = error else {
        return None;
    };
    let reason = match source {
        LlmClientError::ContextWindowExceeded { .. } => RoutingFallbackReason::ContextWindow,
        LlmClientError::Transport { .. } | LlmClientError::Timeout { .. } => {
            RoutingFallbackReason::Unavailable
        }
        LlmClientError::UpstreamHttp { status, .. }
            if matches!(*status, 403 | 408 | 429) || (500..=599).contains(status) =>
        {
            RoutingFallbackReason::Unavailable
        }
        _ => return None,
    };
    Some((target, reason))
}

/// Calls `target`, falling back to the next eligible target after a route-level failure,
/// until a call succeeds or every target has been tried.
///
/// Routing is deliberately not re-run: the fallback replaces the target in place, so the
/// caller's request-side work and retained state still see exactly one turn.
/// `fallback_decision` builds the [`Decision`] published for a `from -> to` hop. Context
/// overflows are recorded for `identity`; unavailable targets remain request-local.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn call_llm_with_fallback(
    mut ctx: Context,
    driver: &Driver,
    targets: &LlmTargetSet,
    mut target: LlmTarget,
    mut decision: Arc<Decision>,
    request: Request,
    identity: Option<&RoutingIdentity>,
    evictions: &SessionEvictions,
    target_unavailable: impl Fn(&Request, &str),
    fallback_decision: impl Fn(&LlmTarget, &LlmTarget, RoutingFallbackReason) -> Arc<Decision>,
) -> Result<Response> {
    loop {
        let result = driver
            .call_llm(RoutedRequest {
                request: request.clone(),
                decision: decision.clone(),
                ctx: ctx.clone(),
            })
            .await;
        let Err(error) = result else { return result };
        let Some((failed, reason)) = classify_fallback(&error) else {
            return Err(error);
        };
        // A target already excluded means the pool is spent; surface the client error
        // so the caller still sees the concrete upstream failure.
        if !ctx.exclude_target(failed) {
            return Err(error);
        }
        match reason {
            RoutingFallbackReason::ContextWindow => evictions.record(identity, failed),
            RoutingFallbackReason::Unavailable => target_unavailable(&request, failed),
        }
        let Ok(next) = targets.resolve_target(&target.semantic_name, &ctx) else {
            return Err(error);
        };
        decision = fallback_decision(&target, &next, reason);
        target = next;
        driver.info(ctx.clone(), decision.clone()).await?;
    }
}

/// An optimization strategy. Implement [`create_run_task`](Self::create_run_task);
/// callers drive it with [`run_stream`](Self::run_stream), serving each [`Step::CallLlm`]
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

    /// Run one request to completion: make model calls with [`Driver::call_llm`],
    /// publish [`Decision`]s with [`Driver::info`], and return the final [`Response`].
    /// The method an algorithm implements; [`run_stream`](Self::run_stream) drives it.
    /// `ctx` carries the request's cross-cutting values (today: the algorithm's
    /// telemetry label in [`Context::values`]).
    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response>;

    /// Feed the algorithm agentic-stack events (tool results, budgets, etc.). The
    /// reference algorithms ignore signals; a stateful algorithm updates its own
    /// (interior-mutable) state. Takes `self: Arc<Self>` like the other run methods.
    #[allow(unused_variables)]
    async fn process_signals(self: Arc<Self>, signals: Signals) -> Result<()> {
        Ok(())
    }

    /// Process a request to completion, returning a stream of [`Step`]s.
    ///
    /// The consumer must fulfill every [`Step::CallLlm`] before the algorithm can
    /// continue. The bounded step channel applies backpressure when the consumer is
    /// not polling. A successful run ends with [`Step::ReturnToAgent`]; a failure is
    /// emitted as an `Err` item. Dropping the stream aborts the spawned algorithm task.
    ///
    /// Every invocation owns a separate [`Driver`].
    fn run_stream(self: Arc<Self>, ctx: Context, request: Request) -> StepStream {
        // Stamp the algorithm's telemetry label into the request context; the
        // context rides on every driver call, so its telemetry is attributed.
        let mut ctx = ctx;
        ctx.values.insert(
            observability::ALGORITHM_KEY.to_string(),
            self.name().to_string(),
        );
        let (driver, step_rx) = Driver::new();
        let task_driver = driver.clone();
        let task_ctx = ctx.clone();
        let stream = ReceiverStream::new(step_rx);
        // One `libsy.run` span covers the whole algorithm task; the driver's
        // `libsy.llm_call` spans and decision logs nest inside it via `tracing`'s
        // contextual parenting.
        let span = observability::run_span(self.name(), &request);
        let handle = tokio::spawn(
            async move {
                observability::observe_run(
                    task_ctx.clone(),
                    self.create_run_task(task_ctx, task_driver, request),
                )
                .await
            }
            .instrument(span),
        );
        // Dropping the stream aborts the algorithm task when its consumer goes away.
        let abort_guard = AbortOnDrop(handle.abort_handle());

        let finish_driver = driver.clone();
        let tail: StepStream = Box::pin(
            futures::stream::once(async move {
                let result = match handle.await {
                    Ok(response) => response,
                    Err(source) => Err(LibsyError::AlgorithmTask { source }),
                };
                finish_driver.finish(result).await
            })
            .filter_map(|finish_result| async move { finish_result.err().map(Err) }),
        );

        let stream: StepStream = Box::pin(stream);
        Box::pin(futures::stream::select(stream, tail).map(move |step| {
            // link abort guard to stream
            let _keep_alive = &abort_guard;
            step
        }))
    }
}

#[cfg(test)]
mod tests {
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

    fn classified_client_error(source: LlmClientError) -> Option<RoutingFallbackReason> {
        classify_fallback(&LibsyError::client_call("target", source)).map(|(_, reason)| reason)
    }

    #[test]
    fn route_fallback_only_accepts_context_and_unavailable_failures() {
        assert_eq!(
            classified_client_error(LlmClientError::ContextWindowExceeded {
                model: "target".to_string(),
                message: "too long".to_string(),
            }),
            Some(RoutingFallbackReason::ContextWindow)
        );
        for source in [
            LlmClientError::Transport {
                source: Box::new(std::io::Error::other("connection failed")),
            },
            LlmClientError::Timeout {
                source: Box::new(std::io::Error::other("request timed out")),
            },
        ] {
            assert_eq!(
                classified_client_error(source),
                Some(RoutingFallbackReason::Unavailable)
            );
        }
        for (status, expected) in [
            (400, None),
            (401, None),
            (403, Some(RoutingFallbackReason::Unavailable)),
            (404, None),
            (408, Some(RoutingFallbackReason::Unavailable)),
            (409, None),
            (429, Some(RoutingFallbackReason::Unavailable)),
            (499, None),
            (500, Some(RoutingFallbackReason::Unavailable)),
            (599, Some(RoutingFallbackReason::Unavailable)),
            (600, None),
        ] {
            assert_eq!(
                classified_client_error(LlmClientError::UpstreamHttp {
                    status,
                    body: "failed".to_string(),
                }),
                expected
            );
        }
        assert_eq!(
            classified_client_error(LlmClientError::InvalidResponse {
                source: Box::new(std::io::Error::other("invalid response")),
            }),
            None
        );
    }

    /// Build a routed decision for orchestration tests.
    fn test_decision(selected_model_id: String) -> Arc<Decision> {
        Arc::new(Decision::new(selected_model_id, None, true))
    }

    /// Trivial algo used only to exercise the orchestrator: calls the first target
    /// and returns its response with a one-item trace.
    struct TestAlgo {
        target_set: LlmTargetSet,
    }

    #[async_trait]
    impl Algorithm for TestAlgo {
        fn name(&self) -> &str {
            "test"
        }

        async fn create_run_task(
            self: Arc<Self>,
            ctx: Context,
            driver: Driver,
            request: Request,
        ) -> Result<Response> {
            let target = self
                .target_set
                .targets()
                .first()
                .ok_or(LibsyError::NoTargets)?
                .clone();
            let decision = test_decision(target.semantic_name.clone());
            driver.info(ctx.clone(), decision.clone()).await?;
            driver
                .call_llm(RoutedRequest {
                    request,
                    decision,
                    ctx,
                })
                .await
        }
    }

    /// Build a shared `TestAlgo` over the given target set.
    fn orch(target_set: LlmTargetSet) -> Arc<dyn Algorithm> {
        Arc::new(TestAlgo { target_set })
    }

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi".to_string()),
            raw_request: None,
            metadata: None,
        }
    }

    fn target_set(names: &[&str]) -> LlmTargetSet {
        let targets = names
            .iter()
            .map(|name| LlmTarget {
                semantic_name: name.to_string(),
            })
            .collect();
        LlmTargetSet::new(targets)
    }

    fn routed(model: &str) -> RoutedRequest {
        RoutedRequest {
            request: request(),
            decision: test_decision(model.to_string()),
            ctx: Context::default(),
        }
    }

    #[tokio::test]
    async fn typed_driver_preserves_call_and_stream_boundaries() -> Result<()> {
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            // Distinct oneshots keep reverse-order replies paired with their producers, and a
            // retained call remains pending until the host responds.
            let (driver, mut step_rx) = Driver::new();
            let first_driver = driver.clone();
            let mut first =
                tokio::spawn(async move { first_driver.call_llm(routed("first")).await });
            let second = tokio::spawn(async move { driver.call_llm(routed("second")).await });

            let mut calls = HashMap::new();
            for _ in 0..2 {
                let step = step_rx.recv().await.ok_or(DriverError::StreamClosed)??;
                let Step::CallLlm(call) = step else {
                    return Err(test_error("expected a CallLlm step"));
                };
                calls.insert(call.get_decision().selected_model_id().to_string(), call);
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
                .map_err(|source| LibsyError::AlgorithmTask { source })??;
            let second_response = second
                .await
                .map_err(|source| LibsyError::AlgorithmTask { source })??;
            assert_eq!(
                first_response.llm_response.as_agg().map(completion_text),
                Some("first response".to_string())
            );
            assert_eq!(
                second_response.llm_response.as_agg().map(completion_text),
                Some("second response".to_string())
            );

            // Dropping the host-facing promise closes only that call's reply channel.
            let (driver, mut step_rx) = Driver::new();
            let producer = tokio::spawn(async move { driver.call_llm(routed("dropped")).await });
            let step = step_rx.recv().await.ok_or(DriverError::StreamClosed)??;
            let Step::CallLlm(call) = step else {
                return Err(test_error("expected a CallLlm step"));
            };
            drop(call);
            let result = producer
                .await
                .map_err(|source| LibsyError::AlgorithmTask { source })?;
            assert!(matches!(
                result,
                Err(LibsyError::Driver(DriverError::ResponseDropped))
            ));

            // A standalone driver reports the typed step receiver disappearing at its next send.
            let (driver, step_rx) = Driver::new();
            drop(step_rx);
            let decision = test_decision("closed".to_string());
            let result = driver.info(Context::default(), decision).await;
            assert!(matches!(
                result,
                Err(LibsyError::Driver(DriverError::StreamClosed))
            ));
            Ok(())
        })
        .await
        .map_err(|error| LibsyError::external("waiting for typed driver boundaries", error))?
    }

    #[test]
    fn target_lookup_returns_the_missing_target() {
        let error = target_set(&[]).get_target("missing").err();
        assert!(matches!(
            error,
            Some(LibsyError::TargetNotFound { target }) if target == "missing"
        ));
    }

    /// Build a single-target algo, plus a `serve` that answers it as a token stream
    /// replaying `chunks` in order (as `Ok` items).
    fn streaming_orch(chunks: Vec<LlmResponseChunk>) -> (Arc<dyn Algorithm>, impl Serve) {
        let algo = orch(target_set(&["stream/model"]));
        let serve = move |_decision: Arc<Decision>, _request: Request| {
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
        // A streaming client -> its chunks flow through the promise and `ReturnToAgent`,
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
        let (trace, response) = test_drive(orch, Context::default(), request(), serve).await?;
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
        let (_, response) = test_drive(orch, Context::default(), request(), serve).await?;
        match response.llm_response.into_agg().await {
            Ok(_) => panic!("expected a mid-stream error, got an aggregate"),
            Err(err) => {
                assert!(err.to_string().contains("upstream exploded"));
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn run_offloads_via_promise_then_returns_to_agent() -> Result<()> {
        // Every call is offloaded via a promise the orchestrator surfaces as a
        // `CallLlm` step for us to fulfill.
        let stream = orch(target_set(&["offload/model"])).run_stream(Context::default(), request());
        tokio::pin!(stream);

        let mut saw_call = false;
        let mut final_completion = None;
        while let Some(step) = stream.next().await {
            match step? {
                Step::CallLlm(call) => {
                    saw_call = true;
                    // The decision rode along with the promise.
                    assert_eq!(call.get_decision().selected_model_id(), "offload/model");
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
                Step::ReturnToAgent(response) => {
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

        assert!(saw_call, "expected a CallLlm step before ReturnToAgent");
        assert_eq!(
            final_completion.ok_or_else(|| test_error("no ReturnToAgent step"))?,
            "fulfilled"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_driven_run_returns_the_trace_and_the_final_response() -> Result<()> {
        let (trace, response) = test_drive(
            orch(target_set(&["direct/model"])),
            Context::default(),
            request(),
            echo(),
        )
        .await?;
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
            let serve = move |decision: Arc<Decision>, _request: Request| {
                let barrier = barrier.clone();
                async move {
                    barrier.wait().await;
                    Ok(reply(decision.selected_model_id()))
                }
            };
            handles.push(tokio::spawn(async move {
                test_drive(algo, Context::default(), request(), serve)
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
                .map_err(|source| LibsyError::AlgorithmTask { source })??;
            assert_eq!(completion, "m");
        }
        Ok(())
    }

    #[tokio::test]
    async fn offload_error_propagates_back_to_the_algorithm() -> Result<()> {
        // A client-less target offloads its call; we fulfill the promise with an
        // Err, which must flow back through `call_llm_target` into the algorithm and
        // out as an error step — not a response.
        let stream = orch(target_set(&["offload/model"])).run_stream(Context::default(), request());
        tokio::pin!(stream);

        let mut saw_error = false;
        while let Some(step) = stream.next().await {
            match step {
                Ok(Step::CallLlm(call)) => {
                    call.respond(Err(test_error("upstream model call failed")))?;
                }
                Ok(Step::Decision(_)) => {}
                Ok(Step::ReturnToAgent(..)) => {
                    return Err(test_error(
                        "expected the offload error to propagate, got a response",
                    ));
                }
                Err(err) => {
                    // The algorithm's `call_llm_target` saw the error via the promise.
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

            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
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

        let stream = algo.run_stream(Context::default(), request());
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
    async fn create_run_task_panic_surfaces_as_a_stream_error() -> Result<()> {
        // An algorithm whose task panics must surface an `Err` step to the stream
        // consumer, not abort the process from an unobserved detached task.
        struct Panicky;

        #[async_trait]
        impl Algorithm for Panicky {
            fn name(&self) -> &str {
                "panicky"
            }

            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                _driver: Driver,
                _request: Request,
            ) -> Result<Response> {
                panic!("boom");
            }
        }

        let algo: Arc<dyn Algorithm> = Arc::new(Panicky);
        let stream = algo.run_stream(Context::default(), request());
        tokio::pin!(stream);

        let mut saw_error = false;
        while let Some(step) = stream.next().await {
            match step {
                Err(err) => {
                    assert!(matches!(err, LibsyError::AlgorithmTask { .. }));
                    saw_error = true;
                }
                Ok(_) => return Err(test_error("expected the panic to surface as an error step")),
            }
        }

        assert!(saw_error, "expected an error step from the panicked task");
        Ok(())
    }

    #[tokio::test]
    async fn run_returns_an_error_when_the_algorithm_task_panics() -> Result<()> {
        // The panic surfaces as an `Err` step inside `run_stream`; `run` propagates it
        // via `?`, so the caller gets an `Err` rather than a hang or a silent panic.
        struct Panicky;

        #[async_trait]
        impl Algorithm for Panicky {
            fn name(&self) -> &str {
                "panicky"
            }

            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                _driver: Driver,
                _request: Request,
            ) -> Result<Response> {
                panic!("boom");
            }
        }

        let algo: Arc<dyn Algorithm> = Arc::new(Panicky);
        match test_drive(algo, Context::default(), request(), echo()).await {
            Ok(_) => Err(test_error(
                "expected the run to surface the algorithm panic as an error",
            )),
            Err(err) => {
                assert!(matches!(err, LibsyError::AlgorithmTask { .. }));
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

            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
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
        let run_task =
            tokio::spawn(
                async move { test_drive(algo, Context::default(), request(), echo()).await },
            );
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
        winner: LlmTarget,
        loser: LlmTarget,
    }

    #[async_trait]
    impl Algorithm for Hedge {
        fn name(&self) -> &str {
            "hedge"
        }

        async fn create_run_task(
            self: Arc<Self>,
            ctx: Context,
            driver: Driver,
            request: Request,
        ) -> Result<Response> {
            let dec_w = test_decision(self.winner.semantic_name.clone());
            let dec_l = test_decision(self.loser.semantic_name.clone());
            let win = driver.call_llm(RoutedRequest {
                request: request.clone(),
                decision: dec_w,
                ctx: ctx.clone(),
            });
            let lose = driver.call_llm(RoutedRequest {
                request,
                decision: dec_l,
                ctx,
            });
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
            winner: LlmTarget {
                semantic_name: "winner".to_string(),
            },
            loser: LlmTarget {
                semantic_name: "loser".to_string(),
            },
        });
        let serve = move |decision: Arc<Decision>, _request: Request| {
            let started = started.clone();
            async move {
                if decision.selected_model_id() == "loser" {
                    started.notify_one();
                    match loser_delay {
                        Some(delay) => tokio::time::sleep(delay).await,
                        None => std::future::pending::<()>().await,
                    }
                } else {
                    started.notified().await;
                }
                Ok(reply(decision.selected_model_id()))
            }
        };
        (algo, serve)
    }

    #[tokio::test]
    async fn run_returns_the_winner_without_a_late_loser_overwriting_it() -> Result<()> {
        // The loser responds 50ms after the winner has already won. `run` must return the
        // winner, not the loser's `respond`-to-a-dropped-receiver error.
        let (algo, serve) = hedge(Some(std::time::Duration::from_millis(50)));
        let (_trace, response) = test_drive(algo, Context::default(), request(), serve).await?;
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
        let run = test_drive(algo, Context::default(), request(), serve);
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

            async fn create_run_task(
                self: Arc<Self>,
                ctx: Context,
                driver: Driver,
                request: Request,
            ) -> Result<Response> {
                let offloads = futures::future::join_all((0..self.n).map(|i| {
                    let decision = test_decision(format!("m{i}"));
                    driver.call_llm(RoutedRequest {
                        request: request.clone(),
                        decision,
                        ctx: ctx.clone(),
                    })
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
        let serve = move |_decision: Arc<Decision>, _request: Request| {
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
        let run = test_drive(algo, Context::default(), request(), serve);
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
