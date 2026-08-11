// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Drive a libsy algorithm to completion, making the model calls it offloads.
//!
//! [`switchyard_libsy::Algorithm::run_stream`] is the whole libsy API: it yields a stream of
//! steps and expects its consumer to serve every offloaded model call. [`run()`] is that
//! consumer — it drives the stream with [`switchyard_libsy::drive`], hands each call to a
//! [`RoutedLlmClient`], and returns the final response with the trace of decisions.
//!
//! libsy owns the stream mechanics; what this module adds is the client call itself and the
//! `libsy.client_call` span around it.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use switchyard_libsy::{Algorithm, CallModel, LibsyError, Result, drive};
use switchyard_protocol::{Decision, LlmClientError, ModelId, Request, Response, RoutedLlmClient};

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

/// Serve one offloaded call. A failed *model* call is forwarded to the algorithm via
/// `respond`; this errors only when the promise itself could not be fulfilled. `serve` makes
/// the one provider call a routed request performs, so it gets its own `libsy.client_call`
/// span.
#[tracing::instrument(
    target = "libsy",
    name = "libsy.client_call",
    skip_all,
    fields(
        algorithm = call.algorithm,
        switchyard.algorithm = call.algorithm,
        selected_model = %call.decision.selected_model_id(),
        otel.kind = "client",
        otel.name = %format_args!("chat {}", call.decision.selected_model_id()),
        openinference.span.kind = "LLM",
        gen_ai.operation.name = "chat",
        gen_ai.request.model = %call.decision.selected_model_id(),
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
async fn serve(
    clients: ClientRouter,
    call: CallModel,
    observer: Option<RunObserver>,
    // Output parameter because `drive` takes a function that returns a plain `Result<()>`.
    routed_calls: Arc<Mutex<RoutedCallWindows>>,
) -> Result<()> {
    let span = tracing::Span::current();
    observability::record_gen_ai_request(&span, &call.request.llm_request);
    if let Some(session_id) = call
        .request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.session_id.as_deref())
    {
        span.record("gen_ai.conversation.id", session_id);
    }
    let request = call.request.clone();
    let decision = call.decision.clone();
    let target = decision.selected_model_id().clone();
    let is_answer_call = decision.is_answer_call();
    // Resolved before the clock starts: picking the client is Switchyard's work, not
    // the provider's, so it belongs in the routing overhead.
    let client = clients.route(&target);
    let started = Instant::now();
    let result = match client {
        Ok(client) => client.call(request, decision).await,
        Err(error) => Err(error),
    }
    .map_err(|source| LibsyError::client_call(target, source));
    let ended = Instant::now();
    let duration = ended - started;

    let result = observability::observe_client_call(result);
    if let Some(observer) = observer {
        observer(RunObservation::LlmCall(LlmCallObservation {
            selected_model: call.decision.selected_model_id().clone(),
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

    call.respond(result)
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
