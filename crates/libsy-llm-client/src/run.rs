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

use switchyard_libsy::{
    Algorithm, CallLlmRequest, LibsyError, Result, RunObserver, algorithm_label, drive,
};
use switchyard_protocol::{Context, Decision, LlmClientError, Request, Response, RoutedLlmClient};

use crate::observability;

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
    ctx: Context,
    request: Request,
    observer: Option<RunObserver>,
) -> Result<(Vec<Arc<dyn Decision>>, Response)> {
    drive(algorithm, ctx, request, observer, move |call| {
        serve(clients.clone(), call)
    })
    .await
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
        algorithm = algorithm_label(&call.get_routed().ctx),
        switchyard.algorithm = algorithm_label(&call.get_routed().ctx),
        switchyard.routing.tier = tracing::field::Empty,
        selected_model = call.get_decision().selected_model(),
        otel.kind = "client",
        otel.name = %format_args!("chat {}", call.get_decision().selected_model()),
        openinference.span.kind = "LLM",
        gen_ai.operation.name = "chat",
        gen_ai.request.model = call.get_decision().selected_model(),
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
async fn serve(clients: ClientRouter, call: CallLlmRequest) -> Result<()> {
    let span = tracing::Span::current();
    observability::record_gen_ai_request(&span, &call.get_routed().request.llm_request);
    if let Some(tier) = call.get_decision().routing_tier() {
        span.record("switchyard.routing.tier", tier);
    }
    if let Some(session_id) = call
        .get_routed()
        .request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.session_id.as_deref())
    {
        span.record("gen_ai.conversation.id", session_id);
    }
    let routed = call.get_routed().clone();
    let target = routed.decision.selected_model().to_string();
    let result = match clients.route(&target) {
        Ok(client) => {
            client
                .call(routed.ctx, routed.request, routed.decision)
                .await
        }
        Err(error) => Err(error),
    }
    .map_err(|source| LibsyError::client_call(target, source));
    let result = observability::observe_client_call(result);
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
    ByModel(HashMap<String, Arc<dyn RoutedLlmClient>>),
}

impl ClientRouter {
    /// Build a router over `model name -> client`, for targets spread across providers.
    pub fn new(by_model: HashMap<String, Arc<dyn RoutedLlmClient>>) -> Self {
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
        model: &str,
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

impl FromIterator<(String, Arc<dyn RoutedLlmClient>)> for ClientRouter {
    fn from_iter<I: IntoIterator<Item = (String, Arc<dyn RoutedLlmClient>)>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}
