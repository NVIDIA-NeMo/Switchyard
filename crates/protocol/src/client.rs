// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The routed-call server trait and the routing decision it carries.
//!
//! [`RoutedLlmClient`] is the one piece of I/O the protocol does not own: a host
//! implements it to actually perform a model call. [`Decision`] is the routing
//! decision that produced the call, carried alongside so the client and any
//! observer can see which model was chosen and why. Both live here — rather than
//! in libsy's orchestration crate — so a client crate that depends only on the
//! protocol can serve routed calls without pulling in the orchestrator.

use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;

use crate::{Context, Request, Response};

/// A decision/trace object produced by an algorithm.
///
/// Carried as a trait object (not a generic parameter) so a stream consumer can
/// inspect any algorithm's decision through this common interface without
/// knowing the concrete type. `as_any` is the escape hatch for a consumer that
/// *does* know the algo and wants to downcast to the concrete decision.
pub trait Decision: Send + Sync {
    /// The model this decision selected (e.g. the routed target's name).
    fn selected_model(&self) -> &str;
    /// A human-readable explanation of the decision, for logs and traces.
    fn reasoning(&self) -> Option<&str>;
    /// Downcast handle: a consumer that knows the algorithm can recover the
    /// concrete decision type via `as_any().downcast_ref::<ConcreteDecision>()`.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Performs the actual model call for a target. This is the one piece of I/O the
/// library does not own — a host implements it over its own transport (HTTP SDK,
/// in-process model, mock). It serves a call the stream consumer chose not to
/// override, reached as a routed request's `default_client`.
#[async_trait]
pub trait RoutedLlmClient: Send + Sync {
    /// Serve the call, returning the model's response. Call the model named by
    /// [`decision.selected_model()`](Decision::selected_model) — the target the algorithm
    /// routed to — mapping it to whatever provider model id this client hits.
    /// `request.llm_request.model` is the agent's original name, carried through for
    /// reference, not a call target. `ctx` carries the request's cross-cutting state.
    async fn call(
        &self,
        ctx: Context,
        request: Request,
        decision: Arc<dyn Decision>,
    ) -> Result<Response, Box<dyn Error + Send + Sync>>;
}
