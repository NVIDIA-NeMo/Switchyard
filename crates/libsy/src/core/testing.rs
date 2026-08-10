// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fake clients for libsy's own tests.
//!
//! libsy offloads every model call, so a test needs something to answer them. [`drive`] is
//! [`crate::drive`] with the promise handling filled in: a test supplies a [`Serve`] closure
//! standing in for the client a real host would use, and gets back the same
//! `(trace, response)` pair a host does — the same path `switchyard-llm-client`'s `run`
//! takes over HTTP.
//!
//! The closure is async so a fake can block on a barrier, wait on a notify, or never
//! resolve, which is what the concurrency, hedging, and fan-out tests need.

use std::future::Future;
use std::sync::Arc;

use futures::future::BoxFuture;
use switchyard_protocol::{
    Decision, LlmClientError, LlmResponse, Request, Response, text_response,
};

use crate::core::algorithm::{Algorithm, CallModel};
use crate::{LibsyError, Result};

/// The result a fake client hands back for one offloaded call.
pub(crate) type ServeResult = std::result::Result<Response, LlmClientError>;

/// Answers offloaded model calls. Returning `Err` propagates a failed *model* call back into
/// the algorithm, which may route around it.
pub(crate) trait Serve: Send + Sync + 'static {
    fn serve(&self, decision: Decision, request: Request) -> BoxFuture<'static, ServeResult>;
}

impl<F, Fut> Serve for F
where
    F: Fn(Decision, Request) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ServeResult> + Send + 'static,
{
    fn serve(&self, decision: Decision, request: Request) -> BoxFuture<'static, ServeResult> {
        Box::pin(self(decision, request))
    }
}

/// Run `algorithm` to completion, serving each offloaded call with `serve`.
pub(crate) async fn test_drive(
    algorithm: Arc<dyn Algorithm>,
    request: Request,
    serve: impl Serve,
) -> Result<(Vec<Decision>, Response)> {
    let serve = Arc::new(serve);
    crate::drive(algorithm, request, move |call| {
        fulfill(Arc::clone(&serve), call)
    })
    .await
}

/// Serve one call and fulfill its promise, mapping failures the way a host does so
/// error-shape assertions match production.
async fn fulfill(serve: Arc<impl Serve>, call: CallModel) -> Result<()> {
    let request = call.request.clone();
    let decision = call.decision.clone();
    let target = decision.selected_model_id().to_string();
    let result = serve
        .serve(decision, request)
        .await
        .map_err(|source| LibsyError::client_call(target, source));
    call.respond(result)
}

/// Answers with the selected model name as the completion — what most routing tests need,
/// since they assert on *which* target was called.
pub(crate) fn echo() -> impl Serve {
    |decision: Decision, _request: Request| async move { Ok(reply(decision.selected_model_id())) }
}

/// A buffered response whose completion text is `completion`.
pub(crate) fn reply(completion: impl Into<String>) -> Response {
    Response {
        llm_response: LlmResponse::Agg(text_response(None, completion.into())),
        metadata: None,
    }
}
