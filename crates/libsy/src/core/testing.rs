// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fake clients for libsy's own tests.
//!
//! libsy offloads every model call, so a test needs something to answer them. [`drive`] is
//! [`crate::drive`] with the promise handling filled in: a test supplies a [`Serve`] closure
//! standing in for the client a real host would use, and gets back the same
//! `(selected model, response)` pair a host does — the same path
//! `switchyard-llm-client`'s `run`
//! takes over HTTP.
//!
//! The closure is async so a fake can block on a barrier, wait on a notify, or never
//! resolve, which is what the concurrency, hedging, and fan-out tests need.

use std::future::Future;
use std::sync::Arc;

use futures::future::BoxFuture;
use switchyard_protocol::{LlmClientError, LlmResponse, ModelId, Request, Response, text_response};

use crate::core::algorithm::{Algorithm, CallModel};
use crate::{LibsyError, Result};

/// The result a fake client hands back for one offloaded call.
pub(crate) type ServeResult = std::result::Result<Response, LlmClientError>;

/// Answers offloaded model calls. Returning `Err` propagates a failed *model* call back into
/// the algorithm, which may route around it.
pub(crate) trait Serve: Send + Sync + 'static {
    fn serve(&self, target: ModelId, request: Request) -> BoxFuture<'static, ServeResult>;
}

impl<F, Fut> Serve for F
where
    F: Fn(ModelId, Request) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ServeResult> + Send + 'static,
{
    fn serve(&self, target: ModelId, request: Request) -> BoxFuture<'static, ServeResult> {
        Box::pin(self(target, request))
    }
}

/// Run `algorithm` to completion, serving each offloaded call with `serve`.
pub(crate) async fn test_drive(
    algorithm: Arc<dyn Algorithm>,
    request: Request,
    serve: impl Serve,
) -> Result<(ModelId, Response)> {
    let serve = Arc::new(serve);
    let routing_serve = Arc::clone(&serve);
    let outcome = crate::drive(algorithm, request, move |call| {
        fulfill(Arc::clone(&routing_serve), call)
    })
    .await?;
    let selected_model = outcome.selected_model_id()?.clone();
    let response = match outcome.response {
        Some(response) => response,
        None => serve
            .serve(selected_model.clone(), outcome.request)
            .await
            .map_err(|source| LibsyError::client_call(selected_model.clone(), source))?,
    };
    Ok((selected_model, response))
}

/// Serve one call and fulfill its promise, mapping failures the way a host does so
/// error-shape assertions match production.
async fn fulfill(serve: Arc<impl Serve>, call: CallModel) -> Result<()> {
    let request = call.request.clone();
    let target = call.models.first().cloned().ok_or(LibsyError::NoTargets)?;
    let result = serve
        .serve(target.clone(), request)
        .await
        .map_err(|source| LibsyError::client_call(target, source));
    call.respond(result)
}

/// Answers with the selected model name as the completion — what most routing tests need,
/// since they assert on *which* target was called.
pub(crate) fn echo() -> impl Serve {
    |target: ModelId, _request: Request| async move { Ok(reply(target)) }
}

/// A buffered response whose completion text is `completion`.
pub(crate) fn reply(completion: impl Into<String>) -> Response {
    Response {
        llm_response: LlmResponse::Agg(text_response(None, completion.into())),
        metadata: None,
    }
}
