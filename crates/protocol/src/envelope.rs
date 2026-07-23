// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request/response envelope: the normalized [`LlmRequest`]/[`LlmResponse`] paired
//! with the original provider payload and correlation [`Metadata`].

use crate::{LlmRequest, LlmResponse, Metadata};
use std::collections::HashMap;

/// [`Context`] for an algorithm or llm client.
/// This struct is fed through the entire process and provides unified information to each step
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Context<S = ()> {
    /// caller specific values
    pub values: HashMap<String, String>,
    /// Caller-defined per-session state; `()` when a consumer carries no session state.
    pub state: S,
}

impl<S> Context<S> {
    /// Create a new [`Context`] with the given state and an empty values map.
    pub fn without_state(&self) -> Context<()> {
        Context {
            values: self.values.clone(),
            state: (),
        }
    }
}

/// Agentic-stack events fed to an algorithm out of band (e.g. tool results, budget
/// updates) — in libsy, via `Algorithm::process_signals`.
///
/// A placeholder today; a stateful algorithm can begin consuming signals as the enum
/// grows without changing the orchestrator contract.
#[derive(Clone)]
pub struct Signals {}

/// A request an algorithm routes: the normalized [`LlmRequest`] plus the original
/// provider payload and correlation [`Metadata`].
#[derive(Clone, Default)]
pub struct Request {
    /// The normalized request an algorithm routes.
    pub llm_request: LlmRequest,
    /// The original provider-shaped request body, if the host wants to forward it
    /// verbatim (e.g. a proxy preserving messages/params). libsy does not read it.
    pub raw_request: Option<serde_json::Value>,
    /// Correlation metadata carried through the request.
    pub metadata: Option<Metadata>,
}

impl Request {
    pub fn requested_model(&self) -> Option<&str> {
        self.llm_request.model.as_deref()
    }
}

/// A response an algorithm returns: the [`LlmResponse`] (streamed or aggregate) plus
/// optional correlation [`Metadata`].
///
/// Not `Clone` — `llm_response` may own a live stream.
pub struct Response {
    /// The neutral model response — a chunk stream or the buffered aggregate.
    pub llm_response: LlmResponse,
    /// Correlation metadata carried through the response.
    pub metadata: Option<Metadata>,
}

impl Response {
    pub fn selected_model(&self) -> Option<&str> {
        self.llm_response.selected_model()
    }
}
