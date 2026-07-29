// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request/response envelope: the normalized [`LlmRequest`]/[`LlmResponse`] paired
//! with the original provider payload and correlation [`Metadata`].

use crate::{LlmRequest, LlmResponse, Metadata};
use std::collections::{HashMap, HashSet};

/// Cross-cutting request context passed through algorithms and LLM clients.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Context {
    /// Caller-specific values propagated through the request.
    pub values: HashMap<String, String>,
    /// Targets this request must not route to. A caller may set these up front to keep a
    /// request off a given target; routing also adds any target that overflows its context
    /// window mid-request.
    excluded_targets: HashSet<String>,
}

impl Context {
    /// Excludes a target from this request, returning whether it was newly excluded.
    /// Re-excluding returns `false`, which is what bounds the overflow retry once every
    /// target has been tried.
    pub fn exclude_target(&mut self, target: impl Into<String>) -> bool {
        self.excluded_targets.insert(target.into())
    }

    /// Whether this request is barred from routing to `target`.
    pub fn is_excluded(&self, target: &str) -> bool {
        self.excluded_targets.contains(target)
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
