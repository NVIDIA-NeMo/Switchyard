// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request/response envelope: the normalized [`LlmRequest`]/[`LlmResponse`] paired
//! with the original provider payload and correlation [`Metadata`].

use crate::{LlmRequest, LlmResponse, WireFormat};
use std::collections::HashMap;

/// Per-request state threaded to an algorithm alongside a request. A placeholder
/// for cross-cutting state (correlation ids, budgets, deadlines) an algorithm will
/// read; empty today.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Context {
    pub values: HashMap<String, String>,
}

/// Correlation and routing metadata attached to a request or response.
///
/// All fields are optional; algorithms and observers use whichever are present
/// (e.g. to key per-session state or emit correlated telemetry). `extra_metadata`
/// is a free-form escape hatch for host-specific keys.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Metadata {
    /// Stable id for a multi-request session/conversation.
    pub session_id: Option<String>,
    /// Id of the agent making the request.
    pub agent_id: Option<String>,
    /// Id of the task the request belongs to.
    pub task_id: Option<String>,
    /// Agent-specific lineage and semantic routing signals.
    pub agent_context: Option<Box<AgentContext>>,
    /// External trace/request id for joining with the host's telemetry.
    pub correlation_id: Option<String>,
    /// Arbitrary host-defined key/value metadata.
    pub extra_metadata: Option<std::collections::BTreeMap<String, String>>,
    /// HTTP headers to attach when forwarding the request/response, if any.
    pub http_headers: Option<std::collections::BTreeMap<String, String>>,
    /// The wire format the request/response was originally encoded in, if known.
    pub wire_format: Option<WireFormat>,
}

/// Optional lineage and semantic signals for affinity-aware routing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AgentContext {
    /// Whether the harness explicitly identified this request as coming from a child agent.
    pub is_subagent: bool,
    /// Id of the parent agent, when this request comes from a child agent.
    pub parent_agent_id: Option<String>,
    /// Harness-defined kind of agent call, such as `collab_spawn` or `review`.
    pub agent_kind: Option<String>,
    /// Semantic agent role, such as `explorer`, `worker`, or `reviewer`.
    pub agent_role: Option<String>,
    /// Semantic task class supplied by the harness.
    pub task_kind: Option<String>,
    /// Id of the current agent turn.
    pub turn_id: Option<String>,
}

/// A request an algorithm routes: the normalized [`LlmRequest`] plus the original
/// provider payload and correlation [`Metadata`].
#[derive(Clone)]
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
