// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal append-only routing traces.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

use crate::{ProxyContext, RequestId};

/// One algorithm-authored routing event with framework-assigned metadata.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoutingEvent {
    /// Zero-based order within the request trace.
    sequence: u64,
    /// Event creation time as Unix epoch milliseconds.
    timestamp_ms: i64,
    /// Stable event name chosen by the routing algorithm.
    name: String,
    /// Arbitrary JSON payload owned by the routing algorithm.
    payload: Value,
}

impl RoutingEvent {
    /// Returns the zero-based order within the request trace.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Returns the event creation time as Unix epoch milliseconds.
    pub fn timestamp_ms(&self) -> i64 {
        self.timestamp_ms
    }

    /// Returns the algorithm-authored event name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the algorithm-authored JSON payload.
    pub fn payload(&self) -> &Value {
        &self.payload
    }
}

/// Ordered routing events for one inbound request.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoutingTrace {
    /// Opaque request identifier used for external correlation.
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<RequestId>,
    /// Events in append order.
    events: Vec<RoutingEvent>,
}

impl RoutingTrace {
    /// Creates an empty trace for an optional request identifier.
    pub fn new(request_id: Option<RequestId>) -> Self {
        Self {
            request_id,
            events: Vec::new(),
        }
    }

    /// Returns the opaque request identifier used for correlation.
    pub fn request_id(&self) -> Option<&RequestId> {
        self.request_id.as_ref()
    }

    /// Returns the events in append order.
    pub fn events(&self) -> &[RoutingEvent] {
        &self.events
    }

    /// Appends an event after assigning its sequence and timestamp.
    pub fn record(
        &mut self,
        name: String,
        payload: Value,
    ) -> Result<&RoutingEvent, RoutingTraceError> {
        self.record_at(name, payload, unix_timestamp_ms())
    }

    fn record_at(
        &mut self,
        name: String,
        payload: Value,
        timestamp_ms: i64,
    ) -> Result<&RoutingEvent, RoutingTraceError> {
        if name.trim().is_empty() {
            return Err(RoutingTraceError::EmptyName);
        }
        self.events.push(RoutingEvent {
            sequence: self.events.len() as u64,
            timestamp_ms,
            name,
            payload,
        });
        self.events
            .last()
            .ok_or(RoutingTraceError::StorageUnavailable)
    }
}

impl ProxyContext {
    /// Appends one routing event without changing routing state.
    pub fn record_routing_event(
        &mut self,
        name: String,
        payload: Value,
    ) -> Result<&RoutingEvent, RoutingTraceError> {
        let request_id = self.request_id.clone();
        if self.get::<RoutingTrace>().is_none() {
            self.insert(RoutingTrace::new(request_id.clone()));
        }
        let trace = self
            .get_mut::<RoutingTrace>()
            .ok_or(RoutingTraceError::StorageUnavailable)?;
        trace.request_id = request_id;
        trace.record(name, payload)
    }

    /// Returns the routing trace recorded for this request, when present.
    pub fn routing_trace(&self) -> Option<&RoutingTrace> {
        self.get::<RoutingTrace>()
    }

    /// Removes and returns the routing trace recorded for this request.
    pub fn take_routing_trace(&mut self) -> Option<RoutingTrace> {
        self.remove::<RoutingTrace>()
    }
}

/// Errors produced by the minimal routing trace envelope.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RoutingTraceError {
    /// Event names must be useful to consumers.
    #[error("routing event name must not be empty")]
    EmptyName,
    /// The typed trace extension could not be recovered after initialization.
    #[error("routing trace storage is unavailable")]
    StorageUnavailable,
}

fn unix_timestamp_ms() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => i64::try_from(duration.as_millis()).unwrap_or(i64::MAX),
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn trace_adds_only_framework_metadata() -> Result<(), RoutingTraceError> {
        let mut trace = RoutingTrace::new(Some(RequestId::from_static("request-1")));
        trace.record_at("classifier.input".to_string(), json!({"prompt": "..."}), 10)?;
        trace.record_at(
            "classifier.output".to_string(),
            json!({"tier": "strong"}),
            20,
        )?;

        assert_eq!(trace.events()[0].sequence(), 0);
        assert_eq!(trace.events()[1].sequence(), 1);
        assert_eq!(trace.events()[1].name(), "classifier.output");
        assert_eq!(trace.events()[1].payload(), &json!({"tier": "strong"}));
        Ok(())
    }

    #[test]
    fn proxy_context_keeps_trace_separate_from_routing_state() -> Result<(), RoutingTraceError> {
        let mut ctx = ProxyContext::with_request_id(RequestId::from_static("request-1"));
        ctx.record_routing_event("router.decision".to_string(), json!({"tier": "strong"}))?;

        assert_eq!(
            ctx.routing_trace().and_then(RoutingTrace::request_id),
            Some(&RequestId::from_static("request-1")),
        );
        assert!(ctx.selected_target().is_none());
        assert!(ctx.take_routing_trace().is_some());
        assert!(ctx.routing_trace().is_none());
        Ok(())
    }

    #[test]
    fn empty_name_is_rejected() {
        let mut trace = RoutingTrace::new(None);
        assert_eq!(
            trace.record("  ".to_string(), Value::Null),
            Err(RoutingTraceError::EmptyName),
        );
    }
}
