// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Append-only audit records for routing evaluation and execution.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::{ProxyContext, RequestId};

/// Current serialized routing-trace schema version.
pub const ROUTING_TRACE_SCHEMA_VERSION: u16 = 1;

/// Stable lifecycle categories shared by all routing algorithms.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingEventKind {
    /// The request or other source data entering routing.
    Input,
    /// A deterministic transformation of earlier trace data.
    Transform,
    /// A scoring, classification, or candidate-evaluation step.
    Evaluation,
    /// A normalized route selection.
    Decision,
    /// One concrete backend attempt made for a decision.
    Attempt,
    /// The terminal request or stream outcome.
    Outcome,
}

/// Optional result status for an evaluation, attempt, or outcome event.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingEventStatus {
    /// The recorded operation completed successfully.
    Success,
    /// The recorded operation failed.
    Error,
    /// The operation was intentionally skipped.
    Skipped,
    /// The operation was cancelled before completion.
    Cancelled,
}

/// Common route-selection fields used for cross-router analysis.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingSelection {
    /// Capability or policy tier selected by the router.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    /// Concrete target identifier selected for dispatch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Upstream model selected for dispatch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl RoutingSelection {
    /// Returns true when the selection identifies no tier, target, or model.
    pub fn is_empty(&self) -> bool {
        self.tier.is_none() && self.target.is_none() && self.model.is_none()
    }
}

/// Router-authored event data before sequence and timestamp assignment.
///
/// Routing implementations own the JSON-compatible `input`, `output`, and
/// `details` payloads. The remaining fields form the stable envelope used by
/// exporters and benchmark analysis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingEventData {
    /// Lifecycle category for this event.
    pub kind: RoutingEventKind,
    /// Component or algorithm that produced the event.
    pub producer: String,
    /// Human-readable, stable step name within the producer.
    pub name: String,
    /// Versioned schema identifier for router-specific payload fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    /// Earlier event sequences that directly informed this event.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub based_on: Vec<u64>,
    /// Decision phase such as `initial`, `retry`, or `fallback`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    /// Router-specific input captured for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// Router-specific output produced by this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// Normalized selection made or attempted by this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection: Option<RoutingSelection>,
    /// Stable machine-readable decision source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Optional human-readable explanation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Optional router confidence in the range `[0, 1]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Wall-clock duration for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    /// Optional operation result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<RoutingEventStatus>,
    /// Sanitized error category or message when the step failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Additional versioned evidence that does not fit the common envelope.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

/// One immutable event in a per-request routing trace.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoutingEvent {
    /// Zero-based order within the request trace.
    sequence: u64,
    /// Event creation time as Unix epoch milliseconds.
    timestamp_ms: i64,
    /// Router-authored event fields.
    #[serde(flatten)]
    data: RoutingEventData,
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

    /// Returns the router-authored event fields.
    pub fn data(&self) -> &RoutingEventData {
        &self.data
    }
}

/// Append-only routing audit for one inbound request.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoutingTrace {
    /// Serialized schema version.
    schema_version: u16,
    /// Opaque request identifier used for external correlation.
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<RequestId>,
    /// Ordered routing lifecycle events.
    events: Vec<RoutingEvent>,
}

impl RoutingTrace {
    /// Creates an empty trace for an optional request identifier.
    pub fn new(request_id: Option<RequestId>) -> Self {
        Self {
            schema_version: ROUTING_TRACE_SCHEMA_VERSION,
            request_id,
            events: Vec::new(),
        }
    }

    /// Returns the serialized schema version.
    pub fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Returns the opaque request identifier used for correlation.
    pub fn request_id(&self) -> Option<&RequestId> {
        self.request_id.as_ref()
    }

    /// Returns the events in append order.
    pub fn events(&self) -> &[RoutingEvent] {
        &self.events
    }

    /// Validates and appends an event, assigning its sequence and timestamp.
    pub fn record(&mut self, data: RoutingEventData) -> Result<&RoutingEvent, RoutingTraceError> {
        self.record_at(data, unix_timestamp_ms())
    }

    /// Returns the most recent normalized routing decision.
    pub fn final_decision(&self) -> Option<&RoutingEvent> {
        self.events
            .iter()
            .rev()
            .find(|event| event.data().kind == RoutingEventKind::Decision)
    }

    /// Returns true when no events have been recorded.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    fn record_at(
        &mut self,
        data: RoutingEventData,
        timestamp_ms: i64,
    ) -> Result<&RoutingEvent, RoutingTraceError> {
        validate_event(&data, self.events.len() as u64)?;
        let event = RoutingEvent {
            sequence: self.events.len() as u64,
            timestamp_ms,
            data,
        };
        self.events.push(event);
        self.events
            .last()
            .ok_or(RoutingTraceError::StorageUnavailable)
    }
}

impl ProxyContext {
    /// Appends one audit event without changing profile routing state.
    pub fn record_routing_event(
        &mut self,
        data: RoutingEventData,
    ) -> Result<&RoutingEvent, RoutingTraceError> {
        let request_id = self.request_id.clone();
        if self.get::<RoutingTrace>().is_none() {
            self.insert(RoutingTrace::new(request_id.clone()));
        }
        let trace = self
            .get_mut::<RoutingTrace>()
            .ok_or(RoutingTraceError::StorageUnavailable)?;
        trace.request_id = request_id;
        trace.record(data)
    }

    /// Returns the routing audit recorded for this request, when present.
    pub fn routing_trace(&self) -> Option<&RoutingTrace> {
        self.get::<RoutingTrace>()
    }

    /// Removes and returns the routing audit recorded for this request.
    pub fn take_routing_trace(&mut self) -> Option<RoutingTrace> {
        self.remove::<RoutingTrace>()
    }
}

/// Validation errors for malformed routing audit events.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RoutingTraceError {
    /// A required string field was empty or whitespace.
    #[error("routing event {field} must not be empty")]
    EmptyField { field: &'static str },
    /// An event referenced itself or a future event.
    #[error("routing event references unavailable sequence {sequence}")]
    InvalidReference { sequence: u64 },
    /// Decision events must identify at least one selected route dimension.
    #[error("routing decision must select a tier, target, or model")]
    MissingDecisionSelection,
    /// Decision events must provide a stable machine-readable source.
    #[error("routing decision source must not be empty")]
    MissingDecisionSource,
    /// Confidence must be finite and within `[0, 1]`.
    #[error("routing event confidence must be finite and between 0 and 1")]
    InvalidConfidence,
    /// Duration must be finite and non-negative.
    #[error("routing event duration_ms must be finite and non-negative")]
    InvalidDuration,
    /// Router-specific payloads require a stable versioned schema identifier.
    #[error("routing event schema is required when input, output, or details are present")]
    MissingPayloadSchema,
    /// The typed trace extension could not be recovered after initialization.
    #[error("routing trace storage is unavailable")]
    StorageUnavailable,
}

fn validate_event(data: &RoutingEventData, next_sequence: u64) -> Result<(), RoutingTraceError> {
    validate_nonempty("producer", &data.producer)?;
    validate_nonempty("name", &data.name)?;
    if let Some(schema) = data.schema.as_deref() {
        validate_nonempty("schema", schema)?;
    }
    if (data.input.is_some() || data.output.is_some() || data.details.is_some())
        && data.schema.is_none()
    {
        return Err(RoutingTraceError::MissingPayloadSchema);
    }
    if let Some(phase) = data.phase.as_deref() {
        validate_nonempty("phase", phase)?;
    }
    for sequence in &data.based_on {
        if *sequence >= next_sequence {
            return Err(RoutingTraceError::InvalidReference {
                sequence: *sequence,
            });
        }
    }
    if let Some(confidence) = data.confidence {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(RoutingTraceError::InvalidConfidence);
        }
    }
    if let Some(duration_ms) = data.duration_ms {
        if !duration_ms.is_finite() || duration_ms < 0.0 {
            return Err(RoutingTraceError::InvalidDuration);
        }
    }
    if data.kind == RoutingEventKind::Decision {
        let Some(selection) = data.selection.as_ref() else {
            return Err(RoutingTraceError::MissingDecisionSelection);
        };
        validate_selection(selection)?;
        if data
            .source
            .as_deref()
            .is_none_or(|source| source.trim().is_empty())
        {
            return Err(RoutingTraceError::MissingDecisionSource);
        }
    }
    Ok(())
}

fn validate_selection(selection: &RoutingSelection) -> Result<(), RoutingTraceError> {
    if selection.is_empty() {
        return Err(RoutingTraceError::MissingDecisionSelection);
    }
    for (field, value) in [
        ("selection.tier", selection.tier.as_deref()),
        ("selection.target", selection.target.as_deref()),
        ("selection.model", selection.model.as_deref()),
    ] {
        if let Some(value) = value {
            validate_nonempty(field, value)?;
        }
    }
    Ok(())
}

fn validate_nonempty(field: &'static str, value: &str) -> Result<(), RoutingTraceError> {
    if value.trim().is_empty() {
        return Err(RoutingTraceError::EmptyField { field });
    }
    Ok(())
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

    fn event(kind: RoutingEventKind, name: &str) -> RoutingEventData {
        RoutingEventData {
            kind,
            producer: "test-router".to_string(),
            name: name.to_string(),
            schema: None,
            based_on: Vec::new(),
            phase: None,
            input: None,
            output: None,
            selection: None,
            source: None,
            reason: None,
            confidence: None,
            duration_ms: None,
            status: None,
            error: None,
            details: None,
        }
    }

    #[test]
    fn trace_assigns_sequence_and_serializes_router_payloads(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut trace = RoutingTrace::new(Some(RequestId::from_static("request-1")));
        let mut input = event(RoutingEventKind::Input, "request");
        input.schema = Some("test.request/v1".to_string());
        input.output = Some(json!({"model": "switchyard"}));
        let first_sequence = trace.record_at(input, 1_700_000_000_000)?.sequence();

        let mut evaluation = event(RoutingEventKind::Evaluation, "score");
        evaluation.schema = Some("test.score/v1".to_string());
        evaluation.based_on = vec![first_sequence];
        evaluation.output = Some(json!({"score": 0.75}));
        evaluation.duration_ms = Some(12.5);
        let second_sequence = trace.record_at(evaluation, 1_700_000_000_100)?.sequence();

        assert_eq!(first_sequence, 0);
        assert_eq!(second_sequence, 1);
        assert_eq!(trace.events()[1].data().based_on, vec![0]);
        assert_eq!(
            trace.events()[1].data().output,
            Some(json!({"score": 0.75}))
        );
        let encoded = serde_json::to_value(&trace)?;
        assert_eq!(encoded["schema_version"], ROUTING_TRACE_SCHEMA_VERSION);
        assert_eq!(encoded["events"][0]["sequence"], 0);
        assert_eq!(encoded["events"][1]["output"], json!({"score": 0.75}));
        Ok(())
    }

    #[test]
    fn final_decision_returns_latest_append_only_selection() -> Result<(), RoutingTraceError> {
        let mut trace = RoutingTrace::new(None);
        let mut initial = event(RoutingEventKind::Decision, "select");
        initial.phase = Some("initial".to_string());
        initial.selection = Some(RoutingSelection {
            tier: Some("weak".to_string()),
            ..RoutingSelection::default()
        });
        initial.source = Some("weighted_draw".to_string());
        trace.record_at(initial, 1)?;

        let mut fallback = event(RoutingEventKind::Decision, "select");
        fallback.phase = Some("context_overflow_fallback".to_string());
        fallback.based_on = vec![0];
        fallback.selection = Some(RoutingSelection {
            tier: Some("strong".to_string()),
            model: Some("large-context-model".to_string()),
            target: None,
        });
        fallback.source = Some("context_overflow".to_string());
        trace.record_at(fallback, 2)?;

        let final_decision = trace.final_decision();
        assert_eq!(final_decision.map(RoutingEvent::sequence), Some(1));
        assert_eq!(trace.events().len(), 2);
        Ok(())
    }

    #[test]
    fn decision_validation_rejects_missing_selection_and_future_references() {
        let mut trace = RoutingTrace::new(None);
        let mut decision = event(RoutingEventKind::Decision, "select");
        decision.source = Some("policy".to_string());
        assert_eq!(
            trace.record_at(decision, 1),
            Err(RoutingTraceError::MissingDecisionSelection),
        );

        let mut input = event(RoutingEventKind::Input, "request");
        input.based_on = vec![0];
        assert_eq!(
            trace.record_at(input, 1),
            Err(RoutingTraceError::InvalidReference { sequence: 0 }),
        );

        let mut output_without_schema = event(RoutingEventKind::Evaluation, "score");
        output_without_schema.output = Some(json!({"score": 0.5}));
        assert_eq!(
            trace.record_at(output_without_schema, 1),
            Err(RoutingTraceError::MissingPayloadSchema),
        );
    }

    #[test]
    fn proxy_context_keeps_trace_separate_from_routing_state() -> Result<(), RoutingTraceError> {
        let mut ctx = ProxyContext::with_request_id(RequestId::from_static("request-1"));
        let mut decision = event(RoutingEventKind::Decision, "select");
        decision.selection = Some(RoutingSelection {
            tier: Some("strong".to_string()),
            ..RoutingSelection::default()
        });
        decision.source = Some("policy".to_string());

        let recorded_sequence = ctx.record_routing_event(decision)?.sequence();
        assert_eq!(recorded_sequence, 0);
        assert_eq!(
            ctx.routing_trace().and_then(RoutingTrace::request_id),
            Some(&RequestId::from_static("request-1")),
        );
        assert!(ctx.selected_target().is_none());
        assert!(ctx.take_routing_trace().is_some());
        assert!(ctx.routing_trace().is_none());
        Ok(())
    }
}
