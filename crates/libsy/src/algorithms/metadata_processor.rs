// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Processor that folds a request's harness HTTP headers into correlation [`Metadata`]
//! and records the merged envelope in [`State`] for later processors and classifiers.

use std::collections::BTreeMap;

use async_trait::async_trait;
use switchyard_protocol::Metadata;

use super::core::{Event, Processor, State};

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Derives correlation [`Metadata`] from the inbound request's HTTP headers, merges it
/// with any metadata already attached to the request, and stores the result in [`State`].
pub struct MetadataProcessor;

#[async_trait]
impl Processor for MetadataProcessor {
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<(), BoxErr> {
        // Only the inbound request carries the harness headers we normalize.
        let Event::Request(request) = event else {
            return Ok(());
        };

        // Normalize whatever harness headers the request carries into a neutral envelope.
        let empty = BTreeMap::new();
        let headers = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.http_headers.as_ref())
            .unwrap_or(&empty);
        let derived = Metadata::from_headers(headers);

        // Merge the request's existing metadata over the header-derived envelope; explicit
        // request metadata wins field-by-field, header-derived values fill any gaps.
        let merged = match request.metadata.as_ref() {
            Some(existing) => merge(existing, derived),
            None => derived,
        };

        state.insert(merged);
        Ok(())
    }
}

/// Merges `overlay` beneath `base`: each field takes `base`'s value when present and
/// falls back to `overlay` otherwise. `is_subagent` is a plain flag, so it is true when
/// either source flags it.
fn merge(base: &Metadata, overlay: Metadata) -> Metadata {
    Metadata {
        session_id: base.session_id.clone().or(overlay.session_id),
        agent_id: base.agent_id.clone().or(overlay.agent_id),
        parent_agent_id: base.parent_agent_id.clone().or(overlay.parent_agent_id),
        is_subagent: base.is_subagent || overlay.is_subagent,
        agent_kind: base.agent_kind.clone().or(overlay.agent_kind),
        agent_role: base.agent_role.clone().or(overlay.agent_role),
        task_id: base.task_id.clone().or(overlay.task_id),
        task_kind: base.task_kind.clone().or(overlay.task_kind),
        turn_id: base.turn_id.clone().or(overlay.turn_id),
        session_final: base.session_final.or(overlay.session_final),
        correlation_id: base.correlation_id.clone().or(overlay.correlation_id),
        extra_metadata: base.extra_metadata.clone().or(overlay.extra_metadata),
        http_headers: base.http_headers.clone().or(overlay.http_headers),
        wire_format: base.wire_format.or(overlay.wire_format),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use switchyard_protocol::{LlmRequest, Request};

    /// Builds a request carrying `metadata`, with no LLM payload of interest.
    fn request(metadata: Option<Metadata>) -> Request {
        Request {
            llm_request: LlmRequest::default(),
            raw_request: None,
            metadata,
        }
    }

    /// A metadata envelope whose only content is the given HTTP headers.
    fn metadata_with_headers(pairs: &[(&str, &str)]) -> Metadata {
        Metadata {
            http_headers: Some(
                pairs
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect(),
            ),
            ..Metadata::default()
        }
    }

    #[tokio::test]
    async fn stores_metadata_derived_from_request_headers() -> Result<(), BoxErr> {
        let request = request(Some(metadata_with_headers(&[(
            "x-switchyard-session-id",
            "sess-1",
        )])));

        let mut state = State::default();
        MetadataProcessor
            .process(&mut state, Event::Request(&request))
            .await?;

        let Some(stored) = state.get::<Metadata>() else {
            panic!("expected merged metadata to be stored in state");
        };
        assert_eq!(stored.session_id.as_deref(), Some("sess-1"));
        Ok(())
    }

    #[tokio::test]
    async fn explicit_request_metadata_wins_and_headers_fill_gaps() -> Result<(), BoxErr> {
        // The request already names a session; a header names a different one and also
        // supplies an agent id the request lacks.
        let mut metadata = metadata_with_headers(&[
            ("x-switchyard-session-id", "from-header"),
            ("x-switchyard-agent-id", "agent-from-header"),
        ]);
        metadata.session_id = Some("explicit-session".to_string());
        let request = request(Some(metadata));

        let mut state = State::default();
        MetadataProcessor
            .process(&mut state, Event::Request(&request))
            .await?;

        let Some(stored) = state.get::<Metadata>() else {
            panic!("expected merged metadata to be stored in state");
        };
        // Explicit request value wins over the header-derived one.
        assert_eq!(stored.session_id.as_deref(), Some("explicit-session"));
        // The header-only field fills the gap the request left.
        assert_eq!(stored.agent_id.as_deref(), Some("agent-from-header"));
        Ok(())
    }

    #[tokio::test]
    async fn request_without_metadata_stores_header_derived_only() -> Result<(), BoxErr> {
        let request = request(None);

        let mut state = State::default();
        MetadataProcessor
            .process(&mut state, Event::Request(&request))
            .await?;

        let Some(stored) = state.get::<Metadata>() else {
            panic!("expected metadata to be stored in state");
        };
        assert_eq!(stored.session_id, None);
        Ok(())
    }

    #[tokio::test]
    async fn ignores_non_request_events() -> Result<(), BoxErr> {
        let signals = crate::Signals {};

        let mut state = State::default();
        MetadataProcessor
            .process(&mut state, Event::Signal(&signals))
            .await?;

        assert!(state.get::<Metadata>().is_none());
        Ok(())
    }
}
