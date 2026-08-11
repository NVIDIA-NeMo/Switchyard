// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use async_trait::async_trait;
use switchyard_protocol::{AggLlmResponse, Decision, Request};

/// An event observed by the algorithm. Events are consumed by [`Processor`] to mutate state.
///
/// Request-bearing variants ([`Event::Request`], [`Event::Decision`]) borrow the request
/// mutably, so a processor may rewrite it in place and the edit propagates to the rest of
/// the chain and to the model call. The observation-only variants stay immutable.
pub enum Event<'a> {
    /// The inbound request that begins a turn.
    Request(&'a mut Request),
    /// A routing decision paired with the request that produced it.
    ///
    /// The request is rewritable: a processor may add instructions or notes here that
    /// are bound to the routing outcome (e.g. a model-specific system prompt).
    Decision {
        /// The request, rewritable in place.
        request: &'a mut Request,
        /// The routing decision produced for `request`.
        decision: &'a Decision,
    },
    /// A buffered response received back from a model.
    ModelResponse(&'a AggLlmResponse),
}

/// Collects events as the algorithm runs and mutates the composition's state.
#[async_trait]
pub trait Processor<S = ()>: Send + Sync {
    /// Process an event, accumulating facts into `state`. Request-bearing events
    /// ([`Event::Request`], [`Event::Decision`]) may also be rewritten in place.
    async fn process(&self, state: &mut S, event: Event<'_>) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use switchyard_protocol::{text_request, text_response};

    type TestState = HashMap<&'static str, u32>;

    /// The key each event variant tallies under.
    fn event_key(event: &Event<'_>) -> &'static str {
        match event {
            Event::Request(_) => "requests",
            Event::Decision { .. } => "decisions",
            Event::ModelResponse(_) => "model_responses",
        }
    }

    /// Reads a count, treating a missing key as zero.
    fn count(state: &TestState, key: &'static str) -> u32 {
        state.get(key).copied().unwrap_or_default()
    }

    /// Tallies each event variant under its own key.
    struct CountingProcessor;

    #[async_trait]
    impl Processor<TestState> for CountingProcessor {
        async fn process(&self, state: &mut TestState, event: Event<'_>) -> Result<()> {
            *state.entry(event_key(&event)).or_default() += 1;
            Ok(())
        }
    }

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        }
    }

    #[tokio::test]
    async fn processor_tallies_each_event_variant_into_state() -> Result<()> {
        let processor = CountingProcessor;
        let mut state = TestState::default();
        let mut req = request();
        let response = text_response(None, "ok");
        let decision = Decision::new("test/model", None, true);
        // Feed one of every event variant through the processor.
        processor
            .process(&mut state, Event::Request(&mut req))
            .await?;
        processor
            .process(&mut state, Event::ModelResponse(&response))
            .await?;
        processor
            .process(
                &mut state,
                Event::Decision {
                    request: &mut req,
                    decision: &decision,
                },
            )
            .await?;
        assert_eq!(count(&state, "requests"), 1);
        assert_eq!(count(&state, "decisions"), 1);
        assert_eq!(count(&state, "model_responses"), 1);
        Ok(())
    }

    #[tokio::test]
    async fn process_accumulates_state_across_repeated_events() -> Result<()> {
        let processor = CountingProcessor;
        let mut state = TestState::default();
        let mut req = request();

        for _ in 0..3 {
            processor
                .process(&mut state, Event::Request(&mut req))
                .await?;
        }

        assert_eq!(count(&state, "requests"), 3);
        Ok(())
    }

    /// Rewrites the requested model on every request-bearing event.
    struct RewritingProcessor;

    #[async_trait]
    impl Processor for RewritingProcessor {
        async fn process(&self, _state: &mut (), event: Event<'_>) -> Result<()> {
            match event {
                Event::Request(request) | Event::Decision { request, .. } => {
                    request.llm_request.model = Some("rewritten".to_string());
                }
                _ => {}
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn processor_rewrites_the_request_in_place() -> Result<()> {
        let mut state = ();
        let mut req = request();
        assert_eq!(req.requested_model(), Some("auto"));

        RewritingProcessor
            .process(&mut state, Event::Request(&mut req))
            .await?;

        // The edit outlives the call, so the next component sees the rewritten request.
        assert_eq!(req.requested_model(), Some("rewritten"));
        Ok(())
    }
}
