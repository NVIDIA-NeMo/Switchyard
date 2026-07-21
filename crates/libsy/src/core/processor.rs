// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::State;
use async_trait::async_trait;
use switchyard_protocol::{AggLlmResponse, Decision, Request, Signals};

/// An event observed by the algorithm. Events are consumed by [`Processor`] to mutate [`State`]
pub enum Event<'a> {
    /// The inbound request that begins a turn.
    Request(&'a Request),
    /// An out-of-band agentic-stack signal (tool results, budget updates, …).
    Signal(&'a Signals),
    /// A routing decision the algorithm just made.
    Decision(&'a dyn Decision),
    /// A request about to be sent to a model.
    ModelRequest(&'a Request),
    /// A buffered response received back from a model.
    ModelResponse(&'a AggLlmResponse),
}

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Collects events as the algorithm runs and mutates [`State`]
#[async_trait]
pub trait Processor: Send + Sync {
    /// Process an event, accumulating facts into `state`.
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<(), BoxErr>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{text_request, text_response};

    /// Per-variant tally the test processor accumulates into `State`.
    #[derive(Default, Debug, PartialEq)]
    struct Counts {
        requests: u32,
        signals: u32,
        decisions: u32,
        model_requests: u32,
        model_responses: u32,
    }

    /// Records every event it observes into a `Counts` kept in `State`.
    struct CountingProcessor;

    #[async_trait]
    impl Processor for CountingProcessor {
        async fn process(&self, state: &mut State, event: Event<'_>) -> Result<(), BoxErr> {
            let counts = state.entry_or_insert_with(Counts::default);
            match event {
                Event::Request(_) => counts.requests += 1,
                Event::Signal(_) => counts.signals += 1,
                Event::Decision(_) => counts.decisions += 1,
                Event::ModelRequest(_) => counts.model_requests += 1,
                Event::ModelResponse(_) => counts.model_responses += 1,
            }
            Ok(())
        }
    }

    /// Minimal [`Decision`] so an `Event::Decision` can be constructed.
    struct TestDecision;

    impl Decision for TestDecision {
        fn selected_model(&self) -> &str {
            "test/model"
        }
        fn reasoning(&self) -> Option<&str> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
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
    async fn processor_tallies_each_event_variant_into_state() -> Result<(), BoxErr> {
        let processor = CountingProcessor;
        let mut state = State::default();
        let req = request();
        let response = text_response(None, "ok");
        let decision = TestDecision;
        let signals = Signals {};

        // Feed one of every event variant through the processor.
        processor.process(&mut state, Event::Request(&req)).await?;
        processor
            .process(&mut state, Event::ModelRequest(&req))
            .await?;
        processor
            .process(&mut state, Event::ModelResponse(&response))
            .await?;
        processor
            .process(&mut state, Event::Decision(&decision))
            .await?;
        processor
            .process(&mut state, Event::Signal(&signals))
            .await?;

        assert_eq!(
            state.get::<Counts>(),
            Some(&Counts {
                requests: 1,
                signals: 1,
                decisions: 1,
                model_requests: 1,
                model_responses: 1,
            })
        );
        Ok(())
    }

    #[tokio::test]
    async fn process_accumulates_state_across_repeated_events() -> Result<(), BoxErr> {
        let processor = CountingProcessor;
        let mut state = State::default();
        let req = request();

        for _ in 0..3 {
            processor.process(&mut state, Event::Request(&req)).await?;
        }

        assert_eq!(state.get::<Counts>().map(|c| c.requests), Some(3));
        Ok(())
    }
}
