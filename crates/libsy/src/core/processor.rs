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
    use crate::StateValue;
    use switchyard_protocol::{text_request, text_response};

    /// The `State::extra` key each event variant tallies under.
    fn event_key(event: &Event<'_>) -> &'static str {
        match event {
            Event::Request(_) => "requests",
            Event::Signal(_) => "signals",
            Event::Decision(_) => "decisions",
            Event::ModelRequest(_) => "model_requests",
            Event::ModelResponse(_) => "model_responses",
        }
    }

    /// Reads a `StateValue::Count` from `extra`, treating a missing key as zero.
    fn count(state: &State, key: &str) -> u32 {
        match state.extra.get(key) {
            Some(StateValue::Count(n)) => *n,
            _ => 0,
        }
    }

    /// Tallies each event variant under its own key in [`State::extra`].
    struct CountingProcessor;

    #[async_trait]
    impl Processor for CountingProcessor {
        async fn process(&self, state: &mut State, event: Event<'_>) -> Result<(), BoxErr> {
            let entry = state
                .extra
                .entry(event_key(&event).to_string())
                .or_insert(StateValue::Count(0));
            if let StateValue::Count(n) = entry {
                *n += 1;
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

        assert_eq!(count(&state, "requests"), 1);
        assert_eq!(count(&state, "signals"), 1);
        assert_eq!(count(&state, "decisions"), 1);
        assert_eq!(count(&state, "model_requests"), 1);
        assert_eq!(count(&state, "model_responses"), 1);
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

        assert_eq!(count(&state, "requests"), 3);
        Ok(())
    }
}
