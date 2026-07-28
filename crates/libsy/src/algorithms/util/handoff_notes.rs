// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage-router handoff notes.
//!
//! When a turn escalates to the strong tier (or hands back to the weak tier),
//! a short **deterministic** note can be handed to the model taking over so it
//! knows *why* — without re-diagnosing (weak→strong) or re-architecting settled
//! work (strong→weak). This is not a model call; the note text comes from config.
//!
//! [`HandoffNoteProcessor`] reacts to [`Event::Decision`] (replayed after the
//! tier is chosen, before the model call), derives escalate vs de-escalate from
//! the chosen tier plus the `decision_source` the picker stashed, and records
//! the matching note on the [`State`] under [`HANDOFF_NOTE_KEY`]. A later
//! request-injection step splices it into the outbound request — that step is
//! intentionally not wired here.

use async_trait::async_trait;

use crate::{Event, Processor, Result, State, StateValue};

/// `State.extra` key under which the computed handoff note is recorded for a
/// later request-injection step to consume.
pub const HANDOFF_NOTE_KEY: &str = "handoff_note";

/// Records a handoff note on the decision, based on the chosen tier and the
/// picker's `decision_source`. Deterministic — no model call.
pub struct HandoffNoteProcessor {
    escalation_note: String,
    deescalation_note: Option<String>,
    only_on_wrong_signal_escalation: bool,
}

impl HandoffNoteProcessor {
    /// Configure the notes: the `escalation_note` handed to the strong tier, an
    /// optional `deescalation_note` handed back to the weak tier, and whether
    /// the escalation note fires only on a signal-driven escalation
    /// (`override` / `dimensions`) rather than a `fall_open` default.
    pub fn new(
        escalation_note: impl Into<String>,
        deescalation_note: Option<String>,
        only_on_wrong_signal_escalation: bool,
    ) -> Self {
        Self {
            escalation_note: escalation_note.into(),
            deescalation_note,
            only_on_wrong_signal_escalation,
        }
    }

    /// The note to record for a decision routed to `tier` with picker `source`,
    /// or `None` when no note applies.
    fn note_for(&self, tier: &str, source: Option<&str>) -> Option<String> {
        match tier {
            // Escalation to the strong tier. When gated, only a signal-driven
            // escalation qualifies — never a `fall_open` default.
            "strong" => {
                let signal_driven = matches!(source, Some("override") | Some("dimensions"));
                (!self.only_on_wrong_signal_escalation || signal_driven)
                    .then(|| self.escalation_note.clone())
            }
            // Hand-back to the weak tier, when a de-escalation note is configured.
            "weak" => self.deescalation_note.clone(),
            _ => None,
        }
    }
}

#[async_trait]
impl Processor for HandoffNoteProcessor {
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<()> {
        let Event::Decision(decision) = event else {
            return Ok(());
        };
        let tier = decision.selected_model().to_string();
        // Read (and release the borrow on) the source the picker stashed.
        let source = match state.extra.get("decision_source") {
            Some(StateValue::String(source)) => Some(source.clone()),
            _ => None,
        };
        if let Some(note) = self.note_for(&tier, source.as_deref()) {
            state
                .extra
                .insert(HANDOFF_NOTE_KEY.to_string(), StateValue::String(note));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Decision;

    struct FakeDecision(&'static str);
    impl Decision for FakeDecision {
        fn selected_model(&self) -> &str {
            self.0
        }
        fn reasoning(&self) -> Option<&str> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn state_with_source(source: &str) -> State {
        let mut state = State::default();
        state.extra.insert(
            "decision_source".to_string(),
            StateValue::String(source.to_string()),
        );
        state
    }

    async fn run(processor: &HandoffNoteProcessor, state: &mut State, tier: &'static str) {
        processor
            .process(state, Event::Decision(&FakeDecision(tier)))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn escalation_note_recorded_on_signal_driven_strong() {
        let processor = HandoffNoteProcessor::new("recovering from an error", None, true);
        for source in ["override", "dimensions"] {
            let mut state = state_with_source(source);
            run(&processor, &mut state, "strong").await;
            assert!(matches!(
                state.extra.get(HANDOFF_NOTE_KEY),
                Some(StateValue::String(note)) if note == "recovering from an error"
            ));
        }
    }

    #[tokio::test]
    async fn no_escalation_note_on_fall_open_default_when_gated() {
        let processor = HandoffNoteProcessor::new("recovering from an error", None, true);
        let mut state = state_with_source("fall_open");
        run(&processor, &mut state, "strong").await;
        assert!(!state.extra.contains_key(HANDOFF_NOTE_KEY));
    }

    #[tokio::test]
    async fn escalation_note_on_fall_open_when_not_gated() {
        let processor = HandoffNoteProcessor::new("recovering from an error", None, false);
        let mut state = state_with_source("fall_open");
        run(&processor, &mut state, "strong").await;
        assert!(state.extra.contains_key(HANDOFF_NOTE_KEY));
    }

    #[tokio::test]
    async fn deescalation_note_recorded_on_weak_when_configured() {
        let processor =
            HandoffNoteProcessor::new("esc", Some("settled — carry on".to_string()), true);
        let mut state = state_with_source("tests_passed");
        run(&processor, &mut state, "weak").await;
        assert!(matches!(
            state.extra.get(HANDOFF_NOTE_KEY),
            Some(StateValue::String(note)) if note == "settled — carry on"
        ));
    }

    #[tokio::test]
    async fn no_deescalation_note_when_unconfigured() {
        let processor = HandoffNoteProcessor::new("esc", None, true);
        let mut state = state_with_source("tests_passed");
        run(&processor, &mut state, "weak").await;
        assert!(!state.extra.contains_key(HANDOFF_NOTE_KEY));
    }
}
