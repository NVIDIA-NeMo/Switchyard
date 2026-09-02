// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The trigger classifier: decides, from the gate's signals, whether the
//! buffered turn warrants a review, and which trigger claims it.
//!
//! Runs after the executor call, on response-side signals: a request-side
//! trigger would review one turn late for mid-task turns and never for the
//! terminal turn — the premature-"done" moment the gate exists to catch.
//! Pure over its inputs; the once-per-conversation stall latch and the
//! review budget live in the ledger.

use crate::Result;

use super::signals::GateSignals;
use super::{AdvisorGateConfig, GateTrigger, algorithm_error};

/// The trigger with its pattern compiled once at construction.
enum CompiledTrigger {
    NoToolCall,
    Pattern(regex::Regex),
}

/// Classifies buffered turns against the configured trigger and the stall
/// checkpoint threshold.
pub(super) struct TriggerClassifier {
    trigger: CompiledTrigger,
    /// For the `no_tool_call` trigger: tool results the conversation must
    /// carry before a terminal turn is reviewable.
    min_tool_results: u32,
    /// Assistant turns at which the stall checkpoint is reached; 0 disables.
    stall_turns: u32,
}

/// What the classifier concluded about one buffered turn.
pub(super) struct TriggerDecision {
    /// The terminal-turn trigger that fired, as the telemetry `trigger`
    /// label ("no_tool_call" | "pattern"); `None` when the turn is not
    /// terminal.
    pub(super) fired: Option<&'static str>,
    /// The conversation reached the stall checkpoint. Threshold only — the
    /// once-per-conversation latch belongs to the budget ledger.
    pub(super) stalled: bool,
}

impl TriggerClassifier {
    /// Validates and compiles the configured trigger.
    pub(super) fn new(config: &AdvisorGateConfig) -> Result<Self> {
        let trigger = match &config.gate_trigger {
            GateTrigger::NoToolCall => CompiledTrigger::NoToolCall,
            GateTrigger::Pattern(pattern) => {
                if pattern.is_empty() {
                    return Err(algorithm_error(
                        "gate_trigger 'pattern' requires a non-empty gate_trigger_pattern",
                    ));
                }
                CompiledTrigger::Pattern(regex::Regex::new(pattern).map_err(|error| {
                    algorithm_error(format!(
                        "gate_trigger_pattern is not a valid regex: {error}"
                    ))
                })?)
            }
        };
        Ok(Self {
            trigger,
            min_tool_results: config.gate_min_tool_results,
            stall_turns: config.gate_stall_turns,
        })
    }

    /// Classifies the turn: does it warrant a review, and on which trigger?
    pub(super) fn classify(&self, signals: &GateSignals) -> TriggerDecision {
        let fired = match &self.trigger {
            CompiledTrigger::Pattern(pattern) => pattern
                .is_match(signals.turn.visible_text.as_deref().unwrap_or(""))
                .then_some("pattern"),
            CompiledTrigger::NoToolCall => (!signals.turn.has_tool_use
                && signals.conversation.tool_result_count >= self.min_tool_results)
                .then_some("no_tool_call"),
        };
        let stalled =
            self.stall_turns > 0 && signals.conversation.assistant_turn_count >= self.stall_turns;
        TriggerDecision { fired, stalled }
    }
}

#[cfg(test)]
mod tests {
    use super::super::signals::TurnSignals;
    use super::*;
    use crate::algorithms::util::tool_signals::ToolSignals;

    fn classifier(config: AdvisorGateConfig) -> TriggerClassifier {
        TriggerClassifier::new(&config).expect("test config is valid")
    }

    fn signals(
        has_tool_use: bool,
        visible_text: Option<&str>,
        tool_results: u32,
        assistant_turns: u32,
    ) -> GateSignals {
        GateSignals {
            conversation: ToolSignals {
                tool_result_count: tool_results,
                assistant_turn_count: assistant_turns,
                ..ToolSignals::default()
            },
            turn: TurnSignals {
                has_tool_use,
                visible_text: visible_text.map(str::to_string),
            },
        }
    }

    #[test]
    fn no_tool_call_fires_on_tool_less_turns_past_the_guard() {
        let classifier = classifier(AdvisorGateConfig {
            gate_min_tool_results: 2,
            ..AdvisorGateConfig::default()
        });
        // A tool-less turn under the guard stays quiet; past it, it fires.
        assert!(
            classifier
                .classify(&signals(false, None, 1, 0))
                .fired
                .is_none()
        );
        assert_eq!(
            classifier.classify(&signals(false, None, 2, 0)).fired,
            Some("no_tool_call")
        );
        // Tool use exempts the turn regardless of the guard.
        assert!(
            classifier
                .classify(&signals(true, None, 5, 0))
                .fired
                .is_none()
        );
    }

    #[test]
    fn pattern_reads_text_only_and_ignores_tool_use() {
        let classifier = classifier(AdvisorGateConfig {
            gate_trigger: GateTrigger::Pattern("task_complete".to_string()),
            ..AdvisorGateConfig::default()
        });
        assert_eq!(
            classifier
                .classify(&signals(true, Some("task_complete: true"), 0, 0))
                .fired,
            Some("pattern")
        );
        assert!(
            classifier
                .classify(&signals(false, Some("still working"), 0, 0))
                .fired
                .is_none()
        );
        // No visible text matches against the empty string, not a panic.
        assert!(
            classifier
                .classify(&signals(false, None, 0, 0))
                .fired
                .is_none()
        );
    }

    #[test]
    fn the_stall_threshold_is_reached_at_the_configured_turn_count() {
        let classifier = classifier(AdvisorGateConfig {
            gate_stall_turns: 3,
            ..AdvisorGateConfig::default()
        });
        assert!(!classifier.classify(&signals(true, None, 0, 2)).stalled);
        assert!(classifier.classify(&signals(true, None, 0, 3)).stalled);
    }

    #[test]
    fn a_zero_stall_threshold_disables_the_checkpoint() {
        let classifier = classifier(AdvisorGateConfig::default());
        assert!(!classifier.classify(&signals(true, None, 0, 100)).stalled);
    }
}
