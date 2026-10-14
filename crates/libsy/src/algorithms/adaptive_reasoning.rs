// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Turn-level reasoning effort for any routing algorithm.
//!
//! [`AdaptiveReasoning`] wraps an algorithm and sets the reasoning effort of the
//! request it routes. It reads the same tool signals the stage router uses:
//!
//! * By default a turn runs at `medium`, or at the caller's effort when that is lower.
//! * A recovery turn runs at the caller's effort, or `high` when the caller sent none.
//!   A recovery turn is one where a recent tool result failed and fewer than
//!   [`RECOVERY_TURNS`] clean results have followed it, or where the same hard
//!   failure repeated.
//!
//! The caller's effort is a ceiling: a turn never runs above it. A target with its own
//! `reasoning_effort` still overrides the request when the client sends it.

use std::sync::Arc;

use serde_json::{Map, Value, json};
use switchyard_protocol::{Request, WireFormat};

use crate::algorithms::util::tool_signals::ToolSignals;
use crate::{Algorithm, Driver, Result, RoutingOutcome};

/// Effort levels from lowest to highest. A request whose effort is not listed is left as is.
const EFFORTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
/// Effort for turns that are not recovering from a failure.
const DEFAULT_EFFORT: &str = "medium";
/// Effort for recovery turns when the caller sent no effort.
const RECOVERY_EFFORT: &str = "high";
/// Clean tool results after a failure that still count as recovery.
pub const RECOVERY_TURNS: u32 = 2;

/// Wraps an algorithm and sets the reasoning effort of each routed request.
pub struct AdaptiveReasoning {
    inner: Arc<dyn Algorithm>,
}

impl AdaptiveReasoning {
    /// Wraps `inner`. Routing decisions are unchanged; only the routed request's effort is.
    pub fn new(inner: Arc<dyn Algorithm>) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl Algorithm for AdaptiveReasoning {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let mut outcome = self.inner.clone().route(driver, request).await?;
        // An answer produced while routing has already been generated.
        if outcome.response.is_none() {
            let signals = ToolSignals::from_request(&outcome.request, None);
            let caller = outcome.request.llm_request.reasoning.effort.clone();
            if let Some((effort, reason)) = choose_effort(&signals, caller.as_deref()) {
                if caller.as_deref() != Some(effort) {
                    set_effort(&mut outcome.request, effort);
                }
                tracing::debug!(
                    target: "libsy",
                    effort,
                    reason,
                    caller_effort = caller.as_deref().unwrap_or(""),
                    "adaptive reasoning effort"
                );
            }
        }
        Ok(outcome)
    }
}

/// Returns the effort for a turn and why, or `None` to leave the request unchanged.
pub fn choose_effort(
    signals: &ToolSignals,
    caller: Option<&str>,
) -> Option<(&'static str, &'static str)> {
    let rank = |effort: &str| EFFORTS.iter().position(|level| *level == effort);
    let ceiling = match caller {
        Some(effort) => EFFORTS[rank(&effort.to_ascii_lowercase())?],
        None => RECOVERY_EFFORT,
    };
    let recovering = signals.repeated_failure
        || (signals.severity > 0.0 && signals.no_error_streak < RECOVERY_TURNS);
    if recovering {
        return Some((ceiling, "recovery"));
    }
    let effort = if rank(ceiling) < rank(DEFAULT_EFFORT) {
        ceiling
    } else {
        DEFAULT_EFFORT
    };
    Some((effort, "default"))
}

/// Sets `effort` on the request and on each preserved body that is replayed as is.
///
/// A preserved body in a format without a known effort field is dropped, so the
/// normalized request is encoded instead and the effort still reaches the provider.
fn set_effort(request: &mut Request, effort: &str) {
    request.llm_request.reasoning.effort = Some(effort.to_string());
    request
        .llm_request
        .preservation
        .requests
        .retain(|format, body| {
            let Some(body) = body.as_object_mut() else {
                return false;
            };
            let format = format.as_str();
            if format == WireFormat::OpenAiChat.as_str() {
                body.insert("reasoning_effort".to_string(), json!(effort));
            } else if format == WireFormat::OpenAiResponses.as_str() {
                set_nested(body, "reasoning", effort);
            } else if format == WireFormat::AnthropicMessages.as_str() {
                set_nested(body, "output_config", effort);
            } else {
                return false;
            }
            true
        });
}

// Replaces only `effort` inside `key`, keeping fields the caller set next to it.
fn set_nested(body: &mut Map<String, Value>, key: &str, effort: &str) {
    let container = body.entry(key.to_string()).or_insert_with(|| json!({}));
    if !container.is_object() {
        *container = json!({});
    }
    if let Value::Object(container) = container {
        container.insert("effort".to_string(), json!(effort));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use switchyard_protocol::{FormatId, ModelId, text_request};

    use super::*;
    use crate::core::testing::empty_driver;

    fn failed(no_error_streak: u32) -> ToolSignals {
        ToolSignals {
            severity: 0.7,
            no_error_streak,
            ..Default::default()
        }
    }

    #[test]
    fn a_turn_without_failures_runs_at_medium_or_lower() {
        let clean = ToolSignals::default();
        assert_eq!(
            choose_effort(&clean, Some("max")),
            Some(("medium", "default"))
        );
        assert_eq!(choose_effort(&clean, None), Some(("medium", "default")));
        assert_eq!(choose_effort(&clean, Some("low")), Some(("low", "default")));
    }

    #[test]
    fn recovery_turns_run_at_the_caller_effort() {
        assert_eq!(
            choose_effort(&failed(0), Some("max")),
            Some(("max", "recovery"))
        );
        assert_eq!(
            choose_effort(&failed(1), Some("xhigh")),
            Some(("xhigh", "recovery"))
        );
        assert_eq!(choose_effort(&failed(0), None), Some(("high", "recovery")));
        assert_eq!(
            choose_effort(&failed(0), Some("low")),
            Some(("low", "recovery"))
        );
    }

    #[test]
    fn recovery_ends_after_two_clean_results_unless_the_failure_repeats() {
        assert_eq!(
            choose_effort(&failed(2), Some("max")),
            Some(("medium", "default"))
        );
        let repeated = ToolSignals {
            repeated_failure: true,
            ..failed(4)
        };
        assert_eq!(
            choose_effort(&repeated, Some("max")),
            Some(("max", "recovery"))
        );
    }

    #[test]
    fn an_unknown_caller_effort_is_left_alone() {
        assert_eq!(choose_effort(&failed(0), Some("turbo")), None);
    }

    #[test]
    fn set_effort_updates_preserved_bodies_in_their_own_format() {
        let mut request = Request {
            llm_request: text_request(Some("m".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        };
        request.llm_request.preservation.requests = BTreeMap::from([
            (
                FormatId::from(WireFormat::OpenAiChat),
                json!({"reasoning_effort": "max"}),
            ),
            (
                FormatId::from(WireFormat::OpenAiResponses),
                json!({"reasoning": {"effort": "max", "summary": "auto"}}),
            ),
            (
                FormatId::from(WireFormat::AnthropicMessages),
                json!({"output_config": {"effort": "max", "format": {"type": "json_schema"}}}),
            ),
            (FormatId::new("custom"), json!({"effort": "max"})),
        ]);
        set_effort(&mut request, "medium");

        let bodies = &request.llm_request.preservation.requests;
        assert_eq!(
            request.llm_request.reasoning.effort.as_deref(),
            Some("medium")
        );
        assert_eq!(
            bodies[&FormatId::from(WireFormat::OpenAiChat)]["reasoning_effort"],
            "medium"
        );
        assert_eq!(
            bodies[&FormatId::from(WireFormat::OpenAiResponses)]["reasoning"],
            json!({"effort": "medium", "summary": "auto"})
        );
        assert_eq!(
            bodies[&FormatId::from(WireFormat::AnthropicMessages)]["output_config"],
            json!({"effort": "medium", "format": {"type": "json_schema"}})
        );
        assert!(!bodies.contains_key(&FormatId::new("custom")));
    }

    /// Routes every request to one model without touching it.
    struct Fixed;

    #[async_trait::async_trait]
    impl Algorithm for Fixed {
        fn name(&self) -> &str {
            "fixed"
        }

        async fn route(
            self: Arc<Self>,
            _driver: Driver,
            request: Request,
        ) -> Result<RoutingOutcome> {
            Ok(RoutingOutcome::route_to(
                ModelId::from("target"),
                Vec::new(),
                request,
            ))
        }
    }

    #[tokio::test]
    async fn the_wrapper_keeps_the_decision_and_sets_the_effort() -> Result<()> {
        let mut request = Request {
            llm_request: text_request(Some("route".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        };
        request.llm_request.reasoning.effort = Some("max".to_string());
        let algorithm = Arc::new(AdaptiveReasoning::new(Arc::new(Fixed)));

        let outcome = algorithm.clone().route(empty_driver(), request).await?;

        assert_eq!(algorithm.name(), "fixed");
        assert_eq!(outcome.selected_model_id()?.as_str(), "target");
        assert_eq!(
            outcome.request.llm_request.reasoning.effort.as_deref(),
            Some("medium")
        );
        Ok(())
    }
}
