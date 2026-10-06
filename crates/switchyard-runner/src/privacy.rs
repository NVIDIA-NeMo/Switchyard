// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Privacy policy used to select a route's execution lane.

mod deterministic;
mod semantic;

use std::collections::HashSet;

use parking_lot::Mutex;
use serde_json::Value;
use strum_macros::{EnumString, IntoStaticStr};
use switchyard_protocol::{LlmClientError, Request, WireFormat};

const EXTERNAL_RESTRICTION_KEY: &str = "switchyard.internal.external_privacy_restriction";
pub(crate) use deterministic::DeterministicDetector;
use deterministic::{Assessment, Inspector};
pub(crate) use semantic::SemanticPrivacyClassifier;
pub(crate) const SELECTED_LANE_KEY: &str = "switchyard.internal.privacy_lane";
const MAX_TASK_ID_BYTES: usize = 512;
const MAX_RESTRICTED_TASKS: usize = 4_096;
const RESPONSES_STATE_FIELDS: [&str; 2] = ["previous_response_id", "conversation"];

pub(crate) struct PrivacyPolicy {
    accept_external_signal: bool,
    inspector: Option<Inspector>,
    classifier: Option<SemanticPrivacyClassifier>,
    task_restrictions: Option<TaskRestrictions>,
}

#[derive(Default)]
struct TaskRestrictions {
    restricted: Mutex<HashSet<String>>,
}

/// Target set allowed to serve one request.
#[derive(Clone, Copy, EnumString, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum PrivacyLane {
    Standard,
    Restricted,
}

impl PrivacyLane {
    pub(crate) fn as_str(self) -> &'static str {
        self.into()
    }
}

/// Records the lane after routing so target metadata can resolve duplicate model IDs.
pub(crate) fn record_selected_lane(request: &mut Request, lane: PrivacyLane) {
    request
        .metadata
        .get_or_insert_default()
        .extra_metadata
        .get_or_insert_default()
        .insert(SELECTED_LANE_KEY.to_string(), lane.as_str().to_string());
}

/// Reads the runner-owned lane marker from a completed routing request.
pub(crate) fn selected_lane(request: &Request) -> Option<PrivacyLane> {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .and_then(|metadata| metadata.get(SELECTED_LANE_KEY))
        .and_then(|lane| lane.parse().ok())
}

pub(crate) struct PrivacyDecision {
    pub(crate) lane: PrivacyLane,
    pub(crate) source: PrivacySource,
    pub(crate) reason_code: &'static str,
    pub(crate) clear_score: Option<f64>,
    pub(crate) clear_threshold: Option<f64>,
}

#[derive(Clone, Copy, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum PrivacySource {
    Policy,
    ExternalSignal,
    Deterministic,
    SemanticClassifier,
    TaskRetention,
}

impl PrivacySource {
    pub(crate) fn as_str(self) -> &'static str {
        self.into()
    }
}

impl PrivacyDecision {
    const fn new(lane: PrivacyLane, source: PrivacySource, reason_code: &'static str) -> Self {
        Self {
            lane,
            source,
            reason_code,
            clear_score: None,
            clear_threshold: None,
        }
    }

    const fn semantic(
        lane: PrivacyLane,
        reason_code: &'static str,
        clear_score: Option<f64>,
        clear_threshold: f64,
    ) -> Self {
        Self {
            lane,
            source: PrivacySource::SemanticClassifier,
            reason_code,
            clear_score,
            clear_threshold: Some(clear_threshold),
        }
    }

    const fn all_clear() -> Self {
        Self::new(PrivacyLane::Standard, PrivacySource::Policy, "all_clear")
    }

    const fn task_restriction(reason_code: &'static str) -> Self {
        Self::new(
            PrivacyLane::Restricted,
            PrivacySource::TaskRetention,
            reason_code,
        )
    }
}

/// Marks a request as requiring a route that accepts external privacy signals.
/// Execution fails if the selected route has not enabled `accept_external_signal`.
/// This marker is available only to trusted in-process hosts, not HTTP clients.
pub fn mark_privacy_restricted(request: &mut Request) {
    request
        .metadata
        .get_or_insert_default()
        .extra_metadata
        .get_or_insert_default()
        .insert(EXTERNAL_RESTRICTION_KEY.to_string(), String::new());
}

pub(crate) fn has_external_restriction(request: &Request) -> bool {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .is_some_and(|metadata| metadata.contains_key(EXTERNAL_RESTRICTION_KEY))
}

pub(crate) fn external_signal_not_accepted() -> LlmClientError {
    LlmClientError::InvalidRequest {
        message: "route does not accept external privacy signals".to_string(),
    }
}

impl PrivacyPolicy {
    pub(crate) fn new(
        accept_external_signal: bool,
        detectors: Option<Vec<DeterministicDetector>>,
        classifier: Option<SemanticPrivacyClassifier>,
    ) -> Result<Self, regex::Error> {
        let inspector = match detectors {
            Some(detectors) => Some(Inspector::new(detectors)?),
            None if classifier.is_some() => Some(Inspector::structural()),
            None => None,
        };
        Ok(Self {
            accept_external_signal,
            inspector,
            classifier,
            task_restrictions: None,
        })
    }

    pub(crate) fn with_task_retention(mut self) -> Self {
        self.task_restrictions = Some(TaskRestrictions::default());
        self
    }

    pub(crate) async fn decide(
        &self,
        request: &Request,
        route: &str,
        algorithm: &str,
    ) -> Result<PrivacyDecision, LlmClientError> {
        if has_external_restriction(request) && !self.accept_external_signal {
            return Err(external_signal_not_accepted());
        }
        let Some(tasks) = &self.task_restrictions else {
            return self.assess(request, route, algorithm).await;
        };
        let task_id = match task_id(request) {
            Ok(task_id) => task_id,
            Err(reason_code) => return Ok(PrivacyDecision::task_restriction(reason_code)),
        };
        if let Some(decision) = tasks.restriction(task_id) {
            return Ok(decision);
        }

        let decision = self.assess(request, route, algorithm).await?;
        Ok(tasks.retain(task_id, decision))
    }

    async fn assess(
        &self,
        request: &Request,
        route: &str,
        algorithm: &str,
    ) -> Result<PrivacyDecision, LlmClientError> {
        if has_external_restriction(request) {
            return Ok(PrivacyDecision::new(
                PrivacyLane::Restricted,
                PrivacySource::ExternalSignal,
                "restricted",
            ));
        }
        if let Some(inspector) = &self.inspector {
            match inspector.inspect(request) {
                Assessment::Restricted(reason_code) => {
                    return Ok(PrivacyDecision::new(
                        PrivacyLane::Restricted,
                        PrivacySource::Deterministic,
                        reason_code,
                    ));
                }
                Assessment::Indeterminate(reason_code) => {
                    return Ok(PrivacyDecision::new(
                        PrivacyLane::Restricted,
                        PrivacySource::Policy,
                        reason_code,
                    ));
                }
                Assessment::Clear => {}
            }
        }

        Ok(match &self.classifier {
            Some(classifier) => classifier.assess(request, route, algorithm).await,
            None => PrivacyDecision::all_clear(),
        })
    }
}

fn task_id(request: &Request) -> Result<&str, &'static str> {
    let Some(task_id) = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.task_id.as_deref())
    else {
        return Err("missing_task_id");
    };
    if task_id.is_empty() || task_id.trim() != task_id || task_id.len() > MAX_TASK_ID_BYTES {
        return Err("invalid_task_id");
    }
    Ok(task_id)
}

impl TaskRestrictions {
    fn restriction(&self, task_id: &str) -> Option<PrivacyDecision> {
        let restricted = self.restricted.lock();
        if restricted.contains(task_id) {
            Some(PrivacyDecision::task_restriction("retained_restriction"))
        } else if restricted.len() >= MAX_RESTRICTED_TASKS {
            Some(PrivacyDecision::task_restriction("capacity_exhausted"))
        } else {
            None
        }
    }

    fn retain(&self, task_id: &str, decision: PrivacyDecision) -> PrivacyDecision {
        // Recheck because another request may restrict the task while assessment awaits.
        let mut restricted = self.restricted.lock();
        if restricted.contains(task_id) {
            return PrivacyDecision::task_restriction("retained_restriction");
        }
        if restricted.len() >= MAX_RESTRICTED_TASKS {
            return PrivacyDecision::task_restriction("capacity_exhausted");
        }
        if matches!(decision.lane, PrivacyLane::Restricted) {
            restricted.insert(task_id.to_string());
        }
        decision
    }
}

pub(crate) fn validate_mixed_request(request: &Request) -> Result<(), LlmClientError> {
    let extensions = &request.llm_request.extensions.fields;
    let preserved = &request.llm_request.preservation.requests;
    let inspect_extensions = match request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.wire_format)
    {
        Some(format) => format == WireFormat::OpenAiResponses,
        None => preserved.is_empty(),
    };
    let has_state = inspect_extensions
        && RESPONSES_STATE_FIELDS
            .iter()
            .any(|field| extensions.get(*field).is_some_and(|value| !value.is_null()))
        || preserved.iter().any(|(format, body)| {
            format.as_str() == WireFormat::OpenAiResponses.as_str() && has_responses_state(body)
        });
    if has_state {
        return Err(LlmClientError::InvalidRequest {
            message:
                "mixed-traffic privacy routes do not support OpenAI Responses continuation state"
                    .to_string(),
        });
    }
    Ok(())
}

fn has_responses_state(body: &Value) -> bool {
    RESPONSES_STATE_FIELDS
        .iter()
        .any(|field| body.get(*field).is_some_and(|value| !value.is_null()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{ContentBlock, Message, Role};

    #[tokio::test]
    async fn external_restriction_requires_route_opt_in() {
        let mut request = Request::default();
        mark_privacy_restricted(&mut request);

        assert!(
            PrivacyPolicy::new(false, None, None)
                .expect("empty detector configuration should compile")
                .with_task_retention()
                .decide(&request, "test/route", "test_algorithm")
                .await
                .is_err()
        );
        assert!(matches!(
            PrivacyPolicy::new(true, None, None)
                .expect("empty detector configuration should compile")
                .decide(&request, "test/route", "test_algorithm")
                .await,
            Ok(PrivacyDecision {
                lane: PrivacyLane::Restricted,
                ..
            })
        ));

        let mut opaque = Request::default();
        opaque.llm_request.messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Unknown {
                provider: "custom".into(),
                raw: Value::Null,
            }],
        });
        let decision = PrivacyPolicy::new(true, None, None)
            .expect("empty detector configuration should compile")
            .decide(&opaque, "test/route", "test_algorithm")
            .await
            .expect("unmarked request should remain valid");
        assert!(matches!(decision.lane, PrivacyLane::Standard));
    }

    #[tokio::test]
    async fn deterministic_inspection_fails_closed_on_opaque_content() {
        let mut request = Request::default();
        request.llm_request.messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Unknown {
                provider: "custom".into(),
                raw: Value::Null,
            }],
        });

        let decision = PrivacyPolicy::new(false, Some(vec![DeterministicDetector::Email]), None)
            .expect("static detector patterns should compile")
            .decide(&request, "test/route", "test_algorithm")
            .await
            .expect("opaque content should select a lane");
        assert!(matches!(decision.lane, PrivacyLane::Restricted));
        assert_eq!(decision.reason_code, "opaque_content");
    }

    #[test]
    fn responses_continuations_are_rejected_from_normalized_and_preserved_state() {
        for field in RESPONSES_STATE_FIELDS {
            for preserved in [false, true] {
                let mut request = Request::default();
                if preserved {
                    request.llm_request.preservation.requests.insert(
                        "openai_responses".into(),
                        serde_json::json!({field: "state_123"}),
                    );
                } else {
                    request.metadata.get_or_insert_default().wire_format =
                        Some(WireFormat::OpenAiResponses);
                    request
                        .llm_request
                        .extensions
                        .fields
                        .insert(field.to_string(), Value::String("state_123".to_string()));
                }

                assert!(matches!(
                    validate_mixed_request(&request),
                    Err(LlmClientError::InvalidRequest { .. })
                ));
            }
        }

        let mut chat = Request::default();
        chat.metadata.get_or_insert_default().wire_format = Some(WireFormat::OpenAiChat);
        chat.llm_request.preservation.requests.insert(
            "openai_chat".into(),
            serde_json::json!({"conversation": "chat extension"}),
        );
        chat.llm_request.extensions.fields.insert(
            "conversation".to_string(),
            Value::String("chat extension".to_string()),
        );
        assert!(validate_mixed_request(&chat).is_ok());

        let mut untagged = Request::default();
        untagged.llm_request.extensions.fields.insert(
            "conversation".to_string(),
            Value::String("responses state".to_string()),
        );
        assert!(validate_mixed_request(&untagged).is_err());
    }

    #[test]
    fn task_retention_rechecks_state_and_capacity() {
        let tasks = TaskRestrictions::default();
        tasks
            .restricted
            .lock()
            .extend((0..MAX_RESTRICTED_TASKS).map(|id| id.to_string()));

        let retained = tasks.retain("0", PrivacyDecision::all_clear());
        let exhausted = tasks.retain("another-task", PrivacyDecision::all_clear());
        assert_eq!(retained.reason_code, "retained_restriction");
        assert_eq!(exhausted.reason_code, "capacity_exhausted");
    }
}
