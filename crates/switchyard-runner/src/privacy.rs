// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trusted restricted admission and bounded task retention.

use std::collections::HashSet;

use parking_lot::Mutex;
use serde_json::Value;
use strum_macros::IntoStaticStr;
use switchyard_protocol::{LlmClientError, Request, WireFormat};

// Keep the key stable so requests assessed by existing callers retain their restriction.
const RESTRICTION_KEY: &str = "switchyard.internal.local_only";
const MAX_TASK_ID_BYTES: usize = 512;
const MAX_TASKS: usize = 4_096;
const RESPONSES_STATE_FIELDS: [&str; 2] = ["previous_response_id", "conversation"];

/// Admits a restricted flag and optionally retains it for the task.
pub(crate) struct RestrictionPolicy {
    require_explicit_signal: bool,
    tasks: Option<Mutex<HashSet<String>>>,
}

impl RestrictionPolicy {
    pub(crate) fn new(require_explicit_signal: bool, task_scope: bool) -> Self {
        Self {
            require_explicit_signal,
            tasks: task_scope.then(Mutex::default),
        }
    }

    // Retain a restriction even if later context validation rejects this turn.
    pub(crate) fn admit(&self, request: &Request) -> Result<bool, LlmClientError> {
        let signal = restriction_signal(request)?;
        if self.require_explicit_signal && signal.is_none() {
            return Err(invalid("route requires an explicit restricted signal"));
        }
        let mut is_restricted = signal.unwrap_or(false);
        if let Some(tasks) = &self.tasks {
            let task_id = request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.task_id.as_deref())
                .filter(|id| !id.is_empty() && id.len() <= MAX_TASK_ID_BYTES && id.trim() == *id)
                .ok_or_else(|| invalid("restricted routing needs a valid task id"))?;
            let mut tasks = tasks.lock();
            let retained = tasks.contains(task_id);
            if !retained && tasks.len() >= MAX_TASKS {
                return Err(invalid("restricted task capacity exhausted"));
            }
            if is_restricted && !retained {
                tasks.insert(task_id.to_owned());
            }
            is_restricted |= retained;
        }
        validate_request_context(request)?;
        Ok(is_restricted)
    }
}

/// Attaches a trusted restricted assessment to an in-process request.
/// False records an explicit signal but cannot remove an earlier true value.
/// The host owns classification, task authorization and target approvals.
pub fn require_restricted_targets(
    request: &mut Request,
    is_restricted: bool,
) -> Result<(), LlmClientError> {
    let is_restricted = restriction_signal(request)?.unwrap_or(false) || is_restricted;
    request
        .metadata
        .get_or_insert_default()
        .extra_metadata
        .get_or_insert_default()
        .insert(RESTRICTION_KEY.to_string(), is_restricted.to_string());
    Ok(())
}

pub(crate) fn has_restriction_signal(request: &Request) -> bool {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .is_some_and(|metadata| metadata.contains_key(RESTRICTION_KEY))
}

fn restriction_signal(request: &Request) -> Result<Option<bool>, LlmClientError> {
    match request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .and_then(|metadata| metadata.get(RESTRICTION_KEY))
        .map(String::as_str)
    {
        None => Ok(None),
        Some("true") => Ok(Some(true)),
        Some("false") => Ok(Some(false)),
        Some(_) => Err(invalid("invalid restricted signal")),
    }
}

#[derive(Clone, Copy, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum PrivacyLane {
    Standard,
    Restricted,
}

#[derive(Clone, Copy, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum PrivacySource {
    Policy,
    ExternalSignal,
    TaskRetention,
}

/// The admitted standard or restricted lane and its trusted source.
pub struct PrivacyDecision {
    lane: PrivacyLane,
    source: PrivacySource,
}

impl PrivacyDecision {
    /// Returns the admitted lane: `restricted` or `standard`.
    pub fn lane(&self) -> &'static str {
        self.lane.into()
    }

    /// Returns the policy component that selected the lane.
    pub fn source(&self) -> &'static str {
        self.source.into()
    }

    /// Returns a bounded reason for the decision.
    pub const fn reason_code(&self) -> &'static str {
        match self.source {
            PrivacySource::Policy => "all_clear",
            PrivacySource::ExternalSignal => "restricted",
            PrivacySource::TaskRetention => "retained_restriction",
        }
    }

    pub(crate) fn from_restriction(
        request: &Request,
        is_restricted: bool,
    ) -> Result<Self, LlmClientError> {
        Ok(Self {
            lane: if is_restricted {
                PrivacyLane::Restricted
            } else {
                PrivacyLane::Standard
            },
            source: if !is_restricted {
                PrivacySource::Policy
            } else if restriction_signal(request)? == Some(true) {
                PrivacySource::ExternalSignal
            } else {
                PrivacySource::TaskRetention
            },
        })
    }
}

fn invalid(message: &'static str) -> LlmClientError {
    LlmClientError::InvalidRequest {
        message: message.to_string(),
    }
}

// Provider-held history may have been admitted under a different restriction.
fn validate_request_context(request: &Request) -> Result<(), LlmClientError> {
    let has_state = |body: &Value| {
        RESPONSES_STATE_FIELDS
            .iter()
            .any(|field| body.get(*field).is_some_and(|value| !value.is_null()))
    };
    if RESPONSES_STATE_FIELDS.iter().any(|field| {
        request
            .llm_request
            .extensions
            .fields
            .get(*field)
            .is_some_and(|value| !value.is_null())
    }) || request
        .llm_request
        .preservation
        .requests
        .iter()
        .any(|(format, body)| {
            format.as_str() == WireFormat::OpenAiResponses.as_str() && has_state(body)
        })
    {
        return Err(invalid(
            "restricted routes do not support OpenAI Responses continuation state",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: &str, is_restricted: bool) -> Request {
        let mut request = Request::default();
        request.metadata.get_or_insert_default().task_id = Some(id.into());
        require_restricted_targets(&mut request, is_restricted).expect("trusted signal");
        request
    }

    #[test]
    fn explicit_signals_cannot_relax_a_request_or_task() {
        let policy = RestrictionPolicy::new(true, true);
        assert!(policy.admit(&Request::default()).is_err());
        assert!(!policy.admit(&task("one", false)).expect("explicit false"));
        let mut request = task("one", true);
        require_restricted_targets(&mut request, false).expect("cannot clear true");
        assert!(policy.admit(&request).expect("restricted"));
        assert!(policy.admit(&task("one", false)).expect("retained"));
        assert!(!policy.admit(&task("two", false)).expect("other task"));
        let retained =
            PrivacyDecision::from_restriction(&task("one", false), true).expect("decision");
        assert_eq!(
            (retained.lane(), retained.source()),
            ("restricted", "task_retention")
        );
    }

    #[test]
    fn malformed_signal_and_full_task_store_fail_closed() {
        let policy = RestrictionPolicy::new(false, true);
        let mut request = task("one", true);
        request
            .metadata
            .as_mut()
            .unwrap()
            .extra_metadata
            .as_mut()
            .unwrap()
            .insert(RESTRICTION_KEY.into(), "unknown".into());
        assert!(require_restricted_targets(&mut request, false).is_err());
        assert!(policy.admit(&request).is_err());
        for id in 0..MAX_TASKS {
            policy
                .admit(&task(&id.to_string(), true))
                .expect("capacity available");
        }
        assert!(policy.admit(&task("new", false)).is_err());
        assert!(policy.admit(&task("0", false)).expect("existing task"));
    }

    #[test]
    fn provider_held_history_is_rejected_without_erasing_task_restriction() {
        let policy = RestrictionPolicy::new(false, true);
        for field in RESPONSES_STATE_FIELDS {
            let mut request = task(field, true);
            request
                .llm_request
                .extensions
                .fields
                .insert(field.into(), Value::String("hidden".into()));
            assert!(policy.admit(&request).is_err());
            assert!(
                policy
                    .admit(&task(field, false))
                    .expect("retained after rejection")
            );
            let mut preserved = task("preserved", false);
            preserved.llm_request.preservation.requests.insert(
                "openai_responses".into(),
                serde_json::json!({field: "hidden"}),
            );
            assert!(policy.admit(&preserved).is_err());
        }
    }
}
