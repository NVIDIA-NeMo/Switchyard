// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trusted local-only admission and bounded task retention.

use std::collections::HashSet;

use parking_lot::Mutex;
use serde_json::Value;
use strum_macros::IntoStaticStr;
use switchyard_protocol::{LlmClientError, Request, WireFormat};

const LOCAL_ONLY_KEY: &str = "switchyard.internal.local_only";
const MAX_TASK_ID_BYTES: usize = 512;
const MAX_TASKS: usize = 4_096;
const RESPONSES_STATE_FIELDS: [&str; 2] = ["previous_response_id", "conversation"];

/// Admits a local-only flag and optionally retains it for the task.
pub(crate) struct LocalPolicy {
    require_explicit_signal: bool,
    tasks: Option<Mutex<HashSet<String>>>,
}

impl LocalPolicy {
    pub(crate) fn new(require_explicit_signal: bool, task_scope: bool) -> Self {
        Self {
            require_explicit_signal,
            tasks: task_scope.then(Mutex::default),
        }
    }

    // Retain a restriction even if later context validation rejects this turn.
    pub(crate) fn admit(&self, request: &Request) -> Result<bool, LlmClientError> {
        let signal = local_requirement(request)?;
        if self.require_explicit_signal && signal.is_none() {
            return Err(invalid("route requires an explicit local-only signal"));
        }
        let mut local_only = signal.unwrap_or(false);
        if let Some(tasks) = &self.tasks {
            let task_id = request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.task_id.as_deref())
                .filter(|id| !id.is_empty() && id.len() <= MAX_TASK_ID_BYTES && id.trim() == *id)
                .ok_or_else(|| invalid("local-only routing needs a valid task id"))?;
            let mut tasks = tasks.lock();
            let retained = tasks.contains(task_id);
            if !retained && tasks.len() >= MAX_TASKS {
                return Err(invalid("local-only task capacity exhausted"));
            }
            if local_only && !retained {
                tasks.insert(task_id.to_owned());
            }
            local_only |= retained;
        }
        validate_request_context(request)?;
        Ok(local_only)
    }
}

/// Attaches a trusted local-only assessment to an in-process request.
/// False records an explicit signal but cannot remove an earlier true value.
/// The host owns classification, task authorization and target approvals.
pub fn require_local_targets(
    request: &mut Request,
    local_only: bool,
) -> Result<(), LlmClientError> {
    let local_only = local_requirement(request)?.unwrap_or(false) || local_only;
    request
        .metadata
        .get_or_insert_default()
        .extra_metadata
        .get_or_insert_default()
        .insert(LOCAL_ONLY_KEY.to_string(), local_only.to_string());
    Ok(())
}

pub(crate) fn has_local_requirement(request: &Request) -> bool {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .is_some_and(|metadata| metadata.contains_key(LOCAL_ONLY_KEY))
}

fn local_requirement(request: &Request) -> Result<Option<bool>, LlmClientError> {
    match request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .and_then(|metadata| metadata.get(LOCAL_ONLY_KEY))
        .map(String::as_str)
    {
        None => Ok(None),
        Some("true") => Ok(Some(true)),
        Some("false") => Ok(Some(false)),
        Some(_) => Err(invalid("invalid local-only signal")),
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

/// The admitted standard or local-only lane and its trusted source.
pub struct PrivacyDecision {
    lane: PrivacyLane,
    source: PrivacySource,
}

impl PrivacyDecision {
    /// Returns `restricted` for local-only handling, otherwise `standard`.
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

    pub(crate) fn from_local_requirement(
        request: &Request,
        local_only: bool,
    ) -> Result<Self, LlmClientError> {
        Ok(Self {
            lane: if local_only {
                PrivacyLane::Restricted
            } else {
                PrivacyLane::Standard
            },
            source: if !local_only {
                PrivacySource::Policy
            } else if local_requirement(request)? == Some(true) {
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
            "local-only routes do not support OpenAI Responses continuation state",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: &str, local_only: bool) -> Request {
        let mut request = Request::default();
        request.metadata.get_or_insert_default().task_id = Some(id.into());
        require_local_targets(&mut request, local_only).expect("trusted signal");
        request
    }

    #[test]
    fn explicit_signals_cannot_relax_a_request_or_task() {
        let policy = LocalPolicy::new(true, true);
        assert!(policy.admit(&Request::default()).is_err());
        assert!(!policy.admit(&task("one", false)).expect("explicit false"));
        let mut request = task("one", true);
        require_local_targets(&mut request, false).expect("cannot clear true");
        assert!(policy.admit(&request).expect("local-only"));
        assert!(policy.admit(&task("one", false)).expect("retained"));
        assert!(!policy.admit(&task("two", false)).expect("other task"));
        let retained =
            PrivacyDecision::from_local_requirement(&task("one", false), true).expect("decision");
        assert_eq!(
            (retained.lane(), retained.source()),
            ("restricted", "task_retention")
        );
    }

    #[test]
    fn malformed_signal_and_full_task_store_fail_closed() {
        let policy = LocalPolicy::new(false, true);
        let mut request = task("one", true);
        request
            .metadata
            .as_mut()
            .unwrap()
            .extra_metadata
            .as_mut()
            .unwrap()
            .insert(LOCAL_ONLY_KEY.into(), "unknown".into());
        assert!(require_local_targets(&mut request, false).is_err());
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
        let policy = LocalPolicy::new(false, true);
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
