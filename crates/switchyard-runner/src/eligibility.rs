// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, host-supplied target requirements and monotonic task admission.
//! Routes support 64 requirement sets over 32 labels of at most 128 bytes each. Task retention
//! stores at most 4,096 nonempty requirement sets under task IDs of at most 512 bytes.

use std::collections::{BTreeSet, HashMap};

use parking_lot::Mutex;
use serde_json::Value;
use switchyard_protocol::{LlmClientError, Request, WireFormat};

use crate::privacy::has_external_restriction;

const REQUIREMENTS_KEY: &str = "switchyard.internal.target_requirements";
const MAX_LABELS: usize = 32;
const MAX_LABEL_BYTES: usize = 128;
const MAX_REQUIREMENT_SETS: usize = 64;
// A JSON Unicode escape can use six bytes per input byte, plus separators and brackets.
const MAX_METADATA_BYTES: usize = MAX_LABELS * (MAX_LABEL_BYTES * 6 + 3) + 2;
const MAX_TASK_ID_BYTES: usize = 512;
const MAX_TASKS: usize = 4_096;
const RESPONSES_STATE_FIELDS: [&str; 2] = ["previous_response_id", "conversation"];

/// Labels that every target serving one request must satisfy.
pub(crate) type Requirements = BTreeSet<String>;

/// Admits configured requirement sets, retaining authoritative requirements by task.
pub(crate) struct EligibilityPolicy {
    require_explicit_requirements: bool,
    accepted_requirement_sets: BTreeSet<Requirements>,
    vocabulary: Requirements,
    tasks: Option<Mutex<HashMap<String, Requirements>>>,
}

impl EligibilityPolicy {
    /// Validates bounded requirement sets and includes the unrestricted empty set.
    pub(crate) fn new(
        require_explicit_requirements: bool,
        accepted_requirement_sets: Vec<Vec<String>>,
        task_scope: bool,
    ) -> Result<Self, LlmClientError> {
        let mut configured = BTreeSet::from([Requirements::new()]);
        let mut vocabulary = Requirements::new();
        for labels in accepted_requirement_sets {
            let labels = labels.into_iter().collect::<Requirements>();
            validate_labels(&labels)?;
            vocabulary.extend(labels.iter().cloned());
            validate_labels(&vocabulary)?;
            configured.insert(labels);
            if configured.len() > MAX_REQUIREMENT_SETS {
                return Err(invalid("too many target requirement sets"));
            }
        }
        Ok(Self {
            require_explicit_requirements,
            accepted_requirement_sets: configured,
            vocabulary,
            tasks: task_scope.then(Mutex::default),
        })
    }

    /// Returns the requirement sets the route must build before serving requests.
    pub(crate) fn accepted_requirement_sets(&self) -> &BTreeSet<Requirements> {
        &self.accepted_requirement_sets
    }

    /// Retains known requirements before rejecting an unsupported accumulated set.
    pub(crate) fn admit(&self, request: &Request) -> Result<Requirements, LlmClientError> {
        let has_signal = has_external_requirements(request) || has_external_restriction(request);
        if self.require_explicit_requirements && !has_signal {
            return Err(invalid(
                "route requires explicit external target requirements",
            ));
        }
        let mut requirements = external_requirements(request)?;
        if !requirements.is_subset(&self.vocabulary) {
            return Err(invalid("target requirements contain an unknown label"));
        }

        if let Some(tasks) = &self.tasks {
            let task_id = task_id(request)?;
            // Persist tightening even when no lane can serve the resulting union. Otherwise a
            // later request could drop the new requirement and reopen a less restrictive lane.
            let mut tasks = tasks.lock();
            if let Some(retained) = tasks.get_mut(task_id) {
                retained.extend(requirements);
                requirements = retained.clone();
            } else {
                // Unknown tasks must stay closed when new restrictions cannot be remembered.
                if tasks.len() >= MAX_TASKS {
                    return Err(invalid("target requirement task capacity exhausted"));
                }
                if !requirements.is_empty() {
                    tasks.insert(task_id.to_string(), requirements.clone());
                }
            }
        }
        if !self.accepted_requirement_sets.contains(&requirements) {
            return Err(invalid("target requirement set is not configured"));
        }
        validate_request_context(request)?;
        Ok(requirements)
    }
}

/// Adds trusted in-process requirements without replacing requirements already on the request.
/// HTTP bodies and headers cannot set this internal marker. Invalid additions leave it untouched.
/// An empty iterator records an explicit signal without removing any requirements.
pub fn require_target_labels(
    request: &mut Request,
    labels: impl IntoIterator<Item = impl AsRef<str>>,
) -> Result<(), LlmClientError> {
    let mut requirements = metadata_labels(request)?.unwrap_or_default();
    for label in labels {
        let label = label.as_ref();
        validate_label(label)?;
        requirements.insert(label.to_string());
        if requirements.len() > MAX_LABELS {
            return Err(invalid("too many target requirement labels"));
        }
    }
    record_labels(request, &requirements)
}

/// Detects the new requirements marker, including malformed or empty encoded sets.
pub(crate) fn has_external_requirements(request: &Request) -> bool {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .is_some_and(|metadata| metadata.contains_key(REQUIREMENTS_KEY))
}

/// Reads current trusted requirements, adapting the legacy privacy marker to `private`.
pub(crate) fn external_requirements(request: &Request) -> Result<Requirements, LlmClientError> {
    let mut requirements = metadata_labels(request)?.unwrap_or_default();
    if has_external_restriction(request) {
        requirements.insert("private".into());
    }
    validate_labels(&requirements)?;
    Ok(requirements)
}

/// Checks label syntax and the bounded size of one target or requirement set.
pub(crate) fn validate_labels(labels: &Requirements) -> Result<(), LlmClientError> {
    if labels.len() > MAX_LABELS {
        return Err(invalid("too many target requirement labels"));
    }
    for label in labels {
        validate_label(label)?;
    }
    Ok(())
}

fn validate_label(label: &str) -> Result<(), LlmClientError> {
    if label.is_empty()
        || label.len() > MAX_LABEL_BYTES
        || label.trim() != label
        || label.chars().any(char::is_control)
    {
        return Err(invalid("invalid target requirement label"));
    }
    Ok(())
}

fn metadata_labels(request: &Request) -> Result<Option<Requirements>, LlmClientError> {
    let Some(value) = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .and_then(|metadata| metadata.get(REQUIREMENTS_KEY))
    else {
        return Ok(None);
    };
    if value.len() > MAX_METADATA_BYTES {
        return Err(invalid(
            "target requirement metadata exceeds the size limit",
        ));
    }
    let labels =
        serde_json::from_str(value).map_err(|_| invalid("invalid target requirement metadata"))?;
    validate_labels(&labels)?;
    Ok(Some(labels))
}

fn record_labels(request: &mut Request, requirements: &Requirements) -> Result<(), LlmClientError> {
    let value = serde_json::to_string(requirements)
        .map_err(|_| invalid("target requirements could not be encoded"))?;
    request
        .metadata
        .get_or_insert_default()
        .extra_metadata
        .get_or_insert_default()
        .insert(REQUIREMENTS_KEY.to_string(), value);
    Ok(())
}

fn task_id(request: &Request) -> Result<&str, LlmClientError> {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.task_id.as_deref())
        .filter(|id| !id.is_empty() && id.len() <= MAX_TASK_ID_BYTES && id.trim() == *id)
        .ok_or_else(|| invalid("target requirements need a valid task id"))
}

fn invalid(message: &'static str) -> LlmClientError {
    LlmClientError::InvalidRequest {
        message: message.to_string(),
    }
}

// Stored provider state can contain content that was admitted under different requirements.
fn validate_request_context(request: &Request) -> Result<(), LlmClientError> {
    let has_state = |body: &Value| {
        RESPONSES_STATE_FIELDS
            .iter()
            .any(|field| body.get(*field).is_some_and(|value| !value.is_null()))
    };
    let extensions = &request.llm_request.extensions.fields;
    if RESPONSES_STATE_FIELDS
        .iter()
        .any(|field| extensions.get(*field).is_some_and(|value| !value.is_null()))
        || request
            .llm_request
            .preservation
            .requests
            .iter()
            .any(|(format, body)| {
                format.as_str() == WireFormat::OpenAiResponses.as_str() && has_state(body)
            })
    {
        return Err(invalid(
            "eligibility routes do not support OpenAI Responses continuation state",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::mark_privacy_restricted;

    #[test]
    fn responses_continuations_are_rejected_from_normalized_and_preserved_state() {
        for field in RESPONSES_STATE_FIELDS {
            for format in [
                None,
                Some(WireFormat::OpenAiChat),
                Some(WireFormat::OpenAiResponses),
                Some(WireFormat::AnthropicMessages),
            ] {
                let mut request = Request::default();
                request.metadata.get_or_insert_default().wire_format = format;
                request
                    .llm_request
                    .extensions
                    .fields
                    .insert(field.into(), Value::String("state_123".into()));
                assert!(
                    validate_request_context(&request).is_err(),
                    "{format:?}: {field}"
                );
                request
                    .llm_request
                    .extensions
                    .fields
                    .insert(field.into(), Value::Null);
                assert!(validate_request_context(&request).is_ok());
            }
            let mut preserved = Request::default();
            preserved.llm_request.preservation.requests.insert(
                "openai_responses".into(),
                serde_json::json!({field: "state_123"}),
            );
            assert!(validate_request_context(&preserved).is_err());
        }
    }

    fn labels(names: &[&str]) -> Requirements {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    fn policy(accepted_requirement_sets: &[&[&str]], task_scope: bool) -> EligibilityPolicy {
        EligibilityPolicy::new(
            false,
            accepted_requirement_sets
                .iter()
                .map(|labels| labels.iter().map(|label| (*label).to_string()).collect())
                .collect(),
            task_scope,
        )
        .expect("test requirement sets should be valid")
    }

    fn request(task: &str, names: &[&str]) -> Request {
        let mut request = Request::default();
        request.metadata.get_or_insert_default().task_id = Some(task.to_string());
        if !names.is_empty() {
            require_target_labels(&mut request, names).expect("test labels should be valid");
        }
        request
    }

    // Requirements accumulate canonically and invalid additions cannot partially replace them.
    #[test]
    fn trusted_requirements_union_without_partial_invalid_additions() -> Result<(), LlmClientError>
    {
        let mut request = request("task", &["private", "eu", "eu"]);
        require_target_labels(&mut request, ["eu"])?;
        let expected = labels(&["eu", "private"]);
        assert_eq!(metadata_labels(&request)?, Some(expected.clone()));
        assert!(require_target_labels(&mut request, ["valid", " invalid "]).is_err());
        assert_eq!(metadata_labels(&request)?, Some(expected));
        Ok(())
    }

    // Both trusted Rust helpers use the route's configured label vocabulary.
    #[test]
    fn external_requirements_need_known_labels() -> Result<(), LlmClientError> {
        let mut marked = request("task", &[]);
        mark_privacy_restricted(&mut marked);
        let configured = vec![vec!["private".to_string()]];
        assert!(!has_external_requirements(&marked));
        let accepts_signal = EligibilityPolicy::new(false, configured, false)?;
        assert_eq!(accepts_signal.admit(&marked)?, labels(&["private"]));
        assert_eq!(
            accepts_signal.admit(&request("task", &["private"]))?,
            labels(&["private"])
        );
        assert!(accepts_signal.admit(&request("task", &[]))?.is_empty());
        let policy = policy(&[&["eu"]], true);
        assert!(policy.admit(&marked).is_err());
        assert_eq!(policy.admit(&request("task", &["eu"]))?, labels(&["eu"]));
        assert!(policy.admit(&request("task", &["unknown"])).is_err());
        let mut malformed = request("task", &[]);
        malformed
            .metadata
            .get_or_insert_default()
            .extra_metadata
            .get_or_insert_default()
            .insert(REQUIREMENTS_KEY.into(), "null".into());
        assert!(has_external_requirements(&malformed));
        assert!(policy.admit(&malformed).is_err());
        assert_eq!(policy.admit(&request("task", &[]))?, labels(&["eu"]));
        Ok(())
    }

    // Explicit empty signals do not clear task history; absent signals are always rejected.
    #[test]
    fn required_signals_distinguish_missing_empty_and_legacy_markers() -> Result<(), LlmClientError>
    {
        let policy = EligibilityPolicy::new(true, vec![vec!["private".into()]], true)?;
        let missing = request("task", &[]);
        assert!(policy.admit(&missing).is_err());

        let mut empty = missing.clone();
        require_target_labels(&mut empty, std::iter::empty::<&str>())?;
        assert!(policy.admit(&empty)?.is_empty());

        let mut legacy = missing.clone();
        mark_privacy_restricted(&mut legacy);
        assert_eq!(policy.admit(&legacy)?, labels(&["private"]));
        assert!(policy.admit(&missing).is_err());
        assert_eq!(policy.admit(&empty)?, labels(&["private"]));

        empty
            .metadata
            .as_mut()
            .expect("metadata")
            .extra_metadata
            .as_mut()
            .expect("signal")
            .insert(REQUIREMENTS_KEY.into(), "null".into());
        assert!(policy.admit(&empty).is_err());
        Ok(())
    }

    // An unsupported union stays retained, so omitting a requirement cannot reopen the task.
    #[test]
    fn task_requirements_tighten_even_without_a_configured_union() -> Result<(), LlmClientError> {
        let restricted = policy(&[&["eu"], &["private"]], true);
        assert_eq!(
            restricted.admit(&request("task", &["eu"]))?,
            labels(&["eu"])
        );
        assert!(restricted.admit(&request("task", &["private"])).is_err());
        assert!(restricted.admit(&request("task", &[])).is_err());
        assert_eq!(
            restricted.admit(&request("other", &[]))?,
            Requirements::new()
        );

        let concurrent = policy(&[&["eu"], &["private"], &["eu", "private"]], true);
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            for label in ["eu", "private"] {
                let policy = &concurrent;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    policy
                        .admit(&request("shared", &[label]))
                        .expect("concurrent requirements should be admitted");
                });
            }
        });
        assert_eq!(
            concurrent.admit(&request("shared", &[]))?,
            labels(&["eu", "private"])
        );
        Ok(())
    }

    // Capacity exhaustion rejects new tasks while existing tasks can still acquire labels.
    #[test]
    fn task_capacity_does_not_prevent_existing_tasks_from_tightening() -> Result<(), LlmClientError>
    {
        let policy = policy(&[&["eu"], &["private"], &["eu", "private"]], true);
        let tasks = policy
            .tasks
            .as_ref()
            .expect("task policy should have state");
        tasks
            .lock()
            .extend((0..MAX_TASKS).map(|task| (task.to_string(), labels(&["eu"]))));
        assert!(policy.admit(&request("new", &["private"])).is_err());
        assert!(policy.admit(&request("new", &[])).is_err());
        assert_eq!(
            policy.admit(&request("0", &["private"]))?,
            labels(&["eu", "private"])
        );
        assert!(policy.admit(&Request::default()).is_err());
        Ok(())
    }

    #[test]
    fn configuration_and_metadata_are_bounded() {
        assert!(
            EligibilityPolicy::new(false, vec![vec!["x".repeat(MAX_LABEL_BYTES + 1)]], false)
                .is_err()
        );
        let vocabulary = (0..=MAX_LABELS)
            .map(|index| vec![format!("label_{index}")])
            .collect();
        assert!(EligibilityPolicy::new(false, vocabulary, false).is_err());
        let accepted_requirement_sets = (1..=MAX_REQUIREMENT_SETS)
            .map(|bits| {
                (0..7)
                    .filter(|bit| bits & (1 << bit) != 0)
                    .map(|bit| format!("label_{bit}"))
                    .collect()
            })
            .collect();
        assert!(EligibilityPolicy::new(false, accepted_requirement_sets, false).is_err());
        let policy = policy(&[], false);
        let mut request = Request::default();
        request
            .metadata
            .get_or_insert_default()
            .extra_metadata
            .get_or_insert_default()
            .insert(REQUIREMENTS_KEY.into(), "{\"not\":\"a set\"}".into());
        assert!(policy.admit(&request).is_err());
    }
}
