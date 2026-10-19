// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, host-supplied target requirements and stateless request admission.
//! Routes support 64 requirement sets over 32 labels of at most 128 bytes each.

use std::collections::BTreeSet;

use serde_json::Value;
use switchyard_protocol::{LlmClientError, Request, WireFormat};

const REQUIREMENTS_KEY: &str = "switchyard.internal.target_requirements";
const MAX_LABELS: usize = 32;
const MAX_LABEL_BYTES: usize = 128;
const MAX_REQUIREMENT_SETS: usize = 64;
// A JSON Unicode escape can use six bytes per input byte, plus separators and brackets.
const MAX_METADATA_BYTES: usize = MAX_LABELS * (MAX_LABEL_BYTES * 6 + 3) + 2;
const RESPONSES_STATE_FIELDS: [&str; 2] = ["previous_response_id", "conversation"];

/// Labels that every target serving one request must satisfy.
pub(crate) type Requirements = BTreeSet<String>;

/// Admits configured requirement sets independently for each request.
pub(crate) struct EligibilityPolicy {
    require_explicit_requirements: bool,
    accepted_requirement_sets: BTreeSet<Requirements>,
    vocabulary: Requirements,
}

impl EligibilityPolicy {
    /// Validates bounded requirement sets and includes the unrestricted empty set.
    pub(crate) fn new(
        require_explicit_requirements: bool,
        accepted_requirement_sets: Vec<Vec<String>>,
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
        })
    }

    /// Returns the requirement sets the route must build before serving requests.
    pub(crate) fn accepted_requirement_sets(&self) -> &BTreeSet<Requirements> {
        &self.accepted_requirement_sets
    }

    /// Validates trusted requirements and rejects unconfigured sets or stored provider state.
    pub(crate) fn admit(&self, request: &Request) -> Result<Requirements, LlmClientError> {
        let has_signal = has_external_requirements(request);
        if self.require_explicit_requirements && !has_signal {
            return Err(invalid(
                "route requires explicit external target requirements",
            ));
        }
        let requirements = external_requirements(request)?;
        if !requirements.is_subset(&self.vocabulary) {
            return Err(invalid("target requirements contain an unknown label"));
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

/// Reads the current request's trusted target requirements.
pub(crate) fn external_requirements(request: &Request) -> Result<Requirements, LlmClientError> {
    Ok(metadata_labels(request)?.unwrap_or_default())
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

    fn policy(accepted_requirement_sets: &[&[&str]]) -> EligibilityPolicy {
        EligibilityPolicy::new(
            false,
            accepted_requirement_sets
                .iter()
                .map(|labels| labels.iter().map(|label| (*label).to_string()).collect())
                .collect(),
        )
        .expect("test requirement sets should be valid")
    }

    fn request(names: &[&str]) -> Request {
        let mut request = Request::default();
        if !names.is_empty() {
            require_target_labels(&mut request, names).expect("test labels should be valid");
        }
        request
    }

    // Requirements accumulate canonically and invalid additions cannot partially replace them.
    #[test]
    fn trusted_requirements_union_without_partial_invalid_additions() -> Result<(), LlmClientError>
    {
        let mut request = request(&["private", "eu", "eu"]);
        require_target_labels(&mut request, ["eu"])?;
        let expected = labels(&["eu", "private"]);
        assert_eq!(metadata_labels(&request)?, Some(expected.clone()));
        assert!(require_target_labels(&mut request, ["valid", " invalid "]).is_err());
        assert_eq!(metadata_labels(&request)?, Some(expected));
        Ok(())
    }

    #[test]
    fn external_requirements_need_known_labels() -> Result<(), LlmClientError> {
        let marked = request(&["private"]);
        let configured = vec![vec!["private".to_string()]];
        let accepts_signal = EligibilityPolicy::new(false, configured)?;
        assert!(has_external_requirements(&marked));
        assert_eq!(accepts_signal.admit(&marked)?, labels(&["private"]));
        assert!(accepts_signal.admit(&request(&[]))?.is_empty());
        let policy = policy(&[&["eu"]]);
        assert!(policy.admit(&marked).is_err());
        assert_eq!(policy.admit(&request(&["eu"]))?, labels(&["eu"]));
        assert!(policy.admit(&request(&["unknown"])).is_err());
        let mut malformed = request(&[]);
        malformed
            .metadata
            .get_or_insert_default()
            .extra_metadata
            .get_or_insert_default()
            .insert(REQUIREMENTS_KEY.into(), "null".into());
        assert!(has_external_requirements(&malformed));
        assert!(policy.admit(&malformed).is_err());
        assert!(policy.admit(&request(&[]))?.is_empty());
        Ok(())
    }

    // Explicit empty signals are accepted; absent and malformed signals are rejected.
    #[test]
    fn required_signals_distinguish_missing_empty_and_malformed_markers()
    -> Result<(), LlmClientError> {
        let policy = EligibilityPolicy::new(true, vec![vec!["private".into()]])?;
        let missing = request(&[]);
        assert!(policy.admit(&missing).is_err());

        let mut empty = missing.clone();
        require_target_labels(&mut empty, std::iter::empty::<&str>())?;
        assert!(policy.admit(&empty)?.is_empty());

        assert_eq!(policy.admit(&request(&["private"]))?, labels(&["private"]));
        assert!(policy.admit(&missing).is_err());
        assert!(policy.admit(&empty)?.is_empty());

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

    // Known labels still require an exact configured set; one request cannot affect another.
    #[test]
    fn request_requirements_are_stateless_and_require_configured_sets() -> Result<(), LlmClientError>
    {
        let policy = policy(&[&["eu"], &["private"]]);
        assert_eq!(policy.admit(&request(&["eu"]))?, labels(&["eu"]));
        assert_eq!(policy.admit(&request(&["private"]))?, labels(&["private"]));
        assert!(policy.admit(&request(&["eu", "private"])).is_err());
        assert!(policy.admit(&request(&[]))?.is_empty());
        Ok(())
    }

    #[test]
    fn configuration_and_metadata_are_bounded() {
        assert!(
            EligibilityPolicy::new(false, vec![vec!["x".repeat(MAX_LABEL_BYTES + 1)]]).is_err()
        );
        let vocabulary = (0..=MAX_LABELS)
            .map(|index| vec![format!("label_{index}")])
            .collect();
        assert!(EligibilityPolicy::new(false, vocabulary).is_err());
        let accepted_requirement_sets = (1..=MAX_REQUIREMENT_SETS)
            .map(|bits| {
                (0..7)
                    .filter(|bit| bits & (1 << bit) != 0)
                    .map(|bit| format!("label_{bit}"))
                    .collect()
            })
            .collect();
        assert!(EligibilityPolicy::new(false, accepted_requirement_sets).is_err());
        let policy = policy(&[]);
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
