// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trusted target requirements for one request.

use std::collections::BTreeSet;

use serde_json::Value;
use strum_macros::IntoStaticStr;
use switchyard_protocol::{LlmClientError, Request, WireFormat};

const REQUIREMENTS_KEY: &str = "switchyard.internal.required_target_labels";
const LEGACY_RESTRICTION_KEY: &str = "switchyard.internal.local_only";
const MAX_LABELS: usize = 64;
const MAX_LABEL_BYTES: usize = 128;
const MAX_REQUIREMENTS_BYTES: usize = MAX_LABELS * (MAX_LABEL_BYTES * 6 + 4) + 2;
const RESPONSES_STATE_FIELDS: [&str; 2] = ["previous_response_id", "conversation"];

pub(crate) struct EligibilityPolicy {
    require_explicit_requirements: bool,
}

impl EligibilityPolicy {
    pub(crate) fn new(require_explicit_requirements: bool) -> Self {
        Self {
            require_explicit_requirements,
        }
    }

    pub(crate) fn admit(&self, request: &Request) -> Result<TargetRequirements, LlmClientError> {
        let supplied = target_requirements(request)?;
        if self.require_explicit_requirements && supplied.is_none() {
            return Err(invalid("route requires explicit target requirements"));
        }
        let source = if supplied.as_ref().is_some_and(BTreeSet::is_empty) {
            EligibilitySource::ExplicitEmpty
        } else if supplied.is_some() {
            EligibilitySource::ExternalSignal
        } else {
            EligibilitySource::Policy
        };
        let required_labels = supplied.unwrap_or_default();
        validate_request_context(request)?;
        Ok(TargetRequirements {
            required_labels,
            source,
        })
    }
}

/// Adds trusted target requirements to an in-process request without removing earlier labels.
/// An empty collection explicitly supplies no new requirements. The host owns classification and approvals.
pub fn require_target_labels(
    request: &mut Request,
    labels: impl IntoIterator<Item = impl AsRef<str>>,
) -> Result<(), LlmClientError> {
    let mut required = target_requirements(request)?.unwrap_or_default();
    for label in labels {
        let label = label.as_ref();
        validate_label(label)?;
        required.insert(label.to_owned());
        if required.len() > MAX_LABELS {
            return Err(invalid("too many target requirement labels"));
        }
    }
    let serialized = serde_json::to_string(&required)
        .map_err(|_| invalid("invalid target requirement labels"))?;
    request
        .metadata
        .get_or_insert_default()
        .extra_metadata
        .get_or_insert_default()
        .insert(REQUIREMENTS_KEY.to_owned(), serialized);
    Ok(())
}

pub(crate) fn has_target_requirements(request: &Request) -> bool {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .is_some_and(|metadata| {
            metadata.contains_key(REQUIREMENTS_KEY) || metadata.contains_key(LEGACY_RESTRICTION_KEY)
        })
}

pub(crate) fn validate_labels(labels: &BTreeSet<String>) -> Result<(), LlmClientError> {
    if labels.len() > MAX_LABELS {
        return Err(invalid("too many target requirement labels"));
    }
    labels.iter().try_for_each(|label| validate_label(label))
}

fn validate_label(label: &str) -> Result<(), LlmClientError> {
    if label.is_empty() || label.len() > MAX_LABEL_BYTES || label.trim() != label {
        return Err(invalid(
            "target labels must be nonblank, trimmed, and at most 128 bytes",
        ));
    }
    Ok(())
}

fn target_requirements(request: &Request) -> Result<Option<BTreeSet<String>>, LlmClientError> {
    let Some(metadata) = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
    else {
        return Ok(None);
    };
    let mut labels: Option<BTreeSet<String>> = metadata
        .get(REQUIREMENTS_KEY)
        .map(|encoded| {
            if encoded.len() > MAX_REQUIREMENTS_BYTES {
                return Err(invalid("target requirement labels exceed the size limit"));
            }
            serde_json::from_str(encoded).map_err(|_| invalid("invalid target requirement labels"))
        })
        .transpose()?;
    match metadata.get(LEGACY_RESTRICTION_KEY).map(String::as_str) {
        Some("true") => {
            labels.get_or_insert_default().insert("private".into());
        }
        Some("false") => {
            labels.get_or_insert_default();
        }
        Some(_) => return Err(invalid("invalid restricted signal")),
        None => {}
    }
    if let Some(labels) = &labels {
        validate_labels(labels)?;
    }
    Ok(labels)
}

#[derive(Clone, Copy, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum EligibilitySource {
    Policy,
    #[strum(serialize = "external_signal")]
    ExplicitEmpty,
    ExternalSignal,
}

/// Validated target requirements supplied by the caller.
pub struct TargetRequirements {
    required_labels: BTreeSet<String>,
    source: EligibilitySource,
}

impl TargetRequirements {
    /// Every returned label must be present on each target receiving the request.
    pub fn required_labels(&self) -> &BTreeSet<String> {
        &self.required_labels
    }

    /// Returns `standard` for empty requirements and `restricted` otherwise.
    pub fn lane(&self) -> &'static str {
        if self.required_labels.is_empty() {
            "standard"
        } else {
            "restricted"
        }
    }

    /// Returns `policy` or `external_signal`.
    pub fn source(&self) -> &'static str {
        self.source.into()
    }

    /// Returns a bounded reason for the effective requirements.
    pub const fn reason_code(&self) -> &'static str {
        match self.source {
            EligibilitySource::Policy | EligibilitySource::ExplicitEmpty => "all_clear",
            EligibilitySource::ExternalSignal => "restricted",
        }
    }
}

fn invalid(message: &'static str) -> LlmClientError {
    LlmClientError::InvalidRequest {
        message: message.to_owned(),
    }
}

// Provider-held history may contain data absent from the request being admitted.
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
            "eligibility routes do not support OpenAI Responses continuation state",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn request(labels: &[&str]) -> Request {
        let mut request = Request::default();
        require_target_labels(&mut request, labels).expect("trusted requirements");
        request
    }

    fn labels(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|label| (*label).to_owned()).collect()
    }

    fn metadata(request: &mut Request) -> &mut BTreeMap<String, String> {
        request
            .metadata
            .get_or_insert_default()
            .extra_metadata
            .get_or_insert_default()
    }

    fn assert_labels(policy: &EligibilityPolicy, request: &Request, expected: &[&str]) {
        assert_eq!(
            policy.admit(request).unwrap().required_labels(),
            &labels(expected)
        );
    }

    #[test]
    fn requirements_are_normalized_and_only_accumulate() {
        let policy = EligibilityPolicy::new(false);
        let mut request = request(&["customer_data", "private", "private"]);
        require_target_labels(&mut request, ["approved"]).expect("additional requirement");
        require_target_labels(&mut request, [] as [&str; 0]).expect("cannot clear requirements");
        assert_labels(&policy, &request, &["approved", "customer_data", "private"]);
        let mut legacy = self::request(&[]);
        metadata(&mut legacy).insert(LEGACY_RESTRICTION_KEY.into(), "true".into());
        assert_labels(&policy, &legacy, &["private"]);
        require_target_labels(&mut legacy, ["customer_data"]).expect("merge legacy restriction");
        assert_eq!(
            target_requirements(&legacy).unwrap(),
            Some(labels(&["private", "customer_data"]))
        );
    }

    #[test]
    fn invalid_requirements_do_not_replace_valid_requirements() {
        let mut request = request(&["private"]);
        for invalid_label in [
            "",
            " ",
            " private",
            "private ",
            &"x".repeat(MAX_LABEL_BYTES + 1),
        ] {
            assert!(require_target_labels(&mut request, [invalid_label]).is_err());
        }
        assert_eq!(
            target_requirements(&request).unwrap(),
            Some(labels(&["private"]))
        );
        for encoded in ["null", "{}", "[1]", "[\"\"]", "["] {
            metadata(&mut request).insert(REQUIREMENTS_KEY.into(), encoded.into());
            assert!(has_target_requirements(&request));
            assert!(EligibilityPolicy::new(false).admit(&request).is_err());
            assert!(require_target_labels(&mut request, ["private"]).is_err());
        }
    }

    #[test]
    fn required_labels_distinguish_missing_from_empty() {
        let policy = EligibilityPolicy::new(true);
        assert!(policy.admit(&Request::default()).is_err());
        let decision = policy
            .admit(&request(&[]))
            .expect("explicit empty requirements");
        let legacy: &crate::EligibilityDecision = &decision;
        let _: &crate::PrivacyDecision = legacy;
        assert!(decision.required_labels().is_empty());
        assert_eq!(
            (decision.lane(), decision.source(), decision.reason_code()),
            ("standard", "external_signal", "all_clear")
        );
        let mut legacy = Request::default();
        metadata(&mut legacy).insert(LEGACY_RESTRICTION_KEY.into(), "false".into());
        assert!(policy.admit(&legacy).unwrap().required_labels().is_empty());
        metadata(&mut legacy).insert(LEGACY_RESTRICTION_KEY.into(), "invalid".into());
        assert!(policy.admit(&legacy).is_err());
    }

    #[test]
    fn rejected_provider_history_does_not_erase_valid_requirements() {
        let policy = EligibilityPolicy::new(false);
        for field in RESPONSES_STATE_FIELDS {
            let mut request = request(&["private"]);
            request
                .llm_request
                .extensions
                .fields
                .insert(field.into(), Value::String("hidden".into()));
            assert!(policy.admit(&request).is_err());
            request.llm_request.extensions.fields.remove(field);
            assert_labels(&policy, &request, &["private"]);
            let mut preserved = self::request(&[]);
            preserved.llm_request.preservation.requests.insert(
                WireFormat::OpenAiResponses.as_str().into(),
                serde_json::json!({field: "hidden"}),
            );
            assert!(policy.admit(&preserved).is_err());
        }
    }

    #[test]
    fn label_limits_reject_without_replacing_valid_requirements() {
        let mut request = request(&["private"]);
        assert!(
            require_target_labels(&mut request, (0..MAX_LABELS).map(|index| index.to_string()))
                .is_err()
        );
        assert_eq!(
            target_requirements(&request).unwrap(),
            Some(labels(&["private"]))
        );
    }
}
