// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned, source-neutral records exchanged by skill-distillation stages.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, SkillDistillationError};
use crate::ids::{SkillEvidenceId, SkillNamespace, SkillVersionId};

/// Current version of serialized skill-distillation records.
pub const SCHEMA_VERSION: u16 = 1;

/// Extensible metadata with deterministic serialized key ordering.
pub type Metadata = BTreeMap<String, Value>;

/// Description of the task attempted by an agent.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskDescriptor {
    /// Human-readable task description.
    pub description: String,
    /// Optional source-specific task identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Additional source-specific task fields.
    #[serde(default)]
    pub metadata: Metadata,
}

/// Runtime metadata associated with one trajectory.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionMetadata {
    /// Agent harness or runner name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness: Option<String>,
    /// Model used for the run, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Run start time; adapters should use RFC3339 UTC.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    /// Run end time; adapters should use RFC3339 UTC.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<String>,
    /// Additional execution fields.
    #[serde(default)]
    pub metadata: Metadata,
}

/// Provenance identifying the system that supplied a trajectory.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectorySourceInfo {
    /// Source adapter name, such as a local runner or benchmark importer.
    pub kind: String,
    /// Optional source-local run or session ID kept when this record is created.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Additional source fields.
    #[serde(default)]
    pub metadata: Metadata,
}

/// Standard event categories understood by distillation implementations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TrajectoryEventKind {
    /// User, assistant, or system message.
    Message,
    /// Tool invocation request.
    ToolCall,
    /// Result returned by a tool invocation.
    ToolResult,
    /// Observation emitted by an agent or environment.
    Observation,
    /// Error encountered during execution.
    Error,
    /// Final output produced by the agent.
    FinalOutput,
}

/// One ordered event in an agent trajectory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryEvent {
    /// Zero-based contiguous event position.
    pub sequence: u64,
    /// Optional event time; adapters should use RFC3339 UTC.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Standard event category.
    pub kind: TrajectoryEventKind,
    /// Source-neutral event payload containing provider-specific data when needed.
    pub payload: Value,
    /// Additional event fields, including source-specific identifiers.
    #[serde(default)]
    pub metadata: Metadata,
}

/// Optional outcome evidence attached to a trajectory.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryOutcome {
    /// Source-defined label, such as `success`, `failure`, or `needs_review`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Optional numeric score or reward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Optional failure or evaluator message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Additional numeric evaluation metrics.
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
}

/// One completed agent run saved as evidence for skill distillation.
///
/// This is not the live session ID carried through request metadata. Code that builds
/// the record must reuse or deterministically derive its [`SkillEvidenceId`] from that
/// session ID.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trajectory {
    /// Serialized record schema version.
    pub schema_version: u16,
    /// Evidence ID used to reject duplicate inputs and record which runs a candidate used.
    pub id: SkillEvidenceId,
    /// Task attempted by the agent.
    pub task: TaskDescriptor,
    /// Runner and model metadata.
    pub execution: ExecutionMetadata,
    /// Source provenance.
    pub source: TrajectorySourceInfo,
    /// Ordered agent events.
    pub events: Vec<TrajectoryEvent>,
    /// Optional score or outcome evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<TrajectoryOutcome>,
    /// Additional trajectory fields.
    #[serde(default)]
    pub metadata: Metadata,
}

impl Trajectory {
    /// Validates the record before it is passed to a distiller.
    pub fn validate(&self) -> Result<()> {
        validate_schema(self.schema_version)?;
        validate_required_text("trajectory", "task.description", &self.task.description)?;
        validate_required_text("trajectory", "source.kind", &self.source.kind)?;
        validate_optional_text("trajectory", "execution.harness", &self.execution.harness)?;
        validate_optional_text("trajectory", "execution.model", &self.execution.model)?;
        validate_optional_text(
            "trajectory",
            "execution.started_at",
            &self.execution.started_at,
        )?;
        validate_optional_text("trajectory", "execution.ended_at", &self.execution.ended_at)?;
        validate_events(&self.events)?;
        if let Some(outcome) = &self.outcome {
            validate_outcome(outcome)?;
        }
        Ok(())
    }
}

/// Input supplied to a distiller for one target namespace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistillationRequest {
    /// Serialized record schema version.
    pub schema_version: u16,
    /// Namespace that receives the resulting candidate skill.
    pub namespace: SkillNamespace,
    /// Source-neutral trajectories to analyze.
    pub trajectories: Vec<Trajectory>,
    /// Existing skill to deepen or revise, when one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_skill: Option<SkillCandidate>,
    /// Additional request fields.
    #[serde(default)]
    pub metadata: Metadata,
}

impl DistillationRequest {
    /// Validates the request and every supplied trajectory.
    pub fn validate(&self) -> Result<()> {
        validate_schema(self.schema_version)?;
        if self.trajectories.is_empty() {
            return Err(invalid_record(
                "distillation request",
                "at least one trajectory is required",
            ));
        }

        let mut ids = HashSet::with_capacity(self.trajectories.len());
        for trajectory in &self.trajectories {
            trajectory.validate()?;
            if !ids.insert(&trajectory.id) {
                return Err(invalid_record(
                    "distillation request",
                    "skill evidence ids must be unique",
                ));
            }
        }

        if let Some(base_skill) = &self.base_skill {
            base_skill.validate()?;
            if base_skill.namespace != self.namespace {
                return Err(invalid_record(
                    "distillation request",
                    "base skill namespace does not match request namespace",
                ));
            }
        }
        Ok(())
    }

    /// Validates a distiller result against this request's namespace and evidence.
    pub fn validate_candidate(&self, candidate: &SkillCandidate) -> Result<()> {
        self.validate()?;
        candidate.validate()?;
        if candidate.namespace != self.namespace {
            return Err(invalid_record(
                "skill candidate",
                "candidate namespace does not match request namespace",
            ));
        }

        let request_ids: HashSet<_> = self
            .trajectories
            .iter()
            .map(|trajectory| &trajectory.id)
            .collect();
        if candidate
            .provenance
            .source_evidence_ids
            .iter()
            .any(|id| !request_ids.contains(id))
        {
            return Err(invalid_record(
                "skill candidate",
                "provenance references skill evidence outside the request",
            ));
        }

        let expected_parent = self.base_skill.as_ref().map(|skill| &skill.version);
        if candidate.provenance.parent_version.as_ref() != expected_parent {
            return Err(invalid_record(
                "skill candidate",
                "parent version does not match the request base skill",
            ));
        }
        Ok(())
    }
}

/// Provenance recorded on a generated skill candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkillProvenance {
    /// Completed runs used as evidence for the candidate.
    pub source_evidence_ids: Vec<SkillEvidenceId>,
    /// Previously active skill version, when this is an update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_version: Option<SkillVersionId>,
    /// Distiller or generator identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator: Option<String>,
    /// Generation time; adapters should use RFC3339 UTC.
    pub generated_at: String,
    /// Additional provenance fields.
    #[serde(default)]
    pub metadata: Metadata,
}

/// State of one validation check or validation report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ValidationStatus {
    /// Validation passed.
    Passed,
    /// Validation failed.
    Failed,
    /// A person must review the result.
    NeedsReview,
}

/// One named validation result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationCheck {
    /// Stable human-readable check name.
    pub name: String,
    /// Check result.
    pub status: ValidationStatus,
    /// Optional result detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Numeric measurements produced by the check.
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
}

/// Validation evidence associated with a skill candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationReport {
    /// Overall validation result.
    pub status: ValidationStatus,
    /// Individual checks supporting the result.
    pub checks: Vec<ValidationCheck>,
    /// Aggregate numeric measurements.
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    /// Human-readable review notes.
    #[serde(default)]
    pub notes: Vec<String>,
    /// Evaluation time; adapters should use RFC3339 UTC.
    pub evaluated_at: String,
}

impl ValidationReport {
    /// Validates names, timestamps, and numeric measurements in the report.
    pub fn validate(&self) -> Result<()> {
        validate_required_text("validation report", "evaluated_at", &self.evaluated_at)?;
        if self.checks.is_empty() {
            return Err(invalid_record(
                "validation report",
                "at least one validation check is required",
            ));
        }
        validate_finite_metrics("validation report.metrics", &self.metrics)?;
        for check in &self.checks {
            validate_required_text("validation report", "check.name", &check.name)?;
            validate_optional_text("validation report", "check.message", &check.message)?;
            validate_finite_metrics("validation check.metrics", &check.metrics)?;
        }
        Ok(())
    }
}

/// Generated skill content and its evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkillCandidate {
    /// Serialized record schema version.
    pub schema_version: u16,
    /// Namespace receiving this candidate.
    pub namespace: SkillNamespace,
    /// Stable candidate version identifier.
    pub version: SkillVersionId,
    /// Portable Agent Skills document content.
    pub skill_md: String,
    /// Evidence and generation metadata.
    pub provenance: SkillProvenance,
    /// Validation result, when validation has run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationReport>,
    /// Additional candidate fields.
    #[serde(default)]
    pub metadata: Metadata,
}

impl SkillCandidate {
    /// Validates the candidate before persistence or activation.
    pub fn validate(&self) -> Result<()> {
        validate_schema(self.schema_version)?;
        validate_required_text("skill candidate", "skill_md", &self.skill_md)?;
        validate_required_text(
            "skill candidate",
            "provenance.generated_at",
            &self.provenance.generated_at,
        )?;
        validate_optional_text(
            "skill candidate",
            "provenance.generator",
            &self.provenance.generator,
        )?;
        if self.provenance.source_evidence_ids.is_empty() {
            return Err(invalid_record(
                "skill candidate",
                "provenance must contain at least one source evidence id",
            ));
        }
        let unique_ids: HashSet<_> = self.provenance.source_evidence_ids.iter().collect();
        if unique_ids.len() != self.provenance.source_evidence_ids.len() {
            return Err(invalid_record(
                "skill candidate",
                "provenance source evidence ids must be unique",
            ));
        }
        if self.provenance.parent_version.as_ref() == Some(&self.version) {
            return Err(invalid_record(
                "skill candidate",
                "parent version must differ from candidate version",
            ));
        }
        if let Some(validation) = &self.validation {
            validation.validate()?;
        }
        Ok(())
    }
}

/// Whether an activation record represents promotion or rollback.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationOperation {
    /// Promote a saved version to active.
    Activate,
    /// Restore the immediately preceding active version.
    Rollback,
}

/// Observable result of changing the active skill version.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationRecord {
    /// Namespace whose active version changed.
    pub namespace: SkillNamespace,
    /// Operation performed by the store.
    pub operation: ActivationOperation,
    /// Version active after the operation.
    pub active_version: SkillVersionId,
    /// Version active immediately before the operation, when one existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_version: Option<SkillVersionId>,
    /// Operation time; stores should use RFC3339 UTC.
    pub recorded_at: String,
}

impl ActivationRecord {
    /// Validates the activation result before it is persisted or emitted.
    pub fn validate(&self) -> Result<()> {
        validate_required_text("activation record", "recorded_at", &self.recorded_at)?;
        if self.previous_version.as_ref() == Some(&self.active_version) {
            return Err(invalid_record(
                "activation record",
                "previous version must differ from active version",
            ));
        }
        Ok(())
    }
}

fn validate_schema(actual: u16) -> Result<()> {
    if actual != SCHEMA_VERSION {
        return Err(SkillDistillationError::UnsupportedSchemaVersion {
            expected: SCHEMA_VERSION,
            actual,
        });
    }
    Ok(())
}

fn validate_required_text(record: &'static str, field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(invalid_record(
            record,
            &format!("{field} must not be empty"),
        ));
    }
    Ok(())
}

fn validate_optional_text(record: &'static str, field: &str, value: &Option<String>) -> Result<()> {
    if let Some(value) = value {
        validate_required_text(record, field, value)?;
    }
    Ok(())
}

fn validate_events(events: &[TrajectoryEvent]) -> Result<()> {
    if events.is_empty() {
        return Err(invalid_record(
            "trajectory",
            "at least one event is required",
        ));
    }
    for (expected, event) in (0_u64..).zip(events) {
        if event.sequence != expected {
            return Err(invalid_record(
                "trajectory",
                "event sequence values must be contiguous and start at zero",
            ));
        }
        validate_optional_text("trajectory", "event.timestamp", &event.timestamp)?;
    }
    Ok(())
}

fn validate_outcome(outcome: &TrajectoryOutcome) -> Result<()> {
    validate_optional_text("trajectory", "outcome.label", &outcome.label)?;
    validate_optional_text("trajectory", "outcome.error", &outcome.error)?;
    if outcome.score.is_some_and(|score| !score.is_finite()) {
        return Err(SkillDistillationError::NonFiniteNumber {
            field: "trajectory outcome score".to_string(),
        });
    }
    validate_finite_metrics("trajectory outcome metrics", &outcome.metrics)
}

fn validate_finite_metrics(field: &str, metrics: &BTreeMap<String, f64>) -> Result<()> {
    if let Some((name, _)) = metrics.iter().find(|(_, value)| !value.is_finite()) {
        return Err(SkillDistillationError::NonFiniteNumber {
            field: format!("{field}.{name}"),
        });
    }
    Ok(())
}

fn invalid_record(record: &'static str, reason: &str) -> SkillDistillationError {
    SkillDistillationError::InvalidRecord {
        record,
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn namespace() -> Result<SkillNamespace> {
        SkillNamespace::new("trialqa")
    }

    fn trajectory() -> Result<Trajectory> {
        Ok(Trajectory {
            schema_version: SCHEMA_VERSION,
            id: SkillEvidenceId::new("run-1")?,
            task: TaskDescriptor {
                description: "answer the task".to_string(),
                ..TaskDescriptor::default()
            },
            execution: ExecutionMetadata::default(),
            source: TrajectorySourceInfo {
                kind: "test".to_string(),
                ..TrajectorySourceInfo::default()
            },
            events: vec![TrajectoryEvent {
                sequence: 0,
                timestamp: None,
                kind: TrajectoryEventKind::Message,
                payload: serde_json::json!({"role": "user", "content": "hello"}),
                metadata: Metadata::default(),
            }],
            outcome: None,
            metadata: Metadata::default(),
        })
    }

    fn candidate() -> Result<SkillCandidate> {
        Ok(SkillCandidate {
            schema_version: SCHEMA_VERSION,
            namespace: namespace()?,
            version: SkillVersionId::new("v1")?,
            skill_md: "# Skill".to_string(),
            provenance: SkillProvenance {
                source_evidence_ids: vec![SkillEvidenceId::new("run-1")?],
                parent_version: None,
                generator: None,
                generated_at: "2026-06-30T00:00:00Z".to_string(),
                metadata: Metadata::default(),
            },
            validation: None,
            metadata: Metadata::default(),
        })
    }

    #[test]
    fn trajectory_validation_accepts_unscored_runs() -> Result<()> {
        trajectory()?.validate()
    }

    #[test]
    fn trajectory_validation_rejects_missing_required_content() -> Result<()> {
        let mut value = trajectory()?;
        value.task.description.clear();
        assert!(matches!(
            value.validate(),
            Err(SkillDistillationError::InvalidRecord { .. })
        ));
        Ok(())
    }

    #[test]
    fn trajectory_validation_rejects_non_contiguous_event_sequences() -> Result<()> {
        let mut value = trajectory()?;
        value.events[0].sequence = 1;
        assert!(matches!(
            value.validate(),
            Err(SkillDistillationError::InvalidRecord { .. })
        ));
        Ok(())
    }

    #[test]
    fn trajectory_validation_rejects_non_finite_scores() -> Result<()> {
        let mut value = trajectory()?;
        value.outcome = Some(TrajectoryOutcome {
            score: Some(f64::NAN),
            ..TrajectoryOutcome::default()
        });
        assert!(matches!(
            value.validate(),
            Err(SkillDistillationError::NonFiniteNumber { .. })
        ));
        Ok(())
    }

    #[test]
    fn request_rejects_duplicate_skill_evidence_ids() -> Result<()> {
        let run = trajectory()?;
        let request = DistillationRequest {
            schema_version: SCHEMA_VERSION,
            namespace: namespace()?,
            trajectories: vec![run.clone(), run],
            base_skill: None,
            metadata: Metadata::default(),
        };
        assert!(matches!(
            request.validate(),
            Err(SkillDistillationError::InvalidRecord { .. })
        ));
        Ok(())
    }

    #[test]
    fn candidate_requires_source_evidence_provenance() -> Result<()> {
        let mut value = candidate()?;
        value.provenance.source_evidence_ids.clear();
        assert!(matches!(
            value.validate(),
            Err(SkillDistillationError::InvalidRecord { .. })
        ));
        Ok(())
    }

    #[test]
    fn candidate_rejects_non_finite_validation_metrics() -> Result<()> {
        let mut value = candidate()?;
        value.validation = Some(ValidationReport {
            status: ValidationStatus::Failed,
            checks: vec![ValidationCheck {
                name: "leakage".to_string(),
                status: ValidationStatus::Failed,
                message: None,
                metrics: BTreeMap::from([("score".to_string(), f64::INFINITY)]),
            }],
            metrics: BTreeMap::new(),
            notes: Vec::new(),
            evaluated_at: "2026-06-30T00:01:00Z".to_string(),
        });
        assert!(matches!(
            value.validate(),
            Err(SkillDistillationError::NonFiniteNumber { .. })
        ));
        Ok(())
    }

    #[test]
    fn request_rejects_candidate_provenance_outside_input() -> Result<()> {
        let request = DistillationRequest {
            schema_version: SCHEMA_VERSION,
            namespace: namespace()?,
            trajectories: vec![trajectory()?],
            base_skill: None,
            metadata: Metadata::default(),
        };
        let mut value = candidate()?;
        value.provenance.source_evidence_ids = vec![SkillEvidenceId::new("other-run")?];
        assert!(matches!(
            request.validate_candidate(&value),
            Err(SkillDistillationError::InvalidRecord { .. })
        ));
        Ok(())
    }

    #[test]
    fn request_requires_matching_base_skill_namespace() -> Result<()> {
        let mut base = candidate()?;
        base.namespace = SkillNamespace::new("other")?;
        let request = DistillationRequest {
            schema_version: SCHEMA_VERSION,
            namespace: namespace()?,
            trajectories: vec![trajectory()?],
            base_skill: Some(base),
            metadata: Metadata::default(),
        };
        assert!(matches!(
            request.validate(),
            Err(SkillDistillationError::InvalidRecord { .. })
        ));
        Ok(())
    }
}
