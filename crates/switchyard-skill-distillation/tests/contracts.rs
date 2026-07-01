// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public-contract tests from an external crate consumer's perspective.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use switchyard_skill_distillation::{
    ActivationOperation, ActivationRecord, DistillationRequest, ExecutionMetadata, Metadata,
    Result, SkillCandidate, SkillDistillationError, SkillDistiller, SkillNamespace,
    SkillProvenance, SkillStore, SkillValidator, SkillVersionId, TaskDescriptor, Trajectory,
    TrajectoryEvent, TrajectoryEventKind, TrajectoryId, TrajectoryOutcome, TrajectorySource,
    TrajectorySourceInfo, ValidationCheck, ValidationReport, ValidationStatus, SCHEMA_VERSION,
};
use tokio::sync::Mutex;

fn namespace() -> Result<SkillNamespace> {
    SkillNamespace::new("trialqa")
}

fn trajectory() -> Result<Trajectory> {
    Ok(Trajectory {
        schema_version: SCHEMA_VERSION,
        id: TrajectoryId::new("run-1")?,
        task: TaskDescriptor {
            description: "answer the task".to_string(),
            ..TaskDescriptor::default()
        },
        execution: ExecutionMetadata {
            harness: Some("test-harness".to_string()),
            model: Some("test-model".to_string()),
            started_at: Some("2026-06-30T00:00:00Z".to_string()),
            ended_at: Some("2026-06-30T00:01:00Z".to_string()),
            metadata: Metadata::default(),
        },
        source: TrajectorySourceInfo {
            kind: "test".to_string(),
            id: Some("source-run-1".to_string()),
            metadata: Metadata::default(),
        },
        events: vec![
            event(0, TrajectoryEventKind::Message, json!({"role": "user"})),
            event(1, TrajectoryEventKind::ToolCall, json!({"name": "search"})),
            event(
                2,
                TrajectoryEventKind::ToolResult,
                json!({"content": "result"}),
            ),
            event(
                3,
                TrajectoryEventKind::Observation,
                json!({"content": "observed"}),
            ),
            event(
                4,
                TrajectoryEventKind::Error,
                json!({"message": "recoverable"}),
            ),
            event(
                5,
                TrajectoryEventKind::FinalOutput,
                json!({"content": "done"}),
            ),
        ],
        outcome: Some(TrajectoryOutcome {
            label: Some("success".to_string()),
            score: Some(0.9),
            error: None,
            metrics: BTreeMap::from([("tokens".to_string(), 42.0)]),
        }),
        metadata: BTreeMap::from([("split".to_string(), json!("train"))]),
    })
}

fn event(sequence: u64, kind: TrajectoryEventKind, payload: serde_json::Value) -> TrajectoryEvent {
    TrajectoryEvent {
        sequence,
        timestamp: None,
        kind,
        payload,
        metadata: Metadata::default(),
    }
}

fn candidate(namespace: SkillNamespace, version: &str) -> Result<SkillCandidate> {
    Ok(SkillCandidate {
        schema_version: SCHEMA_VERSION,
        namespace,
        version: SkillVersionId::new(version)?,
        skill_md: "---\nname: trialqa\n---\n\n# Skill\n".to_string(),
        provenance: SkillProvenance {
            source_trajectory_ids: vec![TrajectoryId::new("run-1")?],
            parent_version: None,
            generator: Some("stub-distiller".to_string()),
            generated_at: "2026-06-30T00:02:00Z".to_string(),
            metadata: Metadata::default(),
        },
        validation: None,
        metadata: Metadata::default(),
    })
}

#[test]
fn all_event_kinds_round_trip_through_stable_json() -> Result<()> {
    let value = trajectory()?;
    value.validate()?;
    let encoded = serde_json::to_value(&value)
        .map_err(|error| SkillDistillationError::Store(error.to_string()))?;

    assert_eq!(encoded["schema_version"], SCHEMA_VERSION);
    assert_eq!(encoded["events"][0]["kind"], "message");
    assert_eq!(encoded["events"][1]["kind"], "tool_call");
    assert_eq!(encoded["events"][2]["kind"], "tool_result");
    assert_eq!(encoded["events"][3]["kind"], "observation");
    assert_eq!(encoded["events"][4]["kind"], "error");
    assert_eq!(encoded["events"][5]["kind"], "final_output");
    assert_eq!(encoded["metadata"]["split"], "train");

    let decoded: Trajectory = serde_json::from_value(encoded)
        .map_err(|error| SkillDistillationError::Store(error.to_string()))?;
    assert_eq!(decoded, value);
    Ok(())
}

#[derive(Clone)]
struct StaticSource {
    values: Vec<Trajectory>,
}

#[async_trait]
impl TrajectorySource for StaticSource {
    async fn load(&self, _namespace: &SkillNamespace) -> Result<Vec<Trajectory>> {
        Ok(self.values.clone())
    }
}

struct StubDistiller;

#[async_trait]
impl SkillDistiller for StubDistiller {
    async fn distill(&self, request: &DistillationRequest) -> Result<SkillCandidate> {
        request.validate()?;
        candidate(request.namespace.clone(), "v1")
    }
}

struct StubValidator;

#[async_trait]
impl SkillValidator for StubValidator {
    async fn validate(
        &self,
        candidate: &SkillCandidate,
        evaluation: &[Trajectory],
    ) -> Result<ValidationReport> {
        candidate.validate()?;
        Ok(ValidationReport {
            status: ValidationStatus::Passed,
            checks: vec![ValidationCheck {
                name: "has-evidence".to_string(),
                status: ValidationStatus::Passed,
                message: Some(format!("{} trajectories", evaluation.len())),
                metrics: BTreeMap::new(),
            }],
            metrics: BTreeMap::new(),
            notes: Vec::new(),
            evaluated_at: "2026-06-30T00:03:00Z".to_string(),
        })
    }
}

#[derive(Default)]
struct StoreState {
    saved: BTreeMap<(SkillNamespace, SkillVersionId), SkillCandidate>,
    active: BTreeMap<SkillNamespace, SkillCandidate>,
    previous: BTreeMap<SkillNamespace, SkillCandidate>,
}

#[derive(Clone, Default)]
struct MemoryStore {
    state: Arc<Mutex<StoreState>>,
}

#[async_trait]
impl SkillStore for MemoryStore {
    async fn active(&self, namespace: &SkillNamespace) -> Result<Option<SkillCandidate>> {
        Ok(self.state.lock().await.active.get(namespace).cloned())
    }

    async fn save_candidate(&self, candidate: &SkillCandidate) -> Result<()> {
        candidate.validate()?;
        self.state.lock().await.saved.insert(
            (candidate.namespace.clone(), candidate.version.clone()),
            candidate.clone(),
        );
        Ok(())
    }

    async fn activate(
        &self,
        namespace: &SkillNamespace,
        version: &SkillVersionId,
    ) -> Result<ActivationRecord> {
        let mut state = self.state.lock().await;
        let key = (namespace.clone(), version.clone());
        let saved = state.saved.get(&key).cloned().ok_or_else(|| {
            SkillDistillationError::VersionNotFound {
                namespace: namespace.to_string(),
                version: version.to_string(),
            }
        })?;
        let previous = state.active.insert(namespace.clone(), saved);
        if let Some(previous) = &previous {
            state.previous.insert(namespace.clone(), previous.clone());
        }
        let record = ActivationRecord {
            namespace: namespace.clone(),
            operation: ActivationOperation::Activate,
            active_version: version.clone(),
            previous_version: previous.map(|value| value.version),
            recorded_at: "2026-06-30T00:04:00Z".to_string(),
        };
        record.validate()?;
        Ok(record)
    }

    async fn rollback(&self, namespace: &SkillNamespace) -> Result<ActivationRecord> {
        let mut state = self.state.lock().await;
        let restored = state.previous.remove(namespace).ok_or_else(|| {
            SkillDistillationError::NoPreviousVersion {
                namespace: namespace.to_string(),
            }
        })?;
        let replaced = state.active.insert(namespace.clone(), restored.clone());
        let record = ActivationRecord {
            namespace: namespace.clone(),
            operation: ActivationOperation::Rollback,
            active_version: restored.version,
            previous_version: replaced.map(|value| value.version),
            recorded_at: "2026-06-30T00:05:00Z".to_string(),
        };
        record.validate()?;
        Ok(record)
    }
}

#[tokio::test]
async fn external_rust_implementations_can_compose_the_contracts() -> Result<()> {
    let namespace = namespace()?;
    let source = StaticSource {
        values: vec![trajectory()?],
    };
    let trajectories = source.load(&namespace).await?;
    let request = DistillationRequest {
        schema_version: SCHEMA_VERSION,
        namespace: namespace.clone(),
        trajectories: trajectories.clone(),
        base_skill: None,
        metadata: Metadata::default(),
    };

    let mut generated = StubDistiller.distill(&request).await?;
    request.validate_candidate(&generated)?;
    let report = StubValidator.validate(&generated, &trajectories).await?;
    report.validate()?;
    generated.validation = Some(report);
    request.validate_candidate(&generated)?;

    let store = MemoryStore::default();
    store.save_candidate(&generated).await?;
    let first_activation = store.activate(&namespace, &generated.version).await?;
    assert_eq!(first_activation.active_version, generated.version);

    let mut second = candidate(namespace.clone(), "v2")?;
    second.provenance.parent_version = Some(generated.version.clone());
    store.save_candidate(&second).await?;
    store.activate(&namespace, &second.version).await?;

    let rollback = store.rollback(&namespace).await?;
    assert_eq!(rollback.operation, ActivationOperation::Rollback);
    assert_eq!(store.active(&namespace).await?, Some(generated));
    Ok(())
}
