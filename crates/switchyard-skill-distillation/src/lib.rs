// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-neutral contracts for turning agent trajectories into reusable skills.
//!
//! This crate owns the serializable records and async extension points shared by
//! trajectory sources, distillers, validators, and skill stores. It deliberately
//! does not choose a provider, storage format, agent runtime, or model implementation.
//!
//! These contracts do not run a workflow by themselves. A Switchyard-owned coordinator
//! loads saved trajectories and the active skill, builds a distillation request, checks
//! the candidate, then saves it and decides whether to make it active. This crate does
//! not define that coordinator, its schedule, or its activation policy.

#![deny(missing_docs)]

mod error;
mod ids;
mod model;
mod ports;

pub use error::{Result, SkillDistillationError};
pub use ids::{SkillEvidenceId, SkillNamespace, SkillVersionId};
pub use model::{
    ActivationOperation, ActivationRecord, DistillationRequest, ExecutionMetadata, Metadata,
    SkillCandidate, SkillProvenance, TaskDescriptor, Trajectory, TrajectoryEvent,
    TrajectoryEventKind, TrajectoryOutcome, TrajectorySourceInfo, ValidationCheck,
    ValidationReport, ValidationStatus, SCHEMA_VERSION,
};
pub use ports::{SkillDistiller, SkillStore, SkillValidator, TrajectorySource};
