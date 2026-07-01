// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Errors returned while validating or executing skill-distillation contracts.

use thiserror::Error;

/// Result type shared by the skill-distillation APIs.
pub type Result<T> = std::result::Result<T, SkillDistillationError>;

/// Error returned by record validation or an extension-point implementation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SkillDistillationError {
    /// An identifier is not a safe local path component.
    #[error("invalid {kind}: {reason}")]
    InvalidIdentifier {
        /// Identifier category, such as `trajectory id`.
        kind: &'static str,
        /// Concrete validation failure.
        reason: String,
    },

    /// A serialized record did not satisfy a required invariant.
    #[error("invalid {record}: {reason}")]
    InvalidRecord {
        /// Record category, such as `trajectory`.
        record: &'static str,
        /// Concrete validation failure.
        reason: String,
    },

    /// A record uses a schema version this crate cannot validate.
    #[error("unsupported schema version {actual}; expected {expected}")]
    UnsupportedSchemaVersion {
        /// Schema version supported by this crate.
        expected: u16,
        /// Schema version found in the record.
        actual: u16,
    },

    /// A score or metric is not a finite number.
    #[error("{field} must be finite")]
    NonFiniteNumber {
        /// Field containing the invalid number.
        field: String,
    },

    /// A trajectory source could not load its records.
    #[error("trajectory source failed: {0}")]
    Source(String),

    /// A distiller could not produce a skill candidate.
    #[error("skill distiller failed: {0}")]
    Distiller(String),

    /// A validator could not evaluate a skill candidate.
    #[error("skill validator failed: {0}")]
    Validator(String),

    /// A skill store could not read or update its state.
    #[error("skill store failed: {0}")]
    Store(String),

    /// A requested skill version was not available.
    #[error("skill version {version} was not found for namespace {namespace}")]
    VersionNotFound {
        /// Namespace searched by the store.
        namespace: String,
        /// Requested version identifier.
        version: String,
    },

    /// Rollback was requested without a previous active version.
    #[error("namespace {namespace} has no previous active skill version")]
    NoPreviousVersion {
        /// Namespace whose history was inspected.
        namespace: String,
    },
}
