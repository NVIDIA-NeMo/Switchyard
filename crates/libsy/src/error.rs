// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed failures surfaced by libsy's orchestration APIs.

use std::{error::Error as StdError, time::Duration};

use thiserror::Error;

/// Result type returned by libsy APIs.
pub type Result<T> = std::result::Result<T, LibsyError>;

/// Failures surfaced while selecting a route, driving an algorithm, or serving a model call.
#[derive(Debug, Error)]
pub enum LibsyError {
    /// A named target was not present in the configured target set.
    #[error("target {target:?} was not found")]
    TargetNotFound {
        /// Missing semantic target name.
        target: String,
    },

    /// Routing was attempted without any configured targets.
    #[error("no routing targets are configured")]
    NoTargets,

    /// Every classifier in a fall-through cascade abstained.
    #[error("every classifier abstained")]
    AllClassifiersAbstained,

    /// A classifier returned an unorderable confidence.
    #[error("classifier returned NaN confidence for target {target:?}")]
    InvalidConfidence {
        /// Target associated with the invalid score.
        target: String,
    },

    /// A routed target had no default client for [`crate::Algorithm::run`].
    #[error("target {target:?} has no client to serve the call")]
    MissingClient {
        /// Target that could not be served.
        target: String,
    },

    /// The type-erased offload driver could not complete an operation.
    #[error(transparent)]
    Driver(#[from] DriverError),

    /// The spawned algorithm task failed before returning normally.
    #[error("algorithm task failed: {source}")]
    AlgorithmTask {
        /// Tokio task failure, including panic and unexpected cancellation details.
        #[from]
        #[source]
        source: tokio::task::JoinError,
    },

    /// An algorithm's step stream ended without a terminal response.
    #[error("algorithm run ended without a final response")]
    MissingFinalResponse,

    /// A target's protocol client failed while serving a routed request.
    #[error("client call to target {target:?} failed: {source}")]
    ClientCall {
        /// Target whose client failed.
        target: String,
        /// Error supplied by the protocol-owned client trait.
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },

    /// A user extension or other foreign operation failed.
    #[error("{operation} failed: {source}")]
    External {
        /// Short description of the operation that failed.
        operation: &'static str,
        /// Original failure.
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
}

impl LibsyError {
    /// Wrap an error returned by the protocol-owned client trait.
    pub fn client_call(target: impl Into<String>, source: Box<dyn StdError + Send + Sync>) -> Self {
        Self::ClientCall {
            target: target.into(),
            source,
        }
    }

    /// Preserve a concrete foreign error with a description of the failed operation.
    pub fn external(
        operation: &'static str,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::External {
            operation,
            source: Box::new(source),
        }
    }

    /// Preserve an already boxed foreign error.
    pub fn external_boxed(
        operation: &'static str,
        source: Box<dyn StdError + Send + Sync>,
    ) -> Self {
        Self::External { operation, source }
    }
}

/// Failures in the type-erased promise-over-stream driver.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DriverError {
    /// A producer operation was attempted before taking the consumer stream.
    #[error("driver stream must be taken before calling producer methods")]
    NotStarted,

    /// The consumer side of the step channel was dropped.
    #[error("driver stream is closed")]
    StreamClosed,

    /// The single-consumer stream had already been taken.
    #[error("driver stream was already taken")]
    StreamAlreadyTaken,

    /// One side of a response promise was dropped before delivery.
    #[error("driver response promise was dropped")]
    ResponseDropped,

    /// A consumer did not fulfill a request before its deadline.
    #[error("driver response timed out after {timeout:?}")]
    ResponseTimedOut {
        /// Maximum time allowed for request fulfillment.
        timeout: Duration,
    },

    /// A type-erased payload did not contain the expected concrete type.
    #[error("driver payload type mismatch: expected {expected}")]
    TypeMismatch {
        /// Human-readable expected payload type or role.
        expected: &'static str,
    },

    /// The driver's receiver lock was poisoned by a panic.
    #[error("driver receiver lock was poisoned")]
    LockPoisoned,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_call_preserves_target_and_source() {
        let error =
            LibsyError::client_call("strong", Box::new(std::io::Error::other("upstream down")));
        match &error {
            LibsyError::ClientCall { target, source } => {
                assert_eq!(target, "strong");
                assert_eq!(source.to_string(), "upstream down");
            }
            other => panic!("expected ClientCall, got {other:?}"),
        }
        assert_eq!(
            StdError::source(&error).map(ToString::to_string),
            Some("upstream down".to_string())
        );
    }

    #[test]
    fn external_preserves_operation_and_source() {
        let error = LibsyError::external(
            "loading extension",
            std::io::Error::other("bad configuration"),
        );
        assert!(matches!(
            &error,
            LibsyError::External { operation, .. } if *operation == "loading extension"
        ));
        assert_eq!(
            StdError::source(&error).map(ToString::to_string),
            Some("bad configuration".to_string())
        );
    }
}
