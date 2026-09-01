// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::PyErr;
use thiserror::Error;

/// Errors produced while extracting prefill features.
#[derive(Debug, Error)]
pub enum PrefillRouterError {
    /// The learned router could not be constructed from its configuration.
    #[error("invalid prefill-router configuration: {0}")]
    InvalidConfiguration(String),

    /// The caller supplied an invalid prefill request.
    #[error("invalid prefill request: {0}")]
    InvalidRequest(String),

    /// An embedded Python operation failed.
    #[error("embedded Python {operation} failed: {source}")]
    Python {
        operation: &'static str,
        #[source]
        source: PyErr,
    },

    /// The embedded implementation returned an invalid result.
    #[error("invalid prefill result: {0}")]
    InvalidResult(String),
}

pub(crate) fn python_error(operation: &'static str, source: PyErr) -> PrefillRouterError {
    PrefillRouterError::Python { operation, source }
}

/// Result type returned by this crate.
pub type Result<T> = std::result::Result<T, PrefillRouterError>;
