// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::PyErr;
use thiserror::Error;

/// Errors produced while extracting prefill features.
#[derive(Debug, Error)]
pub enum PrefillRouterError {
    /// The embedded Transformers implementation failed.
    #[error("Transformers {operation} failed: {source}")]
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
