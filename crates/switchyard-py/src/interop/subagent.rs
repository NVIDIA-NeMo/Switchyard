// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the canonical sub-agent detection policy.
//!
//! Routing implementations must not sniff lineage headers themselves: the fact (explicit
//! `x-switchyard-is-subagent`, Claude Code agent lineage, Codex/relay markers) and the
//! work-vs-maintenance policy both live in the protocol crate, so every engine — the libsy
//! classifier and the server alike — answers "is this delegated work?"
//! identically.

use std::collections::HashMap;

use http::header::{HeaderName, HeaderValue};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Convert Python-owned headers without allowing invalid or excessive entries.
pub(crate) fn header_map_from_python(
    headers: &HashMap<String, String>,
) -> PyResult<http::HeaderMap> {
    let mut result = http::HeaderMap::new();

    for (name, value) in headers {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let value = HeaderValue::from_str(value)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        result
            .try_append(name, value)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    }

    Ok(result)
}

/// Whether `headers` mark this request as delegated sub-agent *work*.
///
/// Wraps [`switchyard_protocol::Metadata::from_headers`] for the lineage fact and
/// `is_subagent_work` for the kind policy, so harness maintenance turns (Codex `compact`,
/// `memory_consolidation`) stay on normal routing rather than being sent to a worker target.
#[pyfunction]
fn is_subagent_request(headers: HashMap<String, String>) -> PyResult<bool> {
    let headers = header_map_from_python(&headers)?;
    Ok(switchyard_protocol::Metadata::from_headers(&headers).is_subagent_work())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(is_subagent_request, module)?)?;
    Ok(())
}
