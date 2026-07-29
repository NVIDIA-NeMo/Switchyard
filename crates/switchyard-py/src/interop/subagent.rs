// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the canonical sub-agent detection policy.
//!
//! Profiles must not sniff lineage headers themselves: the fact (explicit
//! `x-switchyard-is-subagent`, Claude Code agent lineage, Codex/relay markers) and the
//! work-vs-maintenance policy both live in the protocol crate, so every engine — the libsy
//! classifier and the serve-path profile alike — answers "is this delegated work?"
//! identically.

use std::collections::BTreeMap;

use pyo3::prelude::*;

/// Whether `headers` mark this request as delegated sub-agent *work*.
///
/// Wraps [`switchyard_protocol::Metadata::from_headers`] for the lineage fact and
/// `is_subagent_work` for the kind policy, so harness maintenance turns (Codex `compact`,
/// `memory_consolidation`) stay on normal routing rather than being sent to a worker target.
#[pyfunction]
fn is_subagent_request(headers: BTreeMap<String, String>) -> bool {
    switchyard_protocol::Metadata::from_headers(&headers).is_subagent_work()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(is_subagent_request, module)?)?;
    Ok(())
}
