// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the stats route label.
//!
//! The bespoke stats accumulator has been replaced by OpenTelemetry metrics, but
//! `set_stats_route_label` stays: Python routing processors stamp a tier label
//! that the intake payload builder reads for per-request route attribution.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use switchyard_components::StatsRouteLabel;

use crate::core_bindings::context::PyProxyContext;

/// Stamps a generic tier label on the proxy context for route attribution.
///
/// The label feeds the intake payload's route label and the OTel `tier`
/// attribution for Python-driven routers. An empty label is rejected.
#[pyfunction]
fn set_stats_route_label(ctx: PyRef<'_, PyProxyContext>, label: &str) -> PyResult<()> {
    let label = label.trim();
    if label.is_empty() {
        return Err(PyValueError::new_err("stats route label must not be empty"));
    }
    ctx.insert_value(StatsRouteLabel::new(label))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(set_stats_route_label, module)?)?;
    Ok(())
}
