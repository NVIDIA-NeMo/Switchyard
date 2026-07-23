// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the shared stage_router picker.
//!
//! Exposes [`switchyard_components::stage_router::pick_tier`] as
//! `stage_pick_tier(signal, picker_mode, confidence_threshold) -> PickOutcome`
//! so the Python `processor.py` runs the exact same routing decision as the Rust
//! profile. The async classifier and the `no_signal` case stay in Python: this
//! returns a resolved decision, or a request to consult the classifier.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use switchyard_components::stage_router::{
    pick_tier, score_signal, PickOutcome, PickerMode, Tier,
};

use super::dimension_collector::PyToolResultSignal;

fn parse_picker_mode(mode: &str) -> PyResult<PickerMode> {
    match mode {
        "capable_first" => Ok(PickerMode::CapableFirst),
        "efficient_first" => Ok(PickerMode::EfficientFirst),
        other => Err(PyValueError::new_err(format!(
            "unknown picker mode {other:?} (expected capable_first or efficient_first)"
        ))),
    }
}

fn tier_str(tier: Tier) -> &'static str {
    match tier {
        Tier::Capable => "capable",
        Tier::Efficient => "efficient",
    }
}

/// Result of [`stage_pick_tier`].
///
/// `resolved` is `True` when the picker decided without the classifier — then
/// `tier` and `source` are set. `resolved` is `False` when the scorer was not
/// confident: the caller runs its classifier, and falls back to `default_tier`
/// if it has none or it fails. `score` / `confidence` are always the scorer's.
#[pyclass(name = "PickOutcome", frozen)]
pub(crate) struct PyPickOutcome {
    resolved: bool,
    tier: Option<&'static str>,
    source: Option<&'static str>,
    default_tier: &'static str,
    score: f64,
    confidence: Option<f64>,
}

#[pymethods]
impl PyPickOutcome {
    #[getter]
    fn resolved(&self) -> bool {
        self.resolved
    }

    /// Chosen tier (`"capable"` / `"efficient"`) — only when `resolved`.
    #[getter]
    fn tier(&self) -> Option<&'static str> {
        self.tier
    }

    /// Decision source (`"override"` / `"tests_passed"` / `"dimensions"`) — only
    /// when `resolved`.
    #[getter]
    fn source(&self) -> Option<&'static str> {
        self.source
    }

    /// Tier to fall open to when the classifier does not resolve the turn.
    #[getter]
    fn default_tier(&self) -> &'static str {
        self.default_tier
    }

    #[getter]
    fn score(&self) -> f64 {
        self.score
    }

    #[getter]
    fn confidence(&self) -> Option<f64> {
        self.confidence
    }

    fn __repr__(&self) -> String {
        if self.resolved {
            format!(
                "PickOutcome(resolved, tier={:?}, source={:?}, score={:.3})",
                self.tier, self.source, self.score
            )
        } else {
            format!(
                "PickOutcome(consult_classifier, default_tier={:?}, score={:.3})",
                self.default_tier, self.score
            )
        }
    }
}

/// Decide a turn's tier from its [`ToolResultSignal`], up to (but not including)
/// the classifier. `picker_mode` is `"capable_first"` or `"efficient_first"`.
#[pyfunction]
fn stage_pick_tier(
    signal: PyRef<'_, PyToolResultSignal>,
    picker_mode: &str,
    confidence_threshold: f64,
) -> PyResult<PyPickOutcome> {
    let mode = parse_picker_mode(picker_mode)?;
    let outcome = match pick_tier(signal.core(), mode, confidence_threshold) {
        PickOutcome::Resolved {
            tier,
            source,
            score,
            confidence,
        } => PyPickOutcome {
            resolved: true,
            tier: Some(tier_str(tier)),
            source: Some(source.as_str()),
            default_tier: tier_str(tier),
            score,
            confidence,
        },
        PickOutcome::ConsultClassifier {
            score,
            confidence,
            default_tier,
        } => PyPickOutcome {
            resolved: false,
            tier: None,
            source: None,
            default_tier: tier_str(default_tier),
            score,
            confidence: Some(confidence),
        },
    };
    Ok(outcome)
}

/// The pure two-axis scorer for a signal, as `(score, confidence)`. Used by
/// offline analysis to replay the raw score independent of the picker mode.
#[pyfunction]
fn stage_score_signal(signal: PyRef<'_, PyToolResultSignal>) -> (f64, f64) {
    let result = score_signal(signal.core());
    (result.score, result.confidence)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPickOutcome>()?;
    module.add_function(wrap_pyfunction!(stage_pick_tier, module)?)?;
    module.add_function(wrap_pyfunction!(stage_score_signal, module)?)?;
    Ok(())
}
