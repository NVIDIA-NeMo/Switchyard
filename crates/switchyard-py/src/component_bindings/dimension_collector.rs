// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the dimension-collector context-signal layer.
//!
//! Exposes:
//!
//! * `DimensionScore` — one scorer's output (name + score + signal).
//! * `ContextSignals` — the aggregate stamped into `ProxyContext`
//!   (dimensions tuple + token-count estimate + agentic scalar).
//! * `ScoringConfig` — keyword lists + token-count thresholds.
//! * `DimensionCollector` — request-side component that runs the 15 scorers
//!   and stamps `ContextSignals` into the context.
//! * `get_context_signals(ctx)` — Python-facing reader so estimators
//!   built on top (LLM classifier, future rules estimator) can pick up
//!   the stamped signals.

use pyo3::prelude::*;
use pyo3::types::PyList;
use switchyard_components::{
    dimension_collector::{
        extract_response_signals as core_extract_response_signals, ContextSignals, DimensionScore,
        Keywords, ResponseFlag, ResponseSignals, ScoringConfig, ToolResultSignal,
        DEFAULT_RECENT_WINDOW,
    },
    DimensionCollector, ResponseSignalCollector,
};
use switchyard_core::ChatResponse;

use crate::py_serde::value_from_python;

use crate::core_bindings::context::PyProxyContext;
use crate::core_bindings::request::PyChatRequest;
use crate::core_bindings::response::PyChatResponse;

/// One scorer's output as a Python-visible record.
#[pyclass(name = "DimensionScore", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyDimensionScore {
    inner: DimensionScore,
}

#[pymethods]
impl PyDimensionScore {
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name
    }

    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }

    #[getter]
    fn signal(&self) -> Option<&str> {
        self.inner.signal.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "DimensionScore(name={:?}, score={}, signal={:?})",
            self.inner.name, self.inner.score, self.inner.signal,
        )
    }
}

impl PyDimensionScore {
    fn from_core(inner: DimensionScore) -> Self {
        Self { inner }
    }
}

/// Aggregate context-signal record stamped by [`PyDimensionCollector`].
#[pyclass(name = "ContextSignals", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyContextSignals {
    inner: ContextSignals,
}

#[pymethods]
impl PyContextSignals {
    /// The 15 scored dimensions in canonical order.
    #[getter]
    fn dimensions(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let items: Vec<Py<PyDimensionScore>> = self
            .inner
            .dimensions
            .iter()
            .map(|dim| Py::new(py, PyDimensionScore::from_core(dim.clone())))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, items)?.unbind())
    }

    /// Estimated input-token count (`chars / 4` heuristic for now).
    #[getter]
    fn token_count_estimate(&self) -> u32 {
        self.inner.token_count_estimate
    }

    fn __repr__(&self) -> String {
        format!(
            "ContextSignals(dimensions=<{} items>, token_count_estimate={})",
            self.inner.dimensions.len(),
            self.inner.token_count_estimate,
        )
    }
}

impl PyContextSignals {
    fn from_core(inner: ContextSignals) -> Self {
        Self { inner }
    }
}

/// Python-facing scoring config; keyword lists are lower-cased on the way in.
#[pyclass(name = "ScoringConfig", skip_from_py_object)]
#[derive(Clone, Debug, Default)]
pub(crate) struct PyScoringConfig {
    inner: ScoringConfig,
}

#[pymethods]
impl PyScoringConfig {
    #[new]
    #[pyo3(signature = (
        token_count_short = 50,
        token_count_long = 500,
        code_keywords = vec![],
        reasoning_keywords = vec![],
        simple_keywords = vec![],
        technical_keywords = vec![],
        creative_keywords = vec![],
        imperative_verbs = vec![],
        constraint_indicators = vec![],
        output_format_keywords = vec![],
        reference_keywords = vec![],
        negation_keywords = vec![],
        domain_specific_keywords = vec![],
    ))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        token_count_short: u32,
        token_count_long: u32,
        code_keywords: Vec<String>,
        reasoning_keywords: Vec<String>,
        simple_keywords: Vec<String>,
        technical_keywords: Vec<String>,
        creative_keywords: Vec<String>,
        imperative_verbs: Vec<String>,
        constraint_indicators: Vec<String>,
        output_format_keywords: Vec<String>,
        reference_keywords: Vec<String>,
        negation_keywords: Vec<String>,
        domain_specific_keywords: Vec<String>,
    ) -> Self {
        let inner = ScoringConfig {
            token_count: switchyard_components::dimension_collector::TokenCountThresholds {
                short: token_count_short,
                long: token_count_long,
            },
            code_keywords: Keywords::new(code_keywords),
            reasoning_keywords: Keywords::new(reasoning_keywords),
            simple_keywords: Keywords::new(simple_keywords),
            technical_keywords: Keywords::new(technical_keywords),
            creative_keywords: Keywords::new(creative_keywords),
            imperative_verbs: Keywords::new(imperative_verbs),
            constraint_indicators: Keywords::new(constraint_indicators),
            output_format_keywords: Keywords::new(output_format_keywords),
            reference_keywords: Keywords::new(reference_keywords),
            negation_keywords: Keywords::new(negation_keywords),
            domain_specific_keywords: Keywords::new(domain_specific_keywords),
        };
        Self { inner }
    }

    fn __repr__(&self) -> String {
        format!(
            "ScoringConfig(token_count_short={}, token_count_long={})",
            self.inner.token_count.short, self.inner.token_count.long,
        )
    }
}

impl PyScoringConfig {
    fn clone_core(&self) -> ScoringConfig {
        self.inner.clone()
    }
}

/// Request-side component that runs the dimension collector.
#[pyclass(name = "DimensionCollector", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyDimensionCollector {
    inner: DimensionCollector,
}

#[pymethods]
impl PyDimensionCollector {
    #[new]
    #[pyo3(signature = (config = None, *, recent_window = None))]
    fn py_new(config: Option<PyRef<'_, PyScoringConfig>>, recent_window: Option<usize>) -> Self {
        let scoring = config.map(|cfg| cfg.clone_core()).unwrap_or_default();
        let window = recent_window.unwrap_or(DEFAULT_RECENT_WINDOW);
        Self {
            inner: DimensionCollector::with_recent_window(scoring, window),
        }
    }

    fn startup<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) })
    }

    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) })
    }

    fn process<'py>(
        &self,
        py: Python<'py>,
        ctx: PyRef<'_, PyProxyContext>,
        request: PyRef<'_, PyChatRequest>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let processor = self.inner.clone();
        let mut lease = ctx.lease()?;
        let request = request.clone_core();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = processor.process(lease.context_mut()?, request).await;
            let restore_result = lease.restore();
            let request = result.map_err(crate::errors::py_core_error)?;
            restore_result?;
            Python::attach(|py| {
                Py::new(py, PyChatRequest::from_core(request)).map(|request| request.into_any())
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "DimensionCollector()"
    }
}

/// Returns the `ContextSignals` stamped by a `DimensionCollector` run.
///
/// Mirrors the Python idiom `ctx.metadata.get("context_signals")` from
/// the deleted Python implementation, but uses the typed extension bag
/// so consumers don't pay for the dict round-trip.
#[pyfunction]
fn get_context_signals(ctx: PyRef<'_, PyProxyContext>) -> PyResult<Option<PyContextSignals>> {
    Ok(ctx
        .get_cloned::<ContextSignals>()?
        .map(PyContextSignals::from_core))
}

/// Closed set of response-side quality flags emitted by the response
/// signal collector. Mirrors Rust's
/// [`switchyard_components::dimension_collector::ResponseFlag`] enum.
///
/// Python sees these as a class with attribute-style variants
/// (`ResponseFlag.MALFORMED_TOOL_CALL_JSON`, etc.) plus an `__eq__`
/// implementation so set / list membership checks work naturally.
#[pyclass(name = "ResponseFlag", eq, frozen, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PyResponseFlag {
    MalformedToolCallJson,
    EmptyResponse,
    TruncatedCompletion,
    MissingRequiredArgs,
}

impl PyResponseFlag {
    fn from_core(flag: ResponseFlag) -> Self {
        match flag {
            ResponseFlag::MalformedToolCallJson => Self::MalformedToolCallJson,
            ResponseFlag::EmptyResponse => Self::EmptyResponse,
            ResponseFlag::TruncatedCompletion => Self::TruncatedCompletion,
            ResponseFlag::MissingRequiredArgs => Self::MissingRequiredArgs,
        }
    }
}

#[pymethods]
impl PyResponseFlag {
    fn __repr__(&self) -> &'static str {
        match self {
            Self::MalformedToolCallJson => "ResponseFlag.MALFORMED_TOOL_CALL_JSON",
            Self::EmptyResponse => "ResponseFlag.EMPTY_RESPONSE",
            Self::TruncatedCompletion => "ResponseFlag.TRUNCATED_COMPLETION",
            Self::MissingRequiredArgs => "ResponseFlag.MISSING_REQUIRED_ARGS",
        }
    }

    fn __hash__(&self) -> u64 {
        *self as u64
    }
}

/// Aggregate response-side signals stamped by [`PyResponseSignalCollector`].
#[pyclass(name = "ResponseSignals", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyResponseSignals {
    inner: ResponseSignals,
}

#[pymethods]
impl PyResponseSignals {
    /// Failing flags, in the order the checker ran them.
    #[getter]
    fn flags(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let items: Vec<Py<PyResponseFlag>> = self
            .inner
            .flags
            .iter()
            .map(|flag| Py::new(py, PyResponseFlag::from_core(*flag)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, items)?.unbind())
    }

    /// True when at least one check failed; stage_router routers use this as
    /// the per-attempt acceptability gate.
    fn has_failures(&self) -> bool {
        self.inner.has_failures()
    }

    /// Python-friendly membership check; equivalent to
    /// `flag in signals.flags`.
    fn contains(&self, flag: PyResponseFlag) -> bool {
        self.inner
            .flags
            .iter()
            .any(|inner| PyResponseFlag::from_core(*inner) == flag)
    }

    fn __repr__(&self) -> String {
        format!("ResponseSignals(flags=<{} items>)", self.inner.flags.len())
    }
}

impl PyResponseSignals {
    fn from_core(inner: ResponseSignals) -> Self {
        Self { inner }
    }
}

/// Response-side component that runs response-side signal extraction.
#[pyclass(name = "ResponseSignalCollector", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyResponseSignalCollector;

#[pymethods]
impl PyResponseSignalCollector {
    #[new]
    fn py_new() -> Self {
        Self
    }

    fn startup<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) })
    }

    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) })
    }

    fn process<'py>(
        &self,
        py: Python<'py>,
        ctx: PyRef<'_, PyProxyContext>,
        mut response: PyRefMut<'_, PyChatResponse>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut lease = ctx.lease()?;
        let response = response.take_core(py)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = ResponseSignalCollector
                .process(lease.context_mut()?, response)
                .await;
            let restore_result = lease.restore();
            let response = result.map_err(crate::errors::py_core_error)?;
            restore_result?;
            Python::attach(|py| {
                Py::new(py, PyChatResponse::from_core(py, response)?)
                    .map(|response| response.into_any())
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "ResponseSignalCollector()"
    }
}

/// Returns the `ResponseSignals` stamped by a `ResponseSignalCollector` run.
///
/// `None` either means the collector hasn't run yet on this `ctx` or the
/// response was a streaming response (which the buffered-body checks
/// can't introspect).
#[pyfunction]
fn get_response_signals(ctx: PyRef<'_, PyProxyContext>) -> PyResult<Option<PyResponseSignals>> {
    Ok(ctx
        .get_cloned::<ResponseSignals>()?
        .map(PyResponseSignals::from_core))
}

/// Runs the response-side checks against an inline response body dict.
///
/// Intended for the stage-router, which needs to evaluate
/// `ResponseSignals` between attempts without going through a full
/// response-side pass. Accepts the response's `.body` Python
/// dict directly; works for any wire shape because the checks are
/// structure-based, not variant-based (`choices[...]` vs `content[...]`
/// dispatch happens inside the Rust checks themselves).
///
/// Returns an empty `ResponseSignals` (no failures) if the body is
/// `None` or not a dict — same fail-safe posture as the
/// `ResponseSignalCollector` adapter.
#[pyfunction]
fn extract_response_signals(body: Option<&Bound<'_, PyAny>>) -> PyResult<PyResponseSignals> {
    let Some(body) = body else {
        return Ok(PyResponseSignals::from_core(ResponseSignals::default()));
    };
    if body.is_none() {
        return Ok(PyResponseSignals::from_core(ResponseSignals::default()));
    }
    let value = value_from_python(body)?;
    // The four checks dispatch on body structure, not on the
    // `ChatResponse` variant. Wrap in OpenAI-completion just to satisfy
    // the type; Anthropic-shaped bodies still get their dedicated
    // checks via the `content[]` walk.
    let response = ChatResponse::openai_completion(value);
    Ok(PyResponseSignals::from_core(core_extract_response_signals(
        &response,
    )))
}

/// Tool-result context signals stamped by [`PyDimensionCollector`].
///
/// Read via :func:`get_tool_result_signal`. All fields default to neutral
/// values (0.0 / False) when no tool results are present in the request.
#[pyclass(name = "ToolResultSignal", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyToolResultSignal {
    inner: ToolResultSignal,
}

#[pymethods]
impl PyToolResultSignal {
    /// Max error severity across all matched patterns in the last tool result.
    /// ``0.0`` = clean; ``0.3`` = soft; ``0.7`` = hard; ``1.0`` = critical.
    #[getter]
    fn severity(&self) -> f32 {
        self.inner.severity
    }

    /// Pattern names that fired in the most recent tool result.
    #[getter]
    fn patterns(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let items: Vec<Py<pyo3::types::PyString>> = self
            .inner
            .patterns
            .iter()
            .map(|p| Ok(pyo3::types::PyString::new(py, p).unbind()))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, items)?.unbind())
    }

    /// Consecutive clean tool results at the end of history (``0`` if last failed).
    #[getter]
    fn no_error_streak(&self) -> u32 {
        self.inner.no_error_streak
    }

    /// Edit-type tool calls in the conversation (refinement work).
    #[getter]
    fn edit_count(&self) -> u32 {
        self.inner.edit_count
    }

    /// Write/create-type tool calls in the conversation (scaffolding work).
    #[getter]
    fn write_count(&self) -> u32 {
        self.inner.write_count
    }

    /// Read-type tool calls (Read tool + read-like Bash inspections).
    #[getter]
    fn read_count(&self) -> u32 {
        self.inner.read_count
    }

    /// TodoWrite calls — Opus struggle signal used by the strong-default drop gate.
    #[getter]
    fn todowrite_count(&self) -> u32 {
        self.inner.todowrite_count
    }

    /// Edit-type tool calls in the most recent 3 tool calls (sliding window).
    #[getter]
    fn recent_edit_count(&self) -> u32 {
        self.inner.recent_edit_count
    }

    /// Write/create-type tool calls in the most recent 3 tool calls (sliding window).
    #[getter]
    fn recent_write_count(&self) -> u32 {
        self.inner.recent_write_count
    }

    /// Read-type tool calls in the most recent 3 tool calls (sliding window).
    #[getter]
    fn recent_read_count(&self) -> u32 {
        self.inner.recent_read_count
    }

    /// TodoWrite calls in the most recent 3 tool calls (sliding window).
    #[getter]
    fn recent_todowrite_count(&self) -> u32 {
        self.inner.recent_todowrite_count
    }

    /// Consecutive trailing `Other`-category tool calls (build-pit proxy).
    #[getter]
    fn pure_bash_streak(&self) -> u32 {
        self.inner.pure_bash_streak
    }

    /// ``True`` when a recent tool result contained passing test output.
    #[getter]
    fn tests_passed(&self) -> bool {
        self.inner.tests_passed
    }

    /// Total messages in the conversation (turn-depth proxy).
    #[getter]
    fn turn_depth(&self) -> u32 {
        self.inner.turn_depth
    }

    /// Character count of the last user message (current-ask size).
    #[getter]
    fn prompt_char_count(&self) -> u32 {
        self.inner.prompt_char_count
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolResultSignal(severity={:.2}, streak={}, edit={}, write={}, tests_passed={})",
            self.inner.severity,
            self.inner.no_error_streak,
            self.inner.edit_count,
            self.inner.write_count,
            self.inner.tests_passed,
        )
    }
}

impl PyToolResultSignal {
    fn from_core(inner: ToolResultSignal) -> Self {
        Self { inner }
    }
}

/// Returns the :class:`ToolResultSignal` stamped by a :class:`DimensionCollector` run.
///
/// Returns ``None`` when the collector has not run on this context yet.
#[pyfunction]
fn get_tool_result_signal(ctx: PyRef<'_, PyProxyContext>) -> PyResult<Option<PyToolResultSignal>> {
    Ok(ctx
        .get_cloned::<ToolResultSignal>()?
        .map(PyToolResultSignal::from_core))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDimensionScore>()?;
    module.add_class::<PyContextSignals>()?;
    module.add_class::<PyScoringConfig>()?;
    module.add_class::<PyDimensionCollector>()?;
    module.add_class::<PyResponseFlag>()?;
    module.add_class::<PyResponseSignals>()?;
    module.add_class::<PyResponseSignalCollector>()?;
    module.add_class::<PyToolResultSignal>()?;
    module.add_function(wrap_pyfunction!(get_context_signals, module)?)?;
    module.add_function(wrap_pyfunction!(get_response_signals, module)?)?;
    module.add_function(wrap_pyfunction!(extract_response_signals, module)?)?;
    module.add_function(wrap_pyfunction!(get_tool_result_signal, module)?)?;
    Ok(())
}
