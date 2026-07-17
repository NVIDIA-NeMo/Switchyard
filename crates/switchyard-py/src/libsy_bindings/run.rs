// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python execution for Rust-owned libsy algorithms.

use std::sync::Arc;

use libsy::{Algorithm, NoopAlgo, RandomAlgo};
use pyo3::prelude::*;

use super::protocol::PyDecision;
use super::target::PyLlmTargetSet;
use super::values::{PyContext, PyRequest, PyResponse};
use crate::errors::py_libsy_error;

/// Opaque Python handle to a Rust-owned routing algorithm.
#[pyclass(name = "Algorithm", module = "switchyard.libsy", frozen)]
pub(crate) struct PyAlgorithm {
    inner: Arc<dyn Algorithm>,
}

impl PyAlgorithm {
    fn new(inner: Arc<dyn Algorithm>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAlgorithm {
    /// Run the algorithm using its targets' default clients.
    #[pyo3(signature = (request, *, context=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        request: PyRef<'_, PyRequest>,
        context: Option<PyRef<'_, PyContext>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let algorithm = Arc::clone(&self.inner);
        let context = context
            .map(|context| context.clone_core())
            .unwrap_or_default();
        let request = request.clone_core();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (decisions, response) = algorithm
                .run(context, request)
                .await
                .map_err(py_libsy_error)?;
            Python::attach(|py| {
                let decisions = decisions
                    .into_iter()
                    .map(|decision| Py::new(py, PyDecision::new(decision)))
                    .collect::<PyResult<Vec<_>>>()?;
                let response = Py::new(py, PyResponse::from_core(response))?;
                Ok((decisions, response))
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "Algorithm()"
    }
}

/// Construct the no-op reference algorithm.
#[pyfunction(name = "noop")]
fn noop_algorithm() -> PyAlgorithm {
    PyAlgorithm::new(Arc::new(NoopAlgo {}))
}

/// Construct uniform random routing over the supplied targets.
#[pyfunction(name = "random")]
fn random_algorithm(py: Python<'_>, targets: PyRef<'_, PyLlmTargetSet>) -> PyResult<PyAlgorithm> {
    Ok(PyAlgorithm::new(Arc::new(RandomAlgo::new(
        targets.clone_core(py)?,
    ))))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAlgorithm>()?;
    module.add_function(wrap_pyfunction!(noop_algorithm, module)?)?;
    module.add_function(wrap_pyfunction!(random_algorithm, module)?)?;
    Ok(())
}
