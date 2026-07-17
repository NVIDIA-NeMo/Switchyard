// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for libsy routing targets and host-provided clients.

use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;

/// A semantic routing target and its optional Python-hosted routed client.
#[pyclass(name = "LlmTarget", module = "switchyard.libsy")]
pub(crate) struct PyLlmTarget {
    semantic_name: String,
    llm_client: Option<Py<PyAny>>,
}

#[pymethods]
impl PyLlmTarget {
    #[new]
    #[pyo3(signature = (semantic_name, *, llm_client=None))]
    fn new(py: Python<'_>, semantic_name: String, llm_client: Option<Py<PyAny>>) -> PyResult<Self> {
        validate_client(py, llm_client.as_ref())?;
        Ok(Self {
            semantic_name,
            llm_client,
        })
    }

    #[getter]
    fn semantic_name(&self) -> &str {
        &self.semantic_name
    }

    #[getter]
    fn llm_client(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.llm_client.as_ref().map(|client| client.clone_ref(py))
    }

    #[setter]
    fn set_llm_client(&mut self, py: Python<'_>, llm_client: Option<Py<PyAny>>) -> PyResult<()> {
        validate_client(py, llm_client.as_ref())?;
        self.llm_client = llm_client;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "LlmTarget(semantic_name={:?}, llm_client={})",
            self.semantic_name,
            if self.llm_client.is_some() {
                "<configured>"
            } else {
                "None"
            }
        )
    }
}

/// Reusable routing targets. Algorithm factories snapshot their current clients.
#[pyclass(name = "LlmTargetSet", module = "switchyard.libsy", frozen)]
pub(crate) struct PyLlmTargetSet {
    targets: Vec<Py<PyLlmTarget>>,
}

#[pymethods]
impl PyLlmTargetSet {
    #[new]
    fn new(targets: Vec<Py<PyLlmTarget>>) -> Self {
        Self { targets }
    }

    #[getter]
    fn targets(&self, py: Python<'_>) -> Vec<Py<PyLlmTarget>> {
        self.targets
            .iter()
            .map(|target| target.clone_ref(py))
            .collect()
    }

    fn get_target(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyLlmTarget>> {
        for target in &self.targets {
            if target.bind(py).try_borrow()?.semantic_name == name {
                return Ok(target.clone_ref(py));
            }
        }
        Err(PyKeyError::new_err(name.to_string()))
    }

    fn __len__(&self) -> usize {
        self.targets.len()
    }

    fn __repr__(&self) -> String {
        format!("LlmTargetSet(len={})", self.targets.len())
    }
}

fn validate_client(py: Python<'_>, client: Option<&Py<PyAny>>) -> PyResult<()> {
    let Some(client) = client else {
        return Ok(());
    };
    let call = client.bind(py).getattr("call").map_err(|_| {
        PyTypeError::new_err("llm_client must define call(context, request, decision)")
    })?;
    if !call.is_callable() {
        return Err(PyTypeError::new_err(
            "llm_client.call must be an async callable",
        ));
    }
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLlmTarget>()?;
    module.add_class::<PyLlmTargetSet>()?;
    Ok(())
}
