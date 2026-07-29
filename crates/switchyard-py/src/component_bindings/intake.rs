// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Rust-owned intake metadata values.

use std::collections::BTreeMap;

use pyo3::class::basic::CompareOp;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyType};
use serde::Serialize;
use switchyard_components::{IntakeRequestMetadata, RequestMetadata};

use crate::interop::context::insert_into_python;
use crate::py_serde::value_to_python;

#[pyclass(name = "IntakeRequestMetadata", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyIntakeRequestMetadata {
    inner: IntakeRequestMetadata,
}

impl PyIntakeRequestMetadata {
    fn from_core(inner: IntakeRequestMetadata) -> Self {
        Self { inner }
    }

    fn clone_core(&self) -> IntakeRequestMetadata {
        self.inner.clone()
    }
}

#[pymethods]
impl PyIntakeRequestMetadata {
    #[new]
    #[pyo3(signature = (enabled=None, app=None, task=None))]
    fn py_new(enabled: Option<bool>, app: Option<String>, task: Option<String>) -> Self {
        Self {
            inner: IntakeRequestMetadata { enabled, app, task },
        }
    }

    #[getter]
    fn enabled(&self) -> Option<bool> {
        self.inner.enabled
    }

    #[getter]
    fn app(&self) -> Option<String> {
        self.inner.app.clone()
    }

    #[getter]
    fn task(&self) -> Option<String> {
        self.inner.task.clone()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        match op {
            CompareOp::Eq | CompareOp::Ne => {
                let equals = other
                    .extract::<PyRef<'_, Self>>()
                    .map(|other| self.inner == other.inner)
                    .unwrap_or(false);
                let result = if matches!(op, CompareOp::Eq) {
                    equals
                } else {
                    !equals
                };
                Ok(PyBool::new(py, result).to_owned().unbind().into_any())
            }
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "IntakeRequestMetadata(enabled={:?}, app={:?}, task={:?})",
            self.inner.enabled, self.inner.app, self.inner.task,
        )
    }
}

#[pyclass(name = "RequestMetadata", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyRequestMetadata {
    inner: RequestMetadata,
}

impl PyRequestMetadata {
    fn from_core(inner: RequestMetadata) -> Self {
        Self { inner }
    }

    pub(crate) fn clone_core(&self) -> RequestMetadata {
        self.inner.clone()
    }
}

#[pymethods]
impl PyRequestMetadata {
    #[new]
    #[pyo3(signature = (session_id=None, intake=None))]
    fn py_new(
        session_id: Option<String>,
        intake: Option<PyRef<'_, PyIntakeRequestMetadata>>,
    ) -> Self {
        Self {
            inner: RequestMetadata {
                session_id,
                intake: intake
                    .map(|metadata| metadata.clone_core())
                    .unwrap_or_default(),
            },
        }
    }

    #[classmethod]
    fn from_headers(_cls: &Bound<'_, PyType>, headers: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_core(RequestMetadata::from_headers(
            &headers_to_map(headers)?,
        )))
    }

    #[getter]
    fn session_id(&self) -> Option<String> {
        self.inner.session_id.clone()
    }

    #[getter]
    fn intake(&self) -> PyIntakeRequestMetadata {
        PyIntakeRequestMetadata::from_core(self.inner.intake.clone())
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }

    fn apply_to_context(&self, ctx: &Bound<'_, PyAny>) -> PyResult<()> {
        insert_into_python(ctx, self.clone_core())
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        match op {
            CompareOp::Eq | CompareOp::Ne => {
                let equals = other
                    .extract::<PyRef<'_, Self>>()
                    .map(|other| self.inner == other.inner)
                    .unwrap_or(false);
                let result = if matches!(op, CompareOp::Eq) {
                    equals
                } else {
                    !equals
                };
                Ok(PyBool::new(py, result).to_owned().unbind().into_any())
            }
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RequestMetadata(session_id={:?}, intake={:?})",
            self.inner.session_id, self.inner.intake,
        )
    }
}

fn headers_to_map(headers: &Bound<'_, PyAny>) -> PyResult<BTreeMap<String, String>> {
    let dict = PyDict::new(headers.py());
    dict.call_method1("update", (headers,))?;
    dict.iter()
        .map(|(key, value)| Ok((key.extract::<String>()?, value.extract::<String>()?)))
        .collect()
}

fn to_python(py: Python<'_>, value: &impl Serialize) -> PyResult<Py<PyAny>> {
    let value =
        serde_json::to_value(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    value_to_python(py, &value)
}

pub(crate) fn request_metadata_from_mapping(
    py: Python<'_>,
    metadata: &Bound<'_, PyAny>,
) -> PyResult<Option<RequestMetadata>> {
    let dict = PyDict::new(py);
    dict.call_method1("update", (metadata,))?;
    let Some(value) = dict.get_item("_request_metadata")? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let request_metadata = value
        .extract::<PyRef<'_, PyRequestMetadata>>()
        .map_err(|_| {
            PyTypeError::new_err(
                "ProxyContext metadata['_request_metadata'] must be a RequestMetadata value",
            )
        })?;
    Ok(Some(request_metadata.clone_core()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIntakeRequestMetadata>()?;
    module.add_class::<PyRequestMetadata>()?;
    Ok(())
}
