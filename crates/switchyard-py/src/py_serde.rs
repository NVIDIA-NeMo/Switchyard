// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python object and JSON value conversion helpers.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::{depythonize, pythonize};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

/// Converts a Python mapping-like object into a Serde-owned Rust value.
pub(crate) fn from_python<T: DeserializeOwned>(value: &Bound<'_, PyAny>) -> PyResult<T> {
    let normalized = jsonable_python(value)?;
    depythonize(normalized.bind(value.py()))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Converts a serializable Rust value into Python dictionaries and lists.
pub(crate) fn to_python<T: Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    pythonize(py, value)
        .map(|object| object.unbind())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Converts an arbitrary Python object into a JSON value.
pub(crate) fn value_from_python(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    from_python(value)
}

/// Converts a JSON value into a Python object.
pub(crate) fn value_to_python(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    to_python(py, value)
}

fn jsonable_python(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(model_dump) = value.getattr("model_dump") {
        if model_dump.is_callable() {
            let kwargs = PyDict::new(value.py());
            kwargs.set_item("mode", "json")?;
            kwargs.set_item("exclude_none", true)?;
            return model_dump.call((), Some(&kwargs)).map(Bound::unbind);
        }
    }
    if let Ok(to_dict) = value.getattr("to_dict") {
        if to_dict.is_callable() {
            return to_dict.call0().map(Bound::unbind);
        }
    }
    Ok(value.clone().unbind())
}
