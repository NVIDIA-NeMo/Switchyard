// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed request input bindings shared by Rust and Python profile runtimes.
//!
//! Concrete profile config/runtime bindings intentionally live outside this
//! module now. Python owns its profile authoring API, while the profile-config
//! loader exposes Rust-defined profiles through the erased `Profile` runtime in
//! `mod.rs`.

use std::collections::BTreeMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use switchyard_components_v2::{reconcile_session_id, ProfileInput, RequestMetadata};
use switchyard_core::RequestId;

use crate::core_bindings::request::{request_type_from_python, request_type_object, PyChatRequest};
use crate::py_serde::value_to_python;

/// Endpoint-style metadata for direct Python profile calls.
#[pyclass(name = "ProfileRequestMetadata", frozen, skip_from_py_object)]
#[derive(Clone, Default)]
pub(crate) struct PyProfileRequestMetadata {
    inner: RequestMetadata,
}

impl PyProfileRequestMetadata {
    fn from_core(inner: RequestMetadata) -> Self {
        Self { inner }
    }

    pub(crate) fn clone_core(&self) -> RequestMetadata {
        self.inner.clone()
    }
}

#[pymethods]
impl PyProfileRequestMetadata {
    #[new]
    #[pyo3(signature = (request_id=None, inbound_format=None, headers=None, session_id=None))]
    fn new(
        request_id: Option<String>,
        inbound_format: Option<&Bound<'_, PyAny>>,
        headers: Option<&Bound<'_, PyAny>>,
        session_id: Option<String>,
    ) -> PyResult<Self> {
        let headers = match headers {
            Some(headers) if !headers.is_none() => metadata_headers_from_python(headers)?,
            _ => BTreeMap::new(),
        };
        let session_id = reconcile_profile_session_id(session_id.as_deref(), &headers)?;
        Ok(Self {
            inner: RequestMetadata {
                session_id,
                request_id: parse_request_id(request_id)?,
                inbound_format: inbound_format
                    .filter(|value| !value.is_none())
                    .map(request_type_from_python)
                    .transpose()?,
                headers,
            },
        })
    }

    /// Build metadata from headers, inferring request and session IDs.
    #[classmethod]
    #[pyo3(signature = (headers, inbound_format=None))]
    fn from_headers(
        _cls: &Bound<'_, PyType>,
        headers: &Bound<'_, PyAny>,
        inbound_format: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let headers = metadata_headers_from_python(headers)?;
        let request_id = headers
            .get("x-request-id")
            .and_then(|values| values.first())
            .cloned();
        let session_id = reconcile_profile_session_id(None, &headers)?;
        Ok(Self {
            inner: RequestMetadata {
                session_id,
                request_id: parse_request_id(request_id)?,
                inbound_format: inbound_format
                    .filter(|value| !value.is_none())
                    .map(request_type_from_python)
                    .transpose()?,
                headers,
            },
        })
    }

    #[getter]
    fn session_id(&self) -> Option<String> {
        self.inner.session_id.clone()
    }

    #[getter]
    fn request_id(&self) -> Option<String> {
        self.inner
            .request_id
            .as_ref()
            .map(|request_id| request_id.as_str().to_string())
    }

    #[getter]
    fn inbound_format(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .inbound_format
            .map(|request_type| request_type_object(py, request_type))
            .transpose()
    }

    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_python(
            py,
            &serde_json::to_value(&self.inner.headers)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )
    }

    /// Convert metadata into a plain Python dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("request_id", self.request_id())?;
        dict.set_item(
            "inbound_format",
            self.inner
                .inbound_format
                .map(crate::core_bindings::request::request_type_name),
        )?;
        dict.set_item("headers", self.headers(py)?)?;
        dict.set_item("session_id", self.session_id())?;
        Ok(dict.unbind().into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "ProfileRequestMetadata(request_id={:?}, session_id={:?})",
            self.request_id(),
            self.session_id()
        )
    }
}

/// Rust-owned request input for direct Python profile calls.
#[pyclass(name = "ProfileInput", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyProfileInput {
    inner: ProfileInput,
}

#[pymethods]
impl PyProfileInput {
    #[new]
    #[pyo3(signature = (request, metadata=None))]
    fn new(
        request: PyRef<'_, PyChatRequest>,
        metadata: Option<PyRef<'_, PyProfileRequestMetadata>>,
    ) -> Self {
        Self {
            inner: ProfileInput {
                request: request.clone_core(),
                metadata: metadata_to_core(metadata),
            },
        }
    }

    /// Provider-neutral chat request carried by this profile input.
    #[getter]
    fn request(&self) -> PyChatRequest {
        PyChatRequest::from_core(self.inner.request.clone())
    }

    /// Endpoint-style metadata carried by this profile input.
    #[getter]
    fn metadata(&self) -> PyProfileRequestMetadata {
        PyProfileRequestMetadata::from_core(self.inner.metadata.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "ProfileInput(request_model={:?}, request_id={:?})",
            self.inner.request.model(),
            self.inner
                .metadata
                .request_id
                .as_ref()
                .map(|request_id| request_id.as_str())
        )
    }
}

fn metadata_to_core(metadata: Option<PyRef<'_, PyProfileRequestMetadata>>) -> RequestMetadata {
    metadata
        .map(|metadata| metadata.clone_core())
        .unwrap_or_default()
}

fn parse_request_id(request_id: Option<String>) -> PyResult<Option<RequestId>> {
    request_id
        .map(RequestId::new)
        .transpose()
        .map_err(|error| PyValueError::new_err(format!("invalid request_id: {error}")))
}

fn reconcile_profile_session_id(
    session_id: Option<&str>,
    headers: &BTreeMap<String, Vec<String>>,
) -> PyResult<Option<String>> {
    reconcile_session_id(session_id, headers)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn metadata_headers_from_python(
    headers: &Bound<'_, PyAny>,
) -> PyResult<BTreeMap<String, Vec<String>>> {
    let dict = PyDict::new(headers.py());
    dict.call_method1("update", (headers,))?;
    let mut out = BTreeMap::<String, Vec<String>>::new();
    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?.to_ascii_lowercase();
        let values = metadata_header_values_from_python(&value)?;
        out.entry(key).or_default().extend(values);
    }
    Ok(out)
}

fn metadata_header_values_from_python(value: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(value) = value.extract::<String>() {
        return Ok(vec![value]);
    }
    value.extract::<Vec<String>>().map_err(|_| {
        PyTypeError::new_err(
            "ProfileRequestMetadata headers must map strings to strings or lists of strings",
        )
    })
}

/// Registers typed request input bindings with the native Python module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyProfileRequestMetadata>()?;
    module.add_class::<PyProfileInput>()?;
    Ok(())
}
