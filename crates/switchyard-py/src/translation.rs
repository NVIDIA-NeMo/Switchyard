// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Value, json};
use switchyard_translation::{
    AggLlmResponse, LlmRequest, ProviderExtensions, TranslationEngine,
    TranslationError as CoreTranslationError, TranslationOutput, TranslationPolicy,
};

use crate::py_serde::{from_python, to_python};

create_exception!(
    switchyard.translation,
    TranslationError,
    PyException,
    "Translation failed. Attributes: kind (stable variant name), details (variant fields)."
);

/// Translate buffered JSON requests and responses with the default policy.
#[pyclass(name = "TranslationEngine", module = "switchyard.translation", frozen)]
struct PyTranslationEngine {
    inner: TranslationEngine,
    default_policy: TranslationPolicy,
}

#[pymethods]
impl PyTranslationEngine {
    #[new]
    fn new() -> Self {
        Self {
            inner: TranslationEngine::default(),
            default_policy: TranslationPolicy::default(),
        }
    }

    /// Decode a provider request, returning request IR and diagnostics.
    fn decode_request(
        &self,
        py: Python<'_>,
        source: &str,
        body: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let body: Value = from_python(body.as_any())?;
        let output = py.detach(|| {
            self.inner
                .decode_request(source, &body, &self.default_policy)
        });
        let output = into_python_result(py, output)?;
        let result = PyDict::new(py);
        result.set_item("request", to_python(py, &output.request)?)?;
        result.set_item("diagnostics", to_python(py, &output.diagnostics)?)?;
        Ok(result.unbind())
    }

    /// Encode request IR, returning a provider body and diagnostics.
    fn encode_request(
        &self,
        py: Python<'_>,
        target: &str,
        request: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let request: LlmRequest = from_python(request.as_any())?;
        let output = py.detach(|| {
            self.inner
                .encode_request(target, &request, &self.default_policy)
        });
        translation_output(py, output)
    }

    /// Decode a complete provider response, returning response IR and diagnostics.
    fn decode_response(
        &self,
        py: Python<'_>,
        source: &str,
        body: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let body: Value = from_python(body.as_any())?;
        let output = py.detach(|| {
            self.inner
                .decode_response(source, &body, &self.default_policy)
        });
        let output = into_python_result(py, output)?;
        let result = PyDict::new(py);
        result.set_item("response", to_python(py, &output.response)?)?;
        result.set_item("diagnostics", to_python(py, &output.diagnostics)?)?;
        Ok(result.unbind())
    }

    /// Encode response IR, optionally restoring tool details from request extensions.
    #[pyo3(signature = (target, response, *, request_extensions=None))]
    fn encode_response(
        &self,
        py: Python<'_>,
        target: &str,
        response: &Bound<'_, PyDict>,
        request_extensions: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        let response: AggLlmResponse = from_python(response.as_any())?;
        let request_extensions: ProviderExtensions = request_extensions
            .map(|extensions| from_python(extensions.as_any()))
            .transpose()?
            .unwrap_or_default();
        let output = py.detach(|| {
            self.inner.encode_response_with_extensions(
                target,
                &response,
                &request_extensions,
                &self.default_policy,
            )
        });
        translation_output(py, output)
    }

    /// Translate a request dictionary, returning body and diagnostics.
    fn translate_request(
        &self,
        py: Python<'_>,
        source: &str,
        target: &str,
        body: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let body: Value = from_python(body.as_any())?;
        let output = py.detach(|| {
            self.inner
                .translate_request(source, target, &body, &self.default_policy)
        });
        translation_output(py, output)
    }

    /// Translate a complete response dictionary, returning body and diagnostics.
    fn translate_response(
        &self,
        py: Python<'_>,
        source: &str,
        target: &str,
        body: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let body: Value = from_python(body.as_any())?;
        let output = py.detach(|| {
            self.inner
                .translate_response(source, target, &body, &self.default_policy)
        });
        translation_output(py, output)
    }
}

fn translation_output(
    py: Python<'_>,
    result: Result<TranslationOutput, CoreTranslationError>,
) -> PyResult<Py<PyDict>> {
    let output = into_python_result(py, result)?;
    let result = PyDict::new(py);
    result.set_item("body", to_python(py, &output.body)?)?;
    result.set_item("diagnostics", to_python(py, &output.diagnostics)?)?;
    Ok(result.unbind())
}

fn into_python_result<T>(py: Python<'_>, result: Result<T, CoreTranslationError>) -> PyResult<T> {
    match result {
        Ok(output) => Ok(output),
        Err(error) => {
            let exception = TranslationError::new_err(error.to_string());
            exception.value(py).setattr("kind", error.kind())?;
            exception
                .value(py)
                .setattr("details", to_python(py, &error_details(&error))?)?;
            Err(exception)
        }
    }
}

fn error_details(error: &CoreTranslationError) -> Value {
    match error {
        CoreTranslationError::InvalidJson(error) => json!({
            "message": error.to_string(),
            "line": error.line(),
            "column": error.column(),
        }),
        CoreTranslationError::InvalidType { path, expected } => {
            json!({"path": path, "expected": expected})
        }
        CoreTranslationError::UnsupportedTranslation { from, to } => {
            json!({"from": from, "to": to})
        }
        CoreTranslationError::LossyConversion(message) | CoreTranslationError::Other(message) => {
            json!({"message": message})
        }
        CoreTranslationError::UnknownField { path } => json!({"path": path}),
        CoreTranslationError::InvalidValue { path, message } => {
            json!({"path": path, "message": message})
        }
        CoreTranslationError::UpstreamFailure { error } => json!({"error": error}),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let translation = PyModule::new(module.py(), "translation")?;
    translation.add_class::<PyTranslationEngine>()?;
    translation.add(
        "TranslationError",
        module.py().get_type::<TranslationError>(),
    )?;
    module.add_submodule(&translation)
}
