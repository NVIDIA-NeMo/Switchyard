// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for translation.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Value, json};
use switchyard_translation::{
    TranslationEngine, TranslationError as CoreTranslationError, TranslationOutput,
    TranslationPolicy,
};

use crate::py_serde::{from_python, to_python};

create_exception!(
    switchyard.translation,
    TranslationError,
    PyException,
    "Translation failed. Attributes: kind (stable variant name), details (variant fields)."
);

/// Translate buffered JSON requests and responses with the default policy.
///
/// Each call is independent; request context is not retained for later responses.
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

    /// Translate a request dictionary, returning body and diagnostics.
    ///
    /// Formats accept WireFormat members or their string values. Invalid argument
    /// types raise TypeError. Unknown formats and translation failures raise
    /// TranslationError.
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
    ///
    /// Formats and errors follow translate_request. Streaming events and metadata
    /// from a previous request are not supported by this method.
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
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let exception = TranslationError::new_err(error.to_string());
            exception.value(py).setattr("kind", error.kind())?;
            exception
                .value(py)
                .setattr("details", to_python(py, &error_details(&error))?)?;
            return Err(exception);
        }
    };
    let result = PyDict::new(py);
    result.set_item("body", to_python(py, &output.body)?)?;
    result.set_item("diagnostics", to_python(py, &output.diagnostics)?)?;
    Ok(result.unbind())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn details_preserve_json_and_policy_failures() {
        let invalid_json = serde_json::from_str::<Value>("{").unwrap_err();
        let expected = json!({
            "message": invalid_json.to_string(),
            "line": invalid_json.line(),
            "column": invalid_json.column(),
        });
        assert_eq!(
            error_details(&CoreTranslationError::InvalidJson(invalid_json)),
            expected
        );
        for error in [
            CoreTranslationError::LossyConversion("cannot preserve data".into()),
            CoreTranslationError::Other("cannot preserve data".into()),
        ] {
            assert_eq!(
                error_details(&error),
                json!({"message": "cannot preserve data"})
            );
        }
        assert_eq!(
            error_details(&CoreTranslationError::UnknownField {
                path: "$.extra".into()
            }),
            json!({"path": "$.extra"})
        );
    }
}
