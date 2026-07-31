// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bindings for `switchyard-translation`.

use pyo3::prelude::*;
use serde_json::Value;
use switchyard_translation::{
    StreamTranslationState, TranslationEngine, TranslationPolicy, normalize_anthropic_tool_use_ids,
};

use crate::errors::py_translation_error;
use crate::py_serde::{value_from_python, value_to_python};

#[pyclass(name = "TranslationEngine")]
struct PyTranslationEngine {
    inner: TranslationEngine,
    policy: TranslationPolicy,
}

#[pymethods]
impl PyTranslationEngine {
    #[new]
    fn new() -> Self {
        Self {
            inner: TranslationEngine::default(),
            policy: TranslationPolicy::default(),
        }
    }

    fn translate_request(
        &self,
        py: Python<'_>,
        source: &str,
        target: &str,
        body: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let body = value_from_python(body)?;
        let output = self
            .inner
            .translate_request(source, target, &body, &self.policy)
            .map_err(py_translation_error)?;
        value_to_python(py, &output.body)
    }

    fn translate_response(
        &self,
        py: Python<'_>,
        source: &str,
        target: &str,
        body: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let body = value_from_python(body)?;
        let output = self
            .inner
            .translate_response(source, target, &body, &self.policy)
            .map_err(py_translation_error)?;
        value_to_python(py, &output.body)
    }

    #[pyo3(signature = (source, target, model=None, message_id=None))]
    fn stream(
        &self,
        source: &str,
        target: &str,
        model: Option<String>,
        message_id: Option<String>,
    ) -> PyStreamTranslation {
        let mut state = StreamTranslationState::new(source, target);
        state.target_model = model;
        state.target_message_id = message_id;
        PyStreamTranslation {
            source: source.to_string(),
            target: target.to_string(),
            state,
            inner: TranslationEngine::default(),
        }
    }

    fn normalize_anthropic_tool_use_ids(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let messages = value_from_python(messages)?;
        value_to_python(py, &normalize_anthropic_tool_use_ids(messages))
    }
}

#[pyclass(name = "StreamTranslation")]
struct PyStreamTranslation {
    source: String,
    target: String,
    state: StreamTranslationState,
    inner: TranslationEngine,
}

#[pymethods]
impl PyStreamTranslation {
    fn translate_event(&mut self, py: Python<'_>, event: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let event = value_from_python(event)?;
        let output = self
            .inner
            .translate_event(&mut self.state, &self.source, &self.target, &event)
            .map_err(py_translation_error)?;
        value_to_python(py, &Value::Array(output))
    }

    fn finish(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let output = self
            .inner
            .finish_stream(&mut self.state, &self.target)
            .map_err(py_translation_error)?;
        value_to_python(py, &Value::Array(output))
    }
}

/// Registers translation bindings with the native Python module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTranslationEngine>()?;
    module.add_class::<PyStreamTranslation>()?;
    Ok(())
}
