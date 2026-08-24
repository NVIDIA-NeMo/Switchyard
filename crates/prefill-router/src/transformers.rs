// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embedded Python implementation of the Transformers prefill forward path.

use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::error::python_error;
use crate::{ForwardOutput, ForwardRequest, LayerSelection, PrefillForward, Result};

/// Configuration for the embedded Hugging Face Transformers implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransformersForwardConfig {
    /// Hugging Face model identifier or local model path.
    pub model: String,
    /// Torch device override. Auto-detected when omitted.
    pub device: Option<String>,
    /// Optional Hugging Face cache directory.
    pub cache_dir: Option<PathBuf>,
}

impl TransformersForwardConfig {
    /// Creates a configuration using automatic device and cache selection.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            device: None,
            cache_dir: None,
        }
    }
}

/// Embedded Python Transformers implementation of [`PrefillForward`].
pub struct TransformersForward {
    extractor: Py<PyAny>,
    model: String,
    loaded: bool,
}

impl TransformersForward {
    /// Creates a lazily loaded Transformers forward implementation.
    ///
    /// `torch` and `transformers` are imported when the first forward runs.
    pub fn new(config: TransformersForwardConfig) -> Result<Self> {
        Python::attach(|py| {
            add_venv_site_packages(py)
                .map_err(|error| python_error("virtual environment setup", error))?;
            let module = PyModule::from_code(
                py,
                c_str!(include_str!("../python/transformers_forward.py")),
                c"transformers_forward.py",
                c"_switchyard_prefill_transformers",
            )
            .map_err(|error| python_error("module initialization", error))?;
            let class = module
                .getattr("TransformersForward")
                .map_err(|error| python_error("class lookup", error))?;
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("device", config.device)
                .map_err(|error| python_error("configuration", error))?;
            kwargs
                .set_item(
                    "cache_dir",
                    config.cache_dir.map(|path| path.into_os_string()),
                )
                .map_err(|error| python_error("configuration", error))?;
            let extractor = class
                .call((config.model.as_str(),), Some(&kwargs))
                .map_err(|error| python_error("construction", error))?;
            Ok(Self {
                extractor: extractor.unbind(),
                model: config.model,
                loaded: false,
            })
        })
    }
}

// Embedded Python does not discover an activated virtual environment because
// the host executable is Rust. Add its standard site-packages directory before
// importing optional Python dependencies.
fn add_venv_site_packages(py: Python<'_>) -> PyResult<()> {
    let Some(venv) = env::var_os("VIRTUAL_ENV") else {
        return Ok(());
    };

    let site_packages = if cfg!(windows) {
        PathBuf::from(venv).join("Lib").join("site-packages")
    } else {
        let version = PyModule::import(py, "sys")?.getattr("version_info")?;
        let major = version.getattr("major")?.extract::<u8>()?;
        let minor = version.getattr("minor")?.extract::<u8>()?;
        PathBuf::from(venv)
            .join("lib")
            .join(format!("python{major}.{minor}"))
            .join("site-packages")
    };

    PyModule::import(py, "site")?.call_method1("addsitedir", (site_packages,))?;
    Ok(())
}

impl PrefillForward for TransformersForward {
    fn forward(&mut self, request: &ForwardRequest) -> Result<ForwardOutput> {
        Python::attach(|py| {
            if !self.loaded {
                let device = self
                    .extractor
                    .bind(py)
                    .call_method0("_ensure_loaded")
                    .and_then(|value| value.extract::<String>())
                    .map_err(|error| python_error("model loading", error))?;
                tracing::info!(model = %self.model, %device, "prefill model loaded");
                self.loaded = true;
            }

            let kwargs = PyDict::new(py);
            let template_json = serde_json::to_string(&request.chat_template_kwargs)
                .map_err(|error| crate::PrefillRouterError::InvalidResult(error.to_string()))?;
            let template_kwargs = PyModule::import(py, "json")
                .and_then(|json| json.call_method1("loads", (template_json,)))
                .map_err(|error| python_error("chat template conversion", error))?;

            kwargs
                .set_item("chat_template_kwargs", template_kwargs)
                .map_err(|error| python_error("forward configuration", error))?;
            match &request.layers {
                LayerSelection::UpperHalf => kwargs
                    .set_item("extract_layers", "upper_half")
                    .map_err(|error| python_error("forward configuration", error))?,
                LayerSelection::All => kwargs
                    .set_item("extract_layers", "all")
                    .map_err(|error| python_error("forward configuration", error))?,
                LayerSelection::Selected(layers) => kwargs
                    .set_item("extract_layers", layers)
                    .map_err(|error| python_error("forward configuration", error))?,
            }
            let pooling = request
                .pooling
                .iter()
                .map(|mode| mode.as_str())
                .collect::<Vec<_>>();
            kwargs
                .set_item("pooling_modes", pooling)
                .and_then(|()| kwargs.set_item("batch_size", request.batch_size))
                .and_then(|()| kwargs.set_item("max_length", request.max_length))
                .map_err(|error| python_error("forward configuration", error))?;

            let result = self
                .extractor
                .bind(py)
                .call_method("extract_batch", (&request.prompts,), Some(&kwargs))
                .map_err(|error| python_error("forward", error))?;
            extract_output(&result, request)
        })
    }

    fn unload(&mut self) -> Result<()> {
        Python::attach(|py| {
            self.extractor
                .bind(py)
                .call_method0("unload")
                .map_err(|error| python_error("unload", error))?;
            self.loaded = false;
            Ok(())
        })
    }
}

fn extract_output(result: &Bound<'_, PyAny>, request: &ForwardRequest) -> Result<ForwardOutput> {
    let item = |name| {
        result
            .get_item(name)
            .map_err(|error| python_error("result decoding", error))
    };
    ForwardOutput::parse(
        request,
        item("hidden_last")?
            .extract::<BTreeMap<usize, Vec<Vec<f32>>>>()
            .map_err(|error| python_error("result decoding", error))?,
        item("hidden_mean")?
            .extract::<BTreeMap<usize, Vec<Vec<f32>>>>()
            .map_err(|error| python_error("result decoding", error))?,
        item("n_layers")?
            .extract::<usize>()
            .map_err(|error| python_error("result decoding", error))?,
        item("hidden_dim")?
            .extract::<usize>()
            .map_err(|error| python_error("result decoding", error))?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LayerSelection, Pooling};
    use pyo3::exceptions::{PyRuntimeError, PyTypeError};

    const REFERENCE_MODEL: &str = "Qwen/Qwen3.5-0.8B";

    #[test]
    #[ignore = "downloads and runs the Qwen3.5-0.8B reference model"]
    fn matches_the_reference_transformers_tensors_exactly() -> Result<()> {
        let config = TransformersForwardConfig {
            model: REFERENCE_MODEL.to_string(),
            device: Some("cpu".to_string()),
            cache_dir: None,
        };
        let mut backend = TransformersForward::new(config)?;
        let mut request = ForwardRequest::new(vec![
            "Explain routing clearly".into(),
            "Write a short proof".into(),
        ]);
        request
            .chat_template_kwargs
            .insert("enable_thinking".into(), true.into());
        request.layers = LayerSelection::All;
        request.pooling = vec![Pooling::Last, Pooling::Mean];
        request.batch_size = 2;
        request.max_length = 32;

        let actual = backend.forward(&request)?;
        backend.unload()?;
        let expected = Python::attach(|py| {
            add_venv_site_packages(py)?;
            let reference = PyModule::import(py, "model_router_toolkit.prefill.extract")?;
            let constructor_kwargs = PyDict::new(py);
            constructor_kwargs.set_item("device", "cpu")?;
            let extractor = reference
                .getattr("PrefillExtractor")?
                .call((REFERENCE_MODEL,), Some(&constructor_kwargs))?;

            let template_kwargs = PyDict::new(py);
            template_kwargs.set_item("enable_thinking", true)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("chat_template_kwargs", template_kwargs)?;
            kwargs.set_item("extract_layers", "all")?;
            kwargs.set_item("pooling_modes", vec!["last", "mean"])?;
            kwargs.set_item("batch_size", request.batch_size)?;
            kwargs.set_item("max_length", request.max_length)?;
            kwargs.set_item("show_progress", false)?;
            let result =
                extractor.call_method("extract_batch", (&request.prompts,), Some(&kwargs))?;

            ForwardOutput::parse(
                &request,
                extract_tensor_map(&result.getattr("hidden_last")?)?,
                extract_tensor_map(&result.getattr("hidden_mean")?)?,
                result.getattr("n_layers")?.extract()?,
                result.getattr("hidden_dim")?.extract()?,
            )
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
        .map_err(|error| python_error("reference forward", error))?;

        assert_eq!(actual, expected);
        Ok(())
    }

    fn extract_tensor_map(value: &Bound<'_, PyAny>) -> PyResult<BTreeMap<usize, Vec<Vec<f32>>>> {
        let tensors = value
            .cast::<PyDict>()
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        tensors
            .iter()
            .map(|(layer, tensor)| {
                Ok((layer.extract()?, tensor.call_method0("tolist")?.extract()?))
            })
            .collect()
    }
}
