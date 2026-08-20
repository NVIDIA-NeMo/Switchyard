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
                .call((config.model,), Some(&kwargs))
                .map_err(|error| python_error("construction", error))?;
            Ok(Self {
                extractor: extractor.unbind(),
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
                    .set_item("extract_layers", py.None())
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
            let output = extract_output(&result)?;
            output.validate(request.prompts.len())
        })
    }

    fn unload(&mut self) -> Result<()> {
        Python::attach(|py| {
            self.extractor
                .bind(py)
                .call_method0("unload")
                .map_err(|error| python_error("unload", error))?;
            Ok(())
        })
    }
}

fn extract_output(result: &Bound<'_, PyAny>) -> Result<ForwardOutput> {
    let item = |name| {
        result
            .get_item(name)
            .map_err(|error| python_error("result decoding", error))
    };
    Ok(ForwardOutput {
        hidden_last: item("hidden_last")?
            .extract::<BTreeMap<usize, Vec<Vec<f32>>>>()
            .map_err(|error| python_error("result decoding", error))?,
        hidden_mean: item("hidden_mean")?
            .extract::<BTreeMap<usize, Vec<Vec<f32>>>>()
            .map_err(|error| python_error("result decoding", error))?,
        n_layers: item("n_layers")?
            .extract::<usize>()
            .map_err(|error| python_error("result decoding", error))?,
        hidden_dim: item("hidden_dim")?
            .extract::<usize>()
            .map_err(|error| python_error("result decoding", error))?,
    })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::{LayerSelection, Pooling};

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn create() -> Result<Self> {
            let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "switchyard-prefill-parity-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).map_err(|error| {
                crate::PrefillRouterError::InvalidResult(format!(
                    "failed to create test directory {}: {error}",
                    path.display()
                ))
            })?;
            Ok(Self(path))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    #[ignore = "run through the crate-local uv environment"]
    fn matches_the_reference_transformers_tensors_exactly() -> Result<()> {
        let directory = TestDirectory::create()?;
        Python::attach(|py| {
            add_venv_site_packages(py)?;
            PyModule::from_code(
                py,
                c_str!(include_str!("../tests/reference_transformers_forward.py")),
                c"reference_transformers_forward.py",
                c"_switchyard_prefill_reference",
            )?
            .call_method1("create_tiny_model", (directory.path(),))?;
            Ok(())
        })
        .map_err(|error| python_error("test setup", error))?;

        let config = TransformersForwardConfig {
            model: directory.path().to_string_lossy().into_owned(),
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
        request.max_length = 16;

        let actual = backend.forward(&request)?;
        let expected = Python::attach(|py| {
            let reference = PyModule::import(py, "_switchyard_prefill_reference")?;
            let result = reference.call_method1(
                "extract_reference",
                (
                    directory.path(),
                    &request.prompts,
                    vec!["last", "mean"],
                    true,
                ),
            )?;
            extract_output(&result)
                .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
        })
        .map_err(|error| python_error("reference forward", error))?;

        assert_eq!(actual, expected);
        backend.unload()
    }
}
