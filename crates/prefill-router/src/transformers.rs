// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embedded Transformers and checkpoint implementation of the complete forward pass.

use std::env;
use std::mem::size_of;
use std::path::PathBuf;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::error::python_error;
use crate::{PrefillForward, PrefillRouterError, Result};

/// Configuration for embedded Transformers prefill inference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransformersForwardConfig {
    /// Tensor-only router checkpoint path.
    pub checkpoint: PathBuf,
    /// Torch device override. Auto-detected when omitted.
    pub device: Option<String>,
    /// Optional Hugging Face cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Maximum tokenized prompt length.
    pub max_length: usize,
    /// Maximum prompts per encoder forward pass.
    pub batch_size: usize,
}

impl TransformersForwardConfig {
    /// Creates a configuration using automatic device and cache selection.
    pub fn new(checkpoint: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint: checkpoint.into(),
            device: None,
            cache_dir: None,
            max_length: 2_048,
            batch_size: 32,
        }
    }
}

/// Embedded Python implementation of [`PrefillForward`].
pub struct TransformersForward {
    inner: Py<PyAny>,
    model: String,
    output_count: usize,
    max_length: usize,
    batch_size: usize,
    loaded: bool,
}

impl TransformersForward {
    /// Loads checkpoint metadata and defers loading the encoder until first use.
    pub fn new(config: TransformersForwardConfig) -> Result<Self> {
        if config.max_length == 0 || config.batch_size == 0 {
            return Err(PrefillRouterError::InvalidConfiguration(
                "max_length and batch_size must be greater than zero".to_string(),
            ));
        }
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
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("device", config.device)
                .and_then(|()| {
                    kwargs.set_item(
                        "cache_dir",
                        config.cache_dir.map(|path| path.into_os_string()),
                    )
                })
                .map_err(|error| python_error("configuration", error))?;
            let checkpoint = config.checkpoint.into_os_string();
            let inner = module
                .getattr("TransformersForward")
                .and_then(|class| class.call((checkpoint,), Some(&kwargs)))
                .map_err(|error| python_error("construction", error))?;
            let (model, output_count) = inner
                .call_method0("metadata")
                .and_then(|value| value.extract())
                .map_err(|error| python_error("checkpoint metadata", error))?;
            Ok(Self {
                inner: inner.unbind(),
                model,
                output_count,
                max_length: config.max_length,
                batch_size: config.batch_size,
                loaded: false,
            })
        })
    }
}

impl PrefillForward for TransformersForward {
    fn output_count(&self) -> usize {
        self.output_count
    }

    fn forward(&mut self, prompts: &[String]) -> Result<Vec<Vec<f32>>> {
        if prompts.is_empty() || prompts.iter().any(String::is_empty) {
            return Err(PrefillRouterError::InvalidRequest(
                "prompts must be non-empty".to_string(),
            ));
        }
        Python::attach(|py| {
            let bytes = self
                .inner
                .bind(py)
                .call_method1("forward", (prompts, self.batch_size, self.max_length))
                .and_then(|value| value.extract::<Vec<u8>>())
                .map_err(|error| python_error("forward", error))?;
            if !self.loaded {
                tracing::info!(model = %self.model, "prefill model loaded");
                self.loaded = true;
            }
            decode_probabilities(&bytes, prompts.len(), self.output_count)
        })
    }

    fn unload(&mut self) -> Result<()> {
        Python::attach(|py| {
            self.inner
                .bind(py)
                .call_method0("unload")
                .map_err(|error| python_error("unload", error))?;
            self.loaded = false;
            Ok(())
        })
    }
}

// Embedded Python does not discover an activated virtual environment because
// the host executable is Rust. Add its site-packages before importing dependencies.
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

fn decode_probabilities(
    bytes: &[u8],
    prompt_count: usize,
    output_count: usize,
) -> Result<Vec<Vec<f32>>> {
    let expected = prompt_count
        .checked_mul(output_count)
        .and_then(|values| values.checked_mul(size_of::<f32>()))
        .ok_or_else(|| PrefillRouterError::InvalidResult("output is too large".to_string()))?;
    if bytes.len() != expected {
        return Err(PrefillRouterError::InvalidResult(format!(
            "forward returned {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    let values = bytes
        .chunks_exact(size_of::<f32>())
        .map(|bytes| {
            let value = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if value.is_finite() && (0.0..=1.0).contains(&value) {
                Ok(value)
            } else {
                Err(PrefillRouterError::InvalidResult(
                    "forward returned an invalid probability".to_string(),
                ))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(values
        .chunks_exact(output_count)
        .map(<[f32]>::to_vec)
        .collect())
}

#[cfg(test)]
fn decode_expected_probabilities(
    bytes: &[u8],
    prompt_count: usize,
    output_count: usize,
) -> Result<Vec<Vec<f32>>> {
    let expected = prompt_count
        .checked_mul(size_of::<f32>())
        .and_then(|row_bytes| row_bytes.checked_mul(output_count))
        .ok_or_else(|| PrefillRouterError::InvalidResult("expected output is too large".into()))?;
    if bytes.len() != expected {
        return Err(PrefillRouterError::InvalidResult(
            "expected probabilities have an invalid shape".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect::<Vec<_>>()
        .chunks_exact(output_count)
        .map(<[f32]>::to_vec)
        .collect())
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::Path;

    use super::*;

    #[test]
    #[ignore = "requires PREFILL_ROUTER_HANDOFF_DIR"]
    fn matches_handoff_from_raw_prompts_end_to_end() -> Result<()> {
        let directory = env::var_os("PREFILL_ROUTER_HANDOFF_DIR").ok_or_else(|| {
            PrefillRouterError::InvalidConfiguration(
                "PREFILL_ROUTER_HANDOFF_DIR must name the handoff directory".to_string(),
            )
        })?;
        let directory = Path::new(&directory);
        let config = TransformersForwardConfig::new(
            directory.join("selected_dual_oof_router_checkpoint.pt"),
        );
        let mut forward = TransformersForward::new(config)?;
        let test_data = directory.join("fold_0_test_data.npz").into_os_string();
        let (prompts, expected) = Python::attach(|py| {
            let data = PyModule::import(py, "numpy")
                .and_then(|numpy| numpy.call_method1("load", (test_data,)))
                .map_err(|error| python_error("handoff data loading", error))?;
            let prompts = data
                .get_item("prompt")
                .and_then(|value| value.call_method0("tolist"))
                .and_then(|value| value.extract::<Vec<String>>())
                .map_err(|error| python_error("handoff prompt decoding", error))?;
            let probabilities = data
                .get_item("expected_probabilities")
                .map_err(|error| python_error("handoff probability decoding", error))?;
            let (_, output_count) = probabilities
                .getattr("shape")
                .and_then(|shape| shape.extract::<(usize, usize)>())
                .map_err(|error| python_error("handoff probability decoding", error))?;
            let bytes = probabilities
                .call_method0("tobytes")
                .and_then(|value| value.extract::<Vec<u8>>())
                .map_err(|error| python_error("handoff probability decoding", error))?;
            let expected = decode_expected_probabilities(&bytes, prompts.len(), output_count)?;
            Ok((prompts, expected))
        })?;
        let actual = forward.forward(&prompts)?;
        assert_eq!(actual.len(), expected.len());
        // BF16 kernels can perturb confidences; every resulting route must remain exact.
        for (actual, expected) in actual.iter().zip(&expected) {
            let actual_decision = actual
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index);
            let expected_decision = expected
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index);
            assert_eq!(actual_decision, expected_decision);
        }
        let max_delta = actual
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_delta <= 0.04, "probability max delta: {max_delta}");
        Ok(())
    }
}
