// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the components-v2 profile API.
//!
//! Two surfaces are exposed:
//!
//! - The erased serving surface — [`PyProfileConfigDocument`], [`PyProfileConfigPlan`],
//!   and [`PyProfile`] — loads config files and runs any profile through the
//!   object-safe `Profile` contract. It is the right default for servers,
//!   launchers, and generic profile tables that only need `run()`.
//! - The typed request-input surface (see [`typed`]) exposes `ProfileInput` and
//!   `ProfileRequestMetadata` so Python-authored profiles can share the same
//!   request envelope as Rust-authored profiles.
//!
//! Concrete profile authoring now belongs to `switchyard.lib.profiles` on the
//! Python side or `switchyard-components-v2` on the Rust side; this module only
//! bridges config-file execution and the shared input value.

mod typed;

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use switchyard_components_v2::{
    parse_profile_config_path as parse_v2_profile_config_path,
    parse_profile_config_str as parse_v2_profile_config_str, Profile, ProfileConfigDocument,
    ProfileConfigFormat, ProfileConfigPlan, ProfileInput,
};
use switchyard_core::{LlmTargetId, ProfileId, SwitchyardError};

use crate::component_bindings::config::PyLlmTarget;
use crate::core_bindings::request::PyChatRequest;
use crate::core_bindings::response::PyChatResponse;
use crate::errors::py_core_error;
use crate::py_serde::value_to_python;

use self::typed::PyProfileRequestMetadata;

/// Parsed components-v2 profile config document.
#[pyclass(name = "ProfileConfigDocument", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
struct PyProfileConfigDocument {
    inner: ProfileConfigDocument,
}

#[pymethods]
impl PyProfileConfigDocument {
    /// Return parsed profile IDs in deterministic order.
    fn profile_ids(&self) -> Vec<String> {
        self.inner
            .profile_ids()
            .map(|profile_id| profile_id.as_str().to_string())
            .collect()
    }

    /// Return a parsed profile's type discriminator, or `None`.
    fn profile_type(&self, profile_id: &str) -> PyResult<Option<String>> {
        let profile_id = parse_profile_id(profile_id)?;
        Ok(self
            .inner
            .profile_type(&profile_id)
            .map(std::borrow::ToOwned::to_owned))
    }

    /// Return a parsed profile's envelope `subagent_target` reference, or `None`.
    fn profile_subagent_target(&self, profile_id: &str) -> PyResult<Option<String>> {
        let profile_id = parse_profile_id(profile_id)?;
        Ok(self
            .inner
            .profile_subagent_target(&profile_id)
            .map(|target_id| target_id.as_str().to_owned()))
    }

    /// Return a parsed profile's body without the `type` discriminator.
    fn profile_body(&self, py: Python<'_>, profile_id: &str) -> PyResult<Option<Py<PyAny>>> {
        let profile_id = parse_profile_id(profile_id)?;
        self.inner
            .profile_body(&profile_id)
            .map(|body| value_to_python(py, body))
            .transpose()
    }

    /// Return a copy with selected profiles removed before resolution.
    fn without_profiles(&self, profile_ids: Vec<String>) -> PyResult<Self> {
        let profile_ids = profile_ids
            .iter()
            .map(|profile_id| parse_profile_id(profile_id))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: self.inner.without_profiles(&profile_ids),
        })
    }

    /// Resolve endpoint/target references and validate profile-owned configs.
    fn resolve(&self) -> PyResult<PyProfileConfigPlan> {
        self.inner
            .resolve()
            .map(PyProfileConfigPlan::from_core)
            .map_err(py_core_error)
    }

    fn __repr__(&self) -> &'static str {
        "ProfileConfigDocument()"
    }
}

/// Resolved components-v2 profile config plan.
#[pyclass(name = "ProfileConfigPlan", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyProfileConfigPlan {
    inner: ProfileConfigPlan,
}

impl PyProfileConfigPlan {
    fn from_core(inner: ProfileConfigPlan) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyProfileConfigPlan {
    /// Return resolved profile IDs in deterministic order.
    fn profile_ids(&self) -> Vec<String> {
        self.inner
            .profile_ids()
            .map(|profile_id| profile_id.as_str().to_string())
            .collect()
    }

    /// Return resolved target IDs in deterministic order.
    fn target_ids(&self) -> Vec<String> {
        self.inner
            .targets()
            .map(|(target_id, _target)| target_id.as_str().to_string())
            .collect()
    }

    /// Return the serialized profile type for one profile ID, or `None`.
    fn profile_type(&self, profile_id: &str) -> PyResult<Option<String>> {
        let profile_id = parse_profile_id(profile_id)?;
        Ok(self
            .inner
            .profile_type(&profile_id)
            .map(std::borrow::ToOwned::to_owned))
    }

    /// Return a resolved target by ID, or `None`.
    fn target(&self, target_id: &str) -> PyResult<Option<PyLlmTarget>> {
        let target_id = parse_target_id(target_id)?;
        Ok(self
            .inner
            .target(&target_id)
            .cloned()
            .map(PyLlmTarget::from_core))
    }

    /// Build one profile runtime by profile ID.
    fn build_profile(&self, profile_id: &str) -> PyResult<PyProfile> {
        let profile_id = parse_profile_id(profile_id)?;
        let profile = self
            .inner
            .build_profile(&profile_id)
            .map_err(py_core_error)?;
        Ok(PyProfile::from_boxed(
            profile_id.as_str().to_string(),
            profile,
        ))
    }

    /// Build all profile runtimes in deterministic profile-ID order.
    fn build_profiles<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let profiles = self.inner.build_profiles().map_err(py_core_error)?;
        let out = PyDict::new(py);
        for (profile_id, profile) in profiles {
            out.set_item(
                profile_id.as_str(),
                Py::new(
                    py,
                    PyProfile::from_boxed(profile_id.as_str().to_string(), profile),
                )?,
            )?;
        }
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!(
            "ProfileConfigPlan(profiles={}, targets={})",
            self.inner.profile_count(),
            self.inner.target_count(),
        )
    }
}

/// Object-safe components-v2 profile serving runtime.
#[pyclass(name = "Profile", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyProfile {
    profile_id: String,
    inner: Arc<dyn Profile>,
}

impl PyProfile {
    fn from_boxed(profile_id: String, profile: Box<dyn Profile>) -> Self {
        Self {
            profile_id,
            inner: Arc::from(profile),
        }
    }
}

#[pymethods]
impl PyProfile {
    /// Profile ID this runtime was built from.
    #[getter]
    fn profile_id(&self) -> String {
        self.profile_id.clone()
    }

    /// Execute the full profile-owned request lifecycle.
    #[pyo3(signature = (request, metadata=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        request: PyRef<'_, PyChatRequest>,
        metadata: Option<PyRef<'_, PyProfileRequestMetadata>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let profile = Arc::clone(&self.inner);
        let request = request.clone_core();
        let metadata = metadata
            .map(|metadata| metadata.clone_core())
            .unwrap_or_default();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // The erased profile is the metadata-free generic entry point used
            // by config plans and servers. Python-authored profiles can still
            // carry metadata through the shared `ProfileInput` binding.
            let profile_response = profile
                .run(ProfileInput { request, metadata })
                .await
                .map_err(py_core_error)?;
            let (response, _routing_metadata) = profile_response.into_parts();
            Python::attach(|py| {
                Py::new(py, PyChatResponse::from_core(py, response)?)
                    .map(|response| response.into_any())
            })
        })
    }

    fn __repr__(&self) -> String {
        format!("Profile(profile_id={:?})", self.profile_id)
    }
}

/// Parse a components-v2 profile config string.
#[pyfunction(name = "parse_profile_config_str")]
#[pyo3(signature = (input, format = "yaml"))]
fn py_parse_profile_config_str(input: &str, format: &str) -> PyResult<PyProfileConfigDocument> {
    let format = profile_config_format_from_str(format)?;
    parse_v2_profile_config_str(input, format)
        .map(|inner| PyProfileConfigDocument { inner })
        .map_err(py_core_error)
}

/// Parse a components-v2 profile config file, inferring the format from extension.
#[pyfunction(name = "parse_profile_config_path")]
fn py_parse_profile_config_path(path: PathBuf) -> PyResult<PyProfileConfigDocument> {
    parse_v2_profile_config_path(path)
        .map(|inner| PyProfileConfigDocument { inner })
        .map_err(py_core_error)
}

/// Codex sub-agent kinds that carry delegated user work rather than harness
/// maintenance (`compact`, `memory_consolidation`, ...). Unknown kinds fall
/// through to normal routing; extend deliberately, with captured fixtures.
const SUBAGENT_WORK_KINDS: &[&str] = &["collab_spawn", "review"];

/// Report whether normalized request headers mark delegated sub-agent work.
///
/// An explicit `x-switchyard-is-subagent` boolean decides directly. Otherwise
/// detection reuses the canonical harness header normalization
/// (`switchyard_protocol::Metadata::from_headers`) — Claude Code agent
/// lineage and Codex/relay parent or sub-agent markers — and a request whose
/// normalized agent kind names anything other than known delegated-work kinds
/// is not routed as sub-agent work, so Codex `compact` /
/// `memory_consolidation` turns stay on normal profile routing.
#[pyfunction]
fn is_subagent_request(headers: &Bound<'_, PyAny>) -> PyResult<bool> {
    let headers = typed::metadata_headers_from_python(headers)?;
    // Lineage headers are single-valued; keep the first value when repeated.
    let headers: std::collections::BTreeMap<String, String> = headers
        .into_iter()
        .filter_map(|(name, values)| values.into_iter().next().map(|value| (name, value)))
        .collect();
    if let Some(explicit) = headers
        .get("x-switchyard-is-subagent")
        .and_then(|value| parse_bool_header(value))
    {
        return Ok(explicit);
    }
    let metadata = switchyard_protocol::Metadata::from_headers(&headers);
    if !metadata.is_subagent {
        return Ok(false);
    }
    Ok(match metadata.agent_kind.as_deref() {
        None => true,
        Some(kind) => SUBAGENT_WORK_KINDS.contains(&kind),
    })
}

/// Parses the common textual spellings of a boolean header value, mirroring
/// the protocol crate's header normalization.
fn parse_bool_header(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Parse and resolve a components-v2 profile config file.
#[pyfunction]
fn load_profile_config(path: PathBuf) -> PyResult<PyProfileConfigPlan> {
    parse_v2_profile_config_path(path)
        .and_then(|document| document.resolve())
        .map(PyProfileConfigPlan::from_core)
        .map_err(py_core_error)
}

fn profile_config_format_from_str(raw: &str) -> PyResult<ProfileConfigFormat> {
    match raw {
        "json" => Ok(ProfileConfigFormat::Json),
        "toml" => Ok(ProfileConfigFormat::Toml),
        "yaml" | "yml" => Ok(ProfileConfigFormat::Yaml),
        other => Err(PyValueError::new_err(format!(
            "unknown profile config format {other:?}; expected 'yaml', 'json', or 'toml'"
        ))),
    }
}

fn parse_profile_id(value: &str) -> PyResult<ProfileId> {
    ProfileId::new(value).map_err(|error| py_core_error(SwitchyardError::from(error)))
}

fn parse_target_id(value: &str) -> PyResult<LlmTargetId> {
    LlmTargetId::new(value).map_err(|error| py_core_error(SwitchyardError::from(error)))
}

/// Registers profile bindings with the native Python module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyProfileConfigDocument>()?;
    module.add_class::<PyProfileConfigPlan>()?;
    module.add_class::<PyProfile>()?;
    module.add_function(wrap_pyfunction!(py_parse_profile_config_str, module)?)?;
    module.add_function(wrap_pyfunction!(py_parse_profile_config_path, module)?)?;
    module.add_function(wrap_pyfunction!(load_profile_config, module)?)?;
    module.add_function(wrap_pyfunction!(is_subagent_request, module)?)?;
    typed::register(module)?;
    Ok(())
}
