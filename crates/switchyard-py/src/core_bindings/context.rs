// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Rust-owned request context values.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use pyo3::class::basic::CompareOp;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};
use std::time::Duration;
use switchyard_components::{BackendSelection, BackendSelectionReason, StatsBackendLatency};
use switchyard_core::{EvictedTargets, LlmTargetId, ModelId, ProxyContext, RequestId};

use crate::component_bindings::intake::request_metadata_from_mapping;
use crate::py_serde::{value_from_python, value_to_python};

use super::request::{request_type_from_python, request_type_object};

/// Python-owned metadata values kept beside the Rust proxy context.
#[derive(Default)]
struct MetadataStore {
    /// Arbitrary metadata values keyed by Python string.
    values: BTreeMap<String, Py<PyAny>>,
}

/// Cloneable handle to the shared Python metadata store.
#[derive(Clone)]
pub(crate) struct PyProxyMetadataStore {
    /// Shared metadata storage used by context clones.
    inner: Arc<Mutex<MetadataStore>>,
}

/// Dict-like Python object used for `ProxyContext.metadata`.
#[pyclass(name = "ProxyMetadata")]
struct PyProxyMetadata {
    /// Shared metadata storage protected across Python/Rust calls.
    inner: Arc<Mutex<MetadataStore>>,
}

impl PyProxyMetadata {
    /// Creates an empty metadata object.
    fn new() -> Self {
        Self::from_store(PyProxyMetadataStore::default())
    }

    /// Creates a metadata object from an existing shared store.
    fn from_store(store: PyProxyMetadataStore) -> Self {
        Self { inner: store.inner }
    }

    /// Returns the cloneable store handle for sharing with `ProxyContext`.
    fn metadata_store(&self) -> PyProxyMetadataStore {
        PyProxyMetadataStore {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Builds metadata from any Python mapping accepted by `dict.update`.
    fn from_mapping(py: Python<'_>, metadata: &Bound<'_, PyAny>) -> PyResult<Self> {
        let store = Self::new();
        store.update_from_mapping(py, metadata)?;
        Ok(store)
    }

    /// Locks the metadata store with a Python-facing poisoned-lock error.
    fn lock(&self) -> PyResult<MutexGuard<'_, MetadataStore>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("ProxyMetadata lock is poisoned"))
    }

    /// Merges entries from a Python mapping into the metadata store.
    fn update_from_mapping(&self, py: Python<'_>, metadata: &Bound<'_, PyAny>) -> PyResult<()> {
        let entries = metadata_entries(py, metadata)?;
        let mut store = self.lock()?;
        store.values.extend(entries);
        Ok(())
    }

    /// Materializes the metadata store as a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let entries = {
            let store = self.lock()?;
            store
                .values
                .iter()
                .map(|(key, value)| (key.clone(), value.clone_ref(py)))
                .collect::<Vec<_>>()
        };
        let dict = PyDict::new(py);
        for (key, value) in entries {
            dict.set_item(key, value)?;
        }
        Ok(dict.unbind())
    }
}

impl Default for PyProxyMetadataStore {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MetadataStore::default())),
        }
    }
}

#[pymethods]
impl PyProxyMetadata {
    /// Creates metadata from an optional mapping.
    #[new]
    #[pyo3(signature = (metadata=None))]
    fn py_new(py: Python<'_>, metadata: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let store = Self::new();
        if let Some(metadata) = metadata {
            if !metadata.is_none() {
                store.update_from_mapping(py, metadata)?;
            }
        }
        Ok(store)
    }

    /// Implements `dict.get`.
    #[pyo3(signature = (key, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &str,
        default: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let value = self
            .lock()?
            .values
            .get(key)
            .map(|value| value.clone_ref(py));
        match value {
            Some(value) => Ok(value),
            None => Ok(default
                .map(|value| value.clone().unbind())
                .unwrap_or_else(|| py.None())),
        }
    }

    /// Implements `dict.setdefault`.
    #[pyo3(signature = (key, default=None))]
    fn setdefault(
        &self,
        py: Python<'_>,
        key: String,
        default: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let default = default
            .map(|value| value.clone().unbind())
            .unwrap_or_else(|| py.None());
        let mut store = self.lock()?;
        if let Some(value) = store.values.get(&key) {
            return Ok(value.clone_ref(py));
        }
        store.values.insert(key, default.clone_ref(py));
        Ok(default)
    }

    /// Implements `dict.update`.
    fn update(&self, py: Python<'_>, metadata: &Bound<'_, PyAny>) -> PyResult<()> {
        self.update_from_mapping(py, metadata)
    }

    /// Returns a shallow Python dict copy.
    fn copy(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.to_dict(py)
    }

    /// Returns a Python keys view.
    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_dict(py)?
            .bind(py)
            .call_method0("keys")
            .map(Bound::unbind)
    }

    /// Returns a Python values view.
    fn values(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_dict(py)?
            .bind(py)
            .call_method0("values")
            .map(Bound::unbind)
    }

    /// Returns a Python items view.
    fn items(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_dict(py)?
            .bind(py)
            .call_method0("items")
            .map(Bound::unbind)
    }

    /// Implements metadata indexing.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.lock()?
            .values
            .get(key)
            .map(|value| value.clone_ref(py))
            .ok_or_else(|| PyKeyError::new_err(key.to_string()))
    }

    /// Implements metadata assignment.
    fn __setitem__(&self, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.lock()?.values.insert(key, value.clone().unbind());
        Ok(())
    }

    /// Implements metadata deletion.
    fn __delitem__(&self, key: &str) -> PyResult<()> {
        self.lock()?
            .values
            .remove(key)
            .map(|_| ())
            .ok_or_else(|| PyKeyError::new_err(key.to_string()))
    }

    /// Implements `key in metadata`.
    fn __contains__(&self, key: &str) -> PyResult<bool> {
        Ok(self.lock()?.values.contains_key(key))
    }

    /// Implements `len(metadata)`.
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.lock()?.values.len())
    }

    /// Iterates over metadata keys like a Python dict.
    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.keys(py)?
            .bind(py)
            .call_method0("__iter__")
            .map(Bound::unbind)
    }

    /// Compares metadata by Python dict equality semantics.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        match op {
            CompareOp::Eq | CompareOp::Ne => {
                let equals = self
                    .to_dict(py)?
                    .bind(py)
                    .rich_compare(other, CompareOp::Eq)?
                    .extract::<bool>()?;
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

    /// Returns the Python dict representation.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.to_dict(py)?.bind(py).repr()?.extract()
    }
}

/// Extracts string-keyed entries from any mapping accepted by `dict.update`.
fn metadata_entries(
    py: Python<'_>,
    metadata: &Bound<'_, PyAny>,
) -> PyResult<Vec<(String, Py<PyAny>)>> {
    let dict = PyDict::new(py);
    dict.call_method1("update", (metadata,))?;
    dict.iter()
        .map(|(key, value)| Ok((key.extract::<String>()?, value.clone().unbind())))
        .collect()
}

/// Python-facing proxy context backed by the Rust `ProxyContext`.
#[pyclass(name = "ProxyContext", skip_from_py_object)]
pub(crate) struct PyProxyContext {
    /// Shared Rust context guarded across Python and async Rust calls.
    inner: Arc<Mutex<ProxyContext>>,
    /// Borrow flag that prevents Python mutation while Rust owns the context.
    in_use: Arc<AtomicBool>,
    /// Dict-like compatibility metadata store.
    metadata: Py<PyProxyMetadata>,
}

impl PyProxyContext {
    /// Locks the context unless an async Rust component currently owns it.
    fn lock(&self) -> PyResult<MutexGuard<'_, ProxyContext>> {
        if self.in_use.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err(
                "ProxyContext is already borrowed by an async Rust component",
            ));
        }
        self.inner
            .lock()
            .map_err(|_| PyValueError::new_err("ProxyContext lock is poisoned"))
    }

    /// Temporarily moves the Rust context out for async Rust component execution.
    pub(crate) fn lease(&self) -> PyResult<PyProxyContextLease> {
        if self.in_use.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err(
                "ProxyContext is already borrowed by an async Rust component",
            ));
        }
        let context = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| PyValueError::new_err("ProxyContext lock is poisoned"))?;
            std::mem::take(&mut *guard)
        };
        Ok(PyProxyContextLease {
            inner: Arc::clone(&self.inner),
            in_use: Arc::clone(&self.in_use),
            context: Some(context),
        })
    }

    /// Inserts a typed Rust extension into the wrapped context.
    pub(crate) fn insert_value<T>(&self, value: T) -> PyResult<()>
    where
        T: Send + Sync + 'static,
    {
        self.lock()?.insert(value);
        Ok(())
    }

    /// Returns a clone of a typed Rust extension if one is present.
    ///
    /// Companion to [`insert_value`] for Python-facing readers of typed
    /// extensions stamped by Rust processors (e.g. `ContextSignals`).
    pub(crate) fn get_cloned<T>(&self) -> PyResult<Option<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        Ok(self.lock()?.get::<T>().cloned())
    }
}

/// Temporary ownership lease for passing `ProxyContext` into async Rust roles.
pub(crate) struct PyProxyContextLease {
    /// Shared storage that receives the context when the lease is restored.
    inner: Arc<Mutex<ProxyContext>>,
    /// Borrow flag cleared on restore or drop.
    in_use: Arc<AtomicBool>,
    /// Moved-out Rust context.
    context: Option<ProxyContext>,
}

impl PyProxyContextLease {
    /// Returns mutable access to the leased Rust context.
    pub(crate) fn context_mut(&mut self) -> PyResult<&mut ProxyContext> {
        self.context
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("ProxyContext lease has already been restored"))
    }

    /// Restores the leased context into the Python wrapper.
    pub(crate) fn restore(mut self) -> PyResult<()> {
        if let Some(context) = self.context.take() {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| PyValueError::new_err("ProxyContext lock is poisoned"))?;
            *guard = context;
        }
        self.in_use.store(false, Ordering::Release);
        Ok(())
    }
}

impl Drop for PyProxyContextLease {
    /// Restores the context during unwinding and always clears the borrow flag.
    fn drop(&mut self) {
        if let Some(context) = self.context.take() {
            if let Ok(mut guard) = self.inner.lock() {
                *guard = context;
            }
        }
        self.in_use.store(false, Ordering::Release);
    }
}

#[pymethods]
impl PyProxyContext {
    /// Creates a Python proxy context from optional metadata and request ID.
    #[new]
    #[pyo3(signature = (metadata=None, request_id=None))]
    fn new(
        py: Python<'_>,
        metadata: Option<&Bound<'_, PyAny>>,
        request_id: Option<String>,
    ) -> PyResult<Self> {
        let request_metadata = match metadata {
            Some(metadata) if !metadata.is_none() => request_metadata_from_mapping(py, metadata)?,
            _ => None,
        };
        let metadata = match metadata {
            Some(metadata) if !metadata.is_none() => {
                Py::new(py, PyProxyMetadata::from_mapping(py, metadata)?)?
            }
            _ => Py::new(py, PyProxyMetadata::new())?,
        };

        let request_id = request_id
            .map(RequestId::new)
            .transpose()
            .map_err(|error| {
                PyValueError::new_err(format!("invalid request_id for ProxyContext: {error}"))
            })?;

        let mut inner = ProxyContext::default();
        inner.request_id = request_id;
        let metadata_store = metadata.borrow(py).metadata_store();
        inner.insert(metadata_store);
        if let Some(request_metadata) = request_metadata {
            inner.insert(request_metadata);
        }

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            in_use: Arc::new(AtomicBool::new(false)),
            metadata,
        })
    }

    /// Returns the dict-like metadata object.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> Py<PyProxyMetadata> {
        self.metadata.clone_ref(py)
    }

    /// Returns the optional request ID.
    #[getter]
    fn request_id(&self) -> PyResult<Option<String>> {
        Ok(self
            .lock()?
            .request_id
            .as_ref()
            .map(|request_id| request_id.as_str().to_string()))
    }

    /// Updates the optional request ID.
    #[setter]
    fn set_request_id(&self, value: Option<String>) -> PyResult<()> {
        self.lock()?.request_id = value.map(RequestId::new).transpose().map_err(|error| {
            PyValueError::new_err(format!("invalid request_id for ProxyContext: {error}"))
        })?;
        Ok(())
    }

    /// Appends one algorithm-authored routing event.
    fn record_routing_event(
        &self,
        py: Python<'_>,
        name: String,
        payload: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let payload = value_from_python(payload)?;
        let value = {
            let mut context = self.lock()?;
            let recorded = context
                .record_routing_event(name, payload)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            serde_json::to_value(recorded)
                .map_err(|error| PyValueError::new_err(error.to_string()))?
        };
        value_to_python(py, &value)
    }

    /// Returns the routing audit recorded on this context.
    #[getter]
    fn routing_trace(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let value = self
            .lock()?
            .routing_trace()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        value.map(|value| value_to_python(py, &value)).transpose()
    }

    /// Returns the inbound request format as a Python enum object.
    #[getter]
    fn inbound_format(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.lock()?
            .inbound_format
            .map(|request_type| request_type_object(py, request_type))
            .transpose()
    }

    /// Updates the inbound request format from a Python enum object.
    #[setter]
    fn set_inbound_format(&self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.lock()?.inbound_format = value
            .filter(|value| !value.is_none())
            .map(request_type_from_python)
            .transpose()?;
        Ok(())
    }

    /// Returns the selected served model, if backend selection exists.
    #[getter]
    fn selected_model(&self) -> PyResult<Option<String>> {
        Ok(self
            .lock()?
            .get::<BackendSelection>()
            .map(|selection| selection.model.as_str().to_string()))
    }

    /// Updates the selected served model using backend-selection metadata.
    #[setter]
    fn set_selected_model(&self, value: Option<String>) -> PyResult<()> {
        let mut inner = self.lock()?;
        match value {
            Some(value) => {
                let model = ModelId::new(value).map_err(|error| {
                    PyValueError::new_err(format!(
                        "invalid selected_model for ProxyContext: {error}"
                    ))
                })?;
                inner.insert(BackendSelection::for_model(
                    model,
                    None,
                    BackendSelectionReason::PassthroughModel,
                ));
            }
            None => {
                inner.remove::<BackendSelection>();
            }
        }
        Ok(())
    }

    /// Returns the selected target ID, if any.
    #[getter]
    fn selected_target(&self) -> PyResult<Option<String>> {
        Ok(self
            .lock()?
            .selected_target()
            .map(|target| target.as_str().to_string()))
    }

    /// Returns the set of target IDs evicted from the routing pool after a
    /// context-window overflow on the current request.
    /// `None` when no evictions have happened yet.
    #[getter]
    fn evicted_targets(&self) -> PyResult<Option<Vec<String>>> {
        Ok(self.lock()?.get::<EvictedTargets>().map(|evicted| {
            let mut ids: Vec<String> = evicted.iter().map(|id| id.as_str().to_string()).collect();
            ids.sort();
            ids
        }))
    }

    /// Replaces the evicted target set used by the Python compatibility chain.
    #[setter]
    fn set_evicted_targets(&self, value: Option<Vec<String>>) -> PyResult<()> {
        let mut inner = self.lock()?;
        match value {
            Some(values) => {
                let mut evicted = EvictedTargets::default();
                for target in values {
                    evicted.insert(LlmTargetId::new(target).map_err(|error| {
                        PyValueError::new_err(format!(
                            "invalid evicted target for ProxyContext: {error}"
                        ))
                    })?);
                }
                inner.insert(evicted);
            }
            None => {
                inner.remove::<EvictedTargets>();
            }
        }
        Ok(())
    }

    /// Returns the measured backend-call latency in ms, if any backend recorded one.
    #[getter]
    fn backend_call_latency_ms(&self) -> PyResult<Option<f64>> {
        Ok(self
            .lock()?
            .get::<StatsBackendLatency>()
            .map(|latency| latency.as_millis_f64()))
    }

    /// Records the measured backend-call latency in ms.
    ///
    /// The Rust ``StatsLlmBackend`` wraps native backends and writes this
    /// slot automatically. Python-only backends (e.g. ``LatencyServiceLLMBackend``)
    /// that can't be wrapped record their measurement here so the
    /// downstream ``StatsResponseProcessor`` can compute
    /// ``routing_overhead_ms = total_latency - backend_latency`` and emit
    /// it on ``/metrics``. Setting ``None`` clears the slot.
    #[setter]
    fn set_backend_call_latency_ms(&self, value: Option<f64>) -> PyResult<()> {
        let mut inner = self.lock()?;
        match value {
            Some(ms) => {
                if !ms.is_finite() || ms < 0.0 {
                    return Err(PyValueError::new_err(
                        "backend_call_latency_ms must be a finite, non-negative number",
                    ));
                }
                inner.insert(StatsBackendLatency(Duration::from_secs_f64(ms / 1000.0)));
            }
            None => {
                inner.remove::<StatsBackendLatency>();
            }
        }
        Ok(())
    }

    /// Updates the selected target ID.
    #[setter]
    fn set_selected_target(&self, value: Option<String>) -> PyResult<()> {
        let mut inner = self.lock()?;
        match value {
            Some(value) => {
                inner.set_selected_target(LlmTargetId::new(value).map_err(|error| {
                    PyValueError::new_err(format!(
                        "invalid selected_target for ProxyContext: {error}"
                    ))
                })?);
            }
            None => {
                inner.clear_selected_target();
            }
        }
        Ok(())
    }

    /// Returns a compact debug representation for Python users.
    fn __repr__(&self) -> PyResult<String> {
        let inner = self.lock()?;
        let selected_model = inner
            .get::<BackendSelection>()
            .map(|selection| selection.model.as_str().to_string());
        let selected_target = inner
            .selected_target()
            .map(|target| target.as_str().to_string());
        Ok(format!(
            "ProxyContext(request_id={:?}, inbound_format={:?}, selected_model={:?}, selected_target={:?})",
            inner.request_id,
            inner.inbound_format,
            selected_model,
            selected_target,
        ))
    }
}

/// Registers proxy context bindings into the Python module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyProxyMetadata>()?;
    module.add_class::<PyProxyContext>()?;
    Ok(())
}
