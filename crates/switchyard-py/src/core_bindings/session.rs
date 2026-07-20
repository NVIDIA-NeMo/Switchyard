// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Rust-owned session-affinity primitives.

use pyo3::prelude::*;

use crate::py_serde::value_from_python;

/// Derives a stable session key from a request body mapping.
///
/// Accepts any Python mapping (e.g. a request body dict) and returns the key
/// the core derivation produces for it, used to pin requests of one session to
/// a previously selected model. `depth == 0` (default) always yields a key;
/// `depth > 0` extends the hashed prefix with the first `depth` post-first-user
/// messages and returns `None` until that prefix exists.
#[pyfunction]
#[pyo3(signature = (body, depth = 0))]
pub(crate) fn session_key_from_body(
    body: &Bound<'_, PyAny>,
    depth: usize,
) -> PyResult<Option<String>> {
    Ok(switchyard_core::session_key_from_body_with_depth(
        &value_from_python(body)?,
        depth,
    ))
}

/// LRU-bounded cache mapping session keys to arbitrary Python values.
///
/// Stores `Py<PyAny>` so any router can reuse it; the Python consumer keeps
/// `str` model-ids today. `get` refreshes recency, so it requires `&mut self`.
#[pyclass(name = "SessionCache")]
pub(crate) struct PySessionCache {
    /// Core LRU cache holding GIL-bound Python references.
    inner: switchyard_core::SessionCache<Py<PyAny>>,
}

#[pymethods]
impl PySessionCache {
    /// Creates a cache bounded to at most `max_sessions` entries.
    #[new]
    fn new(max_sessions: usize) -> Self {
        Self {
            inner: switchyard_core::SessionCache::new(max_sessions),
        }
    }

    /// Returns the value for `key`, refreshing its recency, or `None`.
    fn get(&mut self, py: Python<'_>, key: &str) -> Option<Py<PyAny>> {
        self.inner.get(key).map(|value| value.clone_ref(py))
    }

    /// Inserts or updates the value for `key`, evicting the oldest if full.
    fn put(&mut self, key: String, value: Py<PyAny>) {
        self.inner.put(key, value);
    }

    /// Returns all cached values (order unspecified).
    fn values(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.inner
            .values()
            .map(|value| value.clone_ref(py))
            .collect()
    }

    /// Returns the configured maximum number of sessions.
    #[getter]
    fn max_sessions(&self) -> usize {
        self.inner.max_sessions()
    }

    /// Returns the current number of cached sessions.
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Registers session-affinity bindings into the Python module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySessionCache>()?;
    module.add_function(wrap_pyfunction!(session_key_from_body, module)?)?;
    Ok(())
}
