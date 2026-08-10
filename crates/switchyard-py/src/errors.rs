// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error mapping for the libsy Python binding.

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

create_exception!(_switchyard_rust, LibsyError, PyRuntimeError);

/// Converts libsy execution failures into one stable Python exception.
pub(crate) fn py_libsy_error(error: impl std::fmt::Display) -> PyErr {
    LibsyError::new_err(error.to_string())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("LibsyError", module.py().get_type::<LibsyError>())
}
