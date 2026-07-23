// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

mod component_bindings;
mod core_bindings;
mod errors;
mod libsy_bindings;
mod py_serde;
mod translation;

#[pymodule]
fn _switchyard_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register(module)?;
    translation::register(module)?;
    core_bindings::register(module)?;
    component_bindings::register(module)?;
    libsy_bindings::register(module)?;
    Ok(())
}
