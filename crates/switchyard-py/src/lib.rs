// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

mod errors;
mod libsy_bindings;
mod py_serde;
mod server_bindings;

#[pymodule]
fn _switchyard_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register(module)?;
    libsy_bindings::register(module)?;
    server_bindings::register(module)?;
    Ok(())
}
