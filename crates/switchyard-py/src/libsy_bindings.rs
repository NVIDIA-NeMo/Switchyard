// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Switchyard's neutral LLM protocol values.

mod protocol;
mod values;

use pyo3::prelude::*;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let protocol_module = PyModule::new(module.py(), "libsy_protocol")?;
    protocol::register(&protocol_module)?;
    values::register(&protocol_module)?;
    module.add_submodule(&protocol_module)?;

    Ok(())
}
