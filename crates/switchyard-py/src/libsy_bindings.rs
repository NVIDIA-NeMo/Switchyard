// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Switchyard's neutral protocol and libsy targets.

mod protocol;
mod target;
mod values;

use pyo3::prelude::*;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let protocol_module = PyModule::new(module.py(), "libsy_protocol")?;
    protocol::register(&protocol_module)?;
    values::register(&protocol_module)?;
    module.add_submodule(&protocol_module)?;

    let libsy_module = PyModule::new(module.py(), "libsy")?;
    target::register(&libsy_module)?;
    module.add_submodule(&libsy_module)?;
    Ok(())
}
