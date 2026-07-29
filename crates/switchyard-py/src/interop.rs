// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private adapters between Python compatibility objects and native components.

use pyo3::prelude::*;

pub(crate) mod context;
pub(crate) mod request;
pub(crate) mod response;
pub(crate) mod roles;
pub(crate) mod subagent;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    context::register(module)?;
    response::register(module)?;
    roles::register(module)?;
    subagent::register(module)?;
    Ok(())
}
