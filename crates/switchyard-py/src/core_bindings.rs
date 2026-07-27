// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bindings for `switchyard-core`.

use pyo3::prelude::*;

pub(crate) mod context;
pub(crate) mod request;
pub(crate) mod response;
pub(crate) mod roles;
pub(crate) mod session;
pub(crate) mod subagent;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    context::register(module)?;
    request::register(module)?;
    response::register(module)?;
    roles::register(module)?;
    session::register(module)?;
    subagent::register(module)?;
    Ok(())
}
