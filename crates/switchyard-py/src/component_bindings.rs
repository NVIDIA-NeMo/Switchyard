// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bindings for concrete `switchyard-components` implementations.

use pyo3::prelude::*;

mod backends;
pub(crate) mod config;
mod dimension_collector;
pub(crate) mod intake;
mod prefill_probe_request_processor;
mod request_processors;
mod response_processors;
pub(crate) mod stats;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    config::register(module)?;
    intake::register(module)?;
    stats::register(module)?;
    prefill_probe_request_processor::register(module)?;
    request_processors::register(module)?;
    response_processors::register(module)?;
    backends::register(module)?;
    dimension_collector::register(module)?;
    Ok(())
}
