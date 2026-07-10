// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the Rust components-v2 profile server.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use switchyard_server::{run_server, ServerRunOptions, DEFAULT_LISTEN_BACKLOG};

use crate::errors::py_core_error;

/// Run the Rust components-v2 profile server from Python.
#[pyfunction]
#[pyo3(signature = (
    config_path,
    host = "127.0.0.1",
    port = 4000,
    backlog = DEFAULT_LISTEN_BACKLOG,
    dry_run = false,
))]
fn run_profile_server(
    py: Python<'_>,
    config_path: String,
    host: &str,
    port: u16,
    backlog: u32,
    dry_run: bool,
) -> PyResult<()> {
    let ip: IpAddr = host.parse().map_err(|error| {
        PyValueError::new_err(format!(
            "host must be an IP address accepted by the Rust server, got {host:?}: {error}"
        ))
    })?;
    let options = ServerRunOptions {
        config: PathBuf::from(config_path),
        addr: SocketAddr::new(ip, port),
        backlog,
        dry_run,
        tls: None,
    };

    // `detach` runs synchronously with the GIL released, so startup errors still
    // return to the Python caller instead of disappearing into a background task.
    py.detach(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        runtime.block_on(run_server(options)).map_err(py_core_error)
    })
}

/// Registers Rust server bindings with the native Python module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_profile_server, module)?)?;
    Ok(())
}
