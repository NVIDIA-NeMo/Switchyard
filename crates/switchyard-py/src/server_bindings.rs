// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the Rust components-v2 profile server.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use switchyard_components_v2::{
    RelaySnapshotLimits, DEFAULT_MAX_ATOF_BATCH_BYTES, DEFAULT_MAX_ATOF_EVENT_BYTES,
    DEFAULT_MAX_RELAY_DEDUPE_ENTRIES, DEFAULT_MAX_RELAY_HISTORY_PER_IDENTITY,
    DEFAULT_MAX_RELAY_IDENTITIES, DEFAULT_MAX_RELAY_RETAINED_BYTES,
    DEFAULT_MAX_RELAY_SNAPSHOT_AGE_MILLIS,
};
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
    atof_bearer_token = None,
    atof_max_identities = DEFAULT_MAX_RELAY_IDENTITIES,
    atof_max_history_per_identity = DEFAULT_MAX_RELAY_HISTORY_PER_IDENTITY,
    atof_max_dedupe_entries = DEFAULT_MAX_RELAY_DEDUPE_ENTRIES,
    atof_max_retained_bytes = DEFAULT_MAX_RELAY_RETAINED_BYTES,
    atof_max_event_bytes = DEFAULT_MAX_ATOF_EVENT_BYTES,
    atof_max_batch_bytes = DEFAULT_MAX_ATOF_BATCH_BYTES,
    atof_max_snapshot_age_millis = DEFAULT_MAX_RELAY_SNAPSHOT_AGE_MILLIS,
))]
// PyO3 exposes these as named Python deployment options; grouping them would
// replace simple keyword arguments with a binding-only configuration class.
#[allow(clippy::too_many_arguments)]
fn run_profile_server(
    py: Python<'_>,
    config_path: String,
    host: &str,
    port: u16,
    backlog: u32,
    dry_run: bool,
    atof_bearer_token: Option<String>,
    atof_max_identities: usize,
    atof_max_history_per_identity: usize,
    atof_max_dedupe_entries: usize,
    atof_max_retained_bytes: usize,
    atof_max_event_bytes: usize,
    atof_max_batch_bytes: usize,
    atof_max_snapshot_age_millis: u64,
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
        atof_bearer_token,
        relay_snapshot_limits: RelaySnapshotLimits {
            max_identities: atof_max_identities,
            max_history_per_identity: atof_max_history_per_identity,
            max_dedupe_entries: atof_max_dedupe_entries,
            max_retained_bytes: atof_max_retained_bytes,
            max_event_bytes: atof_max_event_bytes,
            max_batch_bytes: atof_max_batch_bytes,
            max_snapshot_age_millis: atof_max_snapshot_age_millis,
        },
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
