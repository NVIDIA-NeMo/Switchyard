// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Self-contained Python host for the Rust Switchyard server.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, sync_channel};
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use switchyard_server::config::load_server_state;
use switchyard_server::{
    BoundServer, DEFAULT_LISTEN_BACKLOG, ServerResult, ServerRunOptions, flush_observability,
    initialize_observability,
};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

const DEFAULT_SHUTDOWN_TIMEOUT_SECS: f64 = 2.0;

/// Running loopback server backed entirely by the native Rust implementation.
#[pyclass(name = "Server", module = "switchyard_rust.server", unsendable)]
struct PyServer {
    addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    completion: Option<Receiver<ServerResult<()>>>,
    task: Option<JoinHandle<()>>,
}

#[pymethods]
impl PyServer {
    /// Loads a TOML deployment and starts serving it on loopback.
    #[new]
    #[pyo3(signature = (config, *, port=0))]
    fn new(config: PathBuf, port: u16) -> PyResult<Self> {
        initialize_observability().map_err(server_error)?;
        let state = load_server_state(config).map_err(server_error)?;
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let server = {
            let _guard = runtime.enter();
            BoundServer::bind(
                state,
                ServerRunOptions {
                    addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port),
                    backlog: DEFAULT_LISTEN_BACKLOG,
                    dry_run: false,
                    tls: None,
                },
            )
            .map_err(server_error)?
        };
        let addr = server.local_addr();
        let (shutdown, shutdown_receiver) = oneshot::channel();
        let (completion_sender, completion) = sync_channel(1);
        let task = runtime.spawn(async move {
            let result = server
                .serve(async move {
                    let _ = shutdown_receiver.await;
                })
                .await;
            let _ = completion_sender.send(result);
        });
        Ok(Self {
            addr,
            shutdown: Some(shutdown),
            completion: Some(completion),
            task: Some(task),
        })
    }

    /// Returns the bound loopback port.
    #[getter]
    fn port(&self) -> u16 {
        self.addr.port()
    }

    /// Returns the HTTP base URL for clients.
    #[getter]
    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Gracefully stops the server and flushes pending telemetry.
    #[pyo3(signature = (*, timeout_secs=DEFAULT_SHUTDOWN_TIMEOUT_SECS))]
    fn close(&mut self, py: Python<'_>, timeout_secs: f64) -> PyResult<()> {
        self.close_inner(py, timeout_secs)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exception_type: &Bound<'_, PyAny>,
        _exception: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.close_inner(py, DEFAULT_SHUTDOWN_TIMEOUT_SECS)?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        format!("Server(base_url={:?})", self.base_url())
    }
}

impl PyServer {
    fn close_inner(&mut self, py: Python<'_>, timeout_secs: f64) -> PyResult<()> {
        let timeout = Duration::try_from_secs_f64(timeout_secs).map_err(|_| {
            PyValueError::new_err("timeout_secs must be a supported finite non-negative number")
        })?;
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        let Some(completion) = self.completion.take() else {
            return Ok(());
        };
        let task = self.task.take();
        let result = py.detach(move || {
            let result = completion.recv_timeout(timeout);
            if matches!(result, Err(RecvTimeoutError::Timeout))
                && let Some(task) = task
            {
                task.abort();
            }
            flush_observability();
            result
        });
        match result {
            Ok(result) => result.map_err(server_error),
            Err(RecvTimeoutError::Timeout) => Err(PyRuntimeError::new_err(format!(
                "server did not stop within {timeout_secs} seconds"
            ))),
            Err(RecvTimeoutError::Disconnected) => Err(PyRuntimeError::new_err(
                "server task stopped without reporting a result",
            )),
        }
    }
}

impl Drop for PyServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

fn server_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let server_module = PyModule::new(module.py(), "server")?;
    server_module.add_class::<PyServer>()?;
    module.add_submodule(&server_module)?;
    Ok(())
}
