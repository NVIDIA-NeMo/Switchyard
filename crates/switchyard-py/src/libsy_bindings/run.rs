// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-driven libsy run streams and their Python step variants.

use std::sync::{Arc, Mutex};

use futures_util::{stream, StreamExt};
use libsy::{Algorithm, CallLlmRequest, NoopAlgo, RandomAlgo, Step, StepStream};
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::prelude::*;
use switchyard_protocol::{Context, Decision, Request};

use super::protocol::PyDecision;
use super::target::PyLlmTargetSet;
use super::values::{PyContext, PyRequest, PyResponse};
use crate::errors::py_libsy_error;

#[pyclass(name = "LlmCall", module = "switchyard.libsy")]
pub(crate) struct PyLlmCall {
    inner: Mutex<Option<CallLlmRequest>>,
    context: Context,
    request: Request,
    decision: Arc<dyn Decision>,
}

impl PyLlmCall {
    fn new(inner: CallLlmRequest) -> Self {
        let context = inner.get_routed().ctx.clone();
        let request = inner.get_request().clone();
        let decision = inner.get_routed().decision.clone();
        Self {
            inner: Mutex::new(Some(inner)),
            context,
            request,
            decision,
        }
    }
}

#[pymethods]
impl PyLlmCall {
    #[getter]
    fn context(&self, py: Python<'_>) -> PyResult<Py<PyContext>> {
        Py::new(py, PyContext::from_core(self.context.clone()))
    }

    #[getter]
    fn request(&self, py: Python<'_>) -> PyResult<Py<PyRequest>> {
        Py::new(py, PyRequest::from_core(self.request.clone()))
    }

    #[getter]
    fn decision(&self, py: Python<'_>) -> PyResult<Py<PyDecision>> {
        Py::new(py, PyDecision::new(self.decision.clone()))
    }

    fn respond(&self, response: PyRef<'_, PyResponse>) -> PyResult<()> {
        let mut call = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("LlmCall lock is poisoned"))?;
        if call.is_none() {
            return Err(PyRuntimeError::new_err(
                "LlmCall has already been fulfilled",
            ));
        }
        let response = response.take_core()?;
        let Some(call) = call.take() else {
            return Err(PyRuntimeError::new_err(
                "LlmCall has already been fulfilled",
            ));
        };
        call.respond(Ok(response)).map_err(py_libsy_error)
    }

    fn fail(&self, message: String) -> PyResult<()> {
        let mut call = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("LlmCall lock is poisoned"))?;
        let Some(call) = call.take() else {
            return Err(PyRuntimeError::new_err(
                "LlmCall has already been fulfilled",
            ));
        };
        call.respond(Err(std::io::Error::other(message).into()))
            .map_err(py_libsy_error)
    }

    #[getter]
    fn is_pending(&self) -> PyResult<bool> {
        self.inner
            .lock()
            .map(|call| call.is_some())
            .map_err(|_| PyRuntimeError::new_err("LlmCall lock is poisoned"))
    }

    fn __repr__(&self) -> String {
        format!(
            "LlmCall(selected_model={:?})",
            self.decision.selected_model()
        )
    }
}

/// Python's discriminated-union view of [`libsy::Step`].
#[pyclass(name = "Step", module = "switchyard.libsy", frozen)]
pub(crate) enum PyStep {
    CallLlm { call: Py<PyLlmCall> },
    Decision { decision: Py<PyDecision> },
    ReturnToAgent { response: Py<PyResponse> },
}

fn step_to_python(py: Python<'_>, step: Step) -> PyResult<Py<PyAny>> {
    let step = match step {
        Step::CallLlm(call) => PyStep::CallLlm {
            call: Py::new(py, PyLlmCall::new(*call))?,
        },
        Step::Decision(decision) => PyStep::Decision {
            decision: Py::new(py, PyDecision::new(decision))?,
        },
        Step::ReturnToAgent(response) => PyStep::ReturnToAgent {
            response: Py::new(py, PyResponse::from_core(*response))?,
        },
    };
    step.into_pyobject(py)
        .map(|value| value.unbind().into_any())
}

#[pyclass(name = "RunStream", module = "switchyard.libsy")]
pub(crate) struct PyRunStream {
    stream: Arc<tokio::sync::Mutex<Option<StepStream>>>,
    consumed: bool,
}

impl PyRunStream {
    fn new(algorithm: Arc<dyn Algorithm>, context: Context, request: Request) -> Self {
        let stream = stream::once(async move { algorithm.run_stream(context, request) }).flatten();
        Self {
            stream: Arc::new(tokio::sync::Mutex::new(Some(Box::pin(stream)))),
            consumed: false,
        }
    }
}

#[pymethods]
impl PyRunStream {
    fn __aiter__(mut slf: PyRefMut<'_, Self>) -> PyResult<Py<PyAny>> {
        if slf.consumed {
            return Err(PyRuntimeError::new_err(
                "RunStream has already been consumed",
            ));
        }
        slf.consumed = true;
        let py = slf.py();
        Ok(slf.into_pyobject(py)?.unbind().into_any())
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = Arc::clone(&self.stream);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = stream.lock().await;
            let Some(inner) = guard.as_mut() else {
                return Err(PyStopAsyncIteration::new_err(()));
            };
            match inner.next().await {
                Some(Ok(step)) => {
                    let is_terminal = matches!(&step, Step::ReturnToAgent(_));
                    let step = Python::attach(|py| step_to_python(py, step))?;
                    if is_terminal {
                        guard.take();
                    }
                    drop(guard);
                    Ok(step)
                }
                Some(Err(error)) => {
                    guard.take();
                    Err(py_libsy_error(error))
                }
                None => {
                    guard.take();
                    Err(PyStopAsyncIteration::new_err(()))
                }
            }
        })
    }

    fn aclose<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.consumed = true;
        let stream = Arc::clone(&self.stream);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            stream.lock().await.take();
            Ok(())
        })
    }

    fn __repr__(&self) -> &'static str {
        "RunStream()"
    }
}

/// Opaque Python handle to a Rust-owned routing algorithm.
#[pyclass(name = "Algorithm", module = "switchyard.libsy", frozen)]
pub(crate) struct PyAlgorithm {
    inner: Arc<dyn Algorithm>,
}

impl PyAlgorithm {
    fn new(inner: Arc<dyn Algorithm>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAlgorithm {
    /// Run the algorithm using its targets' default clients.
    #[pyo3(signature = (request, *, context=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        request: PyRef<'_, PyRequest>,
        context: Option<PyRef<'_, PyContext>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let algorithm = Arc::clone(&self.inner);
        let context = context
            .map(|context| context.clone_core())
            .unwrap_or_default();
        let request = request.clone_core();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (decisions, response) = algorithm
                .run(context, request)
                .await
                .map_err(py_libsy_error)?;
            Python::attach(|py| {
                let decisions = decisions
                    .into_iter()
                    .map(|decision| Py::new(py, PyDecision::new(decision)))
                    .collect::<PyResult<Vec<_>>>()?;
                let response = Py::new(py, PyResponse::from_core(response))?;
                Ok((decisions, response))
            })
        })
    }

    /// Return a stream that lets the Python host serve model calls.
    #[pyo3(signature = (request, *, context=None))]
    fn run_stream(
        &self,
        request: PyRef<'_, PyRequest>,
        context: Option<PyRef<'_, PyContext>>,
    ) -> PyRunStream {
        let context = context
            .map(|context| context.clone_core())
            .unwrap_or_default();
        PyRunStream::new(Arc::clone(&self.inner), context, request.clone_core())
    }

    fn __repr__(&self) -> &'static str {
        "Algorithm()"
    }
}

/// Construct the no-op reference algorithm.
#[pyfunction(name = "noop")]
fn noop_algorithm() -> PyAlgorithm {
    PyAlgorithm::new(Arc::new(NoopAlgo {}))
}

/// Construct uniform random routing over the supplied targets.
#[pyfunction(name = "random")]
fn random_algorithm(py: Python<'_>, targets: PyRef<'_, PyLlmTargetSet>) -> PyResult<PyAlgorithm> {
    Ok(PyAlgorithm::new(Arc::new(RandomAlgo::new(
        targets.clone_core(py)?,
    ))))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLlmCall>()?;
    module.add_class::<PyStep>()?;
    module.add_class::<PyRunStream>()?;
    module.add_class::<PyAlgorithm>()?;
    module.add_function(wrap_pyfunction!(noop_algorithm, module)?)?;
    module.add_function(wrap_pyfunction!(random_algorithm, module)?)?;
    Ok(())
}
