// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native configured routing decisions for Python hosts.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll};

use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyUnicodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping, PyString};
use pyo3_async_runtimes::TaskLocals;
use pyo3_async_runtimes::generic::{ContextExt, Runtime};
use switchyard_llm_client::{LlmCallObservation, RunObservation, RunObserver};
use switchyard_protocol::{Metadata, ModelId, Request};
use switchyard_runner::{DecisionTarget, Route, Runner, RunnerError};
use tokio::sync::oneshot;

use crate::libsy_bindings::{PyRoutingOutcome, header_map_from_python, outcome_to_python};
use crate::py_serde::{from_python, to_python};

create_exception!(_switchyard_rust, DecisionError, PyRuntimeError);

/// Configured identity of one selected target.
#[pyclass(name = "DecisionTarget", module = "switchyard.runner", frozen)]
struct PyDecisionTarget {
    inner: DecisionTarget,
}

#[pymethods]
impl PyDecisionTarget {
    #[getter]
    fn target(&self) -> &str {
        &self.inner.target
    }

    #[getter]
    fn model(&self) -> &str {
        self.inner.model.as_str()
    }
}

/// A completed logical model call, including backend retries in its duration.
#[pyclass(name = "RoutingCall", module = "switchyard.runner", frozen)]
struct PyRoutingCall {
    inner: LlmCallObservation,
}

#[pymethods]
impl PyRoutingCall {
    #[getter]
    fn model(&self) -> &str {
        self.inner.selected_model.as_str()
    }

    #[getter]
    fn is_success(&self) -> bool {
        self.inner.is_success
    }

    #[getter]
    fn duration_seconds(&self) -> f64 {
        self.inner.duration.as_secs_f64()
    }

    /// Native normalized usage; missing usage and token fields remain unknown.
    #[getter]
    fn usage(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .usage
            .as_ref()
            .map(|usage| to_python(py, usage))
            .transpose()
    }
}

/// A native outcome, its configured target identities, and request-scoped costs.
#[pyclass(name = "Decision", module = "switchyard.runner", frozen)]
struct PyDecision {
    #[pyo3(get)]
    selected: Py<PyDecisionTarget>,
    #[pyo3(get)]
    fallbacks: Vec<Py<PyDecisionTarget>>,
    #[pyo3(get)]
    outcome: Py<PyRoutingOutcome>,
    #[pyo3(get)]
    calls: Vec<Py<PyRoutingCall>>,
    #[pyo3(get)]
    duration_seconds: f64,
}

#[derive(Default)]
struct Observations {
    calls: Vec<LlmCallObservation>,
    duration_seconds: f64,
}

/// Shared native route configuration and algorithm state.
#[pyclass(name = "Runner", module = "switchyard.runner", frozen)]
struct PyRunner {
    inner: Arc<Runner>,
}

#[derive(Default)]
struct BridgeTasks {
    pending: Mutex<usize>,
    finished: Condvar,
}

impl BridgeTasks {
    fn track(self: &Arc<Self>) -> BridgeTask {
        *self.pending.lock().expect("bridge task lock poisoned") += 1;
        BridgeTask(Arc::clone(self))
    }

    fn wait(&self) {
        let mut pending = self.pending.lock().expect("bridge task lock poisoned");
        while *pending != 0 {
            pending = self
                .finished
                .wait(pending)
                .expect("bridge task lock poisoned");
        }
    }
}

struct BridgeTask(Arc<BridgeTasks>);

impl Drop for BridgeTask {
    fn drop(&mut self) {
        let mut pending = self.0.pending.lock().expect("bridge task lock poisoned");
        *pending -= 1;
        if *pending == 0 {
            self.0.finished.notify_all();
        }
    }
}

// Field order matters: drop the future/closure and its Python references before
// announcing that this bridge task is finished, including during unwinding.
struct Tracked<T> {
    inner: T,
    _task: BridgeTask,
}

impl<F: Future> Future for Tracked<Pin<Box<F>>> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

impl<F: FnOnce()> Tracked<F> {
    fn run(self) {
        (self.inner)();
    }
}

tokio::task_local! {
    static BRIDGE_TASKS: Arc<BridgeTasks>;
    static BRIDGE_LOCALS: TaskLocals;
}

/// Use the existing PyO3 bridge while tracking its detached completion tasks.
struct BridgeRuntime;

impl Runtime for BridgeRuntime {
    type JoinError = tokio::task::JoinError;
    type JoinHandle = tokio::task::JoinHandle<()>;

    fn spawn<F>(future: F) -> Self::JoinHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let tasks = BRIDGE_TASKS.with(Arc::clone);
        let task = tasks.track();
        pyo3_async_runtimes::tokio::get_runtime().spawn(Tracked {
            inner: Box::pin(BRIDGE_TASKS.scope(tasks, future)),
            _task: task,
        })
    }

    fn spawn_blocking<F>(function: F) -> Self::JoinHandle
    where
        F: FnOnce() + Send + 'static,
    {
        let work = Tracked {
            inner: function,
            _task: BRIDGE_TASKS.with(|tasks| tasks.track()),
        };
        pyo3_async_runtimes::tokio::get_runtime().spawn_blocking(move || work.run())
    }
}

impl ContextExt for BridgeRuntime {
    fn scope<F, R>(locals: TaskLocals, future: F) -> Pin<Box<dyn Future<Output = R> + Send>>
    where
        F: Future<Output = R> + Send + 'static,
    {
        Box::pin(BRIDGE_LOCALS.scope(locals, future))
    }

    fn get_task_locals() -> Option<TaskLocals> {
        BRIDGE_LOCALS.try_with(Clone::clone).ok()
    }
}

/// Cancel routing without cancelling the Python completion bridge.
#[pyclass]
struct DecisionCancellation {
    sender: Option<oneshot::Sender<()>>,
    tasks: Arc<BridgeTasks>,
}

#[pymethods]
impl DecisionCancellation {
    fn cancel(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(());
        }
    }

    fn wait(&self, py: Python<'_>) {
        // Called from a Python executor thread after result delivery. Release the
        // GIL while the bridge finishes using Python and drops its references.
        py.detach(|| self.tasks.wait());
    }

    fn is_finished(&self) -> bool {
        *self
            .tasks
            .pending
            .lock()
            .expect("bridge task lock poisoned")
            == 0
    }
}

#[pymethods]
impl PyRunner {
    /// Load and validate a native deployment TOML file without starting a server.
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        py.detach(move || {
            let _guard = pyo3_async_runtimes::tokio::get_runtime().enter();
            Runner::load(path)
                .map(Self::from_runner)
                .map_err(config_error)
        })
    }

    /// Load and validate a native deployment TOML document.
    #[staticmethod]
    fn from_toml(py: Python<'_>, source: &Bound<'_, PyString>) -> PyResult<Self> {
        let source = source
            .to_cow()
            .map_err(|error| {
                if error.is_instance_of::<PyUnicodeError>(py) {
                    // Unicode errors retain the complete configuration in their args.
                    PyValueError::new_err("deployment TOML must be valid UTF-8")
                } else {
                    error
                }
            })?
            .into_owned();
        py.detach(move || {
            let _guard = pyo3_async_runtimes::tokio::get_runtime().enter();
            Runner::from_toml(&source)
                .map(Self::from_runner)
                .map_err(config_error)
        })
    }

    /// Check a route before evaluation can issue any provider calls.
    #[pyo3(signature = (model, *, allow_response=false))]
    fn validate_decision_route(
        &self,
        py: Python<'_>,
        model: &str,
        allow_response: bool,
    ) -> PyResult<Vec<Py<PyDecisionTarget>>> {
        decision_route(&self.inner, model, allow_response)?
            .decision_targets()
            .iter()
            .cloned()
            .map(|inner| Py::new(py, PyDecisionTarget { inner }))
            .collect()
    }

    /// Resolve one normalized IR request using native configuration and transport.
    ///
    /// Response-based algorithms require `allow_response=True` because they can call
    /// an answer model while deciding. Separate runs may execute concurrently. The
    /// caller must give independent tasks distinct session identities and serialize
    /// requests whose shared session state must remain ordered.
    #[pyo3(signature = (request, *, headers=None, allow_response=false))]
    fn decide<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
        headers: Option<&Bound<'py, PyMapping>>,
        allow_response: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let headers = headers
            .map(|mapping| {
                let owned = PyDict::new(py);
                owned.update(mapping)?;
                owned.extract::<HashMap<String, String>>().map_err(|error| {
                    if error.is_instance_of::<PyUnicodeError>(py) {
                        // Unicode errors retain the complete input in their args.
                        PyValueError::new_err("header names and values must be valid UTF-8")
                    } else {
                        error
                    }
                })
            })
            .transpose()?;
        // Start routing only when the coroutine runs, so cancellation before its
        // first step cannot leave an unowned native request running.
        py.import("switchyard_rust.runner")?
            .getattr("_decide")?
            .call1((slf, request, headers, allow_response))
    }

    #[pyo3(signature = (request, *, headers=None, allow_response=false))]
    fn _start_decision<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
        headers: Option<HashMap<String, String>>,
        allow_response: bool,
    ) -> PyResult<(Bound<'py, PyAny>, DecisionCancellation)> {
        let headers = headers.as_ref().map(header_map_from_python).transpose()?;
        let request = Request {
            llm_request: from_python(request)?,
            raw_request: None,
            metadata: headers.map(|headers| {
                let mut metadata = Metadata::from_headers(&headers);
                metadata.http_headers = Some(headers);
                metadata
            }),
        };
        let model = request
            .llm_request
            .model
            .as_deref()
            .filter(|model| !model.trim().is_empty())
            .ok_or_else(|| PyValueError::new_err("request must include a non-empty model"))?;
        decision_route(&self.inner, model, allow_response)?;
        let model = ModelId::from(model);
        let runner = Arc::clone(&self.inner);
        let (sender, cancelled) = oneshot::channel();
        let tasks = Arc::new(BridgeTasks::default());
        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
        let future = async move {
            let observations = Arc::new(Mutex::new(Observations::default()));
            let captured = Arc::clone(&observations);
            let observer: RunObserver = Arc::new(move |event| {
                let mut captured = captured.lock().expect("observation lock poisoned");
                match event {
                    RunObservation::LlmCall(call) | RunObservation::AnswerCall(call) => {
                        captured.calls.push(call);
                    }
                    RunObservation::RoutingOverhead(duration) => {
                        captured.duration_seconds = duration.as_secs_f64();
                    }
                    RunObservation::Outcome(_) => {}
                }
            });
            let route = decision_route(&runner, model.as_str(), allow_response)?;
            let result = tokio::select! {
                biased;
                _ = cancelled => return Ok(None),
                result = route.decide_with_observer(request, Some(observer)) => result,
            };
            let observations =
                std::mem::take(&mut *observations.lock().expect("observation lock poisoned"));
            Python::attach(|py| {
                let calls = observations
                    .calls
                    .into_iter()
                    .map(|inner| Py::new(py, PyRoutingCall { inner }))
                    .collect::<PyResult<Vec<_>>>()?;
                let outcome = result.map_err(|error| {
                    decision_error(py, error, &calls, observations.duration_seconds)
                })?;
                let description = runner.describe_decision(&model, &outcome).ok_or_else(|| {
                    decision_error(
                        py,
                        RunnerError::Algorithm(switchyard_libsy::LibsyError::AlgorithmError {
                            message: "routing outcome has no configured target".to_string(),
                        }),
                        &calls,
                        observations.duration_seconds,
                    )
                })?;
                Py::new(
                    py,
                    PyDecision {
                        selected: Py::new(
                            py,
                            PyDecisionTarget {
                                inner: description.selected,
                            },
                        )?,
                        fallbacks: description
                            .fallbacks
                            .into_iter()
                            .map(|inner| Py::new(py, PyDecisionTarget { inner }))
                            .collect::<PyResult<Vec<_>>>()?,
                        outcome: outcome_to_python(py, outcome)?,
                        calls,
                        duration_seconds: observations.duration_seconds,
                    },
                )
            })
            .map(Some)
        };
        let future = BRIDGE_TASKS.sync_scope(Arc::clone(&tasks), || {
            pyo3_async_runtimes::generic::future_into_py_with_locals::<BridgeRuntime, _, _>(
                py, locals, future,
            )
        })?;
        Ok((
            future,
            DecisionCancellation {
                sender: Some(sender),
                tasks,
            },
        ))
    }
}

impl PyRunner {
    fn from_runner(inner: Runner) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

fn decision_route<'a>(
    runner: &'a Runner,
    model: &str,
    allow_response: bool,
) -> PyResult<&'a Route> {
    let route = runner
        .route(model)
        .ok_or_else(|| PyValueError::new_err(format!("unknown route model {model:?}")))?;
    route.validate_decision_targets().map_err(config_error)?;
    if !allow_response && route.routing_answer_target().is_some() {
        return Err(PyValueError::new_err(
            "route can call an answer model while deciding; set allow_response=True to permit it",
        ));
    }
    Ok(route)
}

fn config_error(error: RunnerError) -> PyErr {
    PyValueError::new_err(error.configuration_diagnostic())
}

fn decision_error(
    py: Python<'_>,
    error: RunnerError,
    calls: &[Py<PyRoutingCall>],
    duration_seconds: f64,
) -> PyErr {
    let summary = error.execution_error_summary();
    // Provider error bodies may echo credentials or request text. The native summary
    // preserves the actionable class and HTTP status without those untrusted details.
    let error = DecisionError::new_err(format!("routing failed: {}", summary.kind.as_str()));
    let value = error.value(py);
    let set_fields = || -> PyResult<()> {
        value.setattr("kind", summary.kind.as_str())?;
        value.setattr("upstream_status", summary.upstream_status)?;
        value.setattr("target", summary.target.as_ref().map(ModelId::as_str))?;
        value.setattr("calls", calls)?;
        value.setattr("duration_seconds", duration_seconds)
    };
    if let Err(error) = set_fields() {
        return error;
    }
    error
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let runner_module = PyModule::new(module.py(), "runner")?;
    runner_module.add("DecisionError", module.py().get_type::<DecisionError>())?;
    runner_module.add_class::<PyRunner>()?;
    runner_module.add_class::<PyDecision>()?;
    runner_module.add_class::<PyDecisionTarget>()?;
    runner_module.add_class::<PyRoutingCall>()?;
    module.add_submodule(&runner_module)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    struct Payload {
        tasks: Arc<BridgeTasks>,
        dropped: Arc<AtomicBool>,
    }

    impl Drop for Payload {
        fn drop(&mut self) {
            assert_eq!(*self.tasks.pending.lock().unwrap(), 1);
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn bridge_completion_follows_payload_drop_when_unpolled_or_panicking() {
        for panic in [false, true] {
            let tasks = Arc::new(BridgeTasks::default());
            let dropped = Arc::new(AtomicBool::new(false));
            let payload = Payload {
                tasks: Arc::clone(&tasks),
                dropped: Arc::clone(&dropped),
            };
            if panic {
                let work = Tracked {
                    inner: move || {
                        let _payload = payload;
                        panic!("bridge worker failed");
                    },
                    _task: tasks.track(),
                };
                assert!(std::panic::catch_unwind(|| work.run()).is_err());
            } else {
                let future = Tracked {
                    inner: Box::pin(async move {
                        let _payload = payload;
                        std::future::pending::<()>().await;
                    }),
                    _task: tasks.track(),
                };
                drop(future);
            }
            assert!(dropped.load(Ordering::SeqCst));
            assert_eq!(*tasks.pending.lock().unwrap(), 0);
        }
    }
}
