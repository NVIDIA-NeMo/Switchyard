// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error mapping helpers for PyO3 bindings.

use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use switchyard_core::SwitchyardError;
use switchyard_translation::TranslationError;

create_exception!(_switchyard_rust, SwitchyardRuntimeError, PyRuntimeError);
create_exception!(_switchyard_rust, LibsyError, SwitchyardRuntimeError);
create_exception!(
    _switchyard_rust,
    SwitchyardConfigError,
    SwitchyardRuntimeError
);
create_exception!(
    _switchyard_rust,
    SwitchyardInvalidIdError,
    SwitchyardRuntimeError
);
create_exception!(
    _switchyard_rust,
    SwitchyardDuplicateRegistrationError,
    SwitchyardRuntimeError
);
create_exception!(
    _switchyard_rust,
    SwitchyardModelNotFoundError,
    SwitchyardRuntimeError
);
create_exception!(
    _switchyard_rust,
    SwitchyardUnsupportedRequestTypeError,
    SwitchyardRuntimeError
);
// Raised by `ChatRequest.validate()` when a structurally valid body fails
// semantic validation (e.g. an empty `messages` array). Endpoints map it to
// a 4xx so agents can distinguish a client bug from a server failure.
create_exception!(
    _switchyard_rust,
    SwitchyardInvalidRequestError,
    SwitchyardRuntimeError
);
create_exception!(
    _switchyard_rust,
    SwitchyardProcessorError,
    SwitchyardRuntimeError
);
create_exception!(
    _switchyard_rust,
    SwitchyardBackendError,
    SwitchyardRuntimeError
);
create_exception!(
    _switchyard_rust,
    SwitchyardUpstreamError,
    SwitchyardRuntimeError
);
// Raised by an LLMBackend when the upstream rejects a request because the
// prompt exceeds the model's context window. Subclasses SwitchyardBackendError
// so existing broad catches still match.
create_exception!(
    _switchyard_rust,
    SwitchyardContextWindowExceededError,
    SwitchyardBackendError
);
// Raised by compatibility/runtime code when every attempted target returned a
// context-window overflow and no fallback remains.
create_exception!(
    _switchyard_rust,
    SwitchyardContextPoolExhaustedError,
    SwitchyardBackendError
);

/// Converts translation crate errors into Python `ValueError`s with stable context.
pub(crate) fn py_translation_error(error: TranslationError) -> PyErr {
    PyValueError::new_err(format!("{}: {}", error.kind(), error))
}

/// Converts libsy execution failures into one stable Python exception.
pub(crate) fn py_libsy_error(error: impl std::fmt::Display) -> PyErr {
    LibsyError::new_err(error.to_string())
}

/// Converts core Switchyard errors into typed Python runtime errors.
///
/// `ContextWindowExceeded` and `ContextPoolExhausted` carry typed fields
/// (`target_id`, `model`, `last_target_id`, `reason`) — we attach those as
/// Python attributes on the raised exception so callers can inspect them
/// programmatically and `backend_error_with_ctx` can recover the typed
/// variant when a Python `LLMBackend` re-raises through the Rust chain.
/// Without these attrs the variant collapses to the string message and the
/// compatibility retry code stamps `"unknown"` for `target_id`/`model`.
pub(crate) fn py_core_error(error: SwitchyardError) -> PyErr {
    let message = error.to_string();
    match error {
        SwitchyardError::InvalidConfig(_) => SwitchyardConfigError::new_err(message),
        SwitchyardError::InvalidId(_) => SwitchyardInvalidIdError::new_err(message),
        SwitchyardError::DuplicateRegistration { .. } => {
            SwitchyardDuplicateRegistrationError::new_err(message)
        }
        SwitchyardError::ModelNotFound { .. } => SwitchyardModelNotFoundError::new_err(message),
        SwitchyardError::UnsupportedRequestType { .. } => {
            SwitchyardUnsupportedRequestTypeError::new_err(message)
        }
        SwitchyardError::InvalidRequest(_) => SwitchyardInvalidRequestError::new_err(message),
        SwitchyardError::Processor(_) => SwitchyardProcessorError::new_err(message),
        SwitchyardError::Backend(_) => SwitchyardBackendError::new_err(message),
        SwitchyardError::Upstream(_) => SwitchyardUpstreamError::new_err(message),
        SwitchyardError::UpstreamHttp {
            status_code, body, ..
        } => {
            let err = SwitchyardUpstreamError::new_err(message);
            attach_upstream_http_attrs(&err, status_code, &body);
            err
        }
        SwitchyardError::ContextWindowExceeded {
            target_id, model, ..
        } => {
            let err = SwitchyardContextWindowExceededError::new_err(message);
            attach_attrs(&err, &[("target_id", &target_id), ("model", &model)]);
            err
        }
        SwitchyardError::ContextPoolExhausted {
            last_target_id,
            reason,
        } => {
            let err = SwitchyardContextPoolExhaustedError::new_err(message);
            attach_attrs(
                &err,
                &[("last_target_id", &last_target_id), ("reason", &reason)],
            );
            err
        }
        SwitchyardError::Other(_) => SwitchyardRuntimeError::new_err(message),
    }
}

/// Set string attributes on a freshly-constructed `PyErr`'s exception value.
/// Errors are intentionally ignored — attribute attachment is best-effort
/// diagnostic metadata, never the path-of-correctness for raising the error.
fn attach_attrs(err: &PyErr, attrs: &[(&str, &str)]) {
    Python::attach(|py| {
        let bound = err.value(py);
        for (name, value) in attrs {
            let _ = bound.setattr(*name, *value);
        }
    });
}

/// Attach typed upstream HTTP details for endpoint error handling.
fn attach_upstream_http_attrs(err: &PyErr, status_code: u16, body: &str) {
    Python::attach(|py| {
        let bound = err.value(py);
        let _ = bound.setattr("status_code", status_code);
        let _ = bound.setattr("body", body);
    });
}

/// Registers typed exception classes in the native extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add(
        "SwitchyardRuntimeError",
        py.get_type::<SwitchyardRuntimeError>(),
    )?;
    module.add("LibsyError", py.get_type::<LibsyError>())?;
    module.add(
        "SwitchyardConfigError",
        py.get_type::<SwitchyardConfigError>(),
    )?;
    module.add(
        "SwitchyardInvalidIdError",
        py.get_type::<SwitchyardInvalidIdError>(),
    )?;
    module.add(
        "SwitchyardDuplicateRegistrationError",
        py.get_type::<SwitchyardDuplicateRegistrationError>(),
    )?;
    module.add(
        "SwitchyardModelNotFoundError",
        py.get_type::<SwitchyardModelNotFoundError>(),
    )?;
    module.add(
        "SwitchyardUnsupportedRequestTypeError",
        py.get_type::<SwitchyardUnsupportedRequestTypeError>(),
    )?;
    module.add(
        "SwitchyardInvalidRequestError",
        py.get_type::<SwitchyardInvalidRequestError>(),
    )?;
    module.add(
        "SwitchyardProcessorError",
        py.get_type::<SwitchyardProcessorError>(),
    )?;
    module.add(
        "SwitchyardBackendError",
        py.get_type::<SwitchyardBackendError>(),
    )?;
    module.add(
        "SwitchyardUpstreamError",
        py.get_type::<SwitchyardUpstreamError>(),
    )?;
    module.add(
        "SwitchyardContextWindowExceededError",
        py.get_type::<SwitchyardContextWindowExceededError>(),
    )?;
    module.add(
        "SwitchyardContextPoolExhaustedError",
        py.get_type::<SwitchyardContextPoolExhaustedError>(),
    )?;
    Ok(())
}
