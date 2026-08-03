// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod client;
mod config;
mod executor;
mod ffi;
mod runtime;
mod translation;

use std::ffi::c_void;
use std::mem;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures_util::FutureExt;
use nemo_relay_plugin::{
    ConfigDiagnostic, DiagnosticLevel, Json, NEMO_RELAY_NATIVE_ABI_VERSION_ASYNC_MIDDLEWARE,
    NativePlugin, NemoRelayNativeAsyncCallbackState, NemoRelayNativeAsyncCompletion,
    NemoRelayNativeAsyncMiddlewareKind, NemoRelayNativeAsyncNext, NemoRelayNativeAsyncStream,
    NemoRelayNativeHostApiV3, NemoRelayNativeString, NemoRelayStatus, PluginContext,
};
use serde::Deserialize;
use serde_json::Map;

use crate::config::SwitchyardConfig;
use crate::executor::PluginExecutor;
use crate::runtime::SwitchyardRuntime;

#[derive(Deserialize)]
struct Invocation {
    name: String,
    request: nemo_relay_plugin::LlmRequest,
}

struct CallbackState {
    host: NemoRelayNativeHostApiV3,
    runtime: Arc<SwitchyardRuntime>,
    executor: PluginExecutor,
}

#[derive(Default)]
struct SwitchyardPlugin;

impl NativePlugin for SwitchyardPlugin {
    fn plugin_kind(&self) -> &str {
        "nvidia.switchyard"
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        match parse_config(plugin_config).and_then(|config| config.validate()) {
            Ok(()) => Vec::new(),
            Err(message) => vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "switchyard.invalid_config".into(),
                component: Some("nvidia.switchyard".into()),
                field: Some("config".into()),
                message,
            }],
        }
    }

    fn register(
        &mut self,
        plugin_config: &Map<String, Json>,
        ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        let host_v1 = ctx.host_api();
        if host_v1.abi_version < NEMO_RELAY_NATIVE_ABI_VERSION_ASYNC_MIDDLEWARE
            || host_v1.struct_size < mem::size_of::<NemoRelayNativeHostApiV3>()
        {
            return Err(
                "Switchyard requires Relay 0.7 or newer with the generic asynchronous native host table"
                    .into(),
            );
        }
        let host = unsafe { *(host_v1 as *const _ as *const NemoRelayNativeHostApiV3) };
        let config = parse_config(plugin_config)?;
        let priority = config.priority;
        let state = Arc::new(CallbackState {
            host,
            runtime: Arc::new(SwitchyardRuntime::new(config)?),
            executor: PluginExecutor::new()?,
        });

        register_buffered(ctx, priority, Arc::clone(&state))?;
        register_stream(ctx, priority, state)?;
        Ok(())
    }
}

fn register_buffered(
    ctx: &mut PluginContext<'_>,
    priority: i32,
    state: Arc<CallbackState>,
) -> Result<(), String> {
    let user_data = Box::into_raw(Box::new(state)).cast::<c_void>();
    let status = unsafe {
        ctx.register_async_middleware_raw(
            NemoRelayNativeAsyncMiddlewareKind::LlmExecutionIntercept,
            "switchyard.run_stream.buffered",
            priority,
            false,
            buffered_callback,
            user_data,
            Some(free_callback_state),
        )
    };
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        Err(format!(
            "failed to register Switchyard buffered execution: {status:?}"
        ))
    }
}

fn register_stream(
    ctx: &mut PluginContext<'_>,
    priority: i32,
    state: Arc<CallbackState>,
) -> Result<(), String> {
    let user_data = Box::into_raw(Box::new(state)).cast::<c_void>();
    let status = unsafe {
        ctx.register_async_stream_middleware_raw(
            "switchyard.run_stream.streaming",
            priority,
            stream_callback,
            user_data,
            Some(free_callback_state),
        )
    };
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        Err(format!(
            "failed to register Switchyard streaming execution: {status:?}"
        ))
    }
}

fn parse_config(plugin_config: &Map<String, Json>) -> Result<SwitchyardConfig, String> {
    match plugin_config.get("version").and_then(Json::as_u64) {
        Some(2) => {}
        Some(version) => {
            return Err(format!(
                "unsupported Switchyard config version {version}; version 1 used switchyard-server; migrate to version = 2"
            ));
        }
        None => {
            return Err("invalid Switchyard configuration: version must be the integer 2".into());
        }
    }
    serde_json::from_value(Json::Object(plugin_config.clone()))
        .map_err(|error| format!("invalid Switchyard configuration: {error}"))
}

unsafe extern "C" fn free_callback_state(user_data: *mut c_void) {
    if !user_data.is_null() {
        unsafe { drop(Box::from_raw(user_data.cast::<Arc<CallbackState>>())) };
    }
}

unsafe extern "C" fn buffered_callback(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> u32 {
    if user_data.is_null() || completion.is_null() || next.is_null() {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    }
    let state = unsafe { &*user_data.cast::<Arc<CallbackState>>() }.clone();
    let invocation = ffi::read_json(&state.host.v1, invocation_json).and_then(|value| {
        serde_json::from_value::<Invocation>(value).map_err(|error| error.to_string())
    });
    let next = next as usize;
    let completion = completion as usize;
    let invocation = match invocation {
        Ok(invocation) => invocation,
        Err(error) => {
            let _ = ffi::reject_completion(
                &state.host,
                completion as *const NemoRelayNativeAsyncCompletion,
                &format!("invalid Relay LLM invocation: {error}"),
            );
            unsafe {
                ffi::release_next(&state.host, next as *const NemoRelayNativeAsyncNext);
                ffi::release_completion(
                    &state.host,
                    completion as *const NemoRelayNativeAsyncCompletion,
                );
            }
            return NemoRelayNativeAsyncCallbackState::Pending as u32;
        }
    };
    let Some(inbound) = state.runtime.managed_protocol(&invocation.name) else {
        if let Err(error) =
            ffi::invoke_next_buffered(&state.host, next, completion, &invocation.request)
        {
            let _ = ffi::reject_completion(
                &state.host,
                completion as *const NemoRelayNativeAsyncCompletion,
                &error,
            );
        }
        unsafe {
            ffi::release_next(&state.host, next as *const NemoRelayNativeAsyncNext);
            ffi::release_completion(
                &state.host,
                completion as *const NemoRelayNativeAsyncCompletion,
            );
        }
        return NemoRelayNativeAsyncCallbackState::Pending as u32;
    };
    let request = match state
        .runtime
        .decode_request(inbound, &invocation.request, false)
    {
        Ok(request) => request,
        Err(error) => {
            let _ = ffi::reject_completion(
                &state.host,
                completion as *const NemoRelayNativeAsyncCompletion,
                &error,
            );
            unsafe {
                ffi::release_next(&state.host, next as *const NemoRelayNativeAsyncNext);
                ffi::release_completion(
                    &state.host,
                    completion as *const NemoRelayNativeAsyncCompletion,
                );
            }
            return NemoRelayNativeAsyncCallbackState::Pending as u32;
        }
    };
    let parent = ffi::ParentScope::capture(&state.host.v1);
    let task_state = Arc::clone(&state);
    state.executor.spawn(async move {
        let execution = AssertUnwindSafe(task_state.runtime.execute_buffered(
            inbound,
            request,
            parent.as_ref(),
        ))
        .catch_unwind();
        tokio::pin!(execution);
        let result = tokio::select! {
            biased;
            () = ffi::wait_for_completion_cancellation(&task_state.host, completion) => None,
            result = &mut execution => Some(
                result.unwrap_or_else(|_| Err("Switchyard buffered execution panicked".into()))
            ),
        };

        let completion_ptr = completion as *const NemoRelayNativeAsyncCompletion;
        if let Some(result) = result {
            match result {
                Ok(response) => {
                    let _ = ffi::resolve_completion(&task_state.host, completion_ptr, &response);
                }
                Err(error) => {
                    let _ = ffi::reject_completion(&task_state.host, completion_ptr, &error);
                }
            }
        }
        unsafe {
            ffi::release_next(&task_state.host, next as *const NemoRelayNativeAsyncNext);
            ffi::release_completion(&task_state.host, completion_ptr);
        }
    });
    NemoRelayNativeAsyncCallbackState::Pending as u32
}

unsafe extern "C" fn stream_callback(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    output: *const NemoRelayNativeAsyncStream,
) -> u32 {
    if user_data.is_null() || output.is_null() || next.is_null() {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    }
    let state = unsafe { &*user_data.cast::<Arc<CallbackState>>() }.clone();
    let invocation = ffi::read_json(&state.host.v1, invocation_json).and_then(|value| {
        serde_json::from_value::<Invocation>(value).map_err(|error| error.to_string())
    });
    let managed_protocol = invocation
        .as_ref()
        .ok()
        .and_then(|invocation| state.runtime.managed_protocol(&invocation.name));
    let parent = managed_protocol.and_then(|_| ffi::ParentScope::capture(&state.host.v1));
    let next = next as usize;
    let output = output as usize;
    let task_state = Arc::clone(&state);
    state.executor.spawn(async move {
        let execution = AssertUnwindSafe(async {
            match invocation {
                Ok(invocation) => {
                    if let Some(inbound) = managed_protocol {
                        match task_state
                            .runtime
                            .decode_request(inbound, &invocation.request, true)
                        {
                            Ok(request) => {
                                task_state
                                    .runtime
                                    .execute_stream(
                                        &task_state.host,
                                        output,
                                        inbound,
                                        request,
                                        parent.as_ref(),
                                    )
                                    .await
                            }
                            Err(error) => Err(error),
                        }
                    } else {
                        ffi::invoke_next_stream(&task_state.host, next, output, &invocation.request)
                            .await
                    }
                }
                Err(error) => Err(format!("invalid Relay LLM stream invocation: {error}")),
            }
        })
        .catch_unwind();
        tokio::pin!(execution);
        let result = tokio::select! {
            biased;
            () = ffi::wait_for_stream_cancellation(&task_state.host, output) => None,
            result = &mut execution => Some(
                result.unwrap_or_else(|_| Err("Switchyard streaming execution panicked".into()))
            ),
        };

        match result {
            Some(Ok(())) => {
                let _ = ffi::finish_stream(
                    &task_state.host,
                    output as *const NemoRelayNativeAsyncStream,
                );
            }
            Some(Err(error)) => {
                let _ = ffi::reject_stream(&task_state.host, output, &error).await;
            }
            None => {}
        }
        unsafe {
            ffi::release_next(&task_state.host, next as *const NemoRelayNativeAsyncNext);
            ffi::release_stream(
                &task_state.host,
                output as *const NemoRelayNativeAsyncStream,
            );
        }
    });
    NemoRelayNativeAsyncCallbackState::Pending as u32
}

nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_register_plugin, SwitchyardPlugin::default);

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn version_one_service_config_gets_a_migration_error_before_v2_deserialization() {
        let value = json!({
            "version": 1,
            "service_url": "http://127.0.0.1:8080",
            "health_endpoint": "/healthz"
        });
        let plugin_config = value.as_object().unwrap();

        let error = parse_config(plugin_config)
            .err()
            .expect("version one must be rejected");
        assert!(error.contains("version 1 used switchyard-server"));
        assert!(error.contains("migrate to version = 2"));
        assert!(!error.contains("unknown field"));
    }

    #[test]
    fn version_must_be_an_integer() {
        let value = json!({"version": "2"});
        let plugin_config = value.as_object().unwrap();

        let error = parse_config(plugin_config)
            .err()
            .expect("non-integer versions must be rejected");
        assert_eq!(
            error,
            "invalid Switchyard configuration: version must be the integer 2"
        );
    }
}
