// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod config;
mod ffi;
mod runtime;
mod translation;

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, OnceLock};

use nemo_relay_plugin::{
    ConfigDiagnostic, DiagnosticLevel, Json, NativePlugin, NemoRelayNativeAsyncCallbackState,
    NemoRelayNativeAsyncCompletion, NemoRelayNativeAsyncNext, NemoRelayNativeAsyncStream,
    NemoRelayNativeHostApiV4, NemoRelayNativeString, NemoRelayStatus, PluginContext,
};
use serde_json::Map;

use crate::config::SwitchyardConfig;
use crate::runtime::{Invocation, SwitchyardRuntime};

static HOST: OnceLock<NemoRelayNativeHostApiV4> = OnceLock::new();

pub(crate) fn host() -> &'static NemoRelayNativeHostApiV4 {
    HOST.get()
        .expect("Switchyard callback invoked before plugin registration")
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
        let host = *ctx
            .host_api_v4()
            .ok_or_else(|| "Switchyard requires Relay native plugin C API v2".to_string())?;
        if let Some(registered) = HOST.get() {
            if registered.v3.v1.abi_version != host.v3.v1.abi_version {
                return Err("Switchyard was initialized with a different Relay host ABI".into());
            }
        } else {
            HOST.set(host)
                .map_err(|_| "failed to retain the Relay native API v2 host table".to_string())?;
        }

        let config = parse_config(plugin_config)?;
        let priority = config.priority;
        let runtime = Arc::new(SwitchyardRuntime::new(config, ctx.runtime())?);

        let buffered_state = Box::into_raw(Box::new(Arc::clone(&runtime))).cast::<c_void>();
        let status = unsafe {
            ctx.register_async_llm_execution_v2_raw(
                "switchyard.run_stream.buffered",
                priority,
                buffered_callback,
                buffered_state,
                Some(free_runtime),
            )
        };
        if status != NemoRelayStatus::Ok {
            return Err(format!(
                "failed to register Switchyard buffered execution: {status:?}"
            ));
        }

        let stream_state = Box::into_raw(Box::new(runtime)).cast::<c_void>();
        let status = unsafe {
            ctx.register_async_llm_stream_execution_v2_raw(
                "switchyard.run_stream.streaming",
                priority,
                stream_callback,
                stream_state,
                Some(free_runtime),
            )
        };
        if status != NemoRelayStatus::Ok {
            return Err(format!(
                "failed to register Switchyard streaming execution: {status:?}"
            ));
        }
        Ok(())
    }
}

fn parse_config(plugin_config: &Map<String, Json>) -> Result<SwitchyardConfig, String> {
    serde_json::from_value(Json::Object(plugin_config.clone()))
        .map_err(|error| format!("invalid Switchyard configuration: {error}"))
}

unsafe extern "C" fn free_runtime(user_data: *mut c_void) {
    if !user_data.is_null() {
        unsafe { drop(Box::from_raw(user_data.cast::<Arc<SwitchyardRuntime>>())) };
    }
}

unsafe extern "C" fn buffered_callback(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> u32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let runtime = unsafe { &*user_data.cast::<Arc<SwitchyardRuntime>>() };
        let invocation = ffi::read_json(&host().v3.v1, invocation_json).and_then(|value| {
            serde_json::from_value::<Invocation>(value).map_err(|e| e.to_string())
        });
        match invocation {
            Ok(invocation) => {
                match futures::executor::block_on(runtime.execute_buffered(
                    invocation,
                    host(),
                    next,
                )) {
                    Ok(response) => {
                        let _ = ffi::resolve_completion(host(), completion, &response);
                    }
                    Err(error) => {
                        let _ = ffi::reject_completion(host(), completion, &error);
                    }
                }
            }
            Err(error) => {
                let _ = ffi::reject_completion(
                    host(),
                    completion,
                    &format!("invalid Relay LLM invocation: {error}"),
                );
            }
        }
    }));
    if result.is_err() {
        let _ =
            ffi::reject_completion(host(), completion, "Switchyard buffered execution panicked");
    }
    unsafe { ffi::release_next(host(), next) };
    NemoRelayNativeAsyncCallbackState::Complete as u32
}

unsafe extern "C" fn stream_callback(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    output: *const NemoRelayNativeAsyncStream,
) -> u32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let runtime = unsafe { &*user_data.cast::<Arc<SwitchyardRuntime>>() };
        let invocation = ffi::read_json(&host().v3.v1, invocation_json).and_then(|value| {
            serde_json::from_value::<Invocation>(value).map_err(|e| e.to_string())
        });
        let result = match invocation {
            Ok(invocation) => futures::executor::block_on(runtime.execute_stream(
                invocation,
                host(),
                next,
                output,
            )),
            Err(error) => Err(format!("invalid Relay LLM stream invocation: {error}")),
        };
        if let Err(error) = result {
            let _ = ffi::reject_stream(host(), output, &error);
        }
    }));
    if result.is_err() {
        let _ = ffi::reject_stream(host(), output, "Switchyard streaming execution panicked");
    }
    unsafe {
        ffi::release_next(host(), next);
        ffi::release_stream(host(), output);
    }
    NemoRelayNativeAsyncCallbackState::Complete as u32
}

nemo_relay_plugin::nemo_relay_plugin_v2!(nemo_relay_register_plugin, SwitchyardPlugin::default);
