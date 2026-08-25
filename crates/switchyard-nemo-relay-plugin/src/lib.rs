// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod config;
mod runtime;
mod translation;

use std::sync::Arc;

use nemo_relay_plugin::{
    ConfigDiagnostic, DiagnosticLevel, Json, LlmJsonAsyncStream, NativePlugin, PluginContext,
    PluginRuntime,
};
use serde_json::Map;

use crate::config::{SwitchyardConfig, protocol_from_call};
use crate::runtime::{SwitchyardRuntime, emit_mark, emit_marks};

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
        match parse_config(plugin_config).and_then(SwitchyardRuntime::new) {
            Ok(_) => Vec::new(),
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
        let config = parse_config(plugin_config)?;
        let priority = config.priority;
        let runtime = Arc::new(SwitchyardRuntime::new(config)?);
        let plugin_runtime = ctx.runtime();
        register_buffered(ctx, priority, Arc::clone(&runtime), plugin_runtime.clone())?;
        register_stream(ctx, priority, runtime, plugin_runtime)?;
        Ok(())
    }
}

fn register_buffered(
    ctx: &mut PluginContext<'_>,
    priority: i32,
    runtime: Arc<SwitchyardRuntime>,
    plugin_runtime: PluginRuntime,
) -> Result<(), String> {
    ctx.register_llm_execution_intercept(
        "switchyard.runner.buffered",
        priority,
        move |name, request, next| {
            let runtime = Arc::clone(&runtime);
            let plugin_runtime = plugin_runtime.clone();
            async move {
                let Some(inbound) = protocol_from_call(&name) else {
                    return next.call(request).await;
                };
                let decoded = runtime.decode_request(inbound, &request, false)?;
                if !runtime.manages(&decoded) {
                    return next.call(request).await;
                }
                let execution = runtime.execute_buffered(inbound, decoded).await;
                emit_marks(&plugin_runtime, execution.marks);
                execution.result
            }
        },
    )
}

fn register_stream(
    ctx: &mut PluginContext<'_>,
    priority: i32,
    runtime: Arc<SwitchyardRuntime>,
    plugin_runtime: PluginRuntime,
) -> Result<(), String> {
    ctx.register_llm_stream_execution_intercept(
        "switchyard.runner.streaming",
        priority,
        move |name, request, next| {
            let runtime = Arc::clone(&runtime);
            let plugin_runtime = plugin_runtime.clone();
            async move {
                let Some(inbound) = protocol_from_call(&name) else {
                    return next.call(request).await;
                };
                let decoded = runtime.decode_request(inbound, &request, true)?;
                if !runtime.manages(&decoded) {
                    return next.call(request).await;
                }
                let stream_plugin_runtime = plugin_runtime.clone();
                let execution = runtime
                    .execute_stream(
                        inbound,
                        decoded,
                        Arc::new(move |mark| emit_mark(&stream_plugin_runtime, mark)),
                    )
                    .await;
                emit_marks(&plugin_runtime, execution.marks);
                execution
                    .result
                    .map(|stream| Box::pin(stream) as LlmJsonAsyncStream)
            }
        },
    )
}

fn parse_config(plugin_config: &Map<String, Json>) -> Result<SwitchyardConfig, String> {
    let mut config = plugin_config.clone();
    config.remove("executor");
    serde_json::from_value(Json::Object(config))
        .map_err(|error| format!("invalid Switchyard configuration: {error}"))
}

nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_register_plugin, SwitchyardPlugin::default);

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn plugin_configuration_requires_a_switchyard_source() {
        let config = json!({"priority": 0});
        assert!(matches!(
            parse_config(config.as_object().unwrap()).and_then(SwitchyardRuntime::new),
            Err(error) if error.contains("switchyard_config_path or switchyard_config")
        ));
    }

    #[test]
    fn plugin_configuration_ignores_the_sdk_executor_override() {
        let config = json!({
            "executor": {"worker_threads": 4},
            "switchyard_config_path": "routes.toml"
        });
        assert!(parse_config(config.as_object().unwrap()).is_ok());
    }
}
