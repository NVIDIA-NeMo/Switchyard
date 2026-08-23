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
use crate::runtime::{RoutingMark, SwitchyardRuntime};

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
                let execution = runtime.execute_stream(inbound, decoded).await;
                emit_marks(&plugin_runtime, execution.marks);
                execution
                    .result
                    .map(|stream| Box::pin(stream) as LlmJsonAsyncStream)
            }
        },
    )
}

fn parse_config(plugin_config: &Map<String, Json>) -> Result<SwitchyardConfig, String> {
    serde_json::from_value(Json::Object(plugin_config.clone()))
        .map_err(|error| format!("invalid Switchyard configuration: {error}"))
}

fn emit_marks(runtime: &PluginRuntime, marks: Vec<RoutingMark>) {
    for mark in marks {
        if let Err(error) = runtime.emit_mark(&mark.name, Some(&mark.data), Some(&mark.metadata)) {
            eprintln!("Switchyard could not emit routing mark {:?}: {error}", mark.name);
        }
    }
}

nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_register_plugin, SwitchyardPlugin::default);

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn plugin_configuration_requires_a_deployment_path() {
        let config = json!({"priority": 0});
        let error = parse_config(config.as_object().unwrap()).unwrap_err();
        assert!(error.contains("deployment_path"));
    }
}
