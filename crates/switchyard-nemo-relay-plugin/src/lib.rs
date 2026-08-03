// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod config;
mod runtime;
mod translation;

use std::sync::Arc;

use nemo_relay_plugin::{ConfigDiagnostic, DiagnosticLevel, Json, NativePlugin, PluginContext};
use serde_json::Map;

use crate::config::SwitchyardConfig;
use crate::runtime::SwitchyardRuntime;

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
        let config = parse_config(plugin_config)?;
        let priority = config.priority;
        let runtime = Arc::new(SwitchyardRuntime::new(config, ctx.runtime())?);

        let buffered_runtime = Arc::clone(&runtime);
        ctx.register_async_llm_execution_v2(
            "switchyard.run_stream.buffered",
            priority,
            move |name, request, continuation| {
                let runtime = Arc::clone(&buffered_runtime);
                async move { runtime.execute_buffered(name, request, continuation).await }
            },
        )?;

        ctx.register_async_llm_stream_execution_v2(
            "switchyard.run_stream.streaming",
            priority,
            move |name, request, continuation| {
                let runtime = Arc::clone(&runtime);
                async move { runtime.execute_stream(name, request, continuation).await }
            },
        )?;
        Ok(())
    }
}

fn parse_config(plugin_config: &Map<String, Json>) -> Result<SwitchyardConfig, String> {
    serde_json::from_value(Json::Object(plugin_config.clone()))
        .map_err(|error| format!("invalid Switchyard configuration: {error}"))
}

nemo_relay_plugin::nemo_relay_plugin_v2!(nemo_relay_register_plugin, SwitchyardPlugin::default);
