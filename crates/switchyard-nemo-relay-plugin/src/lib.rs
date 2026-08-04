// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod client;
mod config;
mod executor;
mod runtime;
mod translation;

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures_util::FutureExt;
use nemo_relay_plugin::{
    ConfigDiagnostic, DiagnosticLevel, Json, LlmJsonStream, LlmNext, LlmRequest, LlmStreamNext,
    NativePlugin, PluginContext, PluginRuntime,
};
use serde_json::Map;
use tokio::task::AbortHandle;

use crate::config::SwitchyardConfig;
use crate::executor::PluginExecutor;
use crate::runtime::{RoutingMark, StreamMessage, SwitchyardRuntime};

struct CallbackState {
    runtime: Arc<SwitchyardRuntime>,
    executor: PluginExecutor,
    relay: PluginRuntime,
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
        let config = parse_config(plugin_config)?;
        let priority = config.priority;
        let state = Arc::new(CallbackState {
            runtime: Arc::new(SwitchyardRuntime::new(config)?),
            executor: PluginExecutor::new()?,
            relay: ctx.runtime(),
        });

        let buffered = Arc::clone(&state);
        ctx.register_llm_execution_intercept(
            "switchyard.run_stream.buffered",
            priority,
            move |name, request, next| execute_buffered(&buffered, name, request, next),
        )?;
        ctx.register_llm_stream_execution_intercept(
            "switchyard.run_stream.streaming",
            priority,
            move |name, request, next| execute_stream(&state, name, request, next),
        )?;
        Ok(())
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

fn execute_buffered(
    state: &CallbackState,
    name: &str,
    request: LlmRequest,
    next: LlmNext<'_>,
) -> nemo_relay_plugin::Result<Json> {
    let Some(inbound) = state.runtime.managed_protocol(name) else {
        return next.call(request);
    };
    let request = state.runtime.decode_request(inbound, &request, false)?;
    let runtime = Arc::clone(&state.runtime);
    let (result, marks) = state.executor.run(async move {
        let mut marks = Vec::new();
        let result = AssertUnwindSafe(runtime.execute_buffered(inbound, request, &mut marks))
            .catch_unwind()
            .await
            .unwrap_or_else(|_| Err("Switchyard buffered execution panicked".into()));
        (result, marks)
    })?;
    emit_marks(&state.relay, marks);
    result
}

fn execute_stream(
    state: &CallbackState,
    name: &str,
    request: LlmRequest,
    next: LlmStreamNext<'_>,
) -> nemo_relay_plugin::Result<LlmJsonStream> {
    let Some(inbound) = state.runtime.managed_protocol(name) else {
        return Ok(Box::new(next.call(request)?));
    };
    let request = state.runtime.decode_request(inbound, &request, true)?;
    let (sender, receiver) = async_channel::bounded(32);
    let runtime = Arc::clone(&state.runtime);
    let task = state.executor.spawn(async move {
        let result = AssertUnwindSafe(runtime.execute_stream(inbound, request, &sender))
            .catch_unwind()
            .await
            .unwrap_or_else(|_| Err("Switchyard streaming execution panicked".into()));
        if let Err(error) = result {
            let _ = sender.send(StreamMessage::Error(error)).await;
        }
    });
    Ok(Box::new(SwitchyardStream {
        receiver,
        task: Some(task),
        relay: state.relay.clone(),
        finished: false,
    }))
}

fn emit_marks(relay: &PluginRuntime, marks: Vec<RoutingMark>) {
    for mark in marks {
        emit_mark(relay, mark);
    }
}

fn emit_mark(relay: &PluginRuntime, mark: RoutingMark) {
    if let Err(error) = relay.emit_mark(&mark.name, Some(&mark.data), Some(&mark.metadata)) {
        eprintln!(
            "Switchyard could not emit routing mark {:?}: {error}",
            mark.name
        );
    }
}

struct SwitchyardStream {
    receiver: async_channel::Receiver<StreamMessage>,
    task: Option<AbortHandle>,
    relay: PluginRuntime,
    finished: bool,
}

impl Iterator for SwitchyardStream {
    type Item = nemo_relay_plugin::Result<Json>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.receiver.recv_blocking() {
                Ok(StreamMessage::Mark(mark)) => emit_mark(&self.relay, mark),
                Ok(StreamMessage::Event(event)) => return Some(Ok(event)),
                Ok(StreamMessage::Error(error)) => {
                    self.finished = true;
                    self.task.take();
                    return Some(Err(error));
                }
                Err(_) => {
                    self.finished = true;
                    self.task.take();
                    return None;
                }
            }
        }
    }
}

impl Drop for SwitchyardStream {
    fn drop(&mut self) {
        self.receiver.close();
        if !self.finished
            && let Some(task) = self.task.take()
        {
            task.abort();
        }
    }
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
