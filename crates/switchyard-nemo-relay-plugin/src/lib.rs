// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod client;
mod config;
mod runtime;
mod translation;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use nemo_relay_plugin::{
    ConfigDiagnostic, DiagnosticLevel, Json, LlmJsonAsyncStream, NativePlugin, PluginContext,
    PluginRuntime,
};
use serde_json::Map;

use crate::config::SwitchyardConfig;
use crate::runtime::{RoutingMark, StreamMessage, SwitchyardRuntime};

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
        "switchyard.run_stream.buffered",
        priority,
        move |name, request, next| {
            let runtime = Arc::clone(&runtime);
            let plugin_runtime = plugin_runtime.clone();
            async move {
                let Some(inbound) = runtime.managed_protocol(&name) else {
                    return next.call(request).await;
                };
                let request = runtime.decode_request(inbound, &request, false)?;
                let mut marks = Vec::new();
                let response = runtime.execute_buffered(inbound, request, &mut marks).await;
                emit_marks(&plugin_runtime, marks);
                response
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
        "switchyard.run_stream.streaming",
        priority,
        move |name, request, next| {
            let runtime = Arc::clone(&runtime);
            let plugin_runtime = plugin_runtime.clone();
            async move {
                let Some(inbound) = runtime.managed_protocol(&name) else {
                    return next.call(request).await;
                };
                let request = runtime.decode_request(inbound, &request, true)?;
                Ok(Box::pin(ManagedStream::new(
                    runtime,
                    plugin_runtime,
                    inbound,
                    request,
                )) as LlmJsonAsyncStream)
            }
        },
    )
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

fn emit_marks(runtime: &PluginRuntime, marks: Vec<RoutingMark>) {
    for mark in marks {
        emit_mark(runtime, mark);
    }
}

fn emit_mark(runtime: &PluginRuntime, mark: RoutingMark) {
    if let Err(error) = runtime.emit_mark(&mark.name, Some(&mark.data), Some(&mark.metadata)) {
        eprintln!(
            "Switchyard could not emit routing mark {:?}: {error}",
            mark.name
        );
    }
}

type StreamExecution = Pin<Box<dyn Future<Output = Result<(), String>> + Send>>;

struct ManagedStream {
    execution: Option<StreamExecution>,
    messages: Pin<Box<async_channel::Receiver<StreamMessage>>>,
    emit_mark: Arc<dyn Fn(RoutingMark) + Send + Sync>,
    terminal_error: Option<String>,
}

impl ManagedStream {
    fn new(
        runtime: Arc<SwitchyardRuntime>,
        plugin_runtime: PluginRuntime,
        inbound: switchyard_protocol::WireFormat,
        request: switchyard_protocol::Request,
    ) -> Self {
        let (sender, messages) = async_channel::bounded(32);
        let execution = async move { runtime.execute_stream(inbound, request, &sender).await };
        let emit_mark = Arc::new(move |mark| emit_mark(&plugin_runtime, mark));
        Self {
            execution: Some(Box::pin(execution)),
            messages: Box::pin(messages),
            emit_mark,
            terminal_error: None,
        }
    }
}

impl futures_util::Stream for ManagedStream {
    type Item = Result<Json, String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(execution) = self.execution.as_mut() {
            match execution.as_mut().poll(cx) {
                Poll::Ready(Ok(())) => self.execution = None,
                Poll::Ready(Err(error)) => {
                    self.execution = None;
                    self.terminal_error = Some(error);
                }
                Poll::Pending => {}
            }
        }

        loop {
            match self.messages.as_mut().poll_next(cx) {
                Poll::Ready(Some(StreamMessage::Mark(mark))) => (self.emit_mark)(mark),
                Poll::Ready(Some(StreamMessage::Event(event))) => {
                    return Poll::Ready(Some(Ok(event)));
                }
                Poll::Ready(None) => {
                    return Poll::Ready(self.terminal_error.take().map(Err));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_register_plugin, SwitchyardPlugin::default);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures_util::StreamExt;
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn managed_stream_delivers_queued_events_before_terminal_error() {
        let (sender, messages) = async_channel::bounded(32);
        let execution = async move {
            sender
                .send(StreamMessage::Event(json!({"id": "committed"})))
                .await
                .expect("queue committed event");
            Err("stream failed after commitment".into())
        };
        let mut stream = ManagedStream {
            execution: Some(Box::pin(execution)),
            messages: Box::pin(messages),
            emit_mark: Arc::new(|_| {}),
            terminal_error: None,
        };

        assert_eq!(stream.next().await, Some(Ok(json!({"id": "committed"}))));
        assert_eq!(
            stream.next().await,
            Some(Err("stream failed after commitment".into()))
        );
        assert_eq!(stream.next().await, None);
    }

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
