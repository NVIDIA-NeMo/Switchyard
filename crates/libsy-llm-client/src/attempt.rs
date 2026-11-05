// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accounting retained until an HTTP attempt's response is consumed or abandoned.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use http::HeaderMap;
use switchyard_protocol::{
    LlmClientError, LlmResponseChunk, LlmResponseStream, LlmResponseStreamEvent, ModelId, Usage,
    WireFormat,
};
use tracing::Span;

use crate::observability::{llm_client_error_type, record_gen_ai_usage};

pub(crate) struct AttemptGuard {
    span: Span,
    usage: Usage,
    accounting: bool,
    final_usage: bool,
    model_finished: bool,
    finished: bool,
}

impl AttemptGuard {
    pub(crate) fn new(
        model: &ModelId,
        format: WireFormat,
        ordinal: u64,
        max_attempts: u64,
        accounting: bool,
    ) -> Self {
        if accounting {
            Span::current().record("switchyard.accounting.unit", "summary");
            Span::current().record("switchyard.accounting.attempts", ordinal);
        }
        let span = tracing::debug_span!(target: "libsy", "libsy.upstream_attempt",
            model = %model, gen_ai.request.model = %model, wire_format = %format,
            attempt = ordinal, max_attempts, retry = ordinal > 1,
            openinference.span.kind = "CHAIN",
            outcome = tracing::field::Empty, status_code = tracing::field::Empty,
            will_retry = tracing::field::Empty, retry_delay_ms = tracing::field::Empty,
            error.type = tracing::field::Empty, otel.status_code = tracing::field::Empty,
            gen_ai.response.id = tracing::field::Empty, gen_ai.response.model = tracing::field::Empty,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
            gen_ai.usage.cache_creation_1h.input_tokens = tracing::field::Empty,
            gen_ai.usage.reasoning.output_tokens = tracing::field::Empty,
            switchyard.usage.total_tokens = tracing::field::Empty,
            switchyard.usage.provider_cost_usd = tracing::field::Empty,
            switchyard.accounting.charge_source = tracing::field::Empty,
            switchyard.accounting.usage_status = tracing::field::Empty,
            switchyard.accounting.unit = if accounting { "completion_attempt" } else { "diagnostic" },
        );
        Self {
            span,
            usage: Usage::default(),
            accounting,
            final_usage: false,
            model_finished: false,
            finished: false,
        }
    }

    pub(crate) fn span(&self) -> &Span {
        &self.span
    }

    pub(crate) fn headers(&mut self, headers: &HeaderMap) {
        if let Some(cost) = headers
            .get("x-litellm-response-cost")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
        {
            self.usage.provider_cost = Some(cost);
            self.span
                .record("switchyard.accounting.charge_source", "header");
        }
    }

    pub(crate) fn record_usage(&mut self, usage: &Usage, final_report: bool) {
        macro_rules! update {
            ($($field:ident),+ $(,)?) => {$(if usage.$field.is_some() { self.usage.$field = usage.$field; })+};
        }
        update!(input_tokens, output_tokens, total_tokens, reasoning_tokens);
        if let Some(cache) = &usage.cache {
            let target = self.usage.cache.get_or_insert_with(Default::default);
            if cache.cached_input_tokens.is_some() {
                target.cached_input_tokens = cache.cached_input_tokens;
            }
            if cache.cache_creation_input_tokens.is_some() {
                target.cache_creation_input_tokens = cache.cache_creation_input_tokens;
            }
            if cache.cache_creation_1h_input_tokens.is_some() {
                target.cache_creation_1h_input_tokens = cache.cache_creation_1h_input_tokens;
            }
        }
        if let Some(cost) = usage
            .provider_cost
            .filter(|value| value.is_finite() && *value >= 0.0)
        {
            self.usage.provider_cost = Some(cost);
            self.span
                .record("switchyard.accounting.charge_source", "provider");
        }
        self.final_usage |= final_report;
    }

    pub(crate) fn response(&mut self, body: &serde_json::Value, format: WireFormat) {
        if let Some(id) = body.get("id").and_then(serde_json::Value::as_str) {
            self.span.record("gen_ai.response.id", id);
        }
        if let Some(model) = body.get("model").and_then(serde_json::Value::as_str) {
            self.span.record("gen_ai.response.model", model);
        }
        self.record_usage(&switchyard_translation::response_usage(body, format), true);
    }

    pub(crate) fn observe(
        &mut self,
        item: &Result<LlmResponseStreamEvent, LlmClientError>,
    ) -> bool {
        let event = match item {
            Ok(event) => event,
            Err(error) => {
                self.error(error);
                return true;
            }
        };
        self.observe_usage(event);
        for chunk in event.normalized() {
            match chunk {
                LlmResponseChunk::MessageStart { id, model } => {
                    if let Some(id) = id {
                        self.span.record("gen_ai.response.id", id.as_str());
                    }
                    if let Some(model) = model {
                        self.span.record("gen_ai.response.model", model.as_str());
                    }
                }
                LlmResponseChunk::MessageStop { .. } => self.model_finished = true,
                LlmResponseChunk::StreamError { .. } => {
                    self.finish("error", Some("provider_stream"));
                    return true;
                }
                LlmResponseChunk::DecodeError { .. } => {
                    self.finish("error", Some("response_translation"));
                    return true;
                }
                _ => (),
            }
        }
        false
    }

    pub(crate) fn observe_usage(&mut self, event: &LlmResponseStreamEvent) {
        // One provider event can contain both its final usage and a terminal error.
        let terminal_report = event.preservation().is_some_and(|raw| {
            matches!(
                raw.raw().get("type").and_then(serde_json::Value::as_str),
                Some("response.completed" | "response.failed" | "response.incomplete")
            )
        });
        for chunk in event.normalized() {
            if let LlmResponseChunk::Usage(usage) = chunk {
                self.record_usage(usage, terminal_report);
            }
        }
    }

    pub(crate) fn error(&mut self, error: &LlmClientError) {
        self.finish("error", Some(llm_client_error_type(error).as_ref()));
    }

    pub(crate) fn success(&mut self) {
        self.finish("ok", None);
    }

    fn finish(&mut self, outcome: &str, error: Option<&str>) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.span.record("outcome", outcome);
        self.span.record(
            "otel.status_code",
            if outcome == "ok" { "OK" } else { "ERROR" },
        );
        if let Some(error) = error {
            self.span.record("error.type", error);
        }
        if !self.accounting {
            return;
        }
        record_gen_ai_usage(&self.span, &self.usage);
        if let Some(total) = self.usage.total_tokens {
            self.span.record("switchyard.usage.total_tokens", total);
        }
        if let Some(cost) = self.usage.provider_cost {
            self.span.record("switchyard.usage.provider_cost_usd", cost);
        }
        let known = self.usage.input_tokens.is_some()
            || self.usage.output_tokens.is_some()
            || self.usage.cached_input_tokens().is_some()
            || self.usage.cache_creation_input_tokens().is_some()
            || self
                .usage
                .cache
                .as_ref()
                .is_some_and(|cache| cache.cache_creation_1h_input_tokens.is_some())
            || self.usage.total_tokens.is_some()
            || self.usage.reasoning_tokens.is_some();
        let complete = self.final_usage
            && self.usage.input_tokens.is_some()
            && self.usage.output_tokens.is_some();
        self.span.record(
            "switchyard.accounting.usage_status",
            if complete {
                "complete"
            } else if known {
                "partial"
            } else {
                "missing"
            },
        );
    }

    pub(crate) fn wrap(self, stream: LlmResponseStream) -> LlmResponseStream {
        Box::pin(AttemptStream {
            inner: stream,
            guard: Some(self),
            dispatch: tracing::dispatcher::get_default(Clone::clone),
        })
    }
}

impl Drop for AttemptGuard {
    fn drop(&mut self) {
        self.finish(
            if self.model_finished {
                "ok"
            } else {
                "cancelled"
            },
            None,
        );
    }
}

struct AttemptStream {
    inner: LlmResponseStream,
    guard: Option<AttemptGuard>,
    dispatch: tracing::Dispatch,
}

impl Stream for AttemptStream {
    type Item = Result<LlmResponseStreamEvent, LlmClientError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let Self {
            inner,
            guard,
            dispatch,
        } = self.get_mut();
        tracing::dispatcher::with_default(dispatch, || match inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(item)) => {
                if guard.as_mut().is_some_and(|guard| guard.observe(&item)) {
                    guard.take();
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                if let Some(mut guard) = guard.take() {
                    guard.final_usage |= guard.model_finished;
                    guard.success();
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        })
    }
}

impl Drop for AttemptStream {
    fn drop(&mut self) {
        let Self {
            inner,
            guard,
            dispatch,
        } = self;
        // Registry parent references can close recursively when the last child
        // drops. Keep that close on the subscriber that owns the response.
        tracing::dispatcher::with_default(dispatch, || {
            drop(guard.take());
            drop(std::mem::replace(inner, Box::pin(futures::stream::empty())));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{StreamExt, stream};
    use serde_json::{Map, Value};
    use std::sync::{Arc, Mutex};
    use tracing::Subscriber;
    use tracing::instrument::WithSubscriber;
    use tracing_subscriber::{
        Layer,
        layer::{Context as LayerContext, SubscriberExt},
        registry::LookupSpan,
    };

    type Records = Arc<Mutex<Vec<Map<String, Value>>>>;
    struct Collector(Records);

    impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Collector {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            id: &tracing::Id,
            ctx: LayerContext<'_, S>,
        ) {
            let mut fields = Map::<String, Value>::new();
            attrs.record(
                &mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                    let text = format!("{value:?}");
                    fields.insert(
                        field.name().into(),
                        serde_json::from_str(&text).unwrap_or(Value::String(text)),
                    );
                },
            );
            ctx.span(id).unwrap().extensions_mut().insert(fields);
        }
        fn on_record(
            &self,
            id: &tracing::Id,
            values: &tracing::span::Record<'_>,
            ctx: LayerContext<'_, S>,
        ) {
            let span = ctx.span(id).unwrap();
            let mut extensions = span.extensions_mut();
            let fields = extensions.get_mut::<Map<String, Value>>().unwrap();
            values.record(
                &mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                    let text = format!("{value:?}");
                    fields.insert(
                        field.name().into(),
                        serde_json::from_str(&text).unwrap_or(Value::String(text)),
                    );
                },
            );
        }
        fn on_close(&self, id: tracing::Id, ctx: LayerContext<'_, S>) {
            let fields = ctx
                .span(&id)
                .unwrap()
                .extensions_mut()
                .remove::<Map<String, Value>>()
                .unwrap();
            self.0.lock().unwrap().push(fields);
        }
    }

    fn usage(input: Option<u64>, output: Option<u64>) -> LlmResponseChunk {
        LlmResponseChunk::Usage(Usage {
            input_tokens: input,
            output_tokens: output,
            ..Usage::default()
        })
    }

    async fn observe(
        events: Vec<LlmResponseStreamEvent>,
        take: Option<usize>,
    ) -> Map<String, Value> {
        observe_results(events.into_iter().map(Ok).collect(), take).await
    }

    async fn observe_results(
        events: Vec<Result<LlmResponseStreamEvent, LlmClientError>>,
        take: Option<usize>,
    ) -> Map<String, Value> {
        let records = Records::default();
        let subscriber = tracing_subscriber::registry().with(Collector(records.clone()));
        async {
            let guard =
                AttemptGuard::new(&ModelId::from("model"), WireFormat::OpenAiChat, 1, 1, true);
            let mut response = guard.wrap(stream::iter(events).boxed());
            if let Some(take) = take {
                for _ in 0..take {
                    response.next().await;
                }
            } else {
                while response.next().await.is_some() {}
            }
            drop(response);
        }
        .with_subscriber(subscriber)
        .await;
        let records = records.lock().unwrap();
        assert_eq!(records.len(), 1);
        records[0].clone()
    }

    #[tokio::test]
    async fn usage_in_a_terminal_error_event_survives_regardless_of_chunk_order() {
        let fields = observe(
            vec![LlmResponseStreamEvent::new(vec![
                LlmResponseChunk::StreamError {
                    message: "private failure".into(),
                },
                usage(Some(11), Some(3)),
            ])],
            None,
        )
        .await;
        assert_eq!(fields["outcome"], "error");
        assert_eq!(fields["gen_ai.usage.input_tokens"], 11);
        assert_eq!(fields["gen_ai.usage.output_tokens"], 3);
        assert!(!fields.contains_key("error"));
    }

    #[tokio::test]
    async fn cumulative_snapshots_preserve_missing_fields_and_final_corrections() {
        let fields = observe(
            vec![
                usage(Some(11), Some(2)).into(),
                usage(None, Some(3)).into(),
                usage(Some(9), Some(1)).into(),
                LlmResponseChunk::MessageStop {
                    reason: Some("stop".into()),
                }
                .into(),
            ],
            None,
        )
        .await;
        assert_eq!(fields["gen_ai.usage.input_tokens"], 9);
        assert_eq!(fields["gen_ai.usage.output_tokens"], 1);
        assert_eq!(fields["switchyard.accounting.usage_status"], "complete");
    }

    #[tokio::test]
    async fn drop_after_finish_is_success_with_partial_accounting() {
        let fields = observe(
            vec![
                usage(Some(11), Some(3)).into(),
                LlmResponseChunk::MessageStop {
                    reason: Some("stop".into()),
                }
                .into(),
                usage(Some(12), Some(4)).into(),
            ],
            Some(2),
        )
        .await;
        assert_eq!(fields["outcome"], "ok");
        assert_eq!(fields["gen_ai.usage.input_tokens"], 11);
        assert_eq!(fields["switchyard.accounting.usage_status"], "partial");
    }

    #[tokio::test]
    async fn missing_usage_is_distinct_from_explicit_zero() {
        for (counts, status) in [(None, "missing"), (Some(0), "complete")] {
            let fields = observe(
                vec![
                    LlmResponseChunk::MessageStop {
                        reason: Some("stop".into()),
                    }
                    .into(),
                    usage(counts, counts).into(),
                ],
                None,
            )
            .await;
            assert_eq!(fields["switchyard.accounting.usage_status"], status);
            assert_eq!(
                fields
                    .get("gen_ai.usage.input_tokens")
                    .and_then(Value::as_u64),
                counts
            );
        }
    }

    #[tokio::test]
    async fn trailing_usage_after_model_finish_is_not_lost_or_added_twice() {
        let fields = observe(
            vec![
                usage(Some(11), Some(3)).into(),
                LlmResponseChunk::MessageStop {
                    reason: Some("stop".into()),
                }
                .into(),
                usage(Some(12), Some(4)).into(),
                usage(Some(12), Some(4)).into(),
            ],
            None,
        )
        .await;
        assert_eq!(fields["gen_ai.usage.input_tokens"], 12);
        assert_eq!(fields["gen_ai.usage.output_tokens"], 4);
        assert_eq!(fields["switchyard.accounting.usage_status"], "complete");
    }

    #[tokio::test]
    async fn timeout_after_usage_retains_partial_consumption_and_error() {
        let fields = observe_results(
            vec![
                Ok(usage(Some(11), Some(3)).into()),
                Err(LlmClientError::Timeout {
                    source: Box::new(std::io::Error::from(std::io::ErrorKind::TimedOut)),
                }),
            ],
            None,
        )
        .await;
        assert_eq!(fields["gen_ai.usage.input_tokens"], 11);
        assert_eq!(fields["outcome"], "error");
        assert_eq!(fields["error.type"], "timeout");
        assert_eq!(fields["switchyard.accounting.usage_status"], "partial");
    }
}
