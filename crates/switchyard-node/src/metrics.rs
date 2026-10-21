// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics for model calls performed by the JavaScript host.

use std::collections::HashSet;
use std::sync::OnceLock;

use napi_derive::napi;
use opentelemetry::{Context, KeyValue, global};
use parking_lot::Mutex;
use serde::Deserialize;
use switchyard_libsy::metrics::{MetricFilter, Metrics};
use switchyard_protocol::Usage;

use crate::{error, parse};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MetricContext {
    session_id: String,
    agent_id: String,
    agent_turn_idx: i64,
    user_turn_idx: i64,
}

impl MetricContext {
    pub(crate) fn context(&self) -> napi::Result<Context> {
        if self.session_id.is_empty()
            || self.agent_id.is_empty()
            || self.agent_turn_idx < 0
            || self.user_turn_idx < 0
        {
            return Err(error(
                "ERR_INVALID_REQUEST",
                "Expected session/agent IDs and nonnegative turn indices",
            ));
        }
        Ok(switchyard_libsy::metric_context::for_turn(
            &self.session_id,
            &self.agent_id,
            self.agent_turn_idx,
            self.user_turn_idx,
        ))
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum CallKind {
    Answer,
    Decision,
    Request,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    id: String,
    context: MetricContext,
    model: String,
    call_kind: CallKind,
    is_success: bool,
    duration_ms: f64,
    ttft_ms: Option<f64>,
    usage: Option<Usage>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    session_id: Option<String>,
    agent_id: Option<String>,
    agent_turn_idx: Option<i64>,
    user_turn_idx: Option<i64>,
}

impl Query {
    fn filter(&self) -> MetricFilter<'_> {
        MetricFilter {
            session_id: self.session_id.as_deref(),
            agent_id: self.agent_id.as_deref(),
            agent_turn_idx: self.agent_turn_idx,
            user_turn_idx: self.user_turn_idx,
        }
    }
}

struct Store {
    metrics: Metrics,
    // Session records are replayed on extension reload and must contribute once.
    recorded: Mutex<HashSet<(String, String)>>,
}

#[napi]
pub struct NativeMetrics {
    store: &'static Store,
}

#[napi]
impl NativeMetrics {
    #[napi(constructor)]
    pub fn new() -> napi::Result<Self> {
        static STORE: OnceLock<Result<Store, String>> = OnceLock::new();
        let store = STORE
            .get_or_init(|| {
                let metrics =
                    Metrics::with_cardinality_limit(20_000).map_err(|error| error.to_string())?;
                global::set_meter_provider(metrics.meter_provider().clone());
                Ok(Store {
                    metrics,
                    recorded: Mutex::default(),
                })
            })
            .as_ref()
            .map_err(|message| error("ERR_METRICS", message))?;
        Ok(Self { store })
    }

    #[napi]
    pub fn record(&self, observation_json: String) -> napi::Result<bool> {
        let observation: Observation = parse(&observation_json)?;
        let context = observation.context.context()?;
        if observation.id.is_empty()
            || observation.model.is_empty()
            || !observation.duration_ms.is_finite()
            || observation.duration_ms < 0.0
            || observation.ttft_ms.is_some_and(|value| {
                !value.is_finite() || value < 0.0 || value > observation.duration_ms
            })
        {
            return Err(error(
                "ERR_INVALID_REQUEST",
                "Expected an ID, model, and finite nonnegative timing",
            ));
        }
        let mut recorded = self.store.recorded.lock();
        if !recorded.insert((observation.context.session_id.clone(), observation.id)) {
            return Ok(false);
        }
        let _guard = context.attach();
        let kind = match observation.call_kind {
            CallKind::Answer => "answer",
            CallKind::Decision => "decision",
            CallKind::Request => "request",
        };
        if let Some(usage) = &observation.usage {
            switchyard_libsy::metrics::record_usage("composite", &observation.model, kind, usage);
        }
        let attributes = switchyard_libsy::metric_context::attributes(&[
            KeyValue::new("algorithm", "composite"),
            KeyValue::new("selected_model", observation.model),
            KeyValue::new(
                "outcome",
                if observation.is_success {
                    "ok"
                } else {
                    "error"
                },
            ),
        ]);
        let meter = global::meter("switchyard");
        let duration_metric = if matches!(observation.call_kind, CallKind::Decision) {
            "switchyard.classifier_duration_ms"
        } else {
            meter
                .u64_counter("switchyard.inference_requests")
                .build()
                .add(1, &attributes);
            if let Some(ttft) = observation.ttft_ms {
                meter
                    .f64_histogram("switchyard.inference_ttft_ms")
                    .build()
                    .record(ttft, &attributes);
            }
            "switchyard.inference_duration_ms"
        };
        meter
            .f64_histogram(duration_metric)
            .build()
            .record(observation.duration_ms, &attributes);
        Ok(true)
    }

    #[napi]
    pub fn stats(&self, query_json: String) -> napi::Result<String> {
        let query: Query = parse(&query_json)?;
        serde_json::to_string(&self.store.metrics.stats(query.filter()))
            .map_err(|err| error("ERR_METRICS", err))
    }

    #[napi]
    pub fn prometheus(&self, query_json: String) -> napi::Result<String> {
        let query: Query = parse(&query_json)?;
        self.store
            .metrics
            .prometheus(query.filter())
            .map_err(|err| error("ERR_METRICS", err))
    }
}
