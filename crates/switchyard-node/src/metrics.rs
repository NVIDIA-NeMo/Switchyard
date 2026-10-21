// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics for model calls performed by the JavaScript host.

use std::collections::HashMap;
use std::sync::OnceLock;

use napi_derive::napi;
use opentelemetry::{Context, KeyValue, global};
use parking_lot::Mutex;
use serde::Deserialize;
use switchyard_costs::{CostSource, CostTotals, Pricing, RecordedCost};
use switchyard_libsy::metrics::{MetricFilter, Metrics};
use switchyard_protocol::Usage;

use crate::{error, parse};

fn pricing(model: &str, rates: Option<&serde_json::Value>) -> napi::Result<Pricing> {
    match rates {
        Some(rates) => {
            let json = serde_json::to_vec(&HashMap::from([(model, rates)]))
                .map_err(|err| error("ERR_INVALID_REQUEST", err))?;
            Pricing::parse(&json).map_err(|err| error("ERR_INVALID_REQUEST", err))
        }
        None => Ok(Pricing::default()),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Estimate {
    usage: Usage,
    rates: serde_json::Value,
}

#[napi]
pub fn estimate_cost(input_json: String) -> napi::Result<String> {
    let input: Estimate = parse(&input_json)?;
    let cost = pricing("forecast", Some(&input.rates))?.estimate("forecast", &input.usage);
    serde_json::to_string(&cost).map_err(|err| error("ERR_METRICS", err))
}

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

#[derive(Clone, Copy, Deserialize)]
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
    rates: Option<serde_json::Value>,
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
    fn matches(&self, context: &MetricContext) -> bool {
        self.session_id
            .as_ref()
            .is_none_or(|id| id == &context.session_id)
            && self
                .agent_id
                .as_ref()
                .is_none_or(|id| id == &context.agent_id)
            && self
                .agent_turn_idx
                .is_none_or(|idx| idx == context.agent_turn_idx)
            && self
                .user_turn_idx
                .is_none_or(|idx| idx == context.user_turn_idx)
    }

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
    recorded: Mutex<HashMap<(String, String), RecordedCall>>,
}

struct RecordedCall {
    context: MetricContext,
    kind: CallKind,
    cost: Option<RecordedCost>,
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
        let pricing = pricing(&observation.model, observation.rates.as_ref())?;
        let cost = observation
            .usage
            .as_ref()
            .and_then(|usage| pricing.cost(&observation.model, usage));
        let mut recorded = self.store.recorded.lock();
        let key = (observation.context.session_id.clone(), observation.id);
        if recorded.contains_key(&key) {
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
        if !matches!(observation.call_kind, CallKind::Request)
            && let Some(cost) = cost
        {
            let mut cost_attributes = attributes.clone();
            cost_attributes.push(KeyValue::new("call_kind", kind));
            cost_attributes.push(KeyValue::new(
                "cost_source",
                match cost.source {
                    CostSource::ProviderReported => "provider_reported",
                    CostSource::Estimated => "estimated",
                },
            ));
            meter
                .f64_counter("switchyard.llm_call.cost_usd")
                .build()
                .add(cost.usd, &cost_attributes);
        }
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
        recorded.insert(
            key,
            RecordedCall {
                context: observation.context,
                kind: observation.call_kind,
                cost,
            },
        );
        Ok(true)
    }

    #[napi]
    pub fn costs(&self, query_json: String) -> napi::Result<String> {
        let query: Query = parse(&query_json)?;
        let mut totals = CostTotals::default();
        for call in self.store.recorded.lock().values() {
            if !matches!(call.kind, CallKind::Request) && query.matches(&call.context) {
                totals.add(call.cost);
            }
        }
        serde_json::to_string(&totals).map_err(|err| error("ERR_METRICS", err))
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
