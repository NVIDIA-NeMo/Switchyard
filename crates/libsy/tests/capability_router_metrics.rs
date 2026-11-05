// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use serde_json::json;
use switchyard_libsy::{
    CapabilityRouter, CapabilityRouterConfig, PickerMode, PoolContext, PoolModel, Result,
    RuleSignalConfig, RuntimeModels, ToolLoopSignalConfig, ToolSemantics, drive,
};
use switchyard_protocol::{
    Category, ContentBlock, LlmRequest, Message, Metadata, ModelId, Request, Role, ToolCall,
    ToolResult, WireFormat,
};

fn router(affinity: bool, hold: u32, mode: PickerMode, rule: bool) -> Arc<CapabilityRouter> {
    let pool = PoolContext {
        models: [
            ("strong", "capable"),
            ("balanced", "balanced"),
            ("weak", "efficient"),
        ]
        .into_iter()
        .map(|(target, tag)| PoolModel {
            target: target.into(),
            provides: vec![tag.parse().unwrap()],
        })
        .collect(),
        default: Category::Efficient,
    };
    Arc::new(
        CapabilityRouter::new(CapabilityRouterConfig {
            pool,
            task: None,
            rules: rule.then(|| RuleSignalConfig {
                capable: vec![],
                balanced: vec!["fix the build".into()],
                named: BTreeMap::new(),
            }),
            tool_loop: Some(ToolLoopSignalConfig {
                mode,
                confidence_threshold: 0.2,
                recent_window: None,
                tool_semantics: ToolSemantics::default(),
                handoff_notes: None,
            }),
            review: None,
            affinity,
        })
        .unwrap()
        .with_escalation_hold_turns(hold),
    )
}

fn request(result: &str) -> Request {
    Request {
        llm_request: LlmRequest {
            messages: vec![
                Message::text(Role::User, "fix the build"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall(ToolCall {
                        id: "call_1".into(),
                        name: "Bash".into(),
                        arguments: json!({"command": "cargo test"}),
                    })],
                },
                Message {
                    role: Role::Tool,
                    content: vec![ContentBlock::ToolResult(ToolResult {
                        tool_call_id: "call_1".into(),
                        content: vec![ContentBlock::Text {
                            text: result.into(),
                        }],
                        is_error: None,
                    })],
                },
            ],
            ..LlmRequest::default()
        },
        raw_request: Some(json!({"messages": [
            {"role": "user", "content": "fix the build"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "Bash", "arguments": "{\"command\":\"cargo test\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": result},
        ]})),
        metadata: Some(Metadata {
            session_id: Some("session".into()),
            wire_format: Some(WireFormat::OpenAiChat),
            ..Metadata::default()
        }),
    }
}

// This integration binary owns its global meter provider, so parallel library
// unit tests cannot add unrelated decisions to the exported counters.
#[tokio::test]
async fn decisions_count_final_provider_once_after_pool_and_retention() -> Result<()> {
    let exporter = InMemoryMetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    opentelemetry::global::set_meter_provider(provider.clone());
    let groups: HashMap<_, _> = [
        (Category::Capable, vec![ModelId::from("strong")]),
        ("balanced".parse().unwrap(), vec![ModelId::from("balanced")]),
        (Category::Efficient, vec![ModelId::from("weak")]),
        (
            Category::Any,
            vec!["strong".into(), "balanced".into(), "weak".into()],
        ),
    ]
    .into();
    let models = Arc::new(
        RuntimeModels::new(groups).with_provider_models(
            ["strong", "balanced", "weak"]
                .map(|name| (name.into(), format!("provider/{name}").into()))
                .into(),
        ),
    );
    let ordinary = router(false, 0, PickerMode::EfficientFirst, false);
    let held = router(false, 3, PickerMode::EfficientFirst, false);
    let retained = router(true, 0, PickerMode::EfficientFirst, false);
    let mut expected = BTreeMap::new();
    for (algorithm, result, target, source) in [
        (ordinary.clone(), "exit code 1", "balanced", "dimensions"),
        (ordinary.clone(), "out of memory", "strong", "override"),
        (ordinary, "ok", "weak", "pool_default"),
        (
            router(false, 0, PickerMode::EfficientFirst, true),
            "out of memory",
            "balanced",
            "rules",
        ),
        (held.clone(), "exit code 1", "balanced", "dimensions"),
        (held, "ok", "balanced", "escalation_hold"),
        (retained.clone(), "ok", "weak", "pool_default"),
        (retained, "out of memory", "weak", "retained"),
        (
            router(false, 0, PickerMode::CapableFirst, false),
            "ok",
            "strong",
            "fall_open",
        ),
    ] {
        let outcome = drive(algorithm, request(result), models.clone(), |_| async {
            unreachable!("these signals make no model calls")
        })
        .await?;
        assert_eq!(outcome.selected_model_id()?.as_str(), target);
        assert_eq!(
            outcome.metadata.unwrap().evidence.unwrap()["source"],
            source
        );
        *expected
            .entry((source.to_string(), format!("provider/{target}")))
            .or_insert(0_u64) += 1;
    }
    provider.force_flush().unwrap();
    let mut actual = BTreeMap::new();
    for resource in exporter.get_finished_metrics().unwrap() {
        for scope in resource.scope_metrics() {
            for metric in scope.metrics() {
                if metric.name() == "switchyard.stage_router.routing_decisions"
                    && let AggregatedMetrics::U64(MetricData::Sum(sum)) = metric.data()
                {
                    for point in sum.data_points() {
                        let attributes: BTreeMap<_, _> = point
                            .attributes()
                            .map(|attribute| (attribute.key.as_str(), attribute.value.to_string()))
                            .collect();
                        *actual
                            .entry((
                                attributes["decision_source"].clone(),
                                attributes["target_name"].clone(),
                            ))
                            .or_insert(0_u64) += point.value();
                    }
                }
            }
        }
    }
    provider.shutdown().unwrap();
    assert_eq!(actual, expected);
    Ok(())
}
