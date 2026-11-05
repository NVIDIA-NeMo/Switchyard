// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native configuration, target identity and local routing for explicit weights.

use std::sync::Arc;
use switchyard_protocol::{Request, text_request};
use switchyard_runner::AlgorithmFactory;

fn config() -> String {
    let weights: libsy::WeightSpaceWeights = serde_json::from_str(include_str!(
        "../../../benchmark/weight_space/reference_weights.json"
    ))
    .expect("reference weights");
    format!(
        r#"
schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
[targets.efficient]
id = "different/efficient-provider-id"
llm_client = "local"
[targets.capable]
id = "different/capable-provider-id"
llm_client = "local"
[routes.neural]
id = "neural"
type = "weight_space"
targets = ["efficient", "capable"]
cost_weight = 0.75
[routes.neural.weights]
{}
"#,
        toml::to_string(&weights).expect("weights TOML")
    )
}

#[tokio::test]
async fn selects_target_without_any_provider_calls() {
    let (algorithm, models) = AlgorithmFactory::from_toml(&config())
        .expect("parse")
        .build()
        .expect("build")
        .remove("neural")
        .expect("route");
    let request = Request {
        llm_request: text_request(None, "hello"),
        ..Default::default()
    };
    let outcome = libsy::drive(algorithm, request, Arc::new(models), |_| async {
        panic!("neural routing makes no provider call")
    })
    .await
    .expect("route");
    assert_eq!(
        outcome.selected_model_id().expect("target").as_str(),
        "efficient"
    );
    assert_eq!(
        outcome.request.llm_request.messages[0].text_content(""),
        Some("hello".into())
    );
}

#[test]
fn execution_status_checkpoint_builds_with_twenty_inputs() {
    let mut config: toml::Value = toml::from_str(&config()).expect("config");
    let weights = &mut config["routes"]["neural"]["weights"];
    weights["schema_version"] = toml::Value::Integer(2);
    for row in weights["input_weights"].as_array_mut().expect("rows") {
        row.as_array_mut()
            .expect("inputs")
            .extend([toml::Value::Float(0.0), toml::Value::Float(0.0)]);
    }
    AlgorithmFactory::from_toml(&toml::to_string(&config).expect("TOML"))
        .expect("parse")
        .build()
        .expect("schema 2 checkpoint");
}

#[test]
fn incompatible_checkpoint_and_duplicate_targets_are_rejected() {
    for config in [
        config().replace(
            "schema_version = 1\ninput_weights",
            "schema_version = 2\ninput_weights",
        ),
        config().replace(
            "targets = [\"efficient\", \"capable\"]",
            "targets = [\"efficient\", \"efficient\"]",
        ),
        config().replace("cost_weight = 0.75", "cost_weight = -1.0"),
    ] {
        assert!(
            AlgorithmFactory::from_toml(&config)
                .expect("parse")
                .build()
                .is_err()
        );
    }
}
