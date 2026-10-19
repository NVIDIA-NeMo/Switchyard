// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage-router knobs that only a composite route can use are rejected at load.

use switchyard_runner::Runner;

const CONFIG: &str = r#"
schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
[targets.fast]
id = "fast-model"
llm_client = "local"
[targets.strong]
id = "strong-model"
llm_client = "local"
[targets.judge]
id = "judge-model"
llm_client = "local"
[routes.stage]
id = "stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "fast"
picker = "efficient_first"
confidence_threshold = 0.5
"#;

const CLASSIFIER: &str = "[routes.stage.classifier]\ntarget = \"judge\"\nbase_threshold = 0.5\n";

fn load_error(config: &str) -> String {
    match Runner::from_toml(config) {
        Ok(_) => panic!("expected a load error for:\n{config}"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn accepts_the_default_classifier_trigger() {
    assert!(Runner::from_toml(&format!("{CONFIG}{CLASSIFIER}")).is_ok());
    let explicit = format!("{CONFIG}{CLASSIFIER}classify_trigger = \"every_request\"\n");
    assert!(Runner::from_toml(&explicit).is_ok());
}

#[test]
fn rejects_classifier_knobs_that_have_no_effect() {
    for setting in [
        "classify_trigger = \"user_turn\"",
        "classify_trigger = \"new_session\"",
        "message_hash_fallback = true",
    ] {
        let error = load_error(&format!("{CONFIG}{CLASSIFIER}{setting}\n"));
        assert!(
            error.contains("have no effect on the stage fallback judge"),
            "{setting}: {error}"
        );
    }
}

#[test]
fn rejects_capable_first_threshold_on_a_standalone_route() {
    let error = load_error(&format!("{CONFIG}capable_first_threshold = 0.2\n"));
    assert!(
        error.contains("capable_first_threshold applies only to composite routes"),
        "{error}"
    );
}
