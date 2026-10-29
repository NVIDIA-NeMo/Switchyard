// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage fallback-judge knobs that cannot take effect are rejected at load.

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

const POOL: &str = r#"
[targets.middle]
id = "middle-model"
llm_client = "local"
[routes.stage.pool]
targets = ["fast", "middle", "strong"]
positions = [0.0, 0.4, 1.0]
"#;

#[test]
fn accepts_configured_lines_and_categorical_fallback() {
    assert!(Runner::from_toml(&format!("{CONFIG}{POOL}")).is_ok());
    assert!(
        Runner::from_toml(&format!(
            "{CONFIG}{POOL}[routes.stage.classifier]\ntarget = \"judge\"\n"
        ))
        .is_ok()
    );
}

#[test]
fn rejects_invalid_pools_and_binary_judge_knobs_for_n_targets() {
    for bad in [
        POOL.replace("0.4", "0.0"),
        POOL.replace("0.4", "nan"),
        POOL.replace("0.4", "1.1"),
        POOL.replace("0.0, 0.4, 1.0", "0.0, 1.0"),
        POOL.replace(
            "[\"fast\", \"middle\", \"strong\"]",
            "[\"strong\", \"middle\", \"fast\"]",
        ),
        POOL.replace(
            "[\"fast\", \"middle\", \"strong\"]",
            "[\"fast\", \"missing\", \"strong\"]",
        ),
        POOL.replace(
            "[\"fast\", \"middle\", \"strong\"]",
            "[\"fast\", \"fast\", \"strong\"]",
        ),
    ] {
        assert!(
            Runner::from_toml(&format!("{CONFIG}{bad}")).is_err(),
            "{bad}"
        );
    }
    assert!(load_error(&format!("{CONFIG}{POOL}{CLASSIFIER}")).contains("categorical"));
}

#[test]
fn rejects_invalid_n_target_classifier_settings_at_load() {
    for (setting, expected) in [
        ("classify_trigger = 'user_turn'", "without retention"),
        ("message_hash_fallback = true", "without retention"),
        ("max_output_tokens = 0", "at least 1"),
        ("prompt = '   '", "prompt must not be empty"),
        ("threshold_step = 0.1", "categorical"),
        ("response_format_type = 'json_object'", "categorical"),
        ("mode = 'custom'", "categorical"),
    ] {
        let error = load_error(&format!(
            "{CONFIG}{POOL}[routes.stage.classifier]\ntarget = 'judge'\n{setting}\n"
        ));
        assert!(error.contains(expected), "{setting}: {error}");
    }
    let error = load_error(&format!(
        "{CONFIG}{POOL}[routes.stage.classifier]\ntarget = 'missing'\n"
    ));
    assert!(error.contains("missing"), "{error}");
}
