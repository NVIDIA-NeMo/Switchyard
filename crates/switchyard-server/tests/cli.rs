// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level regression coverage for the server CLI.

use std::fs;
use std::process::Command;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

#[test]
fn dry_run_rejects_invalid_base_url() -> TestResult {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("routes.toml");
    fs::write(
        &config,
        r#"
schema_version = 1

[llm_clients.invalid]
format = "openai_chat"
base_url = "not a url"

[targets.invalid]
id = "upstream-model"
llm_client = "invalid"

[routes.invalid]
id = "test-route"
type = "passthrough"
target = "invalid"
"#,
    )?;

    let output = Command::new(env!("CARGO_BIN_EXE_switchyard-server"))
        .args(["--config", config.to_string_lossy().as_ref(), "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("base_url must be an absolute HTTP(S) URL"),
        "{stderr}"
    );
    Ok(())
}

// A judge adds no routing choice when every completion category names the same target.
#[test]
fn dry_run_warns_when_custom_classifier_has_one_target() -> TestResult {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("routes.toml");
    for (targets, warning_count) in [("[\"answer\"]", 1), ("[\"answer\", \"second\"]", 0)] {
        fs::write(
            &config,
            format!(
                r#"
schema_version = 1
[llm_clients.upstream]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
[targets]
judge = {{ id = "judge", llm_client = "upstream" }}
answer = {{ id = "answer", llm_client = "upstream" }}
second = {{ id = "second", llm_client = "upstream" }}
[routes.custom]
id = "custom"
type = "llm_classifier"
mode = "custom"
models = {{ judge = ["judge"], any = {targets}, efficient = ["answer"] }}
default_target = "efficient"
prompt = "Choose efficient."
response_schema = '{{"type":"object","properties":{{"target":{{"type":"string"}}}},"required":["target"],"additionalProperties":false}}'
policy = {{ type = "target_selector", selector = "/target" }}
"#
            ),
        )?;
        let output = Command::new(env!("CARGO_BIN_EXE_switchyard-server"))
            .args(["--config", config.to_string_lossy().as_ref(), "--dry-run"])
            .env("RUST_LOG", "warn")
            .output()?;
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout)?;
        let stderr = String::from_utf8(output.stderr)?;
        assert!(stdout.contains("server OK: custom"));
        let logs = format!("{stdout}{stderr}");
        assert_eq!(
            logs.matches("only one routing target").count(),
            warning_count,
            "{logs}"
        );
    }
    Ok(())
}
