// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level regression coverage for the server CLI.

use std::fs;
use std::process::Command;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn dry_run_error(client_config: &str, env: Option<(&str, &str)>) -> TestResult<String> {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("routes.toml");
    fs::write(
        &config,
        format!(
            r#"
schema_version = 1

[llm_clients.invalid]
format = "openai_chat"
{client_config}

[targets.invalid]
id = "upstream-model"
llm_client = "invalid"

[routes.invalid]
id = "test-route"
type = "passthrough"
target = "invalid"
"#
        ),
    )?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_switchyard-server"));
    command.args(["--config", config.to_string_lossy().as_ref(), "--dry-run"]);
    if let Some((name, value)) = env {
        command.env(name, value);
    }
    let output = command.output()?;
    assert!(!output.status.success());
    Ok(String::from_utf8(output.stderr)?)
}

#[test]
fn dry_run_rejects_invalid_base_url() -> TestResult {
    let stderr = dry_run_error("base_url = \"not a url\"", None)?;
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

// Dry-run rejects unsendable headers before startup without exposing credentials.
#[test]
fn dry_run_rejects_unsendable_configured_headers() -> TestResult {
    const INVALID_KEY_ENV: &str = "SWITCHYARD_CLI_TEST_INVALID_HEADER_KEY";
    const INVALID_KEY: &str = "canary\nsecret";
    let cases = [
        (
            "base_url = \"https://example.test/v1\"\n\
             extra_headers = { \"bad header\" = \"value\" }"
                .to_string(),
            None,
            "invalid HTTP header name \"bad header\"",
        ),
        (
            format!("base_url = \"https://example.test/v1\"\napi_key_env = \"{INVALID_KEY_ENV}\""),
            Some((INVALID_KEY_ENV, INVALID_KEY)),
            "api_key cannot be encoded as an HTTP header",
        ),
    ];

    for (client_config, env, expected) in cases {
        let stderr = dry_run_error(&client_config, env)?;
        assert!(stderr.contains(expected), "{stderr}");
        assert!(!stderr.contains(INVALID_KEY), "API key leaked in: {stderr}");
    }
    Ok(())
}

#[test]
fn pricing_requires_a_log_and_validates_before_serving() -> TestResult {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("routes.toml");
    let prices = directory.path().join("prices.json");
    let log = directory.path().join("routing.jsonl");
    fs::write(
        &config,
        r#"
schema_version = 1
[llm_clients.upstream]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
[targets.answer]
id = "answer"
llm_client = "upstream"
[routes.direct]
id = "direct"
type = "passthrough"
target = "answer"
"#,
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_switchyard-server"))
        .arg("--config")
        .arg(&config)
        .arg("--pricing-file")
        .arg(&prices)
        .arg("--dry-run")
        .output()?;
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("--routing-log-file"));
    assert!(!log.exists());
    for (rates, succeeds) in [
        (
            r#"{"answer":{"input":0,"output":0,"cache_read":0,"cache_write":0}}"#,
            true,
        ),
        (r#"{"answer":{"input":0,"output":0,"cache_read":0}}"#, false),
    ] {
        fs::write(&prices, rates)?;
        let output = Command::new(env!("CARGO_BIN_EXE_switchyard-server"))
            .arg("--config")
            .arg(&config)
            .arg("--pricing-file")
            .arg(&prices)
            .arg("--routing-log-file")
            .arg(&log)
            .arg("--dry-run")
            .output()?;
        assert_eq!(output.status.success(), succeeds);
        if !succeeds {
            assert!(String::from_utf8(output.stderr)?.contains("invalid pricing file"));
        }
        assert!(fs::read(&log)?.is_empty());
    }
    Ok(())
}
