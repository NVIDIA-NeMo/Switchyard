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
