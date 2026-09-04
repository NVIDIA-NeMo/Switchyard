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

// Dry-run rejects malformed static header names before binding or routing.
#[test]
fn dry_run_rejects_invalid_configured_header() -> TestResult {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("routes.toml");
    fs::write(
        &config,
        r#"
schema_version = 1

[llm_clients.invalid]
format = "openai_chat"
base_url = "https://example.test/v1"
extra_headers = { "bad header" = "value" }

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
        stderr.contains("invalid HTTP header name \"bad header\""),
        "{stderr}"
    );
    Ok(())
}

// Dry-run rejects malformed credentials without echoing them to stderr.
#[test]
fn dry_run_rejects_api_key_that_cannot_form_auth_header() -> TestResult {
    const INVALID_KEY_ENV: &str = "SWITCHYARD_CLI_TEST_INVALID_HEADER_KEY";
    const INVALID_KEY: &str = "canary\nsecret";

    let directory = tempfile::tempdir()?;
    let config = directory.path().join("routes.toml");
    fs::write(
        &config,
        format!(
            r#"
schema_version = 1

[llm_clients.invalid]
format = "openai_chat"
base_url = "https://example.test/v1"
api_key_env = "{INVALID_KEY_ENV}"

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

    let output = Command::new(env!("CARGO_BIN_EXE_switchyard-server"))
        .args(["--config", config.to_string_lossy().as_ref(), "--dry-run"])
        .env(INVALID_KEY_ENV, INVALID_KEY)
        .output()?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("api_key cannot be encoded as an HTTP header"),
        "{stderr}"
    );
    assert!(!stderr.contains(INVALID_KEY), "API key leaked in: {stderr}");
    Ok(())
}
