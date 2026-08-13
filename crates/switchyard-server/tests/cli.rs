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
    assert!(stderr.contains("llm client invalid base_url"), "{stderr}");
    Ok(())
}
