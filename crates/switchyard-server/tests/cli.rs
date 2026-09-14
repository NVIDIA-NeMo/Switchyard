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

/// Checks that `--dry-run` accepts valid UTF-8 instructions and rejects missing,
/// blank, or invalid UTF-8 files.
#[test]
fn dry_run_validates_codex_instruction_files() -> TestResult {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("routes.toml");
    fs::write(
        &config,
        r#"
schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
[targets.local]
id = "upstream-model"
llm_client = "local"
[routes.local]
id = "test-route"
type = "passthrough"
target = "local"
"#,
    )?;
    let prompt = directory.path().join("instructions.txt");
    for (contents, expected_success) in [
        (None, false),
        (Some(b" \n\t".as_slice()), false),
        (Some(b"\xff".as_slice()), false),
        (
            Some(b"Configured instructions.\n  Keep whitespace.  \n".as_slice()),
            true,
        ),
    ] {
        if let Some(contents) = contents {
            fs::write(&prompt, contents)?;
        }
        let output = Command::new(env!("CARGO_BIN_EXE_switchyard-server"))
            .arg("--config")
            .arg(&config)
            .arg("--dry-run")
            .arg("--codex-base-instructions-file")
            .arg(&prompt)
            .output()?;
        assert_eq!(output.status.success(), expected_success);
        if !expected_success {
            assert!(String::from_utf8(output.stderr)?.contains("codex"));
        }
    }
    Ok(())
}
