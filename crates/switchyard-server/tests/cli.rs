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

/// Checks template loading and rendering before the server binds a socket.
#[test]
fn dry_run_validates_codex_system_templates() -> TestResult {
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
[routes.second]
id = "second-route"
type = "passthrough"
target = "local"
"#,
    )?;
    let prompt = directory.path().join("system.jinja");
    for (contents, expected_error) in [
        (None, Some("invalid --codex-system-template")),
        (
            Some(b" \n\t".as_slice()),
            Some("renders blank instructions"),
        ),
        (
            Some(b"\xff".as_slice()),
            Some("invalid --codex-system-template"),
        ),
        (
            Some(b"{{ model_id".as_slice()),
            Some("invalid Codex system template"),
        ),
        (
            Some(b"{{ model_id | nonexistent }}".as_slice()),
            Some("cannot render Codex system template"),
        ),
        (
            Some(b"{% if typo is true %}tools{% endif %}Custom".as_slice()),
            Some("unknown Codex system template variables: typo"),
        ),
        (
            Some(b"{% if false %}{{ personality }}{% endif %}Custom".as_slice()),
            Some("unknown Codex system template variables: personality"),
        ),
        (
            Some(b"{% if model_id == 'second-route' %}Custom{% endif %}".as_slice()),
            Some("renders blank instructions for model 'test-route'"),
        ),
        (
            Some(b"Custom instructions for {{ model_id }}.\n  Keep whitespace.  \n".as_slice()),
            None,
        ),
    ] {
        if let Some(contents) = contents {
            fs::write(&prompt, contents)?;
        }
        let output = Command::new(env!("CARGO_BIN_EXE_switchyard-server"))
            .arg("--config")
            .arg(&config)
            .arg("--dry-run")
            .arg("--codex-system-template")
            .arg(&prompt)
            .output()?;
        let stderr = String::from_utf8(output.stderr)?;
        assert_eq!(
            output.status.success(),
            expected_error.is_none(),
            "{stderr}"
        );
        if let Some(expected_error) = expected_error {
            assert!(stderr.contains(expected_error), "{stderr}");
        }
    }
    Ok(())
}

#[test]
fn dry_run_requires_routes_for_codex_system_template() -> TestResult {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("routes.toml");
    fs::write(&config, "schema_version = 1\ntargets = {}\nroutes = {}\n")?;
    let prompt = directory.path().join("system.jinja");
    fs::write(&prompt, "Custom instructions for {{ model_id }}.\n")?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_switchyard-server"));
    command.arg("--config").arg(&config).arg("--dry-run");
    let output = command.output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = command
        .arg("--codex-system-template")
        .arg(&prompt)
        .output()?;
    assert!(!output.status.success());
    assert!(
        String::from_utf8(output.stderr)?
            .contains("cannot use a Codex system template without configured routes")
    );
    Ok(())
}
