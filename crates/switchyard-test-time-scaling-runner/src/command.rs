// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shell-free JSON command calls.

use std::process::Stdio;

use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::config::CommandConfig;

/// Sends one JSON value to a command and parses its JSON output.
pub async fn call_json<I, O>(command: &CommandConfig, input: &I) -> Result<O, String>
where
    I: Serialize,
    O: DeserializeOwned,
{
    let Some((program, arguments)) = command.argv.split_first() else {
        return Err("command is empty".to_string());
    };
    let mut child = Command::new(program)
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| format!("could not start {program}: {error}"))?;
    let bytes = serde_json::to_vec(input).map_err(|error| error.to_string())?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "command stdin is unavailable".to_string())?;
    stdin
        .write_all(&bytes)
        .await
        .map_err(|error| format!("could not write command input: {error}"))?;
    drop(stdin);

    let output = child
        .wait_with_output()
        .await
        .map_err(|error| format!("command did not finish: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "command exited with {}; stderr: {}",
            output.status,
            bounded_text(&output.stderr, 8_000)
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| {
        format!(
            "command returned invalid JSON: {error}; stdout: {}",
            bounded_text(&output.stdout, 8_000)
        )
    })
}

fn bounded_text(bytes: &[u8], limit: usize) -> String {
    String::from_utf8_lossy(&bytes[..bytes.len().min(limit)]).into_owned()
}
