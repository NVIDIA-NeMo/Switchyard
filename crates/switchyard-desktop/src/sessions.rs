// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module starts CLI sessions with separate worktrees and model settings.

use std::path::Path;

pub(crate) fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

fn session_root() -> Result<std::path::PathBuf, String> {
    crate::config::expand_home(Path::new("~/.switchyard/worktrees"))
}

pub fn inventory() -> Result<Vec<serde_json::Value>, String> {
    let root = session_root()?;
    let entries = match std::fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("Read session inventory: {error}")),
    };
    let mut sessions = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        if entry
            .file_type()
            .map_err(|error| error.to_string())?
            .is_dir()
            && entry.file_name().to_string_lossy().ends_with("-settings")
        {
            let receipt = entry.path().join("session.json");
            if receipt.exists() {
                let mut value: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(&receipt).map_err(|error| error.to_string())?,
                )
                .map_err(|error| format!("Read {}: {error}", receipt.display()))?;
                value
                    .as_object_mut()
                    .ok_or("The session receipt does not match this worktree.")?
                    .insert("process_status".into(), serde_json::json!("unknown"));
                sessions.push(value);
            }
        }
    }
    sessions.sort_by(|left, right| {
        right["created_at"]
            .as_str()
            .cmp(&left["created_at"].as_str())
    });
    Ok(sessions)
}
