// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Named login directories use the harness's own authentication commands.

use crate::harness::{self, Harness};
use std::path::{Path, PathBuf};

fn root(tool: Harness) -> Result<PathBuf, String> {
    let name = match tool {
        Harness::CodexCli => "codex",
        Harness::Claude => "claude",
        _ => return Err("Named accounts are available for Codex CLI and Claude Code.".into()),
    };
    let home = crate::config::expand_home(Path::new("~"))?;
    let mut path = home;
    for component in [".switchyard", "accounts", name] {
        path.push(component);
        if std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(format!(
                "{} is a symlink. Edit its target yourself.",
                path.display()
            ));
        }
    }
    Ok(path)
}

pub fn validate(tool: Harness, path: &Path) -> Result<(), String> {
    let root = root(tool)?;
    if path.parent() != Some(root.as_path())
        || !std::fs::symlink_metadata(path).is_ok_and(|m| m.is_dir() && !m.file_type().is_symlink())
    {
        return Err("Choose a supported harness.".into());
    }
    Ok(())
}

pub fn discover(tool: Harness) -> Result<Vec<(String, PathBuf)>, String> {
    if !matches!(tool, Harness::CodexCli | Harness::Claude) {
        return Ok(Vec::new());
    }
    let root = root(tool)?;
    let entries = match std::fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(format!(
                "Could not list accounts in {}: {error}",
                root.display()
            ));
        }
    };
    let mut accounts = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with(".archived-")
            && entry
                .file_type()
                .map_err(|error| error.to_string())?
                .is_dir()
            && let Ok(name) = entry.file_name().into_string()
        {
            accounts.push((name, entry.path()));
        }
    }
    accounts.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(accounts)
}

pub fn list(tool: Harness) -> Vec<(String, PathBuf)> {
    discover(tool).unwrap_or_default()
}

pub fn config_paths(tool: Harness, account: Option<&Path>) -> Result<Vec<PathBuf>, String> {
    let Some(account) = account else {
        return harness::paths(tool);
    };
    match tool {
        Harness::CodexCli => Ok(vec![account.join("sy.config.toml")]),
        Harness::Claude => Ok(vec![account.join("settings.json")]),
        _ => harness::paths(tool),
    }
}
