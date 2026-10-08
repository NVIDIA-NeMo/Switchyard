// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Named login directories use the coding tool's own authentication commands.

use crate::harness::{self, Harness};
use std::path::{Path, PathBuf};

// PendingAccount removes only an empty directory if login setup fails.
// The coding tool may already have saved login files, which must stay.
struct PendingAccount(Option<PathBuf>);

impl Drop for PendingAccount {
    fn drop(&mut self) {
        if let Some(path) = &self.0 {
            let _ = std::fs::remove_dir(path);
        }
    }
}

fn root(tool: Harness) -> Result<PathBuf, String> {
    let name = match tool {
        Harness::CodexCli => "codex",
        Harness::Claude => "claude",
        _ => return Err("Named accounts are available for Codex CLI and Claude Code.".into()),
    };
    let home = crate::config::expand_home(Path::new("~"));
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
        return Err("Choose a supported coding tool.".into());
    }
    Ok(())
}

pub fn list(tool: Harness) -> Vec<(String, PathBuf)> {
    let Ok(root) = root(tool) else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut accounts: Vec<_> = entries
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter_map(|e| {
            e.file_name()
                .into_string()
                .ok()
                .map(|name| (name, e.path()))
        })
        .collect();
    accounts.sort_by(|a, b| a.0.cmp(&b.0));
    accounts
}

pub fn config_paths(tool: Harness, account: Option<&Path>) -> Vec<PathBuf> {
    let Some(account) = account else {
        return harness::paths(tool);
    };
    match tool {
        Harness::CodexCli => vec![account.join("sy.config.toml")],
        Harness::Claude => vec![account.join("settings.json")],
        _ => harness::paths(tool),
    }
}

pub fn add(tool: Harness, name: &str) -> Result<String, String> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("Account names need 1–64 letters, numbers, hyphens, or underscores.".into());
    }
    let root = root(tool)?;
    let binary = harness::binary(tool).ok_or("Install the selected coding tool first.")?;
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let path = root.join(name);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new().mode(0o700).create(&path).map_err(|e|format!("Create account: {e}. Pick an existing account after Refresh, or choose a new name."))?;
    }
    #[cfg(not(unix))]
    std::fs::create_dir(&path).map_err(|e| e.to_string())?;
    let mut pending = PendingAccount(Some(path.clone()));
    let variable = match tool {
        Harness::CodexCli => "CODEX_HOME",
        Harness::Claude => "CLAUDE_CONFIG_DIR",
        _ => return Err("Choose a supported coding tool.".into()),
    };
    let login = match tool {
        Harness::CodexCli => "login",
        _ => "auth login",
    };
    let script = format!(
        "#!/bin/bash\nrm -- \"$0\"\nexport PATH=\"/opt/homebrew/bin:/usr/local/bin:$PATH\"\nexport {variable}={}\n{} {login}",
        crate::sessions::quote(&path.display().to_string()),
        crate::sessions::quote(&binary.display().to_string())
    );
    crate::server::open_terminal_script("switchyard-login-", &script)?;
    pending.0 = None;
    Ok(format!(
        "Opened the coding tool's login in Terminal for account {name}. Complete login, click Refresh, and select this account. Switchyard does not read the login tokens. The selected account applies to new sessions only. Account directory: {}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsafe_names_fail_before_any_side_effect() {
        for name in ["", "..", "a/b", "$(touch bad)", "a\nb"] {
            assert!(add(Harness::CodexCli, name).is_err());
        }
        assert!(add(Harness::Pi, "valid").is_err());
    }
    #[test]
    fn selected_account_changes_only_the_owned_config_destination() {
        let account = Path::new("/tmp/named-account");
        assert_eq!(
            config_paths(Harness::CodexCli, Some(account)),
            vec![account.join("sy.config.toml")]
        );
        assert_eq!(
            config_paths(Harness::Claude, Some(account)),
            vec![account.join("settings.json")]
        );
    }
    #[test]
    fn a_login_setup_failure_removes_only_its_empty_directory() {
        let root = tempfile::tempdir().expect("directory");
        let path = root.path().join("account");
        std::fs::create_dir(&path).expect("account");
        drop(PendingAccount(Some(path.clone())));
        assert!(!path.exists());
        std::fs::create_dir(&path).expect("account");
        std::fs::write(path.join("auth.json"), "private").expect("auth");
        drop(PendingAccount(Some(path.clone())));
        assert!(path.join("auth.json").is_file());
    }
    // The subprocess isolates HOME so parallel tests keep their own settings.
    #[cfg(unix)]
    #[test]
    fn symlinked_account_roots_are_not_listed_or_used_for_login() {
        if std::env::var_os("SWITCHYARD_ACCOUNT_FIXTURE").is_some() {
            assert!(list(Harness::CodexCli).is_empty());
            assert!(add(Harness::CodexCli, "new").is_err());
            let path = crate::config::expand_home(Path::new("~/.switchyard/accounts/codex/named"));
            assert!(validate(Harness::CodexCli, &path).is_err());
            return;
        }
        let dir = tempfile::tempdir().expect("fixture");
        let home = dir.path().join("home");
        let target = dir.path().join("target");
        std::fs::create_dir(&home).expect("home");
        std::fs::create_dir_all(target.join("accounts/codex/named")).expect("accounts");
        std::os::unix::fs::symlink(&target, home.join(".switchyard")).expect("symlink");
        let output = std::process::Command::new(std::env::current_exe().expect("runner"))
            .args([
                "--exact",
                "accounts::tests::symlinked_account_roots_are_not_listed_or_used_for_login",
            ])
            .env("SWITCHYARD_ACCOUNT_FIXTURE", "1")
            .env("HOME", &home)
            .output()
            .expect("child");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert!(!target.join("accounts/codex/new").exists());
    }
}
