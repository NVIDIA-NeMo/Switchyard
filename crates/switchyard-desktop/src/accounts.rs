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
    harness::binary(tool).ok_or("Install the selected coding tool first.")?;
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
    login(tool, &path, name)?;
    pending.0 = None;
    Ok(format!(
        "Opened the coding tool's login in Terminal for account {name}. Complete login, click Refresh, and select this account. Switchyard does not read the login tokens. The selected account applies to new sessions only. Account directory: {}",
        path.display()
    ))
}

pub fn login(tool: Harness, path: &Path, name: &str) -> Result<String, String> {
    validate(tool, path)?;
    let binary = harness::binary(tool).ok_or("Install the selected coding tool first.")?;
    let variable = match tool {
        Harness::CodexCli => "CODEX_HOME",
        Harness::Claude => "CLAUDE_CONFIG_DIR",
        _ => return Err("Choose a supported coding tool.".into()),
    };
    #[cfg(windows)]
    crate::windows::open_terminal(crate::windows::TerminalJob {
        binary: binary.clone(),
        args: match tool {
            Harness::CodexCli => vec!["login".into()],
            _ => vec!["auth".into(), "login".into()],
        },
        directory: None,
        environment: vec![(variable.into(), path.display().to_string())],
        remove_environment: vec![],
    })?;
    #[cfg(not(windows))]
    {
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
    }
    Ok(format!(
        "Opened login in Terminal for account {name}. Finish login in the coding tool. Login status remains unknown until the tool confirms it."
    ))
}

pub fn archive(tool: Harness, path: &Path) -> Result<String, String> {
    validate(tool, path)?;
    let archived = path.with_file_name(format!(".archived-{}", uuid::Uuid::now_v7()));
    std::fs::rename(path, &archived).map_err(|error| error.to_string())?;
    Ok(format!(
        "Archived the account folder at {}. Login files are preserved. Existing sessions may still use the old directory and need a new login.",
        archived.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn archived_accounts_leave_the_picker_and_keep_login_files() {
        if std::env::var_os("SWITCHYARD_ACCOUNT_ARCHIVE_FIXTURE").is_some() {
            let entries = discover(Harness::CodexCli).expect("accounts");
            assert_eq!(entries.len(), 1);
            let path = &entries[0].1;
            archive(Harness::CodexCli, path).expect("archive");
            assert!(!path.exists());
            assert!(
                discover(Harness::CodexCli)
                    .expect("accounts after archive")
                    .is_empty()
            );
            let hidden = std::fs::read_dir(root(Harness::CodexCli).expect("root"))
                .expect("archived root")
                .next()
                .expect("archived entry")
                .expect("entry")
                .path();
            assert_eq!(
                std::fs::read_to_string(hidden.join("auth.json")).expect("login"),
                "private-fixture-login"
            );
            return;
        }
        let fixture = tempfile::tempdir().expect("fixture");
        let path = fixture.path().join(".switchyard/accounts/codex/named");
        std::fs::create_dir_all(&path).expect("account");
        std::fs::write(path.join("auth.json"), "private-fixture-login").expect("login");
        let child = std::process::Command::new(std::env::current_exe().expect("runner"))
            .args([
                "--exact",
                "accounts::tests::archived_accounts_leave_the_picker_and_keep_login_files",
            ])
            .env("HOME", fixture.path())
            .env("SWITCHYARD_ACCOUNT_ARCHIVE_FIXTURE", "1")
            .output()
            .expect("child");
        assert!(
            child.status.success(),
            "{}",
            String::from_utf8_lossy(&child.stdout)
        );
    }

    #[test]
    fn unsafe_names_fail_before_any_side_effect() {
        for name in ["", "..", "a/b", "$(touch bad)", "a\nb"] {
            assert!(add(Harness::CodexCli, name).is_err());
        }
        assert!(add(Harness::Pi, "valid").is_err());
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
            assert!(discover(Harness::CodexCli).is_err());
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
