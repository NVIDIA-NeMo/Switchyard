// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module starts CLI sessions with separate worktrees and model settings.

use crate::harness::{self, Harness};
use std::path::Path;

// PendingSession attempts cleanup until open accepts the launch request.
// A successful launch keeps the worktree and settings for the coding session.
struct PendingSession {
    project: std::path::PathBuf,
    worktree: std::path::PathBuf,
    private: std::path::PathBuf,
    branch: String,
    opened: bool,
}

impl Drop for PendingSession {
    fn drop(&mut self) {
        if self.opened {
            return;
        }
        let _ = std::process::Command::new("git")
            .arg("-C")
            .arg(&self.project)
            .args(["worktree", "remove"])
            .arg(&self.worktree)
            .output();
        let _ = std::process::Command::new("git")
            .arg("-C")
            .arg(&self.project)
            .args(["branch", "-d", &self.branch])
            .output();
        let _ = std::fs::remove_dir_all(&self.private);
    }
}

pub(crate) fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

pub fn preview(project: &Path) -> Result<serde_json::Value, String> {
    if project.as_os_str().is_empty() {
        return Err("Choose a Git checkout directory.".into());
    }
    let project = std::fs::canonicalize(crate::config::expand_home(project))
        .map_err(|error| format!("Open project: {error}"))?;
    let git = |args: &[&str]| -> Result<String, String> {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(&project)
            .args(args)
            .output()
            .map_err(|error| error.to_string())?;
        if !output.status.success() {
            return Err("Choose a Git checkout with a committed HEAD.".into());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    };
    let repository = git(&["rev-parse", "--show-toplevel"])?;
    let repository_path = std::fs::canonicalize(&repository).map_err(|error| error.to_string())?;
    let relative = project
        .strip_prefix(&repository_path)
        .map_err(|error| error.to_string())?;
    Ok(
        serde_json::json!({"repository":repository,"project":project,"relative_directory":relative,"revision":git(&["rev-parse","HEAD"])? ,"branch":git(&["branch","--show-current"])? ,"dirty":!git(&["status","--porcelain"])? .is_empty(),"notice":"The session starts from committed HEAD. Uncommitted changes stay in the original checkout."}),
    )
}

fn session_root() -> std::path::PathBuf {
    crate::config::expand_home(Path::new("~/.switchyard/worktrees"))
}

pub fn inventory() -> Result<Vec<serde_json::Value>, String> {
    let root = session_root();
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

pub fn managed_session(id: &str) -> Result<serde_json::Value, String> {
    let suffix = id.strip_prefix("sy-").ok_or("Choose a managed session.")?;
    uuid::Uuid::parse_str(suffix).map_err(|_| "Choose a managed session.")?;
    let root = session_root();
    let private = root.join(format!("{id}-settings"));
    let worktree = root.join(id);
    for path in [&root, &private, &worktree] {
        if std::fs::symlink_metadata(path)
            .map_err(|error| error.to_string())?
            .file_type()
            .is_symlink()
        {
            return Err("Session directories must not be symlinks.".into());
        }
    }
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(private.join("session.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if receipt["id"] != id || receipt["worktree"] != worktree.display().to_string() {
        return Err("The session receipt does not match this worktree.".into());
    }
    Ok(receipt)
}

pub fn cleanup_preview(id: &str) -> Result<serde_json::Value, String> {
    let mut receipt = managed_session(id)?;
    let worktree = receipt["worktree"]
        .as_str()
        .ok_or("Missing session worktree.")?
        .to_string();
    let run = |args: &[&str]| -> Result<String, String> {
        let output = std::process::Command::new("git")
            .args(["-C", &worktree])
            .args(args)
            .output()
            .map_err(|error| error.to_string())?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    };
    receipt["dirty"] = serde_json::json!(!run(&["status", "--porcelain"])?.is_empty());
    receipt["current_revision"] = serde_json::json!(run(&["rev-parse", "HEAD"])?);
    receipt["current_branch"] = serde_json::json!(run(&["branch", "--show-current"])?);
    receipt["cleanup_notice"] = serde_json::json!(
        "Cleanup removes a clean worktree and its unchanged session branch. Recorded usage stays in the routing log. Unknown private files are preserved."
    );
    Ok(receipt)
}

pub fn cleanup(id: &str, reviewed: &serde_json::Value) -> Result<String, String> {
    let current = cleanup_preview(id)?;
    if &current != reviewed {
        return Err("The session changed since review. Review cleanup again.".into());
    }
    if current["dirty"] != false
        || current["current_revision"] != current["revision"]
        || current["current_branch"] != id
    {
        return Err("Save or move the session’s work before cleanup. Cleanup requires a clean worktree with its original commit and branch.".into());
    }
    let repository = current["repository"]
        .as_str()
        .ok_or("Missing session repository.")?;
    let worktree = current["worktree"]
        .as_str()
        .ok_or("Missing session worktree.")?;
    let output = std::process::Command::new("git")
        .args(["-C", repository, "worktree", "remove", worktree])
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let output = std::process::Command::new("git")
        .args(["-C", repository, "branch", "-d", id])
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Ok("Removed the clean session worktree. Git kept the branch; inspect it before deleting it.".into());
    }
    let private = session_root().join(format!("{id}-settings"));
    // Cleanup removes only the receipt because settings can contain later user edits.
    std::fs::remove_file(private.join("session.json")).map_err(|error| error.to_string())?;
    let _ = std::fs::remove_dir(&private);
    Ok("Removed the clean session worktree and branch. Retained private settings and recorded usage.".into())
}

pub fn launch(
    tool: Harness,
    project: &Path,
    model: &str,
    root: &str,
    login: bool,
    account: Option<&Path>,
    revision: &str,
) -> Result<String, String> {
    let root = harness::validate(tool, root, model, login)?;
    if let Some(account) = account {
        crate::accounts::validate(tool, account)?;
    }
    if tool == Harness::CodexApp {
        return Err("Choose Codex CLI to launch a worktree session. Codex app uses its own workspace controls.".into());
    }
    let binary = harness::binary(tool).ok_or("Install the selected coding tool first.")?;
    if project.as_os_str().is_empty() {
        return Err("Choose a Git checkout directory.".into());
    }
    let project = crate::config::expand_home(project);
    let project = std::fs::canonicalize(&project).map_err(|e| format!("Open project: {e}"))?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&project)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("Choose a Git checkout directory.".into());
    }
    let repository = Path::new(
        std::str::from_utf8(&output.stdout)
            .map_err(|error| error.to_string())?
            .trim(),
    )
    .to_path_buf();
    let repository = std::fs::canonicalize(repository).map_err(|error| error.to_string())?;
    let relative = project
        .strip_prefix(&repository)
        .map_err(|error| error.to_string())?
        .to_path_buf();
    let id = format!("sy-{}", uuid::Uuid::now_v7());
    let worktree = crate::config::expand_home(Path::new("~/.switchyard/worktrees")).join(&id);
    std::fs::create_dir_all(
        worktree
            .parent()
            .ok_or("Worktree needs a parent directory")?,
    )
    .map_err(|e| e.to_string())?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&project)
        .args(["worktree", "add", "-b"])
        .arg(&id)
        .arg(&worktree)
        .arg(revision)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "Could not create worktree: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let private = worktree
        .parent()
        .ok_or("Worktree needs a parent directory")?
        .join(format!("{id}-settings"));
    let mut pending = PendingSession {
        project,
        worktree: worktree.clone(),
        private: private.clone(),
        branch: id.clone(),
        opened: false,
    };
    let files = match tool {
        Harness::Pi => vec![private.join("models.json"), private.join("settings.json")],
        _ => vec![private.join("settings.json")],
    };
    if matches!(tool, Harness::Claude | Harness::Pi) {
        std::fs::create_dir_all(&private).map_err(|e| e.to_string())?;
        harness::install(tool, &files, &root, model, login)?;
    }
    std::fs::create_dir_all(&private).map_err(|error| error.to_string())?;
    let receipt = serde_json::json!({"id":id,"tool":tool,"route":model,"repository":repository,"revision":preview(&worktree)?["revision"],"worktree":worktree,"branch":id,"created_at":chrono::Utc::now().to_rfc3339(),"settings_directory":private,"account_directory":account});
    std::fs::write(
        private.join("session.json"),
        serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    #[cfg(windows)]
    {
        let mut environment = Vec::new();
        let mut remove_environment = Vec::new();
        let args = match tool {
            Harness::CodexCli => vec![
                "-c".into(),
                format!("model={}", toml_edit::Value::from(model)),
                "-c".into(),
                "model_provider=\"sy\"".into(),
                "-c".into(),
                format!(
                    "model_providers.sy={{name=\"Switchyard\",base_url={},wire_api=\"responses\",requires_openai_auth={login},http_headers={{\"x-switchyard-session-id\"={},\"x-switchyard-origin\"=\"codex\"}}}}",
                    toml_edit::Value::from(format!("{}/v1", root.trim_end_matches('/'))),
                    toml_edit::Value::from(id.as_str())
                ),
            ],
            Harness::Claude => {
                environment.push((
                    "ANTHROPIC_CUSTOM_HEADERS".into(),
                    format!("x-switchyard-session-id: {id}\nx-switchyard-origin: claude"),
                ));
                if login {
                    remove_environment.extend(
                        [
                            "ANTHROPIC_AUTH_TOKEN",
                            "ANTHROPIC_API_KEY",
                            "CLAUDE_CODE_OAUTH_TOKEN",
                        ]
                        .map(String::from),
                    );
                }
                vec![
                    "--settings".into(),
                    files[0].display().to_string(),
                    "--model".into(),
                    model.into(),
                ]
            }
            Harness::Pi => {
                environment.push(("PI_CODING_AGENT_DIR".into(), private.display().to_string()));
                vec![
                    "--provider".into(),
                    "switchyard".into(),
                    "--model".into(),
                    model.into(),
                ]
            }
            Harness::CodexApp => return Err("Choose a CLI tool.".into()),
        };
        let default_files = harness::paths(tool);
        let account = account.or_else(|| match tool {
            Harness::Claude if std::env::var_os("CLAUDE_CONFIG_DIR").is_none() => None,
            _ => default_files[0].parent(),
        });
        if let Some(path) = account {
            match tool {
                Harness::CodexCli => {
                    environment.push(("CODEX_HOME".into(), path.display().to_string()))
                }
                Harness::Claude => {
                    environment.push(("CLAUDE_CONFIG_DIR".into(), path.display().to_string()))
                }
                _ => {}
            }
        }
        crate::windows::open_terminal(crate::windows::TerminalJob {
            binary,
            args,
            directory: Some(worktree.join(&relative)),
            environment,
            remove_environment,
        })?;
    }
    #[cfg(not(windows))]
    {
        let program = quote(&binary.display().to_string());
        let command = match tool {
            Harness::CodexCli => format!(
                "{program} -c {} -c {} -c {}",
                quote(&format!("model={}", toml_edit::Value::from(model))),
                quote("model_provider=\"sy\""),
                quote(&format!(
                    "model_providers.sy={{name=\"Switchyard\",base_url={},wire_api=\"responses\",requires_openai_auth={login},http_headers={{\"x-switchyard-session-id\"={},\"x-switchyard-origin\"=\"codex\"}}}}",
                    toml_edit::Value::from(format!("{}/v1", root.trim_end_matches('/'))),
                    toml_edit::Value::from(id.as_str())
                ))
            ),
            Harness::Claude => format!(
                "env ANTHROPIC_CUSTOM_HEADERS={} {program} --settings {} --model {}",
                quote(&format!(
                    "x-switchyard-session-id: {id}\nx-switchyard-origin: claude"
                )),
                quote(&files[0].display().to_string()),
                quote(model)
            ),
            Harness::Pi => format!(
                "env PI_CODING_AGENT_DIR={} {program} --provider switchyard --model {}",
                quote(&private.display().to_string()),
                quote(model)
            ),
            Harness::CodexApp => return Err("Choose a CLI tool.".into()),
        };
        let default_files = harness::paths(tool);
        let account = account.or_else(|| match tool {
            Harness::Claude if std::env::var_os("CLAUDE_CONFIG_DIR").is_none() => None,
            _ => default_files[0].parent(),
        });
        let account_env = match (tool, account) {
            (Harness::CodexCli, Some(path)) => {
                format!("export CODEX_HOME={}\n", quote(&path.display().to_string()))
            }
            (Harness::Claude, Some(path)) => format!(
                "export CLAUDE_CONFIG_DIR={}\n",
                quote(&path.display().to_string())
            ),
            _ => String::new(),
        };
        let subscription_env = if tool == Harness::Claude && login {
            "unset ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY CLAUDE_CODE_OAUTH_TOKEN\n"
        } else {
            ""
        };
        let script = format!(
            "#!/bin/bash\nexport PATH=\"$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH\"\nrm -- \"$0\"\ncd {} || exit 1\n{account_env}{subscription_env}{command}",
            quote(&worktree.join(relative).display().to_string())
        );
        crate::server::open_terminal_script("switchyard-session-", &script)?;
    }
    pending.opened = true;
    Ok(format!(
        "Opened Terminal to start the session.\nWorktree: {}\nBranch: {id}\nModel route: {model}\nRemove the worktree with git worktree remove after saving or discarding its changes.",
        worktree.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_session_setup_removes_its_worktree_branch_and_private_settings() {
        let dir = tempfile::tempdir().expect("directory");
        let project = dir.path().join("project");
        std::fs::create_dir(&project).expect("project");
        let git = |args: &[&str]| {
            let output = std::process::Command::new("git")
                .arg("-C")
                .arg(&project)
                .args(args)
                .output()
                .expect("git");
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            output
        };
        git(&["init"]);
        git(&[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-m",
            "fixture",
        ]);
        let worktree = dir.path().join("worktree");
        let private = dir.path().join("settings");
        git(&[
            "worktree",
            "add",
            "-b",
            "session-fixture",
            worktree.to_str().expect("path"),
        ]);
        std::fs::create_dir(&private).expect("private");
        std::fs::write(private.join("settings.json"), "{}").expect("settings");
        drop(PendingSession {
            project: project.clone(),
            worktree: worktree.clone(),
            private: private.clone(),
            branch: "session-fixture".into(),
            opened: false,
        });
        assert!(!worktree.exists());
        assert!(!private.exists());
        assert!(
            !String::from_utf8_lossy(&git(&["branch", "--list", "session-fixture"]).stdout)
                .contains("session-fixture")
        );
    }
    // The subprocess isolates HOME and PATH.
    // The Terminal stub injects a conflicting directory when the script must select one.
    // Claude's default case requires an unset CLAUDE_CONFIG_DIR, not an explicit ~/.claude.
    #[cfg(unix)]
    #[test]
    fn generated_sessions_use_explicit_accounts_without_changing_user_defaults() {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(fixture) = std::env::var("SWITCHYARD_SESSION_FIXTURE") {
            let fixture = std::path::PathBuf::from(fixture);
            let tool = if std::env::var("SWITCHYARD_SESSION_TOOL").expect("tool") == "codex" {
                Harness::CodexCli
            } else {
                Harness::Claude
            };
            let account =
                std::env::var_os("SWITCHYARD_SESSION_ACCOUNT").map(std::path::PathBuf::from);
            let revision = preview(&fixture.join("project")).expect("preview")["revision"]
                .as_str()
                .expect("revision")
                .to_string();
            // Advancing the original checkout makes a launch from HEAD differ from the review.
            if tool == Harness::CodexCli && account.is_none() {
                let committed = std::process::Command::new("git")
                    .arg("-C")
                    .arg(fixture.join("project"))
                    .args([
                        "-c",
                        "user.name=Fixture",
                        "-c",
                        "user.email=fixture@example.invalid",
                        "-c",
                        "commit.gpgsign=false",
                        "commit",
                        "--allow-empty",
                        "-m",
                        "advanced",
                    ])
                    .output()
                    .expect("advance HEAD");
                assert!(committed.status.success());
            }
            launch(
                tool,
                &fixture.join("project"),
                "route'$(literal)",
                "http://localhost:4123",
                tool == Harness::Claude,
                account.as_deref(),
                &revision,
            )
            .expect("launch");
            if tool == Harness::CodexCli && account.is_none() {
                let records = inventory().expect("inventory");
                assert_eq!(records.len(), 1);
                assert_eq!(records[0]["revision"], revision);
                assert_ne!(
                    preview(&fixture.join("project")).expect("advanced project")["revision"],
                    revision
                );
                let id = records[0]["id"].as_str().expect("session id");
                let receipt = session_root().join(format!("{id}-settings/session.json"));
                let original = std::fs::read(&receipt).expect("receipt");
                // Valid JSON roots must still be objects before inventory adds process_status.
                for malformed in ["null", "[]", "false", "42"] {
                    std::fs::write(&receipt, malformed).expect("malformed receipt");
                    assert!(inventory().is_err());
                    assert!(cleanup_preview(id).is_err());
                }
                std::fs::write(&receipt, original).expect("restore receipt");
                let clean = cleanup_preview(id).expect("preview cleanup");
                let worktree = Path::new(records[0]["worktree"].as_str().expect("worktree"));
                let unsaved = worktree.join("unsaved-user-work.txt");
                std::fs::write(&unsaved, "keep this work").expect("user work");
                assert!(cleanup(id, &clean).is_err());
                assert!(unsaved.exists());
                let dirty = cleanup_preview(id).expect("dirty preview");
                assert!(cleanup(id, &dirty).is_err());
                assert!(worktree.exists());
                std::fs::remove_file(&unsaved).expect("save fixture work");
                let clean = cleanup_preview(id).expect("clean preview");
                cleanup(id, &clean).expect("safe cleanup");
                assert!(!worktree.exists());
                assert!(inventory().expect("inventory after cleanup").is_empty());
            }
            return;
        }
        for (name, named, inherited_dir) in [
            ("codex", false, true),
            ("codex", true, true),
            ("claude", false, true),
            ("claude", true, true),
            ("claude", false, false),
            ("claude", true, false),
        ] {
            let fixture = tempfile::tempdir().expect("fixture");
            let home = fixture.path().join("home");
            let bin = fixture.path().join("bin");
            let project = fixture.path().join("project");
            for path in [&home, &bin, &project] {
                std::fs::create_dir(path).expect("directory");
            }
            for args in [
                vec!["init"],
                vec![
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--allow-empty",
                    "-m",
                    "fixture",
                ],
            ] {
                assert!(
                    std::process::Command::new("git")
                        .arg("-C")
                        .arg(&project)
                        .args(args)
                        .output()
                        .expect("git")
                        .status
                        .success()
                );
            }
            let default = home.join("default-login");
            std::fs::create_dir(&default).expect("default");
            std::fs::write(default.join("settings.json"), "{\"user\":true}").expect("settings");
            let account = home.join(".switchyard/accounts").join(name).join("named");
            if named {
                std::fs::create_dir_all(&account).expect("account");
            }
            let output = fixture.path().join("arguments");
            let captured = fixture.path().join("session.command");
            let open = bin.join("open");
            let terminal_claude = if inherited_dir || named {
                "export CLAUDE_CONFIG_DIR=wrong-terminal-login"
            } else {
                "unset CLAUDE_CONFIG_DIR"
            };
            std::fs::write(&open, format!("#!/bin/bash\ncp -- \"$1\" {}\nexport CODEX_HOME=wrong-terminal-login ANTHROPIC_AUTH_TOKEN=wrong-token\n{terminal_claude}\nbash \"$1\"\n",quote(&captured.display().to_string()))).expect("open");
            let tool = bin.join(name);
            std::fs::write(&tool, format!("#!/bin/bash\nprintf '%s\\n' \"${{CODEX_HOME:-}}\" \"${{CLAUDE_CONFIG_DIR:-}}\" \"${{ANTHROPIC_AUTH_TOKEN:-}}\" \"$@\" > {}\n", quote(&output.display().to_string()))).expect("tool");
            for path in [&open, &tool] {
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                    .expect("executable");
            }
            let mut child = std::process::Command::new(std::env::current_exe().expect("runner"));
            child.args(["--exact", "sessions::tests::generated_sessions_use_explicit_accounts_without_changing_user_defaults"])
                    .env("SWITCHYARD_SESSION_FIXTURE", fixture.path()).env("SWITCHYARD_SESSION_TOOL", name)
                    .env("HOME", &home).env("CODEX_HOME", &default).env("CLAUDE_CONFIG_DIR", &default)
                    .env("PATH", format!("{}:/usr/bin:/bin", bin.display()));
            if named {
                child.env("SWITCHYARD_SESSION_ACCOUNT", &account);
            }
            if !inherited_dir {
                child.env_remove("CLAUDE_CONFIG_DIR");
            }
            let result = child.output().expect("child");
            assert!(
                result.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            let arguments = std::fs::read_to_string(output).expect("arguments");
            let lines: Vec<_> = arguments.lines().collect();
            let expected = if named {
                account.to_str().expect("path")
            } else if inherited_dir {
                default.to_str().expect("path")
            } else {
                ""
            };
            assert_eq!(lines[usize::from(name == "claude")], expected);
            if name == "claude" {
                assert_eq!(lines[2], "");
            }
            assert!(arguments.contains("route'$(literal)"));
            let script = std::fs::read_to_string(captured).expect("script");
            assert!(script.contains("x-switchyard-session-id"));
            if name == "claude" && !named && !inherited_dir {
                assert!(!script.contains("export CLAUDE_CONFIG_DIR="));
            }
            assert_eq!(
                std::fs::read_to_string(default.join("settings.json")).expect("settings"),
                "{\"user\":true}"
            );
        }
    }
}
