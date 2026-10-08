// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module starts CLI sessions with separate worktrees and model settings.

use crate::harness::{self, Harness};
use std::io::Write;
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

pub fn launch(
    tool: Harness,
    project: &Path,
    model: &str,
    root: &str,
    login: bool,
    account: Option<&Path>,
) -> Result<String, String> {
    let root = harness::validate(tool, root, model, login)?;
    if let Some(account) = account {
        crate::accounts::validate(tool, account)?;
    }
    if tool == Harness::CodexApp {
        return Err("Choose Codex CLI to launch a worktree session. Codex app uses its own workspace controls.".into());
    }
    let binary = harness::binary(tool).ok_or("Install the selected coding tool first.")?;
    let project = std::fs::canonicalize(project).map_err(|e| format!("Open project: {e}"))?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&project)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("Choose a Git checkout directory.".into());
    }
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
        .arg("HEAD")
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
    let account = account.or_else(|| default_files[0].parent());
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
    let mut script = tempfile::Builder::new()
        .prefix("switchyard-session-")
        .suffix(".command")
        .tempfile()
        .map_err(|e| e.to_string())?;
    writeln!(script,"#!/bin/bash\nexport PATH=\"$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH\"\nrm -- \"$0\"\ncd {} || exit 1\n{account_env}{subscription_env}{command}",quote(&worktree.display().to_string())).map_err(|e|e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        script
            .as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o700))
            .map_err(|e| e.to_string())?;
    }
    crate::server::command("open", &[&script.path().display().to_string()])?;
    pending.opened = true;
    let _ = script.keep().map_err(|e| e.to_string())?;
    Ok(format!(
        "Opened a new session in Terminal.\nWorktree: {}\nBranch: {id}\nModel route: {model}\nRemove the worktree with git worktree remove after saving or discarding its changes.\nPi supplies session/turn IDs only when its integration sends them; unidentified calls remain visible in All sessions.",
        worktree.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quotes_shell_metacharacters_literally() {
        assert_eq!(quote("a'b $(touch /tmp/no)"), "'a'\\''b $(touch /tmp/no)'");
    }
    #[test]
    fn rejected_launch_inputs_create_no_worktree_or_settings() {
        let dir = tempfile::tempdir().expect("directory");
        assert!(
            launch(
                Harness::CodexCli,
                dir.path(),
                "route",
                "https://remote.example",
                true,
                None
            )
            .is_err()
        );
        assert!(
            launch(
                Harness::Pi,
                dir.path(),
                "route",
                "http://localhost:4123",
                true,
                None
            )
            .is_err()
        );
        assert!(
            std::fs::read_dir(dir.path())
                .expect("contents")
                .next()
                .is_none()
        );
    }

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
    // The subprocess isolates HOME and PATH. Its Terminal stub supplies a
    // conflicting login to check that the generated script selects the account.
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
            launch(
                tool,
                &fixture.join("project"),
                "route'$(literal)",
                "http://localhost:4123",
                tool == Harness::Claude,
                account.as_deref(),
            )
            .expect("launch");
            return;
        }
        for (name, named) in [
            ("codex", false),
            ("codex", true),
            ("claude", false),
            ("claude", true),
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
            std::fs::write(&open, format!("#!/bin/bash\ncp -- \"$1\" {}\nexport CODEX_HOME=wrong-terminal-login CLAUDE_CONFIG_DIR=wrong-terminal-login ANTHROPIC_AUTH_TOKEN=wrong-token\nbash \"$1\"\n",quote(&captured.display().to_string()))).expect("open");
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
            let result = child.output().expect("child");
            assert!(
                result.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            let arguments = std::fs::read_to_string(output).expect("arguments");
            let lines: Vec<_> = arguments.lines().collect();
            let expected = if named { &account } else { &default };
            assert_eq!(
                lines[usize::from(name == "claude")],
                expected.to_str().expect("path")
            );
            if name == "claude" {
                assert_eq!(lines[2], "");
            }
            assert!(arguments.contains("route'$(literal)"));
            let script = std::fs::read_to_string(captured).expect("script");
            assert!(script.contains("x-switchyard-session-id"));
            assert_eq!(
                std::fs::read_to_string(default.join("settings.json")).expect("settings"),
                "{\"user\":true}"
            );
        }
    }
}
