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
    revision: String,
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
        delete_session_branch(&self.project, &self.branch, &self.revision);
        let _ = std::fs::remove_dir_all(&self.private);
    }
}

// This deletes the session branch only while it still points at the session's starting commit.
// The commit check makes `branch -D` safe. `branch -d` also requires a merge into the original
// checkout's HEAD, so it fails after the user switches that checkout to an unrelated branch.
fn delete_session_branch(repository: &Path, branch: &str, revision: &str) -> bool {
    let tip = std::process::Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(["rev-parse", "--verify", "--quiet"])
        .arg(format!("refs/heads/{branch}"))
        .output();
    tip.is_ok_and(|tip| {
        tip.status.success() && String::from_utf8_lossy(&tip.stdout).trim() == revision
    }) && std::process::Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(["branch", "-D", branch])
        .output()
        .is_ok_and(|output| output.status.success())
}

pub(crate) fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

pub fn preview(project: &Path) -> Result<serde_json::Value, String> {
    if project.as_os_str().is_empty() {
        return Err("Choose a Git checkout directory.".into());
    }
    let project = std::fs::canonicalize(crate::config::expand_home(project)?)
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
    Ok(serde_json::json!({
        "repository": repository,
        "project": project,
        "relative_directory": relative,
        "revision": git(&["rev-parse", "HEAD"])?,
        "branch": git(&["branch", "--show-current"])?,
        "dirty": !git(&["status", "--porcelain"])?.is_empty(),
        "notice": "The session starts from committed HEAD. \
                   Uncommitted changes stay in the original checkout."
    }))
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

pub fn managed_session(id: &str) -> Result<serde_json::Value, String> {
    let suffix = id.strip_prefix("sy-").ok_or("Choose a managed session.")?;
    uuid::Uuid::parse_str(suffix).map_err(|_| "Choose a managed session.")?;
    let root = session_root()?;
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
    // Plain `git worktree remove` deletes ignored files, so the review lists them.
    let status = run(&["status", "--porcelain", "--ignored=matching"])?;
    receipt["dirty"] = serde_json::json!(status.lines().any(|line| !line.starts_with("!! ")));
    receipt["ignored"] = serde_json::json!(
        status
            .lines()
            .filter_map(|line| line.strip_prefix("!! "))
            .collect::<Vec<_>>()
    );
    receipt["current_revision"] = serde_json::json!(run(&["rev-parse", "HEAD"])?);
    receipt["current_branch"] = serde_json::json!(run(&["branch", "--show-current"])?);
    receipt["cleanup_notice"] = serde_json::json!(
        "Cleanup removes a clean worktree and its unchanged session branch. \
         It also deletes the Git-ignored files in the worktree, which the ignored list names. \
         Recorded usage stays in the routing log. Unknown private files are preserved."
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
        return Err("Save or move the session’s work before cleanup. \
                    Cleanup requires a clean worktree with its original commit and branch."
            .into());
    }
    let repository = current["repository"]
        .as_str()
        .ok_or("Missing session repository.")?;
    let worktree = current["worktree"]
        .as_str()
        .ok_or("Missing session worktree.")?;
    let revision = current["revision"]
        .as_str()
        .ok_or("Missing session commit.")?;
    // Receipt path resolution must succeed before cleanup removes the worktree.
    let private = session_root()?.join(format!("{id}-settings"));
    let output = std::process::Command::new("git")
        .args(["-C", repository, "worktree", "remove", worktree])
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }

    // The worktree is gone, so the receipt goes too; otherwise the inventory keeps the session.
    // Cleanup removes only the receipt because settings can contain later user edits.
    std::fs::remove_file(private.join("session.json")).map_err(|error| error.to_string())?;
    let _ = std::fs::remove_dir(&private);
    if !delete_session_branch(Path::new(repository), id, revision) {
        return Ok(format!(
            "Removed the clean session worktree. Git kept the branch {id}; \
             inspect it before deleting it. Retained private settings and recorded usage."
        ));
    }
    Ok("Removed the clean session worktree and branch. \
        Retained private settings and recorded usage."
        .into())
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
        return Err("Choose Codex CLI to launch a worktree session. \
                    Codex app uses its own workspace controls."
            .into());
    }
    let binary = harness::binary(tool).ok_or("Install the selected harness first.")?;
    if project.as_os_str().is_empty() {
        return Err("Choose a Git checkout directory.".into());
    }
    let project = crate::config::expand_home(project)?;
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
    // Path resolution must succeed before launch creates directories or a worktree.
    let default_files = harness::paths(tool)?;
    let worktree = session_root()?.join(&id);
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
        revision: revision.into(),
        opened: false,
    };
    let files = match tool {
        Harness::Pi => vec![private.join("models.json"), private.join("settings.json")],
        _ => vec![private.join("settings.json")],
    };
    std::fs::create_dir_all(&private).map_err(|error| error.to_string())?;
    if matches!(tool, Harness::Claude | Harness::Pi) {
        harness::install(tool, &files, &root, model, None, login)?;
    }
    let receipt = serde_json::json!({"id":id,"tool":tool,"route":model,"repository":repository,"revision":preview(&worktree)?["revision"],"worktree":worktree,"branch":id,"created_at":chrono::Utc::now().to_rfc3339(),"settings_directory":private,"account_directory":account});
    std::fs::write(
        private.join("session.json"),
        serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let account = account.or_else(|| match tool {
        Harness::Claude if std::env::var_os("CLAUDE_CONFIG_DIR").is_none() => None,
        _ => default_files[0].parent(),
    });
    let codex_args = [
        "-c".to_string(),
        format!("model={}", toml_edit::Value::from(model)),
        "-c".into(),
        "model_provider=\"sy\"".into(),
        "-c".into(),
        format!(
            "model_providers.sy={{name=\"Switchyard\",base_url={},wire_api=\"responses\",\
             requires_openai_auth={login},http_headers={{\"x-switchyard-session-id\"={},\
             \"x-switchyard-origin\"=\"codex\"}}}}",
            toml_edit::Value::from(format!("{}/v1", root.trim_end_matches('/'))),
            toml_edit::Value::from(id.as_str())
        ),
    ];
    #[cfg(windows)]
    {
        let mut environment = Vec::new();
        let mut remove_environment = Vec::new();
        let args = match tool {
            Harness::CodexCli => codex_args.into(),
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
            Harness::CodexApp => return Err("Choose a CLI harness.".into()),
        };
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
                "{program} {}",
                codex_args
                    .iter()
                    .map(|arg| quote(arg))
                    .collect::<Vec<_>>()
                    .join(" ")
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
            Harness::CodexApp => return Err("Choose a CLI harness.".into()),
        };
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
            "#!/bin/bash\n\
             export PATH=\"$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH\"\n\
             rm -- \"$0\"\n\
             cd {} || exit 1\n\
             {account_env}{subscription_env}{command}",
            quote(&worktree.join(relative).display().to_string())
        );
        crate::server::open_terminal_script("switchyard-session-", &script)?;
    }
    pending.opened = true;
    Ok(format!(
        "Opened Terminal to start the session.\n\
         Worktree: {}\n\
         Branch: {id}\n\
         Model route: {model}\n\
         Remove the worktree with git worktree remove after saving or discarding its changes.",
        worktree.display()
    ))
}
