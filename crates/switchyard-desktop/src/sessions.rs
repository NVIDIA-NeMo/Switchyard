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

/// A harness opened in a directory that already exists.
///
/// `launch` creates a worktree and then opens the harness in it. A hand-off later opens
/// another harness in the same directory, so the two steps stay separate.
pub struct OpenRequest<'a> {
    pub tool: Harness,
    pub binary: std::path::PathBuf,
    /// The directory the harness starts in.
    pub directory: std::path::PathBuf,
    /// The public route ID the harness uses as its model.
    pub model: &'a str,
    /// The validated server URL.
    pub root: &'a str,
    pub login: bool,
    /// The login directory the script exports, when the harness must select one.
    pub account: Option<&'a Path>,
    /// The value of the `x-switchyard-session-id` header.
    pub session_id: &'a str,
    /// The private settings files written for Claude and Pi.
    pub files: &'a [std::path::PathBuf],
    /// Pi's private agent directory.
    pub private: &'a Path,
    /// Claude's session UUID. Choosing it here makes the transcript path known before the
    /// session starts.
    pub native_session_id: Option<&'a str>,
    /// Instructions the harness reads before the first turn: Codex `developer_instructions`,
    /// Claude `--append-system-prompt`. A hand-off points at its file here.
    pub instructions: Option<&'a str>,
    /// Directories outside the session directory that Claude may read without asking.
    pub read_dirs: &'a [std::path::PathBuf],
    /// A first prompt that starts the session's first turn at once.
    pub first_prompt: Option<&'a str>,
}

/// Claude Code accepts a UUID for `--session-id`. Its own session IDs are version 4, so the
/// bytes of a time-ordered UUID are relabeled as version 4 rather than enabling another
/// `uuid` feature.
pub fn claude_session_id() -> String {
    uuid::Builder::from_random_bytes(*uuid::Uuid::now_v7().as_bytes())
        .into_uuid()
        .to_string()
}

fn create_worktree(
    project: &Path,
    id: &str,
    revision: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf, PendingSession), String> {
    let worktree = session_root()?.join(id);
    let parent = worktree
        .parent()
        .ok_or("Worktree needs a parent directory")?
        .to_path_buf();
    std::fs::create_dir_all(&parent).map_err(|e| e.to_string())?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(project)
        .args(["worktree", "add", "-b"])
        .arg(id)
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
    let private = parent.join(format!("{id}-settings"));
    let pending = PendingSession {
        project: project.to_path_buf(),
        worktree: worktree.clone(),
        private: private.clone(),
        branch: id.to_string(),
        revision: revision.into(),
        opened: false,
    };
    Ok((worktree, private, pending))
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
    harness::paths(tool)?;
    let (worktree, private, mut pending) = create_worktree(&project, &id, revision)?;
    let files = match tool {
        Harness::Pi => vec![private.join("models.json"), private.join("settings.json")],
        _ => vec![private.join("settings.json")],
    };
    std::fs::create_dir_all(&private).map_err(|error| error.to_string())?;
    if matches!(tool, Harness::Claude | Harness::Pi) {
        harness::install(tool, &files, &root, model, None, login)?;
    }
    let native_session_id = (tool == Harness::Claude).then(claude_session_id);
    let created_at = chrono::Utc::now().to_rfc3339();
    // `launches` records every harness opened in this worktree. The top-level `tool`,
    // `route`, `account_directory`, and `settings_directory` mirror the latest launch.
    let launched = serde_json::json!({
        "tool": tool,
        "route": model,
        "account_directory": account,
        "settings_directory": private,
        "started_at": created_at,
        "native_session_id": native_session_id,
    });
    let receipt = serde_json::json!({
        "id": id,
        "tool": tool,
        "route": model,
        "repository": repository,
        "revision": preview(&worktree)?["revision"],
        "worktree": worktree,
        "branch": id,
        "relative_directory": relative,
        "created_at": created_at,
        "settings_directory": private,
        "account_directory": account,
        "launches": [launched],
    });
    std::fs::write(
        private.join("session.json"),
        serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let account = script_account(tool, account)?;
    open_harness(&OpenRequest {
        tool,
        binary,
        directory: worktree.join(&relative),
        model,
        root: &root,
        login,
        account: account.as_deref(),
        session_id: &id,
        files: &files,
        private: &private,
        native_session_id: native_session_id.as_deref(),
        instructions: None,
        read_dirs: &[],
        first_prompt: None,
    })?;
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

/// This records another harness opened in a managed session's worktree. The top-level `tool`,
/// `route`, `account_directory`, and `settings_directory` follow the latest launch.
pub fn append_launch(id: &str, launched: &serde_json::Value) -> Result<(), String> {
    let mut receipt = managed_session(id)?;
    let path = session_root()?.join(format!("{id}-settings/session.json"));
    let launches = receipt
        .as_object_mut()
        .ok_or("The session receipt does not match this worktree.")?
        .entry("launches")
        .or_insert_with(|| serde_json::json!([]));
    launches
        .as_array_mut()
        .ok_or("The session receipt lists launches in an unexpected form.")?
        .push(launched.clone());
    for key in ["tool", "route", "account_directory", "settings_directory"] {
        receipt[key] = launched[key].clone();
    }
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("Write {}: {error}", path.display()))
}

/// The arguments and environment that start a harness. `open_harness` runs them in a terminal;
/// a review window shows them first.
pub struct Invocation {
    pub args: Vec<String>,
    pub environment: Vec<(String, String)>,
    pub remove_environment: Vec<String>,
}

/// The login directory the launch exports: the named account, or the harness's current
/// directory. Claude keeps an unset `CLAUDE_CONFIG_DIR` unset, so its default login stays
/// the user's default.
pub fn script_account(
    tool: Harness,
    account: Option<&Path>,
) -> Result<Option<std::path::PathBuf>, String> {
    if let Some(account) = account {
        return Ok(Some(account.to_path_buf()));
    }
    let default_files = harness::paths(tool)?;
    Ok(match tool {
        Harness::Claude if std::env::var_os("CLAUDE_CONFIG_DIR").is_none() => None,
        _ => default_files[0].parent().map(Path::to_path_buf),
    })
}

/// This builds the harness command for `request` without opening anything.
pub fn invocation(request: &OpenRequest) -> Result<Invocation, String> {
    let OpenRequest {
        tool,
        binary: _,
        directory: _,
        model,
        root,
        login,
        account,
        session_id,
        files,
        private,
        native_session_id,
        instructions,
        read_dirs,
        first_prompt,
    } = request;
    let (tool, model, login, session_id) = (*tool, *model, *login, *session_id);
    let mut codex_args = vec![
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
            toml_edit::Value::from(session_id)
        ),
    ];
    if let Some(instructions) = instructions {
        codex_args.extend([
            "-c".to_string(),
            format!(
                "developer_instructions={}",
                toml_edit::Value::from(*instructions)
            ),
        ]);
    }
    if let Some(prompt) = first_prompt {
        codex_args.push((*prompt).to_string());
    }
    // `--add-dir` takes every following word as a directory, so it goes first and a flag that
    // takes one value closes it. The prompt is the last word.
    let mut claude_args = Vec::new();
    for dir in read_dirs.iter() {
        claude_args.extend(["--add-dir".to_string(), dir.display().to_string()]);
    }
    claude_args.extend([
        "--settings".to_string(),
        files[0].display().to_string(),
        "--model".into(),
        model.into(),
    ]);
    if let Some(native) = native_session_id {
        claude_args.extend(["--session-id".to_string(), (*native).to_string()]);
    }
    if let Some(instructions) = instructions {
        claude_args.extend([
            "--append-system-prompt".to_string(),
            (*instructions).to_string(),
        ]);
    }
    if let Some(prompt) = first_prompt {
        claude_args.push((*prompt).to_string());
    }
    let mut environment = Vec::new();
    let mut remove_environment = Vec::new();
    let args = match tool {
        Harness::CodexCli => codex_args,
        Harness::Claude => {
            environment.push((
                "ANTHROPIC_CUSTOM_HEADERS".to_string(),
                format!("x-switchyard-session-id: {session_id}\nx-switchyard-origin: claude"),
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
            claude_args
        }
        Harness::Pi => {
            environment.push((
                "PI_CODING_AGENT_DIR".to_string(),
                private.display().to_string(),
            ));
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
    Ok(Invocation {
        args,
        environment,
        remove_environment,
    })
}

/// The shell lines that `open_harness` runs, for a review window. Windows runs the same
/// program, arguments, and environment through its own console.
pub fn display_command(request: &OpenRequest) -> Result<String, String> {
    let invocation = invocation(request)?;
    let mut text = format!("cd {}", quote(&request.directory.display().to_string()));
    for (name, value) in &invocation.environment {
        text.push_str(&format!("\nexport {name}={}", quote(value)));
    }
    if !invocation.remove_environment.is_empty() {
        text.push_str(&format!(
            "\nunset {}",
            invocation.remove_environment.join(" ")
        ));
    }
    text.push('\n');
    text.push_str(&quote(&request.binary.display().to_string()));
    for arg in &invocation.args {
        text.push(' ');
        text.push_str(&quote(arg));
    }
    Ok(text)
}

/// This opens the harness in `request.directory` through the user's terminal.
pub fn open_harness(request: &OpenRequest) -> Result<(), String> {
    let invocation = invocation(request)?;
    #[cfg(windows)]
    {
        crate::windows::open_terminal(crate::windows::TerminalJob {
            binary: request.binary.clone(),
            args: invocation.args,
            directory: Some(request.directory.clone()),
            environment: invocation.environment,
            remove_environment: invocation.remove_environment,
        })?;
    }
    #[cfg(not(windows))]
    {
        let mut environment = String::new();
        for (name, value) in &invocation.environment {
            environment.push_str(&format!("export {name}={}\n", quote(value)));
        }
        if !invocation.remove_environment.is_empty() {
            environment.push_str(&format!(
                "unset {}\n",
                invocation.remove_environment.join(" ")
            ));
        }
        let command = std::iter::once(quote(&request.binary.display().to_string()))
            .chain(invocation.args.iter().map(|arg| quote(arg)))
            .collect::<Vec<_>>()
            .join(" ");
        let script = format!(
            "#!/bin/bash\n\
             export PATH=\"$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH\"\n\
             rm -- \"$0\"\n\
             cd {} || exit 1\n\
             {environment}{command}",
            quote(&request.directory.display().to_string())
        );
        crate::server::open_terminal_script("switchyard-session-", &script)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Claude Code's `--add-dir` swallows every following word as a directory, which once made
    // the first prompt a "path not found". Codex takes one value per `-c`.
    #[test]
    fn the_first_prompt_is_the_last_word_and_no_list_option_precedes_it() {
        let dir = std::path::PathBuf::from("/tmp/hd-1");
        let files = vec![dir.join("settings.json")];
        let request = |tool: Harness| OpenRequest {
            tool,
            binary: std::path::PathBuf::from("/usr/bin/harness"),
            directory: std::path::PathBuf::from("/work"),
            model: "claude",
            root: "http://127.0.0.1:4123",
            login: true,
            account: None,
            session_id: "sy-1",
            files: &files,
            private: &dir,
            native_session_id: Some("00000000-0000-4000-8000-000000000001"),
            instructions: Some("Read the file first."),
            read_dirs: std::slice::from_ref(&dir),
            first_prompt: Some("Read the hand-off file and continue."),
        };
        let claude = invocation(&request(Harness::Claude)).expect("claude").args;
        assert_eq!(&claude[..2], &["--add-dir", "/tmp/hd-1"]);
        assert_eq!(claude[2], "--settings");
        assert_eq!(
            claude.last().map(String::as_str),
            Some("Read the hand-off file and continue.")
        );
        assert_eq!(claude[claude.len() - 3], "--append-system-prompt");
        let codex = invocation(&request(Harness::CodexCli)).expect("codex").args;
        assert_eq!(
            codex.last().map(String::as_str),
            Some("Read the hand-off file and continue.")
        );
        assert!(
            codex[codex.len() - 2].starts_with("developer_instructions="),
            "{codex:?}"
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
        let revision = String::from_utf8_lossy(&git(&["rev-parse", "HEAD"]).stdout)
            .trim()
            .to_string();
        // An unrelated HEAD in the original checkout makes `branch -d` refuse the session branch.
        git(&["checkout", "--orphan", "unrelated"]);
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
            "unrelated",
        ]);
        drop(PendingSession {
            project: project.clone(),
            worktree: worktree.clone(),
            private: private.clone(),
            branch: "session-fixture".into(),
            revision,
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
                assert_eq!(records[0]["relative_directory"], "");
                let launches = records[0]["launches"].as_array().expect("launches");
                assert_eq!(launches.len(), 1);
                assert_eq!(launches[0]["tool"], "codex_cli");
                assert_eq!(launches[0]["route"], records[0]["route"]);
                assert_eq!(launches[0]["started_at"], records[0]["created_at"]);
                assert!(launches[0]["native_session_id"].is_null());
                assert_ne!(
                    preview(&fixture.join("project")).expect("advanced project")["revision"],
                    revision
                );
                let id = records[0]["id"].as_str().expect("session id");
                let receipt = session_root()
                    .expect("session root")
                    .join(format!("{id}-settings/session.json"));
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
                // Cleanup deletes ignored files, so a new ignored file requires a new review.
                let info = fixture.join("project/.git/info");
                std::fs::create_dir_all(&info).expect("git info");
                std::fs::write(info.join("exclude"), "ignored-notes.txt\n").expect("exclude");
                std::fs::write(worktree.join("ignored-notes.txt"), "notes").expect("notes");
                assert!(cleanup(id, &clean).is_err());
                assert!(worktree.join("ignored-notes.txt").exists());
                let reviewed = cleanup_preview(id).expect("ignored preview");
                assert_eq!(
                    reviewed["ignored"],
                    serde_json::json!(["ignored-notes.txt"])
                );
                // An unrelated HEAD in the original checkout makes `branch -d` refuse the branch.
                let project_git = |args: &[&str]| {
                    let output = std::process::Command::new("git")
                        .arg("-C")
                        .arg(fixture.join("project"))
                        .args(args)
                        .output()
                        .expect("git");
                    assert!(output.status.success());
                    output
                };
                project_git(&["checkout", "--orphan", "unrelated"]);
                project_git(&[
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--allow-empty",
                    "-m",
                    "unrelated",
                ]);
                cleanup(id, &reviewed).expect("safe cleanup");
                assert!(!worktree.exists());
                assert!(inventory().expect("inventory after cleanup").is_empty());
                assert!(project_git(&["branch", "--list", id]).stdout.is_empty());
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
            // macOS runs the script through `open`; Linux runs it through the program in
            // TERMINAL, which receives `-e` before the script path.
            let open = bin.join(if cfg!(target_os = "linux") {
                "terminal"
            } else {
                "open"
            });
            let terminal_claude = if inherited_dir || named {
                "export CLAUDE_CONFIG_DIR=wrong-terminal-login"
            } else {
                "unset CLAUDE_CONFIG_DIR"
            };
            std::fs::write(
                &open,
                format!(
                    "#!/bin/bash\n\
                     [ \"$1\" = -e ] && shift\n\
                     cp -- \"$1\" {}\n\
                     export CODEX_HOME=wrong-terminal-login ANTHROPIC_AUTH_TOKEN=wrong-token\n\
                     {terminal_claude}\n\
                     bash \"$1\"\n",
                    quote(&captured.display().to_string())
                ),
            )
            .expect("open");
            let tool = bin.join(name);
            std::fs::write(
                &tool,
                format!(
                    "#!/bin/bash\n\
                     printf '%s\\n' \"${{CODEX_HOME:-}}\" \"${{CLAUDE_CONFIG_DIR:-}}\" \
                     \"${{ANTHROPIC_AUTH_TOKEN:-}}\" \"$@\" > {0}.partial && mv {0}.partial {0}\n",
                    quote(&output.display().to_string())
                ),
            )
            .expect("tool");
            for path in [&open, &tool] {
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                    .expect("executable");
            }
            let mut child = std::process::Command::new(std::env::current_exe().expect("runner"));
            child
                .args([
                    "--exact",
                    "sessions::tests::\
                     generated_sessions_use_explicit_accounts_without_changing_user_defaults",
                ])
                .env("SWITCHYARD_SESSION_FIXTURE", fixture.path())
                .env("SWITCHYARD_SESSION_TOOL", name)
                .env("HOME", &home)
                .env("CODEX_HOME", &default)
                .env("CLAUDE_CONFIG_DIR", &default)
                .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
                .env("TERMINAL", &open);
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
            // A Linux terminal starts apart from the launch, so the arguments can arrive
            // after the child exits. The tool stub renames the file into place when done.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !output.exists() && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
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
            // The receipt names the Claude session UUID that the launch passed on the command
            // line, so the transcript path is known without reading any transcript.
            let receipts: Vec<_> = std::fs::read_dir(home.join(".switchyard/worktrees"))
                .expect("worktrees")
                .filter_map(|entry| {
                    let path = entry.expect("entry").path().join("session.json");
                    std::fs::read(&path).ok().map(|bytes| {
                        serde_json::from_slice::<serde_json::Value>(&bytes).expect("json")
                    })
                })
                .collect();
            if name == "claude" {
                assert_eq!(receipts.len(), 1);
                let native = receipts[0]["launches"][0]["native_session_id"]
                    .as_str()
                    .expect("claude session id");
                assert_eq!(
                    uuid::Uuid::parse_str(native)
                        .expect("uuid")
                        .get_version_num(),
                    4
                );
                assert!(arguments.contains(&format!("--session-id\n{native}\n")));
            } else {
                // The Codex case without an account cleans its session up in the child.
                assert_eq!(receipts.len(), usize::from(named));
                assert!(
                    receipts
                        .iter()
                        .all(|receipt| { receipt["launches"][0]["native_session_id"].is_null() })
                );
                assert!(!arguments.contains("--session-id"));
            }
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
