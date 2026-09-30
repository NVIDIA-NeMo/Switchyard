// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Controls the installed server: check a config with its `--dry-run`, save
//! it, and restart the LaunchAgent that runs it.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use chrono::Local;

use crate::config::Config;
use crate::health::{ServerStatus, probe};
use crate::server_config::{Algorithm, Choice, ServerConfig};

/// How long to wait for the restarted server to answer `/health`.
const RESTART_WAIT: Duration = Duration::from_secs(10);

/// Switches a route to new models, checks the result with the server's
/// `--dry-run`, saves it, and restarts the server.
///
/// Returns what happened on success. An error means the config file was not
/// changed.
pub fn apply(
    settings: &Config,
    route: &str,
    algorithm: &Algorithm,
    choices: &[Choice],
) -> Result<String, String> {
    let path = &settings.config_file;
    // Read the file again so an edit made while the window was open is kept.
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let edited = ServerConfig::parse(&text)?.edit(route, algorithm, choices)?;
    if edited == text {
        return Ok("Nothing to save: the route already uses these models.".to_string());
    }

    let binary = server_binary()?;
    let saved = save_checked(&binary, path, &edited)?;
    let restarted = match restart(&settings.launchd_label) {
        Err(error) => format!("The restart failed: {error}"),
        Ok(()) if wait_for_health(&settings.server_url) => format!(
            "Restarted {}, and the server answers {}/health.",
            settings.launchd_label, settings.server_url
        ),
        Ok(()) => format!(
            "Restarted {}, but the server has not answered {}/health yet. Check its log.",
            settings.launchd_label, settings.server_url
        ),
    };
    Ok(format!("{saved}\n{restarted}"))
}

/// Returns the `switchyard-server` installed next to this app, which is the
/// binary that the server's LaunchAgent runs.
fn server_binary() -> Result<PathBuf, String> {
    std::env::current_exe()
        .map(|exe| exe.with_file_name("switchyard-server"))
        .map_err(|error| format!("find switchyard-server next to this app: {error}"))
}

/// Checks `text` with `switchyard-server --dry-run`, then replaces `path`
/// with it, keeping the old file as a timestamped backup.
///
/// The real file is `path`, or the file it points to when `path` is a
/// symlink. The new text goes to a temporary file next to the real file. The
/// temporary file is synced to disk and then renamed over the real file, so a
/// reader never sees a half-written config. When the check fails, the
/// temporary file is removed, `path` does not change, and the server's error
/// is returned.
fn save_checked(binary: &Path, path: &Path, text: &str) -> Result<String, String> {
    // Resolve a symlink and replace the file it points to, so the link stays.
    let file =
        std::fs::canonicalize(path).map_err(|error| format!("find {}: {error}", path.display()))?;
    let dir = file.parent().unwrap_or(Path::new("/"));
    let mut candidate = tempfile::Builder::new()
        .prefix(".switchyard-picker-")
        .suffix(".toml")
        .tempfile_in(dir)
        .map_err(|error| format!("create a file in {}: {error}", dir.display()))?;
    candidate
        .write_all(text.as_bytes())
        .and_then(|()| candidate.as_file().sync_all())
        .map_err(|error| format!("write {}: {error}", candidate.path().display()))?;
    if let Ok(metadata) = std::fs::metadata(&file) {
        candidate
            .as_file()
            .set_permissions(metadata.permissions())
            .map_err(|error| format!("copy the permissions of {}: {error}", path.display()))?;
    }

    // Name the real file in errors, not the temporary one.
    let summary = check(binary, candidate.path()).map_err(|error| {
        error.replace(
            &candidate.path().display().to_string(),
            &path.display().to_string(),
        )
    })?;

    let backup = PathBuf::from(format!(
        "{}.switchyard-backup.{}",
        file.display(),
        Local::now().format("%Y%m%d%H%M%S")
    ));
    std::fs::copy(&file, &backup)
        .map_err(|error| format!("back up {}: {error}", path.display()))?;
    candidate
        .persist(&file)
        .map_err(|error| format!("replace {}: {}", path.display(), error.error))?;
    Ok(format!(
        "switchyard-server --dry-run: {summary}\nSaved {}, and kept the old file as {}.",
        path.display(),
        backup.display()
    ))
}

/// Runs `switchyard-server --config <config> --dry-run` and returns its
/// summary, or its error output when the config is invalid.
fn check(binary: &Path, config: &Path) -> Result<String, String> {
    let output = Command::new(binary)
        .arg("--config")
        .arg(config)
        .arg("--dry-run")
        .output()
        .map_err(|error| format!("run {}: {error}", binary.display()))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if output.status.success() {
        Ok(stdout)
    } else if stderr.is_empty() {
        Err(stdout)
    } else {
        Err(stderr)
    }
}

/// Restarts the server's LaunchAgent.
pub fn restart(launchd_label: &str) -> Result<(), String> {
    let uid = command("id", &["-u"])?;
    let target = format!("gui/{}/{launchd_label}", uid.trim());
    command("launchctl", &["kickstart", "-k", &target]).map(|_| ())
}

fn wait_for_health(server_url: &str) -> bool {
    let deadline = Instant::now() + RESTART_WAIT;
    loop {
        if probe(server_url) == ServerStatus::Running {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Runs a command, returning its stdout or a message naming what failed.
pub fn command(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// Writes a stand-in for `switchyard-server` that runs `body`.
    fn fake_server(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("switchyard-server");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write script");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("make script executable");
        path
    }

    fn backups(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .expect("read dir")
            .map(|entry| entry.expect("entry").path())
            .filter(|path| path.display().to_string().contains(".switchyard-backup."))
            .collect()
    }

    #[test]
    fn a_config_that_fails_the_check_is_not_saved() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = dir.path().join("composite.toml");
        std::fs::write(&config, "old").expect("write config");
        // The real server names the file it was given; the error must name
        // the user's file instead of the temporary copy.
        let binary = fake_server(
            dir.path(),
            r#"echo "invalid server config $2: route gateway cannot forward both" >&2; exit 1"#,
        );

        let error = save_checked(&binary, &config, "new").expect_err("check fails");

        assert_eq!(
            error,
            format!(
                "invalid server config {}: route gateway cannot forward both",
                config.display()
            )
        );
        assert_eq!(std::fs::read_to_string(&config).expect("read"), "old");
        assert!(backups(dir.path()).is_empty());
        assert_eq!(
            std::fs::read_dir(dir.path()).expect("read dir").count(),
            2,
            "the temporary file is removed"
        );
    }

    #[test]
    fn a_config_that_passes_the_check_replaces_the_file_and_keeps_a_backup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = dir.path().join("composite.toml");
        std::fs::write(&config, "old").expect("write config");
        std::fs::set_permissions(&config, std::fs::Permissions::from_mode(0o640))
            .expect("set permissions");
        let binary = fake_server(
            dir.path(),
            r#"grep -q new "$2" && echo "server OK: gateway""#,
        );

        let message = save_checked(&binary, &config, "new").expect("saved");

        assert!(message.starts_with("switchyard-server --dry-run: server OK: gateway\n"));
        assert_eq!(std::fs::read_to_string(&config).expect("read"), "new");
        let mode = std::fs::metadata(&config)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o640, "the file keeps its permissions");
        let backups = backups(dir.path());
        assert_eq!(backups.len(), 1);
        assert_eq!(std::fs::read_to_string(&backups[0]).expect("read"), "old");
    }

    #[test]
    fn saving_through_a_symlink_replaces_the_linked_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let real = dir.path().join("dotfiles.toml");
        let link = dir.path().join("composite.toml");
        std::fs::write(&real, "old").expect("write config");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        let binary = fake_server(dir.path(), r#"echo "server OK: gateway""#);

        save_checked(&binary, &link, "new").expect("saved");

        let link_type = std::fs::symlink_metadata(&link)
            .expect("metadata")
            .file_type();
        assert!(link_type.is_symlink(), "the link is kept");
        assert_eq!(std::fs::read_to_string(&real).expect("read"), "new");
    }

    #[test]
    fn reports_which_command_failed() {
        let error = command("switchyard-does-not-exist", &[]).expect_err("missing program");

        assert!(error.contains("switchyard-does-not-exist"));
    }
}
