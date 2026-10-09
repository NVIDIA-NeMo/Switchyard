// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module validates and saves server configs and restarts the server LaunchAgent.

use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use chrono::Local;

use crate::config::Config;
use crate::health::{ServerStatus, probe};
use crate::models;
use crate::server_config::{Algorithm, Choice, ServerConfig};

/// RESTART_WAIT limits how long the app waits for the restarted server to answer `/health`.
const RESTART_WAIT: Duration = Duration::from_secs(10);

/// BACKUPS_PER_SECOND limits the number of backups for one file within one second.
const BACKUPS_PER_SECOND: u32 = 100;

/// This function switches a route to new models, checks the result with the server's
/// `--dry-run`, saves it, and restarts the server.
///
/// This function reports what happened on success, one line each: the saved file and its
/// backup, the restart, notes about the edit, and the check. An error means
/// the config file was not changed.
pub fn apply(
    settings: &Config,
    route: &str,
    algorithm: &Algorithm,
    choices: &[Choice],
) -> Result<String, String> {
    let path = &settings.config_file;
    // Read the file again so an edit made while the window was open is kept.
    let text = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "Could not read {}: {error}. Check config_file in the app settings.",
            path.display()
        )
    })?;
    let config = ServerConfig::parse(&text)?;
    let edited = config.edit(route, algorithm, choices)?;
    save_edit(settings, &text, edited)
}

pub fn remove(settings: &Config, route: &str, id: &str) -> Result<String, String> {
    let text = std::fs::read_to_string(&settings.config_file).map_err(|error| error.to_string())?;
    let config = ServerConfig::parse(&text)?;
    if !config.routes().iter().any(|r| r.key == route && r.id == id) {
        return Err("The route changed or was removed. Refresh and choose it again.".into());
    }
    let edited = config.remove(route)?;
    save_edit(settings, &text, edited)
}

fn save_edit(
    settings: &Config,
    text: &str,
    edited: crate::server_config::Edited,
) -> Result<String, String> {
    let path = &settings.config_file;
    if edited.text == text {
        return Ok(
            "Nothing to save. The route already uses this algorithm and these models.".to_string(),
        );
    }

    let binary = server_binary()?;
    let backup = save_checked(&binary, path, text, &edited.text).map_err(explain)?;
    let label = &settings.launchd_label;
    let health = format!("{}/health", settings.server_url);
    let mut lines = vec![format!(
        "Saved {}. The old file is at {}.",
        path.display(),
        backup.display()
    )];
    lines.push(match restart(label) {
        Err(error) => format!(
            "Could not restart {label}: {error}. The server uses the new config the next time \
             it starts. Click Restart server in the menu to try again."
        ),
        Ok(()) if wait_for_health(&settings.server_url) => {
            format!("Restarted {label}, and the server answers {health}.")
        }
        Ok(()) => format!(
            "Restarted {label}, but the server did not answer {health} within {} seconds. \
             Check the server log, by default in ~/.switchyard/logs/. To undo, copy {} over {} \
             and click Restart server.",
            RESTART_WAIT.as_secs(),
            backup.display(),
            path.display()
        ),
    });
    lines.extend(edited.notes);
    let routes = ServerConfig::parse(&edited.text)?.routes().len();
    let plural = if routes == 1 { "" } else { "s" };
    lines.push(format!(
        "The switchyard-server --dry-run check passed for {routes} route{plural}."
    ));
    Ok(lines.join("\n"))
}

/// This function adds advice to a check error that comes from a missing `api_key_env`
/// variable. The check runs with this app's environment, which is not the
/// user's shell environment.
fn explain(error: String) -> String {
    let variable = error
        .split_once("could not read api_key_env ")
        .and_then(|(_, rest)| rest.split([':', ' ', '\n']).next())
        .filter(|variable| !variable.is_empty())
        .map(str::to_string);
    match variable {
        Some(variable) => format!("{error}\n{}", models::missing_env_note(&variable)),
        None => error,
    }
}

/// This function returns the validator installed beside the app executable.
/// The app bundle and server LaunchAgent use separate copies built by the same install.
fn server_binary() -> Result<PathBuf, String> {
    std::env::current_exe()
        .map(|exe| exe.with_file_name("switchyard-server"))
        .map_err(|error| {
            format!(
                "Could not find switchyard-server next to the menu bar app: {error}. Run make \
                 install-macos from the Switchyard repository to install it."
            )
        })
}

/// This function checks `text` with `switchyard-server --dry-run`, then replaces `path`
/// with it, keeping `original`, the text that `path` held, as a timestamped
/// backup. Returns the backup's path.
///
/// The real file is `path`, or the file it points to when `path` is a
/// symlink. The new text goes to a temporary file next to the real file. The
/// temporary file is synced to disk and then renamed over the real file, so a
/// reader never sees a half-written config. When the check fails, or when
/// the real file no longer holds `original` after the backup is written, the
/// temporary file and the new backup are removed, `path` does not change,
/// and the error is returned.
fn save_checked(binary: &Path, path: &Path, original: &str, text: &str) -> Result<PathBuf, String> {
    // Resolve a symlink and replace the file it points to, so the link stays.
    let file = std::fs::canonicalize(path)
        .map_err(|error| format!("Could not find {}: {error}", path.display()))?;
    let dir = file.parent().unwrap_or(Path::new("/"));
    let mut candidate = tempfile::Builder::new()
        .prefix(".switchyard-picker-")
        .suffix(".toml")
        .tempfile_in(dir)
        .map_err(|error| format!("Could not create a file in {}: {error}", dir.display()))?;
    candidate
        .write_all(text.as_bytes())
        .and_then(|()| candidate.as_file().sync_all())
        .map_err(|error| format!("Could not write {}: {error}", candidate.path().display()))?;
    if let Ok(metadata) = std::fs::metadata(&file) {
        candidate
            .as_file()
            .set_permissions(metadata.permissions())
            .map_err(|error| {
                format!(
                    "Could not copy the permissions of {}: {error}",
                    path.display()
                )
            })?;
    }

    // Name the real file in errors, not the temporary one.
    check(binary, candidate.path()).map_err(|error| {
        error.replace(
            &candidate.path().display().to_string(),
            &path.display().to_string(),
        )
    })?;

    let backup = back_up(&file, original)
        .map_err(|error| format!("Could not back up {}: {error}", path.display()))?;
    // The check and the backup take a moment. Keep an edit that another
    // program saved in the meantime. Compare right before the rename, because
    // the rename replaces the file whatever it holds.
    if std::fs::read_to_string(&file).ok().as_deref() != Some(original) {
        let _ = std::fs::remove_file(&backup);
        return Err(format!(
            "Could not replace {}, because it changed on disk while Apply was checking it. \
             Click Apply again to apply your choices to the new file.",
            path.display()
        ));
    }
    candidate
        .persist(&file)
        .map_err(|error| format!("Could not replace {}: {}", path.display(), error.error))?;
    Ok(backup)
}

/// This function writes `text` to a new backup of `file` and returns its path. The backup
/// is `<file>.switchyard-backup.<timestamp>`, the name the installer uses,
/// with `-2`, `-3`, and so on added when that name is taken, so a backup
/// never replaces another one. The backup gets the file's permissions before
/// it gets the text.
fn back_up(file: &Path, text: &str) -> Result<PathBuf, std::io::Error> {
    let permissions = std::fs::metadata(file)?.permissions();
    let stamp = Local::now().format("%Y%m%d%H%M%S");
    for number in 1..=BACKUPS_PER_SECOND {
        let suffix = if number == 1 {
            String::new()
        } else {
            format!("-{number}")
        };
        let backup = PathBuf::from(format!(
            "{}.switchyard-backup.{stamp}{suffix}",
            file.display()
        ));
        let mut out = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&backup)
        {
            Ok(out) => out,
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        };
        let written = out
            .set_permissions(permissions.clone())
            .and_then(|()| out.write_all(text.as_bytes()))
            .and_then(|()| out.sync_all());
        if let Err(error) = written {
            let _ = std::fs::remove_file(&backup);
            return Err(error);
        }
        return Ok(backup);
    }
    Err(std::io::Error::new(
        ErrorKind::AlreadyExists,
        format!("{BACKUPS_PER_SECOND} backups from this second already exist"),
    ))
}

/// This function runs `switchyard-server --config <config> --dry-run`, and returns its
/// error output when the config is invalid.
fn check(binary: &Path, config: &Path) -> Result<(), String> {
    let output = Command::new(binary)
        .arg("--config")
        .arg(config)
        .arg("--dry-run")
        .output()
        .map_err(|error| {
            format!(
                "Could not run {}: {error}. Apply needs switchyard-server next to the menu bar \
                 app. Run make install-macos from the Switchyard repository to install it.",
                binary.display()
            )
        })?;
    if output.status.success() {
        return Ok(());
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Err(if stderr.is_empty() { stdout } else { stderr })
}

/// This function restarts the server's LaunchAgent.
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

/// This function runs a command, returning its stdout or a message naming what failed.
pub fn command(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("Could not run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

// This function saves the script before opening Terminal so a save error cannot launch it.
// If open fails, this function attempts cleanup; callers supply scripts that remove themselves on startup.
pub fn open_terminal_script(prefix: &str, text: &str) -> Result<(), String> {
    let mut script = tempfile::Builder::new()
        .prefix(prefix)
        .suffix(".command")
        .tempfile()
        .map_err(|e| e.to_string())?;
    writeln!(script, "{text}").map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        script
            .as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o700))
            .map_err(|e| e.to_string())?;
    }
    let (_, path) = script.keep().map_err(|e| e.to_string())?;
    if let Err(error) = command("open", &[&path.display().to_string()]) {
        let _ = std::fs::remove_file(path);
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// This function writes a stand-in for `switchyard-server` that runs `body`.
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

        let error = save_checked(&binary, &config, "old", "new").expect_err("check fails");

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

        let backup = save_checked(&binary, &config, "old", "new").expect("saved");

        assert_eq!(std::fs::read_to_string(&config).expect("read"), "new");
        let mode = |path: &Path| {
            std::fs::metadata(path)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777
        };
        assert_eq!(mode(&config), 0o640, "the file keeps its permissions");
        assert_eq!(backups(dir.path()).len(), 1);
        assert_eq!(std::fs::read_to_string(&backup).expect("read"), "old");
        assert_eq!(mode(&backup), 0o640, "the backup gets the same permissions");
        save_checked(&binary, &config, "new", "newer").expect("saved again");
        let mut kept: Vec<_> = backups(dir.path())
            .iter()
            .map(|path| std::fs::read_to_string(path).expect("backup"))
            .collect();
        kept.sort();
        assert_eq!(kept, ["new", "old"]);
        assert_eq!(std::fs::read_to_string(&config).expect("read"), "newer");
    }

    #[test]
    fn a_file_that_changes_during_the_check_is_not_replaced() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = dir.path().join("composite.toml");
        std::fs::write(&config, "old").expect("write config");
        // Another program saves the file while the check runs.
        let binary = fake_server(
            dir.path(),
            &format!("echo edited-elsewhere > '{}'", config.display()),
        );

        let error = save_checked(&binary, &config, "old", "new").expect_err("file changed");

        assert!(error.contains(&config.display().to_string()), "{error}");
        assert_eq!(
            std::fs::read_to_string(&config).expect("read"),
            "edited-elsewhere\n"
        );
        assert!(backups(dir.path()).is_empty());
    }

    #[test]
    fn saving_through_a_symlink_replaces_the_linked_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let real = dir.path().join("dotfiles.toml");
        let link = dir.path().join("composite.toml");
        std::fs::write(&real, "old").expect("write config");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        let binary = fake_server(dir.path(), r#"echo "server OK: gateway""#);

        save_checked(&binary, &link, "old", "new").expect("saved");

        let link_type = std::fs::symlink_metadata(&link)
            .expect("metadata")
            .file_type();
        assert!(link_type.is_symlink(), "the link is kept");
        assert_eq!(std::fs::read_to_string(&real).expect("read"), "new");
    }

    // The subprocess isolates PATH and makes Terminal reject the script without opening an app.
    #[test]
    fn a_rejected_terminal_launch_removes_the_kept_script() {
        if let Ok(record) = std::env::var("SWITCHYARD_TERMINAL_FIXTURE") {
            assert!(open_terminal_script("switchyard-terminal-fixture-", "#!/bin/bash").is_err());
            let path = std::fs::read_to_string(record).expect("script path");
            assert!(!Path::new(path.trim()).exists());
            return;
        }
        let dir = tempfile::tempdir().expect("fixture");
        let record = dir.path().join("script-path");
        let open = dir.path().join("open");
        std::fs::write(
            &open,
            "#!/bin/bash\nprintf '%s' \"$1\" > \"$SWITCHYARD_TERMINAL_FIXTURE\"\nexit 1\n",
        )
        .expect("Terminal stub");
        std::fs::set_permissions(&open, std::fs::Permissions::from_mode(0o700))
            .expect("permissions");
        let output = Command::new(std::env::current_exe().expect("runner"))
            .args([
                "--exact",
                "server::tests::a_rejected_terminal_launch_removes_the_kept_script",
            ])
            .env("SWITCHYARD_TERMINAL_FIXTURE", &record)
            .env("PATH", dir.path())
            .output()
            .expect("child");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
}
