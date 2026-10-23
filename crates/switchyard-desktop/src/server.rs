// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module checks and saves server configs, checks saved key status, and opens files.

use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use chrono::Local;
use switchyard_desktop::install::credentials::KeyStatus;

use crate::config::Config;
use crate::health::{Probe, ServerStatus};
use crate::server_config::{Algorithm, Choice, ServerConfig};

/// RESTART_WAIT limits how long the app waits for the restarted server to answer `/health`.
const RESTART_WAIT: Duration = Duration::from_secs(10);

/// BACKUPS_PER_SECOND limits the number of backups for one file within one second.
const BACKUPS_PER_SECOND: u32 = 100;

/// SaveResult distinguishes a saved config from a failed restart or health check.
pub struct SaveResult {
    pub message: String,
    pub warning: bool,
    pub details: Vec<String>,
}

/// This function switches a route to new models, checks the result with the server's
/// `--dry-run`, saves it, and restarts the server.
///
/// The result names the changed file and reports whether the restart succeeded.
/// The details contain the backup path and edit notes. An error means the file was not changed.
pub fn apply(
    settings: &Config,
    route: &str,
    algorithm: &Algorithm,
    choices: &[Choice],
    expected: &str,
) -> Result<SaveResult, String> {
    let path = &settings.config_file;
    // Read the file again so an edit made while the window was open is kept.
    let text = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "Could not read {}: {error}. Check config_file in the app settings.",
            path.display()
        )
    })?;
    check_original(&text, expected)?;
    let config = ServerConfig::parse(&text)?;
    let edited = config.edit(route, algorithm, choices)?;
    save_edit(settings, &text, edited, "Updated the route.")
}

pub fn remove(
    settings: &Config,
    route: &str,
    id: &str,
    expected: &str,
) -> Result<SaveResult, String> {
    let text = std::fs::read_to_string(&settings.config_file).map_err(|error| error.to_string())?;
    check_original(&text, expected)?;
    let config = ServerConfig::parse(&text)?;
    if !config.routes().iter().any(|r| r.key == route && r.id == id) {
        return Err("The route changed or was removed. Refresh and choose it again.".into());
    }
    let edited = config.remove(route)?;
    save_edit(settings, &text, edited, "Removed the route.")
}

pub fn save_route_change(
    settings: &Config,
    expected: &str,
    edited: crate::server_config::Edited,
) -> Result<SaveResult, String> {
    let text = std::fs::read_to_string(&settings.config_file).map_err(|error| error.to_string())?;
    check_original(&text, expected)?;
    save_edit(settings, &text, edited, "Changed the routes.")
}

fn check_original(text: &str, expected: &str) -> Result<(), String> {
    if text != expected {
        return Err("The server config changed since the editor loaded. \
                    Review it again; your draft is preserved."
            .into());
    }
    Ok(())
}

fn save_edit(
    settings: &Config,
    text: &str,
    edited: crate::server_config::Edited,
    summary: &str,
) -> Result<SaveResult, String> {
    let path = &settings.config_file;
    if edited.text == text {
        return Ok(SaveResult {
            message: "Nothing to save. The route already uses this algorithm and these models."
                .into(),
            warning: false,
            details: Vec::new(),
        });
    }
    service_config(settings).map_err(|error| format!("Nothing was saved. {error}"))?;
    let binary = server_binary()?;
    let backup = save_checked(&binary, path, text, &edited.text)?;
    let saved = format!("{summary} Changed {}.", path.display());
    let health = format!("{}/health", settings.server_url);
    let (message, warning) = match restart(&settings.launchd_label) {
        Err(error) => (
            format!(
                "{saved} Switchyard could not restart: {error}. \
                 Use Restart server in Settings to try again."
            ),
            true,
        ),
        Ok(()) if wait_for_health(&settings.server_url) => {
            (format!("{saved} Restarted Switchyard."), false)
        }
        Ok(()) => (
            format!(
                "{saved} Switchyard restarted but did not answer {health} within {} seconds. \
                 Check the server log in ~/.switchyard/logs/.",
                RESTART_WAIT.as_secs()
            ),
            true,
        ),
    };
    let mut details = vec![format!("Backup: {}", backup.display())];
    details.extend(edited.notes);
    if warning {
        details.push(format!(
            "To undo, copy {} over {} and use Restart server in Settings.",
            backup.display(),
            path.display()
        ));
    }
    Ok(SaveResult {
        message,
        warning,
        details,
    })
}

/// This function returns the file that the installed server loads, after checking that
/// `config_file` names it. The service always starts with `composite.toml` in its installation
/// directory, whatever `config_file` says. Saving, checking, or restarting with a different
/// file would report success while the server kept its old routes, and `/health` would not
/// notice.
fn service_config(settings: &Config) -> Result<&Path, String> {
    let Some(service) = &settings.service_config else {
        return Err(
            "The app found no install.toml next to its settings file, so it cannot \
                    tell which server config the installed server loads."
                .into(),
        );
    };
    // Comparing resolved paths accepts a symlink between the two names.
    let resolved = |path: &Path| std::fs::canonicalize(path).ok();
    if resolved(service).is_none() || resolved(service) != resolved(&settings.config_file) {
        return Err(format!(
            "The installed server loads {}, but the app settings select {}. \
             Remove config_file from the app settings, or set it to {}.",
            service.display(),
            settings.config_file.display(),
            service.display()
        ));
    }
    Ok(service)
}

/// This function finds the installed helper, which is the only program that reads or writes
/// server keys.
fn helper() -> Result<PathBuf, String> {
    Ok(
        server_binary()?.with_file_name(switchyard_desktop::install::executable_name(
            "switchyard-desktop-install",
        )),
    )
}

/// This function runs one of the helper's `key` commands. A key goes to the helper's standard
/// input, never its command line, where other processes could read it.
fn key_command(args: &[&str], key: Option<&str>) -> Result<String, String> {
    let helper = helper()?;
    let mut child = Command::new(&helper)
        .arg("key")
        .args(args)
        .stdin(if key.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "Could not run {}: {error}. \
                 Run cargo desktop install from the Switchyard repository to install it.",
                helper.display()
            )
        })?;
    if let (Some(key), Some(mut stdin)) = (key, child.stdin.take())
        && let Err(error) = stdin.write_all(key.as_bytes())
    {
        // The app tries to stop and reap the helper while preserving the original input error.
        // Cleanup errors may occur if the helper already exited.
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!("Could not pass the key to the helper: {error}"));
    }
    let output = child
        .wait_with_output()
        .map_err(|error| error.to_string())?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }
    Err(String::from_utf8_lossy(&output.stderr)
        .trim()
        .trim_start_matches("switchyard-desktop-install: ")
        .to_string())
}

/// This function reports which `api_key_env` variables in the server config have a saved key.
pub fn server_keys(settings: &Config) -> Result<Vec<KeyStatus>, String> {
    let config = std::fs::canonicalize(&settings.config_file).map_err(|error| error.to_string())?;
    let output = key_command(&["status", "--config", &config.display().to_string()], None)?;
    serde_json::from_str(&output).map_err(|error| error.to_string())
}

pub fn save_server_key(name: &str, key: &str) -> Result<(), String> {
    key_command(&["set", name], Some(key)).map(drop)
}

pub fn import_server_key(name: &str) -> Result<(), String> {
    key_command(&["import", name], None).map(drop)
}

pub fn remove_server_key(name: &str) -> Result<(), String> {
    key_command(&["remove", name], None).map(drop)
}

/// This function finds the installed server from the app's installation metadata.
fn server_binary() -> Result<PathBuf, String> {
    let executable = std::env::current_exe().map_err(|error| {
        format!(
            "Could not find switchyard-server next to the desktop app: {error}. \
             Run cargo desktop install from the Switchyard repository to install it."
        )
    })?;
    let metadata = switchyard_desktop::install::bundle_metadata(&executable);
    if metadata.try_exists().map_err(|error| error.to_string())? {
        return switchyard_desktop::install::Settings::load(&metadata)
            .map(|settings| settings.binary("switchyard-server"));
    }
    Ok(
        executable.with_file_name(switchyard_desktop::install::executable_name(
            "switchyard-server",
        )),
    )
}

/// This function checks `text` with `switchyard-server --dry-run`, then replaces `path`
/// with it, keeping `original`, the text that `path` held, as a timestamped
/// backup. Returns the backup's path.
///
/// The real file is `path`, or the file it points to when `path` is a
/// symlink. The new text goes to a temporary file next to the real file. The
/// temporary file is synced to disk and then renamed over the real file, so a
/// reader never sees a half-written config. On Unix, the directory is synced
/// after the rename, so a power loss keeps both the new file and the backup.
/// When the check fails, or when the real file no longer holds `original`
/// after the backup is written, the temporary file and the new backup are
/// removed, `path` does not change, and the error is returned.
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
    // The rename has already saved the file, so a failed directory sync does not
    // return an error. Windows has no directory sync.
    #[cfg(unix)]
    if let Ok(directory) = std::fs::File::open(dir) {
        let _ = directory.sync_all();
    }
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

/// This function asks the installed helper to run the server's dry-run on `config` with the
/// saved server keys, the same keys the installed server starts with.
fn check(binary: &Path, config: &Path) -> Result<(), String> {
    let config = std::fs::canonicalize(config).map_err(|error| error.to_string())?;
    let helper = binary.with_file_name(switchyard_desktop::install::executable_name(
        "switchyard-desktop-install",
    ));
    let mut validator = Command::new(&helper);
    validator
        .arg("validate")
        .arg("--server")
        .arg(binary)
        .arg("--config")
        .arg(&config);
    let output = validator.output().map_err(|error| {
        format!(
            "Could not run {}: {error}. \
             The config check needs the installed server and credential helper. \
             Run cargo desktop install from the Switchyard repository to install them.",
            helper.display()
        )
    })?;
    if output.status.success() {
        return Ok(());
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Err(if stderr.is_empty() { stdout } else { stderr })
}

/// This function runs the server's config check on the file that the installed server loads.
/// Without `install.toml` no installed server exists, so it checks the selected file.
pub fn check_config(settings: &Config) -> Result<(), String> {
    let file = match settings.service_config {
        Some(_) => service_config(settings)?,
        None => &settings.config_file,
    };
    check(&server_binary()?, file)
}

/// This function checks the file that the installed server loads, then restarts the server.
pub fn restart_checked(settings: &Config) -> Result<String, String> {
    let file = service_config(settings)?;
    check(&server_binary()?, file)?;
    restart(&settings.launchd_label)?;
    if !wait_for_health(&settings.server_url) {
        return Err(format!(
            "Switchyard did not answer {}/health within {} seconds. Open the server log and retry.",
            settings.server_url.trim_end_matches('/'),
            RESTART_WAIT.as_secs()
        ));
    }
    Ok("Restarted Switchyard.".into())
}

/// This function restarts the server's LaunchAgent, systemd user service, or Windows scheduled task.
pub fn restart(launchd_label: &str) -> Result<(), String> {
    #[cfg(windows)]
    {
        let _ = launchd_label;
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let settings = switchyard_desktop::install::Settings::load(
            &switchyard_desktop::install::bundle_metadata(&exe),
        )?;
        switchyard_desktop::install::windows::restart(&settings)
    }
    #[cfg(not(windows))]
    {
        if cfg!(target_os = "linux") {
            return command("systemctl", &["--user", "restart", "switchyard.service"]).map(|_| ());
        }
        let uid = command("id", &["-u"])?;
        let target = format!("gui/{}/{launchd_label}", uid.trim());
        command("launchctl", &["kickstart", "-k", &target]).map(|_| ())
    }
}

pub(crate) fn wait_for_health(server_url: &str) -> bool {
    let deadline = Instant::now() + RESTART_WAIT;
    let probe = Probe::default();
    loop {
        if probe.probe(server_url) == ServerStatus::Running {
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
    if program == "open" {
        #[cfg(windows)]
        {
            return switchyard_desktop::install::windows::open(
                args.first().ok_or("Choose a file or URL.")?,
            )
            .map(|()| String::new());
        }
        #[cfg(target_os = "linux")]
        {
            return command("xdg-open", args);
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// This function installs into `dir`: it writes `install.toml`, the service's
    /// `composite.toml`, and app settings with `settings` as their text. It returns the path of
    /// the app settings.
    fn installed(dir: &Path, settings: &str) -> PathBuf {
        let home = dir.join("home");
        let install = switchyard_desktop::install::Settings {
            sy_home: dir.to_path_buf(),
            codex_home: home.join(".codex"),
            service_dir: home.join("services"),
            app_dir: None,
            source: home.join("source"),
            cargo: home.join("cargo"),
            home,
            port: 4123,
            profile: "sy".into(),
            model: "route".into(),
        };
        let install = toml::to_string(&install).expect("install metadata");
        std::fs::write(dir.join("install.toml"), install).expect("install.toml");
        std::fs::write(dir.join("composite.toml"), "# service config\n").expect("service config");
        let path = dir.join("desktop.toml");
        std::fs::write(&path, settings).expect("app settings");
        path
    }

    // The installed service always loads its own composite.toml. A route edit to any other
    // file must stop before it writes a backup, changes the file, or restarts the server, and
    // Check server config and Restart server must not check the wrong file.
    #[test]
    fn server_config_actions_use_only_the_config_the_installed_server_loads() {
        let dir = tempfile::tempdir().expect("directory");
        let custom = dir.path().join("custom.toml");
        std::fs::write(&custom, "# custom\n").expect("custom config");
        let path = installed(dir.path(), &format!("config_file = {custom:?}\n"));
        let settings = Config::load(&path).expect("settings");
        let before = std::fs::read_dir(dir.path()).expect("files").count();
        let edited = crate::server_config::Edited {
            text: "# edited\n".into(),
            notes: Vec::new(),
        };
        let error = save_edit(&settings, "# custom\n", edited, "Updated the route.")
            .err()
            .expect("a different file");
        assert!(error.contains("The installed server loads"), "{error}");
        assert_eq!(
            std::fs::read_to_string(&custom).expect("custom"),
            "# custom\n"
        );
        assert_eq!(
            std::fs::read_dir(dir.path()).expect("files").count(),
            before
        );
        for error in [
            check_config(&settings).expect_err("check"),
            restart_checked(&settings).expect_err("restart"),
        ] {
            assert!(error.contains("The installed server loads"), "{error}");
        }

        // Without install.toml the app cannot tell which file the server loads.
        let bare = tempfile::tempdir().expect("directory");
        let settings = Config::load(&bare.path().join("desktop.toml")).expect("defaults");
        assert!(
            service_config(&settings)
                .expect_err("no install metadata")
                .contains("no install.toml")
        );

        // The default config_file and a symlink to the service's file both pass.
        let path = installed(dir.path(), "");
        service_config(&Config::load(&path).expect("settings")).expect("default file");
        #[cfg(unix)]
        {
            let link = dir.path().join("link.toml");
            std::os::unix::fs::symlink(dir.path().join("composite.toml"), &link).expect("link");
            std::fs::write(&path, format!("config_file = {link:?}\n")).expect("app settings");
            service_config(&Config::load(&path).expect("settings")).expect("symlink");
        }
    }
}
