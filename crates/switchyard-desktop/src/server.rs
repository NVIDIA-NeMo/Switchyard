// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module validates server configs and restarts the server service.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use switchyard_desktop::install::credentials::KeyStatus;

use crate::config::Config;
use crate::health::{ServerStatus, probe};

/// RESTART_WAIT limits how long the app waits for the restarted server to answer `/health`.
const RESTART_WAIT: Duration = Duration::from_secs(10);

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
    if let (Some(key), Some(mut stdin)) = (key, child.stdin.take()) {
        stdin
            .write_all(key.as_bytes())
            .map_err(|error| format!("Could not pass the key to the helper: {error}"))?;
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
             Apply needs the installed server and credential helper. \
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

pub fn check_config(settings: &Config) -> Result<(), String> {
    check(&server_binary()?, &settings.config_file)
}

pub fn restart_checked(settings: &Config) -> Result<String, String> {
    check_config(settings)?;
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

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[cfg(target_os = "linux")]
    #[test]
    fn restart_uses_the_linux_user_service_and_returns_command_failure() {
        // The subprocess uses a fixture systemctl so this test cannot restart the user's server.
        if let Ok(expected_failure) = std::env::var("SWITCHYARD_RESTART_FIXTURE") {
            let result = restart("unused-on-linux");
            if expected_failure == "1" {
                assert!(result.expect_err("failed restart").contains("systemctl"));
            } else {
                result.expect("restart");
            }
            return;
        }
        let dir = tempfile::tempdir().expect("fixture");
        let record = dir.path().join("arguments");
        let systemctl = dir.path().join("systemctl");
        std::fs::write(
            &systemctl,
            "#!/bin/sh\n\
             printf '%s\\n' \"$@\" > \"$SWITCHYARD_RESTART_ARGUMENTS\"\n\
             exit \"$SWITCHYARD_RESTART_FIXTURE\"\n",
        )
        .expect("service fixture");
        std::fs::set_permissions(&systemctl, std::fs::Permissions::from_mode(0o700))
            .expect("permissions");
        for failure in ["0", "1"] {
            let output = Command::new(std::env::current_exe().expect("runner"))
                .args(["--exact", "server::tests::restart_uses_the_linux_user_service_and_returns_command_failure"])
                .env("SWITCHYARD_RESTART_FIXTURE", failure)
                .env("SWITCHYARD_RESTART_ARGUMENTS", &record)
                .env("PATH", dir.path())
                .output().expect("child");
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stdout)
            );
            assert_eq!(
                std::fs::read_to_string(&record).expect("service arguments"),
                "--user\nrestart\nswitchyard.service\n"
            );
        }
    }
}
