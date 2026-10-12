// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server keys for the installed server.
//!
//! launchd, systemd, or Task Scheduler starts the installed server, never a shell, so the
//! server cannot see the variables that a shell exports. Before it starts the server, the
//! installed helper reads each variable that the config names with `api_key_env` from the
//! system credential store: macOS Keychain, Windows Credential Manager, or the Linux Secret
//! Service (GNOME Keyring or KDE Wallet). A saved key replaces any value the service manager
//! passes, and a variable with no saved key stops the start. Keys are saved under the variable
//! name alone, so editing, moving, or replacing the config keeps them.
//!
//! The helper is the only program that reads or writes these keys. macOS ties Keychain access
//! to the program that saved an item, so one program keeps one identity.

use std::{collections::BTreeSet, path::Path, process::Command};

#[cfg(target_os = "macos")]
pub const STORE: &str = "macOS Keychain";
#[cfg(windows)]
pub const STORE: &str = "Windows Credential Manager";
#[cfg(target_os = "linux")]
pub const STORE: &str = "the Secret Service keyring (GNOME Keyring or KDE Wallet)";
#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
pub const STORE: &str = "a system credential store";

/// KeyStatus reports whether one variable that the config declares has a saved key.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyStatus {
    pub name: String,
    pub saved: bool,
}

/// This function returns the names that the config gives with `api_key_env`.
pub fn variables(config: &Path) -> Result<BTreeSet<String>, String> {
    let text = std::fs::read_to_string(config)
        .map_err(|error| format!("Could not read {}: {error}", config.display()))?;
    switchyard_runner::api_key_variables(&text).map_err(|e| e.to_string())
}

/// This function reports which declared variables have a saved key.
pub fn status(config: &Path) -> Result<Vec<KeyStatus>, String> {
    variables(config)?
        .into_iter()
        .map(|name| {
            Ok(KeyStatus {
                saved: store::exists(&name)?,
                name,
            })
        })
        .collect()
}

/// This function gives `command` the saved key for every variable that the config declares.
/// It fails when any of them has no saved key.
pub fn configure(command: &mut Command, config: &Path) -> Result<(), String> {
    supply(command, &variables(config)?, store::read)
}

/// This function waits up to a minute for the login keyring to unlock. A systemd user service
/// can start before the desktop login unlocks it, and the helper never shows an unlock prompt
/// from the background.
#[cfg(target_os = "linux")]
pub fn wait_for_keyring() -> Result<(), String> {
    store::wait_for_unlock(std::time::Duration::from_secs(60))
}

/// This function runs the server's `--dry-run` on `config` with the saved keys.
pub fn validate(server: &Path, config: &Path) -> Result<(), String> {
    let mut validator = Command::new(server);
    configure(&mut validator, config)?;
    let output = validator
        .arg("--config")
        .arg(config)
        .arg("--dry-run")
        .output()
        .map_err(|e| format!("Could not run {}: {e}", server.display()))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(if stderr.trim().is_empty() {
        String::from_utf8_lossy(&output.stdout).into_owned()
    } else {
        stderr.into_owned()
    })
}

/// This function saves `key` for the variable `name`, replacing any saved key.
pub fn set(name: &str, key: &str) -> Result<(), String> {
    check_name(name)?;
    let key = key.trim();
    if key.is_empty() {
        return Err(format!("The key for {name} is empty."));
    }
    store::write(name, key)
}

/// This function reads `name` once from the user's login shell and saves it.
pub fn import(name: &str) -> Result<(), String> {
    check_name(name)?;
    #[cfg(unix)]
    let key = shell_value(&login_shell()?, name)?;
    #[cfg(windows)]
    let key = shell_value(name)?;
    set(name, &key)
}

/// This function deletes the saved key for `name`. A missing key is not an error.
pub fn remove(name: &str) -> Result<(), String> {
    check_name(name)?;
    store::delete(name)
}

fn supply(
    command: &mut Command,
    names: &BTreeSet<String>,
    saved: impl Fn(&str) -> Result<Option<String>, String>,
) -> Result<(), String> {
    let mut missing = Vec::new();
    for name in names {
        match saved(name)? {
            Some(key) => {
                command.env(name, key);
            }
            None => missing.push(name.as_str()),
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    let helper = std::env::current_exe()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| "switchyard-desktop-install".into());
    let app = if cfg!(any(target_os = "macos", windows)) {
        "Save it in Switchyard under Settings → Server keys, or run"
    } else {
        "Save it with"
    };
    Err(format!(
        "No saved key for {}. The installed server reads api_key_env keys from {STORE}, \
         not from your shell. {app}: {helper} key set {}",
        missing.join(", "),
        if missing.len() == 1 {
            missing[0]
        } else {
            "NAME"
        }
    ))
}

fn check_name(name: &str) -> Result<(), String> {
    if name.trim().is_empty() || name.contains('=') || name.chars().any(char::is_control) {
        return Err("Invalid environment variable name.".into());
    }
    Ok(())
}

/// SHELL_WAIT limits how long a login shell may take to print one variable.
#[cfg(unix)]
const SHELL_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// This function starts `shell` as a login shell, the way a terminal does, and prints one
/// variable. Markers separate the value from anything the shell's startup files print.
#[cfg(unix)]
fn shell_value(shell: &Path, name: &str) -> Result<String, String> {
    use std::io::Read;
    use std::process::Stdio;
    const MARK: &str = "\u{1e}switchyard\u{1e}";
    let mut child = Command::new(shell)
        .args([
            "-i",
            "-l",
            "-c",
            "printf '\\036switchyard\\036'; printenv -- \"$SWITCHYARD_KEY_NAME\"; \
             printf '\\036switchyard\\036'",
        ])
        .env("SWITCHYARD_KEY_NAME", name)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Could not start your login shell {}: {e}", shell.display()))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or("The login shell has no output.")?;
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut output = Vec::new();
        let _ = stdout.read_to_end(&mut output);
        let _ = sender.send(output);
    });
    let output = receiver.recv_timeout(SHELL_WAIT);
    let _ = child.kill();
    let _ = child.wait();
    let output = output.map_err(|_| {
        format!(
            "Your login shell {} did not finish within {} seconds.",
            shell.display(),
            SHELL_WAIT.as_secs()
        )
    })?;
    let text = String::from_utf8(output).map_err(|_| format!("{name} is not UTF-8 text."))?;
    let value = text.split(MARK).nth(1).unwrap_or_default();
    let value = value.strip_suffix('\n').unwrap_or(value);
    if value.trim().is_empty() {
        return Err(format!(
            "{name} is not set in your login shell ({}).",
            shell.display()
        ));
    }
    Ok(value.to_string())
}

/// This function returns the login shell from the user's account record.
#[cfg(unix)]
fn login_shell() -> Result<std::path::PathBuf, String> {
    use std::os::unix::ffi::OsStrExt;
    // SAFETY: getpwuid returns null or a record in static storage. The shell is copied before
    // any other call can reuse that storage.
    let shell = unsafe {
        let entry = libc::getpwuid(libc::getuid());
        if entry.is_null() || (*entry).pw_shell.is_null() {
            return Err("Could not find your login shell.".into());
        }
        std::ffi::CStr::from_ptr((*entry).pw_shell)
            .to_bytes()
            .to_vec()
    };
    if shell.is_empty() {
        return Err("Your account has no login shell.".into());
    }
    Ok(std::ffi::OsStr::from_bytes(&shell).into())
}

/// Windows has no login shell. Task Scheduler and Explorer pass the user's environment
/// variables, so the helper reads its own.
#[cfg(windows)]
fn shell_value(name: &str) -> Result<String, String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{name} is not set in your Windows environment variables."))
}

#[cfg(target_os = "macos")]
mod store {
    /// SERVICE names the server keys in Keychain. Each item's account is the variable name.
    const SERVICE: &str = "Switchyard server key";
    /// NOT_FOUND matches the Keychain's `errSecItemNotFound` status.
    const NOT_FOUND: i32 = -25300;

    /// This function looks for the item without reading its key, so Keychain never asks
    /// for permission.
    pub fn exists(name: &str) -> Result<bool, String> {
        use security_framework::item::{ItemClass, ItemSearchOptions};
        let mut search = ItemSearchOptions::new();
        search
            .class(ItemClass::generic_password())
            .service(SERVICE)
            .account(name)
            .load_attributes(true);
        match search.search() {
            Ok(found) => Ok(!found.is_empty()),
            Err(e) if e.code() == NOT_FOUND => Ok(false),
            Err(e) => Err(format!("Could not search Keychain for {name}: {e}")),
        }
    }

    pub fn read(name: &str) -> Result<Option<String>, String> {
        match security_framework::passwords::get_generic_password(SERVICE, name) {
            Ok(bytes) => String::from_utf8(bytes)
                .map(Some)
                .map_err(|_| format!("The saved key for {name} is not text.")),
            Err(e) if e.code() == NOT_FOUND => Ok(None),
            Err(e) => Err(format!(
                "Could not read the saved key for {name} from Keychain: {e}"
            )),
        }
    }

    pub fn write(name: &str, key: &str) -> Result<(), String> {
        security_framework::passwords::set_generic_password(SERVICE, name, key.as_bytes())
            .map_err(|e| format!("Could not save the key for {name} in Keychain: {e}"))
    }

    pub fn delete(name: &str) -> Result<(), String> {
        match security_framework::passwords::delete_generic_password(SERVICE, name) {
            Ok(()) => Ok(()),
            Err(e) if e.code() == NOT_FOUND => Ok(()),
            Err(e) => Err(format!(
                "Could not delete the saved key for {name} from Keychain: {e}"
            )),
        }
    }
}

#[cfg(windows)]
mod store {
    fn target(name: &str) -> String {
        format!("Switchyard:server-key:{name}")
    }

    pub fn exists(name: &str) -> Result<bool, String> {
        read(name).map(|key| key.is_some())
    }

    pub fn read(name: &str) -> Result<Option<String>, String> {
        crate::install::windows::read_credential(&target(name))
    }

    pub fn write(name: &str, key: &str) -> Result<(), String> {
        crate::install::windows::write_credential(&target(name), key)
    }

    pub fn delete(name: &str) -> Result<(), String> {
        crate::install::windows::delete_credential(&target(name))
    }
}

#[cfg(target_os = "linux")]
mod store {
    use dbus_secret_service::{EncryptionType, SecretService};
    use std::collections::HashMap;

    const LOCKED: &str =
        "The login keyring is locked. Log in to the desktop or unlock the keyring, then try again.";

    fn attributes(name: &str) -> HashMap<&str, &str> {
        HashMap::from([("application", "switchyard"), ("server-key", name)])
    }

    fn connect() -> Result<SecretService, String> {
        SecretService::connect(EncryptionType::Dh).map_err(|e| {
            format!(
                "Could not reach the Secret Service, \
                 which GNOME Keyring or KDE Wallet provides: {e}"
            )
        })
    }

    /// This function finds locked items too, because finding an item does not read its key.
    pub fn exists(name: &str) -> Result<bool, String> {
        let service = connect()?;
        let found = service
            .search_items(attributes(name))
            .map_err(|e| format!("Could not search the keyring for {name}: {e}"))?;
        Ok(!(found.unlocked.is_empty() && found.locked.is_empty()))
    }

    pub fn read(name: &str) -> Result<Option<String>, String> {
        let service = connect()?;
        let found = service
            .search_items(attributes(name))
            .map_err(|e| format!("Could not search the keyring for {name}: {e}"))?;
        if let Some(item) = found.unlocked.first() {
            let bytes = item
                .get_secret()
                .map_err(|e| format!("Could not read the saved key for {name}: {e}"))?;
            return String::from_utf8(bytes)
                .map(Some)
                .map_err(|_| format!("The saved key for {name} is not text."));
        }
        if found.locked.is_empty() {
            Ok(None)
        } else {
            Err(LOCKED.into())
        }
    }

    /// This function saves into the default keyring. Unlocking it can show the desktop's
    /// password prompt, which is expected when the user saves a key.
    pub fn write(name: &str, key: &str) -> Result<(), String> {
        let service = connect()?;
        let collection = service
            .get_default_collection()
            .or_else(|_| service.get_collection_by_alias("login"))
            .map_err(|e| format!("The Secret Service has no default keyring: {e}"))?;
        collection
            .ensure_unlocked()
            .map_err(|e| format!("Could not unlock the default keyring: {e}"))?;
        collection
            .create_item(
                &format!("Switchyard server key {name}"),
                attributes(name),
                key.as_bytes(),
                true,
                "text/plain",
            )
            .map(drop)
            .map_err(|e| format!("Could not save the key for {name} in the keyring: {e}"))
    }

    pub fn delete(name: &str) -> Result<(), String> {
        let service = connect()?;
        let found = service
            .search_items(attributes(name))
            .map_err(|e| format!("Could not search the keyring for {name}: {e}"))?;
        for item in found.unlocked.iter().chain(&found.locked) {
            item.delete()
                .map_err(|e| format!("Could not delete the saved key for {name}: {e}"))?;
        }
        Ok(())
    }

    pub fn wait_for_unlock(limit: std::time::Duration) -> Result<(), String> {
        let deadline = std::time::Instant::now() + limit;
        loop {
            let error = match connect().and_then(|service| {
                service
                    .search_items(HashMap::from([("application", "switchyard")]))
                    .map_err(|e| e.to_string())
                    .map(|found| found.locked.is_empty())
            }) {
                Ok(true) => return Ok(()),
                Ok(false) => LOCKED.to_string(),
                Err(error) => error,
            };
            if std::time::Instant::now() >= deadline {
                return Err(error);
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
mod store {
    const UNSUPPORTED: &str = "Saved server keys need macOS, Windows, or Linux.";

    pub fn exists(_name: &str) -> Result<bool, String> {
        Err(UNSUPPORTED.into())
    }

    pub fn read(_name: &str) -> Result<Option<String>, String> {
        Err(UNSUPPORTED.into())
    }

    pub fn write(_name: &str, _key: &str) -> Result<(), String> {
        Err(UNSUPPORTED.into())
    }

    pub fn delete(_name: &str) -> Result<(), String> {
        Err(UNSUPPORTED.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
schema_version=1
[llm_clients.gateway]
format="openai_chat"
base_url="https://example.test/v1"
api_key_env="CUSTOM_GATEWAY_TOKEN"
[decision_clients.judge]
format="system_one"
endpoint="https://example.test/decision"
api_key_env="CUSTOM_JUDGE_TOKEN"
timeout_ms=1000
[targets]
[routes]
"#;

    #[test]
    fn saved_keys_replace_inherited_values_and_a_missing_key_stops_the_start() {
        let dir = tempfile::tempdir().expect("fixture");
        let config = dir.path().join("composite.toml");
        std::fs::write(&config, CONFIG).expect("config");
        let names = variables(&config).expect("names");

        // A stale value from the service manager must not reach the server.
        let mut command = Command::new("unused");
        command.env("CUSTOM_GATEWAY_TOKEN", "stale-launchd-value");
        supply(&mut command, &names, |name| {
            Ok(Some(format!("saved-{name}")))
        })
        .expect("supplied");
        let mut supplied: Vec<_> = command.get_envs().collect();
        supplied.sort();
        assert_eq!(
            supplied,
            vec![
                (
                    std::ffi::OsStr::new("CUSTOM_GATEWAY_TOKEN"),
                    Some(std::ffi::OsStr::new("saved-CUSTOM_GATEWAY_TOKEN"))
                ),
                (
                    std::ffi::OsStr::new("CUSTOM_JUDGE_TOKEN"),
                    Some(std::ffi::OsStr::new("saved-CUSTOM_JUDGE_TOKEN"))
                ),
            ]
        );

        // An exported value does not stand in for a missing saved key.
        let mut command = Command::new("unused");
        command.env("CUSTOM_JUDGE_TOKEN", "exported-in-shell");
        let error = supply(&mut command, &names, |name| {
            Ok((name == "CUSTOM_GATEWAY_TOKEN").then(|| "saved".to_string()))
        })
        .expect_err("missing key");
        assert!(
            error.contains("No saved key for CUSTOM_JUDGE_TOKEN."),
            "{error}"
        );
        assert!(!error.contains("CUSTOM_GATEWAY_TOKEN"), "{error}");
        assert!(error.contains("key set CUSTOM_JUDGE_TOKEN"), "{error}");

        let error = supply(&mut Command::new("unused"), &names, |_| {
            Err("store unavailable".into())
        })
        .expect_err("store error");
        assert_eq!(error, "store unavailable");
    }

    #[test]
    fn invalid_names_and_empty_keys_are_rejected_before_the_store() {
        for name in ["", " ", "A=B", "A\0B", "A\nB"] {
            assert!(set(name, "key").is_err(), "{name:?}");
            assert!(import(name).is_err(), "{name:?}");
            assert!(remove(name).is_err(), "{name:?}");
        }
        assert_eq!(
            set("SWITCHYARD_UNUSED_TEST_KEY", " \n"),
            Err("The key for SWITCHYARD_UNUSED_TEST_KEY is empty.".into())
        );
    }

    #[cfg(unix)]
    #[test]
    fn login_shell_value_ignores_startup_output() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("fixture");
        let shell = dir.path().join("shell");
        // A chatty shell config prints around the command; $4 is the text after -i -l -c.
        std::fs::write(
            &shell,
            "#!/bin/sh\n\
             echo 'Welcome back'\n\
             export SWITCHYARD_FIXTURE_VALUE=fixture-value\n\
             /bin/sh -c \"$4\"\n\
             echo 'Goodbye'\n",
        )
        .expect("shell");
        std::fs::set_permissions(&shell, std::fs::Permissions::from_mode(0o700))
            .expect("permissions");
        assert_eq!(
            shell_value(&shell, "SWITCHYARD_FIXTURE_VALUE"),
            Ok("fixture-value".into())
        );
        let error = shell_value(&shell, "SWITCHYARD_FIXTURE_UNSET").expect_err("unset");
        assert!(
            error.contains("SWITCHYARD_FIXTURE_UNSET is not set"),
            "{error}"
        );
    }
}
