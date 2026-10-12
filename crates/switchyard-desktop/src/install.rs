// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The source installer owns app bundles, service definitions, and installation metadata.

use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub mod credentials;
#[cfg(windows)]
pub mod windows;

pub fn executable_name(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

const SERVER: &str = "com.nvidia.switchyard.server";
const DESKTOP: &str = "com.nvidia.switchyard.desktop";
// Migration removes this LaunchAgent so the old app cannot restart alongside the desktop app.
const LEGACY_DESKTOP: &str = "com.nvidia.switchyard.menubar";
const UNIT: &str = "switchyard.service";
const BUNDLED_CONFIG: &str = include_str!("../../../scripts/config/composite.toml");
const UPDATE_LOCK: &str = ".switchyard-desktop-update.lock";
const PORT_ERROR: &str = "SY_PORT must be a number from 1 to 65535.";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub home: PathBuf,
    pub sy_home: PathBuf,
    pub codex_home: PathBuf,
    pub service_dir: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_dir: Option<PathBuf>,
    pub source: PathBuf,
    pub cargo: PathBuf,
    pub port: u16,
    pub profile: String,
    pub model: String,
}

impl Settings {
    pub fn for_uninstall() -> Result<Self, String> {
        let defaults = Self::from_env()?;
        let recorded = defaults.install_record();
        if !present(&recorded)? {
            return Ok(defaults);
        }
        let selected = Self::load(&recorded)?;
        for (name, differs) in [
            (
                "SY_HOME",
                defaults.sy_home.as_os_str() != selected.sy_home.as_os_str(),
            ),
            (
                "CODEX_HOME",
                defaults.codex_home.as_os_str() != selected.codex_home.as_os_str(),
            ),
            ("SY_PROFILE", defaults.profile != selected.profile),
        ] {
            if differs && std::env::var_os(name).is_some() {
                return Err(format!(
                    "{name} differs from the recorded installation at {}. \
                     Use uninstall --settings FILE to choose a different installation.",
                    recorded.display()
                ));
            }
        }
        Ok(selected)
    }
    pub fn from_env() -> Result<Self, String> {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
            .ok_or("Could not find your home directory.")?;
        let path = |name, default| std::env::var_os(name).map(PathBuf::from).unwrap_or(default);
        let value = |name: &str, default: &str| {
            std::env::var(name).or_else(|error| match error {
                std::env::VarError::NotPresent => Ok(default.to_owned()),
                std::env::VarError::NotUnicode(_) => Err(format!("{name} must be UTF-8.")),
            })
        };
        let port = value("SY_PORT", "4123")?;
        if !port.bytes().all(|b| b.is_ascii_digit()) {
            return Err(PORT_ERROR.into());
        }
        let settings = Self {
            sy_home: path("SY_HOME", home.join(".switchyard")),
            codex_home: path("CODEX_HOME", home.join(".codex")),
            service_dir: if cfg!(target_os = "macos") {
                home.join("Library/LaunchAgents")
            } else if cfg!(windows) {
                path("APPDATA", home.join("AppData/Roaming"))
                    .join("Microsoft/Windows/Start Menu/Programs")
            } else {
                path("XDG_CONFIG_HOME", home.join(".config")).join("systemd/user")
            },
            app_dir: cfg!(windows).then(|| {
                path("LOCALAPPDATA", home.join("AppData/Local")).join("Programs/Switchyard")
            }),
            home,
            source: Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .ok_or("Could not find the source checkout.")?
                .to_path_buf(),
            cargo: PathBuf::from(env!("CARGO")),
            port: port.parse().map_err(|_| PORT_ERROR)?,
            profile: value("SY_PROFILE", "sy")?,
            model: value("SY_MODEL", "composite-gpt-6-sol-gpt-6-luna")?,
        };
        settings.validate()?;
        Ok(settings)
    }

    pub fn load(file: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(file).map_err(|e| format!("Read {}: {e}", file.display()))?;
        let settings: Self =
            toml::from_str(&text).map_err(|e| format!("Parse {}: {e}", file.display()))?;
        settings.validate()?;
        Ok(settings)
    }

    pub fn validate(&self) -> Result<(), String> {
        let profile = self.profile.as_bytes();
        if !profile.first().is_some_and(u8::is_ascii_alphanumeric)
            || profile.len() > 128
            || !profile
                .iter()
                .all(|&b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(
                "SY_PROFILE must start with an ASCII letter or digit and contain only ASCII \
                 letters, digits, underscores, or hyphens, up to 128 characters."
                    .into(),
            );
        }
        if self.model.is_empty() || self.model.bytes().any(|b| b.is_ascii_control()) {
            return Err(
                "SY_MODEL must be a nonempty public route ID without ASCII control characters."
                    .into(),
            );
        }
        if self.port == 0 {
            return Err(PORT_ERROR.into());
        }
        for (name, path) in [
            ("HOME", &self.home),
            ("SY_HOME", &self.sy_home),
            ("CODEX_HOME", &self.codex_home),
            ("service directory", &self.service_dir),
            ("source checkout", &self.source),
            ("Cargo executable", &self.cargo),
        ]
        .into_iter()
        .chain(self.app_dir.as_ref().map(|path| ("LOCALAPPDATA", path)))
        {
            if !path.is_absolute()
                || path
                    .to_str()
                    .is_none_or(|s| s.bytes().any(|b| b.is_ascii_control()))
            {
                return Err(format!(
                    "{name} must be an absolute UTF-8 path without ASCII control characters."
                ));
            }
        }
        Ok(())
    }

    pub fn app(&self) -> PathBuf {
        self.home.join("Applications/Switchyard.app")
    }
    pub fn bin_dir(&self) -> PathBuf {
        if cfg!(windows) {
            self.app_dir
                .clone()
                .unwrap_or_else(|| self.home.join("AppData/Local/Programs/Switchyard"))
        } else {
            self.sy_home.join("bin")
        }
    }
    pub fn binary(&self, name: &str) -> PathBuf {
        self.bin_dir().join(executable_name(name))
    }
    pub fn desktop_settings(&self) -> PathBuf {
        self.sy_home.join("desktop.toml")
    }
    fn profile_file(&self) -> PathBuf {
        self.codex_home
            .join(format!("{}.config.toml", self.profile))
    }
    fn release(&self, binary: &str) -> PathBuf {
        self.source
            .join("target/release")
            .join(executable_name(binary))
    }
    /// Uninstall reads this copy of the settings to find the installation.
    fn install_record(&self) -> PathBuf {
        self.home.join(".switchyard-desktop-install.toml")
    }
}

/// The app reads this metadata to retain a custom SY_HOME when Finder or a Windows shortcut starts it.
pub fn bundle_metadata(executable: &Path) -> PathBuf {
    // Windows keeps the metadata beside the executable; macOS keeps it in the bundle's Resources.
    let (levels, file) = if cfg!(windows) {
        (1, "install.toml")
    } else {
        (2, "Resources/install.toml")
    };
    executable
        .ancestors()
        .nth(levels)
        .unwrap_or(Path::new("."))
        .join(file)
}

/// The updater runs separately because installation restarts the app that requested it.
pub fn start_update(settings_file: &Path) -> Result<PathBuf, String> {
    let settings = Settings::load(settings_file)?;
    let guard = update_lock(&settings)?;
    if update_starting(&update_status(settings_file)?) {
        return Err(
            "A Switchyard update is already starting. View its progress in Settings.".into(),
        );
    }
    set_update_state(
        &settings,
        "starting",
        "start",
        "Starting the source update.",
    )?;
    let log_path = settings.sy_home.join("logs/update.log");
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(err)?;
    let error_log = log.try_clone().map_err(err)?;
    #[cfg(windows)]
    let updater_path = windows::temporary_updater(&settings.binary("switchyard-desktop-install"))?;
    #[cfg(not(windows))]
    let updater_path = settings.binary("switchyard-desktop-install");
    let mut updater = Command::new(&updater_path);
    #[cfg(windows)]
    windows::hide_console(&mut updater);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // A separate process group keeps launchd from stopping the updater with the app.
        updater.process_group(0);
    }
    // The child must acquire the update lock, so the parent releases it before spawning.
    drop(guard);
    let mut child = updater
        .args(["update", "--settings"])
        .arg(settings_file)
        .stdin(Stdio::null())
        .stdout(log)
        .stderr(error_log)
        .spawn()
        .map_err(|e| {
            let message = format!("Could not start the source update: {e}");
            let _ = set_update_state(&settings, "failed", "start", &message);
            message
        })?;
    #[cfg(windows)]
    updater_path.keep().map_err(err)?;
    // This thread reaps the updater if it exits before installation stops the app.
    thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(log_path)
}

pub fn execute(settings: &Settings, action: &str, dry_run: bool) -> Result<(), String> {
    settings.validate()?;
    if !matches!(action, "install" | "uninstall" | "update") {
        return Err("Choose install, update, or uninstall.".into());
    }
    if !cfg!(any(target_os = "macos", target_os = "linux", windows)) {
        return Err("Installation supports macOS, Linux, and Windows.".into());
    }
    if dry_run {
        println!("{}", installation_plan(settings, action));
        return Ok(());
    }
    if action == "uninstall" {
        return uninstall(settings);
    }
    let _job = update_lock(settings)?;
    if action == "update" {
        set_update_state(settings, "running", "build", "Building the app and server.")?;
    }
    match install(settings, action == "update") {
        Ok(None) => set_update_state(
            settings,
            "completed",
            "completed",
            "Installed the app and server. Switchyard is ready.",
        ),
        Ok(Some(warning)) => set_update_state(settings, "warning", "readiness", &warning),
        Err(error) => {
            // The failure keeps the stage that the install last recorded.
            if settings.sy_home.exists() {
                let stage = update_status(&settings.sy_home.join("install.toml"))
                    .ok()
                    .and_then(|s| s["stage"].as_str().map(str::to_owned))
                    .unwrap_or_else(|| "installation".into());
                let _ = set_update_state(settings, "failed", &stage, &error);
            }
            Err(error)
        }
    }
}

// The persistent file keeps concurrent commands locking the same inode; exit releases the lock.
fn lock(path: &Path, busy: impl FnOnce(fs::TryLockError) -> String) -> Result<fs::File, String> {
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(err)?;
    file.try_lock().map_err(busy)?;
    Ok(file)
}
fn update_lock(settings: &Settings) -> Result<fs::File, String> {
    lock(&settings.home.join(UPDATE_LOCK), |_| {
        "A Switchyard update is already running. Open Settings to view its progress.".into()
    })
}
fn lock_installation(s: &Settings) -> Result<fs::File, String> {
    lock(&s.home.join(".switchyard-desktop-install.lock"), |e| {
        format!("Could not lock the desktop installation: {e}")
    })
}
fn set_update_state(
    settings: &Settings,
    state: &str,
    stage: &str,
    message: &str,
) -> Result<(), String> {
    let updated_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(err)?
        .as_secs();
    let record = serde_json::json!({"state": state, "stage": stage, "message": message,
        "source": settings.source, "updated_at": updated_at});
    write(
        &settings.sy_home.join("update-status.toml"),
        toml::to_string(&record).map_err(err)?.as_bytes(),
        false,
    )
}
// A marker under 10 seconds old marks the child as starting before it acquires the lock.
fn update_starting(record: &serde_json::Value) -> bool {
    record["state"] == "starting"
        && record["updated_at"].as_u64().is_some_and(|stamp| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .is_ok_and(|now| now.as_secs().saturating_sub(stamp) < 10)
        })
}
pub fn update_status(settings_file: &Path) -> Result<serde_json::Value, String> {
    let settings = Settings::load(settings_file)?;
    let Some(text) = read_optional(&settings.sy_home.join("update-status.toml")).map_err(err)?
    else {
        return Ok(serde_json::json!({"state":"idle"}));
    };
    let value: toml::Value = toml::from_str(&text).map_err(err)?;
    let mut record = serde_json::to_value(value).map_err(err)?;
    if !update_starting(&record)
        && matches!(record["state"].as_str(), Some("starting" | "running"))
        && let Ok(lock) = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(settings.home.join(UPDATE_LOCK))
        && lock.try_lock().is_ok()
    {
        record["state"] = serde_json::json!("interrupted");
        record["message"] = serde_json::json!(
            "The update stopped before recording completion. Open the update log, then retry."
        );
    }
    Ok(record)
}
pub fn preview_update(settings_file: &Path) -> Result<serde_json::Value, String> {
    let settings = Settings::load(settings_file)?;
    let git = |args: &[&str]| -> String {
        let mut command = Command::new("git");
        #[cfg(windows)]
        windows::hide_console(&mut command);
        command
            .arg("-C")
            .arg(&settings.source)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "Unavailable".into())
    };
    Ok(
        serde_json::json!({"source":settings.source,"revision":git(&["rev-parse","HEAD"]),"branch":git(&["branch","--show-current"]),"changes":git(&["status","--short"]),"plan":installation_plan(&settings,"update"),"status":update_status(settings_file)?}),
    )
}
fn installation_plan(s: &Settings, action: &str) -> String {
    let removing = action == "uninstall";
    let pick = |remove, install| if removing { remove } else { install };
    let app = if cfg!(target_os = "macos") {
        "macOS desktop app"
    } else if cfg!(windows) {
        "Windows desktop app"
    } else {
        "Linux terminal app"
    };
    let profile = s.profile_file();
    let profile_action = if removing {
        "Preserve recovery copy, then remove"
    } else if !profile.exists() {
        "Create"
    } else if action == "update" {
        "Preserve"
    } else {
        "Back up, then update"
    };
    let binaries = if !removing {
        "Replace"
    } else if cfg!(windows) {
        "Remove"
    } else {
        "Preserve"
    };
    let mut rows = vec![
        format!(
            "{} {app} and local server from {}",
            pick("Remove", "Build and install"),
            s.source.display()
        ),
        format!("Route: {} · Server: http://127.0.0.1:{}", s.model, s.port),
        format!(
            "{} service definitions in {}. The server {}.",
            pick("Remove", "Replace"),
            s.service_dir.display(),
            pick("will stop", "will restart after build and validation")
        ),
    ];
    for (path, label) in [
        (s.desktop_settings(), "app settings"),
        (s.sy_home.join("composite.toml"), "server config"),
        (s.sy_home.join("routing.jsonl"), "usage history"),
    ] {
        let verb = if removing || path.exists() {
            "Preserve"
        } else {
            "Create"
        };
        rows.push(format!("{verb} {label}: {}", path.display()));
    }
    rows.push(format!(
        "{profile_action} Codex profile: {}",
        profile.display()
    ));
    rows.push(format!("{binaries} binaries: {}", s.bin_dir().display()));
    rows.push(format!(
        "Preserve accounts and logs: {}",
        s.sy_home.display()
    ));
    if cfg!(target_os = "macos") {
        rows.push(format!(
            "{} app bundle: {}",
            pick("Remove", "Replace"),
            s.app().display()
        ));
    }
    rows.push(
        "Preview changes no files. Build prerequisites, full server config validation, \
         and readiness are checked during installation."
            .into(),
    );
    if removing {
        rows.extend(removal_notes(s));
    }
    rows.join("\n")
}

/// Lists the coding-tool files that uninstall leaves in place for the user to review.
fn removal_notes(s: &Settings) -> Vec<String> {
    let mut destinations = vec![
        s.codex_home.join("sy.config.toml"),
        s.codex_home.join("config.toml"),
        s.home.join(".claude/settings.json"),
        s.home.join(".pi/agent/models.json"),
        s.home.join(".pi/agent/settings.json"),
    ];
    for (tool, file) in [("codex", "config.toml"), ("claude", "settings.json")] {
        let root = s.home.join(".switchyard/accounts").join(tool);
        // symlink_metadata does not follow a symlink, so a linked accounts directory is skipped.
        if fs::symlink_metadata(&root).is_ok_and(|m| m.is_dir())
            && let Ok(accounts) = fs::read_dir(&root)
        {
            let accounts = accounts
                .flatten()
                .filter(|a| a.file_type().is_ok_and(|t| t.is_dir()));
            destinations.extend(accounts.map(|account| account.path().join(file)));
        }
    }
    let mut notes = Vec::new();
    for destination in destinations {
        let saved = ["switchyard-original", "switchyard-original-missing"]
            .iter()
            .any(|suffix| PathBuf::from(format!("{}.{suffix}", destination.display())).exists());
        if saved {
            notes.push(format!(
                "Preserve coding-tool destination with a Switchyard backup: {}. \
                 Restore it in Connect tools before removal.",
                destination.display()
            ));
        }
    }
    for path in [
        s.codex_home.join("config.sy.toml"),
        s.codex_home.join("config.toml.direct"),
        s.home.join(".zshrc"),
        s.home.join(".bashrc"),
    ] {
        if path.exists() {
            notes.push(format!(
                "Review legacy Switchyard cleanup: {}. Active defaults are copied before \
                 restoring config.toml.direct; marked shell/profile blocks are removed.",
                path.display()
            ));
        }
    }
    notes.push("Custom destinations outside these known locations are not inventoried.".into());
    notes.push(
        "Restore routing from Connect tools before uninstalling. \
         Custom and account tool settings are preserved and may still point at this server."
            .into(),
    );
    notes
}

fn wait_for_health(port: u16) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    let address: SocketAddr = ([127, 0, 0, 1], port).into();
    while Instant::now() < deadline {
        if let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(250)) {
            let _ = stream.set_read_timeout(Some(Duration::from_millis(300)));
            let _ = stream.set_write_timeout(Some(Duration::from_millis(300)));
            if write!(
                stream,
                "GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
            )
            .is_ok()
            {
                let mut response = String::new();
                if stream.take(8192).read_to_string(&mut response).is_ok()
                    && let Some((headers, body)) = response.split_once("\r\n\r\n")
                    && headers
                        .lines()
                        .next()
                        .and_then(|line| line.split_whitespace().nth(1))
                        == Some("200")
                    && serde_json::from_str::<serde_json::Value>(body)
                        .is_ok_and(|v| v["status"] == "ok")
                {
                    return true;
                }
            }
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}

fn run(command: &mut Command) -> Result<(), String> {
    #[cfg(windows)]
    windows::hide_console(command);
    let invocation = format!(
        "{:?} {:?}",
        command.get_program(),
        command.get_args().collect::<Vec<_>>()
    );
    println!("Running {invocation}");
    let status = command
        .status()
        .map_err(|e| format!("Could not run {invocation}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{invocation} exited with {status}"))
    }
}

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn present(path: &Path) -> Result<bool, String> {
    path.try_exists().map_err(err)
}

/// Reads a text file; a missing file is `None`.
fn read_optional(path: &Path) -> std::io::Result<Option<String>> {
    match fs::read_to_string(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        result => result.map(Some),
    }
}

// Atomic replacement keeps existing executables readable while an update is staged.
fn write(path: &Path, bytes: &[u8], executable: bool) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or("The destination has no parent directory.")?;
    fs::create_dir_all(parent).map_err(err)?;
    let mut temp = tempfile::NamedTempFile::new_in(parent).map_err(err)?;
    temp.write_all(bytes).map_err(err)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = if executable { 0o755 } else { 0o600 };
        temp.as_file()
            .set_permissions(fs::Permissions::from_mode(mode))
            .map_err(err)?;
    }
    #[cfg(not(unix))]
    let _ = executable;
    temp.as_file().sync_all().map_err(err)?;
    temp.persist(path)
        .map_err(|e| format!("Write {}: {e}", path.display()))?;
    Ok(())
}

fn write_once(path: &Path, text: &str) -> Result<(), String> {
    if present(path)? {
        return Ok(());
    }
    write(path, text.as_bytes(), false)
}

fn backup(path: &Path, label: &str) -> Result<(), String> {
    let name = path.file_name().ok_or("Missing file name.")?;
    let parent = path.parent().ok_or("Missing parent directory.")?;
    let mut copy = tempfile::Builder::new()
        .prefix(&format!("{}.{label}.", name.to_string_lossy()))
        .tempfile_in(parent)
        .map_err(err)?;
    copy.write_all(&fs::read(path).map_err(err)?).map_err(err)?;
    copy.as_file().sync_all().map_err(err)?;
    let (_, saved) = copy.keep().map_err(err)?;
    println!("Preserved {} at {}", path.display(), saved.display());
    Ok(())
}

fn remove(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("Remove {}: {e}", path.display())),
    }
}

/// What an install gathers and checks before it changes the installation.
struct Prepared {
    /// The Codex profile to save.
    profile: String,
    /// App settings rewritten for a new port, when the port changes.
    desktop_settings: Option<String>,
    /// API key variables in the server config that have no saved key.
    missing_keys: Vec<String>,
}

// Every check and the build run before services stop, so a failure leaves the current
// installation running.
fn install(s: &Settings, preserve_profile: bool) -> Result<Option<String>, String> {
    let prepared = prepare(s, preserve_profile)?;
    let _lock = lock_installation(s)?;
    set_update_state(
        s,
        "running",
        "install",
        "Build and config validation passed. Replacing the installation.",
    )?;
    replace_installation(s, prepared.desktop_settings.as_deref())?;
    install_services(s, prepared.missing_keys.is_empty())?;
    if !preserve_profile || !present(&s.profile_file())? {
        save_profile(s, &prepared.profile)?;
    }
    report_readiness(s, &prepared.missing_keys)
}

fn prepare(s: &Settings, preserve_profile: bool) -> Result<Prepared, String> {
    let profile = s.profile_file();
    let context_window = route_context_window(s)?;
    let profile_text = if profile.exists() && !preserve_profile {
        let current = fs::read_to_string(&profile).map_err(err)?;
        merge_codex_profile(s, &current, context_window)?
    } else {
        codex_profile(s, context_window)
    };
    if cfg!(target_os = "macos") {
        run(Command::new("launchctl").arg("help"))?;
    } else if cfg!(target_os = "linux") {
        run(systemctl(&["show-environment"]).stdout(Stdio::null()))?;
    }
    let desktop_settings = reconcile_desktop_port(s)?;
    build(s)?;
    let missing_keys = check_server_config(s, !preserve_profile || !profile.exists())?;
    Ok(Prepared {
        profile: profile_text,
        desktop_settings,
        missing_keys,
    })
}

fn build(s: &Settings) -> Result<(), String> {
    println!(
        "Building the desktop app and server from {}",
        s.source.display()
    );
    // Cargo and rustc must come from the same toolchain, including GUI-triggered updates.
    let cargo_dir = s.cargo.parent().ok_or("Cargo has no parent directory.")?;
    let mut paths = vec![cargo_dir.to_path_buf()];
    if let Some(path) = std::env::var_os("PATH") {
        paths.extend(std::env::split_paths(&path));
    }
    run(Command::new(&s.cargo)
        .current_dir(&s.source)
        .args(["build", "--locked", "--release", "--manifest-path"])
        .arg(s.source.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(s.source.join("target"))
        .args(["-p", "switchyard-server", "-p", "switchyard-desktop"])
        .env("PATH", std::env::join_paths(paths).map_err(err)?))
}

/// Checks the server config that the install keeps, or the bundled one on a first install, and
/// returns the API key variables that have no saved key.
fn check_server_config(s: &Settings, require_route: bool) -> Result<Vec<String>, String> {
    let config = s.sy_home.join("composite.toml");
    let existing = present(&config)?;
    let staged = tempfile::NamedTempFile::new().map_err(err)?;
    let checked = if existing {
        config.as_path()
    } else {
        fs::write(staged.path(), BUNDLED_CONFIG).map_err(err)?;
        staged.path()
    };
    // Install never saves keys; the app's Settings → Server keys does. Placeholder values let the
    // dry run check the config without reading any key.
    let mut checker = Command::new(s.release("switchyard-server"));
    for name in credentials::variables(checked)? {
        checker.env(name, "placeholder-for-config-check");
    }
    run(checker.arg("--config").arg(checked).arg("--dry-run"))?;
    // Finding an item does not read its key, so this check never asks for Keychain access.
    let missing = credentials::status(checked)?
        .into_iter()
        .filter(|key| !key.saved)
        .map(|key| key.name)
        .collect();
    if require_route {
        check_public_route(checked, &s.model)?;
    }
    Ok(missing)
}

/// Stops the services and replaces the app files. The macOS app bundle is staged and signed
/// before the services stop.
fn replace_installation(s: &Settings, desktop_settings: Option<&str>) -> Result<(), String> {
    let metadata = toml::to_string(s).map_err(err)?;
    let bundle = cfg!(target_os = "macos")
        .then(|| stage_bundle(s, &metadata))
        .transpose()?;
    fs::create_dir_all(&s.sy_home).map_err(err)?;
    set_update_state(
        s,
        "running",
        "services",
        "Stopping services and replacing app files. Existing config and history are retained.",
    )?;
    stop_services(s)?;
    fs::create_dir_all(s.sy_home.join("logs")).map_err(err)?;
    #[cfg(windows)]
    windows::replace_binaries(s)?;
    #[cfg(not(windows))]
    for binary in [
        "switchyard-server",
        "switchyard-desktop",
        "switchyard-desktop-install",
    ] {
        let bytes = fs::read(s.release(binary)).map_err(err)?;
        write(&s.binary(binary), &bytes, true)?;
    }
    migrate_settings(s, desktop_settings)?;
    remove(&s.sy_home.join("bin/switchyard-menubar"))?;
    for record in [s.sy_home.join("install.toml"), s.install_record()] {
        write(&record, metadata.as_bytes(), false)?;
    }
    if let Some(stage) = bundle {
        replace_app(s, stage)?;
    }
    Ok(())
}

/// Creates the server config and app settings when they are missing and moves the old menu bar
/// settings to desktop.toml. `desktop_settings` replaces the app settings when the port changed.
fn migrate_settings(s: &Settings, desktop_settings: Option<&str>) -> Result<(), String> {
    write_once(&s.sy_home.join("composite.toml"), BUNDLED_CONFIG)?;
    let legacy = s.sy_home.join("menubar.toml");
    if present(&legacy)? && !present(&s.desktop_settings())? {
        fs::rename(&legacy, s.desktop_settings()).map_err(err)?;
    }
    write_once(&s.desktop_settings(), &desktop_config(s))?;
    match desktop_settings {
        Some(text) => write(&s.desktop_settings(), text.as_bytes(), false),
        None => Ok(()),
    }
}

fn save_profile(s: &Settings, text: &str) -> Result<(), String> {
    set_update_state(
        s,
        "running",
        "profile",
        "Saving the selected Codex profile.",
    )?;
    let profile = s.profile_file();
    if present(&profile)? && fs::read(&profile).map_err(err)? != text.as_bytes() {
        backup(&profile, "switchyard-backup")?;
    }
    write(&profile, text.as_bytes(), false)
}

/// Returns the warning to record when the server cannot start yet or does not answer /health.
fn report_readiness(s: &Settings, missing_keys: &[String]) -> Result<Option<String>, String> {
    println!(
        "Installed. Server: http://127.0.0.1:{}\nSettings: {}\nUse: codex -p {}",
        s.port,
        s.desktop_settings().display(),
        s.profile
    );
    set_update_state(
        s,
        "running",
        "readiness",
        "Installed. Waiting for the server to answer /health.",
    )?;
    if !missing_keys.is_empty() {
        let next = if cfg!(any(target_os = "macos", windows)) {
            "Open Switchyard, go to Settings → Server keys, save each key, \
             and restart the server."
                .to_string()
        } else {
            format!(
                "Save each key with {} key set NAME, \
                 then run systemctl --user restart switchyard.",
                s.binary("switchyard-desktop-install").display()
            )
        };
        let message = format!(
            "Installed the app and server. The server starts after you save a key for {}. {next}",
            missing_keys.join(", ")
        );
        println!("{message}");
        return Ok(Some(message));
    }
    if wait_for_health(s.port) {
        return Ok(None);
    }
    let message = format!(
        "Installed the app and server, but Switchyard did not answer /health. \
         Check logs at {} and restart the server.",
        s.sy_home.join("logs").display()
    );
    println!("Warning: {message}");
    Ok(Some(message))
}

fn routes(config: &toml::Value) -> impl Iterator<Item = &toml::Value> {
    config
        .get("routes")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flat_map(|routes| routes.values())
}

fn check_public_route(config: &Path, model: &str) -> Result<(), String> {
    let text = fs::read_to_string(config).map_err(err)?;
    let document: toml::Value = toml::from_str(&text).map_err(err)?;
    let ids: Vec<&str> = routes(&document)
        .filter_map(|route| route.get("id")?.as_str())
        .collect();
    if ids.contains(&model) {
        return Ok(());
    }
    Err(format!(
        "Route {model:?} is not in {}. Choose SY_MODEL from: {}. \
         No services or coding-tool settings were changed.",
        config.display(),
        ids.join(", ")
    ))
}

fn reconcile_desktop_port(s: &Settings) -> Result<Option<String>, String> {
    let path = if s.desktop_settings().exists() {
        s.desktop_settings()
    } else {
        s.sy_home.join("menubar.toml")
    };
    let Some(text) = read_optional(&path).map_err(|e| format!("Read {}: {e}", path.display()))?
    else {
        return Ok(None);
    };
    let mut document: toml_edit::DocumentMut = text.parse().map_err(|e| {
        format!(
            "Parse {}: {e}. Fix this file before installing.",
            path.display()
        )
    })?;
    let previous = s.sy_home.join("install.toml");
    let old_port = if previous.exists() {
        Settings::load(&previous)?.port
    } else {
        4123
    };
    let expected = format!("http://127.0.0.1:{old_port}");
    let current = document
        .get("server_url")
        .map(|value| {
            value.as_str().ok_or_else(|| {
                format!(
                    "The server_url value in {} must be a string: invalid data. \
                     Fix this file before installing.",
                    path.display()
                )
            })
        })
        .transpose()?;
    if s.port != old_port && current.is_some_and(|url| url != expected) {
        return Err(format!(
            "{} uses a custom server_url, but SY_PORT changes from {old_port} to {}. \
             Review the app URL and port together before reinstalling. No services were stopped.",
            path.display(),
            s.port
        ));
    }
    // Unchanged and custom URLs must retain the original TOML bytes, including quotes and line endings.
    if current.is_some_and(|url| url != expected || s.port == old_port) {
        return Ok(None);
    }
    document["server_url"] = toml_edit::value(format!("http://127.0.0.1:{}", s.port));
    let updated = document.to_string();
    Ok((updated != text).then_some(updated))
}

// Returns the `context_window` of the route that SY_MODEL names, from the existing server
// config or, on a new install, the bundled one. A missing route returns `None` here;
// check_public_route rejects it later with the list of valid route IDs.
fn route_context_window(s: &Settings) -> Result<Option<i64>, String> {
    let path = s.sy_home.join("composite.toml");
    let text = read_optional(&path).map_err(|e| format!("Read {}: {e}", path.display()))?;
    let document: toml::Value = toml::from_str(text.as_deref().unwrap_or(BUNDLED_CONFIG))
        .map_err(|e| format!("Parse {}: {e}", path.display()))?;
    Ok(routes(&document)
        .find(|route| route.get("id").and_then(toml::Value::as_str) == Some(&s.model))
        .and_then(|route| route.get("context_window")?.as_integer()))
}

// Codex compacts a thread when its usage nears the model's context window. Codex does not
// know a Switchyard route ID, so the profile names the route's window. Without it, Codex
// never compacts on the route, and a long thread fails when the upstream rejects it.
fn codex_profile(s: &Settings, context_window: Option<i64>) -> String {
    let mut value: toml::Value = toml::from_str(
        include_str!("../../../scripts/config/codex.sy.toml")
            .replace("@SY_PORT@", &s.port.to_string())
            .as_str(),
    )
    .expect("bundled profile is valid TOML");
    value["model"] = s.model.clone().into();
    if let (Some(window), Some(table)) = (context_window, value.as_table_mut()) {
        table.insert("model_context_window".into(), window.into());
    }
    toml::to_string(&value).expect("profile values serialize")
}
fn merge_codex_profile(
    s: &Settings,
    current: &str,
    context_window: Option<i64>,
) -> Result<String, String> {
    let mut document: toml_edit::DocumentMut = current
        .parse()
        .map_err(|e| format!("Read Codex profile: {e}"))?;
    let template: toml_edit::DocumentMut = codex_profile(s, context_window)
        .parse()
        .expect("generated profile is valid TOML");
    for key in ["model", "model_provider"] {
        document[key] = template[key].clone();
    }
    // A window kept from an earlier route would make Codex compact at the wrong size.
    match template.get("model_context_window") {
        Some(window) => document["model_context_window"] = window.clone(),
        None => {
            document.remove("model_context_window");
        }
    }
    if document.get("model_providers").is_none() {
        document["model_providers"] = toml_edit::table();
    }
    let providers = &mut document["model_providers"];
    if !providers.is_table_like() {
        return Err(
            "Codex model_providers must be a table. No services or profiles were changed.".into(),
        );
    }
    let sy = providers.get("sy");
    if sy.is_some_and(|item| !item.is_table_like()) {
        return Err(
            "Codex model_providers.sy must be a table. No services or profiles were changed."
                .into(),
        );
    }
    if sy.is_none() {
        providers["sy"] = toml_edit::table();
    }
    for key in ["name", "base_url", "wire_api", "requires_openai_auth"] {
        providers["sy"][key] = template["model_providers"]["sy"][key].clone();
    }
    Ok(document.to_string())
}

fn desktop_config(s: &Settings) -> String {
    format!(
        "server_url = {:?}\n\
         routing_log = {}\n\
         config_file = {}\n\
         launchd_label = {:?}\n\
         refresh_seconds = 30\n\
         \n\
         {}",
        format!("http://127.0.0.1:{}", s.port),
        toml_string(&s.sy_home.join("routing.jsonl").to_string_lossy()),
        toml_string(&s.sy_home.join("composite.toml").to_string_lossy()),
        SERVER,
        include_str!("../prices.toml")
    )
}
fn toml_string(value: &str) -> String {
    toml::Value::String(value.into()).to_string()
}
fn xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
fn plist(label: &str, args: &[String], log: &Path, app: bool) -> String {
    let args: String = args
        .iter()
        .map(|a| format!("<string>{}</string>", xml(a)))
        .collect();
    let keep_alive = if app {
        "<dict><key>SuccessfulExit</key><false/></dict>\
         <key>LimitLoadToSessionType</key><string>Aqua</string>"
    } else {
        "<true/><key>EnvironmentVariables</key><dict>\
         <key>RUST_LOG</key><string>info</string></dict>"
    };
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \
         \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\
         <plist version=\"1.0\"><dict>\
         <key>Label</key><string>{label}</string>\
         <key>ProgramArguments</key><array>{args}</array>\
         <key>RunAtLoad</key><true/>\
         <key>KeepAlive</key>{keep_alive}\
         <key>StandardOutPath</key><string>{}</string>\
         <key>StandardErrorPath</key><string>{}</string>\
         </dict></plist>",
        xml(&log.with_extension("log").to_string_lossy()),
        xml(&log.with_extension("err.log").to_string_lossy())
    )
}

fn domain() -> Result<String, String> {
    let output = Command::new("id").arg("-u").output().map_err(err)?;
    if !output.status.success() {
        return Err("Could not read the current user ID.".into());
    }
    Ok(format!(
        "gui/{}",
        String::from_utf8_lossy(&output.stdout).trim()
    ))
}
fn systemctl(args: &[&str]) -> Command {
    let mut command = Command::new("systemctl");
    command.arg("--user").args(args);
    command
}

fn stop_services(s: &Settings) -> Result<(), String> {
    if cfg!(target_os = "macos") {
        let domain = domain()?;
        for label in [SERVER, DESKTOP, LEGACY_DESKTOP] {
            stop_launchd_job(&format!("{domain}/{label}"))?;
        }
        remove(&s.service_dir.join(format!("{LEGACY_DESKTOP}.plist")))?;
    } else if cfg!(windows) {
        #[cfg(windows)]
        windows::stop(s)?;
    } else if present(&s.service_dir.join(UNIT))? {
        run(&mut systemctl(&["stop", UNIT]))?;
    }
    Ok(())
}
fn install_services(s: &Settings, keys_saved: bool) -> Result<(), String> {
    if cfg!(target_os = "macos") {
        install_launch_agents(s)?;
    } else if cfg!(windows) {
        #[cfg(windows)]
        windows::install(s)?;
    } else {
        install_systemd_unit(s, keys_saved)?;
    }
    Ok(())
}
/// Removes the service definitions and the app. SY_HOME keeps settings, accounts, and history.
fn remove_app_and_services(s: &Settings) -> Result<(), String> {
    if cfg!(target_os = "macos") {
        for label in [SERVER, DESKTOP, LEGACY_DESKTOP] {
            remove(&s.service_dir.join(format!("{label}.plist")))?;
        }
        if present(&s.app())? {
            fs::remove_dir_all(s.app()).map_err(err)?;
        }
    } else if cfg!(windows) {
        #[cfg(windows)]
        windows::uninstall(s)?;
    } else {
        let unit = s.service_dir.join(UNIT);
        if present(&unit)? {
            run(&mut systemctl(&["disable", UNIT]))?;
            remove(&unit)?;
        }
        run(&mut systemctl(&["daemon-reload"]))?;
    }
    Ok(())
}

fn launchd_job_loaded(job: &str) -> Result<bool, String> {
    Command::new("launchctl")
        .args(["print", job])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .map_err(err)
}
fn stop_launchd_job(job: &str) -> Result<(), String> {
    if !launchd_job_loaded(job)? {
        return Ok(());
    }
    run(Command::new("launchctl").args(["bootout", job]))?;
    for _ in 0..50 {
        if !launchd_job_loaded(job)? {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err(format!(
        "{job} did not stop within five seconds. Try again."
    ))
}

fn stage_bundle(s: &Settings, metadata: &str) -> Result<tempfile::TempDir, String> {
    let app = s.app();
    let parent = app.parent().ok_or("The app has no parent directory.")?;
    fs::create_dir_all(parent).map_err(err)?;
    let stage = tempfile::tempdir_in(parent).map_err(err)?;
    let app = stage.path().join("Switchyard.app");
    let info = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <plist version=\"1.0\"><dict>\
         <key>CFBundleIdentifier</key><string>com.nvidia.switchyard</string>\
         <key>CFBundleName</key><string>Switchyard</string>\
         <key>CFBundleDisplayName</key><string>Switchyard</string>\
         <key>CFBundleExecutable</key><string>switchyard-desktop</string>\
         <key>CFBundleIconFile</key><string>Switchyard.icns</string>\
         <key>CFBundlePackageType</key><string>APPL</string>\
         <key>CFBundleShortVersionString</key><string>{}</string>\
         <key>LSMinimumSystemVersion</key><string>12.0</string>\
         <key>NSHighResolutionCapable</key><true/>\
         </dict></plist>",
        env!("CARGO_PKG_VERSION")
    );
    let desktop = fs::read(s.release("switchyard-desktop")).map_err(err)?;
    for (file, bytes, executable) in [
        ("MacOS/switchyard-desktop", &desktop[..], true),
        ("Resources/install.toml", metadata.as_bytes(), false),
        (
            "Resources/Switchyard.icns",
            &include_bytes!("../icons/icon.icns")[..],
            false,
        ),
        ("Info.plist", info.as_bytes(), false),
    ] {
        write(&app.join("Contents").join(file), bytes, executable)?;
    }
    run(Command::new("codesign")
        .args(["--force", "--deep", "--sign", "-"])
        .arg(&app))?;
    Ok(stage)
}

fn replace_app(s: &Settings, stage: tempfile::TempDir) -> Result<(), String> {
    let previous = stage.path().join("Previous.app");
    let existed = present(&s.app())?;
    if existed {
        fs::rename(s.app(), &previous).map_err(err)?;
    }
    let Err(error) = fs::rename(stage.path().join("Switchyard.app"), s.app()) else {
        return Ok(());
    };
    if existed {
        fs::rename(&previous, s.app())
            .map_err(|restore| format!("Replace app: {error}. Restore previous app: {restore}"))?;
    }
    Err(format!("Replace app: {error}"))
}

fn install_launch_agents(s: &Settings) -> Result<(), String> {
    let server = [
        s.binary("switchyard-desktop-install").display().to_string(),
        "serve".into(),
        "--settings".into(),
        s.sy_home.join("install.toml").display().to_string(),
    ];
    let desktop = [
        s.app()
            .join("Contents/MacOS/switchyard-desktop")
            .display()
            .to_string(),
        s.desktop_settings().display().to_string(),
    ];
    for (label, args, log, app) in [
        (SERVER, &server[..], "logs/server", false),
        (DESKTOP, &desktop[..], "logs/desktop", true),
    ] {
        let job = plist(label, args, &s.sy_home.join(log), app);
        write(
            &s.service_dir.join(format!("{label}.plist")),
            job.as_bytes(),
            false,
        )?;
    }
    let domain = domain()?;
    for label in [SERVER, DESKTOP] {
        run(Command::new("launchctl")
            .arg("bootstrap")
            .arg(&domain)
            .arg(s.service_dir.join(format!("{label}.plist"))))?;
        // macOS can defer RunAtLoad jobs until an explicit start is requested.
        run(Command::new("launchctl")
            .arg("kickstart")
            .arg(format!("{domain}/{label}")))?;
    }
    Ok(())
}

fn systemd_arg(path: &Path) -> String {
    // systemd expands percent specifiers and environment variables inside quoted arguments.
    format!(
        "\"{}\"",
        path.to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('%', "%%")
            .replace('$', "$$")
    )
}
fn install_systemd_unit(s: &Settings, expect_running: bool) -> Result<(), String> {
    let unit = format!(
        "[Unit]\n\
         Description=Switchyard local server\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={} serve --settings {}\n\
         Restart=on-failure\n\
         RestartSec=10\n\
         Environment=RUST_LOG=info\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n",
        systemd_arg(&s.binary("switchyard-desktop-install")),
        systemd_arg(&s.sy_home.join("install.toml")),
    );
    write(&s.service_dir.join(UNIT), unit.as_bytes(), false)?;
    run(&mut systemctl(&["daemon-reload"]))?;
    run(&mut systemctl(&["enable", UNIT]))?;
    run(&mut systemctl(&["restart", UNIT]))?;
    // A server waiting for a key exits and restarts until the key is saved.
    if !expect_running {
        return Ok(());
    }
    thread::sleep(Duration::from_secs(2));
    run(&mut systemctl(&["is-active", "--quiet", UNIT]))
        .map_err(|e| format!("{e}. See journalctl --user -u switchyard."))
}

fn strip_block(path: &Path, start: &str, end: &str) -> Result<(), String> {
    let Some(text) = read_optional(path).map_err(err)? else {
        return Ok(());
    };
    let mut output = String::new();
    let mut skipping = false;
    for line in text.split_inclusive('\n') {
        skipping |= line.contains(start);
        if !skipping {
            output.push_str(line);
        }
        if line.contains(end) {
            skipping = false;
        }
    }
    if skipping {
        return Err(format!(
            "{} has an unfinished Switchyard block. No edits were made to this file.",
            path.display()
        ));
    }
    if text != output {
        write(path, output.as_bytes(), false)?;
    }
    Ok(())
}

fn uninstall(s: &Settings) -> Result<(), String> {
    let _lock = lock_installation(s)?;
    println!("{}", installation_plan(s, "uninstall"));
    let profile = s.profile_file();
    if profile.exists() {
        backup(&profile, "switchyard-uninstall-recovery")?;
    }
    stop_services(s)?;
    remove_app_and_services(s)?;
    remove(&profile)?;
    remove_legacy_setup(s)?;
    println!(
        "Removed services, app, and profile. Kept settings, accounts, {}and history at {}.",
        if cfg!(windows) { "" } else { "binaries, " },
        s.sy_home.display()
    );
    let recorded = s.install_record();
    if recorded.exists() && Settings::load(&recorded)?.sy_home == s.sy_home {
        remove(&recorded)?;
    }
    Ok(())
}

/// Removes legacy Switchyard edits to Codex and shell startup files. The active config.toml is
/// backed up before config.toml.direct replaces it.
fn remove_legacy_setup(s: &Settings) -> Result<(), String> {
    remove(&s.codex_home.join("config.sy.toml"))?;
    let config = s.codex_home.join("config.toml");
    strip_block(
        &config,
        "# >>> switchyard sy profile >>>",
        "# <<< switchyard sy profile <<<",
    )?;
    let direct = s.codex_home.join("config.toml.direct");
    if present(&direct)? {
        if present(&config)? {
            backup(&config, "switchyard-current")?;
        }
        write(&config, &fs::read(&direct).map_err(err)?, false)?;
        remove(&direct)?;
    }
    for name in [".zshrc", ".bashrc"] {
        strip_block(
            &s.home.join(name),
            "# >>> switchyard codex alias >>>",
            "# <<< switchyard codex alias <<<",
        )?;
    }
    Ok(())
}
