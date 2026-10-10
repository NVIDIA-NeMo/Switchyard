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

#[cfg(windows)]
pub mod windows;

pub fn executable_name(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

const SERVER: &str = "com.nvidia.switchyard.server";
const DESKTOP: &str = "com.nvidia.switchyard.desktop";
// Migration removes this LaunchAgent so the old app cannot restart alongside the desktop app.
const LEGACY_DESKTOP: &str = "com.nvidia.switchyard.menubar";

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
        let recorded = defaults.home.join(".switchyard-desktop-install.toml");
        let selected = if recorded.try_exists().map_err(|e| e.to_string())? {
            Self::load(&recorded)?
        } else {
            return Ok(defaults);
        };
        for (name, value, saved) in [
            (
                "SY_HOME",
                defaults.sy_home.as_os_str(),
                selected.sy_home.as_os_str(),
            ),
            (
                "CODEX_HOME",
                defaults.codex_home.as_os_str(),
                selected.codex_home.as_os_str(),
            ),
            (
                "SY_PROFILE",
                std::ffi::OsStr::new(&defaults.profile),
                std::ffi::OsStr::new(&selected.profile),
            ),
        ] {
            if std::env::var_os(name).is_some() && value != saved {
                return Err(format!(
                    "{name} differs from the recorded installation at {}. Use uninstall --settings FILE to choose a different installation.",
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
        let config_home = path("XDG_CONFIG_HOME", home.join(".config"));
        let value = |name: &str, default: &str| {
            std::env::var(name).or_else(|error| match error {
                std::env::VarError::NotPresent => Ok(default.to_owned()),
                std::env::VarError::NotUnicode(_) => Err(format!("{name} must be UTF-8.")),
            })
        };
        let port = value("SY_PORT", "4123")?;
        if !port.bytes().all(|b| b.is_ascii_digit()) {
            return Err("SY_PORT must be a number from 1 to 65535.".into());
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
                config_home.join("systemd/user")
            },
            app_dir: if cfg!(windows) {
                Some(path("LOCALAPPDATA", home.join("AppData/Local")).join("Programs/Switchyard"))
            } else {
                None
            },
            home,
            source: Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .ok_or("Could not find the source checkout.")?
                .to_path_buf(),
            cargo: PathBuf::from(env!("CARGO")),
            port: port
                .parse()
                .map_err(|_| "SY_PORT must be a number from 1 to 65535.")?,
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
        if self.profile.is_empty()
            || self.profile.len() > 128
            || !self.profile.as_bytes()[0].is_ascii_alphanumeric()
            || !self
                .profile
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err("SY_PROFILE must start with an ASCII letter or digit and contain only ASCII letters, digits, underscores, or hyphens, up to 128 characters.".into());
        }
        if self.model.is_empty() || self.model.bytes().any(|b| b.is_ascii_control()) {
            return Err(
                "SY_MODEL must be a nonempty public route ID without ASCII control characters."
                    .into(),
            );
        }
        if self.port == 0 {
            return Err("SY_PORT must be a number from 1 to 65535.".into());
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
}

/// The app reads this metadata to retain a custom SY_HOME when Finder or a Windows shortcut starts it.
pub fn bundle_metadata(executable: &Path) -> PathBuf {
    if cfg!(windows) {
        return executable
            .parent()
            .unwrap_or(Path::new("."))
            .join("install.toml");
    }
    executable
        .parent()
        .and_then(Path::parent)
        .unwrap_or(Path::new("."))
        .join("Resources/install.toml")
}

/// The updater runs separately because installation restarts the app that requested it.
pub fn start_update(settings_file: &Path) -> Result<PathBuf, String> {
    let settings = Settings::load(settings_file)?;
    let guard = update_lock(&settings)?;
    let status = update_status(settings_file)?;
    if update_starting(&status) {
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
        .map_err(|e| e.to_string())?;
    let error_log = log.try_clone().map_err(|e| e.to_string())?;
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
    updater_path.keep().map_err(|e| e.to_string())?;
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
        println!("{}", installation_plan(settings, action)?);
        return Ok(());
    }
    if action == "uninstall" {
        uninstall(settings)
    } else {
        let _job = update_lock(settings)?;
        if action == "update" {
            set_update_state(settings, "running", "build", "Building the app and server.")?;
        }
        let result = install(settings, action == "update");
        match &result {
            Ok(ready) => set_update_state(
                settings,
                if *ready { "completed" } else { "warning" },
                if *ready { "completed" } else { "readiness" },
                if *ready {
                    "Installed the app and server. Switchyard is ready."
                } else {
                    "Installed the app and server, but Switchyard did not answer /health. Open the server log and retry its restart."
                },
            )?,
            Err(error) => {
                if settings.sy_home.exists() {
                    let stage = update_status(&settings.sy_home.join("install.toml"))
                        .ok()
                        .and_then(|s| s["stage"].as_str().map(str::to_owned))
                        .unwrap_or_else(|| "installation".into());
                    let _ = set_update_state(settings, "failed", &stage, error);
                }
            }
        }
        result.map(|_| ())
    }
}

fn update_lock(settings: &Settings) -> Result<fs::File, String> {
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(settings.home.join(".switchyard-desktop-update.lock"))
        .map_err(|e| e.to_string())?;
    file.try_lock().map_err(|_| {
        "A Switchyard update is already running. Open Settings to view its progress.".to_string()
    })?;
    Ok(file)
}
fn set_update_state(
    settings: &Settings,
    state: &str,
    stage: &str,
    message: &str,
) -> Result<(), String> {
    let record = serde_json::json!({"state":state,"stage":stage,"message":message,"source":settings.source,"updated_at":SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e|e.to_string())?.as_secs()});
    write(
        &settings.sy_home.join("update-status.toml"),
        toml::to_string(&record)
            .map_err(|e| e.to_string())?
            .as_bytes(),
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
    let path = settings.sy_home.join("update-status.toml");
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(serde_json::json!({"state":"idle"}));
        }
        Err(error) => return Err(error.to_string()),
    };
    let value: toml::Value = toml::from_str(&text).map_err(|e| e.to_string())?;
    let mut record = serde_json::to_value(value).map_err(|e| e.to_string())?;
    let starting = update_starting(&record);
    if !starting
        && matches!(record["state"].as_str(), Some("starting" | "running"))
        && let Ok(lock) = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(settings.home.join(".switchyard-desktop-update.lock"))
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
        serde_json::json!({"source":settings.source,"revision":git(&["rev-parse","HEAD"]),"branch":git(&["branch","--show-current"]),"changes":git(&["status","--short"]),"plan":installation_plan(&settings,"update")?,"status":update_status(settings_file)?}),
    )
}
fn installation_plan(s: &Settings, action: &str) -> Result<String, String> {
    let removing = action == "uninstall";
    let mut rows = vec![
        format!(
            "{} {} and local server from {}",
            if removing {
                "Remove"
            } else {
                "Build and install"
            },
            if cfg!(target_os = "macos") {
                "macOS desktop app"
            } else {
                if cfg!(windows) {
                    "Windows desktop app"
                } else {
                    "Linux terminal app"
                }
            },
            s.source.display()
        ),
        format!("Route: {} · Server: http://127.0.0.1:{}", s.model, s.port),
        format!(
            "{} service definitions in {}. The server {}.",
            if removing { "Remove" } else { "Replace" },
            s.service_dir.display(),
            if removing {
                "will stop"
            } else {
                "will restart after build and validation"
            }
        ),
    ];
    for (path, label) in [
        (s.desktop_settings(), "app settings"),
        (s.sy_home.join("composite.toml"), "server config"),
        (s.sy_home.join("routing.jsonl"), "usage history"),
    ] {
        rows.push(format!(
            "{} {label}: {}",
            if removing || path.exists() {
                "Preserve"
            } else {
                "Create"
            },
            path.display()
        ));
    }
    rows.push(format!(
        "{} Codex profile: {}",
        if removing {
            "Preserve recovery copy, then remove"
        } else if action == "update" && s.profile_file().exists() {
            "Preserve"
        } else if s.profile_file().exists() {
            "Back up, then update"
        } else {
            "Create"
        },
        s.profile_file().display()
    ));
    rows.push(format!(
        "{} binaries: {}",
        if removing { "Preserve" } else { "Replace" },
        s.bin_dir().display()
    ));
    rows.push(format!(
        "Preserve accounts and logs: {}",
        s.sy_home.display()
    ));
    if cfg!(target_os = "macos") {
        rows.push(format!(
            "{} app bundle: {}",
            if removing { "Remove" } else { "Replace" },
            s.app().display()
        ));
    }
    rows.push("Preview changes no files. Build prerequisites, full server config validation, and readiness are checked during installation.".into());
    if removing {
        let mut destinations = vec![
            s.codex_home.join("sy.config.toml"),
            s.codex_home.join("config.toml"),
            s.home.join(".claude/settings.json"),
            s.home.join(".pi/agent/models.json"),
            s.home.join(".pi/agent/settings.json"),
        ];
        for tool in ["codex", "claude"] {
            let root = s.home.join(".switchyard/accounts").join(tool);
            if fs::symlink_metadata(&root).is_ok_and(|m| m.is_dir() && !m.file_type().is_symlink())
                && let Ok(accounts) = fs::read_dir(&root)
            {
                for account in accounts.flatten() {
                    if account.file_type().is_ok_and(|t| t.is_dir()) {
                        destinations.push(account.path().join(if tool == "codex" {
                            "config.toml"
                        } else {
                            "settings.json"
                        }));
                    }
                }
            }
        }
        for destination in destinations {
            let backup = PathBuf::from(format!("{}.switchyard-original", destination.display()));
            let missing = PathBuf::from(format!(
                "{}.switchyard-original-missing",
                destination.display()
            ));
            if backup.exists() || missing.exists() {
                rows.push(format!("Preserve coding-tool destination with a Switchyard backup: {}. Restore it in Connect tools before removal.",destination.display()));
            }
        }
        for path in [
            s.codex_home.join("config.sy.toml"),
            s.codex_home.join("config.toml.direct"),
            s.home.join(".zshrc"),
            s.home.join(".bashrc"),
        ] {
            if path.exists() {
                rows.push(format!("Review legacy Switchyard cleanup: {}. Active defaults are copied before restoring config.toml.direct; marked shell/profile blocks are removed.",path.display()));
            }
        }
        rows.push("Custom destinations outside these known locations are not inventoried.".into());
        rows.push("Restore routing from Connect tools before uninstalling. Custom and account tool settings are preserved and may still point at this server.".into());
    }
    Ok(rows.join("\n"))
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

// Atomic replacement keeps existing executables readable while an update is staged.
fn write(path: &Path, bytes: &[u8], executable: bool) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or("The destination has no parent directory.")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let mut temp = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    temp.write_all(bytes).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        temp.as_file()
            .set_permissions(fs::Permissions::from_mode(if executable {
                0o755
            } else {
                0o600
            }))
            .map_err(|e| e.to_string())?;
    }
    #[cfg(not(unix))]
    let _ = executable;
    temp.as_file().sync_all().map_err(|e| e.to_string())?;
    temp.persist(path)
        .map_err(|e| format!("Write {}: {e}", path.display()))?;
    Ok(())
}

fn write_once(path: &Path, text: &str) -> Result<(), String> {
    if path.try_exists().map_err(|e| e.to_string())? {
        return Ok(());
    }
    write(path, text.as_bytes(), false)
}

fn backup(path: &Path, label: &str) -> Result<(), String> {
    let mut copy = tempfile::Builder::new()
        .prefix(&format!(
            "{}.{label}.",
            path.file_name()
                .ok_or("Missing file name.")?
                .to_string_lossy()
        ))
        .tempfile_in(path.parent().ok_or("Missing parent directory.")?)
        .map_err(|e| e.to_string())?;
    copy.write_all(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    copy.as_file().sync_all().map_err(|e| e.to_string())?;
    let (_, saved) = copy.keep().map_err(|e| e.to_string())?;
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

fn install(s: &Settings, preserve_profile: bool) -> Result<bool, String> {
    let profile = s.profile_file();
    let profile_text = if profile.exists() && !preserve_profile {
        merge_codex_profile(s, &fs::read_to_string(&profile).map_err(|e| e.to_string())?)?
    } else {
        codex_profile(s)
    };
    if cfg!(target_os = "macos") {
        run(Command::new("launchctl").arg("help"))?;
        run(Command::new("codesign").arg("--version"))?;
    } else if cfg!(target_os = "linux") {
        run(Command::new("systemctl").args(["--user", "show-environment"]))?;
    }
    let reconciled = reconcile_desktop_port(s)?;
    println!(
        "Building the desktop app and server from {}",
        s.source.display()
    );
    let mut build = Command::new(&s.cargo);
    build
        .current_dir(&s.source)
        .args(["build", "--locked", "--release", "--manifest-path"])
        .arg(s.source.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(s.source.join("target"))
        .args([
            "-p",
            "switchyard-server",
            "-p",
            "switchyard-desktop",
            "-p",
            "switchyard-desktop-install",
        ]);
    // Cargo and rustc must come from the same toolchain, including GUI-triggered updates.
    let cargo_dir = s.cargo.parent().ok_or("Cargo has no parent directory.")?;
    let mut paths = vec![cargo_dir.to_path_buf()];
    if let Some(path) = std::env::var_os("PATH") {
        paths.extend(std::env::split_paths(&path));
    }
    build.env(
        "PATH",
        std::env::join_paths(paths).map_err(|e| e.to_string())?,
    );
    run(&mut build)?;
    let config = s.sy_home.join("composite.toml");
    let existing = config.try_exists().map_err(|e| e.to_string())?;
    let staged = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
    if !existing {
        fs::write(
            staged.path(),
            include_str!("../../../scripts/config/composite.toml"),
        )
        .map_err(|e| e.to_string())?;
    }
    run(Command::new(s.release("switchyard-server"))
        .arg("--config")
        .arg(if existing {
            config.as_path()
        } else {
            staged.path()
        })
        .arg("--dry-run"))?;
    if !preserve_profile || !s.profile_file().exists() {
        check_public_route(if existing { &config } else { staged.path() }, &s.model)?;
    }
    let _lock = lock_installation(s)?;
    set_update_state(
        s,
        "running",
        "install",
        "Build and config validation passed. Replacing the installation.",
    )?;
    let metadata = toml::to_string(s).map_err(|e| e.to_string())?;
    let bundle = if cfg!(target_os = "macos") {
        Some(stage_bundle(s, &metadata)?)
    } else {
        None
    };
    set_update_state(
        s,
        "running",
        "services",
        "Stopping services and replacing app files. Existing config and history are retained.",
    )?;
    stop_services(s)?;
    fs::create_dir_all(s.sy_home.join("logs")).map_err(|e| e.to_string())?;
    #[cfg(windows)]
    windows::replace_binaries(s)?;
    #[cfg(not(windows))]
    for binary in [
        "switchyard-server",
        "switchyard-desktop",
        "switchyard-desktop-install",
    ] {
        write(
            &s.binary(binary),
            &fs::read(s.release(binary)).map_err(|e| e.to_string())?,
            true,
        )?;
    }
    write_once(
        &config,
        include_str!("../../../scripts/config/composite.toml"),
    )?;
    let legacy = s.sy_home.join("menubar.toml");
    if legacy.try_exists().map_err(|e| e.to_string())?
        && !s
            .desktop_settings()
            .try_exists()
            .map_err(|e| e.to_string())?
    {
        fs::rename(&legacy, s.desktop_settings()).map_err(|e| e.to_string())?;
    }
    write_once(&s.desktop_settings(), &desktop_config(s))?;
    if let Some(text) = reconciled {
        write(&s.desktop_settings(), text.as_bytes(), false)?;
    }
    remove(&s.sy_home.join("bin/switchyard-menubar"))?;
    write(&s.sy_home.join("install.toml"), metadata.as_bytes(), false)?;
    write(
        &s.home.join(".switchyard-desktop-install.toml"),
        metadata.as_bytes(),
        false,
    )?;
    if let Some(bundle) = bundle {
        install_macos(s, bundle)?;
    } else if cfg!(target_os = "linux") {
        install_linux(s)?;
    } else {
        #[cfg(windows)]
        windows::install(s)?;
    }
    if !preserve_profile || !profile.try_exists().map_err(|e| e.to_string())? {
        set_update_state(
            s,
            "running",
            "profile",
            "Saving the selected Codex profile.",
        )?;
        if profile.try_exists().map_err(|e| e.to_string())?
            && fs::read(&profile).map_err(|e| e.to_string())? != profile_text.as_bytes()
        {
            backup(&profile, "switchyard-backup")?;
        }
        write(&profile, profile_text.as_bytes(), false)?;
    }
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
    let ready = wait_for_health(s.port);
    if !ready {
        println!(
            "Warning: installed, but the server did not become ready. Check logs at {} and restart the server.",
            s.sy_home.join("logs").display()
        );
    }
    Ok(ready)
}

fn check_public_route(config: &Path, model: &str) -> Result<(), String> {
    let document: toml::Value =
        toml::from_str(&fs::read_to_string(config).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let ids = document
        .get("routes")
        .and_then(toml::Value::as_table)
        .map(|routes| {
            routes
                .values()
                .filter_map(|route| route.get("id").and_then(toml::Value::as_str))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if ids.contains(&model) {
        Ok(())
    } else {
        Err(format!(
            "Route {model:?} is not in {}. Choose SY_MODEL from: {}. No services or coding-tool settings were changed.",
            config.display(),
            ids.join(", ")
        ))
    }
}

fn reconcile_desktop_port(s: &Settings) -> Result<Option<String>, String> {
    let path = if s.desktop_settings().exists() {
        s.desktop_settings()
    } else {
        s.sy_home.join("menubar.toml")
    };
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("Read {}: {error}", path.display())),
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
            value
                .as_str()
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::InvalidData))
        })
        .transpose()
        .map_err(|e| {
            format!(
                "The server_url value in {} must be a string: {e}. Fix this file before installing.",
                path.display()
            )
        })?;
    if s.port != old_port && current.is_some_and(|url| url != expected) {
        return Err(format!(
            "{} uses a custom server_url, but SY_PORT changes from {old_port} to {}. Review the app URL and port together before reinstalling. No services were stopped.",
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

fn codex_profile(s: &Settings) -> String {
    let mut value: toml::Value = toml::from_str(
        include_str!("../../../scripts/config/codex.sy.toml")
            .replace("@SY_PORT@", &s.port.to_string())
            .as_str(),
    )
    .expect("bundled profile is valid TOML");
    value["model"] = s.model.clone().into();
    toml::to_string(&value).expect("profile values serialize")
}
fn merge_codex_profile(s: &Settings, current: &str) -> Result<String, String> {
    let mut document: toml_edit::DocumentMut = current
        .parse()
        .map_err(|e| format!("Read Codex profile: {e}"))?;
    let template: toml_edit::DocumentMut = codex_profile(s)
        .parse()
        .map_err(|e| format!("Read generated Codex profile: {e}"))?;
    for key in ["model", "model_provider"] {
        document[key] = template[key].clone();
    }
    if document.get("model_providers").is_none() {
        document["model_providers"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    if !document["model_providers"].is_table_like() {
        return Err(
            "Codex model_providers must be a table. No services or profiles were changed.".into(),
        );
    }
    if document["model_providers"]
        .get("sy")
        .is_some_and(|item| !item.is_table_like())
    {
        return Err(
            "Codex model_providers.sy must be a table. No services or profiles were changed."
                .into(),
        );
    }
    if document["model_providers"].get("sy").is_none() {
        document["model_providers"]["sy"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    for key in ["name", "base_url", "wire_api", "requires_openai_auth"] {
        document["model_providers"]["sy"][key] = template["model_providers"]["sy"][key].clone();
    }
    Ok(document.to_string())
}

fn desktop_config(s: &Settings) -> String {
    format!(
        "server_url = {:?}\nrouting_log = {}\nconfig_file = {}\nlaunchd_label = {:?}\nrefresh_seconds = 30\n\n{}",
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
        "<dict><key>SuccessfulExit</key><false/></dict><key>LimitLoadToSessionType</key><string>Aqua</string>"
    } else {
        "<true/><key>EnvironmentVariables</key><dict><key>RUST_LOG</key><string>info</string></dict>"
    };
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\"><plist version=\"1.0\"><dict><key>Label</key><string>{label}</string><key>ProgramArguments</key><array>{args}</array><key>RunAtLoad</key><true/><key>KeepAlive</key>{keep_alive}<key>StandardOutPath</key><string>{}</string><key>StandardErrorPath</key><string>{}</string></dict></plist>",
        xml(&log.with_extension("log").to_string_lossy()),
        xml(&log.with_extension("err.log").to_string_lossy())
    )
}

fn domain() -> Result<String, String> {
    let output = Command::new("id")
        .arg("-u")
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("Could not read the current user ID.".into());
    }
    Ok(format!(
        "gui/{}",
        String::from_utf8_lossy(&output.stdout).trim()
    ))
}
fn stop_services(s: &Settings) -> Result<(), String> {
    #[cfg(windows)]
    return windows::stop(s);
    #[cfg(not(windows))]
    {
        if cfg!(target_os = "macos") {
            let domain = domain()?;
            for label in [SERVER, DESKTOP, LEGACY_DESKTOP] {
                let job = format!("{domain}/{label}");
                let status = Command::new("launchctl")
                    .args(["print", &job])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .map_err(|e| e.to_string())?;
                if !status.success() {
                    continue;
                }
                run(Command::new("launchctl").args(["bootout", &job]))?;
                let mut stopped = false;
                for _ in 0..50 {
                    if !Command::new("launchctl")
                        .args(["print", &job])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status()
                        .map_err(|e| e.to_string())?
                        .success()
                    {
                        stopped = true;
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                if !stopped {
                    return Err(format!(
                        "{job} did not stop within five seconds. Try again."
                    ));
                }
            }
            remove(&s.service_dir.join(format!("{LEGACY_DESKTOP}.plist")))?;
        } else if s
            .service_dir
            .join("switchyard.service")
            .try_exists()
            .map_err(|e| e.to_string())?
        {
            run(Command::new("systemctl").args(["--user", "stop", "switchyard.service"]))?;
        }
        Ok(())
    }
}

fn stage_bundle(s: &Settings, metadata: &str) -> Result<tempfile::TempDir, String> {
    let parent = s
        .app()
        .parent()
        .ok_or("The app has no parent directory.")?
        .to_path_buf();
    fs::create_dir_all(&parent).map_err(|e| e.to_string())?;
    let stage = tempfile::tempdir_in(parent).map_err(|e| e.to_string())?;
    let app = stage.path().join("Switchyard.app");
    let contents = app.join("Contents");
    for binary in ["switchyard-desktop", "switchyard-server"] {
        write(
            &contents.join("MacOS").join(binary),
            &fs::read(s.release(binary)).map_err(|e| e.to_string())?,
            true,
        )?;
    }
    write(
        &contents.join("Resources/install.toml"),
        metadata.as_bytes(),
        false,
    )?;
    write(
        &contents.join("Resources/Switchyard.icns"),
        include_bytes!("../../switchyard-desktop/icons/icon.icns"),
        false,
    )?;
    let info = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><plist version=\"1.0\"><dict><key>CFBundleIdentifier</key><string>com.nvidia.switchyard</string><key>CFBundleName</key><string>Switchyard</string><key>CFBundleDisplayName</key><string>Switchyard</string><key>CFBundleExecutable</key><string>switchyard-desktop</string><key>CFBundleIconFile</key><string>Switchyard.icns</string><key>CFBundlePackageType</key><string>APPL</string><key>CFBundleShortVersionString</key><string>{}</string><key>LSMinimumSystemVersion</key><string>12.0</string><key>NSHighResolutionCapable</key><true/></dict></plist>",
        env!("CARGO_PKG_VERSION")
    );
    write(&contents.join("Info.plist"), info.as_bytes(), false)?;
    run(Command::new("codesign")
        .args(["--force", "--deep", "--sign", "-"])
        .arg(&app))?;
    Ok(stage)
}

fn install_macos(s: &Settings, stage: tempfile::TempDir) -> Result<(), String> {
    let previous = stage.path().join("Previous.app");
    let existed = s.app().try_exists().map_err(|e| e.to_string())?;
    if existed {
        fs::rename(s.app(), &previous).map_err(|e| e.to_string())?;
    }
    if let Err(error) = fs::rename(stage.path().join("Switchyard.app"), s.app()) {
        if existed {
            fs::rename(&previous, s.app()).map_err(|restore| {
                format!("Replace app: {error}. Restore previous app: {restore}")
            })?;
        }
        return Err(format!("Replace app: {error}"));
    }
    let contents = s.app().join("Contents");
    let args = vec![
        s.sy_home
            .join("bin/switchyard-server")
            .display()
            .to_string(),
        "--config".into(),
        s.sy_home.join("composite.toml").display().to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        s.port.to_string(),
        "--routing-log-file".into(),
        s.sy_home.join("routing.jsonl").display().to_string(),
    ];
    write(
        &s.service_dir.join(format!("{SERVER}.plist")),
        plist(SERVER, &args, &s.sy_home.join("logs/server"), false).as_bytes(),
        false,
    )?;
    let args = vec![
        contents
            .join("MacOS/switchyard-desktop")
            .display()
            .to_string(),
        s.desktop_settings().display().to_string(),
    ];
    write(
        &s.service_dir.join(format!("{DESKTOP}.plist")),
        plist(DESKTOP, &args, &s.sy_home.join("logs/desktop"), true).as_bytes(),
        false,
    )?;
    let domain = domain()?;
    for label in [SERVER, DESKTOP] {
        run(Command::new("launchctl")
            .arg("bootstrap")
            .arg(&domain)
            .arg(s.service_dir.join(format!("{label}.plist"))))?;
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
fn install_linux(s: &Settings) -> Result<(), String> {
    let unit = format!(
        "[Unit]\nDescription=Switchyard local server\n\n[Service]\nType=simple\nExecStart={} --config {} --host 127.0.0.1 --port {} --routing-log-file {}\nRestart=on-failure\nEnvironment=RUST_LOG=info\n\n[Install]\nWantedBy=default.target\n",
        systemd_arg(&s.sy_home.join("bin/switchyard-server")),
        systemd_arg(&s.sy_home.join("composite.toml")),
        s.port,
        systemd_arg(&s.sy_home.join("routing.jsonl"))
    );
    write(
        &s.service_dir.join("switchyard.service"),
        unit.as_bytes(),
        false,
    )?;
    run(Command::new("systemctl").args(["--user", "daemon-reload"]))?;
    run(Command::new("systemctl").args(["--user", "enable", "switchyard.service"]))?;
    run(Command::new("systemctl").args(["--user", "restart", "switchyard.service"]))?;
    thread::sleep(Duration::from_secs(2));
    run(Command::new("systemctl").args(["--user", "is-active", "--quiet", "switchyard.service"]))
        .map_err(|e| format!("{e}. See journalctl --user -u switchyard."))
}

fn strip_block(path: &Path, start: &str, end: &str) -> Result<(), String> {
    if !path.try_exists().map_err(|e| e.to_string())? {
        return Ok(());
    }
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut output = String::new();
    let mut skipping = false;
    for line in text.split_inclusive('\n') {
        if line.contains(start) {
            skipping = true;
        }
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
// The persistent file keeps concurrent commands locking the same inode; exit releases the lock.
fn lock_installation(s: &Settings) -> Result<fs::File, String> {
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(s.home.join(".switchyard-desktop-install.lock"))
        .map_err(|e| e.to_string())?;
    lock.try_lock()
        .map_err(|e| format!("Could not lock the desktop installation: {e}"))?;
    Ok(lock)
}

fn uninstall(s: &Settings) -> Result<(), String> {
    let _lock = lock_installation(s)?;
    println!("{}", installation_plan(s, "uninstall")?);
    if s.profile_file().exists() {
        backup(&s.profile_file(), "switchyard-uninstall-recovery")?;
    }
    stop_services(s)?;
    if cfg!(target_os = "macos") {
        for label in [SERVER, DESKTOP, LEGACY_DESKTOP] {
            remove(&s.service_dir.join(format!("{label}.plist")))?;
        }
        if s.app().try_exists().map_err(|e| e.to_string())? {
            fs::remove_dir_all(s.app()).map_err(|e| e.to_string())?;
        }
    } else if cfg!(windows) {
        #[cfg(windows)]
        windows::uninstall(s)?;
    } else {
        let unit = s.service_dir.join("switchyard.service");
        if unit.try_exists().map_err(|e| e.to_string())? {
            run(Command::new("systemctl").args(["--user", "disable", "switchyard.service"]))?;
            remove(&unit)?;
        }
        run(Command::new("systemctl").args(["--user", "daemon-reload"]))?;
    }
    remove(&s.profile_file())?;
    remove(&s.codex_home.join("config.sy.toml"))?;
    let config = s.codex_home.join("config.toml");
    strip_block(
        &config,
        "# >>> switchyard sy profile >>>",
        "# <<< switchyard sy profile <<<",
    )?;
    let direct = s.codex_home.join("config.toml.direct");
    if direct.try_exists().map_err(|e| e.to_string())? {
        if config.try_exists().map_err(|e| e.to_string())? {
            backup(&config, "switchyard-current")?;
        }
        write(
            &config,
            &fs::read(&direct).map_err(|e| e.to_string())?,
            false,
        )?;
        remove(&direct)?;
    }
    for name in [".zshrc", ".bashrc"] {
        strip_block(
            &s.home.join(name),
            "# >>> switchyard codex alias >>>",
            "# <<< switchyard codex alias <<<",
        )?;
    }
    println!(
        "Removed services, app, and profile. Kept settings, accounts, binaries, and history at {}.",
        s.sy_home.display()
    );
    let recorded = s.home.join(".switchyard-desktop-install.toml");
    if recorded.exists() && Settings::load(&recorded)?.sy_home == s.sy_home {
        remove(&recorded)?;
    }
    Ok(())
}
