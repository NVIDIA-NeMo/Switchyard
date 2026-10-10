// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use switchyard_desktop_install::Settings;

// This child process isolates HOME and PATH so service fixtures cannot reach real jobs.
#[test]
fn fixture_process() {
    if let Some(file) = std::env::var_os("INSTALL_FIXTURE_SETTINGS") {
        let settings = Settings::load(Path::new(&file)).expect("fixture settings");
        let action = std::env::var("INSTALL_FIXTURE_ACTION").expect("fixture action");
        if action == "start_update" {
            switchyard_desktop_install::start_update(Path::new(&file)).expect("start updater");
            assert!(switchyard_desktop_install::start_update(Path::new(&file)).is_err());
            for _ in 0..1500 {
                let status =
                    switchyard_desktop_install::update_status(Path::new(&file)).expect("status");
                if status["state"] == "completed" {
                    return;
                }
                assert_ne!(status["state"], "failed", "{status}");
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            panic!(
                "updater did not complete: {:?}\n{:?}",
                switchyard_desktop_install::update_status(Path::new(&file)),
                fs::read_to_string(settings.sy_home.join("logs/update.log")),
            );
        }
        if let Err(error) = switchyard_desktop_install::execute(&settings, &action, false) {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

// Native command fixtures replace build and service commands without invoking a shell.
struct Fixture {
    _temp: tempfile::TempDir,
    settings: Settings,
    bin: PathBuf,
    metadata: PathBuf,
    health: HealthFixture,
}
// The local health server wakes its listener during cleanup so failed assertions cannot hang tests.
struct HealthFixture {
    port: u16,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl HealthFixture {
    fn new() -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("health socket");
        let port = listener.local_addr().expect("address").port();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = stop.clone();
        let thread = std::thread::spawn(move || {
            use std::io::{Read, Write};
            for connection in listener.incoming() {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                if let Ok(mut stream) = connection {
                    stream
                        .set_read_timeout(Some(std::time::Duration::from_secs(1)))
                        .expect("timeout");
                    let mut request = [0; 1024];
                    let _ = stream.read(&mut request);
                    let body = "{\"status\":\"ok\"}";
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                }
            }
        });
        Self {
            port,
            stop,
            thread: Some(thread),
        }
    }
}
impl Drop for HealthFixture {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = std::net::TcpStream::connect(("127.0.0.1", self.port));
        if let Some(thread) = self.thread.take() {
            thread.join().expect("health cleanup");
        }
    }
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().expect("fixture directory");
        let home = temp.path().join("home");
        let bin = temp.path().join("bin");
        fs::create_dir_all(&home).expect("home");
        fs::create_dir_all(&bin).expect("bin");
        // These characters must remain literal in TOML, plist, and systemd arguments.
        let source = temp.path().join("checkout with ' $() & <quotes>");
        fs::create_dir_all(&source).expect("checkout");
        let fixture_src = temp.path().join("fixture.rs");
        fs::write(&fixture_src, include_str!("fixture.txt")).expect("fixture source");
        let status = Command::new("rustc")
            .arg(&fixture_src)
            .arg("-o")
            .arg(bin.join("cargo"))
            .status()
            .expect("compile fixture");
        assert!(status.success());
        for name in ["id", "launchctl", "codesign", "systemctl"] {
            fs::copy(bin.join("cargo"), bin.join(name)).expect("copy native command fixture");
        }
        let health = HealthFixture::new();
        let s = Settings {
            app_dir: None,
            sy_home: home.join("Switchyard ' $(touch unsafe) \\ & <local> % $"),
            codex_home: home.join(".codex"),
            service_dir: if cfg!(target_os = "macos") {
                home.join("Library/LaunchAgents")
            } else {
                home.join(".config/systemd/user")
            },
            home,
            source,
            cargo: bin.join("cargo"),
            port: health.port,
            profile: "team-dev".into(),
            model: "composite-gpt-6-sol-gpt-6-luna".into(),
        };
        let metadata = temp.path().join("install.toml");
        let f = Self {
            _temp: temp,
            settings: s,
            bin,
            metadata,
            health,
        };
        f.save();
        f
    }
    fn save(&self) {
        fs::write(
            &self.metadata,
            toml::to_string(&self.settings).expect("serialize settings"),
        )
        .expect("write metadata");
    }
    fn run(&self, action: &str, fail_validation: bool) -> std::process::Output {
        let path = std::env::join_paths(std::iter::once(self.bin.clone()).chain(
            std::env::split_paths(&std::env::var_os("PATH").expect("PATH")),
        ))
        .expect("fixture PATH");
        Command::new(std::env::current_exe().expect("test binary"))
            .args(["--exact", "fixture_process", "--nocapture"])
            .env("INSTALL_FIXTURE_SETTINGS", &self.metadata)
            .env("INSTALL_FIXTURE_ACTION", action)
            .env("PATH", path)
            .env("HOME", &self.settings.home)
            .env(
                "INSTALLER_BIN",
                env!("CARGO_BIN_EXE_switchyard-desktop-install"),
            )
            .env(
                "FIXTURE_FAIL_VALIDATE",
                if fail_validation { "1" } else { "0" },
            )
            .output()
            .expect("run fixture")
    }
}

#[test]
fn reinstall_migrates_settings_preserves_data_and_generates_native_launchers() {
    let mut f = Fixture::new();
    let s = &f.settings;
    fs::create_dir_all(&s.sy_home).expect("existing settings");
    fs::create_dir_all(&s.codex_home).expect("existing profile");
    fs::write(
        s.sy_home.join("menubar.toml"),
        "baseline_model = 'custom'\n",
    )
    .expect("legacy settings");
    fs::write(s.sy_home.join("routing.jsonl"), "history").expect("history");
    fs::write(s.codex_home.join("config.toml"), "user defaults").expect("user defaults");
    let output = f.run("install", false);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(s.desktop_settings()).expect("settings"),
        format!(
            "baseline_model = 'custom'\nserver_url = \"http://127.0.0.1:{}\"\n",
            s.port
        )
    );
    assert!(!s.sy_home.join("menubar.toml").exists());
    let profile = s.codex_home.join("team-dev.config.toml");
    let original = fs::read_to_string(&profile).expect("profile");
    let parsed: toml::Value = toml::from_str(&original).expect("profile TOML");
    assert_eq!(parsed["model"].as_str(), Some(s.model.as_str()));
    assert!(parsed.get("approval_policy").is_none());
    assert!(parsed.get("sandbox_mode").is_none());
    fs::write(
        s.sy_home.join("composite.toml"),
        include_str!("../../../scripts/config/composite.toml"),
    )
    .expect("server config");
    f.health = HealthFixture::new();
    f.settings.port = f.health.port;
    f.save();
    let output = f.run("install", false);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let s = &f.settings;
    assert_eq!(
        fs::read_to_string(s.sy_home.join("routing.jsonl")).expect("history"),
        "history"
    );
    assert_eq!(
        fs::read_to_string(s.codex_home.join("config.toml")).expect("user defaults"),
        "user defaults"
    );
    assert_eq!(
        fs::read_to_string(s.sy_home.join("composite.toml")).expect("config"),
        include_str!("../../../scripts/config/composite.toml")
    );
    let backups: Vec<_> = fs::read_dir(&s.codex_home)
        .expect("profiles")
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .contains("switchyard-backup")
        })
        .collect();
    assert_eq!(backups.len(), 1);
    assert_eq!(
        fs::read_to_string(backups[0].path()).expect("backup"),
        original
    );
    let recorded = Settings::load(&s.sy_home.join("install.toml")).expect("saved installation");
    assert_eq!(recorded.port, f.health.port);
    assert_eq!(recorded.model, s.model);
    assert_eq!(recorded.source, s.source);
    if cfg!(target_os = "macos") {
        let contents = s.app().join("Contents");
        let info = plist::Value::from_file(contents.join("Info.plist")).expect("Info.plist");
        assert_eq!(
            info.as_dictionary().expect("dictionary")["CFBundleExecutable"].as_string(),
            Some("switchyard-desktop")
        );
        let icon = info.as_dictionary().expect("dictionary")["CFBundleIconFile"]
            .as_string()
            .expect("bundle icon name");
        let bytes = fs::read(contents.join("Resources").join(icon)).expect("bundle icon");
        assert_eq!(
            bytes,
            include_bytes!("../../switchyard-desktop/icons/icon.icns")
        );
        assert_eq!(&bytes[..4], b"icns");
        assert_eq!(
            u32::from_be_bytes(bytes[4..8].try_into().expect("icon size")) as usize,
            bytes.len()
        );
        assert!(contents.join("MacOS/switchyard-desktop").is_file());
        assert!(!contents.join("MacOS/Switchyard").exists());
        assert!(!contents.join("Resources/Update.command").exists());
        let metadata =
            switchyard_desktop_install::bundle_metadata(&contents.join("MacOS/switchyard-desktop"));
        assert_eq!(
            Settings::load(&metadata).expect("bundle settings").sy_home,
            s.sy_home
        );
        let job = plist::Value::from_file(s.service_dir.join("com.nvidia.switchyard.server.plist"))
            .expect("server plist");
        let args = job.as_dictionary().expect("job")["ProgramArguments"]
            .as_array()
            .expect("args");
        assert_eq!(
            args[0].as_string(),
            Some(
                s.sy_home
                    .join("bin/switchyard-server")
                    .to_str()
                    .expect("UTF-8")
            )
        );
        assert!(
            !s.service_dir
                .join("com.nvidia.switchyard.menubar.plist")
                .exists()
        );
    } else {
        let unit = fs::read_to_string(s.service_dir.join("switchyard.service")).expect("unit");
        assert!(unit.contains("%% $$"));
    }
    assert!(!s.home.join("unsafe").exists());
}

// A failed server check must prevent the installer from creating app, service, or settings files.
#[test]
fn validator_failure_leaves_installation_and_profile_untouched() {
    let f = Fixture::new();
    let output = f.run("install", true);
    assert!(!output.status.success());
    assert!(!f.settings.sy_home.exists());
    assert!(!f.settings.codex_home.exists());
    assert!(!f.settings.app().exists());
    assert!(!f.settings.service_dir.exists());
}

#[test]
fn uninstall_restores_snapshot_with_a_recovery_copy_and_keeps_history() {
    let f = Fixture::new();
    assert!(f.run("install", false).status.success());
    let s = &f.settings;
    fs::write(s.codex_home.join("config.toml"), "current settings").expect("current");
    fs::write(s.codex_home.join("config.toml.direct"), "original settings").expect("direct");
    fs::write(s.sy_home.join("routing.jsonl"), "history").expect("history");
    fs::write(s.home.join(".bashrc"), "before\n# >>> switchyard codex alias >>>\nalias codex=old\n# <<< switchyard codex alias <<<\nafter\n").expect("legacy rc");
    let output = f.run("uninstall", false);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(s.codex_home.join("config.toml")).expect("restored"),
        "original settings"
    );
    let backups: Vec<_> = fs::read_dir(&s.codex_home)
        .expect("configs")
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .contains("switchyard-current")
        })
        .collect();
    assert_eq!(backups.len(), 1);
    assert_eq!(
        fs::read_to_string(backups[0].path()).expect("recovery"),
        "current settings"
    );
    assert_eq!(
        fs::read_to_string(s.sy_home.join("routing.jsonl")).expect("history"),
        "history"
    );
    assert_eq!(
        fs::read_to_string(s.home.join(".bashrc")).expect("rc"),
        "before\nafter\n"
    );
    assert!(!s.codex_home.join("team-dev.config.toml").exists());
    assert!(!s.app().exists());
    assert!(f.run("uninstall", false).status.success());
    let unfinished = "before\n# >>> switchyard codex alias >>>\nalias codex=old\nafter\n";
    fs::write(s.home.join(".bashrc"), unfinished).expect("unfinished block");
    assert!(!f.run("uninstall", false).status.success());
    assert_eq!(
        fs::read_to_string(s.home.join(".bashrc")).expect("unchanged rc"),
        unfinished
    );
}

#[test]
fn invalid_inputs_and_dry_run_have_no_installation_side_effects() {
    let f = Fixture::new();
    // Fixture commands come first on PATH so a validation regression cannot stop real services.
    let path = std::env::join_paths(std::iter::once(f.bin.clone()).chain(std::env::split_paths(
        &std::env::var_os("PATH").expect("PATH"),
    )))
    .expect("fixture PATH");
    let binary = env!("CARGO_BIN_EXE_switchyard-desktop-install");
    for (key, value) in [
        ("SY_PROFILE", "../escape"),
        ("SY_PROFILE", "team.dev"),
        ("SY_PROFILE", ""),
        ("SY_MODEL", ""),
        ("SY_MODEL", "model\nname"),
        ("SY_MODEL", "model\x7fname"),
        ("SY_PORT", "0"),
        ("SY_PORT", "65536"),
        ("SY_PORT", "+4123"),
        ("SY_PORT", "4123\n"),
    ] {
        let output = Command::new(binary)
            .arg("install")
            .env("PATH", &path)
            .env("HOME", &f.settings.home)
            .env(key, value)
            .output()
            .expect("reject input");
        assert!(!output.status.success(), "{key}={value:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains(key));
    }
    for key in ["SY_PROFILE", "SY_MODEL", "SY_PORT"] {
        use std::os::unix::ffi::OsStringExt;
        let output = Command::new(binary)
            .arg("install")
            .env("PATH", &path)
            .env("HOME", &f.settings.home)
            .env(key, std::ffi::OsString::from_vec(vec![0xff]))
            .output()
            .expect("reject raw input");
        assert!(!output.status.success(), "{key}");
        assert!(String::from_utf8_lossy(&output.stderr).contains(key));
    }
    for args in [
        vec!["install", "--dryrun"],
        vec!["uninstall", "extra"],
        vec!["install", "--dry-run", "extra"],
    ] {
        assert!(
            !Command::new(binary)
                .args(args)
                .env("PATH", &path)
                .env("HOME", &f.settings.home)
                .output()
                .expect("reject args")
                .status
                .success()
        );
    }
    for action in ["install", "uninstall"] {
        assert!(
            Command::new(binary)
                .args([action, "--dry-run"])
                .env("PATH", &path)
                .env("HOME", &f.settings.home)
                .output()
                .expect("preview")
                .status
                .success()
        );
    }
    assert_eq!(fs::read_dir(&f.settings.home).expect("home").count(), 0);
    let text = fs::read_to_string(&f.metadata).expect("metadata");
    let selected = f.metadata.to_str().expect("settings path");
    for args in [
        ["uninstall", "--settings", selected, "--dry-run"],
        ["uninstall", "--dry-run", "--settings", selected],
    ] {
        let output = Command::new(binary)
            .args(args)
            .env("PATH", &path)
            .env("HOME", &f.settings.home)
            .output()
            .expect("preview selected removal");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("team-dev"));
        assert_eq!(
            fs::read_to_string(&f.metadata).expect("unchanged metadata"),
            text
        );
        assert_eq!(
            fs::read_dir(&f.settings.home)
                .expect("unchanged home")
                .count(),
            0
        );
    }
    fs::write(&f.metadata, format!("{text}\nunknown_field = true\n")).expect("invalid metadata");
    assert!(Settings::load(&f.metadata).is_err());
}

#[test]
fn updater_has_a_separate_process_group() {
    let f = Fixture::new();
    fs::create_dir_all(f.settings.sy_home.join("bin")).expect("installed binaries");
    fs::create_dir_all(f.settings.sy_home.join("logs")).expect("installed logs");
    fs::copy(
        f.bin.join("cargo"),
        f.settings.sy_home.join("bin/switchyard-desktop-install"),
    )
    .expect("updater fixture");
    let parent = Command::new("ps")
        .args(["-o", "pgid=", "-p", &std::process::id().to_string()])
        .output()
        .expect("parent process group");
    assert!(parent.status.success());
    let log = switchyard_desktop_install::start_update(&f.metadata).expect("start updater");
    assert!(switchyard_desktop_install::start_update(&f.metadata).is_err());
    let mut group = String::new();
    for _ in 0..100 {
        group = fs::read_to_string(&log).expect("update log");
        if !group.trim().is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(!group.trim().is_empty());
    assert_ne!(
        group.trim(),
        String::from_utf8(parent.stdout)
            .expect("parent group")
            .trim()
    );
}

// Install and uninstall must use the same lock before stopping services or replacing files.
#[test]
fn concurrent_install_and_uninstall_reject_before_changing_managed_files() {
    let f = Fixture::new();
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(f.settings.home.join(".switchyard-desktop-install.lock"))
        .expect("lock file");
    lock.lock().expect("hold installation lock");
    for action in ["install", "uninstall"] {
        let output = f.run(action, false);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("Could not lock"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let calls = fs::read_to_string(f.settings.home.join("service_calls")).unwrap_or_default();
        assert!(
            !calls.contains("bootout")
                && !calls.contains("bootstrap")
                && !calls.contains("stop")
                && !calls.contains("disable"),
            "{calls}"
        );
        assert!(!f.settings.sy_home.exists());
        assert!(!f.settings.codex_home.exists());
        assert!(!f.settings.app().exists());
        assert!(!f.settings.service_dir.exists());
    }
}

#[test]
fn unknown_route_rejects_before_services_or_profiles_change() {
    let mut f = Fixture::new();
    f.settings.model = "missing-public-route".into();
    f.save();
    let output = f.run("install", false);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing-public-route"));
    assert!(!f.settings.codex_home.exists());
    assert!(!f.settings.app().exists());
    assert!(!f.settings.desktop_settings().exists());
    let calls = fs::read_to_string(f.settings.home.join("service_calls")).unwrap_or_default();
    assert!(!calls.contains("bootout") && !calls.contains("bootstrap") && !calls.contains("stop"));
}
#[test]
fn source_update_preserves_profile_bytes_and_uninstall_keeps_recovery() {
    let f = Fixture::new();
    assert!(f.run("install", false).status.success());
    let profile = f.settings.codex_home.join("team-dev.config.toml");
    let current = "# Personal settings\nmodel='custom-selection'\nexperimental='retained'\n";
    fs::write(&profile, current).expect("edited profile");
    let settings = f.settings.desktop_settings();
    // Mixed line endings and single quotes expose a rewrite even when the parsed settings stay the same.
    let retained_settings = format!(
        "# Personal settings\r\nserver_url = 'http://127.0.0.1:{}'\nbaseline_model='custom'\r\n[prices.custom]\r\ninput_per_mtok=9.25\noutput_per_mtok=12.5\r\n",
        f.settings.port
    );
    fs::write(&settings, &retained_settings).expect("edited settings");
    let output = f.run("update", false);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(fs::read_to_string(&profile).expect("profile"), current);
    assert_eq!(
        fs::read(&settings).expect("settings"),
        retained_settings.as_bytes()
    );
    assert_eq!(
        switchyard_desktop_install::update_status(&f.metadata).expect("status")["state"],
        "completed"
    );
    assert!(f.run("uninstall", false).status.success());
    assert!(!profile.exists());
    let copies: Vec<_> = fs::read_dir(&f.settings.codex_home)
        .expect("profiles")
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .contains("switchyard-uninstall-recovery")
        })
        .collect();
    assert_eq!(copies.len(), 1);
    assert_eq!(
        fs::read_to_string(copies[0].path()).expect("recovery"),
        current
    );
}
#[test]
fn malformed_profile_table_rejects_before_service_changes() {
    let f = Fixture::new();
    fs::create_dir_all(&f.settings.codex_home).expect("codex");
    let profile = f.settings.codex_home.join("team-dev.config.toml");
    let original = "model_providers='bad table'\n";
    fs::write(&profile, original).expect("profile");
    let output = f.run("install", false);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("model_providers must be a table"));
    assert_eq!(
        fs::read_to_string(profile).expect("unchanged profile"),
        original
    );
    assert!(!f.settings.home.join("service_calls").exists());
}

#[test]
// The installed updater must finish after an old running record and refuse a second start.
fn updater_child_completes_through_the_installed_entrypoint() {
    let f = Fixture::new();
    fs::create_dir_all(f.settings.sy_home.join("bin")).expect("installed binaries");
    fs::create_dir_all(f.settings.sy_home.join("logs")).expect("installed logs");
    fs::copy(
        env!("CARGO_BIN_EXE_switchyard-desktop-install"),
        f.settings.sy_home.join("bin/switchyard-desktop-install"),
    )
    .expect("installed updater");
    fs::write(
        f.settings.sy_home.join("update-status.toml"),
        "state='running'\nupdated_at=0\n",
    )
    .expect("interrupted update");
    let output = f.run("start_update", false);
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        switchyard_desktop_install::update_status(&f.metadata).expect("final status")["state"],
        "completed"
    );
}

#[test]
// A non-string server_url must not be replaced with a managed URL during installation.
fn malformed_app_address_rejects_without_replacing_settings() {
    let f = Fixture::new();
    fs::create_dir_all(&f.settings.sy_home).expect("settings directory");
    let path = f.settings.desktop_settings();
    let original = "server_url=42\n# user settings\n";
    fs::write(&path, original).expect("invalid address");
    let output = f.run("install", false);
    assert!(!output.status.success());
    assert_eq!(
        fs::read_to_string(path).expect("preserved settings"),
        original
    );
    assert!(!f.settings.codex_home.join("team-dev.config.toml").exists());
    let calls = fs::read_to_string(f.settings.home.join("service_calls")).expect("preflight");
    assert!(
        !calls.contains("bootout") && !calls.contains("disable") && !calls.contains("bootstrap")
    );
}
