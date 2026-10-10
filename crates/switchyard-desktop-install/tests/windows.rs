// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(windows)]

use std::{
    fs,
    io::{Read, Write},
    net::TcpStream,
    path::Path,
    process::Command,
    time::{Duration, Instant},
};
use switchyard_desktop_install::{Settings, execute, update_status, windows};

fn wait(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(180);
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "Windows operation did not finish within three minutes."
        );
        std::thread::sleep(Duration::from_millis(250));
    }
}
fn health(port: u16) -> bool {
    let Ok(mut socket) = TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().expect("address"),
        Duration::from_secs(1),
    ) else {
        return false;
    };
    socket
        .set_read_timeout(Some(Duration::from_secs(1)))
        .expect("timeout");
    if socket
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut response = String::new();
    socket.read_to_string(&mut response).is_ok() && response.starts_with("HTTP/1.1 200")
}
fn task_xml(s: &Settings, desktop: bool) -> String {
    let output = Command::new("schtasks.exe")
        .args(["/Query", "/TN", &windows::task_name(s, desktop), "/XML"])
        .output()
        .expect("query task");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // schtasks emits UTF-16 XML when stdout is redirected.
    if output.stdout.starts_with(&[0xff, 0xfe]) || output.stdout.contains(&0) {
        let bytes = output
            .stdout
            .strip_prefix(&[0xff, 0xfe])
            .unwrap_or(&output.stdout);
        String::from_utf16(
            &bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect::<Vec<_>>(),
        )
        .expect("task XML")
    } else {
        String::from_utf8(output.stdout).expect("task XML")
    }
}
fn desktop_running(binary: &Path) -> bool {
    // Windows may report an 8.3 path; canonical paths still identify the exact installed executable.
    let Ok(binary) = fs::canonicalize(binary) else {
        return false;
    };
    // PowerShell is used only to observe the test process; installation and launch stay in Rust.
    let output=Command::new("powershell.exe").args(["-NoProfile","-Command", "[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); Get-Process switchyard-desktop -ErrorAction SilentlyContinue | ForEach-Object { $_.Path }"]).output().expect("query desktop");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .any(|p| fs::canonicalize(p.trim()).is_ok_and(|actual| actual == binary))
}
// The helper starts the real updater from the desktop task, retaining its principal and logon settings.
fn start_scheduled_update(s: &Settings, metadata: &Path, fixture: &Path) {
    let source = fixture.join("update-context.rs");
    let helper = fixture.join("update-context.exe");
    let started = fixture.join("update-context-started");
    fs::write(&source, include_str!("windows-update-fixture.txt")).expect("helper source");
    let output = Command::new(s.cargo.parent().expect("toolchain").join("rustc.exe"))
        .args(["--edition=2024", "-Cpanic=abort", "--extern"])
        .arg(format!(
            "switchyard_desktop_install={}",
            s.source
                .join("target/release/libswitchyard_desktop_install.rlib")
                .display()
        ))
        .arg("-L")
        .arg(format!(
            "dependency={}",
            s.source.join("target/release/deps").display()
        ))
        .arg(&source)
        .arg("-o")
        .arg(&helper)
        .output()
        .expect("compile update helper");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut xml = tempfile::NamedTempFile::new().expect("task XML");
    xml.write_all(&[0xff, 0xfe]).expect("XML BOM");
    for word in task_xml(s, true).encode_utf16() {
        xml.write_all(&word.to_le_bytes()).expect("XML");
    }
    let xml = xml.into_temp_path();
    let arguments = format!(
        "{} {}",
        windows::quote_arg(metadata.to_str().expect("metadata path")),
        windows::quote_arg(started.to_str().expect("marker path"))
    );
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-Command", "$ErrorActionPreference='Stop'; $xml=[xml](Get-Content -LiteralPath $env:SWITCHYARD_CONTEXT_XML -Raw); $xml.Task.Actions.Exec.Command=$env:SWITCHYARD_CONTEXT_HELPER; $xml.Task.Actions.Exec.Arguments=$env:SWITCHYARD_CONTEXT_ARGUMENTS; $xml.Save($env:SWITCHYARD_CONTEXT_XML)"])
        .env("SWITCHYARD_CONTEXT_XML", &xml)
        .env("SWITCHYARD_CONTEXT_HELPER", &helper)
        .env("SWITCHYARD_CONTEXT_ARGUMENTS", arguments)
        .output()
        .expect("set task action");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let task = windows::task_name(s, true);
    for args in [
        vec!["/End", "/TN", &task],
        vec![
            "/Create",
            "/F",
            "/TN",
            &task,
            "/XML",
            xml.to_str().expect("XML path"),
        ],
        vec!["/Run", "/TN", &task],
    ] {
        let output = Command::new("schtasks.exe")
            .args(&args)
            .output()
            .expect("scheduled update");
        assert!(
            output.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    // This marker prevents a previous completed update from satisfying the next status check.
    wait(|| started.is_file());
}
// Cleanup preserves logs and removes test tasks and executables even after an assertion fails.
struct Cleanup<'a>(&'a Settings);
impl Drop for Cleanup<'_> {
    fn drop(&mut self) {
        if let Some(destination) = std::env::var_os("SWITCHYARD_WINDOWS_LOG_DIR") {
            let destination = std::path::PathBuf::from(destination);
            let _ = fs::create_dir_all(&destination);
            for name in ["update.log", "desktop.log", "server.log"] {
                let _ = fs::copy(
                    self.0.sy_home.join("logs").join(name),
                    destination.join(name),
                );
            }
        }
        let _ = windows::stop(self.0);
        let _ = windows::uninstall(self.0);
    }
}

#[test]
#[ignore = "Registers real per-user tasks and builds the release app; run explicitly on a Windows test machine."]
fn fresh_install_running_update_restart_and_uninstall_preserve_user_data() {
    let root = tempfile::Builder::new()
        .prefix("Switchyard équipe space ")
        .tempdir()
        .expect("fixture");
    let home = root.path().join("User équipe space");
    fs::create_dir_all(&home).expect("home");
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .expect("port")
        .local_addr()
        .expect("address")
        .port();
    let defaults = Settings::from_env().expect("defaults");
    let s = Settings {
        home: home.clone(),
        sy_home: home.join(".switchyard"),
        codex_home: home.join(".codex"),
        service_dir: home.join("AppData/Roaming/Microsoft/Windows/Start Menu/Programs"),
        app_dir: Some(home.join("AppData/Local/Programs/Switchyard")),
        port,
        ..defaults
    };
    let _cleanup = Cleanup(&s);
    execute(&s, "install", true).expect("preview");
    assert!(!s.sy_home.exists());
    execute(&s, "install", false).expect("fresh install");
    assert!(health(port), "server failed readiness");
    wait(|| desktop_running(&s.binary("switchyard-desktop")));
    for desktop in [false, true] {
        let xml = task_xml(&s, desktop);
        // The registered task's COM principal reports its effective privilege and logon settings.
        let output = Command::new("powershell.exe")
            .args(["-NoProfile", "-Command", r"$ErrorActionPreference='Stop'; $scheduler=New-Object -ComObject Schedule.Service; $scheduler.Connect(); $principal=$scheduler.GetFolder('\').GetTask($env:SWITCHYARD_TEST_TASK).Definition.Principal; @{run_level=[int]$principal.RunLevel; logon_type=[int]$principal.LogonType} | ConvertTo-Json -Compress"])
            .env("SWITCHYARD_TEST_TASK", windows::task_name(&s, desktop))
            .output()
            .expect("task principal");
        assert!(
            output.status.success(),
            "{}\n{xml}",
            String::from_utf8_lossy(&output.stderr)
        );
        let principal: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("principal JSON");
        // Task Scheduler defines LUA privileges as 0 and an interactive-token logon as 3.
        assert_eq!(principal["run_level"], 0, "{principal}\n{xml}");
        assert_eq!(principal["logon_type"], 3, "{principal}\n{xml}");
        assert!(
            xml.contains("LogonTrigger") && xml.contains("PT0S"),
            "{xml}"
        );
    }
    assert!(s.service_dir.join("Switchyard.lnk").is_file());
    let settings = s.desktop_settings();
    let profile = s.codex_home.join("sy.config.toml");
    let original_settings = fs::read(&settings).expect("settings");
    let original_profile = fs::read(&profile).expect("profile");
    let history = s.sy_home.join("routing.jsonl");
    fs::write(&history, b"\n").expect("history");
    let accounts = s.home.join(".switchyard/accounts/codex/team");
    fs::create_dir_all(&accounts).expect("account");
    fs::write(accounts.join("auth.json"), b"preserved-login-fixture").expect("login fixture");
    let metadata = s.sy_home.join("install.toml");
    start_scheduled_update(&s, &metadata, root.path());
    wait(|| {
        let status = update_status(&metadata).expect("update status");
        assert_ne!(
            status["state"],
            "failed",
            "{status}\n{}",
            fs::read_to_string(s.sy_home.join("logs/update.log")).unwrap_or_default()
        );
        status["state"] == "completed"
    });
    assert!(health(port));
    wait(|| desktop_running(&s.binary("switchyard-desktop")));
    assert_eq!(
        fs::read(&settings).expect("retained settings"),
        original_settings
    );
    assert_eq!(
        fs::read(&profile).expect("retained profile"),
        original_profile
    );
    assert_eq!(fs::read(&history).expect("history"), b"\n");
    windows::restart(&s).expect("restart");
    wait(|| health(port));
    // The Cargo alias must rebuild and reinstall from its checkout while the release app is running.
    let status = Command::new(&s.cargo)
        .current_dir(&s.source)
        .args(["desktop", "install"])
        .env("HOME", &s.home)
        .env("USERPROFILE", &s.home)
        .env("SY_HOME", &s.sy_home)
        .env("CODEX_HOME", &s.codex_home)
        .env("SY_PORT", s.port.to_string())
        .env("APPDATA", s.home.join("AppData/Roaming"))
        .env("LOCALAPPDATA", s.home.join("AppData/Local"))
        .status()
        .expect("cargo desktop install");
    assert!(status.success());
    wait(|| {
        let status = update_status(&metadata).expect("reinstall status");
        assert_ne!(status["state"], "failed", "{status}");
        status["state"] == "completed"
    });
    wait(|| health(port));
    execute(&s, "uninstall", false).expect("uninstall");
    assert!(!health(port));
    assert!(!desktop_running(&s.binary("switchyard-desktop")));
    assert!(!s.bin_dir().exists());
    for desktop in [false, true] {
        assert!(
            !Command::new("schtasks.exe")
                .args(["/Query", "/TN", &windows::task_name(&s, desktop)])
                .output()
                .expect("removed task query")
                .status
                .success()
        );
    }
    assert!(!s.service_dir.join("Switchyard.lnk").exists());
    assert_eq!(fs::read(&settings).expect("settings"), original_settings);
    assert_eq!(fs::read(&history).expect("history"), b"\n");
    assert_eq!(
        fs::read(accounts.join("auth.json")).expect("account"),
        b"preserved-login-fixture"
    );
    assert!(fs::read_dir(&s.codex_home).expect("recovery").any(|f| {
        f.expect("file")
            .file_name()
            .to_string_lossy()
            .contains("switchyard-uninstall-recovery")
    }));
    execute(&s, "uninstall", false).expect("repeat uninstall");
}

#[test]
fn executable_replacement_rolls_back_when_another_process_locks_a_file() {
    use std::os::windows::fs::OpenOptionsExt;
    let root = tempfile::tempdir().expect("fixture");
    let defaults = Settings::from_env().expect("defaults");
    let s = Settings {
        home: root.path().join("home"),
        sy_home: root.path().join("settings"),
        codex_home: root.path().join("codex"),
        service_dir: root.path().join("shortcuts"),
        source: root.path().join("source"),
        app_dir: Some(root.path().join("home/AppData/Local/Programs/Switchyard")),
        ..defaults
    };
    fs::create_dir_all(s.source.join("target/release")).expect("release fixtures");
    fs::create_dir_all(s.bin_dir()).expect("binaries");
    for name in [
        "switchyard-server",
        "switchyard-desktop",
        "switchyard-desktop-install",
    ] {
        fs::write(
            s.source
                .join("target/release")
                .join(switchyard_desktop_install::executable_name(name)),
            b"new",
        )
        .expect("new binary");
        fs::write(s.binary(name), b"old").expect("old binary");
    }
    // Locking the second executable requires rollback of the first replacement before a retry.
    let lock = fs::OpenOptions::new()
        .read(true)
        .share_mode(0)
        .open(s.binary("switchyard-desktop"))
        .expect("exclusive lock");
    assert!(windows::replace_binaries(&s).is_err());
    drop(lock);
    for name in [
        "switchyard-server",
        "switchyard-desktop",
        "switchyard-desktop-install",
    ] {
        assert_eq!(
            fs::read(s.binary(name)).expect("previous executable"),
            b"old"
        );
    }
    windows::replace_binaries(&s).expect("retry after releasing lock");
    for name in [
        "switchyard-server",
        "switchyard-desktop",
        "switchyard-desktop-install",
    ] {
        assert_eq!(fs::read(s.binary(name)).expect("new executable"), b"new");
    }
}

#[test]
fn credential_manager_stores_only_the_exact_endpoint_and_survives_a_read() {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Security::Credentials::{CRED_TYPE_GENERIC, CredDeleteW};
    struct Credential(String);
    impl Drop for Credential {
        fn drop(&mut self) {
            let name: Vec<u16> = std::ffi::OsStr::new(&format!("Switchyard:model-list:{}", self.0))
                .encode_wide()
                .chain(Some(0))
                .collect();
            unsafe {
                CredDeleteW(name.as_ptr(), CRED_TYPE_GENERIC, 0);
            }
        }
    }
    let endpoint = Credential(format!(
        "http://localhost/windows-key-fixture-{}-{}",
        std::process::id(),
        fixture_timestamp()
    ));
    assert_eq!(
        windows::saved_key(&endpoint.0).expect("missing credential"),
        None
    );
    windows::save_key(&endpoint.0, "test-key-only").expect("save credential");
    assert_eq!(
        windows::saved_key(&endpoint.0).expect("read credential"),
        Some("test-key-only".into())
    );
    assert_eq!(
        windows::saved_key(&format!("{}/other", endpoint.0)).expect("different endpoint"),
        None
    );
}
fn fixture_timestamp() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("timestamp")
        .as_nanos()
}

#[test]
fn invalid_windows_install_locations_reject_before_creating_files() {
    use std::os::windows::ffi::OsStringExt;
    let root = tempfile::tempdir().expect("fixture");
    let home = root.path().join("home");
    fs::create_dir(&home).expect("home");
    for location in [
        std::ffi::OsString::from("relative"),
        std::ffi::OsString::from("C:\\invalid\npath"),
        std::ffi::OsString::from_wide(&[b'C' as u16, b':' as u16, b'\\' as u16, 0xd800]),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_switchyard-desktop-install"))
            .args(["install", "--dry-run"])
            .env("HOME", &home)
            .env("USERPROFILE", &home)
            .env("LOCALAPPDATA", location)
            .output()
            .expect("invalid location");
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("absolute UTF-8 path"));
        assert_eq!(fs::read_dir(&home).expect("unchanged home").count(), 0);
    }
}

#[test]
fn malformed_private_handoff_records_are_retained_without_dispatch() {
    let root = tempfile::tempdir().expect("fixture");
    let defaults = Settings::from_env().expect("defaults");
    let s = Settings {
        home: root.path().join("home"),
        sy_home: root.path().join("settings"),
        codex_home: root.path().join("codex"),
        service_dir: root.path().join("shortcuts"),
        app_dir: Some(root.path().join("app")),
        ..defaults
    };
    fs::create_dir(&s.home).expect("home");
    let plain = toml::to_string(&s).expect("settings");
    let wrapped = format!("[settings]\n{plain}");
    for (text, pid, reason) in [
        (plain, "1", "unknown"),
        (format!("unknown=true\n{wrapped}"), "1", "unknown"),
        (wrapped.clone(), "0", "parent process"),
    ] {
        let record = tempfile::Builder::new()
            .prefix("switchyard-handoff-")
            .suffix(".toml")
            .tempfile()
            .expect("private record");
        fs::write(record.path(), &text).expect("record");
        let output = Command::new(env!("CARGO_BIN_EXE_switchyard-desktop-install"))
            .arg("apply")
            .arg("--handoff-record")
            .arg(record.path())
            .args(["--action", "uninstall", "--wait", pid])
            .output()
            .expect("rejected handoff");
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .to_lowercase()
                .contains(reason),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            fs::read_to_string(record.path()).expect("record retained"),
            text
        );
        assert_eq!(fs::read_dir(&s.home).expect("home unchanged").count(), 0);
    }
    let outside = root.path().join("switchyard-handoff-outside.toml");
    fs::write(&outside, &wrapped).expect("outside record");
    let output = Command::new(env!("CARGO_BIN_EXE_switchyard-desktop-install"))
        .arg("apply")
        .arg("--handoff-record")
        .arg(&outside)
        .args(["--action", "uninstall", "--wait", "1"])
        .output()
        .expect("outside record");
    assert!(!output.status.success());
    assert_eq!(
        fs::read_to_string(outside).expect("outside record retained"),
        wrapped
    );
}
