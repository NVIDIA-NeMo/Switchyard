// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(windows)]

use std::{fs, process::Command};
use switchyard_desktop::install::{Settings, windows};

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
                .join(switchyard_desktop::install::executable_name(name)),
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
