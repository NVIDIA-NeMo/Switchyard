// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// An unreadable log must fail instead of reporting that no requests were recorded.
#[test]
fn unreadable_usage_fails_without_an_empty_summary() {
    let home = tempfile::tempdir().expect("fixture");
    let settings = home.path().join("desktop.toml");
    std::fs::write(&settings, format!("routing_log={:?}\n", home.path())).expect("settings");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_switchyard-desktop"))
        .arg("--print")
        .arg(&settings)
        .env("HOME", home.path())
        .output()
        .expect("print");
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Usage unavailable"));
    assert_eq!(std::fs::read_dir(home.path()).expect("files").count(), 1);
}
