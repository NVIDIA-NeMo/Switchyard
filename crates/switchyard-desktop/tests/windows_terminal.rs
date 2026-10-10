// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(windows)]

#[test]
fn native_and_npm_launchers_preserve_arguments_accounts_and_working_directory() {
    use std::{
        fs,
        process::{Command, Stdio},
    };
    let root = tempfile::Builder::new()
        .prefix("Terminal équipe space ")
        .tempdir()
        .expect("fixture");
    let source = root.path().join("tool.rs");
    fs::write(&source,r#"fn main() {
        let output=std::env::var_os("FIXTURE_OUTPUT").unwrap();
        let text=format!("{:?}\n{:?}\n{:?}\n{:?}",std::env::args().skip(1).collect::<Vec<_>>(),std::env::current_dir().unwrap(),std::env::var("CODEX_HOME"),std::env::var("ANTHROPIC_AUTH_TOKEN"));
        std::fs::write(output,text).unwrap();
    }"#).expect("tool source");
    let tool = root.path().join("tool.exe");
    assert!(
        Command::new("rustc")
            .arg(&source)
            .arg("-o")
            .arg(&tool)
            .status()
            .expect("compile fixture")
            .success()
    );
    let shim = root.path().join("tool.cmd");
    fs::write(
        &shim,
        format!("@echo off\r\n@\"{}\" %*\r\n", tool.display()),
    )
    .expect("npm shim");
    let args = vec![
        "space équipe".to_string(),
        "a\"b".into(),
        "%PATH%".into(),
        "& echo injected > forbidden-file".into(),
        "C:\\space équipe\\".into(),
        "<input>|output^".into(),
    ];
    let output = root.path().join("arguments.txt");
    let account = root.path().join("account équipe space");
    for binary in [tool, shim] {
        let launch = root.path().join("launch.json");
        fs::write(&launch,serde_json::to_vec(&serde_json::json!({"binary":binary,"args":args,"directory":root.path(),"environment":[["CODEX_HOME",account.display().to_string()],["FIXTURE_OUTPUT",output.display().to_string()]],"remove_environment":["ANTHROPIC_AUTH_TOKEN"]})).expect("launch JSON")).expect("launch record");
        let status = Command::new(env!("CARGO_BIN_EXE_switchyard-desktop"))
            .arg("--terminal-job")
            .arg(&launch)
            .env("ANTHROPIC_AUTH_TOKEN", "must-be-removed")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("terminal job");
        assert!(status.success(), "terminal launch failed");
        let captured = fs::read_to_string(&output).expect("captured tool launch");
        assert!(captured.starts_with(&format!("{args:?}\n")), "{captured}");
        assert!(captured.contains(&format!("Ok({:?})", account.display().to_string())));
        assert!(captured.contains("NotPresent"));
        assert!(captured.contains(&format!("{:?}", root.path())));
        assert!(!launch.exists());
        assert!(!root.path().join("forbidden-file").exists());
        fs::remove_file(&output).expect("reset output");
    }
}
