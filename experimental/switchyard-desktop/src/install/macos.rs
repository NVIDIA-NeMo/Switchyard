// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module installs the app bundle and the LaunchAgents for the server and the desktop app.

use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use crate::config::SERVER_LABEL as SERVER;

use super::{Settings, err, present, remove, run, write, xml};

const DESKTOP: &str = "com.nvidia.switchyard.desktop";

pub fn preflight() -> Result<(), String> {
    run(Command::new("launchctl").arg("help"))
}

pub fn stop(_s: &Settings) -> Result<(), String> {
    let domain = domain()?;
    for label in [SERVER, DESKTOP] {
        stop_launchd_job(&format!("{domain}/{label}"))?;
    }
    Ok(())
}

pub fn install(s: &Settings, _keys_saved: bool) -> Result<(), String> {
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

/// Removes the LaunchAgents and the app bundle.
pub fn uninstall(s: &Settings) -> Result<(), String> {
    for label in [SERVER, DESKTOP] {
        remove(&s.service_dir.join(format!("{label}.plist")))?;
    }
    if present(&s.app())? {
        fs::remove_dir_all(s.app()).map_err(err)?;
    }
    Ok(())
}

pub fn stage_bundle(s: &Settings, metadata: &str) -> Result<tempfile::TempDir, String> {
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
            &include_bytes!("../../icons/icon.icns")[..],
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

pub fn replace_app(s: &Settings, stage: tempfile::TempDir) -> Result<(), String> {
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
