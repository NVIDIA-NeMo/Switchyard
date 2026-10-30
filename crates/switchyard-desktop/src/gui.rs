// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The desktop window shows server health and recorded request and token counts.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use switchyard_desktop::{config::Config, health::Probe, rollup};
use tauri::State;

/// The window retains the probe and reader so refreshes reuse the health worker and log offset.
struct Desktop {
    settings: PathBuf,
    health: Probe,
    reader: rollup::Reader,
}

#[derive(serde::Serialize)]
struct Counts {
    requests: u64,
    tokens: u64,
}

#[derive(serde::Serialize)]
struct Snapshot {
    server_url: String,
    health: serde_json::Value,
    today: Option<Counts>,
    week: Option<Counts>,
    errors: Vec<String>,
    refresh_seconds: u64,
}

type Shared = Arc<Mutex<Desktop>>;

/// The worker reads health and usage while the mutex prevents overlapping refreshes.
#[tauri::command]
async fn snapshot(state: State<'_, Shared>) -> Result<Snapshot, String> {
    let shared = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut desktop = shared
            .try_lock()
            .map_err(|_| "A refresh is running. Try again when it finishes.".to_string())?;
        let config = Config::load(&desktop.settings)?;
        let health = desktop.health.details(&config.server_url);
        let mut errors = Vec::new();
        // A failed read leaves counts absent so the window cannot show incomplete usage as zero.
        let usage = desktop
            .reader
            .read(&config.routing_log, chrono::Local::now().date_naive())
            .map_err(|error| errors.push(format!("Could not read usage: {error}")))
            .ok();
        let counts = |totals: &rollup::Totals| Counts {
            requests: totals.requests(),
            tokens: totals.tokens(),
        };
        Ok(Snapshot {
            server_url: config.server_url,
            health,
            today: usage.as_ref().map(|usage| counts(&usage.today)),
            week: usage.as_ref().map(|usage| counts(&usage.week)),
            errors,
            refresh_seconds: config.refresh_seconds.max(1),
        })
    })
    .await
    .map_err(|error| error.to_string())?
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum FileKind {
    Config,
    Logs,
}

/// The worker waits for any refresh, then opens the configured file after releasing the mutex.
#[tauri::command]
async fn open_file(state: State<'_, Shared>, kind: FileKind) -> Result<(), String> {
    let shared = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let config = {
            let desktop = shared
                .lock()
                .map_err(|_| "Could not access desktop state after a worker failed.".to_string())?;
            Config::load(&desktop.settings)?
        };
        let path = match kind {
            FileKind::Config => config.config_file,
            FileKind::Logs => config.routing_log,
        };
        #[cfg(target_os = "macos")]
        {
            let output = std::process::Command::new("open")
                .arg("--")
                .arg(&path)
                .output()
                .map_err(|error| format!("Could not open {}: {error}", path.display()))?;
            if !output.status.success() {
                return Err(format!(
                    "Could not open {}: {}",
                    path.display(),
                    String::from_utf8_lossy(&output.stderr).trim()
                ));
            }
            Ok(())
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            use windows_sys::Win32::UI::{
                Shell::ShellExecuteW, WindowsAndMessaging::SW_SHOWNORMAL,
            };
            let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
            let status = unsafe {
                ShellExecuteW(
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    path.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    SW_SHOWNORMAL,
                )
            } as isize;
            if status > 32 {
                Ok(())
            } else {
                Err(format!("Windows could not open the file (error {status})."))
            }
        }
    })
    .await
    .map_err(|error| error.to_string())?
}

/// Quitting closes the window and stops the application.
#[tauri::command]
fn quit_app(app: tauri::AppHandle) {
    app.exit(0);
}

/// The application owns one health probe and log reader for the lifetime of the window.
pub fn run(settings: PathBuf) -> Result<(), String> {
    tauri::Builder::default()
        .manage(Arc::new(Mutex::new(Desktop {
            settings,
            health: Probe::default(),
            reader: rollup::Reader::default(),
        })))
        .invoke_handler(tauri::generate_handler![snapshot, open_file, quit_app])
        .run(tauri::generate_context!())
        .map_err(|error| error.to_string())
}
