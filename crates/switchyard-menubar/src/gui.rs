// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tauri manages the window and tray, and runs blocking operations on worker threads.

use crate::controller::{Action, Controller, Reply};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU8, Ordering},
};
use tauri::{
    Manager, State,
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder,
};

struct Shared {
    controller: Arc<Mutex<Controller>>,
    // The state is idle (0), busy (1), or closing (2); closing prevents new operations.
    operation: Arc<AtomicU8>,
}
struct TrayPreview {
    today: MenuItem<tauri::Wry>,
    week: MenuItem<tauri::Wry>,
}
impl TrayPreview {
    fn update(&self, snapshot: &serde_json::Value) -> tauri::Result<()> {
        for (prefix, item) in [("Today —", &self.today), ("This week —", &self.week)] {
            let label = snapshot["summary"]
                .as_array()
                .and_then(|rows| {
                    rows.iter()
                        .filter_map(|row| row.as_str())
                        .find(|row| row.starts_with(prefix) || *row == "No requests recorded yet")
                })
                .unwrap_or("Usage unavailable");
            item.set_text(label)?;
        }
        Ok(())
    }
}
// Operation keeps the app busy from queueing through worker completion.
// Dropping it returns the state to idle, so the user can retry Quit.
struct Operation(Arc<AtomicU8>);
impl Drop for Operation {
    fn drop(&mut self) {
        self.0.store(0, Ordering::Release);
    }
}
impl Shared {
    fn begin(&self) -> Result<Operation, String> {
        self.operation
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| "An operation is running. Try again when it finishes.".to_string())?;
        Ok(Operation(self.operation.clone()))
    }
    fn quit(&self) -> bool {
        self.operation
            .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
            || self.operation.load(Ordering::Acquire) == 2
    }
}

#[tauri::command]
async fn snapshot(
    app: tauri::AppHandle,
    state: State<'_, Shared>,
) -> Result<serde_json::Value, String> {
    let operation = state.begin()?;
    let shared = state.controller.clone();
    let snapshot = tauri::async_runtime::spawn_blocking(move || {
        let _operation = operation;
        shared
            .try_lock()
            .map_err(|_| "An operation is running. Try again when it finishes.".to_string())?
            .snapshot()
    })
    .await
    .map_err(|e| e.to_string())??;
    app.state::<TrayPreview>()
        .update(&snapshot)
        .map_err(|e| e.to_string())?;
    Ok(snapshot)
}
#[tauri::command]
async fn action(state: State<'_, Shared>, action: Action) -> Result<Reply, String> {
    let operation = state.begin()?;
    let shared = state.controller.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _operation = operation;
        shared
            .try_lock()
            .map_err(|_| "An operation is running. Try again when it finishes.".to_string())?
            .dispatch(action)
    })
    .await
    .map_err(|e| e.to_string())?
}
fn open_page(app: &tauri::AppHandle, page: &str) {
    if let Some(window) = app.get_webview_window("main") {
        // Only the fixed page names from the tray menu reach this script.
        let script = format!(
            "window.dispatchEvent(new CustomEvent('switchyard-page', {{detail:{page:?}}}))"
        );
        if let Err(error) = window.eval(&script) {
            eprintln!("switchyard-menubar: could not select the page: {error}");
        }
    }
    show(app);
}
fn show(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main")
        && let Err(e) = window.show().and_then(|()| window.set_focus())
    {
        eprintln!("switchyard-menubar: could not open the window: {e}");
    }
}
pub fn run(controller: Controller) -> Result<(), String> {
    let app = tauri::Builder::default()
        .manage(Shared {
            controller: Arc::new(Mutex::new(controller)),
            operation: Arc::new(AtomicU8::new(0)),
        })
        .invoke_handler(tauri::generate_handler![snapshot, action])
        .setup(|app| {
            let open = MenuItem::with_id(app, "open", "Open Switchyard", true, None::<&str>)?;
            let install = MenuItem::with_id(app, "install", "Install…", true, None::<&str>)?;
            let today =
                MenuItem::with_id(app, "today", "Today — loading usage…", true, None::<&str>)?;
            let week = MenuItem::with_id(
                app,
                "week",
                "This week — loading usage…",
                true,
                None::<&str>,
            )?;
            let usage = MenuItem::with_id(app, "usage", "View usage…", true, None::<&str>)?;
            let settings = MenuItem::with_id(app, "settings", "Settings…", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit Switchyard", true, None::<&str>)?;
            let separator = PredefinedMenuItem::separator(app)?;
            let bottom_separator = PredefinedMenuItem::separator(app)?;
            let menu = Menu::with_items(
                app,
                &[
                    &open,
                    &install,
                    &separator,
                    &today,
                    &week,
                    &usage,
                    &bottom_separator,
                    &settings,
                    &quit,
                ],
            )?;
            app.manage(TrayPreview { today, week });
            if let Ok(snapshot) = app
                .state::<Shared>()
                .controller
                .lock()
                .map_err(|e| e.to_string())?
                .snapshot()
            {
                app.state::<TrayPreview>().update(&snapshot)?;
            }
            TrayIconBuilder::new()
                .icon(tauri::image::Image::new_owned(
                    include_bytes!("../icons/tray.rgba").to_vec(),
                    22,
                    22,
                ))
                .icon_as_template(true)
                .tooltip("Switchyard — routes and usage")
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => open_page(app, "overview"),
                    "install" => open_page(app, "install"),
                    "today" | "week" | "usage" => open_page(app, "usage"),
                    "settings" => open_page(app, "settings"),
                    "quit" if app.state::<Shared>().quit() => app.exit(0),
                    _ => {}
                })
                .build(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                if let Err(e) = window.hide() {
                    eprintln!("switchyard-menubar: could not hide the window: {e}");
                }
            }
        })
        .build(tauri::generate_context!())
        .map_err(|e| e.to_string())?;
    app.run(|app, event| match event {
        tauri::RunEvent::Reopen { .. } => show(app),
        tauri::RunEvent::ExitRequested { api, .. } if !app.state::<Shared>().quit() => {
            api.prevent_exit()
        }
        _ => {}
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    // A queued operation must prevent Quit before its worker starts.
    fn quit_blocks_pending_operations_and_stays_closed() {
        let shared = Shared {
            controller: Arc::new(Mutex::new(Controller::new(
                crate::config::Config::default(),
                std::path::PathBuf::new(),
            ))),
            operation: Arc::new(AtomicU8::new(0)),
        };
        let operation = shared.begin().expect("operation");
        assert!(shared.begin().is_err());
        assert!(!shared.quit());
        drop(operation);
        assert!(shared.quit());
        assert!(shared.quit());
        assert!(shared.begin().is_err());
    }
}
