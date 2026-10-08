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
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
};

struct Shared {
    controller: Arc<Mutex<Controller>>,
    // The state is idle (0), busy (1), or closing (2); closing prevents new operations.
    operation: Arc<AtomicU8>,
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
async fn snapshot(state: State<'_, Shared>) -> Result<serde_json::Value, String> {
    let operation = state.begin()?;
    let shared = state.controller.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _operation = operation;
        shared
            .try_lock()
            .map_err(|_| "An operation is running. Try again when it finishes.".to_string())?
            .snapshot()
    })
    .await
    .map_err(|e| e.to_string())?
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
            let quit = MenuItem::with_id(app, "quit", "Quit Switchyard", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open, &quit])?;
            // The tray keeps a simple high-contrast routing glyph at native resolution.
            let mut pixels = vec![0u8; 22 * 22 * 4];
            for y in 3..19 {
                for x in 3..19 {
                    if x == 10 || x == 11 || (y < 9 && (x == y + 4 || x + y == 17)) {
                        let i = (y * 22 + x) * 4;
                        pixels[i + 3] = 255;
                    }
                }
            }
            TrayIconBuilder::new()
                .icon(tauri::image::Image::new_owned(pixels, 22, 22))
                .icon_as_template(true)
                .tooltip("Switchyard — routes and usage")
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => show(app),
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
