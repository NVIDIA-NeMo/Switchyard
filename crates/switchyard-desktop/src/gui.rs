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
    // The state is idle (0), busy (1), closing (2), or busy with Quit queued (3).
    // The idle state accepts new operations.
    operation: Arc<AtomicU8>,
}
struct TrayPreview {
    menu: Menu<tauri::Wry>,
    models: Mutex<Vec<String>>,
    health: MenuItem<tauri::Wry>,
}
impl TrayPreview {
    fn update(&self, app: &tauri::AppHandle, snapshot: &serde_json::Value) -> Result<(), String> {
        let health = &snapshot["data_state"]["health"];
        self.health
            .set_text(format!(
                "Server: {} · checked {}",
                health["state"].as_str().unwrap_or("unknown"),
                health["checked_at"].as_str().unwrap_or("not yet")
            ))
            .map_err(|e| e.to_string())?;
        let totals: crate::rollup::Totals =
            serde_json::from_value(snapshot["analytics"]["all"].clone())
                .map_err(|error| error.to_string())?;
        for item in self.menu.items().map_err(|error| error.to_string())? {
            if item.id().as_ref().starts_with("model:") {
                self.menu.remove(&item).map_err(|error| error.to_string())?;
            }
        }
        let mut counts = totals.routed.clone();
        for (model, overhead) in &totals.classifier {
            let row = counts.entry(model.clone()).or_default();
            row.input += overhead.input;
            row.cached_input += overhead.cached_input;
            row.output += overhead.output;
        }
        let mut models: Vec<_> = counts.into_iter().collect();
        models.sort_by(|a, b| b.1.total().cmp(&a.1.total()).then_with(|| a.0.cmp(&b.0)));
        *self.models.lock().map_err(|e| e.to_string())? =
            models.into_iter().take(3).map(|(name, _)| name).collect();
        let rows = if snapshot["data_state"]["usage"]["state"] == "unavailable" {
            vec!["Usage unavailable; open Overview…".to_string()]
        } else {
            crate::summary::tray_models(&totals)
        };
        for (index, label) in rows.into_iter().enumerate() {
            let item = MenuItem::with_id(app, format!("model:{index}"), label, true, None::<&str>)
                .map_err(|error| error.to_string())?;
            self.menu
                .insert(&item, 5 + index)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}
// Operation keeps the app busy from queueing through worker completion.
// Drop returns a busy app to idle or closes it when Quit is queued.
struct Operation(Arc<AtomicU8>, Option<tauri::AppHandle>);
impl Drop for Operation {
    fn drop(&mut self) {
        if self
            .0
            .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Acquire)
            == Err(3)
        {
            self.0.store(2, Ordering::Release);
            if let Some(app) = &self.1 {
                app.exit(0);
            }
        }
    }
}
impl Shared {
    fn begin(&self) -> Result<Operation, String> {
        self.operation
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| "An operation is running. Try again when it finishes.".to_string())?;
        Ok(Operation(self.operation.clone(), None))
    }
    fn quit(&self) -> bool {
        if self
            .operation
            .compare_exchange(1, 3, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return false;
        }
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
    let mut operation = state.begin()?;
    operation.1 = Some(app.clone());
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
    app.state::<TrayPreview>().update(&app, &snapshot)?;
    Ok(snapshot)
}
#[tauri::command]
async fn action(
    app: tauri::AppHandle,
    state: State<'_, Shared>,
    action: Action,
) -> Result<Reply, String> {
    let mut operation = state.begin()?;
    operation.1 = Some(app.clone());
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
#[tauri::command]
async fn choose_path(directory: bool) -> Result<Option<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        #[cfg(windows)]
        { let dialog = rfd::FileDialog::new();
        Ok(if directory { dialog.pick_folder() } else { dialog.pick_file() }.map(|p|p.display().to_string())) }
        #[cfg(target_os = "macos")]
        {
        let script = if directory {
            "try\nreturn POSIX path of (choose folder with prompt \"Choose a Git project folder\")\non error number -128\nreturn \"\"\nend try"
        } else {
            "try\nreturn POSIX path of (choose file with prompt \"Choose coding-tool settings\")\non error number -128\nreturn \"\"\nend try"
        };
        let output = std::process::Command::new("osascript").args(["-e",script]).output().map_err(|e| e.to_string())?;
        if !output.status.success() {return Err("The file chooser could not open. Enter the full path in the field.".into());}
        let path=String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
        let path=path.trim_end_matches(['\r','\n']);
        Ok((!path.is_empty()).then(||path.to_owned()))
        }
    }).await.map_err(|e|e.to_string())?
}
fn open_page(app: &tauri::AppHandle, page: &str) {
    if let Some(window) = app.get_webview_window("main") {
        // Only the fixed page names from the tray menu reach this script.
        let script = format!(
            "window.dispatchEvent(new CustomEvent('switchyard-page', {{detail:{page:?}}}))"
        );
        if let Err(error) = window.eval(&script) {
            eprintln!("switchyard-desktop: could not select the page: {error}");
        }
    }
    show(app);
}
fn activation(app: &tauri::AppHandle, visible: bool) -> tauri::Result<()> {
    #[cfg(target_os = "macos")]
    {
        app.set_activation_policy(if visible {
            tauri::ActivationPolicy::Regular
        } else {
            tauri::ActivationPolicy::Accessory
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, visible);
        Ok(())
    }
}
fn show(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main")
        && let Err(e) = activation(app, true)
            .and_then(|()| window.show())
            .and_then(|()| window.set_focus())
    {
        eprintln!("switchyard-desktop: could not open the window: {e}");
    }
}
pub fn run(controller: Controller) -> Result<(), String> {
    let app = tauri::Builder::default()
        .manage(Shared {
            controller: Arc::new(Mutex::new(controller)),
            operation: Arc::new(AtomicU8::new(0)),
        })
        .invoke_handler(tauri::generate_handler![snapshot, action, choose_path])
        .setup(|app| {
            let open = MenuItem::with_id(app, "open", "Open Switchyard", true, None::<&str>)?;
            let install = MenuItem::with_id(app, "install", "Connect tools…", true, None::<&str>)?;
            let health = MenuItem::with_id(app,"health","Server: checking…",false,None::<&str>)?;
            let period = MenuItem::with_id(
                app,
                "period",
                "Models · all retained history",
                false,
                None::<&str>,
            )?;
            let usage = MenuItem::with_id(app, "usage", "View model usage…", true, None::<&str>)?;
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
                    &health,
                    &period,
                    &usage,
                    &bottom_separator,
                    &settings,
                    &quit,
                ],
            )?;
            app.manage(TrayPreview { menu: menu.clone(), models: Mutex::new(Vec::new()), health });
            if let Ok(snapshot) = app
                .state::<Shared>()
                .controller
                .lock()
                .map_err(|e| e.to_string())?
                .snapshot()
            {
                app.state::<TrayPreview>().update(app.handle(), &snapshot)?;
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
                    "usage" => open_page(app, "overview"),
                    id if id.starts_with("model:") => {
                        let model = id.strip_prefix("model:").and_then(|index|index.parse::<usize>().ok()).and_then(|index|app.state::<TrayPreview>().models.lock().ok()?.get(index).cloned());
                        if let Some(window)=app.get_webview_window("main") {
                            let detail=serde_json::json!({"page":"overview","model":model});
                            let _=window.eval(format!("window.dispatchEvent(new CustomEvent('switchyard-page',{{detail:{detail}}}))").as_str());
                        }
                        show(app);
                    },
                    "settings" => open_page(app, "settings"),
                    "quit" => {
                        if app.state::<Shared>().quit() { app.exit(0); }
                        else if let Some(window)=app.get_webview_window("main") {
                            let _=window.eval("window.dispatchEvent(new CustomEvent('switchyard-notice',{detail:'Switchyard will quit when the current operation finishes.'}))");
                            show(app);
                        }
                    },
                    _ => {}
                })
                .build(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                // Accessory removes the running Dock entry while keeping the tray menu.
                if let Err(e) = window.hide().and_then(|()| {
                    activation(window.app_handle(), false)
                }) {
                    eprintln!("switchyard-desktop: could not hide the window: {e}");
                }
            }
        })
        .build(tauri::generate_context!())
        .map_err(|e| e.to_string())?;
    app.run(|app, event| match event {
        #[cfg(target_os = "macos")]
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
