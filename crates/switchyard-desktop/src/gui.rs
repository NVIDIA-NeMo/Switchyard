// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tauri manages the window and tray, and runs blocking operations on worker threads.

#[cfg(target_os = "macos")]
mod panel;

use crate::controller::{Action, Controller, Reply, Snapshot};
use crate::summary::Overview;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU8, Ordering},
};
use std::time::Duration;
use tauri::{
    Manager, State,
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
};

struct Shared {
    controller: Arc<Mutex<Controller>>,
    // The state is idle (0), busy (1), closing (2), or busy with Quit queued (3).
    // The idle state accepts new operations.
    operation: Arc<AtomicU8>,
    // This lock serializes saved appearance changes from the main window and popup.
    appearance: Mutex<()>,
}
/// TRAY_ID names the tray icon so refreshes can replace its menu.
const TRAY_ID: &str = "main";

/// TrayUsage reads the routing log for the tray menu and popup. It loads the settings
/// and keeps its own log reader, so the tray never waits for an app operation and the
/// app never waits for the tray.
struct TrayUsage {
    settings: PathBuf,
    reader: Mutex<crate::rollup::Reader>,
    // The menu is rebuilt only when these rows change, so an open menu does not flicker.
    rows: Mutex<Vec<String>>,
}

/// TrayView holds what the tray shows: the usage overview, or the errors that hide it.
#[derive(serde::Serialize)]
struct TrayView {
    overview: Option<Overview>,
    errors: Vec<String>,
    refresh_seconds: u64,
}

impl TrayUsage {
    /// This function reads new log entries, probes the server, and builds the overview.
    fn read(&self) -> TrayView {
        let config = match switchyard_desktop::config::Config::load(&self.settings) {
            Ok(config) => config,
            Err(error) => {
                return TrayView {
                    overview: None,
                    errors: vec![format!("Cannot read app settings: {error}")],
                    refresh_seconds: MIN_TRAY_REFRESH,
                };
            }
        };
        let refresh_seconds = config.refresh_seconds.max(MIN_TRAY_REFRESH);
        let today = chrono::Local::now().date_naive();
        let usage = match self.reader.lock() {
            Ok(mut reader) => reader.read(&config.routing_log, today),
            Err(_) => Err(std::io::Error::other(
                "the log reader stopped after an earlier failure",
            )),
        };
        let usage = match usage {
            Ok(usage) => usage,
            Err(error) => {
                return TrayView {
                    overview: None,
                    errors: vec![format!("Cannot read routing log: {error}")],
                    refresh_seconds,
                };
            }
        };
        let mut errors = Vec::new();
        // An invalid price file hides savings but keeps traffic and spending visible.
        let rates = crate::pricing::load(config.prices_file.as_deref()).unwrap_or_else(|error| {
            errors.push(format!("Could not read model prices: {error}"));
            Default::default()
        });
        let status = switchyard_desktop::health::probe(&config.server_url);
        TrayView {
            overview: Some(Overview::new(status, &usage, &config, &rates)),
            errors,
            refresh_seconds,
        }
    }
}

/// MIN_TRAY_REFRESH keeps the tray from probing the server more than once every 5 seconds.
const MIN_TRAY_REFRESH: u64 = 5;

/// This function reads the tray usage and replaces the tray menu when its rows changed.
fn refresh_tray(app: &tauri::AppHandle) -> TrayView {
    let usage = app.state::<TrayUsage>();
    let view = usage.read();
    let mut rows = view
        .overview
        .as_ref()
        .map(Overview::rows)
        .unwrap_or_default();
    if !view.errors.is_empty() && !rows.is_empty() {
        rows.push(String::new());
    }
    rows.extend(view.errors.iter().cloned());
    let Ok(mut shown) = usage.rows.lock() else {
        return view;
    };
    if *shown != rows {
        let result = tray_menu(app, &rows).and_then(|menu| match app.tray_by_id(TRAY_ID) {
            Some(tray) => tray.set_menu(Some(menu)),
            None => Ok(()),
        });
        match result {
            Ok(()) => *shown = rows,
            Err(error) => eprintln!("switchyard-desktop: could not update the tray menu: {error}"),
        }
    }
    view
}

/// This function builds the tray menu: the usage rows, then the server and app actions.
fn tray_menu(app: &tauri::AppHandle, rows: &[String]) -> tauri::Result<Menu<tauri::Wry>> {
    let menu = Menu::new(app)?;
    for row in rows {
        if row.is_empty() {
            menu.append(&PredefinedMenuItem::separator(app)?)?;
        } else {
            // Rows are labels, not commands, so they are disabled.
            menu.append(&MenuItem::new(app, row, false, None::<&str>)?)?;
        }
    }
    let groups: [&[(&str, &str)]; 3] = [
        &[
            ("restart", "Restart server"),
            ("open-config", "Open server config…"),
            ("settings", "Settings…"),
        ],
        &[("open", "Open Switchyard"), ("install", "Connect tools…")],
        &[("quit", "Quit Switchyard")],
    ];
    for group in groups {
        menu.append(&PredefinedMenuItem::separator(app)?)?;
        for (id, text) in group {
            menu.append(&MenuItem::with_id(app, *id, *text, true, None::<&str>)?)?;
        }
    }
    Ok(menu)
}

/// This function runs a tray menu action and reports a failure in the app window.
fn run_tray_action(app: &tauri::AppHandle, action: Action) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let result = dispatch(app.clone(), action).await;
        let refresh = app.clone();
        // The refresh probes the server, so it runs on a worker thread.
        let _ = tauri::async_runtime::spawn_blocking(move || refresh_tray(&refresh)).await;
        if let Err(error) = result {
            eprintln!("switchyard-desktop: {error}");
            notice(&app, &error);
        }
    });
}

/// This function shows a message in the app window and brings the window forward.
fn notice(app: &tauri::AppHandle, message: &str) {
    if let Some(window) = app.get_webview_window("main") {
        let detail = serde_json::json!(message);
        let _ = window.eval(format!(
            "window.dispatchEvent(new CustomEvent('switchyard-notice',{{detail:{detail}}}))"
        ));
    }
    show(app);
}

/// This function quits now, or after the running operation when one is running.
fn quit(app: &tauri::AppHandle) {
    if app.state::<Shared>().quit() {
        app.exit(0);
    } else {
        notice(
            app,
            "Switchyard will quit when the current operation finishes.",
        );
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
async fn snapshot(app: tauri::AppHandle, state: State<'_, Shared>) -> Result<Snapshot, String> {
    let mut operation = state.begin()?;
    operation.1 = Some(app.clone());
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
async fn dispatch(app: tauri::AppHandle, action: Action) -> Result<Reply, String> {
    let (operation, shared) = {
        let state = app.state::<Shared>();
        let mut operation = state.begin()?;
        operation.1 = Some(app.clone());
        (operation, state.controller.clone())
    };
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
async fn action(app: tauri::AppHandle, action: Action) -> Result<Reply, String> {
    dispatch(app, action).await
}
#[tauri::command]
async fn tray_overview(app: tauri::AppHandle) -> Result<TrayView, String> {
    // The overview probes the server, so it runs on a worker thread.
    tauri::async_runtime::spawn_blocking(move || refresh_tray(&app))
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
fn quit_app(app: tauri::AppHandle) {
    quit(&app);
}
#[tauri::command]
async fn choose_path(directory: bool) -> Result<Option<String>, String> {
    // The blocking worker waits for the picker; rfd opens macOS panels on the main thread.
    tauri::async_runtime::spawn_blocking(move || {
        let dialog = rfd::FileDialog::new().set_title(if directory {
            "Choose a Git project folder"
        } else {
            "Choose harness settings"
        });
        let path = if directory {
            dialog.pick_folder()
        } else {
            dialog.pick_file()
        };
        #[cfg(windows)]
        {
            Ok(path.map(|path| path.display().to_string()))
        }
        #[cfg(target_os = "macos")]
        {
            path.map(|path| {
                path.into_os_string().into_string().map_err(|_| {
                    "The selected path cannot be represented as UTF-8. \
                     Enter a different path in the field."
                        .to_string()
                })
            })
            .transpose()
        }
    })
    .await
    .map_err(|error| error.to_string())?
}

#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum AppearanceMode {
    #[default]
    System,
    // Dark is saved as "canary".
    Canary,
    Light,
}

#[tauri::command]
fn appearance(
    app: tauri::AppHandle,
    window: tauri::WebviewWindow,
    state: State<'_, Shared>,
    mode: Option<AppearanceMode>,
) -> Result<serde_json::Value, String> {
    let _lock = state.appearance.lock().map_err(|e| e.to_string())?;
    let system = window.theme().map_err(|e| e.to_string())?;
    let path = app
        .path()
        .app_config_dir()
        .map_err(|e| e.to_string())?
        .join("appearance.json");
    let selected = if let Some(mode) = mode {
        let directory = path
            .parent()
            .ok_or("Could not find the appearance settings folder.")?;
        std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
        use std::io::Write;
        let mut temporary =
            tempfile::NamedTempFile::new_in(directory).map_err(|e| e.to_string())?;
        temporary
            .write_all(&serde_json::to_vec(&mode).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        temporary.as_file().sync_all().map_err(|e| e.to_string())?;
        temporary.persist(&path).map_err(|e| e.to_string())?;
        mode
    } else {
        match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| format!("Could not read appearance settings: {e}"))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => AppearanceMode::default(),
            Err(e) => return Err(e.to_string()),
        }
    };
    let result = serde_json::json!({"mode": selected, "system": system});
    if mode.is_some() {
        for window in app.webview_windows().values() {
            window
                .eval(format!(
                    "window.dispatchEvent(new CustomEvent('switchyard-appearance', \
                     {{detail:{result}}}))"
                ))
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(result)
}

#[cfg(target_os = "macos")]
fn material(
    fallback: tauri::window::Effect,
    radius: Option<f64>,
    state: tauri::window::EffectState,
) -> tauri::utils::config::WindowEffectsConfig {
    tauri::utils::config::WindowEffectsConfig {
        effects: vec![tauri::window::Effect::LiquidGlassRegular, fallback],
        state: Some(state),
        radius,
        color: None,
        interactive: false,
    }
}

#[tauri::command]
fn accessibility() -> serde_json::Value {
    #[cfg(target_os = "macos")]
    {
        let workspace = objc2_app_kit::NSWorkspace::sharedWorkspace();
        serde_json::json!({
            "native_material": true,
            "reduce_transparency": workspace.accessibilityDisplayShouldReduceTransparency(),
            "reduce_motion": workspace.accessibilityDisplayShouldReduceMotion(),
        })
    }
    #[cfg(not(target_os = "macos"))]
    serde_json::json!({"native_material": false, "reduce_transparency": true, "reduce_motion": false})
}

/// POPUP_WIDTH matches the width of macOS status item menus such as Wi-Fi and Sound.
const POPUP_WIDTH: f64 = 320.0;

// The webview measures popup height; invalid sizes must not reach AppKit.
fn popup_height(height: f64) -> Result<f64, String> {
    if !height.is_finite() || height <= 0.0 {
        return Err("The popup height must be a positive number.".into());
    }
    Ok(height.clamp(120.0, 620.0))
}

#[tauri::command]
fn tray_size(app: tauri::AppHandle, height: f64) -> Result<(), String> {
    let height = popup_height(height)?;
    #[cfg(target_os = "macos")]
    app.state::<panel::TrayPanel>().resize(height);
    #[cfg(windows)]
    if let Some(window) = app.get_webview_window("tray") {
        let height = window
            .current_monitor()
            .map_err(|e| e.to_string())?
            .map(|monitor| {
                height.min(f64::from(monitor.work_area().size.height) / monitor.scale_factor())
            })
            .unwrap_or(height);
        window
            .set_size(tauri::LogicalSize::new(POPUP_WIDTH, height))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn tray_page(app: tauri::AppHandle, page: String) -> Result<(), String> {
    if !matches!(page.as_str(), "overview" | "install" | "settings") {
        return Err("Choose Overview, Connect tools, or Settings.".into());
    }
    open_page(&app, &page);
    dismiss_tray(app)
}
#[tauri::command]
fn dismiss_tray(app: tauri::AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    app.state::<panel::TrayPanel>().dismiss();
    #[cfg(windows)]
    if let Some(window) = app.get_webview_window("tray") {
        window.hide().map_err(|error| error.to_string())?;
    }
    Ok(())
}
// Windows uses physical coordinates to keep the popup within the target monitor's work area.
#[cfg(any(windows, test))]
fn popup_geometry(
    rect: tauri::Rect,
    area: &tauri::PhysicalRect<i32, u32>,
    scale: f64,
) -> (tauri::PhysicalPosition<i32>, tauri::PhysicalSize<u32>) {
    let origin = rect.position.to_physical::<f64>(scale);
    let icon = rect.size.to_physical::<f64>(scale);
    let size = tauri::PhysicalSize::new(
        (POPUP_WIDTH * scale)
            .min(f64::from(area.size.width))
            .round() as u32,
        (620.0 * scale).min(f64::from(area.size.height)).round() as u32,
    );
    let x = (origin.x + icon.width / 2.0 - f64::from(size.width) / 2.0).clamp(
        f64::from(area.position.x),
        f64::from(area.position.x) + f64::from(area.size.width - size.width),
    );
    let y = (origin.y + icon.height + 6.0 * scale).clamp(
        f64::from(area.position.y),
        f64::from(area.position.y) + f64::from(area.size.height - size.height),
    );
    (
        tauri::PhysicalPosition::new(x.round() as i32, y.round() as i32),
        size,
    )
}

fn toggle_tray(
    tray: &tauri::tray::TrayIcon,
    position: tauri::PhysicalPosition<f64>,
    rect: tauri::Rect,
) -> tauri::Result<()> {
    let app = tray.app_handle();
    #[cfg(target_os = "macos")]
    {
        let _ = (position, rect);
        // AppKit frames avoid converting Tauri's physical, top-left coordinates back to points.
        let Some((anchor, area)) = panel::status_item_frames(tray) else {
            return Ok(());
        };
        if app.state::<panel::TrayPanel>().toggle(anchor, area)
            && let Some(window) = app.get_webview_window("tray")
        {
            window.eval("window.dispatchEvent(new Event('switchyard-tray-open'))")?;
        }
        Ok(())
    }
    #[cfg(windows)]
    {
        let Some(window) = app.get_webview_window("tray") else {
            return Ok(());
        };
        if window.is_visible()? {
            return window.hide();
        }
        if let Some(monitor) = window.available_monitors()?.into_iter().find(|monitor| {
            let origin = monitor.position();
            let size = monitor.size();
            position.x >= f64::from(origin.x)
                && position.x < f64::from(origin.x) + f64::from(size.width)
                && position.y >= f64::from(origin.y)
                && position.y < f64::from(origin.y) + f64::from(size.height)
        }) {
            let (position, size) =
                popup_geometry(rect, monitor.work_area(), monitor.scale_factor());
            window.set_position(position)?;
            window.set_size(size)?;
        }
        window.show()?;
        window.set_focus()?;
        window.eval("window.dispatchEvent(new Event('switchyard-tray-open'))")?;
        Ok(())
    }
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
    if let Err(error) = dismiss_tray(app.clone()) {
        eprintln!("switchyard-desktop: could not hide the preview: {error}");
    }
    if let Some(window) = app.get_webview_window("main")
        && let Err(e) = activation(app, true)
            .and_then(|()| window.show())
            .and_then(|()| window.set_focus())
    {
        eprintln!("switchyard-desktop: could not open the window: {e}");
    }
}
pub fn run(controller: Controller, settings: PathBuf) -> Result<(), String> {
    let app = tauri::Builder::default()
        .manage(Shared {
            controller: Arc::new(Mutex::new(controller)),
            operation: Arc::new(AtomicU8::new(0)),
            appearance: Mutex::new(()),
        })
        .manage(TrayUsage {
            settings,
            reader: Mutex::new(crate::rollup::Reader::default()),
            rows: Mutex::new(Vec::new()),
        })
        .invoke_handler(tauri::generate_handler![
            snapshot,
            action,
            choose_path,
            appearance,
            accessibility,
            tray_size,
            tray_page,
            tray_overview,
            dismiss_tray,
            quit_app
        ])
        .setup(|app| {
            #[cfg(target_os = "macos")]
            if let Some(window) = app.get_webview_window("main") {
                window.set_effects(material(
                    tauri::window::Effect::Sidebar,
                    None,
                    tauri::window::EffectState::FollowsWindowActiveState,
                ))?;
            }
            let tray_window = tauri::WebviewWindowBuilder::new(
                app,
                "tray",
                tauri::WebviewUrl::App("tray.html".into()),
            )
            .title("Switchyard usage")
            .inner_size(POPUP_WIDTH, 620.0)
            .transparent(cfg!(target_os = "macos"))
            .decorations(false)
            .resizable(false)
            .always_on_top(true)
            .skip_taskbar(true)
            .visible(false)
            .focused(false)
            .build()?;
            #[cfg(target_os = "macos")]
            {
                // Like a menu, the popup keeps the active material although it never activates
                // the app.
                tray_window.set_effects(material(
                    tauri::window::Effect::Popover,
                    Some(panel::CORNER_RADIUS),
                    tauri::window::EffectState::Active,
                ))?;
                app.manage(panel::TrayPanel::new(&tray_window).map_err(std::io::Error::other)?);
            }
            #[cfg(windows)]
            let _ = tray_window;

            let tray_pixels = include_bytes!("../icons/tray.rgba");
            TrayIconBuilder::with_id(TRAY_ID)
                .icon(tauri::image::Image::new_owned(tray_pixels.to_vec(), 22, 22))
                .icon_as_template(false)
                .tooltip("Switchyard — routes and usage")
                .menu(&tray_menu(app.handle(), &["Reading usage…".into()])?)
                .show_menu_on_left_click(false)
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        position,
                        rect,
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                        && let Err(error) = toggle_tray(tray, position, rect)
                    {
                        eprintln!(
                            "switchyard-desktop: could not show or hide the usage preview: {error}"
                        );
                    }
                })
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "restart" => run_tray_action(app, Action::Restart {}),
                    "open-config" => run_tray_action(app, Action::OpenConfig {}),
                    "settings" => open_page(app, "settings"),
                    "open" => open_page(app, "overview"),
                    "install" => open_page(app, "install"),
                    "quit" => quit(app),
                    _ => {}
                })
                .build(app)?;
            // Like the menu bar app, the tray rereads usage on the settings interval.
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                loop {
                    let view = refresh_tray(&handle);
                    std::thread::sleep(Duration::from_secs(view.refresh_seconds));
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::ThemeChanged(theme) = event {
                let value = serde_json::json!(theme);
                for target in window.app_handle().webview_windows().values() {
                    if let Err(error) = target.eval(format!(
                        "window.dispatchEvent(new CustomEvent('switchyard-system-theme', \
                         {{detail:{value}}}))"
                    )) {
                        eprintln!("switchyard-desktop: could not update the appearance: {error}");
                    }
                }
            }
            #[cfg(windows)]
            if window.label() == "tray"
                && matches!(event, tauri::WindowEvent::Focused(false))
                && let Err(error) = window.hide()
            {
                eprintln!("switchyard-desktop: could not hide the usage preview: {error}");
            }

            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                if window.label() == "tray" {
                    if let Err(error) = dismiss_tray(window.app_handle().clone()) {
                        eprintln!("switchyard-desktop: could not hide the usage preview: {error}");
                    }
                    return;
                }
                // Accessory removes the running Dock entry while keeping the tray menu.
                if let Err(e) = window.hide().and_then(|()| {
                    if window.label() == "main" {
                        activation(window.app_handle(), false)
                    } else {
                        Ok(())
                    }
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
    fn popup_fits_physical_work_areas_and_restores_size_on_larger_monitors() {
        // The fixture places the icon on a monitor left of the primary display and checks two scales.
        let rect = tauri::Rect {
            position: tauri::Position::Physical(tauri::PhysicalPosition::new(-20, 1000)),
            size: tauri::Size::Physical(tauri::PhysicalSize::new(40, 40)),
        };
        let area = tauri::PhysicalRect {
            position: tauri::PhysicalPosition::new(-1920, 0),
            size: tauri::PhysicalSize::new(1920, 1040),
        };
        let (position, size) = popup_geometry(rect, &area, 2.0);
        assert_eq!(position, tauri::PhysicalPosition::new(-640, 0));
        assert_eq!(size, tauri::PhysicalSize::new(640, 1040));
        let (position, size) = popup_geometry(rect, &area, 1.0);
        assert_eq!(position, tauri::PhysicalPosition::new(-320, 420));
        assert_eq!(size, tauri::PhysicalSize::new(320, 620));
    }

    #[test]
    fn popup_height_rejects_invalid_measurements_and_bounds_valid_sizes() {
        assert!(popup_height(f64::NAN).is_err());
        assert!(popup_height(f64::INFINITY).is_err());
        assert!(popup_height(0.0).is_err());
        assert!(popup_height(-1.0).is_err());
        assert_eq!(popup_height(100.0).unwrap(), 120.0);
        assert_eq!(popup_height(500.0).unwrap(), 500.0);
        assert_eq!(popup_height(900.0).unwrap(), 620.0);
    }

    #[test]
    // A queued operation must prevent Quit before its worker starts.
    fn quit_blocks_pending_operations_and_stays_closed() {
        let directory = tempfile::tempdir().expect("directory");
        let settings = directory.path().join("app.toml");
        let shared = Shared {
            controller: Arc::new(Mutex::new(Controller::new(
                crate::config::Config::defaults_for(&settings).expect("defaults"),
                settings,
            ))),
            operation: Arc::new(AtomicU8::new(0)),
            appearance: Mutex::new(()),
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
