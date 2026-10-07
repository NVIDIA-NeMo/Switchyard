// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The macOS status item: event loop, menu, and the menu's actions.
//!
//! Everything platform-specific lives here and in the routes window, so the
//! rest of the crate builds and is tested on any target.

use objc2::rc::autoreleasepool;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use objc2_app_kit::{NSAlert, NSApplication, NSApplicationActivationPolicy, NSEventMask};
use objc2_foundation::{MainThreadMarker, NSDate, NSDefaultRunLoopMode, NSString};
use tray_icon::menu::{IsMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{Icon, TrayIconBuilder};

use crate::app::refresh;
use crate::config::Config;
use crate::health::ServerStatus;
use crate::icon;
use crate::models::CACHE_FILE;
use crate::picker::Picker;
use crate::server::{command, restart};
use crate::summary::Row;

const EDIT_ROUTES: &str = "edit-routes";
const RESTART: &str = "restart-server";
const OPEN_CONFIG: &str = "open-config";
const OPEN_SETTINGS: &str = "open-settings";
const QUIT: &str = "quit";

/// How long the loop blocks waiting for a UI event before checking the clock.
const EVENT_POLL: f64 = 0.1;

/// Runs the status item until the user quits. `settings` is the settings
/// file that `config` came from. The model list cache sits next to it, and
/// the menu bar reads the file again whenever it changes.
pub fn run(mut config: Config, settings: &Path) -> Result<(), String> {
    let mtm = MainThreadMarker::new().ok_or("the menu bar must run on the main thread")?;
    let ns_app = NSApplication::sharedApplication(mtm);
    // Accessory keeps the process out of the Dock and the app switcher.
    ns_app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
    // An accessory app shows no menu bar, but AppKit still finds keyboard
    // shortcuts, such as Cmd-C in a text field and Cmd-W in the routes
    // window, through the app's main menu.
    let main_menu = main_menu()?;
    main_menu.init_for_nsapp();

    let mut log = crate::rollup::Reader::default();
    let (first, first_rows) = refresh(&config, &mut log);
    let tray = TrayIconBuilder::new()
        .with_icon(glyph(first)?)
        .with_icon_as_template(true)
        .with_tooltip(tooltip(first))
        .with_menu(Box::new(menu(&first_rows)?))
        .build()
        .map_err(|error| format!("create status item: {error}"))?;
    let mut shown = first;
    let mut settings_seen = modified(settings);

    ns_app.finishLaunching();

    let mut picker: Option<Picker> = None;
    let mut next = Instant::now() + interval(&config);
    let mut actions: Vec<(&str, thread::JoinHandle<Result<(), String>>)> = Vec::new();
    loop {
        let quit = autoreleasepool(|_| -> Result<bool, String> {
            pump_events(&ns_app);
            if let Some(picker) = picker.as_mut() {
                picker.poll();
            }

            let mut redraw = false;
            while let Ok(event) = MenuEvent::receiver().try_recv() {
                match event.id.as_ref() {
                    QUIT => {
                        // Quitting between the save and the restart would leave a
                        // saved config that the server has not loaded.
                        if picker.as_ref().is_some_and(Picker::is_busy) {
                            alert(
                                mtm,
                                "Apply is still running",
                                "Quitting now could leave the new config saved while the server \
                             still runs the old one. Wait for the result in the routes window, \
                             then quit.",
                            );
                        } else {
                            return Ok(true);
                        }
                    }
                    EDIT_ROUTES => picker
                        .get_or_insert_with(|| {
                            Picker::new(mtm, &config, settings.with_file_name(CACHE_FILE))
                        })
                        .show(),
                    RESTART => {
                        let label = config.launchd_label.clone();
                        actions.push((
                            "Could not restart the server",
                            thread::spawn(move || restart(&label)),
                        ));
                    }
                    OPEN_CONFIG | OPEN_SETTINGS => {
                        let path = if event.id.as_ref() == OPEN_CONFIG {
                            config.config_file.clone()
                        } else {
                            settings.to_path_buf()
                        };
                        actions.push((
                            "Could not open the file",
                            thread::spawn(move || open(&path)),
                        ));
                    }
                    // Informational rows are disabled, so nothing else fires.
                    _ => continue,
                }
                redraw = true;
            }

            for index in (0..actions.len()).rev() {
                if actions[index].1.is_finished() {
                    let (title, action) = actions.swap_remove(index);
                    report(
                        mtm,
                        title,
                        action
                            .join()
                            .unwrap_or_else(|_| Err("menu action thread panicked".to_string())),
                    );
                    redraw = true;
                }
            }
            if redraw || Instant::now() >= next {
                let changed = modified(settings);
                if changed != settings_seen {
                    settings_seen = changed;
                    match Config::load(settings) {
                        Ok(loaded) => {
                            eprintln!("switchyard-menubar: reloaded {}", settings.display());
                            config = loaded;
                            if let Some(picker) = picker.as_mut() {
                                picker.set_settings(&config);
                            }
                        }
                        Err(error) => eprintln!("switchyard-menubar: {error}"),
                    }
                }
                let (latest, rows) = refresh(&config, &mut log);
                tray.set_menu(Some(Box::new(menu(&rows)?)));
                if latest != shown {
                    shown = latest;
                    eprintln!("switchyard-menubar: the server is {}", state(shown));
                    tray.set_icon_with_as_template(Some(glyph(shown)?), true)
                        .map_err(|error| format!("change the status icon: {error}"))?;
                    // A failed tooltip update is ignored. The tooltip is cosmetic,
                    // and the menu bar keeps running.
                    let _ = tray.set_tooltip(Some(tooltip(shown)));
                }
                next = Instant::now() + interval(&config);
            }
            Ok(false)
        })?;
        if quit {
            return Ok(());
        }
    }
}

/// How long the menu waits between refreshes.
fn interval(config: &Config) -> Duration {
    Duration::from_secs(config.refresh_seconds.max(5))
}

/// Draws the status icon: solid while the server answers, faint when it does not.
fn glyph(status: ServerStatus) -> Result<Icon, String> {
    Icon::from_rgba(
        icon::glyph(status == ServerStatus::Running),
        icon::SIZE,
        icon::SIZE,
    )
    .map_err(|error| format!("build icon: {error}"))
}

/// Says whether the server answers, in the words that the tooltip and the log
/// use.
fn state(status: ServerStatus) -> &'static str {
    match status {
        ServerStatus::Running => "running",
        ServerStatus::Stopped => "not responding",
    }
}

fn tooltip(status: ServerStatus) -> String {
    format!("Switchyard: the server is {}", state(status))
}

/// Returns when the settings file last changed, or `None` when the file is
/// missing. The menu bar compares it between refreshes, so a changed file
/// takes effect without a restart.
fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
}

/// Opens a config file in the user's editor.
fn open(path: &Path) -> Result<(), String> {
    command("open", &["-t", &path.display().to_string()]).map(|_| ())
}

/// Logs the error and shows it in a dialog when a menu action fails. A menu
/// click opens no window that could show the error.
fn report(mtm: MainThreadMarker, title: &str, result: Result<(), String>) {
    if let Err(error) = result {
        eprintln!("switchyard-menubar: {error}");
        alert(mtm, title, &error);
    }
}

/// Shows a dialog and waits for the user to close it. The app activates
/// itself first, because an accessory app's dialog opens behind other windows
/// otherwise.
fn alert(mtm: MainThreadMarker, title: &str, detail: &str) {
    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str(title));
    alert.setInformativeText(&NSString::from_str(detail));
    #[allow(deprecated)]
    NSApplication::sharedApplication(mtm).activateIgnoringOtherApps(true);
    alert.runModal();
}

/// Builds the whole menu: the summary rows, then the fixed actions.
fn menu(rows: &[Row]) -> Result<Menu, String> {
    let menu = Menu::new();
    let append = |item: &dyn IsMenuItem| {
        menu.append(item)
            .map_err(|error| format!("build menu: {error}"))
    };

    for row in rows {
        match row {
            Row::Separator => append(&PredefinedMenuItem::separator())?,
            // Rows are labels, not commands, so they are disabled.
            Row::Label(text) => append(&MenuItem::new(text, false, None))?,
        }
    }
    append(&PredefinedMenuItem::separator())?;
    for (id, text) in [
        (EDIT_ROUTES, "Edit routes…"),
        (RESTART, "Restart server"),
        (OPEN_CONFIG, "Open server config…"),
        (OPEN_SETTINGS, "Open menu bar settings…"),
        (QUIT, "Quit"),
    ] {
        append(&MenuItem::with_id(id, text, true, None))?;
    }
    Ok(menu)
}

/// Builds the app's hidden main menu, which gives text fields their Edit
/// shortcuts and windows their Cmd-W shortcut.
fn main_menu() -> Result<Menu, String> {
    let edit = Submenu::with_items(
        "Edit",
        true,
        &[
            &PredefinedMenuItem::undo(None),
            &PredefinedMenuItem::redo(None),
            &PredefinedMenuItem::separator(),
            &PredefinedMenuItem::cut(None),
            &PredefinedMenuItem::copy(None),
            &PredefinedMenuItem::paste(None),
            &PredefinedMenuItem::select_all(None),
        ],
    )
    .map_err(|error| format!("build the Edit menu: {error}"))?;
    let window = Submenu::with_items("Window", true, &[&PredefinedMenuItem::close_window(None)])
        .map_err(|error| format!("build the Window menu: {error}"))?;
    Menu::with_items(&[&edit, &window]).map_err(|error| format!("build the main menu: {error}"))
}

/// Drains pending AppKit events, blocking briefly when there are none.
fn pump_events(ns_app: &NSApplication) {
    let mut expiration = Some(NSDate::dateWithTimeIntervalSinceNow(EVENT_POLL));
    while let Some(event) = unsafe {
        ns_app.nextEventMatchingMask_untilDate_inMode_dequeue(
            NSEventMask::Any,
            expiration.as_deref(),
            NSDefaultRunLoopMode,
            true,
        )
    } {
        ns_app.sendEvent(&event);
        // Only the first wait blocks; the rest drain what is already queued.
        expiration = Some(NSDate::distantPast());
    }
}
