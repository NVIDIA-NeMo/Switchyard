// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The macOS status item: event loop, menu, and the menu's actions.
//!
//! Everything platform-specific lives here and in the picker window, so the
//! rest of the crate builds and is tested on any target.

use std::path::Path;
use std::time::{Duration, Instant};

use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy, NSEventMask};
use objc2_foundation::{MainThreadMarker, NSDate, NSDefaultRunLoopMode};
use tray_icon::menu::{IsMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{Icon, TrayIconBuilder};

use crate::app::refresh;
use crate::config::Config;
use crate::icon;
use crate::models::CACHE_FILE;
use crate::picker::Picker;
use crate::server::{command, restart};
use crate::summary::Row;

const CHANGE_ROUTING: &str = "change-routing";
const RESTART: &str = "restart-server";
const OPEN_CONFIG: &str = "open-config";
const OPEN_SETTINGS: &str = "open-settings";
const QUIT: &str = "quit";

/// How long the loop blocks waiting for a UI event before checking the clock.
const EVENT_POLL: f64 = 0.1;

/// Runs the status item until the user quits. `settings` is the settings
/// file that `config` came from; the model list cache sits next to it.
pub fn run(config: Config, settings: &Path) -> Result<(), String> {
    let mtm = MainThreadMarker::new().ok_or("the menu bar must run on the main thread")?;
    let ns_app = NSApplication::sharedApplication(mtm);
    // Accessory keeps the process out of the Dock and the app switcher.
    ns_app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
    // An accessory app shows no menu bar, but text fields still find copy
    // and paste shortcuts through the app's main menu.
    let edit_menu = edit_menu()?;
    edit_menu.init_for_nsapp();

    let glyph = Icon::from_rgba(icon::glyph(), icon::SIZE, icon::SIZE)
        .map_err(|error| format!("build icon: {error}"))?;
    let tray = TrayIconBuilder::new()
        .with_icon(glyph)
        .with_icon_as_template(true)
        .with_tooltip("Switchyard")
        .with_menu(Box::new(menu(&refresh(&config))?))
        .build()
        .map_err(|error| format!("create status item: {error}"))?;

    ns_app.finishLaunching();

    let mut picker: Option<Picker> = None;
    let interval = Duration::from_secs(config.refresh_seconds.max(5));
    let mut next = Instant::now() + interval;
    loop {
        pump_events(&ns_app);
        if let Some(picker) = picker.as_mut() {
            picker.poll();
        }

        let mut redraw = false;
        while let Ok(event) = MenuEvent::receiver().try_recv() {
            match event.id.as_ref() {
                QUIT => return Ok(()),
                CHANGE_ROUTING => picker
                    .get_or_insert_with(|| {
                        Picker::new(mtm, &config, settings.with_file_name(CACHE_FILE))
                    })
                    .show(),
                RESTART => report(restart(&config.launchd_label)),
                OPEN_CONFIG => report(open(&config.config_file)),
                OPEN_SETTINGS => report(open(&Config::default_path())),
                // Informational rows are disabled, so nothing else fires.
                _ => continue,
            }
            redraw = true;
        }

        if redraw || Instant::now() >= next {
            tray.set_menu(Some(Box::new(menu(&refresh(&config))?)));
            next = Instant::now() + interval;
        }
    }
}

/// Opens a config file in the user's editor.
fn open(path: &std::path::Path) -> Result<(), String> {
    command("open", &["-t", &path.display().to_string()]).map(|_| ())
}

fn report(result: Result<(), String>) {
    if let Err(error) = result {
        eprintln!("switchyard-menubar: {error}");
    }
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
        (CHANGE_ROUTING, "Change routing…"),
        (RESTART, "Restart server"),
        (OPEN_CONFIG, "Open server config…"),
        (OPEN_SETTINGS, "Open menu bar settings…"),
        (QUIT, "Quit"),
    ] {
        append(&MenuItem::with_id(id, text, true, None))?;
    }
    Ok(menu)
}

/// The app's hidden main menu, which gives text fields their Edit shortcuts.
fn edit_menu() -> Result<Menu, String> {
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
    Menu::with_items(&[&edit]).map_err(|error| format!("build the main menu: {error}"))
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
