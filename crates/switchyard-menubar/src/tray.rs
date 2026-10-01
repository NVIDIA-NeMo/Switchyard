// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The macOS status item: event loop, menu, and the menu's actions.
//!
//! Everything platform-specific lives here, so the rest of the crate builds
//! and is tested on any target.

use std::process::Command;
use std::time::{Duration, Instant};

use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy, NSEventMask};
use objc2_foundation::{MainThreadMarker, NSDate, NSDefaultRunLoopMode};
use tray_icon::menu::{IsMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIconBuilder};

use crate::app::refresh;
use crate::config::Config;
use crate::icon;
use crate::summary::Row;

const RESTART: &str = "restart-server";
const OPEN_CONFIG: &str = "open-config";
const OPEN_SETTINGS: &str = "open-settings";
const QUIT: &str = "quit";

/// How long the loop blocks waiting for a UI event before checking the clock.
const EVENT_POLL: f64 = 0.1;

/// Runs the status item until the user quits.
pub fn run(config: Config) -> Result<(), String> {
    let mtm = MainThreadMarker::new().ok_or("the menu bar must run on the main thread")?;
    let ns_app = NSApplication::sharedApplication(mtm);
    // Accessory keeps the process out of the Dock and the app switcher.
    ns_app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);

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

    let interval = Duration::from_secs(config.refresh_seconds.max(5));
    let mut next = Instant::now() + interval;
    loop {
        pump_events(&ns_app);

        let mut redraw = false;
        while let Ok(event) = MenuEvent::receiver().try_recv() {
            match event.id.as_ref() {
                QUIT => return Ok(()),
                RESTART => report(restart_server(&config)),
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

/// Restarts the server's LaunchAgent.
fn restart_server(config: &Config) -> Result<(), String> {
    let uid = command("id", &["-u"])?;
    let target = format!("gui/{}/{}", uid.trim(), config.launchd_label);
    command("launchctl", &["kickstart", "-k", &target]).map(|_| ())
}

/// Opens a config file in the user's editor.
fn open(path: &std::path::Path) -> Result<(), String> {
    command("open", &["-t", &path.display().to_string()]).map(|_| ())
}

/// Runs a command, returning its stdout or a message naming what failed.
fn command(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
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
        (RESTART, "Restart server"),
        (OPEN_CONFIG, "Open server config…"),
        (OPEN_SETTINGS, "Open menu bar settings…"),
        (QUIT, "Quit"),
    ] {
        append(&MenuItem::with_id(id, text, true, None))?;
    }
    Ok(menu)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_which_command_failed() {
        let error = command("switchyard-does-not-exist", &[]).expect_err("missing program");

        assert!(error.contains("switchyard-does-not-exist"));
    }
}
