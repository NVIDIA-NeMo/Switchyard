// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The macOS app shows usage and manages settings for a local Switchyard server.
//!
//! The server owns routing; this process reads what the server wrote. It
//! shows today's and this week's traffic, and what that traffic would have
//! cost had every call gone to the capable model instead. Its "Edit
//! routes…" window edits a route in the server config and restarts the
//! server.

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod accounts;
mod app;
mod config;
#[cfg(target_os = "macos")]
mod dashboard;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod harness;
mod health;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod history;
mod pricing;
mod rollup;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod sessions;
mod summary;
// The macOS windows use these modules. They build on every target so their
// tests also run in CI, which does not run on macOS.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod models;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod server;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod server_config;
// The status item, its glyph, and the app windows use macOS APIs.
#[cfg(target_os = "macos")]
mod icon;
#[cfg(target_os = "macos")]
mod picker;
#[cfg(target_os = "macos")]
mod sidebar;
#[cfg(target_os = "macos")]
mod tray;

use std::path::PathBuf;
use std::process::ExitCode;

use config::Config;

const USAGE: &str = "\
Usage: switchyard-menubar [--print | --validate-toml] [FILE]

Shows Switchyard usage and estimated savings in the macOS menu bar.

  FILE            Settings file [default: ~/.switchyard/menubar.toml]
  --print         Print the current summary and exit, instead of running
  --validate-toml Parse FILE as TOML and exit";

fn main() -> ExitCode {
    let mut settings = None;
    let mut print_only = false;
    let mut validate_toml = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--print" => print_only = true,
            "--validate-toml" => validate_toml = true,
            _ if arg.starts_with('-') => {
                eprintln!("switchyard-menubar: unknown option {arg}\n\n{USAGE}");
                return ExitCode::FAILURE;
            }
            _ => settings = Some(PathBuf::from(arg)),
        }
    }

    if validate_toml {
        let Some(path) = settings else {
            eprintln!("switchyard-menubar: --validate-toml requires a file path");
            return ExitCode::FAILURE;
        };
        if let Err(error) = config::validate_toml_file(&path) {
            eprintln!("switchyard-menubar: {error}");
            return ExitCode::FAILURE;
        }
        return ExitCode::SUCCESS;
    }

    let settings = settings.unwrap_or_else(Config::default_path);
    let config = match Config::load(&settings) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("switchyard-menubar: {error}");
            return ExitCode::FAILURE;
        }
    };
    // Off macOS there is no menu bar to attach to, so printing is the whole
    // program. That keeps the rollup and pricing logic buildable everywhere.
    if print_only || !cfg!(target_os = "macos") {
        let (_, rows) = app::refresh(&config, &mut rollup::Reader::default());
        for row in rows {
            match row {
                summary::Row::Separator => println!(),
                summary::Row::Label(text) => println!("{text}"),
            }
        }
        return ExitCode::SUCCESS;
    }

    #[cfg(target_os = "macos")]
    if let Err(error) = tray::run(config, &settings) {
        eprintln!("switchyard-menubar: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
