// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The desktop and terminal apps manage a local Switchyard server.
//!
//! Both apps show recorded traffic and compare list-price estimates with the
//! configured baseline model. Their Routes views edit the server config and
//! restart the server.

#![cfg_attr(windows, windows_subsystem = "windows")]

mod accounts;
mod config;
mod controller;
#[cfg(any(target_os = "macos", windows))]
mod gui;
mod harness;
mod health;
mod history;
mod models;
mod pricing;
mod rollup;
mod server;
mod server_config;
mod sessions;
mod summary;
mod tui;
#[cfg(windows)]
mod windows;

use std::path::PathBuf;
use std::process::ExitCode;

use config::Config;

const USAGE: &str = "\
Usage: switchyard-desktop [--tui | --print | --validate-toml] [FILE]

This app manages routes, coding-tool settings, and usage.

  FILE            The app reads settings from FILE [default: ~/.switchyard/desktop.toml].
  --tui           Run the terminal app.
  --print         Print the current summary and exit.
  --validate-toml Parse FILE as TOML and exit.";

fn main() -> ExitCode {
    #[cfg(windows)]
    if let Some(result) = windows::terminal_job() {
        return match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("switchyard-desktop: {error}");
                ExitCode::FAILURE
            }
        };
    }

    #[cfg(windows)]
    windows::attach_console();
    let mut settings = None;
    let mut print_only = false;
    let mut validate_toml = false;
    let mut terminal = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--print" => print_only = true,
            "--tui" => terminal = true,
            "--validate-toml" => validate_toml = true,
            _ if arg.starts_with('-') => {
                eprintln!("switchyard-desktop: unknown option {arg}\n\n{USAGE}");
                return ExitCode::FAILURE;
            }
            _ => settings = Some(PathBuf::from(arg)),
        }
    }

    if terminal && (print_only || validate_toml) {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    }

    if validate_toml {
        let Some(path) = settings else {
            eprintln!("switchyard-desktop: --validate-toml requires a file path");
            return ExitCode::FAILURE;
        };
        if let Err(error) = config::validate_toml_file(&path) {
            eprintln!("switchyard-desktop: {error}");
            return ExitCode::FAILURE;
        }
        return ExitCode::SUCCESS;
    }

    let settings = match settings.map(Ok).unwrap_or_else(Config::default_path) {
        Ok(path) => path,
        Err(error) => {
            eprintln!("switchyard-desktop: {error}");
            return ExitCode::FAILURE;
        }
    };
    let config = match Config::load(&settings) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("switchyard-desktop: {error}");
            if print_only || (!terminal && !cfg!(any(target_os = "macos", windows))) {
                return ExitCode::FAILURE;
            }
            Config::default()
        }
    };
    // The default mode prints a summary on platforms without a desktop app.
    // The --tui option starts the terminal app on those platforms.
    if print_only || (!terminal && !cfg!(any(target_os = "macos", windows))) {
        if !print_only {
            eprintln!(
                "Run switchyard-desktop --tui {} to open the terminal app.",
                sessions::quote(&settings.display().to_string())
            );
        }
        let usage = match rollup::Reader::default()
            .read(&config.routing_log, chrono::Local::now().date_naive())
        {
            Ok(usage) => usage,
            Err(error) => {
                eprintln!(
                    "Usage unavailable. Could not read {}: {error}",
                    config.routing_log.display()
                );
                return ExitCode::FAILURE;
            }
        };
        for row in summary::build(health::probe(&config.server_url), &usage, &config) {
            println!("{row}");
        }
        return ExitCode::SUCCESS;
    }

    let controller = controller::Controller::new(config, settings);
    if terminal {
        #[cfg(windows)]
        windows::allocate_console();
        if let Err(error) = tui::run(controller) {
            eprintln!("switchyard-desktop: {error}");
            return ExitCode::FAILURE;
        }
    } else {
        #[cfg(any(target_os = "macos", windows))]
        if let Err(error) = gui::run(controller) {
            eprintln!("switchyard-desktop: {error}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
