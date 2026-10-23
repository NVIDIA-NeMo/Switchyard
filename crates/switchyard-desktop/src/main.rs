// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The desktop app reports a local Switchyard server's recorded usage.
//!
//! The printed summary shows recorded traffic and compares list-price estimates with the
//! configured baseline model.

#![cfg_attr(windows, windows_subsystem = "windows")]

use switchyard_desktop::{config, health, rollup};
#[cfg(any(target_os = "macos", windows))]
mod gui;
mod pricing;
mod summary;

use std::path::PathBuf;
use std::process::ExitCode;

use config::Config;

const USAGE: &str = "\
Usage: switchyard-desktop [--print | --validate-toml] [FILE]

This app opens the desktop window. Use --print for a usage summary.

  FILE            The app reads settings from FILE.
  --print         Print the current summary and exit.
  --validate-toml Parse FILE as TOML and exit.

Without FILE, the app uses bundled installation metadata to find desktop.toml.
Without that metadata, it uses $SY_HOME/desktop.toml, or ~/.switchyard/desktop.toml
when SY_HOME is unset.

Missing settings use the directory and port from adjacent install.toml, or the
settings file's directory and port 4123 when that metadata is absent.
The default server URL includes the port, such as http://127.0.0.1:4123.
A configured HTTP URL without a port uses port 80.

Savings compare spending on all calls with the cost of the answers at
baseline_model's bundled rates or the table selected by pricing_file.
Recorded spending remains visible when savings are unknown.";

fn main() -> ExitCode {
    // A desktop launch may have no parent console, so attachment failure does not stop the app.
    #[cfg(windows)]
    let _ = unsafe {
        windows_sys::Win32::System::Console::AttachConsole(
            windows_sys::Win32::System::Console::ATTACH_PARENT_PROCESS,
        )
    };
    let mut settings = None;
    let mut validate_toml = false;
    let mut print_only = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--print" => print_only = true,
            "--validate-toml" => validate_toml = true,
            _ if arg.starts_with('-') => {
                eprintln!("switchyard-desktop: unknown option {arg}\n\n{USAGE}");
                return ExitCode::FAILURE;
            }
            _ => settings = Some(PathBuf::from(arg)),
        }
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
    if !print_only {
        #[cfg(any(target_os = "macos", windows))]
        return match gui::run(settings) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("switchyard-desktop: {error}");
                ExitCode::FAILURE
            }
        };
        #[cfg(not(any(target_os = "macos", windows)))]
        {
            eprintln!(
                "The desktop window supports macOS and Windows. Use --print for a usage summary."
            );
            return ExitCode::FAILURE;
        }
    }

    let config = match Config::load(&settings) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("switchyard-desktop: {error}");
            return ExitCode::FAILURE;
        }
    };
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
    let rates = match pricing::load(config.prices_file.as_deref()) {
        Ok(rates) => rates,
        Err(error) => {
            eprintln!("switchyard-desktop: {error}");
            return ExitCode::FAILURE;
        }
    };
    let status = health::Probe::default().probe(&config.server_url);
    for row in summary::build(status, &usage, &config, &rates) {
        println!("{row}");
    }
    ExitCode::SUCCESS
}
