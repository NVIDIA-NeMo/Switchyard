// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The desktop and terminal apps manage a local Switchyard server.
//!
//! Both apps show recorded traffic and compare list-price estimates with the
//! configured baseline model. Their Routes views edit the server config and
//! restart the server.

mod accounts;
mod app;
mod config;
mod controller;
#[cfg(target_os = "macos")]
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

use std::path::PathBuf;
use std::process::ExitCode;

use config::Config;

const USAGE: &str = "\
Usage: switchyard-menubar [--tui | --print | --validate-toml] [FILE]

This app manages routes, coding-tool settings, and usage.

  FILE            The app reads settings from FILE [default: ~/.switchyard/menubar.toml].
  --tui           Run the terminal app.
  --print         Print the current summary and exit.
  --validate-toml Parse FILE as TOML and exit.";

fn main() -> ExitCode {
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
                eprintln!("switchyard-menubar: unknown option {arg}\n\n{USAGE}");
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
    // The default mode prints a summary on platforms without the macOS desktop app.
    // The --tui option starts the terminal app on those platforms.
    if print_only || (!terminal && !cfg!(target_os = "macos")) {
        let (_, rows) = app::refresh(&config, &mut rollup::Reader::default());
        for row in rows {
            match row {
                summary::Row::Separator => println!(),
                summary::Row::Label(text) => println!("{text}"),
            }
        }
        return ExitCode::SUCCESS;
    }

    let controller = controller::Controller::new(config, settings);
    if terminal {
        if let Err(error) = tui::run(controller) {
            eprintln!("switchyard-menubar: {error}");
            return ExitCode::FAILURE;
        }
    } else {
        #[cfg(target_os = "macos")]
        if let Err(error) = gui::run(controller) {
            eprintln!("switchyard-menubar: {error}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
