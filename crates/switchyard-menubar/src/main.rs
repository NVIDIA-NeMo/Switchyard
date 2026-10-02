// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Menu bar companion for a locally running Switchyard server.
//!
//! The server owns routing; this process only reads what the server already
//! wrote. It shows today's and this week's traffic, and what that traffic
//! would have cost had every call gone to the capable model instead.

mod app;
mod config;
mod health;
mod pricing;
mod rollup;
mod summary;
// The status item and the glyph it draws are the only platform-specific code.
#[cfg(target_os = "macos")]
mod icon;
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

    let config = match Config::load(&settings.unwrap_or_else(Config::default_path)) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("switchyard-menubar: {error}");
            return ExitCode::FAILURE;
        }
    };
    // Off macOS there is no menu bar to attach to, so printing is the whole
    // program. That keeps the rollup and pricing logic buildable everywhere.
    if print_only || !cfg!(target_os = "macos") {
        for row in app::refresh(&config) {
            match row {
                summary::Row::Separator => println!(),
                summary::Row::Label(text) => println!("{text}"),
            }
        }
        return ExitCode::SUCCESS;
    }

    #[cfg(target_os = "macos")]
    if let Err(error) = tray::run(config) {
        eprintln!("switchyard-menubar: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
