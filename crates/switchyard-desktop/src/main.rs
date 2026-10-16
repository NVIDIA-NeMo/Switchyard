// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The desktop app reports a local Switchyard server's recorded usage.
//!
//! It shows recorded traffic and compares list-price estimates with the
//! configured baseline model.

mod config;
mod health;
mod history;
mod pricing;
mod rollup;
mod summary;

use std::path::PathBuf;
use std::process::ExitCode;

use config::Config;

const USAGE: &str = "\
Usage: switchyard-desktop [--print | --validate-toml] [FILE]

This app prints the usage that a local Switchyard server recorded.

  FILE            The app reads settings from FILE [default: ~/.switchyard/desktop.toml].
  --print         Print the current summary and exit. This is the default.
  --validate-toml Parse FILE as TOML and exit.";

fn main() -> ExitCode {
    let mut settings = None;
    let mut validate_toml = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--print" => {}
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
    for row in summary::build(health::probe(&config.server_url), &usage, &config) {
        println!("{row}");
    }
    ExitCode::SUCCESS
}
