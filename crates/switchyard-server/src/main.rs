// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary entrypoint for `switchyard-server`.

use std::process::ExitCode;

use tracing_subscriber::EnvFilter;

mod cli;

const DEFAULT_LOG_FILTER: &str = "switchyard_server=info";
const OPENTELEMETRY_LOG_FILTER: &str = "opentelemetry=warn";

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    if let Err(error) = init_logging() {
        eprintln!("failed to initialize logging: {error}");
        return ExitCode::FAILURE;
    }
    match cli::run(cli::ServerArgs::parse_args()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn init_logging() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_FILTER))
        .add_directive(OPENTELEMETRY_LOG_FILTER.parse()?);
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .try_init()?;
    Ok(())
}
