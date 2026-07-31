// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary entrypoint for `switchyard-server`.

use std::process::ExitCode;

mod cli;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    if let Err(error) = switchyard_server::initialize_observability() {
        eprintln!("failed to initialize observability: {error}");
        return ExitCode::FAILURE;
    }
    let exit_code = match cli::run(cli::ServerArgs::parse_args()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    };
    switchyard_server::flush_observability();
    exit_code
}
