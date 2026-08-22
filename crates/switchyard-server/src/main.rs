// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary entrypoint for `switchyard-server`.

use std::process::ExitCode;

mod cli;
mod daemon;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    if let Err(error) = switchyard_server::initialize_observability() {
        eprintln!("failed to initialize observability: {error}");
        return ExitCode::FAILURE;
    }
    let args = cli::ServerArgs::parse_args();
    // Detach must happen before the async runtime does significant work. The
    // parent re-execs under `setsid` and exits; this code only runs again in
    // the detached child, which re-parses args without `--detach`. The call is
    // Unix-only because `detach_into_background` is `#[cfg(unix)]`.
    #[cfg(unix)]
    if args.detach {
        if let Err(error) = daemon::detach_into_background(&args.pidfile) {
            eprintln!("failed to detach switchyard-server: {error}");
            return ExitCode::FAILURE;
        }
        // Unreachable: detach_into_background exits the parent process.
    }
    // If this is the detached child, record its own real PID (the `setsid`
    // parent cannot observe it). A normal foreground run skips this.
    if let Some(pidfile) = daemon::detached_pidfile() {
        if let Err(error) = daemon::write_pidfile(&pidfile, std::process::id()) {
            eprintln!("failed to write pidfile {pidfile:?}: {error}");
            return ExitCode::FAILURE;
        }
    }
    let exit_code = match cli::run(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    };
    switchyard_server::flush_observability();
    exit_code
}
