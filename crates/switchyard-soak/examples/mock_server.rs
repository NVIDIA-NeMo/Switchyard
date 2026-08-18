// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local VidaiMock server used by `scripts/run_local_soak_test.py`.

use std::process::ExitCode;

use clap::Parser;
use vidaimock::MockServer;

/// Start the embedded mock backend used by the local soak test.
#[derive(Parser)]
#[command(
    name = "switchyard-soak-mock",
    about = "Start VidaiMock for the local Switchyard soak test",
    after_long_help = "Example:\n  cargo run --release -p switchyard-soak --example switchyard-soak-mock -- --port 8100 --latency-ms 40",
    version
)]
struct Args {
    /// Local TCP port used by the mock backend.
    #[arg(long, default_value_t = 8100)]
    port: u16,

    /// Artificial delay, in milliseconds, added to every mock response.
    #[arg(long, default_value_t = 40)]
    latency_ms: u64,
}

async fn run(args: Args) -> Result<(), String> {
    if args.port == 0 {
        return Err("--port must be greater than zero".to_string());
    }

    let server = MockServer::builder()
        .bind(format!("127.0.0.1:{}", args.port))
        .mode("realistic")
        .latency_ms(args.latency_ms)
        .start()
        .await
        .map_err(|error| error.to_string())?;

    println!("VidaiMock is ready at {}", server.base_url());
    tokio::signal::ctrl_c()
        .await
        .map_err(|error| format!("could not wait for shutdown: {error}"))?;
    server.shutdown().await.map_err(|error| error.to_string())
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Args::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("switchyard-soak-mock failed: {error}");
            ExitCode::FAILURE
        }
    }
}
