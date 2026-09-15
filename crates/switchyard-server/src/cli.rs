// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI entrypoint for running the configured libsy server.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use clap::Parser;
use switchyard_server::config::load_server_state;
use switchyard_server::{
    DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT, DEFAULT_LISTEN_BACKLOG, ServerError, ServerResult,
    ServerRunOptions, ServerState, TlsOptions, run_server,
};

const DEFAULT_HOST: IpAddr = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
const DEFAULT_PORT: u16 = 4000;

/// Command-line arguments accepted by the Rust server binary.
#[derive(Debug, Parser)]
#[command(
    name = "switchyard-server",
    about = "Serve explicitly configured libsy algorithms",
    version
)]
pub(crate) struct ServerArgs {
    /// TOML file defining LLM clients, targets, and algorithm routes.
    #[arg(long, value_name = "PATH")]
    config: PathBuf,

    /// Host address to bind.
    #[arg(long, default_value_t = DEFAULT_HOST)]
    host: IpAddr,

    /// Port to bind.
    #[arg(short, long, default_value_t = DEFAULT_PORT)]
    port: u16,

    /// TCP listen backlog passed to the socket before Axum accepts traffic.
    #[arg(long, default_value_t = DEFAULT_LISTEN_BACKLOG)]
    backlog: u32,

    /// Maximum time active requests may drain during shutdown.
    #[arg(long, default_value_t = humantime::Duration::from(DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT))]
    shutdown_timeout: humantime::Duration,

    /// Validate the algorithm and client configuration without binding a socket.
    #[arg(long)]
    dry_run: bool,

    /// Append durable per-request routing records to this JSONL file.
    #[arg(long, value_name = "PATH")]
    routing_log_file: Option<PathBuf>,

    /// TLS certificate path in PEM format.
    #[arg(long, requires = "tls_key")]
    tls_cert: Option<PathBuf>,

    /// TLS private-key path in PEM format.
    #[arg(long, requires = "tls_cert")]
    tls_key: Option<PathBuf>,
}

impl ServerArgs {
    /// Parses command-line arguments using clap.
    pub(crate) fn parse_args() -> Self {
        Self::parse()
    }

    fn into_runtime(self) -> ServerResult<(ServerState, ServerRunOptions)> {
        let mut state = load_server_state(&self.config)?;
        if let Some(path) = self.routing_log_file {
            state = state.with_routing_log(path)?;
        }
        let tls = match (self.tls_cert, self.tls_key) {
            (Some(cert), Some(key)) => {
                if !cert.exists() || !key.exists() {
                    return Err(ServerError::new(format!(
                        "invalid --tls-cert {} or --tls-key {}: file does not exist",
                        cert.display(),
                        key.display()
                    )));
                }
                Some(TlsOptions { cert, key })
            }
            _ => None,
        };
        let options = ServerRunOptions {
            addr: SocketAddr::new(self.host, self.port),
            backlog: self.backlog,
            dry_run: self.dry_run,
            shutdown_timeout: self.shutdown_timeout.into(),
            tls,
        };
        Ok((state, options))
    }
}

/// Loads the configured algorithms and starts the server.
pub(crate) async fn run(args: ServerArgs) -> ServerResult<()> {
    let (state, options) = args.into_runtime()?;
    run_server(state, options).await
}
