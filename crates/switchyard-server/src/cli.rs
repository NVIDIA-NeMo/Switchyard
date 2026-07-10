// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI entrypoint for running the components-v2 Rust profile server.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use clap::Parser;
use switchyard_core::{Result, SwitchyardError};
use switchyard_server::{run_server, ServerRunOptions, TLSOptions, DEFAULT_LISTEN_BACKLOG};

const DEFAULT_HOST: IpAddr = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
const DEFAULT_PORT: u16 = 4000;

/// Command-line arguments accepted by the Rust server binary.
#[derive(Debug, Parser)]
#[command(
    name = "switchyard-server",
    about = "Run the Rust Switchyard server from a components-v2 profile config",
    version
)]
pub(crate) struct ServerArgs {
    /// Path to a components-v2 profile config file.
    #[arg(short, long, env = "SWITCHYARD_PROFILE_CONFIG", value_name = "PATH")]
    pub(crate) config: PathBuf,

    /// Host address to bind.
    #[arg(long, default_value_t = DEFAULT_HOST)]
    pub(crate) host: IpAddr,

    /// Port to bind.
    #[arg(short, long, default_value_t = DEFAULT_PORT)]
    pub(crate) port: u16,

    /// TCP listen backlog passed to the socket before Axum accepts traffic.
    #[arg(long, default_value_t = DEFAULT_LISTEN_BACKLOG)]
    pub(crate) backlog: u32,

    /// Validate and build the config without starting the HTTP listener.
    #[arg(long)]
    pub(crate) dry_run: bool,

    /// TLS certificate path, PEM format
    #[arg(long, requires = "tls_key")]
    pub(crate) tls_cert: Option<PathBuf>,

    /// TLS certificate key path, PEM format
    #[arg(long, requires = "tls_cert")]
    pub(crate) tls_key: Option<PathBuf>,
}

impl ServerArgs {
    /// Parses command-line arguments using clap.
    pub(crate) fn parse_args() -> Self {
        Self::parse()
    }

    fn into_options(self) -> Result<ServerRunOptions> {
        let mut tls_options = None;
        if let (Some(cert), Some(key)) = (self.tls_cert, self.tls_key) {
            if !cert.exists() || !key.exists() {
                return Err(SwitchyardError::InvalidConfig(format!(
                    "Invalid path in --tls-cert {} or --tls-key {}. File does not exist.",
                    cert.display(),
                    key.display()
                )));
            }
            tls_options = Some(TLSOptions { cert, key })
        };
        Ok(ServerRunOptions {
            config: self.config,
            addr: SocketAddr::new(self.host, self.port),
            backlog: self.backlog,
            dry_run: self.dry_run,
            tls: tls_options,
        })
    }
}

/// Loads config, optionally validates it, then starts the Rust server.
pub(crate) async fn run(args: ServerArgs) -> Result<()> {
    let opts = args.into_options()?;
    run_server(opts).await
}
