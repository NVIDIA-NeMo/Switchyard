// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI entrypoint for running the components-v2 Rust profile server.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use clap::Parser;
use switchyard_components_v2::{
    RelaySnapshotLimits, DEFAULT_MAX_ATOF_BATCH_BYTES, DEFAULT_MAX_ATOF_EVENT_BYTES,
    DEFAULT_MAX_RELAY_DEDUPE_ENTRIES, DEFAULT_MAX_RELAY_HISTORY_PER_IDENTITY,
    DEFAULT_MAX_RELAY_IDENTITIES, DEFAULT_MAX_RELAY_RETAINED_BYTES,
};
use switchyard_core::Result;
use switchyard_server::{run_server, ServerRunOptions, DEFAULT_LISTEN_BACKLOG};

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

    /// Optional bearer token required by the Relay Decision and ATOF endpoints.
    #[arg(long, env = "SWITCHYARD_ATOF_BEARER_TOKEN", value_name = "TOKEN")]
    pub(crate) atof_bearer_token: Option<String>,

    /// Maximum exact Relay identities retained in memory.
    #[arg(
        long,
        env = "SWITCHYARD_ATOF_MAX_IDENTITIES",
        default_value_t = DEFAULT_MAX_RELAY_IDENTITIES
    )]
    pub(crate) atof_max_identities: usize,

    /// Maximum reconstructed messages retained for each Relay identity.
    #[arg(
        long,
        env = "SWITCHYARD_ATOF_MAX_HISTORY_PER_IDENTITY",
        default_value_t = DEFAULT_MAX_RELAY_HISTORY_PER_IDENTITY
    )]
    pub(crate) atof_max_history_per_identity: usize,

    /// Maximum ATOF event idempotency keys retained in memory.
    #[arg(
        long,
        env = "SWITCHYARD_ATOF_MAX_DEDUPE_ENTRIES",
        default_value_t = DEFAULT_MAX_RELAY_DEDUPE_ENTRIES
    )]
    pub(crate) atof_max_dedupe_entries: usize,

    /// Maximum encoded/string bytes retained across all Relay state.
    #[arg(
        long,
        env = "SWITCHYARD_ATOF_MAX_RETAINED_BYTES",
        default_value_t = DEFAULT_MAX_RELAY_RETAINED_BYTES
    )]
    pub(crate) atof_max_retained_bytes: usize,

    /// Maximum encoded size accepted for one ATOF event.
    #[arg(
        long,
        env = "SWITCHYARD_ATOF_MAX_EVENT_BYTES",
        default_value_t = DEFAULT_MAX_ATOF_EVENT_BYTES
    )]
    pub(crate) atof_max_event_bytes: usize,

    /// Maximum encoded size accepted for one ATOF HTTP POST batch.
    #[arg(
        long,
        env = "SWITCHYARD_ATOF_MAX_BATCH_BYTES",
        default_value_t = DEFAULT_MAX_ATOF_BATCH_BYTES
    )]
    pub(crate) atof_max_batch_bytes: usize,
}

impl ServerArgs {
    /// Parses command-line arguments using clap.
    pub(crate) fn parse_args() -> Self {
        Self::parse()
    }

    fn into_options(self) -> ServerRunOptions {
        ServerRunOptions {
            config: self.config,
            addr: SocketAddr::new(self.host, self.port),
            backlog: self.backlog,
            dry_run: self.dry_run,
            atof_bearer_token: self.atof_bearer_token,
            relay_snapshot_limits: RelaySnapshotLimits {
                max_identities: self.atof_max_identities,
                max_history_per_identity: self.atof_max_history_per_identity,
                max_dedupe_entries: self.atof_max_dedupe_entries,
                max_retained_bytes: self.atof_max_retained_bytes,
                max_event_bytes: self.atof_max_event_bytes,
                max_batch_bytes: self.atof_max_batch_bytes,
            },
        }
    }
}

/// Loads config, optionally validates it, then starts the Rust server.
pub(crate) async fn run(args: ServerArgs) -> Result<()> {
    run_server(args.into_options()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atof_options_are_forwarded_to_server_runtime(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let args = ServerArgs::try_parse_from([
            "switchyard-server",
            "--config",
            "profiles.yaml",
            "--atof-bearer-token",
            "relay-secret",
            "--atof-max-identities",
            "17",
            "--atof-max-history-per-identity",
            "18",
            "--atof-max-dedupe-entries",
            "19",
            "--atof-max-retained-bytes",
            "2000",
            "--atof-max-event-bytes",
            "1024",
            "--atof-max-batch-bytes",
            "2048",
        ])?;

        let options = args.into_options();
        assert_eq!(options.atof_bearer_token.as_deref(), Some("relay-secret"));
        assert_eq!(options.relay_snapshot_limits.max_identities, 17);
        assert_eq!(options.relay_snapshot_limits.max_history_per_identity, 18);
        assert_eq!(options.relay_snapshot_limits.max_dedupe_entries, 19);
        assert_eq!(options.relay_snapshot_limits.max_retained_bytes, 2000);
        assert_eq!(options.relay_snapshot_limits.max_event_bytes, 1024);
        assert_eq!(options.relay_snapshot_limits.max_batch_bytes, 2048);
        Ok(())
    }
}
