// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI entrypoint for running the libsy server with random routing.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, ValueEnum};
use libsy::{Algorithm, LlmTarget, LlmTargetSet, RandomAlgo, RoutedLlmClient};
use switchyard_llm_client::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};
use switchyard_server::{
    run_server, ServerError, ServerResult, ServerRunOptions, ServerState, TlsOptions,
    DEFAULT_LISTEN_BACKLOG,
};

const DEFAULT_HOST: IpAddr = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
const DEFAULT_PORT: u16 = 4000;
const DEFAULT_ROUTE_MODEL: &str = "switchyard/random";

#[derive(Clone, Copy, Debug, ValueEnum)]
enum UpstreamFormat {
    #[value(name = "openai-chat")]
    OpenAiChat,
    #[value(name = "openai-responses")]
    OpenAiResponses,
    #[value(name = "anthropic")]
    Anthropic,
}

impl UpstreamFormat {
    fn backend(self, config: HttpBackendConfig) -> Backend {
        match self {
            Self::OpenAiChat => Backend::OpenAiChat(config),
            Self::OpenAiResponses => Backend::OpenAiResponses(config),
            Self::Anthropic => Backend::Anthropic(config),
        }
    }
}

/// Command-line arguments accepted by the Rust server binary.
#[derive(Debug, Parser)]
#[command(
    name = "switchyard-server",
    about = "Run uniform random routing with libsy",
    version
)]
pub(crate) struct ServerArgs {
    /// Public model id clients send to this server.
    #[arg(long, default_value = DEFAULT_ROUTE_MODEL)]
    route_model: String,

    /// Upstream model id eligible for random selection. Repeat for each target.
    #[arg(long = "target", required = true)]
    targets: Vec<String>,

    /// Base URL shared by the configured upstream targets.
    #[arg(long, env = "SWITCHYARD_UPSTREAM_BASE_URL")]
    base_url: String,

    /// Upstream API key. Omit when the backend needs no authentication.
    #[arg(long, env = "SWITCHYARD_UPSTREAM_API_KEY")]
    api_key: Option<String>,

    /// Provider wire format used by the upstream backend.
    #[arg(long, value_enum, default_value = "openai-chat")]
    upstream_format: UpstreamFormat,

    /// Host address to bind.
    #[arg(long, default_value_t = DEFAULT_HOST)]
    host: IpAddr,

    /// Port to bind.
    #[arg(short, long, default_value_t = DEFAULT_PORT)]
    port: u16,

    /// TCP listen backlog passed to the socket before Axum accepts traffic.
    #[arg(long, default_value_t = DEFAULT_LISTEN_BACKLOG)]
    backlog: u32,

    /// Validate the algorithm and client configuration without binding a socket.
    #[arg(long)]
    dry_run: bool,

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
        let targets = validated_targets(self.targets)?;
        if self.base_url.trim().is_empty() {
            return Err(ServerError::new("--base-url must not be empty"));
        }

        let backend = self.upstream_format.backend(HttpBackendConfig {
            base_url: self.base_url,
            api_key: self.api_key,
            extra_headers: BTreeMap::new(),
        });
        let model_configs = targets
            .iter()
            .map(|model| ModelConfig::new(model, backend.clone(), None))
            .collect::<Vec<_>>();
        let client: Arc<dyn RoutedLlmClient> = Arc::new(
            TranslatingLlmClient::new(&model_configs)
                .map_err(|error| ServerError::new(error.to_string()))?,
        );
        let target_set = LlmTargetSet::new(
            targets
                .iter()
                .map(|model| LlmTarget {
                    semantic_name: model.clone(),
                    llm_client: Some(Arc::clone(&client)),
                })
                .collect(),
        );
        let algorithm: Arc<dyn Algorithm> = Arc::new(RandomAlgo::new(target_set));
        let state = ServerState::new(
            self.route_model,
            format!("uniform random routing across {}", targets.join(", ")),
            algorithm,
        )?;

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
            tls,
        };
        Ok((state, options))
    }
}

fn validated_targets(targets: Vec<String>) -> ServerResult<Vec<String>> {
    if targets.iter().any(|target| target.trim().is_empty()) {
        return Err(ServerError::new("--target values must not be empty"));
    }
    let unique = targets.iter().collect::<BTreeSet<_>>();
    if unique.len() != targets.len() {
        return Err(ServerError::new("--target values must be unique"));
    }
    Ok(targets)
}

/// Builds the random algorithm and starts the server.
pub(crate) async fn run(args: ServerArgs) -> ServerResult<()> {
    let (state, options) = args.into_runtime()?;
    run_server(state, options).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_targets_build_random_runtime() -> ServerResult<()> {
        let args = ServerArgs::try_parse_from([
            "switchyard-server",
            "--target",
            "model/a",
            "--target",
            "model/b",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--dry-run",
        ])
        .map_err(|error| ServerError::new(error.to_string()))?;

        let (state, options) = args.into_runtime()?;

        assert_eq!(state.served_model().id, DEFAULT_ROUTE_MODEL);
        assert!(state
            .served_model()
            .display_name
            .contains("model/a, model/b"));
        assert!(options.dry_run);
        Ok(())
    }

    #[test]
    fn duplicate_targets_are_rejected() -> ServerResult<()> {
        let args = ServerArgs::try_parse_from([
            "switchyard-server",
            "--target",
            "model/a",
            "--target",
            "model/a",
            "--base-url",
            "http://127.0.0.1:9/v1",
        ])
        .map_err(|error| ServerError::new(error.to_string()))?;

        assert!(args.into_runtime().is_err());
        Ok(())
    }
}
