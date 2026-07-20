// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI entrypoint for running the libsy server with random routing.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use clap::{Parser, ValueEnum};
use libsy::algorithms::Random;
use libsy::{Algorithm, LlmTarget, LlmTargetSet, RoutedLlmClient};
use switchyard_llm_client::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};
use switchyard_server::{
    run_server, ServedModel, ServerError, ServerResult, ServerRunOptions, ServerState, TlsOptions,
    DEFAULT_LISTEN_BACKLOG,
};

const DEFAULT_HOST: IpAddr = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
const DEFAULT_PORT: u16 = 4000;

#[derive(Clone, Debug)]
struct RandomRouteSpec {
    model: String,
    targets: Vec<String>,
}

impl FromStr for RandomRouteSpec {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (model, target_list) = value
            .split_once('=')
            .ok_or_else(|| "route must use MODEL=TARGET[,TARGET...]".to_string())?;
        let model = model.trim();
        if model.is_empty() {
            return Err("route model must not be empty".to_string());
        }
        let targets = target_list
            .split(',')
            .map(str::trim)
            .map(str::to_string)
            .collect::<Vec<_>>();
        if targets.is_empty() || targets.iter().any(String::is_empty) {
            return Err(format!("route {model} must contain non-empty targets"));
        }
        if targets.iter().collect::<BTreeSet<_>>().len() != targets.len() {
            return Err(format!("route {model} must contain unique targets"));
        }
        Ok(Self {
            model: model.to_string(),
            targets,
        })
    }
}

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
    /// Random route as MODEL=TARGET[,TARGET...]. Repeat to serve multiple routes.
    #[arg(
        long = "route",
        required = true,
        value_name = "MODEL=TARGET[,TARGET...]"
    )]
    routes: Vec<RandomRouteSpec>,

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
        if self.base_url.trim().is_empty() {
            return Err(ServerError::new("--base-url must not be empty"));
        }

        let backend = self.upstream_format.backend(HttpBackendConfig {
            base_url: self.base_url,
            api_key: self.api_key,
            extra_headers: BTreeMap::new(),
        });
        let target_models = self
            .routes
            .iter()
            .flat_map(|route| route.targets.iter())
            .collect::<BTreeSet<_>>();
        let model_configs = target_models
            .into_iter()
            .map(|model| ModelConfig::new(model.as_str(), backend.clone(), None))
            .collect::<Vec<_>>();
        let client: Arc<dyn RoutedLlmClient> = Arc::new(
            TranslatingLlmClient::new(&model_configs)
                .map_err(|error| ServerError::new(error.to_string()))?,
        );
        let routes = self.routes.into_iter().map(|route| {
            let target_set = LlmTargetSet::new(
                route
                    .targets
                    .iter()
                    .map(|model| LlmTarget {
                        semantic_name: model.clone(),
                        llm_client: Some(Arc::clone(&client)),
                    })
                    .collect(),
            );
            let algorithm: Arc<dyn Algorithm> = Arc::new(Random::new(target_set));
            let model = ServedModel {
                id: route.model,
                display_name: format!("uniform random routing across {}", route.targets.join(", ")),
            };
            (model, algorithm)
        });
        let state = ServerState::new(routes)?;

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

/// Builds the random algorithm and starts the server.
pub(crate) async fn run(args: ServerArgs) -> ServerResult<()> {
    let (state, options) = args.into_runtime()?;
    run_server(state, options).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_routes_build_independent_random_algorithms() -> ServerResult<()> {
        let args = ServerArgs::try_parse_from([
            "switchyard-server",
            "--route",
            "switchyard/general=model/a,model/b",
            "--route",
            "switchyard/coding=model/c,model/d",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--dry-run",
        ])
        .map_err(|error| ServerError::new(error.to_string()))?;

        let (state, options) = args.into_runtime()?;

        assert_eq!(
            state
                .served_models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["switchyard/coding", "switchyard/general"]
        );
        assert!(options.dry_run);
        Ok(())
    }

    #[test]
    fn duplicate_targets_are_rejected() -> ServerResult<()> {
        let result = ServerArgs::try_parse_from([
            "switchyard-server",
            "--route",
            "switchyard/general=model/a,model/a",
            "--base-url",
            "http://127.0.0.1:9/v1",
        ]);

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn duplicate_route_models_are_rejected() -> ServerResult<()> {
        let args = ServerArgs::try_parse_from([
            "switchyard-server",
            "--route",
            "switchyard/general=model/a",
            "--route",
            "switchyard/general=model/b",
            "--base-url",
            "http://127.0.0.1:9/v1",
        ])
        .map_err(|error| ServerError::new(error.to_string()))?;

        assert!(args.into_runtime().is_err());
        Ok(())
    }
}
