// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI: parse flags, build the client from env credentials, and serve.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use clap::{Parser, ValueEnum};
use libsy_llm_client::{Backend, HttpBackendConfig, LlmModelClient};
use switchyard_translation::WireFormat;
use tokio::net::TcpListener;

use crate::{build_router, ProxyState, SERVED_MODEL};

const DEFAULT_HOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
const DEFAULT_PORT: u16 = 4000;
const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";

/// Upstream wire format used as the fallback when an inbound format has no backend.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum UpstreamFormat {
    /// OpenAI Chat Completions.
    OpenaiChat,
    /// OpenAI Responses.
    OpenaiResponses,
    /// Anthropic Messages.
    Anthropic,
}

impl From<UpstreamFormat> for WireFormat {
    fn from(format: UpstreamFormat) -> Self {
        match format {
            UpstreamFormat::OpenaiChat => WireFormat::OpenAiChat,
            UpstreamFormat::OpenaiResponses => WireFormat::OpenAiResponses,
            UpstreamFormat::Anthropic => WireFormat::AnthropicMessages,
        }
    }
}

/// Command-line arguments for the proxy binary.
#[derive(Debug, Parser)]
#[command(
    name = "libsy-proxy",
    about = "Minimal LLM API proxy over libsy-llm-client",
    version
)]
pub struct Args {
    /// Upstream base URL the proxy forwards to (e.g. https://api.openai.com/v1).
    #[arg(long)]
    pub base_url: String,

    /// Upstream model id sent on every call.
    #[arg(long)]
    pub model_name: String,

    /// Fallback upstream format when the inbound format has no backend.
    #[arg(long, value_enum)]
    pub upstream_format: Option<UpstreamFormat>,

    /// Host address to bind.
    #[arg(long, default_value_t = DEFAULT_HOST)]
    pub host: IpAddr,

    /// Port to bind.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,
}

impl Args {
    /// Builds proxy state (and the bind address) from the args and env keys.
    ///
    /// A backend is configured for each wire format whose provider key is set:
    /// OpenAI Chat + Responses from `OPENAI_API_KEY`, Anthropic from
    /// `ANTHROPIC_API_KEY`. Errors if neither key is present.
    fn build(self) -> Result<(ProxyState, SocketAddr), String> {
        let openai_key = env_key(OPENAI_API_KEY_ENV);
        let anthropic_key = env_key(ANTHROPIC_API_KEY_ENV);

        let mut backends: HashMap<WireFormat, Backend> = HashMap::new();
        if let Some(key) = &openai_key {
            backends.insert(
                WireFormat::OpenAiChat,
                Backend::OpenAiChat(self.config(key)),
            );
            backends.insert(
                WireFormat::OpenAiResponses,
                Backend::OpenAiResponses(self.config(key)),
            );
        }
        if let Some(key) = &anthropic_key {
            backends.insert(
                WireFormat::AnthropicMessages,
                Backend::Anthropic(self.config(key)),
            );
        }
        if backends.is_empty() {
            return Err(format!(
                "no upstream credentials: set {OPENAI_API_KEY_ENV} and/or {ANTHROPIC_API_KEY_ENV}"
            ));
        }

        let available: HashSet<WireFormat> = backends.keys().copied().collect();
        let map = HashMap::from([(self.model_name.clone(), backends)]);
        let client = LlmModelClient::new(map).map_err(|error| error.to_string())?;
        let addr = SocketAddr::new(self.host, self.port);
        let state = ProxyState::new(
            client,
            self.model_name.as_str(),
            available,
            self.upstream_format.map(WireFormat::from),
        );
        Ok((state, addr))
    }

    fn config(&self, api_key: &str) -> HttpBackendConfig {
        HttpBackendConfig {
            base_url: self.base_url.clone(),
            api_key: Some(api_key.to_string()),
            extra_headers: BTreeMap::new(),
        }
    }
}

/// Parses args, binds the listener, and serves until Ctrl-C.
pub async fn run(args: Args) -> Result<(), String> {
    let base_url = args.base_url.clone();
    let model_name = args.model_name.clone();
    let (state, addr) = args.build()?;
    let available = state.available.clone();

    let listener = TcpListener::bind(addr)
        .await
        .map_err(|error| format!("failed to bind {addr}: {error}"))?;
    let bound = listener
        .local_addr()
        .map_err(|error| error.to_string())?;
    eprintln!("{}", banner(bound, &base_url, &model_name, &available));

    axum::serve(listener, build_router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| error.to_string())
}

fn env_key(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn banner(
    addr: SocketAddr,
    base_url: &str,
    model_name: &str,
    available: &HashSet<WireFormat>,
) -> String {
    let mut formats: Vec<String> = available.iter().map(|format| format.to_string()).collect();
    formats.sort();
    format!(
        "libsy-proxy\n  \
         listening:     http://{addr}\n  \
         upstream:      {base_url} (model {model_name})\n  \
         upstream fmts: {}\n  \
         serving model: {SERVED_MODEL}\n  \
         endpoints:     GET /health, GET /v1/models, POST /v1/chat/completions, \
         POST /v1/messages, POST /v1/responses\n  \
         stop:          Ctrl-C",
        formats.join(", ")
    )
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        eprintln!("ctrl-c handler unavailable; running without shutdown trigger: {error}");
        std::future::pending::<()>().await;
    }
}
