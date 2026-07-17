// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI: parse flags, build the client from env credentials, and serve.

use std::collections::BTreeMap;
use std::env;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use clap::Parser;
use libsy::{Algorithm, LlmTarget, LlmTargetSet, RandomAlgo, RoutedLlmClient};
use switchyard_llm_client::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};
use switchyard_translation::WireFormat;
use tokio::net::TcpListener;

use crate::{build_router, ProxyState, SERVED_MODEL};

const DEFAULT_HOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
const DEFAULT_PORT: u16 = 4000;
const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";

/// Command-line arguments for the server binary.
#[derive(Debug, Parser)]
#[command(
    name = "libsy-server",
    about = "Minimal LLM API server over switchyard-llm-client",
    version
)]
pub struct Args {
    /// Upstream base URL the server forwards to (e.g. https://api.openai.com/v1).
    #[arg(long)]
    pub base_url: String,

    /// Weak-tier upstream model id — one of the two random-routing targets.
    #[arg(long)]
    pub weak: String,

    /// Wire format the weak tier's upstream speaks (openai-chat, openai-responses,
    /// anthropic-messages). Its provider key selects the credential.
    #[arg(long, value_parser = parse_wire_format)]
    pub weak_format: WireFormat,

    /// Strong-tier upstream model id — one of the two random-routing targets.
    #[arg(long)]
    pub strong: String,

    /// Wire format the strong tier's upstream speaks (openai-chat, openai-responses,
    /// anthropic-messages). Its provider key selects the credential.
    #[arg(long, value_parser = parse_wire_format)]
    pub strong_format: WireFormat,

    /// Log each request's routing decision (the selected tier) to stderr.
    #[arg(long)]
    pub log_routing: bool,

    /// Host address to bind.
    #[arg(long, default_value_t = DEFAULT_HOST)]
    pub host: IpAddr,

    /// Port to bind.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,
}

impl Args {
    /// Builds server state (and the bind address) from the args and env keys.
    ///
    /// Each tier (`weak`, `strong`) is served by a backend of its own
    /// `--*-format`, at the shared `--base-url`, authenticated with that format's
    /// provider key — OpenAI formats from `OPENAI_API_KEY`, Anthropic from
    /// `ANTHROPIC_API_KEY`. The tiers become the [`RandomAlgo`]'s routing targets:
    /// each request picks one uniformly at random. Any inbound format is decoded
    /// to the neutral IR, re-encoded to the chosen tier's format for the upstream
    /// call, and translated back to the inbound format for the client. Errors if a
    /// tier's provider key is unset.
    fn build(self) -> Result<(ProxyState, SocketAddr), String> {
        let weak_backend = backend_for_format(self.weak_format, &self.base_url)?;
        let strong_backend = backend_for_format(self.strong_format, &self.base_url)?;

        // One config per tier — its own format's backend, keyed by the tier's
        // model id. The client resolves a target's name to that config.
        let client = Arc::new(
            TranslatingLlmClient::new(&[
                ModelConfig::new(self.weak.clone(), weak_backend, None),
                ModelConfig::new(self.strong.clone(), strong_backend, None),
            ])
            .map_err(|error| error.to_string())?,
        );

        // The two tiers as routing targets, sharing the client that serves them.
        let targets = LlmTargetSet::new(vec![
            target(&self.weak, client.clone()),
            target(&self.strong, client.clone()),
        ]);
        let algorithm: Arc<dyn Algorithm> = Arc::new(RandomAlgo::new(targets));

        let addr = SocketAddr::new(self.host, self.port);
        Ok((ProxyState::new(algorithm, self.log_routing), addr))
    }
}

/// Parses args, binds the listener, and serves until Ctrl-C.
pub async fn run(args: Args) -> Result<(), String> {
    let base_url = args.base_url.clone();
    let weak = (args.weak.clone(), args.weak_format);
    let strong = (args.strong.clone(), args.strong_format);
    let (state, addr) = args.build()?;

    let listener = TcpListener::bind(addr)
        .await
        .map_err(|error| format!("failed to bind {addr}: {error}"))?;
    let bound = listener.local_addr().map_err(|error| error.to_string())?;
    eprintln!("{}", banner(bound, &base_url, &weak, &strong));

    axum::serve(listener, build_router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| error.to_string())
}

// Parses a `--*-format` flag into a built-in wire format, accepting the canonical
// ids plus common hyphenated aliases.
fn parse_wire_format(value: &str) -> Result<WireFormat, String> {
    match value.to_ascii_lowercase().replace('-', "_").as_str() {
        "openai_chat" | "openai_chat_completions" | "chat" => Ok(WireFormat::OpenAiChat),
        "openai_responses" | "responses" => Ok(WireFormat::OpenAiResponses),
        "anthropic_messages" | "anthropic" | "messages" => Ok(WireFormat::AnthropicMessages),
        _ => Err(format!(
            "unknown wire format '{value}': expected one of \
             openai-chat, openai-responses, anthropic-messages"
        )),
    }
}

// Builds the backend serving `format` at `base_url`, drawing the API key from the
// format's provider env var. Errors if that key is unset.
fn backend_for_format(format: WireFormat, base_url: &str) -> Result<Backend, String> {
    let env_name = match format {
        WireFormat::OpenAiChat | WireFormat::OpenAiResponses => OPENAI_API_KEY_ENV,
        WireFormat::AnthropicMessages => ANTHROPIC_API_KEY_ENV,
    };
    let api_key = env_key(env_name)
        .ok_or_else(|| format!("no credential for {format} tier: set {env_name}"))?;
    let config = HttpBackendConfig {
        base_url: base_url.to_string(),
        api_key: Some(api_key),
        extra_headers: BTreeMap::new(),
    };
    Ok(match format {
        WireFormat::OpenAiChat => Backend::OpenAiChat(config),
        WireFormat::OpenAiResponses => Backend::OpenAiResponses(config),
        WireFormat::AnthropicMessages => Backend::Anthropic(config),
    })
}

// Builds a routing target named `model` that serves its calls through `client`.
fn target(model: &str, client: Arc<TranslatingLlmClient>) -> LlmTarget {
    LlmTarget {
        semantic_name: model.to_string(),
        llm_client: Some(client as Arc<dyn RoutedLlmClient>),
    }
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
    weak: &(String, WireFormat),
    strong: &(String, WireFormat),
) -> String {
    format!(
        "libsy-server\n  \
         listening:     http://{addr}\n  \
         upstream:      {base_url}\n  \
         random routing: weak {} ({}), strong {} ({})\n  \
         serving model: {SERVED_MODEL}\n  \
         endpoints:     GET /health, GET /v1/models, POST /v1/chat/completions, \
         POST /v1/messages, POST /v1/responses\n  \
         stop:          Ctrl-C",
        weak.0, weak.1, strong.0, strong.1
    )
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        eprintln!("ctrl-c handler unavailable; running without shutdown trigger: {error}");
        std::future::pending::<()>().await;
    }
}
