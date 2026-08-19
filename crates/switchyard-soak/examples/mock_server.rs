// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-aware local backend used by `scripts/run_local_soak_test.py`.

use std::collections::HashMap;
use std::convert::Infallible;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use clap::Parser;
use futures_util::{StreamExt, stream};
use parking_lot::Mutex;
use serde_json::{Value, json};

#[derive(Parser)]
#[command(
    name = "switchyard-soak-mock",
    about = "Start the request-aware backend for the local Switchyard scenario test",
    after_long_help = "Example:\n  cargo run --release -p switchyard-soak --example switchyard-soak-mock -- --port 8100 --latency-ms 40",
    version
)]
struct Args {
    /// Local TCP port used by the backend.
    #[arg(long, default_value_t = 8100)]
    port: u16,

    /// Artificial delay, in milliseconds, added to ordinary responses.
    #[arg(long, default_value_t = 40)]
    latency_ms: u64,

    /// Artificial delay between streamed output tokens, in milliseconds.
    #[arg(long, default_value_t = 1)]
    token_latency_ms: u64,
}

#[derive(Clone)]
struct BackendState {
    latency: Duration,
    token_latency: Duration,
    attempts: Arc<Mutex<HashMap<String, u64>>>,
}

// AIPerf requires this OpenAI field; a fixed value keeps local runs deterministic.
const FIXED_CREATED_AT: u64 = 1_700_000_000;

fn scenario_marker(body: &Value) -> Option<&str> {
    body.get("messages")?
        .as_array()?
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .find_map(|content| {
            let marker = content.split_once("[scenario:")?.1;
            marker.split_once(']').map(|(name, _rest)| name)
        })
}

fn classifier_content(marker: Option<&str>) -> &'static str {
    if marker == Some("classifier_invalid") {
        return "not a JSON verdict";
    }
    if marker == Some("classifier_hard") {
        return r#"{"crux":"distributed diagnosis","primary_rule":"LIM-1","capability_boundary":"unsupported","p_solve":0.1}"#;
    }
    r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":1.0}"#
}

fn completion(model: &str, content: &str) -> Value {
    json!({
        "id": "chatcmpl-switchyard-soak",
        "object": "chat.completion",
        "created": FIXED_CREATED_AT,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 32, "completion_tokens": 2, "total_tokens": 34}
    })
}

fn requested_output_tokens(body: &Value) -> u64 {
    body.get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(2)
        .clamp(1, 4_096)
}

fn stream(
    model: &str,
    truncated: bool,
    completion_tokens: u64,
    token_latency: Duration,
) -> Response {
    let mut events = Vec::with_capacity(completion_tokens as usize + 2);
    for index in 0..completion_tokens {
        events.push(
            json!({
                "id": "chatcmpl-switchyard-soak",
                "object": "chat.completion.chunk",
                "created": FIXED_CREATED_AT,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": (index == 0).then_some("assistant"),
                        "content": if index == 0 { "token" } else { " token" }
                    },
                    "finish_reason": null
                }]
            })
            .to_string(),
        );
        if truncated {
            break;
        }
    }
    if !truncated {
        events.push(
            json!({
                "id": "chatcmpl-switchyard-soak",
                "object": "chat.completion.chunk",
                "created": FIXED_CREATED_AT,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 32,
                    "completion_tokens": completion_tokens,
                    "total_tokens": 32 + completion_tokens
                }
            })
            .to_string(),
        );
        events.push("[DONE]".to_string());
    }
    let events =
        stream::iter(events.into_iter().enumerate()).then(move |(index, data)| async move {
            if index > 0 {
                tokio::time::sleep(token_latency).await;
            }
            Ok::<Event, Infallible>(Event::default().data(data))
        });
    Sse::new(events).into_response()
}

async fn chat(State(state): State<BackendState>, Json(body): Json<Value>) -> Response {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let marker = scenario_marker(&body);
    if marker == Some("client_cancellation") {
        tokio::time::sleep(Duration::from_secs(2)).await;
    } else {
        tokio::time::sleep(state.latency).await;
    }

    if model == "mock/classifier" {
        return Json(completion(model, classifier_content(marker))).into_response();
    }
    if model == "mock/weak" && marker == Some("context_overflow") {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "code": "context_length_exceeded",
                    "message": "the weak target context window is too small"
                }
            })),
        )
            .into_response();
    }

    for (name, status) in [
        ("upstream_429", StatusCode::TOO_MANY_REQUESTS),
        ("upstream_500", StatusCode::INTERNAL_SERVER_ERROR),
    ] {
        if marker == Some(name) {
            let key = format!("{model}:{name}");
            let attempt = {
                let mut attempts = state.attempts.lock();
                let attempt = attempts.entry(key).or_default();
                *attempt += 1;
                *attempt
            };
            if attempt <= 2 {
                return (
                    status,
                    Json(
                        json!({"error": {"message": format!("injected {name} attempt {attempt}")}}),
                    ),
                )
                    .into_response();
            }
        }
    }

    if body.get("stream") == Some(&Value::Bool(true)) {
        return stream(
            model,
            marker == Some("truncated_stream"),
            requested_output_tokens(&body),
            state.token_latency,
        );
    }
    Json(completion(model, "OK")).into_response()
}

async fn run(args: Args) -> Result<(), String> {
    if args.port == 0 {
        return Err("--port must be greater than zero".to_string());
    }
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", args.port))
        .await
        .map_err(|error| error.to_string())?;
    let address = listener.local_addr().map_err(|error| error.to_string())?;
    let state = BackendState {
        latency: Duration::from_millis(args.latency_ms),
        token_latency: Duration::from_millis(args.token_latency_ms),
        attempts: Arc::new(Mutex::new(HashMap::new())),
    };
    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route(
            "/reset",
            post(|State(state): State<BackendState>| async move {
                state.attempts.lock().clear();
                Json(json!({"status": "reset"}))
            }),
        )
        .route("/v1/chat/completions", post(chat))
        .with_state(state);
    println!("Scenario backend is ready at http://{address}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                eprintln!("could not register the shutdown signal: {error}");
                std::future::pending::<()>().await;
            }
        })
        .await
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::requested_output_tokens;

    #[test]
    fn output_tokens_follow_openai_limits_and_stay_bounded() {
        assert_eq!(requested_output_tokens(&json!({"max_tokens": 512})), 512);
        assert_eq!(
            requested_output_tokens(&json!({"max_completion_tokens": 8_192})),
            4_096
        );
        assert_eq!(requested_output_tokens(&json!({})), 2);
    }
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
