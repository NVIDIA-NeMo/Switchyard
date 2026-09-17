// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live smoke test against the real TypeSafe `/v1/systemone` API.
//!
//! This is NOT part of the crate's automated test suite (those use `wiremock` and
//! never touch the network). Run it by hand, from an environment that can reach
//! `api.typesafe.ai`, with a real API key:
//!
//! ```text
//! export TYPESAFE_API_KEY=sk-...
//! cargo run -p switchyard-typesafe-client --example live_check
//! ```
//!
//! It sends one real classification request (the same "capable vs. efficient"
//! shape a `type_safe_classifier` route asks in production) and prints the
//! provider's verdict plus token usage. A non-zero exit means the call failed;
//! the printed error is TypeSafe's own response detail (never the API key).

use switchyard_libsy::{TypeSafeClassifierInput, TypeSafeOption, TypeSafeProvider};
use switchyard_typesafe_client::TypeSafeHttpClient;

#[tokio::main]
async fn main() {
    let client = match TypeSafeHttpClient::from_env("TYPESAFE_API_KEY") {
        Ok(client) => client,
        Err(err) => {
            eprintln!("could not build client: {err}");
            std::process::exit(1);
        }
    };

    let options = vec![
        TypeSafeOption::new(
            "capable",
            "The request needs a strong, expensive model: multi-step reasoning, \
             ambiguous or open-ended instructions, or high-stakes correctness.",
        ),
        TypeSafeOption::new(
            "efficient",
            "The request is simple and well-specified: a short factual question, \
             a small formatting or lookup task, or routine chit-chat.",
        ),
    ];

    let input = TypeSafeClassifierInput {
        question: "Which model tier does this conversation need?".to_string(),
        context: "user: What's 2+2?\nassistant:".to_string(),
    };

    println!("Calling TypeSafe (model = jev-latest) ...");
    match client.classify(input, &options).await {
        Ok(verdict) => {
            println!("OK");
            println!("  label:      {}", verdict.label);
            println!("  confidence: {:.3}", verdict.confidence);
        }
        Err(err) => {
            eprintln!("classify() failed: {err}");
            std::process::exit(1);
        }
    }
}
