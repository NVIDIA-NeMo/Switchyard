// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live flat-document posting test. Ignored by default (needs a reachable
//! posting endpoint, typically on the NVIDIA internal network / VPN). Run:
//!
//!   SWITCHYARD_INTAKE_TARGET_URL=<full posting url> \
//!     cargo test -p switchyard-components --test flat_target_live -- --ignored
//!
//! Posting is fail-open, so this drives records through the real sink; verify
//! they landed by querying the target store. No-ops when the env var is unset.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use switchyard_components::intake::to_flat_document;
use switchyard_components::{
    HttpIntakeSink, IntakeFormat, IntakeSink, IntakeSinkConfig, IntakeTarget,
};
use switchyard_core::Result;

// A representative chat-completions intake payload, the input the production
// builder produces before flattening.
fn sample_chat_payload(
    session: &str,
    served: &str,
    routed_to: &str,
    prompt: i64,
    completion: i64,
    cost: f64,
) -> Value {
    json!({
        "request": {
            "model": "openai/openai/gpt-5.2",
            "switchyard": {
                "user_id": "0badf00d",
                "inbound_format": "openai_chat",
                "latency_ms": 1840,
                "routing": {"router_type": "random", "routed_to": routed_to}
            }
        },
        "response": {
            "model": served,
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion
            }
        },
        "session_id": session,
        "cost_usd": cost,
        "cost_input_usd": cost * 0.7,
        "cost_output_usd": cost * 0.3,
        "provider": "switchyard"
    })
}

#[tokio::test]
#[ignore = "live: posts to a real flat-document endpoint; requires VPN. run with --ignored"]
async fn posts_flat_documents_to_live_target() -> Result<()> {
    // No endpoint configured means nothing to exercise; keep the test a no-op.
    let Ok(url) = std::env::var("SWITCHYARD_INTAKE_TARGET_URL") else {
        return Ok(());
    };
    let sink = HttpIntakeSink::new(IntakeSinkConfig {
        target: Some(IntakeTarget {
            url,
            format: IntakeFormat::FlatDocument,
            authenticated: false,
        }),
        ..IntakeSinkConfig::default()
    })?;

    let samples = [
        (
            "claude-live-0001",
            "nvidia/moonshotai/kimi-k2.5",
            "weak",
            25_000,
            6_580,
            0.0142,
        ),
        (
            "claude-live-0002",
            "openai/openai/gpt-5.2",
            "strong",
            41_000,
            9_100,
            0.2300,
        ),
        (
            "codex-live-0003",
            "nvidia/nvidia/nemotron-3-super-v3",
            "weak",
            18_000,
            3_200,
            0.0061,
        ),
    ];
    // The target store may reject backdated documents per retention policy, so
    // stamp now.
    let base_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or_default();
    for (index, (session, served, routed_to, prompt, completion, cost)) in
        samples.iter().enumerate()
    {
        let chat = sample_chat_payload(session, served, routed_to, *prompt, *completion, *cost);
        let doc = to_flat_document(&chat, Some(base_ts + index as i64), None);
        sink.enqueue(doc).await?;
    }
    sink.shutdown().await?;
    Ok(())
}
