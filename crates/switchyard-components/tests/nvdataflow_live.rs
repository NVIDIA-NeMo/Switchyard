// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live NVDataflow sandbox test. Ignored by default (needs the NVIDIA internal
//! network / VPN). Run explicitly:
//!
//!   cargo test -p switchyard-components --test nvdataflow_live -- --ignored
//!
//! Override the project with SWITCHYARD_NVDATAFLOW_PROJECT. Posting is
//! fail-open, so this drives records through the real sink; verify they landed
//! by querying df-<project>-<YYYYMM>.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use switchyard_components::intake::to_nvdataflow_document;
use switchyard_components::{HttpIntakeSink, IntakeSink, IntakeSinkConfig};
use switchyard_core::Result;

// A representative chat-completions intake payload, the input the production
// builder produces before NVDataflow flattening.
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
#[ignore = "live: posts to the NVDataflow sandbox; requires VPN. run with --ignored"]
async fn posts_flat_documents_to_nvdataflow_sandbox() -> Result<()> {
    let project = std::env::var("SWITCHYARD_NVDATAFLOW_PROJECT")
        .unwrap_or_else(|_| "sandbox-switchyard".to_string());
    let sink = HttpIntakeSink::new(IntakeSinkConfig {
        nvdataflow_project: Some(project),
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
    // NVDataflow rejects backdated documents per retention policy, so stamp now.
    let base_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or_default();
    for (index, (session, served, routed_to, prompt, completion, cost)) in
        samples.iter().enumerate()
    {
        let chat = sample_chat_payload(session, served, routed_to, *prompt, *completion, *cost);
        let doc = to_nvdataflow_document(&chat, Some(base_ts + index as i64), None);
        sink.enqueue(doc).await?;
    }
    sink.shutdown().await?;
    Ok(())
}
