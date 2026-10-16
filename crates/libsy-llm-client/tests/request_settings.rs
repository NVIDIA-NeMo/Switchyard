// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request preparation without an HTTP backend.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::json;
use switchyard_llm_client::RequestSettings;
use switchyard_protocol::WireFormat;
use switchyard_translation::{decode_request, encode_request};

// Host clients get the native preparation order without changing the source request.
#[test]
fn prepares_target_body_without_transport() -> Result<(), Box<dyn std::error::Error>> {
    let source = json!({
        "model": "route",
        "input": "hello",
        "previous_response_id": "resp_previous",
        "max_output_tokens": 7,
        "service_tier": "auto",
        "reasoning": {"effort": "low", "summary": "auto"}
    });
    let request = decode_request(WireFormat::OpenAiResponses, &source)?;
    let settings = RequestSettings {
        wire_format: WireFormat::OpenAiResponses,
        extra_body: BTreeMap::from([
            ("max_output_tokens".into(), json!(99)),
            ("service_tier".into(), json!("priority")),
        ]),
        omit_body_fields: BTreeSet::from(["service_tier".into()]),
        reasoning_effort: Some("high".into()),
    };

    let body = settings.prepare_request_body(&request, &"target".into())?;
    assert_eq!(body["model"], "target");
    assert_eq!(body["max_output_tokens"], 7);
    assert_eq!(body["service_tier"], "priority");
    assert_eq!(
        body["reasoning"],
        json!({"effort": "high", "summary": "auto"})
    );
    assert_eq!(body["previous_response_id"], "resp_previous");
    assert_eq!(
        encode_request(&request, WireFormat::OpenAiResponses)?,
        source
    );
    Ok(())
}
