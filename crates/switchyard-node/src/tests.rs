// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host model replies and cleanup for the shared native run.

use serde_json::json;
use switchyard_libsy::{
    DeescalationConfig, EscalationJudgeConfig, LlmClassifierConfig, LlmTaskClassifier,
};
use switchyard_protocol::text_request;

use super::*;

fn run(state: Option<&str>) -> NativeRun {
    let algorithm = LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
        contract: Default::default(),
        config: EscalationJudgeConfig {
            confirmations: 1,
            deescalation: Some(DeescalationConfig {
                strong_min_calls: 1,
                strong_max_calls: None,
                confirmations: 1,
                weak_cooldown_calls: 0,
            }),
            ..Default::default()
        },
        max_output_tokens: 256,
    })
    .unwrap();
    let models = RuntimeModels::new(
        [
            (Category::Efficient, vec!["fast".into()]),
            (Category::Capable, vec!["capable".into()]),
            (Category::Judge, vec!["judge".into()]),
            (Category::Any, vec!["capable".into(), "fast".into()]),
        ]
        .into(),
    );
    NativeRun::new(
        Arc::new(algorithm),
        Request {
            llm_request: text_request(Some("auto".into()), "Fix the test."),
            ..Default::default()
        },
        Arc::new(models),
        state,
        true,
    )
    .unwrap()
}

// The served candidate, not the provider's model name, determines fallback routing.
#[tokio::test]
async fn model_reply_preserves_fallback_answer_and_full_outcome() -> napi::Result<()> {
    let run = run(Some(
        r#"{"escalation_strong_calls":{"count":1},"host_marker":{"string":"kept"}}"#,
    ));
    let call = run.next(None, None).await?;
    assert_eq!(call.kind, "model");
    assert_eq!(call.models.unwrap(), ["capable", "fast"]);
    let request: Value = serde_json::from_str(&call.request.unwrap()).unwrap();
    assert_eq!(request["model"], "capable");
    let reply = json!({
        "servedModel": "fast",
        "response": {
            "id": "answer-1", "model": "provider/reported-model",
            "outputs": [{"role": "assistant", "content": [{"type": "text", "text": "fixed"}]}],
            "usage": {"output_tokens": 2}
        }
    });
    let done = run.next(Some(reply.to_string()), None).await?;
    assert_eq!(done.kind, "done", "the fallback answer needs no judge call");
    let outcome: Value = serde_json::from_str(&done.outcome.unwrap()).unwrap();
    assert_eq!(outcome["selected_model_ids"], json!(["fast"]));
    assert_eq!(outcome["request"]["model"], "fast");
    assert_eq!(outcome["response"]["model"], "provider/reported-model");
    assert_eq!(
        outcome["response"]["outputs"][0]["content"],
        reply["response"]["outputs"][0]["content"]
    );
    assert_eq!(outcome["response"]["usage"]["output_tokens"], 2);
    assert_eq!(
        outcome["metadata"]["evidence"],
        json!({"source": "fallback"})
    );
    assert_eq!(outcome["state"]["host_marker"], json!({"string": "kept"}));
    Ok(())
}

// Invalid host replies and provider failures must not produce a successful route.
#[tokio::test]
async fn rejects_unknown_candidates_and_host_failures() -> napi::Result<()> {
    for (response, failure, code) in [
        (
            Some(json!({"response": {}, "servedModel": "unknown"}).to_string()),
            None,
            "ERR_INVALID_REQUEST",
        ),
        (
            None,
            Some("provider unavailable".to_string()),
            "ERR_ROUTING",
        ),
    ] {
        let run = run(None);
        assert_eq!(run.next(None, None).await?.kind, "model");
        let error = run.next(response, failure).await.err().unwrap();
        assert!(error.reason.starts_with(code), "{error}");
    }
    Ok(())
}

// Closing a pending run rejects further replies.
#[tokio::test]
async fn closes_a_pending_model_call() -> napi::Result<()> {
    let run = run(None);
    assert_eq!(run.next(None, None).await?.kind, "model");
    run.close();
    let error = run.next(None, None).await.err().unwrap();
    assert!(error.reason.starts_with("ERR_INVALID_REQUEST"));
    Ok(())
}
