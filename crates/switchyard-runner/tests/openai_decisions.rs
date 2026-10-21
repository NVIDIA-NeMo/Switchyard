// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use serde_json::json;
use switchyard_llm_client::OpenAiDecisionClient;
use switchyard_protocol::{
    BooleanEstimate, DecisionRequest, DecisionValue, LlmClientError, Probability, Request,
    RoutedDecisionClient, ScoreValue, text_request,
};
use switchyard_runner::{Runner, api_key_variables};
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn openai_decisions_round_trip_and_route_from_toml() -> Result<(), Box<dyn std::error::Error>>
{
    let server = MockServer::start().await;
    let endpoint = format!("{}/v1/decisions", server.uri());
    let client =
        OpenAiDecisionClient::new(endpoint.parse()?, "test-key".into(), Duration::from_secs(2))?;
    let request: DecisionRequest = serde_json::from_value(json!({
        "model": "gpt-6-luna", "context": {"task": "Route this request"},
        "questions": {
            "boolean": {"instructions": "Is it simple?", "kind": {"type": "boolean", "data": {
                "true_description": {"meaning": "Simple"}, "false_description": null
            }}},
            "route": {"instructions": {"task": "Choose"}, "kind": {"type": "choice", "data": {
                "options": [{"id": "advantage", "description": ["A wins"]}, {"id": "no_advantage"}]
            }}},
            "score": {"instructions": "Rate difficulty", "kind": {"type": "score", "data": {
                "levels": ["Easy", {"description": "Hard"}]
            }}}
        }
    }))?;
    let body = json!({
        "model": "gpt-6-luna", "answers": [
            {"type": "predicate", "name": "boolean", "probability": 0.9},
            {"type": "choice", "name": "route", "choice": "no_advantage", "confidence": 0.7,
                "probabilities": [{"value": "advantage", "probability": 0.2},
                    {"value": "no_advantage", "probability": 0.8}]},
            {"type": "score", "name": "score", "score": 0.6, "confidence": 0.3,
                "probabilities": [{"value": 1, "label": "Hard", "probability": 0.6},
                    {"value": 0, "label": "Easy", "probability": 0.4}]}
        ],
        "usage": {"input_tokens": 42, "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 2},
            "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 1}, "total_tokens": 45}
    });
    Mock::given(method("POST")).and(path("/v1/decisions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model": "gpt-6-luna", "input": request.context.to_string(),
            "questions": [
                {"name": "boolean", "type": "predicate", "instructions": json!({
                    "instructions": "Is it simple?", "true": {"meaning": "Simple"}, "false": null
                }).to_string()},
                {"name": "route", "type": "choice", "instructions": "{\"task\":\"Choose\"}",
                    "choices": [{"value": "advantage", "description": "[\"A wins\"]"}, {"value": "no_advantage"}]},
                {"name": "score", "type": "score", "instructions": "Rate difficulty",
                    "levels": [{"label": "Easy"}, {"label": "{\"description\":\"Hard\"}"}]}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body)).expect(1).mount(&server).await;
    let response = client.call(request.clone()).await?;
    assert_eq!(response.model.as_deref(), Some("gpt-6-luna"));
    assert_eq!(
        response.answers["boolean"].value,
        DecisionValue::Boolean(BooleanEstimate::ProbabilityTrue(Probability(0.9)))
    );
    assert_eq!(
        response.answers["score"].value,
        DecisionValue::Score {
            value: ScoreValue(0.6),
            probabilities: Some(vec![Probability(0.4), Probability(0.6)]),
        }
    );
    let DecisionValue::Choice {
        selected,
        probabilities,
    } = &response.answers["route"].value
    else {
        panic!("expected a choice")
    };
    assert_eq!(selected, "no_advantage");
    assert_eq!(
        probabilities.as_ref().unwrap()["advantage"],
        Probability(0.2)
    );
    assert_eq!(
        response.answers["route"].provider_confidence.map(|c| c.0),
        Some(0.7)
    );
    assert_eq!(response.usage.input_tokens, Some(30));
    assert_eq!(response.usage.cached_input_tokens(), Some(10));
    assert_eq!(response.usage.cache_creation_input_tokens(), Some(2));
    assert_eq!(response.usage.output_tokens, Some(3));
    assert_eq!(response.usage.total_tokens, Some(45));
    assert_eq!(response.usage.reasoning_tokens, Some(1));
    server.verify().await;

    for answers in [
        json!([]),
        json!([body["answers"][0], body["answers"][0], body["answers"][2]]),
        json!([body["answers"][0], {"type": "refusal", "name": "route"}, body["answers"][2]]),
        json!([body["answers"][0], {"type": "predicate", "name": "route", "probability": 0.2}, body["answers"][2]]),
    ] {
        server.reset().await;
        let mut invalid = body.clone();
        invalid["answers"] = answers;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(invalid))
            .expect(1)
            .mount(&server)
            .await;
        assert!(matches!(
            client.call(request.clone()).await,
            Err(LlmClientError::ResponseTranslation(_))
        ));
        server.verify().await;
    }
    server.reset().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid test-key"))
        .expect(1)
        .mount(&server)
        .await;
    let Err(LlmClientError::UpstreamHttp {
        status,
        body: error_body,
    }) = client.call(request.clone()).await
    else {
        panic!("expected an upstream error")
    };
    assert_eq!(status.as_u16(), 401);
    assert_eq!(error_body, "invalid [REDACTED]");
    server.verify().await;

    let source = format!(
        r#"
schema_version = 1
[llm_clients.answers]
format = "openai_chat"
base_url = "https://example.test/v1"
[targets]
strong = {{ id = "strong/model", llm_client = "answers" }}
weak = {{ id = "weak/model", llm_client = "answers" }}
[decision_clients.judge]
format = "openai_decisions"
endpoint = "{endpoint}"
api_key_env = "SWITCHYARD_OPENAI_DECISIONS_TEST_KEY"
timeout_ms = 2000
[decision_targets.judge]
id = "gpt-6-luna"
decision_client = "judge"
[routes.classifier]
id = "switchyard/classifier"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "strong"
weak_target = "weak"
fail_open = false
[routes.classifier.decision]
cutoff = 0.4
candidates = {{ a = "strong", b = "weak" }}
evidence = {{ description = "Compare a and b" }}
"#
    );
    assert!(api_key_variables(&source)?.contains("SWITCHYARD_OPENAI_DECISIONS_TEST_KEY"));
    // This unique variable is only read while constructing these clients.
    unsafe {
        std::env::set_var("SWITCHYARD_OPENAI_DECISIONS_TEST_KEY", "test-key");
    }
    let runner = Runner::from_toml(&source);
    let system_one = Runner::from_toml(&source.replace("openai_decisions", "system_one"));
    unsafe {
        std::env::remove_var("SWITCHYARD_OPENAI_DECISIONS_TEST_KEY");
    }
    let runner = runner?;
    system_one?;
    assert_eq!(runner.provider_api_keys(), &["test-key"]);
    let route = runner.route("switchyard/classifier").unwrap();
    for (advantage, expected) in [(0.2, "weak/model"), (0.8, "strong/model")] {
        server.reset().await;
        let mut reply = body.clone();
        reply["answers"] = json!([{
            "type": "choice", "name": "route", "choice": "no_advantage", "confidence": 0.7,
            "probabilities": [{"value": "advantage", "probability": advantage},
                {"value": "no_advantage", "probability": 1.0 - advantage}]
        }]);
        Mock::given(method("POST"))
            .and(path("/v1/decisions"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(reply))
            .expect(1)
            .mount(&server)
            .await;
        let outcome = route
            .decide(Request {
                llm_request: text_request(None, "Route this request"),
                ..Request::default()
            })
            .await?;
        assert_eq!(outcome.selected_model_id()?, expected);
        server.verify().await;
    }
    Ok(())
}
