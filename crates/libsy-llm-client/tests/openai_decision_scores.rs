// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Score bounds depend on the requested rubric, not the probability-weighted average.

use std::time::Duration;

use serde_json::json;
use switchyard_llm_client::{LlmClientError, OpenAiDecisionClient};
use switchyard_protocol::{
    DecisionRequest, DecisionValue, Probability, RoutedDecisionClient, ScoreValue,
};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn scores_stay_within_each_requested_rubric() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    let client = OpenAiDecisionClient::new(
        server.uri().parse()?,
        "test-key".into(),
        Duration::from_secs(2),
    )?;
    for (level_count, score, valid) in [
        (3, 0.0, true),
        (3, 1.5, true),
        (3, 2.0, true),
        (3, -0.1, false),
        (3, 2.0001, false),
        (3, 3.0, false),
        (2, 1.0, true),
        (2, 1.5, false),
        (4, 3.0, true),
        (0, 0.0, false),
    ] {
        let levels: Vec<_> = (0..level_count)
            .map(|index| json!(format!("level-{index}")))
            .collect();
        let probabilities: Vec<_> = (0..level_count)
            .rev()
            .map(|index| json!({"value": index, "probability": 1.0 / level_count as f64}))
            .collect();
        let request: DecisionRequest = serde_json::from_value(json!({
            "model": "judge", "input": "evaluate",
            "questions": {"q": {"instructions": "Rate difficulty", "kind": {
                "type": "score", "data": {"levels": levels}
            }}}
        }))?;
        server.reset().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "judge",
                "answers": [{"name": "q", "type": "score", "score": score,
                    "confidence": 0.73, "probabilities": probabilities}],
                "usage": {"input_tokens": 42, "output_tokens": 3, "total_tokens": 45,
                    "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 2},
                    "output_tokens_details": {"reasoning_tokens": 1}}
            })))
            .expect(1)
            .mount(&server)
            .await;
        let result = client.call(request).await;
        if valid {
            let response = result?;
            assert_eq!(
                response.answers["q"].value,
                DecisionValue::Score {
                    value: ScoreValue(score),
                    probabilities: Some(vec![Probability(1.0 / level_count as f64); level_count]),
                }
            );
            assert_eq!(
                response.answers["q"].provider_confidence.map(|p| p.0),
                Some(0.73)
            );
            assert_eq!(response.usage.input_tokens, Some(30));
        } else {
            assert!(
                matches!(result, Err(LlmClientError::ResponseTranslation(_))),
                "{level_count} levels, score {score}: {result:?}"
            );
        }
        server.verify().await;
    }
    Ok(())
}
