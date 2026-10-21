// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use serde_json::{Value, json};
use switchyard_protocol::{
    DecisionAnswer, DecisionKind, DecisionResponse, DecisionValue, Probability,
};
use switchyard_token_prediction::{
    BUDGET_UPPER_BOUNDS, Error, Prediction, QUESTION, ResponseUsage, prediction, request,
};

fn response(selected: &str, values: &[(&str, f64)]) -> DecisionResponse {
    let mut probabilities: BTreeMap<_, _> = (1..=80)
        .map(|i| (format!("b{i:03}"), Probability(0.0)))
        .collect();
    probabilities.insert("impossible".into(), Probability(0.0));
    for (key, value) in values {
        probabilities.insert((*key).into(), Probability(*value));
    }
    DecisionResponse {
        id: None,
        model: None,
        usage: Default::default(),
        answers: [(
            QUESTION.into(),
            DecisionAnswer {
                value: DecisionValue::Choice {
                    selected: selected.into(),
                    probabilities: Some(probabilities),
                },
                provider_confidence: None,
            },
        )]
        .into(),
    }
}

#[test]
fn matches_python_request_and_forecast() {
    // Generated with token_pred.jev.summarize_for_jev and post_jev_flat fine_request/expected_tokens.
    let fixture: Value = serde_json::from_str(include_str!("fixtures/python.json")).unwrap();
    let history = [
        json!({"source":"user", "message":"Fix the parser", "step_id":1}),
        json!({"source":"agent", "message":"Inspecting parser", "tokens_used":900, "step_id":2}),
    ];
    let request = request(
        "jev-latest".into(),
        &history,
        &[
            ResponseUsage {
                uncached_input_tokens: 800,
                output_tokens: 100,
                reasoning_tokens: 40,
            },
            ResponseUsage {
                uncached_input_tokens: 300,
                output_tokens: 200,
                reasoning_tokens: 50,
            },
        ],
    );
    assert_eq!(
        request.model.unwrap().as_str(),
        fixture["request"]["model"].as_str().unwrap()
    );
    assert_eq!(request.context, fixture["request"]["state"]);
    assert_eq!(history[1]["tokens_used"], 900);
    let question = &request.questions[QUESTION];
    let expected = &fixture["request"]["questions"][QUESTION];
    assert_eq!(question.instructions, expected["instructions"]);
    let DecisionKind::Choice { options } = &question.kind else {
        panic!("expected choice")
    };
    let criteria: serde_json::Map<_, _> = options
        .iter()
        .map(|option| (option.id.clone(), option.description.clone().unwrap()))
        .collect();
    assert_eq!(json!(criteria), expected["criteria"]);
    let answer = &fixture["response"]["answers"][QUESTION];
    let values: Vec<_> = answer["probabilities"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_f64().unwrap()))
        .collect();
    let actual = prediction(&response(answer["choice"].as_str().unwrap(), &values)).unwrap();
    let expected: Prediction = serde_json::from_value(fixture["prediction"].clone()).unwrap();
    assert_eq!(actual.selected_choice, expected.selected_choice);
    assert_eq!(actual.selected_interval, expected.selected_interval);
    for (actual, expected) in [
        (
            actual.predicted_tokens.unwrap(),
            expected.predicted_tokens.unwrap(),
        ),
        (
            actual.expected_upper_tokens.unwrap(),
            expected.expected_upper_tokens.unwrap(),
        ),
        (
            actual.impossible_probability,
            expected.impossible_probability,
        ),
        (
            actual.finite_budget_probability,
            expected.finite_budget_probability,
        ),
    ] {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }
}

#[test]
fn initial_request_and_all_task_budget_ranges() {
    let request = request("jev-latest".into(), &[json!({"message":"task"})], &[]);
    assert_eq!(request.context["prefix"]["end_response"], 0);
    assert_eq!(
        request.context["token_usage"]["tokens_used_so_far"],
        json!(0)
    );
    assert_eq!(request.context["token_usage"]["per_step"], json!([]));
    let DecisionKind::Choice { options } = &request.questions[QUESTION].kind else {
        panic!("expected choice")
    };
    assert_eq!(options.len(), 81);
    assert!(
        options[0]
            .description
            .as_ref()
            .unwrap()
            .as_str()
            .unwrap()
            .contains("(0, 2000]")
    );
    let mut lower = 0;
    for (i, option) in options.iter().take(80).enumerate() {
        let description = option.description.as_ref().unwrap().as_str().unwrap();
        let upper: u64 = description
            .split(", ")
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let result = prediction(&response(&option.id, &[(&option.id, 1.0)])).unwrap();
        assert_eq!(option.id, format!("b{:03}", i + 1));
        assert_eq!(result.predicted_tokens, Some((lower + upper) as f64 / 2.0));
        assert_eq!(result.selected_interval, Some([lower, upper]));
        lower = upper;
    }
    assert_eq!(lower, *BUDGET_UPPER_BOUNDS.last().unwrap());
}

#[test]
fn impossible_is_separate_from_finite_estimates() {
    let result = prediction(&response("impossible", &[("impossible", 1.0)])).unwrap();
    assert_eq!(result.predicted_tokens, None);
    assert_eq!(result.expected_upper_tokens, None);
    assert_eq!(result.selected_interval, None);
    assert_eq!(result.finite_budget_probability, 0.0);
    assert_eq!(result.impossible_probability, 1.0);
    assert_eq!(result.remaining_tokens(900), None);

    let result = prediction(&response(
        "impossible",
        &[("b001", 0.2), ("impossible", 0.8)],
    ))
    .unwrap();
    assert_eq!(result.predicted_tokens, Some(1000.0));
    assert_eq!(result.expected_upper_tokens, Some(2000.0));
    assert_eq!(result.selected_interval, None);
}

#[test]
fn total_budget_includes_prior_usage_and_remaining_subtracts_it_once() {
    let usage = [
        ResponseUsage {
            uncached_input_tokens: 800,
            output_tokens: 100,
            reasoning_tokens: 40,
        },
        ResponseUsage {
            uncached_input_tokens: 300,
            output_tokens: 200,
            reasoning_tokens: 50,
        },
    ];
    let request = request("jev-latest".into(), &[], &usage);
    assert_eq!(request.context["token_usage"]["tokens_used_so_far"], 1400);
    assert_eq!(
        request.context["token_usage"]["reasoning_output_tokens_so_far"],
        90
    );
    assert_eq!(request.context["prefix"]["end_response"], 2);
    let result = prediction(&response("b002", &[("b002", 1.0)])).unwrap();
    assert_eq!(result.predicted_tokens, Some(3000.0));
    assert_eq!(result.remaining_tokens(1400), Some(1600.0));
    assert_eq!(result.remaining_tokens(4000), Some(0.0));
}

#[test]
fn normalizes_rounded_distributions_and_rejects_invalid_values() {
    let result = prediction(&response("b001", &[("b001", 0.5), ("impossible", 0.49)])).unwrap();
    assert!((result.impossible_probability - 0.49 / 0.99).abs() < 1e-12);
    for value in [-0.1, 1.1, f64::NAN, f64::INFINITY, 0.0, 0.8] {
        assert_eq!(
            prediction(&response("b001", &[("b001", value)])),
            Err(Error::InvalidDistribution)
        );
    }
    assert_eq!(
        prediction(&response("unknown", &[("b001", 1.0)])),
        Err(Error::InvalidDistribution)
    );
    assert_eq!(
        prediction(&response("b001", &[("b001", 1.0), ("unknown", 0.0)])),
        Err(Error::InvalidDistribution)
    );

    let mut invalid = response("b001", &[("b001", 1.0)]);
    if let DecisionValue::Choice {
        probabilities: Some(probabilities),
        ..
    } = &mut invalid.answers.get_mut(QUESTION).unwrap().value
    {
        probabilities.remove("b080");
        probabilities.insert("unknown".into(), Probability(0.0));
    }
    assert_eq!(prediction(&invalid), Err(Error::InvalidDistribution));
    invalid.answers.clear();
    assert_eq!(prediction(&invalid), Err(Error::MissingAnswer));
    invalid = response("b001", &[]);
    if let DecisionValue::Choice { probabilities, .. } =
        &mut invalid.answers.get_mut(QUESTION).unwrap().value
    {
        *probabilities = None;
    }
    assert_eq!(prediction(&invalid), Err(Error::MissingAnswer));
}
