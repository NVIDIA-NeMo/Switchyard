// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed deployments use the same role selection, task policy and validation as TOML.

#![cfg(not(target_os = "wasi"))]

use std::collections::BTreeMap;

use libsy::PickerMode;
use switchyard_protocol::{Request, WireFormat, text_request};
use switchyard_runner::{
    AlgorithmSpec, RequirementScope, Runner, RunnerConfig, RunnerError, StageSettingsConfig,
    TargetEligibilityConfig, require_target_labels,
};

const CONFIG: &str = r#"
schema_version = 1
[llm_clients.primary]
format = "openai_chat"
base_url = "http://127.0.0.1:9/v1"
[targets.fast]
id = "cloud/fast"
llm_client = "primary"
[targets.capable]
id = "cloud/capable"
llm_client = "primary"
[targets.private_fast]
id = "private/fast"
llm_client = "primary"
labels = ["private"]
[targets.private_capable]
id = "private/capable"
llm_client = "primary"
labels = ["private"]
[targets.approved_fast]
id = "approved/fast"
llm_client = "primary"
labels = ["private", "customer_data"]
[targets.approved_capable]
id = "approved/capable"
llm_client = "primary"
labels = ["private", "customer_data"]
[routes.coding]
id = "coding"
type = "stage_router"
efficient_target = "fast"
capable_target = "capable"
picker = "efficient_first"
confidence_threshold = 0.6
[routes.coding.eligibility]
target_alternatives = {fast = ["private_fast", "approved_fast"], capable = ["private_capable", "approved_capable"]}
"#;

fn typed_config(
    scope: RequirementScope,
    require_explicit_requirements: bool,
) -> Result<RunnerConfig, RunnerError> {
    let mut config = RunnerConfig::new();
    config.add_http_client(
        "primary",
        WireFormat::OpenAiChat,
        "http://127.0.0.1:9/v1",
        None,
    )?;
    for (name, model, labels) in [
        ("fast", "cloud/fast", &[] as &[&str]),
        ("capable", "cloud/capable", &[]),
        ("private_fast", "private/fast", &["private"]),
        ("private_capable", "private/capable", &["private"]),
        (
            "approved_fast",
            "approved/fast",
            &["private", "customer_data"],
        ),
        (
            "approved_capable",
            "approved/capable",
            &["private", "customer_data"],
        ),
    ] {
        config.add_target(name, model, "primary", labels)?;
    }
    config.add_route(
        "coding",
        "coding",
        AlgorithmSpec::StageRouter {
            efficient_target: "fast".into(),
            capable_target: "capable".into(),
            capable_first_threshold: 0.0,
            settings: StageSettingsConfig {
                confidence_threshold: Some(0.6),
                ..Default::default()
            },
            picker: PickerMode::EfficientFirst,
            classifier: None,
            subagents: None,
        },
        Some(eligibility(scope, require_explicit_requirements)),
    )?;
    Ok(config)
}

fn eligibility(
    scope: RequirementScope,
    require_explicit_requirements: bool,
) -> TargetEligibilityConfig {
    TargetEligibilityConfig {
        scope,
        require_explicit_requirements,
        target_alternatives: BTreeMap::from([
            (
                "fast".into(),
                vec!["private_fast".into(), "approved_fast".into()],
            ),
            (
                "capable".into(),
                vec!["private_capable".into(), "approved_capable".into()],
            ),
        ]),
    }
}

fn request(labels: Option<&[&str]>) -> Request {
    let mut request = Request {
        llm_request: text_request(Some("coding".into()), "Summarize these synthetic notes."),
        ..Default::default()
    };
    request.metadata.get_or_insert_default().task_id = Some("renewal".into());
    if let Some(labels) = labels {
        require_target_labels(&mut request, labels.iter().copied()).expect("requirements");
    }
    request
}

#[tokio::test]
async fn typed_and_toml_routes_select_the_same_ordered_roles() -> Result<(), RunnerError> {
    let typed = Runner::from_config(typed_config(RequirementScope::Request, false)?)?;
    let parsed = Runner::from_toml(CONFIG)?;
    for labels in [
        None,
        Some(&[][..]),
        Some(&["private"][..]),
        Some(&["customer_data", "private"][..]),
    ] {
        let expected = if labels.is_none_or(|labels| labels.is_empty()) {
            ["fast", "capable"]
        } else if labels.is_some_and(|labels| labels.len() == 1) {
            ["private_fast", "private_capable"]
        } else {
            ["approved_fast", "approved_capable"]
        };
        for runner in [&typed, &parsed] {
            let result = runner
                .route("coding")
                .expect("route")
                .decide(request(labels))
                .await?;
            assert_eq!(result.selected_model_ids, expected);
        }
    }
    for runner in [&typed, &parsed] {
        assert!(
            runner
                .route("coding")
                .expect("route")
                .decide(request(Some(&["unknown"])))
                .await
                .is_err()
        );
    }
    let mut edited: RunnerConfig = toml::from_str(CONFIG).expect("typed deployment schema");
    edited.set_target_labels("fast", ["private"])?;
    let runner = Runner::from_config(edited)?;
    let result = runner
        .route("coding")
        .expect("route")
        .decide(request(Some(&["private"])))
        .await?;
    assert_eq!(result.selected_model_ids, ["fast", "private_capable"]);
    Ok(())
}

#[tokio::test]
async fn typed_and_toml_policies_require_labels_and_retain_task_union() -> Result<(), RunnerError> {
    let typed = Runner::from_config(typed_config(RequirementScope::Task, true)?)?;
    let parsed = Runner::from_toml(&format!(
        "{CONFIG}\nscope = 'task'\nrequire_explicit_requirements = true"
    ))?;
    let mut edited: RunnerConfig = toml::from_str(CONFIG).expect("typed deployment schema");
    edited.set_route_eligibility("coding", Some(eligibility(RequirementScope::Task, true)))?;
    let edited = Runner::from_config(edited)?;
    for runner in [&typed, &parsed, &edited] {
        let route = runner.route("coding").expect("route");
        assert!(route.decide(request(None)).await.is_err());
        for (labels, expected) in [
            (&[][..], "fast"),
            (&["private"][..], "private_fast"),
            (&[][..], "private_fast"),
            (&["customer_data"][..], "approved_fast"),
            (&[][..], "approved_fast"),
        ] {
            let result = route.decide(request(Some(labels))).await?;
            assert_eq!(
                result.selected_model_id().expect("selected model"),
                expected
            );
        }
        assert!(route.decide(request(None)).await.is_err());
    }
    Ok(())
}

fn build_error(config: RunnerConfig) -> String {
    match Runner::from_config(config) {
        Ok(_) => panic!("invalid typed configuration was accepted"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn typed_configuration_uses_shared_validation_without_overwriting_names() -> Result<(), RunnerError>
{
    let mut config = typed_config(RequirementScope::Request, false)?;
    config.set_route_eligibility(
        "coding",
        Some(eligibility(RequirementScope::Request, false)),
    )?;
    assert!(config.set_target_labels("missing", ["private"]).is_err());
    assert!(config.set_route_eligibility("missing", None).is_err());
    assert!(
        config
            .add_target("fast", "replacement", "primary", ["private"])
            .is_err()
    );
    assert!(
        config
            .add_http_client(
                "primary",
                WireFormat::OpenAiChat,
                "http://127.0.0.1:9",
                None
            )
            .is_err()
    );
    assert!(
        config
            .add_http_client("invalid", WireFormat::OpenAiChat, "file:///tmp/model", None)
            .is_err()
    );
    config.add_target("invalid", "unapproved", "primary", [" private"])?;
    assert!(build_error(config).contains("target labels"));

    let mut config = typed_config(RequirementScope::Request, false)?;
    config.add_route(
        "bad",
        "bad",
        AlgorithmSpec::Passthrough {
            target: "fast".into(),
            subagents: None,
        },
        Some(TargetEligibilityConfig {
            target_alternatives: BTreeMap::from([("fast".into(), vec!["missing".into()])]),
            ..Default::default()
        }),
    )?;
    let typed = build_error(config);
    let invalid = format!(
        "{CONFIG}\n[routes.bad]\nid = 'bad'\ntype = 'passthrough'\ntarget = 'fast'\n[routes.bad.eligibility]\ntarget_alternatives = {{fast = ['missing']}}\n"
    );
    let parsed = build_error(toml::from_str(&invalid).expect("configuration schema"));
    assert_eq!(typed, parsed);

    let mut config: RunnerConfig = toml::from_str(&format!(
        "{CONFIG}\n[decision_targets.audit]\nid = 'audit'\ndecision_client = 'missing'\n"
    ))
    .expect("typed decision target schema");
    config.set_target_labels("audit", [" private"])?;
    assert!(build_error(config).contains("target labels"));
    Ok(())
}
