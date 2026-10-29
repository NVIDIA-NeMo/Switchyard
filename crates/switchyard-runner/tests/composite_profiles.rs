// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use libsy::{Call, drive};
use switchyard_protocol::{LlmResponse, ModelId, Request, Response, text_request, text_response};
use switchyard_runner::AlgorithmSpec;

const CONFIG: &str = r#"
type = "composite"
[classifier]
target = "judge"
base_threshold = 0.5
classify_trigger = "user_turn"
[stage]
capable_target = "strong"
efficient_target = "weak"
efficient_first_threshold = 0.5
capable_first_threshold = 0.2
[efficient_first]
confidence_threshold = 0.4
capable_hold_turns = 0
efficient_hold_turns = 0
capable_system_prompt = "EF capable prompt"
efficient_system_prompt = "EF efficient prompt"
[capable_first]
confidence_threshold = 0.1
capable_hold_turns = 3
efficient_hold_turns = -1
capable_system_prompt = "CF capable prompt"
efficient_system_prompt = "CF efficient prompt"
"#;

#[tokio::test]
async fn toml_profiles_apply_only_the_selected_prompt() {
    let spec: AlgorithmSpec = toml::from_str(CONFIG).unwrap();
    let targets = BTreeMap::from([
        ("strong".into(), ModelId::from("strong")),
        ("weak".into(), ModelId::from("weak")),
        ("judge".into(), ModelId::from("judge")),
    ]);
    let (algorithm, models) = spec.build_with_runtime_models("auto", &targets).unwrap();
    let models = Arc::new(models);
    for (p_solve, target, prompt) in [
        (1.0, "weak", "EF efficient prompt"),
        (0.0, "strong", "CF capable prompt"),
    ] {
        let outcome = drive(
            algorithm.clone(),
            Request { llm_request: text_request(None, "Help with this task"), ..Default::default() },
            models.clone(),
            |call| async move {
                let Call::Model(call) = call else { panic!("expected an LLM judge") };
                assert_eq!(call.models[0].as_str(), "judge");
                assert!(!format!("{:?}", call.request.llm_request.instructions).contains("CF capable prompt"));
                let verdict = format!(r#"{{"crux":"task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":{p_solve}}}"#);
                call.respond(async { Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(Some("judge".into()), verdict)),
                    metadata: None,
                    upstream_headers: Default::default(),
                }) }).await
            },
        ).await.unwrap();
        assert_eq!(outcome.selected_model_id().unwrap().as_str(), target);
        assert_eq!(outcome.request.llm_request.instructions.len(), 1);
        assert_eq!(
            outcome.request.llm_request.instructions[0].content,
            vec![switchyard_protocol::ContentBlock::Text {
                text: prompt.into()
            }]
        );
        assert_eq!(outcome.selected_model_ids.len(), 2);
    }
}

#[tokio::test]
async fn profile_signals_use_their_threshold_window_tool_semantics_and_notes() {
    use switchyard_protocol::{ContentBlock, Message, Metadata, Role, ToolCall, ToolResult};

    for (threshold, window, expected, note) in [
        (0.0, 1, "weak", Some("Continue the plan")),
        (1.0, 1, "strong", None),
        (0.0, 2, "strong", Some("Review the error")),
    ] {
        let source = CONFIG.replace("confidence_threshold = 0.1", &format!(
            "confidence_threshold = {threshold}\nrecent_turn_window = {window}\n\
             tool_semantics = {{ mutate = ['edit_item'] }}\n\
             handoff_notes = {{ escalation_note = 'Review the error', deescalation_note = 'Continue the plan' }}"
        ));
        let spec: AlgorithmSpec = toml::from_str(&source).unwrap();
        let targets = ["strong", "weak", "judge"]
            .into_iter()
            .map(|target| (target.into(), ModelId::from(target)))
            .collect();
        let (algorithm, models) = spec.build_with_runtime_models("auto", &targets).unwrap();
        let models = Arc::new(models);
        let mut request = Request {
            llm_request: text_request(None, "Fix this task"),
            metadata: Some(Metadata {
                session_id: Some("task".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let judge = |call| async move {
            let Call::Model(call) = call else {
                panic!("expected an LLM judge")
            };
            call.respond(async { Ok(Response {
                llm_response: LlmResponse::Agg(text_response(Some("judge".into()),
                    r#"{"crux":"task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.0}"#)),
                metadata: None,
                upstream_headers: Default::default(),
            }) }).await
        };
        drive(algorithm.clone(), request.clone(), models.clone(), judge)
            .await
            .unwrap();
        for (id, name, text, error) in [
            ("first", "Bash", "fatal runtime error: out of memory", true),
            ("second", "edit_item", "done", false),
        ] {
            request.llm_request.messages.push(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: id.into(),
                    name: name.into(),
                    arguments: serde_json::json!({}),
                })],
            });
            request.llm_request.messages.push(Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult(ToolResult {
                    tool_call_id: id.into(),
                    content: vec![ContentBlock::Text { text: text.into() }],
                    is_error: Some(error),
                })],
            });
        }
        let outcome = drive(algorithm, request, models, |_| async {
            panic!("tool continuations must retain the profile")
        })
        .await
        .unwrap();
        assert_eq!(
            outcome.selected_model_id().unwrap().as_str(),
            expected,
            "threshold={threshold}, window={window}"
        );
        let messages = format!("{:?}", outcome.request.llm_request.messages);
        if let Some(note) = note {
            assert!(messages.contains(note), "{messages}");
        } else {
            assert!(!messages.contains("Review the error"));
            assert!(!messages.contains("Continue the plan"));
        }
    }
}
