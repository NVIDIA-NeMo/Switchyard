// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, mpsc};

use serde_json::json;
use switchyard_llm_client::RunObservation;
use switchyard_protocol::{
    ContentBlock, InstructionBlock, LlmRequest, Message, Request, Role, completion_text,
};
use switchyard_runner::Runner;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request(system: &str, catalog: &str, stream: bool) -> Request {
    Request {
        llm_request: LlmRequest {
            instructions: vec![InstructionBlock {
                role: Role::System,
                content: vec![ContentBlock::Text {
                    text: system.into(),
                }],
            }],
            messages: vec![
                Message::text(Role::User, format!("Catalog\n{catalog}")),
                Message::text(Role::Assistant, "Ready"),
                Message::text(Role::User, "Task: find a song . List names."),
            ],
            stream,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[tokio::test]
async fn selection_rule_answers_and_falls_back() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    let url = server.uri();
    let config = format!(
        r#"
schema_version = 1
[llm_clients.writer]
format = "openai_chat"
base_url = "{url}/v1"
[targets.writer]
id = "llm"
llm_client = "writer"
[decision_clients.scorer]
format = "system_one"
endpoint = "{url}/v1/systemone"
api_key_env = "SWITCHYARD_SELECTION_TEST_KEY"
timeout_ms = 1000
[decision_targets.selector]
id = "decision"
decision_client = "scorer"
[routes.agent]
id = "agent"
type = "decision_select"
target = "writer"
decision_target = "selector"
[routes.agent.rule]
enabled = true
system_instruction = "Choose names"
catalog_prefix = "Catalog\n"
task_prefix = "Task: "
task_suffix = ". List names."
question = "Which names are relevant?"
question_prefix = "required_apis_"
shortlist = 4
top_k = 1
"#
    );
    // Only this test reads this variable, while constructing the two runners.
    unsafe { std::env::set_var("SWITCHYARD_SELECTION_TEST_KEY", "test") };
    let on = Runner::from_toml(&config);
    let off = Runner::from_toml(&config.replace("enabled = true", "enabled = false"));
    unsafe { std::env::remove_var("SWITCHYARD_SELECTION_TEST_KEY") };
    let (on, off) = (on?, off?);
    let agent = on.route("agent").unwrap();
    let catalog = "music:\n  buy: Purchase a track\n  search: Find songs";
    let post = || Mock::given(method("POST")).and(path("/v1/systemone"));
    let success = post()
        .and(body_partial_json(json!({
            "model":"decision", "state":"find a song ", "questions":{
                "required_apis_0":{"type":"noul"}, "required_apis_1":{"type":"noul"}
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id":"ranked", "model":"decision", "answers":{
                "required_apis_0":{"type":"noul","noul":0.1}, "required_apis_1":{"type":"noul","noul":0.9}
            }, "usage":{"input_tokens":20,"output_tokens":2}
        })))
        .expect(2)
        .mount_as_scoped(&server)
        .await;
    for stream in [false, true] {
        let (send, recv) = mpsc::channel();
        let result = agent
            .execute(
                request("Choose names", catalog, stream),
                Some(Arc::new(move |event| {
                    send.send(event).unwrap();
                })),
            )
            .await?;
        assert_eq!(result.response.served_model(), Some(&result.selected_model));
        let answer = result.response.llm_response.into_agg().await?;
        assert_eq!(completion_text(&answer), "music.search");
        assert_eq!(answer.usage.input_tokens, Some(20));
        let calls = recv
            .try_iter()
            .filter_map(|event| match event {
                RunObservation::AnswerCall(call) => Some(call.is_success),
                RunObservation::LlmCall(_) | RunObservation::DecisionCall(_) => Some(false),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(calls, [true]);
    }
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id":"chat", "model":"llm", "choices":[{"index":0,"message":{"role":"assistant","content":"ordinary answer"},"finish_reason":"stop"}]
        }))).expect(7).mount(&server).await;
    let mut capped = request("Choose names", catalog, false);
    capped.llm_request.output.max_output_tokens = Some(1);
    let mut multiple = request("Choose names", catalog, false);
    multiple
        .llm_request
        .extensions
        .fields
        .insert("n".into(), json!(2));
    for (route, input) in [
        (agent, capped),
        (agent, multiple),
        (agent, request("Write code", catalog, false)),
        (agent, request("Choose names", "invalid: [", false)),
        (
            off.route("agent").unwrap(),
            request("Choose names", catalog, false),
        ),
    ] {
        let result = route.execute(input, None).await?;
        assert_eq!(
            completion_text(&result.response.llm_response.into_agg().await?),
            "ordinary answer"
        );
    }
    drop(success);
    for response in [
        ResponseTemplate::new(503),
        ResponseTemplate::new(200).set_body_json(json!({
            "model":"decision","answers":{"required_apis_0":{"type":"noul","noul":0.5}}
        })),
    ] {
        let failure = post()
            .respond_with(response)
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        let result = agent
            .execute(request("Choose names", catalog, false), None)
            .await?;
        assert_eq!(
            completion_text(&result.response.llm_response.into_agg().await?),
            "ordinary answer"
        );
        drop(failure);
    }
    Ok(())
}
