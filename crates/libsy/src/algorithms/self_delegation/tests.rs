// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_protocol::{
    AggLlmResponse, Category, ContentBlock, InstructionBlock, LlmClientError, LlmResponse,
    Metadata, ModelId, Request, Response, ResponseOutput, Role, StopReason, ToolCall,
    ToolDefinition, Usage, WireFormat, completion_text, text_request,
};

use super::*;
use crate::core::algorithm::RuntimeModels;
use crate::core::testing::{Serve, test_drive_with_models};
use crate::{Algorithm, LibsyError, Result};

const STUDENT: &str = "student";
const STRONG: &str = "strong";
const LOCAL_LINE: &str = "{\"route\":\"local\"}";

fn models(with_capable: bool) -> HashMap<Category, Vec<ModelId>> {
    let mut models = HashMap::from([
        (Category::Efficient, vec![ModelId::from(STUDENT)]),
        (Category::Any, vec![ModelId::from(STUDENT)]),
    ]);
    if with_capable {
        models.insert(Category::Capable, vec![ModelId::from(STRONG)]);
        models
            .entry(Category::Any)
            .or_default()
            .push(ModelId::from(STRONG));
    }
    models
}

fn request() -> Request {
    let mut llm_request = text_request(Some("switchyard/edgewise".to_string()), "fix the parser");
    llm_request.instructions = vec![InstructionBlock {
        role: Role::System,
        content: vec![ContentBlock::Text {
            text: "You are a coding agent.".to_string(),
        }],
    }];
    Request {
        llm_request,
        raw_request: None,
        metadata: Some(Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some("agent-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            task_id: Some("task-1".to_string()),
            correlation_id: Some("request-1".to_string()),
            wire_format: Some(WireFormat::AnthropicMessages),
            ..Metadata::default()
        }),
    }
}

fn reply_with_usage(text: &str, tool_calls: usize, input: u64, output: u64) -> AggLlmResponse {
    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(ContentBlock::Text {
            text: text.to_string(),
        });
    }
    for index in 0..tool_calls {
        content.push(ContentBlock::ToolCall(ToolCall {
            id: format!("call_{index}"),
            name: "bash".to_string(),
            arguments: json!({"cmd": "ls"}),
        }));
    }
    AggLlmResponse {
        id: Some("response-1".to_string()),
        outputs: vec![ResponseOutput {
            role: Role::Assistant,
            content,
            url_citations: Vec::new(),
            stop_reason: Some(if tool_calls > 0 {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            }),
        }],
        usage: Usage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            total_tokens: Some(input + output),
            ..Usage::default()
        },
        ..AggLlmResponse::default()
    }
}

fn reply(text: &str, tool_calls: usize) -> AggLlmResponse {
    reply_with_usage(text, tool_calls, 10, 5)
}

fn transport_error() -> LlmClientError {
    LlmClientError::Transport {
        source: Box::new(std::io::Error::other("connection reset")),
    }
}

type Scripted = std::result::Result<AggLlmResponse, LlmClientError>;

/// Scripted replies per target, and every call the algorithm made.
struct Fake {
    replies: Mutex<HashMap<String, VecDeque<Scripted>>>,
    calls: Mutex<Vec<(ModelId, Request)>>,
}

impl Fake {
    fn new(student: Vec<Scripted>, strong: Vec<Scripted>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(HashMap::from([
                (STUDENT.to_string(), student.into()),
                (STRONG.to_string(), strong.into()),
            ])),
            calls: Mutex::new(Vec::new()),
        })
    }

    fn serve(self: &Arc<Self>) -> impl Serve {
        let fake = Arc::clone(self);
        move |target: ModelId, request: Request| {
            let fake = Arc::clone(&fake);
            async move {
                fake.calls.lock().push((target.clone(), request));
                let next = fake
                    .replies
                    .lock()
                    .get_mut(target.as_str())
                    .and_then(VecDeque::pop_front);
                match next {
                    Some(Ok(agg)) => Ok(Response {
                        llm_response: LlmResponse::Agg(agg),
                        metadata: None,
                        upstream_headers: http::HeaderMap::new(),
                    }),
                    Some(Err(error)) => Err(error),
                    None => Err(LlmClientError::General(format!(
                        "unexpected call to {target}"
                    ))),
                }
            }
        }
    }

    fn targets(&self) -> Vec<String> {
        self.calls
            .lock()
            .iter()
            .map(|(target, _)| target.to_string())
            .collect()
    }
}

/// A trace file in the temporary directory, removed with the value.
struct TraceFile(PathBuf);

impl TraceFile {
    fn new(name: &str) -> Self {
        Self(std::env::temp_dir().join(format!(
            "self_delegation_{name}_{}.jsonl",
            uuid::Uuid::now_v7()
        )))
    }

    fn config(&self) -> SelfDelegationConfig {
        SelfDelegationConfig {
            trace_jsonl: Some(self.0.clone()),
            ..SelfDelegationConfig::default()
        }
    }

    fn records(&self) -> Vec<Value> {
        std::fs::read_to_string(&self.0)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).expect("trace line is JSON"))
            .collect()
    }

    fn record(&self) -> Value {
        let records = self.records();
        assert_eq!(records.len(), 1, "one trace record per decision");
        records.into_iter().next().expect("one record")
    }
}

impl Drop for TraceFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

async fn drive_raw(
    config: SelfDelegationConfig,
    request: Request,
    models: impl Into<RuntimeModels>,
    fake: &Arc<Fake>,
) -> Result<(ModelId, Response)> {
    let algorithm: Arc<dyn Algorithm> = Arc::new(SelfDelegation::new(config));
    test_drive_with_models(algorithm, request, models, fake.serve()).await
}

async fn drive(
    config: SelfDelegationConfig,
    request: Request,
    models: impl Into<RuntimeModels>,
    fake: &Arc<Fake>,
) -> Result<(ModelId, AggLlmResponse)> {
    let (selected, response) = drive_raw(config, request, models, fake).await?;
    let agg = aggregate(selected.clone(), response).await?;
    Ok((selected, agg))
}

async fn aggregate(selected: ModelId, response: Response) -> Result<AggLlmResponse> {
    response
        .llm_response
        .into_agg()
        .await
        .map_err(|source| LibsyError::client_call(selected, source))
}

fn tool_calls(agg: &AggLlmResponse) -> usize {
    agg.first_output().map_or(0, |output| {
        output
            .content
            .iter()
            .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
            .count()
    })
}

fn has_text_block(agg: &AggLlmResponse) -> bool {
    agg.first_output().is_some_and(|output| {
        output
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Text { .. }))
    })
}

#[tokio::test]
async fn local_reply_is_served_unchanged() -> Result<()> {
    let trace = TraceFile::new("local");
    let text = "{\"route\":\"local\"}\nI will read the file first.";
    let fake = Fake::new(vec![Ok(reply(text, 1))], vec![]);

    let (selected, agg) = drive(trace.config(), request(), models(true), &fake).await?;

    assert_eq!(selected, STUDENT);
    assert_eq!(completion_text(&agg), text);
    assert_eq!(tool_calls(&agg), 1);
    assert_eq!(agg.usage.input_tokens, Some(10));
    assert_eq!(fake.targets(), [STUDENT]);
    let record = trace.record();
    assert_eq!(record["schema_version"], "native-v1");
    assert!(record["ts"].as_str().is_some_and(|ts| ts.ends_with('Z')));
    assert_eq!(record["route"], "switchyard/edgewise");
    assert_eq!(record["mode"], "serve");
    assert_eq!(record["hide_route"], false);
    assert_eq!(record["lenient_cloud"], true);
    assert_eq!(record["session_id"], "session-1");
    assert_eq!(record["agent_id"], "agent-1");
    assert_eq!(record["turn_id"], "turn-1");
    assert_eq!(record["task_id"], "task-1");
    assert_eq!(record["correlation_id"], "request-1");
    assert_eq!(record["wire_format"], "anthropic_messages");
    assert_eq!(record["is_compaction"], false);
    assert_eq!(record["has_compaction"], false);
    assert_eq!(record["decision"], "local");
    assert_eq!(record["student"]["target"], STUDENT);
    let attempt = &record["student"]["attempts"][0];
    assert_eq!(attempt["content"], text);
    assert_eq!(attempt["classification"], "local");
    assert_eq!(attempt["valid"], true);
    assert_eq!(attempt["finish_reason"], "tool_calls");
    assert_eq!(attempt["tool_calls"][0]["id"], "call_0");
    assert_eq!(attempt["tool_calls"][0]["type"], "function");
    assert_eq!(attempt["tool_calls"][0]["function"]["name"], "bash");
    assert_eq!(
        attempt["tool_calls"][0]["function"]["arguments"],
        "{\"cmd\":\"ls\"}"
    );
    assert_eq!(attempt["usage"]["input_tokens"], 10);
    assert_eq!(attempt["usage"]["output_tokens"], 5);
    assert!(attempt["latency_ms"].is_u64());
    assert!(attempt["error"].is_null());
    assert!(record["strong"].is_null());
    assert_eq!(record["served"]["by"], "student");
    assert_eq!(record["served"]["content"], text);
    assert_eq!(record["served"]["tool_calls"], 1);
    assert_eq!(record["served"]["stripped_route_line"], false);
    Ok(())
}

#[tokio::test]
async fn cloud_reply_calls_strong_with_the_neutral_request_and_merges() -> Result<()> {
    let trace = TraceFile::new("cloud");
    let fake = Fake::new(
        vec![Ok(reply(CLOUD_SENTINEL, 0))],
        vec![Ok(reply_with_usage("I will edit the parser.", 1, 20, 7))],
    );

    let (selected, agg) = drive(trace.config(), request(), models(true), &fake).await?;

    assert_eq!(selected, STRONG);
    assert_eq!(
        completion_text(&agg),
        "{\"route\":\"cloud\"}\nI will edit the parser."
    );
    assert_eq!(tool_calls(&agg), 1);
    // The client-visible usage covers both calls.
    assert_eq!(agg.usage.input_tokens, Some(30));
    assert_eq!(agg.usage.output_tokens, Some(12));
    assert_eq!(agg.usage.total_tokens, Some(42));
    {
        let calls = fake.calls.lock();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, STUDENT);
        assert_eq!(calls[1].0, STRONG);
        let neutral = request();
        for (_, call) in calls.iter() {
            assert!(!call.llm_request.stream, "upstream calls are buffered");
            assert_eq!(call.llm_request.messages, neutral.llm_request.messages);
            assert_eq!(
                call.llm_request.instructions,
                neutral.llm_request.instructions
            );
        }
    }
    let record = trace.record();
    assert_eq!(record["decision"], "cloud");
    assert_eq!(record["student"]["attempts"][0]["classification"], "cloud");
    assert_eq!(record["student"]["attempts"][0]["valid"], true);
    assert_eq!(record["strong"]["target"], STRONG);
    assert_eq!(record["strong"]["content"], "I will edit the parser.");
    assert_eq!(
        record["strong"]["tool_calls"][0]["function"]["name"],
        "bash"
    );
    assert_eq!(record["strong"]["finish_reason"], "tool_calls");
    assert_eq!(record["strong"]["usage"]["input_tokens"], 20);
    assert_eq!(record["strong"]["imitated_route_line"], false);
    assert_eq!(record["strong"]["fallback"], false);
    assert!(record["strong"]["error"].is_null());
    assert_eq!(record["served"]["by"], "strong");
    assert_eq!(
        record["served"]["content"],
        "{\"route\":\"cloud\"}\nI will edit the parser."
    );
    assert_eq!(record["served"]["tool_calls"], 1);
    assert_eq!(record["served"]["stripped_route_line"], false);
    Ok(())
}

#[tokio::test]
async fn strong_imitated_route_line_is_dropped_and_flagged() -> Result<()> {
    let trace = TraceFile::new("imitated");
    let fake = Fake::new(
        vec![Ok(reply(CLOUD_SENTINEL, 0))],
        vec![Ok(reply("{\"route\":\"local\"}\nEditing.", 1))],
    );

    let (_, agg) = drive(trace.config(), request(), models(true), &fake).await?;

    assert_eq!(completion_text(&agg), "{\"route\":\"cloud\"}\nEditing.");
    let record = trace.record();
    assert_eq!(record["strong"]["imitated_route_line"], true);
    assert_eq!(
        record["strong"]["content"],
        "{\"route\":\"local\"}\nEditing."
    );
    assert_eq!(
        record["served"]["content"],
        "{\"route\":\"cloud\"}\nEditing."
    );
    Ok(())
}

#[tokio::test]
async fn strong_tool_only_reply_gets_the_sentinel_ahead_of_its_tool_calls() -> Result<()> {
    let fake = Fake::new(vec![Ok(reply(CLOUD_SENTINEL, 0))], vec![Ok(reply("", 2))]);

    let (_, agg) = drive(
        SelfDelegationConfig::default(),
        request(),
        models(true),
        &fake,
    )
    .await?;

    let content = &agg.outputs[0].content;
    assert_eq!(content.len(), 3);
    assert!(matches!(&content[0], ContentBlock::Text { text } if text == CLOUD_SENTINEL));
    assert_eq!(tool_calls(&agg), 2);
    Ok(())
}

#[tokio::test]
async fn invalid_reply_is_retried_once_then_served() -> Result<()> {
    let trace = TraceFile::new("invalid");
    let fake = Fake::new(
        vec![
            Ok(reply("I think the bug is in the parser.", 0)),
            Ok(reply("{\"route\":\"local\"}\n", 0)),
        ],
        vec![],
    );

    let (selected, agg) = drive(trace.config(), request(), models(true), &fake).await?;

    assert_eq!(selected, STUDENT);
    assert_eq!(completion_text(&agg), "{\"route\":\"local\"}\n");
    assert_eq!(fake.targets(), [STUDENT, STUDENT]);
    let record = trace.record();
    assert_eq!(record["decision"], "invalid");
    let attempts = record["student"]["attempts"]
        .as_array()
        .expect("attempts array");
    assert_eq!(attempts.len(), 2);
    for attempt in attempts {
        assert_eq!(attempt["classification"], "invalid");
        assert_eq!(attempt["valid"], false);
    }
    assert!(record["strong"].is_null());
    assert_eq!(record["served"]["by"], "student");
    Ok(())
}

#[tokio::test]
async fn invalid_reply_escalates_when_configured() -> Result<()> {
    let trace = TraceFile::new("invalid_escalate");
    let config = SelfDelegationConfig {
        on_invalid: OnInvalid {
            retry: 1,
            action: InvalidAction::Escalate,
        },
        ..trace.config()
    };
    let fake = Fake::new(
        vec![Ok(reply("", 0)), Ok(reply("prose without a step", 0))],
        vec![Ok(reply("Running the tests.", 1))],
    );

    let (selected, agg) = drive(config, request(), models(true), &fake).await?;

    assert_eq!(selected, STRONG);
    assert_eq!(
        completion_text(&agg),
        "{\"route\":\"cloud\"}\nRunning the tests."
    );
    assert_eq!(fake.targets(), [STUDENT, STUDENT, STRONG]);
    let record = trace.record();
    assert_eq!(record["decision"], "invalid");
    assert_eq!(record["strong"]["fallback"], true);
    assert_eq!(record["served"]["by"], "strong");
    Ok(())
}

#[tokio::test]
async fn hide_route_strips_the_route_line_from_a_local_reply() -> Result<()> {
    let trace = TraceFile::new("hide_local");
    let config = SelfDelegationConfig {
        hide_route: true,
        ..trace.config()
    };
    let text = "\n{\"route\":\"local\"}\nText after the line.";
    let fake = Fake::new(vec![Ok(reply(text, 1))], vec![]);

    let (_, agg) = drive(config, request(), models(true), &fake).await?;

    assert_eq!(completion_text(&agg), "Text after the line.");
    assert_eq!(tool_calls(&agg), 1);
    let record = trace.record();
    assert_eq!(record["hide_route"], true);
    assert_eq!(record["student"]["attempts"][0]["content"], text);
    assert_eq!(record["served"]["content"], "Text after the line.");
    assert_eq!(record["served"]["stripped_route_line"], true);
    Ok(())
}

#[tokio::test]
async fn hide_route_omits_an_emptied_text_block() -> Result<()> {
    let config = SelfDelegationConfig {
        hide_route: true,
        ..SelfDelegationConfig::default()
    };
    let fake = Fake::new(vec![Ok(reply(LOCAL_LINE, 2))], vec![]);

    let (_, agg) = drive(config, request(), models(true), &fake).await?;

    assert!(!has_text_block(&agg));
    assert_eq!(tool_calls(&agg), 2);
    Ok(())
}

#[tokio::test]
async fn hide_route_serves_the_strong_step_without_the_sentinel() -> Result<()> {
    let trace = TraceFile::new("hide_cloud");
    let config = SelfDelegationConfig {
        hide_route: true,
        ..trace.config()
    };
    let fake = Fake::new(
        vec![Ok(reply(CLOUD_SENTINEL, 0))],
        vec![Ok(reply("{\"route\":\"local\"}\nEditing.", 1))],
    );

    let (selected, agg) = drive(config, request(), models(true), &fake).await?;

    assert_eq!(selected, STRONG);
    assert_eq!(completion_text(&agg), "Editing.");
    let record = trace.record();
    assert_eq!(record["strong"]["imitated_route_line"], true);
    assert_eq!(record["served"]["content"], "Editing.");
    assert_eq!(record["served"]["stripped_route_line"], true);
    Ok(())
}

#[tokio::test]
async fn upstream_calls_are_buffered_and_a_streaming_client_gets_a_stream() -> Result<()> {
    for (student, strong, served_by, text) in [
        (
            reply(CLOUD_SENTINEL, 0),
            Some(reply("Editing.", 1)),
            STRONG,
            "{\"route\":\"cloud\"}\nEditing.",
        ),
        (
            reply("{\"route\":\"local\"}\nLooking.", 1),
            None,
            STUDENT,
            "{\"route\":\"local\"}\nLooking.",
        ),
    ] {
        let mut request = request();
        request.llm_request.stream = true;
        // A same-format client body is replayed verbatim by the LLM client.
        request.llm_request.preservation.requests.insert(
            "openai_chat".into(),
            json!({
                "model": "switchyard/edgewise",
                "messages": [{"role": "user", "content": "fix the parser"}],
                "stream": true,
                "stream_options": {"include_usage": true},
            }),
        );
        let fake = Fake::new(vec![Ok(student)], strong.into_iter().map(Ok).collect());

        let (selected, response) = drive_raw(
            SelfDelegationConfig::default(),
            request,
            models(true),
            &fake,
        )
        .await?;

        assert_eq!(selected, served_by);
        assert!(matches!(response.llm_response, LlmResponse::Stream(_)));
        let agg = aggregate(selected, response).await?;
        assert_eq!(completion_text(&agg), text);
        assert_eq!(tool_calls(&agg), 1);
        for (_, call) in fake.calls.lock().iter() {
            assert!(!call.llm_request.stream);
            for body in call.llm_request.preservation.requests.values() {
                assert!(body.get("stream").is_none());
                assert!(body.get("stream_options").is_none());
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn compaction_flags_are_recorded_and_the_reply_is_served_unchanged() -> Result<()> {
    let trace = TraceFile::new("compaction");
    let mut request = request();
    request.llm_request.is_compaction = true;
    request.llm_request.has_compaction = true;
    let text = "{\"route\":\"local\"}\nSummary of the work so far.";
    let fake = Fake::new(vec![Ok(reply(text, 0))], vec![]);

    let (selected, agg) = drive(trace.config(), request, models(true), &fake).await?;

    assert_eq!(selected, STUDENT);
    assert_eq!(completion_text(&agg), text);
    let record = trace.record();
    assert_eq!(record["is_compaction"], true);
    assert_eq!(record["has_compaction"], true);
    assert_eq!(record["decision"], "local");
    Ok(())
}

#[tokio::test]
async fn collection_mode_serves_the_sentinel_as_is() -> Result<()> {
    let trace = TraceFile::new("collect");
    let fake = Fake::new(vec![Ok(reply(CLOUD_SENTINEL, 0))], vec![]);

    let (selected, agg) = drive(trace.config(), request(), models(false), &fake).await?;

    assert_eq!(selected, STUDENT);
    assert_eq!(completion_text(&agg), CLOUD_SENTINEL);
    assert_eq!(fake.targets(), [STUDENT]);
    let record = trace.record();
    assert_eq!(record["mode"], "collect");
    assert_eq!(record["decision"], "cloud");
    assert!(record["strong"].is_null());
    assert_eq!(record["served"]["by"], "student");
    assert_eq!(record["served"]["content"], CLOUD_SENTINEL);
    Ok(())
}

#[tokio::test]
async fn escalation_policies_need_a_capable_target() -> Result<()> {
    let config = SelfDelegationConfig {
        on_invalid: OnInvalid {
            retry: 0,
            action: InvalidAction::Escalate,
        },
        on_student_error: StudentErrorAction::Escalate,
        ..SelfDelegationConfig::default()
    };
    let fake = Fake::new(vec![Ok(reply("prose without a step", 0))], vec![]);
    let (selected, agg) = drive(config.clone(), request(), models(false), &fake).await?;
    assert_eq!(selected, STUDENT);
    assert_eq!(completion_text(&agg), "prose without a step");
    assert_eq!(fake.targets(), [STUDENT], "retry = 0 asks once");

    let fake = Fake::new(vec![Err(transport_error())], vec![]);
    let result = drive(config, request(), models(false), &fake).await;
    assert!(matches!(result, Err(LibsyError::ClientCall { .. })));
    Ok(())
}

#[tokio::test]
async fn lenient_sentinel_is_cloud_at_serving_and_labelled_in_the_trace() -> Result<()> {
    let trace = TraceFile::new("lenient");
    let fenced = "```json\n{\"route\":\"cloud\"}\n```";
    let fake = Fake::new(vec![Ok(reply(fenced, 0))], vec![Ok(reply("Editing.", 1))]);

    let (selected, agg) = drive(trace.config(), request(), models(true), &fake).await?;

    assert_eq!(selected, STRONG);
    assert_eq!(completion_text(&agg), "{\"route\":\"cloud\"}\nEditing.");
    let record = trace.record();
    assert_eq!(
        record["student"]["attempts"][0]["classification"],
        "cloud_lenient"
    );
    assert_eq!(record["student"]["attempts"][0]["valid"], false);
    assert_eq!(record["decision"], "cloud");

    // With lenient matching off the same reply is invalid and served.
    let config = SelfDelegationConfig {
        lenient_cloud: false,
        on_invalid: OnInvalid {
            retry: 0,
            action: InvalidAction::Serve,
        },
        ..SelfDelegationConfig::default()
    };
    let fake = Fake::new(vec![Ok(reply(fenced, 0))], vec![]);
    let (selected, agg) = drive(config, request(), models(true), &fake).await?;
    assert_eq!(selected, STUDENT);
    assert_eq!(completion_text(&agg), fenced);
    Ok(())
}

#[tokio::test]
async fn cloud_line_with_tool_calls_is_served_as_local() -> Result<()> {
    let trace = TraceFile::new("cloud_tools");
    let fake = Fake::new(vec![Ok(reply(CLOUD_SENTINEL, 1))], vec![]);

    let (selected, agg) = drive(trace.config(), request(), models(true), &fake).await?;

    assert_eq!(selected, STUDENT);
    assert_eq!(completion_text(&agg), CLOUD_SENTINEL);
    assert_eq!(tool_calls(&agg), 1);
    assert_eq!(fake.targets(), [STUDENT]);
    let record = trace.record();
    assert_eq!(record["decision"], "local");
    assert_eq!(record["student"]["attempts"][0]["classification"], "local");
    assert_eq!(record["student"]["attempts"][0]["valid"], false);
    Ok(())
}

#[tokio::test]
async fn student_error_is_returned_or_escalated() -> Result<()> {
    let trace = TraceFile::new("student_error");
    let fake = Fake::new(vec![Err(transport_error())], vec![]);

    let result = drive(trace.config(), request(), models(true), &fake).await;

    assert!(matches!(
        result,
        Err(LibsyError::ClientCall { target, source: LlmClientError::Transport { .. } })
            if target == STUDENT
    ));
    assert_eq!(fake.targets(), [STUDENT]);
    let record = trace.record();
    assert_eq!(record["decision"], "invalid");
    let attempt = &record["student"]["attempts"][0];
    assert!(attempt["classification"].is_null());
    assert_eq!(attempt["valid"], false);
    assert!(
        attempt["error"]
            .as_str()
            .is_some_and(|error| error.contains("connection reset"))
    );
    assert!(record["strong"].is_null());
    assert!(record["served"].is_null());

    let trace = TraceFile::new("student_error_escalate");
    let config = SelfDelegationConfig {
        on_student_error: StudentErrorAction::Escalate,
        ..trace.config()
    };
    let fake = Fake::new(vec![Err(transport_error())], vec![Ok(reply("Editing.", 1))]);

    let (selected, agg) = drive(config, request(), models(true), &fake).await?;

    assert_eq!(selected, STRONG);
    assert_eq!(completion_text(&agg), "{\"route\":\"cloud\"}\nEditing.");
    assert_eq!(agg.usage.input_tokens, Some(10));
    let record = trace.record();
    assert_eq!(record["decision"], "invalid");
    assert!(record["student"]["attempts"][0]["error"].is_string());
    assert_eq!(record["strong"]["fallback"], true);
    assert_eq!(record["served"]["by"], "strong");
    Ok(())
}

#[tokio::test]
async fn strong_error_is_returned_and_traced() -> Result<()> {
    let trace = TraceFile::new("strong_error");
    let fake = Fake::new(
        vec![Ok(reply(CLOUD_SENTINEL, 0))],
        vec![Err(transport_error())],
    );

    let result = drive(trace.config(), request(), models(true), &fake).await;

    assert!(matches!(
        result,
        Err(LibsyError::ClientCall { target, .. }) if target == STRONG
    ));
    let record = trace.record();
    assert_eq!(record["decision"], "cloud");
    assert_eq!(record["strong"]["target"], STRONG);
    assert!(
        record["strong"]["error"]
            .as_str()
            .is_some_and(|error| error.contains("connection reset"))
    );
    assert!(record["served"].is_null());
    Ok(())
}

#[tokio::test]
async fn a_failed_trace_write_does_not_fail_the_request() -> Result<()> {
    let config = SelfDelegationConfig {
        trace_jsonl: Some(
            std::env::temp_dir()
                .join("self_delegation_missing_directory")
                .join("missing")
                .join("trace.jsonl"),
        ),
        ..SelfDelegationConfig::default()
    };
    let fake = Fake::new(vec![Ok(reply("{\"route\":\"local\"}\nok", 1))], vec![]);

    let (selected, agg) = drive(config, request(), models(true), &fake).await?;

    assert_eq!(selected, STUDENT);
    assert_eq!(completion_text(&agg), "{\"route\":\"local\"}\nok");
    Ok(())
}

#[tokio::test]
async fn the_trace_records_the_request_as_the_student_engine_receives_it() -> Result<()> {
    let trace = TraceFile::new("request");
    let models = RuntimeModels::new(models(true)).with_provider_models(HashMap::from([
        (ModelId::from(STUDENT), ModelId::from("edgewise-9b")),
        (ModelId::from(STRONG), ModelId::from("kimi-k3")),
    ]));
    let mut request = request();
    request.llm_request.stream = true;
    request.llm_request.tools = vec![
        ToolDefinition {
            name: "bash".to_string(),
            description: Some("Run a command".to_string()),
            parameters: json!({"type": "object"}),
            strict: None,
        }
        .into(),
    ];
    let fake = Fake::new(vec![Ok(reply("{\"route\":\"local\"}\nok", 1))], vec![]);

    drive(trace.config(), request, models, &fake).await?;

    let record = trace.record();
    let body = &record["request"];
    assert_eq!(body["model"], "edgewise-9b");
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "You are a coding agent.");
    assert_eq!(body["messages"][1]["role"], "user");
    let content = &body["messages"][1]["content"];
    assert!(
        content == "fix the parser" || content[0]["text"] == "fix the parser",
        "user content: {content}"
    );
    assert_eq!(body["tools"][0]["function"]["name"], "bash");
    assert!(body.get("stream").is_none(), "the student call is buffered");
    assert_eq!(record["student"]["target"], STUDENT);
    Ok(())
}

#[test]
fn route_line_requires_exactly_one_route_key() {
    assert_eq!(route_line("{\"route\":\"cloud\"}"), Some(RouteLine::Cloud));
    assert_eq!(
        route_line("  { \"route\": \"local\" }  "),
        Some(RouteLine::Local)
    );
    assert_eq!(route_line("{\"route\":\"cloud\",\"why\":\"hard\"}"), None);
    assert_eq!(route_line("{\"route\":\"remote\"}"), None);
    assert_eq!(route_line("{\"labels\":\"cloud\"}"), None);
    assert_eq!(route_line("[\"route\"]"), None);
    assert_eq!(route_line("{'route':'cloud'}"), None);
}

#[test]
fn usage_sums_counts_and_cache_details() {
    let mut total = Usage {
        input_tokens: Some(10),
        output_tokens: None,
        ..Usage::default()
    };
    let mut more = Usage {
        input_tokens: Some(5),
        output_tokens: Some(7),
        provider_cost: Some(0.5),
        ..Usage::default()
    };
    more.set_cached_input_tokens(3);
    add_usage(&mut total, &more);
    add_usage(&mut total, &more);
    assert_eq!(total.input_tokens, Some(20));
    assert_eq!(total.output_tokens, Some(14));
    assert_eq!(total.total_tokens, None);
    assert_eq!(total.provider_cost, Some(1.0));
    assert_eq!(total.cached_input_tokens(), Some(6));
}

#[derive(Deserialize)]
struct Fixtures {
    sentinel: String,
    local_line: String,
    cases: Vec<Case>,
    strip_cases: Vec<StripCase>,
    merge_cases: Vec<MergeCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    profile: String,
    content: String,
    tool_calls: usize,
    expect: Expect,
}

#[derive(Deserialize)]
struct Expect {
    valid: bool,
    route: String,
    lenient: bool,
}

#[derive(Deserialize)]
struct StripCase {
    name: String,
    content: String,
    stripped: String,
}

#[derive(Deserialize)]
struct MergeCase {
    name: String,
    strong_content: String,
    served: String,
    imitated_route_line: bool,
}

/// The fixture file shared with the Python exporter's validator.
fn fixtures() -> Fixtures {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/route_line_fixtures.json"
    ))
    .expect("fixture file parses")
}

#[test]
fn fixture_cases_match_classification_and_validity() {
    let fixtures = fixtures();
    assert_eq!(fixtures.sentinel, CLOUD_SENTINEL);
    assert_eq!(fixtures.local_line, LOCAL_LINE);
    let serving: Vec<_> = fixtures
        .cases
        .iter()
        .filter(|case| case.profile == "serving")
        .collect();
    assert!(serving.len() >= 18, "serving cases: {}", serving.len());
    for case in serving {
        let class = classify(&case.content, case.tool_calls, true);
        assert_eq!(class.decision(), case.expect.route, "{}: route", case.name);
        assert_eq!(
            class == Classification::CloudLenient,
            case.expect.lenient,
            "{}: lenient",
            case.name
        );
        assert_eq!(
            is_valid(&case.content, case.tool_calls),
            case.expect.valid,
            "{}: valid",
            case.name
        );
        if case.expect.lenient {
            assert_eq!(
                classify(&case.content, case.tool_calls, false),
                Classification::Invalid,
                "{}: strict serving",
                case.name
            );
        }
    }
}

#[test]
fn fixture_strip_cases_match() {
    for case in fixtures().strip_cases {
        let (stripped, _) = strip_route_line(&case.content);
        assert_eq!(stripped, case.stripped, "{}", case.name);
    }
}

#[test]
fn fixture_merge_cases_match() {
    for case in fixtures().merge_cases {
        let (served, imitated) = merge_cloud(&case.strong_content, false);
        assert_eq!(served, case.served, "{}: served", case.name);
        assert_eq!(
            imitated, case.imitated_route_line,
            "{}: imitated_route_line",
            case.name
        );
    }
}
