// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::json;
use switchyard_protocol::{
    ContentBlock, LlmRequest, Message, Role, ToolCall, ToolResult, WireFormat, text_response,
};

use super::*;
use crate::algorithms::util::stage::DECISION_SOURCE_KEY;
use crate::core::algorithm::{Algorithm, LlmTarget};
use crate::core::classifier::Score;
use crate::core::state::StateValue;
use switchyard_protocol::{Context, Decision, LlmResponse, Metadata, Response, RoutedLlmClient};

fn tier_target(name: &str) -> LlmTarget {
    LlmTarget {
        semantic_name: name.to_string(),
        llm_client: None,
    }
}

/// A classifier that always picks `target`, standing in for a cascade member.
struct Fixed(&'static str);

#[async_trait]
impl Classifier<State> for Fixed {
    async fn score(
        &self,
        _state: &mut State,
        _request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        Ok((
            Classification::Scores(vec![Score {
                target: self.0.to_string(),
                confidence: 1.0,
            }]),
            None,
        ))
    }
}

/// A classifier that never decides.
struct Abstains;

#[async_trait]
impl Classifier<State> for Abstains {
    async fn score(
        &self,
        _state: &mut State,
        _request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        Ok((Classification::Ambiguous(vec![]), None))
    }
}

async fn stamped(inner: Arc<dyn Classifier<State>>) -> Result<Option<String>> {
    let stamp = SourceStamp {
        inner,
        source: DecisionSource::LlmClassifier,
    };
    let mut state = State::default();
    stamp
        .score(&mut state, &mut Request::default(), None)
        .await?;
    Ok(match state.extra.get(DECISION_SOURCE_KEY) {
        Some(StateValue::String(source)) => Some(source.clone()),
        _ => None,
    })
}

#[tokio::test]
async fn a_deciding_classifier_is_credited_with_the_turn() -> Result<()> {
    assert_eq!(
        stamped(Arc::new(Fixed("strong"))).await?.as_deref(),
        Some("llm-classifier")
    );
    Ok(())
}

#[tokio::test]
async fn an_abstaining_classifier_claims_nothing() -> Result<()> {
    // It passed the turn on, so the next classifier is the one that decided.
    assert_eq!(stamped(Arc::new(Abstains)).await?, None);
    Ok(())
}

fn config() -> StageRouterConfig {
    StageRouterConfig::new(PickerMode::EfficientFirst, 0.5)
}

#[test]
fn rejects_an_out_of_range_confidence_threshold() {
    let mut config = config();
    config.confidence_threshold = 1.5;
    assert!(matches!(
        StageRouter::new(tier_target("strong"), tier_target("weak"), config),
        Err(LibsyError::AlgorithmError { .. })
    ));
}

#[test]
fn rejects_an_out_of_range_judge_threshold() {
    let mut config = config();
    config.llm_fallback = Some(LlmFallback {
        judge_target: LlmTarget {
            semantic_name: "judge".to_string(),
            llm_client: None,
        },
        config: TaskClassifierConfig {
            base_threshold: -0.1,
            ..Default::default()
        },
    });
    assert!(matches!(
        StageRouter::new(tier_target("strong"), tier_target("weak"), config),
        Err(LibsyError::AlgorithmError { .. })
    ));
}

#[test]
fn builds_over_both_tiers() -> Result<()> {
    let router = StageRouter::new(tier_target("strong"), tier_target("weak"), config())?;
    assert_eq!(router.name(), STAGE_ROUTER);
    Ok(())
}

// ── routing integration tests ────────────────────────────────────────────

const ESCALATION: &str = "the previous model was stalling; pick up the diagnosis";
const JUDGE: &str = "judge";

#[derive(Clone, Debug)]
struct Call {
    target: String,
    messages: Vec<String>,
}

/// Records what each target receives. When called as the judge target it
/// replies with a structured verdict so the fallback classifier gets an answer
/// without a real model.
#[derive(Default)]
struct RecordingClient {
    calls: Mutex<Vec<Call>>,
    judge_p_solve: Mutex<f64>,
}

impl RecordingClient {
    fn routed(&self) -> Vec<Call> {
        self.calls
            .lock()
            .iter()
            .filter(|call| call.target != JUDGE)
            .cloned()
            .collect()
    }
}

#[async_trait]
impl RoutedLlmClient for RecordingClient {
    async fn call(
        &self,
        _ctx: Context,
        request: Request,
        decision: Arc<dyn Decision>,
    ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
        let target = decision.selected_model().to_string();
        self.calls.lock().push(Call {
            target: target.clone(),
            messages: request
                .llm_request
                .messages
                .iter()
                .filter_map(|message| message.text_content("|"))
                .collect(),
        });
        let completion = if target == JUDGE {
            let p_solve = *self.judge_p_solve.lock();
            format!(
                r#"{{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":{p_solve}}}"#
            )
        } else {
            target
        };
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(None, completion)),
            metadata: None,
        })
    }
}

fn recording_target(client: &Arc<RecordingClient>, name: &str) -> LlmTarget {
    LlmTarget {
        semantic_name: name.to_string(),
        llm_client: Some(client.clone() as Arc<dyn RoutedLlmClient>),
    }
}

fn recording_router(
    client: Arc<RecordingClient>,
    config: StageRouterConfig,
) -> Result<Arc<StageRouter>> {
    Ok(Arc::new(StageRouter::new(
        recording_target(&client, "strong"),
        recording_target(&client, "weak"),
        config,
    )?))
}

fn config_with_notes() -> StageRouterConfig {
    let mut c = config();
    c.handoff_notes = Some(HandoffNoteConfig::new(ESCALATION, None, true));
    c
}

fn config_with_judge(client: &Arc<RecordingClient>, p_solve: f64) -> StageRouterConfig {
    *client.judge_p_solve.lock() = p_solve;
    let mut c = config();
    c.llm_fallback = Some(LlmFallback {
        judge_target: recording_target(client, JUDGE),
        config: TaskClassifierConfig {
            base_threshold: 0.5,
            recent_turn_window: Some(3),
            ..Default::default()
        },
    });
    c
}

fn turn_request(failed: bool) -> Request {
    let content = if failed {
        "fatal runtime error: out of memory"
    } else {
        "ok"
    };
    Request {
        llm_request: LlmRequest {
            model: Some("auto".to_string()),
            messages: vec![
                Message::text(Role::User, "fix the build"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall(ToolCall {
                        id: "call_1".to_string(),
                        name: "Bash".to_string(),
                        arguments: json!({"command": "cargo test"}),
                    })],
                },
                Message {
                    role: Role::Tool,
                    content: vec![ContentBlock::ToolResult(ToolResult {
                        tool_call_id: "call_1".to_string(),
                        content: vec![ContentBlock::Text {
                            text: content.to_string(),
                        }],
                        is_error: Some(failed),
                    })],
                },
            ],
            ..LlmRequest::default()
        },
        raw_request: Some(json!({
            "model": "auto",
            "messages": [
                {"role": "user", "content": "fix the build"},
                {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "Bash", "arguments": "{\"command\": \"cargo test\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": content},
            ],
        })),
        metadata: Some(Metadata {
            wire_format: Some(WireFormat::OpenAiChat),
            session_id: Some("session-1".to_string()),
            ..Default::default()
        }),
    }
}

#[tokio::test]
async fn a_signal_driven_escalation_hands_the_note_to_the_model() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = recording_router(client.clone(), config_with_notes())?;
    let ctx = Context::default();

    router.clone().run(ctx.clone(), turn_request(false)).await?;
    router.run(ctx, turn_request(true)).await?;

    let calls = client.routed();
    assert_eq!(calls[0].target, "weak");
    assert_eq!(calls[1].target, "strong");
    assert!(
        !calls[0].messages.iter().any(|t| t.contains(ESCALATION)),
        "steady-state turn should carry no note: {:?}",
        calls[0].messages
    );
    assert!(
        calls[1]
            .messages
            .last()
            .is_some_and(|t| t.ends_with(ESCALATION)),
        "escalating turn should carry the note last: {:?}",
        calls[1].messages
    );
    Ok(())
}

#[tokio::test]
async fn the_judge_decides_a_turn_the_signals_leave_undecided() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = recording_router(client.clone(), config_with_judge(&client, 0.1))?;

    router.run(Context::default(), turn_request(false)).await?;

    assert!(
        client.calls.lock().iter().any(|c| c.target == JUDGE),
        "the judge should be consulted on an undecided turn"
    );
    assert_eq!(client.routed()[0].target, "strong");
    Ok(())
}

#[tokio::test]
async fn a_decisive_signal_never_reaches_the_judge() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = recording_router(client.clone(), config_with_judge(&client, 0.9))?;

    router.run(Context::default(), turn_request(true)).await?;

    assert!(
        !client.calls.lock().iter().any(|c| c.target == JUDGE),
        "a resolved turn should not pay for a judge call"
    );
    assert_eq!(client.routed()[0].target, "strong");
    Ok(())
}

#[tokio::test]
async fn the_judges_verdict_is_not_pinned_to_the_session() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = recording_router(client.clone(), config_with_judge(&client, 0.1))?;
    let ctx = Context::default();

    router.clone().run(ctx.clone(), turn_request(false)).await?;
    *client.judge_p_solve.lock() = 0.9;
    router.run(ctx, turn_request(false)).await?;

    let routed = client.routed();
    assert_eq!(routed[0].target, "strong");
    assert_eq!(routed[1].target, "weak");
    assert_eq!(
        client
            .calls
            .lock()
            .iter()
            .filter(|c| c.target == JUDGE)
            .count(),
        2,
        "each undecided turn is its own question"
    );
    Ok(())
}

#[tokio::test]
async fn a_judge_that_cannot_tell_lands_on_the_picker_default() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = recording_router(client.clone(), config_with_judge(&client, 42.0))?;

    router.run(Context::default(), turn_request(false)).await?;

    assert_eq!(client.routed()[0].target, "weak");
    Ok(())
}

#[tokio::test]
async fn the_judge_reads_the_window_it_was_configured_with() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = recording_router(client.clone(), config_with_judge(&client, 0.9))?;

    router.run(Context::default(), turn_request(false)).await?;

    let judged = client
        .calls
        .lock()
        .iter()
        .find(|c| c.target == JUDGE)
        .map(|c| c.messages.join("|"));
    let Some(judged) = judged else {
        panic!("the judge was never called");
    };
    assert!(
        judged.contains("fix the build"),
        "the judge should see the opening task: {judged}"
    );
    Ok(())
}
