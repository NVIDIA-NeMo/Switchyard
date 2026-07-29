// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for a running [`StageRouter`].
//!
//! The unit tests cover scoring, note selection and prompt placement in
//! isolation. These drive whole turns through the public API to pin what only
//! the assembled cascade can show: which tier served the turn, what the *model*
//! was handed, and which step of the cascade decided.
//!
//! The per-target prompts have their own integration tests in `prompts.rs`,
//! since they are not stage-router-specific.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::json;

use switchyard_libsy::algorithms::{StageRouter, TaskClassifierConfig};
use switchyard_libsy::stage_router::{
    HandoffNoteConfig, LlmFallback, PickerMode, StageRouterConfig,
};
use switchyard_libsy::{
    Algorithm, Context, Decision, LlmResponse, LlmTarget, Metadata, Request, Response, Result,
    RoutedLlmClient,
};
use switchyard_protocol::{
    text_response, ContentBlock, LlmRequest, Message, Role, ToolCall, ToolResult, WireFormat,
};

const ESCALATION: &str = "the previous model was stalling; pick up the diagnosis";
/// Semantic name of the judge target. It is called through its own target and is
/// never a routing destination.
const JUDGE: &str = "judge";

/// One model call as the client saw it.
#[derive(Clone, Debug)]
struct Call {
    /// Target the call was routed to.
    target: String,
    /// Text of each message, so a test can assert on the note.
    messages: Vec<String>,
}

/// A client that records what each target was handed, so a test can assert on
/// what reached the model rather than on router internals.
///
/// It also plays the judge: a call routed to the judge target answers with a
/// verdict, which is how the fallback classifier gets an answer without a real
/// model.
#[derive(Default)]
struct RecordingClient {
    calls: Mutex<Vec<Call>>,
    /// `p_solve` the judge reports. High keeps the turn on the weak tier.
    judge_p_solve: Mutex<f64>,
}

impl RecordingClient {
    /// The calls that routed to a tier, dropping the judge's own.
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
                r#"{{"recommended_route":"efficient","p_solve":{p_solve},"confidence":0.9,"abstain":false,"capability_boundary":"supported","primary_rule":"SUP-1","crux":"bounded task"}}"#
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

fn target(client: &Arc<RecordingClient>, name: &str) -> LlmTarget {
    LlmTarget {
        semantic_name: name.to_string(),
        llm_client: Some(client.clone() as Arc<dyn RoutedLlmClient>),
    }
}

fn router(client: Arc<RecordingClient>, config: StageRouterConfig) -> Result<Arc<StageRouter>> {
    // Only the two tiers are routing destinations; the judge has its own target.
    Ok(Arc::new(StageRouter::new(
        target(&client, "strong"),
        target(&client, "weak"),
        config,
    )?))
}

/// The signal-only configuration these tests start from.
fn config() -> StageRouterConfig {
    StageRouterConfig::new(PickerMode::EfficientFirst, 0.5)
}

/// That configuration plus handoff notes.
fn config_with_notes() -> StageRouterConfig {
    let mut config = config();
    config.handoff_notes = Some(HandoffNoteConfig::new(ESCALATION, None, true));
    config
}

/// One turn of a coding-agent conversation, as the wire delivers it: an
/// assistant tool call answered by a tool result the signal extractor reads.
///
/// `failed` makes the tool result a critical error, the hard override that
/// escalates a turn on the signal alone.
fn turn_request(failed: bool) -> Request {
    let tool_call = json!({
        "role": "assistant",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "Bash", "arguments": "{\"command\": \"cargo test\"}"},
        }],
    });
    let content = if failed {
        "fatal runtime error: out of memory"
    } else {
        "ok"
    };
    let raw_request = json!({
        "model": "auto",
        "messages": [
            {"role": "user", "content": "fix the build"},
            tool_call,
            {"role": "tool", "tool_call_id": "call_1", "content": content},
        ],
    });
    // The neutral IR the router forwards, alongside the raw body the signal
    // extractor parses. An OpenAI tool result is its own `tool` turn, so a note
    // is appended as a fresh user message rather than folded into it.
    let messages = vec![
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
    ];
    Request {
        llm_request: LlmRequest {
            model: Some("auto".to_string()),
            messages,
            ..LlmRequest::default()
        },
        raw_request: Some(raw_request),
        metadata: Some(Metadata {
            wire_format: Some(WireFormat::OpenAiChat),
            // One session, so turns of the same test share the router's state.
            session_id: Some("session-1".to_string()),
            ..Default::default()
        }),
    }
}

#[tokio::test]
async fn a_signal_driven_escalation_hands_the_note_to_the_model() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = router(client.clone(), config_with_notes())?;
    let ctx = Context::default();

    // Turn 1 — a clean tool result is under threshold and falls open to weak.
    router.clone().run(ctx.clone(), turn_request(false)).await?;
    // Turn 2 — a critical tool error escalates on the signals alone.
    router.run(ctx, turn_request(true)).await?;

    let calls = client.routed();
    assert_eq!(calls[0].target, "weak");
    assert_eq!(calls[1].target, "strong");
    assert!(
        !calls[0]
            .messages
            .iter()
            .any(|text| text.contains(ESCALATION)),
        "the steady-state turn should carry no note: {:?}",
        calls[0].messages
    );
    // The note rides on the trailing turn, after the tool result.
    assert!(
        calls[1]
            .messages
            .last()
            .is_some_and(|text| text.ends_with(ESCALATION)),
        "the escalating turn should carry the note last: {:?}",
        calls[1].messages
    );
    Ok(())
}

/// That configuration plus the capability judge behind the signals.
fn config_with_judge(client: &Arc<RecordingClient>, p_solve: f64) -> StageRouterConfig {
    *client.judge_p_solve.lock() = p_solve;
    let mut config = config();
    config.llm_fallback = Some(LlmFallback {
        judge_target: target(client, JUDGE),
        config: TaskClassifierConfig {
            base_threshold: 0.5,
            // The same span the signal scorer reads.
            recent_turn_window: Some(3),
            ..Default::default()
        },
    });
    config
}

#[tokio::test]
async fn the_judge_decides_a_turn_the_signals_leave_undecided() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    // A low p_solve means the judge does not trust the weak tier with the task.
    let router = router(client.clone(), config_with_judge(&client, 0.1))?;

    // A clean turn is under threshold, so without a judge it would fall open to
    // the configured weak default. The judge overrides that.
    router.run(Context::default(), turn_request(false)).await?;

    let judged = client.calls.lock().iter().any(|call| call.target == JUDGE);
    assert!(judged, "the judge should be consulted on an undecided turn");
    assert_eq!(client.routed()[0].target, "strong");
    Ok(())
}

#[tokio::test]
async fn a_decisive_signal_never_reaches_the_judge() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    // The judge would say weak; the signals must win before it is asked at all.
    let router = router(client.clone(), config_with_judge(&client, 0.9))?;

    router.run(Context::default(), turn_request(true)).await?;

    assert!(
        !client.calls.lock().iter().any(|call| call.target == JUDGE),
        "a resolved turn should not pay for a judge call"
    );
    assert_eq!(client.routed()[0].target, "strong");
    Ok(())
}

#[tokio::test]
async fn the_judges_verdict_is_not_pinned_to_the_session() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = router(client.clone(), config_with_judge(&client, 0.1))?;
    let ctx = Context::default();

    // First undecided turn: the judge sends it to the strong tier.
    router.clone().run(ctx.clone(), turn_request(false)).await?;
    // The judge changes its mind; a second undecided turn asks again rather than
    // replaying the first verdict.
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
            .filter(|call| call.target == JUDGE)
            .count(),
        2,
        "each undecided turn is its own question"
    );
    Ok(())
}

#[tokio::test]
async fn a_judge_that_cannot_tell_lands_on_the_picker_default() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    // An out-of-range p_solve is an unusable verdict, so the judge abstains.
    let router = router(client.clone(), config_with_judge(&client, 42.0))?;

    router.run(Context::default(), turn_request(false)).await?;

    // efficient_first, so the turn falls open to weak rather than being pushed
    // to strong by the judge's own fallback.
    assert_eq!(client.routed()[0].target, "weak");
    Ok(())
}

#[tokio::test]
async fn the_judge_reads_the_window_it_was_configured_with() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    // An undecided turn, so the judge is consulted and we can see what it got.
    let router = router(client.clone(), config_with_judge(&client, 0.9))?;

    router.run(Context::default(), turn_request(false)).await?;

    // With a window the judge sees the opening task, not just the newest turn.
    let judged = client
        .calls
        .lock()
        .iter()
        .find(|call| call.target == JUDGE)
        .map(|call| call.messages.join("|"));
    let Some(judged) = judged else {
        panic!("the judge was never called");
    };
    assert!(
        judged.contains("fix the build"),
        "the judge should see the opening task: {judged}"
    );
    Ok(())
}
