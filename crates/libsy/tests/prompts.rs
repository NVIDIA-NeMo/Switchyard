// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the request-text helpers on a running cascade.
//!
//! The unit tests cover placement in isolation. These drive whole turns through
//! a plain [`FallThrough`] — no stage routing — to pin the two things only a
//! running composition can show: that the text survives to the model call, and
//! that it follows the target the cascade settled on rather than the classifier
//! that named it.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use switchyard_libsy::algorithms::{
    append_note, DefaultTarget, FallThrough, SystemPromptProcessor, TargetPrompts,
};
use switchyard_libsy::{
    Algorithm, Classification, Classifier, Context, Decision, Driver, Event, LlmResponse,
    LlmTarget, LlmTargetSet, Processor, Request, Response, Result, RoutedLlmClient,
};
use switchyard_protocol::{text_request, text_response};

const CAPABLE_PROMPT: &str = "diagnose before you edit";
const EFFICIENT_PROMPT: &str = "follow the settled plan";
const NOTE: &str = "the previous model was stalling";

/// One model call as the client saw it.
#[derive(Clone, Debug, Default)]
struct Call {
    target: String,
    messages: Vec<String>,
    instructions: Vec<String>,
}

/// Records what the model was handed, so a test asserts on that rather than on
/// composition internals.
#[derive(Default)]
struct RecordingClient(Mutex<Option<Call>>);

#[async_trait]
impl RoutedLlmClient for RecordingClient {
    async fn call(
        &self,
        _ctx: Context,
        request: Request,
        decision: Arc<dyn Decision>,
    ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
        *self.0.lock() = Some(Call {
            target: decision.selected_model().to_string(),
            messages: request
                .llm_request
                .messages
                .iter()
                .filter_map(|message| message.text_content("|"))
                .collect(),
            instructions: request
                .llm_request
                .instructions
                .iter()
                .filter_map(|block| block.content.iter().find_map(text_of))
                .collect(),
        });
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(None, decision.selected_model())),
            metadata: None,
        })
    }
}

fn text_of(block: &switchyard_protocol::ContentBlock) -> Option<String> {
    match block {
        switchyard_protocol::ContentBlock::Text { text } => Some(text.clone()),
        _ => None,
    }
}

/// A classifier that never decides, so the next one in the cascade gets the turn.
struct Abstains;

#[async_trait]
impl Classifier for Abstains {
    async fn score(
        &self,
        _state: &mut (),
        _request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        Ok((Classification::Ambiguous(Vec::new()), None))
    }
}

/// Appends a note to every outbound request, the way a router would on a turn
/// it wants to explain.
struct Noting;

#[async_trait]
impl Processor for Noting {
    async fn process(&self, _state: &mut (), event: Event<'_>) -> Result<()> {
        if let Event::Decision { request, .. } = event {
            append_note(request, NOTE);
        }
        Ok(())
    }
}

fn targets(client: &Arc<RecordingClient>, names: &[&str]) -> LlmTargetSet {
    LlmTargetSet::new(
        names
            .iter()
            .map(|name| LlmTarget {
                semantic_name: (*name).to_string(),
                llm_client: Some(client.clone() as Arc<dyn RoutedLlmClient>),
            })
            .collect(),
    )
}

fn prompts() -> TargetPrompts {
    TargetPrompts::default()
        .with("capable", CAPABLE_PROMPT)
        .with("efficient", EFFICIENT_PROMPT)
}

/// Runs one turn on a cascade that always routes to `target`.
async fn routed_to(client: &Arc<RecordingClient>, router: FallThrough) -> Result<Call> {
    Arc::new(router)
        .run(
            Context::default(),
            Request {
                llm_request: text_request(Some("auto".to_string()), "fix the build"),
                raw_request: None,
                metadata: None,
            },
        )
        .await?;
    let call = client.0.lock().take();
    match call {
        Some(call) => Ok(call),
        None => panic!("the model was never called"),
    }
}

/// A cascade that routes to `target` and applies `prompts`.
fn router(client: &Arc<RecordingClient>, target: &str, prompts: TargetPrompts) -> FallThrough {
    FallThrough::new(targets(client, &["capable", "efficient"]))
        .with_processor(Arc::new(SystemPromptProcessor::new(prompts)))
        .with_classifier(Arc::new(DefaultTarget::new(target)))
}

#[tokio::test]
async fn each_target_gets_its_own_prompt() -> Result<()> {
    for (target, expected) in [("capable", CAPABLE_PROMPT), ("efficient", EFFICIENT_PROMPT)] {
        let client = Arc::new(RecordingClient::default());
        let call = routed_to(&client, router(&client, target, prompts())).await?;
        assert_eq!(call.target, target);
        assert_eq!(call.instructions, vec![expected.to_string()]);
    }
    Ok(())
}

#[tokio::test]
async fn a_target_with_no_prompt_is_left_untouched() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let only_capable = TargetPrompts::default().with("capable", CAPABLE_PROMPT);

    let call = routed_to(&client, router(&client, "efficient", only_capable)).await?;

    assert!(
        call.instructions.is_empty(),
        "one target's prompt must not leak onto another: {:?}",
        call.instructions
    );
    Ok(())
}

#[tokio::test]
async fn the_prompt_follows_the_target_whichever_classifier_picked_it() -> Result<()> {
    // The first classifier abstains, so the second decides — the prompt still
    // has to follow the target the cascade settled on.
    let client = Arc::new(RecordingClient::default());
    let router = FallThrough::new(targets(&client, &["capable", "efficient"]))
        .with_processor(Arc::new(SystemPromptProcessor::new(prompts())))
        .with_classifier(Arc::new(Abstains))
        .with_classifier(Arc::new(DefaultTarget::new("capable")));

    let call = routed_to(&client, router).await?;

    assert_eq!(call.target, "capable");
    assert_eq!(call.instructions, vec![CAPABLE_PROMPT.to_string()]);
    Ok(())
}

#[tokio::test]
async fn a_note_reaches_the_model_in_the_conversation() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = FallThrough::new(targets(&client, &["capable", "efficient"]))
        .with_processor(Arc::new(Noting))
        .with_classifier(Arc::new(DefaultTarget::new("capable")));

    let call = routed_to(&client, router).await?;

    // Joined onto the trailing user turn rather than opening one of its own.
    assert_eq!(call.messages, vec![format!("fix the build|{NOTE}")]);
    assert!(call.instructions.is_empty(), "a note is not an instruction");
    Ok(())
}
