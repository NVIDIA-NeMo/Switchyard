// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One local routing decision over stdin/stdout for the Claude Code mod.

use std::{
    collections::hash_map::DefaultHasher,
    env, fs,
    hash::{Hash, Hasher},
    io,
    process::ExitCode,
};

use libsy::SessionCheckpoint;
use serde::{Deserialize, Serialize};
use serde_json::json;
use switchyard_protocol::{LlmRequest, Message, Metadata, ModelId, Request};
use switchyard_runner::Runner;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    version: u32,
    session_id: String,
    decision_id: String,
    messages: Vec<Message>,
    checkpoint: Option<Checkpoint>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    version: u32,
    session_id: String,
    config: String,
    state: SessionCheckpoint,
    last: LastDecision,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LastDecision {
    id: String,
    input: String,
    target: String,
    model: String,
}

// Identity checks, not authentication. A different build's hash simply starts
// fresh state; the checkpoint schema is versioned separately.
fn fingerprint(value: &[u8]) -> String {
    let mut hash = DefaultHasher::new();
    value.hash(&mut hash);
    format!("{:016x}", hash.finish())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    match decide().await {
        Ok(decision) => {
            println!("{decision}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

async fn decide() -> Result<serde_json::Value, String> {
    let config = env::args().nth(1).ok_or("Expected a TOML config path")?;
    let source = fs::read_to_string(config).map_err(|error| error.to_string())?;
    let input: Input =
        serde_json::from_reader(io::stdin().lock()).map_err(|_| "Invalid routing input")?;
    select(&source, input).await
}

async fn select(source: &str, input: Input) -> Result<serde_json::Value, String> {
    if input.version != 1 || input.session_id.is_empty() || input.decision_id.is_empty() {
        return Err("Invalid routing input version or identity".into());
    }
    let config = fingerprint(source.as_bytes());
    let input_hash = fingerprint(&serde_json::to_vec(&input.messages).map_err(|e| e.to_string())?);
    let runner =
        Runner::from_toml_for_model_selection(source).map_err(|error| error.to_string())?;
    let route = runner
        .route("switchyard")
        .ok_or("Missing switchyard route")?;
    let mut reset = false;
    if let Some(checkpoint) = &input.checkpoint {
        if checkpoint.version != 1 || checkpoint.session_id != input.session_id {
            return Err("Checkpoint version or session does not match".into());
        }
        if checkpoint.config == config {
            route
                .restore_session(&input.session_id, &checkpoint.state)
                .await
                .map_err(|e| e.to_string())?;
            if checkpoint.last.id == input.decision_id && checkpoint.last.input == input_hash {
                return Ok(
                    json!({"version": 1, "session_id": input.session_id, "decision_id": input.decision_id,
                    "model": checkpoint.last.model, "target": checkpoint.last.target, "checkpoint": checkpoint}),
                );
            }
        } else {
            reset = true;
        }
    }
    let request = Request {
        llm_request: LlmRequest {
            model: Some("switchyard".into()),
            messages: input.messages,
            ..Default::default()
        },
        metadata: Some(Metadata {
            session_id: Some(input.session_id.clone()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut expected = request.llm_request.clone();
    let outcome = route
        .decide(request)
        .await
        .map_err(|error| error.execution_error_summary().kind.as_str().to_owned())?;
    expected.model = outcome.request.llm_request.model.clone();
    if outcome.response.is_some() || outcome.request.llm_request != expected {
        return Err("Routing must only select a model".into());
    }
    let decision = runner
        .describe_decision(&ModelId::from("switchyard"), &outcome)
        .ok_or("Routing selected an unresolved target")?;
    let checkpoint = Checkpoint {
        version: 1,
        session_id: input.session_id.clone(),
        config,
        state: route
            .checkpoint_session(&input.session_id)
            .await
            .map_err(|e| e.to_string())?,
        last: LastDecision {
            id: input.decision_id.clone(),
            input: input_hash,
            target: decision.selected.target.clone(),
            model: decision.selected.model.to_string(),
        },
    };
    Ok(
        json!({ "version": 1, "session_id": input.session_id, "decision_id": input.decision_id,
        "target": decision.selected.target, "model": decision.selected.model,
        "checkpoint": checkpoint, "state_reset": reset }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = include_str!("../../../examples/claude/switchyard.toml");

    fn prompt() -> Vec<Message> {
        serde_json::from_value(
            json!([{ "role": "user", "content": [{ "type": "text", "text": "Fix the parser" }] }]),
        )
        .unwrap()
    }

    fn tool(messages: &mut Vec<Message>, id: &str, name: &str, text: &str, failed: bool) {
        let items: Vec<Message> = serde_json::from_value(json!([
            {"role": "assistant", "content": [{"type": "tool_call", "id": id, "name": name, "arguments": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_call_id": id,
                "content": [{"type": "text", "text": text}], "is_error": failed}]}
        ])).unwrap();
        messages.extend(items);
    }

    fn input(messages: Vec<Message>, step: usize, checkpoint: Option<Checkpoint>) -> Input {
        Input {
            version: 1,
            session_id: "main".into(),
            decision_id: step.to_string(),
            messages,
            checkpoint,
        }
    }

    fn checkpoint(result: &serde_json::Value) -> Checkpoint {
        serde_json::from_value(result["checkpoint"].clone()).unwrap()
    }

    #[tokio::test]
    async fn checkpointed_helpers_match_persistent_runner_across_repeated_cycles() {
        let runner = Runner::from_toml_for_model_selection(CONFIG).unwrap();
        let route = runner.route("switchyard").unwrap();
        let mut histories = vec![prompt()];
        let mut messages = prompt();
        for cycle in 0..3 {
            tool(
                &mut messages,
                &format!("failure-{cycle}"),
                "Bash",
                "out of memory",
                true,
            );
            histories.push(messages.clone());
            for i in 0..3 {
                tool(
                    &mut messages,
                    &format!("write-{cycle}-{i}"),
                    "Write",
                    "wrote file",
                    false,
                );
            }
            // One batch ages the error out. The two-request capable hold must
            // survive fresh helper processes even though the transcript is clean.
            for _ in 0..3 {
                histories.push(messages.clone());
            }
        }
        // A compacted/replaced transcript recomputes signals, retaining the same
        // algorithm state as the persistent host rather than replaying old tools.
        histories.push(prompt());
        let mut saved = None;
        let mut selected = Vec::new();
        for (step, messages) in histories.into_iter().enumerate() {
            let request = Request {
                llm_request: LlmRequest {
                    model: Some("switchyard".into()),
                    messages: messages.clone(),
                    ..Default::default()
                },
                metadata: Some(Metadata {
                    session_id: Some("main".into()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let expected = route.decide(request).await.unwrap();
            let actual = select(CONFIG, input(messages.clone(), step, saved))
                .await
                .unwrap();
            assert_eq!(
                actual["model"].as_str(),
                expected.request.llm_request.model.as_deref()
            );
            let state = route.checkpoint_session("main").await.unwrap();
            assert_eq!(
                actual["checkpoint"]["state"],
                serde_json::to_value(state).unwrap()
            );
            selected.push(actual["model"].as_str().unwrap().to_string());
            // A replay must return exactly the same decision and checkpoint.
            let repeated = select(CONFIG, input(messages, step, Some(checkpoint(&actual))))
                .await
                .unwrap();
            assert_eq!(repeated["checkpoint"], actual["checkpoint"]);
            saved = Some(checkpoint(&actual));
        }
        assert_eq!(selected[0], "claude-sonnet-5-5");
        for cycle in 0..3 {
            assert_eq!(
                &selected[1 + cycle * 4..5 + cycle * 4],
                &[
                    "claude-opus-5-5",
                    "claude-opus-5-5",
                    "claude-opus-5-5",
                    "claude-sonnet-5-5"
                ]
            );
        }
    }

    #[tokio::test]
    async fn rejects_foreign_or_unknown_checkpoints_and_resets_changed_config() {
        let mut messages = prompt();
        tool(&mut messages, "failure", "Bash", "out of memory", true);
        let original = select(CONFIG, input(messages, 0, None)).await.unwrap();
        let mut saved = checkpoint(&original);
        saved.session_id = "another-session".into();
        assert!(
            select(CONFIG, input(prompt(), 1, Some(saved)))
                .await
                .unwrap_err()
                .contains("session")
        );
        let mut saved = checkpoint(&original);
        saved.state.version = 99;
        assert!(
            select(CONFIG, input(prompt(), 1, Some(saved)))
                .await
                .unwrap_err()
                .contains("Incompatible")
        );
        let result = select(
            &format!("{CONFIG}\n# changed configuration\n"),
            input(prompt(), 1, Some(checkpoint(&original))),
        )
        .await
        .unwrap();
        assert_eq!(result["state_reset"], true);
        assert_eq!(result["model"], "claude-sonnet-5-5");
        let cold = select(CONFIG, input(prompt(), 1, None)).await.unwrap();
        assert_eq!(cold["model"], "claude-sonnet-5-5");
    }
}
