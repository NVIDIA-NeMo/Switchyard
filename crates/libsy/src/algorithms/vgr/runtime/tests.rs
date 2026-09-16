// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use switchyard_protocol::{
    LlmClientError, LlmResponse, LlmResponseChunk, LlmResponseStreamEvent, ModelId, Request,
    Response, completion_text, text_request,
};

use super::super::config::{ServingMode, VgrConfig};
use super::super::safety::{KillSwitch, endpoint_failure};
use super::Vgr;
use crate::core::algorithm::Algorithm;
use crate::core::testing::{reply, test_drive};
use crate::{LibsyError, Result};

fn request() -> Request {
    Request {
        llm_request: text_request(Some("auto".into()), "complete the task"),
        ..Default::default()
    }
}

fn runtime(mode: ServingMode) -> Result<Arc<dyn Algorithm>> {
    let mut config = VgrConfig::new(ModelId::new("local"), ModelId::new("cloud"));
    config.targets.judge = Some(ModelId::new("judge"));
    config.mode = mode;
    config.task_typing = false;
    Ok(Arc::new(Vgr::new(config)?))
}

#[tokio::test]
async fn verified_attempt_is_served_and_failed_verification_escalates() -> Result<()> {
    let local = runtime(ServingMode::Evaluate)?;
    let (selected, response) = test_drive(
        local,
        request(),
        |target: ModelId, call: Request| async move {
            Ok(
                match (target.as_str(), call.llm_request.output.max_output_tokens) {
                    ("local", _) => reply("local attempt"),
                    ("judge", Some(512)) => reply("yes"),
                    ("judge", _) => reply("unscored"),
                    ("cloud", _) => reply("cloud answer"),
                    _ => unreachable!(),
                },
            )
        },
    )
    .await?;
    assert_eq!(selected, "local");
    assert_eq!(
        completion_text(
            &response
                .llm_response
                .into_agg()
                .await
                .expect("buffered local response")
        ),
        "local attempt"
    );

    let cloud = runtime(ServingMode::Evaluate)?;
    let (selected, response) = test_drive(cloud, request(), |target: ModelId, _| async move {
        Ok(match target.as_str() {
            "local" => reply("unverified attempt"),
            "judge" => reply("no"),
            "cloud" => reply("cloud answer"),
            _ => unreachable!(),
        })
    })
    .await?;
    assert_eq!(selected, "cloud");
    assert_eq!(
        completion_text(
            &response
                .llm_response
                .into_agg()
                .await
                .expect("buffered cloud response")
        ),
        "cloud answer"
    );
    Ok(())
}

#[tokio::test]
async fn deadline_and_kill_switch_skip_to_cloud() -> Result<()> {
    let mut config = VgrConfig::new(ModelId::new("local"), ModelId::new("cloud"));
    config.mode = ServingMode::Evaluate;
    config.deadline = Duration::from_millis(1);
    config.task_typing = false;
    let timed: Arc<dyn Algorithm> = Arc::new(Vgr::new(config)?);
    let (selected, _) = test_drive(timed, request(), |target: ModelId, _| async move {
        if target == "local" {
            let delayed = futures::stream::once(async {
                tokio::time::sleep(Duration::from_millis(20)).await;
                Ok(LlmResponseStreamEvent::new(vec![
                    LlmResponseChunk::TextDelta {
                        index: 0,
                        text: "late".into(),
                    },
                ]))
            });
            return Ok(Response {
                llm_response: LlmResponse::Stream(Box::pin(delayed)),
                metadata: None,
                upstream_headers: Default::default(),
            });
        }
        Ok(reply(target.to_string()))
    })
    .await?;
    assert_eq!(selected, "cloud");

    let stop = KillSwitch::new();
    stop.engage();
    let mut config = VgrConfig::new(ModelId::new("local"), ModelId::new("cloud"));
    config.mode = ServingMode::Evaluate;
    config.kill_switch = Some(stop);
    let stopped: Arc<dyn Algorithm> = Arc::new(Vgr::new(config)?);
    let (selected, _) = test_drive(stopped, request(), |target: ModelId, _| async move {
        Ok(reply(target.to_string()))
    })
    .await?;
    assert_eq!(selected, "cloud");
    Ok(())
}

#[tokio::test]
async fn locally_committed_stream_is_replayed_as_a_stream() -> Result<()> {
    let runtime = runtime(ServingMode::Evaluate)?;
    let (selected, response) = test_drive(
        runtime,
        request(),
        |target: ModelId, call: Request| async move {
            if target == "local" {
                let event = LlmResponseStreamEvent::new(vec![LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "streamed attempt".into(),
                }]);
                return Ok(Response {
                    llm_response: LlmResponse::Stream(Box::pin(futures::stream::iter([Ok(event)]))),
                    metadata: None,
                    upstream_headers: Default::default(),
                });
            }
            Ok(match call.llm_request.output.max_output_tokens {
                Some(512) => reply("yes"),
                _ => reply("unscored"),
            })
        },
    )
    .await?;
    assert_eq!(selected, "local");
    assert!(matches!(response.llm_response, LlmResponse::Stream(_)));
    assert_eq!(
        completion_text(
            &response
                .llm_response
                .into_agg()
                .await
                .expect("replayed local response")
        ),
        "streamed attempt"
    );
    Ok(())
}

#[test]
fn only_unavailability_affects_endpoint_health() {
    let call = |source| LibsyError::client_call(ModelId::new("local"), source);
    assert!(!endpoint_failure(&call(
        LlmClientError::ContextWindowExceeded {
            model: ModelId::new("local"),
            message: "too long".into(),
        }
    )));
    assert!(!endpoint_failure(&call(LlmClientError::InvalidRequest {
        message: "bad request".into(),
    })));
    assert!(endpoint_failure(&call(LlmClientError::Timeout {
        source: std::io::Error::other("timed out").into(),
    })));
}
