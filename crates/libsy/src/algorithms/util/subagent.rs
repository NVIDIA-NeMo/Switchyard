// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sub-agent override as a single SDK component.
//!
//! [`SubagentOverride`] scores one fixed worker target for requests carrying delegated
//! sub-agent work ([`Metadata::is_subagent_work`]) and abstains for everything else, so a
//! cascade falls through to its later classifiers on ordinary traffic.
//!
//! It is stateless and holds only the worker's *name*: the fall-through cascade resolves it
//! against its
//! target set. Keeping the policy independent of any memory of past decisions is what lets
//! it compose with a stateful classifier such as
//! [`AffinityRouter`](crate::algorithms::AffinityRouter) — the override decides *which*
//! target delegated work belongs on, affinity decides *how long* a decision lives, and
//! neither needs to know about the other.

use std::sync::Arc;

use async_trait::async_trait;

use crate::Result;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier, Score};
use switchyard_protocol::{
    ContentBlock, LlmRequest, Message, Metadata, ModelId, Request, Response, Role,
};

/// Classifies delegated work from its parent-supplied prompt and abstains otherwise.
///
/// The inner classifier receives a prompt-only request clone. The original request remains
/// unchanged for the selected child model.
pub struct SubagentGate<S> {
    inner: Arc<dyn Classifier<S>>,
}

impl<S> SubagentGate<S> {
    /// Wraps `inner` with delegated-work detection.
    pub fn new(inner: Arc<dyn Classifier<S>>) -> Self {
        Self { inner }
    }
}

/// Builds the prompt-only request shown to a delegated-work classifier.
fn delegated_prompt_request(request: &Request) -> Option<Request> {
    // Coding harnesses append the parent's task after their injected user context and reminders.
    let prompt = request
        .llm_request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)?
        .content
        .iter()
        .rev()
        .find_map(|block| match block {
            ContentBlock::Text { text } if !text.trim().is_empty() => Some(text.clone()),
            _ => None,
        })?;

    Some(Request {
        llm_request: LlmRequest {
            model: request.llm_request.model.clone(),
            messages: vec![Message::text(Role::User, prompt)],
            ..LlmRequest::default()
        },
        raw_request: None,
        metadata: request.metadata.clone(),
    })
}

#[async_trait]
impl<S> Classifier<S> for SubagentGate<S>
where
    S: Send + 'static,
{
    fn routing_tier(&self, selected_model_id: &ModelId) -> Option<&'static str> {
        self.inner.routing_tier(selected_model_id)
    }

    async fn score(
        &self,
        state: &mut S,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        if !request
            .metadata
            .as_ref()
            .is_some_and(Metadata::is_subagent_work)
        {
            return Ok((Classification::Scores(Vec::new()), None));
        }
        let Some(mut classifier_request) = delegated_prompt_request(request) else {
            return Ok((Classification::Scores(Vec::new()), None));
        };
        self.inner
            .score(state, &mut classifier_request, driver)
            .await
    }
}

/// Scores a fixed worker target for delegated sub-agent work; abstains otherwise.
pub struct SubagentOverride {
    /// Name of the worker target, resolved by the cascade against its target set.
    worker: ModelId,
}

impl SubagentOverride {
    /// Creates an override scoring `worker` for delegated sub-agent work.
    ///
    /// `worker` must name a target in the cascade's set, or routing a sub-agent request
    /// fails with [`LibsyError::TargetNotFound`](crate::LibsyError::TargetNotFound).
    pub fn new(worker: impl Into<ModelId>) -> Self {
        Self {
            worker: worker.into(),
        }
    }
}

#[async_trait]
impl<S> Classifier<S> for SubagentOverride
where
    S: Send + 'static,
{
    async fn score(
        &self,
        _state: &mut S,
        request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        // Delegated *work* only. A harness maintenance turn (e.g. Codex `compact`) carries
        // sub-agent lineage but is not delegated work, so it abstains and routes normally.
        let is_delegated_work = request
            .metadata
            .as_ref()
            .is_some_and(Metadata::is_subagent_work);
        Ok((
            Classification::Scores(if is_delegated_work {
                vec![Score {
                    confidence: 1.0,
                    target: self.worker.clone(),
                }]
            } else {
                Vec::new()
            }),
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;
    use switchyard_protocol::InstructionBlock;

    use super::*;
    use switchyard_protocol::{slice_to_header_map, text_request};

    #[derive(Default)]
    struct CapturingClassifier {
        requests: Mutex<Vec<Request>>,
    }

    #[async_trait]
    impl Classifier<()> for CapturingClassifier {
        async fn score(
            &self,
            _state: &mut (),
            request: &mut Request,
            _driver: Option<&Driver>,
        ) -> Result<(Classification, Option<Response>)> {
            self.requests.lock().push(request.clone());
            Ok((
                Classification::Scores(vec![Score {
                    confidence: 1.0,
                    target: ModelId::from("worker"),
                }]),
                None,
            ))
        }
    }

    fn request(headers: &[(&str, &str)]) -> Request {
        let metadata =
            (!headers.is_empty()).then(|| Metadata::from_headers(&slice_to_header_map(headers)));
        Request {
            llm_request: text_request(Some(ModelId::from("auto").to_string()), "hi"),
            raw_request: None,
            metadata,
        }
    }

    /// Scores `headers` through the override, returning the winning target if it scored.
    async fn selected(headers: &[(&str, &str)]) -> Result<Option<ModelId>> {
        let mut state = ();
        let classification = SubagentOverride::new("worker")
            .score(&mut state, &mut request(headers), None)
            .await?;
        Ok(classification.0.argmax(false)?.map(|score| score.target))
    }

    #[tokio::test]
    async fn requests_without_metadata_abstain() -> Result<()> {
        assert_eq!(selected(&[]).await?, None);
        Ok(())
    }

    #[tokio::test]
    async fn subagent_work_scores_the_worker() -> Result<()> {
        // Claude Code child-agent lineage.
        let claude = &[
            ("x-claude-code-session-id", "root"),
            ("x-claude-code-agent-id", "child-1"),
        ];
        assert_eq!(selected(claude).await?, Some(ModelId::from("worker")));

        // Codex delegated-work kinds.
        assert_eq!(
            selected(&[("x-openai-subagent", "review")]).await?,
            Some(ModelId::from("worker"))
        );
        assert_eq!(
            selected(&[("x-openai-subagent", "collab_spawn")]).await?,
            Some(ModelId::from("worker"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn harness_maintenance_turns_abstain() -> Result<()> {
        assert_eq!(selected(&[("x-openai-subagent", "compact")]).await?, None);
        assert_eq!(
            selected(&[("x-switchyard-is-subagent", "false")]).await?,
            None
        );
        Ok(())
    }

    #[tokio::test]
    async fn delegated_work_is_scored_definitively() -> Result<()> {
        // Confidence 1.0 under `Scores` (never `Ambiguous`), so the cascade stops here
        // rather than consulting later classifiers.
        let mut state = ();
        let classification = SubagentOverride::new("worker")
            .score(
                &mut state,
                &mut request(&[("x-openai-subagent", "review")]),
                None,
            )
            .await?;
        match classification.0 {
            Classification::Scores(scores) => {
                assert_eq!(scores.len(), 1);
                assert_eq!(scores[0].confidence, 1.0);
            }
            Classification::Ambiguous(_) => panic!("override must score definitively"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn gate_shows_the_classifier_only_the_parent_prompt() -> Result<()> {
        let classifier = Arc::new(CapturingClassifier::default());
        let gate = SubagentGate::new(classifier.clone());
        let mut request = request(&[("x-openai-subagent", "collab_spawn")]);
        request.llm_request.instructions = vec![InstructionBlock {
            role: Role::System,
            content: Message::text(Role::System, "child system instructions").content,
        }];
        request.llm_request.messages = vec![
            Message::text(Role::User, "harness context"),
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text {
                        text: "<system-reminder>tool context</system-reminder>".to_string(),
                    },
                    ContentBlock::Text {
                        text: "implement the delegated task".to_string(),
                    },
                ],
            },
        ];
        let original = request.clone();

        let mut state = ();
        gate.score(&mut state, &mut request, None).await?;

        let captured = classifier.requests.lock();
        assert_eq!(captured.len(), 1);
        assert!(captured[0].llm_request.instructions.is_empty());
        assert_eq!(
            captured[0].llm_request.messages,
            vec![Message::text(Role::User, "implement the delegated task")]
        );
        assert_eq!(request.llm_request, original.llm_request);
        assert_eq!(request.raw_request, original.raw_request);
        assert_eq!(
            request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.agent_id.as_deref()),
            original
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.agent_id.as_deref())
        );
        Ok(())
    }

    #[tokio::test]
    async fn gate_abstains_when_delegated_work_has_no_text_prompt() -> Result<()> {
        let classifier = Arc::new(CapturingClassifier::default());
        let gate = SubagentGate::new(classifier.clone());
        let mut request = request(&[("x-openai-subagent", "collab_spawn")]);
        request.llm_request.messages = vec![Message::text(Role::Assistant, "no user prompt")];

        let mut state = ();
        let (classification, response) = gate.score(&mut state, &mut request, None).await?;

        assert!(classification.argmax(false)?.is_none());
        assert!(response.is_none());
        assert!(classifier.requests.lock().is_empty());
        Ok(())
    }
}
