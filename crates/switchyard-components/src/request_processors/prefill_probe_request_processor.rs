// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-side learned routing from dedicated prefill-probe hidden states.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use switchyard_core::{ChatRequest, LlmTargetId, ProxyContext, Result, SwitchyardError};

use crate::prefill_probe::artifact::InferenceArtifact;
use crate::prefill_probe::policy::CostAwareRoutingPolicy;
use crate::prefill_probe::scorer::{HiddenStateProbeScorer, ProbeScorer};

const LEARNED_SCORE_THRESHOLD: f64 = 0.5;
const TERMINUS_TASK_DESCRIPTION_HEADER: &str = "Task Description:\n";
const TERMINUS_TERMINAL_STATE_HEADER: &str = "\n\nCurrent terminal state:\n";

/// Construction inputs for the learned prefill-probe request processor.
#[derive(Clone, Debug)]
pub struct PrefillProbeProcessorConfig {
    /// Base URL of the dedicated vLLM probe server.
    pub probe_base_url: String,
    /// Probe model whose hidden-state layout matches the checkpoint metadata.
    pub probe_model: String,
    /// Directory shared with vLLM's hidden-state connector.
    pub hidden_states_dir: PathBuf,
    /// Directory containing `router.json` and `router.safetensors`.
    pub checkpoint_dir: PathBuf,
    /// Checkpoint output head corresponding to the strong completion target.
    pub strong_checkpoint_head: String,
    /// Checkpoint output head corresponding to the weak completion target.
    pub weak_checkpoint_head: String,
    /// Target ID stamped when the strong tier is selected.
    pub strong_target_id: LlmTargetId,
    /// Target ID stamped when the weak tier is selected.
    pub weak_target_id: LlmTargetId,
    /// Correctness weight in the cost-aware routing policy.
    pub lambda: f64,
    /// Non-negative weak-target cost in the same units as `strong_cost`.
    pub weak_cost: f64,
    /// Non-negative strong-target cost in the same units as `weak_cost`.
    pub strong_cost: f64,
}

/// Selects a configured completion target from learned prompt hidden states.
#[derive(Clone)]
pub struct PrefillProbeRequestProcessor {
    scorer: Arc<dyn ProbeScorer>,
    strong_target_id: LlmTargetId,
    weak_target_id: LlmTargetId,
    decision_cache: Arc<Mutex<HashMap<String, LlmTargetId>>>,
}

impl PrefillProbeRequestProcessor {
    /// Loads the external checkpoint and builds a dedicated hidden-state scorer.
    pub fn new(config: PrefillProbeProcessorConfig) -> Result<Self> {
        if config.strong_target_id == config.weak_target_id {
            return Err(SwitchyardError::InvalidConfig(
                "prefill probe strong_target_id and weak_target_id must be distinct".to_string(),
            ));
        }

        let policy =
            CostAwareRoutingPolicy::new(config.lambda, config.weak_cost, config.strong_cost)?;
        let artifact = InferenceArtifact::load(&config.checkpoint_dir, &config.probe_model)?;
        let strong_head_index = checkpoint_head_index(
            &artifact,
            "strong_checkpoint_head",
            &config.strong_checkpoint_head,
        )?;
        let weak_head_index = checkpoint_head_index(
            &artifact,
            "weak_checkpoint_head",
            &config.weak_checkpoint_head,
        )?;
        if strong_head_index == weak_head_index {
            return Err(SwitchyardError::InvalidConfig(format!(
                "strong_checkpoint_head and weak_checkpoint_head must map to distinct outputs; \
                 both map to `{}`",
                config.strong_checkpoint_head,
            )));
        }

        let scorer = HiddenStateProbeScorer::new(
            config.probe_base_url,
            config.probe_model,
            config.hidden_states_dir,
            Arc::new(artifact),
            weak_head_index,
            strong_head_index,
            policy,
        );
        Ok(Self::from_scorer(
            config.strong_target_id,
            config.weak_target_id,
            Arc::new(scorer),
        ))
    }

    /// Resolves and scores a probe prompt, stamps the selected target, and preserves the request.
    pub async fn process(
        &self,
        ctx: &mut ProxyContext,
        request: ChatRequest,
    ) -> Result<ChatRequest> {
        let selected_target = match probe_input(&request) {
            Some(input) => self.select_for_input(input).await,
            None => {
                tracing::warn!(
                    fallback_target = %self.strong_target_id,
                    "prefill probe request has no string-valued user instruction; \
                     using uncached strong fallback"
                );
                self.strong_target_id.clone()
            }
        };
        ctx.set_selected_target(selected_target);
        Ok(request)
    }

    async fn select_for_input(&self, input: String) -> LlmTargetId {
        if let Some(selected) = self.decision_cache.lock().get(&input).cloned() {
            return selected;
        }

        match self.scorer.score(&input).await {
            Ok(score) => {
                let selected = if score >= LEARNED_SCORE_THRESHOLD {
                    self.weak_target_id.clone()
                } else {
                    self.strong_target_id.clone()
                };
                self.decision_cache.lock().insert(input, selected.clone());
                selected
            }
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    fallback_target = %self.strong_target_id,
                    "prefill probe failed; using uncached strong fallback"
                );
                self.strong_target_id.clone()
            }
        }
    }

    fn from_scorer(
        strong_target_id: LlmTargetId,
        weak_target_id: LlmTargetId,
        scorer: Arc<dyn ProbeScorer>,
    ) -> Self {
        Self {
            scorer,
            strong_target_id,
            weak_target_id,
            decision_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl fmt::Debug for PrefillProbeRequestProcessor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrefillProbeRequestProcessor")
            .field("strong_target_id", &self.strong_target_id)
            .field("weak_target_id", &self.weak_target_id)
            .finish_non_exhaustive()
    }
}

fn checkpoint_head_index(
    artifact: &InferenceArtifact,
    field: &str,
    checkpoint_head: &str,
) -> Result<usize> {
    artifact
        .output_names()
        .iter()
        .position(|name| name == checkpoint_head)
        .ok_or_else(|| {
            SwitchyardError::InvalidConfig(format!(
                "{field} `{checkpoint_head}` is not present in artifact output_names {:?}",
                artifact.output_names(),
            ))
        })
}

// Returns the first string-valued user instruction.
fn first_user_instruction(request: &ChatRequest) -> Option<&str> {
    let messages = request.body().as_object()?.get("messages")?.as_array()?;
    messages.iter().find_map(|message| {
        (message.get("role").and_then(serde_json::Value::as_str) == Some("user"))
            .then(|| message.get("content").and_then(serde_json::Value::as_str))
            .flatten()
    })
}

// Extracts the task-only portion of a stock Terminus 2 first-user prompt.
fn terminus_task_instruction(instruction: &str) -> Option<&str> {
    let (_, task_and_terminal) = instruction.split_once(TERMINUS_TASK_DESCRIPTION_HEADER)?;
    let (task, _) = task_and_terminal.split_once(TERMINUS_TERMINAL_STATE_HEADER)?;
    (!task.is_empty()).then_some(task)
}

// Resolves the exact text used by both hidden-state scoring and decision caching.
fn probe_input(request: &ChatRequest) -> Option<String> {
    let first_user = first_user_instruction(request)?;
    Some(
        terminus_task_instruction(first_user)
            .unwrap_or(first_user)
            .to_owned(),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;

    enum TestScore {
        Score(f64),
        Failure,
    }

    struct RecordingScorer {
        results: Mutex<VecDeque<TestScore>>,
        inputs: Mutex<Vec<String>>,
    }

    impl RecordingScorer {
        fn new(results: impl IntoIterator<Item = TestScore>) -> Self {
            Self {
                results: Mutex::new(results.into_iter().collect()),
                inputs: Mutex::new(Vec::new()),
            }
        }

        fn observed_inputs(&self) -> Vec<String> {
            self.inputs.lock().clone()
        }
    }

    #[async_trait]
    impl ProbeScorer for RecordingScorer {
        async fn score(&self, probe_input: &str) -> Result<f64> {
            self.inputs.lock().push(probe_input.to_string());
            match self.results.lock().pop_front() {
                Some(TestScore::Score(score)) => Ok(score),
                Some(TestScore::Failure) => {
                    Err(SwitchyardError::Other("test probe failure".to_string()))
                }
                None => Err(SwitchyardError::Other(
                    "test scorer has no result".to_string(),
                )),
            }
        }
    }

    fn processor(scorer: Arc<dyn ProbeScorer>) -> Result<PrefillProbeRequestProcessor> {
        Ok(PrefillProbeRequestProcessor::from_scorer(
            LlmTargetId::new("strong-target").map_err(|error| {
                SwitchyardError::InvalidConfig(format!("invalid strong target ID: {error}"))
            })?,
            LlmTargetId::new("weak-target").map_err(|error| {
                SwitchyardError::InvalidConfig(format!("invalid weak target ID: {error}"))
            })?,
            scorer,
        ))
    }

    fn request(messages: serde_json::Value) -> ChatRequest {
        ChatRequest::openai_chat(json!({
            "model": "router",
            "messages": messages,
        }))
    }

    #[tokio::test]
    async fn binary_scores_select_targets_without_mutating_request() -> Result<()> {
        let weak_processor = processor(Arc::new(RecordingScorer::new([TestScore::Score(1.0)])))?;
        let strong_processor = processor(Arc::new(RecordingScorer::new([TestScore::Score(0.0)])))?;
        let original = request(json!([{"role": "user", "content": "route me"}]));

        let mut weak_ctx = ProxyContext::new();
        let weak_request = weak_processor
            .process(&mut weak_ctx, original.clone())
            .await?;
        let mut strong_ctx = ProxyContext::new();
        let strong_request = strong_processor
            .process(&mut strong_ctx, original.clone())
            .await?;

        assert_eq!(
            weak_ctx.selected_target(),
            Some(&LlmTargetId::new("weak-target").map_err(|error| {
                SwitchyardError::InvalidConfig(format!("invalid weak target ID: {error}"))
            })?)
        );
        assert_eq!(
            strong_ctx.selected_target(),
            Some(&LlmTargetId::new("strong-target").map_err(|error| {
                SwitchyardError::InvalidConfig(format!("invalid strong target ID: {error}"))
            })?)
        );
        assert_eq!(weak_request, original);
        assert_eq!(strong_request, original);
        Ok(())
    }

    #[tokio::test]
    async fn successful_decision_is_cached_by_resolved_probe_text() -> Result<()> {
        let scorer = Arc::new(RecordingScorer::new([TestScore::Score(1.0)]));
        let processor = processor(scorer.clone())?;
        let first = request(json!([
            {"role": "system", "content": "first system"},
            {"role": "user", "content": [{"type": "text", "text": "structured"}]},
            {"role": "user", "content": "same task"},
            {"role": "user", "content": "later task"}
        ]));
        let second = request(json!([
            {"role": "system", "content": "different system"},
            {"role": "user", "content": "same task"},
        ]));

        let mut first_ctx = ProxyContext::new();
        processor.process(&mut first_ctx, first).await?;
        let mut second_ctx = ProxyContext::new();
        processor.process(&mut second_ctx, second).await?;

        assert_eq!(scorer.observed_inputs(), ["same task"]);
        assert_eq!(first_ctx.selected_target(), second_ctx.selected_target());
        Ok(())
    }

    #[tokio::test]
    async fn terminus_envelope_sends_only_task_text_and_preserves_request() -> Result<()> {
        let scorer = Arc::new(RecordingScorer::new([TestScore::Score(0.0)]));
        let processor = processor(scorer.clone())?;
        let original = request(json!([{
            "role": "user",
            "content": concat!(
                "<task>\nTask Description:\n",
                "repair the package",
                "\n\nCurrent terminal state:\n",
                "terminal output\n</task>"
            )
        }]));
        let mut ctx = ProxyContext::new();

        let returned = processor.process(&mut ctx, original.clone()).await?;

        assert_eq!(scorer.observed_inputs(), ["repair the package"]);
        assert_eq!(returned, original);
        Ok(())
    }

    #[tokio::test]
    async fn failed_probe_falls_back_to_strong_without_caching() -> Result<()> {
        let scorer = Arc::new(RecordingScorer::new([
            TestScore::Failure,
            TestScore::Score(1.0),
        ]));
        let processor = processor(scorer.clone())?;
        let input = request(json!([{"role": "user", "content": "retry task"}]));

        let mut failed_ctx = ProxyContext::new();
        processor.process(&mut failed_ctx, input.clone()).await?;
        let mut retry_ctx = ProxyContext::new();
        processor.process(&mut retry_ctx, input).await?;

        assert_eq!(
            failed_ctx.selected_target(),
            Some(&LlmTargetId::new("strong-target").map_err(|error| {
                SwitchyardError::InvalidConfig(format!("invalid strong target ID: {error}"))
            })?)
        );
        assert_eq!(
            retry_ctx.selected_target(),
            Some(&LlmTargetId::new("weak-target").map_err(|error| {
                SwitchyardError::InvalidConfig(format!("invalid weak target ID: {error}"))
            })?)
        );
        assert_eq!(scorer.observed_inputs(), ["retry task", "retry task"]);
        Ok(())
    }

    #[tokio::test]
    async fn missing_string_user_content_is_an_uncached_strong_fallback() -> Result<()> {
        let scorer = Arc::new(RecordingScorer::new([]));
        let processor = processor(scorer.clone())?;
        let input = request(json!([
            {"role": "system", "content": "system only"},
            {"role": "user", "content": [{"type": "text", "text": "structured"}]}
        ]));

        for _ in 0..2 {
            let mut ctx = ProxyContext::new();
            processor.process(&mut ctx, input.clone()).await?;
            assert_eq!(
                ctx.selected_target(),
                Some(&LlmTargetId::new("strong-target").map_err(|error| {
                    SwitchyardError::InvalidConfig(format!("invalid strong target ID: {error}"))
                })?)
            );
        }
        assert!(scorer.observed_inputs().is_empty());
        Ok(())
    }
}
