// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Learned task routing from prompt hidden-state features.
//!
//! The classifier owns checkpoint inference, policy, and bounded task-level
//! decisions. A [`PrefillProbe`] supplies features without coupling `libsy` to
//! an HTTP client, provider SDK, or hidden-state transport.

use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::BuildHasher;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use lru::LruCache;
use parking_lot::Mutex;
use switchyard_protocol::Role;

use self::artifact::InferenceArtifact;
use self::policy::{CostAwareRoutingPolicy, PrefillTier};
use crate::{Classification, Classifier, Driver, LibsyError, Request, Result, Score};

mod artifact;
mod policy;

const TERMINUS_TASK_DESCRIPTION_HEADER: &str = "Task Description:\n";
const TERMINUS_TERMINAL_STATE_HEADER: &str = "\n\nCurrent terminal state:\n";

/// Default maximum number of successful task decisions retained by one classifier.
pub const DEFAULT_PREFILL_PROBE_CACHE_CAPACITY: usize = 4_096;

/// Token-mean hidden-state features produced by a prefill probe.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefillFeatures {
    /// Number of independently extracted hidden-state layers.
    pub layer_count: usize,
    /// Hidden width of each extracted layer.
    pub hidden_size: usize,
    /// Layer-major token-mean features.
    pub values: Vec<f32>,
}

impl PrefillFeatures {
    /// Creates one feature vector with its source layout.
    pub fn new(layer_count: usize, hidden_size: usize, values: Vec<f32>) -> Self {
        Self {
            layer_count,
            hidden_size,
            values,
        }
    }
}

/// Supplies prompt hidden-state features to [`PrefillProbeClassifier`].
///
/// Implementations own transport concerns such as endpoint timeouts and
/// temporary-artifact lifecycle. Errors are treated as an unavailable routing
/// optimization: the classifier selects strong without caching the failure.
#[async_trait]
pub trait PrefillProbe: Send + Sync {
    /// Extracts token-mean features for one task instruction.
    async fn extract(&self, task: &str) -> Result<PrefillFeatures>;
}

/// Construction inputs for [`PrefillProbeClassifier`].
#[derive(Clone, Debug)]
pub struct PrefillProbeClassifierConfig {
    /// Probe model whose hidden-state layout matches the checkpoint metadata.
    pub probe_model: String,
    /// Directory containing `router.json` and `router.safetensors`.
    pub checkpoint_dir: PathBuf,
    /// Checkpoint output head corresponding to the strong completion target.
    pub strong_checkpoint_head: String,
    /// Checkpoint output head corresponding to the weak completion target.
    pub weak_checkpoint_head: String,
    /// Semantic name returned when the strong tier is selected.
    pub strong_target: String,
    /// Semantic name returned when the weak tier is selected.
    pub weak_target: String,
    /// Correctness weight in the cost-aware policy.
    pub lambda: f64,
    /// Non-negative weak-target cost in the same units as `strong_cost`.
    pub weak_cost: f64,
    /// Non-negative strong-target cost in the same units as `weak_cost`.
    pub strong_cost: f64,
    /// Maximum successful task decisions retained in memory.
    pub cache_capacity: usize,
}

struct LearnedRouting {
    artifact: InferenceArtifact,
    weak_head_index: usize,
    strong_head_index: usize,
    policy: CostAwareRoutingPolicy,
}

impl LearnedRouting {
    fn select(&self, features: PrefillFeatures) -> Result<PrefillTier> {
        if features.layer_count != self.artifact.layer_count() {
            return Err(inference_error(format!(
                "feature layer count {} does not match checkpoint layer count {}",
                features.layer_count,
                self.artifact.layer_count(),
            )));
        }
        if features.hidden_size != self.artifact.hidden_size() {
            return Err(inference_error(format!(
                "feature hidden size {} does not match checkpoint hidden size {}",
                features.hidden_size,
                self.artifact.hidden_size(),
            )));
        }
        if features.values.len() != self.artifact.raw_feature_dim() {
            return Err(inference_error(format!(
                "feature length {} does not match checkpoint raw_feature_dim {}",
                features.values.len(),
                self.artifact.raw_feature_dim(),
            )));
        }

        let projected = self.artifact.project(&features.values)?;
        let logits = self.artifact.ensemble_logits(&projected)?;
        let probabilities = self.artifact.ensemble_probabilities(&logits)?;
        let weak_probability = probabilities.get(self.weak_head_index).ok_or_else(|| {
            inference_error(format!(
                "weak checkpoint head index {} is outside prediction length {}",
                self.weak_head_index,
                probabilities.len(),
            ))
        })?;
        let strong_probability = probabilities.get(self.strong_head_index).ok_or_else(|| {
            inference_error(format!(
                "strong checkpoint head index {} is outside prediction length {}",
                self.strong_head_index,
                probabilities.len(),
            ))
        })?;
        self.policy
            .select(f64::from(*weak_probability), f64::from(*strong_probability))
    }
}

/// Classifies a task as strong or weak from learned prompt features.
///
/// Successful decisions are cached under process-randomized hashes, so raw
/// task text is not retained. The LRU bound prevents task cardinality from
/// growing memory without limit. Probe and inference failures select strong and
/// are deliberately not cached.
pub struct PrefillProbeClassifier {
    probe: Arc<dyn PrefillProbe>,
    routing: Arc<LearnedRouting>,
    strong_target: String,
    weak_target: String,
    decision_cache: Mutex<LruCache<u64, String>>,
    cache_hasher: RandomState,
}

impl PrefillProbeClassifier {
    /// Loads the learned checkpoint and constructs a transport-independent classifier.
    pub fn new(config: PrefillProbeClassifierConfig, probe: Arc<dyn PrefillProbe>) -> Result<Self> {
        let cache_capacity = validate_config(&config)?;
        let artifact = InferenceArtifact::load(&config.checkpoint_dir, &config.probe_model)?;
        Self::from_artifact(config, probe, artifact, cache_capacity)
    }

    fn from_artifact(
        config: PrefillProbeClassifierConfig,
        probe: Arc<dyn PrefillProbe>,
        artifact: InferenceArtifact,
        cache_capacity: NonZeroUsize,
    ) -> Result<Self> {
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
            return Err(config_error(
                "strong_checkpoint_head and weak_checkpoint_head must map to distinct outputs",
            ));
        }

        Ok(Self {
            probe,
            routing: Arc::new(LearnedRouting {
                artifact,
                weak_head_index,
                strong_head_index,
                policy: CostAwareRoutingPolicy::new(
                    config.lambda,
                    config.weak_cost,
                    config.strong_cost,
                )?,
            }),
            strong_target: config.strong_target,
            weak_target: config.weak_target,
            decision_cache: Mutex::new(LruCache::new(cache_capacity)),
            cache_hasher: RandomState::new(),
        })
    }

    async fn select_for_task(&self, task: &str) -> String {
        let cache_key = self.cache_hasher.hash_one(task);
        if let Some(target) = self.decision_cache.lock().get(&cache_key).cloned() {
            return target;
        }

        let result = match self.probe.extract(task).await {
            Ok(features) => {
                let routing = Arc::clone(&self.routing);
                tokio::task::spawn_blocking(move || routing.select(features))
                    .await
                    .map_err(|error| inference_error(format!("inference task failed: {error}")))
                    .and_then(|result| result)
            }
            Err(error) => Err(error),
        };

        match result {
            Ok(tier) => {
                let target = match tier {
                    PrefillTier::Weak => self.weak_target.clone(),
                    PrefillTier::Strong => self.strong_target.clone(),
                };
                self.decision_cache.lock().put(cache_key, target.clone());
                target
            }
            Err(error) => {
                tracing::warn!(
                    target: "libsy",
                    error = %error,
                    fallback_target = %self.strong_target,
                    "prefill probe unavailable; using uncached strong fallback"
                );
                self.strong_target.clone()
            }
        }
    }

    fn classification(&self, target: String) -> Classification {
        Classification::Scores(vec![Score {
            confidence: 1.0,
            target,
        }])
    }
}

impl fmt::Debug for PrefillProbeClassifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrefillProbeClassifier")
            .field("strong_target", &self.strong_target)
            .field("weak_target", &self.weak_target)
            .field("cache_capacity", &self.decision_cache.lock().cap())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl<S> Classifier<S> for PrefillProbeClassifier
where
    S: Send + 'static,
{
    fn routing_tier(&self, selected_model: &str) -> Option<&'static str> {
        if selected_model == self.weak_target {
            Some("weak")
        } else if selected_model == self.strong_target {
            Some("strong")
        } else {
            None
        }
    }

    async fn score(
        &self,
        _state: &mut S,
        request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<Classification> {
        let Some(task) = probe_input(request) else {
            tracing::warn!(
                target: "libsy",
                fallback_target = %self.strong_target,
                "prefill probe request has no text user instruction; using strong fallback"
            );
            return Ok(self.classification(self.strong_target.clone()));
        };
        Ok(self.classification(self.select_for_task(&task).await))
    }
}

fn validate_config(config: &PrefillProbeClassifierConfig) -> Result<NonZeroUsize> {
    for (field, value) in [
        ("probe_model", config.probe_model.as_str()),
        (
            "strong_checkpoint_head",
            config.strong_checkpoint_head.as_str(),
        ),
        ("weak_checkpoint_head", config.weak_checkpoint_head.as_str()),
        ("strong_target", config.strong_target.as_str()),
        ("weak_target", config.weak_target.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(config_error(format!("{field} must not be empty")));
        }
    }
    if config.strong_target == config.weak_target {
        return Err(config_error(
            "strong_target and weak_target must be distinct",
        ));
    }
    NonZeroUsize::new(config.cache_capacity)
        .ok_or_else(|| config_error("cache_capacity must be positive"))
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
            config_error(format!(
                "{field} {checkpoint_head:?} is not present in checkpoint output_names {:?}",
                artifact.output_names(),
            ))
        })
}

/// Returns the first text-bearing user message, reduced to a benchmark task when recognized.
fn probe_input(request: &Request) -> Option<String> {
    let instruction = request
        .llm_request
        .messages
        .iter()
        .filter(|message| message.role == Role::User)
        .find_map(|message| message.text_content("").filter(|text| !text.is_empty()))?;
    Some(
        terminus_task_instruction(&instruction)
            .unwrap_or(&instruction)
            .to_owned(),
    )
}

fn terminus_task_instruction(instruction: &str) -> Option<&str> {
    let (_, task_and_terminal) = instruction.split_once(TERMINUS_TASK_DESCRIPTION_HEADER)?;
    let (task, _) = task_and_terminal.split_once(TERMINUS_TERMINAL_STATE_HEADER)?;
    (!task.is_empty()).then_some(task)
}

fn config_error(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: format!("invalid prefill-probe config: {}", message.into()),
    }
}

fn inference_error(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: format!("prefill-probe inference error: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use parking_lot::Mutex;
    use switchyard_protocol::{LlmRequest, Message};

    use super::*;

    enum ProbeResult {
        Features(PrefillFeatures),
        Failure,
    }

    struct RecordingProbe {
        results: Mutex<VecDeque<ProbeResult>>,
        inputs: Mutex<Vec<String>>,
    }

    impl RecordingProbe {
        fn new(results: impl IntoIterator<Item = ProbeResult>) -> Self {
            Self {
                results: Mutex::new(results.into_iter().collect()),
                inputs: Mutex::new(Vec::new()),
            }
        }

        fn inputs(&self) -> Vec<String> {
            self.inputs.lock().clone()
        }
    }

    #[async_trait]
    impl PrefillProbe for RecordingProbe {
        async fn extract(&self, task: &str) -> Result<PrefillFeatures> {
            self.inputs.lock().push(task.to_string());
            match self.results.lock().pop_front() {
                Some(ProbeResult::Features(features)) => Ok(features),
                Some(ProbeResult::Failure) => Err(inference_error("test probe failure")),
                None => Err(inference_error("test probe has no result")),
            }
        }
    }

    fn features() -> PrefillFeatures {
        PrefillFeatures::new(2, 2, vec![0.0; 4])
    }

    fn config(cache_capacity: usize) -> PrefillProbeClassifierConfig {
        PrefillProbeClassifierConfig {
            probe_model: "probe/model".into(),
            checkpoint_dir: "/unused/test/checkpoint".into(),
            strong_checkpoint_head: "opus-4.7".into(),
            weak_checkpoint_head: "nemotron-3-super".into(),
            strong_target: "strong/model".into(),
            weak_target: "weak/model".into(),
            lambda: 1.0,
            weak_cost: 0.0,
            strong_cost: 1.0,
            cache_capacity,
        }
    }

    fn classifier(
        probe: Arc<dyn PrefillProbe>,
        cache_capacity: usize,
    ) -> Result<PrefillProbeClassifier> {
        let config = config(cache_capacity);
        let capacity = validate_config(&config)?;
        PrefillProbeClassifier::from_artifact(
            config,
            probe,
            InferenceArtifact::with_test_probabilities([0.1, 0.8, 0.2, 0.1]),
            capacity,
        )
    }

    fn request(messages: Vec<Message>) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("auto".into()),
                messages,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    async fn selected(
        classifier: &PrefillProbeClassifier,
        request: &mut Request,
    ) -> Result<String> {
        classifier
            .score(&mut (), request, None)
            .await?
            .argmax(false)?
            .map(|score| score.target)
            .ok_or_else(|| inference_error("classifier abstained"))
    }

    #[tokio::test]
    async fn successful_decision_is_cached_by_task_hash() -> Result<()> {
        let probe = Arc::new(RecordingProbe::new([ProbeResult::Features(features())]));
        let classifier = classifier(probe.clone(), 2)?;
        let mut first = request(vec![Message::text(Role::User, "same task")]);
        let mut second = request(vec![Message::text(Role::User, "same task")]);

        assert_eq!(selected(&classifier, &mut first).await?, "weak/model");
        assert_eq!(selected(&classifier, &mut second).await?, "weak/model");
        assert_eq!(probe.inputs(), ["same task"]);
        assert_eq!(classifier.decision_cache.lock().len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn cache_evicts_at_capacity() -> Result<()> {
        let probe = Arc::new(RecordingProbe::new([
            ProbeResult::Features(features()),
            ProbeResult::Features(features()),
            ProbeResult::Features(features()),
        ]));
        let classifier = classifier(probe.clone(), 1)?;

        for task in ["first task", "second task", "first task"] {
            let mut request = request(vec![Message::text(Role::User, task)]);
            assert_eq!(selected(&classifier, &mut request).await?, "weak/model");
        }

        assert_eq!(probe.inputs(), ["first task", "second task", "first task"]);
        assert_eq!(classifier.decision_cache.lock().len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn probe_failure_falls_back_to_strong_without_caching() -> Result<()> {
        let probe = Arc::new(RecordingProbe::new([
            ProbeResult::Failure,
            ProbeResult::Features(features()),
        ]));
        let classifier = classifier(probe.clone(), 2)?;
        let mut first = request(vec![Message::text(Role::User, "retry task")]);
        let mut retry = first.clone();

        assert_eq!(selected(&classifier, &mut first).await?, "strong/model");
        assert_eq!(selected(&classifier, &mut retry).await?, "weak/model");
        assert_eq!(probe.inputs(), ["retry task", "retry task"]);
        Ok(())
    }

    #[tokio::test]
    async fn malformed_features_fall_back_to_strong_without_caching() -> Result<()> {
        let malformed = PrefillFeatures::new(1, 4, vec![0.0; 4]);
        let probe = Arc::new(RecordingProbe::new([
            ProbeResult::Features(malformed),
            ProbeResult::Features(features()),
        ]));
        let classifier = classifier(probe.clone(), 2)?;
        let mut first = request(vec![Message::text(Role::User, "retry shape")]);
        let mut retry = first.clone();

        assert_eq!(selected(&classifier, &mut first).await?, "strong/model");
        assert_eq!(selected(&classifier, &mut retry).await?, "weak/model");
        assert_eq!(probe.inputs(), ["retry shape", "retry shape"]);
        Ok(())
    }

    #[tokio::test]
    async fn terminus_envelope_sends_only_task_text() -> Result<()> {
        let probe = Arc::new(RecordingProbe::new([ProbeResult::Features(features())]));
        let classifier = classifier(probe.clone(), 2)?;
        let mut request = request(vec![Message::text(
            Role::User,
            concat!(
                "<task>\nTask Description:\n",
                "repair the package",
                "\n\nCurrent terminal state:\n",
                "terminal output\n</task>"
            ),
        )]);

        assert_eq!(selected(&classifier, &mut request).await?, "weak/model");
        assert_eq!(probe.inputs(), ["repair the package"]);
        Ok(())
    }

    #[tokio::test]
    async fn missing_user_text_uses_strong_without_probing() -> Result<()> {
        let probe = Arc::new(RecordingProbe::new([]));
        let classifier = classifier(probe.clone(), 2)?;
        let mut request = request(vec![Message::text(Role::System, "system only")]);

        assert_eq!(selected(&classifier, &mut request).await?, "strong/model");
        assert!(probe.inputs().is_empty());
        Ok(())
    }

    #[test]
    fn config_rejects_invalid_targets_heads_and_capacity() -> Result<()> {
        let mut invalid = config(1);
        invalid.weak_target = invalid.strong_target.clone();
        let error = validate_config(&invalid)
            .err()
            .ok_or_else(|| config_error("duplicate targets should fail"))?;
        assert!(error.to_string().contains("must be distinct"));

        let mut invalid = config(0);
        let error = validate_config(&invalid)
            .err()
            .ok_or_else(|| config_error("zero cache capacity should fail"))?;
        assert!(error.to_string().contains("cache_capacity"));

        invalid.cache_capacity = 1;
        invalid.weak_checkpoint_head = "missing".into();
        let capacity = validate_config(&invalid)?;
        let error = PrefillProbeClassifier::from_artifact(
            invalid,
            Arc::new(RecordingProbe::new([])),
            InferenceArtifact::with_test_probabilities([0.1, 0.8, 0.2, 0.1]),
            capacity,
        )
        .err()
        .ok_or_else(|| config_error("missing checkpoint head should fail"))?;
        assert!(error.to_string().contains("output_names"));
        Ok(())
    }
}
