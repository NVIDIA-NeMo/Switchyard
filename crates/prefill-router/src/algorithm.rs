// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! libsy adapter for checkpoint-backed prefill routing.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use libsy::{
    AffinityRouter, Algorithm, Classifier, Driver, Event, LibsyError, Processor, RoutingOutcome,
};
use switchyard_protocol::{ModelId, Request, Role};

use crate::{
    PrefillForward, PrefillRouter, PrefillRouterError, Result, TransformersForward,
    TransformersForwardConfig,
};

const ALGORITHM_NAME: &str = "prefill_router";

/// Configuration for a libsy prefill-router algorithm.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillRouterConfig {
    /// Ordered routing targets, matching the checkpoint head order.
    pub targets: Vec<ModelId>,
    /// Tensor-only router checkpoint path.
    pub checkpoint: PathBuf,
    /// Torch device override. Auto-detected when omitted.
    pub device: Option<String>,
    /// Optional Hugging Face cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Maximum tokenized prompt length.
    pub max_length: usize,
    /// Maximum prompts per encoder forward pass.
    pub batch_size: usize,
}

impl PrefillRouterConfig {
    /// Creates a prefill-router config with sensible Transformers and routing defaults.
    pub fn new(targets: Vec<ModelId>, checkpoint: impl Into<PathBuf>) -> Self {
        let forward = TransformersForwardConfig::new(checkpoint);
        Self {
            targets,
            checkpoint: forward.checkpoint,
            device: forward.device,
            cache_dir: forward.cache_dir,
            max_length: forward.max_length,
            batch_size: forward.batch_size,
        }
    }

    /// Builds the libsy algorithm described by this config.
    pub fn build(self) -> Result<PrefillRouterAlgo> {
        let Self {
            targets,
            checkpoint,
            device,
            cache_dir,
            max_length,
            batch_size,
        } = self;
        let forward = TransformersForward::new(TransformersForwardConfig {
            checkpoint,
            device,
            cache_dir,
            max_length,
            batch_size,
        })?;
        PrefillRouterAlgo::from_forward(targets, forward)
    }
}

/// libsy [`Algorithm`] wrapper around a [`PrefillRouter`].
///
/// The adapter scores only the latest text user turn. With the default user-turn
/// affinity, tool continuations reuse the prior decision instead of re-running
/// prefill inference.
pub struct PrefillRouterAlgo {
    router: Arc<Mutex<PrefillRouter<AnyPrefillForward>>>,
    targets: Vec<ModelId>,
    default_target: ModelId,
    affinity: Arc<AffinityRouter>,
}

impl PrefillRouterAlgo {
    fn from_forward(targets: Vec<ModelId>, forward: impl PrefillForward + 'static) -> Result<Self> {
        let Some(default_target) = targets.first().cloned() else {
            return Err(PrefillRouterError::InvalidConfiguration(
                "at least one target is required".to_string(),
            ));
        };
        let router = PrefillRouter::new(targets.clone(), AnyPrefillForward(Box::new(forward)))?;
        Ok(Self {
            router: Arc::new(Mutex::new(router)),
            targets,
            default_target,
            affinity: Arc::new(AffinityRouter::new().with_release_on_user_turn()),
        })
    }

    #[doc(hidden)]
    pub fn from_test_forward(
        targets: Vec<ModelId>,
        forward: impl PrefillForward + 'static,
    ) -> Result<Self> {
        Self::from_forward(targets, forward)
    }
}

#[async_trait]
impl Algorithm for PrefillRouterAlgo {
    fn name(&self) -> &str {
        ALGORITHM_NAME
    }

    async fn route(
        self: Arc<Self>,
        _driver: Driver,
        mut request: Request,
    ) -> libsy::Result<RoutingOutcome> {
        let mut state = ();
        if let Some(target) = self
            .affinity
            .score(&mut state, &mut request, None)
            .await?
            .0
            .argmax(false)
            .map(|score| score.map(|score| score.target))?
        {
            tracing::info!(target = %target, "prefill affinity selected target");
            return Ok(RoutingOutcome::route_to(
                target.clone(),
                self.targets
                    .iter()
                    .filter(|candidate| *candidate != &target)
                    .cloned()
                    .collect(),
                request,
            ));
        }

        let target = match request
            .llm_request
            .messages
            .iter()
            .rev()
            .filter(|message| message.role == Role::User)
            .filter_map(|message| message.text_content("\n"))
            .find(|text| !text.trim().is_empty())
        {
            Some(prompt) => {
                let router = Arc::clone(&self.router);
                let predictions = tokio::task::spawn_blocking(move || {
                    router
                        .lock()
                        .map_err(|_| "prefill router lock was poisoned".to_string())?
                        .predict(&prompt)
                        .map_err(|error| error.to_string())
                })
                .await
                .map_err(|error| LibsyError::AlgorithmError {
                    message: format!("prefill router prediction task failed: {error}"),
                })?
                .map_err(|message| LibsyError::AlgorithmError {
                    message: format!("prefill router prediction failed: {message}"),
                })?;
                predictions
                    .into_iter()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(target, _)| target)
                    .ok_or(LibsyError::NoTargets)?
            }
            None => {
                tracing::debug!(
                    target = %self.default_target,
                    "prefill route found no text user message; selecting default target"
                );
                self.default_target.clone()
            }
        };
        let mut state = ();
        self.affinity
            .process(
                &mut state,
                Event::Decision {
                    request: &mut request,
                    selected_model_id: &target,
                },
            )
            .await?;
        tracing::info!(target = %target, "prefill router selected target");
        Ok(RoutingOutcome::route_to(
            target.clone(),
            self.targets
                .iter()
                .filter(|candidate| *candidate != &target)
                .cloned()
                .collect(),
            request,
        ))
    }
}

struct AnyPrefillForward(Box<dyn PrefillForward>);

impl PrefillForward for AnyPrefillForward {
    fn output_count(&self) -> usize {
        self.0.output_count()
    }

    fn forward(&mut self, prompts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.0.forward(prompts)
    }

    fn unload(&mut self) -> Result<()> {
        self.0.unload()
    }
}
