// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM-classifier router built on the AgentApi optimizer interfaces.
//!
//! Unlike a local ML classifier (which scores a prompt in-process), an LLM
//! classifier needs its own model call to classify the request. That maps
//! directly onto the optimizer's multi-round `ModelInference` -> `feed` ->
//! `optimize` loop: the router first asks the caller to run a classifier model,
//! then — once the score is fed back — asks the caller to run the routed target
//! model, and finally returns control to the agent.

use async_trait::async_trait;
use std::error::Error;

use crate::{
    AgentApiOptAlgorithm, AgentApiOptInput, AgentApiOptimizer, AgentApiOptimizerResponse,
    AgentApiRequest, Decision, EnrichmentData,
};

/// Preamble prepended to the user prompt when asking the classifier model for a
/// strong-win-rate score.
const CLASSIFIER_PROMPT_PREAMBLE: &str = "Rate how strongly this request needs a frontier model. \
     Reply with a single strong-win-rate score in [0, 1]:\n";

/// The tier a classifier score selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassifierTier {
    Strong,
    Weak,
}

impl ClassifierTier {
    /// Stable string form of the tier, used in decision reasoning.
    pub fn as_str(self) -> &'static str {
        match self {
            ClassifierTier::Strong => "strong",
            ClassifierTier::Weak => "weak",
        }
    }
}

/// Decision info attached to the routed `ModelInference` decision.
///
/// This is the concrete `D` carried by [`Decision`] / [`AgentApiOptimizerResponse`]
/// for the classifier router.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassifierRoutingDecision {
    /// Strong-win-rate score parsed from the classifier response, or `None` when
    /// the response could not be parsed (the router then defaults to strong).
    pub score: Option<f64>,
    /// Threshold the score was compared against.
    pub threshold: f64,
    /// Tier chosen for this request.
    pub tier: ClassifierTier,
    /// Model the request was rewritten to target.
    pub selected_model: String,
}

/// Factory that mints a fresh [`LlmClassifierRouter`] per session.
///
/// Generalizes Switchyard's RouteLLM strong/weak selection to an LLM-driven
/// classifier expressed with the AgentApi optimizer interfaces.
pub struct LlmClassifierAlgorithm {
    /// Model id used to classify the request.
    pub classifier_model: String,
    /// Model id chosen when the score meets the threshold.
    pub strong_model: String,
    /// Model id chosen when the score is below the threshold.
    pub weak_model: String,
    /// Score threshold at or above which the strong tier is selected.
    pub threshold: f64,
}

impl AgentApiOptAlgorithm<ClassifierRoutingDecision> for LlmClassifierAlgorithm {
    fn optimizer(&self) -> Box<dyn AgentApiOptimizer<ClassifierRoutingDecision>> {
        Box::new(LlmClassifierRouter {
            classifier_model: self.classifier_model.clone(),
            strong_model: self.strong_model.clone(),
            weak_model: self.weak_model.clone(),
            threshold: self.threshold,
            phase: Phase::AwaitingRequest,
            pending_request: None,
            score: None,
        })
    }
}

/// Where the router is in its classify -> route -> return lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// No request buffered yet.
    AwaitingRequest,
    /// Request buffered; next `optimize` emits the classifier call.
    Classify,
    /// Classifier call emitted; waiting for its response to be fed.
    AwaitingScore,
    /// Score received; next `optimize` emits the routed target call.
    Route,
    /// Routed call emitted; waiting for the target response to be fed.
    AwaitingResponse,
    /// Target response received; next `optimize` returns control to the agent.
    Done,
}

/// Per-session LLM-classifier router.
///
/// Flow: `feed` the request, then `optimize` (emits the classifier call);
/// `feed` the classifier response, then `optimize` (emits the routed call);
/// `feed` the target response, then `optimize` (returns `Return`).
pub struct LlmClassifierRouter {
    classifier_model: String,
    strong_model: String,
    weak_model: String,
    threshold: f64,
    phase: Phase,
    pending_request: Option<AgentApiRequest>,
    score: Option<f64>,
}

impl LlmClassifierRouter {
    /// Build the routing decision from the parsed score, defaulting to the
    /// strong tier when the classifier output could not be parsed.
    fn decide(&self) -> ClassifierRoutingDecision {
        let (tier, selected_model) = match self.score {
            Some(score) if score >= self.threshold => {
                (ClassifierTier::Strong, self.strong_model.clone())
            }
            Some(_) => (ClassifierTier::Weak, self.weak_model.clone()),
            // Defensive default: keep traffic flowing on the strong tier when the
            // classifier response was unusable.
            None => (ClassifierTier::Strong, self.strong_model.clone()),
        };
        ClassifierRoutingDecision {
            score: self.score,
            threshold: self.threshold,
            tier,
            selected_model,
        }
    }
}

#[async_trait]
impl AgentApiOptimizer<ClassifierRoutingDecision> for LlmClassifierRouter {
    async fn feed(
        &mut self,
        input: AgentApiOptInput,
        _enrichment: EnrichmentData,
    ) -> Result<(), Box<dyn Error>> {
        match input {
            AgentApiOptInput::Request(request) => {
                self.pending_request = Some(request);
                self.phase = Phase::Classify;
            }
            AgentApiOptInput::Response(response) => match self.phase {
                // First response is the classifier's output; parse a score.
                Phase::AwaitingScore => {
                    self.score = response.prompt.trim().parse::<f64>().ok();
                    self.phase = Phase::Route;
                }
                // Second response is the routed model's output; ready to return.
                Phase::AwaitingResponse => self.phase = Phase::Done,
                _ => {
                    return Err(
                        "classifier router received a response outside a pending model call".into(),
                    )
                }
            },
            AgentApiOptInput::Metadata(_) => {}
        }
        Ok(())
    }

    async fn optimize(&mut self) -> Result<Decision<ClassifierRoutingDecision>, Box<dyn Error>> {
        match self.phase {
            Phase::AwaitingRequest => Err("optimize called before a request was fed".into()),
            Phase::Classify => {
                let user_prompt = self
                    .pending_request
                    .as_ref()
                    .ok_or("classifier router has no buffered request")?
                    .prompt
                    .clone();
                self.phase = Phase::AwaitingScore;
                let classifier_request = AgentApiRequest {
                    model: self.classifier_model.clone(),
                    prompt: format!("{CLASSIFIER_PROMPT_PREAMBLE}{user_prompt}"),
                };
                Ok(Decision::ModelInference(AgentApiOptimizerResponse {
                    requests: vec![classifier_request],
                    enrichment_data: Vec::new(),
                    decision_reasoning: Some(format!(
                        "classifying request via {}",
                        self.classifier_model
                    )),
                    decision_info: None,
                }))
            }
            Phase::AwaitingScore => {
                Err("optimize called before the classifier response was fed".into())
            }
            Phase::Route => {
                let mut request = self
                    .pending_request
                    .take()
                    .ok_or("classifier router has no buffered request")?;
                let decision = self.decide();
                request.model = decision.selected_model.clone();
                self.phase = Phase::AwaitingResponse;
                Ok(Decision::ModelInference(AgentApiOptimizerResponse {
                    requests: vec![request],
                    enrichment_data: Vec::new(),
                    decision_reasoning: Some(format!(
                        "classifier score {:?} vs threshold {}; selected {} ({})",
                        decision.score,
                        decision.threshold,
                        decision.selected_model,
                        decision.tier.as_str()
                    )),
                    decision_info: Some(decision),
                }))
            }
            Phase::AwaitingResponse => {
                Err("optimize called before the model response was fed".into())
            }
            Phase::Done => Ok(Decision::Return()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty enrichment payload for feeds under test.
    fn enrichment() -> EnrichmentData {
        EnrichmentData {
            session_id: None,
            agent_id: None,
            task_id: None,
            correlation_id: None,
            extra_metadata: None,
        }
    }

    fn request(prompt: &str, model: &str) -> AgentApiRequest {
        AgentApiRequest {
            prompt: prompt.to_string(),
            model: model.to_string(),
        }
    }

    fn algorithm(threshold: f64) -> LlmClassifierAlgorithm {
        LlmClassifierAlgorithm {
            classifier_model: "router/classifier".to_string(),
            strong_model: "frontier/model".to_string(),
            weak_model: "cheap/model".to_string(),
            threshold,
        }
    }

    /// Drive the classify -> route -> return flow, feeding `score_text` as the
    /// mocked classifier response, and return the routed decision.
    async fn run_flow(
        threshold: f64,
        user_prompt: &str,
        score_text: &str,
    ) -> Result<(String, ClassifierRoutingDecision), Box<dyn Error>> {
        let mut optimizer = algorithm(threshold).optimizer();

        // 1. Feed the user request and ask the router what to do.
        optimizer
            .feed(
                AgentApiOptInput::Request(request(user_prompt, "client/model")),
                enrichment(),
            )
            .await?;
        let classifier_model = match optimizer.optimize().await? {
            Decision::ModelInference(response) => {
                // The first inference is the classifier call itself.
                let call = &response.requests[0];
                assert!(
                    call.prompt.contains(user_prompt),
                    "classifier prompt missing user text"
                );
                call.model.clone()
            }
            Decision::Return() => return Err("expected classifier ModelInference".into()),
        };
        assert_eq!(classifier_model, "router/classifier");

        // 2. Mock the classifier LLM call by feeding its score back.
        optimizer
            .feed(
                AgentApiOptInput::Response(request(score_text, &classifier_model)),
                enrichment(),
            )
            .await?;
        let (routed_model, decision) = match optimizer.optimize().await? {
            Decision::ModelInference(response) => {
                let decision = response
                    .decision_info
                    .ok_or("expected decision info on routed ModelInference")?;
                (response.requests[0].model.clone(), decision)
            }
            Decision::Return() => return Err("expected routed ModelInference".into()),
        };

        // 3. Mock the routed model call; the next optimize returns to the agent.
        optimizer
            .feed(
                AgentApiOptInput::Response(request("mocked completion", &routed_model)),
                enrichment(),
            )
            .await?;
        match optimizer.optimize().await? {
            Decision::Return() => Ok((routed_model, decision)),
            Decision::ModelInference(_) => Err("expected Return after routed response fed".into()),
        }
    }

    #[tokio::test]
    async fn score_at_or_above_threshold_routes_strong() -> Result<(), Box<dyn Error>> {
        let (routed_model, decision) = run_flow(0.5, "solve this proof", "0.9").await?;
        assert_eq!(routed_model, "frontier/model");
        assert_eq!(decision.tier, ClassifierTier::Strong);
        assert_eq!(decision.score, Some(0.9));
        Ok(())
    }

    #[tokio::test]
    async fn score_below_threshold_routes_weak() -> Result<(), Box<dyn Error>> {
        let (routed_model, decision) = run_flow(0.5, "say hello", "0.2").await?;
        assert_eq!(routed_model, "cheap/model");
        assert_eq!(decision.tier, ClassifierTier::Weak);
        assert_eq!(decision.score, Some(0.2));
        Ok(())
    }

    #[tokio::test]
    async fn score_exactly_at_threshold_routes_strong() -> Result<(), Box<dyn Error>> {
        let (routed_model, decision) = run_flow(0.5, "borderline", "0.5").await?;
        assert_eq!(routed_model, "frontier/model");
        assert_eq!(decision.tier, ClassifierTier::Strong);
        Ok(())
    }

    #[tokio::test]
    async fn unparseable_score_defaults_to_strong() -> Result<(), Box<dyn Error>> {
        let (routed_model, decision) = run_flow(0.5, "hi", "not-a-number").await?;
        assert_eq!(routed_model, "frontier/model");
        assert_eq!(decision.tier, ClassifierTier::Strong);
        assert_eq!(decision.score, None);
        Ok(())
    }

    #[tokio::test]
    async fn optimize_before_feed_errors() {
        let mut optimizer = algorithm(0.5).optimizer();
        assert!(optimizer.optimize().await.is_err());
    }

    #[tokio::test]
    async fn optimize_before_classifier_response_errors() -> Result<(), Box<dyn Error>> {
        let mut optimizer = algorithm(0.5).optimizer();
        optimizer
            .feed(
                AgentApiOptInput::Request(request("hi", "client/model")),
                enrichment(),
            )
            .await?;
        // Emit the classifier call, then call optimize again without feeding the score.
        assert!(matches!(
            optimizer.optimize().await?,
            Decision::ModelInference(_)
        ));
        assert!(optimizer.optimize().await.is_err());
        Ok(())
    }
}
