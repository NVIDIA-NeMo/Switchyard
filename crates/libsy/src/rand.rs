// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Weighted random router built on the AgentApi optimizer interfaces.

use async_trait::async_trait;
use std::error::Error;

// `::rand` disambiguates the external crate from this module, which is also
// named `rand`.
use ::rand::rngs::StdRng;
use ::rand::{Rng, SeedableRng};

use crate::{
    AgentApiOptAlgorithm, AgentApiOptInput, AgentApiOptimizer, AgentApiOptimizerResponse,
    ChatRequest, Decision, EnrichementData,
};

/// A routing target and its relative selection weight.
///
/// Weights are relative and need not sum to `1.0`; a target is chosen with
/// probability `weight / sum(weights)`. Weights must be finite and non-negative,
/// and at least one weight must be positive.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedModel {
    /// Model id this target routes to.
    pub model: String,
    /// Relative selection weight; a zero weight is never selected.
    pub weight: f64,
}

impl WeightedModel {
    /// Convenience constructor for a weighted target.
    pub fn new(model: impl Into<String>, weight: f64) -> Self {
        WeightedModel {
            model: model.into(),
            weight,
        }
    }
}

/// Decision info attached to a random-routing `ModelInference` decision.
///
/// This is the concrete `D` carried by [`Decision`] / [`AgentApiOptimizerResponse`]
/// for the random router.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomRoutingDecision {
    /// Model the request was rewritten to target.
    pub selected_model: String,
    /// The random draw in `[0, total_weight)` that produced this decision.
    pub draw: f64,
    /// Sum of all target weights the draw was taken against.
    pub total_weight: f64,
}

/// Factory that mints a fresh [`RandomRouter`] per session.
///
/// Generalizes Switchyard's strong/weak `RandomRoutingProfile` to N targets
/// selected by weighted random choice, expressed with the AgentApi optimizer
/// interfaces.
pub struct RandomRouterAlgorithm {
    /// Weighted set of routing targets.
    pub models: Vec<WeightedModel>,
    /// Optional deterministic RNG seed for reproducible routing.
    pub rng_seed: Option<u64>,
}

impl AgentApiOptAlgorithm<RandomRoutingDecision> for RandomRouterAlgorithm {
    fn optimizer(&self) -> Box<dyn AgentApiOptimizer<RandomRoutingDecision>> {
        let rng = match self.rng_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Box::new(RandomRouter {
            models: self.models.clone(),
            rng,
            pending_request: None,
            completed: false,
        })
    }
}

/// Per-session random router over a weighted set of N targets.
///
/// Flow: the caller `feed`s the inbound request, then calls `optimize`, which
/// draws a weighted target, rewrites the request model, and returns
/// `ModelInference` so the caller performs the model call. The caller `feed`s the
/// response back and calls `optimize` again, which returns `Return` to hand
/// control to the agent.
pub struct RandomRouter {
    models: Vec<WeightedModel>,
    rng: StdRng,
    pending_request: Option<ChatRequest>,
    completed: bool,
}

impl RandomRouter {
    /// Draw a weighted target and build the routing decision for the current
    /// request. Errors if no target carries positive, finite weight.
    fn select(&mut self) -> Result<RandomRoutingDecision, Box<dyn Error>> {
        let total_weight: f64 = self
            .models
            .iter()
            .map(|m| m.weight)
            .filter(|w| w.is_finite() && *w > 0.0)
            .sum();
        // `total_weight` sums only finite positive weights, so it is never NaN;
        // `<= 0.0` therefore holds exactly when no target is selectable.
        if total_weight <= 0.0 {
            return Err("random router has no target with positive weight".into());
        }
        // Draw in [0, total_weight) and walk cumulative weights.
        let draw: f64 = self.rng.gen::<f64>() * total_weight;
        let mut cumulative = 0.0;
        for model in &self.models {
            if !(model.weight.is_finite() && model.weight > 0.0) {
                continue;
            }
            cumulative += model.weight;
            if draw < cumulative {
                return Ok(RandomRoutingDecision {
                    selected_model: model.model.clone(),
                    draw,
                    total_weight,
                });
            }
        }
        // Floating-point rounding can leave the draw at the very top of the
        // range; fall back to the last positively weighted target.
        let last = self
            .models
            .iter()
            .rev()
            .find(|m| m.weight.is_finite() && m.weight > 0.0)
            .ok_or("random router has no target with positive weight")?;
        Ok(RandomRoutingDecision {
            selected_model: last.model.clone(),
            draw,
            total_weight,
        })
    }
}

#[async_trait]
impl AgentApiOptimizer<RandomRoutingDecision> for RandomRouter {
    async fn feed(
        &mut self,
        input: AgentApiOptInput,
        _enrichment: EnrichementData,
    ) -> Result<(), Box<dyn Error>> {
        match input {
            AgentApiOptInput::Request(request) => self.pending_request = Some(request),
            // A fed response means the caller performed the model call; the next
            // optimize should hand control back to the agent.
            AgentApiOptInput::Response(_) => self.completed = true,
            AgentApiOptInput::Metadata(_) => {}
        }
        Ok(())
    }

    async fn optimize(&mut self) -> Result<Decision<RandomRoutingDecision>, Box<dyn Error>> {
        // The model response has already come back; return control to the agent.
        if self.completed {
            return Ok(Decision::Return());
        }
        let mut request = self
            .pending_request
            .take()
            .ok_or("optimize called before a request was fed")?;
        let decision = self.select()?;
        request.model = decision.selected_model.clone();
        Ok(Decision::ModelInference(AgentApiOptimizerResponse {
            requests: vec![request],
            enrichment_data: Vec::new(),
            decision_reasoning: Some(format!(
                "weighted random draw {} of total weight {}; selected {}",
                decision.draw, decision.total_weight, decision.selected_model
            )),
            decision_info: Some(decision),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty enrichment payload for feeds under test.
    fn enrichment() -> EnrichementData {
        EnrichementData {
            session_id: None,
            agent_id: None,
            task_id: None,
            correlation_id: None,
            extra_metadata: None,
        }
    }

    fn request(prompt: &str, model: &str) -> ChatRequest {
        ChatRequest {
            prompt: prompt.to_string(),
            model: model.to_string(),
        }
    }

    fn algorithm(models: Vec<WeightedModel>, seed: u64) -> RandomRouterAlgorithm {
        RandomRouterAlgorithm {
            models,
            rng_seed: Some(seed),
        }
    }

    /// Feed one request and return the model the optimizer routed it to.
    async fn route_once(
        optimizer: &mut Box<dyn AgentApiOptimizer<RandomRoutingDecision>>,
    ) -> Result<String, Box<dyn Error>> {
        optimizer
            .feed(
                AgentApiOptInput::Request(request("hi", "client/model")),
                enrichment(),
            )
            .await?;
        match optimizer.optimize().await? {
            Decision::ModelInference(response) => Ok(response.requests[0].model.clone()),
            Decision::Return() => Err("expected ModelInference".into()),
        }
    }

    #[tokio::test]
    async fn all_weight_on_one_model_always_selects_it() -> Result<(), Box<dyn Error>> {
        let mut optimizer = algorithm(
            vec![
                WeightedModel::new("a/model", 0.0),
                WeightedModel::new("b/model", 1.0),
                WeightedModel::new("c/model", 0.0),
            ],
            7,
        )
        .optimizer();

        for _ in 0..25 {
            assert_eq!(route_once(&mut optimizer).await?, "b/model");
        }
        Ok(())
    }

    /// Over many draws the empirical selection frequencies track the weights.
    #[tokio::test]
    async fn selection_frequencies_track_weights() -> Result<(), Box<dyn Error>> {
        // Weights 1:3:6 -> expected shares 0.1, 0.3, 0.6.
        let mut optimizer = algorithm(
            vec![
                WeightedModel::new("a/model", 1.0),
                WeightedModel::new("b/model", 3.0),
                WeightedModel::new("c/model", 6.0),
            ],
            42,
        )
        .optimizer();

        let draws = 20_000;
        let mut a = 0u32;
        let mut b = 0u32;
        let mut c = 0u32;
        for _ in 0..draws {
            match route_once(&mut optimizer).await?.as_str() {
                "a/model" => a += 1,
                "b/model" => b += 1,
                "c/model" => c += 1,
                other => return Err(format!("unexpected model {other}").into()),
            }
        }

        let total = f64::from(draws);
        assert!(
            (f64::from(a) / total - 0.1).abs() < 0.02,
            "a share off: {a}"
        );
        assert!(
            (f64::from(b) / total - 0.3).abs() < 0.02,
            "b share off: {b}"
        );
        assert!(
            (f64::from(c) / total - 0.6).abs() < 0.02,
            "c share off: {c}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn decision_reports_draw_within_total_weight() -> Result<(), Box<dyn Error>> {
        let mut optimizer = algorithm(
            vec![
                WeightedModel::new("a/model", 2.0),
                WeightedModel::new("b/model", 3.0),
            ],
            7,
        )
        .optimizer();
        optimizer
            .feed(
                AgentApiOptInput::Request(request("hi", "client/model")),
                enrichment(),
            )
            .await?;

        match optimizer.optimize().await? {
            Decision::ModelInference(response) => {
                let decision = response
                    .decision_info
                    .ok_or("expected decision info on ModelInference")?;
                assert_eq!(decision.total_weight, 5.0);
                assert!(decision.draw >= 0.0 && decision.draw < 5.0);
            }
            Decision::Return() => return Err("expected ModelInference, got Return".into()),
        }
        Ok(())
    }

    /// Documented flow: route -> caller performs the (mocked) model call ->
    /// feed the response -> optimize returns control to the agent.
    #[tokio::test]
    async fn returns_to_agent_after_mocked_response_is_fed() -> Result<(), Box<dyn Error>> {
        let mut optimizer =
            algorithm(vec![WeightedModel::new("frontier/model", 1.0)], 7).optimizer();
        optimizer
            .feed(
                AgentApiOptInput::Request(request("hi", "client/model")),
                enrichment(),
            )
            .await?;

        let routed_model = match optimizer.optimize().await? {
            Decision::ModelInference(response) => response.requests[0].model.clone(),
            Decision::Return() => return Err("expected ModelInference on first optimize".into()),
        };
        assert_eq!(routed_model, "frontier/model");

        // Mock the model call by feeding a response back into the optimizer.
        optimizer
            .feed(
                AgentApiOptInput::Response(request("mocked completion", &routed_model)),
                enrichment(),
            )
            .await?;

        match optimizer.optimize().await? {
            Decision::Return() => Ok(()),
            Decision::ModelInference(_) => Err("expected Return after response was fed".into()),
        }
    }

    #[tokio::test]
    async fn optimize_before_feed_errors() {
        let mut optimizer = algorithm(vec![WeightedModel::new("a/model", 1.0)], 7).optimizer();
        assert!(optimizer.optimize().await.is_err());
    }

    #[tokio::test]
    async fn no_positive_weight_errors() -> Result<(), Box<dyn Error>> {
        let mut optimizer = algorithm(
            vec![
                WeightedModel::new("a/model", 0.0),
                WeightedModel::new("b/model", 0.0),
            ],
            7,
        )
        .optimizer();
        optimizer
            .feed(
                AgentApiOptInput::Request(request("hi", "client/model")),
                enrichment(),
            )
            .await?;
        assert!(optimizer.optimize().await.is_err());
        Ok(())
    }

    /// Feed one request and return the draw the optimizer used to route it.
    async fn first_draw(
        optimizer: &mut Box<dyn AgentApiOptimizer<RandomRoutingDecision>>,
    ) -> Result<f64, Box<dyn Error>> {
        optimizer
            .feed(
                AgentApiOptInput::Request(request("hi", "client/model")),
                enrichment(),
            )
            .await?;
        match optimizer.optimize().await? {
            Decision::ModelInference(response) => response
                .decision_info
                .map(|d| d.draw)
                .ok_or_else(|| "missing decision info".into()),
            Decision::Return() => Err("expected ModelInference".into()),
        }
    }

    /// The seed is deterministic, so two optimizers from the same factory make
    /// the same first draw.
    #[tokio::test]
    async fn factory_mints_independent_deterministic_optimizers() -> Result<(), Box<dyn Error>> {
        let factory = algorithm(
            vec![
                WeightedModel::new("a/model", 1.0),
                WeightedModel::new("b/model", 1.0),
            ],
            42,
        );

        let mut first = factory.optimizer();
        let mut second = factory.optimizer();
        let draw_first = first_draw(&mut first).await?;
        let draw_second = first_draw(&mut second).await?;
        assert_eq!(draw_first, draw_second);
        Ok(())
    }
}
