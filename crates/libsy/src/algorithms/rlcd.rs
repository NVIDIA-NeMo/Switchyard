// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RLCD-backed decision routing: a calibrated decision model picks among the
//! route's targets in one pass.
//!
//! RLCD models map a task and a list of options to one calibrated probability
//! per option without writing an answer word by word — the "System One" model
//! class TypeSafe's Jev announcement
//! ([typesafe.ai/blog/introducing-system-one-models-and-jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev))
//! introduced, trained with its Reinforcement Learning for Calibrated
//! Decisions (RLCD) method. TypeSafe has published no RLCD paper.
//!
//! [`Rlcd`] builds a decision request that lists every runtime target as a
//! candidate option, routes to the option with the highest probability, and
//! falls back to the rest of the candidate list — then the configured default
//! target — when the decision verdict is unusable. A decision call that fails
//! in-band (the verdict never arrives) folds to the default target the same
//! way.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use switchyard_protocol::{
    Category, InstructionBlock, LlmRequest, Message, ModelId, OutputParams, Request, Response, Role,
};

use super::fall_through::FallThrough;
use super::llm_class::task_messages;
use super::util::classifier_contract::{ClassifierContract, ClassifierContractConfig};
use super::util::llm_judge::{
    JudgePolicy, JudgeRuntimeConfig, SerdeDecoder, VerdictDecoder, client_error_reason,
    libsy_error_reason, report_fail_open,
};
use super::util::robustness::{safe_client_error, safe_error_summary};
use crate::core::algorithm::{Algorithm, Driver};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::{LibsyError, Result};

const PROMPT_TEMPLATE: &str = include_str!("../prompts/rlcd/prompt.md");
const SCHEMA_TEMPLATE: &str = include_str!("../prompts/rlcd/schema.json");
/// Telemetry label for this algorithm's spans, metrics, and logs.
const ALGORITHM_NAME: &str = "rlcd";
/// The decision model may lose precision rounding probabilities to JSON text.
const PROBABILITY_TOLERANCE: f64 = 0.02;

/// One candidate option and the calibrated probability the decision model assigned it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RlcdOptionScore {
    /// The candidate option this probability belongs to.
    option: String,
    /// Calibrated chance, in `[0, 1]`, that this option is the best target.
    probability: f64,
}

/// The typed decision response from the decision model.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RlcdVerdict {
    /// The option the decision model picked; must match the argmax probability.
    target: String,
    /// One probability per candidate option.
    probabilities: Vec<RlcdOptionScore>,
}

impl RlcdVerdict {
    /// The highest-probability option, or the first when probabilities tie.
    fn best(&self) -> Option<&RlcdOptionScore> {
        let mut best = 0;
        for index in 1..self.probabilities.len() {
            if self.probabilities[index].probability > self.probabilities[best].probability {
                best = index;
            }
        }
        self.probabilities.get(best)
    }

    /// The verdict is usable when it names every candidate exactly once with a
    /// finite `[0, 1]` probability, the probabilities sum to about one, and
    /// `target` is the highest-probability option.
    fn is_valid(&self, candidates: &[ModelId]) -> bool {
        if self.probabilities.len() != candidates.len() {
            return false;
        }
        let mut sum = 0.0;
        let mut seen: HashSet<&str> = HashSet::new();
        for score in &self.probabilities {
            if !score.probability.is_finite() || !(0.0..=1.0).contains(&score.probability) {
                return false;
            }
            if !candidates
                .iter()
                .any(|candidate| candidate.as_str() == score.option.as_str())
            {
                return false;
            }
            if !seen.insert(score.option.as_str()) {
                return false;
            }
            sum += score.probability;
        }
        (sum - 1.0).abs() <= PROBABILITY_TOLERANCE
            && self.best().is_some_and(|best| self.target == best.option)
    }
}

/// Settings that control RLCD decision routing.
#[derive(Clone, Debug)]
pub struct RlcdConfig {
    /// Target chosen when the decision verdict is unusable or the decision
    /// call fails in-band.
    pub default_target: ModelId,
    /// Prompt and verdict contract settings for the decision model.
    pub contract: ClassifierContractConfig,
    /// Maximum completion tokens available to the decision verdict.
    pub max_output_tokens: u64,
}

impl RlcdConfig {
    fn validate(&self) -> Result<()> {
        if self.max_output_tokens == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "max_output_tokens must be at least 1".to_string(),
            });
        }
        Ok(())
    }
}

/// Maps a validated decision verdict to routing scores.
struct RlcdPolicy;

impl JudgePolicy for RlcdPolicy {
    type Verdict = RlcdVerdict;

    fn to_classification(
        &self,
        verdict: Option<&Self::Verdict>,
        driver: &Driver,
    ) -> Result<Classification> {
        // Decision-model output is untrusted. An absent, invalid, or inconsistent verdict is
        // ambiguous so the surrounding router applies its configured fallback.
        let Some(verdict) =
            verdict.filter(|verdict| verdict.is_valid(driver.models_for(&Category::Any)))
        else {
            return Ok(Classification::Ambiguous(vec![]));
        };
        Ok(Classification::Scores(
            verdict
                .probabilities
                .iter()
                .map(|score| Score {
                    target: ModelId::from(score.option.clone()),
                    confidence: score.probability,
                    category: Some(Category::Any),
                })
                .collect(),
        ))
    }
}

/// Returns routing evidence for a usable decision.
fn decision_evidence(verdict: Option<&RlcdVerdict>) -> Option<Value> {
    let verdict = verdict?;
    let best = verdict.best()?;
    Some(serde_json::json!({
        "source": "rlcd",
        "verdict": best.option.clone(),
        "score": best.probability,
        "probabilities": verdict.probabilities.iter().map(|score| serde_json::json!({
            "option": score.option.clone(),
            "probability": score.probability,
        })).collect::<Vec<_>>(),
    }))
}

/// Builds the decision request: the task messages plus a trailing user message
/// that enumerates the candidate options.
fn decision_messages(messages: &[Message], candidates: &[ModelId]) -> Vec<Message> {
    let mut messages = task_messages(messages);
    let options = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| format!("{index}: {candidate}"))
        .collect::<Vec<_>>()
        .join("\n");
    messages.push(Message::text(
        Role::User,
        format!("Choose the best target for the task from these options:\n{options}"),
    ));
    messages
}

/// Consults the runtime decision model and converts its verdict into routing scores.
struct RlcdClassifier {
    contract: ClassifierContract,
    policy: RlcdPolicy,
    runtime: JudgeRuntimeConfig,
}

impl RlcdClassifier {
    fn new(contract: ClassifierContract, policy: RlcdPolicy, runtime: JudgeRuntimeConfig) -> Self {
        Self {
            contract,
            policy,
            runtime,
        }
    }

    fn build_decision_request(&self, request: &Request, driver: &Driver) -> Request {
        Request {
            llm_request: LlmRequest {
                model: request.llm_request.model.clone(),
                instructions: vec![InstructionBlock {
                    role: Role::System,
                    content: Message::text(Role::System, self.contract.system_prompt().to_string())
                        .content,
                }],
                messages: decision_messages(
                    &request.llm_request.messages,
                    driver.models_for(&Category::Any),
                ),
                output: OutputParams {
                    max_output_tokens: Some(self.runtime.max_output_tokens()),
                    response_format: Some(self.contract.response_format().clone()),
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }

    /// Logs and counts a failed decision call.
    fn record_fail_open(&self, driver: &Driver, error: String, reason: &'static str) {
        let judge_target = driver
            .first_model_for(&Category::Judge)
            .map(|c| c.as_str())
            .unwrap_or("missing");
        report_fail_open(judge_target, error, reason);
        driver.set_evidence_if_empty(serde_json::json!({
            "source": "fail_open",
            "reason_code": reason,
        }));
    }

    /// Consults the decision model, yielding `None` when it is unavailable or
    /// unintelligible so the surrounding router applies its fallback.
    async fn decision(
        &self,
        request: &Request,
        driver: &Driver,
        judge_models: &[ModelId],
    ) -> Option<RlcdVerdict> {
        let judge_model = judge_models.first()?.as_str();

        tracing::info!(target = judge_model, "consulting rlcd decision model");
        let response = driver
            .call_model(
                self.build_decision_request(request, driver),
                judge_models.to_vec(),
            )
            .await
            .inspect_err(|error| {
                self.record_fail_open(driver, safe_error_summary(error), libsy_error_reason(error));
            })
            .ok()?;
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .inspect_err(|error| {
                self.record_fail_open(driver, safe_client_error(error), client_error_reason(error));
            })
            .ok()?;
        SerdeDecoder::<RlcdVerdict>::new()
            .decode(&aggregate, &self.contract)
            .inspect_err(|error| {
                self.record_fail_open(driver, safe_error_summary(error), "parse_error");
            })
            .ok()
    }
}

#[async_trait]
impl Classifier<()> for RlcdClassifier {
    async fn score(
        &self,
        _state: &mut (),
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let judge_models = driver.models_for(&Category::Judge);
        if judge_models.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "no models available for category Judge".to_string(),
            });
        }
        let verdict = self.decision(request, driver, judge_models).await;
        let classification = self.policy.to_classification(verdict.as_ref(), driver)?;
        match &classification {
            Classification::Scores(scores) if !scores.is_empty() => {
                if let Some(evidence) = decision_evidence(verdict.as_ref()) {
                    driver.set_evidence(evidence);
                }
            }
            // A present but unusable verdict must not credit the rejected
            // decision: the fallback target decides, and the evidence says why.
            _ if verdict.is_some() => driver.set_evidence_if_empty(serde_json::json!({
                "source": "fail_open",
                "reason_code": "invalid_verdict",
            })),
            _ => {}
        }
        // The decision model is a side call, never the turn's answer.
        Ok((classification, None))
    }
}

/// Terminal classifier that routes the configured default target when the
/// decision model abstains.
struct RlcdFallback(ModelId);

#[async_trait]
impl<S: Send> Classifier<S> for RlcdFallback {
    async fn score(
        &self,
        _state: &mut S,
        _request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        driver.set_evidence_if_empty(serde_json::json!({"source": "fail_open"}));
        Ok((
            Classification::Scores(vec![Score {
                target: self.0.clone(),
                confidence: 0.0,
                category: Some(Category::Any),
            }]),
            None,
        ))
    }
}

/// Routes each request by consulting an RLCD decision model.
pub struct Rlcd {
    route: FallThrough<()>,
}

impl Rlcd {
    /// Builds an RLCD router.
    ///
    /// # Errors
    ///
    /// Returns an error when the decision contract or runtime settings are invalid.
    pub fn new(config: RlcdConfig) -> Result<Self> {
        config.validate()?;
        let contract =
            ClassifierContract::from_config(&config.contract, PROMPT_TEMPLATE, SCHEMA_TEMPLATE)?;
        let classifier = Arc::new(RlcdClassifier::new(
            contract,
            RlcdPolicy,
            JudgeRuntimeConfig::new(config.max_output_tokens)?,
        ));
        Ok(Self {
            route: FallThrough::new()
                .with_name(ALGORITHM_NAME)
                .with_classifier(classifier)
                .with_classifier(Arc::new(RlcdFallback(config.default_target.clone()))),
        })
    }
}

#[async_trait]
impl Algorithm for Rlcd {
    fn name(&self) -> &str {
        ALGORITHM_NAME
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        request: Request,
    ) -> Result<crate::RoutingOutcome> {
        self.route.execute(driver, request).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use futures::StreamExt;
    use parking_lot::Mutex;

    use super::*;
    use switchyard_protocol::{
        LlmClientError, LlmResponse, Metadata, completion_text, text_request, text_response,
    };

    use crate::core::testing::{Serve, test_drive_with_models};

    fn test_config(default_target: &str) -> RlcdConfig {
        RlcdConfig {
            default_target: ModelId::from(default_target.to_string()),
            contract: ClassifierContractConfig::default(),
            max_output_tokens: 128,
        }
    }

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "classify this task"),
            raw_request: None,
            metadata: None,
        }
    }

    fn session_request() -> Request {
        Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                ..Metadata::default()
            }),
            ..request()
        }
    }

    fn runtime_models() -> HashMap<Category, Vec<ModelId>> {
        [
            (Category::Judge, vec![ModelId::from("decision")]),
            (
                Category::Any,
                vec![ModelId::from("efficient"), ModelId::from("capable")],
            ),
        ]
        .into()
    }

    fn router(default_target: &str) -> Result<Arc<dyn Algorithm>> {
        Ok(Arc::new(Rlcd::new(test_config(default_target))?))
    }

    fn verdict(target: &str, scores: &[(&str, f64)]) -> String {
        let probabilities = scores
            .iter()
            .map(|(option, probability)| {
                format!(r#"{{"option":"{option}","probability":{probability}}}"#)
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(r#"{{"target":"{target}","probabilities":[{probabilities}]}}"#)
    }

    /// Answers the decision model with `completion`; every other target echoes its name.
    fn serve_with(completion: String) -> impl Serve {
        move |model: ModelId, _request: Request| {
            let completion = completion.clone();
            async move {
                let model = model.to_string();
                let text = if model == "decision" {
                    completion.to_string()
                } else {
                    format!("answer from {model}")
                };
                Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, text)),
                    metadata: None,
                    upstream_headers: http::HeaderMap::new(),
                })
            }
        }
    }

    /// A decision model that times out; every other target answers normally.
    fn unreachable_decision() -> impl Serve {
        |model: ModelId, _request: Request| async move {
            let model = model.to_string();
            if model == "decision" {
                return Err(LlmClientError::Timeout {
                    source: Box::new(std::io::Error::other("decision model unreachable")),
                });
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, format!("answer from {model}"))),
                metadata: None,
                upstream_headers: http::HeaderMap::new(),
            })
        }
    }

    /// Captures the request a target receives, then answers it with its name.
    fn capturing(into: Arc<Mutex<Option<Request>>>) -> impl Serve {
        move |model: ModelId, request: Request| {
            let into = Arc::clone(&into);
            async move {
                if model.as_str() == "decision" {
                    *into.lock() = Some(request);
                }
                Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, model.to_string())),
                    metadata: None,
                    upstream_headers: http::HeaderMap::new(),
                })
            }
        }
    }

    fn capable_verdict() -> String {
        verdict("capable", &[("efficient", 0.3), ("capable", 0.7)])
    }

    #[tokio::test]
    async fn rlcd_routes_to_the_argmax_option() -> Result<()> {
        let (selected, response) = test_drive_with_models(
            router("efficient")?,
            request(),
            runtime_models(),
            serve_with(capable_verdict()),
        )
        .await?;

        assert_eq!(selected, "capable");
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from capable".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_unreachable_decision_model_routes_the_default_target() -> Result<()> {
        let (selected, response) = test_drive_with_models(
            router("efficient")?,
            request(),
            runtime_models(),
            unreachable_decision(),
        )
        .await?;

        assert_eq!(selected, "efficient");
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from efficient".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_invalid_verdict_routes_the_default_target() -> Result<()> {
        for completion in [
            "not json at all".to_string(),
            verdict("efficient", &[]),
            verdict("efficient", &[("efficient", 1.5), ("capable", -0.5)]),
            verdict("efficient", &[("efficient", 0.3)]),
            verdict("efficient", &[("efficient", 0.3), ("capable", 0.3)]),
            verdict("efficient", &[("efficient", 1.0), ("efficient", 0.0)]),
        ] {
            let (selected, _) = test_drive_with_models(
                router("efficient")?,
                request(),
                runtime_models(),
                serve_with(completion),
            )
            .await?;
            assert_eq!(selected, "efficient", "unusable verdict routed {selected}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_target_that_mismatches_the_argmax_routes_the_default_target() -> Result<()> {
        // The model answered "efficient" yet gave capable the higher probability.
        let (selected, _) = test_drive_with_models(
            router("efficient")?,
            request(),
            runtime_models(),
            serve_with(verdict(
                "efficient",
                &[("efficient", 0.3), ("capable", 0.7)],
            )),
        )
        .await?;
        assert_eq!(selected, "efficient");
        Ok(())
    }

    #[tokio::test]
    async fn the_decision_request_enumerates_every_candidate_option() -> Result<()> {
        let seen = Arc::new(Mutex::new(None));
        let serve = capturing(seen.clone());
        let models = runtime_models();
        test_drive_with_models(router("efficient")?, session_request(), models, serve).await?;

        let request = seen
            .lock()
            .take()
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "the decision model was never called".to_string(),
            })?;
        let text = request
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content("\n"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("efficient"),
            "options missing from decision request: {text}"
        );
        assert!(
            text.contains("capable"),
            "options missing from decision request: {text}"
        );
        assert!(text.contains("Choose the best target"));
        Ok(())
    }

    /// Drives one request, answering the decision model with `completion`, and
    /// returns the outcome evidence.
    async fn evidence_for_decision(completion: String) -> Result<Value> {
        use crate::core::algorithm::Step;

        let models = runtime_models();
        let stream = router("efficient")?.run_stream(request(), Arc::new(models.into()));
        tokio::pin!(stream);

        let mut evidence = None;
        while let Some(step) = stream.next().await {
            match step? {
                Step::CallModel(call) => {
                    let model_name = call
                        .models
                        .first()
                        .map(|model| model.to_string())
                        .unwrap_or_default();
                    let text = if model_name == "decision" {
                        completion.clone()
                    } else {
                        "answer".to_string()
                    };
                    call.respond(Ok(Response {
                        llm_response: LlmResponse::Agg(text_response(None, text)),
                        metadata: None,
                        upstream_headers: http::HeaderMap::new(),
                    }))?;
                }
                Step::Done(outcome) => {
                    evidence = outcome
                        .metadata
                        .as_ref()
                        .expect("run_stream should attach outcome metadata")
                        .evidence
                        .clone();
                }
            }
        }

        evidence.ok_or_else(|| LibsyError::AlgorithmError {
            message: "no evidence recorded".to_string(),
        })
    }

    #[tokio::test]
    async fn the_decision_records_the_probability_distribution_as_evidence() -> Result<()> {
        let evidence = evidence_for_decision(capable_verdict()).await?;
        assert_eq!(
            evidence.pointer("/source").and_then(Value::as_str),
            Some("rlcd")
        );
        assert_eq!(
            evidence.pointer("/verdict").and_then(Value::as_str),
            Some("capable")
        );
        let probabilities = evidence
            .pointer("/probabilities")
            .and_then(Value::as_array)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "no probabilities in evidence".to_string(),
            })?;
        assert_eq!(probabilities.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn an_invalid_verdict_records_fail_open_evidence() -> Result<()> {
        // The probabilities do not sum to one, so the verdict is rejected and
        // the evidence must not credit it.
        let evidence =
            evidence_for_decision(verdict("capable", &[("efficient", 0.3), ("capable", 0.3)]))
                .await?;
        assert_eq!(
            evidence.pointer("/source").and_then(Value::as_str),
            Some("fail_open")
        );
        assert_eq!(
            evidence.pointer("/reason_code").and_then(Value::as_str),
            Some("invalid_verdict")
        );
        Ok(())
    }

    #[test]
    fn rlcd_config_rejects_zero_max_output_tokens() {
        let error = Rlcd::new(RlcdConfig {
            default_target: ModelId::from("capable"),
            contract: ClassifierContractConfig::default(),
            max_output_tokens: 0,
        })
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
        assert!(
            error.contains("max_output_tokens"),
            "unexpected error: {error}"
        );
    }
}
