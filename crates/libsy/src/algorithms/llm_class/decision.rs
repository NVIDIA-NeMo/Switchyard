// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Select a candidate by applying a cutoff to a decision model's choice score, or
//! take the category a decision service selects.

use std::collections::{BTreeMap, HashSet};

use async_trait::async_trait;
use serde_json::{Value, json};
use switchyard_protocol::{
    Category, ChoiceOption, DecisionKind, DecisionQuestion, DecisionRequest, DecisionValue,
    ModelId, Request, Response,
};

use crate::algorithms::llm_class::TaskInput;
use crate::algorithms::util::affinity::ClassifyTrigger;
use crate::algorithms::util::llm_judge::{libsy_error_reason, report_fail_open};
use crate::algorithms::util::robustness::safe_error_summary;
use crate::{Classification, Classifier, Driver, LibsyError, Result, Score, State};

/// Trailing turns sent to a decision service when the route sets no window.
const DEFAULT_TURN_WINDOW: usize = 10;

/// Instructions, evidence, and cutoff policy for a decision judge.
///
/// Candidate labels keep model names out of the generated context. Evidence must use
/// the same labels, with unknown outcomes left unknown. Only the first runtime
/// capable and efficient targets are compared; extra candidates do not add routes.
#[derive(Clone, Debug)]
pub struct DecisionJudgeConfig {
    /// Route capable only when its choice score is strictly above this cutoff.
    /// This score is not a calibrated solve probability. Choose a cutoff from evaluations.
    pub cutoff: f64,
    /// Replaces the packaged instructions. Without custom criteria, preserve
    /// the meaning of `advantage` and `no_advantage`.
    pub instructions: Option<Value>,
    /// Two choice descriptions keyed by candidate label. Requires custom instructions.
    pub criteria: Option<BTreeMap<String, Value>>,
    /// Maps anonymous labels used in evidence (e.g. `"a"`) to runtime model IDs.
    /// Every compared target needs a unique label; extra candidates provide context.
    pub candidates: BTreeMap<String, ModelId>,
    /// JSON passed unchanged to the judge, such as candidate descriptions, reference
    /// cases, outcome/cost summaries, and selection notes. The judge interprets it
    /// using the instructions; the router neither reads its fields nor derives statistics.
    pub evidence: Value,
}

impl DecisionJudgeConfig {
    pub(super) fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.cutoff) {
            return Err(LibsyError::AlgorithmError {
                message: "decision cutoff must be between 0 and 1".into(),
            });
        }
        if self.candidates.values().collect::<HashSet<_>>().len() != self.candidates.len() {
            return Err(LibsyError::AlgorithmError {
                message: "decision candidates must map to distinct targets".into(),
            });
        }
        if let Some(criteria) = &self.criteria
            && (self.instructions.is_none()
                || criteria.len() != 2
                || !criteria.keys().eq(self.candidates.keys()))
        {
            return Err(LibsyError::AlgorithmError {
                message: "custom decision criteria require instructions and exactly two matching candidates".into(),
            });
        }
        Ok(())
    }

    fn candidate(&self, model: &ModelId) -> Result<&str> {
        self.candidates
            .iter()
            .find(|(_, target)| *target == model)
            .map(|(id, _)| id.as_str())
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: format!("decision candidate is missing for target {model}"),
            })
    }
}

pub(super) struct DecisionClassifier {
    config: DecisionJudgeConfig,
    input: TaskInput,
    fail_open: bool,
    question: DecisionQuestion,
}

impl DecisionClassifier {
    pub(super) fn new(
        mut config: DecisionJudgeConfig,
        mut input: TaskInput,
        fail_open: bool,
    ) -> Result<Self> {
        input.recent_turn_window.get_or_insert(DEFAULT_TURN_WINDOW);
        let instructions = match config.instructions.take() {
            Some(instructions) => instructions,
            None => serde_json::from_str(include_str!(
                "../../prompts/capability-classifier/relative_advantage.json"
            ))
            .map_err(|error| LibsyError::external("loading decision judge instructions", error))?,
        };
        let options = match &config.criteria {
            Some(criteria) => criteria
                .iter()
                .map(|(id, description)| ChoiceOption {
                    id: id.clone(),
                    description: Some(description.clone()),
                })
                .collect(),
            None => [
                ("advantage", "The capable candidate succeeds AND the efficient candidate fails."),
                ("no_advantage", "The efficient candidate succeeds OR the capable candidate fails, including shared failure."),
            ]
            .into_iter()
            .map(|(id, description)| ChoiceOption {
                id: id.into(),
                description: Some(json!(description)),
            })
            .collect(),
        };
        let question = DecisionQuestion {
            instructions,
            kind: DecisionKind::Choice { options },
        };
        Ok(Self {
            config,
            input,
            fail_open,
            question,
        })
    }

    async fn try_score(
        &self,
        decision: DecisionRequest,
        driver: &Driver,
        judge: &ModelId,
        score_option: &str,
    ) -> Result<f64> {
        let response = driver.call_decision(decision, judge.clone()).await?;
        let answer = response
            .answers
            .get("route")
            .ok_or_else(|| LibsyError::invalid_decision(judge.clone(), "missing route answer"))?;
        let DecisionValue::Choice {
            probabilities: Some(probabilities),
            ..
        } = &answer.value
        else {
            return Err(LibsyError::invalid_decision(
                judge.clone(),
                "expected a choice with probabilities",
            ));
        };
        // The selected option can differ from the cutoff event, so only the
        // requested option's probability is used.
        let score = probabilities
            .get(score_option)
            .map(|p| p.0)
            .ok_or_else(|| {
                LibsyError::invalid_decision(
                    judge.clone(),
                    format!("missing probability for {score_option}"),
                )
            })?;
        if !(0.0..=1.0).contains(&score) {
            return Err(LibsyError::invalid_decision(
                judge.clone(),
                "decision probability must be between 0 and 1",
            ));
        }
        Ok(score)
    }
}

#[async_trait]
impl Classifier for DecisionClassifier {
    async fn score(
        &self,
        _state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let judge = driver.first_model_for(&Category::Judge)?;
        let capable = driver.first_model_for(&Category::Capable)?;
        let efficient = driver.first_model_for(&Category::Efficient)?;
        let score_option = if self.config.criteria.is_some() {
            self.config.candidate(capable)?
        } else {
            "advantage"
        };
        let decision = DecisionRequest {
            model: None,
            context: json!({
                "task": self.input.messages(request),
                "session": request.metadata.as_ref().and_then(|m| m.session_id.as_deref()),
                "candidates": self.config.candidates.keys().collect::<Vec<_>>(),
                "comparison": {
                    "capable": self.config.candidate(capable)?,
                    "efficient": self.config.candidate(efficient)?,
                },
                "evidence": self.config.evidence,
            }),
            questions: BTreeMap::from([("route".into(), self.question.clone())]),
        };
        let score = match self.try_score(decision, driver, judge, score_option).await {
            Ok(score) => score,
            Err(error) if self.fail_open => {
                return Ok(unavailable(
                    driver,
                    judge,
                    safe_error_summary(&error),
                    libsy_error_reason(&error),
                ));
            }
            Err(error) => return Err(error),
        };
        let (target, category) = if score > self.config.cutoff {
            (capable, Category::Capable)
        } else {
            (efficient, Category::Efficient)
        };
        let evidence = json!({
            "source": "decision_classifier",
            "verdict": if self.config.criteria.is_some() { "candidate_choice" } else { "relative_advantage" },
            "selected_candidate": self.config.candidate(target)?,
            "score_option": score_option,
            "score": score,
            "threshold": self.config.cutoff,
        });
        let selected = Score {
            target: target.clone(),
            confidence: 1.0,
            category: Some(category),
        };
        driver.record_judgment(Some(selected.clone()), Some(evidence.clone()));
        driver.set_evidence(evidence);
        Ok((Classification::Scores(vec![selected]), None))
    }
}

/// Settings for a decision service that selects one model category.
#[derive(Clone, Debug)]
pub struct DirectClassifierConfig {
    /// Categories the service may select. Their names are the option IDs.
    pub options: Vec<Category>,
    /// Routes to the default target when the decision call fails or returns an
    /// invalid choice.
    pub fail_open: bool,
    /// How often the decision service re-decides this session's target.
    pub classify_trigger: ClassifyTrigger,
    /// Uses the first user message when session metadata is unavailable.
    pub message_hash_fallback: bool,
    /// Trailing conversation turns sent to the decision service. Defaults to 10.
    pub recent_turn_window: Option<usize>,
}

pub(super) struct DirectClassifier {
    config: DirectClassifierConfig,
    input: TaskInput,
    question: DecisionQuestion,
}

impl DirectClassifier {
    pub(super) fn new(config: DirectClassifierConfig) -> Result<Self> {
        if config.options.len() < 2
            || config
                .options
                .iter()
                .any(|option| matches!(option, Category::Judge | Category::Any))
        {
            return Err(LibsyError::AlgorithmError {
                message: "direct mode needs at least two options other than judge and any".into(),
            });
        }
        let question = DecisionQuestion {
            instructions: json!("Select the option that should serve this request."),
            kind: DecisionKind::Choice {
                options: config
                    .options
                    .iter()
                    .map(|option| ChoiceOption {
                        id: option.as_str().into(),
                        description: None,
                    })
                    .collect(),
            },
        };
        Ok(Self {
            input: TaskInput {
                recent_turn_window: Some(config.recent_turn_window.unwrap_or(DEFAULT_TURN_WINDOW)),
            },
            config,
            question,
        })
    }

    async fn try_select(
        &self,
        decision: DecisionRequest,
        driver: &Driver,
        judge: &ModelId,
    ) -> Result<&Category> {
        let response = driver.call_decision(decision, judge.clone()).await?;
        let Some(DecisionValue::Choice { selected, .. }) =
            response.answers.get("route").map(|answer| &answer.value)
        else {
            return Err(LibsyError::invalid_decision(
                judge.clone(),
                "expected a route choice",
            ));
        };
        self.config
            .options
            .iter()
            .find(|option| option.as_str() == selected)
            .ok_or_else(|| {
                LibsyError::invalid_decision(judge.clone(), "route choice is not an option")
            })
    }
}

#[async_trait]
impl Classifier for DirectClassifier {
    async fn score(
        &self,
        _state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let judge = driver.first_model_for(&Category::Judge)?;
        let decision = DecisionRequest {
            model: None,
            context: json!({
                "task": self.input.messages(request),
                "session": request.metadata.as_ref().and_then(|m| m.session_id.as_deref()),
            }),
            questions: BTreeMap::from([("route".into(), self.question.clone())]),
        };
        let category = match self.try_select(decision, driver, judge).await {
            Ok(category) => category,
            Err(error) if self.config.fail_open => {
                return Ok(unavailable(
                    driver,
                    judge,
                    safe_error_summary(&error),
                    libsy_error_reason(&error),
                ));
            }
            Err(error) => return Err(error),
        };
        let evidence = json!({
            "source": "decision_classifier",
            "verdict": "direct",
            "selected": category.as_str(),
        });
        let selected = Score {
            target: driver.first_model_for(category)?.clone(),
            confidence: 1.0,
            category: Some(category.clone()),
        };
        driver.record_judgment(Some(selected.clone()), Some(evidence.clone()));
        driver.set_evidence(evidence);
        Ok((Classification::Scores(vec![selected]), None))
    }
}

fn unavailable(
    driver: &Driver,
    judge: &ModelId,
    error: String,
    reason: &'static str,
) -> (Classification, Option<Response>) {
    report_fail_open(judge.as_str(), error, reason);
    let evidence = json!({"source": "fail_open", "reason_code": reason});
    driver.record_judgment(None, Some(evidence.clone()));
    driver.set_evidence_if_empty(evidence);
    (Classification::Ambiguous(vec![]), None)
}
