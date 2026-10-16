// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compare two candidates by the chance that the capable candidate alone succeeds.

use std::collections::{BTreeMap, HashSet};

use async_trait::async_trait;
use serde_json::{Value, json};
use switchyard_protocol::{
    Category, ChoiceOption, ContentBlock, DecisionKind, DecisionQuestion, DecisionRequest,
    DecisionValue, ImageSource, ModelId, Request, Response,
};

use crate::algorithms::llm_class::TaskInput;
use crate::algorithms::util::llm_judge::{libsy_error_reason, report_fail_open};
use crate::algorithms::util::robustness::safe_error_summary;
use crate::{Classification, Classifier, Driver, LibsyError, Result, Score, State};

/// Evidence and policy for a relative-advantage decision judge.
///
/// Candidate labels keep model names out of the generated context. Evidence must use
/// the same labels, with unknown outcomes left unknown. Only the first runtime
/// capable and efficient targets are compared; extra candidates do not add routes.
#[derive(Clone, Debug)]
pub struct DecisionJudgeConfig {
    /// Route capable only when its advantage score is strictly above this cutoff.
    /// This score is not a calibrated solve probability. Choose a cutoff from evaluations.
    pub cutoff: f64,
    /// Replaces the packaged structured instructions; must keep the meaning of
    /// `advantage` (capable succeeds and efficient fails) and `no_advantage`.
    pub instructions: Option<Value>,
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
        input: TaskInput,
        fail_open: bool,
    ) -> Result<Self> {
        let instructions = match config.instructions.take() {
            Some(instructions) => instructions,
            None => serde_json::from_str(include_str!(
                "../../prompts/capability-classifier/relative_advantage.json"
            ))
            .map_err(|error| LibsyError::external("loading decision judge instructions", error))?,
        };
        let question = DecisionQuestion {
            instructions,
            kind: DecisionKind::Choice {
                options: [
                    ("advantage", "The capable candidate succeeds AND the efficient candidate fails."),
                    ("no_advantage", "The efficient candidate succeeds OR the capable candidate fails, including shared failure."),
                ]
                .into_iter()
                .map(|(id, description)| ChoiceOption {
                    id: id.into(),
                    description: Some(json!(description)),
                })
                .collect(),
            },
        };
        Ok(Self {
            config,
            input,
            fail_open,
            question,
        })
    }
}

#[async_trait]
impl Classifier<State> for DecisionClassifier {
    async fn score(
        &self,
        _state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let judge = driver.first_model_for(&Category::Judge)?;
        let capable = driver.first_model_for(&Category::Capable)?;
        let efficient = driver.first_model_for(&Category::Efficient)?;
        let mut messages = self.input.messages(request);
        let mut images = Vec::new();
        for message in &mut messages {
            extract_images(&mut message.content, &mut images);
        }
        let decision = DecisionRequest {
            model: None,
            images,
            context: json!({
                "task": messages,
                "candidates": self.config.candidates.keys().collect::<Vec<_>>(),
                "comparison": {
                    "capable": self.config.candidate(capable)?,
                    "efficient": self.config.candidate(efficient)?,
                },
                "evidence": self.config.evidence,
            }),
            questions: BTreeMap::from([("route".into(), self.question.clone())]),
        };
        let response = match driver.call_decision(decision, judge.clone()).await {
            Ok(response) => response,
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
        // The provider's selected option can differ from the application's cutoff.
        // Only the requested event's score is used; confidence is a separate signal.
        let advantage = response
            .answers
            .get("route")
            .and_then(|answer| match &answer.value {
                DecisionValue::Choice {
                    probabilities: Some(probabilities),
                    ..
                } => probabilities.get("advantage").map(|p| p.0),
                _ => None,
            })
            .filter(|p| (0.0..=1.0).contains(p));
        let Some(advantage) = advantage else {
            return Ok(unavailable(
                driver,
                judge,
                "missing or invalid advantage score".into(),
                "invalid_verdict",
            ));
        };
        let (target, category) = if advantage > self.config.cutoff {
            (capable, Category::Capable)
        } else {
            (efficient, Category::Efficient)
        };
        driver.set_evidence(json!({
            "source": "decision_classifier",
            "verdict": "relative_advantage",
            "score": advantage,
            "threshold": self.config.cutoff,
        }));
        Ok((
            Classification::Scores(vec![Score {
                target: target.clone(),
                confidence: 1.0,
                category: Some(category),
            }]),
            None,
        ))
    }
}

fn extract_images(blocks: &mut [ContentBlock], images: &mut Vec<ImageSource>) {
    for block in blocks {
        match block {
            ContentBlock::Image { source } => {
                images.push(source.clone());
                // Keep the image's position without tokenizing its encoded bytes.
                *block = ContentBlock::Text {
                    text: format!("[Image {}]", images.len()),
                };
            }
            ContentBlock::ToolResult(result) => extract_images(&mut result.content, images),
            _ => {}
        }
    }
}

fn unavailable(
    driver: &Driver,
    judge: &ModelId,
    error: String,
    reason: &'static str,
) -> (Classification, Option<Response>) {
    report_fail_open(judge.as_str(), error, reason);
    driver.set_evidence_if_empty(json!({"source": "fail_open", "reason_code": reason}));
    (Classification::Ambiguous(vec![]), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::ToolResult;

    #[test]
    fn extracts_tool_images_in_order() {
        let image = |url: &str| ImageSource::Url {
            url: url.into(),
            detail: None,
        };
        let mut blocks = vec![
            ContentBlock::Image {
                source: image("https://example.test/first.png"),
            },
            ContentBlock::ToolResult(ToolResult {
                tool_call_id: "capture".into(),
                content: vec![ContentBlock::Image {
                    source: image("https://example.test/second.png"),
                }],
                is_error: Some(false),
            }),
        ];
        let mut images = Vec::new();
        extract_images(&mut blocks, &mut images);
        assert_eq!(
            images,
            vec![
                image("https://example.test/first.png"),
                image("https://example.test/second.png")
            ]
        );
        let ContentBlock::ToolResult(result) = &blocks[1] else {
            panic!("expected tool result")
        };
        assert_eq!(result.tool_call_id, "capture");
        assert_eq!(
            result.content,
            vec![ContentBlock::Text {
                text: "[Image 2]".into()
            }]
        );
        assert_eq!(result.is_error, Some(false));
    }
}
