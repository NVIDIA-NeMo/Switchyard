// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic target selection from a validated JSON classifier verdict.

use jsonptr::PointerBuf;
use serde_json::Value;

use super::llm_judge::JudgePolicy;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Score};
use crate::{LibsyError, Result};
use switchyard_protocol::Category;

/// Maps one string field in a validated verdict to a runtime model category.
pub(crate) struct TargetSelectorPolicy {
    selector: PointerBuf,
}

impl TargetSelectorPolicy {
    /// Parses a JSON Pointer used to read validated verdicts.
    pub(crate) fn new(selector: impl Into<String>) -> Result<Self> {
        let selector =
            PointerBuf::parse(selector.into()).map_err(|error| LibsyError::AlgorithmError {
                message: format!("policy selector is not a valid JSON Pointer: {error}"),
            })?;
        if selector.is_root() {
            return Err(LibsyError::AlgorithmError {
                message: "policy selector must identify a response field".to_string(),
            });
        }
        Ok(Self { selector })
    }
}

impl JudgePolicy for TargetSelectorPolicy {
    type Verdict = Value;

    fn to_classification(
        &self,
        verdict: Option<&Self::Verdict>,
        driver: &Driver,
    ) -> Result<Classification> {
        let target = verdict
            .and_then(|verdict| self.selector.resolve(verdict).ok())
            .and_then(Value::as_str)
            .and_then(|label| label.parse::<Category>().ok())
            // The judge decides the turn; it is not somewhere to route it.
            .filter(|category| *category != Category::Judge)
            .and_then(|category| {
                let target = driver.models_for(&category).first()?.clone();
                Some((category, target))
            });
        match target {
            Some((category, target)) => Ok(Classification::Scores(vec![Score {
                target,
                confidence: 1.0,
                category: Some(category),
            }])),
            None => Ok(Classification::Ambiguous(vec![])),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::algorithm::RuntimeModels;
    use serde_json::json;
    use std::sync::Arc;
    use switchyard_protocol::ModelId;

    use super::*;
    use crate::Result;

    fn driver() -> Driver {
        Driver::new(
            "test",
            Arc::new(RuntimeModels::new(
                [(Category::Capable, vec![ModelId::from("model/opus")])].into(),
            )),
        )
        .0
    }

    #[test]
    fn a_verdict_selects_its_runtime_category() -> Result<()> {
        let policy = TargetSelectorPolicy::new("/decision/target")?;
        let classification = policy.to_classification(
            Some(&json!({
                "decision": {"target": "capable"}
            })),
            &driver(),
        )?;

        assert_eq!(
            classification.argmax(false)?.map(|score| score.target),
            Some(ModelId::from("model/opus"))
        );
        Ok(())
    }

    #[test]
    fn a_missing_unknown_or_unavailable_target_abstains() -> Result<()> {
        let policy = TargetSelectorPolicy::new("/target")?;
        let driver = driver();

        for verdict in [
            json!({"target": "efficient"}),
            json!({"target": "unknown"}),
            json!({"reason": "missing"}),
        ] {
            assert_eq!(
                policy
                    .to_classification(Some(&verdict), &driver)?
                    .argmax(false)?,
                None
            );
        }
        Ok(())
    }

    #[test]
    fn an_invalid_json_pointer_is_rejected() {
        let result = TargetSelectorPolicy::new("/target~2name");
        assert!(matches!(result, Err(LibsyError::AlgorithmError { message })
                if message.contains("valid JSON Pointer")));
    }
}
