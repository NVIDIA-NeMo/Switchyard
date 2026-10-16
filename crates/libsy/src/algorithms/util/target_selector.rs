// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic target selection from a validated JSON classifier verdict.

use std::collections::BTreeMap;

use jsonptr::PointerBuf;
use serde_json::Value;

use super::llm_judge::JudgePolicy;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Score};
use crate::{LibsyError, Result};
use switchyard_protocol::{Category, Request};

/// Selects trusted handoff text by an ID in the judge's verdict.
#[derive(Clone, Debug)]
pub struct HandoffNoteSelector {
    selector: PointerBuf,
    notes: BTreeMap<String, String>,
}

impl HandoffNoteSelector {
    /// Validates a verdict JSON Pointer and an ID-to-note database.
    pub fn new(selector: impl Into<String>, notes: BTreeMap<String, String>) -> Result<Self> {
        let selector =
            PointerBuf::parse(selector.into()).map_err(|error| LibsyError::AlgorithmError {
                message: format!("handoff note selector is not a valid JSON Pointer: {error}"),
            })?;
        if selector.is_root()
            || notes.is_empty()
            || notes
                .iter()
                .any(|(id, text)| id.trim().is_empty() || text.trim().is_empty())
        {
            return Err(LibsyError::AlgorithmError {
                message: "handoff notes require a response field and nonempty note IDs and text"
                    .into(),
            });
        }
        Ok(Self { selector, notes })
    }

    pub(crate) fn extend_prompt(&self, prompt: &str) -> Result<String> {
        let notes =
            serde_json::to_string(&self.notes).map_err(|error| LibsyError::AlgorithmError {
                message: format!("could not serialize handoff notes: {error}"),
            })?;
        Ok(format!(
            "{prompt}\n\nAvailable handoff notes (JSON mapping IDs to text):\n{}\n\nSelect at most one relevant note ID at JSON Pointer {} in your verdict. Use null when no note applies. Follow the supplied response schema.",
            notes, self.selector,
        ))
    }

    fn apply(&self, verdict: &Value, request: &mut Request) {
        if let Some(note) = self
            .selector
            .resolve(verdict)
            .ok()
            .and_then(Value::as_str)
            .and_then(|id| self.notes.get(id))
        {
            super::prompts::append_note(request, note);
        }
    }
}

/// Maps one string field in a validated verdict to a runtime model category.
pub(crate) struct TargetSelectorPolicy {
    selector: PointerBuf,
    handoff_notes: Option<HandoffNoteSelector>,
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
        Ok(Self {
            selector,
            handoff_notes: None,
        })
    }

    pub(crate) fn with_handoff_notes(mut self, notes: Option<HandoffNoteSelector>) -> Self {
        self.handoff_notes = notes;
        self
    }
}

impl JudgePolicy for TargetSelectorPolicy {
    type Verdict = Value;

    fn apply_handoff_note(&self, verdict: &Value, request: &mut Request) {
        if let Some(notes) = &self.handoff_notes {
            notes.apply(verdict, request);
        }
    }

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
