// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Judge-selected handoff notes for a fixed target.

use std::collections::BTreeMap;

use jsonptr::PointerBuf;
use serde_json::Value;
use switchyard_protocol::{ModelId, Request};

use super::llm_judge::JudgePolicy;
use crate::core::algorithm::Driver;
use crate::core::classifier::Classification;
use crate::{LibsyError, Result};

/// Selects trusted handoff text by an ID in the judge's verdict.
#[derive(Clone, Debug)]
pub struct HandoffNotePolicy {
    target: ModelId,
    selector: PointerBuf,
    notes: BTreeMap<String, String>,
}

impl HandoffNotePolicy {
    /// Validates a verdict JSON Pointer and an ID-to-note database.
    pub fn new(
        target: ModelId,
        selector: impl Into<String>,
        notes: BTreeMap<String, String>,
    ) -> Result<Self> {
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
        Ok(Self {
            target,
            selector,
            notes,
        })
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
}

impl JudgePolicy for HandoffNotePolicy {
    type Verdict = Value;

    fn to_classification(
        &self,
        _verdict: Option<&Value>,
        _driver: &Driver,
    ) -> Result<Classification> {
        Ok(super::decisive(&self.target))
    }

    fn apply_handoff_note(&self, verdict: &Value, request: &mut Request) {
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
