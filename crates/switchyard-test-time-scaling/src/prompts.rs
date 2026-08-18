// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small prompt builders used by the controller.

use crate::{Candidate, Summary, Task};

const COMPARISON_INSTRUCTIONS: &str = "Compare the numbered candidate summaries for the original task. Use only recorded evidence and do not assume hidden tests passed. Reject a candidate with a missing patch or artifact, an inconsistent final state, or an unresolved fatal error. Then compare code correctness, completeness, verification after the final edit, test results, root-cause coverage, confirmed command output, and reasonable task interpretation, in that order. Explain the decisive evidence. End with exactly: Final verdict: Solution N";
const REFINEMENT_PREAMBLE: &str = "You are starting a new independent attempt in a fresh environment. The prior summaries may contain successes, failures, conflicting diagnoses, or unverified claims. Use them as evidence, not as ground truth.";
const REFINEMENT_POSTAMBLE: &str = "Combine useful findings, avoid repeated dead ends, and verify the solution in this fresh environment. Prior files and patches are not available unless you recreate them.";

/// Renders the shared prior-attempt context for a refined rollout.
pub fn refinement_prompt(summaries: &[Summary]) -> String {
    let mut prompt = format!("{REFINEMENT_PREAMBLE}\n");
    for (index, summary) in summaries.iter().enumerate() {
        prompt.push_str(&format!(
            "\nPRIOR ATTEMPT SUMMARY {}\n{}\n",
            index + 1,
            serde_json::Value::Object(summary.value.clone())
        ));
    }
    prompt.push_str(&format!("\n{REFINEMENT_POSTAMBLE}"));
    prompt
}

pub(crate) fn comparison_prompt(task: &Task, candidates: &[Candidate]) -> String {
    let mut prompt = format!(
        "{COMPARISON_INSTRUCTIONS}\n\nOriginal task:\n{}\n\nCandidates:\n",
        task.prompt
    );
    for (index, candidate) in candidates.iter().enumerate() {
        prompt.push_str(&format!(
            "\nSolution {}:\n{}\n",
            index + 1,
            serde_json::Value::Object(candidate.summary.value.clone())
        ));
    }
    prompt
}
