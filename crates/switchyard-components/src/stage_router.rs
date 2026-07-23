// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage-router scoring and tier selection — the shared routing core.
//!
//! Given a [`ToolResultSignal`] (extracted by the dimension collector), this
//! module decides whether a coding-agent turn should go to the **capable**
//! (strong) or **efficient** (weak) tier. It is the single source of truth for
//! the decision, shared by the Rust profile and the Python processor via
//! bindings — only the outer shell differs in how it fetches the decision.
//!
//! Two axes:
//!
//! * **error** — did the recent tool results error? (`severity`)
//! * **production** — is the agent producing code? (`spinning` / `exploring`
//!   push toward capable; `production_intensity` pushes toward efficient)
//!
//! Signals are scored with fixed weights, summed, and `tanh`-squashed into
//! `(-1, +1)`; `confidence` is the magnitude. The `confidence_threshold` dials
//! how much corroboration a decisive escalation needs (see [`score_signal`]).

use crate::dimension_collector::ToolResultSignal;

/// Turn depth below which stall signals stay quiet — early no-write turns are
/// normal exploration, not a stall.
const STALL_MIN_TURN_DEPTH: u32 = 8;
/// Gain applied before the tanh squash — spreads the small raw weighted sum
/// across the usable confidence range. Without it confidence would cap near
/// ±0.20 and mid/high thresholds would be unreachable.
const SCORE_GAIN: f64 = 5.0;
/// Strongest error severity the scorer sees: critical (`1.0`) is caught by the
/// override, so hard (`0.7`) normalises `severity` to one signal unit.
const HARD_SEVERITY: f64 = 0.7;
/// Weight one maxed signal contributes. Small enough that no single axis pegs
/// the decision; corroboration across the two axes is what raises confidence.
const SIGNAL_UNIT: f64 = 0.10;
/// Critical severity forces the capable tier regardless of the scorer.
const SEVERITY_CRITICAL: f32 = 1.0;

/// Signed, fixed weights over the two axes. Error signals (`severity`,
/// `spinning`, `exploring`) push toward capable (+); `production_intensity`
/// pushes toward efficient (−). `severity` is normalised by its hard cap so it
/// too contributes one `SIGNAL_UNIT` when maxed.
const DEFAULT_WEIGHTS: &[(&str, f64)] = &[
    ("severity", SIGNAL_UNIT / HARD_SEVERITY),
    ("spinning", SIGNAL_UNIT),
    ("exploring", SIGNAL_UNIT),
    ("production_intensity", -SIGNAL_UNIT),
];

/// The two tiers a turn can route to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    /// Weak / cheap tier.
    Efficient,
    /// Strong / capable tier.
    Capable,
}

/// Which tier to default to when the scorer is not confident.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PickerMode {
    /// Default to capable unless the scorer confidently picks efficient.
    CapableFirst,
    /// Default to efficient unless the scorer confidently picks capable.
    EfficientFirst,
}

impl PickerMode {
    fn default_tier(self) -> Tier {
        match self {
            Self::CapableFirst => Tier::Capable,
            Self::EfficientFirst => Tier::Efficient,
        }
    }
}

/// What produced a decision — for stats and explainability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionSource {
    /// Hard override (critical severity or context compaction).
    Override,
    /// Settled run: recent tests passed with recent production and no error.
    TestsPassed,
    /// Scorer crossed `confidence_threshold`.
    Dimensions,
    /// Scorer was not confident; the caller should consult its classifier.
    FallOpen,
}

impl DecisionSource {
    /// Stable lowercase label used in stats.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Override => "override",
            Self::TestsPassed => "tests_passed",
            Self::Dimensions => "dimensions",
            Self::FallOpen => "fall_open",
        }
    }
}

/// A signed score in `(-1, +1)` and its magnitude. `confidence == score.abs()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreResult {
    /// Signed score: positive → capable, negative → efficient.
    pub score: f64,
    /// Decision certainty, `score.abs()`.
    pub confidence: f64,
}

/// The two-axis feature view of a single [`ToolResultSignal`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CodingAgentDimensions {
    /// Windowed max error severity in `[0, 1]`.
    pub severity: f64,
    /// `1.0` when a deep turn has no reads, plans, writes, or edits (pure churn).
    pub spinning: f64,
    /// `1.0` when a deep turn reads/plans but does not write or edit.
    pub exploring: f64,
    /// Fraction of recent tool ops that produced code (writes + edits).
    pub production_intensity: f64,
}

impl CodingAgentDimensions {
    fn value(&self, name: &str) -> f64 {
        match name {
            "severity" => self.severity,
            "spinning" => self.spinning,
            "exploring" => self.exploring,
            "production_intensity" => self.production_intensity,
            _ => 0.0,
        }
    }
}

/// Outcome of [`pick_tier`]: either a resolved decision, or a signal that the
/// caller should consult its (impl-specific, async) classifier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PickOutcome {
    /// The tier was decided without the classifier.
    Resolved {
        /// The chosen tier.
        tier: Tier,
        /// What produced it.
        source: DecisionSource,
        /// Signed scorer value (`0.0` for override / tests-passed).
        score: f64,
        /// Scorer confidence (`None` where the scorer did not run).
        confidence: Option<f64>,
    },
    /// The scorer was below threshold. The caller runs its classifier; if it
    /// has none (or it fails), fall open to `default_tier`.
    ConsultClassifier {
        /// Signed scorer value, for logging.
        score: f64,
        /// Scorer confidence, for logging.
        confidence: f64,
        /// Tier to fall open to when no classifier resolves it.
        default_tier: Tier,
    },
}

/// Project a [`ToolResultSignal`] onto the two-axis dimension space.
pub fn dimensions_from_signal(signal: &ToolResultSignal) -> CodingAgentDimensions {
    let recent_ops = signal.recent_write_count
        + signal.recent_edit_count
        + signal.recent_read_count
        + signal.recent_todowrite_count;
    let deep_enough = signal.turn_depth >= STALL_MIN_TURN_DEPTH;
    let no_production = signal.recent_write_count == 0 && signal.recent_edit_count == 0;
    let investigating = signal.recent_read_count >= 1 || signal.recent_todowrite_count >= 1;
    // spinning vs exploring partition the "not producing" case by investigative
    // activity, so at most one fires — no double-counting on the production axis.
    let spinning = deep_enough && no_production && !investigating;
    let exploring = deep_enough && no_production && investigating;

    CodingAgentDimensions {
        severity: f64::from(signal.severity),
        spinning: if spinning { 1.0 } else { 0.0 },
        exploring: if exploring { 1.0 } else { 0.0 },
        production_intensity: ratio(
            signal.recent_write_count + signal.recent_edit_count,
            recent_ops,
        ),
    }
}

/// Score a signal: weighted sum of the dimensions, `tanh`-squashed.
///
/// The raw sum is small — one maxed signal is `±0.10`, two corroborating
/// signals `±0.20`. `tanh(gain·raw)` spreads that into a usable range, so the
/// `confidence_threshold` reads roughly: `~0.3` escalates on one signal, `~0.5`
/// needs about one-and-a-half, `~0.7` needs two to corroborate.
pub fn score_signal(signal: &ToolResultSignal) -> ScoreResult {
    let dimensions = dimensions_from_signal(signal);
    let raw: f64 = DEFAULT_WEIGHTS
        .iter()
        .map(|(name, weight)| dimensions.value(name) * weight)
        .sum();
    let score = (SCORE_GAIN * raw).tanh();
    ScoreResult {
        score,
        confidence: score.abs(),
    }
}

/// Hard **escalate** — force the strong tier no matter what the scorer would
/// say. Fires on a critical error or a compacted context.
fn should_escalate(signal: &ToolResultSignal) -> bool {
    // Compaction wipes the accumulated signals, so a task that had escalated
    // would snap back to weak — a context big enough to overflow belongs strong.
    if signal.compacted {
        return true;
    }
    // A critical error is unambiguous.
    signal.severity >= SEVERITY_CRITICAL
}

/// Hard **de-escalate** — drop to the cheap tier on a settled turn: tests
/// passed, code was just written or edited, and nothing errored in the window.
fn should_deescalate(signal: &ToolResultSignal) -> bool {
    signal.tests_passed
        && (signal.recent_write_count + signal.recent_edit_count) >= 1
        && signal.severity <= 0.0
}

/// Decide a turn's tier from its signal.
///
/// The rules run in order; the first that fires wins:
///
/// 1. **Escalate** — a hard reason to go strong (critical error / compaction).
/// 2. **De-escalate** — a hard reason to go cheap (a settled turn).
/// 3. **Scorer** — no hard reason, so weigh the two axes; if confident, follow it.
/// 4. **Fall open** — not confident: hand to the classifier, else the default.
///
/// Rules 1 and 2 are the two hard shortcuts that skip the scorer — one always
/// escalates, one always de-escalates. **Escalate is checked first**, so a
/// critical error still wins on a turn whose tests also happened to pass.
///
/// Deterministic and pure: the async classifier lives in the caller, so rule 4
/// returns [`PickOutcome::ConsultClassifier`] instead of calling it here. The
/// `no_signal` case (no tool activity yet) is handled one level up.
pub fn pick_tier(
    signal: &ToolResultSignal,
    mode: PickerMode,
    confidence_threshold: f64,
) -> PickOutcome {
    // 1. Escalate — a hard reason to go strong, ahead of everything else.
    if should_escalate(signal) {
        return resolved(Tier::Capable, DecisionSource::Override, 0.0, Some(1.0));
    }

    // 2. De-escalate — a hard reason to go cheap (the turn is winding down).
    if should_deescalate(signal) {
        return resolved(Tier::Efficient, DecisionSource::TestsPassed, 0.0, None);
    }

    // 3. Scorer — no hard reason either way, so weigh error vs production. If
    //    confident enough, follow the sign: positive → strong, negative → cheap.
    let scored = score_signal(signal);
    if scored.confidence >= confidence_threshold {
        let tier = if scored.score > 0.0 {
            Tier::Capable
        } else {
            Tier::Efficient
        };
        return resolved(tier, DecisionSource::Dimensions, scored.score, Some(scored.confidence));
    }

    // 4. Fall open — the signals didn't corroborate enough to be sure. Hand off
    //    to the caller's classifier; with none, land on the picker's default.
    PickOutcome::ConsultClassifier {
        score: scored.score,
        confidence: scored.confidence,
        default_tier: mode.default_tier(),
    }
}

/// Build a resolved outcome (a decision made without the classifier).
fn resolved(tier: Tier, source: DecisionSource, score: f64, confidence: Option<f64>) -> PickOutcome {
    PickOutcome::Resolved {
        tier,
        source,
        score,
        confidence,
    }
}

fn ratio(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        f64::from(numerator) / f64::from(denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension_collector::extract_tool_signals;
    use switchyard_core::ChatRequest;
    use serde_json::json;

    fn signal_from(messages: serde_json::Value) -> ToolResultSignal {
        let request = ChatRequest::openai_chat(json!({"model": "m", "messages": messages}));
        extract_tool_signals(&request)
    }

    #[test]
    fn critical_severity_overrides_to_capable() {
        let mut signal = signal_from(json!([{"role": "user", "content": "hi"}]));
        signal.severity = SEVERITY_CRITICAL;
        match pick_tier(&signal, PickerMode::EfficientFirst, 0.5) {
            PickOutcome::Resolved { tier, source, .. } => {
                assert_eq!(tier, Tier::Capable);
                assert_eq!(source, DecisionSource::Override);
            }
            other => panic!("expected override, got {other:?}"),
        }
    }

    #[test]
    fn compaction_overrides_to_capable() {
        let mut signal = signal_from(json!([{"role": "user", "content": "hi"}]));
        signal.compacted = true;
        assert!(matches!(
            pick_tier(&signal, PickerMode::EfficientFirst, 0.5),
            PickOutcome::Resolved { tier: Tier::Capable, source: DecisionSource::Override, .. }
        ));
    }

    #[test]
    fn one_signal_scores_below_half() {
        // A single full wrong signal ≈ 0.46 confidence — just under 0.5.
        let mut signal = signal_from(json!([{"role": "user", "content": "hi"}]));
        signal.severity = HARD_SEVERITY as f32;
        let scored = score_signal(&signal);
        assert!(scored.score > 0.0);
        assert!(scored.confidence < 0.5, "one signal should not clear 0.5: {scored:?}");
    }

    #[test]
    fn quiet_signal_falls_open_to_default() {
        let signal = signal_from(json!([{"role": "user", "content": "hi"}]));
        match pick_tier(&signal, PickerMode::EfficientFirst, 0.5) {
            PickOutcome::ConsultClassifier { default_tier, .. } => {
                assert_eq!(default_tier, Tier::Efficient);
            }
            other => panic!("expected consult-classifier, got {other:?}"),
        }
    }
}
