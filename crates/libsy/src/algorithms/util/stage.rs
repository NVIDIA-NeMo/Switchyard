// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage-router scoring and tier selection — the shared routing core.
//!
//! Given a [`ToolSignals`] (extracted by the dimension collector), this
//! module decides whether a coding-agent turn should go to the **capable**
//! or **efficient** tier. It is the single source of truth for
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

use async_trait::async_trait;
use serde::Deserialize;

use super::prompts;
use super::tool_signals::ToolSignals;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::state::{State, StateValue};
use crate::Result;
use switchyard_protocol::Request;

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

/// The two tiers a turn can route to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    /// Efficient / cheap tier.
    Efficient,
    /// Capable / powerful tier.
    Capable,
}

impl Tier {
    /// Stable label for stats and the [`routing_tier`](Classifier::routing_tier)
    /// hook, independent of what the tiers' targets are called. These are the
    /// strings the capability route reports too, so a deployment running both
    /// sees one tier vocabulary.
    fn label(self) -> &'static str {
        match self {
            Self::Capable => "strong",
            Self::Efficient => "weak",
        }
    }
}

/// The targets a stage router's two tiers route to.
///
/// The tiers are a fixed pair, but their targets are whatever the deployment
/// calls them, so the classifier scores onto those names and the routed call
/// reaches the right model.
#[derive(Clone, Debug)]
pub struct StageTargets {
    capable: String,
    efficient: String,
}

impl StageTargets {
    /// Name the targets the two tiers route to.
    pub fn new(capable: impl Into<String>, efficient: impl Into<String>) -> Self {
        Self {
            capable: capable.into(),
            efficient: efficient.into(),
        }
    }

    /// The target `tier` routes to.
    pub fn name(&self, tier: Tier) -> &str {
        match tier {
            Tier::Capable => &self.capable,
            Tier::Efficient => &self.efficient,
        }
    }

    /// The tier label for a routed target, or `None` for one outside the pair.
    pub fn label_for(&self, target: &str) -> Option<&'static str> {
        if target == self.capable {
            Some(Tier::Capable.label())
        } else if target == self.efficient {
            Some(Tier::Efficient.label())
        } else {
            None
        }
    }
}

/// Which tier to default to when the scorer is not confident.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PickerMode {
    /// Default to capable unless the scorer confidently picks efficient.
    CapableFirst,
    /// Default to efficient unless the scorer confidently picks capable.
    EfficientFirst,
}

impl PickerMode {
    /// Tier a turn routes to when the scorer is not confident enough to pick.
    pub fn default_tier(self) -> Tier {
        match self {
            Self::CapableFirst => Tier::Capable,
            Self::EfficientFirst => Tier::Efficient,
        }
    }
}

/// `State.extra` key under which the turn's [`DecisionSource`] is recorded.
pub const DECISION_SOURCE_KEY: &str = "decision_source";

/// Record which component decided the turn.
pub(crate) fn record_decision_source(state: &mut State, source: DecisionSource) {
    state.extra.insert(
        DECISION_SOURCE_KEY.to_string(),
        StateValue::String(source.as_str().to_string()),
    );
}

/// What produced a decision — for stats and explainability.
///
/// Each is stamped by the component that knows it, so a turn's final label names
/// whoever actually decided it. [`Ambiguous`](Self::Ambiguous) is the exception:
/// the signal scorer records it on its way out, and whichever classifier resolves
/// the turn overwrites it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionSource {
    /// Hard override (critical severity or context compaction).
    Override,
    /// Settled run: recent tests passed with recent production and no error.
    TestsPassed,
    /// Scorer crossed `confidence_threshold`.
    Dimensions,
    /// Scorer was not confident, so the signals did not decide this turn.
    Ambiguous,
    /// An LLM classifier behind the signals decided the turn.
    LlmClassifier,
    /// Nothing resolved the turn, so it landed on the picker's default tier.
    FallOpen,
}

impl DecisionSource {
    /// Stable lowercase label used in stats.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Override => "override",
            Self::TestsPassed => "tests_passed",
            Self::Dimensions => "dimensions",
            Self::Ambiguous => "ambiguous",
            Self::LlmClassifier => "llm-classifier",
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

/// The two-axis feature view of a single [`ToolSignals`].
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

/// Project a [`ToolSignals`] onto the two-axis dimension space.
pub fn dimensions_from_signal(signal: &ToolSignals) -> CodingAgentDimensions {
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
pub fn score_signal(signal: &ToolSignals) -> ScoreResult {
    let d = dimensions_from_signal(signal);
    // Error axis (severity, spinning, exploring → +capable) minus the production
    // axis (→ −efficient). Each maxed signal is one SIGNAL_UNIT; severity is
    // normalised by its hard cap so it lands there too.
    let raw = SIGNAL_UNIT
        * (d.severity / HARD_SEVERITY + d.spinning + d.exploring - d.production_intensity);
    let score = (SCORE_GAIN * raw).tanh();
    ScoreResult {
        score,
        confidence: score.abs(),
    }
}

/// Hard **escalate** — force the capable tier no matter what the scorer would
/// say. Fires on a critical error or a compacted context.
fn should_escalate(signal: &ToolSignals) -> bool {
    // Compaction wipes the accumulated signals, so a task that had escalated
    // would snap back to efficient — a context big enough to overflow belongs capable.
    if signal.compacted {
        return true;
    }
    // A critical error is unambiguous.
    signal.severity >= SEVERITY_CRITICAL
}

/// Hard **de-escalate** — drop to the cheap tier on a settled turn: tests
/// passed, code was just written or edited, and nothing errored in the window.
fn should_deescalate(signal: &ToolSignals) -> bool {
    signal.tests_passed
        && (signal.recent_write_count + signal.recent_edit_count) >= 1
        && signal.severity <= 0.0
}

/// Decide a turn's tier from its signal.
///
/// The rules run in order; the first that fires wins:
///
/// 1. **Escalate** — a hard reason to go capable (critical error / compaction).
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
pub fn pick_tier(signal: &ToolSignals, mode: PickerMode, confidence_threshold: f64) -> PickOutcome {
    // 1. Escalate — a hard reason to go capable, ahead of everything else.
    if should_escalate(signal) {
        return resolved(Tier::Capable, DecisionSource::Override, 0.0, Some(1.0));
    }

    // 2. De-escalate — a hard reason to go cheap (the turn is winding down).
    if should_deescalate(signal) {
        return resolved(Tier::Efficient, DecisionSource::TestsPassed, 0.0, None);
    }

    // 3. Scorer — no hard reason either way, so weigh error vs production. If
    //    confident enough, follow the sign: positive → capable, negative → efficient.
    let scored = score_signal(signal);
    if scored.confidence >= confidence_threshold {
        let tier = if scored.score > 0.0 {
            Tier::Capable
        } else {
            Tier::Efficient
        };
        return resolved(
            tier,
            DecisionSource::Dimensions,
            scored.score,
            Some(scored.confidence),
        );
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
fn resolved(
    tier: Tier,
    source: DecisionSource,
    score: f64,
    confidence: Option<f64>,
) -> PickOutcome {
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

/// The notes a stage router hands the model it routed to, and the gate deciding
/// which one a turn earns.
///
/// Stateless: a note describes the turn's own signals, so every turn they drive
/// carries one. It rides in the forwarded request only, never in the caller's
/// conversation, so notes cannot accumulate across turns.
#[derive(Clone, Debug, Deserialize)]
pub struct HandoffNoteConfig {
    /// Note handed to the capable tier on a signal-driven escalation.
    escalation_note: String,
    /// Optional note handed back to the efficient tier.
    #[serde(default)]
    deescalation_note: Option<String>,
    /// Restricts the escalation note to signal-driven escalations.
    #[serde(default = "escalation_gate_default")]
    only_on_wrong_signal_escalation: bool,
}

/// Gating the escalation note is the safe default: an ungated note can tell the
/// capable model the efficient one was stalling when it wasn't.
fn escalation_gate_default() -> bool {
    true
}

impl HandoffNoteConfig {
    /// Configure the notes: the `escalation_note` handed to the capable tier, an
    /// optional `deescalation_note` handed back to the efficient tier, and
    /// whether the escalation note fires only on a signal-driven escalation
    /// (`override` / `dimensions`) rather than an ambiguous default.
    pub fn new(
        escalation_note: impl Into<String>,
        deescalation_note: Option<String>,
        only_on_wrong_signal_escalation: bool,
    ) -> Self {
        Self {
            escalation_note: escalation_note.into(),
            deescalation_note,
            only_on_wrong_signal_escalation,
        }
    }

    /// The note for a turn routed to `tier` with picker `source`, or `None` when
    /// no note applies.
    fn note_for(&self, tier: Tier, source: DecisionSource) -> Option<&str> {
        match tier {
            // Escalation to the capable tier. When gated, only a signal-driven
            // escalation qualifies — never an ambiguous default, which would tell
            // the capable model the efficient one was stalling when it wasn't.
            Tier::Capable => {
                let signal_driven = matches!(
                    source,
                    DecisionSource::Override | DecisionSource::Dimensions
                );
                (!self.only_on_wrong_signal_escalation || signal_driven)
                    .then_some(self.escalation_note.as_str())
            }
            // Hand-back to the efficient tier, when a de-escalation note is configured.
            Tier::Efficient => self.deescalation_note.as_deref(),
        }
    }
}

/// Signal-only stage-router classifier: scores each turn onto the capable/efficient
/// tiers from tool-result signals, via the configured picker mode and the
/// confidence the scorer must reach before it acts on the signal alone.
///
/// With [`with_handoff_notes`](Self::with_handoff_notes) it also splices a note
/// into the request explaining why the signals sent the turn where they did.
pub struct StageClassifier {
    targets: StageTargets,
    mode: PickerMode,
    confidence_threshold: f64,
    handoff_notes: Option<HandoffNoteConfig>,
}

impl StageClassifier {
    /// Scores onto `targets`, with the given default tier (`mode`) and
    /// `confidence_threshold`.
    pub fn new(targets: StageTargets, mode: PickerMode, confidence_threshold: f64) -> Self {
        Self {
            targets,
            mode,
            confidence_threshold,
            handoff_notes: None,
        }
    }

    /// Hand the routed model a note on a signal-driven escalation, and on a
    /// hand-back to the efficient tier when a de-escalation note is configured.
    pub fn with_handoff_notes(mut self, config: HandoffNoteConfig) -> Self {
        self.handoff_notes = Some(config);
        self
    }

    /// The signals could not decide, so this turn belongs to whatever the cascade
    /// has behind this classifier.
    fn abstain(state: &mut State) -> Classification {
        record_decision_source(state, DecisionSource::Ambiguous);
        Classification::Ambiguous(Vec::new())
    }

    fn apply_handoff_note(&self, request: &mut Request, tier: Tier, source: DecisionSource) {
        let Some(config) = &self.handoff_notes else {
            return;
        };
        if let Some(note) = config.note_for(tier, source) {
            prompts::append_note(request, note);
        }
    }
}

#[async_trait]
impl Classifier<State> for StageClassifier {
    fn routing_tier(&self, selected_model: &str) -> Option<&'static str> {
        self.targets.label_for(selected_model)
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<(Classification, Option<switchyard_protocol::Response>)> {
        let tool_signals = &state.tool_signals;
        let Some(signal) = tool_signals else {
            // No tool activity yet — nothing to score, so the signals have no
            // opinion, same as a below-threshold turn.
            return Ok((Self::abstain(state), None));
        };

        let outcome = pick_tier(signal, self.mode, self.confidence_threshold);
        match outcome {
            PickOutcome::Resolved {
                tier,
                source,
                score,
                ..
            } => {
                let target = self.targets.name(tier);
                record_decision_source(state, source);
                // Only a resolved turn routes on this classifier's target, so it
                // is the only branch whose tier the signals actually chose — an
                // ambiguous turn is decided further down the cascade.
                self.apply_handoff_note(request, tier, source);
                let conf = score.abs();
                Ok((
                    Classification::Scores(vec![Score {
                        target: target.to_string(),
                        confidence: conf,
                    }]),
                    None,
                ))
            }
            PickOutcome::ConsultClassifier { .. } => Ok((Self::abstain(state), None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_protocol::{text_request, Metadata, Request, WireFormat};

    fn signal_from(messages: serde_json::Value) -> ToolSignals {
        let raw_request = Some(json!({"model": "m", "messages": messages}));
        let request = Request {
            raw_request,
            metadata: Some(Metadata {
                wire_format: Some(WireFormat::OpenAiChat),
                ..Default::default()
            }),
            ..Default::default()
        };
        ToolSignals::from_request(&request, None)
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
            PickOutcome::Resolved {
                tier: Tier::Capable,
                source: DecisionSource::Override,
                ..
            }
        ));
    }

    #[test]
    fn one_signal_scores_below_half() {
        // A single full wrong signal ≈ 0.46 confidence — just under 0.5.
        let mut signal = signal_from(json!([{"role": "user", "content": "hi"}]));
        signal.severity = HARD_SEVERITY as f32;
        let scored = score_signal(&signal);
        assert!(scored.score > 0.0);
        assert!(
            scored.confidence < 0.5,
            "one signal should not clear 0.5: {scored:?}"
        );
    }

    #[test]
    fn the_picker_mode_names_the_tier_an_undecided_turn_falls_back_to() {
        assert_eq!(PickerMode::CapableFirst.default_tier(), Tier::Capable);
        assert_eq!(PickerMode::EfficientFirst.default_tier(), Tier::Efficient);
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

    // ─── StageClassifier ─────────────────────────────────────────────────

    /// Tiers named the way a deployment would name them.
    fn tiers() -> StageTargets {
        StageTargets::new("strong", "weak")
    }

    /// A `State` carrying `signal` as its tool signals.
    fn state_with(signal: ToolSignals) -> State {
        State {
            tool_signals: Some(signal),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn classifier_defaults_without_tool_signals() -> Result<()> {
        // No tool activity yet — nothing to score, so the signals have no opinion
        // and the turn belongs to whatever the cascade has behind them.
        let mut state = State::default();
        let classification = StageClassifier::new(tiers(), PickerMode::EfficientFirst, 0.5)
            .score(&mut state, &mut Request::default(), None)
            .await?;
        assert!(classification.0.argmax(false)?.is_none());
        assert!(matches!(
            state.extra.get(DECISION_SOURCE_KEY),
            Some(StateValue::String(source)) if source == "ambiguous"
        ));
        Ok(())
    }

    #[tokio::test]
    async fn classifier_escalates_critical_severity_to_strong() -> Result<()> {
        // Critical severity is a hard override → a definite score for the capable tier.
        let signal = ToolSignals {
            severity: SEVERITY_CRITICAL,
            ..Default::default()
        };
        let mut state = state_with(signal);
        let classification = StageClassifier::new(tiers(), PickerMode::EfficientFirst, 0.5)
            .score(&mut state, &mut Request::default(), None)
            .await?;
        match classification.0 {
            Classification::Scores(scores) => {
                assert_eq!(scores.len(), 1);
                assert_eq!(scores[0].target, "strong");
            }
            _ => panic!("expected a definite classification"),
        }
        // The decision source travels downstream (for handoff-note gating).
        assert!(matches!(
            state.extra.get(DECISION_SOURCE_KEY),
            Some(StateValue::String(source)) if source == "override"
        ));
        Ok(())
    }

    #[tokio::test]
    async fn classifier_deescalates_settled_turn_to_weak() -> Result<()> {
        // Tests passed with recent production and no error → the settled-turn shortcut
        // resolves straight to a definite efficient-tier score.
        let signal = ToolSignals {
            tests_passed: true,
            recent_write_count: 1,
            severity: 0.0,
            ..Default::default()
        };
        let mut state = state_with(signal);
        let classification = StageClassifier::new(tiers(), PickerMode::EfficientFirst, 0.5)
            .score(&mut state, &mut Request::default(), None)
            .await?;
        match classification.0 {
            Classification::Scores(scores) => {
                assert_eq!(scores.len(), 1);
                assert_eq!(scores[0].target, "weak");
            }
            _ => panic!("expected a definite classification"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn classifier_falls_open_to_default_and_records_it() -> Result<()> {
        // A quiet signal corroborates neither axis, so the scorer abstains and
        // records why.
        let mut state = state_with(ToolSignals::default());
        let classification = StageClassifier::new(tiers(), PickerMode::EfficientFirst, 0.5)
            .score(&mut state, &mut Request::default(), None)
            .await?;
        assert!(classification.0.argmax(false)?.is_none());
        assert!(matches!(
            state.extra.get(DECISION_SOURCE_KEY),
            Some(StateValue::String(source)) if source == "ambiguous"
        ));
        Ok(())
    }

    const ESCALATION: &str = "recovering from an error";
    const DEESCALATION: &str = "settled — carry on";

    fn config(only_on_wrong_signal_escalation: bool) -> HandoffNoteConfig {
        HandoffNoteConfig::new(
            ESCALATION,
            Some(DEESCALATION.to_string()),
            only_on_wrong_signal_escalation,
        )
    }

    #[test]
    fn escalation_note_applies_to_signal_driven_capable() {
        for source in [DecisionSource::Override, DecisionSource::Dimensions] {
            assert_eq!(
                config(true).note_for(Tier::Capable, source),
                Some(ESCALATION)
            );
        }
    }

    #[test]
    fn no_escalation_note_on_an_ambiguous_turn_when_gated() {
        assert_eq!(
            config(true).note_for(Tier::Capable, DecisionSource::Ambiguous),
            None
        );
    }

    #[test]
    fn escalation_note_on_an_ambiguous_turn_when_not_gated() {
        assert_eq!(
            config(false).note_for(Tier::Capable, DecisionSource::Ambiguous),
            Some(ESCALATION)
        );
    }

    #[test]
    fn deescalation_note_applies_to_efficient_when_configured() {
        assert_eq!(
            config(true).note_for(Tier::Efficient, DecisionSource::TestsPassed),
            Some(DEESCALATION)
        );
    }

    #[test]
    fn no_deescalation_note_when_unconfigured() {
        let config = HandoffNoteConfig::new(ESCALATION, None, true);
        assert_eq!(
            config.note_for(Tier::Efficient, DecisionSource::TestsPassed),
            None
        );
    }

    // ─── handoff notes ───────────────────────────────────────────────────

    /// A classifier that hands the capable tier an escalation note, gated to
    /// signal-driven escalations.
    fn noting_classifier(mode: PickerMode) -> StageClassifier {
        StageClassifier::new(tiers(), mode, 0.5).with_handoff_notes(HandoffNoteConfig::new(
            ESCALATION,
            Some(DEESCALATION.to_string()),
            true,
        ))
    }

    /// A one-user-turn request, the thing a note gets spliced into.
    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        }
    }

    /// The trailing user turn's text, note included.
    fn trailing_text(request: &Request) -> Option<String> {
        request
            .llm_request
            .messages
            .last()
            .and_then(|message| message.text_content("|"))
    }

    /// The signal that forces an escalation on the override path.
    fn critical() -> ToolSignals {
        ToolSignals {
            severity: SEVERITY_CRITICAL,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn a_signal_driven_escalation_carries_the_note() -> Result<()> {
        let mut state = state_with(critical());
        let mut request = request();

        noting_classifier(PickerMode::EfficientFirst)
            .score(&mut state, &mut request, None)
            .await?;

        assert_eq!(trailing_text(&request), Some(format!("hi|{ESCALATION}")));
        Ok(())
    }

    #[tokio::test]
    async fn every_turn_the_signals_drive_carries_the_note() -> Result<()> {
        // Stateless by design: the note describes this turn's signals, so a run
        // of escalated turns each carries one. Nothing tracks the previous tier.
        let classifier = noting_classifier(PickerMode::EfficientFirst);
        let mut state = state_with(critical());

        for _ in 0..3 {
            let mut request = request();
            classifier.score(&mut state, &mut request, None).await?;
            assert_eq!(trailing_text(&request), Some(format!("hi|{ESCALATION}")));
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_settled_turn_carries_the_deescalation_note() -> Result<()> {
        // Tests passed with recent production resolves to weak on the settled-turn
        // shortcut, which is the hand-back the de-escalation note is for.
        let signal = ToolSignals {
            tests_passed: true,
            recent_write_count: 1,
            ..Default::default()
        };
        let mut state = state_with(signal);
        let mut request = request();

        noting_classifier(PickerMode::EfficientFirst)
            .score(&mut state, &mut request, None)
            .await?;

        assert_eq!(trailing_text(&request), Some(format!("hi|{DEESCALATION}")));
        Ok(())
    }

    #[tokio::test]
    async fn no_note_on_an_ambiguous_turn() -> Result<()> {
        // A quiet signal falls open: the cascade, not these signals, picks the
        // tier, so there is no signal-driven handover to narrate.
        let mut state = state_with(ToolSignals::default());
        let mut request = request();

        let classification = noting_classifier(PickerMode::CapableFirst)
            .score(&mut state, &mut request, None)
            .await?;

        assert!(matches!(classification.0, Classification::Ambiguous(_)));
        assert_eq!(trailing_text(&request), Some("hi".to_string()));
        Ok(())
    }

    #[tokio::test]
    async fn no_note_when_notes_are_unconfigured() -> Result<()> {
        let mut state = state_with(critical());
        let mut request = request();

        StageClassifier::new(tiers(), PickerMode::EfficientFirst, 0.5)
            .score(&mut state, &mut request, None)
            .await?;

        assert_eq!(trailing_text(&request), Some("hi".to_string()));
        Ok(())
    }
}
