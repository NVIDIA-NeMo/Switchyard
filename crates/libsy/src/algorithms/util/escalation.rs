// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trajectory-judge components for the escalation router — the judge, its verdict policy, and
//! the transcript condenser they read.
//!
//! [`build_judge`] is the whole surface; the confirmation policy that consumes its verdicts
//! lives with the assembled algorithm in [`crate::algorithms::escalation`].

use serde::Deserialize;
use serde_json::Value;
use switchyard_protocol::{ContentBlock, Message, ModelId, Role};

use super::classifier_contract::{ClassifierContract, ClassifierContractConfig, validate_prompt};
use super::llm_judge::{
    ClassifierInput, JudgeClassifier, JudgePolicy, JudgeRuntimeConfig, SerdeDecoder,
    StructuredJudge,
};
use crate::core::classifier::{Classification, Score};
use crate::core::state::State;
use crate::{LibsyError, Result};
use switchyard_protocol::Request;

const PROMPT_TEMPLATE: &str = include_str!("../../prompts/escalation/prompt.md");
const DEESCALATION_PROMPT: &str = include_str!("../../prompts/escalation/deescalation.md");
const SCHEMA_TEMPLATE: &str = include_str!("../../prompts/escalation/schema.json");

/// Separator marking where [`truncate_middle`] dropped a message's interior.
const TRIM_MARKER: &str = " ...[trimmed] ";

/// Suffix marking a transcript cut off by [`MAX_REQUEST_CHARS`].
const TRUNCATION_SUFFIX: &str = "...<truncated>";

/// Per-message cap for system and developer anchors, which carry no trajectory signal but
/// which coding-agent harnesses make very large.
const SYSTEM_CHARS: usize = 1_000;

/// Cap for the first user message — the task statement, so it gets the widest anchor budget.
const FIRST_USER_CHARS: usize = 2_000;

/// Backstop on the assembled transcript; the per-message caps normally bind first.
const MAX_REQUEST_CHARS: usize = 18_000;

/// Optional policy for returning an escalated session to the efficient tier.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeescalationConfig {
    /// Minimum number of capable-tier turns before the judge may release the session.
    pub strong_min_calls: u32,
    /// Optional hard limit on capable-tier turns before a forced return.
    #[serde(default)]
    pub strong_max_calls: Option<u32>,
    /// Consecutive judge declines required to return to the efficient tier.
    pub confirmations: u32,
    /// Efficient calls served without judging after a hard-limit return.
    #[serde(default)]
    pub weak_cooldown_calls: u32,
}

impl DeescalationConfig {
    fn validate(&self) -> Result<()> {
        let reject = |message: &str| {
            Err(LibsyError::AlgorithmError {
                message: message.to_string(),
            })
        };
        if self.strong_min_calls == 0 {
            return reject("deescalation.strong_min_calls must be at least 1");
        }
        if self.confirmations == 0 {
            return reject("deescalation.confirmations must be at least 1");
        }
        if self
            .strong_max_calls
            .is_some_and(|strong_max_calls| strong_max_calls < self.strong_min_calls)
        {
            return reject(
                "deescalation.strong_max_calls must be at least deescalation.strong_min_calls",
            );
        }
        Ok(())
    }
}

/// The tuning surface for the trajectory judge.
///
/// The routing settings retain their benchmarked defaults. Everything else is a fixed invariant
/// (the constants above).
#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EscalationJudgeConfig {
    /// Consecutive escalate verdicts required before a turn moves to the capable tier, which
    /// is also the turn that latches the session. Any decline clears the streak.
    /// `1` escalates on the first verdict; the router's main cost dial.
    /// `2` or higher needs a session id, since the streak is retained per session.
    pub confirmations: u32,
    /// Trailing messages shown on top of the anchors. A loop longer than this is invisible.
    pub recent_turn_window: usize,
    /// Per-message cap inside the trailing window.
    pub window_message_chars: usize,
    /// De-escalation policy. `None` preserves permanent latching to the capable tier.
    pub deescalation: Option<DeescalationConfig>,
}

impl EscalationJudgeConfig {
    /// Rejects settings that would leave the judge with nothing useful to read.
    fn validate(&self) -> Result<()> {
        let reject = |message: String| Err(LibsyError::AlgorithmError { message });
        if self.confirmations == 0 {
            return reject("confirmations must be at least 1".to_string());
        }
        if self.recent_turn_window == 0 {
            return reject("recent_turn_window must be at least 1".to_string());
        }
        if self.window_message_chars < 50 {
            return reject(format!(
                "window_message_chars must be at least 50, got {}",
                self.window_message_chars
            ));
        }
        if let Some(deescalation) = self.deescalation {
            deescalation.validate()?;
        }
        Ok(())
    }
}

impl Default for EscalationJudgeConfig {
    fn default() -> Self {
        Self {
            confirmations: 2,
            recent_turn_window: 28,
            window_message_chars: 500,
            deescalation: None,
        }
    }
}

/// Router-controlled phase attached to judge input when de-escalation is enabled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EvaluationPhase {
    Efficient,
    Strong,
}

impl EvaluationPhase {
    fn marker(self) -> &'static str {
        match self {
            Self::Efficient => "EFFICIENT_EVALUATION",
            Self::Strong => "STRONG_EVALUATION",
        }
    }
}

/// The judge's verdict. The schema also requires a `reason`, which makes the judge state its
/// case and measurably sharpens the verdict — routing reads only the boolean, so it is
/// deserialized away rather than carried.
#[derive(Deserialize)]
pub(crate) struct EscalationVerdict {
    escalate: bool,
}

/// Builds the condensed trajectory presented to the escalation judge.
pub(crate) struct EscalationInput {
    config: EscalationJudgeConfig,
    phase: Option<EvaluationPhase>,
}

impl ClassifierInput for EscalationInput {
    fn build_messages(&self, _state: &State, request: &Request) -> Vec<Message> {
        let messages = &request.llm_request.messages;
        let summary = summarize_for_judge(
            messages,
            conversation_turn(request),
            self.phase,
            &self.config,
        );
        vec![Message::text(Role::User, summary)]
    }
}

/// Structured trajectory judge with a typed escalation verdict.
pub(crate) type EscalationJudge = StructuredJudge<EscalationInput, SerdeDecoder<EscalationVerdict>>;

/// Maps the judge's verdict to a classification. A verdict names the tier to serve — capable
/// on escalate, efficient on decline — so the caller reads it straight off the winning score.
/// [`Classification::Ambiguous`] carries the unavailable case, which names no tier: both a
/// decline and an outage stay efficient, but only a decline is evidence, so only a decline
/// clears the streak.
pub(crate) struct EscalationPolicy {
    capable: ModelId,
    efficient: ModelId,
    phase: Option<EvaluationPhase>,
}

impl JudgePolicy for EscalationPolicy {
    type Verdict = EscalationVerdict;

    fn to_classification(&self, verdict: Option<&EscalationVerdict>) -> Classification {
        match verdict {
            Some(verdict) if verdict.escalate => Classification::Scores(vec![Score {
                target: self.capable.clone(),
                confidence: 1.0,
            }]),
            Some(_) => Classification::Scores(vec![Score {
                target: self.efficient.clone(),
                confidence: 1.0,
            }]),
            None => Classification::Ambiguous(Vec::new()),
        }
    }
}

/// Maps present verdicts to phase-aware evidence; absent verdicts add nothing.
fn escalation_evidence(
    policy: &EscalationPolicy,
    verdict: Option<&EscalationVerdict>,
) -> Option<Value> {
    verdict.map(|verdict| {
        let verdict = match (policy.phase, verdict.escalate) {
            (Some(EvaluationPhase::Strong), true) => "retain",
            (Some(EvaluationPhase::Strong), false) => "deescalate",
            (_, true) => "escalate",
            (_, false) => "continue",
        };
        serde_json::json!({
            "source": "escalation",
            "verdict": verdict,
        })
    })
}

/// Builds the trajectory judge over `judge_target`, scoring `capable` when it escalates.
///
/// Loads the packaged prompt and schema, so an unusable asset or an unusable `config` value
/// fails here rather than on the first request.
pub(crate) fn build_judge(
    judge_target: ModelId,
    capable: ModelId,
    efficient: ModelId,
    contract_config: &ClassifierContractConfig,
    config: EscalationJudgeConfig,
    phase: Option<EvaluationPhase>,
    max_output_tokens: u64,
) -> Result<JudgeClassifier<EscalationJudge, EscalationPolicy>> {
    config.validate()?;
    let contract = build_contract(contract_config, phase.is_some())?;
    Ok(JudgeClassifier::new(
        StructuredJudge::new(
            EscalationInput { config, phase },
            contract,
            SerdeDecoder::new(),
            JudgeRuntimeConfig::new(max_output_tokens)?,
        ),
        judge_target,
        EscalationPolicy {
            capable,
            efficient,
            phase,
        },
    )
    .with_evidence(escalation_evidence))
}

fn build_contract(
    contract_config: &ClassifierContractConfig,
    phase_aware: bool,
) -> Result<ClassifierContract> {
    let prompt = contract_config.prompt().unwrap_or(PROMPT_TEMPLATE);
    validate_prompt(prompt)?;
    let phase_aware_config = phase_aware.then(|| {
        contract_config.clone().with_prompt(format!(
            "{}\n\n{}",
            prompt.trim_end(),
            DEESCALATION_PROMPT.trim()
        ))
    });
    let contract_config = phase_aware_config.as_ref().unwrap_or(contract_config);
    ClassifierContract::from_config(contract_config, PROMPT_TEMPLATE, SCHEMA_TEMPLATE)
}

/// The 1-indexed model invocation the transcript ends on: one per assistant reply.
///
/// The judge reads the turn *including* the reply it is judging, so the newest assistant
/// message is this turn — no `+ 1`. Counting the caller's request instead would report the
/// turn after the one under judgement.
///
/// Messages are already normalized by `switchyard-protocol`, so this needs no
/// per-format branching.
pub(crate) fn conversation_turn(request: &Request) -> usize {
    request
        .llm_request
        .messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .count()
}

/// Flattens a message to plain text, tool calls and tool results included.
///
/// [`Message::text_content`] is deliberately not used here: it keeps only text and refusal
/// blocks, which would erase exactly the repeated-command signal the judge's loop detection
/// relies on.
fn message_text(message: &Message) -> String {
    let mut parts = Vec::new();
    collect_text(&message.content, &mut parts);
    parts.join(" ")
}

/// Appends the judge-relevant text of each block, descending into tool results.
fn collect_text(content: &[ContentBlock], parts: &mut Vec<String>) {
    for block in content {
        match block {
            ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
                parts.push(text.clone());
            }
            ContentBlock::ToolCall(call) => {
                parts.push(format!("tool_call {}({})", call.name, call.arguments));
            }
            ContentBlock::ToolResult(result) => collect_text(&result.content, parts),
            _ => {}
        }
    }
}

/// Keeps the head and tail of `text` within `limit` characters.
///
/// The head gets two thirds of the surviving budget: for a trajectory judge the command or
/// error signature that opens a message carries more signal than its trailing output.
fn truncate_middle(text: &str, limit: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        return text.to_string();
    }
    let keep = limit
        .saturating_sub(TRIM_MARKER.chars().count())
        .max(20)
        .min(chars.len());
    let head = keep * 2 / 3;
    let tail = keep - head;
    let mut out: String = chars[..head].iter().collect();
    out.push_str(TRIM_MARKER);
    out.extend(chars[chars.len() - tail..].iter());
    out
}

/// Renders a compact role-labelled transcript for the judge.
///
/// The framing anchors — system/developer messages and the first user message, where agent
/// harnesses put the task statement — are kept unconditionally and capped individually. The
/// trailing window carries recent activity. A coverage header states how much history is not
/// shown, so the judge can reason about pace rather than assuming it sees everything.
///
/// When the assembled text still exceeds `max_request_chars`, the oldest window lines go
/// first: for a trajectory judge the newest evidence is strictly the most valuable.
fn summarize_for_judge(
    messages: &[Message],
    turn: usize,
    phase: Option<EvaluationPhase>,
    config: &EscalationJudgeConfig,
) -> String {
    let mut anchors: Vec<String> = Vec::new();
    let mut window: Vec<String> = Vec::new();
    let mut first_user_seen = false;

    for message in messages {
        let text = message_text(message);
        match message.role {
            Role::System | Role::Developer => anchors.push(format!(
                "[{}] {}",
                role_label(message.role),
                truncate_middle(&text, SYSTEM_CHARS)
            )),
            Role::User if !first_user_seen => {
                first_user_seen = true;
                anchors.push(format!(
                    "[user (task)] {}",
                    truncate_middle(&text, FIRST_USER_CHARS)
                ));
            }
            role => window.push(format!(
                "[{}] {}",
                role_label(role),
                truncate_middle(&text, config.window_message_chars)
            )),
        }
    }

    if window.len() > config.recent_turn_window {
        window.drain(..window.len() - config.recent_turn_window);
    }

    let assemble = |window: &[String]| {
        let header = format!(
            "Conversation turn {turn}; showing the last {} of {} messages after the task framing.",
            window.len(),
            messages.len(),
        );
        phase
            .map(|phase| format!("Routing phase: {}", phase.marker()))
            .into_iter()
            .chain(std::iter::once(header))
            .chain(anchors.iter().cloned())
            .chain(window.iter().cloned())
            .collect::<Vec<_>>()
            .join("\n")
    };

    let mut text = assemble(&window);
    while text.chars().count() > MAX_REQUEST_CHARS && !window.is_empty() {
        window.remove(0);
        text = assemble(&window);
    }
    if text.chars().count() > MAX_REQUEST_CHARS {
        let keep = MAX_REQUEST_CHARS.saturating_sub(TRUNCATION_SUFFIX.chars().count() + 1);
        text = text.chars().take(keep).collect::<String>() + TRUNCATION_SUFFIX;
    }
    text
}

/// The transcript label for a role.
fn role_label(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// A request whose conversation sits at `turn`: `turn - 1` prior assistant replies, each
/// answered by a further user message.
///
/// Shared with the assembled router's tests, which drive the same conversation shape.
#[cfg(test)]
pub(crate) fn request_at_turn(session_id: Option<&str>, turn: usize) -> Request {
    use switchyard_protocol::{LlmRequest, Metadata};

    let mut messages = vec![Message::text(Role::User, "What is 2+2?")];
    for attempt in 1..turn {
        messages.push(Message::text(Role::Assistant, format!("attempt {attempt}")));
        messages.push(Message::text(Role::User, format!("still wrong {attempt}")));
    }
    Request {
        llm_request: LlmRequest {
            model: Some("auto".to_string()),
            messages,
            ..LlmRequest::default()
        },
        raw_request: None,
        metadata: session_id.map(|id| Metadata {
            session_id: Some(id.to_string()),
            ..Metadata::default()
        }),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use switchyard_protocol::{ContentBlock, Message, Role, ToolCall, ToolResult};

    use super::*;
    use crate::algorithms::util::llm_judge::Judge;

    fn escalation_judge(
        max_output_tokens: u64,
        phase: Option<EvaluationPhase>,
        contract_config: &ClassifierContractConfig,
    ) -> Result<EscalationJudge> {
        Ok(StructuredJudge::new(
            EscalationInput {
                config: EscalationJudgeConfig::default(),
                phase,
            },
            build_contract(contract_config, phase.is_some())?,
            SerdeDecoder::new(),
            JudgeRuntimeConfig::new(max_output_tokens)?,
        ))
    }

    #[test]
    fn judge_request_is_rubric_plus_summary_under_a_completion_cap() -> Result<()> {
        let judge = escalation_judge(
            super::super::DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
            None,
            &ClassifierContractConfig::default(),
        )?;

        // As the classifier calls it: the turn's reply is already on the transcript.
        let mut judged = request_at_turn(None, 4);
        judged
            .llm_request
            .messages
            .push(Message::text(Role::Assistant, "this turn's reply"));
        let built = judge.build_request(&State::default(), &judged);

        // Rubric in instructions, condensed trajectory as the sole user message.
        assert_eq!(built.llm_request.instructions.len(), 1);
        assert_eq!(built.llm_request.instructions[0].role, Role::System);
        assert_eq!(
            built.llm_request.instructions[0].content.as_slice(),
            &[ContentBlock::Text {
                text: PROMPT_TEMPLATE.to_string()
            }]
        );
        assert_eq!(built.llm_request.messages.len(), 1);
        assert_eq!(built.llm_request.messages[0].role, Role::User);
        assert!(
            built.llm_request.messages[0]
                .text_content("")
                .is_some_and(|text| text.contains("Conversation turn 4"))
        );
        assert!(
            built.llm_request.messages[0]
                .text_content("")
                .is_none_or(|text| !text.contains("Routing phase:"))
        );
        // Bounded output, so a reasoning judge cannot run away mid-verdict.
        assert_eq!(
            built.llm_request.output.max_output_tokens,
            Some(super::super::DEFAULT_JUDGE_MAX_OUTPUT_TOKENS)
        );
        assert!(built.llm_request.output.response_format.is_some());
        Ok(())
    }

    #[test]
    fn judge_request_uses_the_configured_completion_cap() -> Result<()> {
        let judge = escalation_judge(512, None, &ClassifierContractConfig::default())?;

        let built = judge.build_request(&State::default(), &request_at_turn(None, 1));

        assert_eq!(built.llm_request.output.max_output_tokens, Some(512));
        Ok(())
    }

    #[test]
    fn conversation_turn_counts_assistant_replies() {
        // The judge is handed the transcript with this turn's reply already appended, which
        // is the shape asserted here: a request entering turn N plus its reply *is* turn N.
        for turn in [1, 5] {
            let mut judged = request_at_turn(None, turn);
            judged
                .llm_request
                .messages
                .push(Message::text(Role::Assistant, "this turn's reply"));
            assert_eq!(conversation_turn(&judged), turn);
        }
    }

    #[test]
    fn message_text_keeps_tool_calls_and_results() {
        let call = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "running it".to_string(),
                },
                ContentBlock::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    name: "bash".to_string(),
                    arguments: json!({"cmd": "ls"}),
                }),
            ],
        };
        let text = message_text(&call);
        assert!(text.contains("running it"), "{text}");
        assert!(text.contains(r#"tool_call bash({"cmd":"ls"})"#), "{text}");

        let result = Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "call-1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "no such file".to_string(),
                }],
                is_error: Some(true),
            })],
        };
        assert_eq!(message_text(&result), "no such file");
    }

    #[test]
    fn truncate_middle_keeps_head_and_tail() {
        let text = "a".repeat(40) + &"z".repeat(40);
        let trimmed = truncate_middle(&text, 50);
        assert!(trimmed.chars().count() <= 50, "{trimmed}");
        assert!(trimmed.starts_with('a'));
        assert!(trimmed.ends_with('z'));
        assert!(trimmed.contains("[trimmed]"));

        // Under the limit the text is returned untouched.
        assert_eq!(truncate_middle("short", 50), "short");
    }

    #[test]
    fn deescalation_settings_must_be_valid() {
        let zero = EscalationJudgeConfig {
            deescalation: Some(DeescalationConfig {
                strong_min_calls: 0,
                strong_max_calls: None,
                confirmations: 2,
                weak_cooldown_calls: 0,
            }),
            ..EscalationJudgeConfig::default()
        };
        assert!(
            zero.validate()
                .is_err_and(|error| error.to_string().contains("at least 1"))
        );

        let inverted = EscalationJudgeConfig {
            deescalation: Some(DeescalationConfig {
                strong_min_calls: 4,
                strong_max_calls: Some(3),
                confirmations: 2,
                weak_cooldown_calls: 0,
            }),
            ..EscalationJudgeConfig::default()
        };
        assert!(
            inverted
                .validate()
                .is_err_and(|error| error.to_string().contains("at least deescalation"))
        );
    }

    #[test]
    fn deescalation_contract_marks_both_routing_phases() -> Result<()> {
        let contract = ClassifierContractConfig::default().with_prompt("Custom trajectory rubric.");
        for (phase, marker) in [
            (EvaluationPhase::Efficient, "EFFICIENT_EVALUATION"),
            (EvaluationPhase::Strong, "STRONG_EVALUATION"),
        ] {
            let judge = escalation_judge(512, Some(phase), &contract)?;
            let built = judge.build_request(&State::default(), &request_at_turn(None, 1));
            let system_prompt = built.llm_request.instructions[0].content.iter().find_map(
                |content| match content {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                },
            );
            assert!(system_prompt.is_some_and(|prompt| {
                prompt.starts_with("Custom trajectory rubric.")
                    && prompt.contains("EFFICIENT_EVALUATION")
                    && prompt.contains("STRONG_EVALUATION")
            }));
            assert!(
                built.llm_request.messages[0]
                    .text_content("")
                    .is_some_and(|summary| summary.starts_with(&format!(
                        "Routing phase: {marker}"
                    )))
            );
        }
        Ok(())
    }

    #[test]
    fn summary_keeps_anchors_and_the_recent_window() {
        let mut messages = vec![
            Message::text(Role::System, "you are a coding agent"),
            Message::text(Role::User, "fix the failing test"),
        ];
        for i in 0..10 {
            messages.push(Message::text(Role::Assistant, format!("step {i}")));
        }
        let config = EscalationJudgeConfig {
            recent_turn_window: 3,
            ..EscalationJudgeConfig::default()
        };

        let summary = summarize_for_judge(&messages, 11, None, &config);

        assert!(
            summary.contains("[system] you are a coding agent"),
            "{summary}"
        );
        assert!(
            summary.contains("[user (task)] fix the failing test"),
            "{summary}"
        );
        assert!(summary.contains("Conversation turn 11; showing the last 3 of 12 messages"));
        // Only the newest window entries survive.
        assert!(summary.contains("step 9"), "{summary}");
        assert!(summary.contains("step 7"), "{summary}");
        assert!(!summary.contains("step 6"), "{summary}");
    }

    #[test]
    fn summary_drops_oldest_window_lines_under_the_char_cap() {
        // MAX_REQUEST_CHARS is a backstop, not a dial: at default settings the window caps
        // bind first (28 x 500 plus anchors sits under it), so reaching it takes an unusually
        // wide per-message cap. That is the point — it only fires on pathological input.
        let mut messages = vec![
            Message::text(Role::System, "framing"),
            Message::text(Role::User, "task"),
        ];
        for i in 0..20 {
            messages.push(Message::text(
                Role::Assistant,
                format!("{i} {}", "x".repeat(2_000)),
            ));
        }
        let config = EscalationJudgeConfig {
            window_message_chars: 2_000,
            ..EscalationJudgeConfig::default()
        };

        let summary = summarize_for_judge(&messages, 21, None, &config);

        assert!(
            summary.chars().count() <= MAX_REQUEST_CHARS,
            "{}",
            summary.chars().count()
        );
        // Anchors are never dropped, and the newest activity outlives the oldest.
        assert!(summary.contains("[system] framing"), "{summary}");
        assert!(summary.contains("[user (task)] task"), "{summary}");
        assert!(summary.contains("19 xxx"), "{summary}");
        assert!(!summary.contains("0 xxx"), "{summary}");
    }
}
