// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trajectory-judged escalation: serve the efficient model until an LLM judge finds the run in
//! trouble, then latch the session to the capable model for the rest of the task — re-trying a
//! model that has already failed the task wastes a turn.
//!
//! Each turn is served on the efficient tier first, and the judge reads the conversation
//! *including that reply*. A run that is fine keeps the reply it already has, so only a confirmed
//! escalation pays for a second serving call. A judge failure fails open to the efficient tier
//! and never latches: an outage costs quality risk, never money.
//!
//! Assembled as a [`FallThrough`]: the latch and the judge are its classifiers, and a processor
//! commits the confirmation streak once the turn's decision is final.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmRequest, LlmResponse, Message, OutputParams, Role,
};

use crate::{
    algorithms::util::{load_judge_config, Judge, JudgeClassifier, JudgeConfig, JudgePolicy},
    algorithms::FallThrough,
    Algorithm, Classification, Classifier, Context, Decision, Driver, Event, LibsyError, LlmTarget,
    LlmTargetSet, Processor, Request, Response, Result, RoutedLlmClient, Score, Scored, State,
    StateValue,
};

const PROMPT_TEMPLATE: &str = include_str!("../prompts/escalation/prompt.md");
const SCHEMA_TEMPLATE: &str = include_str!("../prompts/escalation/schema.json");

/// Telemetry label for this algorithm's spans, metrics, and logs.
const ALGORITHM_NAME: &str = "escalation";

/// Session-state key holding the consecutive-escalate streak, written by
/// [`ConfirmationProcessor`] and read by [`EscalationClassifier`].
const STREAK_KEY: &str = "escalation_streak";

/// Session-state key holding the streak this turn's verdict would leave behind, while it
/// travels from the classifier to [`ConfirmationProcessor`].
const PENDING_KEY: &str = "escalation_pending_streak";

/// Separator marking where [`truncate_middle`] dropped a message's interior.
const TRIM_MARKER: &str = " ...[trimmed] ";

/// Suffix marking a transcript cut off by [`MAX_REQUEST_CHARS`].
const TRUNCATION_SUFFIX: &str = "...<truncated>";

/// Per-message cap for system and developer anchors. Coding-agent harnesses inject very large
/// boilerplate system prompts carrying no trajectory signal; uncapped they crowd out the window.
const SYSTEM_CHARS: usize = 1_000;

/// Cap for the first user message — the task statement the judge needs to detect drift, so it
/// gets the most generous anchor budget.
const FIRST_USER_CHARS: usize = 2_000;

/// Backstop on the whole assembled transcript, for a pathological single message. The window
/// caps below normally bind first; when this does bind, the oldest window lines drop.
const MAX_REQUEST_CHARS: usize = 18_000;

/// Completion budget for one judge reply, covering any reasoning emitted alongside the verdict.
///
/// A runaway guard, not a budget: a tight cap truncates mid-reasoning into unparseable JSON and
/// fails the judge open on every call, while a generous one costs only what it generates. The
/// benchmarked judge's `reason` ran ~40 tokens at the median and ~136 at its longest.
const JUDGE_MAX_OUTPUT_TOKENS: u64 = 4_096;

/// The tuning surface for the trajectory judge.
///
/// Deliberately small: an audit of the Python router's twenty-odd knobs found only these three
/// change routing outcomes; the rest were fixed invariants (the `*_CHARS` constants above) or
/// operational plumbing. Defaults are the benchmarked configuration.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct EscalationJudgeSettings {
    /// Consecutive escalate verdicts required before the latch fires.
    ///
    /// `1` latches on the first verdict; higher values filter one-shot eager verdicts — a single
    /// failed command misread as a pattern — while keeping recall on trouble that persists. On
    /// the benchmarked workload `2` suppressed roughly two thirds of escalate verdicts, making
    /// this the main cost dial. The streak is keyed on the session id, so a request without one
    /// can never escalate at `2` or higher.
    pub confirmations: u32,
    /// Trailing messages shown on top of the anchors. Loop detection needs to see the
    /// repeats, so a cycle longer than this window is invisible to the judge.
    pub recent_turn_window: usize,
    /// Per-message cap inside the trailing window. Error signatures and command shapes
    /// survive this easily; full file dumps do not need to.
    pub window_message_chars: usize,
}

impl EscalationJudgeSettings {
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
        Ok(())
    }
}

impl Default for EscalationJudgeSettings {
    fn default() -> Self {
        Self {
            confirmations: 2,
            recent_turn_window: 28,
            window_message_chars: 500,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EscalationVerdict {
    escalate: bool,
    #[allow(dead_code)]
    reason: String,
}

/// The efficient tier's serving call, published so its span and metrics carry the tier.
struct EfficientCall {
    model: String,
}

impl Decision for EfficientCall {
    fn selected_model(&self) -> &str {
        &self.model
    }

    fn routing_tier(&self) -> Option<&str> {
        Some("weak")
    }

    fn reasoning(&self) -> Option<&str> {
        Some("efficient tier serves the turn the judge reads")
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Builds the judge request from the conversation so far.
struct EscalationJudge {
    config: JudgeConfig,
    settings: EscalationJudgeSettings,
}

impl Judge for EscalationJudge {
    type Verdict = EscalationVerdict;

    /// Renders the rubric as the system message and the condensed trajectory as the user
    /// message. `state` is unused: the judge reads only the live request.
    fn build_request(&self, _state: &State, request: &Request) -> Request {
        let messages = &request.llm_request.messages;
        let summary = summarize_for_judge(messages, conversation_turn(request), &self.settings);
        Request {
            llm_request: LlmRequest {
                model: request.llm_request.model.clone(),
                messages: vec![
                    Message::text(Role::System, self.config.system_prompt.clone()),
                    Message::text(Role::User, summary),
                ],
                output: OutputParams {
                    response_format: self.config.response_schema.clone(),
                    max_output_tokens: Some(JUDGE_MAX_OUTPUT_TOKENS),
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }
}

/// Carries the judge's three outcomes as a [`Classification`]. `Ambiguous` marks "judge
/// unavailable": it argmaxes to nothing exactly like a decline's empty score set, but the
/// classifier can still tell the two apart.
struct EscalationPolicy {
    capable: String,
}

impl JudgePolicy for EscalationPolicy {
    type Verdict = EscalationVerdict;

    fn to_classification(&self, verdict: Option<&EscalationVerdict>) -> Classification {
        match verdict {
            Some(verdict) if verdict.escalate => Classification::Scores(vec![Score {
                target: self.capable.clone(),
                confidence: 1.0,
            }]),
            Some(_) => Classification::Scores(Vec::new()),
            None => Classification::Ambiguous(Vec::new()),
        }
    }
}

/// Serves the turn on the efficient tier, then judges the result.
struct EscalationClassifier {
    efficient_target: LlmTarget,
    efficient: String,
    capable: String,
    judge: JudgeClassifier<EscalationJudge, EscalationPolicy>,
    /// Consecutive escalate verdicts required to latch.
    confirmations: u32,
}

#[async_trait]
impl Classifier<State> for EscalationClassifier {
    fn routing_tier(&self, selected_model: &str) -> Option<&'static str> {
        if self.efficient == self.capable {
            None
        } else if selected_model == self.capable {
            Some("strong")
        } else if selected_model == self.efficient {
            Some("weak")
        } else {
            None
        }
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<Scored> {
        // The streak is the latch. It only reaches the threshold on the turn the judge
        // escalated, and a latched turn never consults the judge again to clear it, so an
        // escalated session stays escalated without spending a call to find out.
        if streak(state) >= self.confirmations {
            return Ok((decisive(&self.capable), None));
        }

        // A broken composition, not a routing condition: this cannot decide without a model.
        let Some(driver) = driver else {
            return Err(LibsyError::AlgorithmError {
                message: "escalation classifier requires a driver to call its targets".to_string(),
            });
        };

        // The judge reads this reply, so the call comes before the verdict — and on the common
        // path the reply is also the turn's answer.
        let served = driver
            .call_llm_target(
                Context::default(),
                &self.efficient_target,
                request.clone(),
                Arc::new(EfficientCall {
                    model: self.efficient.clone(),
                }),
            )
            .await?;
        // Judging the reply means buffering it: a stream cannot be read twice.
        let Response {
            llm_response,
            metadata,
        } = served;
        let aggregate = llm_response.into_agg().await.map_err(|error| {
            LibsyError::external("aggregating the efficient tier response", error)
        })?;

        // Append this turn's reply, so the newest evidence is the work being judged.
        let mut judged = request.clone();
        judged
            .llm_request
            .messages
            .push(assistant_message(&aggregate));
        let (classification, _) = self.judge.score(state, &mut judged, Some(driver)).await?;

        // The streak this turn would leave behind, committed by the processor once the decision
        // is final. Strict-consecutive: escalate extends it, any decline clears it, and an
        // unavailable judge is no evidence either way.
        let held = streak(state);
        let (escalate, pending) = match classification {
            Classification::Scores(ref scores) if !scores.is_empty() => (true, held + 1),
            Classification::Scores(_) => (false, 0),
            Classification::Ambiguous(_) => (false, held),
        };
        state
            .extra
            .insert(PENDING_KEY.to_string(), StateValue::Count(pending));

        if escalate && pending >= self.confirmations {
            // The efficient reply is discarded: the capable model answers this turn.
            return Ok((decisive(&self.capable), None));
        }
        Ok((
            decisive(&self.efficient),
            Some(Response {
                llm_response: LlmResponse::Agg(aggregate),
                metadata,
            }),
        ))
    }
}

/// Commits the streak the classifier computed for this turn.
///
/// Split from the classifier so the streak only moves once the decision is final: the classifier
/// reads it to gate its verdict, this writes it back afterwards.
struct ConfirmationProcessor;

#[async_trait]
impl Processor<State> for ConfirmationProcessor {
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<()> {
        match event {
            // Drop a stale pending streak before the classifier records this turn's, so a run
            // that failed between classification and decision cannot leave one behind.
            Event::Request(_) => {
                state.extra.remove(PENDING_KEY);
            }
            // A latched turn never reaches the judge, so it records nothing to commit.
            Event::Decision { .. } => {
                if let Some(pending) = state.extra.remove(PENDING_KEY) {
                    state.extra.insert(STREAK_KEY.to_string(), pending);
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// The session's consecutive-escalate streak, zero until one is recorded.
fn streak(state: &State) -> u32 {
    match state.extra.get(STREAK_KEY) {
        Some(StateValue::Count(count)) => *count,
        _ => 0,
    }
}

/// A full-confidence classification for one target.
fn decisive(target: &str) -> Classification {
    Classification::Scores(vec![Score {
        target: target.to_string(),
        confidence: 1.0,
    }])
}

/// The model's reply as a transcript message, so the judge reads the turn it is judging.
fn assistant_message(response: &AggLlmResponse) -> Message {
    Message {
        role: Role::Assistant,
        content: response
            .first_output()
            .map(|output| output.content.clone())
            .unwrap_or_default(),
    }
}

/// Names the tier rather than the path that chose it: the cascade reports only the winning
/// score, and latched-vs-judged is already visible in the call's tier field.
fn decision_reason(_name: &str, winner: &Score) -> String {
    format!("escalation selected {}", winner.target)
}

/// Serves the efficient model until the judge escalates, then pins the session to the
/// capable model.
pub struct EscalationRouter {
    route: FallThrough<State>,
}

impl EscalationRouter {
    /// Routes between `efficient_target` and `capable_target` using `judge_target`, with the
    /// default [`EscalationJudgeSettings`].
    pub fn new(
        efficient_target: LlmTarget,
        capable_target: LlmTarget,
        judge_target: LlmTarget,
    ) -> Result<Self> {
        Self::with_settings(
            efficient_target,
            capable_target,
            judge_target,
            EscalationJudgeSettings::default(),
        )
    }

    /// Routes between the tiers with explicit judge settings.
    pub fn with_settings(
        efficient_target: LlmTarget,
        capable_target: LlmTarget,
        judge_target: LlmTarget,
        settings: EscalationJudgeSettings,
    ) -> Result<Self> {
        settings.validate()?;
        let config = load_judge_config(PROMPT_TEMPLATE, SCHEMA_TEMPLATE)?;
        let efficient = efficient_target.semantic_name.clone();
        let capable = capable_target.semantic_name.clone();
        let confirmations = settings.confirmations;

        let classifier = Arc::new(EscalationClassifier {
            efficient_target: efficient_target.clone(),
            efficient,
            capable: capable.clone(),
            judge: JudgeClassifier::new(
                EscalationJudge { config, settings },
                judge_target,
                EscalationPolicy { capable },
            ),
            confirmations,
        });
        // The capable target leads the set so `count_tokens_client` prefers its client.
        let targets = LlmTargetSet::new(vec![capable_target, efficient_target]);
        let route = FallThrough::<State>::new_with_state(targets)
            .with_name(ALGORITHM_NAME)
            .with_decision_reason(decision_reason)
            .with_processor(Arc::new(ConfirmationProcessor))
            .with_classifier(classifier);
        Ok(Self { route })
    }
}

#[async_trait]
impl Algorithm for EscalationRouter {
    fn name(&self) -> &str {
        ALGORITHM_NAME
    }

    fn count_tokens_client(&self) -> Option<Arc<dyn RoutedLlmClient>> {
        self.route.count_tokens_client()
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        self.route.execute(ctx, driver, request).await
    }
}

/// The 1-indexed model invocation the transcript ends on: one per assistant reply, the newest
/// being the turn under judgement.
fn conversation_turn(request: &Request) -> usize {
    request
        .llm_request
        .messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .count()
}

/// Flattens a message to plain text, tool calls and tool results included.
///
/// Not [`Message::text_content`]: that keeps only text and refusal blocks, erasing exactly the
/// repeated-command signal the judge's loop detection relies on.
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
/// The head gets two thirds: the command or error signature opening a message carries more
/// signal than its trailing output.
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
/// The framing anchors — system/developer messages and the first user message, where harnesses
/// put the task statement — are kept unconditionally and capped individually; the trailing
/// window carries recent activity, under a coverage header so the judge can reason about pace
/// rather than assume it sees everything. Over [`MAX_REQUEST_CHARS`] the oldest window lines go
/// first: the newest evidence is strictly the most valuable.
fn summarize_for_judge(
    messages: &[Message],
    turn: usize,
    settings: &EscalationJudgeSettings,
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
                truncate_middle(&text, settings.window_message_chars)
            )),
        }
    }

    if window.len() > settings.recent_turn_window {
        window.drain(..window.len() - settings.recent_turn_window);
    }

    let header = format!(
        "Conversation turn {turn}; showing the last {} of {} messages after the task framing.",
        window.len(),
        messages.len(),
    );
    let assemble = |window: &[String]| {
        std::iter::once(header.as_str())
            .chain(anchors.iter().map(String::as_str))
            .chain(window.iter().map(String::as_str))
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

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use futures::StreamExt;
    use parking_lot::Mutex;
    use switchyard_protocol::{
        completion_text, text_response, ContentBlock, LlmClientError, LlmRequest, LlmResponseChunk,
        Message, Metadata, Role, ToolCall, ToolResult,
    };

    use serde_json::json;

    use crate::{
        Algorithm, Context, Decision, LlmResponse, LlmTarget, Request, Response, RoutedLlmClient,
    };

    use super::{
        conversation_turn, load_judge_config, message_text, summarize_for_judge, truncate_middle,
        EscalationJudge, EscalationJudgeSettings, EscalationRouter, Judge, State,
        JUDGE_MAX_OUTPUT_TOKENS, MAX_REQUEST_CHARS, PROMPT_TEMPLATE, SCHEMA_TEMPLATE,
    };

    fn target(name: &str, client: Option<Arc<dyn RoutedLlmClient>>) -> LlmTarget {
        LlmTarget {
            semantic_name: name.to_string(),
            llm_client: client,
        }
    }

    /// A request whose conversation sits at `turn`: `turn - 1` prior assistant replies, each
    /// answered by a further user message.
    fn request_at_turn(session_id: Option<&str>, turn: usize) -> Request {
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

    /// A mid-conversation request: enough trajectory for the judge to have something to read.
    fn request(session_id: Option<&str>) -> Request {
        request_at_turn(session_id, 3)
    }

    /// Settings that latch on a single escalate verdict, for tests not exercising streaks.
    fn latch_immediately() -> EscalationJudgeSettings {
        EscalationJudgeSettings {
            confirmations: 1,
            ..EscalationJudgeSettings::default()
        }
    }

    fn verdict_json(escalate: bool) -> String {
        format!(r#"{{"escalate": {escalate}, "reason": "test"}}"#)
    }

    /// What a [`RecordingClient`] does when a call reaches it.
    enum Reply {
        /// Return this text as an aggregated response.
        Text(String),
        /// Return this text as a one-chunk stream.
        Stream(String),
        /// Fail the call the way a transport error would.
        Fail,
    }

    /// Convenience: a normal text reply.
    fn ok(text: &str) -> Reply {
        Reply::Text(text.to_string())
    }

    /// Records which models were called (in order) and returns pre-supplied replies.
    ///
    /// All three targets share one instance; calls are served in arrival order regardless of
    /// which target the driver is calling. This mirrors the shared `RecordingClient` pattern
    /// used throughout the libsy test suite.
    struct RecordingClient {
        /// Replies in call order.
        replies: Mutex<VecDeque<Reply>>,
        /// Model names in the order each `call()` arrived.
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingClient {
        fn new(replies: impl IntoIterator<Item = Reply>) -> (Arc<Self>, Arc<Mutex<Vec<String>>>) {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let client = Arc::new(Self {
                replies: Mutex::new(replies.into_iter().collect()),
                calls: calls.clone(),
            });
            (client, calls)
        }
    }

    #[async_trait::async_trait]
    impl RoutedLlmClient for RecordingClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, LlmClientError> {
            self.calls
                .lock()
                .push(decision.selected_model().to_string());
            let reply = self
                .replies
                .lock()
                .pop_front()
                .unwrap_or_else(|| Reply::Text("fallback".to_string()));
            match reply {
                Reply::Text(text) => Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, &text)),
                    metadata: None,
                }),
                Reply::Stream(text) => {
                    let chunk = LlmResponseChunk::TextDelta { index: 0, text };
                    Ok(Response {
                        llm_response: LlmResponse::Stream(
                            futures::stream::iter([Ok(chunk)]).boxed(),
                        ),
                        metadata: None,
                    })
                }
                Reply::Fail => Err(LlmClientError::Configuration {
                    message: "test error".into(),
                }),
            }
        }
    }

    /// A router paired with the call log its shared client writes to.
    type InstrumentedRouter = (Arc<EscalationRouter>, Arc<Mutex<Vec<String>>>);

    /// Router where all three targets share one `RecordingClient`. Replies are served in the
    /// order supplied, matching the call order: the efficient tier first, then the judge, then
    /// the capable tier when the judge escalates.
    fn instrumented_router(
        replies: impl IntoIterator<Item = Reply>,
    ) -> crate::Result<InstrumentedRouter> {
        instrumented_router_with(replies, EscalationJudgeSettings::default())
    }

    fn instrumented_router_with(
        replies: impl IntoIterator<Item = Reply>,
        settings: EscalationJudgeSettings,
    ) -> crate::Result<InstrumentedRouter> {
        let (client, calls) = RecordingClient::new(replies);
        let client: Arc<dyn RoutedLlmClient> = client;
        let router = Arc::new(EscalationRouter::with_settings(
            target("efficient", Some(client.clone())),
            target("capable", Some(client.clone())),
            target("judge", Some(client)),
            settings,
        )?);
        Ok((router, calls))
    }

    /// Drives `router` through `turns` requests in one session, returning the last trace.
    async fn run_turns(
        router: &Arc<EscalationRouter>,
        session: Option<&str>,
        turns: usize,
    ) -> crate::Result<Vec<Arc<dyn Decision>>> {
        let mut trace = Vec::new();
        for _ in 0..turns {
            trace = router
                .clone()
                .run(Context::default(), request(session))
                .await?
                .0;
        }
        Ok(trace)
    }

    #[tokio::test]
    async fn pass_judge_keeps_the_efficient_reply() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("4"),                  // efficient serves the turn
            ok(&verdict_json(false)), // judge: no escalation
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s1"))).await?;

        // The efficient reply the judge read is the reply the caller gets — no second call.
        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("4"));
        assert_eq!(*calls.lock(), vec!["efficient", "judge"]);
        Ok(())
    }

    #[tokio::test]
    async fn confirmed_escalation_reserves_the_turn_for_the_capable_model() -> crate::Result<()> {
        let (router, calls) = instrumented_router_with(
            [
                ok("I give up"),         // efficient serves the turn
                ok(&verdict_json(true)), // judge: escalate
                ok("The answer is 4"),   // capable re-serves it
            ],
            latch_immediately(),
        )?;

        let (trace, response) = router.run(Context::default(), request(Some("s2"))).await?;

        // The efficient reply is discarded once the judge escalates.
        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("The answer is 4"));
        assert_eq!(*calls.lock(), vec!["efficient", "judge", "capable"]);
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), "capable");
        assert_eq!(trace[0].routing_tier(), Some("strong"));
        Ok(())
    }

    #[tokio::test]
    async fn pinned_session_skips_both_the_efficient_call_and_the_judge() -> crate::Result<()> {
        let (router, calls) = instrumented_router_with(
            [
                ok("I give up"),         // efficient (first request)
                ok(&verdict_json(true)), // judge: escalate (first request)
                ok("right"),             // capable (first request)
                ok("right again"),       // capable (second request — pinned)
            ],
            latch_immediately(),
        )?;

        let trace = run_turns(&router, Some("s3"), 2).await?;

        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), "capable");
        // The latch is model-agnostic, so escalation labels the tier itself; after a session
        // escalates this is every remaining turn.
        assert_eq!(trace[0].routing_tier(), Some("strong"));
        // A latched turn costs one call: no efficient reply for the judge to read, no judge.
        assert_eq!(
            *calls.lock(),
            vec!["efficient", "judge", "capable", "capable"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn unparseable_judge_reply_fails_open_to_efficient() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("4"),                    // efficient
            ok("sorry, I cannot help"), // judge: unparseable
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s4"))).await?;

        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("4"));
        assert_eq!(*calls.lock(), vec!["efficient", "judge"]);
        Ok(())
    }

    #[tokio::test]
    async fn failed_judge_call_fails_open_to_efficient() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("4"),     // efficient
            Reply::Fail, // judge: transport failure
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s5"))).await?;

        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("4"));
        assert_eq!(*calls.lock(), vec!["efficient", "judge"]);
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_efficient_call_fails_the_turn() -> crate::Result<()> {
        // The efficient call serves the request; unlike the judge it is not optional, so its
        // failure surfaces rather than being routed around.
        let (router, calls) = instrumented_router([Reply::Fail])?;

        let result = router.run(Context::default(), request(Some("s6"))).await;

        assert!(result.is_err(), "expected the efficient failure to surface");
        assert_eq!(*calls.lock(), vec!["efficient"]);
        Ok(())
    }

    #[tokio::test]
    async fn one_escalate_verdict_does_not_latch_at_default_confirmations() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("4"),                 // efficient
            ok(&verdict_json(true)), // judge: escalate, streak 1 of 2
        ])?;

        let (trace, _) = router.run(Context::default(), request(Some("c1"))).await?;

        assert_eq!(trace[0].selected_model(), "efficient");
        assert_eq!(*calls.lock(), vec!["efficient", "judge"]);
        Ok(())
    }

    #[tokio::test]
    async fn two_consecutive_escalate_verdicts_latch() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("4"),                 // efficient
            ok(&verdict_json(true)), // judge: escalate, streak 1
            ok("4"),                 // efficient
            ok(&verdict_json(true)), // judge: escalate, streak 2 — confirmed
            ok("strong answer"),     // capable
            ok("still strong"),      // capable (third request — pinned)
        ])?;

        let trace = run_turns(&router, Some("c2"), 3).await?;

        assert_eq!(trace[0].selected_model(), "capable");
        assert_eq!(
            *calls.lock(),
            vec![
                "efficient",
                "judge",
                "efficient",
                "judge",
                "capable",
                "capable"
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_decline_between_escalate_verdicts_resets_the_streak() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("4"),                  // efficient
            ok(&verdict_json(true)),  // streak 1
            ok("4"),                  // efficient
            ok(&verdict_json(false)), // decline — streak cleared
            ok("4"),                  // efficient
            ok(&verdict_json(true)),  // streak 1 again, not 2
        ])?;

        run_turns(&router, Some("c3"), 3).await?;

        assert!(
            !calls.lock().contains(&"capable".to_string()),
            "strict-consecutive confirmation must not latch across a decline"
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_unavailable_judge_leaves_the_streak_intact() -> crate::Result<()> {
        let (router, _calls) = instrumented_router([
            ok("4"),                 // efficient
            ok(&verdict_json(true)), // streak 1
            ok("4"),                 // efficient
            Reply::Fail,             // judge unavailable — no evidence either way
            ok("4"),                 // efficient
            ok(&verdict_json(true)), // streak 2 — confirmed despite the gap
            ok("strong answer"),     // capable
        ])?;

        let trace = run_turns(&router, Some("c4"), 3).await?;

        assert_eq!(trace[0].selected_model(), "capable");
        Ok(())
    }

    #[tokio::test]
    async fn without_a_session_id_confirmations_can_never_accumulate() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("4"),
            ok(&verdict_json(true)),
            ok("4"),
            ok(&verdict_json(true)),
        ])?;

        run_turns(&router, None, 2).await?;

        // No session id means no retained state, so the capable tier is unreachable at
        // confirmations > 1. Documented behaviour, asserted so it cannot change silently.
        assert_eq!(
            *calls.lock(),
            vec!["efficient", "judge", "efficient", "judge"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn no_session_id_routes_without_pinning() -> crate::Result<()> {
        let (router, calls) = instrumented_router_with(
            [
                ok("I give up"),          // efficient (request 1)
                ok(&verdict_json(true)),  // judge: escalate (request 1)
                ok("4"),                  // capable (request 1)
                ok("4"),                  // efficient (request 2 — not pinned)
                ok(&verdict_json(false)), // judge: no escalation (request 2)
            ],
            latch_immediately(),
        )?;

        run_turns(&router, None, 2).await?;

        assert_eq!(
            *calls.lock(),
            vec!["efficient", "judge", "capable", "efficient", "judge"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_streamed_efficient_reply_is_buffered_for_the_judge() -> crate::Result<()> {
        let (router, _) = instrumented_router([
            Reply::Stream("streamed".to_string()), // efficient
            ok(&verdict_json(false)),              // judge: no escalation
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s8"))).await?;

        // Judging the turn's own reply means reading it, so the stream is folded before the
        // caller sees it — the cost of post-call judging.
        assert!(matches!(response.llm_response, LlmResponse::Agg(_)));
        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("streamed"));
        Ok(())
    }

    #[tokio::test]
    async fn routing_tiers_are_labelled() -> crate::Result<()> {
        let (router, _) = instrumented_router([ok("4"), ok(&verdict_json(false))])?;

        let (trace, _) = router.run(Context::default(), request(Some("s9"))).await?;

        assert_eq!(trace[0].routing_tier(), Some("weak"));
        Ok(())
    }

    #[tokio::test]
    async fn schema_and_prompt_load_at_construction() -> crate::Result<()> {
        EscalationRouter::new(target("a", None), target("b", None), target("c", None))?;
        Ok(())
    }

    #[test]
    fn judge_request_is_rubric_plus_summary_under_a_completion_cap() -> crate::Result<()> {
        let judge = EscalationJudge {
            config: load_judge_config(PROMPT_TEMPLATE, SCHEMA_TEMPLATE)?,
            settings: EscalationJudgeSettings::default(),
        };

        let built = judge.build_request(&State::default(), &request_at_turn(None, 4));

        // Two messages: the rubric as system, the condensed trajectory as user.
        assert_eq!(built.llm_request.messages.len(), 2);
        assert_eq!(built.llm_request.messages[0].role, Role::System);
        assert_eq!(built.llm_request.messages[1].role, Role::User);
        assert!(built.llm_request.messages[1]
            .text_content("")
            .is_some_and(|text| text.contains("Conversation turn 3")));
        // Bounded output, so a reasoning judge cannot run away mid-verdict.
        assert_eq!(
            built.llm_request.output.max_output_tokens,
            Some(JUDGE_MAX_OUTPUT_TOKENS)
        );
        assert!(built.llm_request.output.response_format.is_some());
        Ok(())
    }

    #[test]
    fn conversation_turn_counts_assistant_replies() {
        // The transcript handed to the judge ends on the reply being judged, so the count is
        // the assistant replies present — no lookahead.
        assert_eq!(conversation_turn(&request_at_turn(None, 1)), 0);
        assert_eq!(conversation_turn(&request_at_turn(None, 5)), 4);
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
    fn summary_keeps_anchors_and_the_recent_window() {
        let mut messages = vec![
            Message::text(Role::System, "you are a coding agent"),
            Message::text(Role::User, "fix the failing test"),
        ];
        for i in 0..10 {
            messages.push(Message::text(Role::Assistant, format!("step {i}")));
        }
        let settings = EscalationJudgeSettings {
            recent_turn_window: 3,
            ..EscalationJudgeSettings::default()
        };

        let summary = summarize_for_judge(&messages, 10, &settings);

        assert!(
            summary.contains("[system] you are a coding agent"),
            "{summary}"
        );
        assert!(
            summary.contains("[user (task)] fix the failing test"),
            "{summary}"
        );
        assert!(summary.contains("Conversation turn 10; showing the last 3 of 12 messages"));
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
        let settings = EscalationJudgeSettings {
            window_message_chars: 2_000,
            ..EscalationJudgeSettings::default()
        };

        let summary = summarize_for_judge(&messages, 20, &settings);

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
