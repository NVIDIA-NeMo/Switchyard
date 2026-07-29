// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trajectory-judged escalation: serve the efficient model until an LLM judge finds the run
//! in trouble, then latch the session to the capable model for the rest of the task.
//!
//! The judge reads the conversation as it arrives — which already carries the previous turns'
//! assistant output and tool results — and picks the tier *before* the turn's model call. So a
//! turn costs one model call, and nothing is buffered on the way back: the target's response,
//! streamed or aggregated, reaches the caller untouched.
//!
//! Once a session escalates it stays on the capable model, since re-trying a model that has
//! already failed this task wastes a turn. A judge failure fails open to the efficient tier and
//! never latches: an outage costs quality risk, never money.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::Deserialize;
use switchyard_protocol::{ContentBlock, LlmRequest, Message, OutputParams, Role};

use crate::{
    algorithms::util::{
        load_judge_config, AffinityRouter, Judge, JudgeClassifier, JudgeConfig, JudgePolicy,
    },
    Algorithm, Classification, Classifier, Context, Decision, Driver, Event, LlmTarget, Processor,
    Request, Response, Result, RoutedLlmClient, Score, State,
};

const PROMPT_TEMPLATE: &str = include_str!("../prompts/escalation/prompt.md");
const SCHEMA_TEMPLATE: &str = include_str!("../prompts/escalation/schema.json");

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

/// Ceiling on tracked confirmation streaks, mirroring `AffinityRouter`'s assignment cap.
const MAX_STREAKS: usize = 4_096;

/// The tuning surface for the trajectory judge.
///
/// Deliberately small: these are the three settings an audit of the Python router's twenty-odd
/// knobs found to actually change routing outcomes. Everything else it exposed is either a
/// fixed invariant (see the `*_CHARS` constants above) or operational plumbing.
///
/// Defaults are the ESC7 converged configuration.
#[derive(Clone, Debug)]
pub struct EscalationJudgeSettings {
    /// Consecutive escalate verdicts required before the latch fires.
    ///
    /// `1` latches on the first verdict. Higher values filter one-shot eager verdicts — a
    /// single failed command misread as a pattern — while keeping recall on real trouble,
    /// which by definition persists across turns. On the benchmarked workload `2` suppressed
    /// roughly two thirds of escalate verdicts, so this is the router's main cost dial.
    ///
    /// The streak is keyed on the session id and held per process. A request without a
    /// session id cannot accumulate one, so at `2` or higher it can never escalate.
    pub confirmations: u32,
    /// Trailing messages shown on top of the anchors. Loop detection needs to see the
    /// repeats, so a cycle longer than this window is invisible to the judge.
    pub recent_turn_window: usize,
    /// Per-message cap inside the trailing window. Error signatures and command shapes
    /// survive this easily; full file dumps do not need to.
    pub window_message_chars: usize,
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

struct EscalationDecision {
    model: String,
    tier: &'static str,
    reason: &'static str,
}

impl Decision for EscalationDecision {
    fn selected_model(&self) -> &str {
        &self.model
    }

    fn routing_tier(&self) -> Option<&'static str> {
        Some(self.tier)
    }

    fn reasoning(&self) -> Option<&str> {
        Some(self.reason)
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
                    ..OutputParams::default()
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }
}

/// Maps the judge's verdict to a routing classification, keeping "the judge declined" and
/// "the judge was unavailable" distinguishable.
///
/// Both route to the efficient tier, but only a decline is evidence about the run, so only a
/// decline clears a confirmation streak. [`Classification::Ambiguous`] carries the unavailable
/// case: it argmaxes to nothing, exactly like an empty score set, but the router can tell them
/// apart.
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

/// Serves the efficient model until the judge escalates, then pins the session to the
/// capable model.
pub struct EscalationRouter {
    efficient_target: LlmTarget,
    capable_target: LlmTarget,
    judge_classifier: JudgeClassifier<EscalationJudge, EscalationPolicy>,
    /// Latches sessions to the capable model after their first escalation.
    affinity: AffinityRouter,
    /// Consecutive escalate verdicts required to latch.
    confirmations: u32,
    /// Consecutive escalate verdicts per session, cleared by any decline. Only consulted when
    /// `confirmations > 1`; a single-verdict latch needs no bookkeeping.
    streaks: Mutex<HashMap<String, u32>>,
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
        let config = load_judge_config(PROMPT_TEMPLATE, SCHEMA_TEMPLATE)?;
        let capable_name = capable_target.semantic_name.clone();
        let confirmations = settings.confirmations;
        Ok(Self {
            efficient_target,
            capable_target,
            judge_classifier: JudgeClassifier::new(
                EscalationJudge { config, settings },
                judge_target,
                EscalationPolicy {
                    capable: capable_name.clone(),
                },
            ),
            affinity: AffinityRouter::new().with_latch_only([capable_name]),
            confirmations,
            streaks: Mutex::new(HashMap::new()),
        })
    }

    /// Records one escalate verdict and reports whether the session has now confirmed.
    ///
    /// A session id is required to accumulate across turns, so without one nothing can confirm
    /// beyond a single-verdict latch.
    fn confirm(&self, request: &Request) -> bool {
        if self.confirmations <= 1 {
            return true;
        }
        let Some(session) = session_id(request) else {
            return false;
        };
        let mut streaks = self.streaks.lock();
        if streaks.len() >= MAX_STREAKS && !streaks.contains_key(session) {
            if let Some(evicted) = streaks.keys().next().cloned() {
                streaks.remove(&evicted);
            }
        }
        let streak = streaks.entry(session.to_string()).or_insert(0);
        *streak += 1;
        *streak >= self.confirmations
    }

    /// Clears a session's streak after a decline. Strict-consecutive: any decline resets.
    fn clear_streak(&self, request: &Request) {
        if self.confirmations <= 1 {
            return;
        }
        if let Some(session) = session_id(request) {
            self.streaks.lock().remove(session);
        }
    }

    /// Publishes the routing decision and serves the turn from `target`.
    ///
    /// The target's response is returned as it arrives, so a streamed upstream reply stays a
    /// stream all the way back to the caller.
    async fn call_tier(
        &self,
        ctx: Context,
        driver: &Driver,
        request: Request,
        target: &LlmTarget,
        tier: &'static str,
        reason: &'static str,
    ) -> Result<Response> {
        let decision: Arc<dyn Decision> = Arc::new(EscalationDecision {
            model: target.semantic_name.clone(),
            tier,
            reason,
        });
        driver.info(ctx.clone(), decision.clone()).await?;
        driver.call_llm_target(ctx, target, request, decision).await
    }
}

#[async_trait]
impl Algorithm for EscalationRouter {
    fn name(&self) -> &str {
        "escalation"
    }

    fn count_tokens_client(&self) -> Option<Arc<dyn RoutedLlmClient>> {
        [&self.capable_target, &self.efficient_target]
            .iter()
            .find_map(|t| {
                t.llm_client
                    .as_ref()
                    .filter(|c| c.supports_count_tokens())
                    .cloned()
            })
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        let mut request = request;
        // Neither the affinity router nor the judge keeps anything here: the latch lives in
        // AffinityRouter's own storage and the judge reads only the request. State is passed
        // because the Classifier/Processor traits take it.
        let mut state = State::default();

        // A session that already escalated stays escalated — one-way for the rest of the task.
        let is_pinned = {
            let classification = self.affinity.score(&mut state, &mut request, None).await?;
            matches!(classification, Classification::Scores(ref s) if !s.is_empty())
        };
        if is_pinned {
            return self
                .call_tier(
                    ctx,
                    &driver,
                    request,
                    &self.capable_target,
                    "strong",
                    "session pinned to capable after prior escalation",
                )
                .await;
        }

        // Consult the judge. A decline and an unavailable judge both stay on the efficient
        // tier, but only a decline is evidence, so only a decline clears the streak.
        let classification = self
            .judge_classifier
            .score(&mut state, &mut request, Some(&driver))
            .await?;
        let escalate = match classification {
            Classification::Scores(ref scores) if !scores.is_empty() => self.confirm(&request),
            Classification::Scores(_) => {
                self.clear_streak(&request);
                false
            }
            // Judge unavailable: fail open to efficient without disturbing the streak.
            Classification::Ambiguous(_) => false,
        };
        if !escalate {
            return self
                .call_tier(
                    ctx,
                    &driver,
                    request,
                    &self.efficient_target,
                    "weak",
                    "judge has not confirmed the run is in trouble",
                )
                .await;
        }

        // Latch before serving, so later turns skip the judge and go straight to capable.
        let decision: Arc<dyn Decision> = Arc::new(EscalationDecision {
            model: self.capable_target.semantic_name.clone(),
            tier: "strong",
            reason: "judge escalated the run to the capable model",
        });
        self.affinity
            .process(&mut state, Event::Request(&mut request))
            .await?;
        self.affinity
            .process(
                &mut state,
                Event::Decision {
                    request: &request,
                    decision: &*decision,
                },
            )
            .await?;

        driver.info(ctx.clone(), decision.clone()).await?;
        driver
            .call_llm_target(ctx, &self.capable_target, request, decision)
            .await
    }
}

/// The caller-supplied conversation id, when present and non-empty.
fn session_id(request: &Request) -> Option<&str> {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.session_id.as_deref())
        .filter(|id| !id.is_empty())
}

/// The 1-indexed model invocation this request represents: one past each assistant reply.
///
/// libsy receives messages already normalized by `switchyard-protocol`, so unlike the
/// wire-format-aware Python equivalent this needs no per-format branching.
fn conversation_turn(request: &Request) -> usize {
    request
        .llm_request
        .messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .count()
        + 1
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
    use serde_json::json;
    use switchyard_protocol::{
        completion_text, text_response, ContentBlock, LlmClientError, LlmRequest, LlmResponseChunk,
        Message, Metadata, Role, ToolCall, ToolResult,
    };

    use crate::{
        Algorithm, Context, Decision, LlmResponse, LlmTarget, Request, Response, RoutedLlmClient,
    };

    use super::{
        conversation_turn, message_text, summarize_for_judge, truncate_middle,
        EscalationJudgeSettings, EscalationRouter, MAX_REQUEST_CHARS,
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
        /// Return this text as a one-chunk stream, so buffering anywhere is observable.
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
    /// order supplied, matching the call order: the judge first (when it runs), then the tier.
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

    #[tokio::test]
    async fn pass_judge_serves_efficient() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok(&verdict_json(false)), // judge: no escalation
            ok("4"),                  // efficient
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s1"))).await?;

        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("4"));
        assert_eq!(*calls.lock(), vec!["judge", "efficient"]);
        Ok(())
    }

    #[tokio::test]
    async fn confirmed_escalation_serves_capable_without_calling_efficient() -> crate::Result<()> {
        let (router, calls) = instrumented_router_with(
            [
                ok(&verdict_json(true)), // judge: escalate
                ok("The answer is 4"),   // capable
            ],
            latch_immediately(),
        )?;

        let (trace, response) = router.run(Context::default(), request(Some("s2"))).await?;

        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("The answer is 4"));
        // The efficient tier is never called: the judge decided before the turn's model call.
        assert_eq!(*calls.lock(), vec!["judge", "capable"]);
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), "capable");
        Ok(())
    }

    #[tokio::test]
    async fn pinned_session_skips_the_judge() -> crate::Result<()> {
        let (router, calls) = instrumented_router_with(
            [
                ok(&verdict_json(true)), // judge: escalate (first request)
                ok("right"),             // capable (first request)
                ok("right again"),       // capable (second request — pinned)
            ],
            latch_immediately(),
        )?;

        router
            .clone()
            .run(Context::default(), request(Some("s3")))
            .await?;
        let (trace, _) = router.run(Context::default(), request(Some("s3"))).await?;

        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), "capable");
        assert_eq!(*calls.lock(), vec!["judge", "capable", "capable"]);
        Ok(())
    }

    #[tokio::test]
    async fn unparseable_judge_reply_fails_open_to_efficient() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("sorry, I cannot help"), // judge: unparseable
            ok("4"),                    // efficient
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s4"))).await?;

        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("4"));
        assert_eq!(*calls.lock(), vec!["judge", "efficient"]);
        Ok(())
    }

    #[tokio::test]
    async fn failed_judge_call_fails_open_to_efficient() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            Reply::Fail, // judge: transport failure
            ok("4"),     // efficient
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s5"))).await?;

        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("4"));
        assert_eq!(*calls.lock(), vec!["judge", "efficient"]);
        Ok(())
    }

    #[tokio::test]
    async fn one_escalate_verdict_does_not_latch_at_default_confirmations() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok(&verdict_json(true)), // judge: escalate, streak 1 of 2
            ok("4"),                 // efficient — not yet confirmed
        ])?;

        let (trace, _) = router.run(Context::default(), request(Some("c1"))).await?;

        assert_eq!(trace[0].selected_model(), "efficient");
        assert_eq!(*calls.lock(), vec!["judge", "efficient"]);
        Ok(())
    }

    #[tokio::test]
    async fn two_consecutive_escalate_verdicts_latch() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok(&verdict_json(true)), // judge: escalate, streak 1
            ok("4"),                 // efficient
            ok(&verdict_json(true)), // judge: escalate, streak 2 — confirmed
            ok("strong answer"),     // capable
            ok("still strong"),      // capable (third request — pinned)
        ])?;

        router
            .clone()
            .run(Context::default(), request(Some("c2")))
            .await?;
        router
            .clone()
            .run(Context::default(), request(Some("c2")))
            .await?;
        let (trace, _) = router.run(Context::default(), request(Some("c2"))).await?;

        assert_eq!(trace[0].selected_model(), "capable");
        assert_eq!(
            *calls.lock(),
            vec!["judge", "efficient", "judge", "capable", "capable"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_decline_between_escalate_verdicts_resets_the_streak() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok(&verdict_json(true)),  // streak 1
            ok("4"),                  // efficient
            ok(&verdict_json(false)), // decline — streak cleared
            ok("4"),                  // efficient
            ok(&verdict_json(true)),  // streak 1 again, not 2
            ok("4"),                  // efficient — still no latch
        ])?;

        for _ in 0..3 {
            router
                .clone()
                .run(Context::default(), request(Some("c3")))
                .await?;
        }

        assert!(
            !calls.lock().contains(&"capable".to_string()),
            "strict-consecutive confirmation must not latch across a decline"
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_unavailable_judge_leaves_the_streak_intact() -> crate::Result<()> {
        let (router, _calls) = instrumented_router([
            ok(&verdict_json(true)), // streak 1
            ok("4"),                 // efficient
            Reply::Fail,             // judge unavailable — no evidence either way
            ok("4"),                 // efficient
            ok(&verdict_json(true)), // streak 2 — confirmed despite the gap
            ok("strong answer"),     // capable
        ])?;

        for _ in 0..2 {
            router
                .clone()
                .run(Context::default(), request(Some("c4")))
                .await?;
        }
        let (trace, _) = router.run(Context::default(), request(Some("c4"))).await?;

        assert_eq!(trace[0].selected_model(), "capable");
        Ok(())
    }

    #[tokio::test]
    async fn without_a_session_id_confirmations_can_never_accumulate() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok(&verdict_json(true)),
            ok("4"),
            ok(&verdict_json(true)),
            ok("4"),
        ])?;

        router
            .clone()
            .run(Context::default(), request(None))
            .await?;
        router.run(Context::default(), request(None)).await?;

        // No session id means no streak store, so the capable tier is unreachable at
        // confirmations > 1. Documented behaviour, asserted so it cannot change silently.
        assert_eq!(
            *calls.lock(),
            vec!["judge", "efficient", "judge", "efficient"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn no_session_id_routes_without_pinning() -> crate::Result<()> {
        let (router, calls) = instrumented_router_with(
            [
                ok(&verdict_json(true)),  // judge: escalate (request 1)
                ok("4"),                  // capable (request 1)
                ok(&verdict_json(false)), // judge: no escalation (request 2 — not pinned)
                ok("4"),                  // efficient (request 2)
            ],
            latch_immediately(),
        )?;

        router
            .clone()
            .run(Context::default(), request(None))
            .await?;
        router.run(Context::default(), request(None)).await?;

        assert_eq!(
            *calls.lock(),
            vec!["judge", "capable", "judge", "efficient"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn efficient_response_streams_through_unbuffered() -> crate::Result<()> {
        let (router, _) = instrumented_router([
            ok(&verdict_json(false)),              // judge: no escalation
            Reply::Stream("streamed".to_string()), // efficient
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s8"))).await?;

        // Judging before the call is what lets the tier's stream survive: nothing needs to
        // read the response to make a routing decision.
        assert!(matches!(response.llm_response, LlmResponse::Stream(_)));
        Ok(())
    }

    #[tokio::test]
    async fn routing_tiers_are_labelled() -> crate::Result<()> {
        let (router, _) = instrumented_router([ok(&verdict_json(false)), ok("4")])?;

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
    fn conversation_turn_counts_assistant_replies() {
        assert_eq!(conversation_turn(&request_at_turn(None, 1)), 1);
        assert_eq!(conversation_turn(&request_at_turn(None, 5)), 5);
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

        let summary = summarize_for_judge(&messages, 11, &settings);

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
        let settings = EscalationJudgeSettings {
            window_message_chars: 2_000,
            ..EscalationJudgeSettings::default()
        };

        let summary = summarize_for_judge(&messages, 21, &settings);

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
