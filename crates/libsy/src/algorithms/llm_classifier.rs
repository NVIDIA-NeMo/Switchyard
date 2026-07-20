// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM-classifier routing as a single SDK component.
//!
//! [`LlmClassifier`] is the **heavy fallback stage** of a classifier cascade: it runs after
//! cheaper classifiers (e.g. [`StagedRouter`](super::staged::StagedRouter)) have abstained.
//! It asks an LLM to score the routing targets, then emits one [`Score`] per target — keyed
//! by the target's `semantic_name` (`"strong"`, `"weak"`, …), the same name the algorithm's
//! processors and classifiers route by. An orchestrator picks the argmax.
//!
//! The classifier call mirrors the components-v2 `llm_routing` profile's "judge" call:
//!
//! - a **forced, strict tool call** (`tool_choice` pins one function, `strict: true`, a
//!   closed JSON schema) is how the model returns structured output — read from the response
//!   tool call, not from free-form text. The only departure from v2's schema is the tool's
//!   parameters: instead of v2's `recommended_tier` + feature fields, they are one number per
//!   candidate target, so the reply is a per-target score map;
//! - the user message is a **trimmed conversation summary** — system, tool names and
//!   descriptions, the first user message, and the last `recent_turn_window` messages, as
//!   JSON capped at 16k — the port of v2's `summarize_request` / `condense_body` /
//!   `trim_messages`;
//! - `temperature: 0` and a `max_tokens` cap, as in v2.
//!
//! v2's tier machinery — `RouteSignals` feature schema, the `policy_tier()` variants, tier
//! mapping, confidence floor, alignment bump, tool-planning escalation — is intentionally
//! dropped: the LLM scores the targets directly, so none of it is needed here. v2's three
//! profiles survive as [`Policy`] presets that only swap the system-prompt rubric (the
//! traffic-specific routing guidance that lived in v2's `policy_tier` weights); the LLM call,
//! schema, and per-target output are identical across them.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Map, Value};
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, Context, Decision, InstructionBlock, LlmRequest, Message,
    OutputParams, Request, Role, SamplingParams, ToolChoice, ToolDefinition,
};

use super::core::{Classifier, Score, State};
use crate::{Driver, LlmTarget};

/// Boxed, thread-safe error type used across the SDK.
type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// A traffic-tuned classifier policy. Selects the system-prompt rubric the classifier scores
/// by — the per-target equivalent of components-v2's `general` / `coding_agent` / `openclaw`
/// profiles, whose behavior lived in `policy_tier` weights and now lives in the prompt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Policy {
    /// Mixed general traffic (chat, summarization, extraction, generic coding).
    #[default]
    General,
    /// Coding-agent harness turns (Claude Code, Codex, Cursor).
    CodingAgent,
    /// OpenClaw personal-assistant channels.
    OpenClaw,
}

impl Policy {
    /// The default system-prompt rubric for this policy.
    pub fn system_prompt(self) -> &'static str {
        match self {
            Policy::General => GENERAL_SYSTEM_PROMPT,
            Policy::CodingAgent => CODING_AGENT_SYSTEM_PROMPT,
            Policy::OpenClaw => OPENCLAW_SYSTEM_PROMPT,
        }
    }
}

/// Rubric for mixed general traffic.
pub const GENERAL_SYSTEM_PROMPT: &str = "You are a model-routing classifier for an LLM proxy. \
Call the route-selection tool exactly once, scoring each candidate model in [0, 1] for how \
well suited it is to the next turn (higher = better). \
Favor a more capable candidate for complex or multi-step reasoning, math or algorithmic work, \
careful debugging, high-precision or high-stakes output, structured-output formats, and \
tool- or planning-heavy turns. Favor a cheaper, faster candidate for simple Q&A, chit-chat, \
short lookups, summarization, and straightforward edits.";

/// Rubric for coding-agent harness turns.
pub const CODING_AGENT_SYSTEM_PROMPT: &str = "You are a model-routing classifier inside a \
coding-agent harness. Call the route-selection tool exactly once, scoring each candidate model \
in [0, 1] for how well suited it is to the next turn (higher = better). \
Favor a more capable candidate for planning and debugging turns, multi-file or cross-module \
changes, and genuine multi-step tool orchestration that modifies code. Favor a cheaper, faster \
candidate for chit-chat and clarifications, exploration and file reads, explanations, and small \
single-file or single-line edits.";

/// Rubric for OpenClaw personal-assistant channels.
pub const OPENCLAW_SYSTEM_PROMPT: &str = "You are a model-routing classifier inside an OpenClaw \
personal-assistant agent. Call the route-selection tool exactly once, scoring each candidate \
model in [0, 1] for how well suited it is to the next message (higher = better). \
Favor a more capable candidate for tool orchestration and irreversible external actions \
(especially high-precision ones), heavy memory recall, ambiguous requests, and deliberate \
channels. Favor a cheaper, faster candidate for casual chit-chat, quick lookups, simple memory \
recall, and clarifications on casual channels.";

/// Default classifier system prompt — the [`Policy::General`] rubric.
pub const DEFAULT_SYSTEM_PROMPT: &str = GENERAL_SYSTEM_PROMPT;

/// Default tool name the classifier is forced to call. Mirrors v2's `select_route`.
pub const DEFAULT_TOOL_NAME: &str = "select_route";

/// Default `max_tokens` for the classifier call. Mirrors v2's `DEFAULT_CLASSIFIER_MAX_TOKENS`.
pub const DEFAULT_MAX_TOKENS: u64 = 4096;

/// Default trailing-turn window kept in the summary. Mirrors v2's `default_recent_turn_window`.
pub const DEFAULT_RECENT_TURN_WINDOW: usize = 4;

/// The routing [`Decision`] behind the classifier's own model call: it names the classifier
/// model so [`Driver::call_llm_target`] hits it.
struct ClassifierCall {
    model: String,
}

impl Decision for ClassifierCall {
    fn selected_model(&self) -> &str {
        &self.model
    }

    fn reasoning(&self) -> Option<&str> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// LLM-backed [`Classifier`]: scores each routing target with one offloaded, tool-calling
/// model call.
///
/// Holds the [`Driver`] to offload that call (constructor-injected, since only LLM-backed
/// classifiers need one) and the classifier [`LlmTarget`] naming the scoring model.
pub struct LlmClassifier {
    driver: Driver,
    classifier_target: LlmTarget,
    /// The target `semantic_name`s to score, one [`Score`] emitted per entry.
    targets: Vec<String>,
    system_prompt: String,
    tool_name: String,
    max_tokens: u64,
    recent_turn_window: usize,
    /// On a classifier failure, the target to route to at confidence `1.0`; `None` propagates.
    fail_open_target: Option<String>,
}

impl LlmClassifier {
    /// Creates a classifier that offloads its call through `driver`, scores with
    /// `classifier_target`, and emits a [`Score`] for each name in `targets`.
    pub fn new(driver: Driver, classifier_target: LlmTarget, targets: Vec<String>) -> Self {
        Self {
            driver,
            classifier_target,
            targets,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            tool_name: DEFAULT_TOOL_NAME.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            recent_turn_window: DEFAULT_RECENT_TURN_WINDOW,
            fail_open_target: None,
        }
    }

    /// Selects a traffic-tuned [`Policy`], setting its system-prompt rubric.
    pub fn with_policy(mut self, policy: Policy) -> Self {
        self.system_prompt = policy.system_prompt().to_string();
        self
    }

    /// Overrides the classifier system prompt (takes precedence over [`with_policy`]).
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Overrides the forced tool name.
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self
    }

    /// Sets the classifier call's `max_tokens` cap.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets how many trailing messages the request summary keeps.
    pub fn with_recent_turn_window(mut self, window: usize) -> Self {
        self.recent_turn_window = window;
        self
    }

    /// Routes to `target` at confidence `1.0` when the classifier call/parse fails, instead
    /// of propagating the error.
    pub fn with_fail_open(mut self, target: impl Into<String>) -> Self {
        self.fail_open_target = Some(target.into());
        self
    }

    /// Offloads the tool-calling classification and returns the parsed per-target score map.
    async fn classify(&self, request: &Request) -> Result<HashMap<String, f64>, BoxErr> {
        let classify_request =
            self.build_request(summarize_request(request, self.recent_turn_window));
        let decision: Arc<dyn Decision> = Arc::new(ClassifierCall {
            model: self.classifier_target.semantic_name.clone(),
        });
        let response = self
            .driver
            .call_llm_target(
                Context::default(),
                &self.classifier_target,
                classify_request,
                decision,
            )
            .await?;
        let aggregate = response.llm_response.into_agg().await?;
        tool_call_scores(&aggregate, &self.tool_name)
            .ok_or_else(|| "classifier did not return the route-selection tool call".into())
    }

    /// Builds the classifier's own request: system prompt, the summary, the forced strict
    /// route-selection tool, and `temperature: 0` + `max_tokens`.
    fn build_request(&self, summary: String) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some(self.classifier_target.semantic_name.clone()),
                instructions: vec![InstructionBlock {
                    role: Role::System,
                    content: vec![ContentBlock::Text { text: self.system_prompt.clone() }],
                }],
                messages: vec![Message::text(Role::User, summary)],
                tools: vec![self.route_tool()],
                tool_choice: Some(ToolChoice::Tool { name: self.tool_name.clone() }),
                sampling: SamplingParams { temperature: Some(0.0), ..SamplingParams::default() },
                output: OutputParams {
                    max_output_tokens: Some(self.max_tokens),
                    ..OutputParams::default()
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    /// The strict route-selection tool: one required `number` property per candidate target.
    fn route_tool(&self) -> ToolDefinition {
        let properties: Map<String, Value> = self
            .targets
            .iter()
            .map(|name| (name.clone(), json!({ "type": "number" })))
            .collect();
        let required: Vec<Value> =
            self.targets.iter().map(|name| Value::String(name.clone())).collect();
        ToolDefinition {
            name: self.tool_name.clone(),
            description: Some(
                "Score each candidate model in [0, 1] for how well it can handle the next turn."
                    .to_string(),
            ),
            parameters: json!({
                "type": "object",
                "properties": Value::Object(properties),
                "required": required,
                "additionalProperties": false,
            }),
            strict: Some(true),
        }
    }

    /// Emits one clamped [`Score`] per target, scored by `confidence`.
    fn scores_by(&self, confidence: impl Fn(&str) -> f64) -> Vec<Score> {
        self.targets
            .iter()
            .map(|name| Score {
                confidence: confidence(name).clamp(0.0, 1.0),
                target: name.clone(),
            })
            .collect()
    }
}

#[async_trait]
impl Classifier for LlmClassifier {
    async fn score(&self, _state: &mut State, request: &Request) -> Result<Vec<Score>, BoxErr> {
        match self.classify(request).await {
            // Missing targets score 0.0.
            Ok(parsed) => Ok(self.scores_by(|name| parsed.get(name).copied().unwrap_or(0.0))),
            // On failure, fail open to the configured default target, or propagate.
            Err(err) => match &self.fail_open_target {
                Some(target) => {
                    Ok(self.scores_by(|name| if name == target { 1.0 } else { 0.0 }))
                }
                None => Err(err),
            },
        }
    }
}

// --- classifier output ----------------------------------------------------------------

/// Reads the forced tool call's arguments as a `{target: score}` map, or `None` when the
/// response carries no such tool call or its arguments are not a JSON object.
fn tool_call_scores(response: &AggLlmResponse, tool_name: &str) -> Option<HashMap<String, f64>> {
    let arguments = response
        .outputs
        .iter()
        .flat_map(|output| &output.content)
        .find_map(|block| match block {
            ContentBlock::ToolCall(call) if call.name == tool_name => Some(&call.arguments),
            _ => None,
        })?;
    // Arguments arrive as a parsed object on some formats, a JSON string on others.
    let value = match arguments {
        Value::String(raw) => serde_json::from_str::<Value>(raw).ok()?,
        other => other.clone(),
    };
    Some(
        value
            .as_object()?
            .iter()
            .filter_map(|(name, score)| score.as_f64().map(|score| (name.clone(), score)))
            .collect(),
    )
}

// --- request summary (port of v2 summarize_request / condense_body / trim_messages) ----

/// Serializes a trimmed JSON summary of the request for the classifier, capped at 16k chars.
/// Keeps the model, system instructions, tool names + descriptions, and the trimmed messages.
fn summarize_request(request: &Request, recent_turn_window: usize) -> String {
    let llm_request = &request.llm_request;
    let system: Vec<String> = llm_request
        .instructions
        .iter()
        .filter_map(|instruction| join_text_blocks(&instruction.content))
        .collect();
    let messages = trim_messages(&llm_request.messages, recent_turn_window);
    let tools: Vec<Value> = llm_request
        .tools
        .iter()
        .map(|tool| json!({ "name": tool.name, "description": tool.description }))
        .collect();
    let payload = json!({
        "model": llm_request.model,
        "system": system,
        "messages": messages,
        "tools": tools,
    });

    const MAX_CHARS: usize = 16_000;
    let text = payload.to_string();
    if text.len() <= MAX_CHARS {
        return text;
    }
    let mut end = MAX_CHARS.saturating_sub(15).min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...<truncated>", &text[..end])
}

/// Keeps the routing-signal slice of the conversation: the first user message plus the last
/// `recent_turn_window` messages (or, with window `0`, just the last user message).
fn trim_messages(messages: &[Message], recent_turn_window: usize) -> Vec<Value> {
    let Some(first_user) = messages.iter().position(|message| message.role == Role::User) else {
        return Vec::new();
    };
    let mut out = vec![render_message(&messages[first_user])];
    let tail = &messages[first_user + 1..];
    if recent_turn_window == 0 {
        if let Some(last_user) = tail.iter().rev().find(|message| message.role == Role::User) {
            out.push(render_message(last_user));
        }
    } else {
        let start = tail.len().saturating_sub(recent_turn_window);
        out.extend(tail[start..].iter().map(render_message));
    }
    out
}

/// Renders one message as `{role, content}`, content condensed to text + tool markers.
fn render_message(message: &Message) -> Value {
    json!({ "role": role_str(message.role), "content": message_content_summary(&message.content) })
}

/// Condenses a message's content to one line: text verbatim, tool calls as `<tool_use:name>`,
/// tool results as a truncated `<tool_result:...>` snippet.
fn message_content_summary(content: &[ContentBlock]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text { text } => parts.push(text.clone()),
            ContentBlock::ToolCall(call) => parts.push(format!("<tool_use:{}>", call.name)),
            ContentBlock::ToolResult(result) => {
                let inner = join_text_blocks(&result.content).unwrap_or_default();
                let snippet: String = inner.chars().take(120).collect();
                parts.push(format!("<tool_result:{snippet}>"));
            }
            _ => {}
        }
    }
    parts.join(" ")
}

/// Joins the text blocks of a content list, or `None` when it has no text.
fn join_text_blocks(content: &[ContentBlock]) -> Option<String> {
    let parts: Vec<&str> = content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

/// The wire-level role name for a normalized [`Role`].
fn role_str(role: Role) -> &'static str {
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
    use super::*;

    use futures::StreamExt;
    use switchyard_protocol::{LlmResponse, Response, ResponseOutput, ToolCall};

    use crate::Step;

    /// How the (stubbed) classifier model replies to the offloaded tool-calling call.
    enum ModelReply {
        /// Return a `select_route` tool call whose arguments are this JSON value.
        ToolCall(Value),
        /// Return an assistant message with no tool call.
        NoToolCall,
        /// Fail the model call.
        Error,
    }

    fn target(name: &str) -> LlmTarget {
        LlmTarget { semantic_name: name.to_string(), llm_client: None }
    }

    fn request(prompt: &str) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("auto".to_string()),
                messages: vec![Message::text(Role::User, prompt)],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    /// Builds the aggregate response for a stubbed reply.
    fn reply_response(reply: &ModelReply) -> Result<Response, BoxErr> {
        let content = match reply {
            ModelReply::ToolCall(arguments) => vec![ContentBlock::ToolCall(ToolCall {
                id: "call-1".to_string(),
                name: DEFAULT_TOOL_NAME.to_string(),
                arguments: arguments.clone(),
            })],
            ModelReply::NoToolCall => vec![ContentBlock::Text { text: "no tool".to_string() }],
            ModelReply::Error => return Err("classifier upstream failed".into()),
        };
        Ok(Response {
            llm_response: LlmResponse::Agg(AggLlmResponse {
                outputs: vec![ResponseOutput { role: Role::Assistant, content, stop_reason: None }],
                ..AggLlmResponse::default()
            }),
            metadata: None,
        })
    }

    /// Runs a classifier's `score` against a stubbed reply, returning the scores and the
    /// classifier requests it offloaded.
    async fn run_score(
        classifier: LlmClassifier,
        driver: Driver,
        request: Request,
        reply: ModelReply,
    ) -> Result<(Vec<Score>, Vec<Request>), BoxErr> {
        let stream = driver.stream();
        let handle = tokio::spawn(async move {
            let mut state = State::default();
            classifier.score(&mut state, &request).await
        });

        let mut seen = Vec::new();
        tokio::pin!(stream);
        while let Some(step) = stream.next().await {
            if let Step::CallLlm(call) = step? {
                seen.push(call.get_request().clone());
                call.respond(reply_response(&reply))?;
                break;
            }
        }
        handle.await?.map(|scores| (scores, seen))
    }

    /// A classifier over `["strong", "weak"]` sharing `driver`.
    fn classifier(driver: &Driver) -> LlmClassifier {
        LlmClassifier::new(
            driver.clone(),
            target("classifier-model"),
            vec!["strong".to_string(), "weak".to_string()],
        )
    }

    #[tokio::test]
    async fn scores_each_target_from_the_tool_call() -> Result<(), BoxErr> {
        let driver = Driver::new();
        let (scores, _) = run_score(
            classifier(&driver),
            driver.clone(),
            request("solve this proof"),
            ModelReply::ToolCall(json!({ "strong": 0.9, "weak": 0.2 })),
        )
        .await?;
        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0].target, "strong");
        assert!((scores[0].confidence - 0.9).abs() < 1e-9);
        assert_eq!(scores[1].target, "weak");
        assert!((scores[1].confidence - 0.2).abs() < 1e-9);
        Ok(())
    }

    #[tokio::test]
    async fn tool_call_arguments_as_json_string_are_parsed() -> Result<(), BoxErr> {
        // Some formats deliver `arguments` as a JSON string rather than an object.
        let driver = Driver::new();
        let (scores, _) = run_score(
            classifier(&driver),
            driver.clone(),
            request("hi"),
            ModelReply::ToolCall(Value::String(r#"{"strong": 0.4, "weak": 0.6}"#.to_string())),
        )
        .await?;
        assert!((scores[0].confidence - 0.4).abs() < 1e-9);
        assert!((scores[1].confidence - 0.6).abs() < 1e-9);
        Ok(())
    }

    #[tokio::test]
    async fn missing_target_scores_zero_and_out_of_range_is_clamped() -> Result<(), BoxErr> {
        let driver = Driver::new();
        let (scores, _) = run_score(
            classifier(&driver),
            driver.clone(),
            request("hi"),
            ModelReply::ToolCall(json!({ "strong": 1.5 })),
        )
        .await?;
        assert_eq!(scores[0].confidence, 1.0); // clamped
        assert_eq!(scores[1].target, "weak");
        assert_eq!(scores[1].confidence, 0.0); // missing → 0.0
        Ok(())
    }

    #[tokio::test]
    async fn missing_tool_call_without_fail_open_errors() -> Result<(), BoxErr> {
        let driver = Driver::new();
        let result = run_score(
            classifier(&driver),
            driver.clone(),
            request("hi"),
            ModelReply::NoToolCall,
        )
        .await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn call_error_without_fail_open_propagates() -> Result<(), BoxErr> {
        let driver = Driver::new();
        let result =
            run_score(classifier(&driver), driver.clone(), request("hi"), ModelReply::Error).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn fail_open_routes_to_the_default_target() -> Result<(), BoxErr> {
        let driver = Driver::new();
        let (scores, _) = run_score(
            classifier(&driver).with_fail_open("weak"),
            driver.clone(),
            request("hi"),
            ModelReply::Error,
        )
        .await?;
        assert_eq!(scores[0].target, "strong");
        assert_eq!(scores[0].confidence, 0.0);
        assert_eq!(scores[1].target, "weak");
        assert_eq!(scores[1].confidence, 1.0);
        Ok(())
    }

    #[tokio::test]
    async fn classifier_call_forces_the_strict_route_tool() -> Result<(), BoxErr> {
        let driver = Driver::new();
        let (_, seen) = run_score(
            classifier(&driver),
            driver.clone(),
            request("prove it"),
            ModelReply::ToolCall(json!({ "strong": 0.9, "weak": 0.1 })),
        )
        .await?;
        assert_eq!(seen.len(), 1);
        let llm = &seen[0].llm_request;
        // Forced strict tool with a required number property per target.
        assert_eq!(llm.tool_choice, Some(ToolChoice::Tool { name: DEFAULT_TOOL_NAME.to_string() }));
        assert_eq!(llm.sampling.temperature, Some(0.0));
        assert_eq!(llm.output.max_output_tokens, Some(DEFAULT_MAX_TOKENS));
        let tool = &llm.tools[0];
        assert_eq!(tool.strict, Some(true));
        assert!(tool.parameters["properties"]["strong"].is_object());
        assert!(tool.parameters["properties"]["weak"].is_object());
        // The trimmed summary carries the user turn.
        let user_text = llm
            .messages
            .iter()
            .find_map(|message| message.text_content("\n"))
            .unwrap_or_default();
        assert!(user_text.contains("prove it"));
        Ok(())
    }

    #[tokio::test]
    async fn policy_selects_the_traffic_tuned_system_prompt() -> Result<(), BoxErr> {
        let driver = Driver::new();
        let (_, seen) = run_score(
            classifier(&driver).with_policy(Policy::CodingAgent),
            driver.clone(),
            request("refactor auth"),
            ModelReply::ToolCall(json!({ "strong": 0.8, "weak": 0.2 })),
        )
        .await?;
        let system = seen[0]
            .llm_request
            .instructions
            .iter()
            .find_map(|instruction| instruction.content.iter().find_map(|block| match block {
                ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            }))
            .unwrap_or_default();
        assert!(system.contains("coding-agent harness"));
        // The three policies carry distinct rubrics.
        assert_ne!(Policy::General.system_prompt(), Policy::CodingAgent.system_prompt());
        assert_ne!(Policy::CodingAgent.system_prompt(), Policy::OpenClaw.system_prompt());
        // General is the default.
        assert_eq!(Policy::default(), Policy::General);
        assert_eq!(DEFAULT_SYSTEM_PROMPT, Policy::General.system_prompt());
        Ok(())
    }

    #[tokio::test]
    async fn summary_trims_to_first_user_and_recent_window() -> Result<(), BoxErr> {
        // 1 first user + 8 assistant turns; window keeps the first user + last 4 messages.
        let mut messages = vec![Message::text(Role::User, "FIRST")];
        for index in 0..8 {
            messages.push(Message::text(Role::Assistant, format!("turn {index}")));
        }
        let summary = summarize_request(
            &Request { llm_request: LlmRequest { messages, ..LlmRequest::default() }, raw_request: None, metadata: None },
            4,
        );
        assert!(summary.contains("FIRST")); // first-user anchor kept
        assert!(summary.contains("turn 7")); // last turn kept
        assert!(!summary.contains("turn 2")); // middle dropped by the window
        Ok(())
    }
}
