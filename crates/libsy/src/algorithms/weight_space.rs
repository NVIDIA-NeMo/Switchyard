// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A bounded neural routing policy evaluated entirely in process.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use switchyard_protocol::{
    Category, ContentBlock, LlmRequest, Message, ModelId, Request, ToolCall, ToolResult,
};

use crate::algorithms::fall_through::FallThrough;
use crate::algorithms::util::affinity::{has_new_user_turn, is_user_turn};
use crate::core::algorithm::{Algorithm, Driver};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::state::{State, StateValue};
use crate::{LibsyError, Result, RoutingOutcome, ToolSignals};

/// Ordered request features. Previous-target indicators follow these inputs.
pub const WEIGHT_SPACE_FEATURES: [&str; 16] = [
    "new_user_turn",
    "task_bytes",
    "change_task",
    "analysis_task",
    "history_depth",
    "recent_turns",
    "severity",
    "repeated_failure",
    "recovery",
    "mutation_share",
    "observation_share",
    "planning_share",
    "tests_passed",
    "unknown_streak",
    "has_mutated",
    "truncated",
];

const MAX_MESSAGES: usize = 32;
const MAX_BLOCKS: usize = 8;
const MAX_BYTES: usize = 65_536;

/// Explicit weights for one ReLU hidden layer and one linear output per target.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WeightSpaceWeights {
    /// Version of the feature layout, currently 1.
    pub schema_version: u32,
    /// Hidden rows over request features followed by previous-target indicators.
    pub input_weights: Vec<Vec<f64>>,
    /// One bias per hidden unit.
    pub hidden_bias: Vec<f64>,
    /// One row per target, over hidden activations.
    pub output_weights: Vec<Vec<f64>>,
    /// One bias per target.
    pub output_bias: Vec<f64>,
    /// Nonnegative relative cost preferences, in target order.
    pub cost_bias: Vec<f64>,
}

/// Inspectable activations and preferences, without calibrated probability claims.
#[derive(Clone, Debug, Serialize)]
pub struct WeightSpaceTrace {
    /// Bounded request features and previous-target indicators.
    pub inputs: Vec<f64>,
    /// ReLU activations in checkpoint order.
    pub hidden: Vec<f64>,
    /// Linear preferences after the configured cost penalty.
    pub utilities: Vec<f64>,
    /// Winning target index. Ties keep the first target.
    pub selected: usize,
}

impl WeightSpaceWeights {
    /// Rejects incompatible shapes, unbounded coefficients and unknown versions.
    pub fn validate(&self, target_count: usize) -> Result<()> {
        let hidden = self.hidden_bias.len();
        let inputs = WEIGHT_SPACE_FEATURES.len() + target_count;
        if self.schema_version != 1
            || !(1..=32).contains(&target_count)
            || !(1..=64).contains(&hidden)
            || self.input_weights.len() != hidden
            || self.input_weights.iter().any(|row| row.len() != inputs)
            || self.output_weights.len() != target_count
            || self.output_weights.iter().any(|row| row.len() != hidden)
            || self.output_bias.len() != target_count
            || self.cost_bias.len() != target_count
        {
            return Err(invalid("incompatible version or matrix dimensions"));
        }
        if self
            .input_weights
            .iter()
            .flatten()
            .chain(&self.hidden_bias)
            .chain(self.output_weights.iter().flatten())
            .chain(&self.output_bias)
            .chain(&self.cost_bias)
            .any(|v| !v.is_finite() || v.abs() > 1_000_000.0)
            || self.cost_bias.iter().any(|v| *v < 0.0)
        {
            return Err(invalid(
                "weights must be finite and bounded, costs nonnegative",
            ));
        }
        Ok(())
    }

    fn forward(&self, inputs: Vec<f64>, cost_weight: f64) -> WeightSpaceTrace {
        let hidden: Vec<_> = self
            .input_weights
            .iter()
            .zip(&self.hidden_bias)
            .map(|(row, bias)| (dot(row, &inputs) + bias).max(0.0))
            .collect();
        let utilities: Vec<_> = self
            .output_weights
            .iter()
            .zip(&self.output_bias)
            .zip(&self.cost_bias)
            .map(|((row, bias), cost)| dot(row, &hidden) + bias - cost_weight * cost)
            .collect();
        let mut selected = 0;
        for i in 1..utilities.len() {
            if utilities[i] > utilities[selected] {
                selected = i;
            }
        }
        WeightSpaceTrace {
            inputs,
            hidden,
            utilities,
            selected,
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn invalid(message: &str) -> LibsyError {
    LibsyError::AlgorithmError {
        message: format!("weight_space: {message}"),
    }
}

/// Neural classifier with a bounded snapshot and compact session state.
pub struct WeightSpaceClassifier {
    targets: Vec<ModelId>,
    weights: WeightSpaceWeights,
    cost_weight: f64,
}

impl WeightSpaceClassifier {
    /// Validates and loads an immutable policy. Target order binds output rows.
    pub fn new(
        targets: Vec<ModelId>,
        weights: WeightSpaceWeights,
        cost_weight: f64,
    ) -> Result<Self> {
        weights.validate(targets.len())?;
        if targets
            .iter()
            .enumerate()
            .any(|(i, t)| targets[..i].contains(t))
        {
            return Err(invalid("targets must be unique"));
        }
        if !cost_weight.is_finite() || !(0.0..=1_000_000.0).contains(&cost_weight) {
            return Err(invalid(
                "cost_weight must be finite and between 0 and 1000000",
            ));
        }
        Ok(Self {
            targets,
            weights,
            cost_weight,
        })
    }

    /// Selects a target and records the last routing choice and mutation state.
    pub fn decide(&self, state: &mut State, request: &Request) -> WeightSpaceTrace {
        let (trace, task) = self.evaluate(state, request);
        if let Some(task) = task {
            state
                .extra
                .insert(state_key(request, "task"), StateValue::String(task));
        }
        state.extra.insert(
            state_key(request, "last_target"),
            StateValue::String(self.targets[trace.selected].to_string()),
        );
        state.extra.insert(
            state_key(request, "has_mutated"),
            StateValue::Bool(trace.inputs[14] > 0.0),
        );
        trace
    }

    /// Replays a decision without changing state or invoking a model.
    pub fn trace(&self, state: &State, request: &Request) -> WeightSpaceTrace {
        self.evaluate(state, request).0
    }

    fn evaluate(&self, state: &State, request: &Request) -> (WeightSpaceTrace, Option<String>) {
        let snapshot = snapshot(request);
        let signal = ToolSignals::from_request(&snapshot.request, Some(5));
        let key = state_key(request, "last_target");
        let prior = match state.extra.get(&key) {
            Some(StateValue::String(value)) => Some(value.as_str()),
            _ => None,
        };
        let new_turn = has_new_user_turn(&snapshot.request.llm_request.messages);
        let has_mutated = signal.write_count + signal.edit_count > 0
            || (!new_turn
                && matches!(
                    state.extra.get(&state_key(request, "has_mutated")),
                    Some(StateValue::Bool(true))
                ));
        let ops = signal.recent_write_count
            + signal.recent_edit_count
            + signal.recent_read_count
            + signal.recent_todowrite_count;
        let cached_task = match state.extra.get(&state_key(request, "task")) {
            Some(StateValue::String(value)) => value.as_str(),
            _ => "",
        };
        let task = snapshot
            .task
            .as_deref()
            .unwrap_or(cached_task)
            .to_ascii_lowercase();
        let mut inputs = vec![
            flag(new_turn),
            (task.len() as f64 / 8192.0).min(1.0),
            flag(
                ["implement", "fix", "add", "refactor"]
                    .iter()
                    .any(|s| task.split_whitespace().any(|w| w == *s)),
            ),
            flag(
                ["explain", "investigate", "why", "which"]
                    .iter()
                    .any(|s| task.split_whitespace().any(|w| w == *s)),
            ),
            (request.llm_request.messages.len() as f64 / 256.0).min(1.0),
            (f64::from(signal.assistant_turn_count) / 8.0).min(1.0),
            f64::from(signal.severity),
            flag(signal.repeated_failure),
            (f64::from(signal.no_error_streak) / 4.0).min(1.0),
            ratio(signal.recent_write_count + signal.recent_edit_count, ops),
            ratio(signal.recent_read_count, ops),
            ratio(signal.recent_todowrite_count, ops),
            flag(signal.tests_passed),
            (f64::from(signal.pure_bash_streak) / 8.0).min(1.0),
            flag(has_mutated),
            flag(snapshot.truncated),
        ];
        inputs.extend(
            self.targets
                .iter()
                .map(|t| flag(prior == Some(t.as_str()) && !new_turn)),
        );
        (
            self.weights.forward(inputs, self.cost_weight),
            snapshot.task,
        )
    }
}

fn flag(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
}
fn ratio(a: u32, b: u32) -> f64 {
    f64::from(a) / f64::from(b.max(1))
}

fn state_key(request: &Request, suffix: &str) -> String {
    let agent = request
        .metadata
        .as_ref()
        .and_then(|m| m.agent_id.as_deref());
    format!("weight_space:{agent:?}:{suffix}")
}

#[async_trait]
impl Classifier for WeightSpaceClassifier {
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<switchyard_protocol::Response>)> {
        let available = driver.models_for(&Category::Any);
        if self
            .targets
            .iter()
            .any(|target| !available.contains(target))
        {
            return Err(invalid("checkpoint target missing from runtime models"));
        }
        let trace = self.decide(state, request);
        let target = self.targets[trace.selected].clone();
        driver.set_evidence(serde_json::json!({
            "source": "weight_space", "schema_version": 1,
            "cost_weight": self.cost_weight, "trace": trace,
        }));
        Ok((
            Classification::Scores(vec![Score {
                target,
                confidence: 1.0,
                category: Some(Category::Any),
            }]),
            None,
        ))
    }
}

/// Routes to a named target with one local neural forward pass and no judge calls.
pub struct WeightSpaceRouter {
    inner: FallThrough,
}

impl WeightSpaceRouter {
    /// Loads a validated policy into the standard session-aware routing runtime.
    pub fn new(
        targets: Vec<ModelId>,
        weights: WeightSpaceWeights,
        cost_weight: f64,
    ) -> Result<Self> {
        let classifier = WeightSpaceClassifier::new(targets, weights, cost_weight)?;
        Ok(Self {
            inner: FallThrough::new_with_state()
                .with_name("weight_space")
                .with_classifier(Arc::new(classifier)),
        })
    }
}

#[async_trait]
impl Algorithm for WeightSpaceRouter {
    fn name(&self) -> &str {
        "weight_space"
    }
    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        self.inner.execute(driver, request).await
    }
}

struct Snapshot {
    request: Request,
    task: Option<String>,
    truncated: bool,
}

// The input request already exists. Only bounded text is copied into the scorer.
fn snapshot(request: &Request) -> Snapshot {
    let all = &request.llm_request.messages;
    let window_start = all.len().saturating_sub(MAX_MESSAGES);
    let latest_user = all[window_start..]
        .iter()
        .rposition(is_user_turn)
        .map(|i| i + window_start);
    let start = latest_user.unwrap_or(window_start);
    let mut remaining = MAX_BYTES;
    let mut truncated = window_start > 0 && latest_user.is_none();
    let mut messages = Vec::new();
    for message in all[start..].iter().rev() {
        let mut content = Vec::new();
        for block in message.content.iter().rev().take(MAX_BLOCKS) {
            if let Some(block) = bounded_block(block, &mut remaining, &mut truncated, 0) {
                content.push(block);
            }
        }
        truncated |= message.content.len() > MAX_BLOCKS;
        content.reverse();
        messages.push(Message {
            role: message.role,
            content,
        });
    }
    messages.reverse();
    let task = latest_user.map(|index| {
        let mut task = String::new();
        let mut task_budget = 8192;
        for block in all[index].content.iter().take(MAX_BLOCKS) {
            if let ContentBlock::Text { text } = block {
                task.push_str(&bounded_excerpt(text, &mut task_budget, &mut truncated));
            }
        }
        task
    });
    Snapshot {
        request: Request {
            llm_request: LlmRequest {
                messages,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        },
        task,
        truncated,
    }
}

fn bounded_text(text: &str, remaining: &mut usize, truncated: &mut bool) -> String {
    let mut end = text.len().min(*remaining).min(4096);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    *truncated |= end < text.len();
    *remaining -= end;
    text[..end].to_string()
}

fn bounded_excerpt(text: &str, remaining: &mut usize, truncated: &mut bool) -> String {
    let limit = (*remaining).min(4096);
    if text.len() <= limit || limit < 3 {
        return bounded_text(text, remaining, truncated);
    }
    let mut head = (limit - 1) / 2;
    let mut tail = text.len() - (limit - 1 - head);
    while !text.is_char_boundary(head) {
        head -= 1;
    }
    while !text.is_char_boundary(tail) {
        tail += 1;
    }
    let excerpt = format!("{}\n{}", &text[..head], &text[tail..]);
    *remaining -= excerpt.len();
    *truncated = true;
    excerpt
}

fn bounded_block(
    block: &ContentBlock,
    remaining: &mut usize,
    truncated: &mut bool,
    depth: usize,
) -> Option<ContentBlock> {
    if *remaining == 0 || depth > 1 {
        *truncated = true;
        return None;
    }
    match block {
        ContentBlock::Text { text } => Some(ContentBlock::Text {
            text: bounded_excerpt(text, remaining, truncated),
        }),
        ContentBlock::ToolCall(call) => {
            let parsed;
            let args = match &call.arguments {
                serde_json::Value::String(text) => {
                    let text = bounded_text(text, remaining, truncated);
                    parsed = serde_json::from_str(&text).unwrap_or_default();
                    &parsed
                }
                value => value,
            };
            let command = args
                .get("command")
                .or_else(|| args.get("cmd"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            Some(ContentBlock::ToolCall(ToolCall {
                arguments: serde_json::json!({"command": bounded_text(command, remaining, truncated)}),
                name: bounded_text(&call.name, remaining, truncated),
                id: bounded_text(&call.id, remaining, truncated),
            }))
        }
        ContentBlock::ToolResult(result) => {
            let content = result
                .content
                .iter()
                .take(MAX_BLOCKS)
                .filter_map(|b| bounded_block(b, remaining, truncated, depth + 1))
                .collect();
            *truncated |= result.content.len() > MAX_BLOCKS;
            Some(ContentBlock::ToolResult(ToolResult {
                content,
                tool_call_id: bounded_text(&result.tool_call_id, remaining, truncated),
                is_error: result.is_error,
            }))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::testing::{category_models, echo, test_drive_with_models};
    use switchyard_protocol::{Metadata, Role, text_request};

    fn weights() -> WeightSpaceWeights {
        let mut row = vec![0.0; 18];
        row[6] = 2.0;
        WeightSpaceWeights {
            schema_version: 1,
            input_weights: vec![row],
            hidden_bias: vec![0.0],
            output_weights: vec![vec![0.0], vec![1.0]],
            output_bias: vec![0.1, 0.0],
            cost_bias: vec![0.0, 1.0],
        }
    }

    fn request() -> Request {
        Request {
            llm_request: text_request(None, "Fix the implementation"),
            raw_request: None,
            metadata: None,
        }
    }

    fn classifier(cost: f64) -> Result<WeightSpaceClassifier> {
        WeightSpaceClassifier::new(vec!["efficient".into(), "capable".into()], weights(), cost)
    }

    #[test]
    fn validates_dimensions_and_nonfinite_weights() {
        for mutate in [
            |w: &mut WeightSpaceWeights| w.schema_version = 2,
            |w: &mut WeightSpaceWeights| w.input_weights[0].pop().map(|_| ()).unwrap_or(()),
            |w: &mut WeightSpaceWeights| w.output_bias[0] = f64::NAN,
            |w: &mut WeightSpaceWeights| w.cost_bias[0] = -1.0,
            |w: &mut WeightSpaceWeights| w.hidden_bias.clear(),
        ] {
            let mut w = weights();
            mutate(&mut w);
            assert!(w.validate(2).is_err());
        }
        assert!(classifier(f64::INFINITY).is_err());
        assert!(
            WeightSpaceClassifier::new(vec!["same".into(), "same".into()], weights(), 0.0).is_err()
        );
    }

    #[test]
    fn bounded_snapshot_preserves_recent_errors_without_copying_old_history() -> Result<()> {
        let mut req = request();
        req.llm_request
            .messages
            .extend((0..1000).map(|_| Message::text(Role::Assistant, "old")));
        req.llm_request.messages.push(Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "x".into(),
                is_error: Some(true),
                content: vec![ContentBlock::Text {
                    text: "fatal runtime error".into(),
                }],
            })],
        });
        let bounded = snapshot(&req);
        assert_eq!(bounded.request.llm_request.messages.len(), MAX_MESSAGES);
        assert!(bounded.truncated);
        let trace = classifier(0.0)?.trace(&State::default(), &req);
        assert_eq!(trace.inputs.len(), 18);
        assert_eq!(trace.selected, 1);
        assert!(trace.inputs.iter().all(|v| (0.0..=1.0).contains(v)));
        assert_eq!(classifier(10.0)?.trace(&State::default(), &req).selected, 0);
        Ok(())
    }

    #[test]
    fn utf8_and_nested_tool_payloads_stay_bounded() {
        let mut req = request();
        let huge = "🙂".repeat(100_000);
        req.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: huge.clone(),
                name: "exec_command".into(),
                arguments: serde_json::json!({"command": huge, "unused": huge}),
            })],
        });
        let bounded = snapshot(&req);
        assert!(bounded.truncated);
        let encoded = serde_json::to_string(&bounded.request.llm_request).expect("serialize");
        assert!(encoded.len() < 20_000);
    }

    #[test]
    fn bounded_excerpt_keeps_trailing_diagnostics() {
        let text = format!("{}\nfatal runtime error", "🙂".repeat(5000));
        let mut remaining = 4096;
        let mut truncated = false;
        let excerpt = bounded_excerpt(&text, &mut remaining, &mut truncated);
        assert!(excerpt.ends_with("fatal runtime error"));
        assert!(truncated);
        assert!(excerpt.len() <= 4096);
        assert_eq!(remaining + excerpt.len(), 4096);
    }

    #[test]
    fn new_task_clears_old_mutations_and_preserves_its_excerpt() -> Result<()> {
        let classifier = classifier(0.0)?;
        let mut state = State::default();
        let mut req = request();
        req.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "edit".into(),
                name: "apply_patch".into(),
                arguments: serde_json::json!({}),
            })],
        });
        assert_eq!(classifier.decide(&mut state, &req).inputs[14], 1.0);
        req.llm_request
            .messages
            .push(Message::text(Role::User, "Explain why this works"));
        let trace = classifier.decide(&mut state, &req);
        assert_eq!(trace.inputs[14], 0.0);
        assert_eq!(trace.inputs[2], 0.0);
        assert_eq!(trace.inputs[3], 1.0);
        req.llm_request
            .messages
            .extend((0..40).map(|_| Message::text(Role::Assistant, "working")));
        let later = classifier.decide(&mut state, &req);
        assert_eq!(later.inputs[3], 1.0);
        assert_eq!(later.inputs[14], 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn routes_with_no_judge_and_keeps_request_intact() -> Result<()> {
        let router = Arc::new(WeightSpaceRouter::new(
            vec!["efficient".into(), "capable".into()],
            weights(),
            0.0,
        )?);
        let (model, _) = test_drive_with_models(
            router,
            request(),
            category_models(Category::Any, &["efficient", "capable"]),
            echo(),
        )
        .await?;
        assert_eq!(model.as_str(), "efficient");
        Ok(())
    }

    #[test]
    fn previous_target_state_is_agent_scoped_and_resets_for_new_user_turn() -> Result<()> {
        let mut req = request();
        req.metadata = Some(Metadata {
            agent_id: Some("child".into()),
            ..Default::default()
        });
        let mut state = State::default();
        state.extra.insert(
            state_key(&req, "last_target"),
            StateValue::String("capable".into()),
        );
        assert_eq!(classifier(0.0)?.trace(&state, &req).inputs[17], 0.0);
        req.llm_request
            .messages
            .push(Message::text(Role::Assistant, "working"));
        assert_eq!(classifier(0.0)?.trace(&state, &req).inputs[17], 1.0);
        req.metadata.as_mut().expect("metadata").agent_id = Some("other".into());
        assert_eq!(classifier(0.0)?.trace(&state, &req).inputs[17], 0.0);
        Ok(())
    }
}
