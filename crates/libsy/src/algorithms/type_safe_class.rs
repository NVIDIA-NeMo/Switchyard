// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TypeSafe-backed (Jev "System One Model") routing.
//!
//! A non-generative classifier reached through the [`TypeSafeProvider`] port
//! rather than `driver.call_model`: unlike [`crate::LlmTaskClassifier`], the
//! judgment step here is never one of the deployment's own chat-completion
//! targets, because TypeSafe's API does not speak any of the wire formats
//! `libsy-llm-client` supports. `libsy` stays I/O-free by depending only on the
//! [`TypeSafeProvider`] trait; the runner constructs and injects the concrete,
//! HTTP-performing implementation.
//!
//! Structured the same way as [`crate::LlmTaskClassifier`]'s custom mode — an
//! affinity-gated `FallThrough` cascade terminated by [`DefaultCategoryClassifier`]
//! — so this mode fails open identically and slots into a route the same way
//! `llm_classifier` does.

use std::sync::Arc;

use async_trait::async_trait;
use switchyard_protocol::{Category, ContentBlock, Message, Request, Response, Role};

use super::fall_through::FallThrough;
use super::llm_class::{DefaultCategoryClassifier, affinity_router, task_messages, trim_messages};
use super::util::affinity::ClassifyTrigger;
use super::util::typesafe_provider::{
    TypeSafeClassifierInput, TypeSafeOption, TypeSafeProvider, TypeSafeVerdict,
};
use crate::core::algorithm::{Algorithm, Driver};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::state::State;
use crate::{LibsyError, Result};

/// Telemetry label for this algorithm's spans, metrics, and logs.
const ALGORITHM_NAME: &str = "type_safe_task_classifier";

/// Instruction sent as the provider's question when the deployment leaves
/// [`TypeSafeClassifierConfig::question`] empty.
const DEFAULT_QUESTION: &str = "Which configured model is the best fit for this conversation?";

/// Settings for a [`TypeSafeTaskClassifier`] route.
#[derive(Clone, Debug)]
pub struct TypeSafeClassifierConfig {
    /// Labeled options offered to the provider, e.g. one per routing tier. Each
    /// `label` should parse as a [`Category`] with at least one model configured
    /// for this route.
    pub options: Vec<TypeSafeOption>,
    /// Instruction sent as the provider's question. Falls back to a generic
    /// tier-selection prompt when empty.
    pub question: String,
    /// Lowest confidence that is trusted; below it (or on any provider failure,
    /// or an unresolved label) the request falls through to `default_target`.
    pub base_threshold: f64,
    /// Category served when the provider fails, returns an unrecognised label,
    /// or answers below `base_threshold`.
    pub default_target: Category,
    /// How often the classifier re-decides this session's target.
    pub classify_trigger: ClassifyTrigger,
    /// Uses the first user message as the SessionKey for sticky routing when
    /// session metadata is unavailable. Requires `classify_trigger = NewSession`.
    pub message_hash_fallback: bool,
    /// Trailing conversation turns the provider sees on top of the opening task.
    /// `None` (the default) sends the opening task and latest user follow-up only.
    pub recent_turn_window: Option<usize>,
}

impl Default for TypeSafeClassifierConfig {
    fn default() -> Self {
        Self {
            options: Vec::new(),
            question: String::new(),
            base_threshold: 0.0,
            default_target: Category::Efficient,
            classify_trigger: ClassifyTrigger::default(),
            message_hash_fallback: false,
            recent_turn_window: None,
        }
    }
}

impl TypeSafeClassifierConfig {
    fn validate(&self) -> Result<()> {
        if self.options.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "type_safe_classifier needs at least one option".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.base_threshold) {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "base_threshold must be between 0 and 1, got {}",
                    self.base_threshold
                ),
            });
        }
        if self.message_hash_fallback && self.classify_trigger != ClassifyTrigger::NewSession {
            return Err(LibsyError::AlgorithmError {
                message: "message_hash_fallback requires classify_trigger = new_session"
                    .to_string(),
            });
        }
        Ok(())
    }

    fn question(&self) -> &str {
        if self.question.trim().is_empty() {
            DEFAULT_QUESTION
        } else {
            &self.question
        }
    }
}

/// Flattens windowed task messages into plain text for a provider's `state`/context
/// field. TypeSafe's System One Model classifies free text, not a chat transcript, so
/// each message becomes one labeled line; non-text content (tool calls and results,
/// media, reasoning) is summarized by kind rather than silently dropped, so the
/// provider can still tell that something happened there even though it never sees
/// provider-private or binary payloads.
fn render_context(messages: &[Message]) -> String {
    messages
        .iter()
        .map(render_message)
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_message(message: &Message) -> String {
    let role = role_label(message.role);
    let body = message
        .content
        .iter()
        .map(render_block)
        .collect::<Vec<_>>()
        .join(" ");
    format!("[{role}] {body}")
}

fn role_label(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn render_block(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Text { text } => text.clone(),
        ContentBlock::Reasoning { .. } => "(reasoning omitted)".to_string(),
        ContentBlock::Image { .. } => "(image)".to_string(),
        ContentBlock::Audio { .. } => "(audio)".to_string(),
        ContentBlock::Video { .. } => "(video)".to_string(),
        ContentBlock::File { .. } => "(file)".to_string(),
        ContentBlock::ToolCall(call) => format!("(called tool {})", call.name),
        ContentBlock::ToolResult(result) => format!(
            "(tool result{})",
            if result.is_error == Some(true) {
                ", failed"
            } else {
                ""
            }
        ),
        ContentBlock::Refusal { text } => text.clone(),
        ContentBlock::Unknown { .. } => "(unrecognized content)".to_string(),
    }
}

/// Raw provider-backed scorer.
///
/// Always resolves to a `Classification` — a provider error, an unparseable or
/// unconfigured label, and a low-confidence verdict all become
/// [`Classification::Ambiguous`] so the cascade's [`DefaultCategoryClassifier`]
/// fallback can take over. Nothing here ever returns `Err`, matching
/// [`super::util::llm_judge::JudgeClassifier`]'s fail-open contract.
struct TypeSafeClassifier {
    provider: Arc<dyn TypeSafeProvider>,
    config: TypeSafeClassifierConfig,
}

impl TypeSafeClassifier {
    fn fall_open(driver: &Driver, reason_code: &str, extra: serde_json::Value) -> Classification {
        let mut evidence = serde_json::json!({
            "source": "fail_open",
            "reason_code": reason_code,
        });
        if let (Some(evidence), Some(extra)) = (evidence.as_object_mut(), extra.as_object()) {
            evidence.extend(extra.clone());
        }
        driver.set_evidence_if_empty(evidence);
        Classification::Ambiguous(vec![])
    }

    fn to_classification(&self, verdict: &TypeSafeVerdict, driver: &Driver) -> Classification {
        // `TypeSafeVerdict::confidence` is documented as "not validated at this
        // layer" — this is the caller that promise refers to. A NaN or
        // out-of-range value must not silently pass the `< base_threshold`
        // check below (NaN compares false against everything, so it would
        // otherwise be treated as fully confident).
        if !(0.0..=1.0).contains(&verdict.confidence) {
            tracing::warn!(
                algorithm = ALGORITHM_NAME,
                label = %verdict.label,
                confidence = verdict.confidence,
                "type safe classifier returned an out-of-range confidence; falling through"
            );
            return Self::fall_open(
                driver,
                "invalid_confidence",
                serde_json::json!({
                    "label": verdict.label,
                    "confidence": verdict.confidence,
                    "probabilities": verdict.probabilities,
                    "decision_latency_ms": verdict.decision_latency_ms,
                }),
            );
        }

        let resolved = verdict.label.parse::<Category>().ok().and_then(|category| {
            let target = driver.models_for(&category).first()?.clone();
            Some((category, target))
        });

        let Some((category, target)) = resolved else {
            tracing::warn!(
                algorithm = ALGORITHM_NAME,
                label = %verdict.label,
                "type safe classifier returned an unconfigured label; falling through"
            );
            return Self::fall_open(
                driver,
                "unresolved_label",
                serde_json::json!({
                    "label": verdict.label,
                    "probabilities": verdict.probabilities,
                    "decision_latency_ms": verdict.decision_latency_ms,
                }),
            );
        };

        if verdict.confidence < self.config.base_threshold {
            return Self::fall_open(
                driver,
                "low_confidence",
                serde_json::json!({
                    "label": verdict.label,
                    "confidence": verdict.confidence,
                    "probabilities": verdict.probabilities,
                    "decision_latency_ms": verdict.decision_latency_ms,
                }),
            );
        }

        driver.set_evidence_if_empty(serde_json::json!({
            "source": "type_safe_classifier",
            "label": verdict.label,
            "confidence": verdict.confidence,
            "probabilities": verdict.probabilities,
            "decision_latency_ms": verdict.decision_latency_ms,
        }));
        Classification::Scores(vec![Score {
            target,
            confidence: verdict.confidence,
            category: Some(category),
        }])
    }
}

#[async_trait]
impl Classifier<State> for TypeSafeClassifier {
    async fn score(
        &self,
        _state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let messages = match self.config.recent_turn_window {
            Some(window) => trim_messages(&request.llm_request.messages, window),
            None => task_messages(&request.llm_request.messages),
        };
        let input = TypeSafeClassifierInput {
            question: self.config.question().to_string(),
            context: render_context(&messages),
        };

        let verdict = match self.provider.classify(input, &self.config.options).await {
            Ok(verdict) => verdict,
            Err(error) => {
                tracing::warn!(
                    algorithm = ALGORITHM_NAME,
                    %error,
                    "type safe classifier request failed; falling through"
                );
                let classification =
                    Self::fall_open(driver, "provider_error", serde_json::Value::Null);
                return Ok((classification, None));
            }
        };

        Ok((self.to_classification(&verdict, driver), None))
    }
}

/// Routes requests through a TypeSafe (Jev "System One Model") classifier.
pub struct TypeSafeTaskClassifier {
    route: FallThrough<State>,
    /// Classifier used when this router is embedded in another cascade.
    inner: Arc<dyn Classifier<State>>,
}

impl TypeSafeTaskClassifier {
    /// Builds the classifier described by `config`, scoring through `provider`.
    ///
    /// # Errors
    ///
    /// Returns an error when `config` has no options, an out-of-range
    /// `base_threshold`, or `message_hash_fallback` without a matching trigger.
    pub fn new(
        provider: Arc<dyn TypeSafeProvider>,
        config: TypeSafeClassifierConfig,
    ) -> Result<Self> {
        config.validate()?;
        let default_target = config.default_target.clone();
        let classify_trigger = config.classify_trigger;
        let message_hash_fallback = config.message_hash_fallback;

        let scorer: Arc<dyn Classifier<State>> = Arc::new(TypeSafeClassifier { provider, config });

        // Affinity comes first so a retained assignment short-circuits the provider call.
        let mut route = FallThrough::<State>::new_with_state().with_name(ALGORITHM_NAME);
        if let Some(affinity) = affinity_router(classify_trigger, message_hash_fallback).as_ref() {
            route = route
                .with_processor(affinity.clone())
                .with_classifier(affinity.clone());
        }
        let fallback = DefaultCategoryClassifier(default_target);
        let route = route
            .with_classifier(scorer.clone())
            .with_classifier(Arc::new(fallback));

        Ok(Self {
            route,
            inner: scorer,
        })
    }
}

#[async_trait]
impl Classifier<State> for TypeSafeTaskClassifier {
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        self.inner.score(state, request, driver).await
    }
}

#[async_trait]
impl Algorithm for TypeSafeTaskClassifier {
    fn name(&self) -> &str {
        ALGORITHM_NAME
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        request: Request,
    ) -> Result<crate::RoutingOutcome> {
        self.route.execute(driver, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithm::RuntimeModels;
    use crate::core::testing::empty_driver;
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use switchyard_protocol::{ModelId, text_request};

    use super::super::util::typesafe_provider::TypeSafeProviderError;

    type ProviderResult = std::result::Result<TypeSafeVerdict, TypeSafeProviderError>;

    /// A provider whose response is scripted per call.
    struct StubProvider {
        responses: Mutex<Vec<ProviderResult>>,
    }

    impl StubProvider {
        fn once(response: ProviderResult) -> Self {
            Self {
                responses: Mutex::new(vec![response]),
            }
        }
    }

    #[async_trait]
    impl TypeSafeProvider for StubProvider {
        async fn classify(
            &self,
            _input: TypeSafeClassifierInput,
            _options: &[TypeSafeOption],
        ) -> ProviderResult {
            self.responses
                .lock()
                .pop()
                .expect("StubProvider called more times than scripted")
        }
    }

    /// Builds a test `Driver` whose runtime models group every `target` under its
    /// paired `category`, preserving `targets` order within each group. Pass the
    /// same category more than once to give it several targets.
    fn driver_with(category_targets: &[(Category, &str)]) -> Driver {
        let mut map: std::collections::HashMap<Category, Vec<ModelId>> =
            std::collections::HashMap::new();
        for (category, target) in category_targets {
            map.entry(category.clone())
                .or_default()
                .push(ModelId::from(*target));
        }
        Driver::new("test", Arc::new(RuntimeModels::new(map))).0
    }

    fn config(options: Vec<TypeSafeOption>, base_threshold: f64) -> TypeSafeClassifierConfig {
        TypeSafeClassifierConfig {
            options,
            base_threshold,
            default_target: Category::Efficient,
            ..Default::default()
        }
    }

    fn options() -> Vec<TypeSafeOption> {
        vec![
            TypeSafeOption::new("capable", "complex, multi-step work"),
            TypeSafeOption::new("efficient", "short, simple requests"),
        ]
    }

    fn verdict(label: &str, confidence: f64) -> TypeSafeVerdict {
        TypeSafeVerdict {
            label: label.to_string(),
            confidence,
            probabilities: [(label.to_string(), 1.0)].into(),
            decision_latency_ms: 12,
        }
    }

    #[tokio::test]
    async fn a_confident_verdict_routes_to_its_category() -> Result<()> {
        let provider = Arc::new(StubProvider::once(Ok(verdict("capable", 0.9))));
        let classifier = TypeSafeTaskClassifier::new(provider, config(options(), 0.5))?;
        let driver = driver_with(&[(Category::Capable, "strong"), (Category::Efficient, "weak")]);
        let mut state = State::default();
        let mut request = Request {
            llm_request: text_request(None, "hello"),
            raw_request: None,
            metadata: None,
        };

        let (classification, _) = classifier.score(&mut state, &mut request, &driver).await?;
        assert_eq!(
            classification.argmax(false)?.map(|score| score.target),
            Some(ModelId::from("strong"))
        );
        Ok(())
    }

    /// `TypeSafeTaskClassifier::score` (used when embedded inside another
    /// cascade) exposes only the raw provider-backed scorer, matching
    /// `LlmTaskClassifier`'s contract — its own affinity/default fallback only
    /// runs when the classifier is driven end to end as an [`Algorithm`], via
    /// [`Algorithm::route`]. These three tests exercise that full path.
    async fn routed_target(
        classifier: TypeSafeTaskClassifier,
        driver: Driver,
    ) -> Result<Option<ModelId>> {
        let request = Request {
            llm_request: text_request(None, "hello"),
            raw_request: None,
            metadata: None,
        };
        let outcome = Arc::new(classifier).route(driver, request).await?;
        Ok(outcome.selected_model_ids.first().cloned())
    }

    #[tokio::test]
    async fn a_low_confidence_verdict_falls_through_to_the_default() -> Result<()> {
        let provider = Arc::new(StubProvider::once(Ok(verdict("capable", 0.2))));
        let classifier = TypeSafeTaskClassifier::new(provider, config(options(), 0.5))?;
        let driver = driver_with(&[
            (Category::Any, "strong"),
            (Category::Any, "weak"),
            (Category::Capable, "strong"),
            (Category::Efficient, "weak"),
        ]);

        assert_eq!(
            routed_target(classifier, driver).await?,
            Some(ModelId::from("weak"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_unresolved_label_falls_through_to_the_default() -> Result<()> {
        let provider = Arc::new(StubProvider::once(Ok(verdict(
            "not_a_configured_category",
            0.99,
        ))));
        let classifier = TypeSafeTaskClassifier::new(provider, config(options(), 0.5))?;
        let driver = driver_with(&[
            (Category::Any, "strong"),
            (Category::Any, "weak"),
            (Category::Capable, "strong"),
            (Category::Efficient, "weak"),
        ]);

        assert_eq!(
            routed_target(classifier, driver).await?,
            Some(ModelId::from("weak"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_provider_error_falls_through_to_the_default() -> Result<()> {
        let provider = Arc::new(StubProvider::once(Err(TypeSafeProviderError(
            "network down".to_string(),
        ))));
        let classifier = TypeSafeTaskClassifier::new(provider, config(options(), 0.5))?;
        let driver = driver_with(&[
            (Category::Any, "strong"),
            (Category::Any, "weak"),
            (Category::Capable, "strong"),
            (Category::Efficient, "weak"),
        ]);

        assert_eq!(
            routed_target(classifier, driver).await?,
            Some(ModelId::from("weak"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_nan_confidence_falls_through_to_the_default() -> Result<()> {
        let provider = Arc::new(StubProvider::once(Ok(verdict("capable", f64::NAN))));
        let classifier = TypeSafeTaskClassifier::new(provider, config(options(), 0.5))?;
        let driver = driver_with(&[
            (Category::Any, "strong"),
            (Category::Any, "weak"),
            (Category::Capable, "strong"),
            (Category::Efficient, "weak"),
        ]);

        // Without an explicit range check, `NaN < base_threshold` is `false`, so a
        // naive comparison would treat this as confident enough to route on.
        assert_eq!(
            routed_target(classifier, driver).await?,
            Some(ModelId::from("weak"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_out_of_range_confidence_falls_through_to_the_default() -> Result<()> {
        let provider = Arc::new(StubProvider::once(Ok(verdict("capable", 1.5))));
        let classifier = TypeSafeTaskClassifier::new(provider, config(options(), 0.5))?;
        let driver = driver_with(&[
            (Category::Any, "strong"),
            (Category::Any, "weak"),
            (Category::Capable, "strong"),
            (Category::Efficient, "weak"),
        ]);

        assert_eq!(
            routed_target(classifier, driver).await?,
            Some(ModelId::from("weak"))
        );
        Ok(())
    }

    #[test]
    fn empty_options_are_rejected() {
        let provider = Arc::new(StubProvider::once(Ok(verdict("capable", 1.0))));
        let result = TypeSafeTaskClassifier::new(provider, config(Vec::new(), 0.5));
        assert!(matches!(
            result,
            Err(LibsyError::AlgorithmError { message })
                if message.contains("at least one option")
        ));
    }

    #[test]
    fn out_of_range_threshold_is_rejected() {
        let provider = Arc::new(StubProvider::once(Ok(verdict("capable", 1.0))));
        let result = TypeSafeTaskClassifier::new(provider, config(options(), 1.5));
        assert!(matches!(
            result,
            Err(LibsyError::AlgorithmError { message })
                if message.contains("base_threshold")
        ));
    }

    #[test]
    fn render_context_summarizes_non_text_content() {
        use switchyard_protocol::ToolCall;

        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "checking logs".to_string(),
                },
                ContentBlock::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    name: "grep".to_string(),
                    arguments: serde_json::json!({}),
                }),
            ],
        }];
        let rendered = render_context(&messages);
        assert!(rendered.contains("[assistant]"));
        assert!(rendered.contains("checking logs"));
        assert!(rendered.contains("(called tool grep)"));
    }

    #[test]
    fn empty_driver_smoke() {
        // Exercises the same test-only constructor `llm_class.rs` relies on, to
        // confirm it stays reachable from this sibling module.
        let _ = empty_driver();
    }
}
