// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Semantic preflight for contextual privacy decisions.

use std::collections::BTreeMap;
use std::future::Future;
use std::io::{self, Write};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};
use strum_macros::{EnumString, IntoStaticStr};
use switchyard_protocol::{
    ChoiceOption, ContentBlock, DecisionKind, DecisionQuestion, DecisionRequest, DecisionResponse,
    DecisionValue, InstructionBlock, LlmRequest, LlmResponse, Message, ModelId, OutputParams,
    Request, Role, RoutedDecisionClient, RoutedLlmClient, Usage, completion_text,
};
use tracing::Instrument;

use super::{PrivacyDecision, PrivacyLane};

const MAX_CONTEXT_BYTES: usize = 64 * 1024;
const MAX_OUTPUT_TOKENS: u64 = 512;
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-6;
const QUESTION_ID: &str = "privacy";

pub(crate) struct SemanticPrivacyClassifier {
    backend: SemanticBackend,
    clear_threshold: f64,
}

enum SemanticBackend {
    Decision {
        target: ModelId,
        client: Arc<dyn RoutedDecisionClient>,
        question: Box<DecisionQuestion>,
    },
    Llm {
        target: ModelId,
        client: Arc<dyn RoutedLlmClient>,
    },
}

struct SemanticVerdict {
    selected: PrivacyVerdict,
    clear_score: f64,
}

// Keeps classifier spans complete when the assessment future is cancelled.
struct ClassifierCall {
    span: tracing::Span,
    is_finished: bool,
}

impl ClassifierCall {
    fn new(kind: &'static str, target: &ModelId, route: &str, algorithm: &str) -> Self {
        Self {
            span: tracing::info_span!(
                target: "switchyard_runner",
                "switchyard.privacy_classifier_call",
                switchyard.route = route,
                switchyard.algorithm = algorithm,
                privacy.classifier.kind = kind,
                selected_model = %target,
                openinference.span.kind = "CHAIN",
                outcome = tracing::field::Empty,
                error.type = tracing::field::Empty,
                input_tokens = tracing::field::Empty,
                output_tokens = tracing::field::Empty,
                total_tokens = tracing::field::Empty,
                reasoning_tokens = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
            ),
            is_finished: false,
        }
    }

    async fn observe<T, E>(
        mut self,
        future: impl Future<Output = Result<T, E>>,
        record_response: impl FnOnce(&tracing::Span, &T),
    ) -> Result<T, &'static str> {
        let span = self.span.clone();
        match future.instrument(span).await {
            Ok(response) => {
                record_response(&self.span, &response);
                self.finish("ok", None);
                Ok(response)
            }
            Err(_) => {
                self.finish("error", Some("classifier_failed"));
                Err("classifier_failed")
            }
        }
    }

    fn finish(&mut self, outcome: &'static str, error_type: Option<&'static str>) {
        self.span.record("outcome", outcome);
        if let Some(error_type) = error_type {
            self.span.record("error.type", error_type);
        }
        self.is_finished = true;
    }
}

fn record_response(span: &tracing::Span, id: Option<&str>, model: Option<&str>, usage: &Usage) {
    if let Some(id) = id {
        span.record("gen_ai.response.id", id);
    }
    if let Some(model) = model {
        span.record("gen_ai.response.model", model);
    }
    for (field, value) in [
        ("input_tokens", usage.input_tokens),
        ("output_tokens", usage.output_tokens),
        ("total_tokens", usage.total_tokens),
        ("reasoning_tokens", usage.reasoning_tokens),
    ] {
        if let Some(value) = value {
            span.record(field, value);
        }
    }
}

impl Drop for ClassifierCall {
    fn drop(&mut self) {
        if !self.is_finished {
            self.finish("cancelled", Some("cancelled"));
        }
    }
}

impl SemanticPrivacyClassifier {
    pub(crate) fn decision(
        target: ModelId,
        client: Arc<dyn RoutedDecisionClient>,
        clear_threshold: f64,
    ) -> Self {
        Self {
            backend: SemanticBackend::Decision {
                target,
                client,
                question: Box::new(privacy_question()),
            },
            clear_threshold,
        }
    }

    pub(crate) fn llm(
        target: ModelId,
        client: Arc<dyn RoutedLlmClient>,
        clear_threshold: f64,
    ) -> Self {
        Self {
            backend: SemanticBackend::Llm { target, client },
            clear_threshold,
        }
    }

    pub(crate) async fn assess(
        &self,
        request: &Request,
        route: &str,
        algorithm: &str,
    ) -> PrivacyDecision {
        match self.verdict(request, route, algorithm).await {
            Ok(verdict) => self.apply(verdict),
            Err(reason) => self.restricted(reason, None),
        }
    }

    async fn verdict(
        &self,
        request: &Request,
        route: &str,
        algorithm: &str,
    ) -> Result<SemanticVerdict, &'static str> {
        let context = semantic_context(request).map_err(|()| "input_unavailable")?;
        match &self.backend {
            SemanticBackend::Decision {
                target,
                client,
                question,
            } => {
                let request = decision_request(
                    target,
                    question,
                    context.into_value().map_err(|()| "input_unavailable")?,
                );
                let response = ClassifierCall::new("decision", target, route, algorithm)
                    .observe(client.call(request), |span, response| {
                        record_response(
                            span,
                            response.id.as_deref(),
                            response.model.as_ref().map(ModelId::as_str),
                            &response.usage,
                        );
                    })
                    .await?;
                decision_verdict(&response).ok_or("invalid_verdict")
            }
            SemanticBackend::Llm { target, client } => {
                let request = llm_request(
                    target,
                    context.into_string().map_err(|()| "input_unavailable")?,
                );
                let response = ClassifierCall::new("llm", target, route, algorithm)
                    .observe(client.call(request), |span, response| {
                        if let LlmResponse::Agg(response) = &response.llm_response {
                            record_response(
                                span,
                                response.id.as_deref(),
                                response.model.as_deref(),
                                &response.usage,
                            );
                        }
                    })
                    .await?;
                let LlmResponse::Agg(response) = response.llm_response else {
                    return Err("invalid_verdict");
                };
                let verdict: LlmVerdict = serde_json::from_str(completion_text(&response).trim())
                    .map_err(|_| "invalid_verdict")?;
                SemanticVerdict::new(verdict.reason_code, verdict.clear_score)
                    .ok_or("invalid_verdict")
            }
        }
    }

    fn apply(&self, verdict: SemanticVerdict) -> PrivacyDecision {
        let SemanticVerdict {
            selected,
            clear_score,
        } = verdict;
        if selected == PrivacyVerdict::NoSensitiveContent && clear_score >= self.clear_threshold {
            return PrivacyDecision::semantic(
                PrivacyLane::Standard,
                selected.as_str(),
                Some(clear_score),
                self.clear_threshold,
            );
        }
        let reason = if selected == PrivacyVerdict::NoSensitiveContent {
            "below_clear_threshold"
        } else {
            selected.as_str()
        };
        self.restricted(reason, Some(clear_score))
    }

    fn restricted(&self, reason_code: &'static str, clear_score: Option<f64>) -> PrivacyDecision {
        PrivacyDecision::semantic(
            PrivacyLane::Restricted,
            reason_code,
            clear_score,
            self.clear_threshold,
        )
    }
}

impl SemanticVerdict {
    fn new(selected: PrivacyVerdict, clear_score: f64) -> Option<Self> {
        (clear_score.is_finite() && (0.0..=1.0).contains(&clear_score)).then_some(Self {
            selected,
            clear_score,
        })
    }
}

fn decision_verdict(response: &DecisionResponse) -> Option<SemanticVerdict> {
    let Some(DecisionValue::Choice {
        selected,
        probabilities: Some(probabilities),
    }) = response
        .answers
        .get(QUESTION_ID)
        .map(|answer| &answer.value)
    else {
        return None;
    };
    let Ok(verdict) = selected.parse::<PrivacyVerdict>() else {
        return None;
    };
    let valid_distribution = probabilities.len() == PrivacyVerdict::ALL.len()
        && PrivacyVerdict::ALL.iter().all(|verdict| {
            probabilities
                .get(verdict.as_str())
                .is_some_and(|probability| {
                    probability.0.is_finite() && (0.0..=1.0).contains(&probability.0)
                })
        })
        && (probabilities
            .values()
            .map(|probability| probability.0)
            .sum::<f64>()
            - 1.0)
            .abs()
            <= PROBABILITY_SUM_TOLERANCE;
    if !valid_distribution {
        return None;
    }
    let clear_score = probabilities
        .get(PrivacyVerdict::NoSensitiveContent.as_str())
        .map(|probability| probability.0);
    SemanticVerdict::new(verdict, clear_score?)
}

fn semantic_context(request: &Request) -> Result<BoundedJson, ()> {
    let mut json = BoundedJson::default();
    serde_json::to_writer(&mut json, &request.llm_request).map_err(|_| ())?;
    Ok(json)
}

#[derive(Default)]
struct BoundedJson(Vec<u8>);

impl BoundedJson {
    fn into_value(self) -> Result<Value, ()> {
        serde_json::from_slice(&self.0).map_err(|_| ())
    }

    fn into_string(self) -> Result<String, ()> {
        String::from_utf8(self.0).map_err(|_| ())
    }
}

impl Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_CONTEXT_BYTES.saturating_sub(self.0.len()) {
            return Err(io::Error::other("privacy classifier context exceeds limit"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Deserialize, EnumString, Eq, IntoStaticStr, PartialEq)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
enum PrivacyVerdict {
    NoSensitiveContent,
    PersonalData,
    Credentials,
    ConfidentialData,
    RegulatedData,
    Uncertain,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmVerdict {
    clear_score: f64,
    reason_code: PrivacyVerdict,
}

impl PrivacyVerdict {
    const ALL: [Self; 6] = [
        Self::NoSensitiveContent,
        Self::PersonalData,
        Self::Credentials,
        Self::ConfidentialData,
        Self::RegulatedData,
        Self::Uncertain,
    ];

    fn as_str(self) -> &'static str {
        self.into()
    }

    const fn description(self) -> &'static str {
        match self {
            Self::NoSensitiveContent => "No personal, secret, confidential, or regulated content.",
            Self::PersonalData => "Personal or identifying data is present.",
            Self::Credentials => "Credentials, secrets, or authentication material are present.",
            Self::ConfidentialData => "Confidential organizational data is present.",
            Self::RegulatedData => "Regulated data is present.",
            Self::Uncertain => "The privacy boundary cannot be determined confidently.",
        }
    }
}

fn privacy_question() -> DecisionQuestion {
    let options = PrivacyVerdict::ALL
        .into_iter()
        .map(|verdict| ChoiceOption {
            id: verdict.as_str().into(),
            description: Some(Value::String(verdict.description().into())),
        })
        .collect();
    DecisionQuestion {
        instructions: Value::String(include_str!("semantic_prompt.txt").trim().into()),
        kind: DecisionKind::Choice { options },
    }
}

fn decision_request(
    target: &ModelId,
    question: &DecisionQuestion,
    context: Value,
) -> DecisionRequest {
    DecisionRequest {
        model: Some(target.clone()),
        context,
        questions: BTreeMap::from([(QUESTION_ID.into(), question.clone())]),
    }
}

fn llm_request(target: &ModelId, context: String) -> Request {
    let mut llm_request = LlmRequest {
        model: Some(target.to_string()),
        instructions: vec![InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: format!(
                    "{}\n\n{}",
                    include_str!("semantic_prompt.txt").trim(),
                    include_str!("semantic_llm_prompt.txt").trim()
                ),
            }],
        }],
        messages: vec![Message::text(Role::User, context)],
        output: OutputParams {
            max_output_tokens: Some(MAX_OUTPUT_TOKENS),
            response_format: Some(response_format()),
            ..OutputParams::default()
        },
        ..LlmRequest::default()
    };
    llm_request
        .extensions
        .fields
        .insert("store".into(), Value::Bool(false));
    Request {
        llm_request,
        raw_request: None,
        metadata: None,
    }
}

fn response_format() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "switchyard_privacy_verdict",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "clear_score": { "type": "number", "minimum": 0, "maximum": 1 },
                    "reason_code": {
                        "type": "string",
                        "enum": PrivacyVerdict::ALL.map(PrivacyVerdict::as_str)
                    }
                },
                "required": ["clear_score", "reason_code"]
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::future::{Future, pending};
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use serde_json::json;
    use switchyard_protocol::{
        ContentBlock, DecisionAnswer, InstructionBlock, LlmClientError, Message, Probability,
        Response, Role, ToolCall, ToolDefinition, ToolResult, Usage, text_response,
    };
    use tokio::sync::Notify;
    use tracing_subscriber::fmt::format::FmtSpan;

    use super::*;

    static CLASSIFIER_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    struct ReplyClient(Mutex<Option<Result<DecisionResponse, LlmClientError>>>);

    #[async_trait]
    impl RoutedDecisionClient for ReplyClient {
        async fn call(
            &self,
            _request: DecisionRequest,
        ) -> Result<DecisionResponse, LlmClientError> {
            self.0
                .lock()
                .expect("reply lock")
                .take()
                .expect("one classifier call")
        }
    }

    struct LlmReplyClient(Mutex<Option<Result<Response, LlmClientError>>>);

    #[async_trait]
    impl RoutedLlmClient for LlmReplyClient {
        async fn call(&self, _request: Request) -> Result<Response, LlmClientError> {
            self.0
                .lock()
                .expect("reply lock")
                .take()
                .expect("one classifier call")
        }
    }

    struct PendingClient(Arc<Notify>);

    #[async_trait]
    impl RoutedDecisionClient for PendingClient {
        async fn call(
            &self,
            _request: DecisionRequest,
        ) -> Result<DecisionResponse, LlmClientError> {
            self.0.notify_one();
            pending().await
        }
    }

    #[derive(Clone, Default)]
    struct LogWriter(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for LogWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("log lock").extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn capture_logs<F: Future>(future: F) -> (F::Output, String) {
        let writer = LogWriter::default();
        let logs = Arc::clone(&writer.0);
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .with_span_events(FmtSpan::CLOSE)
            .with_writer(move || writer.clone())
            .finish();
        let dispatch = tracing::Dispatch::new(subscriber);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let output = tracing::dispatcher::with_default(&dispatch, || runtime.block_on(future));
        let logs = logs.lock().expect("log lock").clone();
        (output, String::from_utf8(logs).expect("utf-8 logs"))
    }

    fn response(selected: &str, clear_score: Option<f64>) -> DecisionResponse {
        DecisionResponse {
            id: None,
            model: None,
            answers: BTreeMap::from([(
                QUESTION_ID.into(),
                DecisionAnswer {
                    value: DecisionValue::Choice {
                        selected: selected.into(),
                        probabilities: clear_score.map(|score| {
                            let mut probabilities = PrivacyVerdict::ALL
                                .into_iter()
                                .map(|verdict| (verdict.as_str().into(), Probability(0.0)))
                                .collect::<BTreeMap<_, _>>();
                            probabilities.insert("no_sensitive_content".into(), Probability(score));
                            if selected != "no_sensitive_content" {
                                probabilities.insert(selected.into(), Probability(1.0 - score));
                            } else {
                                probabilities.insert("uncertain".into(), Probability(1.0 - score));
                            }
                            probabilities
                        }),
                    },
                    provider_confidence: None,
                },
            )]),
            usage: Usage::default(),
        }
    }

    fn probabilities(response: &mut DecisionResponse) -> &mut BTreeMap<String, Probability> {
        let DecisionValue::Choice {
            probabilities: Some(probabilities),
            ..
        } = &mut response.answers.get_mut(QUESTION_ID).expect("answer").value
        else {
            panic!("choice answer")
        };
        probabilities
    }

    fn classifier(reply: Result<DecisionResponse, LlmClientError>) -> SemanticPrivacyClassifier {
        SemanticPrivacyClassifier::decision(
            "privacy/model".into(),
            Arc::new(ReplyClient(Mutex::new(Some(reply)))),
            0.9,
        )
    }

    fn llm_classifier(reply: Result<Response, LlmClientError>) -> SemanticPrivacyClassifier {
        SemanticPrivacyClassifier::llm(
            "privacy/model".into(),
            Arc::new(LlmReplyClient(Mutex::new(Some(reply)))),
            0.9,
        )
    }

    fn llm_response(text: &str) -> Response {
        Response {
            llm_response: LlmResponse::Agg(text_response(None, text)),
            metadata: None,
            upstream_headers: Default::default(),
        }
    }

    #[test]
    fn context_is_bounded_and_covers_outbound_request_state() {
        let mut request = Request::default();
        request.llm_request.messages.push(Message::text(
            switchyard_protocol::Role::User,
            "normalized_marker",
        ));
        request.llm_request.instructions.push(InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Reasoning {
                text: "reasoning_marker".into(),
                signature: Some("signature_marker".into()),
                details: vec![json!({"provider_detail_marker": true})],
            }],
        });
        request.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call_marker".into(),
                name: "tool_call_marker".into(),
                arguments: json!({"argument_marker": true}),
            })],
        });
        request.llm_request.messages.push(Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "call_marker".into(),
                content: vec![ContentBlock::Text {
                    text: "result_marker".into(),
                }],
                is_error: Some(false),
            })],
        });
        request.llm_request.tools.push(ToolDefinition {
            name: "tool_definition_marker".into(),
            description: None,
            parameters: json!({}),
            strict: None,
        });
        request
            .llm_request
            .preservation
            .requests
            .insert("openai_chat".into(), json!({"preserved_marker": true}));
        request.raw_request = Some(json!({"raw_marker": true}));
        request
            .metadata
            .get_or_insert_default()
            .extra_metadata
            .get_or_insert_default()
            .insert("metadata_marker".into(), String::new());

        let encoded = semantic_context(&request)
            .expect("bounded context")
            .into_string()
            .expect("json is utf-8");
        assert!(encoded.contains("normalized_marker"));
        assert!(encoded.contains("reasoning_marker"));
        assert!(encoded.contains("tool_call_marker"));
        assert!(encoded.contains("argument_marker"));
        assert!(encoded.contains("result_marker"));
        assert!(encoded.contains("signature_marker"));
        assert!(encoded.contains("provider_detail_marker"));
        assert!(encoded.contains("tool_definition_marker"));
        assert!(encoded.contains("preserved_marker"));
        assert!(!encoded.contains("raw_marker"));
        assert!(!encoded.contains("metadata_marker"));

        let decision_request = decision_request(
            &"privacy/model".into(),
            &privacy_question(),
            serde_json::from_str(&encoded).expect("context value"),
        );
        assert_eq!(decision_request.model.as_deref(), Some("privacy/model"));
        assert!(decision_request.questions.contains_key(QUESTION_ID));
        assert!(
            decision_request
                .context
                .to_string()
                .contains("normalized_marker")
        );

        let classifier_request = llm_request(&"privacy/model".into(), encoded);
        let llm = &classifier_request.llm_request;
        assert_eq!(llm.model.as_deref(), Some("privacy/model"));
        assert_eq!(
            llm.extensions.fields.get("store"),
            Some(&Value::Bool(false))
        );
        assert!(llm.output.response_format.is_some());
        assert!(llm.instructions[0].content.iter().any(
            |block| matches!(block, ContentBlock::Text { text } if text.contains("clear_score"))
        ));
        assert!(!llm.stream);
        assert!(classifier_request.raw_request.is_none());
        assert!(classifier_request.metadata.is_none());

        request.llm_request.messages = vec![Message::text(
            switchyard_protocol::Role::User,
            "x".repeat(MAX_CONTEXT_BYTES),
        )];
        assert!(semantic_context(&request).is_err());
    }

    #[tokio::test]
    async fn typed_verdicts_and_failures_fail_closed() {
        let _guard = CLASSIFIER_TEST_LOCK.lock().await;
        let failure = LlmClientError::Configuration {
            message: "unavailable".into(),
        };
        let mut incomplete = response("no_sensitive_content", Some(0.95));
        probabilities(&mut incomplete).remove("uncertain");
        let mut unnormalized = response("no_sensitive_content", Some(0.95));
        probabilities(&mut unnormalized).insert("personal_data".into(), Probability(0.2));
        for (reply, lane, reason, score) in [
            (
                Ok(response("no_sensitive_content", Some(0.95))),
                PrivacyLane::Standard,
                "no_sensitive_content",
                Some(0.95),
            ),
            (
                Ok(response("no_sensitive_content", Some(0.8))),
                PrivacyLane::Restricted,
                "below_clear_threshold",
                Some(0.8),
            ),
            (
                Ok(response("confidential_data", Some(0.1))),
                PrivacyLane::Restricted,
                "confidential_data",
                Some(0.1),
            ),
            (
                Ok(response("no_sensitive_content", None)),
                PrivacyLane::Restricted,
                "invalid_verdict",
                None,
            ),
            (
                Ok(incomplete),
                PrivacyLane::Restricted,
                "invalid_verdict",
                None,
            ),
            (
                Ok(unnormalized),
                PrivacyLane::Restricted,
                "invalid_verdict",
                None,
            ),
            (
                Err(failure),
                PrivacyLane::Restricted,
                "classifier_failed",
                None,
            ),
        ] {
            let decision = classifier(reply)
                .assess(&Request::default(), "test/route", "test_algorithm")
                .await;
            assert_eq!(decision.lane.as_str(), lane.as_str());
            assert_eq!(decision.reason_code, reason);
            assert_eq!(decision.clear_score, score);
        }
    }

    #[tokio::test]
    async fn llm_verdicts_and_failures_fail_closed() {
        let _guard = CLASSIFIER_TEST_LOCK.lock().await;
        let failure = LlmClientError::Configuration {
            message: "unavailable".into(),
        };
        for (reply, lane, reason, score) in [
            (
                Ok(llm_response(
                    r#"{"clear_score":0.95,"reason_code":"no_sensitive_content"}"#,
                )),
                PrivacyLane::Standard,
                "no_sensitive_content",
                Some(0.95),
            ),
            (
                Ok(llm_response(
                    r#"{"clear_score":1.1,"reason_code":"no_sensitive_content"}"#,
                )),
                PrivacyLane::Restricted,
                "invalid_verdict",
                None,
            ),
            (
                Ok(llm_response("not json")),
                PrivacyLane::Restricted,
                "invalid_verdict",
                None,
            ),
            (
                Err(failure),
                PrivacyLane::Restricted,
                "classifier_failed",
                None,
            ),
        ] {
            let decision = llm_classifier(reply)
                .assess(&Request::default(), "test/route", "test_algorithm")
                .await;
            assert_eq!(decision.lane.as_str(), lane.as_str());
            assert_eq!(decision.reason_code, reason);
            assert_eq!(decision.clear_score, score);
        }
    }

    #[test]
    fn classifier_call_spans_record_terminal_state() {
        let _guard = CLASSIFIER_TEST_LOCK.blocking_lock();
        let mut reply = response("no_sensitive_content", Some(0.95));
        reply.id = Some("decision-123".into());
        reply.model = Some("privacy/provider-model".into());
        reply.usage.input_tokens = Some(17);
        let started = Arc::new(Notify::new());
        let pending = SemanticPrivacyClassifier::decision(
            "privacy/model".into(),
            Arc::new(PendingClient(Arc::clone(&started))),
            0.9,
        );
        let secret = "provider body containing private data";

        let ((success, failure), logs) = capture_logs(async move {
            let success = classifier(Ok(reply))
                .assess(&Request::default(), "test/route", "stage")
                .await;
            let failure = llm_classifier(Err(LlmClientError::Configuration {
                message: secret.into(),
            }))
            .assess(&Request::default(), "test/route", "stage")
            .await;
            let task = tokio::spawn(async move {
                pending
                    .assess(&Request::default(), "test/route", "stage")
                    .await
            });
            started.notified().await;
            task.abort();
            assert!(matches!(task.await, Err(error) if error.is_cancelled()));
            (success, failure)
        });

        assert!(matches!(success.lane, PrivacyLane::Standard));
        assert_eq!(failure.reason_code, "classifier_failed");
        assert!(!logs.contains(secret), "{logs}");
        assert_eq!(
            logs.matches("switchyard.privacy_classifier_call").count(),
            3,
            "{logs}"
        );
        for expected in [
            "test/route",
            "stage",
            "privacy/model",
            "decision-123",
            "privacy/provider-model",
            "input_tokens=17",
            "outcome=\"ok\"",
            "outcome=\"error\"",
            "outcome=\"cancelled\"",
            "error.type=\"classifier_failed\"",
            "error.type=\"cancelled\"",
        ] {
            assert!(logs.contains(expected), "missing {expected:?} in {logs}");
        }
    }
}
