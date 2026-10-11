// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional semantic privacy assessment through host-served judge calls.

use std::collections::BTreeMap;
use std::future::Future;
use std::io::{self, Write};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use switchyard_protocol::{
    Category, ChoiceOption, DecisionKind, DecisionQuestion, DecisionRequest, DecisionResponse,
    DecisionValue, FormatId, InstructionBlock, LlmRequest, LlmResponse, Message, ModelId,
    OutputParams, PreservationMetadata, ProviderExtensions, ReasoningParams, Request, Role, Tool,
    ToolChoice,
};

use crate::algorithms::util::classifier_contract::{ClassifierContract, ClassifierContractConfig};
use crate::algorithms::util::llm_judge::{JudgeRuntimeConfig, SerdeDecoder, StructuredJudge};
use crate::{Call, ClassifierResponseFormat, Driver, Result, RuntimeModels, Step};

const MAX_CONTEXT_BYTES: usize = 64 * 1024;
const QUESTION_ID: &str = "privacy";
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-6;
const DEFAULT_PROMPT: &str = include_str!("prompts/privacy-classifier/prompt.md");

/// A semantic assessment independent of lane selection and task retention.
pub struct PrivacyAssessment {
    reason_code: &'static str,
    clear_score: Option<f64>,
    has_sensitive_content: bool,
}

impl PrivacyAssessment {
    /// Returns a bounded category or failure reason, without request or provider text.
    pub const fn reason_code(&self) -> &'static str {
        self.reason_code
    }

    /// Returns the model's clear-content score when the verdict is valid.
    pub const fn clear_score(&self) -> Option<f64> {
        self.clear_score
    }

    /// Reports an affirmative sensitive category; false also includes failures and is not clearance.
    pub const fn has_sensitive_content(&self) -> bool {
        self.has_sensitive_content
    }

    fn failure(reason_code: &'static str) -> Self {
        Self {
            reason_code,
            clear_score: None,
            has_sensitive_content: false,
        }
    }
}

/// Runs one optional privacy judge call through the host's existing [`Call`] handler.
/// The caller applies thresholds, chooses permitted targets, and retains task restrictions.
/// Routing and externally supplied restrictions do not require this detector.
pub struct PrivacyPreflight {
    target: ModelId,
    models: Arc<RuntimeModels>,
    backend: Backend,
}

enum Backend {
    Decision(DecisionQuestion),
    Llm(StructuredJudge<(), SerdeDecoder<LlmVerdict>>),
}

impl PrivacyPreflight {
    /// Builds a typed decision question, optionally replacing its standing instructions.
    pub fn decision(target: ModelId, instructions: Option<&Value>) -> Self {
        let question = DecisionQuestion {
            instructions: instructions
                .cloned()
                .unwrap_or_else(|| Value::String(DEFAULT_PROMPT.trim().into())),
            kind: DecisionKind::Choice {
                options: PrivacyVerdict::ALL
                    .into_iter()
                    .map(|verdict| ChoiceOption {
                        id: verdict.as_str().into(),
                        description: Some(Value::String(verdict.description().into())),
                    })
                    .collect(),
            },
        };
        Self::new(target, Backend::Decision(question))
    }

    /// Builds a JSON judge using the shared classifier contract and typed decoder.
    /// Returns an error if the prompt or response contract is invalid.
    pub fn llm(
        target: ModelId,
        prompt: Option<&str>,
        format: ClassifierResponseFormat,
    ) -> Result<Self> {
        if prompt.is_some_and(|prompt| prompt.trim().is_empty()) {
            return Err(crate::LibsyError::AlgorithmError {
                message: "privacy classifier prompt must not be empty".into(),
            });
        }
        let prompt = prompt.map(str::to_owned).unwrap_or_else(|| {
            format!(
                "{}\n\n{}",
                DEFAULT_PROMPT.trim(),
                include_str!("prompts/privacy-classifier/llm.md").trim()
            )
        });
        let schema = verdict_schema();
        // Both modes describe the schema, including endpoints that do not enforce it.
        let prompt = match format {
            ClassifierResponseFormat::JsonSchema => format!(
                "{prompt}\n\nReturn exactly one JSON object matching this JSON Schema:\n{schema:#}"
            ),
            ClassifierResponseFormat::JsonObject => prompt,
        };
        let contract = ClassifierContract::from_config(
            &ClassifierContractConfig::default()
                .with_prompt(prompt)
                .with_response_format_type(format),
            DEFAULT_PROMPT,
            &json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "switchyard_privacy_verdict",
                    "strict": true,
                    "schema": schema,
                }
            })
            .to_string(),
        )?;
        let judge = StructuredJudge::new(
            (),
            contract,
            SerdeDecoder::new(),
            JudgeRuntimeConfig::new(512)?,
        );
        Ok(Self::new(target, Backend::Llm(judge)))
    }

    fn new(target: ModelId, backend: Backend) -> Self {
        let models = RuntimeModels::new([(Category::Judge, vec![target.clone()])].into())
            .with_target_restriction();
        Self {
            target,
            models: Arc::new(models),
            backend,
        }
    }

    /// Assesses complete normalized content with one host-served call.
    /// Failures return a bounded reason with no clear score. Dropping this future
    /// cancels assessment and host work; no background task is spawned.
    pub async fn assess<F, Fut>(&self, request: &Request, serve: F) -> PrivacyAssessment
    where
        F: FnOnce(Call) -> Fut + Send,
        Fut: Future<Output = Result<()>> + Send,
    {
        let context = match classifier_context(&request.llm_request) {
            Ok(context) => context,
            Err(()) => return PrivacyAssessment::failure("input_unavailable"),
        };
        let (driver, mut steps) = Driver::new("privacy_classifier", Arc::clone(&self.models));
        // Each assessment publishes one call; poll it with the verdict to preserve cancellation.
        let handled = async {
            let call = match steps.recv().await {
                Some(Ok(Step::CallModel(call))) => Call::Model(call),
                Some(Ok(Step::CallDecision(call))) => Call::Decision(call),
                _ => return Err("classifier_failed"),
            };
            serve(call).await.map_err(|_| "classifier_failed")
        };
        tokio::try_join!(self.verdict(context, &driver), handled)
            .map(|(assessment, ())| assessment)
            .unwrap_or_else(PrivacyAssessment::failure)
    }

    async fn verdict(
        &self,
        context: BoundedJson,
        driver: &Driver,
    ) -> std::result::Result<PrivacyAssessment, &'static str> {
        let (selected, clear_score) = match &self.backend {
            Backend::Decision(question) => {
                let request = DecisionRequest {
                    model: None,
                    context: context.into_value().map_err(|()| "input_unavailable")?,
                    questions: BTreeMap::from([(QUESTION_ID.into(), question.clone())]),
                };
                let response = driver
                    .call_decision(request, self.target.clone())
                    .await
                    .map_err(|_| "classifier_failed")?;
                decision_verdict(&response).ok_or("invalid_verdict")?
            }
            Backend::Llm(judge) => {
                let mut request = judge.build_request_with_messages(vec![Message::text(
                    Role::User,
                    context.into_string().map_err(|()| "input_unavailable")?,
                )]);
                request
                    .llm_request
                    .extensions
                    .fields
                    .insert("store".into(), Value::Bool(false));
                let response = driver
                    .call_model_with_error_recovery(request, vec![self.target.clone()], true)
                    .await
                    .map_err(|_| "classifier_failed")?;
                let LlmResponse::Agg(response) = response.llm_response else {
                    return Err("invalid_verdict");
                };
                let verdict = judge.decode(&response).map_err(|_| "invalid_verdict")?;
                (verdict.reason_code, verdict.clear_score)
            }
        };
        if !clear_score.is_finite() || !(0.0..=1.0).contains(&clear_score) {
            return Err("invalid_verdict");
        }
        Ok(PrivacyAssessment {
            reason_code: selected.as_str(),
            clear_score: Some(clear_score),
            has_sensitive_content: matches!(
                selected,
                PrivacyVerdict::PersonalData
                    | PrivacyVerdict::Credentials
                    | PrivacyVerdict::ConfidentialData
                    | PrivacyVerdict::RegulatedData
            ),
        })
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PrivacyVerdict {
    NoSensitiveContent,
    PersonalData,
    Credentials,
    ConfidentialData,
    RegulatedData,
    Uncertain,
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

    const fn as_str(self) -> &'static str {
        match self {
            Self::NoSensitiveContent => "no_sensitive_content",
            Self::PersonalData => "personal_data",
            Self::Credentials => "credentials",
            Self::ConfidentialData => "confidential_data",
            Self::RegulatedData => "regulated_data",
            Self::Uncertain => "uncertain",
        }
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmVerdict {
    clear_score: f64,
    reason_code: PrivacyVerdict,
}

fn verdict_schema() -> Value {
    json!({
        "type": "object", "additionalProperties": false,
        "properties": {
            "clear_score": {"type": "number", "minimum": 0, "maximum": 1},
            "reason_code": {"type": "string", "enum": PrivacyVerdict::ALL.map(PrivacyVerdict::as_str)}
        },
        "required": ["clear_score", "reason_code"]
    })
}

fn decision_verdict(response: &DecisionResponse) -> Option<(PrivacyVerdict, f64)> {
    let DecisionValue::Choice {
        selected,
        probabilities: Some(probabilities),
    } = &response.answers.get(QUESTION_ID)?.value
    else {
        return None;
    };
    let selected = PrivacyVerdict::deserialize(serde::de::value::StrDeserializer::<
        serde::de::value::Error,
    >::new(selected))
    .ok()?;
    let valid_distribution = probabilities.len() == PrivacyVerdict::ALL.len()
        && PrivacyVerdict::ALL.iter().all(|verdict| {
            probabilities
                .get(verdict.as_str())
                .is_some_and(|p| p.0.is_finite() && (0.0..=1.0).contains(&p.0))
        })
        && (probabilities.values().map(|p| p.0).sum::<f64>() - 1.0).abs()
            <= PROBABILITY_SUM_TOLERANCE;
    if !valid_distribution {
        return None;
    }
    Some((selected, probabilities.get("no_sensitive_content")?.0))
}

// Borrow content-bearing fields; leave transport identity and cached responses out of assessment.
#[derive(Serialize)]
struct PrivacyContext<'a> {
    instructions: &'a [InstructionBlock],
    messages: &'a [Message],
    tools: &'a [Tool],
    tool_choice: Option<&'a ToolChoice>,
    response_format: Option<&'a Value>,
    reasoning_effort: Option<&'a str>,
    reasoning_raw: Option<&'a Value>,
    extensions: &'a Map<String, Value>,
    preserved_requests: &'a BTreeMap<FormatId, Value>,
}

fn classifier_context(request: &LlmRequest) -> std::result::Result<BoundedJson, ()> {
    // Exhaustive destructuring makes new protocol fields require a coverage decision.
    let LlmRequest {
        model: _,
        instructions,
        messages,
        tools,
        tool_choice,
        sampling: _,
        output,
        reasoning,
        stream: _,
        extensions,
        preservation,
    } = request;
    let OutputParams {
        max_output_tokens: _,
        response_format,
        is_schema_enforced: _,
    } = output;
    let ReasoningParams { effort, raw } = reasoning;
    let ProviderExtensions { fields: extensions } = extensions;
    let PreservationMetadata {
        requests: preserved_requests,
        responses: _,
    } = preservation;
    if tools.iter().any(|tool| !matches!(tool, Tool::Function(_)))
        || has_remote_tools(extensions)
        || preserved_requests
            .values()
            .filter_map(Value::as_object)
            .any(has_remote_tools)
    {
        return Err(());
    }
    let context = PrivacyContext {
        instructions,
        messages,
        tools,
        tool_choice: tool_choice.as_ref(),
        response_format: response_format.as_ref(),
        reasoning_effort: effort.as_deref(),
        reasoning_raw: raw.as_ref(),
        extensions,
        preserved_requests,
    };
    let mut json = BoundedJson::default();
    serde_json::to_writer(&mut json, &context).map_err(|_| ())?;
    Ok(json)
}

// Remote tool definitions contain transport credentials, including in preserved wire bodies.
fn has_remote_tools(fields: &Map<String, Value>) -> bool {
    fields.contains_key("mcp_servers")
        || ["tools", "switchyard_codex_additional_tools"]
            .into_iter()
            .any(|key| has_remote_tool(fields.get(key), 0))
        || fields
            .get("input")
            .and_then(Value::as_array)
            .is_some_and(|items| {
                items.iter().any(|item| {
                    item.get("type").and_then(Value::as_str) == Some("additional_tools")
                        && has_remote_tool(item.get("tools"), 0)
                })
            })
}

fn has_remote_tool(tools: Option<&Value>, depth: usize) -> bool {
    tools.and_then(Value::as_array).is_some_and(|tools| {
        tools
            .iter()
            .any(|tool| match tool.get("type").and_then(Value::as_str) {
                Some("mcp" | "mcp_toolset") => true,
                Some("namespace") => depth >= 64 || has_remote_tool(tool.get("tools"), depth + 1),
                _ => false,
            })
    })
}

#[derive(Default)]
struct BoundedJson(Vec<u8>);

impl BoundedJson {
    fn into_value(self) -> std::result::Result<Value, ()> {
        serde_json::from_slice(&self.0).map_err(|_| ())
    }

    fn into_string(self) -> std::result::Result<String, ()> {
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use switchyard_protocol::{
        DecisionAnswer, LlmClientError, Metadata, Probability, Response, ToolDefinition, Usage,
        text_response,
    };
    use tokio::sync::Notify;

    use super::*;
    use crate::LibsyError;

    fn preflights() -> [PrivacyPreflight; 3] {
        [
            PrivacyPreflight::decision("private/classifier".into(), None),
            PrivacyPreflight::llm(
                "private/classifier".into(),
                None,
                ClassifierResponseFormat::JsonSchema,
            )
            .expect("schema judge"),
            PrivacyPreflight::llm(
                "private/classifier".into(),
                None,
                ClassifierResponseFormat::JsonObject,
            )
            .expect("object judge"),
        ]
    }

    async fn reply(call: Call, reason: &str, score: f64) -> Result<()> {
        let failed = || {
            LibsyError::client_call(
                "private/classifier",
                LlmClientError::Configuration {
                    message: "private provider diagnostic".into(),
                },
            )
        };
        match call {
            Call::Model(call) => {
                assert!(call.recover_errors);
                if reason == "failed" {
                    return call.respond(std::future::ready(Err(failed()))).await;
                }
                let content = if reason == "malformed" {
                    "not JSON".into()
                } else {
                    json!({"reason_code": reason, "clear_score": score}).to_string()
                };
                call.respond(std::future::ready(Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, content)),
                    metadata: None,
                    upstream_headers: Default::default(),
                })))
                .await
            }
            Call::Decision(call) => {
                if reason == "failed" {
                    return call.respond(std::future::ready(Err(failed()))).await;
                }
                call.respond(std::future::ready(Ok(decision_response(reason, score))))
                    .await
            }
        }
    }

    fn decision_response(reason: &str, score: f64) -> DecisionResponse {
        let remaining = if reason == "no_sensitive_content" {
            "uncertain"
        } else {
            reason
        };
        let probabilities = PrivacyVerdict::ALL
            .map(|verdict| {
                let probability = match verdict.as_str() {
                    "no_sensitive_content" => score,
                    value if value == remaining => 1.0 - score,
                    _ => 0.0,
                };
                (verdict.as_str().into(), Probability(probability))
            })
            .into();
        DecisionResponse {
            id: None,
            model: None,
            answers: BTreeMap::from([(
                QUESTION_ID.into(),
                DecisionAnswer {
                    value: DecisionValue::Choice {
                        selected: reason.into(),
                        probabilities: Some(probabilities),
                    },
                    provider_confidence: None,
                },
            )]),
            usage: Usage::default(),
        }
    }

    // Invalid verdicts and recovered client errors never manufacture clearance or task evidence.
    #[tokio::test]
    async fn returns_only_valid_evidence_across_backends() {
        for preflight in preflights() {
            for (reason, score, expected, sensitive) in [
                ("no_sensitive_content", 0.95, "no_sensitive_content", false),
                ("personal_data", 0.1, "personal_data", true),
                ("uncertain", 0.5, "uncertain", false),
                ("malformed", 0.0, "invalid_verdict", false),
                ("no_sensitive_content", 1.1, "invalid_verdict", false),
                ("failed", 0.0, "classifier_failed", false),
            ] {
                let assessment = preflight
                    .assess(&Request::default(), |call| async move {
                        reply(call, reason, score).await
                    })
                    .await;
                assert_eq!(assessment.reason_code(), expected);
                assert_eq!(assessment.has_sensitive_content(), sensitive);
                assert_eq!(
                    assessment.clear_score(),
                    (!matches!(expected, "invalid_verdict" | "classifier_failed")).then_some(score)
                );
            }
            let assessment = preflight
                .assess(&Request::default(), |call| async move {
                    reply(call, "no_sensitive_content", 0.95).await?;
                    Err(LibsyError::NoTargets)
                })
                .await;
            assert_eq!(assessment.reason_code(), "classifier_failed");
        }
        for invalid in ["missing", "incomplete", "unnormalized"] {
            let mut response = decision_response("no_sensitive_content", 0.95);
            let DecisionValue::Choice { probabilities, .. } = &mut response
                .answers
                .get_mut(QUESTION_ID)
                .expect("privacy answer")
                .value
            else {
                panic!("expected choice");
            };
            match invalid {
                "missing" => *probabilities = None,
                "incomplete" => {
                    probabilities
                        .as_mut()
                        .expect("distribution")
                        .remove("uncertain");
                }
                _ => {
                    probabilities
                        .as_mut()
                        .expect("distribution")
                        .insert("personal_data".into(), Probability(0.2));
                }
            }
            let assessment = PrivacyPreflight::decision("private/classifier".into(), None)
                .assess(&Request::default(), |call| async move {
                    let Call::Decision(call) = call else {
                        panic!("expected decision call");
                    };
                    call.respond(std::future::ready(Ok(response))).await
                })
                .await;
            assert_eq!(assessment.reason_code(), "invalid_verdict");
            assert_eq!(assessment.clear_score(), None);
            assert!(!assessment.has_sensitive_content());
        }
        let preflight = PrivacyPreflight::llm(
            "private/classifier".into(),
            None,
            ClassifierResponseFormat::JsonSchema,
        )
        .expect("judge");
        let assessment = preflight
            .assess(&Request::default(), |call| async move {
                let Call::Model(call) = call else {
                    panic!("expected LLM call");
                };
                call.respond(std::future::ready(Ok(Response {
                    llm_response: LlmResponse::Stream(Box::pin(futures::stream::empty())),
                    metadata: None,
                    upstream_headers: Default::default(),
                })))
                .await
            })
            .await;
        assert_eq!(assessment.reason_code(), "invalid_verdict");
    }

    // Scan content-bearing fields without forwarding the original transport identity or auth.
    #[tokio::test]
    async fn bounds_full_context_and_preserves_judge_contracts() {
        let mut request = Request {
            llm_request: LlmRequest {
                model: Some("public/answer".into()),
                instructions: vec![InstructionBlock {
                    role: Role::System,
                    content: Message::text(Role::System, "system content").content,
                }],
                messages: vec![Message::text(Role::User, "user content")],
                tools: vec![ToolDefinition {
                    name: "private_tool".into(),
                    description: Some("confidential tool content".into()),
                    parameters: json!({"type": "object", "properties": {"mcp_servers": {"type": "array"}}}),
                    strict: Some(true),
                }
                .into()],
                ..Default::default()
            },
            raw_request: Some(json!({"host_only": "original body"})),
            metadata: Some(Metadata {
                task_id: Some("host-only-task".into()),
                ..Default::default()
            }),
        };
        request
            .llm_request
            .extensions
            .fields
            .insert("provider_content".into(), json!("private"));
        request
            .llm_request
            .preservation
            .requests
            .insert("custom".into(), json!({"private": "preserved", "tools": [{"name": "lookup", "parameters": {"properties": {"mcp_servers": {"type": "mcp"}}}}]}));
        for preflight in preflights() {
            let assessment = preflight
                .assess(&request, |call| async move {
                    let context = match &call {
                        Call::Decision(call) => {
                            assert_eq!(call.model.as_str(), "private/classifier");
                            assert_eq!(call.request.questions.len(), 1);
                            call.request.context.clone()
                        }
                        Call::Model(call) => {
                            assert!(call.request.raw_request.is_none());
                            assert!(call.request.metadata.is_none());
                            assert_eq!(call.request.llm_request.output.max_output_tokens, Some(512));
                            assert_eq!(call.request.llm_request.extensions.fields["store"], false);
                            assert_eq!(call.request.model_id(), Some("private/classifier".into()));
                            let prompt = Message {
                                role: Role::System,
                                content: call.request.llm_request.instructions[0].content.clone(),
                            }.text_content("").expect("prompt");
                            let expected = format!(
                                "{}\n\n{}\n\nReturn exactly one JSON object matching this JSON Schema:\n{:#}",
                                DEFAULT_PROMPT.trim(),
                                include_str!("prompts/privacy-classifier/llm.md").trim(),
                                verdict_schema(),
                            );
                            assert_eq!(prompt, expected);
                            let format = call.request.llm_request.output.response_format.as_ref().expect("format");
                            if format["type"] == "json_schema" {
                                assert_eq!(format["json_schema"]["name"], "switchyard_privacy_verdict");
                                assert_eq!(format["json_schema"]["strict"], true);
                            } else {
                                assert_eq!(format, &json!({"type": "json_object"}));
                            }
                            serde_json::from_str(&call.request.llm_request.messages[0].text_content("").expect("context")).expect("JSON")
                        }
                    };
                    assert_eq!(context["tools"][0]["name"], "private_tool");
                    assert_eq!(context["extensions"]["provider_content"], "private");
                    assert_eq!(context["preserved_requests"]["custom"]["private"], "preserved");
                    let serialized = context.to_string();
                    assert!(serialized.contains("system content"));
                    assert!(serialized.contains("user content"));
                    assert!(!serialized.contains("host-only-task"));
                    assert!(!serialized.contains("host_only"));
                    reply(call, "no_sensitive_content", 0.95).await
                })
                .await;
            assert_eq!(assessment.reason_code(), "no_sensitive_content");
        }
        request.llm_request.messages =
            vec![Message::text(Role::User, "x".repeat(MAX_CONTEXT_BYTES))];
        let assessment = PrivacyPreflight::decision("private/classifier".into(), None)
            .assess(&request, |_| async {
                panic!("oversized input must not leave the host")
            })
            .await;
        assert_eq!(assessment.reason_code(), "input_unavailable");
        assert!(assessment.clear_score().is_none());
    }

    // Normalized and preserved remote-tool definitions must not leave the host for assessment.
    #[tokio::test]
    async fn remote_tool_credentials_never_reach_assessment()
    -> std::result::Result<(), serde_json::Error> {
        let remote = json!({"type": "mcp", "name": "inventory", "url": "https://example.invalid/mcp", "authorization_token": "synthetic-secret"});
        for input in [
            json!({"tools": [remote.clone()]}),
            json!({"tools": [{"type": "provider", "provider": "openai_responses", "raw": remote.clone()}]}),
            json!({"extensions": {"fields": {"mcp_servers": [{"url": "https://example.invalid/mcp", "authorization_token": "synthetic-secret"}]}}}),
            json!({"extensions": {"fields": {"switchyard_codex_additional_tools": [remote.clone()]}}}),
            json!({"preservation": {"requests": {"openai_responses": {"tools": [remote.clone()]}}}}),
            json!({"preservation": {"requests": {"openai_responses": {"input": [{"type": "additional_tools", "tools": [{"type": "namespace", "tools": [remote.clone()]}]}]}}}}),
        ] {
            let request = Request {
                llm_request: serde_json::from_value(input)?,
                ..Default::default()
            };
            for preflight in preflights() {
                let assessment = preflight
                    .assess(&request, |_| async {
                        panic!("remote tool credentials must not reach the classifier")
                    })
                    .await;
                assert_eq!(assessment.reason_code(), "input_unavailable");
                assert_eq!(assessment.clear_score(), None);
            }
        }
        Ok(())
    }

    // Dropping assessment must cancel the host future, not leave a classifier task running.
    #[tokio::test]
    async fn cancellation_drops_pending_host_work() {
        struct OnDrop(Arc<AtomicBool>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let started = Arc::new(Notify::new());
        let host_started = Arc::clone(&started);
        let host_dropped = Arc::clone(&dropped);
        let task = tokio::spawn(async move {
            PrivacyPreflight::decision("private/classifier".into(), None)
                .assess(&Request::default(), |call| async move {
                    let Call::Decision(call) = call else {
                        panic!("expected decision call");
                    };
                    call.respond(async move {
                        let _guard = OnDrop(host_dropped);
                        host_started.notify_one();
                        futures::future::pending().await
                    })
                    .await
                })
                .await
        });
        started.notified().await;
        task.abort();
        let Err(error) = task.await else {
            panic!("assessment was not cancelled");
        };
        assert!(error.is_cancelled());
        assert!(dropped.load(Ordering::SeqCst));
    }
}
