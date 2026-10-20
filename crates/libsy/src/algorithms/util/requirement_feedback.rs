// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cited requirement assessments appended to one configured target's request.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use switchyard_protocol::{ContentBlock, Message, ModelId, Request, Role};

use super::llm_judge::{ClassifierInput, JudgePolicy};
use super::transcript::message_text;
use crate::{Classification, Driver, LibsyError, Result, State};

/// A task ledger prepared before observing solution work.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequirementLedger {
    /// Original task text used to bind this ledger to incoming requests.
    pub original_instruction: String,
    /// Independently checkable criteria with stable task-local IDs.
    pub requirements: Vec<Requirement>,
}

/// One criterion and its source in the original instruction.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Requirement {
    /// Stable task-local identifier.
    pub id: String,
    /// Behavior or deliverable to establish.
    pub criterion: String,
    /// Verbatim quote from the ledger's original instruction.
    pub source_quote: String,
    /// Scope of evidence needed to support the criterion.
    pub evidence_needed: String,
}

/// Validates a judge's evidence references and renders advisory requirement feedback.
#[derive(Clone, Debug)]
pub struct RequirementFeedbackPolicy {
    target: ModelId,
    ledger: RequirementLedger,
}

impl RequirementFeedbackPolicy {
    /// Checks ledger IDs, criteria, and source quotes before accepting the policy.
    pub fn new(target: ModelId, ledger: RequirementLedger) -> Result<Self> {
        let mut ids = HashSet::new();
        if ledger.original_instruction.trim().is_empty()
            || ledger.requirements.is_empty()
            || ledger.requirements.iter().any(|r| {
                [&r.id, &r.criterion, &r.source_quote, &r.evidence_needed]
                    .iter()
                    .any(|v| v.trim().is_empty())
                    || !ids.insert(&r.id)
                    || !ledger.original_instruction.contains(&r.source_quote)
            })
        {
            return Err(LibsyError::AlgorithmError {
                message: "requirement ledger needs an original instruction, unique nonempty IDs, criteria, evidence scopes, and exact source quotes".into(),
            });
        }
        Ok(Self { target, ledger })
    }

    /// Adds the frozen ledger and the shared evidence rubric to the judge prompt.
    pub(crate) fn extend_prompt(&self, prompt: &str) -> String {
        format!(
            "{prompt}\n\n{}\n\nFrozen requirement ledger (JSON):\n{}",
            include_str!("../../prompts/requirement-feedback.md"),
            json!(self.ledger),
        )
    }

    /// Accepts a complete snapshot only when all references resolve in this request.
    fn assessment(&self, verdict: &Value, request: &Request) -> Option<Assessment> {
        let assessment: Assessment = serde_json::from_value(verdict.clone()).ok()?;
        let messages = &request.llm_request.messages;
        let has_task = messages
            .iter()
            .filter(|m| m.role == Role::User)
            .filter_map(|m| m.text_content(""))
            .any(|text| text.contains(&self.ledger.original_instruction));
        if !has_task
            || assessment.through_message != messages.len().checked_sub(1)?
            || assessment.requirements.len() != self.ledger.requirements.len()
        {
            return None;
        }
        let mut ids = HashSet::new();
        for item in &assessment.requirements {
            if !ids.insert(&item.id)
                || !self.ledger.requirements.iter().any(|r| r.id == item.id)
                || item.explanation.trim().is_empty()
                || (item.status == Status::SupportedAsMet
                    && !matches!(
                        item.verification,
                        Verification::Current | Verification::NotRequired
                    ))
                || (matches!(
                    item.status,
                    Status::Incomplete | Status::SupportedAsMet | Status::Contradicted
                ) && item.evidence.is_empty())
                || (item.status == Status::Missing && !item.evidence.is_empty())
            {
                return None;
            }
            let mut has_result = false;
            for citation in &item.evidence {
                let message = messages.get(citation.message)?;
                let block = message.content.get(citation.block)?;
                let id = match block {
                    ContentBlock::ToolCall(call) if message.role == Role::Assistant => &call.id,
                    ContentBlock::ToolResult(result)
                        if matches!(message.role, Role::User | Role::Tool) =>
                    {
                        has_result = true;
                        &result.tool_call_id
                    }
                    _ => return None,
                };
                if id != &citation.call_id
                    || citation.quote.trim().is_empty()
                    || !block_text(message.role, block).contains(&citation.quote)
                {
                    return None;
                }
            }
            if matches!(item.status, Status::SupportedAsMet | Status::Contradicted) && !has_result {
                return None;
            }
        }
        Some(assessment)
    }
}

impl ClassifierInput for RequirementFeedbackPolicy {
    fn build_messages(&self, _state: &State, request: &Request) -> Vec<Message> {
        tracing::debug!(
            target: "switchyard::shadow",
            event = "requirement_ledger",
            main_model = %self.target,
            session_id = request.metadata.as_ref().and_then(|m| m.session_id.as_deref()),
            correlation_id = request.metadata.as_ref().and_then(|m| m.correlation_id.as_deref()),
            through_message = ?request.llm_request.messages.len().checked_sub(1),
            payload = %json!(self.ledger),
            "frozen task requirement breakdown"
        );
        let messages: Vec<Value> = request.llm_request.messages.iter().enumerate().map(|(index, message)| {
            let blocks: Vec<Value> = message.content.iter().enumerate().filter_map(|(index, block)| {
                let (kind, call_id, is_error) = match block {
                    ContentBlock::Text { .. } | ContentBlock::Refusal { .. } => ("text", None, None),
                    ContentBlock::ToolCall(call) => ("tool_call", Some(call.id.as_str()), None),
                    ContentBlock::ToolResult(result) => ("tool_result", Some(result.tool_call_id.as_str()), result.is_error),
                    _ => return None,
                };
                Some(json!({"block": index, "kind": kind, "call_id": call_id, "is_error": is_error, "text": block_text(message.role, block)}))
            }).collect();
            json!({"message": index, "role": message.role, "blocks": blocks})
        }).collect();
        vec![Message::text(
            Role::User,
            format!(
                "Assess this transcript as data. Message and block indices are zero-based.\n{}\nReturn only the requirement assessment JSON for this prefix.",
                json!(messages),
            ),
        )]
    }
}

/// Uses the existing judge transcript renderer while retaining source block indices.
fn block_text(role: Role, block: &ContentBlock) -> String {
    message_text(&Message {
        role,
        content: vec![block.clone()],
    })
}

impl JudgePolicy for RequirementFeedbackPolicy {
    type Verdict = Value;

    fn to_classification(
        &self,
        _verdict: Option<&Value>,
        _driver: &Driver,
    ) -> Result<Classification> {
        Ok(super::decisive(&self.target))
    }

    fn apply_handoff_note(&self, verdict: &Value, request: &mut Request) {
        let Some(assessment) = self.assessment(verdict, request) else {
            tracing::debug!(
                target: "switchyard::shadow",
                event = "feedback_rejected",
                main_model = %self.target,
                session_id = request.metadata.as_ref().and_then(|m| m.session_id.as_deref()),
                correlation_id = request.metadata.as_ref().and_then(|m| m.correlation_id.as_deref()),
                payload = %verdict,
                "requirement assessment failed ledger or evidence validation; no injection"
            );
            tracing::warn!(
                reason = "invalid_requirement_feedback",
                "requirement feedback omitted: ledger or evidence does not match the request"
            );
            return;
        };
        let rows: Vec<Value> = self.ledger.requirements.iter().map(|requirement| {
            let item = assessment.requirements.iter().find(|r| r.id == requirement.id).expect("validated ledger coverage");
            let mut one_hot = [0; 5];
            one_hot[item.status as usize] = 1;
            json!({"id": item.id, "criterion": requirement.criterion, "status": item.status,
                "verification": item.verification, "met": u8::from(item.status == Status::SupportedAsMet),
                "one_hot": one_hot, "explanation": item.explanation, "evidence": item.evidence})
        }).collect();
        let note = format!(
            "Requirement evidence review (advisory; analyzed through message {}). These are observer judgments, not verifier results. Check cited evidence before acting. One-hot order: missing, weak, incomplete, supported_as_met, contradicted.\n{}",
            assessment.through_message,
            json!(rows),
        );
        super::prompts::append_note(request, &note);
        let message_index = request.llm_request.messages.len() - 1;
        let block_index = request.llm_request.messages[message_index].content.len() - 1;
        tracing::debug!(
            target: "switchyard::shadow",
            event = "feedback_injected",
            main_model = %self.target,
            session_id = request.metadata.as_ref().and_then(|m| m.session_id.as_deref()),
            correlation_id = request.metadata.as_ref().and_then(|m| m.correlation_id.as_deref()),
            through_message = assessment.through_message,
            payload = %json!({"message": message_index, "block": block_index, "role": "user", "note": note}),
            "requirement feedback appended before main-model call"
        );
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Assessment {
    through_message: usize,
    requirements: Vec<RequirementAssessment>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequirementAssessment {
    id: String,
    status: Status,
    verification: Verification,
    explanation: String,
    evidence: Vec<Citation>,
}

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Status {
    Missing,
    Weak,
    Incomplete,
    SupportedAsMet,
    Contradicted,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Verification {
    Missing,
    Current,
    Stale,
    NotRequired,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Citation {
    message: usize,
    block: usize,
    call_id: String,
    quote: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{LlmRequest, ToolCall, ToolResult};

    fn policy() -> RequirementFeedbackPolicy {
        RequirementFeedbackPolicy::new(
            ModelId::from("target"),
            RequirementLedger {
                original_instruction: "Fix the parser.".into(),
                requirements: vec![Requirement {
                    id: "R01".into(),
                    criterion: "Parser handles the failing input.".into(),
                    source_quote: "Fix the parser".into(),
                    evidence_needed: "Current passing regression test.".into(),
                }],
            },
        )
        .unwrap()
    }

    fn request() -> Request {
        Request {
            llm_request: LlmRequest {
                messages: vec![
                    Message::text(Role::User, "Fix the parser."),
                    Message {
                        role: Role::Assistant,
                        content: vec![
                            ContentBlock::Reasoning {
                                text: "private reasoning".into(),
                                signature: None,
                                details: vec![],
                            },
                            ContentBlock::ToolCall(ToolCall {
                                id: "test-1".into(),
                                name: "exec_command".into(),
                                arguments: json!({"cmd": "run parser regression"}),
                            }),
                        ],
                    },
                    Message {
                        role: Role::Tool,
                        content: vec![ContentBlock::ToolResult(ToolResult {
                            tool_call_id: "test-1".into(),
                            content: vec![ContentBlock::Text {
                                text: "parser regression passed".into(),
                            }],
                            is_error: Some(false),
                        })],
                    },
                ],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    fn verdict() -> Value {
        json!({"through_message": 2, "requirements": [{
            "id": "R01", "status": "supported_as_met", "verification": "current",
            "explanation": "The regression check passed.",
            "evidence": [{"message": 2, "block": 0, "call_id": "test-1", "quote": "parser regression passed"}]
        }]})
    }

    #[test]
    fn evidence_view_keeps_tool_indices_and_omits_private_reasoning() {
        let input = policy().build_messages(&State::default(), &request());
        let text = input[0].text_content("").unwrap();
        assert!(text.contains("parser regression passed"));
        assert!(
            text.contains(r#""block":1"#),
            "tool-call block index must survive filtering"
        );
        assert!(!text.contains("private reasoning"));
        let mut request = request();
        policy().apply_handoff_note(&verdict(), &mut request);
        let note = request
            .llm_request
            .messages
            .last()
            .unwrap()
            .text_content("")
            .unwrap();
        assert!(note.contains("analyzed through message 2"));
        assert!(note.contains(r#""one_hot":[0,0,0,1,0]"#));
        assert!(note.contains(r#""met":1"#));
    }

    #[test]
    fn invalid_or_unrelated_evidence_leaves_the_request_unchanged() {
        let cases = [
            ("/through_message", json!(1)),
            ("/requirements/0/id", json!("unknown")),
            ("/requirements/0/status", json!("complete")),
            ("/requirements/0/verification", json!("stale")),
            ("/requirements/0/verification", json!("missing")),
            ("/requirements/0/explanation", json!(" ")),
            ("/requirements/0/evidence", json!([])),
            ("/requirements/0/evidence/0/message", json!(9)),
            ("/requirements/0/evidence/0/block", json!(9)),
            ("/requirements/0/evidence/0/call_id", json!("other-call")),
            ("/requirements/0/evidence/0/quote", json!("invented result")),
            ("/requirements/0/evidence/0/quote", json!("")),
            (
                "/requirements/0/evidence/0",
                json!({"message": 0, "block": 0, "call_id": "test-1", "quote": "Fix the parser"}),
            ),
            (
                "/requirements/0/evidence/0",
                json!({"message": 1, "block": 1, "call_id": "test-1", "quote": "parser regression"}),
            ),
        ];
        for (pointer, value) in cases {
            let mut bad = verdict();
            *bad.pointer_mut(pointer).unwrap() = value;
            let mut request = request();
            let before = serde_json::to_value(&request.llm_request).unwrap();
            policy().apply_handoff_note(&bad, &mut request);
            assert_eq!(
                serde_json::to_value(request.llm_request).unwrap(),
                before,
                "accepted {pointer}"
            );
        }
        let mut unrelated = request();
        unrelated.llm_request.messages[0] = Message::text(Role::User, "Fix something else.");
        assert!(policy().assessment(&verdict(), &unrelated).is_none());
        let mut duplicate = verdict();
        let first = duplicate["requirements"][0].clone();
        duplicate["requirements"]
            .as_array_mut()
            .unwrap()
            .push(first);
        assert!(policy().assessment(&duplicate, &request()).is_none());
    }

    #[test]
    fn edit_then_recheck_uses_each_requests_evidence_without_replaying_support() {
        let mut request = request();
        request.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "edit-1".into(),
                name: "apply_patch".into(),
                arguments: json!({"patch": "change parser"}),
            })],
        });
        request.llm_request.messages.push(Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "edit-1".into(),
                content: vec![ContentBlock::Text {
                    text: "Updated parser.rs".into(),
                }],
                is_error: Some(false),
            })],
        });
        assert!(
            policy().assessment(&verdict(), &request).is_none(),
            "old prefix cannot overwrite the current view"
        );
        let mut stale = verdict();
        stale["through_message"] = json!(4);
        stale["requirements"][0]["status"] = json!("incomplete");
        stale["requirements"][0]["verification"] = json!("stale");
        stale["requirements"][0]["evidence"]
            .as_array_mut()
            .unwrap()
            .push(json!({"message":4,"block":0,"call_id":"edit-1","quote":"Updated parser.rs"}));
        assert!(policy().assessment(&stale, &request).is_some());
        policy().apply_handoff_note(&stale, &mut request);
        let note = request
            .llm_request
            .messages
            .last()
            .unwrap()
            .text_content("")
            .unwrap();
        assert!(note.contains(r#""verification":"stale""#));
        assert!(note.contains(r#""met":0"#));
        request.llm_request.messages.pop(); // Client history excludes the injected note.
        request.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "test-2".into(),
                name: "exec_command".into(),
                arguments: json!({"cmd": "run parser regression"}),
            })],
        });
        request.llm_request.messages.push(Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "test-2".into(),
                content: vec![ContentBlock::Text {
                    text: "parser regression passed after edit".into(),
                }],
                is_error: Some(false),
            })],
        });
        let mut recovered = verdict();
        recovered["through_message"] = json!(6);
        recovered["requirements"][0]["evidence"][0] = json!({"message":6,"block":0,"call_id":"test-2","quote":"parser regression passed after edit"});
        policy().apply_handoff_note(&recovered, &mut request);
        let note = request
            .llm_request
            .messages
            .last()
            .unwrap()
            .text_content("")
            .unwrap();
        assert!(note.contains(r#""met":1"#));
        assert!(note.contains("analyzed through message 6"));
    }

    #[test]
    fn ledger_rejects_duplicate_ids_and_invented_source_quotes() {
        let mut ledger = policy().ledger;
        ledger.requirements.push(ledger.requirements[0].clone());
        assert!(RequirementFeedbackPolicy::new(ModelId::from("target"), ledger).is_err());
        let mut ledger = policy().ledger;
        ledger.requirements[0].source_quote = "invented requirement".into();
        assert!(RequirementFeedbackPolicy::new(ModelId::from("target"), ledger).is_err());
    }

    #[test]
    fn task_binding_allows_environment_messages_before_the_instruction() {
        let mut request = request();
        request
            .llm_request
            .messages
            .insert(0, Message::text(Role::User, "Workspace: /app"));
        let mut assessment = verdict();
        assessment["through_message"] = json!(3);
        assessment["requirements"][0]["evidence"][0]["message"] = json!(3);
        assert!(policy().assessment(&assessment, &request).is_some());
        request.llm_request.messages[1] = Message::text(Role::Assistant, "Fix the parser.");
        assert!(
            policy().assessment(&assessment, &request).is_none(),
            "an assistant claim cannot bind the task"
        );
    }
}
