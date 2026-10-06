// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, deterministic request inspection.

use regex::RegexSet;
use serde::Deserialize;
use strum_macros::IntoStaticStr;
use switchyard_protocol::{ContentBlock, LlmRequest, Request, ToolChoice, WireFormat};

const MAX_SCAN_BYTES: usize = 1024 * 1024;
const MAX_SCAN_DEPTH: usize = 64;
const MAX_SCAN_STEPS: usize = 4096;
const STRUCTURAL_SEPARATOR: &str = "\n|\n";

#[derive(Clone, Copy, Debug, Deserialize, Eq, IntoStaticStr, Ord, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub(crate) enum DeterministicDetector {
    Email,
    Phone,
    ApiKey,
    IpAddress,
    Ipv6,
    Url,
    Uuid,
    BearerToken,
    Jwt,
    CreditCard,
    AwsAccessKeyId,
    AwsSecretAccessKey,
    GcpApiKey,
    AzureStorageAccountKey,
    NvidiaApiKey,
}

impl DeterministicDetector {
    fn as_str(self) -> &'static str {
        self.into()
    }

    const fn pattern(self) -> &'static str {
        match self {
            Self::Email => r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            Self::Phone => r"\+?[0-9][0-9()\-\s]{6,}[0-9]",
            Self::ApiKey => r"\b(?:sk|rk|pk|ak)-[A-Za-z0-9_-]{8,}",
            Self::IpAddress => r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            Self::Ipv6 => {
                r"(?:([A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}|([A-Fa-f0-9]{1,4}:){1,7}:|([A-Fa-f0-9]{1,4}:){1,6}:[A-Fa-f0-9]{1,4}|([A-Fa-f0-9]{1,4}:){1,5}(?::[A-Fa-f0-9]{1,4}){1,2}|([A-Fa-f0-9]{1,4}:){1,4}(?::[A-Fa-f0-9]{1,4}){1,3}|([A-Fa-f0-9]{1,4}:){1,3}(?::[A-Fa-f0-9]{1,4}){1,4}|([A-Fa-f0-9]{1,4}:){1,2}(?::[A-Fa-f0-9]{1,4}){1,5}|[A-Fa-f0-9]{1,4}:(?:(?::[A-Fa-f0-9]{1,4}){1,6})|:(?:(?::[A-Fa-f0-9]{1,4}){1,7}|:))"
            }
            Self::Url => r"https?://[^\s]+",
            Self::Uuid => {
                r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
            }
            Self::BearerToken => r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-]{12,}={0,2}\b",
            Self::Jwt => r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
            Self::CreditCard => r"\b(?:\d[ -]?){13,19}\b",
            Self::AwsAccessKeyId => {
                r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA|AGPA|AIDA|AIPA|ANPA|ANVA|APKA|AROA|AUSA)[A-Z0-9]{16}\b"
            }
            Self::AwsSecretAccessKey => r"\b[A-Za-z0-9/+=]{40}\b",
            Self::GcpApiKey => r"\bAIza[0-9A-Za-z\-_]{35}\b",
            Self::AzureStorageAccountKey => r"\b[A-Za-z0-9+/]{86}==",
            Self::NvidiaApiKey => r"\bnvapi-[A-Za-z0-9_-]{20,}\b",
        }
    }
}

#[derive(Debug, PartialEq)]
pub(super) enum Assessment {
    Clear,
    Restricted(&'static str),
    Indeterminate(&'static str),
}

pub(super) struct Inspector {
    rules: InspectionRules,
}

enum InspectionRules {
    Structural,
    Detect {
        patterns: RegexSet,
        detectors: Vec<DeterministicDetector>,
    },
}

impl Inspector {
    pub(super) fn new(detectors: Vec<DeterministicDetector>) -> Result<Self, regex::Error> {
        let patterns = RegexSet::new(
            detectors
                .iter()
                .copied()
                .map(DeterministicDetector::pattern),
        )?;
        Ok(Self {
            rules: InspectionRules::Detect {
                patterns,
                detectors,
            },
        })
    }

    pub(super) fn structural() -> Self {
        Self {
            rules: InspectionRules::Structural,
        }
    }

    pub(super) fn inspect(&self, request: &Request) -> Assessment {
        let mut scan = Scan {
            bytes: 0,
            steps: 0,
            text: matches!(&self.rules, InspectionRules::Detect { .. }).then(String::new),
        };
        match scan.request(&request.llm_request) {
            Ok(()) => self
                .match_text(scan.text.as_deref())
                .map(Assessment::Restricted)
                .unwrap_or(Assessment::Clear),
            Err(assessment) => assessment,
        }
    }

    fn match_text(&self, text: Option<&str>) -> Option<&'static str> {
        let InspectionRules::Detect {
            patterns,
            detectors,
        } = &self.rules
        else {
            return None;
        };
        patterns
            .matches(text?)
            .iter()
            .next()
            .map(|index| detectors[index].as_str())
    }
}

struct Scan {
    bytes: usize,
    steps: usize,
    text: Option<String>,
}

impl Scan {
    fn request(&mut self, request: &LlmRequest) -> Result<(), Assessment> {
        // Keep this exhaustive so each new protocol field gets an explicit privacy decision.
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
        for instruction in instructions {
            self.blocks(&instruction.content, 0)?;
        }
        for message in messages {
            self.blocks(&message.content, 0)?;
        }
        for tool in tools {
            self.text(&tool.name)?;
            if let Some(description) = &tool.description {
                self.text(description)?;
            }
            self.json(&tool.parameters, 0)?;
        }
        if let Some(choice) = tool_choice {
            match choice {
                ToolChoice::Tool { name } => self.text(name)?,
                ToolChoice::Raw(value) => self.json(value, 0)?,
                ToolChoice::Auto | ToolChoice::Required | ToolChoice::None => {}
            }
        }
        if let Some(format) = &output.response_format {
            self.json(format, 0)?;
        }
        if let Some(effort) = &reasoning.effort {
            self.text(effort)?;
        }
        if let Some(raw) = &reasoning.raw {
            self.json(raw, 0)?;
        }
        for (key, value) in &extensions.fields {
            self.text(key)?;
            self.json(value, 0)?;
        }
        // Exact replay may retain fields omitted by normalization, so both
        // representations share the same bounded scan budget.
        for (format, body) in &preservation.requests {
            if ![
                WireFormat::OpenAiChat,
                WireFormat::OpenAiResponses,
                WireFormat::AnthropicMessages,
            ]
            .into_iter()
            .any(|known| format.as_str() == known.as_str())
            {
                return Err(Assessment::Indeterminate("opaque_content"));
            }
            self.json(body, 0)?;
        }
        Ok(())
    }

    fn blocks(&mut self, blocks: &[ContentBlock], depth: usize) -> Result<(), Assessment> {
        let mut continues_text = false;
        for block in blocks {
            self.step(depth)?;
            match block {
                ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
                    self.text_fragment(text, continues_text)?;
                    continues_text = true;
                }
                ContentBlock::Reasoning {
                    text,
                    signature,
                    details,
                } => {
                    continues_text = false;
                    self.text(text)?;
                    if let Some(signature) = signature {
                        self.text(signature)?;
                    }
                    for detail in details {
                        self.json(detail, depth + 1)?;
                    }
                }
                ContentBlock::ToolCall(call) => {
                    continues_text = false;
                    self.text(&call.id)?;
                    self.text(&call.name)?;
                    self.json(&call.arguments, depth + 1)?;
                }
                ContentBlock::ToolResult(result) => {
                    continues_text = false;
                    self.text(&result.tool_call_id)?;
                    self.blocks(&result.content, depth + 1)?;
                }
                ContentBlock::Image { .. }
                | ContentBlock::Audio { .. }
                | ContentBlock::Video { .. }
                | ContentBlock::File { .. }
                | ContentBlock::Unknown { .. } => {
                    return Err(Assessment::Indeterminate("opaque_content"));
                }
            }
        }
        Ok(())
    }

    fn json(&mut self, value: &serde_json::Value, depth: usize) -> Result<(), Assessment> {
        self.step(depth)?;
        match value {
            serde_json::Value::String(value) => self.text(value),
            serde_json::Value::Array(values) => {
                for value in values {
                    self.json(value, depth + 1)?;
                }
                Ok(())
            }
            serde_json::Value::Object(values) => {
                let kind = values.get("type").and_then(serde_json::Value::as_str);
                let encrypted_reasoning = kind == Some("reasoning.encrypted")
                    || kind == Some("reasoning")
                        && values
                            .get("encrypted_content")
                            .is_some_and(|value| !value.is_null());
                if encrypted_reasoning {
                    return Err(Assessment::Indeterminate("opaque_content"));
                }
                for (key, value) in values {
                    self.text(key)?;
                    self.json(value, depth + 1)?;
                }
                Ok(())
            }
            serde_json::Value::Number(value) => self.text(&value.to_string()),
            serde_json::Value::Null | serde_json::Value::Bool(_) => Ok(()),
        }
    }

    fn text(&mut self, text: &str) -> Result<(), Assessment> {
        self.text_fragment(text, false)
    }

    fn text_fragment(&mut self, text: &str, continues_text: bool) -> Result<(), Assessment> {
        self.step(0)?;
        self.bytes = self.bytes.saturating_add(text.len());
        if self.bytes > MAX_SCAN_BYTES {
            return Err(Assessment::Indeterminate("scan_limit"));
        }
        if let Some(scanned) = &mut self.text {
            if !continues_text && !scanned.is_empty() {
                // Keep unrelated values from forming one match across a structural boundary.
                scanned.push_str(STRUCTURAL_SEPARATOR);
            }
            // Adjacent text blocks stay contiguous so splitting one value cannot bypass a rule.
            scanned.push_str(text);
        }
        Ok(())
    }

    fn step(&mut self, depth: usize) -> Result<(), Assessment> {
        if depth > MAX_SCAN_DEPTH {
            return Err(Assessment::Indeterminate("scan_limit"));
        }
        self.steps = self.steps.saturating_add(1);
        if self.steps > MAX_SCAN_STEPS {
            return Err(Assessment::Indeterminate("scan_limit"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use switchyard_protocol::{ContentBlock, Message, Role, ToolCall};

    use super::*;

    fn inspect_with(detectors: &[DeterministicDetector], text: &str) -> Assessment {
        let mut request = Request::default();
        request
            .llm_request
            .messages
            .push(Message::text(Role::User, text));
        Inspector::new(detectors.to_vec())
            .expect("static patterns should compile")
            .inspect(&request)
    }

    #[test]
    fn supports_relay_detector_catalog_and_switchyard_credentials() {
        for (detector, text) in [
            (DeterministicDetector::Email, "user@company.test"),
            (DeterministicDetector::Phone, "+1 (555) 123-4567"),
            (DeterministicDetector::ApiKey, "sk-abcdefgh"),
            (DeterministicDetector::IpAddress, "192.168.1.1"),
            (
                DeterministicDetector::Ipv6,
                "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
            ),
            (DeterministicDetector::Url, "https://example.test/path"),
            (
                DeterministicDetector::Uuid,
                "550e8400-e29b-41d4-a716-446655440000",
            ),
            (
                DeterministicDetector::BearerToken,
                "Bearer abcdefghijklmnop",
            ),
            (DeterministicDetector::Jwt, "eyJheader.payload.signature"),
            (DeterministicDetector::CreditCard, "4111 1111 1111 1111"),
            (
                DeterministicDetector::AwsAccessKeyId,
                "AKIAIOSFODNN7EXAMPLE",
            ),
            (
                DeterministicDetector::AwsSecretAccessKey,
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            ),
            (
                DeterministicDetector::GcpApiKey,
                "AIza01234567890123456789012345678901234",
            ),
            (
                DeterministicDetector::AzureStorageAccountKey,
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==",
            ),
            (
                DeterministicDetector::NvidiaApiKey,
                "nvapi-abcdefghijklmnopqrstuvwxyz",
            ),
        ] {
            assert_eq!(
                inspect_with(&[detector], text),
                Assessment::Restricted(detector.as_str()),
                "{}",
                detector.as_str()
            );
        }

        assert_eq!(
            inspect_with(&[DeterministicDetector::Email], "Bearer abcdefghijklmnop"),
            Assessment::Clear
        );
        assert_eq!(
            inspect_with(&[DeterministicDetector::ApiKey], "task-abcdefgh"),
            Assessment::Clear
        );
    }

    #[test]
    fn inspects_structured_and_preserved_request_content() {
        let inspector = Inspector::new(vec![
            DeterministicDetector::Email,
            DeterministicDetector::BearerToken,
            DeterministicDetector::CreditCard,
        ])
        .expect("static patterns should compile");
        let mut request = Request::default();
        request.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call-1".to_string(),
                name: "lookup".to_string(),
                arguments: json!({"card": 4111111111111111_u64}),
            })],
        });
        assert_eq!(
            inspector.inspect(&request),
            Assessment::Restricted("credit_card")
        );

        let mut request = Request::default();
        request.llm_request.messages.push(Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "user@".to_string(),
                },
                ContentBlock::Text {
                    text: "company.test".to_string(),
                },
            ],
        });
        assert_eq!(inspector.inspect(&request), Assessment::Restricted("email"));

        let mut request = Request::default();
        request.llm_request.preservation.requests.insert(
            "openai_chat".into(),
            json!({"token": "Bearer abcdefghijklmnop"}),
        );
        assert_eq!(
            inspector.inspect(&request),
            Assessment::Restricted("bearer_token")
        );
    }

    #[test]
    fn does_not_join_unrelated_values_into_one_match() {
        let inspector = Inspector::new(vec![DeterministicDetector::CreditCard])
            .expect("static patterns should compile");
        let mut request = Request::default();
        request
            .llm_request
            .messages
            .push(Message::text(Role::User, "4111 1111"));
        request
            .llm_request
            .messages
            .push(Message::text(Role::User, "1111 1111"));

        assert_eq!(inspector.inspect(&request), Assessment::Clear);
    }

    #[test]
    fn opaque_or_oversized_content_is_indeterminate() {
        let inspector = Inspector::new(vec![DeterministicDetector::Email])
            .expect("static patterns should compile");
        let mut opaque = Request::default();
        opaque.llm_request.messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Unknown {
                provider: "custom".into(),
                raw: json!({}),
            }],
        });
        assert_eq!(
            inspector.inspect(&opaque),
            Assessment::Indeterminate("opaque_content")
        );
        let mut encrypted = Request::default();
        encrypted.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Reasoning {
                text: String::new(),
                signature: None,
                details: vec![json!({
                    "type": "reasoning",
                    "encrypted_content": "opaque"
                })],
            }],
        });
        assert_eq!(
            inspector.inspect(&encrypted),
            Assessment::Indeterminate("opaque_content")
        );

        let mut custom = Request::default();
        custom
            .llm_request
            .preservation
            .requests
            .insert("custom".into(), json!({}));
        assert_eq!(
            inspector.inspect(&custom),
            Assessment::Indeterminate("opaque_content")
        );

        let oversized = Request {
            llm_request: LlmRequest {
                messages: vec![Message::text(Role::User, "x".repeat(MAX_SCAN_BYTES + 1))],
                ..LlmRequest::default()
            },
            ..Request::default()
        };
        let mut deep_value = serde_json::Value::Null;
        for _ in 0..=MAX_SCAN_DEPTH {
            deep_value = json!([deep_value]);
        }
        let mut too_deep = Request::default();
        too_deep
            .llm_request
            .extensions
            .fields
            .insert("deep".to_string(), deep_value);

        let mut too_many = Request::default();
        too_many.llm_request.extensions.fields.insert(
            "many".to_string(),
            serde_json::Value::Array(vec![serde_json::Value::Null; MAX_SCAN_STEPS]),
        );

        for request in [&oversized, &too_deep, &too_many] {
            assert_eq!(
                inspector.inspect(request),
                Assessment::Indeterminate("scan_limit")
            );
        }
    }
}
