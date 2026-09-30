// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared output-boundary redaction for configured provider credentials.

use serde_json::Value;

/// Replaces configured provider keys without changing the credentials sent upstream.
///
/// Apply this after response translation, including to preserved provider fields.
/// Each call is independent: keys split across stream events or separate JSON strings
/// are not reconstructed. This is exact substring matching, not a general secret detector.
#[derive(Default)]
pub struct ProviderKeyRedactor {
    text: Vec<String>,
    json: Vec<String>,
}

impl ProviderKeyRedactor {
    /// Registers nonempty keys, matching longer keys before their prefixes.
    pub fn new(keys: &[String]) -> Self {
        let mut raw: Vec<String> = keys.iter().filter(|key| !key.is_empty()).cloned().collect();
        raw.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        raw.dedup();
        let mut json: Vec<String> = raw
            .iter()
            .map(|key| {
                let encoded = Value::String(key.clone()).to_string();
                encoded[1..encoded.len() - 1].to_string()
            })
            .collect();
        json.sort_by_key(|key| std::cmp::Reverse(key.len()));
        // Match escaped forms before any raw-key suffix they contain. Otherwise a
        // leading backslash can survive replacement and invalidate embedded JSON.
        let mut text = raw;
        text.extend(json.iter().cloned());
        text.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        text.dedup();
        Self { text, json }
    }

    /// Returns whether there are no configured secrets to redact.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Redacts JSON-escaped keys in serialized JSON, preserving the server's wire behavior.
    pub fn json(&self, value: String) -> String {
        replace(value, &self.json)
    }

    /// Redacts raw and JSON-escaped keys in text, including headers and error messages.
    pub fn text(&self, value: String) -> String {
        replace(value, &self.text)
    }

    /// Redacts raw and JSON-escaped keys in string values and member names throughout JSON.
    pub fn value(&self, value: Value) -> Value {
        if self.is_empty() {
            return value;
        }
        match value {
            Value::String(text) => Value::String(self.text(text)),
            Value::Array(values) => {
                Value::Array(values.into_iter().map(|value| self.value(value)).collect())
            }
            Value::Object(fields) => Value::Object(
                fields
                    .into_iter()
                    .map(|(name, value)| (self.text(name), self.value(value)))
                    .collect(),
            ),
            value => value,
        }
    }
}

fn replace(mut value: String, secrets: &[String]) -> String {
    for secret in secrets {
        if value.contains(secret) {
            value = value.replace(secret, "[REDACTED]");
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn escaped_credentials_are_redacted_in_names_and_nested_values() {
        let key = "synthetic-\"provider\\key-雪";
        let redactor = ProviderKeyRedactor::new(&[key.into()]);
        let value = json!({
            "error": {"debug": format!("Bearer {key}"), key: [key, "ordinary diagnostics"]},
            "count": 3, "enabled": true, "missing": null,
        });
        let sanitized = redactor.value(value.clone());
        assert_eq!(sanitized["error"]["debug"], "Bearer [REDACTED]");
        assert_eq!(
            sanitized["error"]["[REDACTED]"],
            json!(["[REDACTED]", "ordinary diagnostics"])
        );
        assert_eq!(sanitized["count"], 3);
        assert_eq!(sanitized["enabled"], true);
        assert!(sanitized["missing"].is_null());
        let wire: Value = serde_json::from_str(&redactor.json(value.to_string())).unwrap();
        assert_eq!(wire, sanitized);
        assert_eq!(redactor.text(format!("Bearer {key}")), "Bearer [REDACTED]");
    }

    #[test]
    fn tool_arguments_and_preserved_extensions_are_redacted() {
        let redactor = ProviderKeyRedactor::new(&["qa-token".into()]);
        let value = json!({
            "arguments": json!({"credential": "qa-token", "query": "ordinary text"}).to_string(),
            "provider_extension": {"debug": ["Bearer qa-token"]},
        });
        let sanitized = redactor.value(value.clone());
        let arguments: Value =
            serde_json::from_str(sanitized["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(
            arguments,
            json!({"credential": "[REDACTED]", "query": "ordinary text"})
        );
        assert_eq!(
            sanitized["provider_extension"]["debug"],
            json!(["Bearer [REDACTED]"])
        );
        assert_eq!(
            serde_json::from_str::<Value>(&redactor.json(value.to_string())).unwrap(),
            sanitized
        );
    }

    #[test]
    fn json_escaped_credentials_are_redacted_in_error_text() {
        for key in [r#"provider-"key"#, r"provider-\key", r#"\provider-"key"#] {
            let redactor = ProviderKeyRedactor::new(&[key.into()]);
            let detail = json!({"credential": key, "message": "ordinary diagnostics"});
            let text = format!("raw: {key}; provider error: {detail}");
            let expected = format!(
                "raw: [REDACTED]; provider error: {}",
                json!({"credential": "[REDACTED]", "message": "ordinary diagnostics"})
            );
            assert_eq!(redactor.text(text), expected);
        }
    }

    #[test]
    fn nested_json_strings_cannot_recover_escaped_credentials() {
        for key in [r#"provider-"key"#, r"provider-\key", r#"\provider-"key"#] {
            let redactor = ProviderKeyRedactor::new(&[key.into()]);
            let arguments = json!({key: {"credential": key, "query": "ordinary text"}});
            let response = json!({"result": {"arguments": arguments.to_string()}});
            // Decode both layers, as a client consuming serialized tool arguments would.
            let sanitized = redactor.value(response);
            let decoded: Value =
                serde_json::from_str(sanitized["result"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(
                decoded,
                json!({"[REDACTED]": {
                    "credential": "[REDACTED]", "query": "ordinary text"
                }})
            );
        }
    }

    #[test]
    fn escaped_form_is_replaced_before_an_overlapping_raw_key() {
        let key = r"\provider-key";
        let redactor = ProviderKeyRedactor::new(&[key.into()]);
        let text = json!({"credential": key}).to_string();
        // Replacing the raw suffix first would leave a stray escape in the JSON string.
        let sanitized = redactor.text(text);
        assert_eq!(
            serde_json::from_str::<Value>(&sanitized).unwrap(),
            json!({"credential": "[REDACTED]"})
        );
    }

    #[test]
    fn empty_duplicate_and_overlapping_keys_preserve_nonsecret_content() {
        let redactor = ProviderKeyRedactor::new(&[
            "".into(),
            "qa-token".into(),
            "qa-token-extra".into(),
            "qa-token".into(),
        ]);
        assert_eq!(
            redactor.text("keep qa-token-extra / qa-token".into()),
            "keep [REDACTED] / [REDACTED]"
        );
        assert!(ProviderKeyRedactor::new(&["".into()]).is_empty());
        let value = json!({"choices": [{"delta": {"content": "ordinary text"}}]});
        assert_eq!(redactor.value(value.clone()), value);
        assert_eq!(ProviderKeyRedactor::default().value(value.clone()), value);
    }

    #[test]
    fn independent_events_do_not_reassemble_split_credentials() {
        let redactor = ProviderKeyRedactor::new(&["qa-token".into()]);
        let events = [json!({"delta": "qa-"}), json!({"delta": "token"})];
        for event in events {
            assert_eq!(redactor.value(event.clone()), event);
        }
    }
}
