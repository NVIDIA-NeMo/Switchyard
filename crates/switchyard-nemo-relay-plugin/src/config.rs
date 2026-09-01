// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use serde::Deserialize;
use serde_json::{Map, Value};
use switchyard_protocol::WireFormat;
use switchyard_runner::Runner;

pub(crate) fn protocol_from_call(name: &str) -> Option<WireFormat> {
    match name {
        "openai.chat_completions" => Some(WireFormat::OpenAiChat),
        "openai.responses" => Some(WireFormat::OpenAiResponses),
        "anthropic.messages" => Some(WireFormat::AnthropicMessages),
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SwitchyardConfig {
    #[serde(default)]
    pub(crate) priority: i32,
    #[serde(default)]
    pub(crate) switchyard_config_path: Option<PathBuf>,
    #[serde(default)]
    pub(crate) switchyard_config: Option<Map<String, Value>>,
}

impl SwitchyardConfig {
    pub(crate) fn load_runner(&self) -> Result<Runner, String> {
        match (&self.switchyard_config_path, &self.switchyard_config) {
            (Some(path), None) => Runner::load(path).map_err(|error| error.to_string()),
            (None, Some(config)) => toml::to_string(config)
                .map_err(|error| format!("failed to serialize Switchyard configuration: {error}"))
                .and_then(|source| Runner::from_toml(&source).map_err(|error| error.to_string())),
            (Some(_), Some(_)) => Err(
                "configure exactly one of switchyard_config_path or switchyard_config".to_string(),
            ),
            (None, None) => {
                Err("configure one of switchyard_config_path or switchyard_config".to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use serde_json::json;
    use tempfile::NamedTempFile;

    use super::*;

    fn inline_deployment() -> Map<String, Value> {
        json!({
            "schema_version": 1,
            "llm_clients": {
                "primary": {
                    "format": "openai_chat",
                    "base_url": "https://example.test/v1"
                }
            },
            "targets": {
                "default": {
                    "id": "example/model",
                    "llm_client": "primary"
                }
            },
            "routes": {
                "default": {
                    "id": "switchyard/default",
                    "type": "passthrough",
                    "target": "default"
                }
            }
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn maps_supported_relay_call_names_to_wire_formats() {
        let cases = [
            ("openai.chat_completions", Some(WireFormat::OpenAiChat)),
            ("openai.responses", Some(WireFormat::OpenAiResponses)),
            ("anthropic.messages", Some(WireFormat::AnthropicMessages)),
            ("unsupported.call", None),
        ];

        for (name, expected) in cases {
            assert_eq!(protocol_from_call(name), expected, "{name}");
        }
    }

    #[test]
    fn builds_runner_from_inline_deployment() {
        let config = SwitchyardConfig {
            priority: 0,
            switchyard_config_path: None,
            switchyard_config: Some(inline_deployment()),
        };

        let runner = config.load_runner().unwrap();
        assert!(runner.route("switchyard/default").is_some());
    }

    #[test]
    fn builds_runner_from_deployment_path() {
        let mut deployment = NamedTempFile::new().unwrap();
        write!(
            deployment,
            "{}",
            toml::to_string(&inline_deployment()).unwrap()
        )
        .unwrap();
        let config = SwitchyardConfig {
            priority: 0,
            switchyard_config_path: Some(deployment.path().to_path_buf()),
            switchyard_config: None,
        };

        let runner = config.load_runner().unwrap();
        assert!(runner.route("switchyard/default").is_some());
    }

    #[test]
    fn requires_exactly_one_configuration_source() {
        let neither = SwitchyardConfig {
            priority: 0,
            switchyard_config_path: None,
            switchyard_config: None,
        };
        assert!(matches!(
            neither.load_runner(),
            Err(error) if error.contains("configure one of switchyard_config_path or switchyard_config")
        ));

        let both = SwitchyardConfig {
            priority: 0,
            switchyard_config_path: Some(PathBuf::from("routes.toml")),
            switchyard_config: Some(inline_deployment()),
        };
        assert!(matches!(
            both.load_runner(),
            Err(error) if error.contains("configure exactly one of switchyard_config_path or switchyard_config")
        ));
    }
}
