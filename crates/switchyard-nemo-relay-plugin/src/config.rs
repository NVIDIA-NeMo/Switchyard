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
        let runner = match (&self.switchyard_config_path, &self.switchyard_config) {
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
        }?;
        // Relay keeps caller credentials outside the request passed to execution plugins.
        for model in runner.models() {
            if runner
                .route(model.id.as_str())
                .and_then(|route| route.caller_auth())
                .is_some()
            {
                return Err(format!(
                    "route {} uses forward_auth = true, which the native Relay plugin does not support; \
                     disable forward_auth and configure api_key_env for deployment-owned credentials, \
                     or use standalone switchyard-server for caller-owned credentials",
                    model.id
                ));
            }
        }
        Ok(runner)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use nemo_relay_plugin::{DiagnosticLevel, NativePlugin};
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

    // Both plugin validation and activation must reject every configuration source.
    fn assert_forward_auth_rejected(deployment: Map<String, Value>) {
        let source = toml::to_string(&deployment).unwrap();
        // The same deployment remains supported by the standalone runner.
        let runner = Runner::from_toml(&source).unwrap();
        assert!(
            runner
                .route("switchyard/default")
                .unwrap()
                .caller_auth()
                .is_some()
        );
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{source}").unwrap();

        for plugin_config in [
            json!({"switchyard_config": deployment}),
            json!({"switchyard_config_path": file.path()}),
        ] {
            let plugin_config = plugin_config.as_object().unwrap();
            let diagnostics = crate::SwitchyardPlugin.validate(plugin_config);
            assert_eq!(diagnostics.len(), 1);
            assert!(matches!(diagnostics[0].level, DiagnosticLevel::Error));
            assert_eq!(diagnostics[0].code, "switchyard.invalid_config");
            for detail in [
                "switchyard/default",
                "forward_auth",
                "api_key_env",
                "switchyard-server",
            ] {
                assert!(diagnostics[0].message.contains(detail), "{diagnostics:?}");
            }
            assert!(matches!(
                crate::parse_config(plugin_config).and_then(crate::runtime::SwitchyardRuntime::new),
                Err(error) if error == diagnostics[0].message
            ));
        }
    }

    #[test]
    fn rejects_forward_auth_for_all_provider_formats() {
        for format in ["openai_chat", "openai_responses", "anthropic_messages"] {
            let mut deployment = inline_deployment();
            deployment["llm_clients"]["primary"]["format"] = json!(format);
            deployment["llm_clients"]["primary"]["forward_auth"] = json!(true);
            assert_forward_auth_rejected(deployment);
        }
    }

    #[test]
    fn rejects_forward_auth_on_routing_only_and_alternate_targets() {
        for route in [
            json!({
                "id": "switchyard/default", "type": "advisor",
                "executor_target": "default", "advisor_target": "forwarded"
            }),
            json!({
                "id": "switchyard/default", "type": "random",
                "targets": ["default", "forwarded"]
            }),
        ] {
            let mut deployment = inline_deployment();
            deployment["llm_clients"]["forwarded"] = json!({
                "format": "openai_chat", "base_url": "https://example.test/v1",
                "forward_auth": true
            });
            deployment["targets"]["forwarded"] = json!({
                "id": "example/forwarded", "llm_client": "forwarded"
            });
            deployment["routes"]["default"] = route;
            assert_forward_auth_rejected(deployment);
        }
    }

    #[test]
    fn accepts_deployment_credentials_and_unused_forward_auth_clients() {
        let mut deployment = inline_deployment();
        deployment["llm_clients"]["primary"]["forward_auth"] = json!(false);
        // PATH supplies a non-secret, existing value without mutating process environment.
        deployment["llm_clients"]["primary"]["api_key_env"] = json!("PATH");
        deployment["llm_clients"]["unused"] = json!({
            "format": "openai_chat", "base_url": "https://example.test/v1",
            "forward_auth": true
        });
        deployment["targets"]["unused"] = json!({
            "id": "example/unused", "llm_client": "unused"
        });
        let plugin_config = json!({"switchyard_config": deployment});
        let plugin_config = plugin_config.as_object().unwrap();
        assert!(crate::SwitchyardPlugin.validate(plugin_config).is_empty());
        assert!(
            crate::parse_config(plugin_config)
                .and_then(crate::runtime::SwitchyardRuntime::new)
                .is_ok()
        );
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
