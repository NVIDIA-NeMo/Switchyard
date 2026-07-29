// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed TOML configuration and explicit construction for the Rust server.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use libsy::algorithms::{LlmTaskClassifier, Noop, Passthrough, Random, TaskClassifierConfig};
use libsy::{Algorithm, LlmTarget, LlmTargetSet, RoutedLlmClient};
use serde::Deserialize;
use switchyard_llm_client::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};

use crate::{ServerError, ServerResult, ServerState};

const SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// Loads a TOML deployment file and constructs the complete server state.
pub fn load_server_state(path: impl AsRef<Path>) -> ServerResult<ServerState> {
    let path = path.as_ref();
    let toml = fs::read_to_string(path).map_err(|error| {
        ServerError::new(format!(
            "failed to read server config {}: {error}",
            path.display()
        ))
    })?;
    server_state_from_toml(&toml).map_err(|error| {
        ServerError::new(format!("invalid server config {}: {error}", path.display()))
    })
}

fn server_state_from_toml(toml: &str) -> ServerResult<ServerState> {
    let config: ServerConfig = toml::from_str(toml)
        .map_err(|error| ServerError::new(format!("failed to parse TOML: {error}")))?;
    config.build()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServerConfig {
    schema_version: u32,
    #[serde(default)]
    llm_clients: BTreeMap<String, LlmClientConfig>,
    targets: BTreeMap<String, TargetConfig>,
    routes: BTreeMap<String, RouteConfig>,
}

impl ServerConfig {
    fn build(&self) -> ServerResult<ServerState> {
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(ServerError::new(format!(
                "unsupported schema_version {}; expected {SUPPORTED_SCHEMA_VERSION}",
                self.schema_version
            )));
        }

        let clients = self.build_clients()?;
        let targets = self.build_targets(&clients)?;
        let mut routes = Vec::with_capacity(self.routes.len());
        for (route_name, config) in &self.routes {
            validate_value("route name", route_name)?;
            validate_value(&format!("route {route_name} id"), config.id())?;
            routes.push((
                config.id().to_string(),
                build_algorithm(route_name, config, &targets)?,
            ));
        }
        ServerState::new(routes)
    }

    fn build_clients(&self) -> ServerResult<BTreeMap<String, Arc<dyn RoutedLlmClient>>> {
        let mut models_by_client = self
            .llm_clients
            .keys()
            .map(|name| (name.clone(), Vec::new()))
            .collect::<BTreeMap<String, Vec<ModelConfig>>>();

        for name in self.llm_clients.keys() {
            validate_value("llm client name", name)?;
        }
        for (target_name, target) in &self.targets {
            validate_value("target name", target_name)?;
            validate_value(&format!("target {target_name} id"), &target.id)?;
            let client_config = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                ServerError::new(format!(
                    "target {target_name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            let model_configs = models_by_client
                .get_mut(&target.llm_client)
                .ok_or_else(|| ServerError::new("validated llm client was not initialized"))?;
            model_configs.push(ModelConfig::new(
                &target.id,
                build_backend(&target.llm_client, client_config)?,
                None,
            ));
        }

        let mut clients = BTreeMap::new();
        for (name, model_configs) in models_by_client {
            let client: Arc<dyn RoutedLlmClient> = Arc::new(
                TranslatingLlmClient::new(&model_configs)
                    .map_err(|error| ServerError::new(error.to_string()))?,
            );
            clients.insert(name, client);
        }
        Ok(clients)
    }

    fn build_targets(
        &self,
        clients: &BTreeMap<String, Arc<dyn RoutedLlmClient>>,
    ) -> ServerResult<BTreeMap<String, LlmTarget>> {
        self.targets
            .iter()
            .map(|(name, config)| {
                let client = clients.get(&config.llm_client).ok_or_else(|| {
                    ServerError::new(format!("target {name} has no constructed llm client"))
                })?;
                Ok((
                    name.clone(),
                    LlmTarget {
                        semantic_name: config.id.clone(),
                        llm_client: Some(Arc::clone(client)),
                    },
                ))
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmClientConfig {
    format: ClientFormat,
    base_url: String,
    api_key_env: Option<String>,
    #[serde(default)]
    extra_headers: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetConfig {
    id: String,
    llm_client: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
enum ClientFormat {
    #[serde(rename = "openai_chat")]
    OpenAiChat,
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum RouteConfig {
    Noop {
        id: String,
    },
    Random {
        id: String,
        targets: Vec<String>,
        weights: Option<Vec<f64>>,
        seed: Option<u64>,
    },
    Passthrough {
        id: String,
        target: String,
    },
    LlmClassifier {
        id: String,
        classifier_target: String,
        strong_target: String,
        weak_target: String,
        #[serde(flatten)]
        classifier_config: TaskClassifierConfig,
    },
}

impl RouteConfig {
    fn id(&self) -> &str {
        use RouteConfig::*;
        match self {
            Noop { id } | Random { id, .. } | LlmClassifier { id, .. } | Passthrough { id, .. } => {
                id
            }
        }
    }
}

fn build_backend(client_name: &str, config: &LlmClientConfig) -> ServerResult<Backend> {
    let base_url = config.base_url.trim();
    if base_url.is_empty() {
        return Err(ServerError::new(format!(
            "llm client {client_name} base_url must not be empty"
        )));
    }
    let api_key = config
        .api_key_env
        .as_deref()
        .map(|variable| {
            if variable.trim().is_empty() {
                return Err(ServerError::new(format!(
                    "llm client {client_name} api_key_env must not be empty"
                )));
            }
            let api_key = std::env::var(variable).map_err(|error| {
                ServerError::new(format!(
                    "llm client {client_name} could not read api_key_env {variable}: {error}"
                ))
            })?;
            if api_key.trim().is_empty() {
                return Err(ServerError::new(format!(
                    "llm client {client_name} api_key_env {variable} is empty"
                )));
            }
            Ok(api_key)
        })
        .transpose()?;
    let http = HttpBackendConfig {
        base_url: base_url.to_string(),
        api_key,
        extra_headers: config.extra_headers.clone(),
    };
    Ok(match config.format {
        ClientFormat::OpenAiChat => Backend::OpenAiChat(http),
        ClientFormat::OpenAiResponses => Backend::OpenAiResponses(http),
        ClientFormat::AnthropicMessages => Backend::Anthropic(http),
    })
}

fn build_algorithm(
    route_name: &str,
    config: &RouteConfig,
    targets: &BTreeMap<String, LlmTarget>,
) -> ServerResult<Arc<dyn Algorithm>> {
    match config {
        RouteConfig::Noop { .. } => Ok(Arc::new(Noop {})),
        RouteConfig::Random {
            targets: names,
            weights,
            seed,
            ..
        } => {
            let target_set =
                resolve_targets(route_name, names.iter().map(String::as_str), targets)?;
            let algorithm = Random::new(target_set, weights.clone(), *seed)
                .map_err(|error| ServerError::new(format!("random route {route_name}: {error}")))?;
            Ok(Arc::new(algorithm))
        }
        RouteConfig::Passthrough { target, .. } => {
            let target = resolve_target(route_name, target, targets)?;
            Ok(Arc::new(Passthrough::new(target)))
        }
        RouteConfig::LlmClassifier {
            classifier_target,
            strong_target,
            weak_target,
            classifier_config,
            ..
        } => {
            let classifier = resolve_target(route_name, classifier_target, targets)?;
            let strong = resolve_target(route_name, strong_target, targets)?;
            let weak = resolve_target(route_name, weak_target, targets)?;
            // The weak model is the efficient tier; the strong model is the capable one.
            let algorithm =
                LlmTaskClassifier::new(classifier, weak, strong, classifier_config.clone())
                    .map_err(|error| {
                        ServerError::new(format!("llm_classifier route {route_name}: {error}"))
                    })?;
            Ok(Arc::new(algorithm))
        }
    }
}

fn resolve_targets<'a>(
    route_name: &str,
    names: impl IntoIterator<Item = &'a str>,
    targets: &BTreeMap<String, LlmTarget>,
) -> ServerResult<LlmTargetSet> {
    let resolved = names
        .into_iter()
        .map(|name| resolve_target(route_name, name, targets))
        .collect::<ServerResult<Vec<_>>>()?;
    Ok(LlmTargetSet::new(resolved))
}

fn resolve_target(
    route_name: &str,
    name: &str,
    targets: &BTreeMap<String, LlmTarget>,
) -> ServerResult<LlmTarget> {
    targets.get(name).cloned().ok_or_else(|| {
        ServerError::new(format!(
            "route {route_name} references unknown target {name}"
        ))
    })
}

fn validate_value(label: &str, value: &str) -> ServerResult<()> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(ServerError::new(format!(
            "{label} must be non-empty and have no surrounding whitespace"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_CONFIG: &str = r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[llm_clients.responses]
format = "openai_responses"
base_url = "https://example.test/v1"

[llm_clients.anthropic]
format = "anthropic_messages"
base_url = "https://example.test"

[targets.classifier]
id = "classifier/model"
llm_client = "primary"

[targets.strong]
id = "strong/model"
llm_client = "responses"

[targets.weak]
id = "weak/model"
llm_client = "anthropic"

[routes.noop]
id = "switchyard/noop"
type = "noop"

[routes.random]
id = "switchyard/random"
type = "random"
targets = ["strong", "weak"]

[routes.classifier]
id = "switchyard/classifier"
type = "llm_classifier"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5

[routes.passthrough]
id = "switchyard/passthrough"
type = "passthrough"
target = "weak"
"#;

    fn error_message(toml: &str) -> String {
        match server_state_from_toml(toml) {
            Ok(_) => "configuration unexpectedly succeeded".to_string(),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn builds_all_supported_algorithm_types() -> ServerResult<()> {
        let state = server_state_from_toml(VALID_CONFIG)?;
        // The model id array is sorted alphabetically
        assert_eq!(
            state.models().collect::<Vec<_>>(),
            [
                "switchyard/classifier",
                "switchyard/noop",
                "switchyard/passthrough",
                "switchyard/random",
            ]
        );
        Ok(())
    }

    #[test]
    fn rejects_unknown_fields_and_algorithm_types() {
        let unknown_field =
            VALID_CONFIG.replace("schema_version = 1", "schema_version = 1\nmagic = true");
        assert!(error_message(&unknown_field).contains("unknown field"));

        let unknown_algorithm = VALID_CONFIG.replace("type = \"noop\"", "type = \"imaginary\"");
        assert!(error_message(&unknown_algorithm).contains("unknown variant"));
    }

    #[test]
    fn rejects_invalid_references_and_parameters() {
        let cases = [
            (
                VALID_CONFIG.replace("llm_client = \"primary\"", "llm_client = \"missing\""),
                "unknown llm client missing",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"missing\"]",
                ),
                "unknown target missing",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"strong\"]",
                ),
                "random targets must be unique",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\nweights = [1]",
                ),
                "expected 2 weights, got 1",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\nweights = [0, 0]",
                ),
                "at least one weight must be positive",
            ),
            (
                VALID_CONFIG.replace("base_threshold = 0.5", "base_threshold = 1.5"),
                "base_threshold must be between 0 and 1",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nmessage_hash_fallback = true",
                ),
                "message_hash_fallback requires session_affinity",
            ),
            (
                VALID_CONFIG.replace("schema_version = 1", "schema_version = 2"),
                "unsupported schema_version 2",
            ),
            (
                VALID_CONFIG.replace("[targets.strong]", "[targets.\" strong \"]"),
                "target name must be non-empty and have no surrounding whitespace",
            ),
        ];

        for (toml, expected) in cases {
            assert!(
                error_message(&toml).contains(expected),
                "expected error containing {expected}"
            );
        }
    }

    #[test]
    fn accepts_relative_weights_and_seed() -> ServerResult<()> {
        let weighted = VALID_CONFIG.replace(
            "targets = [\"strong\", \"weak\"]",
            "targets = [\"strong\", \"weak\"]\nweights = [1, 3]\nseed = 42",
        );
        server_state_from_toml(&weighted)?;
        Ok(())
    }

    #[test]
    fn accepts_session_affinity_with_message_hash_fallback() -> ServerResult<()> {
        let configured = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.25\nmin_confidence = 0.0\ncapability_elevated_floor = 0.45\nsession_affinity = true\nmessage_hash_fallback = true",
        );
        server_state_from_toml(&configured)?;
        Ok(())
    }

    #[test]
    fn api_key_environment_reference_is_validated() {
        let missing = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\napi_key_env = \"SWITCHYARD_CONFIG_TEST_KEY_THAT_IS_NOT_SET\"",
            1,
        );
        assert!(error_message(&missing).contains("SWITCHYARD_CONFIG_TEST_KEY_THAT_IS_NOT_SET"));

        const EMPTY_KEY_ENV: &str = "SWITCHYARD_CONFIG_TEST_EMPTY_KEY";
        std::env::set_var(EMPTY_KEY_ENV, "");
        let empty = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            &format!("base_url = \"https://example.test/v1\"\napi_key_env = \"{EMPTY_KEY_ENV}\""),
            1,
        );
        let message = error_message(&empty);
        std::env::remove_var(EMPTY_KEY_ENV);
        assert!(message.contains("is empty"));
    }
}
