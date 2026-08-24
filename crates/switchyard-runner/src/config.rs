// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Version-1 TOML deployment loading for the shared runner.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use switchyard_llm_client::{
    Backend, ClientRouter, DEFAULT_MAX_RETRIES, HttpBackendConfig, ModelConfig,
    TranslatingLlmClient,
};
use switchyard_protocol::{ModelId, RoutedLlmClient, WireFormat};

use crate::{
    AlgorithmSpec, CallerAuthKind, CountTokensTarget, DecisionTarget, ModelCapabilities, Route,
    Runner, RunnerError,
};

const SUPPORTED_SCHEMA_VERSION: u32 = 1;
const MAX_CONFIGURED_RETRIES: u32 = 10;

type RunnerResult<T> = Result<T, RunnerError>;

pub(crate) fn load_runner(path: impl AsRef<Path>) -> RunnerResult<Runner> {
    let path = path.as_ref();
    let source = fs::read_to_string(path).map_err(|error| {
        RunnerError::configuration_source(
            format!("failed to read server config {}: {error}", path.display()),
            error,
        )
    })?;
    runner_from_toml(&source).map_err(|error| {
        RunnerError::configuration_source(
            format!("invalid server config {}: {error}", path.display()),
            error,
        )
    })
}

fn runner_from_toml(source: &str) -> RunnerResult<Runner> {
    let config: DeploymentConfig = toml::from_str(source).map_err(|error| {
        RunnerError::configuration_source(format!("failed to parse TOML: {error}"), error)
    })?;
    config.build()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeploymentConfig {
    schema_version: u32,
    #[serde(default)]
    llm_clients: BTreeMap<String, LlmClientConfig>,
    targets: BTreeMap<String, TargetConfig>,
    routes: BTreeMap<String, RouteConfig>,
}

#[derive(Debug)]
struct RouteConfig {
    id: ModelId,
    context_window: Option<u32>,
    tool_calling: Option<bool>,
    reasoning: Option<bool>,
    algorithm: AlgorithmSpec,
}

impl<'de> Deserialize<'de> for RouteConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut table = toml::Table::deserialize(deserializer)?;
        let id = take_required(&mut table, "id")?;
        let context_window = take_optional(&mut table, "context_window")?;
        let tool_calling = take_optional(&mut table, "tool_calling")?;
        let reasoning = take_optional(&mut table, "reasoning")?;
        let algorithm = AlgorithmSpec::deserialize(toml::Value::Table(table))
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            id,
            context_window,
            tool_calling,
            reasoning,
            algorithm,
        })
    }
}

fn take_required<T, E>(table: &mut toml::Table, name: &'static str) -> Result<T, E>
where
    T: DeserializeOwned,
    E: serde::de::Error,
{
    let value = table.remove(name).ok_or_else(|| E::missing_field(name))?;
    T::deserialize(value).map_err(E::custom)
}

fn take_optional<T, E>(table: &mut toml::Table, name: &'static str) -> Result<Option<T>, E>
where
    T: DeserializeOwned,
    E: serde::de::Error,
{
    table
        .remove(name)
        .map(|value| T::deserialize(value).map_err(E::custom))
        .transpose()
}

impl RouteConfig {
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            context_window: self.context_window,
            tool_calling: self.tool_calling,
            reasoning: self.reasoning,
        }
    }

    fn routing_target_names(&self) -> Vec<&str> {
        self.algorithm.routing_target_names()
    }

    fn callable_target_names(&self) -> Vec<&str> {
        self.algorithm.callable_target_names()
    }
}

impl DeploymentConfig {
    fn decision_target(&self, name: &str) -> Option<DecisionTarget> {
        let target = self.targets.get(name)?;
        let client = self.llm_clients.get(&target.llm_client)?;
        Some(DecisionTarget {
            target: name.to_string(),
            model: target.id.clone(),
            format: client.format.wire_format(),
            base_url: client.base_url.as_str().to_string(),
            extra_body: target.extra_body.clone(),
        })
    }

    fn build(self) -> RunnerResult<Runner> {
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(RunnerError::configuration(format!(
                "unsupported schema_version {}; expected {SUPPORTED_SCHEMA_VERSION}",
                self.schema_version
            )));
        }

        let mut seen_client_model_ids = HashSet::new();
        for (target_name, target) in &self.targets {
            validate_value("target name", target_name)?;
            validate_value(&format!("target {target_name} id"), &target.id)?;
            if !seen_client_model_ids.insert((target.llm_client.as_str(), target.id.as_str())) {
                tracing::warn!(
                    "target {target_name} reuses model id {} on llm client {}; only one target per id is kept and the other is dropped. Give each target a unique model id, or point both routes at one target.",
                    target.id,
                    target.llm_client
                );
            }
        }

        let clients = self.build_clients()?;
        let targets = self.build_targets();
        let mut routes = Vec::with_capacity(self.routes.len());
        for (route_name, config) in &self.routes {
            validate_value("route name", route_name)?;
            validate_value(&format!("route {route_name} id"), &config.id)?;
            for target_name in config.callable_target_names() {
                self.targets.get(target_name).ok_or_else(|| {
                    RunnerError::configuration(format!(
                        "route references unknown target {target_name}"
                    ))
                })?;
            }
            let capabilities = config.capabilities();
            if capabilities.context_window == Some(0) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} context_window must be greater than zero"
                )));
            }
            let algorithm = config
                .algorithm
                .build(route_name, &targets)
                .map_err(|error| RunnerError::configuration_source(error.to_string(), error))?;
            let (route_clients, caller_auth) =
                self.build_route_clients(route_name, config, &clients)?;
            let count_tokens_target = self.build_count_tokens_target(config, &clients);
            let decision_targets = config
                .routing_target_names()
                .into_iter()
                .filter_map(|name| self.decision_target(name))
                .collect();
            let route = Route::new(
                algorithm,
                route_clients,
                caller_auth,
                capabilities,
                count_tokens_target,
                decision_targets,
            );
            routes.push((config.id.clone(), route));
        }
        Ok(Runner::new(routes))
    }

    fn build_clients(&self) -> RunnerResult<BTreeMap<String, Arc<TranslatingLlmClient>>> {
        let mut models_by_client = self
            .llm_clients
            .keys()
            .map(|name| (name.clone(), Vec::new()))
            .collect::<BTreeMap<String, Vec<ModelConfig>>>();

        for (name, client_config) in &self.llm_clients {
            validate_value("llm client name", name)?;
            build_backend(name, client_config, &BTreeMap::new())?;
        }
        for (target_name, target) in &self.targets {
            let client_config = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "target {target_name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            let model_configs = models_by_client
                .get_mut(&target.llm_client)
                .ok_or_else(|| {
                    RunnerError::configuration("validated llm client was not initialized")
                })?;
            model_configs.push(ModelConfig::new(
                target.id.clone(),
                build_backend(&target.llm_client, client_config, &target.extra_body)?,
                None,
            ));
        }

        let mut clients = BTreeMap::new();
        for (name, model_configs) in models_by_client {
            let client = Arc::new(
                TranslatingLlmClient::new(&model_configs)
                    .map_err(|error| RunnerError::configuration(error.to_string()))?,
            );
            clients.insert(name, client);
        }
        Ok(clients)
    }

    fn build_targets(&self) -> BTreeMap<String, ModelId> {
        self.targets
            .iter()
            .map(|(name, config)| (name.clone(), config.id.clone()))
            .collect()
    }

    fn build_route_clients(
        &self,
        route_name: &str,
        route: &RouteConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> RunnerResult<(ClientRouter, Option<CallerAuthKind>)> {
        let mut by_model = HashMap::new();
        let mut caller_auth = None;
        for name in route.callable_target_names() {
            let target = self.targets.get(name).ok_or_else(|| {
                RunnerError::configuration(format!("route references unknown target {name}"))
            })?;
            let client = clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!("target {name} has no constructed llm client"))
            })?;
            let client_config = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "target {name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            if client_config.forward_auth {
                let target_auth = client_config.format.caller_auth_kind();
                if caller_auth.is_some_and(|kind| kind != target_auth) {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} cannot forward both Anthropic and OpenAI caller credentials"
                    )));
                }
                caller_auth = Some(target_auth);
            }
            let client: Arc<dyn RoutedLlmClient> = client.clone();
            by_model.insert(target.id.clone(), client);
        }
        Ok((ClientRouter::new(by_model), caller_auth))
    }

    fn build_count_tokens_target(
        &self,
        route: &RouteConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> Option<CountTokensTarget> {
        route
            .routing_target_names()
            .into_iter()
            .enumerate()
            .filter_map(|(index, name)| {
                let target = self.targets.get(name)?;
                let client = clients.get(&target.llm_client)?;
                client.supports_count_tokens(&target.id).then_some((
                    count_tokens_priority(name, &target.id),
                    index,
                    target,
                    client,
                ))
            })
            .min_by_key(|(priority, index, _, _)| (*priority, *index))
            .map(|(_, _, target, client)| CountTokensTarget {
                model: target.id.clone(),
                client: client.clone(),
            })
    }
}

fn count_tokens_priority(target_name: &str, model_id: &ModelId) -> usize {
    let target_name = target_name.to_ascii_lowercase();
    let model_id = model_id.to_ascii_lowercase();
    ["opus", "sonnet", "haiku"]
        .iter()
        .position(|hint| target_name.contains(hint) || model_id.contains(hint))
        .unwrap_or(3)
}

/// A client endpoint, parsed when the config loads rather than checked afterwards.
///
/// Holding a `HttpBaseUrl` is proof the value is an absolute HTTP(S) URL, so no
/// later stage has to re-check it or can forget to.
#[derive(Clone, Debug)]
struct HttpBaseUrl(reqwest::Url);

impl HttpBaseUrl {
    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl<'de> Deserialize<'de> for HttpBaseUrl {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let url = reqwest::Url::parse(raw.trim()).map_err(|error| {
            serde::de::Error::custom(format!("base_url must be an absolute HTTP(S) URL: {error}"))
        })?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
            return Err(serde::de::Error::custom(
                "base_url must be an absolute HTTP(S) URL",
            ));
        }
        Ok(Self(url))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmClientConfig {
    format: ClientFormat,
    base_url: HttpBaseUrl,
    api_key_env: Option<String>,
    #[serde(default)]
    forward_auth: bool,
    #[serde(default)]
    extra_headers: BTreeMap<String, String>,
    #[serde(default = "default_max_retries")]
    max_retries: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetConfig {
    id: ModelId,
    llm_client: String,
    #[serde(default)]
    extra_body: BTreeMap<String, Value>,
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

impl ClientFormat {
    const fn wire_format(self) -> WireFormat {
        match self {
            Self::OpenAiChat => WireFormat::OpenAiChat,
            Self::OpenAiResponses => WireFormat::OpenAiResponses,
            Self::AnthropicMessages => WireFormat::AnthropicMessages,
        }
    }

    const fn caller_auth_kind(self) -> CallerAuthKind {
        match self {
            Self::AnthropicMessages => CallerAuthKind::Anthropic,
            Self::OpenAiChat | Self::OpenAiResponses => CallerAuthKind::OpenAi,
        }
    }
}
fn build_backend(
    client_name: &str,
    config: &LlmClientConfig,
    extra_body: &BTreeMap<String, Value>,
) -> RunnerResult<Backend> {
    if config.max_retries > MAX_CONFIGURED_RETRIES {
        return Err(RunnerError::configuration(format!(
            "llm client {client_name} max_retries must be at most {MAX_CONFIGURED_RETRIES}"
        )));
    }
    if config.forward_auth && config.api_key_env.is_some() {
        return Err(RunnerError::configuration(format!(
            "llm client {client_name} cannot set both forward_auth and api_key_env"
        )));
    }
    let api_key = config
        .api_key_env
        .as_deref()
        .map(|variable| {
            if variable.trim().is_empty() {
                return Err(RunnerError::configuration(format!(
                    "llm client {client_name} api_key_env must not be empty"
                )));
            }
            let api_key = std::env::var(variable).map_err(|error| {
                RunnerError::configuration(format!(
                    "llm client {client_name} could not read api_key_env {variable}: {error}"
                ))
            })?;
            if api_key.trim().is_empty() {
                return Err(RunnerError::configuration(format!(
                    "llm client {client_name} api_key_env {variable} is empty"
                )));
            }
            Ok(api_key)
        })
        .transpose()?;
    let http = HttpBackendConfig {
        base_url: config.base_url.as_str().to_string(),
        api_key,
        forward_auth: config.forward_auth,
        extra_headers: config.extra_headers.clone(),
        extra_body: extra_body.clone(),
        max_retries: config.max_retries,
    };
    let backend = match config.format {
        ClientFormat::OpenAiChat => Backend::OpenAiChat(http),
        ClientFormat::OpenAiResponses => Backend::OpenAiResponses(http),
        ClientFormat::AnthropicMessages => Backend::Anthropic(http),
    };
    Ok(backend)
}

// A function so that serde default can use it.
const fn default_max_retries() -> u32 {
    DEFAULT_MAX_RETRIES
}

fn validate_value(label: &str, value: &str) -> RunnerResult<()> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(RunnerError::configuration(format!(
            "{label} must be non-empty and have no surrounding whitespace"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_presentation_fields_are_split_from_the_algorithm() {
        let route: RouteConfig = toml::from_str(
            r#"
type = "random"
id = "switchyard/random"
context_window = 128000
tool_calling = true
reasoning = false
targets = ["fast", "strong"]
weights = [1.0, 2.0]
seed = 7
"#,
        )
        .expect("route should deserialize");

        assert_eq!(route.id, "switchyard/random");
        assert_eq!(route.context_window, Some(128_000));
        assert_eq!(route.tool_calling, Some(true));
        assert_eq!(route.reasoning, Some(false));
        assert_eq!(route.algorithm.routing_target_names(), ["fast", "strong"]);
    }

    #[test]
    fn route_algorithm_unknown_fields_are_rejected() {
        let error = toml::from_str::<RouteConfig>(
            r#"
type = "passthrough"
id = "switchyard/fast"
target = "fast"
bogus = true
"#,
        )
        .expect_err("unknown route keys must be rejected");

        assert!(error.to_string().contains("bogus"), "{error}");
    }

    #[test]
    fn deployment_errors_keep_one_path_context() {
        let path = Path::new("/definitely/missing/switchyard-routes.toml");
        let error = match load_runner(path) {
            Ok(_) => panic!("missing config should fail"),
            Err(error) => error,
        };
        let message = error.to_string();

        assert_eq!(message.matches(&path.display().to_string()).count(), 1);
        assert!(message.starts_with("failed to read server config"));
    }
}
#[cfg(test)]
mod deployment_tests {
    use super::*;
    use serde_json::json;

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
        match runner_from_toml(toml) {
            Ok(_) => "configuration unexpectedly succeeded".to_string(),
            Err(error) => error.to_string(),
        }
    }

    fn with_subagent_llm_classifier(config: &str, route: &str, extra: &str) -> String {
        let mut configured = config.to_string();
        configured.push_str(&format!("\n[routes.{route}.subagents]\n"));
        configured.push_str(
            r#"type = "llm_classifier"
mode = "custom"
classifier_target = "classifier"
targets = ["strong", "weak"]
default_target = "weak"
prompt = "Select a target for this delegated task."
response_schema = '{"type":"object","properties":{"target":{"type":"string","enum":["strong","weak"]}},"required":["target"],"additionalProperties":false}'
policy = { type = "target_selector", selector = "/target" }
classify_trigger = "new_session""#,
        );
        configured.push_str(extra);
        configured
    }

    fn with_subagent_passthrough(config: &str, route: &str) -> String {
        format!("{config}\n[routes.{route}.subagents]\ntype = \"passthrough\"\ntarget = \"strong\"")
    }

    fn stage_config() -> String {
        format!(
            r#"{VALID_CONFIG}
[targets.stage_judge]
id = "stage-judge/model"
llm_client = "primary"

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 1.0

[routes.stage.classifier]
target = "stage_judge"
base_threshold = 0.5
"#
        )
    }

    fn hierarchical_config() -> String {
        format!(
            r#"{VALID_CONFIG}
[targets.tier_judge]
id = "tier-judge/model"
llm_client = "primary"

[routes.hier]
id = "switchyard/hier"
type = "hierarchical"

[routes.hier.classifier]
target = "tier_judge"
base_threshold = 0.5

[routes.hier.stage]
capable_target = "strong"
efficient_target = "weak"
confidence_threshold = 0.5
"#
        )
    }

    #[test]
    fn hierarchical_route_builds_and_claims_both_tiers_and_its_judge() -> RunnerResult<()> {
        let runner = runner_from_toml(&hierarchical_config())?;
        assert!(
            runner
                .models()
                .any(|model| model.id.as_str() == "switchyard/hier")
        );
        Ok(())
    }

    #[test]
    fn hierarchical_route_rejects_an_unknown_target() {
        let config = hierarchical_config().replace(
            "capable_target = \"strong\"",
            "capable_target = \"missing\"",
        );
        assert!(runner_from_toml(&config).is_err());
    }

    #[test]
    fn builds_all_supported_algorithm_types() -> RunnerResult<()> {
        let state = runner_from_toml(VALID_CONFIG)?;
        // The model id array is sorted alphabetically
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
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
    fn passthrough_and_stage_accept_subagent_routing() -> RunnerResult<()> {
        let stage = stage_config();
        let stage_with_classifier = with_subagent_llm_classifier(&stage, "stage", "");
        let parsed: DeploymentConfig = toml::from_str(&stage_with_classifier).map_err(|error| {
            RunnerError::configuration(format!("failed to parse stage config: {error}"))
        })?;
        let Some(stage_route) = parsed.routes.get("stage") else {
            return Err(RunnerError::configuration("stage route is missing"));
        };
        let callable_targets = stage_route.callable_target_names();
        for expected in ["strong", "weak", "stage_judge", "classifier"] {
            assert!(callable_targets.contains(&expected));
        }

        for configured in [
            with_subagent_llm_classifier(VALID_CONFIG, "passthrough", ""),
            with_subagent_passthrough(VALID_CONFIG, "passthrough"),
            stage_with_classifier,
            with_subagent_passthrough(&stage, "stage"),
        ] {
            runner_from_toml(&configured)?;
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_unreferenced_llm_client() {
        let invalid = format!(
            "{VALID_CONFIG}\n\
             [llm_clients.unused]\n\
             format = \"openai_chat\"\n\
             base_url = \"not a url\"\n"
        );
        let message = error_message(&invalid);
        assert!(
            message.contains("base_url must be an absolute HTTP(S) URL"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn an_escalation_table_switches_the_classifier_route_to_escalation() -> RunnerResult<()> {
        // Present: the classifier target judges the weak tier's reply each turn instead of
        // picking a tier ahead of it. The route builds either way, so the assertion is that
        // the knob parses and its settings reach the algorithm's validation.
        let escalating = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalating)?;

        // A setting that would starve the judge is rejected here rather than on the first
        // request, the same as any other unusable route configuration.
        let starved = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { confirmations = 0 }",
        );
        assert!(error_message(&starved).contains("confirmations must be at least 1"));
        Ok(())
    }

    #[test]
    fn classifier_judge_completion_caps_are_configurable() -> RunnerResult<()> {
        let capability = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nmax_output_tokens = 512",
        );
        runner_from_toml(&capability)?;

        let escalation = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nmax_output_tokens = 256\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalation)?;
        Ok(())
    }

    #[test]
    fn classifier_prompts_are_configurable_in_both_modes() -> RunnerResult<()> {
        let capability = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"custom capability rubric\"",
        );
        runner_from_toml(&capability)?;

        let escalation = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"custom trajectory rubric\"\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalation)?;

        let empty = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"   \"",
        );
        assert!(error_message(&empty).contains("classifier prompt must not be empty"));

        let schema_placeholder = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"{{RESPONSE_SCHEMA}}\"",
        );
        assert!(
            error_message(&schema_placeholder)
                .contains("Switchyard supplies the schema automatically")
        );
        Ok(())
    }

    #[test]
    fn mode_custom_rejects_capability_fields() {
        let mixed = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "mode = \"custom\"\nbase_threshold = 0.5",
        );

        assert!(
            error_message(&mixed)
                .contains("mode custom cannot use capability or escalation fields")
        );
    }

    #[test]
    fn stage_router_rejects_an_unknown_field() {
        let config = stage_config().replace(
            "picker = \"efficient_first\"",
            "picker = \"efficient_first\"\nmagic = true",
        );
        assert!(error_message(&config).contains("unknown field"));
    }

    #[test]
    fn hierarchical_stage_block_rejects_an_unknown_field() {
        let config = hierarchical_config().replace(
            "confidence_threshold = 0.5",
            "confidence_threshold = 0.5\nmagic = true",
        );
        assert!(error_message(&config).contains("unknown field"));
    }

    #[test]
    fn rejects_unknown_fields_and_algorithm_types() {
        let unknown_field =
            VALID_CONFIG.replace("schema_version = 1", "schema_version = 1\nmagic = true");
        assert!(error_message(&unknown_field).contains("unknown field"));

        let nested_completion_cap = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { max_output_tokens = 256 }",
        );
        assert!(error_message(&nested_completion_cap).contains("unknown field"));

        let unknown_classifier_field = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nclassifier_magic = true",
        );
        assert!(error_message(&unknown_classifier_field).contains("unknown field"));

        let target_capability = VALID_CONFIG.replace(
            "llm_client = \"responses\"",
            "llm_client = \"responses\"\ncontext_window = 1000000",
        );
        assert!(error_message(&target_capability).contains("unknown field `context_window`"));

        let unknown_algorithm = VALID_CONFIG.replace("type = \"noop\"", "type = \"imaginary\"");
        assert!(error_message(&unknown_algorithm).contains("unknown variant"));
    }

    #[test]
    fn rejects_unknown_stage_classifier_fields() {
        // Nested classifier typos must fail instead of silently using a default.
        let config = format!(
            r#"{VALID_CONFIG}

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 1.0

[routes.stage.classifier]
target = "classifier"
base_threshold = 0.5
classifier_magic = true
"#
        );

        let error = error_message(&config);
        assert!(
            error.contains("unknown field `classifier_magic`"),
            "{error}"
        );
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
                VALID_CONFIG.replace("classifier_target = \"classifier\"\n", ""),
                "route references unknown target",
            ),
            (
                VALID_CONFIG.replace(
                    "classifier_target = \"classifier\"",
                    "classifier_target = \"\"",
                ),
                "route references unknown target",
            ),
            (
                VALID_CONFIG.replace(
                    "classifier_target = \"classifier\"",
                    "classifier_target = \"   \"",
                ),
                "route references unknown target",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nthreshold_step = -0.1",
                ),
                "threshold_step must be finite and greater than or equal to 0",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.8\nthreshold_step = 0.11",
                ),
                "base_threshold + 2 * threshold_step must be at most 1",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nmax_output_tokens = 0\nescalation = { confirmations = 2 }",
                ),
                "max_output_tokens must be at least 1",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nmessage_hash_fallback = true",
                ),
                "message_hash_fallback requires classify_trigger = new_session",
            ),
            (
                with_subagent_llm_classifier(
                    VALID_CONFIG,
                    "passthrough",
                    "\nmessage_hash_fallback = true",
                ),
                "cannot use message_hash_fallback",
            ),
            (
                with_subagent_llm_classifier(VALID_CONFIG, "passthrough", "")
                    .replace("mode = \"custom\"", "mode = \"capability\""),
                "mode capability cannot use custom classifier fields",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "escalation = { confirmations = 2 }\nclassify_trigger = \"user_turn\"",
                ),
                "mode escalation cannot use classify_trigger",
            ),
            (
                VALID_CONFIG.replace("schema_version = 1", "schema_version = 2"),
                "unsupported schema_version 2",
            ),
            (
                VALID_CONFIG.replace("[targets.strong]", "[targets.\" strong \"]"),
                "target name must be non-empty and have no surrounding whitespace",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\ncontext_window = 0",
                ),
                "route random context_window must be greater than zero",
            ),
        ];

        for (toml, expected) in cases {
            let error = error_message(&toml);
            assert!(
                error.contains(expected),
                "expected error containing {expected}, got {error}"
            );
        }
    }

    #[test]
    fn accepts_duplicate_target_model_ids_on_one_client() -> RunnerResult<()> {
        // Two targets share one model id on one client. The client keeps one and drops the
        // other, so the build warns and still succeeds, and both routes resolve. Serving one
        // model under two route names this way is allowed; pointing both routes at one target
        // is the tidier form.
        const SAME_MODEL_TWO_ROUTES: &str = r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[targets.fast]
id = "gpt-4o"
llm_client = "primary"

[targets.smart]
id = "gpt-4o"
llm_client = "primary"

[routes.fast]
id = "switchyard/fast"
type = "passthrough"
target = "fast"

[routes.smart]
id = "switchyard/smart"
type = "passthrough"
target = "smart"
"#;
        let state = runner_from_toml(SAME_MODEL_TWO_ROUTES)?;
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["switchyard/fast", "switchyard/smart"]
        );
        Ok(())
    }

    #[test]
    fn accepts_same_model_id_on_different_llm_clients() -> RunnerResult<()> {
        // The same model id served by two llm clients never collides (each client keys its own
        // models), so cross-provider A/B builds with no warning; only a repeat within one client
        // warns.
        const CROSS_PROVIDER: &str = r#"
schema_version = 1

[llm_clients.openai]
format = "openai_chat"
base_url = "https://example.test/v1"

[llm_clients.azure]
format = "openai_chat"
base_url = "https://azure.test/v1"

[targets.openai]
id = "gpt-4o"
llm_client = "openai"

[targets.azure]
id = "gpt-4o"
llm_client = "azure"

[routes.openai]
id = "switchyard/openai-gpt4o"
type = "passthrough"
target = "openai"

[routes.azure]
id = "switchyard/azure-gpt4o"
type = "passthrough"
target = "azure"
"#;
        runner_from_toml(CROSS_PROVIDER)?;
        Ok(())
    }

    #[test]
    fn accepts_relative_weights_and_seed() -> RunnerResult<()> {
        let weighted = VALID_CONFIG.replace(
            "targets = [\"strong\", \"weak\"]",
            "targets = [\"strong\", \"weak\"]\nweights = [1, 3]\nseed = 42",
        );
        runner_from_toml(&weighted)?;
        Ok(())
    }

    #[test]
    fn accepts_new_session_trigger_with_message_hash_fallback() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.25\nthreshold_step = 0.1\nclassify_trigger = \"new_session\"\nmessage_hash_fallback = true",
        );
        runner_from_toml(&configured)?;
        Ok(())
    }

    #[test]
    fn target_extra_body_is_parsed_and_applied_to_its_backend() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "llm_client = \"primary\"",
            "llm_client = \"primary\"\n\
             extra_body = { service_tier = \"priority\", \
             chat_template_kwargs = { enable_thinking = false } }",
            1,
        );
        let config: DeploymentConfig = toml::from_str(&configured).map_err(|error| {
            RunnerError::configuration(format!("failed to parse config: {error}"))
        })?;
        let Some(target) = config.targets.get("classifier") else {
            return Err(RunnerError::configuration("classifier target is missing"));
        };
        let Some(client) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        let backend = build_backend("primary", client, &target.extra_body)?;

        assert_eq!(
            backend.extra_body().get("service_tier"),
            Some(&json!("priority"))
        );
        assert_eq!(
            backend
                .extra_body()
                .get("chat_template_kwargs")
                .and_then(|value| value.get("enable_thinking")),
            Some(&json!(false))
        );
        Ok(())
    }

    #[test]
    fn retry_budget_defaults_and_accepts_an_override() -> RunnerResult<()> {
        let default: DeploymentConfig = toml::from_str(VALID_CONFIG).map_err(|error| {
            RunnerError::configuration(format!("failed to parse default config: {error}"))
        })?;
        let Some(primary) = default.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, DEFAULT_MAX_RETRIES);

        let explicit = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = 0",
            1,
        );
        let config: DeploymentConfig = toml::from_str(&explicit).map_err(|error| {
            RunnerError::configuration(format!("failed to parse explicit retry config: {error}"))
        })?;
        let Some(primary) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, 0);

        let maximum = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            &format!(
                "base_url = \"https://example.test/v1\"\nmax_retries = {MAX_CONFIGURED_RETRIES}"
            ),
            1,
        );
        let config: DeploymentConfig = toml::from_str(&maximum).map_err(|error| {
            RunnerError::configuration(format!("failed to parse maximum retry config: {error}"))
        })?;
        let Some(primary) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, MAX_CONFIGURED_RETRIES);
        Ok(())
    }

    #[test]
    fn rejects_headers_that_switchyard_sets() {
        let cases = [
            (
                "base_url = \"https://example.test/v1\"",
                "base_url = \"https://example.test/v1\"\n\
                 extra_headers = { AUTHORIZATION = \"Bearer custom-key\" }",
                "AUTHORIZATION",
            ),
            (
                "base_url = \"https://example.test\"",
                "base_url = \"https://example.test\"\n\
                 extra_headers = { \"X-Api-Key\" = \"custom-key\" }",
                "X-Api-Key",
            ),
            (
                "base_url = \"https://example.test\"",
                "base_url = \"https://example.test\"\n\
                 extra_headers = { \"ANTHROPIC-VERSION\" = \"custom-version\" }",
                "ANTHROPIC-VERSION",
            ),
        ];

        for (original, replacement, header) in cases {
            let configured = VALID_CONFIG.replacen(original, replacement, 1);
            let error = error_message(&configured);
            assert!(
                error.contains(&format!("extra_headers cannot set {header:?}")),
                "expected {header} to be rejected, got: {error}"
            );
        }
    }

    #[test]
    fn accepts_additional_headers() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\n\
             extra_headers = { X-Inference-Priority = \"batch\" }",
            1,
        );

        runner_from_toml(&configured)?;
        Ok(())
    }

    #[test]
    fn retry_budget_rejects_negative_values() {
        let invalid = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = -1",
            1,
        );
        assert!(error_message(&invalid).contains("max_retries"));
    }

    #[test]
    fn retry_budget_rejects_excessive_values() {
        let invalid = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = 11",
            1,
        );
        assert!(
            error_message(&invalid).contains("llm client primary max_retries must be at most 10")
        );
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
        unsafe {
            // "unsafe" is for concurrent reads and writes, very rare
            std::env::set_var(EMPTY_KEY_ENV, "");
        }
        let empty = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            &format!("base_url = \"https://example.test/v1\"\napi_key_env = \"{EMPTY_KEY_ENV}\""),
            1,
        );
        let message = error_message(&empty);
        unsafe {
            std::env::remove_var(EMPTY_KEY_ENV);
        }
        assert!(message.contains("is empty"));
    }

    #[test]
    fn forward_auth_rejects_conflicting_credentials() {
        let competing_auth = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\n\
                 forward_auth = true\n\
                 api_key_env = \"UNUSED_TEST_KEY\"",
            1,
        );
        assert!(
            error_message(&competing_auth).contains("cannot set both forward_auth and api_key_env")
        );

        let static_auth = VALID_CONFIG.replacen(
            "base_url = \"https://example.test\"",
            "base_url = \"https://example.test\"\n\
                 forward_auth = true\n\
                 extra_headers = { Authorization = \"static-value\" }",
            1,
        );
        assert!(error_message(&static_auth).contains("extra_headers cannot set \"Authorization\""));

        let static_beta = static_auth.replace("Authorization", "anthropic-beta");
        assert!(
            error_message(&static_beta).contains("extra_headers cannot set \"anthropic-beta\"")
        );

        for header in ["chatgpt-account-id", "x-openai-fedramp"] {
            let static_context = VALID_CONFIG.replacen(
                "base_url = \"https://example.test/v1\"",
                &format!(
                    "base_url = \"https://example.test/v1\"\n\
                     forward_auth = true\n\
                     extra_headers = {{ \"{header}\" = \"static-value\" }}"
                ),
                1,
            );
            assert!(
                error_message(&static_context)
                    .contains(&format!("extra_headers cannot set \"{header}\""))
            );
        }
    }

    const ADVISOR_CONFIG: &str = r#"
schema_version = 1

[llm_clients.anthropic]
format = "anthropic_messages"
base_url = "https://example.test"

[targets.executor]
id = "executor/model"
llm_client = "anthropic"

[targets.advisor]
id = "advisor/model"
llm_client = "anthropic"

[routes.gated]
id = "switchyard/advisor"
type = "advisor"
executor_target = "executor"
advisor_target = "advisor"
"#;

    #[test]
    fn advisor_route_parses_with_defaults_and_builds() -> RunnerResult<()> {
        let state = runner_from_toml(ADVISOR_CONFIG)?;
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["switchyard/advisor"]
        );
        Ok(())
    }

    #[test]
    fn advisor_route_accepts_every_gate_knob() -> RunnerResult<()> {
        let tuned = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            concat!(
                "advisor_target = \"advisor\"\n",
                "reviewer_system_prompt = \"review it\"\n",
                "redo_feedback_prefix = \"REVIEWER SAYS: \"\n",
                "gate_trigger = \"pattern\"\n",
                "gate_trigger_pattern = 'task_complete[\"\\s>:]*true'\n",
                "max_reviews = 2\n",
                "gate_stall_turns = 40\n",
                "gate_min_tool_results = 1\n",
                "advisor_max_tokens = 1024\n",
                "advisor_temperature = 0.0\n",
                "transcript_max_chars = 100000\n",
                "fail_open = false\n",
                "context_window = 200000\n",
                "tool_calling = true\n",
                "reasoning = true",
            ),
        );
        runner_from_toml(&tuned)?;
        Ok(())
    }

    #[test]
    fn advisor_route_rejects_unknown_keys() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\nbogus_field = 1",
        );
        assert!(error_message(&invalid).contains("bogus_field"));
    }

    #[test]
    fn advisor_route_requires_both_targets() {
        let missing = ADVISOR_CONFIG.replace("advisor_target = \"advisor\"\n", "");
        assert!(error_message(&missing).contains("advisor_target"));
    }

    #[test]
    fn advisor_route_rejects_unknown_target() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"missing\"",
        );
        assert!(error_message(&invalid).contains("missing"));
    }

    #[test]
    fn advisor_route_rejects_invalid_pattern() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger = \"pattern\"\ngate_trigger_pattern = \"(unclosed\"",
        );
        assert!(error_message(&invalid).contains("not a valid regex"));
    }

    #[test]
    fn advisor_route_rejects_pattern_without_pattern_trigger() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger_pattern = \"done\"",
        );
        assert!(
            error_message(&invalid)
                .contains("gate_trigger_pattern requires gate_trigger = \"pattern\"")
        );
    }

    #[test]
    fn advisor_route_pattern_trigger_requires_pattern() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger = \"pattern\"",
        );
        assert!(error_message(&invalid).contains("non-empty gate_trigger_pattern"));
    }

    #[test]
    fn advisor_route_rejects_zero_max_reviews() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\nmax_reviews = 0",
        );
        assert!(error_message(&invalid).contains("max_reviews must be at least 1"));
    }
}
