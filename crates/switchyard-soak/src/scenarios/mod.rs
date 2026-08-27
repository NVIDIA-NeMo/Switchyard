// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible request scenarios shared by the soak command and AIPerf runner.

mod classifier_mix;
mod client_cancellation;
mod context_overflow;
mod decode_heavy;
mod failure_pressure;
mod growing_conversation;
mod large_tool_catalog;
mod long_context;
mod mixed_traffic;
mod prefix_reuse;
mod short_interactive;
mod stage_transitions;
mod tool_call_burst;

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use clap::ValueEnum;
use serde::Serialize;
use serde_json::{Value, json};

use crate::client::Endpoint;

/// Largest accepted context-window input for generated scenario data.
const MAX_CONTEXT_WINDOW_TOKENS: usize = 1_000_000;
const MIN_CONTEXT_WINDOW_TOKENS: usize = 16_384;
const MAX_PROMPT_BYTES: usize = 1_000_000;

/// Scenario family used by CLI selection and report grouping.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioGroup {
    Core,
    Agentic,
    Resilience,
}

/// Accepted client-visible error-rate range for one scenario.
#[derive(Clone, Copy, Serialize)]
pub struct ErrorExpectation {
    pub min_rate: f64,
    pub max_rate: f64,
}

impl ErrorExpectation {
    pub const SUCCESS: Self = Self {
        min_rate: 0.0,
        max_rate: 0.0,
    };
    pub const MIXED: Self = Self {
        min_rate: 0.01,
        max_rate: 0.75,
    };
    pub const ALL: Self = Self {
        min_rate: 0.8,
        max_rate: 1.0,
    };
}

/// Named scenario collection selected by the soak and benchmark commands.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum ScenarioSet {
    Core,
    Agentic,
    Resilience,
    Standard,
    All,
}

impl ScenarioSet {
    fn contains(self, group: ScenarioGroup) -> bool {
        match self {
            Self::Core => group == ScenarioGroup::Core,
            Self::Agentic => group == ScenarioGroup::Agentic,
            Self::Resilience => group == ScenarioGroup::Resilience,
            Self::Standard => group != ScenarioGroup::Resilience,
            Self::All => true,
        }
    }
}

/// Request-generation limits shared by every scenario builder.
#[derive(Clone, Copy)]
pub struct ScenarioOptions<'a> {
    pub model: &'a str,
    pub prompt_bytes: usize,
    pub max_output_tokens: u32,
    pub context_window_tokens: usize,
}

impl ScenarioOptions<'_> {
    pub fn validate(&self) -> Result<(), String> {
        if self.model.is_empty() {
            return Err("scenario model must not be empty".to_string());
        }
        if self.prompt_bytes == 0 || self.max_output_tokens == 0 {
            return Err("scenario prompt and output limits must be greater than zero".to_string());
        }
        if !(MIN_CONTEXT_WINDOW_TOKENS..=MAX_CONTEXT_WINDOW_TOKENS)
            .contains(&self.context_window_tokens)
        {
            return Err(format!(
                "--context-window-tokens must be between {MIN_CONTEXT_WINDOW_TOKENS} and {MAX_CONTEXT_WINDOW_TOKENS}"
            ));
        }
        let prompt_limit = MAX_PROMPT_BYTES.min(self.context_window_tokens.saturating_mul(2));
        if self.prompt_bytes > prompt_limit {
            return Err(format!(
                "--prompt-bytes must not exceed {prompt_limit} for this context window"
            ));
        }
        Ok(())
    }
}

/// One AIPerf conversation; payloads remain ordered within the session.
#[derive(Clone, Serialize)]
pub struct Session {
    pub session_id: String,
    pub payloads: Vec<Value>,
}

/// One request the Rust soak command can send.
pub struct RequestCase {
    pub endpoint: Endpoint,
    pub session_id: String,
    streaming_body: Value,
    nonstreaming_body: Value,
}

impl RequestCase {
    pub fn new(endpoint: Endpoint, session_id: String, streaming_body: Value) -> Self {
        let mut nonstreaming_body = streaming_body.clone();
        nonstreaming_body["stream"] = Value::Bool(false);
        Self {
            endpoint,
            session_id,
            streaming_body,
            nonstreaming_body,
        }
    }
}

pub struct PreparedRequest<'a> {
    pub endpoint: Endpoint,
    pub session_id: &'a str,
    pub body: &'a Value,
}

/// QPS point expressed relative to the benchmark command's base request rate.
#[derive(Clone, Serialize)]
pub struct RatePoint {
    pub time_s: u32,
    pub rate_multiplier: f64,
}

/// AIPerf schedule applied to a request scenario.
#[derive(Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LoadProfile {
    Fixed {
        id: &'static str,
    },
    ConcurrencyKnee {
        id: &'static str,
        concurrency_steps: Vec<usize>,
    },
    TrafficBurst {
        id: &'static str,
        duration_seconds: u32,
        points: Vec<RatePoint>,
    },
}

/// Immutable scenario data built once before load workers start.
pub struct Scenario {
    pub id: &'static str,
    pub group: ScenarioGroup,
    pub description: &'static str,
    pub expected: &'static str,
    pub expected_error_rate: ErrorExpectation,
    pub sessions: Vec<Session>,
    pub soak_requests: Vec<RequestCase>,
    pub load_profiles: Vec<LoadProfile>,
}

impl Scenario {
    /// Build a Chat Completions scenario whose AIPerf payloads also drive soak traffic.
    pub fn chat(
        id: &'static str,
        group: ScenarioGroup,
        description: &'static str,
        expected: &'static str,
        expected_error_rate: ErrorExpectation,
        sessions: Vec<Session>,
    ) -> Self {
        let soak_requests =
            sessions
                .iter()
                .flat_map(|session| {
                    session.payloads.iter().cloned().map(|body| {
                        RequestCase::new(Endpoint::Chat, session.session_id.clone(), body)
                    })
                })
                .collect();
        Self {
            id,
            group,
            description,
            expected,
            expected_error_rate,
            sessions,
            soak_requests,
            load_profiles: fixed_load(),
        }
    }

    /// Return one prebuilt request with only its streaming flag changed.
    pub fn request(&self, index: usize, stream: bool) -> PreparedRequest<'_> {
        let selected = &self.soak_requests[index % self.soak_requests.len()];
        PreparedRequest {
            endpoint: selected.endpoint,
            session_id: &selected.session_id,
            body: if stream {
                &selected.streaming_body
            } else {
                &selected.nonstreaming_body
            },
        }
    }
}

/// Return the one fixed schedule used by ordinary request scenarios.
pub fn fixed_load() -> Vec<LoadProfile> {
    vec![LoadProfile::Fixed { id: "fixed" }]
}

/// Build the three bounded load schedules used for the short baseline.
pub fn baseline_load() -> Vec<LoadProfile> {
    vec![
        LoadProfile::Fixed { id: "fixed" },
        LoadProfile::ConcurrencyKnee {
            id: "concurrency-knee",
            concurrency_steps: vec![1, 4, 16, 64, 128],
        },
        LoadProfile::TrafficBurst {
            id: "traffic-burst",
            duration_seconds: 30,
            points: vec![
                RatePoint {
                    time_s: 0,
                    rate_multiplier: 1.0,
                },
                RatePoint {
                    time_s: 10,
                    rate_multiplier: 1.0,
                },
                RatePoint {
                    time_s: 11,
                    rate_multiplier: 10.0,
                },
                RatePoint {
                    time_s: 16,
                    rate_multiplier: 10.0,
                },
                RatePoint {
                    time_s: 17,
                    rate_multiplier: 1.0,
                },
                RatePoint {
                    time_s: 30,
                    rate_multiplier: 1.0,
                },
            ],
        },
    ]
}

/// Build one raw Chat Completions payload.
pub fn chat_payload(model: &str, messages: Vec<Value>, max_tokens: u32) -> Value {
    json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": true,
    })
}

/// Build one user message.
pub fn user(content: impl Into<String>) -> Value {
    json!({"role": "user", "content": content.into()})
}

/// Build one assistant message.
pub fn assistant(content: impl Into<String>) -> Value {
    json!({"role": "assistant", "content": content.into()})
}

/// Build repeated context with a stable approximate token count.
pub fn token_text(tokens: usize, label: &str) -> String {
    format!("[scenario:{label}] {}", "x ".repeat(tokens))
}

/// Build the complete catalog in stable report order.
pub fn catalog(options: ScenarioOptions<'_>) -> Result<Vec<Scenario>, String> {
    options.validate()?;
    Ok(vec![
        short_interactive::build(options),
        long_context::build(options),
        decode_heavy::build(options),
        prefix_reuse::build(options),
        mixed_traffic::build(options),
        growing_conversation::build(options),
        large_tool_catalog::build(options),
        tool_call_burst::build(options),
        stage_transitions::build(options),
        classifier_mix::build(options),
        context_overflow::build(options),
        failure_pressure::build(options),
        client_cancellation::build(options),
    ])
}

/// Select explicit scenario IDs or every scenario in a named set.
pub fn select(
    options: ScenarioOptions<'_>,
    set: ScenarioSet,
    requested: &[String],
) -> Result<Vec<Scenario>, String> {
    let mut catalog = catalog(options)?;
    if requested.is_empty() {
        return Ok(catalog
            .into_iter()
            .filter(|scenario| set.contains(scenario.group))
            .collect());
    }

    let mut seen = HashSet::new();
    let mut selected = Vec::new();
    for id in requested {
        if !seen.insert(id) {
            return Err(format!("scenario {id:?} was requested more than once"));
        }
        let index = catalog
            .iter()
            .position(|scenario| scenario.id == id)
            .ok_or_else(|| format!("unknown scenario {id:?}"))?;
        selected.push(catalog.remove(index));
    }
    Ok(selected)
}

#[derive(Serialize)]
struct InputsFile<'a> {
    data: &'a [Session],
}

#[derive(Serialize)]
struct Manifest<'a> {
    schema_version: u32,
    model: &'a str,
    scenarios: Vec<ManifestScenario<'a>>,
}

#[derive(Serialize)]
struct ManifestScenario<'a> {
    id: &'a str,
    group: ScenarioGroup,
    description: &'a str,
    expected: &'a str,
    expected_error_rate: ErrorExpectation,
    input_file: String,
    load_profiles: &'a [LoadProfile],
}

/// Export the selected scenario catalog in AIPerf's verbatim inputs JSON format.
pub fn export(scenarios: &[Scenario], model: &str, output_dir: &Path) -> Result<PathBuf, String> {
    if output_dir.exists() {
        return Err(format!(
            "scenario output directory already exists: {}",
            output_dir.display()
        ));
    }
    let inputs_dir = output_dir.join("inputs");
    fs::create_dir_all(&inputs_dir).map_err(|error| error.to_string())?;

    let mut entries = Vec::new();
    for scenario in scenarios {
        let relative = format!("inputs/{}.json", scenario.id);
        let path = output_dir.join(&relative);
        let body = serde_json::to_string_pretty(&InputsFile {
            data: &scenario.sessions,
        })
        .map_err(|error| error.to_string())?;
        fs::write(path, format!("{body}\n")).map_err(|error| error.to_string())?;
        entries.push(ManifestScenario {
            id: scenario.id,
            group: scenario.group,
            description: scenario.description,
            expected: scenario.expected,
            expected_error_rate: scenario.expected_error_rate,
            input_file: relative,
            load_profiles: &scenario.load_profiles,
        });
    }

    let manifest_path = output_dir.join("manifest.json");
    let body = serde_json::to_string_pretty(&Manifest {
        schema_version: 1,
        model,
        scenarios: entries,
    })
    .map_err(|error| error.to_string())?;
    fs::write(&manifest_path, format!("{body}\n")).map_err(|error| error.to_string())?;
    Ok(manifest_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> ScenarioOptions<'static> {
        ScenarioOptions {
            model: "switchyard/test",
            prompt_bytes: 128,
            max_output_tokens: 32,
            context_window_tokens: 16_384,
        }
    }

    #[test]
    fn catalog_is_unique_bounded_and_serializable() -> Result<(), String> {
        let scenarios = catalog(options())?;
        let names = scenarios
            .iter()
            .map(|scenario| scenario.id)
            .collect::<HashSet<_>>();
        assert_eq!(names.len(), scenarios.len());
        assert!(scenarios.iter().all(|scenario| {
            !scenario.sessions.is_empty()
                && !scenario.soak_requests.is_empty()
                && scenario.sessions.len() <= 32
                && scenario
                    .sessions
                    .iter()
                    .all(|session| !session.payloads.is_empty() && session.payloads.len() <= 32)
        }));

        let parent = tempfile::tempdir().map_err(|error| error.to_string())?;
        let output = parent.path().join("export");
        let manifest = export(&scenarios, options().model, &output)?;
        let payload: Value =
            serde_json::from_str(&fs::read_to_string(manifest).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        assert_eq!(payload["scenarios"].as_array().map(Vec::len), Some(13));
        Ok(())
    }

    #[test]
    fn explicit_selection_keeps_order_and_rejects_bad_names() -> Result<(), String> {
        let requested = vec!["tool-call-burst".to_string(), "long-context".to_string()];
        let selected = select(options(), ScenarioSet::Core, &requested)?;
        assert_eq!(
            selected
                .iter()
                .map(|scenario| scenario.id)
                .collect::<Vec<_>>(),
            ["tool-call-burst", "long-context"]
        );
        assert!(select(options(), ScenarioSet::Core, &["missing".to_string()]).is_err());
        Ok(())
    }

    #[test]
    fn agentic_payloads_keep_their_bounded_contracts() -> Result<(), String> {
        let scenarios = catalog(options())?;
        let find = |id| {
            scenarios
                .iter()
                .find(|scenario| scenario.id == id)
                .ok_or_else(|| format!("missing scenario {id}"))
        };

        let long = find("long-context")?;
        assert!(long.sessions.iter().any(|session| {
            session.payloads[0]["messages"][0]["content"]
                .as_str()
                .is_some_and(|content| content.len() > 8_000)
        }));

        let tools = find("large-tool-catalog")?;
        assert_eq!(
            tools.sessions[1].payloads[0]["tools"]
                .as_array()
                .map(Vec::len),
            Some(64)
        );

        let burst = find("tool-call-burst")?;
        assert_eq!(burst.sessions[0].payloads.len(), 9);
        let final_messages = burst.sessions[0].payloads[8]["messages"]
            .as_array()
            .ok_or_else(|| "tool-call burst messages were not an array".to_string())?;
        assert!(
            final_messages.iter().any(|message| {
                message["role"] == "tool" && message["tool_call_id"] == "call_7"
            })
        );

        let growing = find("growing-conversation")?;
        let message_counts = growing.sessions[0]
            .payloads
            .iter()
            .map(|payload| payload["messages"].as_array().map(Vec::len).unwrap_or(0))
            .collect::<Vec<_>>();
        assert!(message_counts.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(find("failure-pressure")?.expected_error_rate.min_rate > 0.0);
        Ok(())
    }
}
