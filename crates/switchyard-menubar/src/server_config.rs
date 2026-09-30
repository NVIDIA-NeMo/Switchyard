// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reads the server config and rewrites one route's algorithm and models.
//!
//! Edits go through `toml_edit`, so comments, formatting, and every table the
//! edit does not touch stay as the user wrote them. The server's own
//! `--dry-run` decides whether the result is valid; this module only builds it.

use std::collections::HashMap;

use toml_edit::{DocumentMut, Item, Table, TableLike, Value};

/// The part a model plays in a route. A choice made for one algorithm carries
/// over to the role with the same tier when the user switches algorithms.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Tier {
    Judge,
    Capable,
    Efficient,
}

/// A route `type` the picker can write.
#[derive(Debug)]
pub struct Algorithm {
    pub kind: &'static str,
    pub summary: &'static str,
    /// Each role's label, tier, and the key path under the route that names
    /// the role's target. The list is empty for `random`, whose roles are the
    /// entries of its `targets` list.
    roles: &'static [(&'static str, Tier, &'static [&'static str])],
    /// Required settings written when a route switches to this type. They
    /// match the values the routing docs use in their examples.
    settings: &'static [(&'static [&'static str], Setting)],
    /// Whether the type accepts a nested `subagents` policy.
    subagents: bool,
}

#[derive(Debug)]
enum Setting {
    Text(&'static str),
    Number(f64),
}

impl Setting {
    fn value(&self) -> Value {
        match self {
            Self::Text(text) => Value::from(*text),
            Self::Number(number) => Value::from(*number),
        }
    }
}

const RANDOM: &str = "random";

/// The route types from the routing docs, in the order the picker lists them.
/// `noop` and the experimental `prefill_router` are left out: one calls no
/// model, and the other needs a checkpoint file rather than models.
pub const ALGORITHMS: &[Algorithm] = &[
    Algorithm {
        kind: "passthrough",
        summary: "Sends every request to one model.",
        roles: &[("Model", Tier::Capable, &["target"])],
        settings: &[],
        subagents: true,
    },
    Algorithm {
        kind: RANDOM,
        summary: "Splits requests at random between the models, for A/B tests and baselines.",
        roles: &[],
        settings: &[],
        subagents: false,
    },
    Algorithm {
        kind: "llm_classifier",
        summary: "A judge model reads each request and picks the capable or the efficient model.",
        roles: &[
            ("Judge", Tier::Judge, &["classifier_target"]),
            ("Capable", Tier::Capable, &["strong_target"]),
            ("Efficient", Tier::Efficient, &["weak_target"]),
        ],
        settings: &[
            (&["mode"], Setting::Text("capability")),
            (&["base_threshold"], Setting::Number(0.5)),
        ],
        subagents: false,
    },
    Algorithm {
        kind: "composite",
        summary: "A judge picks the default model on each user turn, and tool results move \
                  requests between the capable and the efficient model.",
        roles: &[
            ("Judge", Tier::Judge, &["classifier", "target"]),
            ("Capable", Tier::Capable, &["stage", "capable_target"]),
            ("Efficient", Tier::Efficient, &["stage", "efficient_target"]),
        ],
        settings: &[
            (&["classifier", "base_threshold"], Setting::Number(0.5)),
            (
                &["classifier", "classify_trigger"],
                Setting::Text("user_turn"),
            ),
            (&["stage", "confidence_threshold"], Setting::Number(0.5)),
        ],
        subagents: true,
    },
    Algorithm {
        kind: "stage_router",
        summary: "Tool results and agent progress move each request between the capable and \
                  the efficient model.",
        roles: &[
            ("Capable", Tier::Capable, &["capable_target"]),
            ("Efficient", Tier::Efficient, &["efficient_target"]),
        ],
        settings: &[
            (&["picker"], Setting::Text("efficient_first")),
            (&["confidence_threshold"], Setting::Number(0.5)),
        ],
        subagents: true,
    },
    Algorithm {
        kind: "advisor",
        summary: "The executor answers every turn, and a stronger advisor reviews its final \
                  answers.",
        roles: &[
            ("Executor", Tier::Efficient, &["executor_target"]),
            ("Advisor", Tier::Capable, &["advisor_target"]),
        ],
        settings: &[],
        subagents: false,
    },
    Algorithm {
        kind: "plan_execute",
        summary: "The capable model inspects and plans. The efficient model takes over after \
                  the first file edit.",
        roles: &[
            ("Capable", Tier::Capable, &["capable_target"]),
            ("Efficient", Tier::Efficient, &["efficient_target"]),
        ],
        settings: &[],
        subagents: false,
    },
    Algorithm {
        kind: "auto",
        summary: "Switchyard's recommended preset: a stage router that starts on the efficient \
                  model.",
        roles: &[
            ("Capable", Tier::Capable, &["capable_target"]),
            ("Efficient", Tier::Efficient, &["efficient_target"]),
        ],
        settings: &[],
        subagents: false,
    },
];

/// Route keys every type accepts. Switching types keeps these and removes
/// the rest, so settings of the old type cannot fail the new type's checks.
const COMMON_ROUTE_KEYS: [&str; 6] = [
    "id",
    "type",
    "context_window",
    "tool_calling",
    "reasoning",
    "vision",
];

/// One entry under `[routes]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    /// The route's table name in this file. Callers never send it.
    pub key: String,
    /// The public model ID that callers send.
    pub id: String,
    pub kind: String,
}

/// One entry under `[llm_clients]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Client {
    pub name: String,
    pub format: String,
    pub base_url: String,
    pub api_key_env: Option<String>,
    pub forward_auth: bool,
}

/// A model on an LLM client, as a target names it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Choice {
    pub client: String,
    pub model: String,
}

/// A model the algorithm needs, such as the judge.
#[derive(Clone, Debug, PartialEq)]
pub struct Role {
    pub label: String,
    pub tier: Option<Tier>,
    slot: Slot,
}

/// Where a route names a role's target.
#[derive(Clone, Debug, PartialEq)]
enum Slot {
    /// A key path, such as `stage.capable_target`.
    Key(&'static [&'static str]),
    /// An entry of a random route's `targets` list.
    Listed(usize),
}

/// A parsed server config that keeps the original text's layout.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    doc: DocumentMut,
}

impl ServerConfig {
    pub fn parse(text: &str) -> Result<Self, String> {
        text.parse::<DocumentMut>()
            .map(|doc| Self { doc })
            .map_err(|error| format!("parse the server config: {error}"))
    }

    pub fn routes(&self) -> Vec<Route> {
        entries(self.table("routes"))
            .map(|(key, route)| Route {
                key: key.to_string(),
                id: text(route, "id").unwrap_or(key).to_string(),
                kind: text(route, "type").unwrap_or_default().to_string(),
            })
            .collect()
    }

    pub fn clients(&self) -> Vec<Client> {
        entries(self.table("llm_clients"))
            .map(|(name, client)| Client {
                name: name.to_string(),
                format: text(client, "format").unwrap_or_default().to_string(),
                base_url: text(client, "base_url").unwrap_or_default().to_string(),
                api_key_env: text(client, "api_key_env").map(str::to_string),
                forward_auth: client
                    .get("forward_auth")
                    .and_then(Item::as_bool)
                    .unwrap_or(false),
            })
            .collect()
    }

    /// The algorithm a route uses now, or `None` when the picker cannot show
    /// it, such as a custom-mode classifier whose groups are free-form.
    pub fn algorithm(&self, route: &str) -> Option<&'static Algorithm> {
        let route = self.route(route)?;
        if text(route, "mode") == Some("custom") {
            return None;
        }
        let kind = text(route, "type")?;
        ALGORITHMS.iter().find(|algorithm| algorithm.kind == kind)
    }

    /// The roles `algorithm` needs on `route`. A random route keeps its
    /// number of targets, and has at least two.
    pub fn roles(&self, route: &str, algorithm: &Algorithm) -> Vec<Role> {
        if algorithm.kind != RANDOM {
            return algorithm
                .roles
                .iter()
                .map(|(label, tier, path)| Role {
                    label: (*label).to_string(),
                    tier: Some(*tier),
                    slot: Slot::Key(path),
                })
                .collect();
        }
        let listed = self
            .route(route)
            .filter(|route| text(*route, "type") == Some(RANDOM))
            .and_then(|route| {
                route
                    .get("targets")?
                    .as_array()
                    .map(|targets| targets.len())
            })
            .unwrap_or(0);
        (0..listed.max(2))
            .map(|index| Role {
                label: format!("Model {}", index + 1),
                tier: [Tier::Capable, Tier::Efficient].get(index).copied(),
                slot: Slot::Listed(index),
            })
            .collect()
    }

    /// The model each role of the route's current algorithm uses. A role
    /// whose target is missing gets an empty choice.
    pub fn choices(&self, route: &str) -> Vec<Choice> {
        let Some(algorithm) = self.algorithm(route) else {
            return Vec::new();
        };
        self.roles(route, algorithm)
            .iter()
            .map(|role| {
                self.target_at(route, &role.slot)
                    .and_then(|name| self.target(name))
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Returns the config text with `route` switched to `algorithm`, using
    /// one choice per role.
    ///
    /// For each role, in order of preference: keep the route's target when it
    /// already names the model; use another target that names exactly this
    /// model on this client; change the route's target in place when no other
    /// route or role uses it, including a role earlier in this edit; otherwise
    /// add a new target that copies it. A changed or copied target keeps
    /// settings such as `extra_body` and `omit_body_fields`, including
    /// settings meant for the old model. Targets no route uses any more are
    /// left in the file.
    pub fn edit(
        &self,
        route: &str,
        algorithm: &Algorithm,
        choices: &[Choice],
    ) -> Result<String, String> {
        if self.route(route).is_none() {
            return Err(format!("The config has no route {route}."));
        }
        let roles = self.roles(route, algorithm);
        if roles.len() != choices.len() {
            return Err(format!(
                "{} needs {} models, not {}.",
                algorithm.kind,
                roles.len(),
                choices.len()
            ));
        }
        let clients = self.clients();
        for (role, choice) in roles.iter().zip(choices) {
            if choice.model.trim().is_empty() {
                return Err(format!("Choose a model for {}.", role.label));
            }
            if !clients.iter().any(|client| client.name == choice.client) {
                return Err(format!("The config has no llm client {}.", choice.client));
            }
        }

        let current = self.algorithm(route);
        let same_type = current.is_some_and(|current| current.kind == algorithm.kind);
        // When the type changes, each new role starts from the old role with
        // the same tier, so a composite's capable target stays the capable one.
        let by_tier: Vec<(Option<Tier>, &str)> = current
            .map(|current| {
                self.roles(route, current)
                    .iter()
                    .filter_map(|role| Some((role.tier, self.target_at(route, &role.slot)?)))
                    .collect()
            })
            .unwrap_or_default();
        let mut references = self.target_references();

        let mut doc = self.doc.clone();
        let mut names = Vec::with_capacity(roles.len());
        for (role, choice) in roles.iter().zip(choices) {
            let existing = if same_type {
                self.target_at(route, &role.slot)
            } else {
                role.tier.and_then(|tier| {
                    by_tier
                        .iter()
                        .find(|(old, _)| *old == Some(tier))
                        .map(|(_, name)| *name)
                })
            };
            let choice = Choice {
                client: choice.client.clone(),
                model: choice.model.trim().to_string(),
            };
            names.push(choose_target(
                &mut doc,
                &mut references,
                route,
                role,
                &choice,
                existing,
            )?);
        }

        let table = doc
            .get_mut("routes")
            .and_then(Item::as_table_like_mut)
            .and_then(|routes| routes.get_mut(route))
            .and_then(Item::as_table_like_mut)
            .ok_or_else(|| format!("The config has no route {route}."))?;
        if !same_type {
            reset_route(table, algorithm);
        }
        for (role, name) in roles.iter().zip(&names) {
            set_slot(table, &role.slot, name);
        }
        if !same_type {
            for (path, setting) in algorithm.settings {
                set_at(table, path, setting.value());
            }
        }
        Ok(doc.to_string())
    }

    fn table(&self, name: &str) -> Option<&dyn TableLike> {
        self.doc.get(name)?.as_table_like()
    }

    fn route(&self, route: &str) -> Option<&dyn TableLike> {
        self.table("routes")?.get(route)?.as_table_like()
    }

    fn target(&self, name: &str) -> Option<Choice> {
        let target = self.table("targets")?.get(name)?.as_table_like()?;
        Some(Choice {
            client: text(target, "llm_client")?.to_string(),
            model: text(target, "id")?.to_string(),
        })
    }

    fn target_at(&self, route: &str, slot: &Slot) -> Option<&str> {
        let route = self.route(route)?;
        match slot {
            Slot::Key(path) => {
                let (last, parents) = path.split_last()?;
                let mut table = route;
                for key in parents {
                    table = table.get(key)?.as_table_like()?;
                }
                text(table, last)
            }
            Slot::Listed(index) => route.get("targets")?.as_array()?.get(*index)?.as_str(),
        }
    }

    /// Counts how many times each target name appears in any route, nested
    /// policies included. A name that appears once belongs to one role.
    fn target_references(&self) -> HashMap<&str, usize> {
        let mut counts: HashMap<&str, usize> = entries(self.table("targets"))
            .map(|(name, _)| (name, 0))
            .collect();
        if let Some(routes) = self.doc.get("routes") {
            visit_strings(routes, &mut |text| {
                if let Some(count) = counts.get_mut(text) {
                    *count += 1;
                }
            });
        }
        counts
    }
}

/// Picks or writes the target that serves `choice` for one role, and returns
/// its name. Reusing a target adds one to its reference count, so a later
/// role in the same edit copies that target instead of changing it in place.
fn choose_target(
    doc: &mut DocumentMut,
    references: &mut HashMap<&str, usize>,
    route: &str,
    role: &Role,
    choice: &Choice,
    existing: Option<&str>,
) -> Result<String, String> {
    let targets = doc
        .get_mut("targets")
        .and_then(Item::as_table_like_mut)
        .ok_or("The config has no [targets] table.")?;

    if let Some(name) = existing
        && targets
            .get(name)
            .is_some_and(|target| names_choice(target, choice))
    {
        return Ok(name.to_string());
    }
    if let Some((name, _)) = targets
        .iter()
        .find(|(_, target)| names_choice(target, choice))
    {
        if let Some(count) = references.get_mut(name) {
            *count += 1;
        }
        return Ok(name.to_string());
    }
    if let Some(name) = existing
        && references.get(name) == Some(&1)
        && let Some(target) = targets.get_mut(name).and_then(Item::as_table_like_mut)
    {
        set_value(target, "llm_client", Value::from(choice.client.as_str()));
        set_value(target, "id", Value::from(choice.model.as_str()));
        return Ok(name.to_string());
    }

    let mut table = Table::new();
    if let Some(source) = existing
        .and_then(|name| targets.get(name))
        .and_then(Item::as_table_like)
    {
        for (key, item) in source.iter() {
            table.insert(key, item.clone());
        }
    }
    set_value(
        &mut table,
        "llm_client",
        Value::from(choice.client.as_str()),
    );
    set_value(&mut table, "id", Value::from(choice.model.as_str()));
    let base = format!("{route}_{}", role.label.to_lowercase().replace(' ', "_"));
    let mut name = base.clone();
    let mut suffix = 1;
    while targets.contains_key(&name) {
        suffix += 1;
        name = format!("{base}_{suffix}");
    }
    targets.insert(&name, Item::Table(table));
    Ok(name)
}

/// Whether a target names exactly this model on this client.
fn names_choice(target: &Item, choice: &Choice) -> bool {
    target.as_table_like().is_some_and(|target| {
        text(target, "llm_client") == Some(choice.client.as_str())
            && text(target, "id") == Some(choice.model.as_str())
    })
}

/// Removes the old type's settings and sets the new `type`.
fn reset_route(route: &mut dyn TableLike, algorithm: &Algorithm) {
    let kept =
        |key: &str| COMMON_ROUTE_KEYS.contains(&key) || (algorithm.subagents && key == "subagents");
    let stale: Vec<String> = route
        .iter()
        .map(|(key, _)| key)
        .filter(|key| !kept(key))
        .map(str::to_string)
        .collect();
    for key in stale {
        route.remove(&key);
    }
    set_value(route, "type", Value::from(algorithm.kind));
}

fn set_slot(route: &mut dyn TableLike, slot: &Slot, name: &str) {
    match slot {
        Slot::Key(path) => set_at(route, path, Value::from(name)),
        Slot::Listed(index) => {
            if !route.get("targets").is_some_and(Item::is_array) {
                route.insert("targets", toml_edit::value(toml_edit::Array::new()));
            }
            if let Some(targets) = route.get_mut("targets").and_then(Item::as_array_mut) {
                if *index < targets.len() {
                    targets.replace(*index, name);
                } else {
                    targets.push(name);
                }
            }
        }
    }
}

/// Sets a value at a key path, creating missing tables on the way.
fn set_at(table: &mut dyn TableLike, path: &[&str], value: Value) {
    match path {
        [] => {}
        [key] => set_value(table, key, value),
        [key, rest @ ..] => {
            if !table.get(key).is_some_and(Item::is_table_like) {
                table.insert(key, Item::Table(Table::new()));
            }
            if let Some(child) = table.get_mut(key).and_then(Item::as_table_like_mut) {
                set_at(child, rest, value);
            }
        }
    }
}

/// Replaces a value, keeping the spacing and trailing comment around it.
fn set_value(table: &mut dyn TableLike, key: &str, mut value: Value) {
    if let Some(old) = table.get_mut(key).and_then(Item::as_value_mut) {
        *value.decor_mut() = old.decor().clone();
        *old = value;
    } else {
        table.insert(key, Item::Value(value));
    }
}

fn entries(table: Option<&dyn TableLike>) -> impl Iterator<Item = (&str, &dyn TableLike)> {
    table
        .into_iter()
        .flat_map(|table| table.iter())
        .filter_map(|(key, item)| Some((key, item.as_table_like()?)))
}

fn text<'a>(table: &'a dyn TableLike, key: &str) -> Option<&'a str> {
    table.get(key)?.as_str()
}

fn visit_strings(item: &Item, visit: &mut dyn FnMut(&str)) {
    match item {
        Item::Value(value) => visit_value(value, visit),
        Item::Table(table) => table
            .iter()
            .for_each(|(_, item)| visit_strings(item, visit)),
        Item::ArrayOfTables(tables) => tables
            .iter()
            .flat_map(Table::iter)
            .for_each(|(_, item)| visit_strings(item, visit)),
        Item::None => {}
    }
}

fn visit_value(value: &Value, visit: &mut dyn FnMut(&str)) {
    match value {
        Value::String(text) => visit(text.value()),
        Value::Array(array) => array.iter().for_each(|value| visit_value(value, visit)),
        Value::InlineTable(table) => table
            .iter()
            .for_each(|(_, value)| visit_value(value, visit)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A composite route with a GPT judge on a Responses client and Claude on
    /// a Chat Completions client, both on one gateway.
    const GATEWAY: &str = r#"schema_version = 1

[llm_clients.gateway]
format = "openai_responses"
base_url = "https://gateway.example.com/v1"
forward_auth = true

# Claude caches prompts on the chat endpoint.
[llm_clients.gateway_chat]
format = "openai_chat"
base_url = "https://gateway.example.com/v1"
forward_auth = true

# The gateway rejects any thinking setting for these models.
[targets.gateway_capable]
id = "claude-opus-5-5"
llm_client = "gateway_chat"  # chat, for caching
omit_body_fields = ["reasoning_effort"]

[targets.gateway_efficient]
id = "claude-sonnet-5"
llm_client = "gateway_chat"
omit_body_fields = ["reasoning_effort"]

[targets.gateway_judge]
id = "gpt-5.6-terra"
llm_client = "gateway"

[routes.gateway]
id = "switchyard-gateway"
type = "composite"
context_window = 200000

[routes.gateway.classifier]
target = "gateway_judge"
base_threshold = 0.5
classify_trigger = "user_turn"
message_hash_fallback = true

[routes.gateway.stage]
capable_target = "gateway_capable"
efficient_target = "gateway_efficient"
confidence_threshold = 0.5
"#;

    fn choice(client: &str, model: &str) -> Choice {
        Choice {
            client: client.to_string(),
            model: model.to_string(),
        }
    }

    fn algorithm(kind: &str) -> &'static Algorithm {
        ALGORITHMS
            .iter()
            .find(|algorithm| algorithm.kind == kind)
            .expect("known algorithm")
    }

    /// Edits the config and checks the result with the server's parser.
    fn edit(text: &str, route: &str, kind: &str, choices: &[Choice]) -> String {
        let edited = ServerConfig::parse(text)
            .expect("parse")
            .edit(route, algorithm(kind), choices)
            .expect("edit");
        if let Err(error) = switchyard_runner::Runner::from_toml(&edited) {
            panic!("the server rejects the edited config: {error}\n{edited}");
        }
        edited
    }

    #[test]
    fn every_algorithm_writes_a_config_the_server_accepts() {
        let gateway = ServerConfig::parse(GATEWAY).expect("parse");
        let models = [
            choice("gateway", "gpt-5.6-terra"),
            choice("gateway", "gpt-5.6-sol"),
            choice("gateway_chat", "claude-sonnet-5"),
        ];
        for algorithm in ALGORITHMS {
            let roles = gateway.roles("gateway", algorithm).len();

            let edited = edit(GATEWAY, "gateway", algorithm.kind, &models[3 - roles..]);

            let config = ServerConfig::parse(&edited).expect("parse");
            assert_eq!(
                config.algorithm("gateway").map(|a| a.kind),
                Some(algorithm.kind)
            );
            assert_eq!(config.choices("gateway"), models[3 - roles..]);
        }
    }

    #[test]
    fn a_target_one_role_reuses_is_not_changed_for_another_role() {
        // Capable moves to the efficient model, and Efficient to a new one.
        let chosen = [
            choice("gateway", "gpt-5.6-terra"),
            choice("gateway_chat", "claude-sonnet-5"),
            choice("gateway", "gpt-5.6-luna"),
        ];

        let edited = edit(GATEWAY, "gateway", "composite", &chosen);

        assert_eq!(
            ServerConfig::parse(&edited)
                .expect("parse")
                .choices("gateway"),
            chosen
        );
    }

    #[test]
    fn switching_models_and_back_restores_the_file() {
        let gpt = edit(
            GATEWAY,
            "gateway",
            "composite",
            &[
                choice("gateway", "gpt-5.6-terra"),
                choice("gateway", "gpt-5.6-sol"),
                choice("gateway", "gpt-5.6-luna"),
            ],
        );

        let config = ServerConfig::parse(&gpt).expect("parse");
        assert_eq!(
            config.choices("gateway"),
            [
                choice("gateway", "gpt-5.6-terra"),
                choice("gateway", "gpt-5.6-sol"),
                choice("gateway", "gpt-5.6-luna"),
            ]
        );
        assert!(
            gpt.contains(
                "[targets.gateway_capable]\nid = \"gpt-5.6-sol\"\nllm_client = \"gateway\"  # chat, for caching\nomit_body_fields = [\"reasoning_effort\"]"
            ),
            "the target keeps its other settings and comments:\n{gpt}"
        );

        let claude = edit(
            &gpt,
            "gateway",
            "composite",
            &[
                choice("gateway", "gpt-5.6-terra"),
                choice("gateway_chat", "claude-opus-5-5"),
                choice("gateway_chat", "claude-sonnet-5"),
            ],
        );
        assert_eq!(claude, GATEWAY);
    }

    #[test]
    fn changing_the_algorithm_replaces_only_the_route_settings() {
        let edited = edit(
            GATEWAY,
            "gateway",
            "stage_router",
            &[
                choice("gateway_chat", "claude-opus-5-5"),
                choice("gateway_chat", "claude-sonnet-5"),
            ],
        );

        let config = ServerConfig::parse(&edited).expect("parse");
        assert_eq!(
            config.algorithm("gateway").map(|a| a.kind),
            Some("stage_router")
        );
        let route = config.route("gateway").expect("route");
        assert_eq!(text(route, "id"), Some("switchyard-gateway"));
        assert_eq!(
            route.get("context_window").and_then(Item::as_integer),
            Some(200000)
        );
        assert_eq!(text(route, "capable_target"), Some("gateway_capable"));
        assert_eq!(text(route, "efficient_target"), Some("gateway_efficient"));
        assert_eq!(text(route, "picker"), Some("efficient_first"));
        assert!(route.get("classifier").is_none() && route.get("stage").is_none());
        assert!(
            edited.contains("[targets.gateway_judge]"),
            "unused targets stay in the file"
        );
    }

    #[test]
    fn a_target_another_route_uses_is_copied_not_changed() {
        let shared = format!(
            "{GATEWAY}\n[routes.direct]\nid = \"direct\"\ntype = \"passthrough\"\ntarget = \"gateway_capable\"\n"
        );

        let edited = edit(
            &shared,
            "gateway",
            "composite",
            &[
                choice("gateway", "gpt-5.6-terra"),
                choice("gateway", "gpt-5.6-sol"),
                choice("gateway_chat", "claude-sonnet-5"),
            ],
        );

        let config = ServerConfig::parse(&edited).expect("parse");
        assert_eq!(
            config.choices("direct"),
            [choice("gateway_chat", "claude-opus-5-5")]
        );
        assert_eq!(
            config.target_at("gateway", &Slot::Key(&["stage", "capable_target"])),
            Some("gateway_capable_2")
        );
        assert!(
            edited.contains(
                "[targets.gateway_capable_2]\nid = \"gpt-5.6-sol\"\nllm_client = \"gateway\"  # chat, for caching\nomit_body_fields = [\"reasoning_effort\"]"
            ),
            "the copy keeps the shared target's settings:\n{edited}"
        );
    }

    #[test]
    fn rejects_a_missing_model_or_an_unknown_client() {
        let config = ServerConfig::parse(GATEWAY).expect("parse");
        for (capable, expected) in [
            (choice("gateway", " "), "Choose a model for Capable."),
            (
                choice("typo", "gpt-5.6-sol"),
                "The config has no llm client typo.",
            ),
        ] {
            let error = config
                .edit(
                    "gateway",
                    algorithm("composite"),
                    &[
                        choice("gateway", "gpt-5.6-terra"),
                        capable,
                        choice("gateway", "gpt-5.6-luna"),
                    ],
                )
                .expect_err("invalid choice");

            assert_eq!(error, expected);
        }
    }
}
