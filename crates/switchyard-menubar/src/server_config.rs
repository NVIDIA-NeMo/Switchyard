// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module reads the server config and rewrites one route's algorithm and models.
//!
//! Edits go through `toml_edit`, so comments, formatting, and every table the
//! edit does not touch stay as the user wrote them. The server's own
//! `--dry-run` decides whether the result is valid; this module only builds it.

use std::collections::HashMap;

use toml_edit::{DocumentMut, Item, Table, TableLike, Value};

/// Tier identifies the part a model plays in a route. A choice made for one algorithm carries
/// over to the role with the same tier when the user switches algorithms.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Tier {
    Judge,
    Capable,
    Efficient,
}

/// Algorithm describes a route `type` the apps can write.
#[derive(Debug)]
pub struct Algorithm {
    pub kind: &'static str,
    /// This field sets the name shown in both apps.
    pub title: &'static str,
    pub summary: &'static str,
    /// This field lists roles in display order. The list is empty for
    /// `random`, whose roles are the entries of its `targets` list.
    roles: &'static [RoleSpec],
    /// This field lists settings written when a route switches to this type. They
    /// match the values the routing docs use in their examples.
    settings: &'static [(&'static [&'static str], Setting)],
    /// This field records whether the type accepts a nested `subagents` policy.
    subagents: bool,
}

/// RoleSpec describes one role of an algorithm.
#[derive(Debug)]
struct RoleSpec {
    label: &'static str,
    /// This field explains what the role does.
    hint: &'static str,
    tier: Tier,
    /// This field names the key path to the role's target under the route.
    path: &'static [&'static str],
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

/// ALGORITHMS lists supported route types in display order.
/// `noop` and the experimental `prefill_router` are left out: one calls no
/// model, and the other needs a checkpoint file rather than models.
pub const ALGORITHMS: &[Algorithm] = &[
    Algorithm {
        kind: "passthrough",
        title: "Single model",
        summary: "This algorithm sends every request to one model.",
        roles: &[RoleSpec {
            label: "Model",
            hint: "This model answers every request.",
            tier: Tier::Capable,
            path: &["target"],
        }],
        settings: &[],
        subagents: true,
    },
    Algorithm {
        kind: RANDOM,
        title: "Random split",
        summary: "This algorithm splits requests at random between models for A/B tests and baselines.",
        roles: &[],
        settings: &[],
        subagents: false,
    },
    Algorithm {
        kind: "llm_classifier",
        title: "Judge per request",
        summary: "A judge model reads each request and picks the capable or the efficient model.",
        roles: &[
            RoleSpec {
                label: "Judge",
                hint: "The judge reads each request and picks a model.",
                tier: Tier::Judge,
                path: &["classifier_target"],
            },
            RoleSpec {
                label: "Capable",
                hint: "The capable model answers the hard requests.",
                tier: Tier::Capable,
                path: &["strong_target"],
            },
            RoleSpec {
                label: "Efficient",
                hint: "The efficient model answers the easy requests.",
                tier: Tier::Efficient,
                path: &["weak_target"],
            },
        ],
        settings: &[
            (&["mode"], Setting::Text("capability")),
            (&["base_threshold"], Setting::Number(0.5)),
        ],
        subagents: false,
    },
    Algorithm {
        kind: "composite",
        title: "Judge plus stage router",
        summary: "A judge picks the starting model on each user turn, and tool results move \
                  requests between the capable and the efficient model.",
        roles: &[
            RoleSpec {
                label: "Judge",
                hint: "The judge picks the starting model on each user turn.",
                tier: Tier::Judge,
                path: &["classifier", "target"],
            },
            RoleSpec {
                label: "Capable",
                hint: "The capable model takes the hard steps.",
                tier: Tier::Capable,
                path: &["stage", "capable_target"],
            },
            RoleSpec {
                label: "Efficient",
                hint: "The efficient model takes the easy steps.",
                tier: Tier::Efficient,
                path: &["stage", "efficient_target"],
            },
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
        title: "Stage router",
        summary: "Tool results and agent progress move each request between the capable and \
                  the efficient model.",
        roles: &[
            RoleSpec {
                label: "Capable",
                hint: "The capable model takes the hard steps.",
                tier: Tier::Capable,
                path: &["capable_target"],
            },
            RoleSpec {
                label: "Efficient",
                hint: "The efficient model takes the easy steps.",
                tier: Tier::Efficient,
                path: &["efficient_target"],
            },
        ],
        settings: &[
            (&["picker"], Setting::Text("efficient_first")),
            (&["confidence_threshold"], Setting::Number(0.5)),
        ],
        subagents: true,
    },
    Algorithm {
        kind: "advisor",
        title: "Executor and advisor",
        summary: "The executor answers every turn, and a stronger advisor reviews its final \
                  answers.",
        roles: &[
            RoleSpec {
                label: "Executor",
                hint: "The executor answers every turn.",
                tier: Tier::Efficient,
                path: &["executor_target"],
            },
            RoleSpec {
                label: "Advisor",
                hint: "The advisor reviews the executor's final answers.",
                tier: Tier::Capable,
                path: &["advisor_target"],
            },
        ],
        settings: &[],
        subagents: false,
    },
    Algorithm {
        kind: "plan_execute",
        title: "Plan, then execute",
        summary: "The capable model inspects and plans. The efficient model takes over after \
                  the first file edit.",
        roles: &[
            RoleSpec {
                label: "Capable",
                hint: "The capable model inspects the code and plans.",
                tier: Tier::Capable,
                path: &["capable_target"],
            },
            RoleSpec {
                label: "Efficient",
                hint: "The efficient model takes over after the first edit.",
                tier: Tier::Efficient,
                path: &["efficient_target"],
            },
        ],
        settings: &[],
        subagents: false,
    },
    Algorithm {
        kind: "auto",
        title: "Recommended preset",
        summary: "This stage router starts each request on the efficient model.",
        roles: &[
            RoleSpec {
                label: "Capable",
                hint: "The capable model takes the hard steps.",
                tier: Tier::Capable,
                path: &["capable_target"],
            },
            RoleSpec {
                label: "Efficient",
                hint: "The efficient model starts each request.",
                tier: Tier::Efficient,
                path: &["efficient_target"],
            },
        ],
        settings: &[],
        subagents: false,
    },
];

/// COMMON_ROUTE_KEYS lists keys that every route type accepts. Switching types keeps these and removes
/// the rest, so settings of the old type cannot fail the new type's checks.
const COMMON_ROUTE_KEYS: [&str; 6] = [
    "id",
    "type",
    "context_window",
    "tool_calling",
    "reasoning",
    "vision",
];

/// REQUEST_SETTINGS lists target settings that change the request body. The server keeps one
/// target per model on a client, so it rejects two targets that name the
/// same model on the same client with different values for these.
const REQUEST_SETTINGS: [&str; 3] = ["omit_body_fields", "reasoning_effort", "extra_body"];

/// TARGET_SETTINGS lists settings that a route takes on when it uses the target.
const TARGET_SETTINGS: [&str; 4] = [
    "system_prompt",
    "reasoning_effort",
    "extra_body",
    "omit_body_fields",
];

/// Route describes one entry under `[routes]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    /// This field names the route's table in this file. Callers never send it.
    pub key: String,
    /// This field sets the public model ID that callers send.
    pub id: String,
    pub kind: String,
}

/// Client describes one entry under `[llm_clients]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Client {
    pub name: String,
    pub format: String,
    pub base_url: String,
    pub api_key_env: Option<String>,
    pub forward_auth: bool,
}

impl Client {
    /// This function returns the host in `base_url`, such as `chatgpt.com`. A port stays.
    pub fn host(&self) -> &str {
        let rest = self
            .base_url
            .split_once("://")
            .map_or(self.base_url.as_str(), |(_, rest)| rest);
        let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
        authority.rsplit('@').next().unwrap_or(authority)
    }
}

/// Choice identifies a model and the client that serves it.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Choice {
    pub client: String,
    pub model: String,
}

/// Role describes a model the algorithm needs, such as the judge.
#[derive(Clone, Debug, PartialEq)]
pub struct Role {
    pub label: String,
    /// This field explains what the role does.
    pub hint: &'static str,
    pub tier: Option<Tier>,
    slot: Slot,
}

/// Slot identifies where a route names a role's target.
#[derive(Clone, Debug, PartialEq)]
enum Slot {
    /// Key identifies a key path, such as `stage.capable_target`.
    Key(&'static [&'static str]),
    /// Listed identifies an entry of a random route's `targets` list.
    Listed(usize),
}

impl Slot {
    /// This function returns the slot's key path under the route, such as
    /// `stage.capable_target`.
    fn path(&self) -> String {
        match self {
            Self::Key(path) => path.join("."),
            Self::Listed(_) => "targets".to_string(),
        }
    }
}

/// Edited stores the result of [`ServerConfig::edit`].
#[derive(Debug)]
pub struct Edited {
    /// This field stores the new config text.
    pub text: String,
    /// This field stores user-facing notes about the edit.
    pub notes: Vec<String>,
}

/// ServerConfig keeps the parsed config and the original text's layout.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    doc: DocumentMut,
    /// This field maps tables inside generated blocks to each block's first marker.
    generated: HashMap<String, String>,
}

impl ServerConfig {
    pub fn parse(text: &str) -> Result<Self, String> {
        text.parse::<DocumentMut>()
            .map(|doc| Self {
                doc,
                generated: generated_tables(text),
            })
            .map_err(|error| {
                format!(
                    "Could not parse the server config. Fix this TOML error and try again:\n\
                     {error}"
                )
            })
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

    pub fn remove(&self, route: &str) -> Result<Edited, String> {
        let mut doc = self.doc.clone();
        let routes = doc
            .get_mut("routes")
            .and_then(Item::as_table_like_mut)
            .ok_or("No routes are configured.")?;
        let removed = routes
            .remove(route)
            .ok_or("The route was already removed. Refresh and choose it again.")?;
        retain_block_markers(&mut doc, &removed);
        Ok(Edited {
            text: doc.to_string(),
            notes: vec![
                "Removed the route. Its targets and endpoint settings are preserved.".into(),
            ],
        })
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

    /// This function returns the text of the marker that opens the block another tool
    /// writes around `[table.key]`, when the table sits inside one. That tool
    /// overwrites the table the next time it runs.
    pub fn generated_by(&self, table: &str, key: &str) -> Option<&str> {
        self.generated
            .get(&format!("{table}.{key}"))
            .map(String::as_str)
    }

    /// This function returns the model IDs that targets name on `client`, sorted, each once.
    pub fn models_on(&self, client: &str) -> Vec<String> {
        let mut models: Vec<String> = entries(self.table("targets"))
            .filter(|(_, target)| text(*target, "llm_client") == Some(client))
            .filter_map(|(_, target)| text(target, "id").map(str::to_string))
            .collect();
        models.sort();
        models.dedup();
        models
    }

    /// This function returns the algorithm a route uses now, or `None` when the picker
    /// cannot show it, such as a custom-mode classifier whose groups are
    /// free-form.
    pub fn algorithm(&self, route: &str) -> Option<&'static Algorithm> {
        let route = self.route(route)?;
        if text(route, "mode") == Some("custom") {
            return None;
        }
        let kind = text(route, "type")?;
        ALGORITHMS.iter().find(|algorithm| algorithm.kind == kind)
    }

    /// This function returns the roles that `algorithm` needs on `route`. A random route
    /// keeps its number of targets, and has at least two.
    pub fn roles(&self, route: &str, algorithm: &Algorithm) -> Vec<Role> {
        if algorithm.kind != RANDOM {
            return algorithm
                .roles
                .iter()
                .map(|spec| Role {
                    label: spec.label.to_string(),
                    hint: spec.hint,
                    tier: Some(spec.tier),
                    slot: Slot::Key(spec.path),
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
                hint: "This model gets a share of the requests.",
                tier: [Tier::Capable, Tier::Efficient].get(index).copied(),
                slot: Slot::Listed(index),
            })
            .collect()
    }

    /// This function returns the model that each role of the route's current algorithm
    /// uses. A role whose target is missing gets an empty choice.
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

    /// This function returns the config text with `route` switched to `algorithm`, using
    /// one choice per role, and notes about the edit.
    ///
    /// Each role keeps its target if it already names the selected model.
    /// Otherwise, this function reuses a matching target with equivalent settings,
    /// or with request settings that prevent the server from accepting a duplicate.
    /// If neither exists, this function edits an unshared target or copies a shared one.
    /// Roles earlier in this edit also count as users of a target. A changed or copied target keeps settings
    /// such as `extra_body` and `omit_body_fields`, including settings meant
    /// for the old model. Targets no route uses any more are left in the file.
    pub fn edit(
        &self,
        route: &str,
        algorithm: &Algorithm,
        choices: &[Choice],
    ) -> Result<Edited, String> {
        if self.route(route).is_none() {
            return Err(format!("The config has no [routes.{route}] table."));
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
                return Err(format!("Pick a model for {}.", role.label));
            }
            if choice.client.is_empty() {
                return Err(format!("Pick an endpoint for {}.", role.label));
            }
            if !clients.iter().any(|client| client.name == choice.client) {
                return Err(format!(
                    "The config has no LLM client named \"{}\".",
                    choice.client
                ));
            }
        }

        let current = self.algorithm(route);
        let same_type = current.is_some_and(|current| current.kind == algorithm.kind);
        let old_roles = current
            .map(|current| self.roles(route, current))
            .unwrap_or_default();
        // When the type changes, each new role starts from the old role with
        // the same tier, so a composite's capable target stays the capable one.
        let by_tier: Vec<(Option<Tier>, &str)> = old_roles
            .iter()
            .filter_map(|role| Some((role.tier, self.target_at(route, &role.slot)?)))
            .collect();
        let mut users = self.target_users();
        let place = self.new_target_position();

        let mut doc = self.doc.clone();
        let mut names = Vec::with_capacity(roles.len());
        let mut notes = Vec::new();
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
            let (name, moved_from) =
                choose_target(&mut doc, &mut users, route, role, &choice, existing, place)?;
            if let Some(old) = moved_from
                && let Some(note) = kept_settings_note(&doc, &clients, &name, &old, &choice)
            {
                notes.push(note);
            }
            let mut others: Vec<&str> = users
                .get(name.as_str())
                .into_iter()
                .flatten()
                .copied()
                .filter(|other| *other != route)
                .collect();
            others.dedup();
            if Some(name.as_str()) != existing && !others.is_empty() {
                let plural = if others.len() == 1 { "" } else { "s" };
                notes.push(format!(
                    "{} now shares [targets.{name}] with route{plural} {}.",
                    role.label,
                    others.join(", ")
                ));
            }
            names.push(name);
        }
        notes.extend(self.login_note(route, choices));
        notes.extend(self.generated_note(route, &names));

        let table = doc
            .get_mut("routes")
            .and_then(Item::as_table_like_mut)
            .and_then(|routes| routes.get_mut(route))
            .and_then(Item::as_table_like_mut)
            .ok_or_else(|| format!("The config has no [routes.{route}] table."))?;
        if !same_type {
            let old_slots: Vec<String> = old_roles.iter().map(|role| role.slot.path()).collect();
            let removed = reset_route(table, algorithm, &old_slots);
            if !removed.is_empty() {
                notes.push(format!(
                    "Switching to {} removed these settings from [routes.{route}]: {}. The \
                     backup still has them.",
                    algorithm.kind,
                    removed.join(", ")
                ));
            }
        }
        for (role, name) in roles.iter().zip(&names) {
            set_slot(table, &role.slot, name);
        }
        if !same_type {
            for (path, setting) in algorithm.settings {
                set_at(table, path, setting.value());
            }
        }
        Ok(Edited {
            text: doc.to_string(),
            notes,
        })
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

    /// Returns, for each target name, the routes that name it: one entry per
    /// mention in any route, nested policies included. A target with one
    /// entry belongs to one role.
    fn target_users(&self) -> HashMap<&str, Vec<&str>> {
        let mut users: HashMap<&str, Vec<&str>> = entries(self.table("targets"))
            .map(|(name, _)| (name, Vec::new()))
            .collect();
        for (route, item) in self
            .table("routes")
            .into_iter()
            .flat_map(|routes| routes.iter())
        {
            visit_strings(item, &mut |text| {
                if let Some(routes) = users.get_mut(text) {
                    routes.push(route);
                }
            });
        }
        users
    }

    /// This function returns the file position for a new `[targets.*]` table. The table
    /// goes after the last target outside every generated block, so that the
    /// next rewrite of a block cannot delete it. When every target sits
    /// inside a block, any other place is next to a generated target, so the
    /// table goes before the first table of the file.
    fn new_target_position(&self) -> Option<isize> {
        self.table("targets")?
            .iter()
            .filter(|(key, _)| self.generated_by("targets", key).is_none())
            .filter_map(|(_, item)| item.as_table()?.position())
            .max()
            .or_else(|| {
                self.generated
                    .keys()
                    .any(|key| key.starts_with("targets."))
                    .then_some(-1)
            })
    }

    /// This function says so when the edit sends the callers' logins to another host. A
    /// client with `forward_auth` sends each caller's own login, and a caller
    /// that has none for the new host gets HTTP 401.
    fn login_note(&self, route: &str, choices: &[Choice]) -> Option<String> {
        let clients = self.clients();
        let before = login_hosts(&clients, &self.choices(route));
        let after = login_hosts(&clients, choices);
        if after.iter().all(|host| before.contains(host)) {
            return None;
        }
        let was = if before.is_empty() {
            String::new()
        } else {
            format!(" It sent them to {} before.", before.join(" and "))
        };
        Some(format!(
            "[routes.{route}] now sends each caller's own login to {}.{was} A caller that has \
             no login for the new host gets HTTP 401.",
            after.join(" and ")
        ))
    }

    /// This function returns a warning when the route sits inside a block that another tool
    /// writes, which overwrites the route, or when the route uses targets
    /// inside such a block, which that tool can change or remove.
    fn generated_note(&self, route: &str, targets: &[String]) -> Option<String> {
        if let Some(block) = self.generated_by("routes", route) {
            return Some(format!(
                "[routes.{route}] sits inside a block that another tool writes. That tool \
                 overwrites this change the next time it runs. Block marker: \"{block}\"."
            ));
        }
        let inside: Vec<String> = targets
            .iter()
            .filter(|name| self.generated_by("targets", name).is_some())
            .map(|name| format!("[targets.{name}]"))
            .collect();
        let (verb, pronoun) = if inside.len() == 1 {
            ("sits", "it")
        } else {
            ("sit", "them")
        };
        (!inside.is_empty()).then(|| {
            format!(
                "[routes.{route}] uses {}, which {verb} inside a block that another tool \
                 writes. The next time that tool runs, it can change or remove {pronoun}.",
                inside.join(", ")
            )
        })
    }
}

/// This function picks or writes the target that serves `choice` for one role. Returns its
/// name, and the model that the target named before when the edit changed
/// the target in place or copied it. Reusing a target adds the route to the
/// target's users, so a later role in the same edit copies that target
/// instead of changing it in place.
/// A new target goes at file position `place`, where the next rewrite of a
/// generated block cannot delete it.
fn choose_target<'a>(
    doc: &mut DocumentMut,
    users: &mut HashMap<&'a str, Vec<&'a str>>,
    route: &'a str,
    role: &Role,
    choice: &Choice,
    existing: Option<&str>,
    place: Option<isize>,
) -> Result<(String, Option<Choice>), String> {
    let targets = doc
        .get_mut("targets")
        .and_then(Item::as_table_like_mut)
        .ok_or("The config has no [targets] table.")?;

    let current = existing
        .and_then(|name| targets.get(name))
        .and_then(Item::as_table_like);
    if let Some(name) = existing
        && current.is_some_and(|target| names_choice(target, choice))
    {
        return Ok((name.to_string(), None));
    }
    let old = current.and_then(|target| {
        Some(Choice {
            client: text(target, "llm_client")?.to_string(),
            model: text(target, "id")?.to_string(),
        })
    });
    if let Some(name) = reusable(&*targets, current, choice).map(str::to_string) {
        if let Some(routes) = users.get_mut(name.as_str()) {
            routes.push(route);
        }
        return Ok((name, None));
    }
    if let Some(name) = existing
        && users.get(name).is_some_and(|routes| routes.len() == 1)
        && let Some(target) = targets.get_mut(name).and_then(Item::as_table_like_mut)
    {
        set_value(target, "id", Value::from(choice.model.as_str()));
        set_value(target, "llm_client", Value::from(choice.client.as_str()));
        return Ok((name.to_string(), old));
    }

    let mut table = Table::new();
    if let Some(source) = existing
        .and_then(|name| targets.get(name))
        .and_then(Item::as_table_like)
    {
        for (key, item) in source.iter() {
            let mut item = item.clone();
            unplace(&mut item);
            table.insert(key, item);
        }
    }
    set_value(&mut table, "id", Value::from(choice.model.as_str()));
    set_value(
        &mut table,
        "llm_client",
        Value::from(choice.client.as_str()),
    );
    let base = format!("{route}_{}", role.label.to_lowercase().replace(' ', "_"));
    let mut name = base.clone();
    let mut suffix = 1;
    while targets.contains_key(&name) {
        suffix += 1;
        name = format!("{base}_{suffix}");
    }
    // A table with no position is written after the last [targets.*] table in
    // the file, which can sit inside a block that another tool writes.
    table.set_position(place);
    targets.insert(&name, Item::Table(table));
    Ok((name, old))
}

/// This function returns a note about the request settings that a target kept when the
/// edit moved it from `old` to `new`, a model of another family or on a
/// client with another request format. A setting meant for the old model can
/// make the provider reject the new model's requests, and `--dry-run` does
/// not catch that. So the note lists the kept settings, or says that the
/// target has none, and the user decides.
fn kept_settings_note(
    doc: &DocumentMut,
    clients: &[Client],
    name: &str,
    old: &Choice,
    new: &Choice,
) -> Option<String> {
    let format = |name: &str| {
        clients
            .iter()
            .find(|client| client.name == name)
            .map(|client| client.format.as_str())
    };
    if family(&old.model) == family(&new.model) && format(&old.client) == format(&new.client) {
        return None;
    }
    let target = doc
        .get("targets")?
        .as_table_like()?
        .get(name)?
        .as_table_like()?;
    let kept: Vec<String> = REQUEST_SETTINGS
        .iter()
        .filter_map(|key| Some(format!("{key} = {}", setting(target, key)?)))
        .collect();
    Some(if kept.is_empty() {
        format!(
            "[targets.{name}] now names {} and has no omit_body_fields, reasoning_effort, or \
             extra_body. Check whether the new model needs one of them.",
            new.model
        )
    } else {
        format!(
            "[targets.{name}] now names {} and kept {} from {}. Check that the new model \
             accepts these settings.",
            new.model,
            kept.join(", "),
            old.model
        )
    })
}

/// This function returns a model's family: the first word of the last part of its ID,
/// such as `gpt` for `openai/gpt-5.6-sol`.
fn family(model: &str) -> String {
    let name = model.rsplit('/').next().unwrap_or(model);
    name.split(['-', '.', '_', ':'])
        .next()
        .unwrap_or(name)
        .to_lowercase()
}

/// This function returns another target that names `choice` and that a role whose target
/// is `current` may use. The role takes on that target's settings. So it
/// uses a target with the same settings as `current`, or else a target whose
/// request settings differ from `current`, because the server would reject a
/// second target for this model with other request settings.
fn reusable<'t>(
    targets: &'t dyn TableLike,
    current: Option<&dyn TableLike>,
    choice: &Choice,
) -> Option<&'t str> {
    let same = |target: &dyn TableLike, keys: &[&str]| {
        keys.iter()
            .all(|key| setting(target, key) == current.and_then(|current| setting(current, key)))
    };
    let candidates: Vec<(&str, &dyn TableLike)> = targets
        .iter()
        .filter_map(|(name, target)| Some((name, target.as_table_like()?)))
        .filter(|(_, target)| names_choice(*target, choice))
        .collect();
    candidates
        .iter()
        .find(|(_, target)| same(*target, &TARGET_SETTINGS))
        .or_else(|| {
            candidates
                .iter()
                .find(|(_, target)| !same(*target, &REQUEST_SETTINGS))
        })
        .map(|(name, _)| *name)
}

/// This function returns a target's setting as plain data, so the same value written in
/// another layout compares equal.
fn setting(target: &dyn TableLike, key: &str) -> Option<toml::Value> {
    let mut table = Table::new();
    table.insert(key, target.get(key)?.clone());
    DocumentMut::from(table)
        .to_string()
        .parse::<toml::Table>()
        .ok()?
        .remove(key)
}

/// This function clears the file position of every table in `item`. A copied sub-table
/// then follows the header of the table it is copied into, not the header
/// of the table it came from.
fn unplace(item: &mut Item) {
    if let Some(table) = item.as_table_mut() {
        table.set_position(None);
        for (_, child) in table.iter_mut() {
            unplace(child);
        }
    }
}

/// This function returns whether a target names exactly this model on this client.
fn names_choice(target: &dyn TableLike, choice: &Choice) -> bool {
    text(target, "llm_client") == Some(choice.client.as_str())
        && text(target, "id") == Some(choice.model.as_str())
}

/// This function removes the old type's settings, sets the new `type`, and returns the
/// removed settings as key paths, such as `classifier.base_threshold`. The
/// list leaves out `old_slots`, the old roles' targets, which the new roles
/// replace.
fn reset_route(
    route: &mut dyn TableLike,
    algorithm: &Algorithm,
    old_slots: &[String],
) -> Vec<String> {
    let kept =
        |key: &str| COMMON_ROUTE_KEYS.contains(&key) || (algorithm.subagents && key == "subagents");
    let stale: Vec<String> = route
        .iter()
        .map(|(key, _)| key)
        .filter(|key| !kept(key))
        .map(str::to_string)
        .collect();
    let mut removed = Vec::new();
    for key in stale {
        if let Some(item) = route.remove(&key) {
            key_paths(&key, &item, &mut removed);
        }
    }
    removed.retain(|path| !old_slots.contains(path));
    set_value(route, "type", Value::from(algorithm.kind));
    removed
}

/// This function adds the key path of every value in `item` to `paths`.
fn key_paths(path: &str, item: &Item, paths: &mut Vec<String>) {
    match item.as_table_like() {
        Some(table) => {
            for (key, child) in table.iter() {
                key_paths(&format!("{path}.{key}"), child, paths);
            }
        }
        None => paths.push(path.to_string()),
    }
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

/// This function sets a value at a key path, creating missing tables on the way.
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

/// This function replaces a value, keeping the spacing and trailing comment around it.
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

/// This function returns the hosts that get the callers' own logins: the hosts of the
/// `forward_auth` clients that `choices` name, sorted, each once.
fn login_hosts<'c>(clients: &'c [Client], choices: &[Choice]) -> Vec<&'c str> {
    let mut hosts: Vec<&str> = choices
        .iter()
        .filter_map(|choice| clients.iter().find(|client| client.name == choice.client))
        .filter(|client| client.forward_auth)
        .map(Client::host)
        .collect();
    hosts.sort_unstable();
    hosts.dedup();
    hosts
}

// Generated-block markers can occur on nested route tables and still apply to retained targets.
fn retain_block_markers(doc: &mut DocumentMut, removed: &Item) {
    let mut markers = Vec::new();
    let mut tables: Vec<_> = removed.as_table().into_iter().collect();
    while let Some(table) = tables.pop() {
        if let Some(position) = table.position()
            && let Some(prefix) = table.decor().prefix().and_then(|value| value.as_str())
            && prefix
                .lines()
                .any(|line| line.trim().starts_with("# >>>") || line.trim().starts_with("# <<<"))
        {
            markers.push((position, prefix.to_string()));
        }
        tables.extend(table.iter().filter_map(|(_, item)| item.as_table()));
    }
    markers.sort_by_key(|(position, _)| *position);
    let mut positions = Vec::new();
    let mut tables = vec![doc.as_table()];
    while let Some(table) = tables.pop() {
        positions.extend(table.position());
        tables.extend(table.iter().filter_map(|(_, item)| item.as_table()));
    }
    let mut prefixes: HashMap<isize, String> = HashMap::new();
    let mut trailing = String::new();
    for (removed_position, prefix) in markers {
        if let Some(next) = positions
            .iter()
            .copied()
            .filter(|position| *position > removed_position)
            .min()
        {
            prefixes.entry(next).or_default().push_str(&prefix);
        } else {
            trailing.push_str(&prefix);
        }
    }
    let mut tables = vec![doc.as_table_mut()];
    while let Some(table) = tables.pop() {
        if let Some(position) = table.position()
            && let Some(mut prefix) = prefixes.remove(&position)
        {
            prefix.push_str(
                table
                    .decor()
                    .prefix()
                    .and_then(|value| value.as_str())
                    .unwrap_or_default(),
            );
            table.decor_mut().set_prefix(prefix);
        }
        tables.extend(table.iter_mut().filter_map(|(_, item)| item.as_table_mut()));
    }
    trailing.push_str(doc.trailing().as_str().unwrap_or_default());
    doc.set_trailing(trailing);
}

/// This function finds the route and target tables that sit inside blocks another tool
/// writes. A block starts at a line that begins with `# >>>` and ends at the
/// next line that begins with `# <<<`, the markers that the installer scripts
/// use. Each table maps to the text of its block's first marker, with the `#`
/// and the `>>>` removed.
fn generated_tables(text: &str) -> HashMap<String, String> {
    let mut tables = HashMap::new();
    let mut block: Option<String> = None;
    for line in text.lines().map(str::trim) {
        if line.starts_with("# >>>") {
            block = Some(
                line.trim_matches(|c: char| c == '#' || c == '>' || c.is_whitespace())
                    .to_string(),
            );
        } else if line.starts_with("# <<<") {
            block = None;
        } else if let Some(block) = &block
            && let Some(header) = line.strip_prefix('[')
            && let Some((table, rest)) = header.split_once('.')
            && matches!(table, "routes" | "targets")
            && let Some(key) = table_key(rest)
        {
            tables
                .entry(format!("{table}.{key}"))
                .or_insert_with(|| block.clone());
        }
    }
    tables
}

/// This function returns the first key of a table header's dotted path, bare or in double
/// quotes: `name` for `name.child]`, and `a.b` for `"a.b"]`.
fn table_key(path: &str) -> Option<&str> {
    if let Some(quoted) = path.strip_prefix('"') {
        return quoted.split_once('"').map(|(key, _)| key);
    }
    path.split(['.', ']']).next().filter(|key| !key.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This fixture uses a GPT judge and Claude on one gateway with different API formats.
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

    /// This function edits the config and checks the result with the server's parser.
    fn edit(text: &str, route: &str, kind: &str, choices: &[Choice]) -> Edited {
        let edited = ServerConfig::parse(text)
            .expect("parse")
            .edit(route, algorithm(kind), choices)
            .expect("edit");
        if let Err(error) = switchyard_runner::Runner::from_toml(&edited.text) {
            panic!(
                "the server rejects the edited config: {error}\n{}",
                edited.text
            );
        }
        edited
    }

    fn capable_target(config: &ServerConfig) -> Option<&str> {
        config.target_at("gateway", &Slot::Key(&["stage", "capable_target"]))
    }

    #[test]
    fn removing_a_route_keeps_generated_block_markers_for_retained_targets() {
        for (suffix, target, owned) in [
            (
                "# >>> generated >>>\n[routes.generated]\nid='generated'\ntype='passthrough'\ntarget='owned'\n[targets.owned]\nid='m'\nllm_client='gateway'\n# <<< generated <<<\n",
                "owned",
                true,
            ),
            (
                "# >>> generated >>>\n[targets.owned]\nid='m'\nllm_client='gateway'\n# <<< generated <<<\n[routes.generated]\nid='generated'\ntype='passthrough'\ntarget='owned'\n[targets.unowned]\nid='other'\nllm_client='gateway'\n",
                "unowned",
                false,
            ),
            (
                "[routes.generated]\nid='generated'\ntype='composite'\n# >>> generated >>>\n[routes.generated.classifier]\ntarget='owned'\n[targets.owned]\nid='m'\nllm_client='gateway'\n# <<< generated <<<\n",
                "owned",
                true,
            ),
        ] {
            let config = ServerConfig::parse(&format!("{GATEWAY}\n{suffix}")).expect("config");
            assert_eq!(config.generated_by("targets", target).is_some(), owned);
            let edited = config.remove("generated").expect("remove");
            let remaining = ServerConfig::parse(&edited.text).expect("config");
            assert_eq!(
                remaining.generated_by("targets", target).is_some(),
                owned,
                "{}",
                edited.text
            );
        }
    }

    #[test]
    // Routes can share targets, so deleting one route must leave those targets usable.
    fn removing_a_route_preserves_shared_targets_and_other_routes() {
        let original = format!(
            "{GATEWAY}\n[routes.keep]\nid='keep'\ntype='passthrough'\ntarget='gateway_capable'\n"
        );
        let config = ServerConfig::parse(&original).expect("config");
        let edited = config.remove("gateway").expect("remove");
        let remaining = ServerConfig::parse(&edited.text).expect("config");
        assert_eq!(remaining.routes().len(), 1);
        assert_eq!(remaining.routes()[0].id, "keep");
        assert_eq!(remaining.models_on("gateway"), config.models_on("gateway"));
        switchyard_runner::Runner::from_toml(&edited.text).expect("server accepts remaining route");
        assert!(config.remove("missing").is_err());
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

            let config = ServerConfig::parse(&edited.text).expect("parse");
            assert_eq!(
                config.algorithm("gateway").map(|a| a.kind),
                Some(algorithm.kind)
            );
            assert_eq!(config.choices("gateway"), models[3 - roles..]);
        }
    }

    #[test]
    fn a_target_one_role_reuses_is_not_changed_for_another_role() {
        // Capable moves to the efficient model, and Efficient moves to a new model.
        let chosen = [
            choice("gateway", "gpt-5.6-terra"),
            choice("gateway_chat", "claude-sonnet-5"),
            choice("gateway", "gpt-5.6-luna"),
        ];

        let edited = edit(GATEWAY, "gateway", "composite", &chosen);

        assert_eq!(
            ServerConfig::parse(&edited.text)
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
        )
        .text;

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
        assert_eq!(claude.text, GATEWAY);
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

        let config = ServerConfig::parse(&edited.text).expect("parse");
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
            edited.text.contains("[targets.gateway_judge]"),
            "unused targets stay in the file"
        );
        let removed = edited.notes.join("\n");
        assert!(
            removed.contains("classifier.message_hash_fallback")
                && !removed.contains("stage.capable_target"),
            "the result lists the removed settings, not the replaced targets: {removed}"
        );
    }

    #[test]
    fn a_target_another_route_uses_is_copied_not_changed() {
        // The shared target has a sub-table, which the copy must keep under
        // its own header.
        let shared = GATEWAY.replace(
            "omit_body_fields = [\"reasoning_effort\"]\n\n[targets.gateway_efficient]",
            "omit_body_fields = [\"reasoning_effort\"]\n\n[targets.gateway_capable.extra_body]\n\
             top_k = 5\n\n[targets.gateway_efficient]",
        ) + "\n[routes.direct]\nid = \"direct\"\ntype = \"passthrough\"\ntarget = \"gateway_capable\"\n";

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

        let config = ServerConfig::parse(&edited.text).expect("parse");
        assert_eq!(
            config.choices("direct"),
            [choice("gateway_chat", "claude-opus-5-5")]
        );
        assert_eq!(
            config.choices("gateway")[1],
            choice("gateway", "gpt-5.6-sol")
        );
        let copy = capable_target(&config).expect("capable target");
        assert_ne!(copy, "gateway_capable");
        let targets = config.table("targets").expect("targets");
        let (source, copied) = (
            targets.get("gateway_capable").and_then(Item::as_table_like),
            targets.get(copy).and_then(Item::as_table_like),
        );
        for key in TARGET_SETTINGS {
            assert_eq!(
                source.and_then(|target| setting(target, key)),
                copied.and_then(|target| setting(target, key)),
                "the copy keeps {key}"
            );
        }
        let header = |name: &str| edited.text.find(&format!("[{name}]")).expect(name);
        assert!(
            header(&format!("targets.{copy}")) < header(&format!("targets.{copy}.extra_body")),
            "the copied sub-table follows the copy's header:\n{}",
            edited.text
        );
    }

    #[test]
    fn uses_another_routes_target_only_when_the_route_keeps_its_settings_or_must_share() {
        // `prompted` names the model that Capable moves to, and has a
        // system prompt that the gateway route does not have.
        for (omit, shared) in [
            // Same request settings: the server accepts a second target for
            // the model, so Capable keeps its own target.
            ("omit_body_fields = [\"reasoning_effort\"]\n", false),
            // Other request settings: the server would reject a second
            // target, so Capable shares `prompted`, and the result says so.
            ("", true),
        ] {
            let text = format!(
                "{GATEWAY}\n[targets.prompted]\nid = \"gpt-5.6-sol\"\nllm_client = \"gateway\"\n\
                 system_prompt = \"You answer for route other.\"\n{omit}\n[routes.other]\n\
                 id = \"other\"\ntype = \"passthrough\"\ntarget = \"prompted\"\n"
            );

            let edited = edit(
                &text,
                "gateway",
                "composite",
                &[
                    choice("gateway", "gpt-5.6-terra"),
                    choice("gateway", "gpt-5.6-sol"),
                    choice("gateway_chat", "claude-sonnet-5"),
                ],
            );

            let config = ServerConfig::parse(&edited.text).expect("parse");
            assert_eq!(
                config.choices("gateway")[1],
                choice("gateway", "gpt-5.6-sol")
            );
            assert_eq!(
                capable_target(&config) == Some("prompted"),
                shared,
                "{omit}"
            );
            assert_eq!(
                edited
                    .notes
                    .iter()
                    .any(|note| note.contains("prompted") && note.contains("other")),
                shared,
                "{:?}",
                edited.notes
            );
        }
    }

    #[test]
    fn a_move_to_another_model_family_reports_the_settings_the_target_kept() {
        let gpt = edit(
            GATEWAY,
            "gateway",
            "composite",
            &[
                choice("gateway", "gpt-5.6-terra"),
                choice("gateway", "gpt-5.6-sol"),
                choice("gateway", "gpt-5.6-luna"),
            ],
        )
        .text;

        // The judge target has no request settings, and the capable target
        // keeps omit_body_fields from its GPT model. Efficient stays.
        let edited = edit(
            &gpt,
            "gateway",
            "composite",
            &[
                choice("gateway_chat", "claude-haiku-4-5"),
                choice("gateway_chat", "claude-opus-5-5"),
                choice("gateway", "gpt-5.6-luna"),
            ],
        );

        let note = |target: &str| {
            edited
                .notes
                .iter()
                .find(|note| note.contains(&format!("[targets.{target}]")))
                .cloned()
        };
        let capable = note("gateway_capable").expect("a note for the capable target");
        assert!(
            capable.contains("claude-opus-5-5")
                && capable.contains(r#"omit_body_fields = ["reasoning_effort"]"#),
            "{capable}"
        );
        let judge = note("gateway_judge").expect("a note for the judge target");
        assert!(
            judge.contains("claude-haiku-4-5") && !judge.contains(" = "),
            "{judge}"
        );
        assert_eq!(note("gateway_efficient"), None);
    }

    #[test]
    fn rejects_a_missing_model_or_an_unknown_client() {
        let config = ServerConfig::parse(GATEWAY).expect("parse");
        for (capable, named) in [
            (choice("gateway", " "), "Capable"),
            (choice("typo", "gpt-5.6-sol"), "typo"),
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

            assert!(error.contains(named), "{error}");
        }
    }

    /// This fixture places the last target and route inside a generated block.
    /// A hand-written route shares the capable target.
    fn with_generated_block() -> String {
        format!(
            "{GATEWAY}\n[routes.direct]\nid = \"direct\"\ntype = \"passthrough\"\n\
             target = \"gateway_capable\"\n\n\
             # >>> sync: generated by sync.py; edits between the markers are overwritten >>>\n\n\
             [targets.generated]\nid = \"gpt-5.5\"\nllm_client = \"gateway\"\n\n\
             [routes.generated]\nid = \"generated\"\ntype = \"passthrough\"\n\
             target = \"generated\"\n\n# <<< sync <<<\n"
        )
    }

    #[test]
    fn a_new_target_goes_before_a_generated_block_not_inside_it() {
        // Another route uses the capable target, so Capable gets a copy.
        let edited = edit(
            &with_generated_block(),
            "gateway",
            "composite",
            &[
                choice("gateway", "gpt-5.6-terra"),
                choice("gateway", "gpt-5.6-sol"),
                choice("gateway_chat", "claude-sonnet-5"),
            ],
        );

        let config = ServerConfig::parse(&edited.text).expect("parse");
        let copy = capable_target(&config).expect("capable target");
        assert_ne!(copy, "gateway_capable");
        assert_eq!(config.generated_by("targets", copy), None);
        let at = |needle: &str| edited.text.find(needle).expect(needle);
        assert!(
            at(&format!("[targets.{copy}]")) < at("# >>> sync"),
            "the next rewrite of the block would delete the copy:\n{}",
            edited.text
        );
    }

    #[test]
    fn a_new_target_goes_before_the_first_table_when_every_target_is_generated() {
        // A script writes every target in a block above the hand-written
        // routes. Another route uses the capable target, so Capable gets a
        // copy.
        let text = GATEWAY
            .replace(
                "# The gateway rejects",
                "# >>> sync: generated by sync.py; edits between the markers are overwritten >>>\n\
                 # The gateway rejects",
            )
            .replace("[routes.gateway]\n", "# <<< sync <<<\n\n[routes.gateway]\n")
            + "\n[routes.direct]\nid = \"direct\"\ntype = \"passthrough\"\n\
               target = \"gateway_capable\"\n";

        let edited = edit(
            &text,
            "gateway",
            "composite",
            &[
                choice("gateway", "gpt-5.6-terra"),
                choice("gateway", "gpt-5.6-sol"),
                choice("gateway_chat", "claude-sonnet-5"),
            ],
        );

        let config = ServerConfig::parse(&edited.text).expect("parse");
        let copy = capable_target(&config).expect("capable target");
        assert_ne!(copy, "gateway_capable");
        assert_eq!(config.generated_by("targets", copy), None);
        let at = |needle: &str| edited.text.find(needle).expect(needle);
        assert!(
            at(&format!("[targets.{copy}]")) < at("# >>> sync"),
            "the next rewrite of the block would delete the copy:\n{}",
            edited.text
        );
    }

    #[test]
    fn finds_the_tables_inside_generated_blocks() {
        let text = with_generated_block().replace(
            "# <<< sync <<<",
            "[routes.\"quoted.key\"]\nid = \"q\"\ntype = \"passthrough\"\ntarget = \"generated\"\n\n\
             # <<< sync <<<",
        );

        let config = ServerConfig::parse(&text).expect("parse");

        let block = "sync: generated by sync.py; edits between the markers are overwritten";
        assert_eq!(config.generated_by("routes", "generated"), Some(block));
        assert_eq!(config.generated_by("targets", "generated"), Some(block));
        assert_eq!(config.generated_by("routes", "quoted.key"), Some(block));
        assert_eq!(config.generated_by("routes", "gateway"), None);
        assert_eq!(config.generated_by("routes", "direct"), None);
        assert_eq!(config.generated_by("targets", "gateway_capable"), None);
    }

    #[test]
    fn warns_when_a_change_is_inside_a_generated_block() {
        let text = with_generated_block();

        // The tool overwrites changes to a route inside its generated block.
        let inside = edit(
            &text,
            "generated",
            "passthrough",
            &[choice("gateway", "gpt-5.6-sol")],
        );
        assert!(
            inside
                .notes
                .iter()
                .any(|note| note.contains("[routes.generated]") && note.contains("overwrites")),
            "{:?}",
            inside.notes
        );

        // The hand-written route now uses a target inside the generated block.
        let outside = edit(
            &text,
            "direct",
            "passthrough",
            &[choice("gateway", "gpt-5.5")],
        );
        assert!(
            outside.notes.iter().any(
                |note| note.contains("[routes.direct]") && note.contains("[targets.generated]")
            ),
            "{:?}",
            outside.notes
        );
    }

    #[test]
    fn says_when_the_callers_logins_go_to_another_host() {
        let text = GATEWAY.replace(
            "[targets.gateway_capable]",
            "[llm_clients.other]\nformat = \"openai_responses\"\nbase_url = \
             \"https://login.example.org/v1\"\nforward_auth = true\n\n[targets.gateway_capable]",
        );
        let login = |edited: &Edited| {
            edited
                .notes
                .iter()
                .find(|note| note.contains("login"))
                .cloned()
        };

        let moved = edit(
            &text,
            "gateway",
            "composite",
            &[
                choice("other", "a"),
                choice("other", "b"),
                choice("other", "c"),
            ],
        );
        let note = login(&moved).expect("a note about the login");
        assert!(
            note.contains("login.example.org") && note.contains("gateway.example.com"),
            "{note}"
        );

        let same_host = edit(
            &text,
            "gateway",
            "composite",
            &[
                choice("gateway", "gpt-5.6-terra"),
                choice("gateway_chat", "claude-opus-5-5"),
                choice("gateway_chat", "claude-haiku-4-5"),
            ],
        );
        assert_eq!(login(&same_host), None);
    }

    #[test]
    fn lists_the_models_a_client_already_serves() {
        let config = ServerConfig::parse(GATEWAY).expect("parse");

        assert_eq!(
            config.models_on("gateway_chat"),
            ["claude-opus-5-5", "claude-sonnet-5"]
        );
        assert_eq!(config.models_on("gateway"), ["gpt-5.6-terra"]);
        assert!(config.models_on("missing").is_empty());
    }

    #[test]
    fn names_the_host_and_port_of_a_client_without_its_path_or_login() {
        let client = |base_url: &str| Client {
            name: "c".to_string(),
            format: "openai_chat".to_string(),
            base_url: base_url.to_string(),
            api_key_env: None,
            forward_auth: false,
        };

        assert_eq!(
            client("https://chatgpt.com/backend-api/codex").host(),
            "chatgpt.com"
        );
        assert_eq!(
            client("http://user@127.0.0.1:8000/v1?x=1").host(),
            "127.0.0.1:8000"
        );
    }
}
