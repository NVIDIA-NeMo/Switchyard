// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module reads the server config and rewrites one route's algorithm and models.
//!
//! Edits go through `toml_edit`, so comments, formatting, and every table the
//! edit does not touch stay as the user wrote them. The server's own
//! `--dry-run` decides whether the result is valid; this module only builds it.

use toml_edit::{DocumentMut, TableLike};

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
    /// This field lists roles in display order. The list is empty for
    /// `random`, whose roles are the entries of its `targets` list.
    roles: &'static [RoleSpec],
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

const RANDOM: &str = "random";

/// ALGORITHMS lists supported route types in display order.
/// `noop` and the experimental `prefill_router` are left out: one calls no
/// model, and the other needs a checkpoint file rather than models.
pub const ALGORITHMS: &[Algorithm] = &[
    Algorithm {
        kind: "passthrough",
        roles: &[RoleSpec {
            label: "Model",
            hint: "This model answers every request.",
            tier: Tier::Capable,
            path: &["target"],
        }],
    },
    Algorithm {
        kind: RANDOM,
        roles: &[],
    },
    Algorithm {
        kind: "llm_classifier",
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
    },
    Algorithm {
        kind: "composite",
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
    },
    Algorithm {
        kind: "stage_router",
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
    },
    Algorithm {
        kind: "advisor",
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
    },
    Algorithm {
        kind: "plan_execute",
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
    },
    Algorithm {
        kind: "auto",
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
    },
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

impl Slot {}

impl Algorithm {
    /// This function returns the roles of this type. `listed` is the number of entries
    /// in a random route's `targets` list; other types ignore it.
    fn roles_for(&self, listed: usize) -> Vec<Role> {
        if self.kind != RANDOM {
            return self
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
        (0..listed.max(2))
            .map(|index| Role {
                label: format!("Model {}", index + 1),
                hint: "This model gets a share of the requests.",
                tier: [Tier::Capable, Tier::Efficient].get(index).copied(),
                slot: Slot::Listed(index),
            })
            .collect()
    }
}

/// ServerConfig keeps the parsed config and the original text's layout.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    doc: DocumentMut,
}

impl ServerConfig {
    pub fn parse(text: &str) -> Result<Self, String> {
        text.parse::<DocumentMut>()
            .map(|doc| Self { doc })
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
        let listed = self
            .route(route)
            .filter(|route| algorithm.kind == RANDOM && text(*route, "type") == Some(RANDOM))
            .and_then(|route| {
                route
                    .get("targets")?
                    .as_array()
                    .map(|targets| targets.len())
            })
            .unwrap_or(0);
        algorithm.roles_for(listed)
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
