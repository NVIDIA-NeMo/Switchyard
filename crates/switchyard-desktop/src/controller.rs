// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The terminal controller reloads settings and combines usage and server health into snapshots.
//! It retains the log reader and health probe across refreshes.

use crate::{
    accounts,
    config::Config,
    handoffs,
    harness::{self, Harness},
    history, models, pricing, rollup, server,
    server_config::{ALGORITHMS, Choice, Route, ServerConfig},
    sessions,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use switchyard_desktop::transcripts;

#[derive(Clone, Debug, Deserialize, strum::EnumDiscriminants)]
#[strum_discriminants(name(ActionKind), derive(strum::EnumIter))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    Editor {
        generation: u64,
        route: String,
        algorithm: String,
    },
    Apply {
        generation: u64,
        route: String,
        algorithm: String,
        choices: Vec<Choice>,
        revision: Option<String>,
    },
    Remove {
        generation: u64,
        route: String,
        id: String,
        revision: Option<String>,
    },
    PreviewRoute {
        generation: u64,
        route: String,
        algorithm: String,
        choices: Vec<Choice>,
        revision: Option<String>,
    },
    PreviewRouteChange {
        generation: u64,
        operation: String,
        route: String,
        name: String,
        id: String,
        algorithm: Option<String>,
        choices: Vec<Choice>,
    },
    RouteChange {
        generation: u64,
        operation: String,
        route: String,
        name: String,
        id: String,
        algorithm: Option<String>,
        choices: Vec<Choice>,
        preview_token: Option<String>,
    },
    Models {
        client: String,
        refresh: bool,
        key: Option<String>,
    },
    PreviewInstall {
        tool: Harness,
        account: Option<String>,
        route: String,
        id: String,
        settings_file: Option<String>,
    },
    Install {
        tool: Harness,
        account: Option<String>,
        route: String,
        id: String,
        settings_file: Option<String>,
        preview_token: Option<String>,
    },
    PreviewRestore {
        tool: Harness,
        account: Option<String>,
        settings_file: Option<String>,
    },
    Restore {
        tool: Harness,
        account: Option<String>,
        settings_file: Option<String>,
        preview_token: Option<String>,
    },
    AddAccount {
        tool: Harness,
        name: String,
    },
    Launch {
        preview_token: Option<String>,
        tool: Harness,
        account: Option<String>,
        route: String,
        id: String,
        project: String,
    },
    PreviewLaunch {
        tool: Harness,
        account: Option<String>,
        route: String,
        id: String,
        project: String,
    },
    /// This shows the hand-off that another harness reads to continue `source`: the text,
    /// the destination, the warnings, and the command. `tool`, `account`, and `route` select
    /// the destination.
    PreviewHandoff {
        source: HandoffSource,
        tool: Harness,
        account: Option<String>,
        route: String,
        id: String,
    },
    /// This writes the reviewed hand-off and opens the destination harness in the session's
    /// directory. With `attach`, it returns the harness command instead of opening a terminal,
    /// for the `handoff` subcommand, which runs the harness in its own terminal.
    Handoff {
        source: HandoffSource,
        tool: Harness,
        account: Option<String>,
        route: String,
        id: String,
        preview_token: Option<String>,
        #[serde(default)]
        attach: bool,
    },
    /// This lists recent sessions from harness transcripts, newest first: those of one harness
    /// and account, or of both command line harnesses and every saved account when `tool` is
    /// absent. Each session names the harness that continues it by default. The list is read on
    /// demand because it opens every transcript.
    RecentSessions {
        tool: Option<Harness>,
        account: Option<String>,
    },
    LoginAccount {
        tool: Harness,
        name: String,
    },
    PreviewAccountArchive {
        tool: Harness,
        name: String,
    },
    ArchiveAccount {
        tool: Harness,
        name: String,
        preview_token: Option<String>,
    },
    OpenAccount {
        tool: Harness,
        name: String,
    },
    OpenSession {
        id: String,
    },
    PreviewSessionCleanup {
        id: String,
    },
    RemoveSession {
        id: String,
        preview_token: Option<String>,
    },
    Restart {},
    PreviewUpdate {},
    Update {
        preview_token: Option<String>,
    },
    OpenLogs {},
    OpenUpdateLog {},
    OpenConfig {},
    OpenPrices {},
    OpenSettings {},
    CheckConfig {},
    ServerKeys {},
    SaveServerKey {
        name: String,
        key: String,
    },
    ImportServerKey {
        name: String,
    },
    RemoveServerKey {
        name: String,
    },
    SaveConfiguration {
        config: Config,
        create_server: bool,
    },
    PreviewResetSettings {},
    ResetSettings {
        preview_token: Option<String>,
    },
    SavePreferences {
        baseline_model: String,
        refresh_seconds: u64,
    },
}

/// HandoffSource names the session that a hand-off continues.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HandoffSource {
    /// A session that the app launched, by its `sy-` ID. The hand-off reads the transcript of
    /// the session's latest launch and keeps the session ID.
    Managed { id: String },
    /// A transcript file that the user chose: a Claude Code `<uuid>.jsonl` or a Codex rollout.
    /// The hand-off continues in the directory the transcript records, under a new session ID.
    Transcript { path: String },
}

/// Snapshot holds the data shown by the apps. The terminal app destructures every field,
/// so adding a field requires an update to `draw`.
#[derive(Clone, Default, Serialize)]
pub struct Snapshot {
    pub needs_setup: bool,
    pub configuration_error: Option<String>,
    pub update_status: Value,
    /// Each value is the `session.json` file that `sessions::launch` wrote for a managed
    /// worktree session, plus `process_status`. The snapshot passes these files through as JSON.
    pub session_inventory: Vec<Value>,
    /// This field holds the error that stopped the inventory read. `session_inventory` is empty
    /// when this field is set.
    pub session_inventory_error: Option<String>,
    /// Each value is the `handoff.json` record of a hand-off, newest first. Records older than
    /// `handoffs::RETENTION_DAYS` are removed on the first snapshot after the app starts.
    pub handoffs: Vec<Value>,
    pub settings: Option<AppSettings>,
    pub data_state: DataState,
    /// The controller increases this number when the app settings select another server
    /// config file. Route actions send it back, so a draft cannot change the new file.
    pub generation: u64,
    pub routes: Vec<RouteInfo>,
    pub clients: Vec<ClientInfo>,
    pub tools: Vec<ToolInfo>,
    pub algorithms: Vec<AlgorithmInfo>,
    pub metrics: Metrics,
    pub today: chrono::NaiveDate,
    pub analytics: rollup::Usage,
    /// The keys are `today`, `week`, and `all`. A value is null when usage could not be read.
    pub estimates: BTreeMap<&'static str, Option<pricing::Estimate>>,
    pub baseline_model: String,
    pub activity: [u64; 24],
    pub sessions: Vec<String>,
    pub entries: Vec<history::Entry>,
    pub skipped: usize,
    pub limited: bool,
    pub errors: Vec<String>,
    pub refresh_seconds: u64,
}

/// AppSettings holds the app settings and their file path. `Controller::snapshot` destructures
/// every `Config` field, so adding a setting requires an update here too.
#[derive(Clone, Default, Serialize)]
pub struct AppSettings {
    pub path: PathBuf,
    pub server_url: String,
    pub routing_log: PathBuf,
    pub config_file: PathBuf,
    pub launchd_label: String,
    pub refresh_seconds: u64,
    pub baseline_model: String,
    pub prices_file: Option<PathBuf>,
}

/// RouteInfo describes one entry under `[routes]` in the server config.
#[derive(Clone, Default, Serialize)]
pub struct RouteInfo {
    /// This field names the route's table in the server config.
    pub key: String,
    /// This field holds the public model ID that harnesses send.
    pub id: String,
    pub kind: String,
    /// Route edits send this token back. The controller rejects it after the file changes.
    pub revision: String,
    pub label: String,
    pub choices: Vec<Choice>,
    /// This field lists the tier of each role in role order. It is `None` when the apps
    /// cannot edit the route's algorithm.
    pub tiers: Option<Vec<Option<String>>>,
    pub editable: bool,
    /// This field holds the marker of the generated block that contains the route. The tool
    /// that wrote the block replaces the route the next time it runs.
    pub generated: Option<String>,
    /// This field maps each harness to the reason it cannot use the route, or to `None`
    /// when it can.
    pub compatibility: BTreeMap<Harness, Option<String>>,
    /// This field lists the harness settings files that select the route's public model ID.
    pub dependents: Vec<Dependent>,
}

/// Dependent names harness settings that select a route, so deleting the route breaks them.
#[derive(Clone, Serialize)]
pub struct Dependent {
    pub tool: Harness,
    pub label: &'static str,
    /// This field is `None` for the harness's own settings and names a saved account otherwise.
    pub account: Option<String>,
    pub files: Vec<PathBuf>,
}

/// ToolInfo describes one harness whose settings the apps can change.
#[derive(Clone, Serialize)]
pub struct ToolInfo {
    pub tool: Harness,
    pub label: &'static str,
    /// This field is true when the settings files resolve and the harness program is installed.
    pub available: bool,
    pub files: Vec<PathBuf>,
    /// This field summarizes the current settings, or holds the error that stopped the read.
    pub status: String,
    pub accounts: Vec<String>,
    pub account_details: Vec<AccountInfo>,
    pub discovery_error: Option<String>,
}

/// AccountInfo describes one saved account folder of a harness.
#[derive(Clone, Serialize)]
pub struct AccountInfo {
    pub name: String,
    pub path: PathBuf,
    /// The apps cannot ask a harness whether a login finished, so this field is `unknown`.
    pub login_status: &'static str,
    /// This field summarizes the account's settings, or holds the error that stopped the read.
    pub status: String,
}

/// ClientInfo describes one endpoint under `[llm_clients]` in the server config.
#[derive(Clone, Default, Serialize)]
pub struct ClientInfo {
    pub name: String,
    pub host: String,
    /// This field lists the model IDs that targets in the server config name on this endpoint.
    pub models: Vec<String>,
    /// This field is true when the app cannot ask the endpoint for its model list.
    pub unlisted: bool,
    pub accepts_key: bool,
    /// This field explains why the endpoint cannot list models without a saved key.
    pub note: Option<String>,
}

/// AlgorithmInfo describes one routing algorithm that the apps can write and the roles
/// that a new route of that algorithm needs.
#[derive(Clone, Default, Serialize)]
pub struct AlgorithmInfo {
    pub kind: &'static str,
    pub title: &'static str,
    pub summary: &'static str,
    pub roles: Vec<RoleInfo>,
}

#[derive(Clone, Default, Serialize)]
pub struct RoleInfo {
    pub label: String,
    pub hint: &'static str,
    pub tier: Option<String>,
}

#[derive(Clone, Default, Serialize)]
pub struct DataState {
    pub usage: Source,
    pub history: Source,
    /// This field holds the `/health` result with `state`, `reason`, `checked_at`, and `url`.
    pub health: Value,
}

/// Source records whether the app could read one data file.
#[derive(Clone, Default, Serialize)]
pub struct Source {
    pub state: Availability,
    pub checked_at: String,
    pub path: PathBuf,
}

/// The default is unavailable, so a data file that the app has not read shows no usage.
#[derive(Clone, Copy, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    Available,
    #[default]
    Unavailable,
}

#[derive(Clone, Default, Serialize)]
pub struct Metrics {
    pub server_url: String,
    pub running: bool,
    pub today: Counts,
    pub week: Counts,
}

#[derive(Clone, Default, Serialize)]
pub struct Counts {
    pub requests: u64,
    pub tokens: u64,
}

#[derive(Serialize)]
pub struct Reply {
    pub message: String,
    pub data: Value,
}
impl Reply {
    fn saved_edit(result: server::SaveResult) -> Self {
        Self {
            message: result.message,
            data: json!({"warning": result.warning, "details": result.details}),
        }
    }

    fn message(message: String) -> Self {
        Self {
            message,
            data: Value::Null,
        }
    }
}

// Review keeps exact file bytes in the controller so later edits require a new review.
// Only the UUID token goes to the frontend; reviewed files can contain credentials.
struct Review {
    token: String,
    inputs: Value,
    files: Vec<harness::ReviewedFile>,
}
impl Review {
    fn capture(inputs: Value, paths: Vec<PathBuf>) -> Result<Self, String> {
        let files = paths
            .into_iter()
            .map(|path| {
                let bytes = match std::fs::read(&path) {
                    Ok(bytes) => Some(std::sync::Arc::<[u8]>::from(bytes)),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                    Err(error) => return Err(format!("Read {}: {error}", path.display())),
                };
                Ok((path, bytes))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            token: uuid::Uuid::now_v7().to_string(),
            inputs,
            files,
        })
    }
    fn original(&self) -> Result<&str, String> {
        std::str::from_utf8(
            self.files
                .first()
                .and_then(|(_, bytes)| bytes.as_deref())
                .ok_or("The reviewed server config is missing.")?,
        )
        .map_err(|error| error.to_string())
    }
    // Decoded settings omit comments and unrelated fields, so this check compares raw bytes.
    fn check(&self, token: Option<&str>, inputs: &Value) -> Result<(), String> {
        if token != Some(self.token.as_str()) || inputs != &self.inputs {
            return Err("Review these changes again before applying them.".into());
        }
        for (path, bytes) in &self.files {
            let current = match std::fs::read(path) {
                Ok(bytes) => Some(std::sync::Arc::<[u8]>::from(bytes)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(error.to_string()),
            };
            if &current != bytes {
                return Err(format!(
                    "{} changed since your review. Review it again; your draft is preserved.",
                    path.display()
                ));
            }
        }
        Ok(())
    }
}

// Each Controller retains one launch review until the next preview or a selected config change.
// Review keeps raw settings private; Launch uses these values and the reviewed Git commit.
struct ReviewedLaunch {
    review: Review,
    project: Value,
    model: String,
    server_url: String,
    login: bool,
    account: Option<PathBuf>,
}

// The hand-off review keeps the rendered text, so Handoff writes exactly what the user read.
// The transcript is a live file, so the review remembers its size and modification time
// rather than its bytes.
struct ReviewedHandoff {
    review: Review,
    plan: HandoffPlan,
}
struct HandoffPlan {
    id: String,
    markdown: String,
    kind: transcripts::Kind,
    transcript: PathBuf,
    fingerprint: transcripts::Fingerprint,
    source: Value,
    managed: Option<String>,
    tool: Harness,
    model: String,
    root: String,
    login: bool,
    account: Option<PathBuf>,
    directory: PathBuf,
    session_id: String,
    native_session_id: Option<String>,
}

pub struct Controller {
    settings: PathBuf,
    config: Config,
    reader: rollup::Reader,
    health: crate::health::Probe,
    // The generation ties a draft to the server config file selected by the settings.
    generation: u64,
    route_reviews: std::collections::HashMap<String, Review>,
    tool_review: Option<Review>,
    session_cleanup: Option<(String, String, Value)>,
    account_archive: Option<(Harness, String, String)>,
    route_change: Option<Review>,
    update_review: Option<(String, Value)>,
    launch_review: Option<ReviewedLaunch>,
    handoff_review: Option<ReviewedHandoff>,
    settings_review: Option<Review>,
    /// Old hand-offs are removed once per run, on the first snapshot.
    purged_handoffs: bool,
}
impl Controller {
    pub fn new(config: Config, settings: PathBuf) -> Self {
        Self {
            config,
            settings,
            reader: rollup::Reader::default(),
            health: crate::health::Probe::default(),
            generation: 0,
            route_reviews: Default::default(),
            tool_review: None,
            session_cleanup: None,
            account_archive: None,
            route_change: None,
            update_review: None,
            launch_review: None,
            handoff_review: None,
            settings_review: None,
            purged_handoffs: false,
        }
    }
    fn reload(&mut self) -> Result<(), String> {
        let config = Config::load(&self.settings)?;
        if config.config_file != self.config.config_file {
            self.generation += 1;
            self.route_reviews.clear();
            self.tool_review = None;
            self.route_change = None;
            self.launch_review = None;
            self.handoff_review = None;
        }
        self.config = config;

        Ok(())
    }
    fn check_generation(&self, generation: u64) -> Result<(), String> {
        if generation != self.generation {
            return Err("The route changed or was removed. Refresh and choose it again.".into());
        }
        Ok(())
    }
    // Key actions accept only names that the server config declares.
    fn declared_key(&self, name: &str) -> Result<(), String> {
        if switchyard_desktop::install::credentials::variables(&self.config.config_file)?
            .contains(name)
        {
            Ok(())
        } else {
            Err(format!(
                "The server config does not name {name} with api_key_env. Refresh and try again."
            ))
        }
    }
    fn routes(&self) -> Result<ServerConfig, String> {
        let text = std::fs::read_to_string(&self.config.config_file).map_err(|e| e.to_string())?;
        ServerConfig::parse(&text).map_err(|_| {
            "The server config contains invalid TOML. Open it in Settings to fix it.".into()
        })
    }
    // Unchanged bytes keep the draft revision when the editor switches algorithms.
    fn remember_route(&mut self, route: &str, mut review: Review) -> String {
        if let Some(previous) = self.route_reviews.get(route)
            && previous.files == review.files
        {
            review.token = previous.token.clone();
        }
        let revision = review.token.clone();
        self.route_reviews.insert(route.to_string(), review);
        revision
    }
    pub fn snapshot(&mut self) -> Result<Snapshot, String> {
        let configuration_error = self.reload().err();
        let mut errors: Vec<String> = configuration_error.iter().cloned().collect();
        let needs_setup = crate::config::needs_setup(&self.settings)?;
        // One local date keeps aggregate totals and frontend filters aligned across midnight.
        let today = chrono::Local::now().date_naive();
        let mut usage_state = Availability::Available;
        let usage = self
            .reader
            .read(&self.config.routing_log, today)
            .unwrap_or_else(|error| {
                usage_state = Availability::Unavailable;
                errors.push(format!("Could not read usage: {error}"));
                rollup::Usage::default()
            });
        // Invalid overrides leave baseline rates unknown and add an error to the snapshot.
        let rates = pricing::load(self.config.prices_file.as_deref()).unwrap_or_else(|error| {
            errors.push(format!("Could not read model prices: {error}"));
            Default::default()
        });
        let estimates = [
            ("today", &usage.today),
            ("week", &usage.week),
            ("all", &usage.all),
        ]
        .into_iter()
        .map(|(period, totals)| {
            let estimate = (usage_state == Availability::Available)
                .then(|| pricing::Estimate::new(totals, &rates, &self.config.baseline_model));
            (period, estimate)
        })
        .collect();
        let checked_at = chrono::Utc::now().to_rfc3339();
        let mut health = self.health.details(&self.config.server_url);
        health["checked_at"] = json!(checked_at);
        health["url"] = json!(self.config.server_url);
        let server_status = if health["state"] == "running" {
            crate::health::ServerStatus::Running
        } else {
            crate::health::ServerStatus::Stopped
        };
        let mut routes = Vec::new();
        let mut clients = Vec::new();
        let route_review = Review::capture(json!({}), vec![self.config.config_file.clone()]);
        let route_config = route_review
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|review| {
                ServerConfig::parse(review.original()?).map_err(|_| {
                    "The server config contains invalid TOML. Open it in Settings to fix it."
                        .to_string()
                })
            });
        match route_config {
            Ok(config) => {
                let listed = config.routes();
                self.route_reviews
                    .retain(|key, _| listed.iter().any(|route| &route.key == key));
                for route in listed {
                    let original = route_review.as_ref().map_err(Clone::clone)?;
                    let review = Review {
                        token: uuid::Uuid::now_v7().to_string(),
                        inputs: json!({"route":route.key}),
                        files: original.files.clone(),
                    };
                    let revision = self.remember_route(&route.key, review);
                    let choices = config.choices(&route.key);
                    let models = choices
                        .iter()
                        .map(|c| format!("{} / {}", c.client, c.model))
                        .collect::<Vec<_>>()
                        .join(" → ");
                    let algorithm = config.algorithm(&route.key);
                    routes.push(RouteInfo {
                        label: format!("{} · {} · {}", route.id, route.kind, models),
                        tiers: algorithm.map(|a| {
                            config
                                .roles(&route.key, a)
                                .iter()
                                .map(|r| r.tier.map(|t| format!("{t:?}")))
                                .collect()
                        }),
                        compatibility: harness::HARNESSES
                            .iter()
                            .map(|&(tool, _)| (tool, login_mode(tool, &config, &route).err()))
                            .collect(),
                        dependents: known_dependents(&route.id),
                        editable: algorithm.is_some(),
                        generated: config
                            .generated_by("routes", &route.key)
                            .map(str::to_string),
                        key: route.key,
                        id: route.id,
                        kind: route.kind,
                        revision,
                        choices,
                    });
                }
                clients = config
                    .clients()
                    .iter()
                    .map(|c| ClientInfo {
                        name: c.name.clone(),
                        host: c.host().to_string(),
                        models: config.models_on(&c.name),
                        unlisted: models::unlisted(c),
                        accepts_key: cfg!(any(target_os = "macos", windows))
                            && !models::unlisted(c)
                            && (c.forward_auth || c.api_key_env.is_some()),
                        note: models::missing_env(c).map(models::missing_env_note),
                    })
                    .collect();
            }
            Err(e) => errors.push(e),
        }
        let mut history_state = Availability::Available;
        let history = history::load(&self.config.routing_log).unwrap_or_else(|error| {
            history_state = Availability::Unavailable;
            errors.push(format!("Could not read recent calls: {error}"));
            history::History::default()
        });
        let activity = history::hourly_calls(&history.entries, chrono::Utc::now());
        let tools = harness::HARNESSES
            .iter()
            .map(|&(tool, label)| {
                let discovered = accounts::discover(tool);
                let discovery_error = discovered.as_ref().err().cloned();
                let discovered = discovered.unwrap_or_default();
                let account_details = discovered
                    .iter()
                    .map(|(name, path)| AccountInfo {
                        name: name.clone(),
                        path: path.clone(),
                        login_status: "unknown",
                        status: accounts::config_paths(tool, Some(path))
                            .and_then(|files| harness::inspect(tool, &files))
                            .unwrap_or_else(|error| error),
                    })
                    .collect();
                // A path error stays in status; files remains an array and the tool is unavailable.
                let files = harness::paths(tool);
                let status = files
                    .as_ref()
                    .map_err(Clone::clone)
                    .and_then(|files| harness::inspect(tool, files))
                    .unwrap_or_else(|error| error);
                ToolInfo {
                    tool,
                    label,
                    available: files.is_ok() && harness::binary(tool).is_some(),
                    files: files.unwrap_or_default(),
                    status,
                    accounts: discovered.into_iter().map(|(name, _)| name).collect(),
                    account_details,
                    discovery_error,
                }
            })
            .collect();
        let (session_inventory, session_inventory_error) = match sessions::inventory() {
            Ok(inventory) => (inventory, None),
            Err(error) => (Vec::new(), Some(error)),
        };
        if !self.purged_handoffs {
            self.purged_handoffs = true;
            if let Err(error) = handoffs::purge(chrono::Utc::now()) {
                errors.push(format!("Could not remove old hand-offs: {error}"));
            }
        }
        let handoffs = handoffs::list().unwrap_or_else(|error| {
            errors.push(format!("Could not read hand-offs: {error}"));
            Vec::new()
        });
        let Config {
            server_url,
            routing_log,
            config_file,
            launchd_label,
            refresh_seconds,
            baseline_model,
            prices_file,
        } = self.config.clone();
        let update_status = install_metadata(&self.settings)
            .and_then(|path| switchyard_desktop::install::update_status(&path))
            .unwrap_or_else(|error| json!({"state":"unavailable","message":error}));
        Ok(Snapshot {
            needs_setup,
            configuration_error,
            update_status,
            settings: Some(AppSettings {
                path: self.settings.clone(),
                server_url: server_url.clone(),
                routing_log: routing_log.clone(),
                config_file,
                launchd_label,
                refresh_seconds,
                baseline_model: baseline_model.clone(),
                prices_file,
            }),
            session_inventory,
            session_inventory_error,
            handoffs,
            data_state: DataState {
                usage: Source {
                    state: usage_state,
                    checked_at: checked_at.clone(),
                    path: routing_log.clone(),
                },
                history: Source {
                    state: history_state,
                    checked_at,
                    path: routing_log.clone(),
                },
                health,
            },
            generation: self.generation,
            routes,
            clients,
            tools,
            algorithms: ALGORITHMS
                .iter()
                .map(|a| AlgorithmInfo {
                    kind: a.kind,
                    title: a.title,
                    summary: a.summary,
                    roles: a.new_route_roles().iter().map(role_info).collect(),
                })
                .collect(),
            metrics: Metrics {
                server_url: server_url.clone(),
                running: server_status == crate::health::ServerStatus::Running,
                today: Counts {
                    requests: usage.today.requests(),
                    tokens: usage.today.tokens(),
                },
                week: Counts {
                    requests: usage.week.requests(),
                    tokens: usage.week.tokens(),
                },
            },
            today,
            analytics: usage,
            estimates,
            baseline_model,
            activity,
            sessions: history.sessions(),
            entries: history.entries,
            skipped: history.skipped,
            limited: history.limited,
            errors,
            refresh_seconds: refresh_seconds.max(1),
        })
    }
    /// The TUI uses these file and folder paths to run an editor in the foreground.
    pub fn editor_target(&mut self, action: &Action) -> Result<Option<PathBuf>, String> {
        match action {
            Action::OpenSettings {} => Ok(Some(self.settings.clone())),
            Action::OpenConfig {} => {
                self.reload()?;
                Ok(Some(self.config.config_file.clone()))
            }
            // When `prices_file` is unset, this arm returns `None` and `dispatch` describes the
            // bundled rates.
            Action::OpenPrices {} => {
                self.reload()?;
                Ok(self.config.prices_file.clone())
            }
            Action::OpenUpdateLog {} => self.update_log().map(Some),
            Action::OpenLogs {} => logs_dir().map(Some),
            // Account and session folders open in the editor too, because a terminal may run
            // over SSH, where `open` would start a file manager on another screen.
            Action::OpenAccount { tool, name } => account_path(*tool, Some(name)),
            Action::OpenSession { id } => {
                let receipt = sessions::managed_session(id)?;
                let worktree = receipt["worktree"]
                    .as_str()
                    .ok_or("Missing session directory.")?;
                Ok(Some(PathBuf::from(worktree)))
            }
            _ => Ok(None),
        }
    }
    fn update_log(&self) -> Result<PathBuf, String> {
        let metadata =
            switchyard_desktop::install::Settings::load(&install_metadata(&self.settings)?)?;
        Ok(metadata.sy_home.join("logs/update.log"))
    }
    pub fn dispatch(&mut self, action: Action) -> Result<Reply, String> {
        if matches!(action, Action::OpenSettings {}) {
            server::command("open", &[&self.settings.display().to_string()])?;
            return Ok(Reply::message("Opened app settings.".into()));
        }
        // Configuration actions remain available while a saved file needs repair.
        match &action {
            Action::SaveConfiguration {
                config,
                create_server,
            } => {
                crate::config::save_configuration(&self.settings, config, *create_server)?;
                self.reload()?;
                return Ok(Reply::message("Saved configuration. The installed server loads these files when it starts. Check Server keys before restarting.".into()));
            }
            Action::PreviewResetSettings {} => {
                let review = Review::capture(json!({}), vec![self.settings.clone()])?;
                let data = json!({"preview_token":review.token,"file":self.settings,"message":"This removes only the app preferences. A recovery copy is kept. Server configs, keys, history, accounts, and tool settings are preserved."});
                self.settings_review = Some(review);
                return Ok(Reply {
                    message: String::new(),
                    data,
                });
            }
            Action::ResetSettings { preview_token } => {
                let review = self
                    .settings_review
                    .as_ref()
                    .ok_or("Review app preference removal first.")?;
                review.check(preview_token.as_deref(), &json!({}))?;
                let backup = self
                    .settings
                    .with_extension(format!("removed-{}.toml", uuid::Uuid::now_v7()));
                match self.settings.symlink_metadata() {
                    Ok(_) => std::fs::rename(&self.settings, &backup)
                        .map_err(|error| error.to_string())?,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.to_string()),
                }
                self.settings_review = None;
                self.config = Config::defaults_for(&self.settings)?;
                self.generation += 1;
                self.route_reviews.clear();
                self.route_change = None;
                self.tool_review = None;
                self.launch_review = None;
                return Ok(Reply::message(format!(
                    "Removed app preferences. Recovery copy: {}. Server files are preserved.",
                    backup.display()
                )));
            }
            _ => {}
        }
        self.reload()?;
        match action {
            Action::SaveConfiguration { .. }
            | Action::PreviewResetSettings {}
            | Action::ResetSettings { .. } => unreachable!(),
            Action::OpenLogs {} => {
                server::command("open", &[&logs_dir()?.display().to_string()])?;
                Ok(Reply::message("Opened Switchyard logs.".into()))
            }
            Action::Editor {
                generation,
                route,
                algorithm,
            } => {
                self.check_generation(generation)?;
                let review = Review::capture(
                    json!({"route":route}),
                    vec![self.config.config_file.clone()],
                )?;
                let config = self.routes()?;
                review.check(Some(&review.token), &review.inputs)?;
                current_route(&config, &route, None)?;
                config
                    .algorithm(&route)
                    .ok_or("This route must be edited in the server config file.")?;
                let algorithm = algorithm_by_id(&algorithm)?;
                let roles = config.roles(&route, algorithm);
                let revision = self.remember_route(&route, review);
                let data = json!({"revision":revision,"roles": roles.iter().map(role_info).collect::<Vec<_>>(),"choices":config.choices(&route)});
                Ok(Reply {
                    message: algorithm.summary.into(),
                    data,
                })
            }
            Action::Apply {
                generation,
                route,
                algorithm,
                choices,
                revision,
            } => {
                self.check_generation(generation)?;
                let config = self.routes()?;
                current_route(&config, &route, None)?;
                config
                    .algorithm(&route)
                    .ok_or("This route must be edited in the server config file.")?;
                let algorithm = algorithm_by_id(&algorithm)?;
                self.route_reviews
                    .get(&route)
                    .ok_or("Open the route editor before saving.")?
                    .check(revision.as_deref(), &json!({"route":route}))?;
                server::apply(
                    &self.config,
                    &self.health,
                    &route,
                    algorithm,
                    &choices,
                    self.route_reviews[&route].original()?,
                )
                .map(Reply::saved_edit)
            }
            Action::Remove {
                generation,
                route,
                id,
                revision,
            } => {
                self.check_generation(generation)?;
                current_route(&self.routes()?, &route, Some(&id))?;
                self.route_reviews
                    .get(&route)
                    .ok_or("Open the route editor before removing it.")?
                    .check(revision.as_deref(), &json!({"route":route}))?;
                server::remove(
                    &self.config,
                    &self.health,
                    &route,
                    &id,
                    self.route_reviews[&route].original()?,
                )
                .map(Reply::saved_edit)
            }
            Action::PreviewRouteChange {
                generation,
                operation,
                route,
                name,
                id,
                algorithm,
                choices,
            } => {
                self.check_generation(generation)?;
                let inputs = json!({"operation":operation,"route":route,"name":name,"id":id,"algorithm":algorithm,"choices":choices});
                let review = Review::capture(inputs, vec![self.config.config_file.clone()])?;
                let edited = self.routes()?.change_route(
                    &operation,
                    &route,
                    &name,
                    &id,
                    algorithm.as_deref().map(algorithm_by_id).transpose()?,
                    &choices,
                )?;
                review.check(Some(&review.token), &review.inputs)?;
                let data = json!({"notes":edited.notes,"file":self.config.config_file,"preview_token":review.token});
                self.route_change = Some(review);
                Ok(Reply {
                    message: String::new(),
                    data,
                })
            }
            Action::RouteChange {
                generation,
                operation,
                route,
                name,
                id,
                algorithm,
                choices,
                preview_token,
            } => {
                self.check_generation(generation)?;
                let inputs = json!({"operation":operation,"route":route,"name":name,"id":id,"algorithm":algorithm,"choices":choices});
                let review = self
                    .route_change
                    .as_ref()
                    .ok_or("Review the route change first.")?;
                review.check(preview_token.as_deref(), &inputs)?;
                let edited = self.routes()?.change_route(
                    &operation,
                    &route,
                    &name,
                    &id,
                    algorithm.as_deref().map(algorithm_by_id).transpose()?,
                    &choices,
                )?;
                server::save_route_change(&self.config, &self.health, review.original()?, edited)
                    .map(Reply::saved_edit)
            }
            Action::Models {
                client,
                refresh,
                key,
            } => {
                let config = self.routes()?;
                let client = config
                    .clients()
                    .into_iter()
                    .find(|c| c.name == client)
                    .ok_or("Unknown endpoint.")?;
                if let Some(key) = &key {
                    if key.trim().is_empty() || models::has_line_break(key) {
                        return Err("Enter a nonempty key without line breaks.".into());
                    }
                    if models::unlisted(&client)
                        || !(client.forward_auth || client.api_key_env.is_some())
                    {
                        return Err("This endpoint does not accept a model-list key. \
                                    Choose an endpoint configured for model-list authentication."
                            .into());
                    }
                    #[cfg(not(any(target_os = "macos", windows)))]
                    models::save_key(&client.base_url, key)?;
                }
                let cache = self.settings.with_file_name(models::CACHE_FILE);
                let loaded =
                    models::load(&cache, &client, refresh || key.is_some(), key.as_deref());
                if let Some(key) = &key {
                    match &loaded.error {
                        None | Some(models::ListError::NotCached(_)) => {
                            models::save_key(&client.base_url, key)?
                        }
                        Some(error) => return Err(error.reason()),
                    }
                }
                // Fetch and cache-write failures need a warning even when a model list remains usable.
                let warning = loaded.error.is_some();
                let fetched_at = loaded.list.as_ref().map(|list| list.fetched_at);
                let source = loaded.list.as_ref().map_or("config", |_| loaded.source);
                let message = loaded
                    .error
                    .map(|e| match e.advice() {
                        "" => e.reason(),
                        advice => format!("{} {advice}", e.reason()),
                    })
                    .unwrap_or_else(|| "Model list loaded.".into());
                let models = loaded
                    .list
                    .map(|l| l.models)
                    .unwrap_or_else(|| config.models_on(&client.name));
                Ok(Reply {
                    message,
                    data: json!({"client":client.name,"count":models.len(),"models":models,"warning":warning,"source":source,"fetched_at":fetched_at}),
                })
            }
            Action::PreviewRoute {
                generation,
                route,
                algorithm,
                choices,
                revision,
            } => {
                self.check_generation(generation)?;
                self.route_reviews
                    .get(&route)
                    .ok_or("Open the route editor before reviewing it.")?
                    .check(revision.as_deref(), &json!({"route":route}))?;
                let edited = self
                    .routes()?
                    .edit(&route, algorithm_by_id(&algorithm)?, &choices)?;
                Ok(Reply {
                    message: String::new(),
                    data: json!({"notes":edited.notes,"revision":revision,"file":self.config.config_file}),
                })
            }
            Action::PreviewInstall {
                tool,
                account,
                route,
                id,
                settings_file,
            } => {
                let files = install_paths(tool, account.as_deref(), settings_file.as_deref())?;
                let inputs = json!({"tool":tool,"account":account,"route":route,"id":id,"settings_file":settings_file,"server_url":self.config.server_url});
                let mut paths = harness::review_files(&files);
                paths.push(self.config.config_file.clone());
                let review = Review::capture(inputs, paths)?;
                let config = self.routes()?;
                let selected = current_route(&config, &route, Some(&id))?;
                let mut data = harness::preview(
                    tool,
                    &files,
                    &self.config.server_url,
                    &selected.id,
                    config.context_window(&selected.key),
                    login_mode(tool, &config, &selected)?,
                )?;
                review.check(Some(&review.token), &review.inputs)?;
                data["preview_token"] = json!(review.token);
                self.tool_review = Some(review);
                Ok(Reply {
                    message: String::new(),
                    data,
                })
            }
            Action::Install {
                tool,
                account,
                route,
                id,
                settings_file,
                preview_token,
            } => {
                let files = install_paths(tool, account.as_deref(), settings_file.as_deref())?;
                let inputs = json!({"tool":tool,"account":account,"route":route,"id":id,"settings_file":settings_file,"server_url":self.config.server_url});
                self.tool_review
                    .as_ref()
                    .ok_or("Preview these settings before applying them.")?
                    .check(preview_token.as_deref(), &inputs)?;
                let config = self.routes()?;
                let route = current_route(&config, &route, Some(&id))?;
                harness::install_reviewed(
                    tool,
                    &files,
                    &self.config.server_url,
                    &route.id,
                    config.context_window(&route.key),
                    login_mode(tool, &config, &route)?,
                    self.tool_review
                        .as_ref()
                        .map(|review| review.files.as_slice()),
                )
                .map(Reply::message)
            }
            Action::PreviewRestore {
                tool,
                account,
                settings_file,
            } => {
                let files = install_paths(tool, account.as_deref(), settings_file.as_deref())?;
                let review = Review::capture(
                    json!({"restore":true,"tool":tool,"account":account,"settings_file":settings_file}),
                    harness::review_files(&files),
                )?;
                let mut data = harness::preview_restore(tool, &files)?;
                review.check(Some(&review.token), &review.inputs)?;
                data["preview_token"] = json!(review.token);
                self.tool_review = Some(review);
                Ok(Reply {
                    message: String::new(),
                    data,
                })
            }
            Action::Restore {
                tool,
                account,
                settings_file,
                preview_token,
            } => {
                let files = install_paths(tool, account.as_deref(), settings_file.as_deref())?;
                self.tool_review.as_ref().ok_or("Preview the restore before applying it.")?.check(preview_token.as_deref(),&json!({"restore":true,"tool":tool,"account":account,"settings_file":settings_file}))?;
                match harness::restore_reviewed(
                    &files,
                    self.tool_review
                        .as_ref()
                        .map(|review| review.files.as_slice()),
                ) {
                    Ok(message) => Ok(Reply::message(message)),
                    Err(error) if error.starts_with("Restored settings, but") => Ok(Reply {
                        message: error,
                        data: json!({"warning":true}),
                    }),
                    Err(error) => Err(error),
                }
            }
            Action::AddAccount { tool, name } => {
                accounts::add(tool, name.trim()).map(Reply::message)
            }
            Action::Launch {
                tool,
                account,
                route,
                id,
                project,
                preview_token,
            } => {
                let inputs =
                    json!({"tool":tool,"account":account,"route":route,"id":id,"project":project});
                let plan = self
                    .launch_review
                    .as_ref()
                    .ok_or("Review the session before opening Terminal.")?;
                plan.review.check(preview_token.as_deref(), &inputs)?;
                let current = sessions::preview(std::path::Path::new(&project))?;
                if current != plan.project {
                    return Err(
                        "The project changed since review. Review the session again.".into(),
                    );
                }
                sessions::launch(
                    tool,
                    std::path::Path::new(
                        plan.project["project"]
                            .as_str()
                            .ok_or("Missing reviewed project.")?,
                    ),
                    &plan.model,
                    &plan.server_url,
                    plan.login,
                    plan.account.as_deref(),
                    plan.project["revision"]
                        .as_str()
                        .ok_or("Missing reviewed commit.")?,
                )
                .map(Reply::message)
            }
            Action::PreviewLaunch {
                tool,
                account,
                route,
                id,
                project,
            } => {
                if tool == Harness::CodexApp {
                    return Err("Use Codex app’s workspace controls or choose Codex CLI.".into());
                }
                if harness::binary(tool).is_none() {
                    return Err("Install the selected harness first.".into());
                }
                let inputs =
                    json!({"tool":tool,"account":account,"route":route,"id":id,"project":project});
                let review = Review::capture(
                    inputs.clone(),
                    vec![self.config.config_file.clone(), self.settings.clone()],
                )?;
                let app_config = Config::load(&self.settings)?;
                review.check(Some(&review.token), &inputs)?;
                if app_config.config_file != self.config.config_file {
                    return Err("Review these changes again before applying them.".into());
                }
                let config = ServerConfig::parse(review.original()?).map_err(
                    |_| "The server config contains invalid TOML. Open it in Settings to fix it.",
                )?;
                let route = current_route(&config, &route, Some(&id))?;
                let login = login_mode(tool, &config, &route)?;
                let account = account_path(tool, account.as_deref())?;
                let project_review = sessions::preview(std::path::Path::new(&project))?;
                let mut data = project_review.clone();
                data["tool"] = json!(tool);
                data["route"] = json!(route.id);
                data["account_directory"] = json!(account);
                data["server_url"] = json!(app_config.server_url);
                data["login"] = json!(login);
                if tool == Harness::Pi {
                    data["settings_notice"] = json!(
                        "Pi uses isolated session settings. \
                         Personal settings and extensions are not copied."
                    );
                }
                review.check(Some(&review.token), &inputs)?;
                data["preview_token"] = json!(review.token);
                self.launch_review = Some(ReviewedLaunch {
                    review,
                    project: project_review,
                    model: route.id,
                    server_url: app_config.server_url,
                    login,
                    account,
                });
                Ok(Reply {
                    message: String::new(),
                    data,
                })
            }
            Action::PreviewHandoff {
                source,
                tool,
                account,
                route,
                id,
            } => {
                let inputs =
                    json!({"source":source,"tool":tool,"account":account,"route":route,"id":id});
                if matches!(tool, Harness::CodexApp | Harness::Pi) {
                    return Err("Choose Codex CLI or Claude Code to continue a session.".into());
                }
                let binary = harness::binary(tool).ok_or("Install the selected harness first.")?;
                let review = Review::capture(
                    inputs.clone(),
                    vec![self.config.config_file.clone(), self.settings.clone()],
                )?;
                let app_config = Config::load(&self.settings)?;
                if app_config.config_file != self.config.config_file {
                    return Err("Review these changes again before applying them.".into());
                }
                let config = ServerConfig::parse(review.original()?).map_err(
                    |_| "The server config contains invalid TOML. Open it in Settings to fix it.",
                )?;
                let route = current_route(&config, &route, Some(&id))?;
                let login = login_mode(tool, &config, &route)?;
                let root = harness::validate(tool, &app_config.server_url, &route.id, login)?;
                let account = account_path(tool, account.as_deref())?;
                let mut warnings = Vec::new();
                let (kind, transcript, managed, directory, session_id) = match &source {
                    HandoffSource::Managed { id } => {
                        let receipt = sessions::managed_session(id)?;
                        let (kind, transcript) = managed_transcript(&receipt)?;
                        let directory = session_directory(&receipt)?;
                        warnings.push(format!(
                            "The new session keeps the session ID {id}, so the server's \
                             per-session routing state carries over and Usage lists both \
                             harnesses under it."
                        ));
                        (
                            kind,
                            transcript,
                            Some(id.clone()),
                            Some(directory),
                            id.clone(),
                        )
                    }
                    HandoffSource::Transcript { path } => {
                        let path = PathBuf::from(path);
                        if !path.is_absolute()
                            || !path.is_file()
                            || path.extension().is_none_or(|e| e != "jsonl")
                        {
                            return Err(
                                "Choose a .jsonl transcript file by its absolute path.".into()
                            );
                        }
                        let kind = transcripts::detect(&path)?;
                        let session_id = format!("sy-{}", uuid::Uuid::now_v7());
                        (kind, path, None, None, session_id)
                    }
                };
                let limits = transcripts::Limits::default();
                let excerpt = transcripts::excerpt(kind, &transcript, &limits)?;
                let markdown = transcripts::render(&excerpt, &limits);
                let directory = match directory {
                    Some(directory) => directory,
                    None => excerpt.session.cwd.clone().ok_or(
                        "The transcript does not record a working directory, so the app \
                         cannot tell where to continue. Hand off a managed session instead.",
                    )?,
                };
                if !directory.is_dir() {
                    return Err(format!(
                        "The session directory {} is not on this computer. The transcript may \
                         come from a remote session or from a project that was removed.",
                        directory.display()
                    ));
                }
                if excerpt.truncated {
                    warnings.push(
                        "Parts of the transcript were cut to keep the hand-off short.".into(),
                    );
                }
                if let Some(branch) = &excerpt.session.branch
                    && let Some(current) = current_branch(&directory)
                    && &current != branch
                {
                    warnings.push(format!(
                        "The transcript records branch {branch}, but {} is on {current}.",
                        directory.display()
                    ));
                }
                if chrono::Utc::now().timestamp_millis() - excerpt.fingerprint.modified_ms
                    < RECENT_MS
                {
                    warnings.push(
                        "The transcript changed in the last two minutes, so the source session \
                         may still be running. Finish or quit it before continuing elsewhere."
                            .into(),
                    );
                }
                if kind == transcripts::Kind::Claude
                    && claude_open(&transcript, &excerpt.session.native_id)
                {
                    warnings.push("Claude Code lists this session as open.".into());
                }
                let handoff_id = handoffs::new_id();
                let dir = handoffs::directory(&handoff_id)?;
                let file = dir.join(handoffs::FILE);
                let native_session_id = (tool == Harness::Claude).then(sessions::claude_session_id);
                let script_account = sessions::script_account(tool, account.as_deref())?;
                let files = vec![dir.join("settings.json")];
                let instructions = pointer(kind, &file);
                let first = first_prompt(&file);
                let command = sessions::display_command(&sessions::OpenRequest {
                    tool,
                    binary,
                    directory: directory.clone(),
                    model: &route.id,
                    root: &root,
                    login,
                    account: script_account.as_deref(),
                    session_id: &session_id,
                    files: &files,
                    private: &dir,
                    native_session_id: native_session_id.as_deref(),
                    instructions: Some(&instructions),
                    read_dirs: std::slice::from_ref(&dir),
                    first_prompt: Some(&first),
                })?;
                let mut source_info =
                    serde_json::to_value(&excerpt.session).map_err(|error| error.to_string())?;
                source_info["tool"] = json!(source_harness(kind));
                source_info["label"] = json!(kind.label());
                source_info["managed_id"] = json!(managed);
                let data = json!({
                    "preview_token": review.token,
                    "handoff": markdown,
                    "bytes": markdown.len(),
                    "truncated": excerpt.truncated,
                    "turns": excerpt.turns.len(),
                    "files_touched": excerpt.files.len(),
                    "commands": excerpt.commands.len(),
                    "source": source_info,
                    "destination": {
                        "tool": tool,
                        "label": harness_name(tool),
                        "route": route.id,
                        "account_directory": account,
                        "directory": directory,
                        "session_id": session_id,
                        "native_session_id": native_session_id,
                        "login": login,
                        "server_url": app_config.server_url,
                    },
                    "handoff_file": file,
                    "settings_file": (tool == Harness::Claude).then(|| files[0].clone()),
                    "command": command,
                    "warnings": warnings,
                    "notice": HANDOFF_NOTICE,
                });
                self.handoff_review = Some(ReviewedHandoff {
                    review,
                    plan: HandoffPlan {
                        id: handoff_id,
                        markdown,
                        kind,
                        transcript,
                        fingerprint: excerpt.fingerprint,
                        source: source_info,
                        managed,
                        tool,
                        model: route.id,
                        root,
                        login,
                        account,
                        directory,
                        session_id,
                        native_session_id,
                    },
                });
                Ok(Reply {
                    message: String::new(),
                    data,
                })
            }
            Action::Handoff {
                source,
                tool,
                account,
                route,
                id,
                preview_token,
                attach,
            } => {
                let inputs =
                    json!({"source":source,"tool":tool,"account":account,"route":route,"id":id});
                let reviewed = self
                    .handoff_review
                    .as_ref()
                    .ok_or("Review the hand-off before opening Terminal.")?;
                reviewed.review.check(preview_token.as_deref(), &inputs)?;
                if transcripts::fingerprint(&reviewed.plan.transcript)? != reviewed.plan.fingerprint
                {
                    return Err(
                        "The transcript changed since your review. Review the hand-off again."
                            .into(),
                    );
                }
                let binary = harness::binary(tool).ok_or("Install the selected harness first.")?;
                // A review opens one session. Another hand-off needs another review.
                let plan = self
                    .handoff_review
                    .take()
                    .ok_or("Review the hand-off before opening Terminal.")?
                    .plan;
                let started_at = chrono::Utc::now().to_rfc3339();
                let dir = handoffs::directory(&plan.id)?;
                let file = dir.join(handoffs::FILE);
                let mut record = json!({
                    "id": plan.id,
                    "created_at": started_at,
                    "file": file,
                    "source": plan.source,
                    "destination": {
                        "tool": plan.tool,
                        "label": harness_name(plan.tool),
                        "route": plan.model,
                        "account_directory": plan.account,
                        "directory": plan.directory,
                        "session_id": plan.session_id,
                        "native_session_id": plan.native_session_id,
                    },
                    "bytes": plan.markdown.len(),
                    "opened": false,
                });
                handoffs::create(&plan.id, &plan.markdown, &record)?;
                let files = vec![dir.join("settings.json")];
                let instructions = pointer(plan.kind, &file);
                let first = first_prompt(&file);
                let opened = sessions::script_account(plan.tool, plan.account.as_deref()).and_then(
                    |script_account| {
                        if plan.tool == Harness::Claude {
                            harness::install(
                                plan.tool,
                                &files,
                                &plan.root,
                                &plan.model,
                                None,
                                plan.login,
                            )?;
                        }
                        let request = sessions::OpenRequest {
                            tool: plan.tool,
                            binary: binary.clone(),
                            directory: plan.directory.clone(),
                            model: &plan.model,
                            root: &plan.root,
                            login: plan.login,
                            account: script_account.as_deref(),
                            session_id: &plan.session_id,
                            files: &files,
                            private: &dir,
                            native_session_id: plan.native_session_id.as_deref(),
                            instructions: Some(&instructions),
                            read_dirs: std::slice::from_ref(&dir),
                            first_prompt: Some(&first),
                        };
                        if attach {
                            sessions::invocation(&request).map(Some)
                        } else {
                            sessions::open_harness(&request).map(|()| None)
                        }
                    },
                );
                // A hand-off that never opened holds only an unread excerpt, so it goes away.
                let invocation = match opened {
                    Ok(invocation) => invocation,
                    Err(error) => {
                        let _ = std::fs::remove_dir_all(&dir);
                        return Err(error);
                    }
                };
                record["opened"] = json!(true);
                handoffs::update(&plan.id, &record)?;
                if let Some(managed) = &plan.managed {
                    sessions::append_launch(
                        managed,
                        &json!({
                            "tool": plan.tool,
                            "route": plan.model,
                            "account_directory": plan.account,
                            "settings_directory": dir,
                            "started_at": started_at,
                            "native_session_id": plan.native_session_id,
                            "handoff": {
                                "id": plan.id,
                                "from_tool": plan.source["tool"],
                                "from_native_id": plan.source["native_id"],
                                "file": file,
                            },
                        }),
                    )?;
                }
                let message = format!(
                    "{} {}.\n\
                     Hand-off: {}\n\
                     Directory: {}\n\
                     Session ID: {}\n\
                     The new session reads the hand-off under its own login. Hand-offs are \
                     removed after {} days.",
                    if attach {
                        "Continuing the session here in"
                    } else {
                        "Opened Terminal to continue the session in"
                    },
                    harness_name(plan.tool),
                    file.display(),
                    plan.directory.display(),
                    plan.session_id,
                    handoffs::RETENTION_DAYS
                );
                let data = match invocation {
                    Some(invocation) => json!({"invocation": {
                        "program": binary,
                        "directory": plan.directory,
                        "args": invocation.args,
                        "environment": invocation.environment,
                        "remove_environment": invocation.remove_environment,
                    }}),
                    None => Value::Null,
                };
                Ok(Reply { message, data })
            }
            Action::RecentSessions { tool, account } => Ok(Reply {
                message: String::new(),
                data: recent_sessions(&self.config.config_file, tool, account.as_deref())?,
            }),
            Action::LoginAccount { tool, name } => {
                let path = account_path(tool, Some(&name))?.ok_or("Choose an account.")?;
                accounts::login(tool, &path, &name).map(Reply::message)
            }
            Action::PreviewAccountArchive { tool, name } => {
                let path = account_path(tool, Some(&name))?.ok_or("Choose an account.")?;
                let token = uuid::Uuid::now_v7().to_string();
                self.account_archive = Some((tool, name.clone(), token.clone()));
                Ok(Reply {
                    message: String::new(),
                    data: json!({
                        "name": name,
                        "path": path,
                        "preview_token": token,
                        "notice": "Archive moves this account out of the picker and preserves all \
                                   login files. Existing sessions using its old directory may \
                                   need a new login."
                    }),
                })
            }
            Action::ArchiveAccount {
                tool,
                name,
                preview_token,
            } => {
                let (expected_tool, expected_name, token) = self
                    .account_archive
                    .as_ref()
                    .ok_or("Review account archive first.")?;
                if tool != *expected_tool
                    || name != *expected_name
                    || preview_token.as_deref() != Some(token.as_str())
                {
                    return Err("Review account archive again.".into());
                }
                let path = account_path(tool, Some(&name))?.ok_or("Choose an account.")?;
                accounts::archive(tool, &path).map(Reply::message)
            }
            Action::OpenAccount { tool, name } => {
                let path = account_path(tool, Some(&name))?.ok_or("Choose an account.")?;
                server::command("open", &[&path.display().to_string()])?;
                Ok(Reply::message("Opened the account folder.".into()))
            }
            Action::OpenSession { id } => {
                let receipt = sessions::managed_session(&id)?;
                server::command(
                    "open",
                    &[receipt["worktree"]
                        .as_str()
                        .ok_or("Missing session directory.")?],
                )?;
                Ok(Reply::message("Opened the session folder.".into()))
            }
            Action::PreviewSessionCleanup { id } => {
                let data = sessions::cleanup_preview(&id)?;
                let token = uuid::Uuid::now_v7().to_string();
                self.session_cleanup = Some((id, token.clone(), data.clone()));
                let mut data = data;
                data["preview_token"] = json!(token);
                Ok(Reply {
                    message: String::new(),
                    data,
                })
            }
            Action::RemoveSession { id, preview_token } => {
                let (expected_id, token, data) = self
                    .session_cleanup
                    .as_ref()
                    .ok_or("Review session cleanup first.")?;
                if &id != expected_id || preview_token.as_deref() != Some(token.as_str()) {
                    return Err("Review session cleanup again.".into());
                }
                sessions::cleanup(&id, data).map(Reply::message)
            }
            Action::Restart {} => {
                server::restart_checked(&self.config, &self.health).map(Reply::message)
            }
            Action::CheckConfig {} => server::check_config(&self.config)
                .map(|()| Reply::message("Server config check passed.".into())),
            Action::ServerKeys {} => Ok(Reply {
                message: String::new(),
                data: json!({"keys": server::server_keys(&self.config)?, "store": switchyard_desktop::install::credentials::STORE}),
            }),
            Action::SaveServerKey { name, key } => {
                self.declared_key(&name)?;
                server::save_server_key(&name, &key)?;
                Ok(Reply::message(format!(
                    "Saved the key for {name}. Restart the server to use it."
                )))
            }
            Action::ImportServerKey { name } => {
                self.declared_key(&name)?;
                server::import_server_key(&name)?;
                Ok(Reply::message(format!(
                    "Saved {name} from your login shell. Restart the server to use it."
                )))
            }
            Action::RemoveServerKey { name } => {
                self.declared_key(&name)?;
                server::remove_server_key(&name)?;
                Ok(Reply::message(format!(
                    "Removed the saved key for {name}. The server will not start until it has one."
                )))
            }
            Action::SavePreferences {
                baseline_model,
                refresh_seconds,
            } => {
                crate::config::save_preferences(&self.settings, &baseline_model, refresh_seconds)?;
                self.reload()?;
                Ok(Reply::message(format!(
                    "Changed {}.",
                    self.settings.display()
                )))
            }
            Action::OpenSettings {} => unreachable!(),
            Action::PreviewUpdate {} => {
                let mut data = switchyard_desktop::install::preview_update(&install_metadata(
                    &self.settings,
                )?)?;
                let token = uuid::Uuid::now_v7().to_string();
                let mut reviewed = data.clone();
                reviewed
                    .as_object_mut()
                    .ok_or("Invalid update preview.")?
                    .remove("status");
                self.update_review = Some((token.clone(), reviewed));
                data["preview_token"] = json!(token);
                Ok(Reply {
                    message: String::new(),
                    data,
                })
            }
            Action::Update { preview_token } => {
                // Each update attempt uses up the reviewed preview, so a second Update needs a new
                // PreviewUpdate even when the checkout has not changed.
                let (token, reviewed) = self
                    .update_review
                    .take()
                    .ok_or("Review the source update first.")?;
                let mut current = switchyard_desktop::install::preview_update(&install_metadata(
                    &self.settings,
                )?)?;
                current
                    .as_object_mut()
                    .ok_or("Invalid update preview.")?
                    .remove("status");
                if preview_token != Some(token) || current != reviewed {
                    return Err(
                        "The source checkout or installation changed. Review the update again."
                            .into(),
                    );
                }
                update(&self.settings).map(Reply::message)
            }
            Action::OpenUpdateLog {} => {
                let log = self.update_log()?;
                server::command("open", &[&log.display().to_string()])?;
                Ok(Reply::message("Opened the update log.".into()))
            }
            Action::OpenConfig {} => {
                server::command("open", &[&self.config.config_file.display().to_string()])?;
                Ok(Reply::message(
                    "Opened the server config in your default editor.".into(),
                ))
            }
            Action::OpenPrices {} => {
                let Some(path) = self.config.prices_file.as_ref() else {
                    return Ok(Reply::message("Bundled model prices are active. To use custom rates, set pricing_file in app settings and pass the same file to the server with --pricing-file.".into()));
                };
                server::command("open", &[&path.display().to_string()])?;
                Ok(Reply::message(
                    "Opened the model prices in your default editor. Restart the server after \
                     you save them; it reads the prices when it starts."
                        .into(),
                ))
            }
        }
    }
}
fn logs_dir() -> Result<PathBuf, String> {
    crate::config::expand_home(std::path::Path::new("~/.switchyard/logs"))
}
fn known_dependents(id: &str) -> Vec<Dependent> {
    let mut dependents = Vec::new();
    for &(tool, label) in harness::HARNESSES {
        let mut destinations = vec![(None, harness::paths(tool))];
        destinations.extend(
            accounts::list(tool)
                .into_iter()
                .map(|(name, path)| (Some(name), accounts::config_paths(tool, Some(&path)))),
        );
        for (account, files) in destinations {
            let Ok(files) = files else {
                continue;
            };
            if harness::inspect(tool, &files)
                .is_ok_and(|status| status.lines().next() == Some(format!("Model: {id}").as_str()))
            {
                dependents.push(Dependent {
                    tool,
                    label,
                    account,
                    files,
                });
            }
        }
    }
    dependents
}

fn role_info(role: &crate::server_config::Role) -> RoleInfo {
    RoleInfo {
        label: role.label.clone(),
        hint: role.hint,
        tier: role.tier.map(|t| format!("{t:?}")),
    }
}

fn algorithm_by_id(kind: &str) -> Result<&'static crate::server_config::Algorithm, String> {
    ALGORITHMS
        .iter()
        .find(|a| a.kind == kind)
        .ok_or_else(|| "Unknown routing algorithm.".into())
}
fn current_route(config: &ServerConfig, key: &str, id: Option<&str>) -> Result<Route, String> {
    config
        .routes()
        .into_iter()
        .find(|r| r.key == key && id.is_none_or(|id| id == r.id))
        .ok_or_else(|| "The route changed or was removed. Refresh and choose it again.".into())
}
// A custom file selects one settings location; saved accounts select their own folders.
fn install_paths(
    tool: Harness,
    account: Option<&str>,
    file: Option<&str>,
) -> Result<Vec<PathBuf>, String> {
    if let Some(file) = file {
        if account.is_some() {
            return Err("Choose a saved account or a custom settings file, not both.".into());
        }
        let path = PathBuf::from(file);
        if !path.is_absolute() || file.trim().is_empty() || file.contains(['\n', '\r']) {
            return Err("Enter an absolute settings file path without line breaks.".into());
        }
        if path.is_dir() {
            return Err("Choose a settings file, not a folder.".into());
        }
        if tool == Harness::Pi {
            if path.file_name().is_none_or(|name| name != "models.json") {
                return Err("Choose Pi’s models.json; settings.json is read beside it.".into());
            }
            return Ok(vec![path.clone(), path.with_file_name("settings.json")]);
        }
        return Ok(vec![path]);
    }
    let account = account_path(tool, account)?;
    accounts::config_paths(tool, account.as_deref())
}

fn account_path(tool: Harness, name: Option<&str>) -> Result<Option<PathBuf>, String> {
    let Some(name) = name else {
        return Ok(None);
    };
    let path = accounts::list(tool)
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, p)| p)
        .ok_or("Unknown account. Refresh and choose an existing account.")?;
    accounts::validate(tool, &path)?;
    Ok(Some(path))
}
fn login_mode(
    tool: Harness,
    config: &ServerConfig,
    route: &crate::server_config::Route,
) -> Result<bool, String> {
    let clients = config.clients();
    let choices = config.choices(&route.key);
    if choices.is_empty() {
        return Err(
            "This route cannot be installed by the app. Choose a route with models.".into(),
        );
    }
    let mut login = false;
    for choice in choices {
        let client = clients
            .iter()
            .find(|c| c.name == choice.client)
            .ok_or("The route has an unknown endpoint.")?;
        if client.forward_auth {
            let url = url::Url::parse(&client.base_url)
                .map_err(|_| "The route has an invalid endpoint URL.")?;
            let host = url
                .host_str()
                .ok_or("The route has an invalid endpoint URL.")?;
            let supported = url.scheme() == "https"
                && url.port().is_none_or(|port| port == 443)
                && match tool {
                    Harness::CodexCli | Harness::CodexApp => host == "chatgpt.com",
                    Harness::Claude => host == "api.anthropic.com",
                    Harness::Pi => false,
                };
            if !supported {
                return Err(format!(
                    "This route forwards the caller's login to {host}. \
                     Pick a matching harness or a route with server API credentials."
                ));
            }
            login = true;
        }
    }
    Ok(login)
}

/// A transcript changed this recently suggests that its session is still running.
const RECENT_MS: i64 = 2 * 60 * 1000;
/// `HarnessSessions` lists at most this many sessions, newest first.
const LISTED_SESSIONS: usize = 200;
const HANDOFF_NOTICE: &str = "Switchyard writes the hand-off text above to a private file and opens the harness in the \
     session directory. The new session reads the file under its own login and summarizes it; \
     nothing is uploaded, and no login is shared.";

/// The short harness name that hand-offs use. `HARNESSES` labels describe settings files.
pub(crate) fn harness_name(tool: Harness) -> &'static str {
    match tool {
        Harness::CodexCli => "Codex CLI",
        Harness::CodexApp => "Codex app",
        Harness::Claude => "Claude Code",
        Harness::Pi => "Pi",
    }
}

fn source_harness(kind: transcripts::Kind) -> Harness {
    match kind {
        transcripts::Kind::Claude => Harness::Claude,
        transcripts::Kind::Codex => Harness::CodexCli,
    }
}

/// The transcripts that a harness writes. Codex app and Codex CLI share one session store.
fn transcript_kind(tool: Harness) -> Result<transcripts::Kind, String> {
    match tool {
        Harness::Claude => Ok(transcripts::Kind::Claude),
        Harness::CodexCli | Harness::CodexApp => Ok(transcripts::Kind::Codex),
        Harness::Pi => Err("Pi sessions cannot be continued in another harness yet.".into()),
    }
}

/// The directory that holds a harness's transcripts: the account directory, or the harness's
/// current config directory.
fn transcript_root(tool: Harness, account: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(account) = account {
        return Ok(account.to_path_buf());
    }
    harness::paths(tool)?[0]
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "The harness settings have no parent directory.".into())
}

/// The harness that continues a session by default: the other command line harness.
pub(crate) fn other_harness(tool: Harness) -> Harness {
    match tool {
        Harness::Claude => Harness::CodexCli,
        _ => Harness::Claude,
    }
}

/// This lists sessions from harness transcripts, newest first: those of one harness and
/// account, or of both command line harnesses and every saved account when `tool` is `None`.
/// Each session names the harness that continues it by default, with the first route that
/// harness can use, so a caller can offer one action per session.
pub(crate) fn recent_sessions(
    config_file: &Path,
    tool: Option<Harness>,
    account: Option<&str>,
) -> Result<Value, String> {
    let mut roots = Vec::new();
    match tool {
        Some(tool) => {
            let path = account_path(tool, account)?;
            roots.push((
                tool,
                account.map(str::to_owned),
                transcript_root(tool, path.as_deref())?,
            ));
        }
        None => {
            for tool in [Harness::CodexCli, Harness::Claude] {
                if let Ok(root) = transcript_root(tool, None) {
                    roots.push((tool, None, root));
                }
                for (name, path) in accounts::list(tool) {
                    roots.push((tool, Some(name), path));
                }
            }
        }
    }
    let text = std::fs::read_to_string(config_file)
        .map_err(|error| format!("Read {}: {error}", config_file.display()))?;
    let config = ServerConfig::parse(&text)
        .map_err(|_| "The server config contains invalid TOML. Open it in Settings to fix it.")?;
    let routes = config.routes();
    let destination = |tool: Harness| {
        let name = harness_name(tool);
        let mut usable = None;
        let mut rejected = None;
        for route in &routes {
            match login_mode(tool, &config, route) {
                Ok(_) => {
                    usable = Some((route.key.clone(), route.id.clone()));
                    break;
                }
                Err(reason) => {
                    rejected.get_or_insert(reason);
                }
            }
        }
        let available = harness::binary(tool).is_some();
        let reason = if !available {
            Some(format!(
                "{name} is not detected. Install it or add it to PATH."
            ))
        } else if usable.is_none() {
            Some(match rejected {
                Some(why) => format!("No route in the server config can serve {name}. {why}"),
                None => format!("Add a route that {name} can use to the server config."),
            })
        } else {
            None
        };
        json!({
            "tool": tool,
            "label": name,
            "available": available,
            "route": usable.as_ref().map(|(key, _)| key.clone()),
            "id": usable.as_ref().map(|(_, id)| id.clone()),
            "reason": reason,
        })
    };
    let destinations: BTreeMap<Harness, Value> = [Harness::CodexCli, Harness::Claude]
        .into_iter()
        .map(|tool| (tool, destination(tool)))
        .collect();
    // A session in a managed worktree keeps its session ID when it continues.
    let managed: Vec<(PathBuf, String)> = sessions::inventory()
        .unwrap_or_default()
        .iter()
        .filter_map(|receipt| {
            Some((
                canonical(Path::new(receipt["worktree"].as_str()?)),
                receipt["id"].as_str()?.to_owned(),
            ))
        })
        .collect();
    let mut listed = Vec::new();
    let mut errors = Vec::new();
    let mut found = Vec::new();
    let mut skipped = 0;
    let mut elsewhere = 0;
    for (tool, account, root) in &roots {
        let kind = transcript_kind(*tool)?;
        listed.push(json!({
            "tool": tool,
            "label": kind.label(),
            "account": account,
            "transcripts": kind.transcript_dir(root),
        }));
        let listing = match transcripts::list(kind, root) {
            Ok(listing) => listing,
            Err(error) => {
                errors.push(format!("{}: {error}", kind.transcript_dir(root).display()));
                continue;
            }
        };
        skipped += listing.skipped;
        for session in &listing.sessions {
            // A session continues in its own directory, so one recorded on another computer,
            // such as a remote session, or in a folder that is gone, is counted and left out.
            if !session.cwd.as_deref().is_some_and(Path::is_dir) {
                elsewhere += 1;
                continue;
            }
            let mut value = serde_json::to_value(session).map_err(|error| error.to_string())?;
            let managed_id = session.cwd.as_deref().map(canonical).and_then(|cwd| {
                managed
                    .iter()
                    .find(|(worktree, _)| cwd.starts_with(worktree))
                    .map(|(_, id)| id.clone())
            });
            value["tool"] = json!(tool);
            value["label"] = json!(kind.label());
            value["account"] = json!(account);
            value["managed_id"] = json!(managed_id);
            value["destination"] = destinations[&other_harness(*tool)].clone();
            found.push(value);
        }
    }
    found.sort_by(|left, right| {
        right["last_activity"]
            .as_str()
            .cmp(&left["last_activity"].as_str())
    });
    let total = found.len();
    found.truncate(LISTED_SESSIONS);
    Ok(json!({
        "sessions": found,
        "total": total,
        "skipped": skipped,
        "elsewhere": elsewhere,
        "roots": listed,
        "errors": errors,
        "limit": LISTED_SESSIONS,
    }))
}

// Harnesses record the resolved directory, which differs from the recorded one through
// symlinks such as macOS's /var → /private/var.
pub(crate) fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn parse_time(text: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|time| time.with_timezone(&chrono::Utc))
}

/// This finds the transcript of a managed session's latest launch. Claude launches record
/// their session UUID; Codex names its thread itself, so the newest rollout started in the
/// session directory after the launch stands in for it.
fn managed_transcript(receipt: &Value) -> Result<(transcripts::Kind, PathBuf), String> {
    let launch = receipt["launches"]
        .as_array()
        .and_then(|launches| launches.last())
        .unwrap_or(receipt);
    let tool: Harness = serde_json::from_value(launch["tool"].clone())
        .map_err(|_| "The session receipt names an unknown harness.")?;
    let kind = transcript_kind(tool)?;
    let account = launch["account_directory"].as_str().map(PathBuf::from);
    let root = transcript_root(tool, account.as_deref())?;
    if let Some(native) = launch["native_session_id"].as_str() {
        return Ok((kind, transcripts::locate(kind, &root, native)?));
    }
    let directory = canonical(&session_directory(receipt)?);
    let launched = launch["started_at"].as_str().and_then(parse_time);
    let listing = transcripts::list(kind, &root)?;
    listing
        .sessions
        .into_iter()
        .filter(|session| {
            session
                .cwd
                .as_deref()
                .is_some_and(|cwd| canonical(cwd) == directory)
        })
        .find(|session| {
            match (launched, session.started.as_deref().and_then(parse_time)) {
                // A rollout's own timestamp can precede the launch record by the time the
                // harness took to start.
                (Some(launched), Some(begun)) => begun + chrono::Duration::minutes(1) >= launched,
                _ => true,
            }
        })
        .map(|session| (kind, session.transcript))
        .ok_or_else(|| {
            format!(
                "No {} transcript started in {} after the launch. Send one message in that \
                 session first, or choose its transcript file.",
                kind.label(),
                directory.display()
            )
        })
}

/// The directory a managed session's harness starts in: the worktree, or the recorded folder
/// within it.
fn session_directory(receipt: &Value) -> Result<PathBuf, String> {
    let worktree = PathBuf::from(
        receipt["worktree"]
            .as_str()
            .ok_or("Missing session worktree.")?,
    );
    Ok(match receipt["relative_directory"].as_str() {
        Some(relative) if !relative.is_empty() => worktree.join(relative),
        _ => worktree,
    })
}

fn current_branch(directory: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(["branch", "--show-current"])
        .output()
        .ok()?;
    let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (output.status.success() && !branch.is_empty()).then_some(branch)
}

/// Claude Code registers each open session under `<config dir>/sessions`. The transcript
/// lives at `<config dir>/projects/<slug>/<uuid>.jsonl`.
fn claude_open(transcript: &Path, native_id: &str) -> bool {
    let Some(config_dir) = transcript.ancestors().nth(3) else {
        return false;
    };
    let Ok(entries) = std::fs::read_dir(config_dir.join("sessions")) else {
        return false;
    };
    entries.flatten().any(|entry| {
        std::fs::read(entry.path())
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .is_some_and(|record| record["sessionId"] == native_id)
    })
}

/// The instructions that point the destination at the hand-off. The excerpt itself never goes
/// on the command line.
fn pointer(kind: transcripts::Kind, file: &Path) -> String {
    format!(
        "Switchyard handed a {} session to you. Its state is in the file {}. Read that file \
         before anything else. Then summarize the state in at most five lines, confirm the \
         next step, and continue the work. Do not redo completed work.",
        kind.label(),
        file.display()
    )
}

fn first_prompt(file: &Path) -> String {
    format!(
        "Read the hand-off file {} and continue from where the previous session stopped.",
        file.display()
    )
}

fn install_metadata(settings: &std::path::Path) -> Result<PathBuf, String> {
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let bundle = switchyard_desktop::install::bundle_metadata(&executable);
    let metadata = if bundle.is_file() {
        bundle
    } else {
        settings
            .parent()
            .ok_or("The settings file has no parent directory.")?
            .join("install.toml")
    };
    Ok(metadata)
}
fn update(settings: &std::path::Path) -> Result<String, String> {
    let metadata = install_metadata(settings)?;
    let log = switchyard_desktop::install::start_update(&metadata)?;
    Ok(format!(
        "Started rebuilding and reinstalling from the saved source checkout. \
         Progress and errors are recorded in {}. \
         The app restarts after a successful install.",
        log.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recent_call_read_errors_name_the_source_and_keep_the_io_details() {
        let directory = tempfile::tempdir().expect("directory");
        let settings = directory.path().join("desktop.toml");
        let path = directory.path().to_str().expect("UTF-8 path");
        std::fs::write(
            &settings,
            format!("routing_log = {path:?}\nserver_url = 'http://127.0.0.1:9'\n"),
        )
        .expect("settings");
        let config = Config::load(&settings).expect("config");
        let expected = crate::history::load(directory.path())
            .err()
            .expect("directory read fails");
        let snapshot = Controller::new(config, settings)
            .snapshot()
            .expect("snapshot");
        assert!(
            snapshot
                .errors
                .contains(&format!("Could not read recent calls: {expected}")),
            "{:?}",
            snapshot.errors
        );
        assert_eq!(snapshot.activity, [0; 24]);
    }

    // An update attempt uses up the reviewed preview, so the same token cannot start a second one.
    #[test]
    fn update_preview_tokens_work_once() {
        let directory = tempfile::tempdir().expect("directory");
        let settings = directory.path().join("desktop.toml");
        std::fs::write(&settings, "server_url = 'http://127.0.0.1:9'\n").expect("settings");
        let config = Config::load(&settings).expect("config");
        let mut controller = Controller::new(config, settings);
        // The temporary directory has no install.toml, so the first attempt fails after it takes
        // the review.
        controller.update_review = Some(("token".into(), serde_json::json!({})));
        let update = || Action::Update {
            preview_token: Some("token".into()),
        };
        let first = controller.dispatch(update()).err().expect("no install");
        assert_ne!(first, "Review the source update first.");
        assert_eq!(
            controller.dispatch(update()).err().as_deref(),
            Some("Review the source update first.")
        );
    }

    #[test]
    fn setup_and_settings_share_saves_and_removal_requires_the_reviewed_bytes() {
        let dir = tempfile::tempdir().expect("fixture");
        let path = dir.path().join("desktop.toml");
        let mut config = Config::defaults_for(&path).expect("defaults");
        config.server_url = "http://127.0.0.1:1".into();
        let mut controller = Controller::new(config.clone(), path.clone());
        assert!(controller.snapshot().expect("initial snapshot").needs_setup);
        let mut unknown = serde_json::to_value(&config).expect("config");
        unknown["unknown_file"] = json!("ignored.toml");
        assert!(
            serde_json::from_value::<Action>(
                json!({"kind":"save_configuration","config":unknown,"create_server":true})
            )
            .is_err()
        );
        assert!(!path.exists());
        controller
            .dispatch(Action::SaveConfiguration {
                config: config.clone(),
                create_server: true,
            })
            .expect("setup save");
        assert!(!controller.snapshot().expect("saved snapshot").needs_setup);
        let server_bytes = std::fs::read(&config.config_file).expect("server");
        let selected = dir.path().join("selected.toml");
        std::fs::write(&selected, &server_bytes).expect("another server config");
        config.config_file = selected;
        config.routing_log = dir.path().join("selected.jsonl");
        controller
            .dispatch(Action::SaveConfiguration {
                config: config.clone(),
                create_server: false,
            })
            .expect("settings save");
        let snapshot = controller.snapshot().expect("changed snapshot");
        assert_eq!(
            snapshot.settings.expect("settings").config_file,
            config.config_file
        );
        assert_eq!(
            snapshot.generation, 1,
            "selecting a different config invalidates route drafts"
        );
        let preview = controller
            .dispatch(Action::PreviewResetSettings {})
            .expect("preview");
        let original = std::fs::read_to_string(&path).expect("preferences");
        std::fs::write(&path, format!("{original}# external change\n")).expect("external edit");
        assert!(
            controller
                .dispatch(Action::ResetSettings {
                    preview_token: preview.data["preview_token"].as_str().map(str::to_owned)
                })
                .is_err()
        );
        assert!(path.exists());
        let preview = controller
            .dispatch(Action::PreviewResetSettings {})
            .expect("new preview");
        controller
            .dispatch(Action::ResetSettings {
                preview_token: preview.data["preview_token"].as_str().map(str::to_owned),
            })
            .expect("remove preferences");
        assert!(!path.exists());
        assert_eq!(
            std::fs::read(&config.config_file).expect("server preserved"),
            server_bytes
        );
        assert!(
            std::fs::read_dir(dir.path())
                .expect("files")
                .flatten()
                .any(|entry| entry.file_name().to_string_lossy().contains("removed-"))
        );
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(dir.path().join("missing-settings.toml"), &path)
                .expect("broken settings link");
            let snapshot = controller.snapshot().expect("repair snapshot");
            assert!(!snapshot.needs_setup);
            assert!(snapshot.configuration_error.is_some());
            assert!(controller.dispatch(Action::Restart {}).is_err());
            assert!(
                path.symlink_metadata()
                    .expect("preserved link")
                    .is_symlink()
            );
        }
    }

    // The subprocess isolates harness discovery and every launch destination.
    #[cfg(unix)]
    #[test]
    fn session_launch_requires_an_unchanged_review_before_creating_files() {
        use std::os::unix::fs::PermissionsExt;
        use std::{fs, process::Command};
        if let Some(home) = std::env::var_os("SWITCHYARD_LAUNCH_REVIEW_FIXTURE") {
            let home = PathBuf::from(home);
            let project = home.join("project");
            let git = |args: &[&str]| {
                let result = Command::new("/usr/bin/git")
                    .arg("-C")
                    .arg(&project)
                    .args(args)
                    .output()
                    .expect("git");
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
            };
            fs::create_dir(&project).expect("project");
            git(&["init"]);
            git(&[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "-m",
                "initial",
            ]);
            let settings = home.join("desktop.toml");
            let server = home.join("server.toml");
            let text = "[llm_clients.c]\n\
                        base_url='http://localhost:1234'\n\
                        [targets.m]\n\
                        id='actual'\n\
                        llm_client='c'\n\
                        [routes.r]\n\
                        id='public'\n\
                        type='passthrough'\n\
                        target='m'\n";
            fs::write(&server, text).expect("server");
            fs::write(
                &settings,
                format!(
                    "config_file={server:?}\nrouting_log={:?}\n",
                    home.join("routing.jsonl")
                ),
            )
            .expect("settings");
            let mut controller =
                Controller::new(Config::load(&settings).expect("settings"), settings.clone());
            let preview = |controller: &mut Controller| {
                controller
                    .dispatch(Action::PreviewLaunch {
                        tool: Harness::CodexCli,
                        account: None,
                        route: "r".into(),
                        id: "public".into(),
                        project: project.display().to_string(),
                    })
                    .expect("preview")
            };
            let launch = |token: Option<String>, path: String| Action::Launch {
                tool: Harness::CodexCli,
                account: None,
                route: "r".into(),
                id: "public".into(),
                project: path,
                preview_token: token,
            };
            let reviewed = preview(&mut controller);
            let token = reviewed.data["preview_token"]
                .as_str()
                .expect("token")
                .to_string();
            for rejected in [
                launch(None, project.display().to_string()),
                launch(Some("wrong".into()), project.display().to_string()),
                launch(Some(token.clone()), home.display().to_string()),
            ] {
                assert!(controller.dispatch(rejected).is_err());
            }
            fs::write(&server, format!("{text}# external edit\n")).expect("edit");
            assert!(
                controller
                    .dispatch(launch(Some(token), project.display().to_string()))
                    .is_err()
            );
            assert_eq!(
                fs::read_to_string(&server).expect("preserved"),
                format!("{text}# external edit\n")
            );
            let reviewed = preview(&mut controller);
            let token = reviewed.data["preview_token"]
                .as_str()
                .expect("token")
                .to_string();
            let original = fs::read_to_string(&settings).expect("settings");
            fs::write(&settings, format!("{original}# external settings\n")).expect("edit");
            assert!(
                controller
                    .dispatch(launch(Some(token), project.display().to_string()))
                    .is_err()
            );
            let reviewed = preview(&mut controller);
            let token = reviewed.data["preview_token"]
                .as_str()
                .expect("token")
                .to_string();
            git(&[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "-m",
                "changed",
            ]);
            assert!(
                controller
                    .dispatch(launch(Some(token), project.display().to_string()))
                    .err()
                    .expect("changed HEAD")
                    .contains("project changed")
            );
            assert!(!home.join(".switchyard/worktrees").exists());
            assert!(!home.join("terminal-called").exists());
            assert_eq!(fs::read_dir(&home).expect("files").count(), 4);
            return;
        }
        let home = tempfile::tempdir().expect("fixture");
        let bin = home.path().join("bin");
        fs::create_dir(&bin).expect("bin");
        for name in ["codex", "open", "osascript"] {
            let path = bin.join(name);
            fs::write(
                &path,
                "#!/bin/sh\ntouch \"$HOME/terminal-called\"\nexit 0\n",
            )
            .expect("stub");
            fs::set_permissions(path, fs::Permissions::from_mode(0o755)).expect("executable");
        }
        let output=Command::new(std::env::current_exe().expect("test binary"))
            .args(["--exact", "controller::tests::session_launch_requires_an_unchanged_review_before_creating_files"])
            .env("SWITCHYARD_LAUNCH_REVIEW_FIXTURE",home.path()).env("HOME",home.path())
            .env("PATH",format!("{}:/usr/bin:/bin",bin.display())).output().expect("child");
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    // The subprocess isolates harness discovery, transcripts, hand-off files, and the terminal.
    // The stubs record their arguments and environment, so the test sees what the destination
    // harness receives.
    #[cfg(unix)]
    #[test]
    fn session_handoff_requires_an_unchanged_review_and_opens_the_other_harness() {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;
        use std::{fs, process::Command};
        if let Some(home) = std::env::var_os("SWITCHYARD_HANDOFF_REVIEW_FIXTURE") {
            let home = PathBuf::from(home);
            let project = home.join("project");
            let git = |args: &[&str]| {
                let result = Command::new("/usr/bin/git")
                    .arg("-C")
                    .arg(&project)
                    .args(args)
                    .output()
                    .expect("git");
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
            };
            fs::create_dir(&project).expect("project");
            git(&["init"]);
            git(&["symbolic-ref", "HEAD", "refs/heads/main"]);
            git(&[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "-m",
                "initial",
            ]);
            let settings = home.join("desktop.toml");
            let server = home.join("server.toml");
            fs::write(
                &server,
                "[llm_clients.c]\n\
                 base_url='http://localhost:1234'\n\
                 [targets.m]\n\
                 id='actual'\n\
                 llm_client='c'\n\
                 [routes.r]\n\
                 id='public'\n\
                 type='passthrough'\n\
                 target='m'\n",
            )
            .expect("server");
            fs::write(
                &settings,
                format!(
                    "config_file={server:?}\nrouting_log={:?}\n",
                    home.join("routing.jsonl")
                ),
            )
            .expect("settings");
            let mut controller =
                Controller::new(Config::load(&settings).expect("settings"), settings.clone());
            let revision = sessions::preview(&project).expect("preview")["revision"]
                .as_str()
                .expect("revision")
                .to_string();
            let server_url = controller.config.server_url.clone();
            sessions::launch(
                Harness::CodexCli,
                &project,
                "public",
                &server_url,
                false,
                None,
                &revision,
            )
            .expect("launch");
            let receipt = &sessions::inventory().expect("inventory")[0];
            let id = receipt["id"].as_str().expect("id").to_string();
            let worktree = PathBuf::from(receipt["worktree"].as_str().expect("worktree"));
            // Codex names its thread itself, so a rollout started in the worktree after the
            // launch stands in for the session.
            let codex_home = PathBuf::from(std::env::var_os("CODEX_HOME").expect("CODEX_HOME"));
            let day = codex_home.join("sessions/2026/10/26");
            fs::create_dir_all(&day).expect("rollouts");
            let thread = uuid::Uuid::now_v7().to_string();
            let rollout = day.join(format!("rollout-2026-10-26T10-00-00-{thread}.jsonl"));
            let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
            let line = |record: Value| format!("{record}\n");
            fs::write(
                &rollout,
                [
                    line(
                        json!({"timestamp": now, "type": "session_meta", "payload": {
                        "id": thread, "cwd": worktree, "timestamp": now, "git": {"branch": id}}}),
                    ),
                    line(json!({"timestamp": now, "type": "event_msg", "payload": {
                        "type": "user_message", "message": "Fix the flaky login test"}})),
                    line(json!({"timestamp": now, "type": "event_msg", "payload": {
                        "type": "agent_message",
                        "message": "I changed auth.rs and the test passes now."}})),
                ]
                .concat(),
            )
            .expect("rollout");
            let preview = |controller: &mut Controller, source: HandoffSource, tool: Harness| {
                controller.dispatch(Action::PreviewHandoff {
                    source,
                    tool,
                    account: None,
                    route: "r".into(),
                    id: "public".into(),
                })
            };
            let handoff =
                |source: HandoffSource, tool: Harness, token: Option<String>| Action::Handoff {
                    source,
                    tool,
                    account: None,
                    route: "r".into(),
                    id: "public".into(),
                    preview_token: token,
                    attach: false,
                };
            let managed = HandoffSource::Managed { id: id.clone() };
            assert!(preview(&mut controller, managed.clone(), Harness::Pi).is_err());
            assert!(preview(&mut controller, managed.clone(), Harness::CodexApp).is_err());
            let reviewed =
                preview(&mut controller, managed.clone(), Harness::Claude).expect("preview");
            let data = &reviewed.data;
            let token = data["preview_token"].as_str().expect("token").to_string();
            let text = data["handoff"].as_str().expect("hand-off text");
            assert!(text.starts_with("# Hand-off from Codex"), "{text}");
            assert!(text.contains("Fix the flaky login test") && text.contains("auth.rs"));
            assert_eq!(data["destination"]["directory"], json!(worktree));
            assert_eq!(data["destination"]["session_id"], json!(id));
            assert_eq!(data["source"]["tool"], "codex_cli");
            assert_eq!(data["source"]["native_id"], json!(thread));
            assert_eq!(data["source"]["managed_id"], json!(id));
            let warnings = data["warnings"].to_string();
            assert!(warnings.contains("keeps the session ID"), "{warnings}");
            assert!(warnings.contains("may still be running"), "{warnings}");
            assert!(!warnings.contains("records branch"), "{warnings}");
            let command = data["command"].as_str().expect("command");
            for flag in [
                "--append-system-prompt",
                "--add-dir",
                "--session-id",
                "--settings",
            ] {
                assert!(command.contains(flag), "{command}");
            }
            // The excerpt stays out of the command line.
            assert!(!command.contains("Fix the flaky"), "{command}");
            assert!(!home.join(".switchyard/handoffs").exists());
            for rejected in [
                handoff(managed.clone(), Harness::Claude, None),
                handoff(managed.clone(), Harness::Claude, Some("wrong".into())),
                handoff(managed.clone(), Harness::CodexCli, Some(token.clone())),
            ] {
                assert!(controller.dispatch(rejected).is_err());
            }
            assert!(!home.join(".switchyard/handoffs").exists());
            // The transcript grew after the review.
            let mut file = fs::OpenOptions::new()
                .append(true)
                .open(&rollout)
                .expect("append");
            writeln!(
                file,
                "{}",
                json!({"timestamp": now, "type": "event_msg", "payload": {
                    "type": "user_message", "message": "later"}})
            )
            .expect("write");
            drop(file);
            let error = controller
                .dispatch(handoff(managed.clone(), Harness::Claude, Some(token)))
                .err()
                .expect("stale review");
            assert!(error.contains("changed since"), "{error}");
            assert!(!home.join(".switchyard/handoffs").exists());
            let reviewed =
                preview(&mut controller, managed.clone(), Harness::Claude).expect("second preview");
            let token = reviewed.data["preview_token"]
                .as_str()
                .expect("token")
                .to_string();
            let file = PathBuf::from(reviewed.data["handoff_file"].as_str().expect("file"));
            let done = controller
                .dispatch(handoff(
                    managed.clone(),
                    Harness::Claude,
                    Some(token.clone()),
                ))
                .expect("hand-off");
            assert!(done.message.contains("Claude Code"), "{}", done.message);
            // One review opens one session.
            assert!(
                controller
                    .dispatch(handoff(managed.clone(), Harness::Claude, Some(token)))
                    .is_err()
            );
            let dir = file.parent().expect("hand-off directory");
            assert_eq!(
                fs::metadata(dir).expect("dir").permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(&file).expect("file").permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                fs::read_to_string(&file).expect("hand-off"),
                reviewed.data["handoff"].as_str().expect("text")
            );
            assert!(dir.join("settings.json").is_file());
            let record: Value =
                serde_json::from_slice(&fs::read(dir.join("handoff.json")).expect("record"))
                    .expect("json");
            assert_eq!(record["opened"], true);
            assert_eq!(record["file"], json!(file));
            assert_eq!(record["destination"]["session_id"], json!(id));
            assert_eq!(record["source"]["native_id"], json!(thread));
            let receipt = sessions::managed_session(&id).expect("receipt");
            let launches = receipt["launches"].as_array().expect("launches");
            assert_eq!(launches.len(), 2);
            assert_eq!(launches[1]["tool"], "claude");
            assert_eq!(launches[1]["route"], "public");
            assert_eq!(launches[1]["handoff"]["from_tool"], "codex_cli");
            assert_eq!(launches[1]["handoff"]["from_native_id"], json!(thread));
            assert_eq!(launches[1]["handoff"]["file"], json!(file));
            assert_eq!(launches[1]["settings_directory"], json!(dir));
            assert!(
                uuid::Uuid::parse_str(
                    launches[1]["native_session_id"]
                        .as_str()
                        .expect("claude session")
                )
                .is_ok()
            );
            assert_eq!(receipt["tool"], "claude");
            assert!(sessions::cleanup_preview(&id).is_ok());
            let arguments = wait_for(&home.join("claude-arguments"), |_| true);
            assert!(
                arguments.contains(&format!("--add-dir\n{}\n", dir.display())),
                "{arguments}"
            );
            assert!(
                arguments.contains(&format!(
                    "--settings\n{}\n",
                    dir.join("settings.json").display()
                )),
                "{arguments}"
            );
            assert!(
                arguments.contains("--append-system-prompt\nSwitchyard handed a Codex session"),
                "{arguments}"
            );
            assert!(
                arguments.contains(&format!("x-switchyard-session-id: {id}\n")),
                "{arguments}"
            );
            assert!(
                arguments.contains(&format!("Read the hand-off file {}", file.display())),
                "{arguments}"
            );
            assert!(!arguments.contains("Fix the flaky login test"));
            let snapshot = controller.snapshot().expect("snapshot");
            assert_eq!(snapshot.handoffs.len(), 1);
            assert_eq!(snapshot.handoffs[0]["id"], record["id"]);

            // A transcript that the app did not start continues in its own directory under a
            // new session ID, and no receipt is written for it.
            let claude_dir =
                PathBuf::from(std::env::var_os("CLAUDE_CONFIG_DIR").expect("CLAUDE_CONFIG_DIR"));
            let session = sessions::claude_session_id();
            let slug = claude_dir.join("projects/-fixture-project");
            fs::create_dir_all(&slug).expect("projects");
            let transcript = slug.join(format!("{session}.jsonl"));
            fs::write(
                &transcript,
                [
                    line(json!({"type": "user", "cwd": project, "sessionId": session,
                        "gitBranch": "main", "timestamp": now,
                        "message": {"role": "user", "content": "Add retries to the uploader"}})),
                    line(
                        json!({"type": "assistant", "cwd": project, "sessionId": session,
                        "timestamp": now, "message": {"id": "m1", "role": "assistant",
                        "model": "claude-x",
                        "content": [{"type": "text", "text": "Added retries in uploader.rs."}]}}),
                    ),
                ]
                .concat(),
            )
            .expect("transcript");
            // A transcript from another computer records a directory that is not here, so it
            // is counted and left out of the listing.
            let remote_slug = claude_dir.join("projects/-remote-box-home");
            fs::create_dir_all(&remote_slug).expect("projects");
            let remote = sessions::claude_session_id();
            fs::write(
                remote_slug.join(format!("{remote}.jsonl")),
                line(
                    json!({"type": "user", "cwd": home.join("not-on-this-computer"),
                    "sessionId": remote, "timestamp": now,
                    "message": {"role": "user", "content": "Remote work"}}),
                ),
            )
            .expect("remote transcript");
            let listing = controller
                .dispatch(Action::RecentSessions {
                    tool: Some(Harness::Claude),
                    account: None,
                })
                .expect("listing");
            assert_eq!(listing.data["total"], 1);
            assert_eq!(listing.data["elsewhere"], 1);
            assert_eq!(listing.data["sessions"][0]["transcript"], json!(transcript));
            assert_eq!(listing.data["sessions"][0]["cwd"], json!(project));
            // Without a harness, both harnesses list together, and each session names the
            // other harness with the first route it can use.
            let merged = controller
                .dispatch(Action::RecentSessions {
                    tool: None,
                    account: None,
                })
                .expect("merged listing");
            let all = merged.data["sessions"].as_array().expect("sessions");
            assert_eq!(all.len(), 2, "{all:?}");
            assert_eq!(merged.data["elsewhere"], 1);
            let claude = all
                .iter()
                .find(|s| s["tool"] == "claude")
                .expect("claude session");
            let codex = all
                .iter()
                .find(|s| s["tool"] == "codex_cli")
                .expect("codex session");
            assert_eq!(claude["destination"]["tool"], "codex_cli");
            assert_eq!(codex["destination"]["tool"], "claude");
            assert_eq!(codex["destination"]["route"], "r");
            assert_eq!(codex["destination"]["id"], "public");
            assert_eq!(codex["destination"]["available"], true);
            assert!(codex["destination"]["reason"].is_null(), "{codex}");
            assert_eq!(codex["managed_id"], json!(id));
            assert!(claude["managed_id"].is_null());
            assert_eq!(merged.data["roots"].as_array().map(Vec::len), Some(2));
            assert!(
                preview(
                    &mut controller,
                    HandoffSource::Transcript {
                        path: settings.display().to_string()
                    },
                    Harness::CodexCli
                )
                .is_err()
            );
            let source = HandoffSource::Transcript {
                path: transcript.display().to_string(),
            };
            let reviewed = preview(&mut controller, source.clone(), Harness::CodexCli)
                .expect("transcript preview");
            let new_id = reviewed.data["destination"]["session_id"]
                .as_str()
                .expect("session id")
                .to_string();
            assert!(new_id.starts_with("sy-") && new_id != id);
            assert_eq!(reviewed.data["destination"]["directory"], json!(project));
            assert_eq!(reviewed.data["source"]["tool"], "claude");
            assert!(reviewed.data["source"]["managed_id"].is_null());
            let command = reviewed.data["command"].as_str().expect("command");
            assert!(command.contains("developer_instructions="), "{command}");
            assert!(command.contains("model_providers.sy="), "{command}");
            let token = reviewed.data["preview_token"]
                .as_str()
                .expect("token")
                .to_string();
            controller
                .dispatch(handoff(source, Harness::CodexCli, Some(token)))
                .expect("transcript hand-off");
            assert_eq!(sessions::inventory().expect("inventory").len(), 1);
            assert!(sessions::managed_session(&new_id).is_err());
            let arguments = wait_for(&home.join("codex-arguments"), |text| {
                text.contains("developer_instructions")
            });
            assert!(arguments.contains(&new_id) && !arguments.contains(&id));
            assert!(arguments.contains("Read the hand-off file"), "{arguments}");
            assert!(!arguments.contains("Add retries to the uploader"));
            assert_eq!(controller.snapshot().expect("snapshot").handoffs.len(), 2);
            return;
        }
        let home = tempfile::tempdir().expect("fixture");
        let bin = home.path().join("bin");
        fs::create_dir(&bin).expect("bin");
        for name in ["codex", "claude"] {
            let path = bin.join(name);
            fs::write(
                &path,
                format!(
                    "#!/bin/bash\n\
                     printf '%s\\n' \"${{CODEX_HOME:-}}\" \"${{CLAUDE_CONFIG_DIR:-}}\" \
                     \"${{ANTHROPIC_CUSTOM_HEADERS:-}}\" \"$@\" > \"$HOME/{name}-arguments.partial\" \
                     && mv \"$HOME/{name}-arguments.partial\" \"$HOME/{name}-arguments\"\n"
                ),
            )
            .expect("stub");
            fs::set_permissions(path, fs::Permissions::from_mode(0o755)).expect("executable");
        }
        // macOS runs the script through `open`; Linux runs it through the program in TERMINAL,
        // which receives `-e` before the script path.
        let terminal = bin.join(if cfg!(target_os = "linux") {
            "terminal"
        } else {
            "open"
        });
        fs::write(
            &terminal,
            "#!/bin/bash\n[ \"$1\" = -e ] && shift\nbash \"$1\"\n",
        )
        .expect("terminal");
        fs::set_permissions(&terminal, fs::Permissions::from_mode(0o755)).expect("executable");
        let output = Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "controller::tests::session_handoff_requires_an_unchanged_review_and_opens_the_other_harness",
            ])
            .env("SWITCHYARD_HANDOFF_REVIEW_FIXTURE", home.path())
            .env("HOME", home.path())
            .env("CODEX_HOME", home.path().join("codex-home"))
            .env("CLAUDE_CONFIG_DIR", home.path().join("claude-config"))
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .env("TERMINAL", &terminal)
            .output()
            .expect("child");
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// The terminal starts apart from the hand-off, so a stub's record can arrive after the
    /// action returns. This waits for the file to hold what `ready` expects.
    #[cfg(unix)]
    fn wait_for(path: &std::path::Path, ready: impl Fn(&str) -> bool) -> String {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Ok(text) = std::fs::read_to_string(path)
                && ready(&text)
            {
                return text;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "{} was not written in time",
                path.display()
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    #[test]
    // An algorithm switch must not give stale choices a revision for externally edited bytes.
    fn algorithm_switches_keep_the_first_revision_and_external_edits_reject_it() {
        let dir = tempfile::tempdir().expect("fixture");
        let settings = dir.path().join("desktop.toml");
        let config = dir.path().join("server.toml");
        let text = "[llm_clients.c]\n\
                    base_url='http://localhost:1234'\n\
                    [targets.m]\n\
                    id='actual'\n\
                    llm_client='c'\n\
                    [routes.r]\n\
                    id='public'\n\
                    type='passthrough'\n\
                    target='m'\n";
        std::fs::write(&config, text).expect("config");
        std::fs::write(
            &settings,
            format!(
                "config_file={config:?}\nrouting_log={:?}",
                dir.path().join("log")
            ),
        )
        .expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("load"), settings);
        let first = controller
            .dispatch(Action::Editor {
                generation: 0,
                route: "r".into(),
                algorithm: "passthrough".into(),
            })
            .expect("editor");
        let second = controller
            .dispatch(Action::Editor {
                generation: 0,
                route: "r".into(),
                algorithm: "random".into(),
            })
            .expect("switch roles");
        assert_eq!(first.data["revision"], second.data["revision"]);
        assert_eq!(
            controller.snapshot().expect("snapshot").routes[0].revision,
            first.data["revision"]
        );
        std::fs::write(&config, format!("{text}# external edit\n")).expect("external edit");
        let latest = controller
            .dispatch(Action::Editor {
                generation: 0,
                route: "r".into(),
                algorithm: "passthrough".into(),
            })
            .expect("new editor");
        assert_ne!(latest.data["revision"], first.data["revision"]);
        let rejected = controller.dispatch(Action::PreviewRoute {
            generation: 0,
            route: "r".into(),
            algorithm: "passthrough".into(),
            choices: vec![Choice {
                client: "c".into(),
                model: "replacement".into(),
            }],
            revision: first.data["revision"].as_str().map(str::to_string),
        });
        assert!(rejected.is_err());
        assert_eq!(
            std::fs::read_to_string(&config).expect("config"),
            format!("{text}# external edit\n")
        );
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 2);
    }

    #[test]
    // A custom destination must use the same files for preview, installation, backup, and restore.
    fn custom_settings_are_validated_and_restored() {
        let dir = tempfile::tempdir().expect("directory");
        for (tool, _) in harness::HARNESSES {
            let file = dir.path().join(if *tool == Harness::Pi {
                "models.json"
            } else {
                "custom-settings"
            });
            assert_eq!(
                install_paths(*tool, None, Some(file.to_str().expect("path"))).expect("custom")[0],
                file
            );
        }
        for (account, file) in [
            (None, "relative.json"),
            (None, ""),
            (Some("account"), "/tmp/settings.json"),
            (None, "/tmp/settings\n.json"),
        ] {
            assert!(install_paths(Harness::Claude, account, Some(file)).is_err());
        }
        assert!(install_paths(Harness::Pi, None, Some("/tmp/settings.json")).is_err());
        assert!(install_paths(Harness::Claude, None, dir.path().to_str()).is_err());
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 0);
        let settings = dir.path().join("app.toml");
        let config = dir.path().join("server.toml");
        std::fs::write(
            &config,
            "[llm_clients.c]\n\
             base_url='https://example.com/v1'\n\
             format='openai_chat'\n\
             [targets.m]\n\
             id='actual'\n\
             llm_client='c'\n\
             [routes.r]\n\
             id='public'\n\
             type='passthrough'\n\
             target='m'\n",
        )
        .expect("config");
        std::fs::write(
            &settings,
            format!(
                "config_file={config:?}\nrouting_log={:?}",
                dir.path().join("log")
            ),
        )
        .expect("settings");
        let custom = dir.path().join("custom.json");
        std::fs::write(&custom, "{\"user\":true}").expect("custom");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        let file = Some(custom.display().to_string());
        let preview = controller
            .dispatch(Action::PreviewInstall {
                tool: Harness::Claude,
                account: None,
                settings_file: file.clone(),
                route: "r".into(),
                id: "public".into(),
            })
            .expect("preview");
        assert!(
            preview.data["changes"]
                .as_array()
                .expect("diff")
                .iter()
                .any(|change| change["file"] == custom.display().to_string())
        );
        std::fs::write(&custom, "{\"external\":true}").expect("external edit");
        let rejected = controller.dispatch(Action::Install {
            tool: Harness::Claude,
            account: None,
            settings_file: file.clone(),
            route: "r".into(),
            id: "public".into(),
            preview_token: preview.data["preview_token"].as_str().map(str::to_string),
        });
        assert!(rejected.is_err());
        assert_eq!(
            std::fs::read_to_string(&custom).expect("custom"),
            "{\"external\":true}"
        );
        assert!(
            !custom
                .with_file_name("custom.json.switchyard-original")
                .exists()
        );
        std::fs::write(&custom, "{\"user\":true}").expect("restore fixture");
        controller
            .dispatch(Action::Install {
                tool: Harness::Claude,
                account: None,
                settings_file: file.clone(),
                route: "r".into(),
                id: "public".into(),
                preview_token: preview.data["preview_token"].as_str().map(str::to_string),
            })
            .expect("install");
        assert_eq!(
            preview.data["proposed"],
            harness::inspect(Harness::Claude, std::slice::from_ref(&custom)).expect("inspect")
        );
        let restore_preview = controller
            .dispatch(Action::PreviewRestore {
                tool: Harness::Claude,
                account: None,
                settings_file: file.clone(),
            })
            .expect("restore preview");
        controller
            .dispatch(Action::Restore {
                tool: Harness::Claude,
                account: None,
                settings_file: file,
                preview_token: restore_preview.data["preview_token"]
                    .as_str()
                    .map(str::to_string),
            })
            .expect("restore");
        assert_eq!(
            std::fs::read_to_string(custom).expect("custom"),
            "{\"user\":true}"
        );
    }

    #[cfg(any(target_os = "macos", windows))]
    #[test]
    // A cached list must not let an untested key reach Keychain.
    fn submitting_a_key_checks_upstream_even_when_refresh_is_false() {
        use std::io::{Read, Write};
        let dir = tempfile::tempdir().expect("directory");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener");
        listener.set_nonblocking(true).expect("nonblocking");
        let url = format!("http://{}/v1", listener.local_addr().expect("address"));
        #[cfg(windows)]
        assert_eq!(
            switchyard_desktop::install::windows::saved_key(&url).expect("no saved key"),
            None
        );
        let worker = std::thread::spawn(move || {
            for _ in 0..200 {
                if let Ok((mut stream, _)) = listener.accept() {
                    stream.set_nonblocking(false).expect("blocking request");
                    stream
                        .set_read_timeout(Some(std::time::Duration::from_secs(1)))
                        .expect("timeout");
                    let mut request = Vec::new();
                    let mut chunk = [0; 1024];
                    for _ in 0..4 {
                        let n = stream.read(&mut chunk).expect("request");
                        request.extend_from_slice(&chunk[..n]);
                        if n == 0 || request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                            break;
                        }
                    }
                    stream
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\n\
                              Content-Length: 0\r\n\
                              Connection: close\r\n\
                              \r\n",
                        )
                        .expect("response");
                    return String::from_utf8_lossy(&request).into_owned();
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            String::new()
        });
        let settings = dir.path().join("settings.toml");
        let routes = dir.path().join("routes.toml");
        std::fs::write(
            &routes,
            format!(
                "[llm_clients.c]\n\
                 base_url={url:?}\n\
                 format='openai_responses'\n\
                 forward_auth=true\n\
                 [targets.m]\n\
                 id='actual'\n\
                 llm_client='c'\n\
                 [routes.r]\n\
                 id='public'\n\
                 type='passthrough'\n\
                 target='m'\n"
            ),
        )
        .expect("routes");
        std::fs::write(
            &settings,
            format!(
                "config_file={routes:?}\nrouting_log={:?}",
                dir.path().join("log")
            ),
        )
        .expect("settings");
        let cached =
            json!({format!("{url}/models"): {"models":["cached"],"fetched_at":models::now()}})
                .to_string();
        let cache = dir.path().join(models::CACHE_FILE);
        std::fs::write(&cache, &cached).expect("cache");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        let cached_reply = controller
            .dispatch(Action::Models {
                client: "c".into(),
                refresh: false,
                key: None,
            })
            .expect("cached list");
        assert_eq!(cached_reply.data["warning"], false);
        assert_eq!(cached_reply.data["source"], "cache");
        // A failed refresh must not report the cached list as a successful upstream fetch.
        let failed_refresh = controller
            .dispatch(Action::Models {
                client: "c".into(),
                refresh: true,
                key: None,
            })
            .expect("cached fallback");
        assert_eq!(failed_refresh.data["warning"], true);
        assert_eq!(failed_refresh.data["source"], "cache");
        assert_eq!(failed_refresh.data["models"], json!(["cached"]));
        let result = controller.dispatch(Action::Models {
            client: "c".into(),
            refresh: false,
            key: Some("rejected-fixture-key".into()),
        });
        let request = worker.join().expect("worker");
        assert!(result.err().expect("rejected key").contains("401"));
        assert!(request.starts_with("GET /v1/models "));
        assert!(request.contains("Authorization: Bearer rejected-fixture-key"));
        assert_eq!(std::fs::read_to_string(cache).expect("cache"), cached);
        #[cfg(windows)]
        assert_eq!(
            switchyard_desktop::install::windows::saved_key(&url).expect("rejected key not saved"),
            None
        );
        std::fs::write(
            &routes,
            "[llm_clients.c]\n\
             base_url='http://127.0.0.1:9/v1'\n\
             format='openai_responses'\n\
             [targets.m]\n\
             id='actual'\n\
             llm_client='c'\n\
             [routes.r]\n\
             id='public'\n\
             type='passthrough'\n\
             target='m'\n",
        )
        .expect("routes");
        let result = controller.dispatch(Action::Models {
            client: "c".into(),
            refresh: false,
            key: Some("unused-fixture-key".into()),
        });
        assert!(
            result
                .err()
                .expect("unused key")
                .contains("does not accept a model-list key")
        );
    }
    #[test]
    // Matching route names in two files must not let an old draft edit the newly selected file.
    fn stale_editor_rejects_after_settings_change_without_writes() {
        let dir = tempfile::tempdir().expect("directory");
        let settings = dir.path().join("settings.toml");
        let first = dir.path().join("first.toml");
        let second = dir.path().join("second.toml");
        let text = "[llm_clients.c]\n\
                    base_url='https://example.com/v1'\n\
                    format='openai_responses'\n\
                    [targets.m]\n\
                    id='actual'\n\
                    llm_client='c'\n\
                    [routes.r]\n\
                    id='public'\n\
                    type='passthrough'\n\
                    target='m'\n";
        std::fs::write(&first, text).expect("first");
        std::fs::write(&second, text).expect("second");
        let write_settings = |path: &std::path::Path| {
            std::fs::write(
                &settings,
                format!(
                    "config_file={path:?}\nrouting_log={:?}",
                    dir.path().join("log")
                ),
            )
            .expect("settings");
        };
        write_settings(&first);
        let mut controller =
            Controller::new(Config::load(&settings).expect("settings"), settings.clone());
        let snapshot = controller.snapshot().expect("snapshot");
        assert_eq!(
            snapshot.routes[0].tiers,
            Some(vec![Some("Capable".to_string())])
        );
        let generation = snapshot.generation;
        write_settings(&second);
        for action in [
            Action::Editor {
                generation,
                route: "r".into(),
                algorithm: "passthrough".into(),
            },
            Action::Apply {
                generation,
                route: "r".into(),
                algorithm: "passthrough".into(),
                choices: vec![Choice {
                    client: "c".into(),
                    model: "changed".into(),
                }],
                revision: None,
            },
        ] {
            assert!(controller.dispatch(action).is_err());
        }
        assert_eq!(std::fs::read_to_string(&second).expect("second"), text);
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 3);
        assert_eq!(
            controller.snapshot().expect("snapshot").generation,
            generation + 1
        );
    }

    #[test]
    fn unsupported_routes_are_read_only_before_any_write() {
        let dir = tempfile::tempdir().expect("directory");
        let config = dir.path().join("server.toml");
        let settings = dir.path().join("app.toml");
        std::fs::write(
            &settings,
            format!("config_file={config:?}\nserver_url='http://127.0.0.1:0'"),
        )
        .expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        // These fixtures use valid server algorithms that the app cannot edit.
        for fields in [
            "type='noop'",
            "type='prefill_router'\ntargets=['answer']\ncheckpoint='router.pt'",
            "type='llm_classifier'\n\
             mode='custom'\n\
             models={judge=['judge'],any=['answer']}\n\
             default_target='any'\n\
             prompt='Select a target.'\n\
             response_schema='{\"type\":\"object\"}'\n\
             policy={type='target_selector',selector='/target'}",
        ] {
            let _: switchyard_runner::AlgorithmSpec =
                toml::from_str(fields).expect("server route schema");
            let original = format!("[routes.r]\nid='public'\n{fields}\n");
            std::fs::write(&config, &original).expect("config");
            assert!(!controller.snapshot().expect("snapshot").routes[0].editable);
            for kind in ["editor", "apply"] {
                let mut raw =
                    json!({"kind":kind,"generation":0,"route":"r","algorithm":"passthrough"});
                if kind == "apply" {
                    raw["choices"] = json!([]);
                }
                let action = serde_json::from_value(raw).expect("action");
                assert_eq!(
                    controller.dispatch(action).err().expect("read only"),
                    "This route must be edited in the server config file."
                );
            }
            assert_eq!(std::fs::read_to_string(&config).expect("config"), original);
            assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 2);
        }
    }
}
