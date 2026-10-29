// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The terminal controller reloads settings and combines usage and server health into snapshots.
//! It retains the log reader and health probe across refreshes.

use crate::{
    accounts,
    config::Config,
    harness::{self, Harness},
    history, models, pricing, rollup, server,
    server_config::{ALGORITHMS, Choice, Route, ServerConfig},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

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
        /// This field holds the endpoint's base URL when the user opened the key form. The
        /// controller refuses `key` unless the endpoint still has this URL.
        base_url: Option<String>,
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
    SavePreferences {
        baseline_model: String,
        refresh_seconds: u64,
    },
}

/// Snapshot holds the data shown by the apps. The terminal app destructures every field,
/// so adding a field requires an update to `draw`.
#[derive(Clone, Default, Serialize)]
pub struct Snapshot {
    pub update_status: Value,
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
    /// This field lists the models that answer requests, as `client / model`. The runner
    /// finds them, so it includes models the editor cannot change, such as sub-agent models.
    pub models: Vec<String>,
    /// This field lists the models the route calls only to decide, such as a judge or an advisor.
    pub judges: Vec<String>,
    /// This field lists the editor's current model for each role. It is empty when the apps
    /// cannot edit the route's algorithm.
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
    /// This field holds the URL that the app sends a model-list key to.
    pub base_url: String,
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
        let (path, bytes) = self
            .files
            .first()
            .ok_or("The app did not read a server config.")?;
        let bytes = bytes
            .as_deref()
            .ok_or_else(|| format!("The server config {} does not exist.", path.display()))?;
        std::str::from_utf8(bytes).map_err(|error| {
            format!(
                "The server config {} is not valid UTF-8: {error}",
                path.display()
            )
        })
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

pub struct Controller {
    health_probe: crate::health::Probe,
    settings: PathBuf,
    config: Config,
    reader: rollup::Reader,
    // The generation ties a draft to the server config file selected by the settings.
    generation: u64,
    route_reviews: std::collections::HashMap<String, Review>,
    tool_review: Option<Review>,

    route_change: Option<Review>,
    update_review: Option<(String, Value)>,
}
impl Controller {
    pub fn new(config: Config, settings: PathBuf) -> Self {
        Self {
            config,
            settings,
            health_probe: crate::health::Probe::default(),
            reader: rollup::Reader::default(),
            generation: 0,
            route_reviews: Default::default(),
            tool_review: None,

            route_change: None,
            update_review: None,
        }
    }
    fn reload(&mut self) -> Result<(), String> {
        let config = Config::load(&self.settings)?;
        if config.config_file != self.config.config_file {
            self.generation += 1;
            self.route_reviews.clear();
            self.tool_review = None;
            self.route_change = None;
        }
        self.config = config;

        Ok(())
    }
    fn check_generation(&self, generation: u64) -> Result<(), String> {
        if generation != self.generation {
            return Err("The app settings now select a different server config. \
                        Refresh and choose the route again."
                .into());
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
    /// This function reads the routes and endpoints in the server config. The runner lists each
    /// route's targets, so the Routes page shows every model the server would call. The editor
    /// fields come from the same text and do not limit which models the page shows.
    fn server_config(&mut self, errors: &mut Vec<String>) -> (Vec<RouteInfo>, Vec<ClientInfo>) {
        let review = match Review::capture(json!({}), vec![self.config.config_file.clone()]) {
            Ok(review) => review,
            Err(error) => {
                errors.push(error);
                return Default::default();
            }
        };
        let text = match review.original() {
            Ok(text) => text,
            Err(error) => {
                errors.push(error);
                return Default::default();
            }
        };
        // Parse errors can quote config lines, and those lines can hold header credentials, so
        // the snapshot names only the action that shows the full error.
        let Ok(config) = ServerConfig::parse(text) else {
            errors.push(
                "The server config contains invalid TOML. Open it in Settings to fix it.".into(),
            );
            return Default::default();
        };
        let clients = config
            .clients()
            .iter()
            .map(|c| ClientInfo {
                name: c.name.clone(),
                base_url: c.base_url.clone(),
                host: c.host().to_string(),
                models: config.models_on(&c.name),
                unlisted: models::unlisted(c),
                accepts_key: cfg!(any(target_os = "macos", windows))
                    && !models::unlisted(c)
                    && (c.forward_auth || c.api_key_env.is_some()),
                note: models::missing_env(c).map(models::missing_env_note),
            })
            .collect();
        let Ok(listed) = switchyard_runner::route_targets(text) else {
            errors.push(
                "The server cannot read the routes in the server config. \
                 Run Check server config in Settings to see why."
                    .into(),
            );
            return (Vec::new(), clients);
        };
        self.route_reviews
            .retain(|key, _| listed.iter().any(|route| &route.name == key));
        let selections = harness_selections();
        let mut routes = Vec::with_capacity(listed.len());
        for route in listed {
            let revision = self.remember_route(
                &route.name,
                Review {
                    token: uuid::Uuid::now_v7().to_string(),
                    inputs: json!({"route": route.name}),
                    files: review.files.clone(),
                },
            );
            let models: Vec<String> = route.answer_targets.iter().map(target_label).collect();
            let judges: Vec<String> = route.judge_targets.iter().map(target_label).collect();
            let algorithm = config.algorithm(&route.name);
            routes.push(RouteInfo {
                label: format!(
                    "{} · {} · {}",
                    route.id,
                    route.kind,
                    [models.as_slice(), judges.as_slice()].concat().join(" → ")
                ),
                tiers: algorithm.map(|a| {
                    config
                        .roles(&route.name, a)
                        .iter()
                        .map(|r| r.tier.map(|t| format!("{t:?}")))
                        .collect()
                }),
                compatibility: harness::HARNESSES
                    .iter()
                    .map(|&(tool, _)| (tool, login_mode(tool, &config, &route.name).err()))
                    .collect(),
                dependents: selections
                    .iter()
                    .filter(|(model, _)| model.as_str() == &*route.id)
                    .map(|(_, dependent)| dependent.clone())
                    .collect(),
                editable: algorithm.is_some(),
                generated: config
                    .generated_by("routes", &route.name)
                    .map(str::to_string),
                choices: config.choices(&route.name),
                id: route.id.to_string(),
                kind: route.kind.to_string(),
                key: route.name,
                revision,
                models,
                judges,
            });
        }
        (routes, clients)
    }
    pub fn snapshot(&mut self) -> Result<Snapshot, String> {
        self.reload()?;
        let mut errors = Vec::new();
        // One local date keeps the daily and weekly usage totals aligned across midnight.
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
        let mut health = self.health_probe.details(&self.config.server_url);
        health["checked_at"] = json!(checked_at);
        health["url"] = json!(self.config.server_url);
        let server_status = if health["state"] == "running" {
            crate::health::ServerStatus::Running
        } else {
            crate::health::ServerStatus::Stopped
        };
        let (routes, clients) = self.server_config(&mut errors);
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
        let Config {
            server_url,
            routing_log,
            config_file,
            launchd_label,
            refresh_seconds,
            baseline_model,
            prices_file,
            service_config: _, // server::save_edit checks route edits against it
        } = self.config.clone();
        let update_status = install_metadata(&self.settings)
            .and_then(|path| switchyard_desktop::install::update_status(&path))
            .unwrap_or_else(|error| json!({"state":"unavailable","message":error}));
        Ok(Snapshot {
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
        self.reload()?;
        match action {
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
                server::save_route_change(&self.config, review.original()?, edited)
                    .map(Reply::saved_edit)
            }
            Action::Models {
                client,
                refresh,
                key,
                base_url,
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
                    if base_url.as_deref() != Some(client.base_url.as_str()) {
                        return Err(format!(
                            "The URL for {} changed after this form opened, so the app did \
                             not send the key. Open the form again to check the endpoint.",
                            client.name
                        ));
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
                    login_mode(tool, &config, &selected.key)?,
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
                    login_mode(tool, &config, &route.key)?,
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
            Action::Restart {} => server::restart_checked(&self.config).map(Reply::message),
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
/// This function shows a target as `client / model`, or names it when no target entry has its
/// name, which the server rejects at startup.
fn target_label(target: &switchyard_runner::ConfiguredTarget) -> String {
    match &target.model {
        Some(model) => format!("{} / {}", model.client, model.id),
        None => format!("{} (no target has this name)", target.name),
    }
}
/// This function reads each harness and account's settings once and pairs each one with the
/// model it selects, so a snapshot does not reread every settings file for every route.
fn harness_selections() -> Vec<(String, Dependent)> {
    let mut selections = Vec::new();
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
            let Some(model) = harness::inspect(tool, &files).ok().and_then(|status| {
                Some(status.lines().next()?.strip_prefix("Model: ")?.to_owned())
            }) else {
                continue;
            };
            selections.push((
                model,
                Dependent {
                    tool,
                    label,
                    account,
                    files,
                },
            ));
        }
    }
    selections
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
fn login_mode(tool: Harness, config: &ServerConfig, route: &str) -> Result<bool, String> {
    let clients = config.clients();
    let choices = config.choices(route);
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

    // The editor supports only some route fields, so the page must not limit models to them:
    // a sub-agent model, a stage router's judge, and a custom classifier's groups all show.
    #[test]
    fn routes_list_every_model_the_runner_calls() {
        let directory = tempfile::tempdir().expect("directory");
        let settings = directory.path().join("desktop.toml");
        let server_config = directory.path().join("server.toml");
        std::fs::write(
            &settings,
            format!("server_url = 'http://127.0.0.1:9'\nconfig_file = {server_config:?}\n"),
        )
        .expect("settings");
        std::fs::write(
            &server_config,
            r#"schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[targets.strong]
id = "strong/model"
llm_client = "primary"

[targets.weak]
id = "weak/model"
llm_client = "primary"

[targets.judge]
id = "judge/model"
llm_client = "primary"

[routes.direct]
id = "direct"
type = "passthrough"
target = "strong"

[routes.direct.subagents]
type = "passthrough"
target = "weak"

[routes.stage]
id = "stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"

[routes.stage.classifier]
target = "judge"

[routes.custom]
id = "custom"
type = "llm_classifier"
mode = "custom"
models = { judge = ["judge"], capable = ["strong"], efficient = ["weak"], any = ["strong", "weak"] }
default_target = "efficient"
prompt = "Pick a target."
response_schema = '{"type":"object"}'
policy = { type = "target_selector", selector = "/target" }
"#,
        )
        .expect("server config");
        let config = Config::load(&settings).expect("config");
        let snapshot = Controller::new(config, settings)
            .snapshot()
            .expect("snapshot");
        let routes: Vec<_> = snapshot
            .routes
            .iter()
            .map(|route| {
                (
                    route.key.as_str(),
                    route.models.clone(),
                    route.judges.clone(),
                )
            })
            .collect();
        let both = || {
            vec![
                "primary / strong/model".to_string(),
                "primary / weak/model".to_string(),
            ]
        };
        let judge = || vec!["primary / judge/model".to_string()];
        assert_eq!(
            routes,
            [
                ("custom", both(), judge()),
                ("direct", both(), Vec::new()),
                ("stage", both(), judge()),
            ],
            "{:?}",
            snapshot.errors
        );
    }

    // The key form keeps the URL it opened with. If the server config moves the endpoint
    // before the user submits, the controller must not send the key to the new URL.
    #[test]
    fn model_keys_go_only_to_the_url_the_form_opened_with() {
        let directory = tempfile::tempdir().expect("directory");
        let settings = directory.path().join("desktop.toml");
        let server_config = directory.path().join("server.toml");
        std::fs::write(
            &settings,
            format!("server_url = 'http://127.0.0.1:9'\nconfig_file = {server_config:?}\n"),
        )
        .expect("settings");
        std::fs::write(
            &server_config,
            "schema_version = 1\n\n[llm_clients.primary]\nformat = \"openai_chat\"\n\
             base_url = \"https://new.example.test/v1\"\napi_key_env = \"PRIMARY_KEY\"\n",
        )
        .expect("server config");
        let config = Config::load(&settings).expect("config");
        let mut controller = Controller::new(config, settings);
        for base_url in [Some("https://old.example.test/v1".to_string()), None] {
            let error = controller
                .dispatch(Action::Models {
                    client: "primary".into(),
                    refresh: true,
                    key: Some("secret".into()),
                    base_url,
                })
                .err()
                .expect("the URL changed");
            assert!(error.starts_with("The URL for primary changed"), "{error}");
        }
    }

    #[test]
    // An algorithm switch must not give stale choices a revision for externally edited bytes.
    fn algorithm_switches_keep_the_first_revision_and_external_edits_reject_it() {
        let dir = tempfile::tempdir().expect("fixture");
        let settings = dir.path().join("desktop.toml");
        let config = dir.path().join("server.toml");
        let text = "schema_version = 1\n\
                    [llm_clients.c]\n\
                    base_url='http://localhost:1234'\n\
                    format='openai_chat'\n\
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
                base_url: None,
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
                base_url: None,
            })
            .expect("cached fallback");
        assert_eq!(failed_refresh.data["warning"], true);
        assert_eq!(failed_refresh.data["source"], "cache");
        assert_eq!(failed_refresh.data["models"], json!(["cached"]));
        let result = controller.dispatch(Action::Models {
            client: "c".into(),
            refresh: false,
            key: Some("rejected-fixture-key".into()),
            base_url: Some(url.clone()),
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
            base_url: Some("http://127.0.0.1:9/v1".into()),
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
        let text = "schema_version = 1\n\
                    [llm_clients.c]\n\
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
            let original =
                format!("schema_version = 1\ntargets = {{}}\n[routes.r]\nid='public'\n{fields}\n");
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
