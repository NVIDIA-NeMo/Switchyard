// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Both frontends call these operations so validation stays independent of widgets.

use crate::{
    accounts,
    config::Config,
    harness::{self, Harness},
    history, models, rollup, server,
    server_config::{ALGORITHMS, Choice, Route, ServerConfig},
    sessions,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize)]
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
    OpenLogs {},
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
    OpenUpdateLog {},
    OpenConfig {},
    OpenSettings {},
    RetrySettings {},
    CheckConfig {},
    SavePreferences {
        baseline_model: String,
        refresh_seconds: u64,
        prices: crate::pricing::PriceTable,
    },
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

pub struct Controller {
    settings: PathBuf,
    config: Config,
    reader: rollup::Reader,
    // The generation ties a draft to the server config file selected by the settings.
    generation: u64,
    route_reviews: std::collections::HashMap<String, Review>,
    tool_review: Option<Review>,
    session_cleanup: Option<(String, String, Value)>,
    account_archive: Option<(Harness, String, String)>,
    route_change: Option<Review>,
    update_review: Option<(String, Value)>,
    launch_review: Option<ReviewedLaunch>,
}
impl Controller {
    pub fn new(config: Config, settings: PathBuf) -> Self {
        Self {
            config,
            settings,
            reader: rollup::Reader::default(),
            generation: 0,
            route_reviews: Default::default(),
            tool_review: None,
            session_cleanup: None,
            account_archive: None,
            route_change: None,
            update_review: None,
            launch_review: None,
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
    pub fn snapshot(&mut self) -> Result<Value, String> {
        self.reload()?;
        let mut errors = Vec::new();
        // One local date keeps aggregate totals and frontend filters aligned across midnight.
        let today = chrono::Local::now().date_naive();
        let mut usage_available = true;
        let usage = self
            .reader
            .read(&self.config.routing_log, today)
            .unwrap_or_else(|error| {
                usage_available = false;
                errors.push(format!("Could not read usage: {error}"));
                rollup::Usage::default()
            });
        let estimates: std::collections::BTreeMap<_, _> = [
            ("today", &usage.today), ("week", &usage.week), ("all", &usage.all),
        ].into_iter().map(|(period, totals)| {
            let missing = crate::pricing::unpriced(
                totals.routed.keys().chain(totals.classifier.keys())
                    .map(String::as_str).chain(std::iter::once(self.config.baseline_model.as_str())),
                &self.config.prices,
            );
            (period, json!({"cost":if usage_available {crate::pricing::estimate(totals, &self.config.prices, &self.config.baseline_model)} else {None},"missing":missing}))
        }).collect();
        let checked_at = chrono::Utc::now().to_rfc3339();
        let mut health = crate::health::details(&self.config.server_url);
        health["checked_at"] = json!(checked_at);
        health["url"] = json!(self.config.server_url);
        let server_status = if health["state"] == "running" {
            crate::health::ServerStatus::Running
        } else {
            crate::health::ServerStatus::Stopped
        };
        let summary = crate::summary::build(server_status, &usage, &self.config);
        let metrics = json!({"server_url":self.config.server_url,"running":server_status == crate::health::ServerStatus::Running,
            "today":{"requests":usage.today.requests(),"tokens":usage.today.tokens()},
            "week":{"requests":usage.week.requests(),"tokens":usage.week.tokens()}});
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
                    routes.push(json!({"key": route.key, "id": route.id, "kind": route.kind,"revision":revision,
                        "label": format!("{} · {} · {}", route.id, route.kind, models), "choices": choices,
                        "tiers": config.algorithm(&route.key).map(|a| config.roles(&route.key, a).iter().map(|r| r.tier.map(|t| format!("{t:?}"))).collect::<Vec<_>>()),
                        "compatibility": harness::HARNESSES.iter().map(|(tool,_)| (serde_json::to_value(tool).unwrap().as_str().unwrap().to_string(),login_mode(*tool,&config,&route).err())).collect::<std::collections::BTreeMap<_,_>>(),
                        "dependents":known_dependents(&route.id),
                        "editable": config.algorithm(&route.key).is_some(), "generated": config.generated_by("routes", &route.key)}));
                }
                clients = config.clients().iter().map(|c| json!({"name":c.name,"host":c.host(),"models":config.models_on(&c.name),"unlisted":models::unlisted(c),"accepts_key":cfg!(any(target_os = "macos", windows)) && !models::unlisted(c) && (c.forward_auth || c.api_key_env.is_some()),"note":models::missing_env(c).map(models::missing_env_note)})).collect();
            }
            Err(e) => errors.push(e),
        }
        let mut history_available = true;
        let history = history::load(&self.config.routing_log).unwrap_or_else(|e| {
            history_available = false;
            errors.push(e);
            history::History::default()
        });
        let activity = history::hourly_calls(&history.entries, chrono::Utc::now());
        let tools: Vec<_> = harness::HARNESSES.iter().map(|(tool,label)| {
            let discovered = accounts::discover(*tool);
            let error = discovered.as_ref().err().cloned();
            let discovered = discovered.unwrap_or_default();
            let names: Vec<_> = discovered.iter().map(|(name,_)|name).collect();
            let details: Vec<_> = discovered.iter().map(|(name,path)|json!({"name":name,"path":path,"login_status":"unknown","status":harness::inspect(*tool,&accounts::config_paths(*tool,Some(path))).unwrap_or_else(|error|error)})).collect();
            json!({"tool":tool,"label":label,"available":harness::binary(*tool).is_some(),"files":harness::paths(*tool),"status":harness::inspect(*tool,&harness::paths(*tool)).unwrap_or_else(|error|error),"accounts":names,"account_details":details,"discovery_error":error})
        }).collect();
        let update_status = install_metadata(&self.settings)
            .and_then(|path| switchyard_desktop_install::update_status(&path))
            .unwrap_or_else(|error| json!({"state":"unavailable","message":error}));
        let session_inventory = sessions::inventory();
        let inventory_error = session_inventory.as_ref().err().cloned();
        let session_inventory = session_inventory.unwrap_or_default();
        Ok(
            json!({"update_status":update_status,"session_inventory":session_inventory,"session_inventory_error":inventory_error,"settings":{"path":self.settings,"config_file":self.config.config_file,"routing_log":self.config.routing_log,"baseline_model":self.config.baseline_model,"refresh_seconds":self.config.refresh_seconds,"prices":self.config.prices},
            "data_state":{"usage":{"state":if usage_available {"available"} else {"unavailable"},"checked_at":checked_at,"path":self.config.routing_log},"history":{"state":if history_available {"available"} else {"unavailable"},"checked_at":checked_at,"path":self.config.routing_log},"health":health},
            "generation":self.generation,"summary":summary,"metrics":metrics,"routes":routes,"clients":clients,"tools":tools,
            "algorithms":ALGORITHMS.iter().map(|a|json!({"kind":a.kind,"title":a.title,"summary":a.summary})).collect::<Vec<_>>(),
            "today":today,"analytics":usage,"estimates":estimates,"baseline_model":self.config.baseline_model,
            "activity":activity,"sessions":history.sessions(),"entries":history.entries,"limited":history.limited,"skipped":history.skipped,
            "errors":errors,"refresh_seconds":self.config.refresh_seconds.max(1)}),
        )
    }
    pub fn dispatch(&mut self, action: Action) -> Result<Reply, String> {
        if matches!(action, Action::OpenSettings {}) {
            server::command("open", &[&self.settings.display().to_string()])?;
            return Ok(Reply::message("Opened app settings.".into()));
        }
        self.reload()?;
        match action {
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
                let data = json!({"revision":revision,"roles": roles.iter().map(|r|json!({"label":r.label,"hint":r.hint,"tier":r.tier.map(|t|format!("{t:?}"))})).collect::<Vec<_>>(),"choices":config.choices(&route)});
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
                        return Err("This endpoint does not accept a model-list key. Choose an endpoint configured for model-list authentication.".into());
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
                    .map(|e| format!("{} {}", e.reason(), e.advice()))
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
                    return Err("Install the selected coding tool first.".into());
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
                        "Pi uses isolated session settings. Personal settings and extensions are not copied."
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
                    data: json!({"name":name,"path":path,"preview_token":token,"notice":"Archive moves this account out of the picker and preserves all login files. Existing sessions using its old directory may need a new login."}),
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
            Action::OpenLogs {} => {
                let path = crate::config::expand_home(std::path::Path::new("~/.switchyard/logs"));
                server::command("open", &[&path.display().to_string()])?;
                Ok(Reply::message("Opened Switchyard logs.".into()))
            }
            Action::Restart {} => server::restart_checked(&self.config).map(Reply::message),
            Action::CheckConfig {} => server::check_config(&self.config)
                .map(|()| Reply::message("Server config check passed.".into())),
            Action::SavePreferences {
                baseline_model,
                refresh_seconds,
                prices,
            } => {
                crate::config::save_preferences(
                    &self.settings,
                    &baseline_model,
                    refresh_seconds,
                    &prices,
                )?;
                self.reload()?;
                Ok(Reply::message(format!(
                    "Changed {}.",
                    self.settings.display()
                )))
            }
            Action::RetrySettings {} => Ok(Reply::message("App settings loaded.".into())),
            Action::OpenSettings {} => unreachable!(),
            Action::PreviewUpdate {} => {
                let mut data =
                    switchyard_desktop_install::preview_update(&install_metadata(&self.settings)?)?;
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
                let (token, reviewed) = self
                    .update_review
                    .as_ref()
                    .ok_or("Review the source update first.")?;
                let mut current =
                    switchyard_desktop_install::preview_update(&install_metadata(&self.settings)?)?;
                current
                    .as_object_mut()
                    .ok_or("Invalid update preview.")?
                    .remove("status");
                if preview_token.as_deref() != Some(token.as_str()) || &current != reviewed {
                    return Err(
                        "The source checkout or installation changed. Review the update again."
                            .into(),
                    );
                }
                update(&self.settings).map(Reply::message)
            }
            Action::OpenUpdateLog {} => {
                let metadata =
                    switchyard_desktop_install::Settings::load(&install_metadata(&self.settings)?)?;
                let log = metadata.sy_home.join("logs/update.log");
                server::command("open", &[&log.display().to_string()])?;
                Ok(Reply::message("Opened the update log.".into()))
            }
            Action::OpenConfig {} => {
                server::command("open", &[&self.config.config_file.display().to_string()])?;
                Ok(Reply::message(
                    "Opened the server config in your default editor.".into(),
                ))
            }
        }
    }
}
fn known_dependents(id: &str) -> Vec<Value> {
    let mut dependents = Vec::new();
    for (tool, label) in harness::HARNESSES {
        let mut destinations = vec![(None, harness::paths(*tool))];
        destinations.extend(
            accounts::list(*tool)
                .into_iter()
                .map(|(name, path)| (Some(name), accounts::config_paths(*tool, Some(&path)))),
        );
        for (account, files) in destinations {
            if harness::inspect(*tool, &files)
                .is_ok_and(|status| status.lines().next() == Some(format!("Model: {id}").as_str()))
            {
                dependents.push(json!({"tool":tool,"label":label,"account":account,"files":files}));
            }
        }
    }
    dependents
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
    Ok(accounts::config_paths(tool, account.as_deref()))
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
                    "This route forwards the caller's login to {host}. Pick a matching coding tool or a route with server API credentials."
                ));
            }
            login = true;
        }
    }
    Ok(login)
}

fn install_metadata(settings: &std::path::Path) -> Result<PathBuf, String> {
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let bundle = switchyard_desktop_install::bundle_metadata(&executable);
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
    let log = switchyard_desktop_install::start_update(&metadata)?;
    Ok(format!(
        "Started rebuilding and reinstalling from the saved source checkout. Progress and errors are recorded in {}. The app restarts after a successful install.",
        log.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    // The subprocess isolates coding-tool discovery and every launch destination.
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
            let text = "[llm_clients.c]\nbase_url='http://localhost:1234'\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n";
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

    #[test]
    // An algorithm switch must not give stale choices a revision for externally edited bytes.
    fn algorithm_switches_keep_the_first_revision_and_external_edits_reject_it() {
        let dir = tempfile::tempdir().expect("fixture");
        let settings = dir.path().join("desktop.toml");
        let config = dir.path().join("server.toml");
        let text = "[llm_clients.c]\nbase_url='http://localhost:1234'\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n";
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
            controller.snapshot().expect("snapshot")["routes"][0]["revision"],
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
    // The review includes backup files because changed backups also change recovery behavior.
    fn changed_review_files_and_inputs_reject_without_side_effects() {
        let dir = tempfile::tempdir().expect("fixture");
        let settings = dir.path().join("tool.json");
        let backup = dir.path().join("backup");
        std::fs::write(&settings, "private-original").expect("settings");
        let inputs = json!({"tool":"claude","route":"first"});
        let review = Review::capture(inputs.clone(), vec![settings.clone(), backup.clone()])
            .expect("review");
        assert!(review.check(Some(&review.token), &inputs).is_ok());
        for token in [None, Some("unknown")] {
            assert!(review.check(token, &inputs).is_err());
        }
        assert!(
            review
                .check(
                    Some(&review.token),
                    &json!({"tool":"claude","route":"second"})
                )
                .is_err()
        );
        std::fs::write(&settings, "external-edit").expect("external edit");
        assert!(review.check(Some(&review.token), &inputs).is_err());
        std::fs::write(&settings, "private-original").expect("restore fixture");
        std::fs::write(&backup, "new-backup").expect("new backup");
        assert!(review.check(Some(&review.token), &inputs).is_err());
        assert_eq!(
            std::fs::read_to_string(&settings).expect("settings"),
            "private-original"
        );
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 2);
    }

    #[test]
    // The Apply fixture supplies a generation so only its extra choice field causes rejection.
    fn malformed_actions_reject_before_dispatch() {
        for input in [
            r#"{"kind":"restart","command":"rm"}"#,
            r#"{"kind":"install","tool":"unknown","route":"r","id":"m","account":null}"#,
            r#"{"kind":"apply","generation":0,"route":"r","algorithm":"passthrough","choices":[{"client":"c","model":"m","extra":1}]}"#,
            r#"{"kind":"unknown"}"#,
        ] {
            assert!(serde_json::from_str::<Action>(input).is_err());
        }
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
        std::fs::write(&config,"[llm_clients.c]\nbase_url='https://example.com/v1'\nformat='openai_chat'\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n").expect("config");
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
            switchyard_desktop_install::windows::saved_key(&url).expect("no saved key"),
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
                    stream.write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").expect("response");
                    return String::from_utf8_lossy(&request).into_owned();
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            String::new()
        });
        let settings = dir.path().join("settings.toml");
        let routes = dir.path().join("routes.toml");
        std::fs::write(&routes, format!("[llm_clients.c]\nbase_url={url:?}\nformat='openai_responses'\nforward_auth=true\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n")).expect("routes");
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
            switchyard_desktop_install::windows::saved_key(&url).expect("rejected key not saved"),
            None
        );
        std::fs::write(&routes, "[llm_clients.c]\nbase_url='http://127.0.0.1:9/v1'\nformat='openai_responses'\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n").expect("routes");
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
    #[cfg(not(any(target_os = "macos", windows)))]
    // Rejecting a typed key must leave the upstream endpoint, model cache, and route config unchanged.
    fn key_persistence_rejects_without_network_or_file_writes() {
        let dir = tempfile::tempdir().expect("directory");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener");
        listener.set_nonblocking(true).expect("nonblocking");
        let url = format!("http://{}/v1", listener.local_addr().expect("address"));
        let settings = dir.path().join("settings.toml");
        let routes = dir.path().join("routes.toml");
        let text = format!(
            "[llm_clients.c]\nbase_url={url:?}\nformat='openai_responses'\nforward_auth=true\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n"
        );
        std::fs::write(&routes, &text).expect("routes");
        std::fs::write(
            &settings,
            format!(
                "config_file={routes:?}\nrouting_log={:?}",
                dir.path().join("log")
            ),
        )
        .expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        let result = controller.dispatch(Action::Models {
            client: "c".into(),
            refresh: true,
            key: Some("fixture-key".into()),
        });
        assert_eq!(
            result.err(),
            Some("Saving model-list keys requires macOS Keychain or Windows Credential Manager. On this platform, set the environment variable named by api_key_env and make it available to the app.".into())
        );
        assert_eq!(
            listener.accept().expect_err("no upstream request").kind(),
            std::io::ErrorKind::WouldBlock
        );
        assert_eq!(std::fs::read_to_string(routes).expect("routes"), text);
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 2);
    }
    #[test]
    fn usage_snapshot_survives_missing_and_malformed_route_configs() {
        let dir = tempfile::tempdir().expect("directory");
        let settings = dir.path().join("settings.toml");
        let routes = dir.path().join("routes.toml");
        let log = dir.path().join("log");
        std::fs::write(
            &log,
            "{\"ts\":\"2026-10-08T12:00:00Z\",\"model\":\"actual\",\"session_id\":\"session\"}\n",
        )
        .expect("log");
        std::fs::write(
            &settings,
            format!("config_file={routes:?}\nrouting_log={log:?}"),
        )
        .expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        for malformed in [false, true] {
            if malformed {
                std::fs::write(&routes, "secret = 'UNTERMINATED_PRIVATE_VALUE").expect("routes");
            }
            let snapshot = controller.snapshot().expect("snapshot");
            assert_eq!(snapshot["sessions"], json!(["session"]));
            assert_eq!(snapshot["entries"][0]["model"], "actual");
            assert_eq!(
                snapshot["analytics"]["all"]["routed"]["actual"]["requests"],
                1
            );
            assert!(snapshot["estimates"]["all"]["cost"].is_null());
            assert!(!snapshot["errors"].as_array().expect("errors").is_empty());
            assert!(!snapshot.to_string().contains("PRIVATE_VALUE"));
        }
    }
    #[test]
    // Matching route names in two files must not let an old draft edit the newly selected file.
    fn stale_editor_rejects_after_settings_change_without_writes() {
        let dir = tempfile::tempdir().expect("directory");
        let settings = dir.path().join("settings.toml");
        let first = dir.path().join("first.toml");
        let second = dir.path().join("second.toml");
        let text = "[llm_clients.c]\nbase_url='https://example.com/v1'\nformat='openai_responses'\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n";
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
        assert_eq!(snapshot["routes"][0]["tiers"], json!(["Capable"]));
        let generation = snapshot["generation"].as_u64().expect("generation");
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
            controller.snapshot().expect("snapshot")["generation"],
            generation + 1
        );
    }
    #[test]
    fn invalid_route_choices_reject_before_saving_or_restarting() {
        let dir = tempfile::tempdir().expect("directory");
        let settings = dir.path().join("settings.toml");
        let config = dir.path().join("server.toml");
        let original = "[llm_clients.c]\nbase_url='http://localhost:1234'\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n";
        std::fs::write(&config, original).expect("config");
        std::fs::write(&settings, format!("config_file={config:?}")).expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        let editor = controller
            .dispatch(Action::Editor {
                generation: 0,
                route: "r".into(),
                algorithm: "passthrough".into(),
            })
            .expect("editor");
        let revision = editor.data["revision"].as_str().map(str::to_string);
        for (choices, expected) in [
            (vec![], "passthrough needs 1 models, not 0."),
            (
                vec![Choice {
                    client: "c".into(),
                    model: " ".into(),
                }],
                "Pick a model for Model.",
            ),
            (
                vec![Choice {
                    client: "unknown".into(),
                    model: "model".into(),
                }],
                "The config has no LLM client named \"unknown\".",
            ),
        ] {
            let error = controller
                .dispatch(Action::Apply {
                    generation: 0,
                    route: "r".into(),
                    algorithm: "passthrough".into(),
                    choices,
                    revision: revision.clone(),
                })
                .err()
                .expect("rejection");
            assert_eq!(error, expected);
            assert_eq!(std::fs::read_to_string(&config).expect("config"), original);
            assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 2);
        }
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
            "type='llm_classifier'\nmode='custom'\nmodels={judge=['judge'],any=['answer']}\ndefault_target='any'\nprompt='Select a target.'\nresponse_schema='{\"type\":\"object\"}'\npolicy={type='target_selector',selector='/target'}",
        ] {
            let _: switchyard_runner::AlgorithmSpec =
                toml::from_str(fields).expect("server route schema");
            let original = format!("[routes.r]\nid='public'\n{fields}\n");
            std::fs::write(&config, &original).expect("config");
            assert_eq!(
                controller.snapshot().expect("snapshot")["routes"][0]["editable"],
                false
            );
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

    #[test]
    fn rejected_install_has_no_writes() {
        let dir = tempfile::tempdir().expect("directory");
        let settings = dir.path().join("settings.toml");
        let config = dir.path().join("server.toml");
        let text = "[llm_clients.c]\nbase_url='http://chatgpt.com/backend-api/codex'\nforward_auth=true\nformat='openai_responses'\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n";
        std::fs::write(&config, text).expect("config");
        std::fs::write(
            &settings,
            format!(
                "config_file={:?}\nrouting_log={:?}",
                config,
                dir.path().join("log")
            ),
        )
        .expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        for id in ["old", "public"] {
            assert!(
                controller
                    .dispatch(Action::Install {
                        tool: Harness::CodexCli,
                        account: None,
                        settings_file: None,
                        route: "r".into(),
                        id: id.into(),
                        preview_token: None,
                    })
                    .is_err()
            );
            assert_eq!(std::fs::read_to_string(&config).expect("config"), text);
            assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 2);
        }
        let snapshot = controller.snapshot().expect("snapshot");
        assert!(
            snapshot["routes"][0]["label"]
                .as_str()
                .expect("label")
                .contains("passthrough · c / actual")
        );
        assert!(!snapshot.to_string().contains("backend-api"));
    }
}
