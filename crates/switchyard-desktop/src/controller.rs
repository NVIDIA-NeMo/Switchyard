// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Both frontends call these operations so validation stays independent of widgets.
//! Every change that an app makes to the server or to saved settings is an `Action`. The
//! terminal app matches `ActionKind` without a wildcard arm, so adding an action breaks the
//! build until the terminal app can run it.

use crate::{config::Config, history, pricing, rollup, server};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Clone, Debug, Deserialize, strum::EnumDiscriminants)]
#[strum_discriminants(name(ActionKind), derive(strum::EnumIter))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
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

/// Snapshot holds everything that the apps show. The terminal app destructures it without
/// `..`, so adding a field breaks the build until the terminal app shows it.
#[derive(Clone, Default, Serialize)]
pub struct Snapshot {
    pub update_status: Value,
    /// This field is `None` only in the terminal app's placeholder snapshot, which it shows
    /// when the app settings cannot be loaded.
    pub settings: Option<AppSettings>,
    pub data_state: DataState,
    pub summary: Vec<String>,
    pub metrics: Metrics,
    pub today: chrono::NaiveDate,
    pub analytics: rollup::Usage,
    /// The keys are `today`, `week`, and `all`. A value is null when usage could not be read.
    pub estimates: BTreeMap<&'static str, Option<pricing::Estimate>>,
    pub baseline_model: String,
    pub activity: [u64; 24],
    pub sessions: Vec<String>,
    pub entries: Vec<history::Entry>,
    pub limited: bool,
    pub skipped: usize,
    pub errors: Vec<String>,
    pub refresh_seconds: u64,
}

/// AppSettings holds every app setting and the file that stores them. `Controller::snapshot`
/// destructures `Config` without `..`, so adding a setting to `Config` breaks the build until
/// it is added here too.
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
    fn message(message: String) -> Self {
        Self {
            message,
            data: Value::Null,
        }
    }
}

pub struct Controller {
    settings: PathBuf,
    config: Config,
    reader: rollup::Reader,
    update_review: Option<(String, Value)>,
}
impl Controller {
    pub fn new(config: Config, settings: PathBuf) -> Self {
        Self {
            config,
            settings,
            reader: rollup::Reader::default(),
            update_review: None,
        }
    }
    fn reload(&mut self) -> Result<(), String> {
        self.config = Config::load(&self.settings)?;
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
    pub fn snapshot(&mut self) -> Result<Snapshot, String> {
        self.reload()?;
        let mut errors = Vec::new();
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
        let mut health = crate::health::details(&self.config.server_url);
        health["checked_at"] = json!(checked_at);
        health["url"] = json!(self.config.server_url);
        let server_status = if health["state"] == "running" {
            crate::health::ServerStatus::Running
        } else {
            crate::health::ServerStatus::Stopped
        };
        let summary = crate::summary::build(server_status, &usage, &self.config, &rates);
        let mut history_state = Availability::Available;
        let history = history::load(&self.config.routing_log).unwrap_or_else(|e| {
            history_state = Availability::Unavailable;
            errors.push(e);
            history::History::default()
        });
        let activity = history::hourly_calls(&history.entries, chrono::Utc::now());
        let update_status = install_metadata(&self.settings)
            .and_then(|path| switchyard_desktop::install::update_status(&path))
            .unwrap_or_else(|error| json!({"state":"unavailable","message":error}));
        let sessions = history.sessions();
        let Config {
            server_url,
            routing_log,
            config_file,
            launchd_label,
            refresh_seconds,
            baseline_model,
            prices_file,
        } = self.config.clone();
        Ok(Snapshot {
            update_status,
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
            summary,
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
            baseline_model: baseline_model.clone(),
            activity,
            sessions,
            entries: history.entries,
            limited: history.limited,
            skipped: history.skipped,
            errors,
            refresh_seconds: refresh_seconds.max(1),
            settings: Some(AppSettings {
                path: self.settings.clone(),
                server_url,
                routing_log,
                config_file,
                launchd_label,
                refresh_seconds,
                baseline_model,
                prices_file,
            }),
        })
    }
    /// The TUI uses these file paths to run an editor in the foreground.
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
                let (token, reviewed) = self
                    .update_review
                    .as_ref()
                    .ok_or("Review the source update first.")?;
                let mut current = switchyard_desktop::install::preview_update(&install_metadata(
                    &self.settings,
                )?)?;
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
