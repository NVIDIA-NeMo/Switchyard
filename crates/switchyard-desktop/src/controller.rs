// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Both frontends call these operations so validation stays independent of widgets.

use crate::{config::Config, history, rollup, server};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    OpenLogs {},
    Restart {},
    PreviewUpdate {},
    Update {
        preview_token: Option<String>,
    },
    OpenUpdateLog {},
    OpenConfig {},
    OpenPrices {},
    OpenSettings {},
    RetrySettings {},
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
        // Invalid overrides leave baseline rates unknown and add an error to the snapshot.
        let rates =
            crate::pricing::load(self.config.prices_file.as_deref()).unwrap_or_else(|error| {
                errors.push(format!("Could not read model prices: {error}"));
                Default::default()
            });
        let estimates: std::collections::BTreeMap<_, _> = [
            ("today", &usage.today),
            ("week", &usage.week),
            ("all", &usage.all),
        ]
        .into_iter()
        .map(|(period, totals)| {
            let estimate =
                crate::pricing::Estimate::new(totals, &rates, &self.config.baseline_model);
            (
                period,
                if usage_available {
                    json!(estimate)
                } else {
                    Value::Null
                },
            )
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
        let metrics = json!({"server_url":self.config.server_url,"running":server_status == crate::health::ServerStatus::Running,
            "today":{"requests":usage.today.requests(),"tokens":usage.today.tokens()},
            "week":{"requests":usage.week.requests(),"tokens":usage.week.tokens()}});
        let mut history_available = true;
        let history = history::load(&self.config.routing_log).unwrap_or_else(|e| {
            history_available = false;
            errors.push(e);
            history::History::default()
        });
        let activity = history::hourly_calls(&history.entries, chrono::Utc::now());
        let update_status = install_metadata(&self.settings)
            .and_then(|path| switchyard_desktop::install::update_status(&path))
            .unwrap_or_else(|error| json!({"state":"unavailable","message":error}));
        Ok(
            json!({"update_status":update_status,"settings":{"path":self.settings,"config_file":self.config.config_file,"routing_log":self.config.routing_log,"baseline_model":self.config.baseline_model,"refresh_seconds":self.config.refresh_seconds,"prices_file":self.config.prices_file},
            "data_state":{"usage":{"state":if usage_available {"available"} else {"unavailable"},"checked_at":checked_at,"path":self.config.routing_log},"history":{"state":if history_available {"available"} else {"unavailable"},"checked_at":checked_at,"path":self.config.routing_log},"health":health},
            "summary":summary,"metrics":metrics,
            "today":today,"analytics":usage,"estimates":estimates,"baseline_model":self.config.baseline_model,
            "activity":activity,"sessions":history.sessions(),"entries":history.entries,"limited":history.limited,"skipped":history.skipped,
            "errors":errors,"refresh_seconds":self.config.refresh_seconds.max(1)}),
        )
    }
    /// The TUI uses these file paths to run an editor in the foreground.
    pub fn editor_target(&mut self, action: &Action) -> Result<Option<PathBuf>, String> {
        match action {
            Action::OpenSettings {} => Ok(Some(self.settings.clone())),
            Action::OpenConfig {} => {
                self.reload()?;
                Ok(Some(self.config.config_file.clone()))
            }
            _ => Ok(None),
        }
    }
    pub fn dispatch(&mut self, action: Action) -> Result<Reply, String> {
        if matches!(action, Action::OpenSettings {}) {
            server::command("open", &[&self.settings.display().to_string()])?;
            return Ok(Reply::message("Opened app settings.".into()));
        }
        self.reload()?;
        match action {
            Action::OpenLogs {} => {
                let path = crate::config::expand_home(std::path::Path::new("~/.switchyard/logs"))?;
                server::command("open", &[&path.display().to_string()])?;
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
            Action::RetrySettings {} => Ok(Reply::message("App settings loaded.".into())),
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
                let metadata = switchyard_desktop::install::Settings::load(&install_metadata(
                    &self.settings,
                )?)?;
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
    // The information action must leave settings unchanged and create no price file.
    #[test]
    fn opening_prices_without_an_override_explains_bundled_rates_without_writes() {
        let dir = tempfile::tempdir().expect("fixture");
        let settings = dir.path().join("desktop.toml");
        std::fs::write(&settings, "server_url='http://127.0.0.1:1'\n").expect("settings");
        let before = std::fs::read(&settings).expect("settings");
        let mut controller =
            Controller::new(Config::load(&settings).expect("load"), settings.clone());
        let reply = controller
            .dispatch(Action::OpenPrices {})
            .expect("bundled prices");
        assert!(reply.message.contains("Bundled model prices"));
        assert!(reply.message.contains("--pricing-file"));
        assert_eq!(
            std::fs::read(&settings).expect("unchanged settings"),
            before
        );
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 1);
    }

    #[test]
    fn malformed_actions_reject_before_dispatch() {
        for input in [
            r#"{"kind":"restart","command":"rm"}"#,
            r#"{"kind":"unknown"}"#,
        ] {
            assert!(serde_json::from_str::<Action>(input).is_err());
        }
    }
}
