// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Both frontends call these operations so validation stays independent of widgets.
//! Every change that an app makes to the server or to saved settings is an `Action`. The
//! terminal app matches `ActionKind` without a wildcard arm, so adding an action breaks the
//! build until the terminal app can run it.

use crate::{config::Config, history, pricing, rollup};
use serde::Serialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

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

pub struct Controller {
    settings: PathBuf,
    config: Config,
    reader: rollup::Reader,
    health_probe: crate::health::Probe,
}
impl Controller {
    pub fn new(config: Config, settings: PathBuf) -> Self {
        Self {
            config,
            settings,
            reader: rollup::Reader::default(),
            health_probe: crate::health::Probe::default(),
        }
    }
    fn reload(&mut self) -> Result<(), String> {
        self.config = Config::load(&self.settings)?;
        Ok(())
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
        let mut health = self.health_probe.details(&self.config.server_url);
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
