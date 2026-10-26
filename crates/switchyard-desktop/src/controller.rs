// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The terminal controller reloads settings and combines usage and server health into snapshots.
//! It retains the log reader and health probe across refreshes.

use crate::{config::Config, rollup};
use serde::Serialize;
use serde_json::{Value, json};
use std::path::PathBuf;

/// Snapshot holds the terminal app's usage, server health, errors, and refresh interval.
#[derive(Clone, Default, Serialize)]
pub struct Snapshot {
    pub data_state: DataState,
    pub metrics: Metrics,
    pub errors: Vec<String>,
    pub refresh_seconds: u64,
}

#[derive(Clone, Default, Serialize)]
pub struct DataState {
    pub usage: Source,
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
        let checked_at = chrono::Utc::now().to_rfc3339();
        let mut health = self.health_probe.details(&self.config.server_url);
        health["checked_at"] = json!(checked_at);
        health["url"] = json!(self.config.server_url);
        let server_status = if health["state"] == "running" {
            crate::health::ServerStatus::Running
        } else {
            crate::health::ServerStatus::Stopped
        };
        let server_url = self.config.server_url.clone();
        let routing_log = self.config.routing_log.clone();
        let refresh_seconds = self.config.refresh_seconds;
        Ok(Snapshot {
            data_state: DataState {
                usage: Source {
                    state: usage_state,
                    checked_at: checked_at.clone(),
                    path: routing_log.clone(),
                },
                health,
            },
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
            errors,
            refresh_seconds: refresh_seconds.max(1),
        })
    }
}
