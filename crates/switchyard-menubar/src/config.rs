// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module reads app settings from `~/.switchyard/menubar.toml`.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::pricing::PriceTable;

/// Config stores the settings the apps use to find the server and estimate costs.
/// Missing keys fall back to [`Config::default`].
#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// This field sets the server base URL used for health checks.
    pub server_url: String,
    /// This field names the JSONL file where the server appends routing records.
    pub routing_log: PathBuf,
    /// This field names the server TOML file opened by "Open server config".
    pub config_file: PathBuf,
    /// This field names the LaunchAgent used to restart the server.
    pub launchd_label: String,
    /// This field sets the usage refresh interval in seconds.
    pub refresh_seconds: u64,
    /// This field names the model used to estimate costs without routing.
    pub baseline_model: String,
    /// This field stores per-model rates; missing rates hide dollar figures.
    pub prices: PriceTable,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server_url: "http://127.0.0.1:4123".to_string(),
            routing_log: PathBuf::from("~/.switchyard/routing.jsonl"),
            config_file: PathBuf::from("~/.switchyard/composite.toml"),
            launchd_label: "com.nvidia.switchyard.server".to_string(),
            refresh_seconds: 30,
            baseline_model: "gpt-5.6-sol".to_string(),
            prices: PriceTable::new(),
        }
    }
}

impl Config {
    /// This function parses settings and uses defaults when the file is absent.
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut config: Self = match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text)
                .map_err(|error| format!("parse {}: {error}", path.display()))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(error) => return Err(format!("read {}: {error}", path.display())),
        };
        config.routing_log = expand_home(&config.routing_log);
        config.config_file = expand_home(&config.config_file);
        Ok(config)
    }

    /// This function returns the default settings path, `~/.switchyard/menubar.toml`.
    pub fn default_path() -> PathBuf {
        expand_home(Path::new("~/.switchyard/menubar.toml"))
    }
}

pub fn validate_toml_file(path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    toml::from_str::<toml::Value>(&text)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    Ok(())
}

/// This function rewrites a leading `~`. Paths are written by hand in the settings file,
/// where `~` is the natural way to spell a home path.
pub fn expand_home(path: &Path) -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    match path.strip_prefix("~") {
        Ok(rest) => home.join(rest),
        Err(_) => path.to_path_buf(),
    }
}
