// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module reads desktop settings and supplies installation defaults.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::install::Settings;

pub(crate) const DEFAULT_PORT: u16 = 4123;
pub(crate) const HOME_DIR: &str = ".switchyard";
pub(crate) const SERVER_LABEL: &str = "com.nvidia.switchyard.server";
pub(crate) const ROUTING_LOG: &str = "routing.jsonl";
pub(crate) const SERVER_CONFIG: &str = "composite.toml";
pub(crate) const PRICES_FILE: &str = "prices.json";
pub(crate) const DESKTOP_CONFIG: &str = "desktop.toml";

pub(crate) fn server_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

pub(crate) fn home_dir() -> Result<PathBuf, String> {
    std::env::home_dir()
        .filter(|home| home.is_absolute())
        .ok_or_else(|| "Could not find your home directory.".into())
}

/// Config stores the settings the apps use to find the server and compare costs.
/// Missing keys use the selected installation directory and port.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// This field sets the HTTP base URL used for health checks; an omitted port means 80.
    pub server_url: String,
    /// This field names the JSONL file where the server appends routing records.
    pub routing_log: PathBuf,
    /// This field names the server TOML file opened by "Open server config".
    pub config_file: PathBuf,
    /// This field names the LaunchAgent used to restart the server.
    pub launchd_label: String,
    /// This field sets the usage refresh interval in seconds.
    pub refresh_seconds: u64,
    /// This field names the model whose rates price the answers as if routing had not run.
    pub baseline_model: String,
    /// This optional file replaces bundled rates for the baseline estimate.
    #[serde(rename = "pricing_file", skip_serializing_if = "Option::is_none")]
    pub prices_file: Option<PathBuf>,
}

impl Config {
    pub(crate) fn for_directory(directory: &Path, port: u16) -> Self {
        Self {
            server_url: server_url(port),
            routing_log: directory.join(ROUTING_LOG),
            config_file: directory.join(SERVER_CONFIG),
            launchd_label: SERVER_LABEL.into(),
            refresh_seconds: 30,
            baseline_model: "gpt-6-sol".into(),
            prices_file: None,
        }
    }

    /// This function parses settings and uses adjacent `install.toml` for directory and port defaults.
    /// Without installation metadata, default paths use the settings directory and the port is 4123.
    /// An omitted pricing_file selects bundled rates.
    pub fn load(path: &Path) -> Result<Self, String> {
        let directory = path.parent().unwrap_or(Path::new("."));
        let metadata = directory.join("install.toml");
        let defaults = if metadata.try_exists().map_err(|e| e.to_string())? {
            let settings = Settings::load(&metadata)?;
            Self::for_directory(&settings.sy_home, settings.port)
        } else {
            Self::for_directory(directory, DEFAULT_PORT)
        };
        let mut values = toml::Table::try_from(defaults).map_err(|e| e.to_string())?;
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let overrides: toml::Table = toml::from_str(&text)
                    .map_err(|error| format!("parse {}: {error}", path.display()))?;
                values.extend(overrides);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("read {}: {error}", path.display())),
        }
        let moved = if values.contains_key("prices") {
            " Remove the prices table to use bundled rates, or convert it to prices.toml and select it with pricing_file."
        } else {
            ""
        };
        let mut config: Self = values
            .try_into()
            .map_err(|error| format!("parse {}: {error}{moved}", path.display()))?;
        crate::health::validate_url(&config.server_url)?;
        config.routing_log = expand_home(&config.routing_log)?;
        config.config_file = expand_home(&config.config_file)?;
        config.prices_file = config.prices_file.as_deref().map(expand_home).transpose()?;
        Ok(config)
    }

    /// The app reads bundled installation metadata so shortcuts retain a custom `SY_HOME`.
    /// Without that metadata, `SY_HOME` selects the directory; if unset, it is `~/.switchyard`.
    pub fn default_path() -> Result<PathBuf, String> {
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let metadata = crate::install::bundle_metadata(&executable);
        if metadata.try_exists().map_err(|e| e.to_string())? {
            return Settings::load(&metadata).map(|s| s.desktop_settings());
        }
        let directory = match std::env::var_os("SY_HOME") {
            Some(value) => expand_home(Path::new(&value))?,
            None => home_dir()?.join(HOME_DIR),
        };
        if !directory.is_absolute() {
            return Err("SY_HOME must be an absolute path.".into());
        }
        Ok(directory.join(DESKTOP_CONFIG))
    }
}

pub fn validate_toml_file(path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    toml::from_str::<toml::Value>(&text)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    Ok(())
}

/// This function expands a leading `~` in manually written settings and reports a missing home.
pub(crate) fn expand_home(path: &Path) -> Result<PathBuf, String> {
    match path.strip_prefix("~") {
        Ok(rest) => Ok(home_dir()?.join(rest)),
        Err(_) => Ok(path.to_path_buf()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_settings_use_the_selected_directory_without_creating_files() {
        let dir = tempfile::tempdir().expect("fixture");
        let config = Config::load(&dir.path().join("desktop.toml")).expect("defaults");
        assert_eq!(config.routing_log, dir.path().join("routing.jsonl"));
        assert!(config.prices_file.is_none());
        assert_eq!(config.config_file, dir.path().join("composite.toml"));
        assert_eq!(config.server_url, "http://127.0.0.1:4123");
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 0);
    }

    #[test]
    fn invalid_price_input_forms_reject_settings() {
        let dir = tempfile::tempdir().expect("fixture");
        let path = dir.path().join("desktop.toml");
        std::fs::write(&path, "refresh_second=30").expect("settings");
        assert!(Config::load(&path).is_err());
        std::fs::write(
            &path,
            "[prices.model]\ninput_per_mtok=1.0\noutput_per_mtok=1.0\n",
        )
        .expect("settings");
        let error = Config::load(&path).expect_err("prices moved");
        assert!(error.contains("prices.toml"), "{error}");
    }
}
