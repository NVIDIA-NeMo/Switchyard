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
    expand_under(path, &home)
}

fn expand_under(path: &Path, home: &Path) -> PathBuf {
    match path.strip_prefix("~") {
        Ok(rest) => home.join(rest),
        Err(_) => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_settings_and_prices() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("menubar.toml");
        std::fs::write(
            &path,
            r#"
server_url = "http://127.0.0.1:9000"
baseline_model = "sol"

[prices.luna]
input_per_mtok = 0.25
cached_input_per_mtok = 0.025
output_per_mtok = 2.0
"#,
        )
        .expect("write settings");

        let config = Config::load(&path).expect("load");

        assert_eq!(config.server_url, "http://127.0.0.1:9000");
        assert_eq!(config.baseline_model, "sol");
        assert_eq!(config.prices["luna"].cached_input_per_mtok, Some(0.025));
        assert_eq!(config.refresh_seconds, 30, "unset keys keep their default");
    }

    #[test]
    fn a_missing_file_yields_defaults() {
        let dir = tempfile::tempdir().expect("tempdir");

        let config = Config::load(&dir.path().join("absent.toml")).expect("load");

        assert_eq!(config.server_url, "http://127.0.0.1:4123");
        assert!(config.prices.is_empty());
        let empty = dir.path().join("empty.toml");
        std::fs::write(&empty, "").expect("write settings");
        let loaded = Config::load(&empty).expect("load empty settings");
        assert_eq!(config.routing_log, loaded.routing_log);
        assert_eq!(config.config_file, loaded.config_file);
    }

    #[test]
    fn rejects_unparsable_settings() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("menubar.toml");
        std::fs::write(&path, "server_url = ").expect("write settings");

        assert!(Config::load(&path).is_err());
    }

    #[test]
    fn validates_arbitrary_toml_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("codex.toml");
        std::fs::write(&path, "[model_providers.\"sy\"]\nname = \"Switchyard\"\n")
            .expect("write config");

        assert!(validate_toml_file(&path).is_ok());

        std::fs::write(&path, "[model_providers.\"sy\"]\nname = ").expect("write config");
        assert!(validate_toml_file(&path).is_err());
    }

    #[test]
    fn expands_a_leading_tilde() {
        let home = Path::new("/Users/example");

        assert_eq!(
            expand_under(Path::new("~/.switchyard/routing.jsonl"), home),
            PathBuf::from("/Users/example/.switchyard/routing.jsonl")
        );
        assert_eq!(
            expand_under(Path::new("/tmp/routing.jsonl"), home),
            PathBuf::from("/tmp/routing.jsonl")
        );
    }
}
