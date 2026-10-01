// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Menu bar settings, read from `~/.switchyard/menubar.toml`.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::pricing::PriceTable;

/// Everything the menu bar needs to find the server and price its traffic.
/// Missing keys fall back to [`Config::default`].
#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Base URL the server listens on, used for the health check.
    pub server_url: String,
    /// JSONL file the server appends routing records to.
    pub routing_log: PathBuf,
    /// Server TOML, opened by the "Open server config" menu item.
    pub config_file: PathBuf,
    /// LaunchAgent label used to restart the server.
    pub launchd_label: String,
    /// How often the menu contents are recomputed.
    pub refresh_seconds: u64,
    /// Model the traffic is assumed to have used without Switchyard.
    pub baseline_model: String,
    /// Per-model rates. Dollar figures are hidden while a seen model is absent.
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
    /// Parses settings, falling back to defaults when the file is absent.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => return Err(format!("read {}: {error}", path.display())),
        };
        let mut config: Self =
            toml::from_str(&text).map_err(|error| format!("parse {}: {error}", path.display()))?;
        config.routing_log = expand_home(&config.routing_log);
        config.config_file = expand_home(&config.config_file);
        Ok(config)
    }

    /// Default settings path, `~/.switchyard/menubar.toml`.
    pub fn default_path() -> PathBuf {
        expand_home(Path::new("~/.switchyard/menubar.toml"))
    }
}

/// Rewrites a leading `~`. Paths are written by hand in the settings file,
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
    }

    #[test]
    fn rejects_unparsable_settings() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("menubar.toml");
        std::fs::write(&path, "server_url = ").expect("write settings");

        assert!(Config::load(&path).is_err());
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
