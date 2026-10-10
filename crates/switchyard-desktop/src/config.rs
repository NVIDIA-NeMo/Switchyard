// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module reads app settings from `~/.switchyard/desktop.toml`.

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
        crate::health::validate_url(&config.server_url)?;
        for (model, price) in &config.prices {
            price
                .validate()
                .map_err(|error| format!("Price for {model}: {error}"))?;
        }
        config.routing_log = expand_home(&config.routing_log);
        config.config_file = expand_home(&config.config_file);
        Ok(config)
    }

    /// The app reads installation metadata so Finder and Windows shortcuts retain a custom SY_HOME.
    /// An executable without that metadata uses `~/.switchyard/desktop.toml`.
    pub fn default_path() -> Result<PathBuf, String> {
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let metadata = switchyard_desktop_install::bundle_metadata(&executable);
        if metadata.try_exists().map_err(|e| e.to_string())? {
            return switchyard_desktop_install::Settings::load(&metadata)
                .map(|s| s.desktop_settings());
        }
        Ok(expand_home(Path::new("~/.switchyard/desktop.toml")))
    }
}

pub fn save_preferences(
    path: &Path,
    baseline: &str,
    refresh: u64,
    prices: &PriceTable,
) -> Result<(), String> {
    use std::io::Write;
    if baseline.trim().is_empty() || baseline.contains(['\n', '\r']) {
        return Err("Choose a nonempty baseline model ID without line breaks.".into());
    }
    if refresh == 0 {
        return Err("Refresh interval must be at least one second.".into());
    }
    for (model, price) in prices {
        price
            .validate()
            .map_err(|error| format!("Price for {model}: {error}"))?;
    }
    let original = match std::fs::read_to_string(path) {
        Ok(text) => Some(text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.to_string()),
    };
    let mut doc = original
        .as_deref()
        .unwrap_or("")
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| error.to_string())?;
    doc["baseline_model"] = toml_edit::value(baseline.trim());
    doc["refresh_seconds"] =
        toml_edit::value(i64::try_from(refresh).map_err(|error| error.to_string())?);
    let price_text = toml::to_string(&serde_json::json!({"prices":prices}))
        .map_err(|error| error.to_string())?;
    let price_doc = price_text
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| error.to_string())?;
    doc["prices"] = price_doc["prices"].clone();
    let text = doc.to_string();
    let config: Config = toml::from_str(&text).map_err(|error| error.to_string())?;
    crate::health::validate_url(&config.server_url)?;
    let destination = if original.is_some() {
        std::fs::canonicalize(path).map_err(|error| error.to_string())?
    } else {
        path.to_path_buf()
    };
    let parent = destination
        .parent()
        .ok_or("App settings need a parent folder.")?;
    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let mut temp = tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
    temp.write_all(text.as_bytes())
        .and_then(|()| temp.as_file().sync_all())
        .map_err(|error| error.to_string())?;
    if let Ok(metadata) = std::fs::metadata(path) {
        temp.as_file()
            .set_permissions(metadata.permissions())
            .map_err(|error| error.to_string())?;
    }
    let current = match std::fs::read_to_string(path) {
        Ok(text) => Some(text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.to_string()),
    };
    if current != original {
        return Err("App settings changed while saving. Reload them and try again.".into());
    }
    temp.persist(destination)
        .map_err(|error| error.to_string())?;
    Ok(())
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
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    match path.strip_prefix("~") {
        Ok(rest) => home.join(rest),
        Err(_) => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preference_edits_keep_unrelated_settings_and_reject_invalid_values() {
        let dir = tempfile::tempdir().expect("fixture");
        let path = dir.path().join("desktop.toml");
        let original =
            "# user comment\nserver_url='http://[::1]:4123'\nrouting_log='/custom/log'\n";
        std::fs::write(&path, original).expect("settings");
        assert!(save_preferences(&path, "", 30, &PriceTable::new()).is_err());
        assert!(save_preferences(&path, "baseline", 0, &PriceTable::new()).is_err());
        assert_eq!(std::fs::read_to_string(&path).expect("settings"), original);
        let prices = PriceTable::from([(
            "baseline".into(),
            crate::pricing::ModelPrice {
                input_per_mtok: 1.0,
                cached_input_per_mtok: None,
                output_per_mtok: 2.0,
            },
        )]);
        save_preferences(&path, "baseline", 12, &prices).expect("save");
        let saved = std::fs::read_to_string(&path).expect("saved");
        assert!(saved.contains("# user comment"));
        let config = Config::load(&path).expect("config");
        assert_eq!(config.server_url, "http://[::1]:4123");
        assert_eq!(config.routing_log, PathBuf::from("/custom/log"));
        assert_eq!(config.baseline_model, "baseline");
        assert_eq!(config.refresh_seconds, 12);
        assert_eq!(config.prices, prices);
    }

    #[test]
    fn invalid_price_input_forms_reject_settings() {
        let dir = tempfile::tempdir().expect("fixture");
        let path = dir.path().join("desktop.toml");
        for field in ["input_per_mtok", "cached_input_per_mtok", "output_per_mtok"] {
            for value in ["-1.0", "nan", "inf", "-inf"] {
                let mut rates = vec![
                    "input_per_mtok=0.0".to_string(),
                    "output_per_mtok=0.0".to_string(),
                ];
                rates.retain(|line| !line.starts_with(field));
                rates.push(format!("{field}={value}"));
                std::fs::write(&path, format!("[prices.model]\n{}", rates.join("\n")))
                    .expect("settings");
                assert!(Config::load(&path).is_err(), "{field}={value}");
            }
        }
        std::fs::write(&path,"[prices.model]\ninput_per_mtok=0.0\noutput_per_mtok=1.0\ncached_input_per_mtok_typo=0.0").expect("settings");
        assert!(Config::load(&path).is_err());
        std::fs::write(
            &path,
            "[prices.model]\ninput_per_mtok=0.0\noutput_per_mtok=1.0\ncached_input_per_mtok=0.0",
        )
        .expect("settings");
        assert!(Config::load(&path).is_ok());
    }
}
