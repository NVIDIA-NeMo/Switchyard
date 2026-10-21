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

    /// This function builds defaults for the app settings file at `path` without reading it.
    /// Adjacent `install.toml` selects the installation directory and port.
    /// Missing metadata uses the file's parent directory and port 4123.
    /// Invalid metadata returns an error.
    /// The terminal app uses these defaults while the user repairs invalid app settings.
    pub fn defaults_for(path: &Path) -> Result<Self, String> {
        let directory = path.parent().unwrap_or(Path::new("."));
        let metadata = directory.join("install.toml");
        if metadata.try_exists().map_err(|e| e.to_string())? {
            let settings = Settings::load(&metadata)?;
            Ok(Self::for_directory(&settings.sy_home, settings.port))
        } else {
            Ok(Self::for_directory(directory, DEFAULT_PORT))
        }
    }

    /// This function parses settings and uses adjacent `install.toml` for directory and port defaults.
    /// Without installation metadata, default paths use the settings directory and the port is 4123.
    /// An omitted pricing_file selects bundled rates.
    pub fn load(path: &Path) -> Result<Self, String> {
        let defaults = Self::defaults_for(path)?;
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

pub fn save_preferences(path: &Path, baseline: &str, refresh: u64) -> Result<(), String> {
    use std::io::Write;
    if baseline.trim().is_empty() || baseline.contains(['\n', '\r']) {
        return Err("Choose a nonempty baseline model ID without line breaks.".into());
    }
    if refresh == 0 {
        return Err("Refresh interval must be at least one second.".into());
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
    let text = doc.to_string();
    // Validation fills missing fields with installation defaults, just as Config::load does.
    // The saved text retains comments and changes only the requested preferences.
    let mut values =
        toml::Table::try_from(Config::defaults_for(path)?).map_err(|error| error.to_string())?;
    values.extend(toml::from_str::<toml::Table>(&text).map_err(|error| error.to_string())?);
    let config: Config = values.try_into().map_err(|error| error.to_string())?;
    crate::health::validate_url(&config.server_url)?;
    let destination = match std::fs::symlink_metadata(path) {
        Ok(_) => std::fs::canonicalize(path).map_err(|error| error.to_string())?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => path.to_path_buf(),
        Err(error) => return Err(error.to_string()),
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

/// This function expands a leading `~` in manually written settings and reports a missing home.
pub fn expand_home(path: &Path) -> Result<PathBuf, String> {
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
    fn preference_edits_keep_unrelated_settings_and_reject_invalid_values() {
        let dir = tempfile::tempdir().expect("fixture");
        let path = dir.path().join("desktop.toml");
        std::fs::write(&path, "refresh_second=30").expect("settings");
        assert!(Config::load(&path).is_err());
        let original =
            "# user comment\nserver_url='http://[::1]:4123'\nrouting_log='/custom/log'\n";
        std::fs::write(&path, original).expect("settings");
        assert!(save_preferences(&path, "", 30).is_err());
        assert!(save_preferences(&path, "baseline", 0).is_err());
        assert_eq!(std::fs::read_to_string(&path).expect("settings"), original);
        save_preferences(&path, "baseline", 12).expect("save");
        let saved = std::fs::read_to_string(&path).expect("saved");
        assert!(saved.contains("# user comment"));
        let config = Config::load(&path).expect("config");
        assert_eq!(config.server_url, "http://[::1]:4123");
        assert_eq!(config.routing_log, PathBuf::from("/custom/log"));
        assert_eq!(config.baseline_model, "baseline");
        assert_eq!(config.refresh_seconds, 12);

        std::fs::write(
            &path,
            "[prices.model]\ninput_per_mtok=1.0\noutput_per_mtok=1.0\n",
        )
        .expect("settings");
        let error = Config::load(&path).expect_err("prices moved");
        assert!(error.contains("prices.toml"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn preference_edits_preserve_symlinks_and_reject_dangling_targets() {
        let dir = tempfile::tempdir().expect("fixture");
        let path = dir.path().join("desktop.toml");
        let target = dir.path().join("missing.toml");
        std::os::unix::fs::symlink(&target, &path).expect("symlink");
        assert!(save_preferences(&path, "baseline", 30).is_err());
        assert_eq!(
            std::fs::read_link(&path).expect("preserved symlink"),
            target
        );
        assert!(!target.exists());
        std::fs::write(&target, "refresh_seconds=30\n").expect("target");
        save_preferences(&path, "baseline", 12).expect("save through symlink");
        assert_eq!(
            std::fs::read_link(&path).expect("preserved symlink"),
            target
        );
        assert_eq!(
            Config::load(&target)
                .expect("target settings")
                .refresh_seconds,
            12
        );
    }
}
