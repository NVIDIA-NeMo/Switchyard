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
    /// Callers can use these defaults while the user repairs invalid app settings.
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
        // A dangling settings link needs repair instead of silently using defaults.
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let overrides: toml::Table = toml::from_str(&text)
                    .map_err(|error| format!("parse {}: {error}", path.display()))?;
                values.extend(overrides);
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound
                    && std::fs::symlink_metadata(path)
                        .is_err_and(|lookup| lookup.kind() == std::io::ErrorKind::NotFound) => {}
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

/// Setup is required only when neither the app settings nor its default server config exists.
/// Existing files, including broken symlinks, skip setup and remain available for repair in Settings.
pub fn needs_setup(path: &Path) -> Result<bool, String> {
    fn present(path: &Path) -> Result<bool, String> {
        match std::fs::symlink_metadata(path) {
            Ok(_) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(format!("Read {}: {error}", path.display())),
        }
    }
    Ok(!present(path)? && !present(&Config::defaults_for(path)?.config_file)?)
}

/// This function saves the complete form while retaining comments and unrelated settings.
pub fn save_configuration(path: &Path, config: &Config, create_server: bool) -> Result<(), String> {
    crate::health::validate_url(&config.server_url)?;
    for file in [&config.config_file, &config.routing_log]
        .into_iter()
        .chain(config.prices_file.iter())
    {
        if !file.is_absolute()
            || file
                .as_os_str()
                .to_string_lossy()
                .chars()
                .any(char::is_control)
        {
            return Err("Choose absolute file paths without control characters.".into());
        }
    }
    if config.launchd_label.is_empty() || config.launchd_label.chars().any(char::is_control) {
        return Err("Choose a server service label without control characters.".into());
    }
    let paths = [
        Some(path),
        Some(config.config_file.as_path()),
        Some(config.routing_log.as_path()),
        config.prices_file.as_deref(),
    ];
    let resolved = paths
        .into_iter()
        .flatten()
        .map(|path| resolved_destination(path, 0))
        .collect::<Result<Vec<_>, _>>()?;
    let distinct: std::collections::HashSet<_> = resolved.iter().collect();
    if distinct.len() != resolved.len() {
        return Err("App settings, server config, routing log, and price override must use different files.".into());
    }
    validate_preferences(&config.baseline_model, config.refresh_seconds)?;
    let mut values = toml::Table::try_from(config).map_err(|error| error.to_string())?;
    // An absent price file removes a previous override and restores bundled prices.
    if config.prices_file.is_none() {
        values.insert("pricing_file".into(), toml::Value::String(String::new()));
    }
    if let Some(prices) = &config.prices_file {
        switchyard_costs::Pricing::load(prices).map_err(|error| error.to_string())?;
    }
    if !create_server {
        let text = std::fs::read_to_string(&config.config_file)
            .map_err(|error| format!("Read server config: {error}"))?;
        switchyard_runner::api_key_variables(&text).map_err(|error| error.to_string())?;
    }
    edit_settings(
        path,
        values,
        create_server.then_some(config.config_file.as_path()),
    )
}

// Canonical paths also catch existing symlinks and aliases to a file that setup will create.
fn resolved_destination(path: &Path, depth: usize) -> Result<PathBuf, String> {
    if depth > 40 {
        return Err("A selected file has too many symbolic links or parent folders.".into());
    }
    match std::fs::canonicalize(path) {
        Ok(path) => return Ok(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    if let Ok(target) = std::fs::read_link(path) {
        return resolved_destination(
            &path.parent().ok_or("Missing file folder.")?.join(target),
            depth + 1,
        );
    }
    let name = path
        .file_name()
        .ok_or("Choose a file path without unresolved parent components.")?;
    Ok(resolved_destination(path.parent().ok_or("Missing file folder.")?, depth + 1)?.join(name))
}

fn validate_preferences(baseline: &str, refresh: u64) -> Result<(), String> {
    if baseline.trim().is_empty() || baseline.contains(['\n', '\r']) {
        return Err("Choose a nonempty baseline model ID without line breaks.".into());
    }
    if refresh == 0 {
        return Err("Refresh interval must be at least one second.".into());
    }
    Ok(())
}

pub fn save_preferences(path: &Path, baseline: &str, refresh: u64) -> Result<(), String> {
    validate_preferences(baseline, refresh)?;
    let mut edits = toml::Table::new();
    edits.insert("baseline_model".into(), baseline.trim().into());
    edits.insert(
        "refresh_seconds".into(),
        i64::try_from(refresh)
            .map_err(|error| error.to_string())?
            .into(),
    );
    edit_settings(path, edits, None)
}

fn edit_settings(
    path: &Path,
    edits: toml::Table,
    create_server: Option<&Path>,
) -> Result<(), String> {
    use std::io::Write;
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
    for (key, value) in edits {
        if key == "pricing_file" && value.as_str() == Some("") {
            doc.remove(&key);
        } else {
            doc[&key] = match value {
                toml::Value::String(value) => toml_edit::value(value),
                toml::Value::Integer(value) => toml_edit::value(value),
                _ => return Err(format!("Unsupported app setting: {key}")),
            };
        }
    }
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
    let created = if let Some(server) = create_server {
        let parent = server
            .parent()
            .ok_or("Server config needs a parent folder.")?;
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        let mut staged =
            tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
        staged
            .write_all(crate::install::BUNDLED_CONFIG.as_bytes())
            .and_then(|()| staged.as_file().sync_all())
            .map_err(|error| error.to_string())?;
        staged.persist_noclobber(server).map_err(|error| {
            format!(
                "Create {}: {error}. Choose an existing config without creating a new file.",
                server.display()
            )
        })?;
        Some(server)
    } else {
        None
    };
    if let Err(error) = temp.persist(destination) {
        if let Some(server) = created
            && std::fs::read(server).ok().as_deref()
                == Some(crate::install::BUNDLED_CONFIG.as_bytes())
        {
            let _ = std::fs::remove_file(server);
        }
        return Err(error.to_string());
    }
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
    use super::{Config, needs_setup, save_configuration, save_preferences};

    #[test]
    fn existing_primary_files_skip_setup_even_when_they_need_repair() {
        let dir = tempfile::tempdir().expect("fixture");
        let path = dir.path().join("desktop.toml");
        assert!(needs_setup(&path).expect("fresh"));
        std::fs::write(dir.path().join("routing.jsonl"), "").expect("log");
        std::fs::write(dir.path().join("prices.json"), "{}").expect("prices");
        assert!(needs_setup(&path).expect("optional files"));
        std::fs::write(dir.path().join("composite.toml"), "invalid").expect("server config");
        assert!(!needs_setup(&path).expect("server file"));
        std::fs::remove_file(dir.path().join("composite.toml")).expect("remove server");
        std::fs::write(&path, "invalid [").expect("broken settings");
        assert!(!needs_setup(&path).expect("existing settings"));
    }

    #[test]
    fn setup_saves_selected_files_and_rejected_edits_have_no_file_side_effects() {
        let dir = tempfile::tempdir().expect("fixture");
        let path = dir.path().join("desktop.toml");
        let mut config = Config::defaults_for(&path).expect("defaults");
        config.config_file = dir.path().join("chosen.toml");
        config.routing_log = dir.path().join("chosen.jsonl");
        save_configuration(&path, &config, true).expect("setup");
        assert!(!needs_setup(&path).expect("completed"));
        assert_eq!(
            Config::load(&path).expect("load").config_file,
            config.config_file
        );
        assert_eq!(
            std::fs::read_to_string(&config.config_file).expect("template"),
            crate::install::BUNDLED_CONFIG
        );
        let original = std::fs::read(&path).expect("settings");
        assert!(
            save_configuration(&path, &config, true).is_err(),
            "creating must not replace a file"
        );
        assert_eq!(std::fs::read(&path).expect("unchanged"), original);
        for (url, file) in [
            ("https://invalid.test", dir.path().join("absent.toml")),
            ("http://localhost", path.clone()),
            ("http://localhost", config.routing_log.clone()),
        ] {
            let mut invalid = config.clone();
            invalid.server_url = url.into();
            invalid.config_file = file.clone();
            assert!(save_configuration(&path, &invalid, true).is_err());
            assert_eq!(std::fs::read(&path).expect("unchanged"), original);
            assert!(!dir.path().join("absent.toml").exists());
            assert!(!config.routing_log.exists());
        }
        std::fs::write(&path, "# keep my comment\nunknown_key=true\n").expect("unknown field");
        let before = std::fs::read(&path).expect("raw settings");
        config.config_file = dir.path().join("absent.toml");
        assert!(save_configuration(&path, &config, true).is_err());
        assert_eq!(std::fs::read(&path).expect("raw unchanged"), before);
        assert!(!config.config_file.exists());
    }

    #[cfg(unix)]
    #[test]
    fn preference_edits_preserve_symlinks_and_reject_dangling_targets() {
        let dir = tempfile::tempdir().expect("fixture");
        let path = dir.path().join("desktop.toml");
        let target = dir.path().join("missing.toml");
        std::os::unix::fs::symlink(&target, &path).expect("symlink");
        assert!(!needs_setup(&path).expect("existing settings link"));
        assert!(Config::load(&path).is_err());
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
