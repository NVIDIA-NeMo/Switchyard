// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module installs and restores user settings for the supported coding tools.

use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};
use toml_edit::{DocumentMut, value};

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Harness {
    CodexCli,
    CodexApp,
    Claude,
    Pi,
}

pub const HARNESSES: &[(Harness, &str)] = &[
    (Harness::CodexCli, "Codex CLI profile"),
    (Harness::CodexApp, "Codex app and CLI defaults"),
    (Harness::Claude, "Claude Code"),
    (Harness::Pi, "Pi"),
];

pub fn paths(tool: Harness) -> Vec<PathBuf> {
    let home = crate::config::expand_home(Path::new("~"));
    let codex = std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".codex"));
    let claude = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".claude"));
    let pi = std::env::var_os("PI_CODING_AGENT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".pi/agent"));
    match tool {
        Harness::CodexCli => vec![codex.join("sy.config.toml")],
        Harness::CodexApp => vec![codex.join("config.toml")],
        Harness::Claude => vec![claude.join("settings.json")],
        Harness::Pi => vec![pi.join("models.json"), pi.join("settings.json")],
    }
}

pub fn inspect(tool: Harness, files: &[PathBuf]) -> Result<String, String> {
    let texts = files
        .iter()
        .map(|path| read(path))
        .collect::<Result<Vec<_>, _>>()?;
    inspect_settings(tool, files, &texts)
}

fn inspect_settings(tool: Harness, files: &[PathBuf], texts: &[String]) -> Result<String, String> {
    let text = &texts[0];
    let model;
    let endpoint;
    match tool {
        Harness::CodexCli | Harness::CodexApp => {
            let doc = text
                .parse::<DocumentMut>()
                .map_err(|e| e.message().to_owned())?;
            model = doc
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("Default")
                .to_string();
            let provider = doc
                .get("model_provider")
                .and_then(|v| v.as_str())
                .unwrap_or("openai");
            endpoint = doc
                .get("model_providers")
                .and_then(|v| v.get(provider))
                .and_then(|v| v.get("base_url"))
                .and_then(|v| v.as_str())
                .unwrap_or("Provider default")
                .to_string();
        }
        Harness::Claude => {
            let doc = object(text)?;
            model = doc
                .pointer("/env/ANTHROPIC_MODEL")
                .and_then(Value::as_str)
                .or_else(|| doc.get("model").and_then(Value::as_str))
                .unwrap_or("Default")
                .to_string();
            endpoint = doc
                .pointer("/env/ANTHROPIC_BASE_URL")
                .and_then(Value::as_str)
                .unwrap_or("Anthropic default")
                .to_string();
        }
        Harness::Pi => {
            let doc = object(&texts[1])?;
            model = doc
                .get("defaultModel")
                .and_then(Value::as_str)
                .unwrap_or("Default")
                .to_string();
            let provider = doc
                .get("defaultProvider")
                .and_then(Value::as_str)
                .unwrap_or("Default");
            let models = object(text)?;
            endpoint = models
                .get("providers")
                .and_then(|v| v.get(provider))
                .and_then(|v| v.get("baseUrl"))
                .and_then(Value::as_str)
                .unwrap_or("Provider default")
                .to_string();
        }
    }
    // The display keeps the URL scheme, host, and port to omit credentials and queries.
    let endpoint = url::Url::parse(&endpoint)
        .ok()
        .map(|u| {
            format!(
                "{}://{}{}",
                u.scheme(),
                u.host_str().unwrap_or("?"),
                u.port().map(|p| format!(":{p}")).unwrap_or_default()
            )
        })
        .unwrap_or_else(|| "Provider default".into());
    Ok(format!(
        "Model: {model}\nEndpoint: {endpoint}\nSettings: {}",
        files
            .iter()
            .map(|file| file.display().to_string())
            .collect::<Vec<_>>()
            .join("\n          ")
    ))
}

fn read_file(path: &Path) -> Result<Option<String>, String> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => {
            return Err(format!(
                "{} is a symlink. Edit its target yourself.",
                path.display()
            ));
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.to_string()),
        _ => {}
    }
    std::fs::read_to_string(path)
        .map(Some)
        .map_err(|e| format!("Read {}: {e}", path.display()))
}

fn read(path: &Path) -> Result<String, String> {
    Ok(read_file(path)?.unwrap_or_default())
}

fn object(text: &str) -> Result<Value, String> {
    let doc: Value = if text.is_empty() {
        json!({})
    } else {
        serde_json::from_str(text).map_err(|e| e.to_string())?
    };
    if !doc.is_object() {
        return Err("Settings must be a JSON object.".into());
    }
    Ok(doc)
}

fn map<'a>(doc: &'a mut Value, key: &str) -> Result<&'a mut Value, String> {
    if doc.get(key).is_none() {
        doc[key] = json!({});
    }
    if !doc[key].is_object() {
        return Err(format!("{key} must be a JSON object."));
    }
    Ok(&mut doc[key])
}

pub fn validate(tool: Harness, root: &str, model: &str, login: bool) -> Result<String, String> {
    let url = url::Url::parse(root).map_err(|e| e.to_string())?;
    if url.scheme() != "http"
        || !matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/"
    {
        return Err(
            "Install requires a local HTTP server URL, such as http://127.0.0.1:4123.".into(),
        );
    }
    if model.trim().is_empty() {
        return Err("Pick a route first.".into());
    }
    if tool == Harness::Pi && login {
        return Err("Pi's custom provider needs a route with server API credentials. Use Codex or Claude for subscription login forwarding.".into());
    }
    // validate returns the parsed URL without a trailing slash so callers can append API paths.
    Ok(url.to_string().trim_end_matches('/').to_owned())
}

fn codex_settings(text: &str, root: &str, model: &str, login: bool) -> Result<String, String> {
    let mut doc = text
        .parse::<DocumentMut>()
        .map_err(|e| e.message().to_owned())?;
    if doc
        .get("model_providers")
        .is_some_and(|item| item.as_table_like().is_none())
    {
        return Err("model_providers must be a TOML table.".into());
    }
    if doc
        .get("model_providers")
        .and_then(|v| v.get("sy"))
        .is_some_and(|item| item.as_table_like().is_none())
    {
        return Err("sy must be a TOML provider table.".into());
    }
    doc["model"] = value(model);
    doc["model_provider"] = value("sy");
    doc["model_providers"]["sy"]["name"] = value("Switchyard");
    doc["model_providers"]["sy"]["base_url"] = value(format!("{root}/v1"));
    doc["model_providers"]["sy"]["wire_api"] = value("responses");
    doc["model_providers"]["sy"]["requires_openai_auth"] = value(login);
    // A stale env_key would make Codex require a local API key even when the
    // server owns the credentials or forwards the coding tool’s subscription login.
    doc["model_providers"]["sy"]
        .as_table_like_mut()
        .ok_or("sy must be a provider table")?
        .remove("env_key");
    Ok(doc.to_string())
}

fn claude_settings(text: &str, root: &str, model: &str, login: bool) -> Result<String, String> {
    let mut doc = object(text)?;
    let env = map(&mut doc, "env")?;
    let role_models = [
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    ];
    // The complete installation signature distinguishes our placeholder from a user credential.
    let prior_switchyard = env["ANTHROPIC_BASE_URL"].as_str() == Some(root)
        && env["ANTHROPIC_AUTH_TOKEN"].as_str() == Some("switchyard-local")
        && env["ANTHROPIC_API_KEY"].as_str() == Some("")
        && env["ANTHROPIC_MODEL"]
            .as_str()
            .is_some_and(|model| !model.is_empty())
        && role_models
            .iter()
            .all(|key| env[*key] == env["ANTHROPIC_MODEL"]);
    env["ANTHROPIC_BASE_URL"] = json!(root);
    env["ANTHROPIC_MODEL"] = json!(model);
    // All model roles use the selected public route, including subagents.
    for key in role_models {
        env[key] = json!(model);
    }
    if login {
        if prior_switchyard {
            let values = env.as_object_mut().ok_or("env must be a JSON object.")?;
            values.remove("ANTHROPIC_AUTH_TOKEN");
            values.remove("ANTHROPIC_API_KEY");
        }
        for key in ["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"] {
            if env.get(key).is_some_and(|value| value.as_str() != Some("")) {
                return Err(format!(
                    "{key} overrides subscription login. Remove it from Claude settings before installing this route."
                ));
            }
        }
    } else {
        env["ANTHROPIC_AUTH_TOKEN"] = json!("switchyard-local");
        env["ANTHROPIC_API_KEY"] = json!("");
    }
    Ok(serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())? + "\n")
}

// The preview uses the installation transformations without creating files or backups.
pub fn preview(
    tool: Harness,
    files: &[PathBuf],
    root: &str,
    model: &str,
    login: bool,
) -> Result<Value, String> {
    let root = validate(tool, root, model, login)?;
    let originals = files
        .iter()
        .map(|path| read_file(path))
        .collect::<Result<Vec<_>, _>>()?;
    let incoming = configured(tool, &originals, &root, model, login)?;
    let current = originals
        .iter()
        .map(|text| text.clone().unwrap_or_default())
        .collect::<Vec<_>>();
    Ok(json!({
        "changes": settings_changes(tool, files, &current, &incoming)?,
        "current": inspect_settings(tool, files, &current)?,
        "proposed": inspect_settings(tool, files, &incoming)?,
        "authentication": if login { "This route uses the coding tool’s subscription login." } else { "This route uses the API credentials configured on the Switchyard server." },
        "files": files.iter().map(|path| path.display().to_string()).collect::<Vec<_>>()
    }))
}

// PreviewModel and its nested types omit unknown fields that could contain credentials.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct PreviewModel {
    id: Option<String>,
    name: Option<String>,
    reasoning: Option<bool>,
    input: Option<Vec<PreviewInput>>,
    context_window: Option<u64>,
    max_tokens: Option<u64>,
    cost: Option<PreviewCosts>,
}
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
enum PreviewInput {
    Text,
    Image,
}
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct PreviewCosts {
    input: Option<f64>,
    output: Option<f64>,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
}

// settings_changes compares supported routing fields and hides credential values before display.
fn settings_changes(
    tool: Harness,
    files: &[PathBuf],
    before: &[String],
    after: &[String],
) -> Result<Vec<Value>, String> {
    let mut changes = Vec::new();
    for (index, file) in files.iter().enumerate() {
        let decode = |text: &str| -> Result<Value, String> {
            if matches!(tool, Harness::CodexCli | Harness::CodexApp) {
                let doc: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
                serde_json::to_value(doc).map_err(|e| e.to_string())
            } else {
                object(text)
            }
        };
        let old = decode(&before[index])?;
        let new = decode(&after[index])?;
        let keys: &[&str] = match tool {
            Harness::CodexCli | Harness::CodexApp => &[
                "/model",
                "/model_provider",
                "/model_providers/sy/name",
                "/model_providers/sy/env_key",
                "/model_providers/sy/base_url",
                "/model_providers/sy/requires_openai_auth",
                "/model_providers/sy/wire_api",
            ],
            Harness::Claude => &[
                "/model",
                "/env/ANTHROPIC_MODEL",
                "/env/ANTHROPIC_DEFAULT_OPUS_MODEL",
                "/env/ANTHROPIC_DEFAULT_SONNET_MODEL",
                "/env/ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "/env/ANTHROPIC_BASE_URL",
                "/env/ANTHROPIC_AUTH_TOKEN",
                "/env/ANTHROPIC_API_KEY",
            ],
            Harness::Pi if index == 0 => &[
                "/providers/switchyard/baseUrl",
                "/providers/switchyard/api",
                "/providers/switchyard/apiKey",
                "/providers/switchyard/headers",
                "/providers/switchyard/models",
            ],
            Harness::Pi => &["/defaultModel", "/defaultProvider"],
        };
        for key in keys {
            let left = old.pointer(key);
            let right = new.pointer(key);
            if left == right {
                continue;
            }
            let display = |value: Option<&Value>| -> String {
                let Some(value) = value else {
                    return "(unset)".into();
                };
                match *key {
                    "/env/ANTHROPIC_AUTH_TOKEN"
                    | "/env/ANTHROPIC_API_KEY"
                    | "/providers/switchyard/apiKey" => if value.as_str() == Some("") {
                        "(empty)"
                    } else {
                        "(set; hidden)"
                    }
                    .into(),
                    "/providers/switchyard/headers" => value
                        .as_object()
                        .map(|headers| {
                            headers
                                .keys()
                                .map(|name| format!("{name}: (set; hidden)"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_else(|| "(configured headers; values hidden)".into()),
                    "/providers/switchyard/models" => value
                        .as_array()
                        .map(|models| {
                            models
                                .iter()
                                .map(|model| {
                                    serde_json::from_value::<PreviewModel>(model.clone())
                                        .and_then(serde_json::to_value)
                                        .unwrap_or_else(|_| {
                                            json!("(unsupported model metadata; hidden)")
                                        })
                                })
                                .collect::<Vec<_>>()
                        })
                        .map(|models| Value::Array(models).to_string())
                        .unwrap_or_default(),
                    "/env/ANTHROPIC_BASE_URL"
                    | "/model_providers/sy/base_url"
                    | "/providers/switchyard/baseUrl" => value
                        .as_str()
                        .and_then(|v| url::Url::parse(v).ok())
                        .map(|mut url| {
                            let _ = url.set_username("");
                            let _ = url.set_password(None);
                            url.set_query(None);
                            url.set_fragment(None);
                            url.to_string()
                        })
                        .unwrap_or_else(|| "(configured endpoint)".into()),
                    _ => match value {
                        Value::String(value) => value.clone(),
                        Value::Bool(value) => value.to_string(),
                        _ => "(set; hidden)".into(),
                    },
                }
            };
            changes.push(json!({"file":file.display().to_string(),"key":key.trim_start_matches('/').replace('/',"."),"before":display(left),"after":display(right)}));
        }
    }
    Ok(changes)
}

// Both preview and installation use these transformations so the proposed settings match the saved settings.
fn configured(
    tool: Harness,
    originals: &[Option<String>],
    root: &str,
    model: &str,
    login: bool,
) -> Result<Vec<String>, String> {
    let mut incoming = Vec::new();
    match tool {
        Harness::CodexCli | Harness::CodexApp => incoming.push(codex_settings(
            originals[0].as_deref().unwrap_or_default(),
            root,
            model,
            login,
        )?),
        Harness::Claude => incoming.push(claude_settings(
            originals[0].as_deref().unwrap_or_default(),
            root,
            model,
            login,
        )?),
        Harness::Pi => {
            let mut doc = object(originals[0].as_deref().unwrap_or_default())?;
            let providers = map(&mut doc, "providers")?;
            providers["switchyard"] = json!({"baseUrl":format!("{root}/v1"), "api":"openai-completions", "apiKey":"switchyard-local", "headers":{"x-switchyard-origin":"pi"}, "models":[{"id":model,"name":model,"reasoning":true,"input":["text"],"contextWindow":128000,"maxTokens":16384,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0}}]});
            incoming.push(serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())? + "\n");
            let mut settings = object(originals[1].as_deref().unwrap_or_default())?;
            settings["defaultProvider"] = json!("switchyard");
            settings["defaultModel"] = json!(model);
            incoming
                .push(serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())? + "\n");
        }
    }
    Ok(incoming)
}

pub fn install(
    tool: Harness,
    files: &[PathBuf],
    root: &str,
    model: &str,
    login: bool,
) -> Result<String, String> {
    let root = validate(tool, root, model, login)?;
    for path in files {
        if read_file(&restore_pending_path(path))?.is_some() {
            return Err("A previous restore did not finish retiring its backups. Retry Restore before installing another route.".into());
        }
    }
    let originals = files
        .iter()
        .map(|p| read_file(p))
        .collect::<Result<Vec<_>, _>>()?;
    let incoming = configured(tool, &originals, &root, model, login)?;
    let incoming: Vec<_> = incoming.into_iter().map(Some).collect();
    for (path, before) in files.iter().zip(&originals) {
        let backup = backup_path(path);
        let absent = absent_path(path);
        if read_file(&backup)?.is_some() || read_file(&absent)?.is_some() {
            continue;
        }
        let (destination, text) = match before {
            Some(text) => (&backup, text.as_str()),
            None => (&absent, ""),
        };
        prepare(destination, text)?
            .persist_noclobber(destination)
            .map_err(|e| e.to_string())?;
    }
    replace(files, &originals, &incoming)?;
    Ok("Saved the Switchyard settings and kept the original backup. Restart the coding tool. For Codex CLI, use codex -p sy. Pi can reload models with /model. Project settings and shell variables can override these user defaults.".into())
}

fn backup_path(path: &Path) -> PathBuf {
    path.with_file_name(format!(
        "{}.switchyard-original",
        path.file_name().unwrap_or_default().to_string_lossy()
    ))
}

// The marker distinguishes an originally missing file from an empty file.
// Restore removes installed settings only when the original file was missing.
fn absent_path(path: &Path) -> PathBuf {
    backup_path(path).with_extension("switchyard-original-missing")
}

fn restore_pending_path(path: &Path) -> PathBuf {
    backup_path(path).with_extension("switchyard-restore-pending")
}

// Restore receipts keep the original contents or absence available until backup cleanup finishes.
pub fn restore(files: &[PathBuf]) -> Result<String, String> {
    let pending = files
        .iter()
        .map(|path| read_file(&restore_pending_path(path)))
        .collect::<Result<Vec<_>, _>>()?;
    let resuming = pending.iter().any(Option::is_some);
    let mut restored = Vec::new();
    for (path, pending) in files.iter().zip(pending) {
        let original = read_file(&backup_path(path))?;
        let absent = read_file(&absent_path(path))?.is_some();
        let original = match pending {
            Some(text) => serde_json::from_str::<Option<String>>(&text)
                .map_err(|e| format!("Read pending restore: {e}"))?,
            None if original.is_none() && !absent => {
                if !resuming {
                    return Err("No original backup is available for this installation.".into());
                }
                // A cleared receipt means this file finished restoring, so a retry preserves later edits.
                read_file(path)?
            }
            None => original,
        };
        restored.push(original);
    }
    let current = files
        .iter()
        .map(|p| read_file(p))
        .collect::<Result<Vec<_>, _>>()?;
    let mut kept_paths = Vec::new();
    for (path, before) in files.iter().zip(&current) {
        let Some(before) = before else {
            continue;
        };
        let mut recovery = tempfile::Builder::new()
            .prefix("switchyard-before-restore-")
            .tempfile_in(path.parent().ok_or("Settings need a parent directory")?)
            .map_err(|e| e.to_string())?;
        recovery
            .write_all(before.as_bytes())
            .map_err(|e| e.to_string())?;
        let (_, kept) = recovery.keep().map_err(|e| e.to_string())?;
        kept_paths.push(kept.display().to_string());
    }
    // Each private receipt stores the original contents or absence and blocks installation during cleanup.
    for (path, original) in files.iter().zip(&restored) {
        let pending = restore_pending_path(path);
        if read_file(&pending)?.is_none() {
            let text = serde_json::to_string(original).map_err(|e| e.to_string())?;
            prepare(&pending, &text)?
                .persist_noclobber(&pending)
                .map_err(|e| e.to_string())?;
        }
    }
    let recovery = format!("Previous settings: {}", kept_paths.join(", "));
    replace(files, &current, &restored).map_err(|error| format!("{error}. {recovery}"))?;
    for path in files {
        for retired in [backup_path(path), absent_path(path)] {
            match std::fs::remove_file(&retired) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(format!(
                        "Restored settings, but could not retire {}: {error}. Retry Restore before installing another route. {recovery}",
                        retired.display()
                    ));
                }
            }
        }
    }
    for path in files {
        let pending = restore_pending_path(path);
        std::fs::remove_file(&pending).map_err(|error| {
            format!(
                "Restored settings, but could not clear {}: {error}. {recovery}",
                pending.display()
            )
        })?;
    }
    Ok(format!(
        "Restored original settings. Restart the coding tool. {recovery}"
    ))
}

fn prepare(path: &Path, text: &str) -> Result<tempfile::NamedTempFile, String> {
    let parent = path.parent().ok_or("Settings need a parent directory")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let mut temp = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    temp.write_all(text.as_bytes())
        .and_then(|_| temp.as_file().sync_all())
        .map_err(|e| e.to_string())?;
    Ok(temp)
}

// Each file replacement is atomic, but the set of replacements is not.
// A later failure triggers an attempt to restore files already replaced.
fn replace(
    files: &[PathBuf],
    before: &[Option<String>],
    after: &[Option<String>],
) -> Result<(), String> {
    let staged = files
        .iter()
        .zip(after)
        .map(|(path, text)| text.as_deref().map(|text| prepare(path, text)).transpose())
        .collect::<Result<Vec<_>, _>>()?;
    for (path, expected) in files.iter().zip(before) {
        if read_file(path)? != *expected {
            return Err(format!(
                "{} changed while installing. Refresh and try again.",
                path.display()
            ));
        }
    }
    for (index, (path, temp)) in files.iter().zip(staged).enumerate() {
        let result = match temp {
            Some(temp) => temp.persist(path).map(|_| ()).map_err(|e| e.to_string()),
            None if before[index].is_some() => {
                std::fs::remove_file(path).map_err(|e| e.to_string())
            }
            None => Ok(()),
        };
        if let Err(error) = result {
            for (path, original) in files[..index].iter().zip(before) {
                match original {
                    Some(text) => {
                        prepare(path, text)?
                            .persist(path)
                            .map_err(|e| e.to_string())?;
                    }
                    None => std::fs::remove_file(path).map_err(|e| e.to_string())?,
                }
            }
            return Err(error);
        }
    }
    Ok(())
}

pub fn binary(tool: Harness) -> Option<PathBuf> {
    if tool == Harness::CodexApp {
        return [
            PathBuf::from("/Applications/Codex.app"),
            crate::config::expand_home(Path::new("~/Applications/Codex.app")),
        ]
        .into_iter()
        .find(|p| p.exists());
    }
    let name = match tool {
        Harness::CodexCli => "codex",
        Harness::Claude => "claude",
        Harness::Pi => "pi",
        Harness::CodexApp => return None,
    };
    let mut dirs: Vec<_> = std::env::var_os("PATH")
        .map(|s| std::env::split_paths(&s).collect())
        .unwrap_or_default();
    dirs.extend([
        PathBuf::from("/opt/homebrew/bin"),
        PathBuf::from("/usr/local/bin"),
        crate::config::expand_home(Path::new("~/.local/bin")),
        crate::config::expand_home(Path::new("~/.npm-global/bin")),
    ]);
    dirs.into_iter().map(|p| p.join(name)).find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    // Nested unknown fields and malformed metadata must not expose credentials in the preview.
    fn pi_preview_includes_model_limits_and_redacts_headers() {
        let dir = tempfile::tempdir().expect("directory");
        let files = vec![
            dir.path().join("models.json"),
            dir.path().join("settings.json"),
        ];
        std::fs::write(&files[0], r#"{"providers":{"switchyard":{"baseUrl":"http://localhost:4123/v1","api":"openai-completions","apiKey":"SECRET","headers":{"Authorization":"SECRET"},"models":[{"id":"route","name":"Route","reasoning":true,"input":["text","image"],"contextWindow":1000,"maxTokens":20,"cost":{"input":12,"output":3,"cacheRead":1,"cacheWrite":2,"unknown":{"key":"SECRET"}}},{"id":"bad-name","name":{"key":"SECRET"}},{"id":"bad-cost","cost":{"input":{"key":"SECRET"}}},{"id":"bad-input","input":["text",{"key":"SECRET"}]}]}}}"#).expect("models");
        let result =
            preview(Harness::Pi, &files, "http://localhost:4123", "route", false).expect("preview");
        assert!(!result.to_string().contains("SECRET"));
        let changes = result["changes"].as_array().expect("changes");
        let models = changes
            .iter()
            .find(|change| change["key"] == "providers.switchyard.models")
            .expect("models diff");
        assert!(models["before"].as_str().expect("before").contains("1000"));
        assert!(models["after"].as_str().expect("after").contains("128000"));
        let projected: Value =
            serde_json::from_str(models["before"].as_str().expect("before")).expect("models");
        assert_eq!(
            projected[0]["cost"],
            json!({"input":12.0,"output":3.0,"cacheRead":1.0,"cacheWrite":2.0})
        );
        assert_eq!(projected[0]["input"], json!(["text", "image"]));
        assert_eq!(projected[0]["name"], "Route");
        for model in projected.as_array().expect("models").iter().skip(1) {
            assert_eq!(model, "(unsupported model metadata; hidden)");
        }

        assert!(
            changes
                .iter()
                .any(|change| change["key"] == "providers.switchyard.headers")
        );
    }

    #[test]
    // Preview must match the installed settings without creating files or backups.
    fn preview_matches_installation_without_writes() {
        for (tool, _) in HARNESSES {
            let dir = tempfile::tempdir().expect("directory");
            let files = if *tool == Harness::Pi {
                vec![
                    dir.path().join("models.json"),
                    dir.path().join("settings.json"),
                ]
            } else {
                vec![dir.path().join("config")]
            };
            let original = if matches!(tool, Harness::CodexCli | Harness::CodexApp) {
                "# user comment\nmodel='before'\nsandbox_mode='workspace-write'\n"
            } else {
                "{\"custom\":true}"
            };
            std::fs::write(&files[0], original).expect("settings");
            let result = preview(*tool, &files, "http://localhost:4123", "new-route", false)
                .expect("preview");
            assert!(
                result["changes"]
                    .as_array()
                    .expect("diff")
                    .iter()
                    .any(|change| change["after"]
                        .as_str()
                        .is_some_and(|value| value.contains("new-route")))
            );
            assert_eq!(read(&files[0]).expect("unchanged"), original);
            assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 1);
            install(*tool, &files, "http://localhost:4123", "new-route", false).expect("install");
            assert_eq!(
                result["proposed"],
                inspect(*tool, &files).expect("installed")
            );
            install(
                *tool,
                &files,
                "http://localhost:4123",
                "refreshed-route",
                false,
            )
            .expect("refresh");
            let installed = read(&files[0]).expect("settings");
            assert!(installed.contains("refreshed-route"));
            assert!(
                installed.contains(if matches!(tool, Harness::CodexCli | Harness::CodexApp) {
                    "sandbox_mode"
                } else {
                    "custom"
                })
            );
            assert_eq!(read(&backup_path(&files[0])).expect("backup"), original);
            restore(&files).expect("restore");
            assert_eq!(read(&files[0]).expect("original"), original);
            if *tool == Harness::Pi {
                assert!(!files[1].exists());
            }
            std::fs::write(&files[0], original.replace("before", "new-user-model"))
                .expect("user edit");
            install(*tool, &files, "http://localhost:4123", "again", false).expect("reinstall");
            restore(&files).expect("restore again");
            assert_eq!(
                read(&files[0]).expect("new original"),
                original.replace("before", "new-user-model")
            );
            assert!(
                result["authentication"]
                    .as_str()
                    .expect("authentication")
                    .contains("API credentials")
            );
        }
        let dir = tempfile::tempdir().expect("directory");
        let files = vec![dir.path().join("settings.json")];
        let original =
            r#"{"env":{"ANTHROPIC_API_KEY":"SECRET","ANTHROPIC_MODEL":{"key":"SECRET"}}}"#;
        std::fs::write(&files[0], original).expect("settings");
        assert!(
            preview(
                Harness::Claude,
                &files,
                "http://localhost:4123",
                "route",
                true
            )
            .is_err()
        );
        assert_eq!(read(&files[0]).expect("unchanged"), original);
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 1);
        let safe = preview(
            Harness::Claude,
            &files,
            "http://localhost:4123",
            "route",
            false,
        )
        .expect("API preview");
        assert!(!safe.to_string().contains("SECRET"));
        assert!(
            safe["changes"]
                .as_array()
                .expect("diff")
                .iter()
                .any(|change| change["key"] == "env.ANTHROPIC_API_KEY"
                    && change["before"] == "(set; hidden)")
        );
        std::fs::write(&files[0], "{}").expect("settings");
        let result = preview(
            Harness::Claude,
            &files,
            "http://localhost:4123",
            "route",
            true,
        )
        .expect("login preview");
        assert!(
            result["authentication"]
                .as_str()
                .expect("authentication")
                .contains("subscription login")
        );
    }

    #[test]
    fn a_pending_restore_blocks_install_and_retires_stale_backups_on_retry() {
        let dir = tempfile::tempdir().expect("directory");
        let path = dir.path().join("settings.json");
        let files = std::slice::from_ref(&path);
        std::fs::write(&path, "{}").expect("settings");
        install(
            Harness::Claude,
            files,
            "http://localhost:4123",
            "api",
            false,
        )
        .expect("install");
        let pending = restore_pending_path(&path);
        std::fs::write(
            &pending,
            serde_json::to_string(&Some("{}".to_string())).expect("receipt"),
        )
        .expect("pending retirement");
        let installed = read(&path).expect("installed");
        assert!(
            install(
                Harness::Claude,
                files,
                "http://localhost:4123",
                "another",
                false
            )
            .is_err()
        );
        assert_eq!(read(&path).expect("unchanged"), installed);
        // The backup was already retired, so retry uses the saved restoration receipt.
        std::fs::remove_file(backup_path(&path)).expect("retired backup");
        restore(files).expect("retry restore");
        assert_eq!(read(&path).expect("restored"), "{}");
        assert!(!pending.exists());
    }

    // Pi cleanup can stop between files; retry must keep edits to a file that already finished.
    #[test]
    fn pi_restore_retry_preserves_a_file_whose_receipt_was_already_cleared() {
        let dir = tempfile::tempdir().expect("directory");
        let files = vec![
            dir.path().join("models.json"),
            dir.path().join("settings.json"),
        ];
        for path in &files {
            std::fs::write(path, "{}").expect("original");
        }
        install(Harness::Pi, &files, "http://localhost:4123", "route", false).expect("install");
        for path in &files {
            std::fs::remove_file(backup_path(path)).expect("retired backup");
            std::fs::write(path, "{}").expect("restored settings");
        }
        std::fs::write(&files[0], "{\"user_edit\":true}").expect("edit after restore");
        std::fs::write(restore_pending_path(&files[1]), r#""{}""#).expect("remaining receipt");
        restore(&files).expect("retry cleanup");
        assert_eq!(read(&files[0]).expect("first file"), "{\"user_edit\":true}");
        assert_eq!(read(&files[1]).expect("second file"), "{}");
        assert!(
            files
                .iter()
                .all(|path| !restore_pending_path(path).exists())
        );
    }

    #[test]
    fn subscription_install_removes_only_switchyard_credentials() {
        let dir = tempfile::tempdir().expect("directory");
        let path = dir.path().join("settings.json");
        let files = std::slice::from_ref(&path);
        std::fs::write(&path, "{}").expect("settings");
        install(
            Harness::Claude,
            files,
            "http://localhost:4123",
            "api-route",
            false,
        )
        .expect("API install");
        install(
            Harness::Claude,
            files,
            "http://localhost:4123",
            "subscription",
            true,
        )
        .expect("subscription install");
        let doc = object(&read(&path).expect("settings")).expect("JSON");
        assert!(doc["env"].get("ANTHROPIC_AUTH_TOKEN").is_none());
        assert!(doc["env"].get("ANTHROPIC_API_KEY").is_none());
        assert_eq!(doc["env"]["ANTHROPIC_MODEL"], "subscription");
        for key in ["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"] {
            for value in [
                json!("user-secret"),
                json!("switchyard-local"),
                json!(null),
                json!(false),
                json!(123),
                json!([]),
                json!({}),
            ] {
                let original = json!({"env":{key:value}}).to_string();
                std::fs::write(&path, &original).expect("settings");
                let before = std::fs::read_dir(dir.path()).expect("files").count();
                assert!(
                    install(
                        Harness::Claude,
                        files,
                        "http://localhost:4123",
                        "subscription",
                        true
                    )
                    .is_err()
                );
                assert_eq!(read(&path).expect("unchanged"), original);
                assert_eq!(
                    std::fs::read_dir(dir.path()).expect("files").count(),
                    before
                );
            }
        }
    }

    #[test]
    fn invalid_inputs_change_no_files() {
        let dir = tempfile::tempdir().expect("directory");
        let path = dir.path().join("settings.json");
        for invalid in ["[]", "null", "{", "{\"env\":false}"] {
            std::fs::write(&path, invalid).expect("settings");
            assert!(
                install(
                    Harness::Claude,
                    std::slice::from_ref(&path),
                    "http://localhost:4123",
                    "route",
                    false
                )
                .is_err()
            );
            assert_eq!(read(&path).expect("read"), invalid);
            assert!(!backup_path(&path).exists());
        }
        std::fs::write(&path, "{}").expect("settings");
        for root in [
            "https://example.com",
            "http://localhost.evil",
            "http://user:secret@localhost",
            "http://127.0.0.1/path",
            "http://localhost?q=a",
        ] {
            assert!(
                install(
                    Harness::Claude,
                    std::slice::from_ref(&path),
                    root,
                    "route",
                    false
                )
                .is_err()
            );
            assert_eq!(read(&path).expect("read"), "{}");
        }
        assert!(!backup_path(&path).exists());
    }
    #[test]
    fn pi_preflights_both_files() {
        let dir = tempfile::tempdir().expect("directory");
        let files = vec![
            dir.path().join("models.json"),
            dir.path().join("settings.json"),
        ];
        std::fs::write(&files[0], "{}").expect("models");
        std::fs::write(&files[1], "[]").expect("settings");
        assert!(install(Harness::Pi, &files, "http://localhost:4123", "route", false).is_err());
        assert_eq!(read(&files[0]).expect("read"), "{}");
        assert!(!backup_path(&files[0]).exists());
    }
    #[test]
    fn malformed_codex_provider_tables_do_not_panic_or_write() {
        let dir = tempfile::tempdir().expect("directory");
        let path = dir.path().join("config.toml");
        for text in [
            "model_providers = 1",
            "model_providers = []",
            "[model_providers]\nsy = false",
            "[[model_providers.sy]]\nname = 'old'",
        ] {
            std::fs::write(&path, text).expect("settings");
            assert!(
                install(
                    Harness::CodexCli,
                    std::slice::from_ref(&path),
                    "http://localhost:4123",
                    "route",
                    false
                )
                .is_err()
            );
            assert_eq!(read(&path).expect("read"), text);
            assert!(!backup_path(&path).exists());
        }
        std::fs::write(
            &path,
            "model_providers = { sy = { name = 'old', env_key = 'OLD_KEY' } }\n",
        )
        .expect("settings");
        install(
            Harness::CodexCli,
            std::slice::from_ref(&path),
            "http://localhost:4123",
            "route",
            false,
        )
        .expect("inline table");
        let doc = read(&path)
            .expect("read")
            .parse::<DocumentMut>()
            .expect("parse");
        assert_eq!(
            doc["model_providers"]["sy"]["base_url"].as_str(),
            Some("http://localhost:4123/v1")
        );
        assert!(doc["model_providers"]["sy"].get("env_key").is_none());
    }

    #[test]
    fn inspection_omits_url_secrets_and_invalid_toml_source() {
        let dir = tempfile::tempdir().expect("directory");
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "model_provider = 'p'\n[model_providers.p]\nbase_url = 'https://user:SECRET@example.com:8443/v1?key=SECRET'\n").expect("settings");
        let info = inspect(Harness::CodexCli, std::slice::from_ref(&path)).expect("inspect");
        assert!(info.contains("https://example.com:8443"));
        assert!(!info.contains("SECRET"));
        std::fs::write(&path, "api_key = \"SECRET\n").expect("invalid");
        let error = inspect(Harness::CodexCli, std::slice::from_ref(&path)).expect_err("invalid");
        assert!(!error.contains("SECRET"));
    }

    #[test]
    fn replacement_prepares_every_file_and_rejects_changed_inputs() {
        let dir = tempfile::tempdir().expect("directory");
        let first = dir.path().join("first");
        let blocked = dir.path().join("blocked");
        std::fs::write(&first, "original").expect("first");
        std::fs::write(&blocked, "not a directory").expect("blocked");
        let files = vec![first.clone(), blocked.join("second")];
        assert!(
            replace(
                &files,
                &[Some("original".into()), None],
                &[Some("new".into()), Some("new".into())]
            )
            .is_err()
        );
        assert_eq!(read(&first).expect("read"), "original");
        assert!(
            replace(
                std::slice::from_ref(&first),
                &[Some("stale".into())],
                &[Some("new".into())]
            )
            .is_err()
        );
        assert_eq!(read(&first).expect("read"), "original");
    }

    #[cfg(unix)]
    #[test]
    fn restore_preflights_symlinks_before_changing_any_settings() {
        let dir = tempfile::tempdir().expect("directory");
        let files = vec![
            dir.path().join("models.json"),
            dir.path().join("settings.json"),
        ];
        for path in &files {
            std::fs::write(path, "{}").expect("original");
        }
        install(Harness::Pi, &files, "http://localhost:4123", "route", false).expect("install");
        let before = read(&files[0]).expect("read");
        std::fs::remove_file(&files[1]).expect("remove");
        let target = dir.path().join("target");
        std::fs::write(&target, "private").expect("target");
        std::os::unix::fs::symlink(&target, &files[1]).expect("symlink");
        assert!(restore(&files).is_err());
        assert!(files.iter().all(|path| backup_path(path).is_file()));
        assert_eq!(read(&files[0]).expect("read"), before);
        assert_eq!(std::fs::read_to_string(target).expect("target"), "private");
    }
}
