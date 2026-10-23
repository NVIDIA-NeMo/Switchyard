// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module installs and restores user settings for the supported harnesses.

use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};
use toml_edit::{DocumentMut, value};

pub type ReviewedFile = (PathBuf, Option<std::sync::Arc<[u8]>>);

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Deserialize, serde::Serialize,
)]
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

/// This function uses the selected harness's environment variable when it is present.
/// This function resolves the home directory only when that variable is absent.
pub fn paths(tool: Harness) -> Result<Vec<PathBuf>, String> {
    let (variable, default, files): (&str, &str, &[&str]) = match tool {
        Harness::CodexCli => ("CODEX_HOME", "~/.codex", &["sy.config.toml"]),
        Harness::CodexApp => ("CODEX_HOME", "~/.codex", &["config.toml"]),
        Harness::Claude => ("CLAUDE_CONFIG_DIR", "~/.claude", &["settings.json"]),
        Harness::Pi => (
            "PI_CODING_AGENT_DIR",
            "~/.pi/agent",
            &["models.json", "settings.json"],
        ),
    };
    let directory = match std::env::var_os(variable) {
        Some(path) => PathBuf::from(path),
        None => crate::config::expand_home(Path::new(default))?,
    };
    Ok(files.iter().map(|file| directory.join(file)).collect())
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
        return Err(
            "Pi's custom provider needs a route with server API credentials. \
             Use Codex or Claude for subscription login forwarding."
                .into(),
        );
    }
    // validate returns the parsed URL without a trailing slash so callers can append API paths.
    Ok(url.to_string().trim_end_matches('/').to_owned())
}

fn codex_settings(
    text: &str,
    root: &str,
    model: &str,
    context_window: Option<u64>,
    login: bool,
) -> Result<String, String> {
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
    // Codex does not know a route ID, so it compacts only when the settings name the
    // route's window. A window kept from another route would compact at the wrong size.
    match context_window.and_then(|window| i64::try_from(window).ok()) {
        Some(window) => doc["model_context_window"] = value(window),
        None => {
            doc.remove("model_context_window");
        }
    }
    doc["model_providers"]["sy"]["name"] = value("Switchyard");
    doc["model_providers"]["sy"]["base_url"] = value(format!("{root}/v1"));
    doc["model_providers"]["sy"]["wire_api"] = value("responses");
    doc["model_providers"]["sy"]["requires_openai_auth"] = value(login);
    // A stale env_key would make Codex require a local API key even when the
    // server owns the credentials or forwards the harness’s subscription login.
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
                    "{key} overrides subscription login. \
                     Remove it from Claude settings before installing this route."
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
    context_window: Option<u64>,
    login: bool,
) -> Result<Value, String> {
    let root = validate(tool, root, model, login)?;
    for path in files {
        if read_file(&restore_pending_path(path))?.is_some() {
            return Err("A previous restore still needs backup cleanup. \
                        Retry Restore before connecting another route."
                .into());
        }
    }
    let originals = files
        .iter()
        .map(|path| read_file(path))
        .collect::<Result<Vec<_>, _>>()?;
    let incoming = configured(tool, &originals, &root, model, context_window, login)?;
    let current = originals
        .iter()
        .map(|text| text.clone().unwrap_or_default())
        .collect::<Vec<_>>();
    Ok(json!({
        "changes": settings_changes(tool, files, &current, &incoming)?,
        "current": inspect_settings(tool, files, &current)?,
        "proposed": inspect_settings(tool, files, &incoming)?,
        "authentication": if login {
            "This route uses the harness’s subscription login."
        } else {
            "This route uses the API credentials configured on the Switchyard server."
        },
        "files": files.iter().map(|path| path.display().to_string()).collect::<Vec<_>>(),
        "file_actions": files.iter().zip(originals.iter().zip(&incoming)).map(|(path,(before,after))|json!({"file":path,"action":if before.as_deref()==Some(after.as_str()) {"preserve"} else if before.is_none() {"create"} else {"update"}})).collect::<Vec<_>>()
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
                "/model_context_window",
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
                        Value::Number(value) => value.to_string(),
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
    context_window: Option<u64>,
    login: bool,
) -> Result<Vec<String>, String> {
    let mut incoming = Vec::new();
    match tool {
        Harness::CodexCli | Harness::CodexApp => incoming.push(codex_settings(
            originals[0].as_deref().unwrap_or_default(),
            root,
            model,
            context_window,
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
            providers["switchyard"] = json!({"baseUrl":format!("{root}/v1"), "api":"openai-completions", "apiKey":"switchyard-local", "headers":{"x-switchyard-origin":"pi"}, "models":[{"id":model,"name":model,"reasoning":true,"input":["text"],"contextWindow":context_window.unwrap_or(128000),"maxTokens":16384,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0}}]});
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

pub fn install_reviewed(
    tool: Harness,
    files: &[PathBuf],
    root: &str,
    model: &str,
    context_window: Option<u64>,
    login: bool,
    expected: Option<&[ReviewedFile]>,
) -> Result<String, String> {
    let root = validate(tool, root, model, login)?;
    for path in files {
        if read_file(&restore_pending_path(path))?.is_some() {
            return Err("A previous restore did not finish retiring its backups. \
                        Retry Restore before installing another route."
                .into());
        }
    }
    let originals = files
        .iter()
        .map(|p| read_file(p))
        .collect::<Result<Vec<_>, _>>()?;
    check_review(expected)?;
    let incoming = configured(tool, &originals, &root, model, context_window, login)?;
    let incoming: Vec<_> = incoming.into_iter().map(Some).collect();
    for (path, original) in files.iter().zip(&originals) {
        if let Some(expected) = expected
            .and_then(|files| files.iter().find(|(file, _)| file == path))
            .map(|(_, bytes)| bytes)
            && original.as_ref().map(|text| text.as_bytes()) != expected.as_deref()
        {
            return Err("Harness settings changed after the preview. Review them again.".into());
        }
    }
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
    let next = match tool {
        Harness::CodexCli => {
            let profile = files[0]
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.strip_suffix(".config.toml"));
            profile
                .map(|name| format!(" Start Codex CLI with codex -p {name}."))
                .unwrap_or_else(|| " Restart Codex CLI.".into())
        }
        Harness::Pi => " Reload Pi models with /model.".into(),
        Harness::CodexApp => " Restart Codex app.".into(),
        Harness::Claude => " Restart Claude Code.".into(),
    };
    Ok(format!(
        "Connected route {model}. Changed {}.{next}",
        files
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
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

pub fn review_files(files: &[PathBuf]) -> Vec<PathBuf> {
    files
        .iter()
        .flat_map(|path| {
            [
                path.clone(),
                backup_path(path),
                absent_path(path),
                restore_pending_path(path),
            ]
        })
        .collect()
}

pub fn preview_restore(tool: Harness, files: &[PathBuf]) -> Result<Value, String> {
    let resuming = files.iter().any(|path| restore_pending_path(path).exists());
    let mut plan = Vec::new();
    let mut before = Vec::new();
    let mut after = Vec::new();
    for path in files {
        let current = read_file(path)?;
        let pending = read_file(&restore_pending_path(path))?;
        let original = match pending {
            Some(text) => {
                serde_json::from_str::<Option<String>>(&text).map_err(|error| error.to_string())?
            }
            None => match read_file(&backup_path(path))? {
                Some(text) => Some(text),
                None if read_file(&absent_path(path))?.is_some() => None,
                None if resuming => current.clone(),
                None => return Err("No original backup is available for this destination.".into()),
            },
        };
        let action = if current == original {
            "unchanged"
        } else {
            match (&current, &original) {
                (_, None) => "remove",
                (None, _) => "create",
                _ => "replace",
            }
        };
        before.push(current.unwrap_or_default());
        after.push(original.unwrap_or_default());
        plan.push(json!({"file":path,"action":action,"backup":backup_path(path)}));
    }
    Ok(json!({
        "files": plan,
        "changes": settings_changes(tool, files, &before, &after).ok(),
        "warning": "Restore replaces each listed file with its original version. \
                    Later edits are kept in a recovery copy before replacement."
    }))
}

pub fn restore_reviewed(
    files: &[PathBuf],
    expected: Option<&[ReviewedFile]>,
) -> Result<String, String> {
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
    check_review(expected)?;
    for (path, original) in files.iter().zip(&current) {
        if let Some(expected) = expected
            .and_then(|files| files.iter().find(|(file, _)| file == path))
            .map(|(_, bytes)| bytes)
            && original.as_ref().map(|text| text.as_bytes()) != expected.as_deref()
        {
            return Err(
                "Harness settings changed after the restore preview. Review them again.".into(),
            );
        }
    }
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
                        "Restored settings, but could not retire {}: {error}. \
                         Retry Restore before installing another route. {recovery}",
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
        "Restored original settings. Restart the harness. {recovery}"
    ))
}

fn check_review(expected: Option<&[ReviewedFile]>) -> Result<(), String> {
    if let Some(files) = expected {
        for (path, expected) in files {
            let current = match std::fs::read(path) {
                Ok(bytes) => Some(std::sync::Arc::<[u8]>::from(bytes)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(error.to_string()),
            };
            if &current != expected {
                return Err(format!(
                    "{} changed after the preview. Review it again.",
                    path.display()
                ));
            }
        }
    }
    Ok(())
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
        #[cfg(windows)]
        return crate::windows::codex_app();
        #[cfg(not(windows))]
        {
            #[cfg(target_os = "macos")]
            if let Some(path) = objc2_app_kit::NSWorkspace::sharedWorkspace()
                .URLForApplicationWithBundleIdentifier(&objc2_foundation::NSString::from_str(
                    "com.openai.codex",
                ))
                .and_then(|url| url.path())
                .map(|path| PathBuf::from(path.to_string()))
                .filter(|path| path.is_dir())
            {
                return Some(path);
            }
            return [
                Some(PathBuf::from("/Applications/Codex.app")),
                crate::config::expand_home(Path::new("~/Applications/Codex.app")).ok(),
            ]
            .into_iter()
            .flatten()
            .find(|p| p.exists());
        }
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
    ]);
    dirs.extend(
        [
            crate::config::expand_home(Path::new("~/.local/bin")).ok(),
            crate::config::expand_home(Path::new("~/.npm-global/bin")).ok(),
        ]
        .into_iter()
        .flatten(),
    );
    #[cfg(windows)]
    {
        dirs.extend(
            [
                crate::config::expand_home(Path::new("~/AppData/Roaming/npm")).ok(),
                crate::config::expand_home(Path::new("~/.cargo/bin")).ok(),
                crate::config::expand_home(Path::new("~/.local/bin")).ok(),
            ]
            .into_iter()
            .flatten(),
        );
        dirs.into_iter()
            .flat_map(|p| [p.join(format!("{name}.exe")), p.join(format!("{name}.cmd"))])
            .find(|p| p.is_file())
    }
    #[cfg(not(windows))]
    dirs.into_iter().map(|p| p.join(name)).find(|p| p.is_file())
}
