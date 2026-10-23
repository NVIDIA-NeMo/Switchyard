// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module installs and restores user settings for the supported harnesses.

use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use toml_edit::DocumentMut;

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
