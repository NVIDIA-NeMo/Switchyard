// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module lists the models an LLM client offers from its `GET /models` endpoint,
//! and keeps each fetched list in a cache file.
//!
//! The request goes through the system `curl`, the same way the app runs
//! `launchctl`, so the app needs no HTTP or TLS stack of its own. The key is
//! written to curl's stdin, never to its command line, where other processes
//! could read it.
//!
//! The cache file holds model IDs and fetch times, never a key. The app uses
//! a list from the cache file until the user refreshes it, so it fetches a
//! list on its own only when the file does not have it.

use crate::server_config::Client;

/// This function explains why model suggestions lack a key. Model-list keys stay separate from
/// the server keys that the installed helper keeps.
pub fn missing_env_note(variable: &str) -> String {
    if cfg!(any(target_os = "macos", windows)) {
        format!(
            "Switchyard does not read {variable} from your shell. \
             To load model suggestions, save a model-list key below. \
             The server's own key is under Settings → Server keys."
        )
    } else {
        format!(
            "{variable} is not set in this terminal, so this endpoint cannot list models. \
             Type the model ID instead."
        )
    }
}

/// This function returns the client's `api_key_env` variable when this process does not
/// have it set.
pub fn missing_env(client: &Client) -> Option<&str> {
    client
        .api_key_env
        .as_deref()
        .filter(|_| env_key(client).is_none())
}

/// This function returns whether the client's endpoint has no model list that this app can
/// ask for. The ChatGPT Codex endpoint answers only a ChatGPT login, never an
/// API key, and the app does not hold the login.
pub fn unlisted(client: &Client) -> bool {
    client.host() == "chatgpt.com"
}

fn env_key(client: &Client) -> Option<String> {
    client
        .api_key_env
        .as_deref()
        .and_then(|variable| std::env::var(variable).ok())
        .filter(|key| !key.trim().is_empty())
}
