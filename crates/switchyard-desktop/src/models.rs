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

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::config::expand_home;
use crate::server_config::Client;

/// CACHE_FILE names the cache file. It sits next to the settings file that the
/// app was started with.
pub const CACHE_FILE: &str = "model-lists.json";

/// KEYCHAIN_SERVICE identifies keys saved from Routes. Each
/// item's account is the client's `base_url`, so clients that share a
/// gateway share its key.
#[cfg(target_os = "macos")]
const KEYCHAIN_SERVICE: &str = "Switchyard model list";

/// TIMEOUT_SECONDS limits how long curl may spend listing models.
const TIMEOUT_SECONDS: &str = "20";

/// CACHE_WRITE stays locked while a thread rewrites the cache file, so threads that
/// finish at the same time do not drop each other's lists.
static CACHE_WRITE: Mutex<()> = Mutex::new(());

/// ModelList stores one fetched model list in the cache file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelList {
    /// This field stores sorted model IDs.
    pub models: Vec<String>,
    /// This field records the fetch time in seconds since 1970-01-01 UTC.
    pub fetched_at: u64,
}

/// Loaded stores the model list and fetch result for one URL.
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded {
    /// This field identifies an endpoint list as `endpoint` or a Switchyard cache list as `cache`.
    /// A Codex cache list has the source `local`; `error` separately reports fetch or cache-write failures.
    pub source: &'static str,
    /// This field names the URL that serves the list.
    pub url: String,
    /// This field stores the newest list fetched or read from the cache file.
    pub list: Option<ModelList>,
    /// This field records a fetch or cache-write error, if either occurred.
    pub error: Option<ListError>,
}

/// ListError describes why a client's models could not be listed.
#[derive(Clone, Debug, PartialEq)]
pub enum ListError {
    /// The client needs a key, and none is set or saved.
    NoKey,
    /// The client reads its key from this `api_key_env` variable, which this
    /// process does not have, and no key is saved.
    NoEnv(String),
    /// The platform's credential store did not return the saved key.
    Keychain(String),
    /// The models endpoint answered 401 or 403.
    Rejected(u16),
    /// The request failed, or the answer was not a model list.
    Failed(String),
    /// The list was fetched, but the cache file could not be written.
    NotCached(String),
}

impl ListError {
    /// This function says what went wrong, in one sentence.
    pub fn reason(&self) -> String {
        match self {
            Self::NoKey => {
                "This endpoint needs an API key to list models, and the app does not have one."
                    .to_string()
            }
            Self::NoEnv(variable) => missing_env_note(variable),
            Self::Keychain(error) => {
                format!("Could not read the saved key from secure credential storage: {error}.")
            }
            Self::Rejected(status) => {
                format!("The models endpoint rejected the key (HTTP {status}).")
            }
            Self::Failed(error) => format!("Could not list models: {error}."),
            Self::NotCached(error) => format!("Could not save the list: {error}."),
        }
    }

    /// This function says what the user can do while the window has no list to show. The
    /// window accepts a typed model ID, so most answers offer that. A missing
    /// variable also stops Apply, so that answer offers only the key field.
    pub fn advice(&self) -> &'static str {
        match self {
            Self::NoEnv(_) => "Or paste the key below to list models.",
            _ if self.needs_key() => {
                "Enter a valid API key and choose Save key and load models, or type a model ID."
            }
            _ => "Click Refresh models to try again, or type a model ID.",
        }
    }

    /// This function returns whether saving a key could fix this error.
    pub fn needs_key(&self) -> bool {
        matches!(
            self,
            Self::NoKey | Self::NoEnv(_) | Self::Keychain(_) | Self::Rejected(_)
        )
    }
}

/// This function says that this process lacks a client's `api_key_env` variable. Apply's
/// `--dry-run` check runs in this process's environment, so the check cannot
/// read the key either.
pub fn missing_env_note(variable: &str) -> String {
    format!(
        "The app's environment has no {variable}, which Apply needs: add it to the desktop \
         app's LaunchAgent."
    )
}

/// This function returns the client's `api_key_env` variable when this process does not
/// have it set.
pub fn missing_env(client: &Client) -> Option<&str> {
    client
        .api_key_env
        .as_deref()
        .filter(|_| env_key(client).is_none())
}

/// This function loads one endpoint's models and retains the cached list when refresh fails.
/// The selected endpoint alone receives the entered key.
pub fn load(cache: &Path, client: &Client, refresh: bool, typed_key: Option<&str>) -> Loaded {
    let url = list_url(client);
    let old = read_cache(cache).remove(&url);
    if unlisted(client) {
        // ChatGPT has no list to fetch, so nothing goes in the cache file.
        return Loaded {
            source: "local",
            url,
            list: codex_models(),
            error: None,
        };
    }
    if old.is_some() && !refresh {
        return Loaded {
            source: "cache",
            url,
            list: old,
            error: None,
        };
    }
    match key(client, typed_key).and_then(|key| list(client, key.as_deref())) {
        Ok(models) => {
            let new = ModelList {
                models,
                fetched_at: now(),
            };
            let error = write_cache(cache, &url, &new)
                .err()
                .map(ListError::NotCached);
            Loaded {
                source: "endpoint",
                url,
                list: Some(new),
                error,
            }
        }
        Err(error) => Loaded {
            source: "cache",
            url,
            list: old,
            error: Some(error),
        },
    }
}

/// This function returns the URL that serves the client's model list. It is also the
/// list's key in the cache file and in [`Loaded::url`].
pub fn list_url(client: &Client) -> String {
    models_url(&client.format, &client.base_url)
}

/// This function returns whether the client's endpoint has no model list that this app can
/// ask for. The ChatGPT Codex endpoint answers only a ChatGPT login, never an
/// API key, and the app does not hold the login.
pub fn unlisted(client: &Client) -> bool {
    client.host() == "chatgpt.com"
}

/// This function reads the model list that Codex saved in `$CODEX_HOME/models_cache.json`,
/// or in `~/.codex/models_cache.json` when `CODEX_HOME` is unset. Codex
/// fetched that list with the user's ChatGPT login. For a ChatGPT client this
/// file is the only list of its models that the app can read.
fn codex_models() -> Option<ModelList> {
    let home = std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| expand_home(Path::new("~/.codex")));
    read_codex_models(&home.join("models_cache.json"))
}

/// This function parses a Codex model list file. Returns every model's `slug`, sorted and
/// without duplicates, and the time Codex fetched the file.
fn read_codex_models(path: &Path) -> Option<ModelList> {
    #[derive(Deserialize)]
    struct Saved {
        fetched_at: String,
        models: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        slug: String,
    }

    let saved: Saved = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let fetched_at = chrono::DateTime::parse_from_rfc3339(&saved.fetched_at).ok()?;
    let mut models: Vec<String> = saved.models.into_iter().map(|entry| entry.slug).collect();
    models.sort();
    models.dedup();
    Some(ModelList {
        models,
        fetched_at: u64::try_from(fetched_at.timestamp()).ok()?,
    })
}

/// This function lists the client's model IDs, sorted. When the client needs a key and
/// `key` is `None`, returns [`ListError::NoEnv`] or [`ListError::NoKey`]
/// without sending a request.
fn list(client: &Client, key: Option<&str>) -> Result<Vec<String>, ListError> {
    match (key, &client.api_key_env) {
        (None, Some(variable)) => Err(ListError::NoEnv(variable.clone())),
        (None, None) if client.forward_auth => Err(ListError::NoKey),
        _ => fetch(&list_url(client), &client.format, key),
    }
}

/// This function returns the models endpoint for a client. It builds the URL the way the
/// server builds its request URLs. OpenAI clients drop a trailing
/// `/chat/completions` or `/responses`. Anthropic clients drop a trailing
/// `/v1/messages` to `/v1`, and add `/v1` to any other path.
/// Anthropic returns 20 models unless the URL sets `limit`, so the URL adds
/// `limit=1000`, the most that Anthropic allows.
fn models_url(format: &str, base_url: &str) -> String {
    let (base, query) = match base_url.split_once('?') {
        Some((base, query)) => (base, format!("?{query}")),
        None => (base_url, String::new()),
    };
    let base = base.trim_end_matches('/');
    let path = if format == "anthropic_messages" {
        let root = base
            .strip_suffix("/messages")
            .filter(|root| root.ends_with("/v1"))
            .unwrap_or(base);
        if root.ends_with("/v1") {
            format!("{root}/models")
        } else {
            format!("{root}/v1/models")
        }
    } else {
        let root = base
            .strip_suffix("/chat/completions")
            .or_else(|| base.strip_suffix("/responses"))
            .unwrap_or(base);
        format!("{root}/models")
    };
    let query = if format == "anthropic_messages" && query.is_empty() {
        "?limit=1000".to_string()
    } else if format == "anthropic_messages" {
        format!("{query}&limit=1000")
    } else {
        query
    };
    format!("{path}{query}")
}

/// This function returns whether the server sends this client a key. If it does, listing
/// the client's models needs a key too.
fn sends_key(client: &Client) -> bool {
    client.forward_auth || client.api_key_env.is_some()
}

/// This function returns the key to list the client's models with, or `None` when the
/// client needs no key or none is available.
///
/// The entered key takes precedence over the environment and saved credentials.
fn key(client: &Client, typed_key: Option<&str>) -> Result<Option<String>, ListError> {
    if !sends_key(client) {
        return Ok(None);
    }
    if let Some(key) = typed_key.map(str::to_string).or_else(|| env_key(client)) {
        return Ok(Some(key));
    }
    saved_key(&client.base_url).map_err(ListError::Keychain)
}

fn env_key(client: &Client) -> Option<String> {
    client
        .api_key_env
        .as_deref()
        .and_then(|variable| std::env::var(variable).ok())
        .filter(|key| !key.trim().is_empty())
}

/// This function returns the key saved in the Keychain for `base_url`, or `None` when no
/// key is saved.
#[cfg(target_os = "macos")]
fn saved_key(base_url: &str) -> Result<Option<String>, String> {
    /// NOT_FOUND matches the Keychain's `errSecItemNotFound` status.
    const NOT_FOUND: i32 = -25300;
    match security_framework::passwords::get_generic_password(KEYCHAIN_SERVICE, base_url) {
        Ok(bytes) => String::from_utf8(bytes)
            .map(|key| Some(key).filter(|key| !key.trim().is_empty()))
            .map_err(|_| "the saved key is not text".to_string()),
        Err(error) if error.code() == NOT_FOUND => Ok(None),
        Err(error) => Err(keychain_error(&error)),
    }
}

#[cfg(not(any(target_os = "macos", windows)))]
fn saved_key(_base_url: &str) -> Result<Option<String>, String> {
    Ok(None)
}

/// This function saves a key in the login Keychain for every client with this `base_url`.
#[cfg(target_os = "macos")]
pub fn save_key(base_url: &str, key: &str) -> Result<(), String> {
    security_framework::passwords::set_generic_password(
        KEYCHAIN_SERVICE,
        base_url,
        key.trim().as_bytes(),
    )
    .map_err(|error| keychain_error(&error))
}

#[cfg(not(any(target_os = "macos", windows)))]
/// This function rejects saved keys on platforms without Keychain or Credential Manager support.
pub fn save_key(_base_url: &str, _key: &str) -> Result<(), String> {
    Err("Saving model-list keys requires macOS Keychain or Windows Credential Manager. On this platform, set the environment variable named by api_key_env and make it available to the app.".into())
}

#[cfg(windows)]
/// This function reads a saved key for this exact base_url from Windows Credential Manager.
fn saved_key(base_url: &str) -> Result<Option<String>, String> {
    switchyard_desktop_install::windows::saved_key(base_url)
}
#[cfg(windows)]
/// This function saves the key for this exact base_url in Windows Credential Manager.
pub fn save_key(base_url: &str, key: &str) -> Result<(), String> {
    switchyard_desktop_install::windows::save_key(base_url, key)
}

/// This function returns the Keychain's message without its final period, so the window
/// can end its own sentence after it.
#[cfg(target_os = "macos")]
fn keychain_error(error: &security_framework::base::Error) -> String {
    error.to_string().trim_end_matches('.').to_string()
}

/// This function returns whether a key has a line break, which would end the header that
/// carries the key and start another one.
pub fn has_line_break(key: &str) -> bool {
    key.contains(['\r', '\n'])
}

/// This function reads the cache file. A missing or unreadable file reads as empty, so the
/// app fetches its lists again and writes a new file.
fn read_cache(path: &Path) -> BTreeMap<String, ModelList> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// This function adds the list at `url` to the cache file. The list replaces the one at
/// the same URL unless that one is newer: two loads can overlap, and the
/// load that fetched a list first can write it last. The text goes to a
/// temporary file that is renamed over the cache file, so a reader never
/// sees half a file.
fn write_cache(path: &Path, url: &str, list: &ModelList) -> Result<(), String> {
    let _writing = CACHE_WRITE.lock().unwrap_or_else(PoisonError::into_inner);
    let mut all = read_cache(path);
    if all
        .get(url)
        .is_some_and(|cached| cached.fetched_at > list.fetched_at)
    {
        return Ok(());
    }
    all.insert(url.to_string(), list.clone());
    let text = serde_json::to_string_pretty(&all)
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    let dir = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(dir)
        .map_err(|error| format!("create a file in {}: {error}", dir.display()))?;
    file.write_all(text.as_bytes())
        .and_then(|()| file.as_file().sync_all())
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    file.persist(path)
        .map_err(|error| format!("replace {}: {}", path.display(), error.error))?;
    Ok(())
}

/// This function returns the current time in seconds since 1970-01-01 UTC.
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

fn fetch(url: &str, format: &str, key: Option<&str>) -> Result<Vec<String>, ListError> {
    let failed = |error: String| ListError::Failed(error);
    if key.is_some_and(has_line_break) {
        return Err(failed(
            "the key has a line break, so the app did not send it".to_string(),
        ));
    }
    let mut command = Command::new("curl");
    #[cfg(windows)]
    switchyard_desktop_install::windows::hide_console(&mut command);
    let mut child = command
        // curl honors `-q` only as its first argument. `-q` makes curl ignore
        // ~/.curlrc, where a `verbose` line would print the key into the
        // error text that the window shows.
        .args([
            "-q",
            "--silent",
            "--show-error",
            "--max-time",
            TIMEOUT_SECONDS,
        ])
        // curl reads request headers from stdin, one per line.
        .args(["--header", "@-"])
        // `--url` keeps a URL that starts with `-` from being read as an
        // option.
        .args(["--write-out", "\n%{http_code}", "--url", url])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| failed(format!("run curl: {error}")))?;
    if let Some(mut stdin) = child.stdin.take() {
        let headers = match (key, format) {
            (Some(key), "anthropic_messages") => {
                format!("x-api-key: {key}\nanthropic-version: 2023-06-01\n")
            }
            (Some(key), _) => format!("Authorization: Bearer {key}\n"),
            (None, _) => String::new(),
        };
        let sent = stdin.write_all(headers.as_bytes());
        drop(stdin);
        if let Err(error) = sent {
            let _ = child.kill();
            let _ = child.wait();
            return Err(failed(format!("send headers to curl: {error}")));
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|error| failed(format!("run curl: {error}")))?;
    if !output.status.success() {
        return Err(failed(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let (body, status) = stdout.rsplit_once('\n').unwrap_or(("", &stdout));
    match status.trim().parse::<u16>() {
        Ok(200) => parse_ids(body).map_err(failed),
        Ok(status @ (401 | 403)) => Err(ListError::Rejected(status)),
        Ok(status) => Err(failed(format!("{url} answered HTTP {status}"))),
        Err(_) => Err(failed(format!("{url} sent no HTTP status"))),
    }
}

/// This function parses the model IDs. OpenAI and Anthropic both answer with
/// `{"data": [{"id": ...}, ...]}`.
fn parse_ids(body: &str) -> Result<Vec<String>, String> {
    #[derive(Deserialize)]
    struct Listing {
        data: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        id: String,
    }

    let listing: Listing = serde_json::from_str(body)
        .map_err(|error| format!("the model list is not the expected JSON: {error}"))?;
    let mut ids: Vec<String> = listing.data.into_iter().map(|entry| entry.id).collect();
    ids.sort();
    ids.dedup();
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    /// Stub returns the configured response and records each request header.
    /// Drop wakes its listener and joins the worker so a failed assertion cannot leave it running.
    struct Stub {
        /// This field sets the client base URL used to list models from this server.
        url: String,
        requests: Arc<Mutex<Vec<Vec<String>>>>,
        address: SocketAddr,
        stopping: Arc<AtomicBool>,
        worker: Option<std::thread::JoinHandle<()>>,
    }

    impl Stub {
        fn start(response: String) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let address = listener.local_addr().expect("addr");
            let url = format!("http://{address}/v1");
            let requests = Arc::new(Mutex::new(Vec::new()));
            let stopping = Arc::new(AtomicBool::new(false));
            let stop = Arc::clone(&stopping);
            let record = Arc::clone(&requests);
            let worker = std::thread::spawn(move || {
                for mut stream in listener.incoming().map_while(Result::ok) {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let mut head = Vec::new();
                    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
                    let mut line = String::new();
                    while reader.read_line(&mut line).is_ok_and(|read| read > 0) && line != "\r\n" {
                        head.push(line.trim_end().to_string());
                        line.clear();
                    }
                    record.lock().expect("lock").push(head);
                    let _ = stream.write_all(response.as_bytes());
                }
            });
            Self {
                url,
                requests,
                address,
                stopping,
                worker: Some(worker),
            }
        }

        fn requests(&self) -> Vec<Vec<String>> {
            self.requests.lock().expect("lock").clone()
        }
    }

    impl Drop for Stub {
        fn drop(&mut self) {
            self.stopping.store(true, Ordering::Release);
            let _ = TcpStream::connect(self.address);
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
    }

    fn listing(models: &[&str]) -> String {
        let data: Vec<_> = models
            .iter()
            .map(|id| serde_json::json!({ "id": id }))
            .collect();
        format!(
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{}",
            serde_json::json!({ "data": data })
        )
    }

    fn client(name: &str, format: &str, base_url: &str) -> Client {
        Client {
            name: name.to_string(),
            format: format.to_string(),
            base_url: base_url.to_string(),
            api_key_env: None,
            forward_auth: true,
        }
    }

    // The subprocess isolates PATH; its curl fixture exits before reading the oversized header.
    #[cfg(unix)]
    #[test]
    fn failed_header_write_reaps_the_curl_process() {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(record) = std::env::var("SWITCHYARD_CURL_FIXTURE") {
            let error = fetch(
                "http://localhost/models",
                "openai_chat",
                Some(&"k".repeat(1024 * 1024)),
            )
            .expect_err("header write fails");
            assert!(
                matches!(error, ListError::Failed(message) if message.starts_with("send headers to curl:"))
            );
            let pid = std::fs::read_to_string(record).expect("curl PID");
            assert!(
                !Command::new("/bin/kill")
                    .args(["-0", pid.trim()])
                    .status()
                    .expect("process status")
                    .success()
            );
            return;
        }
        let dir = tempfile::tempdir().expect("fixture");
        let record = dir.path().join("pid");
        let curl = dir.path().join("curl");
        std::fs::write(
            &curl,
            "#!/bin/sh\nprintf '%s' \"$$\" > \"$SWITCHYARD_CURL_FIXTURE\"\nexit 1\n",
        )
        .expect("curl fixture");
        std::fs::set_permissions(&curl, std::fs::Permissions::from_mode(0o700))
            .expect("permissions");
        let output = Command::new(std::env::current_exe().expect("runner"))
            .args([
                "--exact",
                "models::tests::failed_header_write_reaps_the_curl_process",
            ])
            .env("SWITCHYARD_CURL_FIXTURE", &record)
            .env("PATH", dir.path())
            .output()
            .expect("child");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    #[test]
    fn entered_key_is_sent_only_to_its_selected_base_url() {
        let selected = Stub::start(listing(&["selected"]));
        let unrelated = Stub::start(listing(&["unrelated"]));
        let unauthenticated = Stub::start(listing(&["public"]));
        let dir = tempfile::tempdir().expect("fixture");
        let clients = [
            client("selected", "openai_chat", &selected.url),
            Client {
                api_key_env: Some("SWITCHYARD_MENUBAR_TEST_UNSET_KEY".into()),
                ..client("unrelated", "openai_chat", &unrelated.url)
            },
            Client {
                forward_auth: false,
                ..client("public", "openai_chat", &unauthenticated.url)
            },
        ];
        let loaded: Vec<_> = clients
            .iter()
            .enumerate()
            .map(|(index, client)| {
                load(
                    &dir.path().join(CACHE_FILE),
                    client,
                    true,
                    (index == 0).then_some("selected-secret"),
                )
            })
            .collect();
        assert_eq!(loaded[0].error, None);
        assert_eq!(loaded[0].source, "endpoint");
        assert!(
            !std::fs::read_to_string(dir.path().join(CACHE_FILE))
                .expect("cache")
                .contains("selected-secret")
        );
        assert_eq!(selected.requests().len(), 1);
        assert!(selected.requests()[0].contains(&"Authorization: Bearer selected-secret".into()));
        assert_eq!(
            loaded[1].error,
            Some(ListError::NoEnv("SWITCHYARD_MENUBAR_TEST_UNSET_KEY".into()))
        );
        assert!(unrelated.requests().is_empty());
        assert_eq!(loaded[2].error, None);
        assert_eq!(loaded[2].source, "endpoint");
        assert_eq!(unauthenticated.requests().len(), 1);
        assert!(
            unauthenticated.requests()[0]
                .iter()
                .all(|header| !header.starts_with("Authorization:"))
        );
    }

    #[test]
    fn reports_a_rejected_key_and_a_missing_key() {
        let stub =
            Stub::start("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n{}".to_string());
        let gateway = client("gateway", "openai_chat", &stub.url);
        let from_env = Client {
            api_key_env: Some("SWITCHYARD_MENUBAR_TEST_UNSET_KEY".to_string()),
            forward_auth: false,
            ..gateway.clone()
        };

        assert_eq!(list(&gateway, Some("wrong")), Err(ListError::Rejected(401)));
        assert_eq!(list(&gateway, None), Err(ListError::NoKey));
        assert_eq!(
            list(&from_env, None),
            Err(ListError::NoEnv(
                "SWITCHYARD_MENUBAR_TEST_UNSET_KEY".to_string()
            ))
        );
        assert!(matches!(
            list(&gateway, Some("key\r\nX-Injected: yes")),
            Err(ListError::Failed(_))
        ));
        assert_eq!(
            stub.requests().len(),
            1,
            "a missing key or a key with a line break sends no request"
        );
    }
}
