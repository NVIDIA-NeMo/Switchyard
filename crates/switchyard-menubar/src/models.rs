// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lists the models an LLM client offers, from its `GET /models` endpoint,
//! and keeps each fetched list in a cache file.
//!
//! The request goes through the system `curl`, the same way the menu runs
//! `launchctl`, so the app needs no HTTP or TLS stack of its own. The key is
//! written to curl's stdin, never to its command line, where other processes
//! could read it.
//!
//! The cache file holds model IDs and fetch times, never a key. The app uses
//! a list from the cache file until the user refreshes it, so it fetches a
//! list on its own only when the file does not have it.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::server_config::Client;

/// The name of the cache file. It sits next to the settings file that the
/// app was started with.
pub const CACHE_FILE: &str = "model-lists.json";

/// The Keychain service that holds keys saved from the route picker. Each
/// item's account is the client's `base_url`, so clients that share a
/// gateway share its key.
#[cfg(target_os = "macos")]
const KEYCHAIN_SERVICE: &str = "Switchyard model list";

/// How long one listing may take before curl gives up.
const TIMEOUT_SECONDS: &str = "20";

/// Held while a thread rewrites the cache file, so that two threads that
/// finish at the same time do not drop each other's lists.
static CACHE_WRITE: Mutex<()> = Mutex::new(());

/// One fetched model list, as the cache file stores it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelList {
    /// The model IDs, sorted.
    pub models: Vec<String>,
    /// When the list was fetched, in seconds since 1970-01-01 UTC.
    pub fetched_at: u64,
}

/// The model list at one URL, as [`load`] found it.
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded {
    /// The URL that serves the list.
    pub url: String,
    /// The newest list: fetched just now, or read from the cache file.
    pub list: Option<ModelList>,
    /// Why the list could not be fetched or saved in the cache file, when
    /// that happened.
    pub error: Option<ListError>,
}

/// Why a client's models could not be listed.
#[derive(Clone, Debug, PartialEq)]
pub enum ListError {
    /// The client needs a key, and none is set or saved.
    NoKey,
    /// The client reads its key from this `api_key_env` variable, which this
    /// process does not have, and no key is saved.
    NoEnv(String),
    /// The Keychain did not return the saved key.
    Keychain(String),
    /// The models endpoint answered 401 or 403.
    Rejected(u16),
    /// The request failed, or the answer was not a model list.
    Failed(String),
    /// The list was fetched, but the cache file could not be written.
    NotCached(String),
}

impl ListError {
    /// Says what went wrong, in one sentence.
    pub fn reason(&self) -> String {
        match self {
            Self::NoKey => {
                "This LLM client needs an API key to list models, and the app does not have one."
                    .to_string()
            }
            Self::NoEnv(variable) => missing_env_note(variable),
            Self::Keychain(error) => {
                format!("Could not read the saved key from your login Keychain: {error}.")
            }
            Self::Rejected(status) => {
                format!("The models endpoint rejected the key (HTTP {status}).")
            }
            Self::Failed(error) => format!("Could not list models: {error}."),
            Self::NotCached(error) => format!("Could not save the list: {error}."),
        }
    }

    /// Says what the user can do while the window has no list to show. The
    /// window accepts a typed model ID, so most answers offer that. A missing
    /// variable also stops Apply, so that answer offers only the key field.
    pub fn advice(&self) -> &'static str {
        match self {
            Self::NoEnv(_) => "Or paste the key below to list models.",
            _ if self.needs_key() => {
                "Paste a valid API key below and click Save key, or type a model ID."
            }
            _ => "Click Refresh models to try again, or type a model ID.",
        }
    }

    /// Returns whether saving a key could fix this error.
    pub fn needs_key(&self) -> bool {
        matches!(
            self,
            Self::NoKey | Self::NoEnv(_) | Self::Keychain(_) | Self::Rejected(_)
        )
    }
}

/// Says that this process lacks a client's `api_key_env` variable. Apply's
/// `--dry-run` check runs in this process's environment, so the check cannot
/// read the key either.
pub fn missing_env_note(variable: &str) -> String {
    format!(
        "The app's environment has no {variable}, which Apply needs: add it to the menu bar \
         app's LaunchAgent."
    )
}

/// Returns the client's `api_key_env` variable when this process does not
/// have it set.
pub fn missing_env(client: &Client) -> Option<&str> {
    client
        .api_key_env
        .as_deref()
        .filter(|_| env_key(client).is_none())
}

/// The Keychain's answer for each `base_url`. The threads of one [`load`]
/// share it.
type SavedKeys = Mutex<HashMap<String, Result<Option<String>, String>>>;

/// Returns the model list at each URL that `clients` use, in the order the
/// URLs first appear. Clients whose models share a URL share one list and
/// one request.
///
/// Each URL loads on its own thread, so a slow URL does not hold back the
/// others. `loaded` gets each URL's result as soon as it is ready.
///
/// Without `refresh`, a list in the cache file comes back without a
/// request, and only a URL with no cached list is fetched. With `refresh`,
/// every URL is fetched. A fetched list replaces the one in the cache file
/// unless the file holds a newer list for that URL. When the file cannot be
/// written, the fetched list still comes back, with
/// [`ListError::NotCached`], and the file keeps its old list. When a fetch
/// fails, the cached list comes back with the error.
///
/// `typed_key` is a key the user just entered. When it is set, clients that
/// send a key use it instead of their environment variable or the Keychain.
pub fn load(
    cache: &Path,
    clients: &[Client],
    refresh: bool,
    typed_key: Option<&str>,
    loaded: &(dyn Fn(&Loaded) + Sync),
) -> Vec<Loaded> {
    let cached = read_cache(cache);
    let saved_keys = SavedKeys::default();
    std::thread::scope(|scope| {
        let workers: Vec<_> = by_url(clients)
            .into_iter()
            .map(|(url, client)| {
                let old = cached.get(&url).cloned();
                let saved_keys = &saved_keys;
                scope.spawn(move || {
                    let entry = load_url(cache, url, client, old, refresh, typed_key, saved_keys);
                    loaded(&entry);
                    entry
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    })
}

/// Loads the list at one URL: from the cache file, or fetched with the
/// client's key and then added to the cache file.
fn load_url(
    cache: &Path,
    url: String,
    client: &Client,
    old: Option<ModelList>,
    refresh: bool,
    typed_key: Option<&str>,
    saved_keys: &SavedKeys,
) -> Loaded {
    if old.is_some() && !refresh {
        return Loaded {
            url,
            list: old,
            error: None,
        };
    }
    match key(client, typed_key, saved_keys).and_then(|key| list(client, key.as_deref())) {
        Ok(models) => {
            let new = ModelList {
                models,
                fetched_at: now(),
            };
            let error = write_cache(cache, &url, &new)
                .err()
                .map(ListError::NotCached);
            Loaded {
                url,
                list: Some(new),
                error,
            }
        }
        Err(error) => Loaded {
            url,
            list: old,
            error: Some(error),
        },
    }
}

/// Returns the URL that serves the client's model list. It is also the
/// list's key in the cache file and in [`Loaded::url`].
pub fn list_url(client: &Client) -> String {
    models_url(&client.format, &client.base_url)
}

/// Lists the client's model IDs, sorted. When the client needs a key and
/// `key` is `None`, returns [`ListError::NoEnv`] or [`ListError::NoKey`]
/// without sending a request.
fn list(client: &Client, key: Option<&str>) -> Result<Vec<String>, ListError> {
    match (key, &client.api_key_env) {
        (None, Some(variable)) => Err(ListError::NoEnv(variable.clone())),
        (None, None) if client.forward_auth => Err(ListError::NoKey),
        _ => fetch(&list_url(client), &client.format, key),
    }
}

/// Returns the models that contain every word of `query`, ignoring case.
pub fn matching<'a>(models: &'a [String], query: &str) -> Vec<&'a str> {
    let query = query.to_lowercase();
    let words: Vec<&str> = query.split_whitespace().collect();
    models
        .iter()
        .map(String::as_str)
        .filter(|model| {
            let model = model.to_lowercase();
            words.iter().all(|word| model.contains(word))
        })
        .collect()
}

/// Returns the models endpoint for a client. It builds the URL the way the
/// server builds its request URLs. OpenAI clients drop a trailing
/// `/chat/completions` or `/responses`. Anthropic clients drop a trailing
/// `/v1/messages` to `/v1`, and add `/v1` to any other path.
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
    format!("{path}{query}")
}

/// Pairs each models URL with the client whose settings fetch it: the first
/// client with that URL that sends a key, or else the first client with it.
fn by_url(clients: &[Client]) -> Vec<(String, &Client)> {
    let mut urls: Vec<(String, &Client)> = Vec::new();
    for client in clients {
        let url = list_url(client);
        match urls.iter_mut().find(|(seen, _)| *seen == url) {
            Some(entry) if !sends_key(entry.1) && sends_key(client) => entry.1 = client,
            Some(_) => {}
            None => urls.push((url, client)),
        }
    }
    urls
}

/// Returns whether the server sends this client a key. If it does, listing
/// the client's models needs a key too.
fn sends_key(client: &Client) -> bool {
    client.forward_auth || client.api_key_env.is_some()
}

/// Returns the key to list the client's models with, or `None` when the
/// client needs no key or none is available.
///
/// The typed key comes first, then the client's `api_key_env` variable when
/// this process has it, then the Keychain item for the client's `base_url`.
/// `saved` stores each Keychain answer, and a thread holds its lock while
/// it reads the Keychain. So macOS asks at most once per `base_url` when it
/// needs the user's permission to hand over a key.
fn key(
    client: &Client,
    typed_key: Option<&str>,
    saved: &SavedKeys,
) -> Result<Option<String>, ListError> {
    if !sends_key(client) {
        return Ok(None);
    }
    if let Some(key) = typed_key.map(str::to_string).or_else(|| env_key(client)) {
        return Ok(Some(key));
    }
    saved
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(client.base_url.clone())
        .or_insert_with(|| saved_key(&client.base_url))
        .clone()
        .map_err(ListError::Keychain)
}

fn env_key(client: &Client) -> Option<String> {
    client
        .api_key_env
        .as_deref()
        .and_then(|variable| std::env::var(variable).ok())
        .filter(|key| !key.trim().is_empty())
}

/// Returns the key saved in the Keychain for `base_url`, or `None` when no
/// key is saved.
#[cfg(target_os = "macos")]
fn saved_key(base_url: &str) -> Result<Option<String>, String> {
    /// The Keychain's `errSecItemNotFound` status.
    const NOT_FOUND: i32 = -25300;
    match security_framework::passwords::get_generic_password(KEYCHAIN_SERVICE, base_url) {
        Ok(bytes) => String::from_utf8(bytes)
            .map(|key| Some(key).filter(|key| !key.trim().is_empty()))
            .map_err(|_| "the saved key is not text".to_string()),
        Err(error) if error.code() == NOT_FOUND => Ok(None),
        Err(error) => Err(keychain_error(&error)),
    }
}

#[cfg(not(target_os = "macos"))]
fn saved_key(_base_url: &str) -> Result<Option<String>, String> {
    Ok(None)
}

/// Saves a key in the login Keychain for every client with this `base_url`.
#[cfg(target_os = "macos")]
pub fn save_key(base_url: &str, key: &str) -> Result<(), String> {
    security_framework::passwords::set_generic_password(
        KEYCHAIN_SERVICE,
        base_url,
        key.trim().as_bytes(),
    )
    .map_err(|error| keychain_error(&error))
}

/// Returns the Keychain's message without its final period, so the window
/// can end its own sentence after it.
#[cfg(target_os = "macos")]
fn keychain_error(error: &security_framework::base::Error) -> String {
    error.to_string().trim_end_matches('.').to_string()
}

/// Returns whether a key has a line break, which would end the header that
/// carries the key and start another one.
pub fn has_line_break(key: &str) -> bool {
    key.contains(['\r', '\n'])
}

/// Reads the cache file. A missing or unreadable file reads as empty, so the
/// app fetches its lists again and writes a new file.
fn read_cache(path: &Path) -> BTreeMap<String, ModelList> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Adds the list at `url` to the cache file. The list replaces the one at
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

/// Returns the current time in seconds since 1970-01-01 UTC.
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
    let mut child = Command::new("curl")
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
        // Read request headers from stdin, one per line.
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
        stdin
            .write_all(headers.as_bytes())
            .map_err(|error| failed(format!("send headers to curl: {error}")))?;
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

/// Parses the model IDs. OpenAI and Anthropic both answer with
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
    use std::net::TcpListener;
    use std::sync::Arc;

    /// A local HTTP server that answers every request with the current
    /// `response` and records the head of each request. Its thread waits in
    /// `accept` until the test process exits.
    struct Stub {
        /// The `base_url` of a client that lists models here.
        url: String,
        response: Arc<Mutex<String>>,
        requests: Arc<Mutex<Vec<Vec<String>>>>,
    }

    impl Stub {
        fn start(response: String) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let url = format!("http://{}/v1", listener.local_addr().expect("addr"));
            let response = Arc::new(Mutex::new(response));
            let requests = Arc::new(Mutex::new(Vec::new()));
            let (answer, record) = (Arc::clone(&response), Arc::clone(&requests));
            std::thread::spawn(move || {
                for mut stream in listener.incoming().map_while(Result::ok) {
                    let mut head = Vec::new();
                    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
                    let mut line = String::new();
                    while reader.read_line(&mut line).is_ok_and(|read| read > 0) && line != "\r\n" {
                        head.push(line.trim_end().to_string());
                        line.clear();
                    }
                    record.lock().expect("lock").push(head);
                    let response = answer.lock().expect("lock").clone();
                    let _ = stream.write_all(response.as_bytes());
                }
            });
            Self {
                url,
                response,
                requests,
            }
        }

        fn answer(&self, response: String) {
            *self.response.lock().expect("lock") = response;
        }

        fn requests(&self) -> Vec<Vec<String>> {
            self.requests.lock().expect("lock").clone()
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

    fn models(loaded: &[Loaded]) -> Vec<&str> {
        loaded
            .iter()
            .flat_map(|entry| entry.list.iter().flat_map(|list| &list.models))
            .map(String::as_str)
            .collect()
    }

    #[test]
    fn fetches_a_missing_list_once_and_then_uses_the_cached_list() {
        let stub = Stub::start(listing(&["model-b", "model-a"]));
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join(CACHE_FILE);
        // Both clients list their models at the same URL.
        let clients = [
            client("gateway", "openai_responses", &stub.url),
            client("gateway_chat", "openai_chat", &stub.url),
        ];

        let first = load(&cache, &clients, false, Some("test-key"), &|_| {});
        // No key this time: a cached list needs neither a request nor a key.
        let second = load(&cache, &clients, false, None, &|_| {});

        let requests = stub.requests();
        assert_eq!(requests.len(), 1, "{requests:?}");
        assert_eq!(requests[0][0], "GET /v1/models HTTP/1.1");
        assert!(
            requests[0].contains(&"Authorization: Bearer test-key".to_string()),
            "{requests:?}"
        );
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].url, format!("{}/models", stub.url));
        assert_eq!(first[0].error, None);
        assert_eq!(models(&first), ["model-a", "model-b"]);
        assert_eq!(second, first);
        let text = std::fs::read_to_string(&cache).expect("read cache");
        assert!(!text.contains("test-key"), "the cache holds no key: {text}");
    }

    #[test]
    fn refresh_fetches_again_and_replaces_the_cached_list() {
        let stub = Stub::start(listing(&["old-model"]));
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join(CACHE_FILE);
        let clients = [client("gateway", "openai_chat", &stub.url)];
        load(&cache, &clients, false, Some("test-key"), &|_| {});
        stub.answer(listing(&["new-model"]));

        let refreshed = load(&cache, &clients, true, Some("test-key"), &|_| {});
        let later = load(&cache, &clients, false, None, &|_| {});

        assert_eq!(stub.requests().len(), 2);
        assert_eq!(models(&refreshed), ["new-model"]);
        assert_eq!(refreshed[0].error, None);
        assert_eq!(later, refreshed, "the cache file holds the new list");
    }

    #[test]
    fn a_failed_refresh_keeps_the_cached_list() {
        let stub = Stub::start(listing(&["old-model"]));
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join(CACHE_FILE);
        let clients = [client("gateway", "openai_chat", &stub.url)];
        let first = load(&cache, &clients, false, Some("test-key"), &|_| {});
        stub.answer(
            "HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n{}".to_string(),
        );

        let refreshed = load(&cache, &clients, true, Some("test-key"), &|_| {});
        let later = load(&cache, &clients, false, None, &|_| {});

        assert_eq!(stub.requests().len(), 2);
        assert_eq!(refreshed[0].list, first[0].list);
        assert_eq!(
            refreshed[0].error,
            Some(ListError::Failed(format!(
                "{}/models answered HTTP 500",
                stub.url
            )))
        );
        assert_eq!(later, first, "the cache file still holds the old list");
    }

    #[test]
    fn fetches_a_shared_url_with_the_client_that_sends_a_key() {
        let stub = Stub::start(listing(&["model-a"]));
        let dir = tempfile::tempdir().expect("tempdir");
        // The first client at the URL sends no key, so listing with it would
        // leave out the key the second client needs.
        let keyless = Client {
            forward_auth: false,
            ..client("open", "openai_chat", &stub.url)
        };
        let clients = [keyless, client("gateway", "openai_responses", &stub.url)];

        let loaded = load(
            &dir.path().join(CACHE_FILE),
            &clients,
            false,
            Some("test-key"),
            &|_| {},
        );

        let requests = stub.requests();
        assert_eq!(requests.len(), 1, "{requests:?}");
        assert!(
            requests[0].contains(&"Authorization: Bearer test-key".to_string()),
            "{requests:?}"
        );
        assert_eq!(models(&loaded), ["model-a"]);
    }

    #[test]
    fn returns_a_fetched_list_that_the_cache_file_cannot_hold() {
        let stub = Stub::start(listing(&["model-a"]));
        let dir = tempfile::tempdir().expect("tempdir");
        // The cache file's directory does not exist, so the file cannot be
        // written.
        let cache = dir.path().join("missing").join(CACHE_FILE);
        let clients = [client("gateway", "openai_chat", &stub.url)];

        let loaded = load(&cache, &clients, false, Some("test-key"), &|_| {});

        assert_eq!(models(&loaded), ["model-a"]);
        assert!(
            matches!(loaded[0].error, Some(ListError::NotCached(_))),
            "{loaded:?}"
        );
        assert!(!cache.exists());
    }

    #[test]
    fn the_cache_file_keeps_the_newer_of_two_lists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join(CACHE_FILE);
        let url = "https://api.example/v1/models";
        let list = |model: &str, fetched_at| ModelList {
            models: vec![model.to_string()],
            fetched_at,
        };

        // The load that fetched first writes last.
        write_cache(&cache, url, &list("new", 200)).expect("write");
        write_cache(&cache, url, &list("old", 100)).expect("write");

        assert_eq!(read_cache(&cache).get(url), Some(&list("new", 200)));
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

    #[test]
    fn builds_the_models_url_with_the_servers_url_rules() {
        for (format, base_url, expected) in [
            (
                "openai_chat",
                "https://api.example/v1/",
                "https://api.example/v1/models",
            ),
            (
                "openai_responses",
                "https://api.example/v1/responses",
                "https://api.example/v1/models",
            ),
            (
                "anthropic_messages",
                "https://api.example",
                "https://api.example/v1/models",
            ),
            (
                "anthropic_messages",
                "https://api.example/v1/messages",
                "https://api.example/v1/models",
            ),
            // The server sends messages to /foo/messages/v1/messages here.
            (
                "anthropic_messages",
                "https://api.example/foo/messages",
                "https://api.example/foo/messages/v1/models",
            ),
            (
                "openai_chat",
                "https://api.example/v1?api-version=1",
                "https://api.example/v1/models?api-version=1",
            ),
        ] {
            assert_eq!(
                models_url(format, base_url),
                expected,
                "{format} {base_url}"
            );
        }
    }

    #[test]
    fn matches_every_word_ignoring_case() {
        let models = [
            "claude-opus-5-5".to_string(),
            "claude-sonnet-5".to_string(),
            "gpt-5.6-sol".to_string(),
        ];

        assert_eq!(matching(&models, "CLAUDE opus"), ["claude-opus-5-5"]);
        assert_eq!(matching(&models, "").len(), 3);
    }
}
