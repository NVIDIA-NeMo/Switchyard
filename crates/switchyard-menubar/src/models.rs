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
    /// The Keychain did not return the saved key.
    Keychain(String),
    /// The server answered 401 or 403.
    Rejected(u16),
    Failed(String),
    /// The list was fetched, but the cache file could not be written.
    NotCached(String),
}

impl ListError {
    /// Says what went wrong, in one sentence.
    pub fn reason(&self) -> String {
        match self {
            Self::NoKey => "This client needs a key to list models, and none is saved.".to_string(),
            Self::Keychain(error) => {
                format!("Could not read the saved key from the Keychain: {error}.")
            }
            Self::Rejected(status) => format!("The server rejected the key (HTTP {status})."),
            Self::Failed(error) => format!("Could not list models: {error}."),
            Self::NotCached(error) => format!("Could not save the list: {error}."),
        }
    }

    /// Says what the user can do while the window has no list to show. The
    /// window always accepts a typed model ID, so every answer offers that.
    pub fn advice(&self) -> &'static str {
        if self.needs_key() {
            "Paste the key below and click Save key, or type a model ID."
        } else {
            "Click Refresh models to try again, or type a model ID."
        }
    }

    /// Returns whether saving a key could fix this error.
    pub fn needs_key(&self) -> bool {
        matches!(self, Self::NoKey | Self::Keychain(_) | Self::Rejected(_))
    }
}

/// Returns the model list at each URL that `clients` use, in the order the
/// URLs first appear. Clients whose models share a URL share one list and
/// one request.
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
) -> Vec<Loaded> {
    let mut cached = read_cache(cache);
    let mut saved_keys = HashMap::new();
    let mut fetched = BTreeMap::new();
    let mut loaded = Vec::new();
    for (url, client) in by_url(clients) {
        let old = cached.remove(&url);
        if old.is_some() && !refresh {
            loaded.push(Loaded {
                url,
                list: old,
                error: None,
            });
            continue;
        }
        let result =
            key(client, typed_key, &mut saved_keys).and_then(|key| list(client, key.as_deref()));
        loaded.push(match result {
            Ok(models) => {
                let new = ModelList {
                    models,
                    fetched_at: now(),
                };
                fetched.insert(url.clone(), new.clone());
                Loaded {
                    url,
                    list: Some(new),
                    error: None,
                }
            }
            Err(error) => Loaded {
                url,
                list: old,
                error: Some(error),
            },
        });
    }
    if !fetched.is_empty()
        && let Err(error) = write_cache(cache, &fetched)
    {
        for entry in loaded
            .iter_mut()
            .filter(|entry| fetched.contains_key(&entry.url))
        {
            entry.error = Some(ListError::NotCached(error.clone()));
        }
    }
    loaded
}

/// Returns the URL that serves the client's model list. It is also the
/// list's key in the cache file and in [`Loaded::url`].
pub fn list_url(client: &Client) -> String {
    models_url(&client.format, &client.base_url)
}

/// Lists the client's model IDs, sorted. When the client needs a key and
/// `key` is `None`, returns [`ListError::NoKey`] without sending a request.
fn list(client: &Client, key: Option<&str>) -> Result<Vec<String>, ListError> {
    if sends_key(client) && key.is_none() {
        return Err(ListError::NoKey);
    }
    fetch(&list_url(client), &client.format, key)
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
/// server builds its request URLs: OpenAI clients drop a trailing
/// `/chat/completions` or `/responses`, and Anthropic clients use
/// `/v1/models`.
fn models_url(format: &str, base_url: &str) -> String {
    let (base, query) = match base_url.split_once('?') {
        Some((base, query)) => (base, format!("?{query}")),
        None => (base_url, String::new()),
    };
    let base = base.trim_end_matches('/');
    let path = if format == "anthropic_messages" {
        let root = base.strip_suffix("/messages").unwrap_or(base);
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
/// `saved` remembers each Keychain answer, so macOS asks at most once per
/// `base_url` when it needs the user's permission to hand over a key.
fn key(
    client: &Client,
    typed_key: Option<&str>,
    saved: &mut HashMap<String, Result<Option<String>, String>>,
) -> Result<Option<String>, ListError> {
    if !sends_key(client) {
        return Ok(None);
    }
    if let Some(key) = typed_key.map(str::to_string).or_else(|| env_key(client)) {
        return Ok(Some(key));
    }
    saved
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
        Err(error) => Err(error.to_string()),
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
    .map_err(|error| error.to_string())
}

/// Reads the cache file. A missing or unreadable file reads as empty, so the
/// app fetches its lists again and writes a new file.
fn read_cache(path: &Path) -> BTreeMap<String, ModelList> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Adds `lists` to the cache file. A list replaces the one at the same URL
/// unless that one is newer: two loads can overlap, and the load that
/// fetched a list first can write it last. The text goes to a temporary file
/// that is renamed over the cache file, so a reader never sees half a file.
fn write_cache(path: &Path, lists: &BTreeMap<String, ModelList>) -> Result<(), String> {
    let _writing = CACHE_WRITE.lock().unwrap_or_else(PoisonError::into_inner);
    let mut all = read_cache(path);
    for (url, list) in lists {
        if all
            .get(url)
            .is_none_or(|cached| cached.fetched_at <= list.fetched_at)
        {
            all.insert(url.clone(), list.clone());
        }
    }
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
        .args(["--write-out", "\n%{http_code}", url])
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

        let first = load(&cache, &clients, false, Some("test-key"));
        // No key this time: a cached list needs neither a request nor a key.
        let second = load(&cache, &clients, false, None);

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
        load(&cache, &clients, false, Some("test-key"));
        stub.answer(listing(&["new-model"]));

        let refreshed = load(&cache, &clients, true, Some("test-key"));
        let later = load(&cache, &clients, false, None);

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
        let first = load(&cache, &clients, false, Some("test-key"));
        stub.answer(
            "HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n{}".to_string(),
        );

        let refreshed = load(&cache, &clients, true, Some("test-key"));
        let later = load(&cache, &clients, false, None);

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

        let loaded = load(&cache, &clients, false, Some("test-key"));

        assert_eq!(models(&loaded), ["model-a"]);
        assert!(
            matches!(loaded[0].error, Some(ListError::NotCached(_))),
            "{loaded:?}"
        );
        assert!(!cache.exists());
    }

    #[test]
    fn reports_a_rejected_key_and_a_missing_key() {
        let stub =
            Stub::start("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n{}".to_string());
        let gateway = client("gateway", "openai_chat", &stub.url);

        assert_eq!(list(&gateway, Some("wrong")), Err(ListError::Rejected(401)));
        assert_eq!(list(&gateway, None), Err(ListError::NoKey));
        assert_eq!(stub.requests().len(), 1, "a missing key sends no request");
    }

    #[test]
    fn builds_the_models_url_like_the_server_builds_request_urls() {
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
