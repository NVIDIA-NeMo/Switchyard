// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Both frontends call these operations so validation stays independent of widgets.

use crate::{
    accounts, app,
    config::Config,
    harness::{self, Harness},
    history, models, rollup, server,
    server_config::{ALGORITHMS, Choice, Route, ServerConfig},
    sessions,
    summary::Row,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    Editor {
        generation: u64,
        route: String,
        algorithm: String,
    },
    Apply {
        generation: u64,
        route: String,
        algorithm: String,
        choices: Vec<Choice>,
    },
    Models {
        client: String,
        refresh: bool,
        key: Option<String>,
    },
    Install {
        tool: Harness,
        account: Option<String>,
        route: String,
        id: String,
    },
    Restore {
        tool: Harness,
        account: Option<String>,
    },
    AddAccount {
        tool: Harness,
        name: String,
    },
    Launch {
        tool: Harness,
        account: Option<String>,
        route: String,
        id: String,
        project: String,
    },
    Restart {},
    Update {},
    OpenConfig {},
}

#[derive(Serialize)]
pub struct Reply {
    pub message: String,
    pub data: Value,
}
impl Reply {
    fn message(message: String) -> Self {
        Self {
            message,
            data: Value::Null,
        }
    }
}

pub struct Controller {
    settings: PathBuf,
    config: Config,
    reader: rollup::Reader,
    // The generation ties a draft to the server config file selected by the settings.
    generation: u64,
}
impl Controller {
    pub fn new(config: Config, settings: PathBuf) -> Self {
        Self {
            config,
            settings,
            reader: rollup::Reader::default(),
            generation: 0,
        }
    }
    fn reload(&mut self) -> Result<(), String> {
        let config = Config::load(&self.settings)?;
        if config.config_file != self.config.config_file {
            self.generation += 1;
        }
        self.config = config;
        Ok(())
    }
    fn check_generation(&self, generation: u64) -> Result<(), String> {
        if generation != self.generation {
            return Err("The route changed or was removed. Refresh and choose it again.".into());
        }
        Ok(())
    }
    fn routes(&self) -> Result<ServerConfig, String> {
        let text = std::fs::read_to_string(&self.config.config_file).map_err(|e| e.to_string())?;
        ServerConfig::parse(&text).map_err(|_| {
            "The server config contains invalid TOML. Open it in Settings to fix it.".into()
        })
    }
    pub fn snapshot(&mut self) -> Result<Value, String> {
        self.reload()?;
        let (_, summary) = app::refresh(&self.config, &mut self.reader);
        let summary: Vec<_> = summary
            .into_iter()
            .map(|r| match r {
                Row::Label(s) => s,
                Row::Separator => String::new(),
            })
            .collect();
        let mut errors = Vec::new();
        let mut routes = Vec::new();
        let mut clients = Vec::new();
        match self.routes() {
            Ok(config) => {
                for route in config.routes() {
                    let choices = config.choices(&route.key);
                    let models = choices
                        .iter()
                        .map(|c| format!("{} / {}", c.client, c.model))
                        .collect::<Vec<_>>()
                        .join(" → ");
                    routes.push(json!({"key": route.key, "id": route.id, "kind": route.kind,
                        "label": format!("{} · {} · {}", route.id, route.kind, models), "choices": choices,
                        "tiers": config.algorithm(&route.key).map(|a| config.roles(&route.key, a).iter().map(|r| r.tier.map(|t| format!("{t:?}"))).collect::<Vec<_>>()),
                        "editable": true, "generated": config.generated_by("routes", &route.key)}));
                }
                clients = config.clients().iter().map(|c| json!({"name":c.name,"host":c.host(),"models":config.models_on(&c.name),"unlisted":models::unlisted(c),"accepts_key":cfg!(target_os = "macos") && !models::unlisted(c) && (c.forward_auth || c.api_key_env.is_some()),"note":models::missing_env(c).map(models::missing_env_note)})).collect();
            }
            Err(e) => errors.push(e),
        }
        let history = history::load(&self.config.routing_log).unwrap_or_else(|e| {
            errors.push(e);
            history::History::default()
        });
        let tools: Vec<_> = harness::HARNESSES.iter().map(|(tool,label)| {
            let accounts: Vec<_> = accounts::list(*tool).into_iter().map(|(name,_)| name).collect();
            json!({"tool":tool,"label":label,"available":harness::binary(*tool).is_some(),
                "status":harness::inspect(*tool,&harness::paths(*tool)).unwrap_or_else(|e| e),"accounts":accounts})
        }).collect();
        Ok(
            json!({"generation":self.generation,"summary":summary,"routes":routes,"clients":clients,"tools":tools,
            "algorithms":ALGORITHMS.iter().map(|a|json!({"kind":a.kind,"title":a.title,"summary":a.summary})).collect::<Vec<_>>(),
            "sessions":history.sessions(),"entries":history.entries,"limited":history.limited,"skipped":history.skipped,
            "errors":errors,"refresh_seconds":self.config.refresh_seconds.max(1)}),
        )
    }
    pub fn dispatch(&mut self, action: Action) -> Result<Reply, String> {
        self.reload()?;
        match action {
            Action::Editor {
                generation,
                route,
                algorithm,
            } => {
                self.check_generation(generation)?;
                let config = self.routes()?;
                current_route(&config, &route, None)?;
                let algorithm = algorithm_by_id(&algorithm)?;
                let roles = config.roles(&route, algorithm);
                let data = json!({"roles": roles.iter().map(|r|json!({"label":r.label,"hint":r.hint,"tier":r.tier.map(|t|format!("{t:?}"))})).collect::<Vec<_>>(),"choices":config.choices(&route)});
                Ok(Reply {
                    message: algorithm.summary.into(),
                    data,
                })
            }
            Action::Apply {
                generation,
                route,
                algorithm,
                choices,
            } => {
                self.check_generation(generation)?;
                let config = self.routes()?;
                current_route(&config, &route, None)?;
                let algorithm = algorithm_by_id(&algorithm)?;
                config.edit(&route, algorithm, &choices)?;
                server::apply(&self.config, &route, algorithm, &choices).map(Reply::message)
            }
            Action::Models {
                client,
                refresh,
                key,
            } => {
                let config = self.routes()?;
                let client = config
                    .clients()
                    .into_iter()
                    .find(|c| c.name == client)
                    .ok_or("Unknown endpoint.")?;
                if let Some(key) = &key {
                    if key.trim().is_empty() || models::has_line_break(key) {
                        return Err("Enter a nonempty key without line breaks.".into());
                    }
                    if models::unlisted(&client)
                        || !(client.forward_auth || client.api_key_env.is_some())
                    {
                        return Err("This endpoint does not accept a model-list key. Choose an endpoint configured for model-list authentication.".into());
                    }
                    #[cfg(not(target_os = "macos"))]
                    models::save_key(&client.base_url, key)?;
                }
                let cache = self.settings.with_file_name(models::CACHE_FILE);
                let result = models::load(
                    &cache,
                    std::slice::from_ref(&client),
                    refresh || key.is_some(),
                    key.as_deref(),
                    &|_| {},
                );
                let loaded = result.into_iter().next().ok_or("No model list returned.")?;
                if let Some(key) = &key {
                    match &loaded.error {
                        None | Some(models::ListError::NotCached(_)) => {
                            models::save_key(&client.base_url, key)?
                        }
                        _ => {
                            return Err(loaded
                                .error
                                .as_ref()
                                .map(models::ListError::reason)
                                .unwrap_or_default());
                        }
                    }
                }
                let message = loaded
                    .error
                    .map(|e| format!("{} {}", e.reason(), e.advice()))
                    .unwrap_or_else(|| "Model list loaded.".into());
                let models = loaded
                    .list
                    .map(|l| l.models)
                    .unwrap_or_else(|| config.models_on(&client.name));
                Ok(Reply {
                    message,
                    data: json!({"client":client.name,"models":models}),
                })
            }
            Action::Install {
                tool,
                account,
                route,
                id,
            } => {
                let account = account_path(tool, account.as_deref())?;
                let config = self.routes()?;
                let route = current_route(&config, &route, Some(&id))?;
                let login = login_mode(tool, &self.config, &route)?;
                harness::install(
                    tool,
                    &accounts::config_paths(tool, account.as_deref()),
                    &self.config.server_url,
                    &route.id,
                    login,
                )
                .map(Reply::message)
            }
            Action::Restore { tool, account } => {
                let account = account_path(tool, account.as_deref())?;
                harness::restore(&accounts::config_paths(tool, account.as_deref()))
                    .map(Reply::message)
            }
            Action::AddAccount { tool, name } => {
                accounts::add(tool, name.trim()).map(Reply::message)
            }
            Action::Launch {
                tool,
                account,
                route,
                id,
                project,
            } => {
                let account = account_path(tool, account.as_deref())?;
                let config = self.routes()?;
                let route = current_route(&config, &route, Some(&id))?;
                let login = login_mode(tool, &self.config, &route)?;
                sessions::launch(
                    tool,
                    &PathBuf::from(project),
                    &route.id,
                    &self.config.server_url,
                    login,
                    account.as_deref(),
                )
                .map(Reply::message)
            }
            Action::Restart {} => server::restart(&self.config.launchd_label)
                .map(|()| Reply::message("Server restart requested.".into())),
            Action::Update {} => update().map(Reply::message),
            Action::OpenConfig {} => {
                server::command("open", &[&self.config.config_file.display().to_string()])?;
                Ok(Reply::message(
                    "Opened the server config in your default editor.".into(),
                ))
            }
        }
    }
}
fn algorithm_by_id(kind: &str) -> Result<&'static crate::server_config::Algorithm, String> {
    ALGORITHMS
        .iter()
        .find(|a| a.kind == kind)
        .ok_or_else(|| "Unknown routing algorithm.".into())
}
fn current_route(config: &ServerConfig, key: &str, id: Option<&str>) -> Result<Route, String> {
    config
        .routes()
        .into_iter()
        .find(|r| r.key == key && id.is_none_or(|id| id == r.id))
        .ok_or_else(|| "The route changed or was removed. Refresh and choose it again.".into())
}
fn account_path(tool: Harness, name: Option<&str>) -> Result<Option<PathBuf>, String> {
    let Some(name) = name else {
        return Ok(None);
    };
    let path = accounts::list(tool)
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, p)| p)
        .ok_or("Unknown account. Refresh and choose an existing account.")?;
    accounts::validate(tool, &path)?;
    Ok(Some(path))
}
fn login_mode(
    tool: Harness,
    settings: &Config,
    route: &crate::server_config::Route,
) -> Result<bool, String> {
    let text = std::fs::read_to_string(&settings.config_file).map_err(|e| e.to_string())?;
    let config = ServerConfig::parse(&text)?;
    let clients = config.clients();
    let choices = config.choices(&route.key);
    if choices.is_empty()
        || !config
            .routes()
            .iter()
            .any(|current| current.key == route.key && current.id == route.id)
    {
        return Err(
            "This route cannot be installed by the app. Choose a route with models.".into(),
        );
    }
    let mut login = false;
    for choice in choices {
        let client = clients
            .iter()
            .find(|c| c.name == choice.client)
            .ok_or("The route has an unknown endpoint.")?;
        if client.forward_auth {
            let url = url::Url::parse(&client.base_url)
                .map_err(|_| "The route has an invalid endpoint URL.")?;
            let host = url
                .host_str()
                .ok_or("The route has an invalid endpoint URL.")?;
            let supported = url.scheme() == "https"
                && url.port().is_none_or(|port| port == 443)
                && match tool {
                    Harness::CodexCli | Harness::CodexApp => host == "chatgpt.com",
                    Harness::Claude => host == "api.anthropic.com",
                    Harness::Pi => false,
                };
            if !supported {
                return Err(format!(
                    "This route forwards the caller's login to {host}. Pick a matching coding tool or a route with server API credentials."
                ));
            }
            login = true;
        }
    }
    Ok(login)
}

fn update() -> Result<String, String> {
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let resource = executable
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("Resources/Update.command"))
        .filter(|p| p.is_file())
        .or_else(|| {
            let installed = crate::config::expand_home(std::path::Path::new(
                "~/Applications/Switchyard.app/Contents/Resources/Update.command",
            ));
            installed.is_file().then_some(installed)
        })
        .ok_or(
            "Install Switchyard.app with scripts/macos/install.sh before updating from the app.",
        )?;
    crate::server::command("open", &[&resource.display().to_string()])?;
    Ok("Terminal is rebuilding and reinstalling from the source checkout used for this install. The app restarts after a successful build. Your routing and prices stay saved.".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    // The Apply fixture supplies a generation so only its extra choice field causes rejection.
    fn malformed_actions_reject_before_dispatch() {
        for input in [
            r#"{"kind":"restart","command":"rm"}"#,
            r#"{"kind":"install","tool":"unknown","route":"r","id":"m","account":null}"#,
            r#"{"kind":"apply","generation":0,"route":"r","algorithm":"passthrough","choices":[{"client":"c","model":"m","extra":1}]}"#,
            r#"{"kind":"unknown"}"#,
        ] {
            assert!(serde_json::from_str::<Action>(input).is_err());
        }
    }
    #[cfg(target_os = "macos")]
    #[test]
    // A cached list must not let an untested key reach Keychain.
    fn submitting_a_key_checks_upstream_even_when_refresh_is_false() {
        use std::io::{Read, Write};
        let dir = tempfile::tempdir().expect("directory");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener");
        listener.set_nonblocking(true).expect("nonblocking");
        let url = format!("http://{}/v1", listener.local_addr().expect("address"));
        let worker = std::thread::spawn(move || {
            for _ in 0..200 {
                if let Ok((mut stream, _)) = listener.accept() {
                    stream
                        .set_read_timeout(Some(std::time::Duration::from_secs(1)))
                        .expect("timeout");
                    let mut request = Vec::new();
                    let mut chunk = [0; 1024];
                    for _ in 0..4 {
                        let n = stream.read(&mut chunk).expect("request");
                        request.extend_from_slice(&chunk[..n]);
                        if n == 0 || request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                            break;
                        }
                    }
                    stream.write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").expect("response");
                    return String::from_utf8_lossy(&request).into_owned();
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            String::new()
        });
        let settings = dir.path().join("settings.toml");
        let routes = dir.path().join("routes.toml");
        std::fs::write(&routes, format!("[llm_clients.c]\nbase_url={url:?}\nformat='openai_responses'\nforward_auth=true\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n")).expect("routes");
        std::fs::write(
            &settings,
            format!(
                "config_file={routes:?}\nrouting_log={:?}",
                dir.path().join("log")
            ),
        )
        .expect("settings");
        let cached =
            json!({format!("{url}/models"): {"models":["cached"],"fetched_at":models::now()}})
                .to_string();
        let cache = dir.path().join(models::CACHE_FILE);
        std::fs::write(&cache, &cached).expect("cache");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        let result = controller.dispatch(Action::Models {
            client: "c".into(),
            refresh: false,
            key: Some("rejected-fixture-key".into()),
        });
        let request = worker.join().expect("worker");
        assert!(result.err().expect("rejected key").contains("401"));
        assert!(request.starts_with("GET /v1/models "));
        assert!(request.contains("Authorization: Bearer rejected-fixture-key"));
        assert_eq!(std::fs::read_to_string(cache).expect("cache"), cached);
        std::fs::write(&routes, "[llm_clients.c]\nbase_url='http://127.0.0.1:9/v1'\nformat='openai_responses'\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n").expect("routes");
        let result = controller.dispatch(Action::Models {
            client: "c".into(),
            refresh: false,
            key: Some("unused-fixture-key".into()),
        });
        assert!(
            result
                .err()
                .expect("unused key")
                .contains("does not accept a model-list key")
        );
    }
    #[test]
    #[cfg(not(target_os = "macos"))]
    // Rejecting a typed key must leave the upstream endpoint, model cache, and route config unchanged.
    fn key_persistence_rejects_without_network_or_file_writes() {
        let dir = tempfile::tempdir().expect("directory");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener");
        listener.set_nonblocking(true).expect("nonblocking");
        let url = format!("http://{}/v1", listener.local_addr().expect("address"));
        let settings = dir.path().join("settings.toml");
        let routes = dir.path().join("routes.toml");
        let text = format!(
            "[llm_clients.c]\nbase_url={url:?}\nformat='openai_responses'\nforward_auth=true\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n"
        );
        std::fs::write(&routes, &text).expect("routes");
        std::fs::write(
            &settings,
            format!(
                "config_file={routes:?}\nrouting_log={:?}",
                dir.path().join("log")
            ),
        )
        .expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        let result = controller.dispatch(Action::Models {
            client: "c".into(),
            refresh: true,
            key: Some("fixture-key".into()),
        });
        assert_eq!(
            result.err(),
            Some("Saving model-list keys requires macOS Keychain. On this platform, set the environment variable named by api_key_env and make it available to the app.".into())
        );
        assert_eq!(
            listener.accept().expect_err("no upstream request").kind(),
            std::io::ErrorKind::WouldBlock
        );
        assert_eq!(std::fs::read_to_string(routes).expect("routes"), text);
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 2);
    }
    #[test]
    fn usage_snapshot_survives_missing_and_malformed_route_configs() {
        let dir = tempfile::tempdir().expect("directory");
        let settings = dir.path().join("settings.toml");
        let routes = dir.path().join("routes.toml");
        let log = dir.path().join("log");
        std::fs::write(
            &log,
            "{\"ts\":\"2026-10-08T12:00:00Z\",\"model\":\"actual\",\"session_id\":\"session\"}\n",
        )
        .expect("log");
        std::fs::write(
            &settings,
            format!("config_file={routes:?}\nrouting_log={log:?}"),
        )
        .expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        for malformed in [false, true] {
            if malformed {
                std::fs::write(&routes, "secret = 'UNTERMINATED_PRIVATE_VALUE").expect("routes");
            }
            let snapshot = controller.snapshot().expect("snapshot");
            assert_eq!(snapshot["sessions"], json!(["session"]));
            assert_eq!(snapshot["entries"][0]["model"], "actual");
            assert!(!snapshot["errors"].as_array().expect("errors").is_empty());
            assert!(!snapshot.to_string().contains("PRIVATE_VALUE"));
        }
    }
    #[test]
    // Matching route names in two files must not let an old draft edit the newly selected file.
    fn stale_editor_rejects_after_settings_change_without_writes() {
        let dir = tempfile::tempdir().expect("directory");
        let settings = dir.path().join("settings.toml");
        let first = dir.path().join("first.toml");
        let second = dir.path().join("second.toml");
        let text = "[llm_clients.c]\nbase_url='https://example.com/v1'\nformat='openai_responses'\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n";
        std::fs::write(&first, text).expect("first");
        std::fs::write(&second, text).expect("second");
        let write_settings = |path: &std::path::Path| {
            std::fs::write(
                &settings,
                format!(
                    "config_file={path:?}\nrouting_log={:?}",
                    dir.path().join("log")
                ),
            )
            .expect("settings");
        };
        write_settings(&first);
        let mut controller =
            Controller::new(Config::load(&settings).expect("settings"), settings.clone());
        let snapshot = controller.snapshot().expect("snapshot");
        assert_eq!(snapshot["routes"][0]["tiers"], json!(["Capable"]));
        let generation = snapshot["generation"].as_u64().expect("generation");
        write_settings(&second);
        for action in [
            Action::Editor {
                generation,
                route: "r".into(),
                algorithm: "passthrough".into(),
            },
            Action::Apply {
                generation,
                route: "r".into(),
                algorithm: "passthrough".into(),
                choices: vec![Choice {
                    client: "c".into(),
                    model: "changed".into(),
                }],
            },
        ] {
            assert!(controller.dispatch(action).is_err());
        }
        assert_eq!(std::fs::read_to_string(&second).expect("second"), text);
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 3);
        assert_eq!(
            controller.snapshot().expect("snapshot")["generation"],
            generation + 1
        );
    }
    #[test]
    fn rejected_install_has_no_writes() {
        let dir = tempfile::tempdir().expect("directory");
        let settings = dir.path().join("settings.toml");
        let config = dir.path().join("server.toml");
        let text = "[llm_clients.c]\nbase_url='http://chatgpt.com/backend-api/codex'\nforward_auth=true\nformat='openai_responses'\n[targets.m]\nid='actual'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n";
        std::fs::write(&config, text).expect("config");
        std::fs::write(
            &settings,
            format!(
                "config_file={:?}\nrouting_log={:?}",
                config,
                dir.path().join("log")
            ),
        )
        .expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        for id in ["old", "public"] {
            assert!(
                controller
                    .dispatch(Action::Install {
                        tool: Harness::CodexCli,
                        account: None,
                        route: "r".into(),
                        id: id.into()
                    })
                    .is_err()
            );
            assert_eq!(std::fs::read_to_string(&config).expect("config"), text);
            assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 2);
        }
        let snapshot = controller.snapshot().expect("snapshot");
        assert!(
            snapshot["routes"][0]["label"]
                .as_str()
                .expect("label")
                .contains("passthrough · c / actual")
        );
        assert!(!snapshot.to_string().contains("backend-api"));
    }
}
