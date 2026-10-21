// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ratatui renders snapshots and submits typed operations to the shared controller.

use crate::controller::{Action, ActionKind, AppSettings, Availability, Controller, Snapshot};
use ratatui::{
    crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    layout::{Constraint, Layout},
    style::{Color, Style},
    widgets::{Block, Paragraph, Sparkline, Tabs, Wrap},
};
use serde_json::Value;
use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};
const PAGES: &[&str] = &["Overview", "Usage", "Settings"];
/// SETTINGS_KEYS lists the Settings page keys and the controller action that each key runs,
/// either directly or through a form. The Settings page lists these keys as its help text.
const SETTINGS_KEYS: &[(char, &str, ActionKind)] = &[
    ('o', "Open server config", ActionKind::OpenConfig),
    ('c', "Check server config", ActionKind::CheckConfig),
    ('b', "Restart server", ActionKind::Restart),
    ('a', "Open app settings", ActionKind::OpenSettings),
    (
        'e',
        "Change baseline model and refresh interval",
        ActionKind::SavePreferences,
    ),
    ('p', "Open model prices", ActionKind::OpenPrices),
    ('k', "Check saved server keys", ActionKind::ServerKeys),
    ('s', "Save a server key", ActionKind::SaveServerKey),
    (
        'i',
        "Import a server key from your login shell",
        ActionKind::ImportServerKey,
    ),
    (
        'x',
        "Remove a saved server key",
        ActionKind::RemoveServerKey,
    ),
    ('u', "Update from source", ActionKind::PreviewUpdate),
    ('l', "Open update log", ActionKind::OpenUpdateLog),
];
#[derive(Clone)]
struct Form {
    title: String,
    fields: Vec<(String, String)>,
    index: usize,
    kind: FormKind,
    cursor: usize,
    page_scroll: u16,
}
impl Form {
    fn byte_cursor(&self) -> usize {
        self.fields[self.index]
            .1
            .char_indices()
            .nth(self.cursor)
            .map_or(self.fields[self.index].1.len(), |(index, _)| index)
    }
    fn insert(&mut self, text: &str) {
        let text = text.replace(['\r', '\n'], "");
        let index = self.byte_cursor();
        self.fields[self.index].1.insert_str(index, &text);
        self.cursor += text.chars().count();
    }
    fn edit(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Tab | KeyCode::BackTab => {
                self.index = if key.code == KeyCode::Tab {
                    (self.index + 1) % self.fields.len()
                } else {
                    (self.index + self.fields.len() - 1) % self.fields.len()
                };
                self.cursor = self.fields[self.index].1.chars().count();
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => {
                self.cursor = (self.cursor + 1).min(self.fields[self.index].1.chars().count())
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.fields[self.index].1.chars().count(),
            KeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                let index = self.byte_cursor();
                self.fields[self.index].1.remove(index);
            }
            KeyCode::Delete if self.cursor < self.fields[self.index].1.chars().count() => {
                let index = self.byte_cursor();
                self.fields[self.index].1.remove(index);
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.fields[self.index].1.clear();
                self.cursor = 0;
            }
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.insert(&c.to_string())
            }
            _ => {}
        }
    }
}
#[derive(Clone)]
enum FormKind {
    Filter,
    Update,
    Preferences,
    SaveKey,
    ImportKey,
    RemoveKey,
}
#[derive(Clone)]
struct View {
    snapshot: Snapshot,
    page: usize,
    session: usize,
    filter: String,
    preview: Option<Value>,
    scroll: u16,
    message: String,
    form: Option<Form>,
    details: String,
    showing_details: bool,
    selected_session: Option<String>,
    editor_file: Option<std::path::PathBuf>,
    /// This field holds the last `ServerKeys` reply. It is `None` until the user checks the keys.
    server_keys: Option<Value>,
}
impl View {
    fn dispatch(&mut self, controller: &mut Controller, action: Action) {
        match controller.editor_target(&action) {
            Ok(Some(path)) => {
                self.editor_file = Some(path);
                return;
            }
            Ok(None) => {}
            Err(error) => {
                self.message = error;
                return;
            }
        }
        match controller.dispatch(action) {
            Ok(reply) => {
                self.message = reply.message;
                self.details = reply.data["details"]
                    .as_array()
                    .map(|details| {
                        details
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                if let Err(e) = self.refresh(controller) {
                    self.message = e;
                }
            }
            Err(e) => self.message = e,
        }
    }
    fn refresh(&mut self, controller: &mut Controller) -> Result<(), String> {
        self.snapshot = controller.snapshot()?;
        if let Some(session) = &self.selected_session {
            self.session = self
                .snapshot
                .sessions
                .iter()
                .position(|s| s == session)
                .map_or(usize::MAX, |i| i + 1);
        }
        Ok(())
    }
    fn form(&mut self, title: &str, kind: FormKind, fields: Vec<(String, String)>) {
        let cursor = fields.first().map_or(0, |f| f.1.chars().count());
        self.form = Some(Form {
            title: title.into(),
            fields,
            index: 0,
            kind,
            cursor,
            page_scroll: self.scroll,
        });
        self.scroll = 0;
    }
    fn submit(&mut self, controller: &mut Controller) -> Result<(), String> {
        let retained = self.form.clone().ok_or("No form is open.")?;
        let result = self.submit_form(controller);
        if result.is_err() {
            self.form = Some(retained);
        } else if self.form.is_none() {
            self.scroll = retained.page_scroll;
        }
        result
    }
    fn submit_form(&mut self, controller: &mut Controller) -> Result<(), String> {
        let form = self.form.take().ok_or("No form is open.")?;
        let Some(action) = self.form_action(&form)? else {
            return Ok(());
        };
        let reply = controller.dispatch(action)?;
        self.message = reply.message;
        self.details = reply.data["details"]
            .as_array()
            .map(|d| {
                d.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        if matches!(
            form.kind,
            FormKind::SaveKey | FormKind::ImportKey | FormKind::RemoveKey
        ) {
            // The key change succeeded. A failed recheck only clears the list.
            self.load_server_keys(controller).ok();
        }
        self.refresh(controller)?;
        Ok(())
    }
    /// This function builds the controller action that a form submits.
    /// The filter form changes only the view, so it returns `None`.
    fn form_action(&mut self, form: &Form) -> Result<Option<Action>, String> {
        let value = |i: usize| form.fields[i].1.trim().to_string();
        Ok(Some(match form.kind {
            FormKind::Update => {
                if value(0) != "UPDATE" {
                    return Err("Type UPDATE to start the reviewed source update.".into());
                }
                Action::Update {
                    preview_token: self
                        .preview
                        .as_ref()
                        .and_then(|p| p["preview_token"].as_str())
                        .map(str::to_owned),
                }
            }
            FormKind::Filter => {
                self.filter = value(0);
                self.scroll = 0;
                return Ok(None);
            }
            FormKind::Preferences => Action::SavePreferences {
                baseline_model: value(0),
                refresh_seconds: value(1).parse().map_err(|_| {
                    "Enter the refresh interval as a whole number of seconds.".to_string()
                })?,
            },
            FormKind::SaveKey => {
                if value(1).is_empty() {
                    return Err("Enter a key first.".into());
                }
                Action::SaveServerKey {
                    name: value(0),
                    key: value(1),
                }
            }
            FormKind::ImportKey => Action::ImportServerKey { name: value(0) },
            FormKind::RemoveKey => Action::RemoveServerKey { name: value(0) },
        }))
    }
    /// A failed check clears the list, so the Settings page never shows outdated key states.
    fn load_server_keys(&mut self, controller: &mut Controller) -> Result<(), String> {
        self.server_keys = None;
        self.server_keys = Some(controller.dispatch(Action::ServerKeys {})?.data);
        Ok(())
    }
    fn key_list(&self) -> String {
        let Some(data) = &self.server_keys else {
            return "Press k to check which keys are saved.".into();
        };
        let keys = data["keys"]
            .as_array()
            .map(|keys| {
                keys.iter()
                    .map(|key| {
                        format!(
                            "{} · {}",
                            string(key, "name"),
                            if key["saved"] == true {
                                "Saved"
                            } else {
                                "Not saved"
                            }
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if keys.is_empty() {
            return "The server config names no keys with api_key_env.".into();
        }
        format!("Stored in {}\n{}", string(data, "store"), keys.join("\n"))
    }
    fn connection(&self) -> String {
        let metrics = &self.snapshot.metrics;
        if metrics.running {
            return format!("Connected · {}", metrics.server_url);
        }
        match self.snapshot.data_state.health["reason"].as_str() {
            Some(reason) => format!("Not responding · {} · {reason}", metrics.server_url),
            None => format!("Not responding · {}", metrics.server_url),
        }
    }
    fn body(&self) -> String {
        let errors = self.snapshot.errors.join("\n");
        let body = if self.showing_details {
            format!("{}\n\n{}", self.message, self.details)
        } else {
            self.page_body()
        };
        if errors.is_empty() {
            body
        } else {
            format!("DATA ERRORS\n{errors}\n\n{body}")
        }
    }
    fn page_body(&self) -> String {
        // This destructuring has no `..`, so adding a snapshot field breaks the build until the
        // terminal app shows it. When another place shows a field, the pattern binds it to `_`
        // and a comment names that place.
        let Snapshot {
            update_status,
            settings,
            data_state,
            summary,
            metrics,
            today,
            analytics,
            estimates,
            baseline_model: _, // The Settings page shows settings.baseline_model.
            activity: _,       // run() draws it as the Overview sparkline.
            sessions,
            entries,
            limited,
            skipped,
            errors: _,          // body() shows errors above every page.
            refresh_seconds: _, // run() refreshes on this interval.
        } = &self.snapshot;
        let cost = |period: &str| -> Vec<String> {
            estimates
                .get(period)
                .copied()
                .flatten()
                .map(|estimate| {
                    crate::summary::cost_rows(&estimate)
                        .into_iter()
                        .map(|row| row.trim().to_string())
                        .collect()
                })
                .unwrap_or_default()
        };
        let groups =
            |title: &str,
             groups: &std::collections::BTreeMap<String, crate::rollup::ModelTokens>| {
                let mut lines = vec![String::new(), title.to_string()];
                if groups.is_empty() {
                    lines.push("None recorded".into());
                }
                for (name, tokens) in groups {
                    lines.push(format!(
                        "{}  {} calls · {} tokens",
                        if name.is_empty() {
                            "Not recorded"
                        } else {
                            name
                        },
                        tokens.requests,
                        tokens.total()
                    ));
                }
                lines
            };
        match self.page {
            0 if data_state.usage.state == Availability::Unavailable => {
                format!("CONNECTION\n{}", self.connection())
            }
            1 if data_state.history.state == Availability::Unavailable => String::new(),
            0 => {
                let mut lines = vec![
                    "CONNECTION".to_string(),
                    self.connection(),
                    String::new(),
                    format!("TODAY · {today}"),
                    format!(
                        "{} model calls · {} tokens",
                        metrics.today.requests, metrics.today.tokens
                    ),
                ];
                lines.extend(cost("today"));
                lines.extend([
                    String::new(),
                    "PAST 7 DAYS".into(),
                    format!(
                        "{} model calls · {} tokens",
                        metrics.week.requests, metrics.week.tokens
                    ),
                ]);
                lines.extend(cost("week"));
                lines.extend([
                    String::new(),
                    "ALL RETAINED HISTORY".into(),
                    format!(
                        "{} model calls · {} tokens",
                        analytics.all.requests(),
                        analytics.all.tokens()
                    ),
                ]);
                lines.extend(cost("all"));
                lines.extend(groups(
                    "ROUTES · PAST 7 DAYS · answers and routing calls",
                    &analytics.week.routes,
                ));
                lines.extend(groups(
                    "ROUTING CALLS BY MODEL · PAST 7 DAYS",
                    &analytics.week.classifier,
                ));
                let notes: Vec<&str> = summary
                    .iter()
                    .map(|line| line.trim())
                    .filter(|line| line.to_lowercase().contains("savings") || line.ends_with('%'))
                    .collect();
                if !notes.is_empty() {
                    lines.extend([String::new(), "USAGE DETAILS".into()]);
                    lines.extend(notes.into_iter().map(str::to_string));
                }
                lines.join("\n")
            }
            1 => {
                let selected = self.selected_session.as_deref();
                let mut lines = vec![format!(
                    "s Next session   f Filter turn/model/route\n\
                     Session: {}{}   Filter: {}\n\
                     {} recent calls{}; {} unreadable records skipped.\n\
                     Each row is one model call; a turn can contain several calls.\n",
                    selected.unwrap_or("All sessions"),
                    if self.session == usize::MAX {
                        " (outside recent history)".to_string()
                    } else if self.session > 0 {
                        format!(" ({} of {})", self.session, sessions.len())
                    } else {
                        String::new()
                    },
                    self.filter,
                    entries.len(),
                    if *limited {
                        " (8 MiB / 5,000 record limit)"
                    } else {
                        ""
                    },
                    skipped
                )];
                for e in entries.iter().rev() {
                    if selected.is_some_and(|s| e.session_id.as_deref() != Some(s)) {
                        continue;
                    }
                    if !format!(
                        "{} {} {}",
                        e.model,
                        e.turn_id.as_deref().unwrap_or_default(),
                        e.route_id
                    )
                    .to_lowercase()
                    .contains(&self.filter.to_lowercase())
                    {
                        continue;
                    }
                    lines.push(format!(
                        "{}  {}{}\n\
                         Session {}  Turn {}  Route {}\n\
                         Uncached input {}  Cached input {}  Output {}\n",
                        e.ts,
                        e.model,
                        if e.tier == "classifier" {
                            " (routing overhead)"
                        } else {
                            " (answer call)"
                        },
                        not_recorded(e.session_id.as_deref()),
                        not_recorded(e.turn_id.as_deref()),
                        not_recorded(Some(&e.route_id)),
                        e.prompt_tokens.saturating_sub(e.cached_tokens),
                        e.cached_tokens,
                        e.completion_tokens
                    ));
                }
                lines.join("\n")
            }
            _ => {
                let actions = SETTINGS_KEYS
                    .iter()
                    .map(|(key, label, _)| format!("{key} {label}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                // This destructuring has no `..`, so adding an app setting breaks the build until
                // the Settings page shows it.
                let Some(AppSettings {
                    path,
                    server_url,
                    routing_log,
                    config_file,
                    launchd_label,
                    refresh_seconds,
                    baseline_model,
                    prices_file,
                }) = settings
                else {
                    return format!(
                        "App settings could not be loaded. Press a to open them, \
                         then F5 to check them again.\n\nACTIONS\n{actions}"
                    );
                };
                let update = [
                    update_status["state"].as_str(),
                    update_status["stage"].as_str(),
                    update_status["message"].as_str(),
                ]
                .into_iter()
                .flatten()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join(" · ");
                format!(
                    "APP SETTINGS · {}\n\
                     Server URL: {server_url}\n\
                     Server config: {}\n\
                     Routing log: {}\n\
                     Service label: {launchd_label}\n\
                     Baseline model: {baseline_model}\n\
                     Refresh interval: {refresh_seconds} seconds\n\
                     Model prices: {}\n\n\
                     UPDATE FROM SOURCE\n{}\n\n\
                     SERVER KEYS\n{}\n\n\
                     ACTIONS\n{actions}\n\n\
                     Files open with $VISUAL, or $EDITOR if $VISUAL is unset or blank.\n\
                     The default is vi (Unix) or notepad (Windows).\n\
                     Update uses the saved installation checkout.",
                    path.display(),
                    config_file.display(),
                    routing_log.display(),
                    prices_file
                        .as_ref()
                        .map_or("Bundled rates".into(), |file| file.display().to_string()),
                    if update.is_empty() {
                        "No update has run"
                    } else {
                        &update
                    },
                    self.key_list()
                )
            }
        }
    }
    fn key(&mut self, controller: &mut Controller, code: KeyCode) -> Result<(), String> {
        if self.page == 2
            && let KeyCode::Char(c) = code
            && let Some(&(.., kind)) = SETTINGS_KEYS.iter().find(|(key, ..)| *key == c)
        {
            return self.start(controller, kind);
        }
        match code {
            KeyCode::PageDown => self.scroll = self.scroll.saturating_add(10),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::F(5) => self.refresh(controller)?,
            KeyCode::Char('s') if self.page == 1 => {
                self.session = self.session.saturating_add(1) % (self.snapshot.sessions.len() + 1);
                self.selected_session = self
                    .session
                    .checked_sub(1)
                    .and_then(|i| self.snapshot.sessions.get(i))
                    .cloned();
                self.scroll = 0;
            }
            KeyCode::Char('f') if self.page == 1 => self.form(
                "Filter turn, model, or route",
                FormKind::Filter,
                vec![("Search".into(), self.filter.clone())],
            ),
            KeyCode::Char('h') => {
                self.showing_details = !self.showing_details;
                self.scroll = 0;
            }
            KeyCode::Esc => {
                self.message.clear();
                self.showing_details = false;
                self.scroll = 0;
            }
            _ => {}
        }
        Ok(())
    }
    /// This function starts the action for a Settings page key. The match has no wildcard
    /// arm, so adding a controller action breaks the build until the terminal app can run it.
    fn start(&mut self, controller: &mut Controller, kind: ActionKind) -> Result<(), String> {
        match kind {
            ActionKind::OpenConfig => self.dispatch(controller, Action::OpenConfig {}),
            ActionKind::CheckConfig => self.dispatch(controller, Action::CheckConfig {}),
            ActionKind::Restart => self.dispatch(controller, Action::Restart {}),
            ActionKind::OpenSettings => self.dispatch(controller, Action::OpenSettings {}),
            ActionKind::OpenPrices => self.dispatch(controller, Action::OpenPrices {}),
            ActionKind::OpenUpdateLog => self.dispatch(controller, Action::OpenUpdateLog {}),
            ActionKind::ServerKeys => {
                self.load_server_keys(controller)?;
                self.message = "Checked the saved server keys.".into();
            }
            ActionKind::SavePreferences => {
                let (baseline, refresh) = self
                    .snapshot
                    .settings
                    .as_ref()
                    .map(|s| (s.baseline_model.clone(), s.refresh_seconds.to_string()))
                    .unwrap_or_default();
                self.form(
                    "Change app preferences",
                    FormKind::Preferences,
                    vec![
                        ("Baseline model ID".into(), baseline),
                        ("Refresh interval in seconds".into(), refresh),
                    ],
                );
            }
            ActionKind::SaveServerKey => self.form(
                "Save a server key · restart the server to use it",
                FormKind::SaveKey,
                vec![
                    ("Key name".into(), String::new()),
                    ("Key".into(), String::new()),
                ],
            ),
            ActionKind::ImportServerKey => self.form(
                "Import a server key from your login shell",
                FormKind::ImportKey,
                vec![("Key name".into(), String::new())],
            ),
            ActionKind::RemoveServerKey => self.form(
                "Remove a saved server key · the server does not start until it is saved again",
                FormKind::RemoveKey,
                vec![("Key name".into(), String::new())],
            ),
            // The update form submits Update after the user reads this preview.
            ActionKind::PreviewUpdate | ActionKind::Update => {
                let reply = controller.dispatch(Action::PreviewUpdate {})?;
                self.preview = Some(reply.data);
                self.form(
                    "Review source update · type UPDATE to start",
                    FormKind::Update,
                    vec![("Confirmation".into(), String::new())],
                );
            }
        }
        Ok(())
    }
}
fn string(v: &Value, key: &str) -> String {
    v[key].as_str().unwrap_or_default().into()
}

// The view shows empty or missing IDs as "Not recorded".
fn not_recorded(value: Option<&str>) -> &str {
    value
        .filter(|value| !value.is_empty())
        .unwrap_or("Not recorded")
}
enum WorkKind {
    Key(KeyCode),
    Submit,
    Refresh,
}
struct Work {
    view: View,
    kind: WorkKind,
}
struct PasteMode;
impl Drop for PasteMode {
    fn drop(&mut self) {
        let _ = ratatui::crossterm::execute!(std::io::stdout(), event::DisableBracketedPaste);
    }
}
// The caller restores normal terminal input before starting the editor.
// The absolute file path stays one literal argument, even if its name starts with '-'.
fn edit_file(path: &std::path::Path) -> Result<(), String> {
    let editor = std::env::var("VISUAL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::env::var("EDITOR")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .unwrap_or_else(|| if cfg!(windows) { "notepad" } else { "vi" }.into());
    let parts = shlex::split(&editor).ok_or("The editor command has an unmatched quote.")?;
    let (program, arguments) = parts
        .split_first()
        .ok_or("Set VISUAL or EDITOR to an editor command.")?;
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(path)
    };
    let status = std::process::Command::new(program)
        .args(arguments)
        .arg(path)
        .status()
        .map_err(|error| format!("Could not start the editor {program}: {error}"))?;
    if !status.success() {
        return Err(format!("The editor {program} exited with {status}."));
    }
    Ok(())
}
pub fn run(mut controller: Controller) -> Result<(), String> {
    let (snapshot, page) = match controller.snapshot() {
        Ok(snapshot) => (snapshot, 0),
        Err(error) => (
            Snapshot {
                errors: vec![error],
                refresh_seconds: 30,
                ..Snapshot::default()
            },
            2,
        ),
    };
    let mut view = View {
        snapshot,
        page,
        session: 0,
        filter: String::new(),
        preview: None,
        scroll: 0,
        message: String::new(),
        form: None,
        details: String::new(),
        showing_details: false,
        selected_session: None,
        editor_file: None,
        server_keys: None,
    };
    ratatui::crossterm::execute!(std::io::stdout(), event::EnableBracketedPaste)
        .map_err(|e| e.to_string())?;
    let _paste_mode = PasteMode;
    let (requests, jobs) = mpsc::channel::<Work>();
    let (completed, results) = mpsc::channel::<View>();
    // One worker owns the controller while the terminal continues to draw and read input.
    let worker = thread::spawn(move || {
        while let Ok(mut job) = jobs.recv() {
            let result = match job.kind {
                WorkKind::Key(code) => job.view.key(&mut controller, code),
                WorkKind::Submit => job.view.submit(&mut controller),
                WorkKind::Refresh => job.view.refresh(&mut controller),
            };
            if let Err(error) = result {
                job.view.message = error;
            }
            if completed.send(job.view).is_err() {
                break;
            }
        }
    });
    let result = ratatui::run(|terminal| -> Result<(), String> {
        let mut last = Instant::now();
        let mut busy = false;
        let mut quit_pending = false;
        let mut work_page = 0;
        loop {
            if let Ok(mut updated) = results.try_recv() {
                if view.page != work_page {
                    updated.page = view.page;
                    updated.scroll = view.scroll;
                }
                view = updated;
                busy = false;
                if quit_pending {
                    return Ok(());
                }
                if let Some(path) = view.editor_file.take() {
                    ratatui::crossterm::execute!(std::io::stdout(), event::DisableBracketedPaste)
                        .map_err(|error| error.to_string())?;
                    terminal.show_cursor().map_err(|error| error.to_string())?;
                    ratatui::try_restore().map_err(|error| error.to_string())?;
                    let result = edit_file(&path);
                    ratatui::crossterm::terminal::enable_raw_mode()
                        .map_err(|error| error.to_string())?;
                    ratatui::crossterm::execute!(
                        std::io::stdout(),
                        ratatui::crossterm::terminal::EnterAlternateScreen,
                        event::EnableBracketedPaste
                    )
                    .map_err(|error| error.to_string())?;
                    terminal.clear().map_err(|error| error.to_string())?;
                    view.message = match result {
                        Ok(()) => format!("Closed the editor for {}.", path.display()),
                        Err(error) => error,
                    };
                    work_page = view.page;
                    requests
                        .send(Work {
                            view: view.clone(),
                            kind: WorkKind::Refresh,
                        })
                        .map_err(|error| error.to_string())?;
                    busy = true;
                }
            }
            terminal
                .draw(|frame| {
                    let areas = Layout::vertical([
                        Constraint::Length(3),
                        Constraint::Min(5),
                        Constraint::Length(5),
                    ])
                    .split(frame.area());
                    frame.render_widget(
                        Tabs::new(PAGES.iter().copied())
                            .select(view.page)
                            .highlight_style(
                                Style::default().fg(Color::White).bg(Color::Rgb(40, 56, 76)),
                            )
                            .block(Block::bordered().title("Switchyard")),
                        areas[0],
                    );
                    let mut body_area = areas[1];
                    if view.page == 0 && view.form.is_none() {
                        let split = Layout::vertical([Constraint::Min(5), Constraint::Length(6)])
                            .split(body_area);
                        body_area = split[0];
                        frame.render_widget(
                            Sparkline::default()
                                .data(view.snapshot.activity)
                                .style(Style::default().fg(Color::LightBlue))
                                .block(
                                    Block::bordered()
                                        .title("Recent model calls · past 24h · hourly"),
                                ),
                            split[1],
                        );
                    }
                    let (body, title) = if let Some(form) = &view.form {
                        let rows = form
                            .fields
                            .iter()
                            .enumerate()
                            .map(|(i, (label, value))| {
                                let marker = if i == form.index { ">" } else { " " };
                                let mut value = value.clone();
                                // The key field shows one bullet per character.
                                if matches!(form.kind, FormKind::SaveKey) && i == 1 {
                                    value = "•".repeat(value.chars().count());
                                }
                                if i == form.index {
                                    let cursor = value
                                        .char_indices()
                                        .nth(form.cursor)
                                        .map_or(value.len(), |(index, _)| index);
                                    value.insert(cursor, '▏');
                                }
                                format!("{marker} {label}: {value}")
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        let hints = match &form.kind {
                            FormKind::Update => view
                                .preview
                                .as_ref()
                                .map(|p| {
                                    format!(
                                        "\n\n{}",
                                        serde_json::to_string_pretty(p).unwrap_or_default()
                                    )
                                })
                                .unwrap_or_default(),
                            FormKind::SaveKey | FormKind::ImportKey | FormKind::RemoveKey
                                if view.server_keys.is_some() =>
                            {
                                format!("\n\nSERVER KEYS\n{}", view.key_list())
                            }
                            _ => String::new(),
                        };
                        (format!("{rows}{hints}"), form.title.clone())
                    } else {
                        (view.body(), PAGES[view.page].into())
                    };
                    frame.render_widget(
                        Paragraph::new(body)
                            .wrap(Wrap { trim: false })
                            .scroll((view.scroll, 0))
                            .block(Block::bordered().title(title)),
                        body_area,
                    );
                    frame.render_widget(
                        Paragraph::new(format!(
                            "←/→ or Tab: section · PgUp/PgDn: scroll · F5: refresh\n\
                             h: details · Esc: dismiss · q: quit\n\
                             Forms: Tab: field · ←/→: cursor · \
                             Ctrl-U: clear · Enter: submit · Esc: cancel\n\
                             {}",
                            if busy {
                                if quit_pending {
                                    "Finishing the operation before quitting…"
                                } else {
                                    "Working… Navigation is available; \
                                     wait before starting another action."
                                }
                            } else {
                                &view.message
                            }
                        )),
                        areas[2],
                    );
                })
                .map_err(|e| e.to_string())?;
            if event::poll(Duration::from_millis(200)).map_err(|e| e.to_string())? {
                let event = event::read().map_err(|e| e.to_string())?;
                if let Event::Paste(text) = &event {
                    if !busy && let Some(form) = &mut view.form {
                        form.insert(text);
                    }
                    continue;
                }
                let Event::Key(key) = event else {
                    continue;
                };
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Tab | KeyCode::Right if view.form.is_none() => {
                        view.page = (view.page + 1) % PAGES.len();
                        view.scroll = 0;
                        continue;
                    }
                    KeyCode::BackTab | KeyCode::Left if view.form.is_none() => {
                        view.page = (view.page + PAGES.len() - 1) % PAGES.len();
                        view.scroll = 0;
                        continue;
                    }
                    _ => {}
                }
                if busy {
                    match key.code {
                        KeyCode::Char('q') if view.form.is_none() => quit_pending = true,
                        KeyCode::PageDown => view.scroll = view.scroll.saturating_add(10),
                        KeyCode::PageUp => view.scroll = view.scroll.saturating_sub(10),
                        _ => {}
                    }
                    continue;
                }
                if let Some(form) = view.form.as_mut() {
                    match key.code {
                        KeyCode::Esc => {
                            view.scroll = form.page_scroll;
                            view.form = None;
                        }
                        KeyCode::Enter => {
                            work_page = view.page;
                            requests
                                .send(Work {
                                    view: view.clone(),
                                    kind: WorkKind::Submit,
                                })
                                .map_err(|e| e.to_string())?;
                            busy = true;
                        }
                        KeyCode::PageDown => view.scroll = view.scroll.saturating_add(10),
                        KeyCode::PageUp => view.scroll = view.scroll.saturating_sub(10),
                        _ => form.edit(key),
                    }
                    continue;
                }
                if key.code == KeyCode::Char('q') {
                    return Ok(());
                }
                work_page = view.page;
                requests
                    .send(Work {
                        view: view.clone(),
                        kind: WorkKind::Key(key.code),
                    })
                    .map_err(|e| e.to_string())?;
                busy = true;
            }
            if last.elapsed() > Duration::from_secs(view.snapshot.refresh_seconds)
                && view.form.is_none()
                && !busy
            {
                work_page = view.page;
                requests
                    .send(Work {
                        view: view.clone(),
                        kind: WorkKind::Refresh,
                    })
                    .map_err(|e| e.to_string())?;
                busy = true;
                last = Instant::now();
            }
        }
    });
    drop(requests);
    worker
        .join()
        .map_err(|_| "The terminal operation worker stopped unexpectedly.".to_string())?;
    if result.is_err() {
        let _ = ratatui::try_restore();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    fn view() -> View {
        View {
            snapshot: Snapshot::default(),
            page: 1,
            session: 0,
            filter: String::new(),
            preview: None,
            scroll: 0,
            message: String::new(),
            form: None,
            details: String::new(),
            showing_details: false,
            selected_session: None,
            editor_file: None,
            server_keys: None,
        }
    }

    #[test]
    // A read failure must not appear as an empty log or zero recorded usage.
    fn unavailable_data_does_not_render_empty_usage() {
        let mut view = view();
        view.snapshot.data_state.usage.state = Availability::Unavailable;
        view.snapshot.data_state.history.state = Availability::Unavailable;
        view.snapshot.errors = vec!["Could not read usage".into()];
        view.page = 0;
        assert!(view.body().contains("Could not read usage"));
        assert!(!view.body().contains("model calls"));
        view.page = 1;
        assert!(view.body().contains("Could not read usage"));
        assert!(!view.body().contains("recent calls"));
    }

    #[test]
    // Every controller action needs a Settings key, or the desktop app can do something
    // that the terminal app cannot.
    fn every_controller_action_has_a_settings_key() {
        let mut keys: Vec<char> = SETTINGS_KEYS.iter().map(|&(key, ..)| key).collect();
        // run() handles q before Settings keys, and View::key handles h on every page.
        keys.extend(['q', 'h']);
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(
            keys.len(),
            SETTINGS_KEYS.len() + 2,
            "Settings keys must be unique and must not use q or h"
        );

        // Update is the only action without its own key: the u key reviews the update
        // first, and the review form submits Update.
        let missing: Vec<ActionKind> = ActionKind::iter()
            .filter(|&kind| {
                kind != ActionKind::Update && !SETTINGS_KEYS.iter().any(|&(.., k)| k == kind)
            })
            .collect();
        assert!(missing.is_empty(), "No Settings key runs {missing:?}");
        let review = Form {
            title: String::new(),
            fields: vec![(String::new(), "UPDATE".into())],
            index: 0,
            kind: FormKind::Update,
            cursor: 0,
            page_scroll: 0,
        };
        assert!(matches!(
            view().form_action(&review),
            Ok(Some(Action::Update { .. }))
        ));
    }
}
