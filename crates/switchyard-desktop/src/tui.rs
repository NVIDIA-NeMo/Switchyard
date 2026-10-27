// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The terminal app shows recorded usage, recent calls, and server status.
//!
//! One worker thread owns the controller and refreshes its snapshots while the UI thread
//! draws and reads input.

use crate::controller::{
    Action, ActionKind, AppSettings, Availability, Controller, Reply, Snapshot,
};
use crate::history::Entry;
use crate::pricing::Estimate;
use crate::rollup::{CLASSIFIER_TIER, ModelTokens, Totals, Usage};
use crate::summary::{money, tokens};
use chrono::{DateTime, Utc};
use ratatui::{
    DefaultTerminal, Frame,
    crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{
        Bar, BarChart, Block, BorderType, Cell, Clear, List, ListItem, ListState, Padding,
        Paragraph, Row, Scrollbar, ScrollbarOrientation, ScrollbarState, Table, TableState, Tabs,
        Wrap,
    },
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};
use switchyard_costs::{CostSource, RecordedCost};

/// Smaller terminals show a resize message instead of clipped panels.
const MIN_WIDTH: u16 = 60;
const MIN_HEIGHT: u16 = 16;
/// The calls page puts the detail pane beside the table at this width or wider.
const SIDE_DETAIL_WIDTH: u16 = 120;
const TOAST_TIME: Duration = Duration::from_secs(4);
const SPINNER: [char; 8] = ['⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽', '⣾'];

/// Each entry holds a key, its status-bar label, and a longer description.
type Key = (&'static str, &'static str, &'static str);
const GLOBAL_KEYS: &[Key] = &[
    (
        "1-4",
        "page",
        "Switch pages. Tab, Shift-Tab, ← and → also switch pages.",
    ),
    (
        "r",
        "refresh",
        "Reload usage, calls, and server health now. F5 also refreshes.",
    ),
    ("?", "help", "Show or hide this help."),
    (
        "q",
        "quit",
        "Quit the app. The server keeps running. Ctrl-C also quits.",
    ),
];
const CALLS_KEYS: &[Key] = &[
    (
        "↑↓",
        "select",
        "Move the selection. j, k, PgUp, PgDn, g, and G also move it.",
    ),
    (
        "/",
        "filter",
        "Filter as you type. Every word must appear in the model, route, session, or turn. \
         A word with a capital letter must match case exactly.",
    ),
    (
        "s",
        "session",
        "Show one session at a time, then all sessions again.",
    ),
    ("⏎", "details", "Open the selected call's details."),
    ("esc", "clear", "Clear the filter and the session."),
];
const MODELS_KEYS: &[Key] = &[(
    "p",
    "period",
    "Switch the period: today, the past 7 days, or the whole log. The [ and ] keys also work.",
)];
const SETTINGS_KEYS: &[Key] = &[
    ("↑↓", "select", "Select an action. j, g, and G also move."),
    (
        "⏎",
        "run",
        "Run the selected action. The letter beside each action also runs it.",
    ),
];
const FILTER_KEYS: &[Key] = &[
    ("⏎", "keep", "Keep the filter and go back to the list."),
    ("esc", "clear", "Clear the filter."),
    ("ctrl-u", "erase", "Erase the filter text."),
];
const FORM_KEYS: &[Key] = &[
    ("⏎", "submit", ""),
    ("tab", "next field", ""),
    ("esc", "cancel", ""),
];
const UPDATE_KEYS: &[Key] = &[
    ("⏎", "start", ""),
    ("↑↓", "scroll", ""),
    ("esc", "cancel", ""),
];
const POPUP_KEYS: &[Key] = &[("↑↓", "scroll", ""), ("esc", "close", "")];
const CONFIRM_KEYS: &[Key] = &[("y", "confirm", ""), ("n", "cancel", "")];

/// The Settings page uses these entries for action labels, help text, and shortcut keys.
const ACTIONS: &[(char, &str, &str, ActionKind)] = &[
    (
        'o',
        "Open server config",
        "Opens the server config in $VISUAL or $EDITOR. Restart the server after you save it.",
        ActionKind::OpenConfig,
    ),
    (
        'c',
        "Check server config",
        "Runs the installed server's config check. The running server keeps running.",
        ActionKind::CheckConfig,
    ),
    (
        'b',
        "Restart server",
        "Checks the server config, then restarts the Switchyard service after you confirm. \
         Requests in progress fail.",
        ActionKind::Restart,
    ),
    (
        'a',
        "Open app settings",
        "Opens this app's settings file in $VISUAL or $EDITOR.",
        ActionKind::OpenSettings,
    ),
    (
        'e',
        "Change preferences",
        "Changes the refresh interval and the baseline model used to estimate savings.",
        ActionKind::SavePreferences,
    ),
    (
        'p',
        "Open model prices",
        "Opens the custom price file in $VISUAL or $EDITOR. With bundled prices, this explains \
         how to set a custom file.",
        ActionKind::OpenPrices,
    ),
    (
        'k',
        "Check server keys",
        "Lists the API key variables that the server config names and whether each is saved.",
        ActionKind::ServerKeys,
    ),
    (
        's',
        "Save a server key",
        "Saves an API key for a variable that the server config names. Restart the server to \
         use it.",
        ActionKind::SaveServerKey,
    ),
    (
        'i',
        "Import a server key",
        "Saves an API key from your login shell's environment. Restart the server to use it.",
        ActionKind::ImportServerKey,
    ),
    (
        'x',
        "Remove a server key",
        "Removes a saved API key. The server does not start until the key is saved again.",
        ActionKind::RemoveServerKey,
    ),
    (
        'u',
        "Update from source",
        "Shows what the saved source checkout would install. Type UPDATE to start it.",
        ActionKind::PreviewUpdate,
    ),
    (
        'l',
        "Open update log",
        "Opens the log of the last source update in $VISUAL or $EDITOR.",
        ActionKind::OpenUpdateLog,
    ),
    (
        'v',
        "Open server logs",
        "Opens the Switchyard server logs folder in $VISUAL or $EDITOR.",
        ActionKind::OpenLogs,
    ),
];

/// Column is a table heading, its width, and whether its values are right-aligned.
type Column = (&'static str, Constraint, bool);
/// The calls table drops the columns marked `true` below 90 columns.
const CALL_COLUMNS: [(Column, bool); 8] = [
    (("TIME", Constraint::Length(8), true), false),
    (("MODEL", Constraint::Fill(2), false), false),
    (("ROUTE", Constraint::Fill(1), false), true),
    (("KIND", Constraint::Length(7), false), false),
    (("IN", Constraint::Length(7), true), false),
    (("CACHED", Constraint::Length(7), true), true),
    (("OUT", Constraint::Length(7), true), false),
    (("COST", Constraint::Length(9), true), false),
];

/// Theme maps each kind of text to one style. NO_COLOR keeps the text modifiers.
#[derive(Clone, Copy)]
struct Theme {
    muted: Style,
    key: Style,
    ok: Style,
    warn: Style,
    error: Style,
    selected: Style,
    border: Style,
    bar: Style,
}
impl Theme {
    fn new(color: bool) -> Self {
        let plain = Style::new();
        if !color {
            return Self {
                muted: plain.add_modifier(Modifier::DIM),
                key: plain.add_modifier(Modifier::BOLD),
                ok: plain,
                warn: plain.add_modifier(Modifier::BOLD),
                error: plain.add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                selected: plain.add_modifier(Modifier::REVERSED),
                border: plain,
                bar: plain,
            };
        }
        // Named ANSI colors follow the terminal's palette, so they work on light and dark themes.
        Self {
            muted: plain.fg(Color::DarkGray),
            key: plain.fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ok: plain.fg(Color::Green),
            warn: plain.fg(Color::Yellow),
            error: plain.fg(Color::Red),
            // A fixed foreground keeps dim cells readable on the selected row.
            selected: plain.fg(Color::Black).bg(Color::Cyan),
            border: plain.fg(Color::DarkGray),
            bar: plain.fg(Color::Cyan),
        }
    }
}
/// This function follows https://no-color.org: any non-empty NO_COLOR value turns color off.
fn color_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none_or(|value| value.is_empty())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    Overview,
    Calls,
    Models,
    Settings,
}
const TABS: [Tab; 4] = [Tab::Overview, Tab::Calls, Tab::Models, Tab::Settings];
impl Tab {
    fn name(self) -> &'static str {
        match self {
            Tab::Overview => "Overview",
            Tab::Calls => "Calls",
            Tab::Models => "Models",
            Tab::Settings => "Settings",
        }
    }
    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Period {
    Today,
    Week,
    All,
}
impl Period {
    fn name(self) -> &'static str {
        match self {
            Period::Today => "Today",
            Period::Week => "Past 7 days",
            Period::All => "Whole log",
        }
    }
    fn next(self) -> Self {
        match self {
            Period::Today => Period::Week,
            Period::Week => Period::All,
            Period::All => Period::Today,
        }
    }
    fn previous(self) -> Self {
        self.next().next()
    }
    fn totals(self, usage: &Usage) -> &Totals {
        match self {
            Period::Today => &usage.today,
            Period::Week => &usage.week,
            Period::All => &usage.all,
        }
    }
    /// The estimate is `None` when the usage log could not be read.
    fn estimate(self, snapshot: &Snapshot) -> Option<Estimate> {
        let key = match self {
            Period::Today => "today",
            Period::Week => "week",
            Period::All => "all",
        };
        snapshot.estimates.get(key).copied().flatten()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FormKind {
    Preferences,
    SaveKey,
    ImportKey,
    RemoveKey,
    Update,
}
/// Form holds the text fields for an action that needs input.
struct Form {
    kind: FormKind,
    title: &'static str,
    fields: Vec<(&'static str, String)>,
    index: usize,
    /// This field counts characters, not bytes, from the start of the selected field.
    cursor: usize,
    /// This field shows the last validation or controller error below the fields.
    error: String,
}
impl Form {
    fn new(kind: FormKind, title: &'static str, fields: Vec<(&'static str, String)>) -> Self {
        let cursor = fields.first().map_or(0, |field| field.1.chars().count());
        Self {
            kind,
            title,
            fields,
            index: 0,
            cursor,
            error: String::new(),
        }
    }
    fn byte_cursor(&self) -> usize {
        let text = &self.fields[self.index].1;
        text.char_indices()
            .nth(self.cursor)
            .map_or(text.len(), |(index, _)| index)
    }
    fn insert(&mut self, text: &str) {
        let text = text.replace(['\r', '\n'], "");
        let index = self.byte_cursor();
        self.fields[self.index].1.insert_str(index, &text);
        self.cursor += text.chars().count();
    }
    fn edit(&mut self, key: KeyEvent) {
        let length = self.fields[self.index].1.chars().count();
        match key.code {
            KeyCode::Tab | KeyCode::BackTab => {
                let count = self.fields.len();
                self.index = if key.code == KeyCode::Tab {
                    (self.index + 1) % count
                } else {
                    (self.index + count - 1) % count
                };
                self.cursor = self.fields[self.index].1.chars().count();
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(length),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = length,
            KeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                let index = self.byte_cursor();
                self.fields[self.index].1.remove(index);
            }
            KeyCode::Delete if self.cursor < length => {
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
                self.insert(&c.to_string());
            }
            _ => {}
        }
    }
    fn value(&self, index: usize) -> String {
        self.fields[index].1.trim().to_string()
    }
    /// This function builds the controller action that the form submits. An update needs the
    /// token from the preview that the user read.
    fn action(&self, preview: Option<&Value>) -> Result<Action, String> {
        Ok(match self.kind {
            FormKind::Preferences => Action::SavePreferences {
                baseline_model: self.value(0),
                refresh_seconds: self.value(1).parse().map_err(|_| {
                    "Enter the refresh interval as a whole number of seconds.".to_string()
                })?,
            },
            FormKind::SaveKey => {
                if self.value(1).is_empty() {
                    return Err("Enter a key first.".into());
                }
                Action::SaveServerKey {
                    name: self.value(0),
                    key: self.value(1),
                }
            }
            FormKind::ImportKey => Action::ImportServerKey {
                name: self.value(0),
            },
            FormKind::RemoveKey => Action::RemoveServerKey {
                name: self.value(0),
            },
            FormKind::Update => {
                if self.value(0) != "UPDATE" {
                    return Err("Type UPDATE to start the reviewed source update.".into());
                }
                Action::Update {
                    preview_token: preview
                        .and_then(|preview| preview["preview_token"].as_str())
                        .map(str::to_owned),
                }
            }
        })
    }
    fn label(&self) -> &'static str {
        match self.kind {
            FormKind::Preferences => "Saving the preferences",
            FormKind::SaveKey => "Saving the server key",
            FormKind::ImportKey => "Importing the server key",
            FormKind::RemoveKey => "Removing the server key",
            FormKind::Update => "Starting the update",
        }
    }
}

enum Popup {
    Help {
        scroll: u16,
    },
    Detail {
        // The popup owns this call because a refresh can remove it from recent history.
        entry: Box<Entry>,
        scroll: u16,
    },
    Restart,
    Form(Form),
    Update {
        preview: Value,
        scroll: u16,
        confirm: Form,
    },
    Message {
        title: &'static str,
        text: String,
        error: bool,
        scroll: u16,
    },
}
impl Popup {
    fn scroll_mut(&mut self) -> Option<&mut u16> {
        match self {
            Popup::Help { scroll }
            | Popup::Detail { scroll, .. }
            | Popup::Update { scroll, .. }
            | Popup::Message { scroll, .. } => Some(scroll),
            Popup::Restart | Popup::Form(_) => None,
        }
    }
    fn form_mut(&mut self) -> Option<&mut Form> {
        match self {
            Popup::Form(form) | Popup::Update { confirm: form, .. } => Some(form),
            _ => None,
        }
    }
}

enum Job {
    Refresh,
    Run(Action),
}
enum Done {
    Editor(PathBuf),
    Finished {
        reply: Option<Result<Reply, String>>,
        /// The box keeps this message small; a snapshot holds every model and route total.
        snapshot: Result<Box<Snapshot>, String>,
    },
}
/// Request is a job for the worker plus the label that the status bar shows while it runs.
struct Request {
    job: Job,
    label: &'static str,
    automatic: bool,
}
impl Request {
    fn run(action: Action, label: &'static str) -> Self {
        Self {
            job: Job::Run(action),
            label,
            automatic: false,
        }
    }
    fn refresh(automatic: bool) -> Self {
        Self {
            job: Job::Refresh,
            label: "Refreshing",
            automatic,
        }
    }
}
struct Busy {
    label: &'static str,
    /// This field names the running action, so its reply can update the right view.
    kind: Option<ActionKind>,
    /// This field records the popup generation when the action starts.
    popup_generation: u64,
    started: Instant,
    /// Automatic refreshes report errors in the status bar, and automatic key checks update
    /// Settings without opening a popup.
    automatic: bool,
}
struct Toast {
    text: String,
    error: bool,
    shown: Instant,
}

struct App {
    theme: Theme,
    data: Snapshot,
    tab: Tab,
    period: Period,
    filter: String,
    filtering: bool,
    session: Option<String>,
    /// These indexes point into `data.entries`, newest first, after both filters are applied.
    visible: Vec<usize>,
    selected: usize,
    offset: usize,
    action: usize,
    /// This field holds the last `ServerKeys` reply. It is `None` until the user checks the keys,
    /// and a failed check clears it so the page never shows outdated key states.
    server_keys: Option<Value>,
    popup: Option<Popup>,
    /// This counter changes whenever a popup opens, closes, or is replaced.
    popup_generation: u64,
    toast: Option<Toast>,
    busy: Option<Busy>,
    /// A quit request waits for the active refresh to finish.
    quit: bool,
    refreshed: Option<Instant>,
    last_refresh: Instant,
    width: u16,
}

impl App {
    fn new(theme: Theme) -> Self {
        Self {
            theme,
            data: Snapshot::default(),
            tab: Tab::Overview,
            period: Period::Week,
            filter: String::new(),
            filtering: false,
            session: None,
            visible: Vec::new(),
            selected: 0,
            offset: 0,
            action: 0,
            server_keys: None,
            popup: None,
            popup_generation: 0,
            toast: None,
            busy: None,
            quit: false,
            refreshed: None,
            last_refresh: Instant::now(),
            width: 80,
        }
    }

    fn rebuild_calls(&mut self) {
        let session = self.session.as_deref();
        let filter = self.filter.as_str();
        self.visible = self
            .data
            .entries
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, entry)| {
                session.is_none_or(|session| entry.session_id.as_deref() == Some(session))
                    && matches_filter(entry, filter)
            })
            .map(|(index, _)| index)
            .collect();
        self.selected = self.selected.min(self.visible.len().saturating_sub(1));
    }

    fn selected_entry(&self) -> Option<&Entry> {
        self.visible
            .get(self.selected)
            .map(|&index| &self.data.entries[index])
    }

    // Every replacement or dismissal changes the generation, even when the same form reopens.
    fn set_popup(&mut self, popup: Option<Popup>) {
        self.popup_generation += 1;
        self.popup = popup;
    }

    /// A reply from a dismissed or replaced popup uses the status bar to preserve the current draft.
    fn report(&mut self, text: String, error: bool, popup_matches: bool) {
        if popup_matches {
            self.notify(text, error);
        } else if !text.is_empty() {
            self.toast = Some(Toast {
                text,
                error,
                shown: Instant::now(),
            });
        }
    }

    /// Short single-line messages appear in the status bar; others open a scrollable window.
    fn notify(&mut self, text: String, error: bool) {
        if text.is_empty() {
            return;
        }
        if text.contains('\n') || text.chars().count() + 8 > usize::from(self.width) * 2 / 3 {
            self.set_popup(Some(Popup::Message {
                title: if error { "Error" } else { "Done" },
                text,
                error,
                scroll: 0,
            }));
        } else {
            self.toast = Some(Toast {
                text,
                error,
                shown: Instant::now(),
            });
        }
    }

    fn finish(
        &mut self,
        reply: Option<Result<Reply, String>>,
        snapshot: Result<Box<Snapshot>, String>,
    ) -> Option<Request> {
        let busy = self.busy.take();
        let automatic = busy.as_ref().is_some_and(|busy| busy.automatic);
        let kind = busy.as_ref().and_then(|busy| busy.kind);
        // The reply's popup must be identified before snapshot errors can change the popup.
        let popup_matches = busy
            .as_ref()
            .is_none_or(|busy| !busy.automatic && busy.popup_generation == self.popup_generation);
        match snapshot {
            Ok(data) => {
                // A refresh follows the newest call or restores an older selection if it is still visible.
                let anchor = self
                    .selected_entry()
                    .filter(|_| self.selected > 0)
                    .map(|entry| (entry.ts.clone(), entry.model.clone(), entry.turn_id.clone()));
                self.data = *data;
                self.refreshed = Some(Instant::now());
                self.rebuild_calls();
                if let Some((ts, model, turn)) = anchor
                    && let Some(position) = self.visible.iter().position(|&index| {
                        let entry = &self.data.entries[index];
                        entry.ts == ts && entry.model == model && entry.turn_id == turn
                    })
                {
                    self.selected = position;
                }
            }
            Err(error) => {
                // Settings offers the actions needed to repair a failed initial load.
                if self.refreshed.is_none() && self.data.errors.is_empty() {
                    self.tab = Tab::Settings;
                }
                // Overview keeps the error after the status message is dismissed.
                self.data.errors = vec![error.clone()];
                if automatic || !popup_matches || self.popup.is_some() {
                    self.toast = Some(Toast {
                        text: error.lines().next().unwrap_or_default().into(),
                        error: true,
                        shown: Instant::now(),
                    });
                } else {
                    self.notify(error, true);
                }
            }
        }
        let reply = reply?;
        match (kind, reply) {
            (Some(ActionKind::ServerKeys), Err(error)) => {
                self.server_keys = None;
                self.report(error, true, popup_matches);
            }
            // A form stays open with the error, so the user can fix the input and submit again.
            (_, Err(error)) => {
                if popup_matches && let Some(form) = self.popup.as_mut().and_then(Popup::form_mut) {
                    form.error = error;
                } else {
                    self.report(error, true, popup_matches);
                }
            }
            (Some(ActionKind::PreviewUpdate), Ok(reply)) => {
                if !popup_matches {
                    self.report(
                        "The source update preview finished. Press u in Settings to review it."
                            .into(),
                        false,
                        false,
                    );
                    return None;
                }
                self.set_popup(Some(Popup::Update {
                    preview: reply.data,
                    scroll: 0,
                    confirm: Form::new(
                        FormKind::Update,
                        "Update from source?",
                        vec![("Type UPDATE to start", String::new())],
                    ),
                }));
            }
            (Some(ActionKind::ServerKeys), Ok(reply)) => {
                let text = key_lines(self.theme, Some(&reply.data))
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("\n");
                self.server_keys = Some(reply.data);
                if !popup_matches {
                    if !automatic {
                        self.report("Checked the saved server keys.".into(), false, false);
                    }
                    return None;
                }
                self.set_popup(Some(Popup::Message {
                    title: "Server keys",
                    text,
                    error: false,
                    scroll: 0,
                }));
            }
            (
                Some(
                    ActionKind::SaveServerKey
                    | ActionKind::ImportServerKey
                    | ActionKind::RemoveServerKey,
                ),
                Ok(reply),
            ) => {
                if popup_matches {
                    self.set_popup(None);
                }
                self.report(reply.message, false, popup_matches);
                let mut request = Request::run(Action::ServerKeys {}, "Checking the server keys");
                request.automatic = true;
                return Some(request);
            }
            (_, Ok(reply)) => {
                if popup_matches
                    && matches!(self.popup, Some(Popup::Form(_) | Popup::Update { .. }))
                {
                    self.set_popup(None);
                }
                let details: Vec<&str> = reply.data["details"]
                    .as_array()
                    .map(|items| items.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                if details.is_empty() {
                    self.report(reply.message, false, popup_matches);
                } else if !popup_matches {
                    self.report(
                        format!("{}\n\n{}", reply.message, details.join("\n")),
                        false,
                        false,
                    );
                } else {
                    self.set_popup(Some(Popup::Message {
                        title: "Details",
                        text: format!("{}\n\n{}", reply.message, details.join("\n")),
                        error: false,
                        scroll: 0,
                    }));
                }
            }
        }
        None
    }

    fn poll_results(
        &mut self,
        results: &mpsc::Receiver<Done>,
        jobs: &mpsc::Sender<Job>,
    ) -> Result<Option<PathBuf>, String> {
        loop {
            match results.try_recv() {
                Ok(Done::Finished { reply, snapshot }) => {
                    if let Some(request) = self.finish(reply, snapshot)
                        && !self.quit
                    {
                        self.start(jobs, request)?;
                    }
                }
                Ok(Done::Editor(path)) => {
                    self.busy = None;
                    return Ok((!self.quit).then_some(path));
                }
                Err(mpsc::TryRecvError::Empty) => return Ok(None),
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err("The terminal operation worker stopped unexpectedly.".into());
                }
            }
        }
    }

    fn refresh_due(&self) -> bool {
        // The initial snapshot has no interval, so refreshes use 30 seconds until one succeeds.
        let seconds = match self.data.refresh_seconds {
            0 => 30,
            seconds => seconds,
        };
        self.busy.is_none() && self.last_refresh.elapsed() >= Duration::from_secs(seconds)
    }

    fn start(&mut self, jobs: &mpsc::Sender<Job>, request: Request) -> Result<(), String> {
        if let Some(busy) = &self.busy {
            let text = format!("Wait for \"{}\" to finish.", busy.label);
            self.notify(text, true);
            return Ok(());
        }
        let kind = match &request.job {
            Job::Refresh => {
                self.last_refresh = Instant::now();
                None
            }
            Job::Run(action) => Some(ActionKind::from(action)),
        };
        jobs.send(request.job).map_err(|error| error.to_string())?;
        self.busy = Some(Busy {
            label: request.label,
            kind,
            started: Instant::now(),
            automatic: request.automatic,
            popup_generation: self.popup_generation,
        });
        Ok(())
    }

    fn keys(&self) -> &'static [Key] {
        match (&self.popup, self.filtering, self.tab) {
            (Some(Popup::Restart), _, _) => CONFIRM_KEYS,
            (Some(Popup::Update { .. }), _, _) => UPDATE_KEYS,
            (Some(Popup::Form(_)), _, _) => FORM_KEYS,
            (Some(_), _, _) => POPUP_KEYS,
            (None, true, _) => FILTER_KEYS,
            (None, false, Tab::Overview) => &[],
            (None, false, Tab::Calls) => CALLS_KEYS,
            (None, false, Tab::Models) => MODELS_KEYS,
            (None, false, Tab::Settings) => SETTINGS_KEYS,
        }
    }

    fn paste(&mut self, text: &str) {
        if let Some(form) = self.popup.as_mut().and_then(Popup::form_mut) {
            form.insert(text);
        } else if self.filtering && self.popup.is_none() {
            self.filter.extend(text.chars().filter(|c| !c.is_control()));
            self.selected = 0;
            self.rebuild_calls();
        }
    }

    fn key(&mut self, key: KeyEvent) -> Option<Request> {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.quit = true;
            return None;
        }
        if self.popup.is_some() {
            return self.popup_key(key);
        }
        if self.filtering {
            self.filter_key(key);
            return None;
        }
        match key.code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Char('?') => self.set_popup(Some(Popup::Help { scroll: 0 })),
            KeyCode::Char(digit @ '1'..='4') => {
                self.tab = TABS[digit as usize - '1' as usize];
            }
            KeyCode::Tab | KeyCode::Right => self.tab = TABS[(self.tab.index() + 1) % TABS.len()],
            KeyCode::BackTab | KeyCode::Left => {
                self.tab = TABS[(self.tab.index() + TABS.len() - 1) % TABS.len()];
            }
            KeyCode::Char('r') | KeyCode::F(5) => return Some(Request::refresh(false)),
            KeyCode::Esc => {
                self.toast = None;
                if self.tab == Tab::Calls {
                    self.filter.clear();
                    self.session = None;
                    self.rebuild_calls();
                }
            }
            code => match self.tab {
                Tab::Overview => {}
                Tab::Calls => self.calls_key(code),
                Tab::Models => match code {
                    KeyCode::Char('p' | ']') => self.period = self.period.next(),
                    KeyCode::Char('[') => self.period = self.period.previous(),
                    _ => {}
                },
                Tab::Settings => return self.settings_key(code),
            },
        }
        None
    }

    fn calls_key(&mut self, code: KeyCode) {
        if let Some(selected) = step(code, self.selected, self.visible.len().saturating_sub(1)) {
            self.selected = selected;
            return;
        }
        match code {
            KeyCode::Char('/') => self.filtering = true,
            KeyCode::Enter => {
                if let Some(entry) = self.selected_entry() {
                    self.set_popup(Some(Popup::Detail {
                        entry: Box::new(entry.clone()),
                        scroll: 0,
                    }));
                }
            }
            KeyCode::Char('s') => {
                let sessions = &self.data.sessions;
                self.session = match &self.session {
                    None => sessions.first().cloned(),
                    Some(current) => sessions
                        .iter()
                        .position(|session| session == current)
                        .and_then(|index| sessions.get(index + 1))
                        .cloned(),
                };
                self.selected = 0;
                self.rebuild_calls();
            }
            _ => {}
        }
    }

    fn filter_key(&mut self, key: KeyEvent) {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Enter => self.filtering = false,
            KeyCode::Esc => {
                self.filtering = false;
                self.filter.clear();
            }
            KeyCode::Backspace => {
                self.filter.pop();
            }
            KeyCode::Char('u') if control => self.filter.clear(),
            KeyCode::Char('w') if control => {
                let kept = self
                    .filter
                    .trim_end()
                    .rfind(' ')
                    .map_or(0, |index| index + 1);
                self.filter.truncate(kept);
            }
            KeyCode::Char(c) if !control && !key.modifiers.contains(KeyModifiers::ALT) => {
                self.filter.push(c);
            }
            _ => return,
        }
        self.selected = 0;
        self.rebuild_calls();
    }

    /// Action letters come first, so `k` checks the server keys here instead of moving up.
    fn settings_key(&mut self, code: KeyCode) -> Option<Request> {
        if let KeyCode::Char(letter) = code
            && let Some(index) = ACTIONS.iter().position(|action| action.0 == letter)
        {
            self.action = index;
            return self.start_action(ACTIONS[index].3);
        }
        if let Some(action) = step(code, self.action, ACTIONS.len() - 1) {
            self.action = action;
            return None;
        }
        if code == KeyCode::Enter {
            return self.start_action(ACTIONS[self.action].3);
        }
        None
    }

    /// This function starts a Settings action. The match has no wildcard arm, so adding a
    /// controller action breaks the build until the terminal app can run it.
    fn start_action(&mut self, kind: ActionKind) -> Option<Request> {
        let form = |kind, title, fields| Some(Popup::Form(Form::new(kind, title, fields)));
        let request = |action, label| Some(Request::run(action, label));
        match kind {
            ActionKind::OpenConfig => request(Action::OpenConfig {}, "Opening the server config"),
            ActionKind::CheckConfig => {
                request(Action::CheckConfig {}, "Checking the server config")
            }
            ActionKind::Restart => {
                self.set_popup(Some(Popup::Restart));
                None
            }
            ActionKind::OpenSettings => {
                request(Action::OpenSettings {}, "Opening the app settings")
            }
            ActionKind::SavePreferences => {
                let (baseline, refresh) = self
                    .data
                    .settings
                    .as_ref()
                    .map(|s| (s.baseline_model.clone(), s.refresh_seconds.to_string()))
                    .unwrap_or_default();
                self.set_popup(form(
                    FormKind::Preferences,
                    "Change preferences",
                    vec![
                        ("Baseline model ID", baseline),
                        ("Refresh interval in seconds", refresh),
                    ],
                ));
                None
            }
            ActionKind::OpenPrices => request(Action::OpenPrices {}, "Opening the model prices"),
            ActionKind::ServerKeys => request(Action::ServerKeys {}, "Checking the server keys"),
            ActionKind::SaveServerKey => {
                self.set_popup(form(
                    FormKind::SaveKey,
                    "Save a server key",
                    vec![("Key name", String::new()), ("Key", String::new())],
                ));
                None
            }
            ActionKind::ImportServerKey => {
                self.set_popup(form(
                    FormKind::ImportKey,
                    "Import a server key from your login shell",
                    vec![("Key name", String::new())],
                ));
                None
            }
            ActionKind::RemoveServerKey => {
                self.set_popup(form(
                    FormKind::RemoveKey,
                    "Remove a saved server key",
                    vec![("Key name", String::new())],
                ));
                None
            }
            // The update window submits Update after the user reads this preview.
            ActionKind::PreviewUpdate | ActionKind::Update => {
                request(Action::PreviewUpdate {}, "Reading the source checkout")
            }
            ActionKind::OpenUpdateLog => {
                request(Action::OpenUpdateLog {}, "Opening the update log")
            }
            ActionKind::OpenLogs => request(Action::OpenLogs {}, "Opening the server logs"),
        }
    }

    fn popup_key(&mut self, key: KeyEvent) -> Option<Request> {
        let popup = self.popup.as_mut()?;
        match popup {
            Popup::Restart => {
                match key.code {
                    KeyCode::Char('y' | 'Y') => {
                        self.set_popup(None);
                        return Some(Request::run(Action::Restart {}, "Restarting the server"));
                    }
                    KeyCode::Char('n' | 'N' | 'q') | KeyCode::Esc | KeyCode::Enter => {
                        self.set_popup(None);
                    }
                    _ => {}
                }
                return None;
            }
            Popup::Form(form) | Popup::Update { confirm: form, .. } => {
                match key.code {
                    KeyCode::Esc => self.set_popup(None),
                    KeyCode::Enter => {
                        let preview = match &self.popup {
                            Some(Popup::Update { preview, .. }) => Some(preview),
                            _ => None,
                        };
                        let form = match &self.popup {
                            Some(Popup::Form(form) | Popup::Update { confirm: form, .. }) => form,
                            _ => return None,
                        };
                        match form.action(preview) {
                            Ok(action) => return Some(Request::run(action, form.label())),
                            Err(error) => {
                                if let Some(form) = self.popup.as_mut().and_then(Popup::form_mut) {
                                    form.error = error;
                                }
                            }
                        }
                    }
                    // Only the update window scrolls; its text field ignores the arrow keys.
                    KeyCode::Up | KeyCode::Down | KeyCode::PageUp | KeyCode::PageDown => {
                        if let Some(scroll) = popup.scroll_mut()
                            && let Some(next) =
                                step(key.code, usize::from(*scroll), usize::from(u16::MAX))
                        {
                            *scroll = next as u16;
                        }
                    }
                    _ => form.edit(key),
                }
                return None;
            }
            _ => {}
        }
        match key.code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q' | '?') => self.set_popup(None),
            // Drawing clamps the scroll to the text, so the last line here has no fixed value.
            code => {
                if let Some(scroll) = popup.scroll_mut()
                    && let Some(next) = step(code, usize::from(*scroll), usize::from(u16::MAX))
                {
                    *scroll = next as u16;
                }
            }
        }
        None
    }
}

/// This function moves a list position for the arrow, vim, page, and jump keys.
fn step(code: KeyCode, current: usize, last: usize) -> Option<usize> {
    Some(match code {
        KeyCode::Down | KeyCode::Char('j') => current.saturating_add(1).min(last),
        KeyCode::Up | KeyCode::Char('k') => current.saturating_sub(1),
        KeyCode::PageDown => current.saturating_add(10).min(last),
        KeyCode::PageUp => current.saturating_sub(10),
        KeyCode::Home | KeyCode::Char('g') => 0,
        KeyCode::End | KeyCode::Char('G') => last,
        _ => return None,
    })
}

/// Every filter word must appear in the call's model, route, session, or turn ID.
/// Words with uppercase letters match case exactly; other words ignore ASCII case.
fn matches_filter(entry: &Entry, filter: &str) -> bool {
    let fields = [
        entry.model.as_str(),
        entry.route_id.as_str(),
        entry.session_id.as_deref().unwrap_or_default(),
        entry.turn_id.as_deref().unwrap_or_default(),
    ];
    filter.split_whitespace().all(|word| {
        let exact = word.chars().any(char::is_uppercase);
        fields.iter().any(|field| {
            if exact {
                field.contains(word)
            } else {
                field
                    .as_bytes()
                    .windows(word.len())
                    .any(|window| window.eq_ignore_ascii_case(word.as_bytes()))
            }
        })
    })
}

fn count(value: u64) -> String {
    let digits = value.to_string();
    let mut text = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            text.push(',');
        }
        text.push(digit);
    }
    text
}

fn plural(value: u64, one: &str, many: &str) -> String {
    format!("{} {}", count(value), if value == 1 { one } else { many })
}

/// Single calls often cost fractions of a cent, so `call_money` shows more digits than `money`.
fn call_money(usd: f64) -> String {
    if usd == 0.0 {
        "$0".into()
    } else if usd >= 0.01 {
        money(usd)
    } else if usd >= 0.0001 {
        format!("${usd:.4}")
    } else {
        "<$0.0001".into()
    }
}

/// This function shows a total only when every call has a cost. Otherwise it shows the known
/// part as a lower bound, or "unknown", so a missing cost never reads as $0.
fn cost(theme: Theme, total: Option<f64>, known: f64) -> Span<'static> {
    match total {
        Some(total) => Span::raw(money(total)),
        None if known > 0.0 => Span::styled(format!("≥ {}", money(known)), theme.warn),
        None => Span::styled("unknown", theme.warn),
    }
}

/// This function shows the savings and their share of the baseline, or "hidden" when
/// `Estimate::savings` has no figure. With `baseline`, it also shows the baseline cost.
fn saved(theme: Theme, estimate: &Estimate, baseline: bool) -> Vec<Span<'static>> {
    let Some(savings) = estimate.savings() else {
        return vec![Span::styled("hidden", theme.warn)];
    };
    let style = if savings.saved() < 0.0 {
        theme.error
    } else {
        theme.ok
    };
    let mut spans = vec![Span::styled(money(savings.saved()), style.bold())];
    if let Some(percent) = savings.percent() {
        let text = if baseline {
            format!(" · {percent:.0}% of {}", money(savings.baseline))
        } else {
            format!(" · {percent:.0}%")
        };
        spans.push(Span::styled(text, theme.muted));
    }
    spans
}

/// This function explains why `Estimate::savings` has no figure, in the order that
/// `summary::build` checks the reasons. Call it only when the savings are hidden.
fn hidden_savings(estimate: &Estimate, baseline: &str) -> String {
    if estimate.baseline.is_none() {
        format!("the price table has no rates for {baseline}")
    } else if estimate.unknown_calls > 0 {
        format!(
            "{} no recorded cost",
            plural(estimate.unknown_calls, "call has", "calls have")
        )
    } else {
        format!(
            "{} no token counts",
            plural(estimate.uncounted_answers, "answer has", "answers have")
        )
    }
}

fn age(ts: &str, now: DateTime<Utc>) -> String {
    let Ok(time) = DateTime::parse_from_rfc3339(ts) else {
        return "—".into();
    };
    let seconds = now.signed_duration_since(time).num_seconds().max(0);
    match seconds {
        0..60 => format!("{seconds}s ago"),
        60..3_600 => format!("{}m ago", seconds / 60),
        3_600..86_400 => format!("{}h ago", seconds / 3_600),
        _ => format!("{}d ago", seconds / 86_400),
    }
}

fn server_state(theme: Theme, running: bool) -> [Span<'static>; 2] {
    [
        Span::styled("● ", if running { theme.ok } else { theme.error }),
        Span::raw(if running { "running" } else { "not responding" }),
    ]
}

/// This function draws a muted label in a fixed-width column, then the value or "Not recorded".
fn labeled(theme: Theme, name: &str, width: usize, value: String) -> Line<'static> {
    let value = if value.is_empty() {
        "Not recorded".to_string()
    } else {
        value
    };
    Line::from(vec![
        Span::styled(format!("{name:<width$}"), theme.muted),
        Span::raw(value),
    ])
}

fn right<'a>(line: impl Into<Line<'a>>) -> Cell<'a> {
    Cell::from(line.into().right_aligned())
}

/// Model tables end with the share column. Narrow terminals drop it so model IDs stay readable;
/// `Table` ignores the extra cell in each row.
fn share_columns(columns: &[Column], area: Rect) -> &[Column] {
    let keep = if area.width >= 80 {
        columns.len()
    } else {
        columns.len() - 1
    };
    &columns[..keep]
}

fn table<'a>(theme: Theme, columns: &[Column], rows: Vec<Row<'a>>) -> Table<'a> {
    let header = columns.iter().map(|&(title, _, align_right)| {
        if align_right {
            right(title)
        } else {
            Cell::from(title)
        }
    });
    Table::new(rows, columns.iter().map(|column| column.1))
        .header(Row::new(header).style(theme.muted))
}

fn block(theme: Theme, title: &str) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme.border)
        .title(Span::styled(format!(" {title} "), Modifier::BOLD))
}

fn draw(frame: &mut Frame, app: &mut App) {
    // This exhaustive destructuring requires an update when Snapshot gains a field.
    let Snapshot {
        update_status: _,   // draw_settings shows the last update status.
        settings: _,        // draw_settings shows the app settings.
        data_state: _,      // draw_overview shows health; draw_models checks usage
        metrics: _,         // draw_header and the Overview cards
        today: _,           // the Overview "Today" card title
        analytics: _,       // draw_top_models and draw_models
        estimates: _,       // the Overview cards, draw_top_models, and draw_models
        baseline_model: _,  // draw_top_models and draw_models
        activity: _,        // draw_activity
        sessions: _,        // The Calls title and the `s` key use these IDs.
        entries: _,         // draw_call_table and the call details show these entries.
        skipped: _,         // calls_block shows the number of unreadable records.
        limited: _,         // calls_block and draw_activity show this limit.
        errors: _,          // draw_errors
        refresh_seconds: _, // App::refresh_due
    } = &app.data;
    let area = frame.area();
    app.width = area.width;
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        draw_too_small(frame, app.theme, area);
        return;
    }
    let [header, body, status] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(1),
        Constraint::Length(1),
    ])
    .areas(area);
    draw_header(frame, app, header);
    match app.tab {
        Tab::Overview => draw_overview(frame, app, body),
        Tab::Calls => draw_calls(frame, app, body),
        Tab::Models => draw_models(frame, app, body),
        Tab::Settings => draw_settings(frame, app, body),
    }
    draw_status(frame, app, status);
    draw_popup(frame, app, body);
}

fn draw_too_small(frame: &mut Frame, theme: Theme, area: Rect) {
    let size = |value: u16, needed: u16| {
        let style = if value < needed {
            theme.error
        } else {
            theme.ok
        };
        Span::styled(value.to_string(), style)
    };
    let text = vec![
        Line::from("Terminal too small"),
        Line::from(vec![
            size(area.width, MIN_WIDTH),
            Span::raw(" × "),
            size(area.height, MIN_HEIGHT),
        ]),
        Line::styled(format!("Needs {MIN_WIDTH} × {MIN_HEIGHT}"), theme.muted),
    ];
    let height = text.len() as u16;
    frame.render_widget(
        Paragraph::new(text).centered(),
        area.centered_vertically(Constraint::Length(height)),
    );
}

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let titles = TABS.map(|tab| {
        Line::from(vec![
            Span::styled(format!("{} ", tab.index() + 1), theme.key),
            Span::raw(tab.name()),
        ])
    });
    // Tabs pads each title by one space on both sides and puts a divider between titles.
    let tabs_width: usize = titles.iter().map(|title| title.width() + 3).sum();
    let running = app.data.metrics.running;
    let mut status = server_state(theme, running).to_vec();
    let updated = match app.refreshed {
        Some(time) => format!(" · updated {}s ago ", time.elapsed().as_secs()),
        None => " · no data yet ".into(),
    };
    let url = format!(" · {}", app.data.metrics.server_url);
    // Narrow terminals drop the URL first, then the update time, so the tabs stay readable.
    let room = usize::from(area.width).saturating_sub(tabs_width);
    let base = status.iter().map(Span::width).sum::<usize>();
    if base > room {
        status[1] = Span::raw(if running { "up" } else { "down" });
    }
    if !app.data.metrics.server_url.is_empty()
        && base + url.chars().count() + updated.chars().count() <= room
    {
        status.push(Span::styled(url, theme.muted));
    }
    if base + updated.chars().count() <= room {
        status.push(Span::styled(updated, theme.muted));
    }
    let status = Line::from(status).right_aligned();
    let [tabs, status_area] = Layout::horizontal([
        Constraint::Fill(1),
        Constraint::Length(status.width() as u16),
    ])
    .areas(area);
    frame.render_widget(
        Tabs::new(titles)
            .select(app.tab.index())
            .highlight_style(theme.selected)
            .divider(Span::styled("│", theme.border)),
        tabs,
    );
    frame.render_widget(Paragraph::new(status), status_area);
}

/// This function draws the Data errors panel at the top of `area` and returns the rest.
fn draw_errors(frame: &mut Frame, app: &App, area: Rect) -> Rect {
    let theme = app.theme;
    let errors = &app.data.errors;
    let lines: Vec<Line> = errors
        .iter()
        .flat_map(|error| error.lines())
        .map(|line| Line::styled(line, theme.error))
        .collect();
    let panel = Paragraph::new(lines).wrap(Wrap { trim: false });
    let height = if errors.is_empty() {
        0
    } else {
        // The panel needs rows for wrapped text as well as its top and bottom borders.
        panel
            .line_count(area.width.saturating_sub(2))
            .saturating_add(2)
            .min(usize::from(area.height / 2)) as u16
    };
    let [errors_area, rest] =
        Layout::vertical([Constraint::Length(height), Constraint::Fill(1)]).areas(area);
    if height > 0 {
        frame.render_widget(
            panel.block(block(theme, "Data errors").border_style(theme.error)),
            errors_area,
        );
    }
    rest
}

fn draw_overview(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let data = &app.data;
    let rest = draw_errors(frame, app, area);
    let usage = data.data_state.usage.state == Availability::Available;
    let chart = if usage && rest.height >= 24 { 9 } else { 0 };
    let [cards, chart_area, models] = Layout::vertical([
        Constraint::Length(5),
        Constraint::Length(chart),
        Constraint::Fill(1),
    ])
    .areas(rest);
    let mut server = vec![
        Line::from(server_state(theme, data.metrics.running).to_vec()),
        Line::styled(data.metrics.server_url.as_str(), theme.muted),
    ];
    if let Some(reason) = data.data_state.health["reason"].as_str() {
        server.push(Line::styled(reason.to_string(), theme.muted));
    }
    // Without usage data, the page shows only the connection so nothing reads as zero usage.
    if !usage {
        let [card, _] = Layout::horizontal([Constraint::Fill(1), Constraint::Fill(2)]).areas(cards);
        draw_card(frame, theme, "Server", server, card);
        return;
    }
    // The header already shows the server state, so narrow terminals give its card to usage.
    let periods = if cards.width >= 100 {
        let [server_card, rest] =
            Layout::horizontal([Constraint::Fill(1), Constraint::Fill(2)]).areas(cards);
        draw_card(frame, theme, "Server", server, server_card);
        rest
    } else {
        cards
    };
    let [today, week] = Layout::horizontal([Constraint::Fill(1); 2]).areas(periods);
    let today_title = format!("Today · {}", data.today.format("%b %-d"));
    for (period, title, card) in [
        (Period::Today, today_title.as_str(), today),
        (Period::Week, Period::Week.name(), week),
    ] {
        let lines = period_lines(app, period, card.width >= 36);
        draw_card(frame, theme, title, lines, card);
    }
    if chart > 0 {
        draw_activity(frame, app, chart_area);
    }
    draw_top_models(frame, app, models);
}

fn draw_card(frame: &mut Frame, theme: Theme, title: &str, lines: Vec<Line>, area: Rect) {
    frame.render_widget(
        Paragraph::new(lines).block(block(theme, title).padding(Padding::horizontal(1))),
        area,
    );
}

/// This function builds one period's calls, spending, and savings for an Overview card.
fn period_lines(app: &App, period: Period, baseline: bool) -> Vec<Line<'static>> {
    let theme = app.theme;
    let counts = match period {
        Period::Today => &app.data.metrics.today,
        _ => &app.data.metrics.week,
    };
    let mut lines = vec![Line::from(vec![
        Span::styled(count(counts.requests), Modifier::BOLD),
        Span::styled(" calls · ", theme.muted),
        Span::styled(tokens(counts.tokens), Modifier::BOLD),
        Span::styled(" tokens", theme.muted),
    ])];
    if let Some(estimate) = period.estimate(&app.data) {
        lines.push(Line::from(vec![
            Span::styled("Spent ", theme.muted),
            cost(theme, estimate.actual, estimate.known).bold(),
        ]));
        let mut savings = vec![Span::styled("Saved ", theme.muted)];
        savings.extend(saved(theme, &estimate, baseline));
        lines.push(Line::from(savings));
    }
    lines
}

fn draw_activity(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let data = &app.data.activity;
    let peak = data.iter().copied().max().unwrap_or_default();
    let mut outer = block(theme, "Calls per hour · past 24h")
        .title_top(Line::styled(format!(" peak {} ", count(peak)), theme.muted).right_aligned())
        .title_bottom(Line::styled(" 24h ago ", theme.muted))
        .title_bottom(Line::styled(" now ", theme.muted).right_aligned());
    if app.data.limited {
        outer = outer
            .title_bottom(Line::styled(" Recent history is limited. ", theme.muted).centered());
    }
    let inner = outer.inner(area);
    frame.render_widget(outer, area);
    let bars = data.len() as u16;
    let gap = u16::from(inner.width >= bars * 3);
    let width = ((inner.width + gap) / bars).saturating_sub(gap).max(1);
    let chart = inner.centered_horizontally(Constraint::Length(bars * (width + gap) - gap));
    let bars: Vec<Bar> = data
        .iter()
        .map(|&value| Bar::new(value).text_value(String::new()))
        .collect();
    frame.render_widget(
        BarChart::new(bars)
            .bar_width(width)
            .bar_gap(gap)
            .bar_style(theme.bar),
        chart,
    );
}

/// Models are ordered by their share of the period's tokens; the model ID breaks ties.
fn by_tokens(models: &BTreeMap<String, ModelTokens>) -> Vec<(&str, &ModelTokens)> {
    let mut rows: Vec<_> = models
        .iter()
        .map(|(name, usage)| (name.as_str(), usage))
        .collect();
    rows.sort_by(|left, right| {
        right
            .1
            .total()
            .cmp(&left.1.total())
            .then(left.0.cmp(right.0))
    });
    rows
}

/// This function builds a model row. With `kind`, the row also names the answer or routing kind.
fn model_row(
    theme: Theme,
    model: &str,
    usage: &ModelTokens,
    total: f64,
    routing: bool,
    kind: bool,
) -> Row<'static> {
    let share = usage.total() as f64 / total;
    let filled = ((share * 10.0).round() as usize).min(10);
    let mut cells = vec![Cell::from(Span::styled(
        model.to_string(),
        if routing { theme.muted } else { Style::new() },
    ))];
    if kind {
        cells.push(Cell::from(Span::styled(
            if routing { "routing" } else { "answer" },
            theme.muted,
        )));
    }
    cells.extend([
        right(count(usage.requests)),
        right(tokens(usage.total())),
        right(cost(theme, usage.cost.total_usd, usage.cost.known_usd)),
        Cell::from(Line::from(vec![
            Span::styled(
                format!("{}{}", "█".repeat(filled), "░".repeat(10 - filled)),
                theme.bar,
            ),
            Span::styled(format!(" {:>3.0}%", share * 100.0), theme.muted),
        ])),
    ]);
    Row::new(cells)
}

fn draw_top_models(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let week = &app.data.analytics.week;
    let mut notes = Vec::new();
    if !app.data.baseline_model.is_empty() {
        notes.push(Line::styled(
            format!(
                "Savings compare answer calls with {} prices.",
                app.data.baseline_model
            ),
            theme.muted,
        ));
    }
    for period in [Period::Today, Period::Week] {
        if let Some(estimate) = period.estimate(&app.data)
            && estimate.savings().is_none()
        {
            notes.push(Line::styled(
                format!(
                    "{}: savings hidden because {}.",
                    period.name(),
                    hidden_savings(&estimate, &app.data.baseline_model)
                ),
                theme.warn,
            ));
        }
    }
    let outer = block(theme, "Models · past 7 days");
    let inner = outer.inner(area);
    frame.render_widget(outer, area);
    let total = week.tokens().max(1) as f64;
    let mut rows: Vec<Row> = by_tokens(&week.routed)
        .into_iter()
        .map(|(model, usage)| model_row(theme, model, usage, total, false, false))
        .collect();
    if !week.classifier.is_empty() {
        let mut routing = ModelTokens::default();
        for usage in week.classifier.values() {
            routing.add(usage);
        }
        rows.push(model_row(
            theme,
            "routing calls",
            &routing,
            total,
            true,
            false,
        ));
    }
    let height = if week.is_empty() { 1 } else { rows.len() + 1 };
    let [table_area, _, notes_area] = Layout::vertical([
        Constraint::Length(height as u16),
        Constraint::Length(1),
        Constraint::Fill(1),
    ])
    .areas(inner);
    frame.render_widget(Paragraph::new(notes).wrap(Wrap { trim: false }), notes_area);
    if week.is_empty() {
        let text = if app.data.analytics.all.is_empty() {
            "No calls recorded yet."
        } else {
            "No calls recorded in the past 7 days."
        };
        frame.render_widget(Paragraph::new(text).style(theme.muted), table_area);
        return;
    }
    let columns = [
        ("MODEL", Constraint::Fill(1), false),
        ("CALLS", Constraint::Length(8), true),
        ("TOKENS", Constraint::Length(8), true),
        ("COST", Constraint::Length(10), true),
        ("SHARE", Constraint::Length(17), false),
    ];
    frame.render_widget(
        table(theme, share_columns(&columns, inner), rows),
        table_area,
    );
}

fn draw_calls(frame: &mut Frame, app: &mut App, area: Rect) {
    let area = draw_errors(frame, app, area);
    if app.data.data_state.history.state == Availability::Unavailable {
        return;
    }
    if area.width >= SIDE_DETAIL_WIDTH {
        let [table, detail] =
            Layout::horizontal([Constraint::Fill(1), Constraint::Length(48)]).areas(area);
        draw_call_table(frame, app, table);
        let lines = app
            .selected_entry()
            .map(|entry| detail_lines(app.theme, entry))
            .unwrap_or_else(|| vec![Line::styled("No call selected.", app.theme.muted)]);
        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(block(app.theme, "Call details").padding(Padding::horizontal(1))),
            detail,
        );
    } else {
        draw_call_table(frame, app, area);
    }
}

fn calls_block(app: &App) -> Block<'static> {
    let theme = app.theme;
    let mut title = vec![
        Span::styled(" Calls ", Modifier::BOLD),
        Span::styled(
            format!("[{}] ", count(app.visible.len() as u64)),
            theme.muted,
        ),
    ];
    if let Some(session) = &app.session {
        let note = if app.data.sessions.contains(session) {
            ""
        } else {
            " (not in recent calls)"
        };
        title.push(Span::styled(format!("session {session}{note} "), theme.key));
    }
    if !app.filter.is_empty() || app.filtering {
        title.push(Span::styled(format!("/{} ", app.filter), theme.key));
    }
    let mut outer = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(if app.filtering {
            theme.key
        } else {
            theme.border
        })
        .title(Line::from(title));
    if !app.visible.is_empty() {
        outer = outer.title_bottom(
            Line::styled(
                format!(
                    " {} of {} ",
                    app.selected + 1,
                    count(app.visible.len() as u64)
                ),
                theme.muted,
            )
            .right_aligned(),
        );
    }
    let mut notes = Vec::new();
    if app.data.limited {
        notes.push("latest 5,000 calls".to_string());
    }
    if app.data.skipped > 0 {
        notes.push(format!(
            "{} skipped",
            plural(
                app.data.skipped as u64,
                "unreadable record",
                "unreadable records"
            )
        ));
    }
    if !notes.is_empty() {
        outer = outer.title_bottom(Line::styled(
            format!(" {} ", notes.join(" · ")),
            theme.muted,
        ));
    }
    outer
}

fn draw_call_table(frame: &mut Frame, app: &mut App, area: Rect) {
    // The table shows this many calls; the borders and the header row take three rows.
    let height = usize::from(area.height.saturating_sub(3)).max(1);
    if app.selected < app.offset {
        app.offset = app.selected;
    } else if app.selected >= app.offset + height {
        app.offset = app.selected + 1 - height;
    }
    app.offset = app.offset.min(app.visible.len().saturating_sub(height));
    let app = &*app;
    let theme = app.theme;
    let outer = calls_block(app);
    if app.visible.is_empty() {
        let text = if app.data.entries.is_empty() {
            "No calls recorded yet."
        } else {
            "No calls match the session and filter. Press Esc to clear them."
        };
        frame.render_widget(
            Paragraph::new(Line::styled(text, theme.muted))
                .centered()
                .block(outer.padding(Padding::top(1))),
            area,
        );
        return;
    }
    let wide = area.width >= 90;
    let shown = |column: &(Column, bool)| wide || !column.1;
    let now = Utc::now();
    let rows: Vec<Row> = app.visible[app.offset..]
        .iter()
        .take(height)
        .map(|&index| {
            let entry = &app.data.entries[index];
            let usage = ModelTokens::from(entry);
            let routing = entry.tier == CLASSIFIER_TIER;
            let dim = if routing { theme.muted } else { Style::new() };
            let cells = [
                right(Span::styled(age(&entry.ts, now), theme.muted)),
                Cell::from(Span::styled(entry.model.clone(), dim)),
                Cell::from(entry.route_id.clone()),
                Cell::from(Span::styled(
                    if routing { "routing" } else { "answer" },
                    theme.muted,
                )),
                right(tokens(usage.input)),
                right(tokens(usage.cached_input)),
                right(tokens(usage.output)),
                match entry.cost {
                    Some(cost) => right(call_money(cost.usd)),
                    None => right(Span::styled("—", theme.warn)),
                },
            ];
            Row::new(
                cells
                    .into_iter()
                    .zip(&CALL_COLUMNS)
                    .filter(|(_, column)| shown(column))
                    .map(|(cell, _)| cell),
            )
        })
        .collect();
    let columns: Vec<Column> = CALL_COLUMNS
        .iter()
        .filter(|column| shown(column))
        .map(|column| column.0)
        .collect();
    let mut state = TableState::default().with_selected(Some(app.selected - app.offset));
    frame.render_stateful_widget(
        table(theme, &columns, rows)
            .row_highlight_style(theme.selected)
            .highlight_symbol("› ")
            .block(outer),
        area,
        &mut state,
    );
    if app.visible.len() > height {
        let mut scroll = ScrollbarState::new(app.visible.len()).position(app.selected);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .thumb_style(theme.key)
                .track_symbol(Some("│"))
                .track_style(theme.border),
            // The scrollbar covers the data rows, below the top border and the header row.
            Rect {
                y: area.y + 2,
                height: area.height.saturating_sub(3),
                ..area
            },
            &mut scroll,
        );
    }
}

fn detail_lines(theme: Theme, entry: &Entry) -> Vec<Line<'static>> {
    let usage = ModelTokens::from(entry);
    let field = |name: &str, value: String| labeled(theme, name, 9, value);
    let time = DateTime::parse_from_rfc3339(&entry.ts).map_or_else(
        |_| entry.ts.clone(),
        |time| {
            format!(
                "{} ({})",
                time.with_timezone(&chrono::Local)
                    .format("%Y-%m-%d %H:%M:%S"),
                age(&entry.ts, Utc::now())
            )
        },
    );
    let kind = if entry.tier == CLASSIFIER_TIER {
        "Routing call (classifier or judge)"
    } else {
        "Answer call"
    };
    let mut lines = vec![
        field("Time", time),
        field("Model", entry.model.clone()),
        field("Kind", kind.into()),
        field("Route", entry.route_id.clone()),
        field("Session", entry.session_id.clone().unwrap_or_default()),
        field("Turn", entry.turn_id.clone().unwrap_or_default()),
        Line::default(),
        field(
            "Input",
            format!(
                "{} uncached · {} cached",
                count(usage.input),
                count(usage.cached_input)
            ),
        ),
    ];
    if usage.cache_write > 0 {
        lines.push(field(
            "Cache",
            format!(
                "{} written ({} at the 1-hour rate)",
                count(usage.cache_write),
                count(usage.cache_write_1h)
            ),
        ));
    }
    lines.push(field("Output", count(usage.output)));
    lines.push(match entry.cost {
        Some(RecordedCost { usd, source }) => field(
            "Cost",
            format!(
                "{} · {}",
                call_money(usd),
                match source {
                    CostSource::ProviderReported => "reported by the provider",
                    _ => "estimated from token prices",
                }
            ),
        ),
        None => Line::from(vec![
            Span::styled(format!("{:<9}", "Cost"), theme.muted),
            Span::styled("Not recorded", theme.warn),
        ]),
    });
    lines
}

fn draw_models(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let area = draw_errors(frame, app, area);
    if app.data.data_state.usage.state == Availability::Unavailable {
        return;
    }
    let totals = app.period.totals(&app.data.analytics);
    let mut summary = vec![
        Span::styled(count(totals.requests()), Modifier::BOLD),
        Span::styled(" answer calls · ", theme.muted),
        Span::styled(tokens(totals.tokens()), Modifier::BOLD),
        Span::styled(" tokens", theme.muted),
    ];
    if let Some(estimate) = app.period.estimate(&app.data) {
        summary.push(Span::styled(" · spent ", theme.muted));
        summary.push(cost(theme, estimate.actual, estimate.known).bold());
        summary.push(Span::styled(" · saved ", theme.muted));
        summary.extend(saved(theme, &estimate, true));
        if estimate.savings().is_none() {
            summary.push(Span::styled(
                format!(
                    " because {}",
                    hidden_savings(&estimate, &app.data.baseline_model)
                ),
                theme.warn,
            ));
        }
    }
    let outer = block(theme, &format!("Models · {}", app.period.name())).title_top(
        Line::from(vec![
            Span::styled(" p ", theme.key),
            Span::styled("period ", theme.muted),
        ])
        .right_aligned(),
    );
    let inner = outer.inner(area);
    frame.render_widget(outer, area);
    let total = totals.tokens().max(1) as f64;
    let rows: Vec<Row> = by_tokens(&totals.routed)
        .into_iter()
        .map(|(model, usage)| model_row(theme, model, usage, total, false, true))
        .chain(
            by_tokens(&totals.classifier)
                .into_iter()
                .map(|(model, usage)| model_row(theme, model, usage, total, true, true)),
        )
        .collect();
    // The models table stops early enough to leave the routes heading and one route visible.
    let routes_height = (totals.routes.len() as u16 + 1).min(2);
    let models_height = (rows.len() as u16 + 1).min(inner.height.saturating_sub(3 + routes_height));
    let [summary_area, models_area, _, routes_area] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(models_height),
        Constraint::Length(1),
        Constraint::Fill(1),
    ])
    .areas(inner);
    frame.render_widget(
        Paragraph::new(Line::from(summary)).wrap(Wrap { trim: false }),
        summary_area,
    );
    if totals.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::styled(
                "No calls recorded in this period.",
                theme.muted,
            )),
            models_area,
        );
        return;
    }
    let columns = [
        ("MODEL", Constraint::Fill(1), false),
        ("KIND", Constraint::Length(8), false),
        ("CALLS", Constraint::Length(8), true),
        ("TOKENS", Constraint::Length(8), true),
        ("COST", Constraint::Length(10), true),
        ("SHARE OF TOKENS", Constraint::Length(17), false),
    ];
    frame.render_widget(
        table(theme, share_columns(&columns, inner), rows),
        models_area,
    );
    let routes: Vec<Row> = by_tokens(&totals.routes)
        .into_iter()
        .map(|(route, usage)| {
            Row::new(vec![
                Cell::from(
                    if route.is_empty() {
                        "Not recorded"
                    } else {
                        route
                    }
                    .to_string(),
                ),
                right(count(usage.requests)),
                right(tokens(usage.total())),
                right(cost(theme, usage.cost.total_usd, usage.cost.known_usd)),
            ])
        })
        .collect();
    let columns = [
        (
            "ROUTE (answer and routing calls)",
            Constraint::Fill(1),
            false,
        ),
        ("CALLS", Constraint::Length(8), true),
        ("TOKENS", Constraint::Length(8), true),
        ("COST", Constraint::Length(10), true),
    ];
    frame.render_widget(table(theme, &columns, routes), routes_area);
}

/// This function lists the server key variables from the last `ServerKeys` reply.
fn key_lines(theme: Theme, keys: Option<&Value>) -> Vec<Line<'static>> {
    let Some(data) = keys else {
        return vec![Line::styled(
            "Press k to check which keys are saved.",
            theme.muted,
        )];
    };
    let keys = data["keys"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    if keys.is_empty() {
        return vec![Line::from(
            "The server config names no keys with api_key_env.",
        )];
    }
    let mut lines = vec![Line::styled(
        format!(
            "Stored in {}",
            data["store"].as_str().unwrap_or("Not recorded")
        ),
        theme.muted,
    )];
    lines.extend(keys.iter().map(|key| {
        let saved = key["saved"] == true;
        Line::from(vec![
            Span::raw(format!("{} ", key["name"].as_str().unwrap_or_default())),
            if saved {
                Span::styled("saved", theme.ok)
            } else {
                Span::styled("not saved", theme.warn)
            },
        ])
    }));
    lines
}

fn draw_settings(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let [list_area, info_area] =
        Layout::horizontal([Constraint::Length(30), Constraint::Fill(1)]).areas(area);
    let items: Vec<ListItem> = ACTIONS
        .iter()
        .map(|(letter, name, ..)| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{letter} "), theme.key),
                Span::raw(*name),
            ]))
        })
        .collect();
    let mut state = ListState::default().with_selected(Some(app.action));
    frame.render_stateful_widget(
        List::new(items)
            .highlight_style(theme.selected)
            .highlight_symbol("› ")
            .block(block(theme, "Actions")),
        list_area,
        &mut state,
    );
    let row = |name: &str, value: String| labeled(theme, name, 15, value);
    let heading = |text: &'static str| Line::styled(text, theme.muted.bold());
    let mut lines = vec![Line::from(ACTIONS[app.action].2), Line::default()];
    if !app.data.errors.is_empty() {
        lines.extend(
            app.data
                .errors
                .iter()
                .map(|error| Line::styled(error.clone(), theme.error)),
        );
        lines.push(Line::default());
    }
    // This destructuring has no `..`, so adding an app setting requires an update here.
    match &app.data.settings {
        Some(AppSettings {
            path,
            server_url,
            routing_log,
            config_file,
            launchd_label,
            refresh_seconds,
            baseline_model,
            prices_file,
        }) => lines.extend([
            heading("APP SETTINGS"),
            row("Settings file", path.display().to_string()),
            row("Server URL", server_url.clone()),
            row("Server config", config_file.display().to_string()),
            row("Routing log", routing_log.display().to_string()),
            row("Service label", launchd_label.clone()),
            row("Baseline model", baseline_model.clone()),
            row(
                "Refresh",
                format!("every {}", plural(*refresh_seconds, "second", "seconds")),
            ),
            row(
                "Model prices",
                prices_file
                    .as_ref()
                    .map_or("Bundled rates".into(), |file| file.display().to_string()),
            ),
            row("Editor", editor()),
        ]),
        None => lines.push(Line::styled(
            "App settings could not be loaded. Press a to open them, then r to check them again.",
            theme.warn,
        )),
    }
    let update = &app.data.update_status;
    let update = ["state", "stage", "message"]
        .into_iter()
        .filter_map(|field| update[field].as_str())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" · ");
    lines.extend([Line::default(), heading("UPDATE FROM SOURCE")]);
    lines.push(Line::from(if update.is_empty() {
        "No update has run".to_string()
    } else {
        update
    }));
    lines.extend([Line::default(), heading("SERVER KEYS")]);
    lines.extend(key_lines(theme, app.server_keys.as_ref()));
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block(theme, ACTIONS[app.action].1).padding(Padding::horizontal(1))),
        info_area,
    );
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let left = if app.filtering {
        Line::from(vec![
            Span::styled(" / ", theme.key),
            Span::raw(app.filter.clone()),
            Span::styled("▏", theme.key),
            Span::styled(
                format!(
                    "  {} ",
                    plural(app.visible.len() as u64, "match", "matches")
                ),
                theme.muted,
            ),
        ])
    } else if let Some(busy) = &app.busy {
        let frame_index = busy.started.elapsed().as_millis() / 100 % SPINNER.len() as u128;
        let mut spans = vec![
            Span::styled(format!(" {} ", SPINNER[frame_index as usize]), theme.key),
            Span::raw(format!("{}…", busy.label)),
        ];
        if app.quit {
            spans.push(Span::styled(" quitting when done", theme.muted));
        }
        Line::from(spans)
    } else if let Some(toast) = &app.toast {
        let style = if toast.error { theme.error } else { theme.ok };
        Line::from(vec![
            Span::styled(if toast.error { " ✗ " } else { " ✓ " }, style),
            Span::styled(toast.text.clone(), style),
        ])
    } else {
        Line::default()
    };
    let room = usize::from(area.width).saturating_sub(left.width() + 2);
    let global = app.popup.is_none() && !app.filtering;
    let hints = fit_keys(theme, app.keys(), global, room);
    let [left_area, right_area] =
        Layout::horizontal([Constraint::Length(left.width() as u16), Constraint::Fill(1)])
            .areas(area);
    frame.render_widget(Paragraph::new(left), left_area);
    frame.render_widget(Paragraph::new(hints.right_aligned()), right_area);
}

/// This function adds page and global key hints while they fit in the status bar.
/// It reserves room for the help key when global hints are shown.
fn fit_keys(theme: Theme, keys: &[Key], global: bool, room: usize) -> Line<'static> {
    let hint = |key: &'static str, label: &str| {
        [
            Span::styled(key, theme.key),
            Span::styled(format!(" {label}  "), theme.muted),
        ]
    };
    const HELP_WIDTH: usize = "? help  ".len();
    let mut spans = Vec::new();
    let mut used = if global { HELP_WIDTH } else { 0 };
    let globals = if global { GLOBAL_KEYS } else { &[] };
    for (key, label, _) in keys.iter().chain(globals) {
        if *key == "?" {
            continue;
        }
        let width = key.chars().count() + label.chars().count() + 3;
        if used + width > room {
            break;
        }
        used += width;
        spans.extend(hint(key, label));
    }
    if global && room >= HELP_WIDTH {
        spans.extend(hint("?", "help"));
    }
    Line::from(spans)
}

/// This function draws a form's fields with a cursor in the selected one, then its error.
fn form_lines(theme: Theme, form: &Form) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = form
        .fields
        .iter()
        .enumerate()
        .map(|(index, (label, value))| {
            // The key field shows one bullet per character, so the key never appears on screen.
            let mut value = if form.kind == FormKind::SaveKey && index == 1 {
                "•".repeat(value.chars().count())
            } else {
                value.clone()
            };
            let selected = index == form.index;
            if selected {
                let cursor = value
                    .char_indices()
                    .nth(form.cursor)
                    .map_or(value.len(), |(byte, _)| byte);
                value.insert(cursor, '▏');
            }
            Line::from(vec![
                Span::styled(
                    if selected { "› " } else { "  " },
                    if selected { theme.key } else { theme.muted },
                ),
                Span::styled(format!("{label}: "), theme.muted),
                Span::raw(value),
            ])
        })
        .collect();
    if !form.error.is_empty() {
        lines.push(Line::default());
        lines.push(Line::styled(form.error.clone(), theme.error));
    }
    lines
}

fn draw_popup(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = app.theme;
    let Some(popup) = &app.popup else {
        return;
    };
    let footer = |pairs: &[(&'static str, &'static str)]| {
        let mut spans = Vec::new();
        for (key, label) in pairs {
            spans.push(Span::styled(format!(" {key} "), theme.key));
            spans.push(Span::raw(*label));
        }
        spans.push(Span::raw(" "));
        Line::from(spans)
    };
    let close = footer(&[("esc", "close")]);
    // The close key stays on the border while the popup text scrolls.
    let (title, lines, pinned, border, preferred, footer) = match popup {
        Popup::Help { .. } => {
            let width = 76;
            let text_width = width.min(area.width.saturating_sub(2)).saturating_sub(4);
            let lines = help_lines(theme, usize::from(text_width));
            ("Help", lines, Vec::new(), theme.key, width, close)
        }
        Popup::Detail { entry, .. } => (
            "Call details",
            detail_lines(theme, entry),
            Vec::new(),
            theme.key,
            72,
            close,
        ),
        Popup::Restart => (
            "Restart the server?",
            vec![
                Line::from(format!(
                    "This checks the server config, then restarts the Switchyard service at {}.",
                    app.data.metrics.server_url
                )),
                Line::from("Requests in progress fail."),
            ],
            Vec::new(),
            theme.warn,
            64,
            footer(&[("y", "Restart"), ("n", "Cancel")]),
        ),
        Popup::Form(form) => {
            let mut lines = Vec::new();
            if matches!(
                form.kind,
                FormKind::SaveKey | FormKind::ImportKey | FormKind::RemoveKey
            ) {
                lines.push(Line::styled("Server keys", theme.muted));
                lines.extend(key_lines(theme, app.server_keys.as_ref()));
                lines.push(Line::default());
            }
            (
                form.title,
                lines,
                form_lines(theme, form),
                theme.key,
                72,
                footer(&[("⏎", "Submit"), ("esc", "Cancel")]),
            )
        }
        Popup::Update {
            preview, confirm, ..
        } => {
            let mut pinned = vec![Line::default()];
            pinned.extend(form_lines(theme, confirm));
            (
                confirm.title,
                update_lines(theme, preview),
                pinned,
                theme.warn,
                84,
                footer(&[("⏎", "Start"), ("esc", "Cancel")]),
            )
        }
        Popup::Message {
            title, text, error, ..
        } => (
            *title,
            text.lines()
                .map(|line| Line::from(line.to_string()))
                .collect(),
            Vec::new(),
            if *error { theme.error } else { theme.key },
            76,
            close,
        ),
    };
    let width = preferred.min(area.width.saturating_sub(2));
    // The text area is inside the two border columns and the two padding columns.
    let text_width = width.saturating_sub(4);
    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    let pinned = Paragraph::new(pinned).wrap(Wrap { trim: false });
    let rows = paragraph.line_count(text_width) as u16;
    let pinned_rows = pinned.line_count(text_width) as u16;
    let rect = area.centered(
        Constraint::Length(width),
        Constraint::Length((rows + pinned_rows + 2).min(area.height.saturating_sub(2))),
    );
    let outer = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(border)
        .title(Span::styled(format!(" {title} "), border.bold()))
        .title_bottom(footer.right_aligned())
        .padding(Padding::horizontal(1));
    let [text_area, pinned_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(pinned_rows)])
            .areas(outer.inner(rect));
    // Scrolling stops at the last line, so scrolling back up responds at once.
    let last = rows.saturating_sub(text_area.height);
    let scroll = match app.popup.as_mut().and_then(Popup::scroll_mut) {
        Some(scroll) => {
            *scroll = (*scroll).min(last);
            *scroll
        }
        None => 0,
    };
    frame.render_widget(Clear, rect);
    frame.render_widget(outer, rect);
    frame.render_widget(paragraph.scroll((scroll, 0)), text_area);
    frame.render_widget(pinned, pinned_area);
}

fn update_lines(theme: Theme, preview: &Value) -> Vec<Line<'static>> {
    // `git status --short` starts some lines with a space, so only trailing space is removed.
    let field = |name: &str| {
        preview[name]
            .as_str()
            .unwrap_or_default()
            .trim_end()
            .to_string()
    };
    let row = |name: &str| labeled(theme, name, 10, field(name.to_lowercase().as_str()));
    let mut lines = vec![
        row("Source"),
        row("Branch"),
        row("Revision"),
        Line::default(),
        Line::styled("Uncommitted changes", theme.muted),
    ];
    let changes = field("changes");
    if changes.is_empty() {
        lines.push(Line::from("None"));
    } else {
        lines.extend(changes.lines().map(|line| Line::from(line.to_string())));
    }
    lines.push(Line::default());
    lines.push(Line::styled("Plan", theme.muted));
    lines.extend(
        field("plan")
            .lines()
            .map(|line| Line::from(line.to_string())),
    );
    if let Some(message) = preview["status"]["message"].as_str() {
        lines.push(Line::default());
        lines.push(Line::styled(format!("Last update: {message}"), theme.warn));
    }
    lines.push(Line::default());
    lines.push(Line::from("The app restarts after a successful install."));
    lines
}

/// The help window indents wrapped descriptions under their first line, so keys stay in one column.
fn help_lines(theme: Theme, width: usize) -> Vec<Line<'static>> {
    const KEY_WIDTH: usize = 8;
    let room = width.saturating_sub(KEY_WIDTH).max(20);
    let mut lines = Vec::new();
    for (section, keys) in [
        ("Every page", GLOBAL_KEYS),
        ("Calls", CALLS_KEYS),
        ("While filtering calls", FILTER_KEYS),
        ("Models", MODELS_KEYS),
        ("Settings", SETTINGS_KEYS),
    ] {
        if !lines.is_empty() {
            lines.push(Line::default());
        }
        lines.push(Line::styled(format!(" {section} "), theme.selected.bold()));
        for (key, _, about) in keys {
            let mut rows = Vec::new();
            let mut row = String::new();
            for word in about.split_whitespace() {
                if !row.is_empty() && row.chars().count() + 1 + word.chars().count() > room {
                    rows.push(std::mem::take(&mut row));
                }
                if !row.is_empty() {
                    row.push(' ');
                }
                row.push_str(word);
            }
            rows.push(row);
            for (index, row) in rows.into_iter().enumerate() {
                let key = if index == 0 { *key } else { "" };
                lines.push(Line::from(vec![
                    Span::styled(format!("{key:<KEY_WIDTH$}"), theme.key),
                    Span::raw(row),
                ]));
            }
        }
    }
    lines
}

struct PasteMode;
impl Drop for PasteMode {
    fn drop(&mut self) {
        let _ = ratatui::crossterm::execute!(std::io::stdout(), event::DisableBracketedPaste);
    }
}

/// This function returns $VISUAL, then $EDITOR, then the platform default.
fn editor() -> String {
    ["VISUAL", "EDITOR"]
        .into_iter()
        .filter_map(|name| std::env::var(name).ok())
        .find(|value| !value.trim().is_empty())
        .unwrap_or_else(|| if cfg!(windows) { "notepad" } else { "vi" }.into())
}

/// This function builds the editor command using Windows or POSIX quoting without starting it.
/// The Windows file check keeps an existing unquoted executable path with spaces as one program.
fn editor_command(editor: &str, windows: bool) -> Result<std::process::Command, String> {
    let parts = if windows {
        let whole = editor.trim();
        if whole.is_empty() {
            Vec::new()
        } else if Path::new(whole).is_file() {
            vec![whole.to_string()]
        } else {
            windows_args::Args::parse_cmd(whole).collect()
        }
    } else {
        shlex::split(editor).ok_or("The editor command has an unmatched quote.")?
    };
    let (program, arguments) = parts
        .split_first()
        .ok_or("Set VISUAL or EDITOR to an editor command.")?;
    let mut command = std::process::Command::new(program);
    command.args(arguments);
    Ok(command)
}

/// This function runs the editor on an absolute path. The caller restores normal terminal input
/// first. The path stays one literal argument, even if its name starts with '-'.
fn edit_file(path: &Path) -> Result<(), String> {
    let mut command = editor_command(&editor(), cfg!(windows))?;
    let program = command.get_program().to_string_lossy().into_owned();
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(path)
    };
    let status = command
        .arg(path)
        .status()
        .map_err(|error| format!("Could not start the editor {program}: {error}"))?;
    if !status.success() {
        return Err(format!("The editor {program} exited with {status}."));
    }
    Ok(())
}

/// This function gives the terminal to the editor and takes it back afterward. The outer error
/// means the terminal could not be switched; the inner result comes from the editor.
fn edit_in_foreground(
    terminal: &mut DefaultTerminal,
    path: &Path,
) -> std::io::Result<Result<(), String>> {
    use ratatui::crossterm::{
        execute,
        terminal::{EnterAlternateScreen, enable_raw_mode},
    };
    execute!(std::io::stdout(), event::DisableBracketedPaste)?;
    terminal.show_cursor()?;
    ratatui::try_restore()?;
    let edited = edit_file(path);
    enable_raw_mode()?;
    execute!(
        std::io::stdout(),
        EnterAlternateScreen,
        event::EnableBracketedPaste
    )?;
    terminal.clear()?;
    Ok(edited)
}

pub fn run(mut controller: Controller) -> Result<(), String> {
    let mut app = App::new(Theme::new(color_enabled()));
    ratatui::crossterm::execute!(std::io::stdout(), event::EnableBracketedPaste)
        .map_err(|e| e.to_string())?;
    let _paste_mode = PasteMode;
    let (jobs, inbox) = mpsc::channel::<Job>();
    let (done, results) = mpsc::channel::<Done>();
    // One worker owns the controller while the terminal continues to draw and read input.
    let worker = thread::spawn(move || {
        while let Ok(job) = inbox.recv() {
            let finished = match job {
                Job::Refresh => Done::Finished {
                    reply: None,
                    snapshot: controller.snapshot().map(Box::new),
                },
                Job::Run(action) => match controller.editor_target(&action) {
                    Ok(Some(path)) => Done::Editor(path),
                    target => Done::Finished {
                        reply: Some(target.and_then(|_| controller.dispatch(action))),
                        snapshot: controller.snapshot().map(Box::new),
                    },
                },
            };
            if done.send(finished).is_err() {
                break;
            }
        }
    });
    let result = ratatui::run(|terminal| -> Result<(), String> {
        // The initial refresh starts after terminal setup so a blocked read cannot delay drawing.
        app.start(&jobs, Request::refresh(true))?;
        loop {
            if let Some(path) = app.poll_results(&results, &jobs)? {
                match edit_in_foreground(terminal, &path).map_err(|error| error.to_string())? {
                    Ok(()) => {
                        app.notify(format!("Closed the editor for {}.", path.display()), false)
                    }
                    Err(error) => app.notify(error, true),
                }
                app.start(&jobs, Request::refresh(false))?;
            }
            if app.quit && app.busy.is_none() {
                return Ok(());
            }
            if app
                .toast
                .as_ref()
                .is_some_and(|toast| !toast.error && toast.shown.elapsed() > TOAST_TIME)
            {
                app.toast = None;
            }
            terminal
                .draw(|frame| draw(frame, &mut app))
                .map_err(|e| e.to_string())?;
            // The spinner needs a faster redraw only while a job runs.
            let wait = Duration::from_millis(if app.busy.is_some() { 100 } else { 250 });
            if event::poll(wait).map_err(|e| e.to_string())? {
                match event::read().map_err(|e| e.to_string())? {
                    Event::Paste(text) => app.paste(&text),
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        if let Some(request) = app.key(key) {
                            app.start(&jobs, request)?;
                        }
                    }
                    _ => {}
                }
            }
            if !app.quit && app.refresh_due() {
                app.start(&jobs, Request::refresh(true))?;
            }
        }
    });
    drop(jobs);
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
    use super::{
        ACTIONS, Action, ActionKind, App, Availability, BTreeMap, Done, Estimate, Form, FormKind,
        KeyCode, KeyEvent, KeyModifiers, ModelTokens, Popup, Request, Snapshot, Tab, Theme, Totals,
        Usage, draw, mpsc,
    };
    use crate::history::Entry;
    use ratatui::{Terminal, backend::TestBackend};
    use serde_json::json;
    use strum::IntoEnumIterator;

    fn screen(app: &mut App, tab: Tab) -> String {
        app.tab = tab;
        let mut terminal = Terminal::new(TestBackend::new(140, 40)).expect("terminal");
        terminal.draw(|frame| draw(frame, app)).expect("draw");
        let buffer = terminal.backend().buffer();
        buffer
            .content()
            .chunks(usize::from(buffer.area.width))
            .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    // A refresh must follow new calls at the top without moving a selection on an older call.
    #[test]
    fn successful_refresh_populates_calls_and_preserves_selection() {
        let entry = |turn: &str| -> Entry {
            serde_json::from_value(json!({
                "ts": "2026-10-21T12:00:00Z",
                "model": "gemini-4-argon",
                "route_id": "answer",
                "turn_id": turn
            }))
            .expect("entry")
        };
        let snapshot = |turns: &[&str]| {
            Box::new(Snapshot {
                entries: turns.iter().map(|turn| entry(turn)).collect(),
                ..Snapshot::default()
            })
        };
        let mut app = App::new(Theme::new(false));
        app.finish(None, Ok(snapshot(&["first", "second"])));
        assert_eq!(app.visible.len(), 2);
        assert_eq!(
            app.selected_entry().expect("newest").turn_id.as_deref(),
            Some("second")
        );
        app.finish(None, Ok(snapshot(&["first", "second", "third"])));
        assert_eq!(
            app.selected_entry().expect("newest").turn_id.as_deref(),
            Some("third")
        );
        app.calls_key(KeyCode::Down);
        app.finish(None, Ok(snapshot(&["first", "second", "third", "fourth"])));
        assert_eq!(
            app.selected_entry().expect("selected").turn_id.as_deref(),
            Some("second")
        );
    }

    #[test]
    fn detail_popup_keeps_the_opened_call_across_refreshes() {
        let entry = |turn: &str| -> Entry {
            serde_json::from_value(json!({
                "ts": "2026-10-21T12:00:00Z",
                "model": "gemini-4-argon",
                "route_id": "answer",
                "turn_id": turn
            }))
            .expect("entry")
        };
        let mut app = App::new(Theme::new(false));
        app.tab = Tab::Calls;
        app.finish(
            None,
            Ok(Box::new(Snapshot {
                entries: vec![entry("opened-turn")],
                ..Snapshot::default()
            })),
        );
        app.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        app.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        // The popup must retain its call when newer calls arrive or history no longer contains it.
        for entries in [vec![entry("opened-turn"), entry("newer-turn")], vec![]] {
            app.finish(
                None,
                Ok(Box::new(Snapshot {
                    entries,
                    ..Snapshot::default()
                })),
            );
            assert!(matches!(app.popup, Some(Popup::Detail { scroll: 1, .. })));
            let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("terminal");
            terminal
                .draw(|frame| super::draw_popup(frame, &mut app, frame.area()))
                .expect("draw popup");
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(text.contains("opened-turn"), "{text}");
            assert!(!text.contains("newer-turn"), "{text}");
            app.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        }
        app.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(app.popup.is_none());
        app.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.popup.is_none());
    }

    // The last key must remain reachable when the results exceed the Settings pane's height.
    #[test]
    fn server_key_results_remain_readable_in_a_scrollable_popup() {
        let mut app = App::new(Theme::new(false));
        app.tab = Tab::Settings;
        let request = app
            .key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE))
            .expect("key action");
        let (jobs, _inbox) = mpsc::channel();
        app.start(&jobs, request).expect("start key check");
        let keys = json!({
            "store": "test credential store",
            "keys": (0..20).map(|index| json!({"name": format!("API_KEY_{index:02}"), "saved": index % 2 == 0})).collect::<Vec<_>>()
        });
        app.finish(
            Some(Ok(super::Reply {
                message: String::new(),
                data: keys.clone(),
            })),
            Ok(Box::new(Snapshot::default())),
        );
        assert_eq!(app.server_keys, Some(keys));
        assert!(matches!(app.popup, Some(Popup::Message { scroll: 0, .. })));
        app.key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("terminal");
        terminal.draw(|frame| draw(frame, &mut app)).expect("draw");
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("API_KEY_19 not saved"), "{text}");
    }

    // A late reply must preserve a new draft even when the user reopens the same kind of form.
    #[test]
    fn action_replies_only_change_the_popup_that_submitted_them() {
        for action in ['e', 's'] {
            for success in [false, true] {
                for replacement in [
                    None,
                    Some(None),
                    Some(Some('e')),
                    Some(Some('s')),
                    Some(Some('?')),
                ] {
                    let mut app = App::new(Theme::new(false));
                    app.tab = Tab::Settings;
                    let key = |code| KeyEvent::new(code, KeyModifiers::NONE);
                    app.key(key(KeyCode::Char(action)));
                    app.paste("original");
                    app.key(key(KeyCode::Tab));
                    app.paste(if action == 'e' { "30" } else { "secret" });
                    let request = app.key(key(KeyCode::Enter)).expect("submit form");
                    let (jobs, _inbox) = mpsc::channel();
                    let (done, results) = mpsc::channel();
                    app.start(&jobs, request).expect("start action");
                    if let Some(next) = replacement {
                        app.key(key(KeyCode::Esc));
                        if let Some(letter) = next {
                            app.key(key(KeyCode::Char(letter)));
                            app.paste("new draft");
                        }
                    }
                    let message = "The original action completed with a result that is too long for a short status message.";
                    done.send(Done::Finished {
                        reply: Some(if success {
                            Ok(super::Reply {
                                message: message.into(),
                                data: json!({"details": ["Original details"]}),
                            })
                        } else {
                            Err(message.into())
                        }),
                        snapshot: Ok(Box::new(Snapshot::default())),
                    })
                    .expect("completion");
                    app.poll_results(&results, &jobs).expect("result");
                    match replacement {
                        None if !success => {
                            let form = app
                                .popup
                                .as_mut()
                                .and_then(Popup::form_mut)
                                .expect("original form");
                            assert_eq!(form.error, message);
                            assert_eq!(form.value(0), "original");
                        }
                        None => assert!(matches!(app.popup, Some(Popup::Message { .. }))),
                        Some(None) => assert!(app.popup.is_none()),
                        Some(Some('?')) => assert!(matches!(app.popup, Some(Popup::Help { .. }))),
                        Some(Some(letter)) => {
                            let form = app
                                .popup
                                .as_mut()
                                .and_then(Popup::form_mut)
                                .expect("replacement form");
                            assert_eq!(form.value(0), "new draft");
                            assert!(form.error.is_empty(), "{letter}: {}", form.error);
                            let text = screen(&mut app, Tab::Settings);
                            assert!(text.contains("new draft"), "{text}");
                        }
                    }
                    if replacement.is_some() {
                        let toast = app.toast.as_ref().expect("reported completion");
                        assert!(toast.text.contains(message));
                        assert_eq!(toast.error, !success);
                    }
                }
            }
        }
    }

    // The automatic key check must preserve the form opened while the save was running.
    #[test]
    fn key_refresh_after_saving_does_not_replace_a_new_form() {
        let mut app = App::new(Theme::new(false));
        app.tab = Tab::Settings;
        let key = |code| KeyEvent::new(code, KeyModifiers::NONE);
        app.key(key(KeyCode::Char('s')));
        app.paste("API_KEY");
        app.key(key(KeyCode::Tab));
        app.paste("secret");
        let request = app.key(key(KeyCode::Enter)).expect("save key");
        let (jobs, inbox) = mpsc::channel();
        let (done, results) = mpsc::channel();
        app.start(&jobs, request).expect("start save");
        inbox.recv().expect("save job");
        app.key(key(KeyCode::Esc));
        app.key(key(KeyCode::Char('e')));
        app.paste("new draft");
        for reply in [
            super::Reply {
                message: "Saved the key.".into(),
                data: json!(null),
            },
            super::Reply {
                message: String::new(),
                data: json!({"keys": [{"name": "API_KEY", "saved": true}]}),
            },
        ] {
            done.send(Done::Finished {
                reply: Some(Ok(reply)),
                snapshot: Ok(Box::new(Snapshot::default())),
            })
            .expect("completion");
            app.poll_results(&results, &jobs).expect("result");
            let form = app
                .popup
                .as_mut()
                .and_then(Popup::form_mut)
                .expect("new form");
            assert_eq!(form.value(0), "new draft");
            assert!(form.error.is_empty());
        }
        assert!(matches!(
            inbox.try_recv(),
            Ok(super::Job::Run(Action::ServerKeys {}))
        ));
        assert_eq!(
            app.server_keys.as_ref().expect("key status")["keys"][0]["saved"],
            true
        );
    }

    // A late preview must preserve a replacement popup instead of opening update confirmation.
    #[test]
    fn update_previews_preserve_replacement_popups_and_require_confirmation() {
        for replacement in [None, Some('e'), Some('?')] {
            let mut app = App::new(Theme::new(false));
            app.tab = Tab::Settings;
            let key = |code| KeyEvent::new(code, KeyModifiers::NONE);
            let request = app.key(key(KeyCode::Char('u'))).expect("preview request");
            let (jobs, _inbox) = mpsc::channel();
            let (done, results) = mpsc::channel();
            app.start(&jobs, request).expect("start preview");
            if let Some(letter) = replacement {
                app.key(key(KeyCode::Char(letter)));
                app.paste("new draft");
            }
            done.send(Done::Finished {
                reply: Some(Ok(super::Reply {
                    message: String::new(),
                    data: json!({"preview_token": "reviewed-token"}),
                })),
                snapshot: Ok(Box::new(Snapshot::default())),
            })
            .expect("preview result");
            app.poll_results(&results, &jobs).expect("result");
            match replacement {
                None => {
                    assert!(matches!(app.popup, Some(Popup::Update { .. })));
                    assert!(app.key(key(KeyCode::Enter)).is_none());
                    app.paste("UPDATE");
                    let request = app.key(key(KeyCode::Enter)).expect("confirmed update");
                    assert!(matches!(
                        request.job,
                        super::Job::Run(Action::Update { preview_token: Some(token) })
                            if token == "reviewed-token"
                    ));
                }
                Some('e') => {
                    let form = app.popup.as_mut().and_then(Popup::form_mut).expect("draft");
                    assert_eq!(form.value(0), "new draft");
                    assert!(form.error.is_empty());
                    assert!(app.toast.is_some());
                }
                Some('?') => {
                    assert!(matches!(app.popup, Some(Popup::Help { .. })));
                    assert!(app.toast.is_some());
                }
                Some(_) => unreachable!(),
            }
        }
    }

    // A snapshot failure must preserve a replacement draft and leave a dismissed popup closed.
    #[test]
    fn snapshot_and_action_errors_preserve_dismissed_and_replacement_forms() {
        for replacement in [false, true] {
            let mut app = App::new(Theme::new(false));
            app.tab = Tab::Settings;
            let key = |code| KeyEvent::new(code, KeyModifiers::NONE);
            app.key(key(KeyCode::Char('e')));
            app.paste("original");
            app.key(key(KeyCode::Tab));
            app.paste("30");
            let request = app.key(key(KeyCode::Enter)).expect("preferences request");
            let (jobs, _inbox) = mpsc::channel();
            let (done, results) = mpsc::channel();
            app.start(&jobs, request).expect("start preferences");
            app.key(key(KeyCode::Esc));
            if replacement {
                app.key(key(KeyCode::Char('s')));
                app.paste("new draft");
            }
            done.send(Done::Finished {
                reply: Some(Err("The original action failed.".into())),
                snapshot: Err("Could not read settings.\nCheck the TOML file.".into()),
            })
            .expect("failed result");
            app.poll_results(&results, &jobs).expect("result");
            if replacement {
                let form = app.popup.as_mut().and_then(Popup::form_mut).expect("draft");
                assert_eq!(form.value(0), "new draft");
                assert!(form.error.is_empty());
            } else {
                assert!(app.popup.is_none());
            }
            assert_eq!(
                app.data.errors,
                ["Could not read settings.\nCheck the TOML file."]
            );
            assert!(app.toast.as_ref().expect("failure").error);
        }
    }

    // These fixtures check the parsed arguments without requiring a Windows editor to run.
    #[test]
    fn editor_commands_preserve_platform_quoting_and_paths() {
        for (windows, raw, program, arguments) in [
            (
                true,
                r"C:\Windows\System32\notepad.exe",
                r"C:\Windows\System32\notepad.exe",
                vec![],
            ),
            (
                true,
                r#""C:\Program Files\Editor\editor.exe" --wait "two words""#,
                r"C:\Program Files\Editor\editor.exe",
                vec!["--wait", "two words"],
            ),
            (
                true,
                r#"C:\Tools\editor.exe --wait "\\server\share\file with spaces""#,
                r"C:\Tools\editor.exe",
                vec!["--wait", r"\\server\share\file with spaces"],
            ),
            (
                true,
                r#"C:\Tools\editor.exe --title "" --wait"#,
                r"C:\Tools\editor.exe",
                vec!["--title", "", "--wait"],
            ),
            (
                true,
                r#"C:\Tools\editor.exe --title "two words"#,
                r"C:\Tools\editor.exe",
                vec!["--title", "two words"],
            ),
            (
                false,
                "'editor with spaces' --wait 'two words'",
                "editor with spaces",
                vec!["--wait", "two words"],
            ),
        ] {
            let command = super::editor_command(raw, windows).expect("editor command");
            assert_eq!(
                command.get_program(),
                std::ffi::OsStr::new(program),
                "{raw}"
            );
            assert_eq!(
                command.get_args().collect::<Vec<_>>(),
                arguments
                    .iter()
                    .map(std::ffi::OsStr::new)
                    .collect::<Vec<_>>(),
                "{raw}"
            );
        }
        for (windows, raw, error) in [
            (true, " ", "Set VISUAL or EDITOR to an editor command."),
            (false, " ", "Set VISUAL or EDITOR to an editor command."),
            (
                false,
                "'unterminated",
                "The editor command has an unmatched quote.",
            ),
        ] {
            assert_eq!(
                super::editor_command(raw, windows).err().as_deref(),
                Some(error)
            );
        }
        let directory = tempfile::tempdir().expect("directory");
        let path = directory.path().join("editor with spaces.exe");
        std::fs::write(&path, "").expect("editor file");
        let command =
            super::editor_command(path.to_str().expect("UTF-8 path"), true).expect("editor path");
        assert_eq!(command.get_program(), path.as_os_str());
        assert_eq!(command.get_args().count(), 0);
    }

    // Esc dismisses the status message while Overview keeps the refresh error.
    #[test]
    fn refresh_errors_remain_in_overview_after_dismissing_the_toast() {
        let mut app = App::new(Theme::new(false));
        app.finish(None, Err("Could not read settings.".into()));
        app.tab = Tab::Overview;
        app.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("terminal");
        terminal.draw(|frame| draw(frame, &mut app)).expect("draw");
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("Could not read settings."), "{text}");
    }

    // An empty queue can still receive a result; a disconnected queue cannot finish the refresh.
    #[test]
    fn worker_disconnect_returns_an_error_with_a_refresh_in_flight() {
        let mut app = App::new(Theme::new(false));
        let (jobs, _inbox) = mpsc::channel();
        app.start(&jobs, Request::refresh(false)).expect("refresh");
        let (done, results) = mpsc::channel();
        assert!(app.poll_results(&results, &jobs).is_ok());
        drop(done);
        assert_eq!(
            app.poll_results(&results, &jobs),
            Err("The terminal operation worker stopped unexpectedly.".into())
        );
    }

    #[test]
    fn automatic_refresh_errors_leave_dismissed_popups_and_help_alone() {
        let mut app = App::new(Theme::new(false));
        let error = "Could not read settings.\nCheck the TOML file.";
        let (jobs, _inbox) = mpsc::channel();
        let (done, results) = mpsc::channel();
        app.start(&jobs, Request::refresh(false))
            .expect("manual refresh");
        done.send(Done::Finished {
            reply: None,
            snapshot: Err(error.into()),
        })
        .expect("result");
        app.poll_results(&results, &jobs).expect("manual result");
        assert!(matches!(app.popup, Some(Popup::Message { .. })));
        app.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        for help in [false, true] {
            if help {
                app.key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));
                app.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
            }
            app.start(&jobs, Request::refresh(true))
                .expect("automatic refresh");
            done.send(Done::Finished {
                reply: None,
                snapshot: Err(error.into()),
            })
            .expect("result");
            app.poll_results(&results, &jobs).expect("automatic result");
            assert!(if help {
                matches!(app.popup, Some(Popup::Help { scroll: 1 }))
            } else {
                app.popup.is_none()
            });
            assert_eq!(
                app.toast.as_ref().expect("error toast").text,
                "Could not read settings."
            );
            assert_eq!(app.data.errors, [error]);
        }
    }

    #[test]
    fn data_errors_split_newlines_and_reserve_wrapped_rows_at_minimum_size() {
        for error in [
            "Could not read model prices.\nSpecific cause".to_string(),
            format!(
                "Could not read model prices at {}: Specific cause",
                "long-directory/".repeat(8)
            ),
        ] {
            let mut app = App::new(Theme::new(false));
            app.data.errors = vec![error.clone()];
            let mut terminal = Terminal::new(TestBackend::new(60, 16)).expect("terminal");
            terminal.draw(|frame| draw(frame, &mut app)).expect("draw");
            let rows: Vec<String> = terminal
                .backend()
                .buffer()
                .content
                .chunks(60)
                .map(|row| row.iter().map(|cell| cell.symbol()).collect())
                .collect();
            assert!(
                rows[3..].iter().any(|row| row.contains("Specific cause")),
                "{error}: {rows:?}"
            );
        }
    }

    // A call without a recorded cost must never add up to, or display as, $0.
    #[test]
    fn unknown_costs_do_not_render_as_zero() {
        let entry: Entry = serde_json::from_value(json!({"ts": "2026-10-21T12:00:00Z",
            "model": "gemini-4-argon", "route_id": "answer", "prompt_tokens": 10,
            "completion_tokens": 2}))
        .expect("entry");
        let usage = ModelTokens::from(&entry);
        let totals = Totals {
            routed: [("gemini-4-argon".to_string(), usage.clone())].into(),
            classifier: BTreeMap::new(),
            routes: [("answer".to_string(), usage)].into(),
        };
        let estimate = Estimate {
            actual: None,
            known: 0.0,
            has_known_cost: false,
            unknown_calls: 1,
            baseline: Some(0.00004),
            uncounted_answers: 0,
        };
        let mut app = App::new(Theme::new(false));
        app.data = Snapshot {
            analytics: Usage {
                today: totals.clone(),
                week: totals.clone(),
                all: totals,
                ..Usage::default()
            },
            estimates: [
                ("today", Some(estimate)),
                ("week", Some(estimate)),
                ("all", Some(estimate)),
            ]
            .into(),
            baseline_model: "gemini-4-argon".into(),
            entries: vec![entry],
            ..Snapshot::default()
        };
        app.data.data_state.usage.state = Availability::Available;
        app.data.data_state.history.state = Availability::Available;
        app.rebuild_calls();
        for tab in [Tab::Overview, Tab::Calls, Tab::Models] {
            let text = screen(&mut app, tab);
            assert!(!text.contains("$0"), "{tab:?} shows a zero cost:\n{text}");
        }
        let overview = screen(&mut app, Tab::Overview);
        assert!(overview.contains("Spent unknown"));
        assert!(overview.contains("no recorded cost"));
    }

    #[test]
    // Every controller action needs a Settings key, or the desktop app can do something
    // that the terminal app cannot.
    fn every_controller_action_has_a_settings_key() {
        let mut keys: Vec<char> = ACTIONS.iter().map(|&(key, ..)| key).collect();
        // The Settings page reads its letters before the global keys and the j, g, and G moves.
        let reserved = ['q', 'r', '?', '1', '2', '3', '4', 'j', 'g', 'G'];
        keys.extend(reserved);
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(
            keys.len(),
            ACTIONS.len() + reserved.len(),
            "Settings keys must be unique and must not use a global or movement key"
        );

        // Update is the only action without its own key: the u key reviews the update
        // first, and the review window submits Update.
        let missing: Vec<ActionKind> = ActionKind::iter()
            .filter(|&kind| kind != ActionKind::Update && !ACTIONS.iter().any(|&(.., k)| k == kind))
            .collect();
        assert!(missing.is_empty(), "No Settings key runs {missing:?}");
        let mut review = Form::new(FormKind::Update, "", vec![("", "UPDATE".into())]);
        assert!(matches!(
            review.action(Some(&json!({"preview_token": "t"}))),
            Ok(Action::Update { preview_token: Some(token) }) if token == "t"
        ));
        review.fields[0].1 = "update".into();
        assert!(review.action(None).is_err());
    }
}
