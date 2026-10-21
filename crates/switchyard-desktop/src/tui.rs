// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The terminal app shows usage, recent calls, routes, and harness settings from controller
//! snapshots, and it runs server, route, harness, and account actions.
//!
//! The UI thread owns every view setting. One worker thread owns the controller and runs
//! refreshes and actions, so moving around and filtering never wait for the log or a service.

use crate::controller::{
    AccountInfo, Action, ActionKind, AppSettings, Availability, Controller, Dependent, Reply,
    RouteInfo, Snapshot, ToolInfo,
};
use crate::harness::Harness;
use crate::history::Entry;
use crate::pricing::Estimate;
use crate::rollup::{CLASSIFIER_TIER, ModelTokens, Totals, Usage};
use crate::server_config::Choice;
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
    borrow::Cow,
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

/// Each key list has the key, a short label for the status bar, and a sentence for the help
/// window. Both read these lists, so the status bar and the help window always agree.
type Key = (&'static str, &'static str, &'static str);
const GLOBAL_KEYS: &[Key] = &[
    (
        "1-6",
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
const ROUTES_KEYS: &[Key] = &[
    (
        "↑↓",
        "select",
        "Select a route. j, PgUp, PgDn, g, and G also move it.",
    ),
    (
        "/",
        "filter",
        "Filter as you type. Every word must appear in the route's public model ID, config \
         name, algorithm, endpoints, or models. A word with a capital letter must match case \
         exactly.",
    ),
    ("esc", "clear", "Clear the filter."),
];
const HARNESSES_KEYS: &[Key] = &[
    (
        "↑↓",
        "route",
        "Choose the route to install. j, k, PgUp, PgDn, g, and G also move it. The Routes page \
         selects the same route, and its filter limits the choices.",
    ),
    ("t", "harness", "Select the next harness."),
    (
        "a",
        "account",
        "Select the next saved account, then the harness's own settings again. Install, \
         restore, and the account keys use the selection.",
    ),
    (
        "x",
        "custom file",
        "Install to or restore a custom settings file instead. Leave it empty to use the \
         detected settings again.",
    ),
];
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
/// The status bar shows these keys while the selected form field has a list of choices.
const CHOICE_KEYS: &[Key] = &[
    ("⏎", "submit", ""),
    ("↑↓", "choose", ""),
    ("tab", "next field", ""),
    ("esc", "cancel", ""),
];
const REVIEW_KEYS: &[Key] = &[
    ("⏎", "confirm", ""),
    ("↑↓", "scroll", ""),
    ("esc", "cancel", ""),
];
const POPUP_KEYS: &[Key] = &[("↑↓", "scroll", ""), ("esc", "close", "")];
const CONFIRM_KEYS: &[Key] = &[("y", "confirm", ""), ("n", "cancel", "")];

/// Each row is a page, a key, a short label, a help sentence, and the actions that the key's
/// flow can run. `App::start_action` receives the first action. Settings lists its rows and
/// runs the selected one with Enter; other pages run a row's flow on the selected item when its
/// key is pressed. A new controller action breaks the build until `App::start_action` runs it,
/// and `every_controller_action_has_a_key` fails until a row's flow includes it.
type ActionRow = (Tab, char, &'static str, &'static str, &'static [ActionKind]);
const ACTIONS: &[ActionRow] = &[
    (
        Tab::Routes,
        'e',
        "edit",
        "Changes the selected route's algorithm and models. Enter reviews the change, and \
         typing SAVE saves it and restarts the server.",
        &[
            ActionKind::Editor,
            ActionKind::PreviewRoute,
            ActionKind::Apply,
        ],
    ),
    (
        Tab::Routes,
        'a',
        "create",
        "Creates a route from a public model ID, an algorithm, and its models. Type SAVE after \
         the review.",
        &[ActionKind::PreviewRouteChange, ActionKind::RouteChange],
    ),
    (
        Tab::Routes,
        'n',
        "rename",
        "Changes the selected route's public model ID. Harnesses that send the old ID must \
         change too. Type SAVE after the review.",
        &[ActionKind::PreviewRouteChange, ActionKind::RouteChange],
    ),
    (
        Tab::Routes,
        'c',
        "duplicate",
        "Copies the selected route under a new public model ID. The copy shares its targets. \
         Type SAVE after the review.",
        &[ActionKind::PreviewRouteChange, ActionKind::RouteChange],
    ),
    (
        Tab::Routes,
        'd',
        "delete",
        "Deletes the selected route after you type its public model ID. Its targets and \
         endpoints stay, and the server config is backed up.",
        &[ActionKind::Remove],
    ),
    (
        Tab::Routes,
        'm',
        "models",
        "Loads the model IDs that an endpoint offers.",
        &[ActionKind::Models],
    ),
    (
        Tab::Routes,
        'k',
        "model key",
        "Saves an API key that lets the app load an endpoint's model IDs, then loads them.",
        &[ActionKind::Models],
    ),
    (
        Tab::Harnesses,
        'p',
        "preview",
        "Shows the settings diff that installing the selected route would make. Nothing is \
         saved.",
        &[ActionKind::PreviewInstall],
    ),
    (
        Tab::Harnesses,
        'i',
        "install",
        "Shows the settings diff, then saves it after you type INSTALL. A backup keeps the \
         original settings.",
        &[ActionKind::PreviewInstall, ActionKind::Install],
    ),
    (
        Tab::Harnesses,
        'u',
        "restore",
        "Shows which files return to their original versions, then restores them after you \
         type RESTORE.",
        &[ActionKind::PreviewRestore, ActionKind::Restore],
    ),
    (
        Tab::Harnesses,
        'n',
        "add account",
        "Creates a saved account and opens the harness's login in a new terminal window.",
        &[ActionKind::AddAccount],
    ),
    (
        Tab::Harnesses,
        'l',
        "login",
        "Opens the selected account's login again in a new terminal window.",
        &[ActionKind::LoginAccount],
    ),
    (
        Tab::Harnesses,
        'o',
        "open account",
        "Opens the selected account folder in $VISUAL or $EDITOR.",
        &[ActionKind::OpenAccount],
    ),
    (
        Tab::Harnesses,
        'd',
        "archive",
        "Moves the selected account out of the list after you type its name. Its login files \
         are kept.",
        &[
            ActionKind::PreviewAccountArchive,
            ActionKind::ArchiveAccount,
        ],
    ),
    (
        Tab::Settings,
        'o',
        "Open server config",
        "Opens the server config in $VISUAL or $EDITOR. Restart the server after you save it.",
        &[ActionKind::OpenConfig],
    ),
    (
        Tab::Settings,
        'c',
        "Check server config",
        "Runs the installed server's config check. The running server keeps running.",
        &[ActionKind::CheckConfig],
    ),
    (
        Tab::Settings,
        'b',
        "Restart server",
        "Checks the server config, then restarts the Switchyard service after you confirm. \
         Requests in progress fail.",
        &[ActionKind::Restart],
    ),
    (
        Tab::Settings,
        'a',
        "Open app settings",
        "Opens this app's settings file in $VISUAL or $EDITOR.",
        &[ActionKind::OpenSettings],
    ),
    (
        Tab::Settings,
        'e',
        "Change preferences",
        "Changes the baseline model that prices savings and the refresh interval.",
        &[ActionKind::SavePreferences],
    ),
    (
        Tab::Settings,
        'p',
        "Open model prices",
        "Opens the custom price file in $VISUAL or $EDITOR. With bundled prices, this explains \
         how to set a custom file.",
        &[ActionKind::OpenPrices],
    ),
    (
        Tab::Settings,
        'k',
        "Check server keys",
        "Lists the API key variables that the server config names and whether each is saved.",
        &[ActionKind::ServerKeys],
    ),
    (
        Tab::Settings,
        's',
        "Save a server key",
        "Saves an API key for a variable that the server config names. Restart the server to \
         use it.",
        &[ActionKind::SaveServerKey],
    ),
    (
        Tab::Settings,
        'i',
        "Import a server key",
        "Saves an API key from your login shell's environment. Restart the server to use it.",
        &[ActionKind::ImportServerKey],
    ),
    (
        Tab::Settings,
        'x',
        "Remove a server key",
        "Removes a saved API key. The server does not start until the key is saved again.",
        &[ActionKind::RemoveServerKey],
    ),
    (
        Tab::Settings,
        'u',
        "Update from source",
        "Shows what the saved source checkout would install. Type UPDATE to start it.",
        &[ActionKind::PreviewUpdate, ActionKind::Update],
    ),
    (
        Tab::Settings,
        'l',
        "Open update log",
        "Opens the log of the last source update in $VISUAL or $EDITOR.",
        &[ActionKind::OpenUpdateLog],
    ),
    (
        Tab::Settings,
        'v',
        "Open server logs",
        "Opens the Switchyard server logs folder in $VISUAL or $EDITOR.",
        &[ActionKind::OpenLogs],
    ),
];
/// This function lists one page's rows in `ACTIONS` order.
fn rows(tab: Tab) -> impl Iterator<Item = &'static ActionRow> + Clone {
    ACTIONS.iter().filter(move |row| row.0 == tab)
}

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
/// The routes table drops the columns marked `true` below 80 columns; the route details still
/// show every model.
const ROUTE_COLUMNS: [(Column, bool); 3] = [
    (("ROUTE", Constraint::Fill(1), false), false),
    (("ALGORITHM", Constraint::Length(24), false), false),
    (("MODELS", Constraint::Fill(2), false), true),
];
/// The harnesses table drops the columns marked `true` below 80 columns; the details still
/// show the current settings.
const HARNESS_COLUMNS: [(Column, bool); 4] = [
    (("HARNESS", Constraint::Fill(2), false), false),
    (("STATE", Constraint::Length(13), false), false),
    (("CURRENT SETTINGS", Constraint::Fill(3), false), true),
    (("ACCOUNTS", Constraint::Length(8), true), false),
];

/// Theme maps each kind of text to one style. NO_COLOR keeps only bold, dim, and reverse.
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
    Routes,
    Harnesses,
    Settings,
}
const TABS: [Tab; 6] = [
    Tab::Overview,
    Tab::Calls,
    Tab::Models,
    Tab::Routes,
    Tab::Harnesses,
    Tab::Settings,
];
impl Tab {
    fn name(self) -> &'static str {
        match self {
            Tab::Overview => "Overview",
            Tab::Calls => "Calls",
            Tab::Models => "Models",
            Tab::Routes => "Routes",
            Tab::Harnesses => "Harnesses",
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

enum FormKind {
    Preferences,
    SaveKey,
    ImportKey,
    RemoveKey,
    /// The token comes from the update preview that the user read.
    Update(Option<String>),
    /// The route is the snapshot's copy from when the user opened the editor. The models form
    /// keeps its revision and choices, so a server config that changed since then fails the
    /// review instead of mixing old choices into the new file.
    Algorithm {
        generation: u64,
        route: Box<RouteInfo>,
    },
    RouteModels {
        generation: u64,
        route: String,
        algorithm: String,
        revision: Option<String>,
    },
    /// The fields hold a new route's public model ID, config name, and algorithm.
    NewRoute {
        generation: u64,
    },
    NewRouteModels {
        generation: u64,
        name: String,
        id: String,
        algorithm: String,
    },
    /// The operation is `rename` or `duplicate`, and `route` is the selected route's config name.
    RouteIdentity {
        generation: u64,
        operation: &'static str,
        route: String,
    },
    /// The fields hold the word that the user must type and the reviewed action, which the
    /// form submits unchanged.
    Confirm(Cow<'static, str>, Box<Action>),
    Remove {
        generation: u64,
        route: String,
        id: String,
        revision: String,
    },
    Models,
    ModelKey,
    /// The harness is the one selected when the form opened.
    Account(Harness),
    SettingsFile,
}
/// Choices names the list that ↑ and ↓ cycle through in a form field.
enum Choices {
    Algorithms,
    Endpoints,
}
/// Submit is what a form sends: a controller action, the next form of a flow that needs more
/// input before the controller can review it, or a custom settings file, which changes only
/// where the Harnesses page installs.
enum Submit {
    Run(Action),
    Next(Form),
    SettingsFile(String),
}
/// Form holds the text fields for an action that needs input.
struct Form {
    kind: FormKind,
    title: Cow<'static, str>,
    fields: Vec<(Cow<'static, str>, String)>,
    index: usize,
    /// This field counts characters, not bytes, from the start of the selected field.
    cursor: usize,
    /// This field shows the last validation or controller error below the fields.
    error: String,
}
impl Form {
    fn new<L: Into<Cow<'static, str>>>(
        kind: FormKind,
        title: impl Into<Cow<'static, str>>,
        fields: Vec<(L, String)>,
    ) -> Self {
        let fields: Vec<_> = fields
            .into_iter()
            .map(|(label, value)| (label.into(), value))
            .collect();
        let cursor = fields.first().map_or(0, |field| field.1.chars().count());
        Self {
            kind,
            title: title.into(),
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
    /// This function reads the choices from fields that alternate endpoint and model.
    fn choices(&self) -> Vec<Choice> {
        self.fields
            .chunks_exact(2)
            .map(|pair| Choice {
                client: pair[0].1.trim().into(),
                model: pair[1].1.trim().into(),
            })
            .collect()
    }
    /// This function names the list that the selected field chooses from, if any.
    fn options(&self) -> Option<Choices> {
        match self.kind {
            FormKind::Algorithm { .. } => Some(Choices::Algorithms),
            FormKind::NewRoute { .. } if self.index == 2 => Some(Choices::Algorithms),
            FormKind::RouteModels { .. } | FormKind::NewRouteModels { .. }
                if self.index.is_multiple_of(2) =>
            {
                Some(Choices::Endpoints)
            }
            FormKind::Models | FormKind::ModelKey if self.index == 0 => Some(Choices::Endpoints),
            _ => None,
        }
    }
    /// This function replaces the selected field with the next or previous name in `names`.
    fn choose<'a>(&mut self, mut names: impl Iterator<Item = &'a str> + Clone, forward: bool) {
        let count = names.clone().count();
        if count == 0 {
            return;
        }
        let field = &mut self.fields[self.index].1;
        let next = match names.clone().position(|name| name == field.trim()) {
            Some(index) if forward => (index + 1) % count,
            Some(index) => (index + count - 1) % count,
            None => 0,
        };
        *field = names.nth(next).unwrap_or_default().to_string();
        self.cursor = field.chars().count();
    }
    /// This function builds what the form submits. The snapshot supplies the roles and
    /// endpoints that a new route's models form starts with.
    fn action(&self, data: &Snapshot) -> Result<Submit, String> {
        let confirm = |word: &str, error: &str| {
            if self.value(0) == word {
                Ok(())
            } else {
                Err(error.to_string())
            }
        };
        Ok(Submit::Run(match &self.kind {
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
            FormKind::Update(preview_token) => {
                confirm("UPDATE", "Type UPDATE to start the reviewed source update.")?;
                Action::Update {
                    preview_token: preview_token.clone(),
                }
            }
            FormKind::Algorithm { generation, route } => Action::Editor {
                generation: *generation,
                route: route.key.clone(),
                algorithm: self.value(0),
            },
            FormKind::RouteModels {
                generation,
                route,
                algorithm,
                revision,
            } => Action::PreviewRoute {
                generation: *generation,
                route: route.clone(),
                algorithm: algorithm.clone(),
                choices: self.choices(),
                revision: revision.clone(),
            },
            FormKind::NewRoute { generation } => {
                let id = self.value(0);
                let algorithm = self.value(2);
                let roles = &data
                    .algorithms
                    .iter()
                    .find(|known| known.kind == algorithm)
                    .ok_or("Unknown routing algorithm. Press ↑ or ↓ to choose one.")?
                    .roles;
                let choice = Choice {
                    client: data
                        .clients
                        .first()
                        .map(|client| client.name.clone())
                        .unwrap_or_default(),
                    model: String::new(),
                };
                let fields = role_fields(roles.iter().map(|role| (&*role.label, choice.clone())));
                let title = format!("Models for the new route {id}");
                let kind = FormKind::NewRouteModels {
                    generation: *generation,
                    name: config_name(&id, &self.value(1)),
                    id,
                    algorithm,
                };
                return Ok(Submit::Next(Form::new(kind, title, fields)));
            }
            FormKind::NewRouteModels {
                generation,
                name,
                id,
                algorithm,
            } => Action::PreviewRouteChange {
                generation: *generation,
                operation: "create".into(),
                route: String::new(),
                name: name.clone(),
                id: id.clone(),
                algorithm: Some(algorithm.clone()),
                choices: self.choices(),
            },
            FormKind::RouteIdentity {
                generation,
                operation,
                route,
            } => {
                let id = self.value(0);
                // A rename keeps the config name, so its form has no config name field.
                let name = match self.fields.get(1) {
                    Some((_, typed)) => config_name(&id, typed.trim()),
                    None => String::new(),
                };
                Action::PreviewRouteChange {
                    generation: *generation,
                    operation: operation.to_string(),
                    route: route.clone(),
                    name,
                    id,
                    algorithm: None,
                    choices: Vec::new(),
                }
            }
            FormKind::Confirm(word, action) => {
                confirm(word, &format!("Type {word} to apply the reviewed changes."))?;
                (**action).clone()
            }
            FormKind::Remove {
                generation,
                route,
                id,
                revision,
            } => {
                confirm(id, "Route name did not match. Nothing was deleted.")?;
                Action::Remove {
                    generation: *generation,
                    route: route.clone(),
                    id: id.clone(),
                    revision: Some(revision.clone()),
                }
            }
            FormKind::Models => Action::Models {
                client: self.value(0),
                refresh: true,
                key: None,
            },
            FormKind::ModelKey => Action::Models {
                client: self.value(0),
                refresh: true,
                key: Some(self.value(1)),
            },
            FormKind::Account(tool) => Action::AddAccount {
                tool: *tool,
                name: self.value(0),
            },
            FormKind::SettingsFile => return Ok(Submit::SettingsFile(self.value(0))),
        }))
    }
    fn label(&self) -> &'static str {
        match &self.kind {
            FormKind::Preferences => "Saving the preferences",
            FormKind::SaveKey => "Saving the server key",
            FormKind::ImportKey => "Importing the server key",
            FormKind::RemoveKey => "Removing the server key",
            FormKind::Update(_) => "Starting the update",
            FormKind::Algorithm { .. } => "Opening the route editor",
            FormKind::RouteModels { .. } => "Reviewing the route",
            FormKind::NewRoute { .. }
            | FormKind::NewRouteModels { .. }
            | FormKind::RouteIdentity { .. } => "Reviewing the route change",
            // The reviewed action names what typing the confirmation word saves.
            FormKind::Confirm(_, action) => match **action {
                Action::Install { .. } => "Installing the harness settings",
                Action::Restore { .. } => "Restoring the harness settings",
                Action::ArchiveAccount { .. } => "Archiving the account",
                _ => "Saving the route",
            },
            FormKind::Remove { .. } => "Deleting the route",
            FormKind::Models => "Loading models",
            FormKind::ModelKey => "Saving the key and loading models",
            FormKind::Account(_) => "Adding the account",
            FormKind::SettingsFile => "Choosing the settings file",
        }
    }
}

/// This function returns an endpoint field and a model field for each role and its choice.
fn role_fields<'a>(roles: impl Iterator<Item = (&'a str, Choice)>) -> Vec<(String, String)> {
    roles
        .flat_map(|(label, choice)| {
            let model = if label == "Model" {
                label.to_string()
            } else {
                format!("{label} model")
            };
            [
                (format!("{label} endpoint"), choice.client),
                (model, choice.model),
            ]
        })
        .collect()
}

/// This function returns the typed config name, or one made from the public model ID the same
/// way the desktop app fills a blank config name.
fn config_name(id: &str, typed: &str) -> String {
    if !typed.is_empty() {
        return typed.to_string();
    }
    id.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-")
        .trim_matches('-')
        .chars()
        .take(64)
        .collect()
}

/// This function opens the models form after the controller returns the roles of `algorithm`.
fn route_models_form(
    generation: u64,
    route: &RouteInfo,
    algorithm: String,
    editor: &Value,
) -> Result<Form, String> {
    let roles = editor["roles"].as_array().ok_or("No roles returned.")?;
    let fields = role_fields(roles.iter().enumerate().map(|(index, role)| {
        // A role keeps its model by tier when the new algorithm changes the role order.
        let matching = route.tiers.as_ref().and_then(|tiers| {
            let tier = role["tier"].as_str()?;
            tiers.iter().position(|t| t.as_deref() == Some(tier))
        });
        let choice = matching
            .and_then(|index| route.choices.get(index))
            .or_else(|| (algorithm == route.kind).then(|| route.choices.get(index))?)
            .or_else(|| route.choices.first())
            .cloned()
            .unwrap_or_default();
        (role["label"].as_str().unwrap_or_default(), choice)
    }));
    let kind = FormKind::RouteModels {
        generation,
        route: route.key.clone(),
        algorithm,
        revision: Some(route.revision.clone()),
    };
    Ok(Form::new(kind, format!("Models for {}", route.id), fields))
}

/// This function returns the action that a preview's review window submits after the user
/// types the confirmation word: the previewed inputs, plus the preview's token for every flow
/// except a route edit, which sends its revision instead.
fn reviewed(previewed: Action, preview: &Value) -> Option<Action> {
    let preview_token = preview["preview_token"].as_str().map(str::to_owned);
    Some(match previewed {
        Action::PreviewRoute {
            generation,
            route,
            algorithm,
            choices,
            revision,
        } => Action::Apply {
            generation,
            route,
            algorithm,
            choices,
            revision,
        },
        Action::PreviewRouteChange {
            generation,
            operation,
            route,
            name,
            id,
            algorithm,
            choices,
        } => Action::RouteChange {
            generation,
            operation,
            route,
            name,
            id,
            algorithm,
            choices,
            preview_token,
        },
        Action::PreviewInstall {
            tool,
            account,
            route,
            id,
            settings_file,
        } => Action::Install {
            tool,
            account,
            route,
            id,
            settings_file,
            preview_token,
        },
        Action::PreviewRestore {
            tool,
            account,
            settings_file,
        } => Action::Restore {
            tool,
            account,
            settings_file,
            preview_token,
        },
        Action::PreviewAccountArchive { tool, name } => Action::ArchiveAccount {
            tool,
            name,
            preview_token,
        },
        _ => return None,
    })
}

enum Popup {
    Help {
        scroll: u16,
    },
    Detail {
        scroll: u16,
    },
    Restart,
    Form(Form),
    /// The user reads the preview lines, then types the confirmation word into `confirm`.
    Review {
        lines: Vec<Line<'static>>,
        /// This field names what Enter does, such as Save or Delete.
        verb: &'static str,
        scroll: u16,
        confirm: Form,
    },
    Message {
        title: &'static str,
        text: String,
        tone: Tone,
        scroll: u16,
    },
}
impl Popup {
    fn scroll_mut(&mut self) -> Option<&mut u16> {
        match self {
            Popup::Help { scroll }
            | Popup::Detail { scroll }
            | Popup::Review { scroll, .. }
            | Popup::Message { scroll, .. } => Some(scroll),
            Popup::Restart | Popup::Form(_) => None,
        }
    }
    fn form_mut(&mut self) -> Option<&mut Form> {
        match self {
            Popup::Form(form) | Popup::Review { confirm: form, .. } => Some(form),
            _ => None,
        }
    }
    /// This function opens a review window whose one field takes the confirmation word.
    fn review(
        lines: Vec<Line<'static>>,
        verb: &'static str,
        kind: FormKind,
        title: impl Into<Cow<'static, str>>,
        field: &'static str,
    ) -> Self {
        Popup::Review {
            lines,
            verb,
            scroll: 0,
            confirm: Form::new(kind, title, vec![(field, String::new())]),
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
    /// This field holds the `ACTIONS` key that started the job. Keys that share a first action
    /// continue differently when the reply arrives, such as `p` and `i` on the Harnesses page.
    key: Option<char>,
}
impl Request {
    fn run(action: Action, label: &'static str) -> Self {
        Self {
            job: Job::Run(action),
            label,
            key: None,
        }
    }
    fn refresh() -> Self {
        Self {
            job: Job::Refresh,
            label: "Refreshing",
            key: None,
        }
    }
}
struct Busy {
    label: &'static str,
    /// This field holds the running action, so its reply can update the right view and a
    /// preview's review window can submit exactly what the controller reviewed.
    action: Option<Action>,
    key: Option<char>,
    started: Instant,
}
struct Toast {
    text: String,
    tone: Tone,
    shown: Instant,
}
/// Tone tells the user how an action went. A warning means the action did only part of its
/// work, such as saving a route when the server could not restart. Warnings and errors stay in
/// the status bar until Esc, because the user must act on them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tone {
    Done,
    Warning,
    Error,
}
impl Tone {
    /// The controller marks a reply as a warning when the action did only part of its work.
    fn of(reply: &Reply) -> Self {
        if reply.data["warning"] == true {
            Tone::Warning
        } else {
            Tone::Done
        }
    }
}

/// Destination is where the Harnesses page installs and restores settings.
enum Destination {
    /// The harness's own settings files.
    Harness,
    /// A saved account, by name. The name stays after a refresh stops listing the account, so
    /// its actions still send it and the controller rejects them instead of changing other
    /// settings.
    Account(String),
    /// A custom settings file path.
    File(String),
}

struct App {
    theme: Theme,
    data: Snapshot,
    tab: Tab,
    period: Period,
    filter: String,
    filtering: bool,
    session: Option<String>,
    /// These indexes point into `data.entries`, newest first, after the session and filter.
    visible: Vec<usize>,
    selected: usize,
    offset: usize,
    action: usize,
    route_filter: String,
    /// These indexes point into `data.routes`, in config order, after the route filter.
    route_visible: Vec<usize>,
    route_selected: usize,
    route_offset: usize,
    /// This field indexes `data.tools`, which lists every harness in a fixed order.
    harness: usize,
    destination: Destination,
    /// This field holds the last install preview and the action that produced it. The
    /// Harnesses page shows it only while that action matches the selection.
    install_preview: Option<(Action, Value)>,
    /// This field holds the last `ServerKeys` reply. It is `None` until the user checks the keys,
    /// and a failed check clears it so the page never shows outdated key states.
    server_keys: Option<Value>,
    popup: Option<Popup>,
    toast: Option<Toast>,
    busy: Option<Busy>,
    /// A quit request waits for the running job, so a restart or update is never cut off.
    quit: bool,
    refreshed: Option<Instant>,
    last_refresh: Instant,
    width: u16,
}

impl App {
    fn new(theme: Theme, data: Snapshot) -> Self {
        let mut app = Self {
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
            route_filter: String::new(),
            route_visible: Vec::new(),
            route_selected: 0,
            route_offset: 0,
            harness: 0,
            destination: Destination::Harness,
            install_preview: None,
            server_keys: None,
            popup: None,
            toast: None,
            busy: None,
            quit: false,
            refreshed: None,
            last_refresh: Instant::now(),
            width: 80,
        };
        app.apply_snapshot(Ok(data));
        app
    }

    fn apply_snapshot(&mut self, snapshot: Result<Snapshot, String>) {
        let data = match snapshot {
            Ok(data) => data,
            Err(error) => return self.notify(error, Tone::Error),
        };
        // A refresh keeps the same call selected unless the newest call was selected.
        let anchor = self
            .selected_entry()
            .filter(|_| self.selected > 0)
            .map(|entry| (entry.ts.clone(), entry.model.clone(), entry.turn_id.clone()));
        let route = self.selected_route_key();
        self.data = data;
        self.refreshed = Some(Instant::now());
        self.rebuild_calls();
        self.rebuild_routes(route);
        if let Some((ts, model, turn)) = anchor
            && let Some(position) = self.visible.iter().position(|&index| {
                let entry = &self.data.entries[index];
                entry.ts == ts && entry.model == model && entry.turn_id == turn
            })
        {
            self.selected = position;
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

    /// A route keeps its selection by config name, so a refresh or a rename keeps it selected.
    fn selected_route_key(&self) -> Option<String> {
        self.selected_route().ok().map(|route| route.key.clone())
    }

    /// This function filters the routes and selects `anchor` if it is still listed. Otherwise
    /// the selection keeps its position, so deleting a route selects the next one; `refilter`
    /// starts from the first match instead.
    fn rebuild_routes(&mut self, anchor: Option<String>) {
        let algorithms = &self.data.algorithms;
        let filter = self.route_filter.as_str();
        self.route_visible = self
            .data
            .routes
            .iter()
            .enumerate()
            .filter(|(_, route)| {
                let title = algorithms
                    .iter()
                    .find(|algorithm| algorithm.kind == route.kind)
                    .map_or("", |algorithm| algorithm.title);
                matches_words(&[&route.label, &route.key, title], filter)
            })
            .map(|(index, _)| index)
            .collect();
        let routes = &self.data.routes;
        self.route_selected = anchor
            .and_then(|key| {
                self.route_visible
                    .iter()
                    .position(|&index| routes[index].key == key)
            })
            .unwrap_or(self.route_selected)
            .min(self.route_visible.len().saturating_sub(1));
    }

    fn selected_route(&self) -> Result<&RouteInfo, String> {
        match self.route_visible.get(self.route_selected) {
            Some(&index) => Ok(&self.data.routes[index]),
            None if !self.data.routes.is_empty() => {
                Err("No route matches the filter. Clear or change the filter first.".into())
            }
            None => Err("Choose a route first.".into()),
        }
    }

    fn harness(&self) -> Result<&ToolInfo, String> {
        self.data
            .tools
            .get(self.harness)
            .ok_or_else(|| "No harness is listed. Fix the app settings, then press r.".into())
    }

    /// This function returns the selected account's name, or `None` when the harness's own
    /// settings or a custom settings file is the destination.
    fn account(&self) -> Option<String> {
        match &self.destination {
            Destination::Account(name) => Some(name.clone()),
            Destination::Harness | Destination::File(_) => None,
        }
    }

    fn custom_file(&self) -> Option<String> {
        match &self.destination {
            Destination::File(path) => Some(path.clone()),
            Destination::Harness | Destination::Account(_) => None,
        }
    }

    /// Account actions change one saved account, so this function rejects other destinations.
    fn selected_account(&self) -> Result<(Harness, String), String> {
        let name = self
            .account()
            .ok_or("Press a to select a saved account first.")?;
        Ok((self.harness()?.tool, name))
    }

    /// This function builds the install preview of the selected route for the selected
    /// harness and destination.
    fn install_action(&self) -> Result<Action, String> {
        let tool = self.harness()?.tool;
        let route = self.selected_route()?;
        Ok(Action::PreviewInstall {
            tool,
            account: self.account(),
            route: route.key.clone(),
            id: route.id.clone(),
            settings_file: self.custom_file(),
        })
    }

    /// This function returns the last install preview while it still describes the selection.
    fn install_preview(&self) -> Option<&Value> {
        let (
            Action::PreviewInstall {
                tool,
                account,
                route,
                id,
                settings_file,
            },
            preview,
        ) = self.install_preview.as_ref()?
        else {
            return None;
        };
        let selected = self.selected_route().ok()?;
        let destination = match (&self.destination, account, settings_file) {
            (Destination::Harness, None, None) => true,
            (Destination::Account(name), Some(previewed), None) => name == previewed,
            (Destination::File(path), None, Some(previewed)) => path == previewed,
            _ => false,
        };
        let current = destination
            && self.harness().is_ok_and(|harness| harness.tool == *tool)
            && selected.key == *route
            && selected.id == *id;
        current.then_some(preview)
    }

    /// Short messages appear in the status bar; long ones open a window so nothing is cut off.
    fn notify(&mut self, text: String, tone: Tone) {
        if text.is_empty() {
            return;
        }
        if text.contains('\n') || text.chars().count() + 8 > usize::from(self.width) * 2 / 3 {
            self.popup = Some(Popup::Message {
                title: match tone {
                    Tone::Done => "Done",
                    Tone::Warning => "Attention",
                    Tone::Error => "Error",
                },
                text,
                tone,
                scroll: 0,
            });
        } else {
            self.toast = Some(Toast {
                text,
                tone,
                shown: Instant::now(),
            });
        }
    }

    /// This function applies a finished job. It returns the follow-up request, if any.
    fn finish(
        &mut self,
        reply: Option<Result<Reply, String>>,
        snapshot: Result<Box<Snapshot>, String>,
    ) -> Option<Request> {
        let (action, key) = match self.busy.take() {
            Some(busy) => (busy.action, busy.key),
            None => (None, None),
        };
        self.apply_snapshot(snapshot.map(|snapshot| *snapshot));
        let reply = reply?;
        match (&action, reply.is_ok()) {
            // Saved settings no longer match an earlier diff, and a failed preview has none.
            (Some(Action::Install { .. } | Action::Restore { .. }), true)
            | (Some(Action::PreviewInstall { .. }), false) => self.install_preview = None,
            // The archived account leaves the list, so the harness's own settings are next.
            (Some(Action::ArchiveAccount { .. }), true) => self.destination = Destination::Harness,
            _ => {}
        }
        match (action.as_ref().map(ActionKind::from), reply) {
            (Some(ActionKind::ServerKeys), Err(error)) => {
                self.server_keys = None;
                self.notify(error, Tone::Error);
            }
            // A form stays open with the error, so the user can fix the input and submit again.
            (_, Err(error)) => match self.popup.as_mut().and_then(Popup::form_mut) {
                Some(form) => form.error = error,
                None => self.notify(error, Tone::Error),
            },
            (Some(ActionKind::PreviewUpdate), Ok(reply)) => {
                let token = reply.data["preview_token"].as_str().map(str::to_owned);
                self.popup = Some(Popup::review(
                    update_lines(self.theme, &reply.data),
                    "Start",
                    FormKind::Update(token),
                    "Update from source?",
                    "Type UPDATE to start",
                ));
            }
            // The p key shows the diff on the page. The i key also opens the review, unless the
            // settings already match the route.
            (Some(ActionKind::PreviewInstall), Ok(reply)) => {
                let previewed = action?;
                let changed = reply.data["changes"]
                    .as_array()
                    .is_some_and(|changes| !changes.is_empty());
                if !changed {
                    self.notify("The settings already match this route.".into(), Tone::Done);
                } else if key == Some('i')
                    && let Some(install @ Action::Install { tool, .. }) =
                        reviewed(previewed.clone(), &reply.data)
                {
                    let known = self.data.tools.iter().find(|known| known.tool == tool);
                    let label = known.map_or("the harness", |known| known.label);
                    let title = format!("Apply routing to {label}?");
                    self.popup = Some(Popup::review(
                        install_lines(self.theme, &reply.data),
                        "Install",
                        FormKind::Confirm("INSTALL".into(), Box::new(install)),
                        title,
                        "Type INSTALL to save the settings",
                    ));
                } else {
                    let text = "Press i to apply this diff.";
                    self.notify(text.into(), Tone::Done);
                }
                self.install_preview = Some((previewed, reply.data));
            }
            (Some(ActionKind::PreviewRestore), Ok(reply)) => {
                if let Some(restore) = action.and_then(|action| reviewed(action, &reply.data)) {
                    self.popup = Some(Popup::review(
                        restore_lines(self.theme, &reply.data),
                        "Restore",
                        FormKind::Confirm("RESTORE".into(), Box::new(restore)),
                        "Restore harness settings?",
                        "Type RESTORE to replace the files",
                    ));
                }
            }
            (Some(ActionKind::PreviewAccountArchive), Ok(reply)) => {
                if let Some(archive) = action.and_then(|action| reviewed(action, &reply.data))
                    && let Action::ArchiveAccount { name, .. } = &archive
                {
                    let name = name.clone();
                    self.popup = Some(Popup::review(
                        archive_lines(self.theme, &reply.data),
                        "Archive",
                        FormKind::Confirm(name.clone().into(), Box::new(archive)),
                        format!("Archive the account {name}?"),
                        "Type the account name to archive",
                    ));
                }
            }
            // The reply lists the roles of the chosen algorithm. The models form replaces the
            // algorithm form only while that form is still open for the same route.
            (Some(ActionKind::Editor), Ok(reply)) => {
                if let Some(Action::Editor {
                    route: key,
                    algorithm,
                    ..
                }) = action
                    && let Some(Popup::Form(form)) = &self.popup
                    && let FormKind::Algorithm { generation, route } = &form.kind
                    && route.key == key
                {
                    match route_models_form(*generation, route, algorithm, &reply.data) {
                        Ok(next) => self.popup = Some(Popup::Form(next)),
                        Err(error) => self.notify(error, Tone::Error),
                    }
                }
            }
            // The review window replaces the form that asked for the preview, so a form that
            // the user closed meanwhile stays closed.
            (Some(ActionKind::PreviewRoute | ActionKind::PreviewRouteChange), Ok(reply)) => {
                if matches!(self.popup, Some(Popup::Form(_)))
                    && let Some(reviewed) = action.and_then(|action| reviewed(action, &reply.data))
                {
                    self.popup = Some(Popup::review(
                        change_lines(self.theme, &reply.data),
                        "Save",
                        FormKind::Confirm("SAVE".into(), Box::new(reviewed)),
                        "Review route changes",
                        "Type SAVE to apply",
                    ));
                }
            }
            (Some(ActionKind::Models), Ok(reply)) => {
                let models: Vec<&str> = reply.data["models"]
                    .as_array()
                    .map(|items| items.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                let list = if models.is_empty() {
                    "No model IDs found.".to_string()
                } else {
                    models.join("\n")
                };
                let client = reply.data["client"].as_str().unwrap_or_default();
                self.popup = Some(Popup::Message {
                    title: "Models",
                    text: format!("{}\n\nModel IDs on {client}:\n{list}", reply.message),
                    tone: Tone::of(&reply),
                    scroll: 0,
                });
            }
            (Some(ActionKind::ServerKeys), Ok(reply)) => {
                self.server_keys = Some(reply.data);
                self.notify("Checked the saved server keys.".into(), Tone::Done);
            }
            (
                Some(
                    ActionKind::SaveServerKey
                    | ActionKind::ImportServerKey
                    | ActionKind::RemoveServerKey,
                ),
                Ok(reply),
            ) => {
                self.popup = None;
                self.notify(reply.message, Tone::Done);
                return Some(Request::run(
                    Action::ServerKeys {},
                    "Checking the server keys",
                ));
            }
            (_, Ok(reply)) => {
                if matches!(self.popup, Some(Popup::Form(_) | Popup::Review { .. })) {
                    self.popup = None;
                }
                let tone = Tone::of(&reply);
                let details: Vec<&str> = reply.data["details"]
                    .as_array()
                    .map(|items| items.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                if details.is_empty() {
                    self.notify(reply.message, tone);
                } else {
                    self.popup = Some(Popup::Message {
                        title: if tone == Tone::Warning {
                            "Attention"
                        } else {
                            "Details"
                        },
                        text: format!("{}\n\n{}", reply.message, details.join("\n")),
                        tone,
                        scroll: 0,
                    });
                }
            }
        }
        None
    }

    fn refresh_due(&self) -> bool {
        // The placeholder snapshot after a failed start has no interval; 30 seconds is the default.
        let seconds = match self.data.refresh_seconds {
            0 => 30,
            seconds => seconds,
        };
        self.busy.is_none() && self.last_refresh.elapsed() >= Duration::from_secs(seconds)
    }

    fn start(&mut self, jobs: &mpsc::Sender<Job>, request: Request) -> Result<(), String> {
        if let Some(busy) = &self.busy {
            let text = format!("Wait for \"{}\" to finish.", busy.label);
            self.notify(text, Tone::Error);
            return Ok(());
        }
        let action = match &request.job {
            Job::Refresh => {
                self.last_refresh = Instant::now();
                None
            }
            Job::Run(action) => Some(action.clone()),
        };
        jobs.send(request.job).map_err(|error| error.to_string())?;
        self.busy = Some(Busy {
            label: request.label,
            action,
            key: request.key,
            started: Instant::now(),
        });
        Ok(())
    }

    fn keys(&self) -> &'static [Key] {
        match (&self.popup, self.filtering, self.tab) {
            (Some(Popup::Restart), _, _) => CONFIRM_KEYS,
            (Some(Popup::Review { .. }), _, _) => REVIEW_KEYS,
            (Some(Popup::Form(form)), _, _) if form.options().is_some() => CHOICE_KEYS,
            (Some(Popup::Form(_)), _, _) => FORM_KEYS,
            (Some(_), _, _) => POPUP_KEYS,
            (None, true, _) => FILTER_KEYS,
            (None, false, Tab::Overview) => &[],
            (None, false, Tab::Calls) => CALLS_KEYS,
            (None, false, Tab::Models) => MODELS_KEYS,
            (None, false, Tab::Routes) => ROUTES_KEYS,
            (None, false, Tab::Harnesses) => HARNESSES_KEYS,
            (None, false, Tab::Settings) => SETTINGS_KEYS,
        }
    }

    /// The Calls and Routes pages each keep their own filter; `/` edits the current page's.
    fn filter_mut(&mut self) -> &mut String {
        if self.tab == Tab::Routes {
            &mut self.route_filter
        } else {
            &mut self.filter
        }
    }

    fn refilter(&mut self) {
        if self.tab == Tab::Routes {
            let anchor = self.selected_route_key();
            self.route_selected = 0;
            self.rebuild_routes(anchor);
        } else {
            self.selected = 0;
            self.rebuild_calls();
        }
    }

    fn paste(&mut self, text: &str) {
        if let Some(form) = self.popup.as_mut().and_then(Popup::form_mut) {
            form.insert(text);
        } else if self.filtering && self.popup.is_none() {
            self.filter_mut()
                .extend(text.chars().filter(|c| !c.is_control()));
            self.refilter();
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
            KeyCode::Char('?') => self.popup = Some(Popup::Help { scroll: 0 }),
            KeyCode::Char(digit @ '1'..='6') => {
                self.tab = TABS[digit as usize - '1' as usize];
            }
            KeyCode::Tab | KeyCode::Right => self.tab = TABS[(self.tab.index() + 1) % TABS.len()],
            KeyCode::BackTab | KeyCode::Left => {
                self.tab = TABS[(self.tab.index() + TABS.len() - 1) % TABS.len()];
            }
            KeyCode::Char('r') | KeyCode::F(5) => return Some(Request::refresh()),
            KeyCode::Esc => {
                self.toast = None;
                match self.tab {
                    Tab::Calls => {
                        self.filter.clear();
                        self.session = None;
                        self.rebuild_calls();
                    }
                    Tab::Routes => {
                        self.route_filter.clear();
                        self.refilter();
                    }
                    Tab::Overview | Tab::Models | Tab::Harnesses | Tab::Settings => {}
                }
            }
            code => {
                // Action letters come before the page's own keys, so `k` runs an action on
                // Routes and Settings instead of moving up.
                if let KeyCode::Char(letter) = code
                    && let Some((index, &(_, key, _, _, flow))) =
                        rows(self.tab).enumerate().find(|(_, row)| row.1 == letter)
                {
                    if self.tab == Tab::Settings {
                        self.action = index;
                    }
                    return self.run_row(flow[0], key);
                }
                match self.tab {
                    Tab::Overview => {}
                    Tab::Calls => self.calls_key(code),
                    Tab::Models => match code {
                        KeyCode::Char('p' | ']') => self.period = self.period.next(),
                        KeyCode::Char('[') => self.period = self.period.previous(),
                        _ => {}
                    },
                    Tab::Routes => self.routes_key(code),
                    Tab::Harnesses => self.harnesses_key(code),
                    Tab::Settings => return self.settings_key(code),
                }
            }
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
            KeyCode::Enter if self.selected_entry().is_some() => {
                self.popup = Some(Popup::Detail { scroll: 0 });
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

    fn routes_key(&mut self, code: KeyCode) {
        let last = self.route_visible.len().saturating_sub(1);
        if let Some(selected) = step(code, self.route_selected, last) {
            self.route_selected = selected;
        } else if code == KeyCode::Char('/') {
            self.filtering = true;
        }
    }

    /// The arrows choose the route, which is the longer list; `t` cycles through the few
    /// harnesses. A new harness starts at its own settings, because accounts belong to one
    /// harness.
    fn harnesses_key(&mut self, code: KeyCode) {
        let last = self.route_visible.len().saturating_sub(1);
        if let Some(selected) = step(code, self.route_selected, last) {
            self.route_selected = selected;
            return;
        }
        match code {
            KeyCode::Char('t') => {
                self.harness = (self.harness + 1) % self.data.tools.len().max(1);
                self.destination = Destination::Harness;
            }
            // The cycle runs through the harness's own settings and then each saved account.
            KeyCode::Char('a') => {
                let names = self.harness().map_or(&[][..], |tool| &tool.accounts);
                let current = match &self.destination {
                    Destination::Account(name) => names.iter().position(|known| known == name),
                    Destination::Harness | Destination::File(_) => None,
                };
                let next = names.get(current.map_or(0, |index| index + 1)).cloned();
                self.destination = next.map_or(Destination::Harness, Destination::Account);
            }
            KeyCode::Char('x') => {
                self.popup = Some(Popup::Form(Form::new(
                    FormKind::SettingsFile,
                    "Custom settings file · leave empty to use the detected settings",
                    vec![(
                        "Absolute file path (Pi: models.json)",
                        self.custom_file().unwrap_or_default(),
                    )],
                )));
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
                self.filter_mut().clear();
            }
            KeyCode::Backspace => {
                self.filter_mut().pop();
            }
            KeyCode::Char('u') if control => self.filter_mut().clear(),
            KeyCode::Char('w') if control => {
                let filter = self.filter_mut();
                let kept = filter.trim_end().rfind(' ').map_or(0, |index| index + 1);
                filter.truncate(kept);
            }
            KeyCode::Char(c) if !control && !key.modifiers.contains(KeyModifiers::ALT) => {
                self.filter_mut().push(c);
            }
            _ => return,
        }
        self.refilter();
    }

    fn settings_key(&mut self, code: KeyCode) -> Option<Request> {
        if let Some(action) = step(code, self.action, rows(Tab::Settings).count() - 1) {
            self.action = action;
            return None;
        }
        if code == KeyCode::Enter
            && let Some(&(_, key, _, _, flow)) = rows(Tab::Settings).nth(self.action)
        {
            return self.run_row(flow[0], key);
        }
        None
    }

    /// This function runs a row's flow and records its key on the request, so the reply can
    /// continue that key's flow.
    fn run_row(&mut self, kind: ActionKind, key: char) -> Option<Request> {
        match self.start_action(kind, key) {
            Ok(request) => request.map(|request| Request {
                key: Some(key),
                ..request
            }),
            Err(error) => {
                self.notify(error, Tone::Error);
                None
            }
        }
    }

    fn open(&mut self, form: Form) -> Result<Option<Request>, String> {
        self.popup = Some(Popup::Form(form));
        Ok(None)
    }

    /// This function starts the flow of an `ACTIONS` row. The match has no wildcard arm, so
    /// adding a controller action breaks the build until the terminal app can run it. Only the
    /// arms of actions that several keys start read `key`, and they return an error for a key
    /// that they do not know, so a new row must also be added to its arm.
    fn start_action(&mut self, kind: ActionKind, key: char) -> Result<Option<Request>, String> {
        let request = |action, label| Ok(Some(Request::run(action, label)));
        let generation = self.data.generation;
        match kind {
            // The algorithm form submits Editor, the models form submits PreviewRoute, and the
            // review window submits Apply.
            ActionKind::Editor | ActionKind::PreviewRoute | ActionKind::Apply => {
                let route = Box::new(self.selected_route()?.clone());
                let title = format!("Routing algorithm for {}", route.id);
                let fields = vec![("Algorithm", route.kind.clone())];
                self.open(Form::new(
                    FormKind::Algorithm { generation, route },
                    title,
                    fields,
                ))
            }
            // The route form submits PreviewRouteChange, and the review window submits
            // RouteChange. A new route also needs a models form before the review.
            ActionKind::PreviewRouteChange | ActionKind::RouteChange => {
                const CONFIG_NAME: &str = "Config name (blank uses the public model ID)";
                let form = match key {
                    'a' => {
                        let algorithm = self.data.algorithms.first().map(|a| a.kind);
                        Form::new(
                            FormKind::NewRoute { generation },
                            "Create a route",
                            vec![
                                ("Public model ID", String::new()),
                                (CONFIG_NAME, String::new()),
                                ("Algorithm", algorithm.unwrap_or_default().to_string()),
                            ],
                        )
                    }
                    'n' => {
                        let route = self.selected_route()?;
                        Form::new(
                            FormKind::RouteIdentity {
                                generation,
                                operation: "rename",
                                route: route.key.clone(),
                            },
                            format!("Rename {}", route.id),
                            vec![("Public model ID", route.id.clone())],
                        )
                    }
                    'c' => {
                        let route = self.selected_route()?;
                        Form::new(
                            FormKind::RouteIdentity {
                                generation,
                                operation: "duplicate",
                                route: route.key.clone(),
                            },
                            format!("Duplicate {}", route.id),
                            vec![
                                ("Public model ID", String::new()),
                                (CONFIG_NAME, String::new()),
                            ],
                        )
                    }
                    _ => return Err(format!("The {key} key does not start a route change.")),
                };
                self.open(form)
            }
            ActionKind::Remove => {
                let route = self.selected_route()?;
                let lines = remove_lines(self.theme, route, self.data.settings.as_ref());
                let title = format!("Delete route {}?", route.id);
                let kind = FormKind::Remove {
                    generation,
                    route: route.key.clone(),
                    id: route.id.clone(),
                    revision: route.revision.clone(),
                };
                self.popup = Some(Popup::review(
                    lines,
                    "Delete",
                    kind,
                    title,
                    "Type the route name to delete",
                ));
                Ok(None)
            }
            ActionKind::Models => {
                let client = self.data.clients.first();
                let client = client.map(|c| c.name.clone()).unwrap_or_default();
                let form = match key {
                    'm' => Form::new(
                        FormKind::Models,
                        "Load an endpoint's models",
                        vec![("Endpoint name", client)],
                    ),
                    'k' => Form::new(
                        FormKind::ModelKey,
                        "Save a model-list key",
                        vec![("Endpoint name", client), ("API key", String::new())],
                    ),
                    _ => return Err(format!("The {key} key does not load models.")),
                };
                self.open(form)
            }
            // Both keys preview the install. App::finish shows the diff for p and opens the
            // review, which submits Install, for i.
            ActionKind::PreviewInstall | ActionKind::Install => {
                let label = match key {
                    'p' => "Previewing the settings",
                    'i' => "Reviewing the settings",
                    _ => return Err(format!("The {key} key does not preview harness settings.")),
                };
                request(self.install_action()?, label)
            }
            // The review window lists the files that return to their backups and submits
            // Restore.
            ActionKind::PreviewRestore | ActionKind::Restore => {
                let tool = self.harness()?.tool;
                let restore = Action::PreviewRestore {
                    tool,
                    account: self.account(),
                    settings_file: self.custom_file(),
                };
                request(restore, "Reading the original settings")
            }
            ActionKind::AddAccount => {
                let harness = self.harness()?;
                let title = format!("Sign in to another {} account", harness.label);
                let kind = FormKind::Account(harness.tool);
                self.open(Form::new(
                    kind,
                    title,
                    vec![("Account name", String::new())],
                ))
            }
            ActionKind::LoginAccount => {
                let (tool, name) = self.selected_account()?;
                request(Action::LoginAccount { tool, name }, "Opening the login")
            }
            // Controller::editor_target returns the account folder, so it opens in the editor.
            ActionKind::OpenAccount => {
                let (tool, name) = self.selected_account()?;
                request(
                    Action::OpenAccount { tool, name },
                    "Opening the account folder",
                )
            }
            // The review window names the account folder and submits ArchiveAccount.
            ActionKind::PreviewAccountArchive | ActionKind::ArchiveAccount => {
                let (tool, name) = self.selected_account()?;
                let archive = Action::PreviewAccountArchive { tool, name };
                request(archive, "Reviewing the account archive")
            }
            ActionKind::OpenConfig => request(Action::OpenConfig {}, "Opening the server config"),
            ActionKind::CheckConfig => {
                request(Action::CheckConfig {}, "Checking the server config")
            }
            ActionKind::Restart => {
                self.popup = Some(Popup::Restart);
                Ok(None)
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
                self.open(Form::new(
                    FormKind::Preferences,
                    "Change preferences",
                    vec![
                        ("Baseline model ID", baseline),
                        ("Refresh interval in seconds", refresh),
                    ],
                ))
            }
            ActionKind::OpenPrices => request(Action::OpenPrices {}, "Opening the model prices"),
            ActionKind::ServerKeys => request(Action::ServerKeys {}, "Checking the server keys"),
            ActionKind::SaveServerKey => self.open(Form::new(
                FormKind::SaveKey,
                "Save a server key",
                vec![("Key name", String::new()), ("Key", String::new())],
            )),
            ActionKind::ImportServerKey => self.open(Form::new(
                FormKind::ImportKey,
                "Import a server key from your login shell",
                vec![("Key name", String::new())],
            )),
            ActionKind::RemoveServerKey => self.open(Form::new(
                FormKind::RemoveKey,
                "Remove a saved server key",
                vec![("Key name", String::new())],
            )),
            // The review window submits Update after the user reads this preview.
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
                        self.popup = None;
                        return Some(Request::run(Action::Restart {}, "Restarting the server"));
                    }
                    KeyCode::Char('n' | 'N' | 'q') | KeyCode::Esc | KeyCode::Enter => {
                        self.popup = None;
                    }
                    _ => {}
                }
                return None;
            }
            Popup::Form(form) | Popup::Review { confirm: form, .. } => {
                match key.code {
                    KeyCode::Esc => self.popup = None,
                    KeyCode::Enter => match form.action(&self.data) {
                        Ok(Submit::Run(action)) => return Some(Request::run(action, form.label())),
                        Ok(Submit::Next(next)) => self.popup = Some(Popup::Form(next)),
                        Ok(Submit::SettingsFile(path)) => {
                            self.popup = None;
                            self.destination = if path.is_empty() {
                                Destination::Harness
                            } else {
                                Destination::File(path)
                            };
                            let text = "Press p to preview or i to install.";
                            self.notify(text.into(), Tone::Done);
                        }
                        Err(error) => form.error = error,
                    },
                    // A field with choices cycles through them. Otherwise only the review
                    // window scrolls; text fields ignore these keys.
                    KeyCode::Up | KeyCode::Down | KeyCode::PageUp | KeyCode::PageDown => {
                        let forward = key.code == KeyCode::Down;
                        match (form.options(), key.code) {
                            (Some(Choices::Algorithms), KeyCode::Up | KeyCode::Down) => {
                                form.choose(self.data.algorithms.iter().map(|a| a.kind), forward);
                            }
                            (Some(Choices::Endpoints), KeyCode::Up | KeyCode::Down) => {
                                let names = self.data.clients.iter().map(|c| c.name.as_str());
                                form.choose(names, forward);
                            }
                            _ => {
                                if let Some(scroll) = popup.scroll_mut()
                                    && let Some(next) =
                                        step(key.code, usize::from(*scroll), usize::from(u16::MAX))
                                {
                                    *scroll = next as u16;
                                }
                            }
                        }
                    }
                    _ => form.edit(key),
                }
                return None;
            }
            _ => {}
        }
        match key.code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q' | '?') => self.popup = None,
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

/// Every word must appear in one of the call's IDs.
fn matches_filter(entry: &Entry, filter: &str) -> bool {
    let fields = [
        entry.model.as_str(),
        entry.route_id.as_str(),
        entry.session_id.as_deref().unwrap_or_default(),
        entry.turn_id.as_deref().unwrap_or_default(),
    ];
    matches_words(&fields, filter)
}

/// Every word must appear in one of the fields. A word with a capital letter matches case
/// exactly; other words ignore ASCII case, which covers model, route, and session IDs.
fn matches_words(fields: &[&str], filter: &str) -> bool {
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

/// Single calls often cost fractions of a cent, so this shows more digits than `money`.
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
    // This destructuring has no `..`, so adding a snapshot field breaks the build until a page
    // shows it. Each comment names the place that shows the field.
    let Snapshot {
        update_status: _,   // draw_settings
        settings: _,        // draw_settings
        data_state: _,      // every page checks availability; draw_overview shows health
        generation: _,      // every route form sends it back with its action
        routes: _,          // draw_routes and the route details
        clients: _,         // draw_endpoints, the endpoint fields, and the endpoint forms
        tools: _,           // draw_harnesses and harness_lines
        algorithms: _,      // draw_routes, the route details, and the algorithm fields
        summary: _,         // --print shows these rows; draw_top_models builds the same notes
        metrics: _,         // draw_header and the Overview cards
        today: _,           // the Overview "Today" card title
        analytics: _,       // draw_top_models and draw_models
        estimates: _,       // the Overview cards and draw_models
        baseline_model: _,  // draw_top_models; draw_settings shows the saved setting
        activity: _,        // draw_activity
        sessions: _,        // the calls table title and the `s` key
        entries: _,         // draw_call_table and the call details
        limited: _,         // calls_block and draw_activity
        skipped: _,         // calls_block
        errors: _,          // draw_errors and draw_settings
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
        Tab::Routes => draw_routes(frame, app, body),
        Tab::Harnesses => draw_harnesses(frame, app, body),
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
    // Each title is the tab's digit and at most `letters` letters of its name.
    let titles = |letters: usize| {
        TABS.map(|tab| {
            let name = tab.name().get(..letters).unwrap_or(tab.name());
            Line::from(vec![
                Span::styled((tab.index() + 1).to_string(), theme.key),
                Span::raw(if name.is_empty() { "" } else { " " }),
                Span::raw(name),
            ])
        })
    };
    // Tabs pads each title by one space on both sides and puts a divider between titles.
    let width = |titles: &[Line]| titles.iter().map(|title| title.width() + 3).sum::<usize>();
    // Narrow terminals shorten the names to three letters, then show the digits alone, so no
    // tab is cut off.
    let titles = [usize::MAX, 3]
        .into_iter()
        .map(titles)
        .find(|titles| width(titles) <= usize::from(area.width))
        .unwrap_or_else(|| titles(0));
    let tabs_width = width(&titles);
    let running = app.data.metrics.running;
    let mut status = server_state(theme, running).to_vec();
    let updated = match app.refreshed {
        Some(time) => format!(" · updated {}s ago ", time.elapsed().as_secs()),
        None => " · no data yet ".into(),
    };
    let url = format!(" · {}", app.data.metrics.server_url);
    // Narrow terminals drop the URL first, then the update time, then shorten the server state.
    // The narrowest ones drop the server state too, which the Overview page still shows, so the
    // tabs stay readable.
    let room = usize::from(area.width).saturating_sub(tabs_width);
    let base = status.iter().map(Span::width).sum::<usize>();
    if base > room {
        status[1] = Span::raw(if running { "up" } else { "down" });
        if status.iter().map(Span::width).sum::<usize>() > room {
            status.clear();
        }
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

fn draw_errors(frame: &mut Frame, app: &App, area: Rect) {
    let lines: Vec<Line> = app
        .data
        .errors
        .iter()
        .map(|error| Line::styled(error.as_str(), app.theme.error))
        .collect();
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block(app.theme, "Data errors").border_style(app.theme.error)),
        area,
    );
}

fn draw_overview(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let data = &app.data;
    let errors = if data.errors.is_empty() {
        0
    } else {
        (data.errors.len() as u16 + 2).min(area.height / 2)
    };
    let [errors_area, rest] =
        Layout::vertical([Constraint::Length(errors), Constraint::Fill(1)]).areas(area);
    if errors > 0 {
        draw_errors(frame, app, errors_area);
    }
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

/// This function draws one period's calls, spending, and savings for an Overview card.
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
            .title_bottom(Line::styled(" from the latest 5,000 calls ", theme.muted).centered());
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
    if app.data.data_state.history.state == Availability::Unavailable {
        draw_errors(frame, app, area);
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

/// This function returns the first shown row of a list, moved only as far as needed to keep
/// the selection inside the `height` rows that fit.
fn keep_visible(selected: usize, offset: usize, len: usize, height: usize) -> usize {
    let offset = if selected < offset {
        selected
    } else if selected >= offset + height {
        selected + 1 - height
    } else {
        offset
    };
    offset.min(len.saturating_sub(height))
}

/// This function draws a scrollbar over a bordered table's data rows, below the top border
/// and the header row, when the rows do not fit.
fn draw_scrollbar(frame: &mut Frame, theme: Theme, area: Rect, len: usize, position: usize) {
    if len <= usize::from(area.height.saturating_sub(3)) {
        return;
    }
    let mut scroll = ScrollbarState::new(len).position(position);
    frame.render_stateful_widget(
        Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .thumb_style(theme.key)
            .track_symbol(Some("│"))
            .track_style(theme.border),
        Rect {
            y: area.y + 2,
            height: area.height.saturating_sub(3),
            ..area
        },
        &mut scroll,
    );
}

fn draw_call_table(frame: &mut Frame, app: &mut App, area: Rect) {
    // The table shows this many calls; the borders and the header row take three rows.
    let height = usize::from(area.height.saturating_sub(3)).max(1);
    app.offset = keep_visible(app.selected, app.offset, app.visible.len(), height);
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
    draw_scrollbar(frame, theme, area, app.visible.len(), app.selected);
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
    if app.data.data_state.usage.state == Availability::Unavailable {
        draw_errors(frame, app, area);
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

/// This function returns the algorithm's name, or the route type for a type the apps cannot
/// write.
fn algorithm_title<'a>(data: &'a Snapshot, kind: &'a str) -> &'a str {
    data.algorithms
        .iter()
        .find(|algorithm| algorithm.kind == kind)
        .map_or(kind, |algorithm| algorithm.title)
}

fn draw_routes(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = app.theme;
    // A server config that cannot be read lists no routes, so its error explains the empty list.
    let errors = if app.data.routes.is_empty() && !app.data.errors.is_empty() {
        (app.data.errors.len() as u16 + 2).min(area.height / 3)
    } else {
        0
    };
    let notes = endpoint_notes(theme, &app.data);
    let endpoints = (app.data.clients.len().max(1) + notes.len() + 3) as u16;
    let [errors_area, top, endpoints_area] = Layout::vertical([
        Constraint::Length(errors),
        Constraint::Fill(1),
        Constraint::Length(endpoints.min(area.height / 3)),
    ])
    .areas(area);
    if errors > 0 {
        draw_errors(frame, app, errors_area);
    }
    draw_endpoints(frame, app, notes, endpoints_area);
    let (title, lines) = match app.selected_route() {
        Ok(route) => (route.id.clone(), route_lines(theme, &app.data, route)),
        Err(_) => (
            "Route".to_string(),
            vec![Line::styled("No route selected.", theme.muted)],
        ),
    };
    let detail = Paragraph::new(lines).wrap(Wrap { trim: false });
    // Wide terminals put the details beside the list; others put them below it.
    let (list, detail_area) = if top.width >= SIDE_DETAIL_WIDTH {
        let [list, side] =
            Layout::horizontal([Constraint::Fill(1), Constraint::Length(52)]).areas(top);
        (list, side)
    } else {
        // The list keeps its borders, header, and the selected row; the details get the rest.
        let rows = detail.line_count(top.width.saturating_sub(4)) as u16 + 2;
        let [list, below] = Layout::vertical([
            Constraint::Fill(1),
            Constraint::Length(rows.min(top.height.saturating_sub(4))),
        ])
        .areas(top);
        (list, below)
    };
    draw_route_table(frame, app, list);
    frame.render_widget(
        detail.block(block(theme, &title).padding(Padding::horizontal(1))),
        detail_area,
    );
}

fn draw_route_table(frame: &mut Frame, app: &mut App, area: Rect) {
    // The table shows this many routes; the borders and the header row take three rows.
    let height = usize::from(area.height.saturating_sub(3)).max(1);
    let visible = app.route_visible.len();
    app.route_offset = keep_visible(app.route_selected, app.route_offset, visible, height);
    let app = &*app;
    let theme = app.theme;
    let mut title = vec![
        Span::styled(" Routes ", Modifier::BOLD),
        Span::styled(format!("[{}] ", count(visible as u64)), theme.muted),
    ];
    if !app.route_filter.is_empty() || app.filtering {
        title.push(Span::styled(format!("/{} ", app.route_filter), theme.key));
    }
    let mut outer = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(if app.filtering {
            theme.key
        } else {
            theme.border
        })
        .title(Line::from(title));
    if visible == 0 {
        let text = if app.data.routes.is_empty() {
            "No routes in the server config. Press a to create one."
        } else {
            "No routes match the filter. Press Esc to clear it."
        };
        frame.render_widget(
            Paragraph::new(Line::styled(text, theme.muted))
                .centered()
                .block(outer.padding(Padding::top(1))),
            area,
        );
        return;
    }
    outer = outer.title_bottom(
        Line::styled(
            format!(" {} of {} ", app.route_selected + 1, count(visible as u64)),
            theme.muted,
        )
        .right_aligned(),
    );
    let wide = area.width >= 80;
    let shown = |column: &(Column, bool)| wide || !column.1;
    let rows: Vec<Row> = app.route_visible[app.route_offset..]
        .iter()
        .take(height)
        .map(|&index| {
            let route = &app.data.routes[index];
            let models = if route.editable {
                let models: Vec<String> = route
                    .choices
                    .iter()
                    .map(|choice| format!("{} / {}", choice.client, choice.model))
                    .collect();
                Span::raw(models.join(", "))
            } else {
                Span::styled("set in the server config", theme.muted)
            };
            let cells = [
                Cell::from(route.id.clone()),
                Cell::from(algorithm_title(&app.data, &route.kind).to_string()),
                Cell::from(models),
            ];
            Row::new(
                cells
                    .into_iter()
                    .zip(&ROUTE_COLUMNS)
                    .filter(|(_, column)| shown(column))
                    .map(|(cell, _)| cell),
            )
        })
        .collect();
    let columns: Vec<Column> = ROUTE_COLUMNS
        .iter()
        .filter(|column| shown(column))
        .map(|column| column.0)
        .collect();
    let mut state =
        TableState::default().with_selected(Some(app.route_selected - app.route_offset));
    frame.render_stateful_widget(
        table(theme, &columns, rows)
            .row_highlight_style(theme.selected)
            .highlight_symbol("› ")
            .block(outer),
        area,
        &mut state,
    );
    draw_scrollbar(frame, theme, area, visible, app.route_selected);
}

/// This function shows the selected route's algorithm and the model of each role first, so a
/// short terminal still shows them, then the notes that limit editing it here.
fn route_lines(theme: Theme, data: &Snapshot, route: &RouteInfo) -> Vec<Line<'static>> {
    const WIDTH: usize = 11;
    // This destructuring has no `..`, so adding a route field breaks the build until the
    // terminal app shows it. Each comment names the place that shows the field.
    let RouteInfo {
        key,
        id: _, // draw_routes titles these details with it
        kind,
        revision: _, // the edit and delete flows send it back
        label: _,    // harness_lines names the route with it, and the filter reads it
        choices,
        tiers: _, // route_models_form keeps each role's model by tier
        editable,
        generated,
        compatibility: _, // harness_lines explains it for the selected harness
        dependents: _,    // remove_lines lists them in the delete review
    } = route;
    let field = |name: &str, value: String| labeled(theme, name, WIDTH, value);
    let algorithm = data.algorithms.iter().find(|known| known.kind == *kind);
    let mut lines = vec![
        field("Config", format!("[routes.{key}]")),
        field(
            "Algorithm",
            algorithm.map_or_else(
                || kind.clone(),
                |known| format!("{} ({})", known.title, known.kind),
            ),
        ),
    ];
    // A random route can list more models than a new one starts with, so later roles are
    // numbered the same way.
    for (index, choice) in choices.iter().enumerate() {
        let role = algorithm
            .and_then(|known| known.roles.get(index))
            .map_or_else(|| format!("Model {}", index + 1), |role| role.label.clone());
        lines.push(if choice.model.is_empty() {
            Line::from(vec![
                Span::styled(format!("{role:<WIDTH$}"), theme.muted),
                Span::styled("No target", theme.warn),
            ])
        } else {
            field(&role, format!("{} / {}", choice.client, choice.model))
        });
    }
    if !editable {
        lines.push(Line::default());
        lines.push(Line::styled(
            "This routing method has no editor here. Edit its fields in the server config.",
            theme.warn,
        ));
    }
    if let Some(generated) = generated {
        lines.push(Line::default());
        lines.push(Line::styled(
            format!("Generated by {generated}. That tool can replace edits made here."),
            theme.warn,
        ));
    }
    if let Some(known) = algorithm {
        lines.push(Line::default());
        lines.push(Line::styled(known.summary, theme.muted));
    }
    lines
}

/// This function explains which endpoints cannot list their models.
fn endpoint_notes(theme: Theme, data: &Snapshot) -> Vec<Line<'static>> {
    let unlisted = "The app cannot list this endpoint's models. Type model IDs by hand.";
    data.clients
        .iter()
        .flat_map(|client| {
            let unlisted = client.unlisted.then(|| unlisted.to_string());
            unlisted.into_iter().chain(client.note.clone()).map(|note| {
                Line::from(vec![
                    Span::styled(format!("{}: ", client.name), theme.muted),
                    Span::styled(note, theme.warn),
                ])
            })
        })
        .collect()
}

fn draw_endpoints(frame: &mut Frame, app: &App, notes: Vec<Line<'static>>, area: Rect) {
    let theme = app.theme;
    let clients = &app.data.clients;
    let outer = block(theme, &format!("Endpoints [{}]", clients.len()));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);
    if clients.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::styled(
                "The server config names no endpoints.",
                theme.muted,
            )),
            inner,
        );
        return;
    }
    let [table_area, notes_area] = Layout::vertical([
        Constraint::Length(clients.len() as u16 + 1),
        Constraint::Fill(1),
    ])
    .areas(inner);
    let rows: Vec<Row> = clients
        .iter()
        .map(|client| {
            Row::new([
                client.name.clone(),
                client.host.clone(),
                client.models.join(", "),
            ])
        })
        .collect();
    let columns = [
        ("NAME", Constraint::Fill(1), false),
        ("HOST", Constraint::Fill(1), false),
        ("MODELS IN THE SERVER CONFIG", Constraint::Fill(3), false),
    ];
    frame.render_widget(table(theme, &columns, rows), table_area);
    frame.render_widget(Paragraph::new(notes).wrap(Wrap { trim: false }), notes_area);
}

fn draw_harnesses(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    // The borders and the header row take three rows.
    let rows = app.data.tools.len().max(1) as u16 + 3;
    let [list, rest] = Layout::vertical([
        Constraint::Length(rows.min(area.height / 2)),
        Constraint::Fill(1),
    ])
    .areas(area);
    draw_harness_table(frame, app, list);
    let Ok(harness) = app.harness() else {
        return;
    };
    let detail = Paragraph::new(harness_lines(app, harness))
        .wrap(Wrap { trim: false })
        .block(block(theme, harness.label).padding(Padding::horizontal(1)));
    let Some(preview) = app.install_preview() else {
        frame.render_widget(detail, rest);
        return;
    };
    let diff = Paragraph::new(install_lines(theme, preview))
        .wrap(Wrap { trim: false })
        .block(block(theme, "Settings diff").padding(Padding::horizontal(1)));
    // Wide terminals put the diff beside the details; others put it below them.
    let [details_area, diff_area] = if rest.width >= SIDE_DETAIL_WIDTH {
        Layout::horizontal([Constraint::Fill(1); 2]).areas(rest)
    } else {
        Layout::vertical([Constraint::Fill(1); 2]).areas(rest)
    };
    frame.render_widget(detail, details_area);
    frame.render_widget(diff, diff_area);
}

fn draw_harness_table(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let tools = &app.data.tools;
    let outer = block(theme, &format!("Harnesses [{}]", tools.len()));
    if tools.is_empty() {
        let text = "No harness is listed. Fix the app settings, then press r.";
        frame.render_widget(
            Paragraph::new(Line::styled(text, theme.muted))
                .centered()
                .block(outer),
            area,
        );
        return;
    }
    let wide = area.width >= 80;
    let shown = |column: &(Column, bool)| wide || !column.1;
    let rows: Vec<Row> = tools
        .iter()
        .map(|tool| {
            let state = if tool.available {
                Span::styled("Detected", theme.ok)
            } else {
                Span::styled("Not detected", theme.warn)
            };
            // The first two lines name the model and the endpoint, or hold the read error.
            let settings: Vec<&str> = tool.status.lines().take(2).collect();
            let cells = [
                Cell::from(tool.label),
                Cell::from(state),
                Cell::from(settings.join(" · ")),
                right(count(tool.accounts.len() as u64)),
            ];
            Row::new(
                cells
                    .into_iter()
                    .zip(&HARNESS_COLUMNS)
                    .filter(|(_, column)| shown(column))
                    .map(|(cell, _)| cell),
            )
        })
        .collect();
    let columns: Vec<Column> = HARNESS_COLUMNS
        .iter()
        .filter(|column| shown(column))
        .map(|column| column.0)
        .collect();
    let mut state = TableState::default().with_selected(Some(app.harness));
    frame.render_stateful_widget(
        table(theme, &columns, rows)
            .row_highlight_style(theme.selected)
            .highlight_symbol("› ")
            .block(outer),
        area,
        &mut state,
    );
}

/// This function shows the route to install and whether the harness can use it, then where
/// the settings go and what they hold now, then the saved accounts.
fn harness_lines(app: &App, harness: &ToolInfo) -> Vec<Line<'static>> {
    const WIDTH: usize = 12;
    let theme = app.theme;
    let field = |name: &str, value: String| labeled(theme, name, WIDTH, value);
    let heading = |text: &'static str| Line::styled(text, theme.muted.bold());
    // This destructuring has no `..`, so adding a harness field breaks the build until this
    // page shows it.
    let ToolInfo {
        tool,
        label: _, // draw_harnesses titles these details with it
        available,
        files,
        status,
        accounts: _, // the a key cycles through them, and account_details has the same names
        account_details,
        discovery_error,
    } = harness;
    let mut lines = Vec::new();
    match app.selected_route() {
        Ok(route) => {
            lines.push(field("Route", route.label.clone()));
            match route.compatibility.get(tool) {
                Some(Some(reason)) => lines.push(Line::styled(reason.clone(), theme.warn)),
                Some(None) => lines.push(Line::styled("This harness can use the route.", theme.ok)),
                None => {}
            }
        }
        Err(error) => lines.push(Line::from(vec![
            Span::styled(format!("{:<WIDTH$}", "Route"), theme.muted),
            Span::styled(error, theme.warn),
        ])),
    }
    if !available {
        lines.push(Line::styled(
            "Not detected. You can prepare settings and install the harness later.",
            theme.warn,
        ));
    }
    // The current settings summary of the destination, or why there is none, and its style.
    let (current, style) = match &app.destination {
        Destination::Harness => {
            let files: Vec<String> = files.iter().map(|f| f.display().to_string()).collect();
            lines.push(field("Install to", "The harness's own settings".into()));
            lines.push(field("Files", files.join(", ")));
            (status.clone(), Style::new())
        }
        Destination::Account(name) => {
            match account_details.iter().find(|account| account.name == *name) {
                Some(account) => {
                    lines.push(field("Install to", format!("Account {name}")));
                    lines.push(field("Folder", account.path.display().to_string()));
                    (account.status.clone(), Style::new())
                }
                None => (
                    format!(
                        "Account {name} is no longer listed. Press a to choose another \
                         destination."
                    ),
                    theme.warn,
                ),
            }
        }
        Destination::File(path) => {
            lines.push(field("Install to", "Custom file".into()));
            lines.push(field("File", path.clone()));
            ("Press p to read it.".into(), theme.muted)
        }
    };
    lines.extend([Line::default(), heading("CURRENT SETTINGS")]);
    // A summary spans several lines, and Paragraph does not start a new line at '\n'.
    lines.extend(
        current
            .lines()
            .map(|line| Line::styled(line.to_string(), style)),
    );
    lines.extend([
        Line::default(),
        heading("SAVED ACCOUNTS · a folder does not confirm a completed login"),
    ]);
    if let Some(error) = discovery_error {
        lines.push(Line::styled(error.clone(), theme.error));
    }
    for account in account_details {
        // This destructuring has no `..`, so adding an account field breaks the build until
        // this page shows it.
        let AccountInfo {
            name,
            path,
            login_status,
            status: _, // CURRENT SETTINGS shows it for the selected account
        } = account;
        let selected = matches!(&app.destination, Destination::Account(chosen) if chosen == name);
        lines.push(Line::from(vec![
            Span::styled(if selected { "› " } else { "  " }, theme.key),
            Span::raw(name.clone()),
            Span::styled(
                format!(" · login status {login_status} · {}", path.display()),
                theme.muted,
            ),
        ]));
    }
    if account_details.is_empty() {
        lines.push(Line::styled(
            "None. Press n to sign in to another account.",
            theme.muted,
        ));
    }
    lines.extend([
        Line::default(),
        Line::styled(
            "Login opens in a new terminal window, and account folders open in $VISUAL or \
             $EDITOR. Native account limits still apply. Switchyard does not rotate login \
             tokens.",
            theme.muted,
        ),
    ]);
    lines
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
    let items: Vec<ListItem> = rows(Tab::Settings)
        .map(|(_, letter, name, ..)| {
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
    let (_, _, name, about, _) = rows(Tab::Settings).nth(app.action).unwrap_or(&ACTIONS[0]);
    let mut lines = vec![Line::from(*about), Line::default()];
    if !app.data.errors.is_empty() {
        lines.extend(
            app.data
                .errors
                .iter()
                .map(|error| Line::styled(error.clone(), theme.error)),
        );
        lines.push(Line::default());
    }
    // This destructuring has no `..`, so adding an app setting breaks the build until the
    // Settings page shows it.
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
            .block(block(theme, name).padding(Padding::horizontal(1))),
        info_area,
    );
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let left = if app.filtering {
        let (filter, matches) = match app.tab {
            Tab::Routes => (&app.route_filter, app.route_visible.len()),
            _ => (&app.filter, app.visible.len()),
        };
        Line::from(vec![
            Span::styled(" / ", theme.key),
            Span::raw(filter.clone()),
            Span::styled("▏", theme.key),
            Span::styled(
                format!("  {} ", plural(matches as u64, "match", "matches")),
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
        let (mark, style) = match toast.tone {
            Tone::Done => (" ✓ ", theme.ok),
            Tone::Warning => (" ! ", theme.warn),
            Tone::Error => (" ✗ ", theme.error),
        };
        Line::from(vec![
            Span::styled(mark, style),
            Span::styled(toast.text.clone(), style),
        ])
    } else {
        Line::default()
    };
    let room = usize::from(area.width).saturating_sub(left.width() + 2);
    // Page keys do nothing while a window is open or the filter takes the keyboard. The
    // Settings page lists its action letters beside its rows, so only other pages put them here.
    let global = app.popup.is_none() && !app.filtering;
    let actions = rows(app.tab).filter(|_| global && app.tab != Tab::Settings);
    let hints = fit_keys(theme, actions, app.keys(), global, room);
    let [left_area, right_area] =
        Layout::horizontal([Constraint::Length(left.width() as u16), Constraint::Fill(1)])
            .areas(area);
    frame.render_widget(Paragraph::new(left), left_area);
    frame.render_widget(Paragraph::new(hints.right_aligned()), right_area);
}

/// This function lists the page's action keys, its other keys, then the global keys, and stops
/// before a hint would be cut. It keeps room for the help key, so the status bar always says how
/// to find every key.
fn fit_keys(
    theme: Theme,
    actions: impl Iterator<Item = &'static ActionRow>,
    keys: &[Key],
    global: bool,
    room: usize,
) -> Line<'static> {
    let hint = |key: Cow<'static, str>, label: &str| {
        [
            Span::styled(key, theme.key),
            Span::styled(format!(" {label}  "), theme.muted),
        ]
    };
    const HELP_WIDTH: usize = "? help  ".len();
    let mut spans = Vec::new();
    let mut used = if global { HELP_WIDTH } else { 0 };
    let globals = if global { GLOBAL_KEYS } else { &[] };
    let hints = actions
        .map(|&(_, key, label, ..)| (Cow::Owned(key.to_string()), label))
        .chain(
            keys.iter()
                .chain(globals)
                .map(|&(key, label, _)| (Cow::Borrowed(key), label)),
        );
    for (key, label) in hints {
        if key == "?" {
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
        spans.extend(hint(Cow::Borrowed("?"), "help"));
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
            let masked = matches!(form.kind, FormKind::SaveKey | FormKind::ModelKey);
            let mut value = if masked && index == 1 {
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
    // A window that asks for a decision keeps its keys on the border and its text fields below
    // the scrolling text, so both stay visible when the text is long.
    let (title, lines, pinned, border, preferred, footer) = match popup {
        Popup::Help { .. } => {
            let width = 76;
            let text_width = width.min(area.width.saturating_sub(2)).saturating_sub(4);
            let lines = help_lines(theme, usize::from(text_width));
            ("Help", lines, Vec::new(), theme.key, width, close)
        }
        Popup::Detail { .. } => (
            "Call details",
            app.selected_entry()
                .map(|entry| detail_lines(theme, entry))
                .unwrap_or_default(),
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
        Popup::Form(form) => (
            &*form.title,
            form_notes(app, form),
            form_lines(theme, form),
            theme.key,
            72,
            footer(&[("⏎", "Submit"), ("esc", "Cancel")]),
        ),
        Popup::Review {
            lines,
            verb,
            confirm,
            ..
        } => {
            let mut pinned = vec![Line::default()];
            pinned.extend(form_lines(theme, confirm));
            (
                &*confirm.title,
                lines.clone(),
                pinned,
                theme.warn,
                84,
                footer(&[("⏎", *verb), ("esc", "Cancel")]),
            )
        }
        Popup::Message {
            title, text, tone, ..
        } => (
            *title,
            text.lines()
                .map(|line| Line::from(line.to_string()))
                .collect(),
            Vec::new(),
            match tone {
                Tone::Done => theme.key,
                Tone::Warning => theme.warn,
                Tone::Error => theme.error,
            },
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

/// This function returns the text above a form's fields: what the fields accept and the lists
/// that ↑ and ↓ cycle through.
fn form_notes(app: &App, form: &Form) -> Vec<Line<'static>> {
    let theme = app.theme;
    let data = &app.data;
    let algorithms = |selected: &str| {
        let names: Vec<String> = data
            .algorithms
            .iter()
            .map(|known| format!("{} ({})", known.title, known.kind))
            .collect();
        let mut lines = vec![Line::styled(
            format!("Algorithms: {}", names.join(" · ")),
            theme.muted,
        )];
        if let Some(known) = data.algorithms.iter().find(|a| a.kind == selected.trim()) {
            lines.push(Line::from(known.summary));
        }
        lines.push(Line::default());
        lines
    };
    let identity = |config_name: bool| {
        let mut lines = vec![Line::from(
            "Harnesses send the public model ID as the model. Each route needs its own ID.",
        )];
        if config_name {
            lines.push(Line::from(
                "The config name is the route's table name in the server config, as in \
                 [routes.team-router]. It uses 1–64 letters, digits, hyphens, or underscores.",
            ));
        }
        lines.push(Line::default());
        lines
    };
    match &form.kind {
        FormKind::SaveKey | FormKind::ImportKey | FormKind::RemoveKey => {
            let mut lines = vec![Line::styled("Server keys", theme.muted)];
            lines.extend(key_lines(theme, app.server_keys.as_ref()));
            lines.push(Line::default());
            lines
        }
        FormKind::Algorithm { .. } => algorithms(&form.fields[0].1),
        FormKind::NewRoute { .. } => {
            let mut lines = identity(true);
            lines.extend(algorithms(&form.fields[2].1));
            lines
        }
        FormKind::RouteIdentity { operation, .. } => identity(*operation != "rename"),
        FormKind::RouteModels { .. } | FormKind::NewRouteModels { .. } => {
            let names: Vec<&str> = data.clients.iter().map(|c| c.name.as_str()).collect();
            vec![
                Line::styled(format!("Endpoints: {}", names.join(", ")), theme.muted),
                Line::from("Use the endpoint name and a model ID. Unlisted model IDs are allowed."),
                Line::default(),
            ]
        }
        FormKind::Models | FormKind::ModelKey => {
            let key = matches!(form.kind, FormKind::ModelKey);
            let mut lines: Vec<Line> = data
                .clients
                .iter()
                .map(|client| {
                    let mut spans = vec![
                        Span::raw(client.name.clone()),
                        Span::styled(format!(" · {}", client.host), theme.muted),
                    ];
                    if key && client.accepts_key {
                        spans.push(Span::styled(" · accepts a model-list key", theme.ok));
                    } else if key {
                        spans.push(Span::styled(" · no model-list key", theme.warn));
                    }
                    Line::from(spans)
                })
                .collect();
            lines.push(Line::default());
            lines
        }
        FormKind::SettingsFile => vec![
            Line::from(
                "The harness must load a custom file itself. Pi reads settings.json from the \
                 folder of models.json.",
            ),
            Line::default(),
        ],
        FormKind::Account(_) => vec![
            Line::from(
                "Use 1–64 ASCII letters, digits, hyphens, or underscores. Login opens in a new \
                 terminal window; finish it there. API routes use credentials configured on \
                 the server. The Codex app manages its own account.",
            ),
            Line::default(),
        ],
        FormKind::Preferences
        | FormKind::Update(_)
        | FormKind::Confirm(..)
        | FormKind::Remove { .. } => Vec::new(),
    }
}

/// This function shows the file and notes of a route preview before the user types SAVE.
fn change_lines(theme: Theme, preview: &Value) -> Vec<Line<'static>> {
    let file = preview["file"].as_str().unwrap_or_default().to_string();
    let mut lines = vec![labeled(theme, "Changed file", 13, file)];
    let notes = preview["notes"].as_array().map(Vec::as_slice);
    for note in notes.unwrap_or_default().iter().filter_map(Value::as_str) {
        lines.push(Line::default());
        lines.push(Line::from(note.to_string()));
    }
    lines.push(Line::default());
    lines.push(Line::from(
        "Switchyard validates the config, saves a backup, and restarts the server.",
    ));
    lines
}

/// This function explains what deleting a route keeps and who must change.
fn remove_lines(
    theme: Theme,
    route: &RouteInfo,
    settings: Option<&AppSettings>,
) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(format!(
        "This removes the public model ID from the server. Targets and harness settings are \
         kept. Any harness using {} must choose another route.",
        route.id
    ))];
    if !route.dependents.is_empty() {
        lines.push(Line::default());
        lines.push(Line::styled(
            format!(
                "These harness settings use {} and must choose another route:",
                route.id
            ),
            theme.warn,
        ));
        for dependent in &route.dependents {
            // This destructuring has no `..`, so adding a field breaks the build until this
            // review shows it.
            let Dependent {
                tool: _, // the label names the harness
                label,
                account,
                files,
            } = dependent;
            let account = account.as_ref().map(|name| format!(" ({name})"));
            let files: Vec<String> = files.iter().map(|f| f.display().to_string()).collect();
            lines.push(Line::from(format!(
                "{label}{}: {}",
                account.unwrap_or_default(),
                files.join(", ")
            )));
        }
    }
    if let Some(generated) = &route.generated {
        lines.push(Line::default());
        lines.push(Line::styled(
            format!("Generated by {generated}. That tool can add the route back."),
            theme.warn,
        ));
    }
    if let Some(settings) = settings {
        lines.push(Line::default());
        lines.push(labeled(
            theme,
            "Changed file",
            13,
            settings.config_file.display().to_string(),
        ));
    }
    lines.push(Line::default());
    lines.push(Line::from(
        "Switchyard validates the config, saves a backup, and restarts the server.",
    ));
    lines
}

/// This function returns a text field of a controller reply, or nothing when it is missing.
fn text(value: &Value, key: &str) -> String {
    value[key].as_str().unwrap_or_default().to_string()
}

/// This function lists each setting that a harness preview changes: the file and setting, the
/// current value, and the value afterward.
fn diff_lines(theme: Theme, preview: &Value, after: &str) -> Vec<Line<'static>> {
    let mut lines = vec![Line::styled(
        format!("SETTINGS DIFF · − current / + {after}"),
        theme.muted.bold(),
    )];
    for change in preview["changes"].as_array().into_iter().flatten() {
        lines.extend([
            Line::default(),
            Line::styled(
                format!("{} · {}", text(change, "file"), text(change, "key")),
                theme.muted,
            ),
            Line::styled(format!("− {}", text(change, "before")), theme.error),
            Line::styled(format!("+ {}", text(change, "after")), theme.ok),
        ]);
    }
    lines
}

/// This function shows an install preview: the settings afterward, each changed setting, what
/// happens to each file, and how the harness signs in.
fn install_lines(theme: Theme, preview: &Value) -> Vec<Line<'static>> {
    let mut lines = vec![Line::styled("AFTER APPLYING ROUTING", theme.muted.bold())];
    let proposed = text(preview, "proposed");
    lines.extend(proposed.lines().map(|line| Line::from(line.to_string())));
    lines.push(Line::default());
    lines.extend(diff_lines(theme, preview, "proposed"));
    lines.push(Line::default());
    for file in preview["file_actions"].as_array().into_iter().flatten() {
        let (action, file) = (text(file, "action"), text(file, "file"));
        lines.push(Line::from(format!("Will {action}: {file}")));
    }
    lines.push(Line::default());
    lines.push(Line::from(text(preview, "authentication")));
    lines
}

/// This function shows a restore preview: what restoring does to each file, the backup it
/// uses, and the settings that change.
fn restore_lines(theme: Theme, preview: &Value) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(text(preview, "warning"))];
    for file in preview["files"].as_array().into_iter().flatten() {
        lines.extend([
            Line::default(),
            Line::from(format!("{}: {}", text(file, "action"), text(file, "file"))),
            Line::styled(
                format!("Original backup: {}", text(file, "backup")),
                theme.muted,
            ),
        ]);
    }
    // The controller leaves the diff out when a file cannot be compared.
    if preview["changes"].is_array() {
        lines.push(Line::default());
        lines.extend(diff_lines(theme, preview, "restored"));
    }
    lines
}

fn archive_lines(theme: Theme, preview: &Value) -> Vec<Line<'static>> {
    vec![
        Line::from(text(preview, "notice")),
        Line::default(),
        labeled(theme, "Account", 11, text(preview, "name")),
        labeled(theme, "Directory", 11, text(preview, "path")),
    ]
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
/// Each page lists its other keys first, then its action keys from `ACTIONS`.
fn help_lines(theme: Theme, width: usize) -> Vec<Line<'static>> {
    const KEY_WIDTH: usize = 8;
    let room = width.saturating_sub(KEY_WIDTH).max(20);
    let mut lines = Vec::new();
    for (section, keys, page) in [
        ("Every page", GLOBAL_KEYS, None),
        ("Calls", CALLS_KEYS, Some(Tab::Calls)),
        ("While filtering calls or routes", FILTER_KEYS, None),
        ("Models", MODELS_KEYS, Some(Tab::Models)),
        ("Routes", ROUTES_KEYS, Some(Tab::Routes)),
        ("Harnesses", HARNESSES_KEYS, Some(Tab::Harnesses)),
        ("Settings", SETTINGS_KEYS, Some(Tab::Settings)),
    ] {
        if !lines.is_empty() {
            lines.push(Line::default());
        }
        lines.push(Line::styled(format!(" {section} "), theme.selected.bold()));
        let actions = page.into_iter().flat_map(rows);
        let entries = keys
            .iter()
            .map(|&(key, _, about)| (Cow::Borrowed(key), about))
            .chain(actions.map(|&(_, key, _, about, _)| (Cow::Owned(key.to_string()), about)));
        for (key, about) in entries {
            let mut wrapped = Vec::new();
            let mut row = String::new();
            for word in about.split_whitespace() {
                if !row.is_empty() && row.chars().count() + 1 + word.chars().count() > room {
                    wrapped.push(std::mem::take(&mut row));
                }
                if !row.is_empty() {
                    row.push(' ');
                }
                row.push_str(word);
            }
            wrapped.push(row);
            for (index, row) in wrapped.into_iter().enumerate() {
                let key = if index == 0 { &*key } else { "" };
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

/// This function runs the editor on an absolute path. The caller restores normal terminal input
/// first. The path stays one literal argument, even if its name starts with '-'.
fn edit_file(path: &Path) -> Result<(), String> {
    let editor = editor();
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
    let (snapshot, tab) = match controller.snapshot() {
        Ok(snapshot) => (snapshot, Tab::Overview),
        // Settings shows the error and the actions that can fix the app settings.
        Err(error) => (
            Snapshot {
                errors: vec![error],
                ..Snapshot::default()
            },
            Tab::Settings,
        ),
    };
    let mut app = App::new(Theme::new(color_enabled()), snapshot);
    app.tab = tab;
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
        loop {
            while let Ok(finished) = results.try_recv() {
                match finished {
                    Done::Finished { reply, snapshot } => {
                        if let Some(request) = app.finish(reply, snapshot)
                            && !app.quit
                        {
                            app.start(&jobs, request)?;
                        }
                    }
                    // A quit request made while the editor target loaded skips the editor.
                    Done::Editor(_) if app.quit => app.busy = None,
                    Done::Editor(path) => {
                        app.busy = None;
                        match edit_in_foreground(terminal, &path).map_err(|e| e.to_string())? {
                            Ok(()) => app.notify(
                                format!("Closed the editor for {}.", path.display()),
                                Tone::Done,
                            ),
                            Err(error) => app.notify(error, Tone::Error),
                        }
                        app.start(&jobs, Request::refresh())?;
                    }
                }
            }
            if app.quit && app.busy.is_none() {
                return Ok(());
            }
            if app
                .toast
                .as_ref()
                .filter(|toast| toast.tone == Tone::Done)
                .is_some_and(|toast| toast.shown.elapsed() > TOAST_TIME)
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
                app.start(&jobs, Request::refresh())?;
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
    use super::*;
    use crate::config::Config;
    use crate::controller::{AlgorithmInfo, RoleInfo, Source};
    use ratatui::{Terminal, backend::TestBackend};
    use serde_json::json;
    use strum::IntoEnumIterator;

    fn press(app: &mut App, code: KeyCode) -> Option<Request> {
        app.key(KeyEvent::from(code))
    }

    fn type_text(app: &mut App, text: &str) {
        for c in text.chars() {
            press(app, KeyCode::Char(c));
        }
    }

    /// This function runs one request the way the worker thread does.
    fn work(app: &mut App, controller: &mut Controller, request: Option<Request>) {
        let (jobs, inbox) = mpsc::channel();
        app.start(&jobs, request.expect("a request"))
            .expect("start");
        let Ok(Job::Run(action)) = inbox.try_recv() else {
            panic!("the request runs no action");
        };
        let reply = controller.dispatch(action);
        let next = app.finish(Some(reply), controller.snapshot().map(Box::new));
        assert!(next.is_none());
    }

    fn screen(app: &mut App, tab: Tab) -> String {
        screen_sized(app, tab, 140, 40)
    }

    fn screen_sized(app: &mut App, tab: Tab, width: u16, height: u16) -> String {
        app.tab = tab;
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        let buffer = terminal.backend().buffer();
        buffer
            .content()
            .chunks(usize::from(buffer.area.width))
            .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    // A read failure must not appear as an empty log or zero recorded usage.
    fn unavailable_data_does_not_render_empty_usage() {
        let mut app = App::new(
            Theme::new(false),
            Snapshot {
                errors: vec!["Could not read usage".into()],
                ..Snapshot::default()
            },
        );
        let overview = screen(&mut app, Tab::Overview);
        assert!(overview.contains("Could not read usage"));
        assert!(!overview.contains("calls ·"));
        assert!(!overview.contains("Spent"));
        let calls = screen(&mut app, Tab::Calls);
        assert!(calls.contains("Could not read usage"));
        assert!(!calls.contains("No calls recorded"));
    }

    #[test]
    // A call without a recorded cost must never add up to, or display as, $0.
    fn unknown_costs_do_not_render_as_zero() {
        let entry: Entry = serde_json::from_value(json!({"ts": "2026-10-21T12:00:00Z",
            "model": "gemini-4-argon", "route_id": "answer", "prompt_tokens": 10,
            "completion_tokens": 2}))
        .unwrap();
        let usage = ModelTokens::from(&entry);
        let totals = Totals {
            routed: [("gemini-4-argon".to_string(), usage.clone())].into(),
            classifier: BTreeMap::new(),
            routes: [("answer".to_string(), usage)].into(),
        };
        let estimate = Estimate {
            actual: None,
            known: 0.0,
            unknown_calls: 1,
            baseline: Some(0.00004),
            uncounted_answers: 0,
        };
        let available = Source {
            state: Availability::Available,
            ..Source::default()
        };
        let mut snapshot = Snapshot {
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
        snapshot.data_state.usage = available.clone();
        snapshot.data_state.history = available;
        let mut app = App::new(Theme::new(false), snapshot);
        for tab in [Tab::Overview, Tab::Calls, Tab::Models] {
            let text = screen(&mut app, tab);
            assert!(!text.contains("$0"), "{tab:?} shows a zero cost:\n{text}");
        }
        let overview = screen(&mut app, Tab::Overview);
        assert!(overview.contains("Spent unknown"));
        assert!(overview.contains("no recorded cost"));
    }

    #[test]
    fn filter_requires_every_word_and_matches_case_only_for_capitals() {
        let entry: Entry = serde_json::from_value(json!({"ts": "", "model": "claude-opus-4-8",
            "route_id": "answer", "session_id": "S1"}))
        .unwrap();
        assert!(matches_filter(&entry, "opus answer s1"));
        assert!(matches_filter(&entry, "S1"));
        assert!(!matches_filter(&entry, "Opus"));
        assert!(!matches_filter(&entry, "opus haiku"));
    }

    #[test]
    // Every controller action needs a key, or the desktop app can do something that the
    // terminal app cannot.
    fn every_controller_action_has_a_key() {
        let mut pairs: Vec<(usize, char)> = ACTIONS
            .iter()
            .map(|&(tab, key, ..)| (tab.index(), key))
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        assert_eq!(pairs.len(), ACTIONS.len(), "A page uses one key twice");

        // App::key handles the global keys and the page digits before action letters, and
        // j, g, and G must keep moving the selection. Each page's own keys come after its
        // action letters, so a row with one of them would hide it.
        let global = ['q', 'r', '?', 'j', 'g', 'G'];
        let digits: Vec<char> = (1..=TABS.len() as u32)
            .filter_map(|digit| char::from_digit(digit, 10))
            .collect();
        let page_keys = |tab| -> &[char] {
            match tab {
                Tab::Calls => &['/', 's'],
                Tab::Models => &['p', '[', ']'],
                Tab::Routes => &['/'],
                Tab::Harnesses => &['t', 'a', 'x'],
                Tab::Overview | Tab::Settings => &[],
            }
        };
        let reserved: Vec<(Tab, char)> = ACTIONS
            .iter()
            .filter(|&&(tab, key, ..)| {
                global.contains(&key) || digits.contains(&key) || page_keys(tab).contains(&key)
            })
            .map(|&(tab, key, ..)| (tab, key))
            .collect();
        assert!(
            reserved.is_empty(),
            "These rows use a reserved key: {reserved:?}"
        );

        let missing: Vec<ActionKind> = ActionKind::iter()
            .filter(|kind| !ACTIONS.iter().any(|&(.., flow)| flow.contains(kind)))
            .collect();
        assert!(missing.is_empty(), "No key runs {missing:?}");
    }

    #[test]
    // A review window submits only after its exact word, and it sends the reviewed token or
    // revision unchanged.
    fn reviews_submit_only_after_the_confirmation_word() {
        let data = Snapshot::default();
        let submit = |kind, typed: &str| {
            let form = Form::new(kind, "", vec![("", typed.to_string())]);
            match form.action(&data) {
                Ok(Submit::Run(action)) => Some(action),
                _ => None,
            }
        };
        assert!(matches!(
            submit(FormKind::Update(Some("t".into())), "UPDATE"),
            Some(Action::Update { preview_token: Some(token) }) if token == "t"
        ));
        assert!(submit(FormKind::Update(None), "update").is_none());
        let save = || FormKind::Confirm("SAVE".into(), Box::new(Action::CheckConfig {}));
        assert!(matches!(
            submit(save(), "SAVE"),
            Some(Action::CheckConfig {})
        ));
        assert!(submit(save(), "save").is_none());
        let remove = || FormKind::Remove {
            generation: 3,
            route: "r".into(),
            id: "public".into(),
            revision: "reviewed".into(),
        };
        assert!(matches!(
            submit(remove(), "public"),
            Some(Action::Remove { generation: 3, route, id, revision: Some(revision) })
                if route == "r" && id == "public" && revision == "reviewed"
        ));
        assert!(submit(remove(), "Public").is_none());
    }

    #[test]
    // Filtering keeps a selected route that still matches and otherwise selects the first
    // match, so a route action never runs on a route that the filter hides.
    fn route_filter_keeps_a_matching_selection() {
        let route = |key: &str, label: &str| RouteInfo {
            key: key.into(),
            id: key.into(),
            label: label.into(),
            ..RouteInfo::default()
        };
        let snapshot = Snapshot {
            routes: vec![
                route("a", "alpha"),
                route("b", "beta"),
                route("c", "alpine"),
            ],
            ..Snapshot::default()
        };
        let mut app = App::new(Theme::new(false), snapshot);
        app.tab = Tab::Routes;
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Char('/'));
        let selected = |app: &App| app.selected_route().map(|route| route.key.clone());
        for (filter, expected) in [("beta", "b"), ("", "b"), ("al", "a")] {
            app.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
            type_text(&mut app, filter);
            assert_eq!(
                selected(&app),
                Ok(expected.to_string()),
                "filter {filter:?}"
            );
        }
        type_text(&mut app, " missing");
        press(&mut app, KeyCode::Enter);
        assert!(selected(&app).is_err());
        assert!(press(&mut app, KeyCode::Char('d')).is_none());
        assert!(
            !matches!(app.popup, Some(Popup::Review { .. })),
            "a hidden route opened the delete review"
        );
    }

    #[test]
    // The details must name a model for every role, including random-split models beyond the
    // two that a new random route starts with.
    fn route_details_list_every_model() {
        let role = |label: &str| RoleInfo {
            label: label.into(),
            ..RoleInfo::default()
        };
        let choice = |model: &str| Choice {
            client: "c".into(),
            model: model.into(),
        };
        let snapshot = Snapshot {
            routes: vec![RouteInfo {
                key: "split".into(),
                id: "team-split".into(),
                kind: "random".into(),
                editable: true,
                choices: vec![choice("one"), choice("two"), choice("three")],
                generated: Some("switchyard-setup".into()),
                ..RouteInfo::default()
            }],
            algorithms: vec![AlgorithmInfo {
                kind: "random",
                title: "Random split",
                roles: vec![role("Model 1"), role("Model 2")],
                ..AlgorithmInfo::default()
            }],
            ..Snapshot::default()
        };
        let mut app = App::new(Theme::new(false), snapshot);
        let text = screen(&mut app, Tab::Routes);
        for line in [
            "Model 1    c / one",
            "Model 2    c / two",
            "Model 3    c / three",
        ] {
            assert!(text.contains(line), "missing {line:?}:\n{text}");
        }
        assert!(text.contains("Generated by switchyard-setup"));
    }

    #[test]
    // A model-list key must never appear on screen.
    fn model_list_key_stays_masked() {
        let mut app = App::new(Theme::new(false), Snapshot::default());
        app.tab = Tab::Routes;
        press(&mut app, KeyCode::Char('k'));
        press(&mut app, KeyCode::Tab);
        type_text(&mut app, "sk-secret");
        let text = screen(&mut app, Tab::Routes);
        assert!(text.contains("API key: •••••••••"), "{text}");
        assert!(!text.contains("sk-secret"));
    }

    #[test]
    // A route that saved while the server failed to restart must read as a warning, not as a
    // success, in the details window and in the status bar.
    fn warning_replies_do_not_read_as_success() {
        let mut app = App::new(Theme::new(false), Snapshot::default());
        let reply = |details: Value| {
            Some(Ok(Reply {
                message: "Saved the route, but the restart failed.".into(),
                data: json!({"warning": true, "details": details}),
            }))
        };
        app.finish(
            reply(json!(["Backup: server.toml.bak"])),
            Ok(Box::default()),
        );
        let text = screen(&mut app, Tab::Routes);
        assert!(text.contains(" Attention "), "{text}");
        app.popup = None;
        app.finish(reply(json!([])), Ok(Box::default()));
        let text = screen(&mut app, Tab::Routes);
        assert!(text.contains(" ! Saved the route"), "{text}");
    }

    #[test]
    // The narrowest supported terminal must still show every tab, so no page's digit or name is
    // cut off, whichever page is selected.
    fn narrow_header_shows_every_tab() {
        let mut app = App::new(Theme::new(false), Snapshot::default());
        for selected in TABS {
            let text = screen_sized(&mut app, selected, MIN_WIDTH, MIN_HEIGHT);
            let header = text.lines().next().unwrap_or_default();
            for tab in TABS {
                let title = format!("{} {}", tab.index() + 1, &tab.name()[..3]);
                assert!(
                    header.contains(&title),
                    "{selected:?}: {title:?} is cut off in {header:?}"
                );
            }
        }
    }

    #[test]
    // The models form must keep the revision from when the editor opened, so a server config
    // that changed since then fails the review and keeps the draft instead of saving old
    // choices into the new file.
    fn externally_changed_routes_cannot_bless_snapshot_choices() {
        let dir = tempfile::tempdir().expect("fixture");
        let settings = dir.path().join("desktop.toml");
        let routes = dir.path().join("server.toml");
        let original = "[llm_clients.c]\n\
                        base_url='http://localhost:1'\n\
                        format='openai_responses'\n\
                        [targets.m]\n\
                        id='original'\n\
                        llm_client='c'\n\
                        [routes.r]\n\
                        id='public'\n\
                        type='passthrough'\n\
                        target='m'\n";
        std::fs::write(&routes, original).expect("routes");
        std::fs::write(
            &settings,
            format!(
                "config_file={routes:?}\nrouting_log={:?}\nserver_url='http://127.0.0.1:0'",
                dir.path().join("log")
            ),
        )
        .expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("config"), settings);
        let mut app = App::new(Theme::new(false), controller.snapshot().expect("snapshot"));
        app.tab = Tab::Routes;
        assert!(press(&mut app, KeyCode::Char('e')).is_none());
        let external = original.replace("id='original'", "id='external'");
        std::fs::write(&routes, &external).expect("external edit");
        let editor = press(&mut app, KeyCode::Enter);
        work(&mut app, &mut controller, editor);
        let preview = press(&mut app, KeyCode::Enter);
        work(&mut app, &mut controller, preview);
        let Some(Popup::Form(draft)) = &app.popup else {
            panic!("the draft closed");
        };
        assert!(matches!(draft.kind, FormKind::RouteModels { .. }));
        assert!(draft.error.contains("Review"), "{}", draft.error);
        assert_eq!(
            std::fs::read_to_string(&routes).expect("preserved"),
            external
        );
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 2);
    }

    #[test]
    // The p and i keys run the same preview, so only the recorded key decides whether its reply
    // opens the install review. The review saves nothing until INSTALL is typed.
    fn only_i_opens_the_install_review() {
        let dir = tempfile::tempdir().expect("fixture");
        let settings = dir.path().join("desktop.toml");
        let routes = dir.path().join("server.toml");
        std::fs::write(
            &routes,
            "[llm_clients.c]\n\
             base_url='https://example.com/v1'\n\
             format='openai_chat'\n\
             [targets.m]\n\
             id='actual'\n\
             llm_client='c'\n\
             [routes.r]\n\
             id='public'\n\
             type='passthrough'\n\
             target='m'\n",
        )
        .expect("routes");
        std::fs::write(
            &settings,
            format!(
                "config_file={routes:?}\nrouting_log={:?}\nserver_url='http://127.0.0.1:0'",
                dir.path().join("log")
            ),
        )
        .expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("config"), settings);
        let mut app = App::new(Theme::new(false), controller.snapshot().expect("snapshot"));
        app.tab = Tab::Harnesses;
        // A custom file keeps the install inside the fixture instead of the user's settings.
        let claude = app
            .data
            .tools
            .iter()
            .position(|t| t.tool == Harness::Claude);
        app.harness = claude.expect("Claude Code");
        let custom = dir.path().join("claude.json");
        std::fs::write(&custom, "{\"user\":true}").expect("custom");
        press(&mut app, KeyCode::Char('x'));
        type_text(&mut app, custom.to_str().expect("path"));
        press(&mut app, KeyCode::Enter);

        let preview = press(&mut app, KeyCode::Char('p'));
        work(&mut app, &mut controller, preview);
        assert!(app.popup.is_none(), "p opened a window");
        assert!(screen(&mut app, Tab::Harnesses).contains("SETTINGS DIFF"));

        let review = press(&mut app, KeyCode::Char('i'));
        work(&mut app, &mut controller, review);
        assert!(matches!(app.popup, Some(Popup::Review { .. })));
        type_text(&mut app, "install");
        assert!(press(&mut app, KeyCode::Enter).is_none());
        let unchanged = std::fs::read_to_string(&custom).expect("custom");
        assert_eq!(unchanged, "{\"user\":true}");
        app.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        type_text(&mut app, "INSTALL");
        let install = press(&mut app, KeyCode::Enter);
        work(&mut app, &mut controller, install);
        let installed = std::fs::read_to_string(&custom).expect("custom");
        assert!(installed.contains("public"), "{installed}");
    }
}
