// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The routes window. A sidebar lists the routes. For the open route, the
//! window shows its algorithm and a model for each role, and Apply checks,
//! saves, and restarts the server.
//!
//! The window only collects choices. `server_config` edits the config,
//! `models` lists and caches models, and `server` checks, saves, and
//! restarts, so all of that runs without a GUI. Model lists, Keychain access,
//! and Apply run on worker threads, which send their results through a
//! channel that the menu bar's event loop drains by calling [`Picker::poll`].

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{
    NSAccessibility, NSApplication, NSBackingStoreType, NSBox, NSBoxType, NSButton, NSColor,
    NSComboBox, NSComboBoxDelegate, NSControl, NSControlTextEditingDelegate, NSFont, NSPopUpButton,
    NSResponder, NSScreen, NSScrollView, NSSecureTextField, NSTableView, NSTextField,
    NSTextFieldDelegate, NSTextView, NSView, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{
    MainThreadMarker, NSNotification, NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize,
    NSString, ns_string,
};

use crate::config::Config;
use crate::models::{self, ListError, Loaded, ModelList};
use crate::server;
use crate::server_config::{ALGORITHMS, Choice, Client, Role, Route, ServerConfig, Tier};
use crate::sidebar::{self, SidebarSource};

// Control tags. Role rows use `CLIENT_TAG + row` for the endpoint popup and
// `MODEL_TAG + row` for the model box. A pick from the model list arrives as
// `SELECTED_TAG + row`.
const ALGORITHM_TAG: isize = 2;
const SAVE_KEY_TAG: isize = 3;
const APPLY_TAG: isize = 4;
const REFRESH_TAG: isize = 5;
const CLOSE_TAG: isize = 6;
const REVERT_TAG: isize = 7;
const CLIENT_TAG: isize = 100;
const MODEL_TAG: isize = 200;
const SELECTED_TAG: isize = 300;

const SIDEBAR_WIDTH: f64 = sidebar::WIDTH;
const DETAIL_WIDTH: f64 = 750.0;
const WIDTH: f64 = SIDEBAR_WIDTH + DETAIL_WIDTH;
/// The window's content height, or less on a short screen.
const MAX_HEIGHT: f64 = 640.0;
/// The least content height that leaves room for one role row when the
/// generated-block banner and the key area both show: 232 above the rows,
/// 80 for the key area, 84 for the row, and `ROWS_BOTTOM` below.
const MIN_HEIGHT: f64 = 582.0;
const MARGIN: f64 = 24.0;
const CONTENT_WIDTH: f64 = DETAIL_WIDTH - 2.0 * MARGIN;
// The columns of a role row, measured from the left edge of the content.
const ROLE_WIDTH: f64 = 130.0;
const ENDPOINT_X: f64 = ROLE_WIDTH + 12.0;
const ENDPOINT_WIDTH: f64 = 240.0;
const MODEL_X: f64 = ENDPOINT_X + ENDPOINT_WIDTH + 10.0;
const MODEL_WIDTH: f64 = CONTENT_WIDTH - MODEL_X - 6.0;
/// Height of one role row: the role and its two dropdowns, then a note of up
/// to two lines under them.
const ROLE_HEIGHT: f64 = 84.0;
const NOTE_HEIGHT: f64 = 34.0;
/// Height of the key area: three lines of text above the key field.
const KEY_HEIGHT: f64 = 80.0;
/// Height of the algorithm's description: up to two lines.
const SUMMARY_HEIGHT: f64 = 36.0;
/// Height of the result area. A longer result scrolls.
const STATUS_HEIGHT: f64 = 120.0;
const BUTTONS_HEIGHT: f64 = 32.0;
// Distances from the bottom of the window to the edges of the areas that sit
// above the buttons.
const BUTTONS_Y: f64 = 16.0;
const STATUS_Y: f64 = BUTTONS_Y + BUTTONS_HEIGHT + 10.0;
const ROWS_BOTTOM: f64 = STATUS_Y + STATUS_HEIGHT + 8.0;

define_class!(
    /// Receives every control's action and text change, and queues the
    /// control's tag for [`Picker::poll`]. The picker handles the tags later,
    /// outside AppKit's callback, so an event that AppKit sends while the
    /// picker updates its controls only adds a tag and does not reenter it.
    // SAFETY: NSObject has no subclassing requirements, and `ControlTarget`
    // does not implement `Drop`.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "SwitchyardPickerControlTarget"]
    #[ivars = RefCell<Vec<isize>>]
    struct ControlTarget;

    impl ControlTarget {
        // SAFETY: AppKit calls an action method with the control that sent it.
        #[unsafe(method(controlChanged:))]
        fn control_changed(&self, sender: &NSControl) {
            self.ivars().borrow_mut().push(sender.tag());
        }
    }

    unsafe impl NSObjectProtocol for ControlTarget {}

    unsafe impl NSControlTextEditingDelegate for ControlTarget {
        // SAFETY: the signature matches the protocol method.
        #[unsafe(method(controlTextDidChange:))]
        fn control_text_did_change(&self, notification: &NSNotification) {
            let control = notification
                .object()
                .and_then(|object| object.downcast::<NSControl>().ok());
            if let Some(control) = control {
                self.ivars().borrow_mut().push(control.tag());
            }
        }
    }

    unsafe impl NSTextFieldDelegate for ControlTarget {}

    unsafe impl NSComboBoxDelegate for ControlTarget {
        // SAFETY: the signature matches the protocol method.
        #[unsafe(method(comboBoxSelectionDidChange:))]
        fn combo_box_selection_did_change(&self, notification: &NSNotification) {
            let control = notification
                .object()
                .and_then(|object| object.downcast::<NSControl>().ok());
            if let Some(control) = control {
                self.ivars()
                    .borrow_mut()
                    .push(control.tag() - MODEL_TAG + SELECTED_TAG);
            }
        }
    }
);

impl ControlTarget {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(RefCell::new(Vec::new()));
        // SAFETY: NSObject's `init` has this signature.
        unsafe { msg_send![super(this), init] }
    }
}

define_class!(
    /// A model box that scrolls itself into view when it gets keyboard
    /// focus, so Tab can reach a role row that the scroll view hides.
    // SAFETY: NSComboBox has no subclassing requirements, and `ModelBox`
    // does not implement `Drop`.
    #[unsafe(super(NSComboBox, NSTextField, NSControl, NSView, NSResponder, NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "SwitchyardPickerModelBox"]
    struct ModelBox;

    impl ModelBox {
        // SAFETY: the signature matches NSResponder's method.
        #[unsafe(method(becomeFirstResponder))]
        fn become_first_responder(&self) -> bool {
            // SAFETY: NSResponder's `becomeFirstResponder` takes no
            // arguments and returns a BOOL.
            let became: bool = unsafe { msg_send![super(self), becomeFirstResponder] };
            if became {
                self.scrollRectToVisible(self.bounds());
            }
            became
        }
    }
);

impl ModelBox {
    fn new(mtm: MainThreadMarker, frame: NSRect) -> Retained<Self> {
        // SAFETY: NSComboBox's `initWithFrame:` has this signature.
        unsafe { msg_send![Self::alloc(mtm), initWithFrame: frame] }
    }
}

/// A result that a worker thread sends back to the window.
enum Done {
    /// The model list at one URL, from load number `load`.
    Listed {
        load: u64,
        loaded: Loaded,
    },
    /// The outcome of a load that the user asked for, for the result area.
    Status(String),
    Applied(Result<String, String>),
}

/// What the window knows about the model list at one URL.
#[derive(Default)]
struct ListState {
    /// The newest list: fetched, or read from the cache file.
    list: Option<ModelList>,
    /// Why the last load failed, when it did.
    error: Option<ListError>,
    /// `started` is the newest load started for this list, and `shown` is the
    /// newest load whose result the window shows. Loads are numbered in the
    /// order they start.
    started: u64,
    shown: u64,
}

impl ListState {
    /// Returns whether a worker thread is loading this list now.
    fn loading(&self) -> bool {
        self.started > self.shown
    }

    /// Records the result of load number `load`. A worker that started
    /// earlier can finish later, so the window ignores a result from a load
    /// older than the one it shows. Otherwise the error comes from this
    /// result, and the newer of the two lists stays, because a failed load
    /// returns the list from the cache file, which can be older than the one
    /// the window has. The result of the newest load ends the loading state.
    fn finish(&mut self, load: u64, list: Option<ModelList>, error: Option<ListError>) {
        if load < self.shown {
            return;
        }
        self.shown = load;
        let fetched_at = |list: &Option<ModelList>| list.as_ref().map(|list| list.fetched_at);
        if fetched_at(&list) >= fetched_at(&self.list) {
            self.list = list;
        }
        self.error = error;
    }
}

/// A route's algorithm and one choice per role: what the user picked and has
/// not applied yet, or what the config holds.
#[derive(Default)]
struct Draft {
    algorithm: usize,
    choices: Vec<Choice>,
}

/// The controls of one role row.
struct RoleRow {
    client: Retained<NSPopUpButton>,
    model: Retained<ModelBox>,
    note: Retained<NSTextField>,
}

/// The controls that the picker reads or updates after it lays them out.
struct Controls {
    algorithm: Retained<NSPopUpButton>,
    rows: Vec<RoleRow>,
    /// The scroll view that holds the role rows. Its room is the space
    /// between `rows_top`, the rows' top edge, and the result area.
    scroll: Retained<NSScrollView>,
    rows_view: Retained<NSView>,
    rows_top: f64,
    key_label: Retained<NSTextField>,
    key: Retained<NSSecureTextField>,
    save_key: Retained<NSButton>,
    status: Retained<NSTextView>,
    revert: Retained<NSButton>,
    apply: Retained<NSButton>,
}

/// The parts of the window that stay while the detail pane changes.
struct Shell {
    content: Retained<NSView>,
    table: Retained<NSTableView>,
    detail: Retained<NSView>,
    height: f64,
}

/// The routes window and what the user has chosen in it.
pub struct Picker {
    mtm: MainThreadMarker,
    window: Retained<NSWindow>,
    target: Retained<ControlTarget>,
    source: Retained<SidebarSource>,
    settings: Config,
    /// The file that keeps fetched model lists.
    cache: PathBuf,
    config: Option<ServerConfig>,
    routes: Vec<Route>,
    clients: Vec<Client>,
    /// The open route's index in `routes`.
    route: usize,
    algorithm: usize,
    roles: Vec<Role>,
    /// One choice per role, in role order.
    choices: Vec<Choice>,
    /// The algorithm and choices that the config holds for the open route.
    saved: Draft,
    /// What the user picked on the routes they left, by route key.
    drafts: HashMap<String, Draft>,
    /// The latest choice for each tier on this route. When the next algorithm
    /// has no role for a tier, its choice stays here, so switching back
    /// restores it. `select_route` clears the map.
    by_tier: HashMap<Tier, Choice>,
    /// Model lists by URL, kept while the app runs. Clients that list their
    /// models at the same URL share one entry.
    lists: HashMap<String, ListState>,
    /// How many list loads the window has started. It numbers the loads.
    loads: u64,
    /// What the last Apply, or a failed config load, said.
    result: String,
    /// What the last Refresh models or Save key said. It shows under `result`.
    list_notes: String,
    /// Whether an Apply is running.
    busy: bool,
    /// Whether the sidebar shows the open route as edited.
    shown_edited: bool,
    sender: Sender<Done>,
    receiver: Receiver<Done>,
    shell: Shell,
    controls: Option<Controls>,
}

impl Picker {
    pub fn new(mtm: MainThreadMarker, settings: &Config, cache: PathBuf) -> Self {
        // SAFETY: the window is not released when closed, because `Picker`
        // owns it and shows it again from the menu.
        let window = unsafe {
            let window = NSWindow::initWithContentRect_styleMask_backing_defer(
                NSWindow::alloc(mtm),
                rect(0.0, 0.0, WIDTH, MAX_HEIGHT),
                NSWindowStyleMask::Titled
                    | NSWindowStyleMask::Closable
                    | NSWindowStyleMask::Miniaturizable,
                NSBackingStoreType::Buffered,
                false,
            );
            window.setReleasedWhenClosed(false);
            window
        };
        window.setTitle(ns_string!("Switchyard routes"));
        let source = SidebarSource::new(mtm);
        let shell = build_shell(mtm, &window, &source);
        window.center();
        let (sender, receiver) = channel();
        Self {
            mtm,
            window,
            target: ControlTarget::new(mtm),
            source,
            settings: settings.clone(),
            cache,
            config: None,
            routes: Vec::new(),
            clients: Vec::new(),
            route: 0,
            algorithm: 0,
            roles: Vec::new(),
            choices: Vec::new(),
            saved: Draft::default(),
            drafts: HashMap::new(),
            by_tier: HashMap::new(),
            lists: HashMap::new(),
            loads: 0,
            result: String::new(),
            list_notes: String::new(),
            busy: false,
            shown_edited: false,
            sender,
            receiver,
            shell,
            controls: None,
        }
    }

    /// Whether an Apply is running.
    pub fn is_busy(&self) -> bool {
        self.busy
    }

    /// Stores the menu bar settings after the user edited the settings file.
    /// Apply writes the file that `config_file` names. So when `config_file`
    /// changes while the window is open and no Apply is running, the window
    /// loads the new file and drops changes that the user has not applied.
    pub fn set_settings(&mut self, settings: &Config) {
        let moved = settings.config_file != self.settings.config_file;
        self.settings = settings.clone();
        if moved && !self.busy && self.is_open() {
            self.drafts.clear();
            self.load(None);
            self.layout();
        }
    }

    /// Whether the window is on the screen or in the Dock.
    fn is_open(&self) -> bool {
        self.window.isVisible() || self.window.isMiniaturized()
    }

    /// Brings the window to the front. A window that is closed rereads the
    /// server config first and drops what the user picked but did not apply.
    /// A window that is open keeps its picks, so the menu item is a way back
    /// to it.
    pub fn show(&mut self) {
        if !self.is_open() {
            if !self.busy {
                self.drafts.clear();
                // Opening the window again retries a list that failed and has
                // nothing to show. The window fetches a list that it already
                // has only when the user clicks Refresh models or Save key.
                self.lists
                    .retain(|_, state| state.loading() || state.list.is_some());
                let route = self.routes.get(self.route).map(|route| route.key.clone());
                self.load(route.as_deref());
            }
            self.layout();
        }
        // An accessory app must activate itself before its window can take
        // keyboard focus. `activate` needs macOS 14, so use the older call.
        #[allow(deprecated)]
        NSApplication::sharedApplication(self.mtm).activateIgnoringOtherApps(true);
        self.window.makeKeyAndOrderFront(None);
    }

    /// Handles queued control events and finished background work.
    pub fn poll(&mut self) {
        let tags = std::mem::take(&mut *self.target.ivars().borrow_mut());
        let picks = self.source.take_picks();
        let mut changed = !tags.is_empty() || !picks.is_empty();
        for tag in tags {
            self.on_control(tag);
        }
        for row in picks {
            self.on_sidebar(row);
        }
        while let Ok(done) = self.receiver.try_recv() {
            changed = true;
            self.on_done(done);
        }
        if changed {
            if self.is_edited() != self.shown_edited {
                self.sync_sidebar();
            }
            self.update_buttons();
        }
    }

    fn load(&mut self, route: Option<&str>) {
        let path = &self.settings.config_file;
        let loaded = std::fs::read_to_string(path)
            .map_err(|error| {
                format!(
                    "Could not read {}: {error}. Check config_file in the menu bar settings.",
                    path.display()
                )
            })
            .and_then(|text| ServerConfig::parse(&text));
        match loaded {
            Ok(config) => {
                self.routes = config.routes();
                self.clients = config.clients();
                self.route = route
                    .and_then(|key| self.routes.iter().position(|route| route.key == key))
                    .unwrap_or(0);
                self.config = Some(config);
                self.select_route();
                if self.routes.is_empty() {
                    self.result = format!(
                        "{} has no routes to change.",
                        self.settings.config_file.display()
                    );
                }
            }
            Err(error) => {
                self.config = None;
                self.routes.clear();
                self.clients.clear();
                self.select_route();
                self.result = error;
            }
        }
    }

    /// Shows the open route's algorithm and models: what the user picked on
    /// it before, if anything, or else what the config holds.
    fn select_route(&mut self) {
        self.by_tier.clear();
        self.list_notes.clear();
        let (Some(config), Some(route)) = (&self.config, self.routes.get(self.route)) else {
            self.roles.clear();
            self.choices.clear();
            self.saved = Draft::default();
            return;
        };
        match config.algorithm(&route.key) {
            Some(current) => {
                self.algorithm = ALGORITHMS
                    .iter()
                    .position(|algorithm| algorithm.kind == current.kind)
                    .unwrap_or(0);
                self.roles = config.roles(&route.key, current);
                self.choices = config.choices(&route.key);
                self.result.clear();
            }
            None => {
                self.algorithm = 0;
                self.roles = config.roles(&route.key, &ALGORITHMS[0]);
                self.choices = vec![Choice::default(); self.roles.len()];
                self.result = format!(
                    "This window cannot show the current settings of route {}: its type is not \
                     in the Algorithm list, or it is a custom-mode llm_classifier. Apply replaces \
                     those settings with the algorithm and models you pick.",
                    route.id
                );
            }
        }
        self.fill_clients();
        self.saved = Draft {
            algorithm: self.algorithm,
            choices: self.choices.clone(),
        };
        if let Some(config) = &self.config
            && let Some(route) = self.routes.get(self.route)
            && let Some(draft) = self.drafts.get(&route.key)
            && let Some(algorithm) = ALGORITHMS.get(draft.algorithm)
        {
            let roles = config.roles(&route.key, algorithm);
            if roles.len() == draft.choices.len() {
                self.algorithm = draft.algorithm;
                self.roles = roles;
                self.choices = draft.choices.clone();
            }
        }
        self.list_models();
    }

    /// Switches the algorithm. Each new role starts from the latest choice
    /// for its tier on this route, so the capable model stays the capable one.
    /// Choices for tiers the new algorithm lacks stay in `by_tier`.
    fn select_algorithm(&mut self, index: usize) {
        self.read_models();
        for (role, choice) in self.roles.iter().zip(&self.choices) {
            if let Some(tier) = role.tier {
                self.by_tier.insert(tier, choice.clone());
            }
        }
        let (Some(config), Some(route), Some(algorithm)) = (
            &self.config,
            self.routes.get(self.route),
            ALGORITHMS.get(index),
        ) else {
            return;
        };
        let roles = config.roles(&route.key, algorithm);
        self.choices = roles
            .iter()
            .map(|role| {
                role.tier
                    .and_then(|tier| self.by_tier.get(&tier).cloned())
                    .unwrap_or_default()
            })
            .collect();
        self.roles = roles;
        self.algorithm = index;
        self.fill_clients();
        self.list_models();
        self.layout();
    }

    /// Gives every role without a client the first role's client.
    fn fill_clients(&mut self) {
        let fallback = self
            .choices
            .iter()
            .map(|choice| &choice.client)
            .find(|client| !client.is_empty())
            .or_else(|| self.clients.first().map(|client| &client.name))
            .cloned()
            .unwrap_or_default();
        for choice in &mut self.choices {
            if choice.client.is_empty() {
                choice.client = fallback.clone();
            }
        }
    }

    /// Opens the route that the user picked in the sidebar, and keeps what
    /// the user picked on the route they leave.
    fn on_sidebar(&mut self, row: usize) {
        let Some(route) = self.source.route_at(row) else {
            return;
        };
        if route == self.route {
            return;
        }
        if self.busy {
            // The result of a running Apply belongs to the route that
            // started it, so the sidebar stays on that route.
            self.sync_sidebar();
            return;
        }
        self.stash();
        self.route = route;
        self.select_route();
        self.layout();
    }

    /// Returns whether the open route differs from what the config holds.
    fn is_edited(&self) -> bool {
        differs(&self.saved, self.algorithm, &self.choices)
    }

    /// Keeps what the user picked on the open route, or drops the old pick
    /// when the route matches the config again.
    fn stash(&mut self) {
        let Some(route) = self.routes.get(self.route) else {
            return;
        };
        if self.is_edited() {
            self.drafts.insert(
                route.key.clone(),
                Draft {
                    algorithm: self.algorithm,
                    choices: self.choices.clone(),
                },
            );
        } else {
            self.drafts.remove(&route.key);
        }
    }

    /// Drops what the user picked on the open route.
    fn revert(&mut self) {
        if let Some(route) = self.routes.get(self.route) {
            self.drafts.remove(&route.key);
        }
        self.select_route();
        self.layout();
    }

    /// Updates the sidebar's rows and selects the open route's row.
    fn sync_sidebar(&mut self) {
        self.shown_edited = self.is_edited();
        let entries: Vec<sidebar::Entry> = self
            .routes
            .iter()
            .enumerate()
            .map(|(index, route)| sidebar::Entry {
                title: &route.id,
                single_model: route.kind == "passthrough",
                edited: if index == self.route {
                    self.shown_edited
                } else {
                    self.drafts.contains_key(&route.key)
                },
            })
            .collect();
        self.source
            .show(&self.shell.table, sidebar::rows(&entries), self.route);
    }

    /// Loads the lists of the rows' clients that the window does not have:
    /// from the cache file, or fetched when the cache file has none.
    fn list_models(&mut self) {
        let clients = self.row_clients(|state| state.is_none());
        if !clients.is_empty() {
            self.load_lists(clients, false, None);
        }
    }

    /// Fetches the lists that the rows use again, one request per URL.
    fn refresh_models(&mut self) {
        let clients = self.row_clients(|state| !state.is_some_and(ListState::loading));
        if clients.is_empty() {
            return;
        }
        self.set_list_notes("Refreshing the model lists…".to_string());
        self.load_lists(clients, true, None);
    }

    /// Returns the clients that the rows use and whose list state passes
    /// `wanted`, each client once.
    fn row_clients(&self, wanted: impl Fn(Option<&ListState>) -> bool) -> Vec<Client> {
        let mut clients: Vec<Client> = Vec::new();
        for choice in &self.choices {
            if let Some(client) = self.client(&choice.client)
                && wanted(self.lists.get(&models::list_url(client)))
                && !clients.contains(client)
            {
                clients.push(client.clone());
            }
        }
        clients
    }

    /// Loads the lists that `clients` use on a worker thread, which loads
    /// each URL on its own thread and reports each list as it arrives. With
    /// `refresh`, the worker fetches the lists even when the cache file has
    /// them, and the outcome replaces the result area's list notes, because
    /// the user asked for it. `key` is a key the user just typed, with its
    /// `base_url`. The worker lists the models with it first, and saves it in
    /// the Keychain only when the models endpoint did not reject it.
    fn load_lists(&mut self, clients: Vec<Client>, refresh: bool, key: Option<(String, String)>) {
        self.loads += 1;
        let load = self.loads;
        for client in &clients {
            self.lists
                .entry(models::list_url(client))
                .or_default()
                .started = load;
        }
        self.update_rows();
        let cache = self.cache.clone();
        let sender = self.sender.clone();
        std::thread::spawn(move || {
            let typed_key = key.as_ref().map(|(_, key)| key.as_str());
            let lists = models::load(&cache, &clients, refresh, typed_key, &|loaded| {
                let _ = sender.send(Done::Listed {
                    load,
                    loaded: loaded.clone(),
                });
            });
            if !refresh {
                return;
            }
            let mut status = Vec::new();
            if let Some((base_url, key)) = &key {
                status.push(keep_key(base_url, key, &lists));
            }
            status.extend(lists.iter().map(|loaded| outcome(loaded, &clients)));
            let _ = sender.send(Done::Status(status.join("\n")));
        });
    }

    fn client(&self, name: &str) -> Option<&Client> {
        self.clients.iter().find(|client| client.name == name)
    }

    fn on_control(&mut self, tag: isize) {
        let Some(controls) = &self.controls else {
            return;
        };
        match tag {
            ALGORITHM_TAG => {
                if let Some(index) = selected(&controls.algorithm)
                    && index != self.algorithm
                {
                    self.select_algorithm(index);
                }
            }
            SAVE_KEY_TAG => self.save_key(),
            REFRESH_TAG => self.refresh_models(),
            APPLY_TAG => self.apply(),
            REVERT_TAG => self.revert(),
            CLOSE_TAG => self.window.close(),
            _ if (CLIENT_TAG..MODEL_TAG).contains(&tag) => {
                let row = tag.abs_diff(CLIENT_TAG);
                let client = controls
                    .rows
                    .get(row)
                    .and_then(|ui| selected(&ui.client))
                    .and_then(|index| self.clients.get(index))
                    .map(|client| client.name.clone());
                if let (Some(client), Some(choice)) = (client, self.choices.get_mut(row)) {
                    choice.client = client;
                    self.list_models();
                    // A new client can show or hide the key field and
                    // enable or disable Apply.
                    self.update_rows();
                }
            }
            _ if (MODEL_TAG..SELECTED_TAG).contains(&tag) => {
                let row = tag.abs_diff(MODEL_TAG);
                if let (Some(ui), Some(choice)) =
                    (controls.rows.get(row), self.choices.get_mut(row))
                {
                    choice.model = ui.model.stringValue().to_string();
                }
                self.update_row(row);
            }
            _ if tag >= SELECTED_TAG => {
                // The box shows the picked model only after this event, so
                // read the picked item rather than the box's text.
                let row = tag.abs_diff(SELECTED_TAG);
                let picked = controls
                    .rows
                    .get(row)
                    .and_then(|ui| ui.model.objectValueOfSelectedItem())
                    .and_then(|value| value.downcast::<NSString>().ok());
                if let (Some(picked), Some(choice)) = (picked, self.choices.get_mut(row)) {
                    choice.model = picked.to_string();
                    self.update_row(row);
                }
            }
            _ => {}
        }
    }

    fn on_done(&mut self, done: Done) {
        match done {
            Done::Listed {
                load,
                loaded: Loaded { url, list, error },
            } => {
                self.lists.entry(url).or_default().finish(load, list, error);
                self.update_rows();
            }
            Done::Status(status) => self.set_list_notes(status),
            Done::Applied(result) => {
                self.busy = false;
                match result {
                    Ok(message) => {
                        if let Some(route) = self.routes.get(self.route) {
                            self.drafts.remove(&route.key);
                        }
                        let prices = self.unpriced_models();
                        let route = self.routes.get(self.route).map(|route| route.key.clone());
                        self.load(route.as_deref());
                        self.layout();
                        self.set_result(format!("{message}{prices}"));
                    }
                    Err(error) => self.set_result(format!("Not saved. {error}")),
                }
            }
        }
    }

    /// Returns a note that names the chosen models with no price in
    /// menubar.toml, or an empty note when every model has one.
    fn unpriced_models(&self) -> String {
        let models = self.choices.iter().map(|choice| choice.model.trim());
        let missing = crate::pricing::unpriced(models, &self.settings.prices);
        if missing.is_empty() {
            return String::new();
        }
        let pronoun = if missing.len() == 1 { "it" } else { "them" };
        format!(
            "\nmenubar.toml has no price for {}. Savings stay hidden until you add {pronoun}.",
            missing.join(", ")
        )
    }

    /// Lists the models of every client with the key field's `base_url`
    /// using the typed key, and then saves the key unless the models
    /// endpoint rejected it.
    fn save_key(&mut self) {
        let (Some(controls), Some(base_url)) = (&self.controls, self.key_url().map(str::to_string))
        else {
            return;
        };
        let key = controls.key.stringValue().to_string().trim().to_string();
        controls.key.setStringValue(ns_string!(""));
        if key.is_empty() {
            self.set_list_notes("Paste a key into the field first.".to_string());
            return;
        }
        if models::has_line_break(&key) {
            self.set_list_notes(
                "Could not use the key, because it has a line break. Paste the key again \
                 without line breaks."
                    .to_string(),
            );
            return;
        }
        let clients: Vec<Client> = self
            .clients
            .iter()
            .filter(|client| client.base_url == base_url)
            .cloned()
            .collect();
        self.set_list_notes(format!(
            "Checking the key by listing the models at {base_url}…"
        ));
        self.load_lists(clients, true, Some((base_url, key)));
    }

    fn apply(&mut self) {
        if !self.can_apply() {
            return;
        }
        self.read_models();
        let (Some(route), Some(algorithm)) =
            (self.routes.get(self.route), ALGORITHMS.get(self.algorithm))
        else {
            return;
        };
        let route = route.key.clone();
        let choices = self.choices.clone();
        let settings = self.settings.clone();
        let sender = self.sender.clone();
        std::thread::spawn(move || {
            let result = server::apply(&settings, &route, algorithm, &choices);
            let _ = sender.send(Done::Applied(result));
        });
        self.busy = true;
        self.list_notes.clear();
        self.set_result("Checking the config with switchyard-server --dry-run…".to_string());
    }

    /// Copies the typed model IDs into the choices.
    fn read_models(&mut self) {
        let Some(controls) = &self.controls else {
            return;
        };
        for (row, choice) in controls.rows.iter().zip(&mut self.choices) {
            choice.model = row.model.stringValue().to_string();
        }
    }

    fn set_result(&mut self, result: String) {
        self.result = result;
        self.show_status();
    }

    fn set_list_notes(&mut self, notes: String) {
        self.list_notes = notes;
        self.show_status();
    }

    /// Writes the result and the list notes in the result area.
    fn show_status(&self) {
        let Some(controls) = &self.controls else {
            return;
        };
        let text: Vec<&str> = [self.result.as_str(), self.list_notes.as_str()]
            .into_iter()
            .filter(|text| !text.is_empty())
            .collect();
        controls
            .status
            .setString(&NSString::from_str(&text.join("\n")));
        self.update_buttons();
    }

    /// Returns whether Apply can run: no Apply is running, the config has
    /// loaded, the user changed the open route, and every role has a model
    /// and an endpoint that the config defines.
    fn can_apply(&self) -> bool {
        !self.busy
            && self.config.is_some()
            && self.is_edited()
            && self.choices.iter().all(|choice| {
                self.client(&choice.client).is_some() && !choice.model.trim().is_empty()
            })
    }

    fn update_buttons(&self) {
        if let Some(controls) = &self.controls {
            controls.apply.setEnabled(self.can_apply());
            controls.revert.setEnabled(!self.busy && self.is_edited());
        }
    }

    fn update_rows(&self) {
        for row in 0..self.choices.len() {
            self.update_row(row);
        }
        let Some(controls) = &self.controls else {
            return;
        };
        self.update_buttons();
        let key_url = self.key_url();
        let hidden = key_url.is_none();
        controls.key_label.setHidden(hidden);
        controls.key.setHidden(hidden);
        controls.save_key.setHidden(hidden);
        if let Some(url) = key_url {
            controls
                .key_label
                .setStringValue(&NSString::from_str(&format!(
                    "To list the models at {url}, paste the API key for that address and click \
                     Save key. The app uses the key only to list models there and keeps it in \
                     your login Keychain, not in a file."
                )));
            controls
                .key
                .setAccessibilityLabel(Some(&NSString::from_str(&format!("API key for {url}"))));
        }
        self.fit_rows();
    }

    /// Sizes the role rows' scroll view to the room above the result area,
    /// less the key area's room while the key area shows, and shows the first
    /// role. A scroll view that already has the right height keeps its scroll
    /// position, so a model list that arrives does not move the row that the
    /// user is editing.
    fn fit_rows(&self) {
        let Some(controls) = &self.controls else {
            return;
        };
        let key = if self.key_url().is_some() {
            KEY_HEIGHT
        } else {
            0.0
        };
        let rows_height = ROLE_HEIGHT * controls.rows.len() as f64;
        let room = controls.rows_top - ROWS_BOTTOM - key;
        let shown = rows_height.min(room.max(ROLE_HEIGHT));
        if controls.scroll.frame().size.height == shown {
            return;
        }
        controls.scroll.setFrame(rect(
            MARGIN,
            controls.rows_top - shown,
            CONTENT_WIDTH,
            shown,
        ));
        // The rows view's origin is its bottom left, so the scroll view
        // starts at the last role. Show the first role instead.
        controls
            .rows_view
            .scrollPoint(NSPoint::new(0.0, rows_height - shown));
    }

    /// Returns the `base_url` that the key field saves a key for: the
    /// `base_url` of the first row whose list failed because it needs a
    /// working key.
    fn key_url(&self) -> Option<&str> {
        self.choices.iter().find_map(|choice| {
            let client = self.client(&choice.client)?;
            let error = self.lists.get(&models::list_url(client))?.error.as_ref()?;
            error.needs_key().then_some(client.base_url.as_str())
        })
    }

    /// Fills a row's model list and writes the note under it.
    fn update_row(&self, row: usize) {
        let (Some(controls), Some(choice)) = (&self.controls, self.choices.get(row)) else {
            return;
        };
        let Some(ui) = controls.rows.get(row) else {
            return;
        };
        ui.model.removeAllItems();
        let client = self.client(&choice.client);
        let state = client.and_then(|client| self.lists.get(&models::list_url(client)));
        let loading = state.is_some_and(ListState::loading);
        let note = match (client, state) {
            (None, _) if self.clients.is_empty() => "The config has no [llm_clients] entries. \
                                                     Add an LLM client to the server config, \
                                                     then open this window again."
                .to_string(),
            (None, _) => format!(
                "The config has no LLM client named \"{}\". Pick an endpoint in this row.",
                choice.client
            ),
            (Some(client), _) if models::unlisted(client) => {
                let saved = state.and_then(|state| state.list.as_ref());
                self.unlisted_note(ui, client, saved, &choice.model)
            }
            (
                Some(client),
                Some(ListState {
                    list: Some(list),
                    error,
                    ..
                }),
            ) => {
                let shown = show_models(ui, &list.models, &choice.model);
                let mut note = match_note(shown, list.models.len()).unwrap_or_else(|| {
                    format!(
                        "{} models on {}, fetched {}. Type to search.",
                        list.models.len(),
                        client.host(),
                        age(list.fetched_at)
                    )
                });
                if loading {
                    note.push_str(" Refreshing…");
                } else if let Some(error) = error {
                    note.push('\n');
                    note.push_str(&error.reason());
                }
                // A cached list loads without a key, so the note must name
                // the missing variable that Apply needs.
                if let Some(variable) = models::missing_env(client) {
                    note.push('\n');
                    note.push_str(&models::missing_env_note(variable));
                }
                note
            }
            (
                Some(_),
                Some(ListState {
                    list: None,
                    error: Some(error),
                    ..
                }),
            ) if !loading => format!("No model list yet. {} {}", error.reason(), error.advice()),
            (Some(_), _) => "Loading models…".to_string(),
        };
        let note = NSString::from_str(&note);
        ui.note.setStringValue(&note);
        // The note fits two lines, and its tooltip shows the whole text.
        ui.note.setToolTip(Some(&note));
        ui.client.setToolTip(
            client
                .map(|client| NSString::from_str(&endpoint_tip(client)))
                .as_deref(),
        );
    }

    /// Fills the list of a row whose client has no model list, and returns
    /// its note. The list holds the models that the config already names on
    /// the client, and the models that Codex saved.
    fn unlisted_note(
        &self,
        ui: &RoleRow,
        client: &Client,
        saved: Option<&ModelList>,
        query: &str,
    ) -> String {
        let mut models = self
            .config
            .as_ref()
            .map(|config| config.models_on(&client.name))
            .unwrap_or_default();
        if let Some(saved) = saved {
            models.extend(saved.models.iter().cloned());
        }
        models.sort();
        models.dedup();
        let shown = show_models(ui, &models, query);
        match_note(shown, models.len()).unwrap_or_else(|| {
            let source = match saved {
                Some(saved) => format!(
                    "the models in your config and the ones that Codex saved {}",
                    age(saved.fetched_at)
                ),
                None => "the models in your config".to_string(),
            };
            format!(
                "{} has no model list that this app can read. The list shows {source}. Type \
                 any other model ID.",
                client.host()
            )
        })
    }

    /// Rebuilds the detail pane for the open route and algorithm. The role
    /// rows sit in a scroll view, so a route with many models still leaves
    /// the buttons on the screen.
    fn layout(&mut self) {
        let mtm = self.mtm;
        let height = self.shell.height;
        let (content, old_detail) = (self.shell.content.clone(), self.shell.detail.clone());
        let detail = NSView::initWithFrame(
            NSView::alloc(mtm),
            rect(SIDEBAR_WIDTH, 0.0, DETAIL_WIDTH, height),
        );
        // Controls are placed from the top; AppKit's origin is bottom left.
        let mut top = 20.0;
        let mut next = |row_height: f64| {
            let y = height - top - row_height;
            top += row_height;
            y
        };

        let route = self.routes.get(self.route);
        let y = next(30.0);
        detail.addSubview(&self.heading(
            route.map_or("", |route| route.id.as_str()),
            rect(MARGIN, y, CONTENT_WIDTH, 26.0),
        ));
        let y = next(24.0);
        let about = route.map_or(String::new(), |route| {
            let table = if route.key == route.id {
                String::new()
            } else {
                format!(" It is [routes.{}] in the config.", route.key)
            };
            format!(
                "Callers choose this route by sending \"{}\" as the model.{table}",
                route.id
            )
        });
        detail.addSubview(&self.small(&about, rect(MARGIN, y + 2.0, CONTENT_WIDTH, 16.0)));
        let block = self
            .config
            .as_ref()
            .zip(route)
            .and_then(|(config, route)| config.generated_by("routes", &route.key));
        if let Some(block) = block {
            let y = next(44.0);
            let banner = self.wrapping(
                &format!(
                    "Another tool writes this route and overwrites changes made here the \
                     next time it runs. Block marker: \"{block}\"."
                ),
                rect(MARGIN, y + 4.0, CONTENT_WIDTH, 36.0),
            );
            banner.setTextColor(Some(&NSColor::systemOrangeColor()));
            detail.addSubview(&banner);
        }

        let y = next(44.0);
        detail.addSubview(&self.label("Algorithm", rect(MARGIN, y + 16.0, 80.0, 18.0)));
        let titles: Vec<String> = ALGORITHMS
            .iter()
            .map(|algorithm| format!("{} ({})", algorithm.title, algorithm.kind))
            .collect();
        let algorithm = self.popup(
            &titles,
            Some(self.algorithm),
            rect(MARGIN + 90.0, y + 12.0, CONTENT_WIDTH - 90.0, 26.0),
        );
        algorithm.setTag(ALGORITHM_TAG);
        algorithm.setAccessibilityLabel(Some(ns_string!("Algorithm")));
        detail.addSubview(&algorithm);
        let y = next(SUMMARY_HEIGHT);
        detail.addSubview(
            &self.small(
                ALGORITHMS
                    .get(self.algorithm)
                    .map_or("", |algorithm| algorithm.summary),
                rect(MARGIN + 90.0, y, CONTENT_WIDTH - 90.0, SUMMARY_HEIGHT - 2.0),
            ),
        );

        let y = next(34.0);
        detail.addSubview(&self.bold("Models", rect(MARGIN, y + 8.0, ROLE_WIDTH, 18.0)));
        detail.addSubview(&self.column_title(
            "Endpoint",
            rect(MARGIN + ENDPOINT_X + 2.0, y + 8.0, ENDPOINT_WIDTH, 16.0),
        ));
        detail.addSubview(&self.column_title(
            "Model",
            rect(MARGIN + MODEL_X + 2.0, y + 8.0, MODEL_WIDTH, 16.0),
        ));
        let rows_top = height - top;

        let rows_height = ROLE_HEIGHT * self.roles.len() as f64;
        let rows_view = NSView::initWithFrame(
            NSView::alloc(mtm),
            rect(0.0, 0.0, CONTENT_WIDTH, rows_height),
        );
        let endpoints: Vec<String> = self.clients.iter().map(endpoint_title).collect();
        let mut rows = Vec::with_capacity(self.roles.len());
        for (row, (role, choice)) in self.roles.iter().zip(&self.choices).enumerate() {
            let y = rows_height - ROLE_HEIGHT * (row + 1) as f64;
            let tag = isize::try_from(row).unwrap_or(0);
            // The two dropdowns sit at the top of the row, above the note.
            let controls_y = y + ROLE_HEIGHT - 34.0;
            rows_view
                .addSubview(&self.bold(&role.label, rect(0.0, controls_y + 4.0, ROLE_WIDTH, 18.0)));
            rows_view.addSubview(&self.small(
                role.hint,
                rect(0.0, y + 6.0, ROLE_WIDTH - 8.0, ROLE_HEIGHT - 42.0),
            ));
            // When the config has no client with the choice's name, the
            // popup selects nothing, and the note names the missing client.
            let selected = self
                .clients
                .iter()
                .position(|client| client.name == choice.client);
            let client = self.popup(
                &endpoints,
                selected,
                rect(ENDPOINT_X, controls_y, ENDPOINT_WIDTH, 26.0),
            );
            client.setTag(CLIENT_TAG + tag);
            client.setAccessibilityLabel(Some(&NSString::from_str(&format!(
                "{} endpoint",
                role.label
            ))));
            rows_view.addSubview(&client);

            let model = ModelBox::new(mtm, rect(MODEL_X, controls_y, MODEL_WIDTH, 26.0));
            model.setStringValue(&NSString::from_str(&choice.model));
            model.setPlaceholderString(Some(ns_string!("Search, or type a model ID")));
            model.setNumberOfVisibleItems(14);
            model.setCompletes(false);
            model.setTag(MODEL_TAG + tag);
            model
                .setAccessibilityLabel(Some(&NSString::from_str(&format!("{} model", role.label))));
            // SAFETY: the target lives as long as the window.
            unsafe { model.setDelegate(Some(ProtocolObject::from_ref(&*self.target))) };
            rows_view.addSubview(&model);

            let note = self.small(
                "",
                rect(
                    ENDPOINT_X + 2.0,
                    y + 4.0,
                    CONTENT_WIDTH - ENDPOINT_X - 8.0,
                    NOTE_HEIGHT,
                ),
            );
            rows_view.addSubview(&note);
            rows.push(RoleRow {
                client,
                model,
                note,
            });
        }
        // `fit_rows` gives the scroll view its size, because the room depends
        // on whether the key field shows.
        let scroll =
            NSScrollView::initWithFrame(NSScrollView::alloc(mtm), rect(0.0, 0.0, 0.0, 0.0));
        scroll.setHasVerticalScroller(true);
        scroll.setAutohidesScrollers(true);
        scroll.setDrawsBackground(false);
        scroll.setDocumentView(Some(&rows_view));
        detail.addSubview(&scroll);

        let key_label = self.wrapping("", rect(MARGIN, ROWS_BOTTOM + 30.0, CONTENT_WIDTH, 48.0));
        detail.addSubview(&key_label);
        let key = NSSecureTextField::initWithFrame(
            NSSecureTextField::alloc(mtm),
            rect(MARGIN, ROWS_BOTTOM + 4.0, CONTENT_WIDTH - 110.0, 22.0),
        );
        key.setPlaceholderString(Some(ns_string!("Paste the API key")));
        // Return in the field saves the key instead of running Apply.
        key.setTag(SAVE_KEY_TAG);
        // SAFETY: `ControlTarget` implements `controlChanged:` and lives as
        // long as the window.
        unsafe {
            key.setTarget(Some(&self.target));
            key.setAction(Some(sel!(controlChanged:)));
        }
        detail.addSubview(&key);
        let save_key = self.button(
            "Save key",
            SAVE_KEY_TAG,
            rect(MARGIN + CONTENT_WIDTH - 100.0, ROWS_BOTTOM, 100.0, 30.0),
        );
        detail.addSubview(&save_key);

        // The result can be longer than the area, such as a long --dry-run
        // error, so the area scrolls.
        let status_scroll = NSScrollView::initWithFrame(
            NSScrollView::alloc(mtm),
            rect(MARGIN, STATUS_Y, CONTENT_WIDTH, STATUS_HEIGHT),
        );
        status_scroll.setHasVerticalScroller(true);
        status_scroll.setAutohidesScrollers(true);
        status_scroll.setDrawsBackground(false);
        let size = status_scroll.contentSize();
        let status = NSTextView::initWithFrame(
            NSTextView::alloc(mtm),
            rect(0.0, 0.0, size.width, size.height),
        );
        // The text view grows down as the text gets longer, and wraps at the
        // area's width.
        status.setMinSize(size);
        status.setMaxSize(NSSize::new(size.width, f64::MAX));
        status.setVerticallyResizable(true);
        status.setHorizontallyResizable(false);
        status.setEditable(false);
        status.setDrawsBackground(false);
        status.setFont(Some(&NSFont::systemFontOfSize(NSFont::systemFontSize())));
        // The text must follow the light or dark appearance, because the area
        // draws no background of its own.
        status.setTextColor(Some(&NSColor::labelColor()));
        status_scroll.setDocumentView(Some(&status));
        detail.addSubview(&status_scroll);

        let right = MARGIN + CONTENT_WIDTH;
        let refresh = self.button(
            "Refresh models",
            REFRESH_TAG,
            rect(MARGIN, BUTTONS_Y, 150.0, BUTTONS_HEIGHT),
        );
        refresh.setEnabled(!self.choices.is_empty());
        detail.addSubview(&refresh);
        let revert = self.button(
            "Revert",
            REVERT_TAG,
            rect(right - 320.0, BUTTONS_Y, 100.0, BUTTONS_HEIGHT),
        );
        detail.addSubview(&revert);
        let close = self.button(
            "Close",
            CLOSE_TAG,
            rect(right - 210.0, BUTTONS_Y, 100.0, BUTTONS_HEIGHT),
        );
        // Escape closes the window, as in a macOS dialog.
        close.setKeyEquivalent(&NSString::from_str("\u{1b}"));
        detail.addSubview(&close);
        let apply = self.button(
            "Apply",
            APPLY_TAG,
            rect(right - 100.0, BUTTONS_Y, 100.0, BUTTONS_HEIGHT),
        );
        // Return runs Apply, the window's default button.
        apply.setKeyEquivalent(ns_string!("\r"));
        detail.addSubview(&apply);

        content.replaceSubview_with(&old_detail, &detail);
        self.shell.detail = detail;
        self.controls = Some(Controls {
            algorithm,
            rows,
            scroll,
            rows_view,
            rows_top,
            key_label,
            key,
            save_key,
            status,
            revert,
            apply,
        });
        // Tab moves between the controls in the order they appear, including
        // the controls inside the scroll view.
        self.window.recalculateKeyViewLoop();
        self.sync_sidebar();
        self.update_rows();
        self.show_status();
    }

    fn label(&self, text: &str, frame: NSRect) -> Retained<NSTextField> {
        let label = NSTextField::labelWithString(&NSString::from_str(text), self.mtm);
        label.setFrame(frame);
        label
    }

    fn wrapping(&self, text: &str, frame: NSRect) -> Retained<NSTextField> {
        let label = NSTextField::wrappingLabelWithString(&NSString::from_str(text), self.mtm);
        label.setFrame(frame);
        label
    }

    /// Returns a wrapping label in the small system font and the secondary color.
    fn small(&self, text: &str, frame: NSRect) -> Retained<NSTextField> {
        let label = self.wrapping(text, frame);
        label.setFont(Some(&NSFont::systemFontOfSize(
            NSFont::smallSystemFontSize(),
        )));
        label.setTextColor(Some(&NSColor::secondaryLabelColor()));
        label
    }

    fn heading(&self, text: &str, frame: NSRect) -> Retained<NSTextField> {
        let label = self.label(text, frame);
        label.setFont(Some(&NSFont::boldSystemFontOfSize(20.0)));
        label
    }

    fn bold(&self, text: &str, frame: NSRect) -> Retained<NSTextField> {
        let label = self.label(text, frame);
        label.setFont(Some(
            &NSFont::boldSystemFontOfSize(NSFont::systemFontSize()),
        ));
        label
    }

    fn column_title(&self, text: &str, frame: NSRect) -> Retained<NSTextField> {
        let label = self.label(text, frame);
        label.setFont(Some(&NSFont::boldSystemFontOfSize(
            NSFont::smallSystemFontSize(),
        )));
        label.setTextColor(Some(&NSColor::secondaryLabelColor()));
        label
    }

    /// Returns a popup that lists `items` with `selected` chosen, or with no
    /// item chosen when `selected` is `None`.
    fn popup(
        &self,
        items: &[String],
        selected: Option<usize>,
        frame: NSRect,
    ) -> Retained<NSPopUpButton> {
        let popup =
            NSPopUpButton::initWithFrame_pullsDown(NSPopUpButton::alloc(self.mtm), frame, false);
        for item in items {
            popup.addItemWithTitle(&NSString::from_str(item));
        }
        popup.selectItemAtIndex(
            selected
                .and_then(|index| isize::try_from(index).ok())
                .unwrap_or(-1),
        );
        // SAFETY: `ControlTarget` implements `controlChanged:` and lives as
        // long as the window.
        unsafe {
            popup.setTarget(Some(&self.target));
            popup.setAction(Some(sel!(controlChanged:)));
        }
        popup
    }

    fn button(&self, title: &str, tag: isize, frame: NSRect) -> Retained<NSButton> {
        // SAFETY: `ControlTarget` implements `controlChanged:` and lives as
        // long as the window.
        let button = unsafe {
            NSButton::buttonWithTitle_target_action(
                &NSString::from_str(title),
                Some(&self.target),
                Some(sel!(controlChanged:)),
                self.mtm,
            )
        };
        button.setFrame(frame);
        button.setTag(tag);
        button
    }
}

/// The selected item of a popup, if any.
fn selected(popup: &NSPopUpButton) -> Option<usize> {
    usize::try_from(popup.indexOfSelectedItem()).ok()
}

pub fn rect(x: f64, y: f64, width: f64, height: f64) -> NSRect {
    NSRect::new(NSPoint::new(x, y), NSSize::new(width, height))
}

/// Builds the window's fixed content: the sidebar, its divider, and an empty
/// detail pane. The window is as tall as the screen allows, up to
/// `MAX_HEIGHT`.
fn build_shell(mtm: MainThreadMarker, window: &NSWindow, source: &SidebarSource) -> Shell {
    let screen = window.screen().or_else(|| NSScreen::mainScreen(mtm));
    let room = screen.as_ref().map_or(MAX_HEIGHT, |screen| {
        window
            .contentRectForFrameRect(screen.visibleFrame())
            .size
            .height
            - 20.0
    });
    let height = MAX_HEIGHT.min(room).max(MIN_HEIGHT);
    let content = NSView::initWithFrame(NSView::alloc(mtm), rect(0.0, 0.0, WIDTH, height));
    let (panel, table) = sidebar::build(mtm, source, height);
    content.addSubview(&panel);
    let divider = NSBox::initWithFrame(NSBox::alloc(mtm), rect(SIDEBAR_WIDTH, 0.0, 1.0, height));
    divider.setBoxType(NSBoxType::Separator);
    content.addSubview(&divider);
    let detail = NSView::initWithFrame(
        NSView::alloc(mtm),
        rect(SIDEBAR_WIDTH, 0.0, DETAIL_WIDTH, height),
    );
    content.addSubview(&detail);
    window.setContentSize(NSSize::new(WIDTH, height));
    window.setContentView(Some(&content));
    Shell {
        content,
        table,
        detail,
        height,
    }
}

/// Returns whether the open route differs from the algorithm and choices
/// that the config holds. Leading and trailing spaces in a model ID do not
/// count.
fn differs(saved: &Draft, algorithm: usize, choices: &[Choice]) -> bool {
    algorithm != saved.algorithm
        || choices.len() != saved.choices.len()
        || choices.iter().zip(&saved.choices).any(|(now, saved)| {
            now.client != saved.client || now.model.trim() != saved.model.trim()
        })
}

/// Names the request format of a client in one short word, for the popup and
/// its tooltip.
fn format_word(client: &Client) -> &str {
    match client.format.as_str() {
        "openai_chat" => "Chat",
        "openai_responses" => "Responses",
        "anthropic_messages" => "Anthropic",
        other => other,
    }
}

/// Names an endpoint in a role's popup: the client's name and the request
/// format that it speaks.
fn endpoint_title(client: &Client) -> String {
    format!("{} ({})", client.name, format_word(client))
}

/// Says where the client sends requests, in which format, and whose key it
/// uses.
fn endpoint_tip(client: &Client) -> String {
    let key = if client.forward_auth {
        "It uses each caller's own login.".to_string()
    } else if let Some(variable) = &client.api_key_env {
        format!("It uses the API key from the environment variable {variable}.")
    } else {
        "It sends no key.".to_string()
    };
    format!(
        "Sends requests to {} in the {} format. {key}",
        client.base_url,
        format_word(client)
    )
}

/// Puts the models that match `query` in the row's list, and returns how
/// many. A complete ID shows the whole list, so another model is one click
/// away.
fn show_models(ui: &RoleRow, models: &[String], query: &str) -> usize {
    let query = query.trim();
    let shown: Vec<&str> = if models.iter().any(|model| model == query) {
        models.iter().map(String::as_str).collect()
    } else {
        models::matching(models, query)
    };
    for model in &shown {
        // SAFETY: the combo box stores NSString values.
        unsafe { ui.model.addItemWithObjectValue(&NSString::from_str(model)) };
    }
    shown.len()
}

/// Says how many models match what the user typed, or returns `None` when
/// the whole list shows.
fn match_note(shown: usize, total: usize) -> Option<String> {
    if shown == total {
        None
    } else if shown == 0 {
        Some("No listed model matches. Apply uses the ID as typed.".to_string())
    } else {
        Some(format!("{shown} of {total} models match."))
    }
}

/// Saves a typed key in the Keychain unless the models endpoint rejected
/// it while the app listed `lists` with it, and says what happened.
fn keep_key(base_url: &str, key: &str, lists: &[Loaded]) -> String {
    if lists
        .iter()
        .any(|loaded| matches!(loaded.error, Some(ListError::Rejected(_))))
    {
        return format!(
            "Did not save the key for {base_url}, because the models endpoint rejected it."
        );
    }
    match models::save_key(base_url, key) {
        Ok(()) => format!("Saved the key for {base_url} in your login Keychain."),
        Err(error) => format!(
            "Could not save the key in your login Keychain: {error}. The app used the key this \
             time only and did not keep it. To try again, paste the key and click Save key."
        ),
    }
}

/// Describes one loaded list for the result area. A ChatGPT client has no
/// list to fetch, so its line names the file that Codex saved instead.
fn outcome(loaded: &Loaded, clients: &[Client]) -> String {
    let chatgpt = clients
        .iter()
        .find(|client| models::list_url(client) == loaded.url && models::unlisted(client));
    if let Some(client) = chatgpt {
        return match &loaded.list {
            Some(list) => format!(
                "{}: read the {} models that Codex saved {}.",
                client.host(),
                list.models.len(),
                age(list.fetched_at)
            ),
            None => format!("{}: Codex has not saved a model list.", client.host()),
        };
    }
    let mut line = format!("{}:", loaded.url);
    if let Some(list) = &loaded.list {
        line.push_str(&format!(
            " {} models, fetched {}.",
            list.models.len(),
            age(list.fetched_at)
        ));
    }
    if let Some(error) = &loaded.error {
        line.push(' ');
        line.push_str(&error.reason());
    }
    line
}

/// Says how long ago `fetched_at` was, such as "3 hours ago".
fn age(fetched_at: u64) -> String {
    let minutes = models::now().saturating_sub(fetched_at) / 60;
    let (count, unit) = match minutes {
        0 => return "just now".to_string(),
        1..60 => (minutes, "minute"),
        60..1440 => (minutes / 60, "hour"),
        _ => (minutes / 1440, "day"),
    };
    let plural = if count == 1 { "" } else { "s" };
    format!("{count} {unit}{plural} ago")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(fetched_at: u64) -> Option<ModelList> {
        Some(ModelList {
            models: vec!["model".to_string()],
            fetched_at,
        })
    }

    fn choice(client: &str, model: &str) -> Choice {
        Choice {
            client: client.to_string(),
            model: model.to_string(),
        }
    }

    #[test]
    fn a_failed_load_keeps_the_newer_list_and_shows_the_error() {
        // Load 1 fetched a list that the cache file does not have, so load 2
        // fails with no list at all.
        let mut state = ListState {
            list: list(100),
            started: 2,
            shown: 1,
            ..ListState::default()
        };
        let failed = ListError::Failed("connection refused".to_string());

        state.finish(2, None, Some(failed.clone()));

        assert!(!state.loading());
        assert_eq!(state.list, list(100));
        assert_eq!(state.error, Some(failed));
    }

    #[test]
    fn ignores_a_load_that_finishes_after_a_newer_one() {
        let mut state = ListState {
            started: 2,
            ..ListState::default()
        };

        state.finish(2, list(200), None);
        state.finish(1, None, Some(ListError::Rejected(401)));

        assert!(!state.loading());
        assert_eq!(state.list, list(200));
        assert_eq!(state.error, None);
    }

    #[test]
    fn a_route_is_edited_when_the_algorithm_endpoint_or_model_changes() {
        let saved = Draft {
            algorithm: 3,
            choices: vec![choice("hub", "sonnet"), choice("hub", "opus")],
        };
        let same = [choice("hub", "sonnet"), choice("hub", "opus")];

        assert!(!differs(&saved, 3, &same));
        // A space typed after a model ID is not a change.
        assert!(!differs(
            &saved,
            3,
            &[choice("hub", "sonnet "), choice("hub", "opus")]
        ));
        assert!(differs(&saved, 4, &same));
        assert!(differs(
            &saved,
            3,
            &[choice("chat", "sonnet"), choice("hub", "opus")]
        ));
        assert!(differs(
            &saved,
            3,
            &[choice("hub", "sonnet"), choice("hub", "haiku")]
        ));
        assert!(differs(&saved, 3, &same[..1]));
    }
}
