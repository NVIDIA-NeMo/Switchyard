// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The "Change routing…" window. The user picks a route, its algorithm, and
//! a model for each role, and Apply checks, saves, and restarts the server.
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
    NSAccessibility, NSApplication, NSBackingStoreType, NSButton, NSColor, NSComboBox,
    NSComboBoxDelegate, NSControl, NSControlTextEditingDelegate, NSFont, NSPopUpButton,
    NSResponder, NSScreen, NSScrollView, NSSecureTextField, NSTextField, NSTextFieldDelegate,
    NSTextView, NSView, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{
    MainThreadMarker, NSNotification, NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize,
    NSString, ns_string,
};

use crate::config::Config;
use crate::models::{self, ListError, Loaded, ModelList};
use crate::server;
use crate::server_config::{ALGORITHMS, Choice, Client, Role, Route, ServerConfig, Tier};

// Control tags. Role rows use `CLIENT_TAG + row` for the client popup and
// `MODEL_TAG + row` for the model box. A pick from the model list arrives as
// `SELECTED_TAG + row`.
const ROUTE_TAG: isize = 1;
const ALGORITHM_TAG: isize = 2;
const SAVE_KEY_TAG: isize = 3;
const APPLY_TAG: isize = 4;
const REFRESH_TAG: isize = 5;
const CLOSE_TAG: isize = 6;
const CLIENT_TAG: isize = 100;
const MODEL_TAG: isize = 200;
const SELECTED_TAG: isize = 300;

const WIDTH: f64 = 780.0;
const MARGIN: f64 = 20.0;
const FIELD_X: f64 = 110.0;
const FIELD_WIDTH: f64 = WIDTH - FIELD_X - MARGIN;
const CLIENT_WIDTH: f64 = 290.0;
const MODEL_X: f64 = FIELD_X + CLIENT_WIDTH + 10.0;
const MODEL_WIDTH: f64 = WIDTH - MODEL_X - MARGIN;
/// Height of one role row: the popup and combo box, then a note of up to
/// three lines under them.
const ROLE_HEIGHT: f64 = 80.0;
const NOTE_HEIGHT: f64 = 44.0;
/// Height of the key area: three lines of text above the key field.
const KEY_HEIGHT: f64 = 80.0;
/// Height of the algorithm's description: up to two lines, then its roles.
const SUMMARY_HEIGHT: f64 = 60.0;
/// Height of the result area. A longer result scrolls.
const STATUS_HEIGHT: f64 = 124.0;

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

/// The controls of one role row.
struct RoleRow {
    client: Retained<NSPopUpButton>,
    model: Retained<ModelBox>,
    note: Retained<NSTextField>,
}

/// The controls that the picker reads or updates after it lays them out.
struct Controls {
    route: Retained<NSPopUpButton>,
    algorithm: Retained<NSPopUpButton>,
    rows: Vec<RoleRow>,
    key_label: Retained<NSTextField>,
    key: Retained<NSSecureTextField>,
    save_key: Retained<NSButton>,
    status: Retained<NSTextView>,
    apply: Retained<NSButton>,
}

/// The picker window and what the user has chosen in it.
pub struct Picker {
    mtm: MainThreadMarker,
    window: Retained<NSWindow>,
    target: Retained<ControlTarget>,
    settings: Config,
    /// The file that keeps fetched model lists.
    cache: PathBuf,
    config: Option<ServerConfig>,
    routes: Vec<Route>,
    clients: Vec<Client>,
    route: usize,
    algorithm: usize,
    roles: Vec<Role>,
    /// One choice per role, in role order.
    choices: Vec<Choice>,
    /// The latest choice for each tier on this route. When the next algorithm
    /// has no role for a tier, its choice stays here, so switching back
    /// restores it. `select_route` clears the map.
    by_tier: HashMap<Tier, Choice>,
    /// Model lists by URL, kept while the app runs. Clients that list their
    /// models at the same URL share one entry.
    lists: HashMap<String, ListState>,
    /// How many list loads the window has started. It numbers the loads.
    loads: u64,
    status: String,
    /// Whether an Apply is running.
    busy: bool,
    sender: Sender<Done>,
    receiver: Receiver<Done>,
    controls: Option<Controls>,
}

impl Picker {
    pub fn new(mtm: MainThreadMarker, settings: &Config, cache: PathBuf) -> Self {
        // SAFETY: the window is not released when closed, because `Picker`
        // owns it and shows it again from the menu.
        let window = unsafe {
            let window = NSWindow::initWithContentRect_styleMask_backing_defer(
                NSWindow::alloc(mtm),
                rect(0.0, 0.0, WIDTH, 400.0),
                NSWindowStyleMask::Titled
                    | NSWindowStyleMask::Closable
                    | NSWindowStyleMask::Miniaturizable,
                NSBackingStoreType::Buffered,
                false,
            );
            window.setReleasedWhenClosed(false);
            window
        };
        window.setTitle(ns_string!("Change routing"));
        let (sender, receiver) = channel();
        Self {
            mtm,
            window,
            target: ControlTarget::new(mtm),
            settings: settings.clone(),
            cache,
            config: None,
            routes: Vec::new(),
            clients: Vec::new(),
            route: 0,
            algorithm: 0,
            roles: Vec::new(),
            choices: Vec::new(),
            by_tier: HashMap::new(),
            lists: HashMap::new(),
            loads: 0,
            status: String::new(),
            busy: false,
            sender,
            receiver,
            controls: None,
        }
    }

    /// Rereads the server config and brings the window to the front.
    pub fn show(&mut self) {
        if !self.busy {
            // Opening the window again retries a list that failed and has
            // nothing to show. The window fetches a list that it already has
            // only when the user clicks Refresh models or Save key.
            self.lists
                .retain(|_, state| state.loading() || state.list.is_some());
            let route = self.routes.get(self.route).map(|route| route.key.clone());
            self.load(route.as_deref());
        }
        let first = self.controls.is_none();
        self.layout();
        if first {
            self.window.center();
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
        for tag in tags {
            self.on_control(tag);
        }
        while let Ok(done) = self.receiver.try_recv() {
            self.on_done(done);
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
                    self.status = format!(
                        "{} has no routes to change.",
                        self.settings.config_file.display()
                    );
                }
            }
            Err(error) => {
                self.config = None;
                self.routes.clear();
                self.clients.clear();
                self.roles.clear();
                self.choices.clear();
                self.status = error;
            }
        }
    }

    /// Shows the selected route's current algorithm and models.
    fn select_route(&mut self) {
        self.by_tier.clear();
        let (Some(config), Some(route)) = (&self.config, self.routes.get(self.route)) else {
            self.roles.clear();
            self.choices.clear();
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
                self.status.clear();
            }
            None => {
                self.algorithm = 0;
                self.roles = config.roles(&route.key, &ALGORITHMS[0]);
                self.choices = vec![Choice::default(); self.roles.len()];
                self.status = format!(
                    "This window cannot show the current settings of route {}: its type is not \
                     in the Algorithm list, or it is a custom-mode llm_classifier. Apply replaces \
                     those settings with the algorithm and models you pick.",
                    route.id
                );
            }
        }
        self.fill_clients();
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
        self.set_status("Refreshing the model lists…".to_string());
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
    /// them, and the outcome replaces the result area's text, because the
    /// user asked for it. `key` is a key the user just typed, with its
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
            status.extend(lists.iter().map(outcome));
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
            ROUTE_TAG => {
                if let Some(index) = selected(&controls.route)
                    && index != self.route
                {
                    self.route = index;
                    self.select_route();
                    self.layout();
                }
            }
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
            Done::Status(status) => self.set_status(status),
            Done::Applied(result) => {
                self.busy = false;
                match result {
                    Ok(message) => {
                        let unpriced = self.unpriced_models();
                        let route = self.routes.get(self.route).map(|route| route.key.clone());
                        self.load(route.as_deref());
                        self.layout();
                        self.set_status(format!("{message}{unpriced}"));
                    }
                    Err(error) => self.set_status(format!("Not saved. {error}")),
                }
            }
        }
    }

    /// Returns a note for each chosen model that has no price in menubar.toml.
    fn unpriced_models(&self) -> String {
        let mut notes = String::new();
        let mut seen = Vec::new();
        for choice in &self.choices {
            let model = choice.model.trim();
            if !self.settings.prices.contains_key(model) && !seen.contains(&model) {
                seen.push(model);
                notes.push_str(&format!(
                    "\nmenubar.toml has no price for {model}. Savings stay hidden until you add \
                     one and restart the menu bar app."
                ));
            }
        }
        notes
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
            self.set_status("Paste a key into the field first.".to_string());
            return;
        }
        if models::has_line_break(&key) {
            self.set_status(
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
        self.set_status(format!(
            "Checking the key by listing the models at {base_url}…"
        ));
        self.load_lists(clients, true, Some((base_url, key)));
    }

    fn apply(&mut self) {
        if self.busy {
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
        self.set_status("Checking the config with switchyard-server --dry-run…".to_string());
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

    fn set_status(&mut self, status: String) {
        self.status = status;
        if let Some(controls) = &self.controls {
            controls.status.setString(&NSString::from_str(&self.status));
            controls.apply.setEnabled(self.can_apply());
        }
    }

    /// Returns whether Apply can run: no Apply is running, the config has
    /// loaded, and every role names an LLM client from the config.
    fn can_apply(&self) -> bool {
        !self.busy
            && self.config.is_some()
            && self
                .choices
                .iter()
                .all(|choice| self.client(&choice.client).is_some())
    }

    fn update_rows(&self) {
        for row in 0..self.choices.len() {
            self.update_row(row);
        }
        let Some(controls) = &self.controls else {
            return;
        };
        controls.apply.setEnabled(self.can_apply());
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
                "The config has no LLM client named \"{}\". Pick an LLM client from the list on \
                 the left.",
                choice.client
            ),
            (
                Some(client),
                Some(ListState {
                    list: Some(list),
                    error,
                    ..
                }),
            ) => {
                let mut note = fill(ui, list, &choice.model);
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
        // The note fits three lines, and its tooltip shows the whole text.
        ui.note.setToolTip(Some(&note));
    }

    /// Rebuilds the window's controls for the current route and algorithm.
    /// The role rows sit in a scroll view. It is as tall as the rows when the
    /// screen has room, and shorter when it does not, so the buttons under
    /// the rows stay on the screen.
    fn layout(&mut self) {
        let mtm = self.mtm;
        let fixed = 2.0 * MARGIN + 34.0 + 32.0 + SUMMARY_HEIGHT + KEY_HEIGHT + STATUS_HEIGHT + 32.0;
        let rows_height = ROLE_HEIGHT * self.roles.len() as f64;
        let screen = self.window.screen().or_else(|| NSScreen::mainScreen(mtm));
        let room = screen.as_ref().map_or(f64::INFINITY, |screen| {
            self.window
                .contentRectForFrameRect(screen.visibleFrame())
                .size
                .height
                - fixed
        });
        let rows_shown = rows_height.min(room.max(ROLE_HEIGHT));
        let height = fixed + rows_shown;
        let view = NSView::initWithFrame(NSView::alloc(mtm), rect(0.0, 0.0, WIDTH, height));
        // Controls are placed from the top; AppKit's origin is bottom left.
        let mut top = MARGIN;
        let mut next = |row_height: f64| {
            let y = height - top - row_height;
            top += row_height;
            y
        };

        let y = next(34.0);
        view.addSubview(&self.label("Route", rect(MARGIN, y + 12.0, 80.0, 18.0)));
        let routes: Vec<String> = self.routes.iter().map(|route| route.id.clone()).collect();
        let route = self.popup(
            &routes,
            Some(self.route),
            rect(FIELD_X, y + 8.0, FIELD_WIDTH, 26.0),
        );
        route.setTag(ROUTE_TAG);
        route.setAccessibilityLabel(Some(ns_string!("Route")));
        view.addSubview(&route);

        let y = next(32.0);
        view.addSubview(&self.label("Algorithm", rect(MARGIN, y + 10.0, 80.0, 18.0)));
        let kinds: Vec<String> = ALGORITHMS.iter().map(|a| a.kind.to_string()).collect();
        let algorithm = self.popup(
            &kinds,
            Some(self.algorithm),
            rect(FIELD_X, y + 6.0, FIELD_WIDTH, 26.0),
        );
        algorithm.setTag(ALGORITHM_TAG);
        algorithm.setAccessibilityLabel(Some(ns_string!("Algorithm")));
        view.addSubview(&algorithm);

        let y = next(SUMMARY_HEIGHT);
        let roles: Vec<&str> = self.roles.iter().map(|role| role.label.as_str()).collect();
        let summary = format!(
            "{}\nRoles: {}.",
            ALGORITHMS.get(self.algorithm).map_or("", |a| a.summary),
            roles.join(", ")
        );
        let summary = self.wrapping(
            &summary,
            rect(FIELD_X, y + 4.0, FIELD_WIDTH, SUMMARY_HEIGHT - 8.0),
        );
        view.addSubview(&summary);

        let clients: Vec<String> = self
            .clients
            .iter()
            .map(|client| format!("{} ({})", client.name, client.format))
            .collect();
        let rows_y = next(rows_shown);
        let rows_view =
            NSView::initWithFrame(NSView::alloc(mtm), rect(0.0, 0.0, WIDTH, rows_height));
        let mut rows = Vec::with_capacity(self.roles.len());
        for (row, (role, choice)) in self.roles.iter().zip(&self.choices).enumerate() {
            let y = rows_height - ROLE_HEIGHT * (row + 1) as f64;
            let tag = isize::try_from(row).unwrap_or(0);
            // The popup and the combo box sit above the note.
            let controls_y = y + NOTE_HEIGHT + 8.0;
            rows_view
                .addSubview(&self.label(&role.label, rect(MARGIN, controls_y + 4.0, 86.0, 18.0)));
            // When the config has no client with the choice's name, the
            // popup selects nothing, and the note names the missing client.
            let selected = self
                .clients
                .iter()
                .position(|client| client.name == choice.client);
            let client = self.popup(
                &clients,
                selected,
                rect(FIELD_X, controls_y, CLIENT_WIDTH, 26.0),
            );
            client.setTag(CLIENT_TAG + tag);
            client.setAccessibilityLabel(Some(&NSString::from_str(&format!(
                "{} LLM client",
                role.label
            ))));
            rows_view.addSubview(&client);

            let model = ModelBox::new(mtm, rect(MODEL_X, controls_y, MODEL_WIDTH, 26.0));
            model.setStringValue(&NSString::from_str(&choice.model));
            model.setPlaceholderString(Some(ns_string!("Type to filter, or type any model ID")));
            model.setNumberOfVisibleItems(14);
            model.setCompletes(false);
            model.setTag(MODEL_TAG + tag);
            model
                .setAccessibilityLabel(Some(&NSString::from_str(&format!("{} model", role.label))));
            // SAFETY: the target lives as long as the window.
            unsafe { model.setDelegate(Some(ProtocolObject::from_ref(&*self.target))) };
            rows_view.addSubview(&model);

            let note = self.wrapping(
                "",
                rect(MODEL_X + 2.0, y + 4.0, MODEL_WIDTH - 2.0, NOTE_HEIGHT),
            );
            note.setFont(Some(&NSFont::systemFontOfSize(
                NSFont::smallSystemFontSize(),
            )));
            note.setTextColor(Some(&NSColor::secondaryLabelColor()));
            rows_view.addSubview(&note);
            rows.push(RoleRow {
                client,
                model,
                note,
            });
        }
        let scroll = NSScrollView::initWithFrame(
            NSScrollView::alloc(mtm),
            rect(0.0, rows_y, WIDTH, rows_shown),
        );
        scroll.setHasVerticalScroller(true);
        scroll.setAutohidesScrollers(true);
        scroll.setDrawsBackground(false);
        scroll.setDocumentView(Some(&rows_view));
        view.addSubview(&scroll);
        // The rows view's origin is its bottom left, so the scroll view
        // starts at the last role. Show the first role instead.
        rows_view.scrollPoint(NSPoint::new(0.0, rows_height - rows_shown));

        let y = next(KEY_HEIGHT);
        let key_label = self.wrapping("", rect(FIELD_X, y + 30.0, FIELD_WIDTH, 48.0));
        view.addSubview(&key_label);
        let key = NSSecureTextField::initWithFrame(
            NSSecureTextField::alloc(mtm),
            rect(FIELD_X, y + 4.0, FIELD_WIDTH - 110.0, 22.0),
        );
        key.setPlaceholderString(Some(ns_string!("Paste the API key")));
        view.addSubview(&key);
        let save_key = self.button(
            "Save key",
            SAVE_KEY_TAG,
            rect(WIDTH - MARGIN - 100.0, y, 100.0, 30.0),
        );
        view.addSubview(&save_key);

        let y = next(STATUS_HEIGHT);
        // The result can be longer than the area, such as a long --dry-run
        // error, so the area scrolls.
        let status_scroll = NSScrollView::initWithFrame(
            NSScrollView::alloc(mtm),
            rect(MARGIN, y + 4.0, WIDTH - 2.0 * MARGIN, STATUS_HEIGHT - 8.0),
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
        status.setString(&NSString::from_str(&self.status));
        status_scroll.setDocumentView(Some(&status));
        view.addSubview(&status_scroll);

        let refresh = self.button(
            "Refresh models",
            REFRESH_TAG,
            rect(MARGIN, MARGIN - 4.0, 150.0, 32.0),
        );
        refresh.setEnabled(!self.choices.is_empty());
        view.addSubview(&refresh);
        let close = self.button(
            "Close",
            CLOSE_TAG,
            rect(WIDTH - MARGIN - 210.0, MARGIN - 4.0, 100.0, 32.0),
        );
        // Escape closes the window, as in a macOS dialog.
        close.setKeyEquivalent(&NSString::from_str("\u{1b}"));
        view.addSubview(&close);
        let apply = self.button(
            "Apply",
            APPLY_TAG,
            rect(WIDTH - MARGIN - 100.0, MARGIN - 4.0, 100.0, 32.0),
        );
        // Return runs Apply, the window's default button.
        apply.setKeyEquivalent(ns_string!("\r"));
        view.addSubview(&apply);

        // Keep the window's top edge in place as its height changes.
        let frame = self.window.frame();
        let top_left = NSPoint::new(frame.origin.x, frame.origin.y + frame.size.height);
        self.window.setContentSize(NSSize::new(WIDTH, height));
        self.window.setContentView(Some(&view));
        // Tab moves between the controls in the order they appear, including
        // the controls inside the scroll view.
        self.window.recalculateKeyViewLoop();
        self.window.setFrameTopLeftPoint(top_left);
        // A taller window can reach past the bottom of the screen. Move it
        // back onto the screen, so that Apply stays in reach.
        if let Some(screen) = &screen {
            let visible = screen.visibleFrame();
            let mut frame = self.window.frame();
            frame.origin.x = frame
                .origin
                .x
                .min(visible.origin.x + visible.size.width - frame.size.width)
                .max(visible.origin.x);
            frame.origin.y = frame
                .origin
                .y
                .max(visible.origin.y)
                .min(visible.origin.y + visible.size.height - frame.size.height);
            self.window.setFrameOrigin(frame.origin);
        }

        self.controls = Some(Controls {
            route,
            algorithm,
            rows,
            key_label,
            key,
            save_key,
            status,
            apply,
        });
        self.update_rows();
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

fn rect(x: f64, y: f64, width: f64, height: f64) -> NSRect {
    NSRect::new(NSPoint::new(x, y), NSSize::new(width, height))
}

/// Puts the listed models that match `query` in the row's list, and says how
/// many there are.
fn fill(ui: &RoleRow, list: &ModelList, query: &str) -> String {
    let query = query.trim();
    let listed = &list.models;
    // A complete ID shows the whole list, so another model is one click away.
    let shown: Vec<&str> = if listed.iter().any(|model| model == query) {
        listed.iter().map(String::as_str).collect()
    } else {
        models::matching(listed, query)
    };
    for model in &shown {
        // SAFETY: the combo box stores NSString values.
        unsafe { ui.model.addItemWithObjectValue(&NSString::from_str(model)) };
    }
    if query.is_empty() || shown.len() == listed.len() {
        format!(
            "{} models, fetched {}. Type to filter.",
            listed.len(),
            age(list.fetched_at)
        )
    } else if shown.is_empty() {
        "No listed model matches. Apply uses the ID as typed.".to_string()
    } else {
        format!("{} of {} models match.", shown.len(), listed.len())
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

/// Describes one loaded list for the result area.
fn outcome(loaded: &Loaded) -> String {
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
}
