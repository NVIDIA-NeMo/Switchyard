// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The app window shares the tray event loop and loads routes and usage on workers.

use crate::accounts;
use crate::config::Config;
use crate::harness::{self, HARNESSES, Harness};
use crate::history::{self, History};
use crate::picker::rect;
use crate::server_config::ServerConfig;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{
    NSApplication, NSApplicationDelegate, NSBackingStoreType, NSButton, NSControl, NSFont,
    NSPopUpButton, NSScrollView, NSTextField, NSTextView, NSView, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{MainThreadMarker, NSObject, NSObjectProtocol, NSSize, NSString};
use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

const USAGE: isize = 1;
const INSTALL: isize = 2;
const REFRESH: isize = 3;
const SAVE: isize = 4;
const RESTORE: isize = 5;
const ROUTE: isize = 6;
const TOOL: isize = 7;
const SESSION: isize = 8;
const LAUNCH: isize = 9;
const UPDATE: isize = 10;
const ACCOUNT: isize = 11;
const ADD_ACCOUNT: isize = 12;
const REOPEN: isize = 13;

define_class!(
    // SAFETY: NSObject has no subclassing requirements. The target only queues tags.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "SwitchyardAppTarget"]
    #[ivars = RefCell<Vec<isize>>]
    struct Target;
    impl Target {
        // SAFETY: AppKit sends the control that invoked this action.
        #[unsafe(method(changed:))]
        fn changed(&self, sender: &NSControl) { self.ivars().borrow_mut().push(sender.tag()); }
    }
    unsafe impl NSObjectProtocol for Target {}
    unsafe impl NSApplicationDelegate for Target {
        #[unsafe(method(applicationShouldHandleReopen:hasVisibleWindows:))]
        fn reopen(&self, _application: &NSApplication, _visible: bool) -> bool {
            self.ivars().borrow_mut().push(REOPEN);
            true
        }
    }
);

enum Finished {
    Refresh(Snapshot),
    Status(Result<String, String>),
}

struct Snapshot {
    routes: Result<Vec<crate::server_config::Route>, String>,
    history: Result<History, String>,
}

fn load(settings: &Config) -> Snapshot {
    Snapshot {
        routes: std::fs::read_to_string(&settings.config_file)
            .map_err(|e| e.to_string())
            .and_then(|text| ServerConfig::parse(&text))
            .map(|config| config.routes()),
        history: history::load(&settings.routing_log),
    }
}

pub struct Dashboard {
    mtm: MainThreadMarker,
    window: Retained<NSWindow>,
    // Dashboard retains the delegate because NSApplication holds a weak reference.
    target: Retained<Target>,
    text: Retained<NSTextView>,
    tool: Retained<NSPopUpButton>,
    route: Retained<NSPopUpButton>,
    session: Retained<NSPopUpButton>,
    project: Retained<NSTextField>,
    account: Retained<NSPopUpButton>,
    account_name: Retained<NSTextField>,
    add_account: Retained<NSButton>,
    accounts: Vec<(String, PathBuf)>,
    save: Retained<NSButton>,
    restore: Retained<NSButton>,
    launch: Retained<NSButton>,
    settings: Config,
    history: History,
    sessions: Vec<String>,
    routes: Vec<crate::server_config::Route>,
    install: bool,
    busy: bool,
    refresh_pending: bool,
    route_error: Option<String>,
    usage_error: Option<String>,
    status: Option<String>,
    sender: Sender<Finished>,
    receiver: Receiver<Finished>,
}

impl Dashboard {
    pub fn new(mtm: MainThreadMarker, settings: &Config) -> Self {
        // SAFETY: The app owns this window and keeps it after a close action.
        let window = unsafe {
            let w = NSWindow::initWithContentRect_styleMask_backing_defer(
                NSWindow::alloc(mtm),
                rect(0., 0., 940., 720.),
                NSWindowStyleMask::Titled
                    | NSWindowStyleMask::Closable
                    | NSWindowStyleMask::Miniaturizable,
                NSBackingStoreType::Buffered,
                false,
            );
            w.setReleasedWhenClosed(false);
            w
        };
        window.setTitle(&NSString::from_str("Switchyard"));
        window.center();
        let allocated = Target::alloc(mtm).set_ivars(RefCell::new(Vec::new()));
        // SAFETY: NSObject init has this signature.
        let target: Retained<Target> = unsafe { msg_send![super(allocated), init] };
        NSApplication::sharedApplication(mtm).setDelegate(Some(ProtocolObject::from_ref(&*target)));
        let content = NSView::initWithFrame(NSView::alloc(mtm), rect(0., 0., 940., 720.));
        for (title, tag, x, width) in [
            ("Usage", USAGE, 20., 100.),
            ("Install…", INSTALL, 130., 110.),
            ("Refresh", REFRESH, 250., 100.),
            ("Update from source…", UPDATE, 690., 230.),
        ] {
            content.addSubview(&button(mtm, &target, title, tag, rect(x, 666., width, 32.)));
        }
        let tool = popup(mtm, &target, TOOL, rect(20., 619., 240., 30.));
        for (_, name) in HARNESSES {
            tool.addItemWithTitle(&NSString::from_str(name));
        }
        let route = popup(mtm, &target, ROUTE, rect(280., 619., 310., 30.));
        let session = popup(mtm, &target, SESSION, rect(20., 619., 900., 30.));
        let save = button(
            mtm,
            &target,
            "Install / refresh",
            SAVE,
            rect(600., 619., 150., 30.),
        );
        let restore = button(
            mtm,
            &target,
            "Restore original",
            RESTORE,
            rect(760., 619., 160., 30.),
        );
        let project =
            NSTextField::initWithFrame(NSTextField::alloc(mtm), rect(20., 573., 680., 30.));
        project.setPlaceholderString(Some(&NSString::from_str(
            "Git project path for a separate worktree session",
        )));
        let launch = button(
            mtm,
            &target,
            "New worktree session",
            LAUNCH,
            rect(710., 573., 210., 30.),
        );
        let account = popup(mtm, &target, ACCOUNT, rect(20., 530., 280., 30.));
        let account_name =
            NSTextField::initWithFrame(NSTextField::alloc(mtm), rect(320., 530., 370., 30.));
        account_name.setPlaceholderString(Some(&NSString::from_str("Name for a new account")));
        let add_account = button(
            mtm,
            &target,
            "Add account…",
            ADD_ACCOUNT,
            rect(710., 530., 210., 30.),
        );
        for view in [
            &*account as &NSView,
            &*account_name,
            &*add_account,
            &*tool as &NSView,
            &*route,
            &*session,
            &*save,
            &*restore,
            &*project,
            &*launch,
        ] {
            content.addSubview(view);
        }
        let scroll =
            NSScrollView::initWithFrame(NSScrollView::alloc(mtm), rect(20., 20., 900., 495.));
        scroll.setHasVerticalScroller(true);
        scroll.setHasHorizontalScroller(true);
        let text = NSTextView::initWithFrame(NSTextView::alloc(mtm), rect(0., 0., 880., 495.));
        text.setEditable(false);
        text.setSelectable(true);
        text.setFont(Some(&NSFont::monospacedSystemFontOfSize_weight(13., 0.)));
        text.setTextContainerInset(NSSize::new(12., 12.));
        scroll.setDocumentView(Some(&text));
        content.addSubview(&scroll);
        window.setContentView(Some(&content));
        let (sender, receiver) = channel();
        let mut app = Self {
            mtm,
            window,
            target,
            text,
            tool,
            route,
            session,
            project,
            account,
            account_name,
            add_account,
            accounts: Vec::new(),
            save,
            restore,
            launch,
            settings: settings.clone(),
            history: History::default(),
            sessions: Vec::new(),
            routes: Vec::new(),
            install: false,
            busy: false,
            refresh_pending: false,
            route_error: None,
            usage_error: None,
            status: None,
            sender,
            receiver,
        };
        app.mode(false);
        app
    }

    pub fn is_busy(&self) -> bool {
        self.busy
    }
    pub fn set_settings(&mut self, settings: &Config) {
        self.settings = settings.clone();
        self.refresh();
    }
    pub fn show(&mut self, install: bool) {
        self.mode(install);
        #[allow(deprecated)]
        NSApplication::sharedApplication(self.mtm).activateIgnoringOtherApps(true);
        self.window.makeKeyAndOrderFront(None);
        self.refresh();
    }

    fn mode(&mut self, install: bool) {
        self.install = install;
        for view in [
            &*self.tool as &NSView,
            &*self.route,
            &*self.save,
            &*self.restore,
            &*self.project,
            &*self.launch,
            &*self.account,
            &*self.account_name,
            &*self.add_account,
        ] {
            view.setHidden(!install);
        }
        self.session.setHidden(install);
        if install {
            self.refresh_accounts();
            self.inspect();
        } else {
            self.render_history();
        }
    }

    fn tool(&self) -> Harness {
        HARNESSES
            .get(self.tool.indexOfSelectedItem() as usize)
            .map(|(tool, _)| *tool)
            .unwrap_or(Harness::CodexCli)
    }

    fn selected_account(&self) -> Option<&std::path::Path> {
        let index = self.account.indexOfSelectedItem();
        if index > 0 {
            self.accounts
                .get(index as usize - 1)
                .map(|(_, p)| p.as_path())
        } else {
            None
        }
    }
    fn refresh_accounts(&mut self) {
        let selected = self.selected_account().map(std::path::Path::to_path_buf);
        self.accounts = accounts::list(self.tool());
        self.account.removeAllItems();
        self.account
            .addItemWithTitle(&NSString::from_str("Default coding-tool login"));
        for (name, _) in &self.accounts {
            self.account.addItemWithTitle(&NSString::from_str(name));
        }
        if let Some(index) = self
            .accounts
            .iter()
            .position(|(_, p)| Some(p) == selected.as_ref())
        {
            self.account.selectItemAtIndex(index as isize + 1);
        }
        let supported = matches!(self.tool(), Harness::CodexCli | Harness::Claude);
        self.account.setEnabled(supported && !self.busy);
        self.add_account.setEnabled(supported && !self.busy);
        self.account_name.setEnabled(supported && !self.busy);
    }
    fn inspect(&self) {
        if let Some(error) = &self.route_error {
            self.text.setString(&NSString::from_str(error));
            return;
        }
        if let Some(status) = &self.status {
            self.text.setString(&NSString::from_str(status));
            return;
        }
        let tool = self.tool();
        let found = harness::binary(tool).map(|p| format!("Installed: {}",p.display())).unwrap_or_else(|| "Executable not found in the usual install directories. Install the coding tool first if needed.".into());
        let info = harness::inspect(tool, &accounts::config_paths(tool, self.selected_account()))
            .unwrap_or_else(|e| format!("Could not read settings: {e}"));
        self.text.setString(&NSString::from_str(&format!("{found}\n{info}\n\nSelect a public Switchyard route, then click Install / refresh.\nCodex CLI gets a standalone sy profile. Codex app changes the base user config.\nClaude and Pi get user settings. Existing settings are backed up.\nRestart the coding tool to apply defaults. Existing sessions keep their login. Select a named account to change where settings are saved and which login new sessions use. Add account… opens the coding tool’s own login in Terminal.\n\nA subscription route uses the coding tool's own saved login.\nA route with server API credentials works across tools.\nSwitchyard records observed token usage; provider quota and billing remain with the provider.\n\nNew worktree session creates an independent checkout from the project's HEAD\nand opens the selected tool in Terminal. Uncommitted changes stay in the original checkout.\nUse git worktree list and git worktree remove to manage these checkouts.\nProject settings and shell variables can override user defaults.")));
    }

    fn refresh(&mut self) {
        if self.busy {
            self.refresh_pending = true;
            return;
        }
        self.status = None;
        let settings = self.settings.clone();
        let tx = self.sender.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Finished::Refresh(load(&settings)));
        });
        self.set_busy(true);
    }

    fn apply_refresh(&mut self, snapshot: Snapshot) {
        let previous = self
            .routes
            .get(self.route.indexOfSelectedItem() as usize)
            .map(|r| r.id.clone());
        self.route_error = None;
        match snapshot.routes {
            Ok(routes) => {
                self.routes = routes;
                self.route.removeAllItems();
                for route in &self.routes {
                    self.route.addItemWithTitle(&NSString::from_str(&route.id));
                }
                if let Some(index) = self
                    .routes
                    .iter()
                    .position(|r| Some(&r.id) == previous.as_ref())
                {
                    self.route.selectItemAtIndex(index as isize);
                }
            }
            Err(e) => {
                self.routes.clear();
                self.route.removeAllItems();
                self.route_error = Some(format!("Could not read routes: {e}"));
            }
        }
        self.usage_error = None;
        match snapshot.history {
            Ok(history) => {
                let selected = self
                    .sessions
                    .get(self.session.indexOfSelectedItem().saturating_sub(1) as usize)
                    .cloned();
                self.history = history;
                self.sessions = self.history.sessions();
                self.session.removeAllItems();
                self.session.addItemWithTitle(&NSString::from_str(
                    "All sessions (includes calls without a session ID)",
                ));
                for session in &self.sessions {
                    self.session.addItemWithTitle(&NSString::from_str(session));
                }
                if let Some(index) = self
                    .sessions
                    .iter()
                    .position(|s| Some(s) == selected.as_ref())
                {
                    self.session.selectItemAtIndex(index as isize + 1);
                }
            }
            Err(error) => self.usage_error = Some(format!("Could not read usage: {error}")),
        }
        self.mode(self.install);
    }

    fn render_history(&self) {
        if let Some(error) = &self.usage_error {
            self.text.setString(&NSString::from_str(error));
            return;
        }
        let index = self.session.indexOfSelectedItem();
        let selected = if index > 0 {
            self.sessions.get(index as usize - 1).map(String::as_str)
        } else {
            None
        };
        self.text
            .setString(&NSString::from_str(&self.history.display(selected)));
    }

    fn set_busy(&mut self, busy: bool) {
        self.busy = busy;
        for button in [&self.save, &self.restore, &self.launch, &self.add_account] {
            button.setEnabled(!busy);
        }
        self.tool.setEnabled(!busy);
        self.route.setEnabled(!busy);
        self.refresh_accounts();
    }

    pub fn poll(&mut self) {
        let tags = std::mem::take(&mut *self.target.ivars().borrow_mut());
        for tag in tags {
            if self.busy && !matches!(tag, USAGE | INSTALL | SESSION | REOPEN) {
                continue;
            }
            match tag {
                USAGE => self.mode(false),
                INSTALL => {
                    self.mode(true);
                }
                REFRESH => self.refresh(),
                TOOL | ROUTE | ACCOUNT => {
                    if tag == TOOL {
                        self.refresh_accounts();
                    }
                    self.status = None;
                    self.inspect();
                }
                SESSION => self.render_history(),
                REOPEN => self.show(self.install),
                SAVE | RESTORE | LAUNCH | ADD_ACCOUNT => self.action(tag),
                UPDATE => {
                    let result = update();
                    self.text
                        .setString(&NSString::from_str(&result.unwrap_or_else(|e| e)));
                }
                _ => {}
            }
        }
        while let Ok(result) = self.receiver.try_recv() {
            self.set_busy(false);
            match result {
                Finished::Refresh(snapshot) => self.apply_refresh(snapshot),
                Finished::Status(result) => {
                    self.status = Some(result.unwrap_or_else(|e| e));
                    self.mode(self.install);
                }
            }
            if self.refresh_pending {
                self.refresh_pending = false;
                self.refresh();
            }
        }
    }

    fn action(&mut self, tag: isize) {
        let tool = self.tool();
        let account = self.selected_account().map(std::path::Path::to_path_buf);
        let files = accounts::config_paths(tool, account.as_deref());
        let account_name = self.account_name.stringValue().to_string();
        let tx = self.sender.clone();
        let settings = self.settings.clone();
        let route = self
            .routes
            .get(self.route.indexOfSelectedItem() as usize)
            .cloned();
        let project = PathBuf::from(self.project.stringValue().to_string());
        self.set_busy(true);
        self.text.setString(&NSString::from_str("Working…"));
        std::thread::spawn(move || {
            let result = (|| {
                if tag == ADD_ACCOUNT {
                    return accounts::add(tool, account_name.trim());
                }
                if let Some(account) = account.as_deref() {
                    accounts::validate(tool, account)?;
                }
                if tag == RESTORE {
                    return harness::restore(&files);
                }
                let route = route.ok_or_else(|| "Pick a route first.".to_string())?;
                let login = login_mode(tool, &settings, &route)?;
                if tag == LAUNCH {
                    crate::sessions::launch(
                        tool,
                        &project,
                        &route.id,
                        &settings.server_url,
                        login,
                        account.as_deref(),
                    )
                } else {
                    harness::install(tool, &files, &settings.server_url, &route.id, login)
                }
            })();
            let _ = tx.send(Finished::Status(result));
        });
    }
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

fn button(
    mtm: MainThreadMarker,
    target: &Target,
    title: &str,
    tag: isize,
    frame: objc2_foundation::NSRect,
) -> Retained<NSButton> {
    // SAFETY: Target implements changed: and lives as long as the controls.
    let b = unsafe {
        NSButton::buttonWithTitle_target_action(
            &NSString::from_str(title),
            Some(target),
            Some(sel!(changed:)),
            mtm,
        )
    };
    b.setTag(tag);
    b.setFrame(frame);
    b
}
fn popup(
    mtm: MainThreadMarker,
    target: &Target,
    tag: isize,
    frame: objc2_foundation::NSRect,
) -> Retained<NSPopUpButton> {
    let p = NSPopUpButton::initWithFrame_pullsDown(NSPopUpButton::alloc(mtm), frame, false);
    p.setTag(tag);
    // SAFETY: Target implements changed: and lives as long as the controls.
    unsafe {
        p.setTarget(Some(target));
        p.setAction(Some(sel!(changed:)));
    }
    p
}
fn update() -> Result<String, String> {
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let resource = executable
        .parent()
        .and_then(|p| p.parent())
        .ok_or("Could not locate the app bundle.")?
        .join("Resources/Update.command");
    if !resource.is_file() {
        return Err(
            "Install Switchyard.app with scripts/macos/install.sh before updating from the app."
                .into(),
        );
    }
    crate::server::command("open", &[&resource.display().to_string()])?;
    Ok("Terminal is rebuilding and reinstalling from the source checkout used for this install. The app restarts after a successful build. Your routing and prices stay saved.".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_load_does_not_depend_on_a_valid_server_config() {
        let dir = tempfile::tempdir().expect("directory");
        let log = dir.path().join("log");
        std::fs::write(
            &log,
            "{\"ts\":\"2026-10-08T12:00:00Z\",\"model\":\"actual\",\"session_id\":\"session\"}\n",
        )
        .expect("log");
        let settings = Config {
            config_file: dir.path().join("config"),
            routing_log: log,
            ..Config::default()
        };
        let snapshot = load(&settings);
        assert!(snapshot.routes.is_err());
        assert_eq!(
            snapshot.history.expect("history").sessions(),
            vec!["session"]
        );
        std::fs::write(&settings.config_file, "not valid TOML").expect("config");
        let snapshot = load(&settings);
        assert!(snapshot.routes.is_err());
        assert!(
            snapshot
                .history
                .expect("history")
                .display(None)
                .contains("actual")
        );
    }
    #[test]
    fn installing_rejects_stale_ids_and_insecure_subscription_endpoints() {
        let dir = tempfile::tempdir().expect("directory");
        let path = dir.path().join("server.toml");
        let settings = Config {
            config_file: path.clone(),
            ..Config::default()
        };
        for endpoint in [
            "https://chatgpt.com/backend-api/codex",
            "http://chatgpt.com/backend-api/codex",
        ] {
            let text = format!(
                "[llm_clients.upstream]\nbase_url = '{endpoint}'\nformat = 'openai_responses'\nforward_auth = true\n[targets.model]\nid = 'actual'\nllm_client = 'upstream'\n[routes.public]\nid = 'route'\ntype = 'passthrough'\ntarget = 'model'\n"
            );
            std::fs::write(&path, &text).expect("config");
            let route = ServerConfig::parse(&text)
                .expect("parse")
                .routes()
                .remove(0);
            assert_eq!(
                login_mode(Harness::CodexCli, &settings, &route).is_ok(),
                endpoint.starts_with("https:")
            );
            let mut stale = route;
            stale.id = "old-public-id".into();
            assert!(login_mode(Harness::CodexCli, &settings, &stale).is_err());
            assert_eq!(std::fs::read_to_string(&path).expect("unchanged"), text);
        }
    }
}
