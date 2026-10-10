// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ratatui renders snapshots and submits typed operations to the shared controller.

use crate::{
    controller::{Action, Controller},
    harness::Harness,
    server_config::Choice,
};
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
const PAGES: &[&str] = &[
    "Overview",
    "Routes",
    "Coding tools",
    "Usage",
    "Sessions",
    "Settings",
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
    Algorithm,
    Apply(String, Option<String>),
    Project,
    Account,
    Models,
    Key,
    Filter,
    RouteFilter,
    SettingsFile,
    Remove,
    Restore,
    Update,
    ConfirmApply(Box<Action>),
    ConfirmLaunch(Box<Action>),
}
#[derive(Clone)]
struct View {
    snapshot: Value,
    page: usize,
    route: usize,
    tool: usize,
    account: usize,
    session: usize,
    filter: String,
    route_filter: String,
    settings_file: String,
    preview: Option<Value>,
    scroll: u16,
    message: String,
    form: Option<Form>,
    details: String,
    showing_details: bool,
    selected_session: Option<String>,
    missing_account: Option<String>,
}
impl View {
    fn routes(&self) -> &[Value] {
        self.snapshot["routes"]
            .as_array()
            .map_or(&[], Vec::as_slice)
    }
    fn tools(&self) -> &[Value] {
        self.snapshot["tools"].as_array().map_or(&[], Vec::as_slice)
    }
    fn matches_route(&self, route: &Value) -> bool {
        let kind = string(route, "kind");
        let title = crate::server_config::ALGORITHMS
            .iter()
            .find(|algorithm| algorithm.kind == kind)
            .map_or("", |algorithm| algorithm.title);
        format!("{} {title}", string(route, "label"))
            .to_lowercase()
            .contains(&self.route_filter.to_lowercase())
    }
    fn route(&self) -> Result<&Value, String> {
        if self.page == 1
            && self
                .routes()
                .get(self.route)
                .is_some_and(|route| !self.matches_route(route))
        {
            return Err("No route matches the search. Clear or change the search first.".into());
        }
        self.routes()
            .get(self.route)
            .ok_or_else(|| "Choose a route first.".into())
    }
    fn tool(&self) -> Harness {
        serde_json::from_value(self.tools()[self.tool]["tool"].clone()).unwrap_or(Harness::CodexCli)
    }
    fn account(&self) -> Option<String> {
        if self.missing_account.is_some() {
            return self.missing_account.clone();
        }
        self.tools().get(self.tool)?["accounts"]
            .as_array()
            .and_then(|a| self.account.checked_sub(1).and_then(|i| a.get(i)))
            .and_then(Value::as_str)
            .map(str::to_string)
    }
    fn dispatch(&mut self, controller: &mut Controller, action: Action) {
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
        let route = self.routes().get(self.route).map(|r| string(r, "key"));
        let tool = self.tools().get(self.tool).map(|t| string(t, "tool"));
        let account = self.account();
        self.snapshot = controller.snapshot()?;
        if let Some(route) = route {
            self.route = self
                .routes()
                .iter()
                .position(|r| string(r, "key") == route)
                .unwrap_or(usize::MAX);
        }
        if let Some(tool) = tool {
            self.tool = self
                .tools()
                .iter()
                .position(|t| string(t, "tool") == tool)
                .unwrap_or(0);
        }
        if let Some(account) = account {
            self.account = self
                .tools()
                .get(self.tool)
                .and_then(|t| t["accounts"].as_array())
                .and_then(|a| a.iter().position(|v| v.as_str() == Some(&account)))
                .map_or(usize::MAX, |i| i + 1);
            self.missing_account = (self.account == usize::MAX).then_some(account);
        }
        if let Some(session) = &self.selected_session {
            self.session = self.snapshot["sessions"]
                .as_array()
                .and_then(|a| a.iter().position(|v| v.as_str() == Some(session)))
                .map_or(usize::MAX, |i| i + 1);
        }
        Ok(())
    }
    fn preview_install(&mut self, controller: &mut Controller) -> Result<(), String> {
        self.preview = None;
        let route = self.route()?;
        let reply = controller.dispatch(Action::PreviewInstall {
            tool: self.tool(),
            account: self.account(),
            settings_file: self.custom_file(),
            route: string(route, "key"),
            id: string(route, "id"),
        })?;
        self.preview = Some(reply.data);
        self.message =
            "Review the diff. Press i to install; changing the selection clears the preview."
                .into();
        Ok(())
    }
    fn custom_file(&self) -> Option<String> {
        (!self.settings_file.is_empty()).then(|| self.settings_file.clone())
    }
    fn move_route(&mut self, forward: bool) {
        let visible: Vec<_> = self
            .routes()
            .iter()
            .enumerate()
            .filter(|(_, r)| self.page != 1 || self.matches_route(r))
            .map(|(i, _)| i)
            .collect();
        if visible.is_empty() {
            return;
        }
        let index = visible.iter().position(|i| *i == self.route).unwrap_or(0);
        let next = if forward {
            (index + 1) % visible.len()
        } else {
            index.saturating_sub(1)
        };
        self.route = visible[next];
        self.preview = None;
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
        let value = |i: usize| form.fields[i].1.trim().to_string();
        let action = match form.kind {
            FormKind::Algorithm => {
                let route = self.route()?;
                let algorithm = value(0);
                let reply = controller.dispatch(Action::Editor {
                    generation: self.snapshot["generation"].as_u64().unwrap_or_default(),
                    route: string(route, "key"),
                    algorithm: algorithm.clone(),
                })?;
                let roles = reply.data["roles"].as_array().ok_or("No roles returned.")?;
                let choices: Vec<Choice> =
                    serde_json::from_value(route["choices"].clone()).map_err(|e| e.to_string())?;
                let mut fields = Vec::new();
                for (i, role) in roles.iter().enumerate() {
                    // A role keeps its model by tier when the new algorithm changes role order.
                    let matching = route["tiers"].as_array().and_then(|tiers| {
                        role["tier"]
                            .as_str()
                            .and_then(|tier| tiers.iter().position(|t| t.as_str() == Some(tier)))
                    });
                    let choice = matching
                        .and_then(|index| choices.get(index))
                        .or_else(|| {
                            (algorithm == string(route, "kind"))
                                .then(|| choices.get(i))
                                .flatten()
                        })
                        .or_else(|| choices.first())
                        .cloned()
                        .unwrap_or_default();
                    fields.push((format!("{} endpoint", string(role, "label")), choice.client));
                    fields.push((
                        if string(role, "label") == "Model" {
                            "Model".into()
                        } else {
                            format!("{} model", string(role, "label"))
                        },
                        choice.model,
                    ));
                }
                self.form(
                    "Route models · Enter applies",
                    FormKind::Apply(algorithm, route["revision"].as_str().map(str::to_owned)),
                    fields,
                );
                return Ok(());
            }
            FormKind::Apply(algorithm, revision) => {
                let choices = form
                    .fields
                    .chunks(2)
                    .map(|f| Choice {
                        client: f[0].1.trim().into(),
                        model: f[1].1.trim().into(),
                    })
                    .collect();
                let action = Action::Apply {
                    generation: self.snapshot["generation"].as_u64().unwrap_or_default(),
                    route: string(self.route()?, "key"),
                    algorithm,
                    choices,
                    revision: revision.clone(),
                };
                let preview_action = match &action {
                    Action::Apply {
                        generation,
                        route,
                        algorithm,
                        choices,
                        ..
                    } => Action::PreviewRoute {
                        generation: *generation,
                        route: route.clone(),
                        algorithm: algorithm.clone(),
                        choices: choices.clone(),
                        revision,
                    },
                    _ => unreachable!(),
                };
                let reply = controller.dispatch(preview_action)?;
                self.preview = Some(reply.data);
                self.form(
                    "Review route changes · type SAVE to apply",
                    FormKind::ConfirmApply(Box::new(action)),
                    vec![("Confirmation".into(), String::new())],
                );
                return Ok(());
            }
            FormKind::ConfirmApply(action) => {
                if value(0) != "SAVE" {
                    return Err("Type SAVE to apply the reviewed changes.".into());
                }
                *action
            }
            FormKind::ConfirmLaunch(action) => {
                if value(0) != "LAUNCH" {
                    return Err("Type LAUNCH to open the reviewed session.".into());
                }
                *action
            }
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
            FormKind::Project => {
                let r = self.route()?;
                let reply = controller.dispatch(Action::PreviewLaunch {
                    tool: self.tool(),
                    account: self.account(),
                    route: string(r, "key"),
                    id: string(r, "id"),
                    project: value(0),
                })?;
                let action = Action::Launch {
                    preview_token: reply.data["preview_token"].as_str().map(str::to_owned),
                    tool: self.tool(),
                    account: self.account(),
                    route: string(r, "key"),
                    id: string(r, "id"),
                    project: value(0),
                };
                self.preview = Some(reply.data);
                self.form(
                    "Review committed HEAD session · type LAUNCH",
                    FormKind::ConfirmLaunch(Box::new(action)),
                    vec![("Confirmation".into(), String::new())],
                );
                return Ok(());
            }
            FormKind::Account => Action::AddAccount {
                tool: self.tool(),
                name: value(0),
            },
            FormKind::Models | FormKind::Key => {
                let key = if matches!(form.kind, FormKind::Key) {
                    Some(value(1))
                } else {
                    None
                };
                let reply = controller.dispatch(Action::Models {
                    client: value(0),
                    refresh: true,
                    key,
                })?;
                self.message = reply.message;
                self.details = reply.data["models"]
                    .as_array()
                    .map(|m| {
                        m.iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                self.showing_details = true;
                self.scroll = 0;
                return Ok(());
            }
            FormKind::RouteFilter => {
                self.route_filter = value(0);
                if !self
                    .routes()
                    .get(self.route)
                    .is_some_and(|r| self.matches_route(r))
                {
                    self.route = self
                        .routes()
                        .iter()
                        .position(|r| self.matches_route(r))
                        .unwrap_or(0);
                }
                self.preview = None;
                self.scroll = 0;
                return Ok(());
            }
            FormKind::SettingsFile => {
                self.settings_file = value(0);
                self.account = 0;
                self.preview = None;
                self.preview_install(controller)?;
                return Ok(());
            }
            FormKind::Restore => {
                if value(0) != "RESTORE" {
                    return Err("Confirmation did not match. Nothing was restored.".into());
                }
                Action::Restore {
                    tool: self.tool(),
                    account: self.account(),
                    settings_file: self.custom_file(),
                    preview_token: self
                        .preview
                        .as_ref()
                        .and_then(|p| p["preview_token"].as_str())
                        .map(str::to_owned),
                }
            }
            FormKind::Remove => {
                let route = self.route()?;
                if value(0) != string(route, "id") {
                    return Err("Route name did not match. Nothing was deleted.".into());
                }
                Action::Remove {
                    generation: self.snapshot["generation"].as_u64().unwrap_or_default(),
                    route: string(route, "key"),
                    id: string(route, "id"),
                    revision: self
                        .preview
                        .as_ref()
                        .and_then(|p| p["revision"].as_str())
                        .map(str::to_owned),
                }
            }
            FormKind::Filter => {
                self.filter = value(0);
                self.scroll = 0;
                return Ok(());
            }
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
        self.refresh(controller)?;
        Ok(())
    }
    fn body(&self) -> String {
        let errors = self.snapshot["errors"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
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
        match self.page {
            0 if self.snapshot["data_state"]["usage"]["state"] == "unavailable" => {
                format!("CONNECTION\n{} · {}", if self.snapshot["metrics"]["running"] == true {"Connected"} else {"Not responding"}, string(&self.snapshot["metrics"], "server_url"))
            }
            3 if self.snapshot["data_state"]["history"]["state"] == "unavailable" => String::new(),
            0=>{
                let mut period="";
                let details=self.snapshot["summary"].as_array().map(|rows|rows.iter().filter_map(Value::as_str).filter_map(|line| {
                    if line.starts_with("Today —") {period="Today";}
                    if line.starts_with("This week —") {period="Past 7 days";}
                    if line.contains("Saved") {Some(format!("{period}: {}",line.trim()))}
                    else if line.contains("prices")||line.contains("Savings hidden")||line.trim().ends_with('%') {Some(line.trim().into())}
                    else {None}
                }).collect::<Vec<String>>().join("\n")).unwrap_or_default();
                format!("CONNECTION\n{} · {}\n\nTODAY\n{} model calls · {} tokens\n\nPAST 7 DAYS\n{} model calls · {} tokens\n\nUSAGE DETAILS\n{}",if self.snapshot["metrics"]["running"]==true {"Connected"} else {"Not responding"},string(&self.snapshot["metrics"],"server_url"),self.snapshot["metrics"]["today"]["requests"],self.snapshot["metrics"]["today"]["tokens"],self.snapshot["metrics"]["week"]["requests"],self.snapshot["metrics"]["week"]["tokens"],details)
            },
            1=>{
                let visible:Vec<_>=self.routes().iter().enumerate().filter(|(_,r)|self.matches_route(r)).collect();
                let index=visible.iter().position(|(i,_)|*i==self.route).unwrap_or(0);
                let mut rows=vec![format!("/ Search routes   e Edit   d Delete route   m Refresh models   k Save endpoint key\n{} of {} routes · Search: {}\n",visible.len(),self.routes().len(),self.route_filter)];
                for(i,r) in visible.iter().skip(index.saturating_sub(4)).take(10){rows.push(format!("{} {}",if *i==self.route {">"}else{" "},string(r,"label")));}
                if visible.is_empty(){ rows.push("No routes match your search.".into()); }
                if let Ok(route)=self.route(){rows.push(format!("\nSELECTED ROUTE\n{}",string(route,"label")));}
                rows.join("\n")
            }
            2|4=>{
                let mut body=format!("t Next tool   r Next route   a Next account\n{}\n\nTool: {} ({})\nAccount: {}\nRoute: {}\nSettings: {}\n\nDetected user settings:\n{}",if self.page==2 {"p Preview diff   i Review / install   u Restore backup   x Custom settings file\nn Sign in to another subscription account"}else{"l Launch worktree session   n Sign in to another subscription account"},string(&self.tools()[self.tool],"label"),if self.tools()[self.tool]["available"]==true {"Detected"}else{"Binary not found"},self.account().unwrap_or_else(||"Existing user settings".into()),self.route().map(|r|string(r,"label")).unwrap_or_default(),if self.settings_file.is_empty(){"Detected location"}else{&self.settings_file},string(&self.tools()[self.tool],"status"));
                if let Some(preview)=&self.preview {
                    body.push_str(&format!("\n\nSELECTED SETTINGS\n{}",string(preview,"current")));
                    body.push_str("\n\nSETTINGS DIFF · − current / + proposed\n");
                    if let Some(error)=preview["error"].as_str(){body.push_str(error);}
                    if let Some(changes)=preview["changes"].as_array(){for change in changes{body.push_str(&format!("\n{} · {}\n− {}\n+ {}\n",string(change,"file"),string(change,"key"),string(change,"before"),string(change,"after")));}}
                    body.push_str(&format!("\n{}",string(preview,"authentication")));
                }
                body
            }
            3=>{
                let selected=self.selected_session.as_deref();
                let mut lines=vec![format!("s Next session   f Filter turn/model/route\nSession: {}{}   Filter: {}\n{} recent calls{}; {} unreadable records skipped.\nEach row is one model call; a turn can contain several calls.\n",selected.unwrap_or("All sessions"),if self.session==usize::MAX {" (outside recent history)"} else {""},self.filter,self.snapshot["entries"].as_array().map_or(0,Vec::len),if self.snapshot["limited"]==true{" (8 MiB / 5,000 record limit)"}else{""},self.snapshot["skipped"])];
                if let Some(entries)=self.snapshot["entries"].as_array(){for e in entries.iter().rev(){
                    if selected.is_some_and(|s|e["session_id"].as_str()!=Some(s)){continue;}
                    if !format!("{} {} {}",string(e,"model"),string(e,"turn_id"),string(e,"route_id")).to_lowercase().contains(&self.filter.to_lowercase()){continue;}
                    lines.push(format!("{}  {}{}\nSession {}  Turn {}  Route {}\nUncached input {}  Cached input {}  Output {}\n",string(e,"ts"),string(e,"model"),if e["tier"]=="classifier"{" (routing overhead)"}else{" (answer call)"},e["session_id"].as_str().filter(|s|!s.is_empty()).unwrap_or("Not recorded"),e["turn_id"].as_str().filter(|s|!s.is_empty()).unwrap_or("Not recorded"),e["route_id"].as_str().filter(|s|!s.is_empty()).unwrap_or("Not recorded"),e["prompt_tokens"].as_u64().unwrap_or_default().saturating_sub(e["cached_tokens"].as_u64().unwrap_or_default()),e["cached_tokens"],e["completion_tokens"]));
                }}lines.join("\n")
            }
            _=>"o Open server config\na Open app settings\nc Check config\nb Restart server\nu Update from source\n\nUpdate uses the checkout that installed Switchyard.app.\nNative account limits still apply. Switchyard does not rotate login tokens.".into(),
        }
    }
    fn key(&mut self, controller: &mut Controller, code: KeyCode) -> Result<(), String> {
        match code {
            KeyCode::Tab => {
                self.page = (self.page + 1) % PAGES.len();
                self.scroll = 0;
            }
            KeyCode::BackTab => {
                self.page = (self.page + PAGES.len() - 1) % PAGES.len();
                self.scroll = 0;
            }
            KeyCode::Down | KeyCode::Char('r') => self.move_route(true),
            KeyCode::Up => self.move_route(false),
            KeyCode::PageDown => self.scroll = self.scroll.saturating_add(10),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::F(5) => self.refresh(controller)?,
            KeyCode::Char('t') => {
                self.tool = (self.tool + 1) % self.tools().len();
                self.account = 0;
                self.missing_account = None;
                self.settings_file.clear();
                self.preview = None;
            }
            KeyCode::Char('a') if self.page == 2 || self.page == 4 => {
                self.account = self.account.saturating_add(1)
                    % (self.tools()[self.tool]["accounts"]
                        .as_array()
                        .map_or(0, Vec::len)
                        + 1);
                self.missing_account = None;
                self.settings_file.clear();
                self.preview = None;
            }
            KeyCode::Char('e') if self.page == 1 => {
                let kind = string(self.route()?, "kind");
                self.form(
                    "Routing algorithm",
                    FormKind::Algorithm,
                    vec![("Algorithm".into(), kind)],
                );
            }
            KeyCode::Char('m' | 'k') if self.page == 1 => {
                let mut fields = vec![(
                    "Endpoint name".into(),
                    self.snapshot["clients"][0]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .into(),
                )];
                if code == KeyCode::Char('k') {
                    fields.push(("API key".into(), String::new()));
                }
                self.form(
                    "Models",
                    if code == KeyCode::Char('k') {
                        FormKind::Key
                    } else {
                        FormKind::Models
                    },
                    fields,
                );
            }
            KeyCode::Char('p') if self.page == 2 => self.preview_install(controller)?,
            KeyCode::Char('x') if self.page == 2 => self.form(
                "Custom settings file; empty uses detected settings",
                FormKind::SettingsFile,
                vec![(
                    "Absolute file path (Pi: models.json)".into(),
                    self.settings_file.clone(),
                )],
            ),
            KeyCode::Char('/') if self.page == 1 => self.form(
                "Find a route",
                FormKind::RouteFilter,
                vec![(
                    "Name, method, endpoint, or model".into(),
                    self.route_filter.clone(),
                )],
            ),
            KeyCode::Char('d') if self.page == 1 => {
                let route = self.route()?;
                self.preview = Some(serde_json::json!({"revision":route["revision"]}));
                self.form(
                    &format!(
                        "Delete {}? Targets stay; config is backed up",
                        string(self.route()?, "id")
                    ),
                    FormKind::Remove,
                    vec![("Type the route name to delete".into(), String::new())],
                );
            }
            KeyCode::Char('i') if self.page == 2 => {
                if self.preview.is_none() {
                    self.preview_install(controller)?;
                } else if let Some(error) = self.preview.as_ref().and_then(|p| p["error"].as_str())
                {
                    return Err(error.into());
                } else {
                    let r = self.route()?;
                    self.dispatch(
                        controller,
                        Action::Install {
                            tool: self.tool(),
                            account: self.account(),
                            settings_file: self.custom_file(),
                            route: string(r, "key"),
                            id: string(r, "id"),
                            preview_token: self
                                .preview
                                .as_ref()
                                .and_then(|p| p["preview_token"].as_str())
                                .map(str::to_owned),
                        },
                    );
                    self.preview = None;
                }
            }
            KeyCode::Char('u') if self.page == 2 => {
                let reply = controller.dispatch(Action::PreviewRestore {
                    tool: self.tool(),
                    account: self.account(),
                    settings_file: self.custom_file(),
                })?;
                self.preview = Some(reply.data);
                self.form(
                    "Restore coding-tool settings? Switchyard routes are kept",
                    FormKind::Restore,
                    vec![("Type RESTORE to confirm".into(), String::new())],
                );
            }
            KeyCode::Char('n') if self.page == 2 || self.page == 4 => self.form(
                "Sign in to another subscription account",
                FormKind::Account,
                vec![("Account name".into(), String::new())],
            ),
            KeyCode::Char('l') if self.page == 4 => self.form(
                "Launch worktree session",
                FormKind::Project,
                vec![("Git project absolute path".into(), String::new())],
            ),
            KeyCode::Char('s') if self.page == 3 => {
                self.session = self.session.saturating_add(1)
                    % (self.snapshot["sessions"].as_array().map_or(0, Vec::len) + 1);
                self.selected_session = self
                    .session
                    .checked_sub(1)
                    .and_then(|i| self.snapshot["sessions"].as_array()?.get(i))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                self.scroll = 0;
            }
            KeyCode::Char('f') if self.page == 3 => self.form(
                "Filter turn, model, or route",
                FormKind::Filter,
                vec![("Search".into(), self.filter.clone())],
            ),
            KeyCode::Char('o') if self.page == 5 => {
                self.dispatch(controller, Action::OpenConfig {})
            }
            KeyCode::Char('c') if self.page == 5 => {
                self.dispatch(controller, Action::CheckConfig {})
            }
            KeyCode::Char('a') if self.page == 5 => {
                self.dispatch(controller, Action::OpenSettings {})
            }
            KeyCode::Char('b') if self.page == 5 => self.dispatch(controller, Action::Restart {}),
            KeyCode::Char('u') if self.page == 5 => {
                let reply = controller.dispatch(Action::PreviewUpdate {})?;
                self.preview = Some(reply.data);
                self.form(
                    "Review source update · type UPDATE to start",
                    FormKind::Update,
                    vec![("Confirmation".into(), String::new())],
                );
            }
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
}
fn string(v: &Value, key: &str) -> String {
    v[key].as_str().unwrap_or_default().into()
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
pub fn run(mut controller: Controller) -> Result<(), String> {
    let (snapshot, page) = match controller.snapshot() {
        Ok(snapshot) => (snapshot, 0),
        Err(error) => (
            serde_json::json!({"generation":0,"errors":[error],"routes":[],"clients":[],"sessions":[],"history":[],"tools":crate::harness::HARNESSES.iter().map(|(tool,label)|serde_json::json!({"tool":tool,"label":label,"accounts":[],"status":"Fix app settings, then refresh."})).collect::<Vec<_>>(),"summary":[],"settings":{}}),
            5,
        ),
    };
    let mut view = View {
        snapshot,
        page,
        route: 0,
        tool: 0,
        account: 0,
        session: 0,
        filter: String::new(),
        route_filter: String::new(),
        settings_file: String::new(),
        preview: None,
        scroll: 0,
        message: String::new(),
        form: None,
        details: String::new(),
        showing_details: false,
        selected_session: None,
        missing_account: None,
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
            }
            terminal.draw(|frame| {
                let areas = Layout::vertical([
                    Constraint::Length(3),
                    Constraint::Min(5),
                    Constraint::Length(5),
                ]).split(frame.area());
                frame.render_widget(Tabs::new(PAGES.iter().copied()).select(view.page).highlight_style(Style::default().fg(Color::White).bg(Color::Rgb(40,56,76))).block(Block::bordered().title("Switchyard")),areas[0]);
                let mut body_area=areas[1];
                if view.page==0 && view.form.is_none() {
                    let split=Layout::vertical([Constraint::Min(5),Constraint::Length(6)]).split(body_area);
                    body_area=split[0];
                    let data=view.snapshot["activity"].as_array().map(|items|items.iter().filter_map(Value::as_u64).collect::<Vec<_>>()).unwrap_or_default();
                    frame.render_widget(Sparkline::default().data(&data).style(Style::default().fg(Color::LightBlue)).block(Block::bordered().title("Recent model calls · past 24h · hourly")),split[1]);
                }
                let (body, title) = if let Some(form) = &view.form {
                    let rows = form.fields.iter().enumerate().map(|(i, (label, value))| {
                        let marker = if i == form.index { ">" } else { " " };
                        let mut value = if matches!(form.kind, FormKind::Key) && i == 1 {
                            "•".repeat(value.chars().count())
                        } else { value.clone() };
                        if i==form.index {
                            let cursor=value.char_indices().nth(form.cursor).map_or(value.len(),|(index,_)|index);
                            value.insert(cursor,'▏');
                        }
                        format!("{marker} {label}: {value}")
                    }).collect::<Vec<_>>().join("\n");
                    let hints=match &form.kind {
                        FormKind::ConfirmApply(..) | FormKind::ConfirmLaunch(..) | FormKind::Update => view.preview.as_ref().map(|p| format!("\n\n{}", serde_json::to_string_pretty(p).unwrap_or_default())).unwrap_or_default(),
                        FormKind::Algorithm=>format!("\n\nAlgorithms: {}",crate::server_config::ALGORITHMS.iter().map(|a|format!("{} ({})",a.title,a.kind)).collect::<Vec<_>>().join(" · ")),
                        FormKind::Apply(..)=>format!("\n\nEndpoints: {}\nUse the endpoint name and a model ID. Unlisted model IDs are allowed.",view.snapshot["clients"].as_array().map(|a|a.iter().map(|c|string(c,"name")).collect::<Vec<_>>().join(", ")).unwrap_or_default()),
                        FormKind::Restore=>view.preview.as_ref().map(|p|format!("\n\n{}\n{}",string(p,"warning"),p["files"].as_array().map(|a|a.iter().map(|f|format!("{}: {}",string(f,"action"),string(f,"file"))).collect::<Vec<_>>().join("\n")).unwrap_or_default())).unwrap_or_default(),
                        _=>String::new(),
                    };
                    (format!("{rows}{hints}"), form.title.clone())
                } else { (view.body(), PAGES[view.page].into()) };
                frame.render_widget(
                    Paragraph::new(body)
                        .wrap(Wrap { trim: false })
                        .scroll((view.scroll, 0))
                        .block(Block::bordered().title(title)),
                    body_area,
                );
                frame.render_widget(
                    Paragraph::new(format!("Tab: section · PgUp/PgDn: scroll · F5: refresh · h: details · Esc: dismiss · q: quit\nForms: Tab: field · ↑/↓: choices · ←/→: cursor · Ctrl-U: clear · Enter: submit · Esc: cancel\n{}", if busy {if quit_pending {"Finishing the operation before quitting…"} else {"Working… Navigation is available; wait before starting another action."}} else {&view.message})),
                    areas[2],
                );
            }).map_err(|e|e.to_string())?;
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
                if busy {
                    match key.code {
                        KeyCode::Char('q') if view.form.is_none() => quit_pending = true,
                        KeyCode::Tab if view.form.is_none() => {
                            view.page = (view.page + 1) % PAGES.len();
                            view.scroll = 0;
                        }
                        KeyCode::BackTab if view.form.is_none() => {
                            view.page = (view.page + PAGES.len() - 1) % PAGES.len();
                            view.scroll = 0;
                        }
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
                        KeyCode::Up | KeyCode::Down => {
                            let options: Vec<String> = match &form.kind {
                                FormKind::Algorithm => crate::server_config::ALGORITHMS
                                    .iter()
                                    .map(|a| a.kind.to_string())
                                    .collect(),
                                FormKind::Apply(..) if form.index % 2 == 0 => {
                                    view.snapshot["clients"]
                                        .as_array()
                                        .map(|clients| {
                                            clients
                                                .iter()
                                                .filter_map(|c| {
                                                    c["name"].as_str().map(str::to_owned)
                                                })
                                                .collect()
                                        })
                                        .unwrap_or_default()
                                }
                                _ => Vec::new(),
                            };
                            if !options.is_empty() {
                                let index = options
                                    .iter()
                                    .position(|option| option == &form.fields[form.index].1)
                                    .unwrap_or(0);
                                let next = if key.code == KeyCode::Down {
                                    (index + 1) % options.len()
                                } else {
                                    (index + options.len() - 1) % options.len()
                                };
                                form.fields[form.index].1 = options[next].clone();
                                form.cursor = options[next].chars().count();
                            }
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
            if last.elapsed()
                > Duration::from_secs(view.snapshot["refresh_seconds"].as_u64().unwrap_or(30))
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
    use crate::config::Config;
    use serde_json::json;

    fn view() -> View {
        View {
            snapshot: json!({"routes":[{"key":"a","id":"a","label":"alpha"},{"key":"b","id":"b","label":"beta"}]}),
            page: 1,
            route: 0,
            tool: 0,
            account: 0,
            session: 0,
            filter: String::new(),
            route_filter: "alpha".into(),
            settings_file: String::new(),
            preview: Some(json!({"proposed":"old settings"})),
            scroll: 0,
            message: String::new(),
            form: None,
            details: String::new(),
            showing_details: false,
            selected_session: None,
            missing_account: None,
        }
    }

    #[test]
    fn route_search_preserves_matching_selection_and_selects_the_first_match() {
        let dir = tempfile::tempdir().expect("directory");
        let mut controller = Controller::new(Config::default(), dir.path().join("app.toml"));
        let mut view = view();
        view.route = 1;
        for (filter, expected) in [("", 1), ("beta", 1), ("alpha", 0), ("missing", 0)] {
            view.form(
                "Search",
                FormKind::RouteFilter,
                vec![("Search".into(), filter.into())],
            );
            view.submit(&mut controller).expect("search");
            assert_eq!(view.route, expected);
            assert!(view.preview.is_none());
        }
        assert!(view.route().is_err());
    }

    #[test]
    // A mismatched confirmation must fail before controller dispatch can create files.
    fn unconfirmed_restore_leaves_files_untouched() {
        let directory = tempfile::tempdir().expect("directory");
        let mut controller = Controller::new(Config::default(), directory.path().join("app.toml"));
        let mut view = view();
        view.form(
            "Restore",
            FormKind::Restore,
            vec![("Confirmation".into(), "no".into())],
        );
        assert_eq!(
            view.submit(&mut controller).expect_err("reject"),
            "Confirmation did not match. Nothing was restored."
        );
        assert_eq!(
            std::fs::read_dir(directory.path()).expect("files").count(),
            0
        );
    }

    #[test]
    fn refreshing_the_snapshot_keeps_the_install_preview_readable() {
        let directory = tempfile::tempdir().expect("directory");
        let settings = directory.path().join("app.toml");
        std::fs::write(
            &settings,
            format!(
                "config_file={:?}\nrouting_log={:?}\nserver_url='http://127.0.0.1:0'\n",
                directory.path().join("server.toml"),
                directory.path().join("history.jsonl")
            ),
        )
        .expect("settings");
        let mut controller = Controller::new(Config::load(&settings).expect("settings"), settings);
        let mut view = view();
        view.refresh(&mut controller).expect("refresh");
        assert_eq!(
            view.preview.expect("retained preview")["proposed"],
            "old settings"
        );
    }

    #[test]
    // A read failure must not appear as an empty log or zero recorded usage.
    fn unavailable_data_does_not_render_empty_usage() {
        let mut view = view();
        view.snapshot["data_state"] =
            json!({"usage":{"state":"unavailable"},"history":{"state":"unavailable"}});
        view.snapshot["errors"] = json!(["Could not read usage"]);
        view.page = 0;
        assert!(view.body().contains("Could not read usage"));
        assert!(!view.body().contains("model calls"));
        view.page = 3;
        assert!(view.body().contains("Could not read usage"));
        assert!(!view.body().contains("recent calls"));
    }

    #[test]
    // Snapshot choices must keep their original revision when the file changes before editing.
    fn externally_changed_routes_cannot_bless_snapshot_choices() {
        let dir = tempfile::tempdir().expect("fixture");
        let settings = dir.path().join("desktop.toml");
        let routes = dir.path().join("server.toml");
        let original = "[llm_clients.c]\nbase_url='http://localhost:1'\nformat='openai_responses'\n[targets.m]\nid='original'\nllm_client='c'\n[routes.r]\nid='public'\ntype='passthrough'\ntarget='m'\n";
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
        let mut view = view();
        view.route_filter.clear();
        view.refresh(&mut controller).expect("snapshot");
        view.route = 0;
        let external = original.replace("id='original'", "id='external'");
        std::fs::write(&routes, &external).expect("external edit");
        view.form(
            "Algorithm",
            FormKind::Algorithm,
            vec![("Algorithm".into(), "passthrough".into())],
        );
        view.submit(&mut controller).expect("editor");
        assert!(view.submit(&mut controller).is_err());
        assert!(matches!(
            view.form.as_ref().expect("retained draft").kind,
            FormKind::Apply(..)
        ));
        assert_eq!(
            std::fs::read_to_string(&routes).expect("preserved"),
            external
        );
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 2);
    }

    #[test]
    fn editing_a_pasted_path_preserves_unicode_and_the_suffix() {
        let mut view = view();
        view.form(
            "Project",
            FormKind::Project,
            vec![("Folder".into(), String::new())],
        );
        let form = view.form.as_mut().expect("form");
        form.insert("/tmp/équipe/project");
        form.cursor = 6;
        form.edit(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));
        form.insert("x");
        assert_eq!(form.fields[0].1, "/tmp/éxuipe/project");
        form.edit(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        form.insert("prefix");
        assert_eq!(form.fields[0].1, "prefix/tmp/éxuipe/project");
    }
}
