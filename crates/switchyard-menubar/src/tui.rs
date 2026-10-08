// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ratatui renders snapshots and submits typed operations to the shared controller.

use crate::{
    controller::{Action, Controller},
    harness::Harness,
    server_config::Choice,
};
use ratatui::{
    crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    layout::{Constraint, Layout},
    widgets::{Block, Paragraph, Wrap},
};
use serde_json::Value;
use std::time::{Duration, Instant};
const PAGES: &[&str] = &[
    "Overview", "Routes", "Install", "Usage", "Sessions", "Settings",
];
struct Form {
    title: String,
    fields: Vec<(String, String)>,
    index: usize,
    kind: FormKind,
}
enum FormKind {
    Algorithm,
    Apply(String),
    Project,
    Account,
    Models,
    Key,
    Filter,
}
struct View {
    snapshot: Value,
    page: usize,
    route: usize,
    tool: usize,
    account: usize,
    session: usize,
    filter: String,
    scroll: u16,
    message: String,
    form: Option<Form>,
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
    fn route(&self) -> Result<&Value, String> {
        self.routes()
            .get(self.route)
            .ok_or_else(|| "Choose a route first.".into())
    }
    fn tool(&self) -> Harness {
        serde_json::from_value(self.tools()[self.tool]["tool"].clone()).unwrap_or(Harness::CodexCli)
    }
    fn account(&self) -> Option<String> {
        self.tools()[self.tool]["accounts"]
            .as_array()
            .and_then(|a| self.account.checked_sub(1).and_then(|i| a.get(i)))
            .and_then(Value::as_str)
            .map(str::to_string)
    }
    fn dispatch(&mut self, controller: &mut Controller, action: Action) {
        match controller.dispatch(action) {
            Ok(reply) => {
                self.message = reply.message;
                if let Err(e) = self.refresh(controller) {
                    self.message = e;
                }
            }
            Err(e) => self.message = e,
        }
    }
    fn refresh(&mut self, controller: &mut Controller) -> Result<(), String> {
        self.snapshot = controller.snapshot()?;
        self.route = self.route.min(self.routes().len().saturating_sub(1));
        Ok(())
    }
    fn form(&mut self, title: &str, kind: FormKind, fields: Vec<(String, String)>) {
        self.form = Some(Form {
            title: title.into(),
            fields,
            index: 0,
            kind,
        });
    }
    fn submit(&mut self, controller: &mut Controller) -> Result<(), String> {
        let form = self.form.take().ok_or("No form is open.")?;
        let value = |i: usize| form.fields[i].1.trim().to_string();
        match form.kind {
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
                    FormKind::Apply(algorithm),
                    fields,
                );
            }
            FormKind::Apply(algorithm) => {
                let choices = form
                    .fields
                    .chunks(2)
                    .map(|f| Choice {
                        client: f[0].1.trim().into(),
                        model: f[1].1.trim().into(),
                    })
                    .collect();
                self.dispatch(
                    controller,
                    Action::Apply {
                        generation: self.snapshot["generation"].as_u64().unwrap_or_default(),
                        route: string(self.route()?, "key"),
                        algorithm,
                        choices,
                    },
                );
            }
            FormKind::Project => {
                let r = self.route()?;
                self.dispatch(
                    controller,
                    Action::Launch {
                        tool: self.tool(),
                        account: self.account(),
                        route: string(r, "key"),
                        id: string(r, "id"),
                        project: value(0),
                    },
                );
            }
            FormKind::Account => self.dispatch(
                controller,
                Action::AddAccount {
                    tool: self.tool(),
                    name: value(0),
                },
            ),
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
                self.message = format!(
                    "{}\n{}",
                    reply.message,
                    reply.data["models"]
                        .as_array()
                        .map(|m| m
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join("\n"))
                        .unwrap_or_default()
                );
            }
            FormKind::Filter => {
                self.filter = value(0);
                self.scroll = 0;
            }
        }
        Ok(())
    }
    fn body(&self) -> String {
        match self.page {
            0=>strings(&self.snapshot["summary"]),
            1=>{
                let mut rows=vec!["e Edit route   m Refresh models   k Save endpoint key\nAlgorithms: passthrough, random, llm_classifier, composite, stage_router, advisor, plan_execute, auto\n".into()];
                for (i,r) in self.routes().iter().enumerate(){rows.push(format!("{} {}",if i==self.route {">"}else{" "},string(r,"label")));}
                rows.push(format!("\nEndpoints: {}",self.snapshot["clients"].as_array().map(|a|a.iter().map(|c|format!("{} ({})",string(c,"name"),string(c,"host"))).collect::<Vec<_>>().join(", ")).unwrap_or_default()));rows.join("\n")
            }
            2|4=>format!("t Next tool   r Next route   a Next account\n{}\n\nTool: {}\nAccount: {}\nRoute: {}\n\n{}",if self.page==2 {"i Install / update   u Restore backup   n Add native login"}else{"l Launch worktree session   n Add native login"},string(&self.tools()[self.tool],"label"),self.account().unwrap_or_else(||"Current login".into()),self.route().map(|r|string(r,"label")).unwrap_or_default(),string(&self.tools()[self.tool],"status")),
            3=>{
                let selected=self.session.checked_sub(1).and_then(|i|self.snapshot["sessions"].as_array()?.get(i)).and_then(Value::as_str);
                let mut lines=vec![format!("s Next session   f Filter turn/model/route\nSession: {}   Filter: {}\n{} recent calls{}; {} unreadable records skipped.\nEach row is one model call; a turn can contain several calls. Input includes cached reads.\n",selected.unwrap_or("All sessions"),self.filter,self.snapshot["entries"].as_array().map_or(0,Vec::len),if self.snapshot["limited"]==true{" (8 MiB / 5,000 record limit)"}else{""},self.snapshot["skipped"])];
                if let Some(entries)=self.snapshot["entries"].as_array(){for e in entries.iter().rev(){
                    if selected.is_some_and(|s|e["session_id"].as_str()!=Some(s)){continue;}
                    if !format!("{} {} {}",string(e,"model"),string(e,"turn_id"),string(e,"route_id")).to_lowercase().contains(&self.filter.to_lowercase()){continue;}
                    lines.push(format!("{}  {}{}\nSession {}  Turn {}  Route {}\nInput {}  Cached {}  Output {}\n",string(e,"ts"),string(e,"model"),if e["tier"]=="classifier"{" (routing overhead)"}else{""},e["session_id"].as_str().unwrap_or("Not recorded"),e["turn_id"].as_str().unwrap_or("Not recorded"),string(e,"route_id"),e["prompt_tokens"],e["cached_tokens"],e["completion_tokens"]));
                }}lines.join("\n")
            }
            _=>"o Open server config\nb Restart server\nu Update from source\n\nUpdate uses the checkout that installed Switchyard.app.\nNative account limits still apply. Switchyard does not rotate login tokens.".into(),
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
            KeyCode::Down | KeyCode::Char('r') => {
                self.route = (self.route + 1) % self.routes().len().max(1)
            }
            KeyCode::Up => self.route = self.route.saturating_sub(1),
            KeyCode::PageDown => self.scroll = self.scroll.saturating_add(10),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::F(5) => self.refresh(controller)?,
            KeyCode::Char('t') => {
                self.tool = (self.tool + 1) % self.tools().len();
                self.account = 0;
            }
            KeyCode::Char('a') => {
                self.account = (self.account + 1)
                    % (self.tools()[self.tool]["accounts"]
                        .as_array()
                        .map_or(0, Vec::len)
                        + 1)
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
            KeyCode::Char('i') if self.page == 2 => {
                let r = self.route()?;
                self.dispatch(
                    controller,
                    Action::Install {
                        tool: self.tool(),
                        account: self.account(),
                        route: string(r, "key"),
                        id: string(r, "id"),
                    },
                );
            }
            KeyCode::Char('u') if self.page == 2 => self.dispatch(
                controller,
                Action::Restore {
                    tool: self.tool(),
                    account: self.account(),
                },
            ),
            KeyCode::Char('n') if self.page == 2 || self.page == 4 => self.form(
                "Add native login",
                FormKind::Account,
                vec![("Account name".into(), String::new())],
            ),
            KeyCode::Char('l') if self.page == 4 => self.form(
                "Launch worktree session",
                FormKind::Project,
                vec![("Git project absolute path".into(), String::new())],
            ),
            KeyCode::Char('s') if self.page == 3 => {
                self.session = (self.session + 1)
                    % (self.snapshot["sessions"].as_array().map_or(0, Vec::len) + 1);
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
            KeyCode::Char('b') if self.page == 5 => self.dispatch(controller, Action::Restart {}),
            KeyCode::Char('u') if self.page == 5 => self.dispatch(controller, Action::Update {}),
            _ => {}
        }
        Ok(())
    }
}
fn string(v: &Value, key: &str) -> String {
    v[key].as_str().unwrap_or_default().into()
}
fn strings(v: &Value) -> String {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}
pub fn run(mut controller: Controller) -> Result<(), String> {
    let snapshot = controller.snapshot()?;
    let mut view = View {
        snapshot,
        page: 0,
        route: 0,
        tool: 0,
        account: 0,
        session: 0,
        filter: String::new(),
        scroll: 0,
        message: String::new(),
        form: None,
    };
    let result = ratatui::run(|terminal| -> Result<(), String> {
        let mut last = Instant::now();
        loop {
            terminal.draw(|frame| {
                let areas = Layout::vertical([
                    Constraint::Length(3),
                    Constraint::Min(5),
                    Constraint::Length(5),
                ]).split(frame.area());
                let title = PAGES.iter().enumerate().map(|(i, page)| {
                    if i == view.page { format!("[{page}]") } else { page.to_string() }
                }).collect::<Vec<_>>().join("   ");
                frame.render_widget(
                    Paragraph::new(title).block(Block::bordered().title("Switchyard")),
                    areas[0],
                );
                let (body, title) = if let Some(form) = &view.form {
                    let rows = form.fields.iter().enumerate().map(|(i, (label, value))| {
                        let marker = if i == form.index { ">" } else { " " };
                        let value = if matches!(form.kind, FormKind::Key) && i == 1 {
                            "•".repeat(value.chars().count())
                        } else { value.clone() };
                        format!("{marker} {label}: {value}")
                    }).collect::<Vec<_>>().join("\n");
                    (rows, form.title.clone())
                } else { (view.body(), PAGES[view.page].into()) };
                frame.render_widget(
                    Paragraph::new(body)
                        .wrap(Wrap { trim: false })
                        .scroll((view.scroll, 0))
                        .block(Block::bordered().title(title)),
                    areas[1],
                );
                frame.render_widget(
                    Paragraph::new(format!("Tab: section · ↑/↓: route · PgUp/PgDn: scroll · F5: refresh · q: quit\nForms: Tab: field · Ctrl-U: clear · Enter: submit · Esc: cancel\n{}", view.message)),
                    areas[2],
                );
            }).map_err(|e|e.to_string())?;
            if event::poll(Duration::from_millis(200)).map_err(|e| e.to_string())? {
                let Event::Key(key) = event::read().map_err(|e| e.to_string())? else {
                    continue;
                };
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if let Some(form) = view.form.as_mut() {
                    match key.code {
                        KeyCode::Esc => view.form = None,
                        KeyCode::Tab => form.index = (form.index + 1) % form.fields.len(),
                        KeyCode::BackTab => {
                            form.index = (form.index + form.fields.len() - 1) % form.fields.len()
                        }
                        KeyCode::Enter => {
                            if let Err(e) = view.submit(&mut controller) {
                                view.message = e;
                            }
                        }
                        KeyCode::Backspace => {
                            form.fields[form.index].1.pop();
                        }
                        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            form.fields[form.index].1.clear()
                        }
                        KeyCode::Char(c) => form.fields[form.index].1.push(c),
                        _ => {}
                    }
                    continue;
                }
                if key.code == KeyCode::Char('q') {
                    return Ok(());
                }
                if let Err(e) = view.key(&mut controller, key.code) {
                    view.message = e;
                }
            }
            if last.elapsed()
                > Duration::from_secs(view.snapshot["refresh_seconds"].as_u64().unwrap_or(30))
                && view.form.is_none()
            {
                if let Err(e) = view.refresh(&mut controller) {
                    view.message = e;
                }
                last = Instant::now();
            }
        }
    });
    if result.is_err() {
        let _ = ratatui::try_restore();
    }
    result
}
