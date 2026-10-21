// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ratatui renders snapshots and submits typed operations to the shared controller.

use crate::controller::{Action, Controller};
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
}
#[derive(Clone)]
struct View {
    snapshot: Value,
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
            self.session = self.snapshot["sessions"]
                .as_array()
                .and_then(|a| a.iter().position(|v| v.as_str() == Some(session)))
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
        let value = |i: usize| form.fields[i].1.trim().to_string();
        let action = match form.kind {
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
                format!(
                    "CONNECTION\n{} · {}",
                    if self.snapshot["metrics"]["running"] == true {
                        "Connected"
                    } else {
                        "Not responding"
                    },
                    string(&self.snapshot["metrics"], "server_url")
                )
            }
            1 if self.snapshot["data_state"]["history"]["state"] == "unavailable" => String::new(),
            0 => {
                let mut period = "";
                let details = self.snapshot["summary"]
                    .as_array()
                    .map(|rows| {
                        rows.iter()
                            .filter_map(Value::as_str)
                            .filter_map(|line| {
                                if line.starts_with("Today —") {
                                    period = "Today";
                                }
                                if line.starts_with("This week —") {
                                    period = "Past 7 days";
                                }
                                if line.contains("Saved") {
                                    Some(format!("{period}: {}", line.trim()))
                                } else if line.contains("prices")
                                    || line.contains("Savings hidden")
                                    || line.trim().ends_with('%')
                                {
                                    Some(line.trim().into())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<String>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                format!(
                    "CONNECTION\n\
                     {} · {}\n\n\
                     TODAY\n\
                     {} model calls · {} tokens\n\n\
                     PAST 7 DAYS\n\
                     {} model calls · {} tokens\n\n\
                     USAGE DETAILS\n\
                     {}",
                    if self.snapshot["metrics"]["running"] == true {
                        "Connected"
                    } else {
                        "Not responding"
                    },
                    string(&self.snapshot["metrics"], "server_url"),
                    self.snapshot["metrics"]["today"]["requests"],
                    self.snapshot["metrics"]["today"]["tokens"],
                    self.snapshot["metrics"]["week"]["requests"],
                    self.snapshot["metrics"]["week"]["tokens"],
                    details
                )
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
                        " (outside recent history)"
                    } else {
                        ""
                    },
                    self.filter,
                    self.snapshot["entries"].as_array().map_or(0, Vec::len),
                    if self.snapshot["limited"] == true {
                        " (8 MiB / 5,000 record limit)"
                    } else {
                        ""
                    },
                    self.snapshot["skipped"]
                )];
                if let Some(entries) = self.snapshot["entries"].as_array() {
                    for e in entries.iter().rev() {
                        if selected.is_some_and(|s| e["session_id"].as_str() != Some(s)) {
                            continue;
                        }
                        if !format!(
                            "{} {} {}",
                            string(e, "model"),
                            string(e, "turn_id"),
                            string(e, "route_id")
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
                            string(e, "ts"),
                            string(e, "model"),
                            if e["tier"] == "classifier" {
                                " (routing overhead)"
                            } else {
                                " (answer call)"
                            },
                            e["session_id"]
                                .as_str()
                                .filter(|s| !s.is_empty())
                                .unwrap_or("Not recorded"),
                            e["turn_id"]
                                .as_str()
                                .filter(|s| !s.is_empty())
                                .unwrap_or("Not recorded"),
                            e["route_id"]
                                .as_str()
                                .filter(|s| !s.is_empty())
                                .unwrap_or("Not recorded"),
                            e["prompt_tokens"]
                                .as_u64()
                                .unwrap_or_default()
                                .saturating_sub(e["cached_tokens"].as_u64().unwrap_or_default()),
                            e["cached_tokens"],
                            e["completion_tokens"]
                        ));
                    }
                }
                lines.join("\n")
            }
            _ => "o Open server config\n\
                  a Open app settings\n\
                  c Check config\n\
                  b Restart server\n\
                  u Update from source\n\n\
                  Files open with $VISUAL, or $EDITOR if $VISUAL is unset or blank.\n\
                  The default is vi (Unix) or notepad (Windows).\n\n\
                  Update uses the saved installation checkout."
                .into(),
        }
    }
    fn key(&mut self, controller: &mut Controller, code: KeyCode) -> Result<(), String> {
        match code {
            KeyCode::PageDown => self.scroll = self.scroll.saturating_add(10),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::F(5) => self.refresh(controller)?,
            KeyCode::Char('s') if self.page == 1 => {
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
            KeyCode::Char('f') if self.page == 1 => self.form(
                "Filter turn, model, or route",
                FormKind::Filter,
                vec![("Search".into(), self.filter.clone())],
            ),
            KeyCode::Char('o') if self.page == 2 => {
                self.dispatch(controller, Action::OpenConfig {})
            }
            KeyCode::Char('c') if self.page == 2 => {
                self.dispatch(controller, Action::CheckConfig {})
            }
            KeyCode::Char('a') if self.page == 2 => {
                self.dispatch(controller, Action::OpenSettings {})
            }
            KeyCode::Char('b') if self.page == 2 => self.dispatch(controller, Action::Restart {}),
            KeyCode::Char('u') if self.page == 2 => {
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
            serde_json::json!({"errors":[error],"sessions":[],"history":[],"summary":[],"settings":{}}),
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
                        let data = view.snapshot["activity"]
                            .as_array()
                            .map(|items| items.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
                            .unwrap_or_default();
                        frame.render_widget(
                            Sparkline::default()
                                .data(&data)
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
    use serde_json::json;

    fn view() -> View {
        View {
            snapshot: json!({}),
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
        }
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
        view.page = 1;
        assert!(view.body().contains("Could not read usage"));
        assert!(!view.body().contains("recent calls"));
    }
}
