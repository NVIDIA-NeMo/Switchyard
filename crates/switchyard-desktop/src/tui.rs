// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The terminal app shows usage, recent calls, and server actions from controller snapshots.
//!
//! The UI thread owns every view setting. One worker thread owns the controller and runs
//! refreshes and actions, so moving around and filtering never wait for the log or a service.

use crate::controller::{Availability, Controller, Snapshot};
use ratatui::{
    Frame,
    crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Padding, Paragraph, Tabs, Wrap},
};
use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

/// Smaller terminals show a resize message instead of clipped panels.
const MIN_WIDTH: u16 = 60;
const MIN_HEIGHT: u16 = 16;
const TOAST_TIME: Duration = Duration::from_secs(4);
const SPINNER: [char; 8] = ['⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽', '⣾'];

/// Each key list has the key, a short label for the status bar, and a sentence for the help
/// window. Both read these lists, so the status bar and the help window always agree.
type Key = (&'static str, &'static str, &'static str);
const GLOBAL_KEYS: &[Key] = &[
    (
        "1",
        "page",
        "Switch pages. Tab, Shift-Tab, ← and → also switch pages.",
    ),
    (
        "r",
        "refresh",
        "Reload usage, calls, and server health now. F5 also refreshes.",
    ),
    (
        "q",
        "quit",
        "Quit the app. The server keeps running. Ctrl-C also quits.",
    ),
];

/// Theme maps each kind of text to one style. NO_COLOR keeps only bold, dim, and reverse.
#[derive(Clone, Copy)]
struct Theme {
    muted: Style,
    key: Style,
    ok: Style,
    error: Style,
    selected: Style,
    border: Style,
}
impl Theme {
    fn new(color: bool) -> Self {
        let plain = Style::new();
        if !color {
            return Self {
                muted: plain.add_modifier(Modifier::DIM),
                key: plain.add_modifier(Modifier::BOLD),
                ok: plain,
                error: plain.add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                selected: plain.add_modifier(Modifier::REVERSED),
                border: plain,
            };
        }
        // Named ANSI colors follow the terminal's palette, so they work on light and dark themes.
        Self {
            muted: plain.fg(Color::DarkGray),
            key: plain.fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ok: plain.fg(Color::Green),
            error: plain.fg(Color::Red),
            // A fixed foreground keeps dim cells readable on the selected row.
            selected: plain.fg(Color::Black).bg(Color::Cyan),
            border: plain.fg(Color::DarkGray),
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
}
const TABS: [Tab; 1] = [Tab::Overview];
impl Tab {
    fn name(self) -> &'static str {
        match self {
            Tab::Overview => "Overview",
        }
    }
    fn index(self) -> usize {
        self as usize
    }
}

enum Job {
    Refresh,
}
enum Done {
    Finished {
        /// The box keeps this message small; a snapshot holds every model and route total.
        snapshot: Result<Box<Snapshot>, String>,
    },
}
/// Request is a job for the worker plus the label that the status bar shows while it runs.
struct Request {
    job: Job,
    label: &'static str,
}
impl Request {
    fn refresh() -> Self {
        Self {
            job: Job::Refresh,
            label: "Refreshing",
        }
    }
}
struct Busy {
    label: &'static str,
    started: Instant,
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
        match snapshot {
            Ok(data) => {
                self.data = data;
                self.refreshed = Some(Instant::now());
            }
            Err(error) => self.notify(error, true),
        }
    }

    /// Messages appear in the status bar.
    fn notify(&mut self, text: String, error: bool) {
        if !text.is_empty() {
            self.toast = Some(Toast {
                text,
                error,
                shown: Instant::now(),
            });
        }
    }

    fn finish(&mut self, snapshot: Result<Box<Snapshot>, String>) {
        self.busy = None;
        self.apply_snapshot(snapshot.map(|snapshot| *snapshot));
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
            self.notify(text, true);
            return Ok(());
        }
        self.last_refresh = Instant::now();
        jobs.send(request.job).map_err(|error| error.to_string())?;
        self.busy = Some(Busy {
            label: request.label,
            started: Instant::now(),
        });
        Ok(())
    }

    fn keys(&self) -> &'static [Key] {
        &[]
    }

    fn key(&mut self, key: KeyEvent) -> Option<Request> {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.quit = true;
            return None;
        }
        match key.code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Char(digit @ '1'..='1') => {
                self.tab = TABS[digit as usize - '1' as usize];
            }
            KeyCode::Tab | KeyCode::Right => self.tab = TABS[(self.tab.index() + 1) % TABS.len()],
            KeyCode::BackTab | KeyCode::Left => {
                self.tab = TABS[(self.tab.index() + TABS.len() - 1) % TABS.len()];
            }
            KeyCode::Char('r') | KeyCode::F(5) => return Some(Request::refresh()),
            KeyCode::Esc => self.toast = None,
            _ => {}
        }
        None
    }
}

fn server_state(theme: Theme, running: bool) -> [Span<'static>; 2] {
    [
        Span::styled("● ", if running { theme.ok } else { theme.error }),
        Span::raw(if running { "running" } else { "not responding" }),
    ]
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
        data_state: _,
        metrics: _,
        errors: _,
        refresh_seconds: _,
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
    }
    draw_status(frame, app, status);
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

fn draw_overview(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let data = &app.data;
    let mut server = vec![
        Line::from(server_state(theme, data.metrics.running).to_vec()),
        Line::styled(data.metrics.server_url.as_str(), theme.muted),
    ];
    if let Some(reason) = data.data_state.health["reason"].as_str() {
        server.push(Line::styled(reason.to_string(), theme.muted));
    }
    for error in &data.errors {
        server.push(Line::styled(error.as_str(), theme.error));
    }
    if data.data_state.usage.state == Availability::Available {
        server.push(Line::from(format!(
            "Today: {} calls · {} tokens",
            data.metrics.today.requests, data.metrics.today.tokens
        )));
        server.push(Line::from(format!(
            "Past 7 days: {} calls · {} tokens",
            data.metrics.week.requests, data.metrics.week.tokens
        )));
    }
    frame.render_widget(
        Paragraph::new(server)
            .wrap(Wrap { trim: false })
            .block(block(theme, "Server").padding(Padding::horizontal(1))),
        area,
    );
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let left = if let Some(busy) = &app.busy {
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
    // Page keys do nothing while a window is open or the filter takes the keyboard.
    let global = true;
    let hints = fit_keys(theme, app.keys(), global, room);
    let [left_area, right_area] =
        Layout::horizontal([Constraint::Length(left.width() as u16), Constraint::Fill(1)])
            .areas(area);
    frame.render_widget(Paragraph::new(left), left_area);
    frame.render_widget(Paragraph::new(hints.right_aligned()), right_area);
}

/// This function lists the page's keys, then the global keys, and stops before a hint would be
/// cut.
fn fit_keys(theme: Theme, keys: &[Key], global: bool, room: usize) -> Line<'static> {
    let hint = |key: &'static str, label: &str| {
        [
            Span::styled(key, theme.key),
            Span::styled(format!(" {label}  "), theme.muted),
        ]
    };
    let mut spans = Vec::new();
    let mut used = 0;
    let globals = if global { GLOBAL_KEYS } else { &[] };
    for (key, label, _) in keys.iter().chain(globals) {
        let width = key.chars().count() + label.chars().count() + 3;
        if used + width > room {
            break;
        }
        used += width;
        spans.extend(hint(key, label));
    }
    Line::from(spans)
}

struct PasteMode;
impl Drop for PasteMode {
    fn drop(&mut self) {
        let _ = ratatui::crossterm::execute!(std::io::stdout(), event::DisableBracketedPaste);
    }
}

pub fn run(mut controller: Controller) -> Result<(), String> {
    let snapshot = controller.snapshot().unwrap_or_else(|error| Snapshot {
        errors: vec![error],
        ..Snapshot::default()
    });
    let mut app = App::new(Theme::new(color_enabled()), snapshot);
    ratatui::crossterm::execute!(std::io::stdout(), event::EnableBracketedPaste)
        .map_err(|e| e.to_string())?;
    let _paste_mode = PasteMode;
    let (jobs, inbox) = mpsc::channel::<Job>();
    let (done, results) = mpsc::channel::<Done>();
    // One worker owns the controller while the terminal continues to draw and read input.
    let worker = thread::spawn(move || {
        while let Ok(Job::Refresh) = inbox.recv() {
            let finished = Done::Finished {
                snapshot: controller.snapshot().map(Box::new),
            };
            if done.send(finished).is_err() {
                break;
            }
        }
    });
    let result = ratatui::run(|terminal| -> Result<(), String> {
        loop {
            while let Ok(Done::Finished { snapshot }) = results.try_recv() {
                app.finish(snapshot);
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
