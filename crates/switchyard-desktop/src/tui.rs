// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The terminal app shows recorded usage and server status on the Overview and Models pages.
//!
//! One worker thread owns the controller and refreshes its snapshots while the UI thread
//! draws and reads input.

use crate::controller::{Availability, Controller, Snapshot};
use crate::pricing::Estimate;
use crate::rollup::{ModelTokens, Totals, Usage};
use crate::summary::{money, tokens};
use ratatui::{
    Frame,
    crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{
        Bar, BarChart, Block, BorderType, Cell, Clear, Padding, Paragraph, Row, Table, Tabs, Wrap,
    },
};
use std::{
    collections::BTreeMap,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

/// Smaller terminals show a resize message instead of clipped panels.
const MIN_WIDTH: u16 = 60;
const MIN_HEIGHT: u16 = 16;
const TOAST_TIME: Duration = Duration::from_secs(4);
const SPINNER: [char; 8] = ['⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽', '⣾'];

/// Each entry holds a key, its status-bar label, and a longer description.
type Key = (&'static str, &'static str, &'static str);
const GLOBAL_KEYS: &[Key] = &[
    (
        "1-2",
        "page",
        "Switch pages. Tab, Shift-Tab, ← and → also switch pages.",
    ),
    (
        "r",
        "refresh",
        "Reload usage and server health now. F5 also refreshes.",
    ),
    ("?", "help", "Show or hide this help."),
    (
        "q",
        "quit",
        "Quit the app. The server keeps running. Ctrl-C also quits.",
    ),
];
const MODELS_KEYS: &[Key] = &[(
    "p",
    "period",
    "Switch the period: today, the past 7 days, or the whole log. The [ and ] keys also work.",
)];
const POPUP_KEYS: &[Key] = &[("↑↓", "scroll", ""), ("esc", "close", "")];

/// Column is a table heading, its width, and whether its values are right-aligned.
type Column = (&'static str, Constraint, bool);

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
    Models,
}
const TABS: [Tab; 2] = [Tab::Overview, Tab::Models];
impl Tab {
    fn name(self) -> &'static str {
        match self {
            Tab::Overview => "Overview",
            Tab::Models => "Models",
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

enum Popup {
    Help {
        scroll: u16,
    },
    Message {
        title: &'static str,
        text: String,
        error: bool,
        scroll: u16,
    },
}
impl Popup {
    fn scroll_mut(&mut self) -> &mut u16 {
        match self {
            Popup::Help { scroll } | Popup::Message { scroll, .. } => scroll,
        }
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
    automatic: bool,
}
impl Request {
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
    started: Instant,
    /// The UI retains the refresh origin so automatic failures preserve the current popup.
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
    popup: Option<Popup>,
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
            popup: None,
            toast: None,
            busy: None,
            quit: false,
            refreshed: None,
            last_refresh: Instant::now(),
            width: 80,
        }
    }

    /// Short single-line messages appear in the status bar; others open a scrollable window.
    fn notify(&mut self, text: String, error: bool) {
        if text.is_empty() {
            return;
        }
        if text.contains('\n') || text.chars().count() + 8 > usize::from(self.width) * 2 / 3 {
            self.popup = Some(Popup::Message {
                title: if error { "Error" } else { "Done" },
                text,
                error,
                scroll: 0,
            });
        } else {
            self.toast = Some(Toast {
                text,
                error,
                shown: Instant::now(),
            });
        }
    }

    fn finish(&mut self, snapshot: Result<Box<Snapshot>, String>) {
        let automatic = self.busy.take().is_some_and(|busy| busy.automatic);
        match snapshot {
            Ok(data) => {
                self.data = *data;
                self.refreshed = Some(Instant::now());
            }
            Err(error) => {
                // Overview keeps the error after the status message is dismissed.
                self.data.errors = vec![error.clone()];
                if automatic {
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
    }

    fn poll_results(&mut self, results: &mpsc::Receiver<Done>) -> Result<(), String> {
        loop {
            match results.try_recv() {
                Ok(Done::Finished { snapshot }) => self.finish(snapshot),
                Err(mpsc::TryRecvError::Empty) => return Ok(()),
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
        self.last_refresh = Instant::now();
        jobs.send(request.job).map_err(|error| error.to_string())?;
        self.busy = Some(Busy {
            label: request.label,
            started: Instant::now(),
            automatic: request.automatic,
        });
        Ok(())
    }

    fn keys(&self) -> &'static [Key] {
        match (&self.popup, self.tab) {
            (Some(_), _) => POPUP_KEYS,
            (None, Tab::Overview) => &[],
            (None, Tab::Models) => MODELS_KEYS,
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
        match key.code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Char('?') => self.popup = Some(Popup::Help { scroll: 0 }),
            KeyCode::Char(digit @ '1'..='2') => {
                self.tab = TABS[digit as usize - '1' as usize];
            }
            KeyCode::Tab | KeyCode::Right => self.tab = TABS[(self.tab.index() + 1) % TABS.len()],
            KeyCode::BackTab | KeyCode::Left => {
                self.tab = TABS[(self.tab.index() + TABS.len() - 1) % TABS.len()];
            }
            KeyCode::Char('r') | KeyCode::F(5) => return Some(Request::refresh(false)),
            KeyCode::Esc => self.toast = None,
            code => match self.tab {
                Tab::Overview => {}
                Tab::Models => match code {
                    KeyCode::Char('p' | ']') => self.period = self.period.next(),
                    KeyCode::Char('[') => self.period = self.period.previous(),
                    _ => {}
                },
            },
        }
        None
    }

    fn popup_key(&mut self, key: KeyEvent) -> Option<Request> {
        let popup = self.popup.as_mut()?;
        match key.code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q' | '?') => self.popup = None,
            // Drawing clamps the scroll to the text, so the last line here has no fixed value.
            code => {
                let scroll = popup.scroll_mut();
                if let Some(next) = step(code, usize::from(*scroll), usize::from(u16::MAX)) {
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

fn server_state(theme: Theme, running: bool) -> [Span<'static>; 2] {
    [
        Span::styled("● ", if running { theme.ok } else { theme.error }),
        Span::raw(if running { "running" } else { "not responding" }),
    ]
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
        data_state: _,      // draw_overview shows health; draw_models checks usage
        metrics: _,         // draw_header and the Overview cards
        today: _,           // the Overview "Today" card title
        analytics: _,       // draw_top_models and draw_models
        estimates: _,       // the Overview cards, draw_top_models, and draw_models
        baseline_model: _,  // draw_top_models and draw_models
        activity: _,        // draw_activity
        limited: _,         // draw_activity
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
        Tab::Models => draw_models(frame, app, body),
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
    let global = app.popup.is_none();
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
    let scroll = match app.popup.as_mut().map(Popup::scroll_mut) {
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

/// The help window indents wrapped descriptions under their first line, so keys stay in one column.
fn help_lines(theme: Theme, width: usize) -> Vec<Line<'static>> {
    const KEY_WIDTH: usize = 8;
    let room = width.saturating_sub(KEY_WIDTH).max(20);
    let mut lines = Vec::new();
    for (section, keys) in [("Every page", GLOBAL_KEYS), ("Models", MODELS_KEYS)] {
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

pub fn run(mut controller: Controller) -> Result<(), String> {
    let mut app = App::new(Theme::new(color_enabled()));
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
        // The initial refresh starts after terminal setup so a blocked read cannot delay drawing.
        app.start(&jobs, Request::refresh(true))?;
        loop {
            app.poll_results(&results)?;
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
        App, Availability, BTreeMap, Done, Estimate, KeyCode, KeyEvent, KeyModifiers, ModelTokens,
        Popup, Request, Snapshot, Tab, Theme, Totals, Usage, draw, mpsc,
    };
    use crate::history::Entry;
    use ratatui::{Terminal, backend::TestBackend};
    use serde_json::json;

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

    // Esc dismisses the status message while Overview keeps the refresh error.
    #[test]
    fn refresh_errors_remain_in_overview_after_dismissing_the_toast() {
        let mut app = App::new(Theme::new(false));
        app.finish(Err("Could not read settings.".into()));
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
        assert!(app.poll_results(&results).is_ok());
        drop(done);
        assert_eq!(
            app.poll_results(&results),
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
            snapshot: Err(error.into()),
        })
        .expect("result");
        app.poll_results(&results).expect("manual result");
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
                snapshot: Err(error.into()),
            })
            .expect("result");
            app.poll_results(&results).expect("automatic result");
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
            ..Snapshot::default()
        };
        app.data.data_state.usage.state = Availability::Available;
        for tab in [Tab::Overview, Tab::Models] {
            let text = screen(&mut app, tab);
            assert!(!text.contains("$0"), "{tab:?} shows a zero cost:\n{text}");
        }
        let overview = screen(&mut app, Tab::Overview);
        assert!(overview.contains("Spent unknown"));
        assert!(overview.contains("no recorded cost"));
    }
}
