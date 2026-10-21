// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module turns totals and recorded costs into the usage summary that the tray and
//! `--print` show. The terminal app uses its count, money, and missing-savings wording too.

use serde::Serialize;
use switchyard_costs::Pricing;

use crate::pricing::{Estimate, Savings};
use crate::rollup::{Totals, Usage};
use switchyard_desktop::config::Config;
use switchyard_desktop::health::ServerStatus;

/// MAX_MODEL_ROWS limits the number of models shown in the usage summary.
const MAX_MODEL_ROWS: usize = 5;

/// Overview holds the server state, today's and this week's traffic, and the share of
/// this week's tokens that each model answered.
#[derive(Debug, Serialize)]
pub struct Overview {
    pub running: bool,
    pub server_url: String,
    /// This field explains why the overview has no periods.
    pub empty: Option<String>,
    pub periods: Vec<Period>,
    pub models: Vec<Share>,
    /// This field explains why savings are missing.
    pub note: Option<String>,
}

/// Period holds one period's traffic, spending, and savings as display text.
#[derive(Debug, Serialize)]
pub struct Period {
    pub name: &'static str,
    pub traffic: String,
    pub spent: String,
    pub saved: Option<String>,
}

/// Share holds one model's percentage of this week's answer tokens.
#[derive(Debug, Serialize)]
pub struct Share {
    pub model: String,
    pub percent: f64,
}

impl Overview {
    pub fn new(status: ServerStatus, usage: &Usage, config: &Config, rates: &Pricing) -> Self {
        let mut overview = Self {
            running: status == ServerStatus::Running,
            server_url: config.server_url.clone(),
            empty: None,
            periods: Vec::new(),
            models: Vec::new(),
            note: None,
        };
        if usage.week.is_empty() {
            overview.empty = Some(if usage.all.is_empty() {
                "No requests recorded yet".into()
            } else {
                "No requests recorded in the last 7 days".into()
            });
            return overview;
        }
        let estimate = |totals: &Totals| Estimate::new(totals, rates, &config.baseline_model);
        let week = estimate(&usage.week);
        overview.periods = vec![
            period("Today", &usage.today, &estimate(&usage.today)),
            period("This week", &usage.week, &week),
        ];
        overview.models = model_shares(&usage.week);
        overview.note = savings_note(&week, config, rates);
        overview
    }

    /// This function formats the overview as text lines; an empty line separates groups.
    pub fn rows(&self) -> Vec<String> {
        let state = if self.running {
            "running"
        } else {
            "not responding"
        };
        let mut rows = vec![
            format!("Server: {state} · {}", self.server_url),
            String::new(),
        ];
        if let Some(empty) = &self.empty {
            rows.push(empty.clone());
            return rows;
        }
        for (index, period) in self.periods.iter().enumerate() {
            if index > 0 {
                rows.push(String::new());
            }
            rows.push(format!("{} — {}", period.name, period.traffic));
            rows.push(format!("    {}", period.spent));
            if let Some(saved) = &period.saved {
                rows.push(format!("    {saved}"));
            }
        }
        if !self.models.is_empty() {
            rows.push(String::new());
            rows.push("This week by model".into());
            rows.extend(
                self.models
                    .iter()
                    .map(|share| format!("    {} — {:.0}%", share.model, share.percent)),
            );
        }
        if let Some(note) = &self.note {
            rows.push(String::new());
            rows.push(note.clone());
        }
        rows
    }
}

/// This function builds the summary rows for the current day and trailing week.
pub fn build(status: ServerStatus, usage: &Usage, config: &Config, rates: &Pricing) -> Vec<String> {
    Overview::new(status, usage, config, rates).rows()
}

fn period(name: &'static str, totals: &Totals, estimate: &Estimate) -> Period {
    let (spent, saved) = cost_rows(estimate);
    Period {
        name,
        traffic: format!(
            "{} · {} tokens",
            plural(totals.requests(), "request", "requests"),
            tokens(totals.tokens())
        ),
        spent,
        saved,
    }
}

/// This function formats a period's recorded spending and, when it is known, the savings,
/// for the tray and `--print`.
fn cost_rows(estimate: &Estimate) -> (String, Option<String>) {
    let spent = match estimate.actual {
        Some(actual) => format!("Spent {}", money(actual)),
        None => format!(
            "Known spending {} · total unknown ({} no recorded cost)",
            money(estimate.known),
            plural(estimate.unknown_calls, "call has", "calls have"),
        ),
    };
    let saved = estimate
        .savings()
        .map(|savings| format!("Saved {}", saved(&savings)));
    (spent, saved)
}

/// This function explains why this week's savings are missing. A price table without the
/// baseline model gets advice about the price file; `hidden_savings` words the other reasons.
fn savings_note(week: &Estimate, config: &Config, rates: &Pricing) -> Option<String> {
    if !rates.contains(&config.baseline_model) {
        return Some(match config.prices_file.as_ref() {
            Some(path) => format!(
                "Add {} to {} to see savings",
                config.baseline_model,
                path.display()
            ),
            None => format!(
                "Bundled prices have no rates for {}. Set pricing_file to a custom table to see savings.",
                config.baseline_model
            ),
        });
    }
    week.savings().is_none().then(|| {
        format!(
            "Savings hidden for this week: {}",
            hidden_savings(week, &config.baseline_model)
        )
    })
}

/// This function explains why `Estimate::savings` has no figure. The tray, `--print`, and the
/// terminal app all call it, so they give the same reason. Call it only when the savings are
/// hidden.
pub(crate) fn hidden_savings(estimate: &Estimate, baseline_model: &str) -> String {
    if estimate.baseline.is_none() {
        format!("the price table has no rates for {baseline_model}")
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

fn saved(savings: &Savings) -> String {
    let amount = money(savings.saved());
    match savings.percent() {
        Some(percent) => format!("{amount} ({percent:.0}% of {})", money(savings.baseline)),
        None => amount,
    }
}

fn model_shares(week: &Totals) -> Vec<Share> {
    let total: u64 = week.routed.values().map(|tokens| tokens.total()).sum();
    if total == 0 {
        return Vec::new();
    }
    let mut models: Vec<_> = week
        .routed
        .iter()
        .map(|(model, tokens)| (model, tokens.total()))
        .collect();
    // Models appear by decreasing share; the model ID breaks ties for a stable order.
    models.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
    models
        .into_iter()
        .take(MAX_MODEL_ROWS)
        .map(|(model, count)| Share {
            model: model.clone(),
            percent: count as f64 / total as f64 * 100.0,
        })
        .collect()
}

/// This function separates thousands with commas, so a count such as 2,062 reads at a glance.
pub(crate) fn grouped(value: u64) -> String {
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

/// This function writes a grouped count with the singular or plural words that follow it.
pub(crate) fn plural(value: u64, one: &str, many: &str) -> String {
    format!("{} {}", grouped(value), if value == 1 { one } else { many })
}

/// This function formats a token count compactly, since exact totals are not useful here.
pub(crate) fn tokens(value: u64) -> String {
    match value {
        0..=9_999 => grouped(value),
        10_000..=999_999 => format!("{:.0}K", value as f64 / 1_000.0),
        1_000_000..=999_999_999 => format!("{:.1}M", value as f64 / 1_000_000.0),
        _ => format!("{:.1}B", value as f64 / 1_000_000_000.0),
    }
}

/// This function formats dollars, keeping the sign readable when routing cost more than it saved.
pub(crate) fn money(value: f64) -> String {
    let cents = (value.abs() * 100.0).round() as u64;
    let sign = if value < 0.0 { "-" } else { "" };
    format!("{sign}${}.{:02}", grouped(cents / 100), cents % 100)
}
