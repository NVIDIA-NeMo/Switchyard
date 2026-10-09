// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module turns totals and prices into usage summary rows.

use crate::config::Config;
use crate::health::ServerStatus;
use crate::pricing::{Savings, estimate, unpriced};
use crate::rollup::{Totals, Usage};

/// MAX_MODEL_ROWS limits the number of models shown in the usage summary.
const MAX_MODEL_ROWS: usize = 5;

/// This function builds the menu body for the current day and trailing week.
pub fn build(status: ServerStatus, usage: &Usage, config: &Config) -> Vec<String> {
    let state = match status {
        ServerStatus::Running => "running",
        ServerStatus::Stopped => "not responding",
    };
    let mut rows = vec![
        format!("Server: {state} · {}", config.server_url),
        String::new(),
    ];

    if usage.week.is_empty() {
        rows.push("No requests recorded yet".into());
        return rows;
    }

    rows.extend(period("Today", &usage.today, config));
    rows.push(String::new());
    rows.extend(period("This week", &usage.week, config));

    let models = model_shares(&usage.week);
    if !models.is_empty() {
        rows.push(String::new());
        rows.push("This week by model".into());
        rows.extend(models);
    }

    if config.prices.is_empty() {
        rows.push(String::new());
        rows.push("Add prices to menubar.toml to see savings".into());
    } else {
        let seen = usage
            .week
            .routed
            .keys()
            .chain(usage.week.classifier.keys())
            .map(String::as_str);
        let baseline = std::iter::once(config.baseline_model.as_str());
        let missing = unpriced(seen.chain(baseline), &config.prices);
        if !missing.is_empty() {
            rows.push(String::new());
            rows.push(format!(
                "Savings hidden for this week: menubar.toml has no price for {}",
                listed(&missing)
            ));
        }
    }
    rows
}

/// This function names at most three models, so that the menu row stays short.
fn listed(models: &[&str]) -> String {
    match models {
        [first, second, third, rest @ ..] if !rest.is_empty() => {
            format!("{first}, {second}, {third}, and {} more", rest.len())
        }
        _ => models.join(", "),
    }
}

fn period(name: &str, totals: &Totals, config: &Config) -> Vec<String> {
    let requests = totals.requests();
    let mut rows = vec![format!(
        "{name} — {requests} request{} · {} tokens",
        if requests == 1 { "" } else { "s" },
        tokens(totals.tokens())
    )];
    if let Some(savings) = estimate(totals, &config.prices, &config.baseline_model) {
        rows.push(format!("    Saved {}", saved(&savings)));
    }
    rows
}

fn saved(savings: &Savings) -> String {
    let amount = money(savings.saved());
    match savings.percent() {
        Some(percent) => format!("{amount} ({percent:.0}% of {})", money(savings.baseline)),
        None => amount,
    }
}

fn model_shares(week: &Totals) -> Vec<String> {
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
        .map(|(model, count)| format!("    {model} — {:.0}%", count as f64 / total as f64 * 100.0))
        .collect()
}

/// This function formats a token count compactly, since exact totals are not useful here.
fn tokens(value: u64) -> String {
    match value {
        0..=9_999 => value.to_string(),
        10_000..=999_999 => format!("{:.0}K", value as f64 / 1_000.0),
        1_000_000..=999_999_999 => format!("{:.1}M", value as f64 / 1_000_000.0),
        _ => format!("{:.1}B", value as f64 / 1_000_000_000.0),
    }
}

/// This function formats dollars, keeping the sign readable when routing cost more than it saved.
fn money(value: f64) -> String {
    if value < 0.0 {
        format!("-${:.2}", -value)
    } else {
        format!("${value:.2}")
    }
}

/// The tray ranks models by recorded tokens and combines models below the top three.
#[cfg(any(target_os = "macos", test))]
pub fn tray_models(totals: &Totals) -> Vec<String> {
    let mut models = totals.routed.clone();
    for (model, overhead) in &totals.classifier {
        let counts = models.entry(model.clone()).or_default();
        counts.requests += overhead.requests;
        counts.input += overhead.input;
        counts.cached_input += overhead.cached_input;
        counts.output += overhead.output;
    }
    let mut models: Vec<_> = models.into_iter().collect();
    models.sort_by(|left, right| {
        right
            .1
            .total()
            .cmp(&left.1.total())
            .then_with(|| left.0.cmp(&right.0))
    });
    if models.is_empty() {
        return vec!["No model calls recorded".into()];
    }
    let mut rows: Vec<_> = models
        .iter()
        .take(3)
        .map(|(model, counts)| {
            format!(
                "{model} — {} tokens · calls: {}",
                tokens(counts.total()),
                counts.requests
            )
        })
        .collect();
    if models.len() > 3 {
        let remaining: u64 = models
            .iter()
            .skip(3)
            .map(|(_, counts)| counts.total())
            .sum();
        rows.push(format!(
            "{} more models — {} tokens",
            models.len() - 3,
            tokens(remaining)
        ));
    }
    rows
}
