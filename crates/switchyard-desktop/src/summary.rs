// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module turns totals and recorded costs into usage summary rows.

use switchyard_costs::Pricing;

use crate::pricing::{Estimate, Savings};
use crate::rollup::{Totals, Usage};
use switchyard_desktop::config::Config;
use switchyard_desktop::health::ServerStatus;

const MAX_MODEL_ROWS: usize = 5;

/// This function builds the menu body for the current day and trailing week.
pub fn build(status: ServerStatus, usage: &Usage, config: &Config, rates: &Pricing) -> Vec<String> {
    let state = match status {
        ServerStatus::Running => "running",
        ServerStatus::Stopped => "not responding",
    };
    let mut rows = vec![
        format!("Server: {state} · {}", config.server_url),
        String::new(),
    ];

    if usage.week.is_empty() {
        rows.push(if usage.all.is_empty() {
            "No requests recorded yet".into()
        } else {
            "No requests recorded in the last 7 days".into()
        });
        return rows;
    }

    rows.extend(period("Today", &usage.today, config, rates));
    rows.push(String::new());
    rows.extend(period("This week", &usage.week, config, rates));

    let models = model_shares(&usage.week);
    if !models.is_empty() {
        rows.push(String::new());
        rows.push("This week by model".into());
        rows.extend(models);
    }

    let unknown = usage.week.cost().unknown_calls;
    let uncounted: u64 = usage
        .week
        .routed
        .values()
        .map(|tokens| tokens.uncounted)
        .sum();
    if !rates.contains(&config.baseline_model) {
        rows.push(String::new());
        rows.push(match config.prices_file.as_ref() {
            Some(path) => format!("Add {} to {} to see savings", config.baseline_model, path.display()),
            None => format!("Bundled prices have no rates for {}. Set pricing_file to a custom table to see savings.", config.baseline_model),
        });
    } else if unknown > 0 {
        rows.push(String::new());
        rows.push(format!(
            "Savings hidden for this week: {unknown} {} no recorded cost",
            if unknown == 1 {
                "call has"
            } else {
                "calls have"
            }
        ));
    } else if uncounted > 0 {
        rows.push(String::new());
        let answers = if uncounted == 1 {
            "answer has"
        } else {
            "answers have"
        };
        rows.push(format!(
            "Savings hidden for this week: {uncounted} {answers} no token counts"
        ));
    }
    rows
}

/// Each period shows its complete cost or labels its known subtotal as incomplete.
fn period(name: &str, totals: &Totals, config: &Config, rates: &Pricing) -> Vec<String> {
    let requests = totals.requests();
    let mut rows = vec![format!(
        "{name} — {requests} request{} · {} tokens",
        if requests == 1 { "" } else { "s" },
        tokens(totals.tokens())
    )];
    rows.extend(cost_rows(&Estimate::new(
        totals,
        rates,
        &config.baseline_model,
    )));
    rows
}

/// This function formats a period's recorded spending and, when it is known, the savings.
pub fn cost_rows(estimate: &Estimate) -> Vec<String> {
    let mut rows = vec![match estimate.actual {
        Some(actual) => format!("    Spent {}", money(actual)),
        None if estimate.has_known_cost => format!(
            "    Known spending {} · total unknown ({} {} no recorded cost)",
            money(estimate.known),
            estimate.unknown_calls,
            if estimate.unknown_calls == 1 {
                "call has"
            } else {
                "calls have"
            },
        ),
        None => "    Spending unknown".into(),
    }];
    if let Some(savings) = estimate.savings() {
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

/// The summary ranks up to five answer models by token share and excludes classifier calls.
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
pub(crate) fn tokens(value: u64) -> String {
    match value {
        0..=9_999 => value.to_string(),
        10_000..=999_999 => format!("{:.0}K", value as f64 / 1_000.0),
        1_000_000..=999_999_999 => format!("{:.1}M", value as f64 / 1_000_000.0),
        _ => format!("{:.1}B", value as f64 / 1_000_000_000.0),
    }
}

pub(crate) fn money(value: f64) -> String {
    if value < 0.0 {
        format!("-${:.2}", -value)
    } else {
        format!("${value:.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::build;
    use crate::rollup::{ModelTokens, Totals, Usage};
    use switchyard_costs::{CostSource, Pricing, RecordedCost};
    use switchyard_desktop::config::Config;
    use switchyard_desktop::health::ServerStatus;

    /// The fixture uses both cost sources so each can contribute a partial subtotal.
    fn usage_with_costs(answer_usd: Option<f64>, classifier_usd: Option<f64>) -> Usage {
        let mut totals = Totals::default();
        for (bucket, model, usd, source) in [
            (
                &mut totals.routed,
                "answer",
                answer_usd,
                CostSource::Estimated,
            ),
            (
                &mut totals.classifier,
                "classifier",
                classifier_usd,
                CostSource::ProviderReported,
            ),
        ] {
            let mut tokens = ModelTokens {
                requests: 1,
                input: 100,
                output: 10,
                ..ModelTokens::default()
            };
            tokens.cost.add(usd.map(|usd| RecordedCost { usd, source }));
            bucket.insert(model.into(), tokens);
        }
        Usage {
            today: totals.clone(),
            week: totals,
            ..Usage::default()
        }
    }

    /// The test checks complete, partial, missing, and zero costs in both summary periods.
    #[test]
    fn summary_shows_known_subtotals_and_missing_cost_warnings() {
        let directory = tempfile::tempdir().expect("settings directory");
        let config =
            Config::defaults_for(&directory.path().join("desktop.toml")).expect("default settings");
        let rates = Pricing::bundled().expect("bundled prices");
        for (answer_usd, classifier_usd, expected, unknown) in [
            (
                Some(1.25),
                None,
                "    Known spending $1.25 · total unknown (1 call has no recorded cost)",
                1,
            ),
            (
                None,
                Some(1.25),
                "    Known spending $1.25 · total unknown (1 call has no recorded cost)",
                1,
            ),
            (Some(1.25), Some(0.25), "    Spent $1.50", 0),
            (None, None, "    Spending unknown", 2),
            (
                Some(0.0),
                None,
                "    Known spending $0.00 · total unknown (1 call has no recorded cost)",
                1,
            ),
        ] {
            let usage = usage_with_costs(answer_usd, classifier_usd);
            let rows = build(ServerStatus::Running, &usage, &config, &rates);
            let spending: Vec<_> = rows
                .iter()
                .filter(|row| {
                    row.starts_with("    Spent ")
                        || row.starts_with("    Known spending ")
                        || row.starts_with("    Spending ")
                })
                .map(String::as_str)
                .collect();
            assert_eq!(
                spending,
                [expected, expected],
                "each period should show the expected spending row"
            );
            let warnings: Vec<_> = rows
                .iter()
                .filter(|row| row.starts_with("Savings hidden for this week:"))
                .map(String::as_str)
                .collect();
            let expected_warnings = match unknown {
                0 => vec![],
                1 => vec!["Savings hidden for this week: 1 call has no recorded cost"],
                2 => vec!["Savings hidden for this week: 2 calls have no recorded cost"],
                _ => unreachable!(),
            };
            assert_eq!(
                warnings, expected_warnings,
                "the weekly summary should report the expected missing-cost warning"
            );
        }
    }
}
