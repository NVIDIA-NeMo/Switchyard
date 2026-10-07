// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Turns totals and prices into the rows the menu shows.

use crate::config::Config;
use crate::health::ServerStatus;
use crate::pricing::{Savings, estimate, unpriced};
use crate::rollup::{Totals, Usage};

/// Most model rows shown, so the menu stays short.
const MAX_MODEL_ROWS: usize = 5;

/// One line of the menu. Rows are informational; the actions are fixed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Row {
    Separator,
    Label(String),
}

fn label(text: impl Into<String>) -> Row {
    Row::Label(text.into())
}

/// Builds the menu body for the current day and trailing week.
pub fn build(status: ServerStatus, usage: &Usage, config: &Config) -> Vec<Row> {
    let state = match status {
        ServerStatus::Running => "running",
        ServerStatus::Stopped => "not responding",
    };
    let mut rows = vec![
        label(format!("Server: {state} · {}", config.server_url)),
        Row::Separator,
    ];

    if usage.week.is_empty() {
        rows.push(label("No requests recorded yet"));
        return rows;
    }

    rows.extend(period("Today", &usage.today, config));
    rows.push(Row::Separator);
    rows.extend(period("This week", &usage.week, config));

    let models = model_shares(&usage.week);
    if !models.is_empty() {
        rows.push(Row::Separator);
        rows.push(label("This week by model"));
        rows.extend(models);
    }

    if config.prices.is_empty() {
        rows.push(Row::Separator);
        rows.push(label("Add prices to menubar.toml to see savings"));
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
            rows.push(Row::Separator);
            rows.push(label(format!(
                "Savings hidden for this week: menubar.toml has no price for {}",
                listed(&missing)
            )));
        }
    }
    rows
}

/// Names at most three models, so that the menu row stays short.
fn listed(models: &[&str]) -> String {
    match models {
        [first, second, third, rest @ ..] if !rest.is_empty() => {
            format!("{first}, {second}, {third}, and {} more", rest.len())
        }
        _ => models.join(", "),
    }
}

fn period(name: &str, totals: &Totals, config: &Config) -> Vec<Row> {
    let requests = totals.requests();
    let mut rows = vec![label(format!(
        "{name} — {requests} request{} · {} tokens",
        if requests == 1 { "" } else { "s" },
        tokens(totals.tokens())
    ))];
    if let Some(savings) = estimate(totals, &config.prices, &config.baseline_model) {
        rows.push(label(format!("    Saved {}", saved(&savings))));
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

fn model_shares(week: &Totals) -> Vec<Row> {
    let total: u64 = week.routed.values().map(|tokens| tokens.total()).sum();
    if total == 0 {
        return Vec::new();
    }
    let mut models: Vec<_> = week
        .routed
        .iter()
        .map(|(model, tokens)| (model, tokens.total()))
        .collect();
    // Largest share first; model id breaks ties so the order is stable.
    models.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
    models
        .into_iter()
        .take(MAX_MODEL_ROWS)
        .map(|(model, count)| {
            label(format!(
                "    {model} — {:.0}%",
                count as f64 / total as f64 * 100.0
            ))
        })
        .collect()
}

/// Formats a token count compactly, since exact totals are not useful here.
fn tokens(value: u64) -> String {
    match value {
        0..=9_999 => value.to_string(),
        10_000..=999_999 => format!("{:.0}K", value as f64 / 1_000.0),
        1_000_000..=999_999_999 => format!("{:.1}M", value as f64 / 1_000_000.0),
        _ => format!("{:.1}B", value as f64 / 1_000_000_000.0),
    }
}

/// Formats dollars, keeping the sign readable when routing cost more than it saved.
fn money(value: f64) -> String {
    if value < 0.0 {
        format!("-${:.2}", -value)
    } else {
        format!("${value:.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::{ModelPrice, PriceTable};
    use crate::rollup::ModelTokens;

    fn counted(input: u64, output: u64, requests: u64) -> ModelTokens {
        ModelTokens {
            requests,
            input,
            cached_input: 0,
            output,
        }
    }

    fn config(prices: PriceTable) -> Config {
        Config {
            baseline_model: "sol".to_string(),
            prices,
            ..Config::default()
        }
    }

    fn priced() -> PriceTable {
        PriceTable::from([
            (
                "sol".to_string(),
                ModelPrice {
                    input_per_mtok: 1.25,
                    cached_input_per_mtok: None,
                    output_per_mtok: 10.0,
                },
            ),
            (
                "luna".to_string(),
                ModelPrice {
                    input_per_mtok: 0.25,
                    cached_input_per_mtok: None,
                    output_per_mtok: 2.0,
                },
            ),
        ])
    }

    fn labels(rows: &[Row]) -> Vec<String> {
        rows.iter()
            .filter_map(|row| match row {
                Row::Label(text) => Some(text.clone()),
                Row::Separator => None,
            })
            .collect()
    }

    fn usage_of(today: Totals, week: Totals) -> Usage {
        Usage { today, week }
    }

    #[test]
    fn shows_requests_tokens_and_savings_for_both_periods() {
        let mut today = Totals::default();
        today
            .routed
            .insert("luna".to_string(), counted(1_000_000, 100_000, 3));
        let mut week = Totals::default();
        week.routed
            .insert("luna".to_string(), counted(4_000_000, 400_000, 12));

        let rows = build(
            ServerStatus::Running,
            &usage_of(today, week),
            &config(priced()),
        );
        let labels = labels(&rows);

        assert_eq!(labels[0], "Server: running · http://127.0.0.1:4123");
        assert_eq!(labels[1], "Today — 3 requests · 1.1M tokens");
        assert_eq!(labels[2], "    Saved $1.80 (80% of $2.25)");
        assert_eq!(labels[3], "This week — 12 requests · 4.4M tokens");
        assert_eq!(labels[4], "    Saved $7.20 (80% of $9.00)");
    }

    #[test]
    fn hides_savings_and_explains_why_when_prices_are_missing() {
        let mut week = Totals::default();
        week.routed
            .insert("luna".to_string(), counted(1_000, 100, 1));

        week.classifier
            .insert("sol".to_string(), counted(100, 0, 1));
        for prices in [
            PriceTable::new(),
            PriceTable::from([("sol".to_string(), priced()["sol"])]),
            PriceTable::from([("luna".to_string(), priced()["luna"])]),
        ] {
            let rows = build(
                ServerStatus::Running,
                &usage_of(week.clone(), week.clone()),
                &config(prices),
            );
            let labels = labels(&rows);
            assert!(labels.iter().all(|label| !label.contains("Saved")));
            assert!(labels.iter().any(|label| label.contains("menubar.toml")));
        }
    }

    #[test]
    fn ranks_models_by_token_share() {
        let mut week = Totals::default();
        week.routed
            .insert("luna".to_string(), counted(700_000, 0, 7));
        week.routed
            .insert("sol".to_string(), counted(300_000, 0, 3));

        let rows = build(
            ServerStatus::Running,
            &usage_of(Totals::default(), week),
            &config(priced()),
        );
        let models: Vec<_> = labels(&rows)
            .into_iter()
            .filter(|label| label.starts_with("    ") && label.ends_with('%'))
            .collect();

        assert_eq!(models, vec!["    luna — 70%", "    sol — 30%"]);
    }

    #[test]
    fn reports_a_stopped_server_with_no_traffic() {
        let rows = build(ServerStatus::Stopped, &Usage::default(), &config(priced()));
        let labels = labels(&rows);

        assert!(labels[0].starts_with("Server: not responding"));
        assert_eq!(labels[1], "No requests recorded yet");
    }

    #[test]
    fn shows_a_loss_when_routing_overhead_exceeds_the_saving() {
        let mut today = Totals::default();
        today.routed.insert("sol".to_string(), counted(1_000, 0, 1));
        today
            .classifier
            .insert("sol".to_string(), counted(1_000_000, 0, 1));

        let rows = build(
            ServerStatus::Running,
            &usage_of(today.clone(), today),
            &config(priced()),
        );

        assert!(labels(&rows)[2].starts_with("    Saved -$1.25"));
    }

    #[test]
    fn formats_tokens_and_money_for_reading() {
        assert_eq!(tokens(412), "412");
        assert_eq!(tokens(12_345), "12K");
        assert_eq!(tokens(1_400_000), "1.4M");
        assert_eq!(tokens(9_700_000_000), "9.7B");

        assert_eq!(money(2.181), "$2.18");
        assert_eq!(money(-0.125), "-$0.12");
    }

    #[test]
    fn names_the_models_that_keep_the_weeks_savings_hidden() {
        let mut today = Totals::default();
        today
            .routed
            .insert("luna".to_string(), counted(1_000, 100, 1));
        let mut week = today.clone();
        week.routed
            .insert("mystery".to_string(), counted(1_000, 100, 1));

        let rows = build(
            ServerStatus::Running,
            &usage_of(today, week),
            &config(priced()),
        );
        let labels = labels(&rows);

        // Every model of today has a price, so today still shows its saving.
        let saved = labels
            .iter()
            .filter(|label| label.contains("Saved"))
            .count();
        assert_eq!(saved, 1, "{labels:?}");
        assert!(
            labels.iter().any(|label| label
                == "Savings hidden for this week: menubar.toml has no price for mystery"),
            "{labels:?}"
        );
    }

    #[test]
    fn names_the_baseline_model_when_it_has_no_price() {
        let mut week = Totals::default();
        week.routed
            .insert("luna".to_string(), counted(1_000, 100, 1));
        let mut prices = priced();
        prices.remove("sol");

        let rows = build(
            ServerStatus::Running,
            &usage_of(week.clone(), week),
            &config(prices),
        );

        let labels = labels(&rows);
        assert!(
            labels.contains(
                &"Savings hidden for this week: menubar.toml has no price for sol".to_string()
            ),
            "{labels:?}"
        );
    }

    #[test]
    fn shortens_a_long_list_of_unpriced_models() {
        let names = ["a", "b", "c", "d", "e"];

        assert_eq!(listed(&names), "a, b, c, and 2 more");
        assert_eq!(listed(&names[..3]), "a, b, c");
        assert_eq!(listed(&names[..1]), "a");
    }
}
