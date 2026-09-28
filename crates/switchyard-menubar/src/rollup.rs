// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token totals for today and the past week, read from the server's routing log.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use chrono::{DateTime, Local, NaiveDate};
use serde::Deserialize;

/// Tier value the server writes for classifier and judge calls.
const CLASSIFIER_TIER: &str = "classifier";

/// Billable token counts for one model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ModelTokens {
    pub requests: u64,
    /// Input tokens billed at the full rate. Cache writes bill at that rate
    /// too, so they are counted here rather than tracked separately.
    pub input: u64,
    /// Input tokens served from the provider cache.
    pub cached_input: u64,
    /// Generated tokens, including reasoning.
    pub output: u64,
}

impl ModelTokens {
    /// Every token the provider counted, cached reads included.
    pub fn total(&self) -> u64 {
        self.input + self.cached_input + self.output
    }

    fn add(&mut self, other: &Self) {
        self.requests += other.requests;
        self.input += other.input;
        self.cached_input += other.cached_input;
        self.output += other.output;
    }
}

/// Tokens for a period, split by who asked for them.
///
/// The split matters for savings: without Switchyard the `routed` calls would
/// still have happened, but the `classifier` calls would not exist at all.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    /// Calls that served the caller, keyed by the model that answered.
    pub routed: BTreeMap<String, ModelTokens>,
    /// Switchyard's own classifier and judge calls, keyed by model.
    pub classifier: BTreeMap<String, ModelTokens>,
}

impl Totals {
    fn add(&mut self, record: &Record) {
        let bucket = if record.tier == CLASSIFIER_TIER {
            &mut self.classifier
        } else {
            &mut self.routed
        };
        bucket
            .entry(record.model.clone())
            .or_default()
            .add(&record.tokens());
    }

    /// Calls made on the caller's behalf.
    pub fn requests(&self) -> u64 {
        self.routed.values().map(|tokens| tokens.requests).sum()
    }

    /// Every token spent, Switchyard's own routing overhead included.
    pub fn tokens(&self) -> u64 {
        self.routed
            .values()
            .chain(self.classifier.values())
            .map(ModelTokens::total)
            .sum()
    }

    /// True when nothing was recorded.
    pub fn is_empty(&self) -> bool {
        self.routed.is_empty() && self.classifier.is_empty()
    }
}

/// What the menu shows.
#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub today: Totals,
    pub week: Totals,
}

/// Fields the rollup needs from one routing log line.
#[derive(Deserialize)]
struct Record {
    ts: String,
    model: String,
    #[serde(default)]
    tier: String,
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    reasoning_tokens: u64,
}

impl Record {
    fn tokens(&self) -> ModelTokens {
        ModelTokens {
            requests: 1,
            // `prompt_tokens` is the whole input, cache reads included.
            input: self.prompt_tokens.saturating_sub(self.cached_tokens),
            cached_input: self.cached_tokens,
            output: self.completion_tokens + self.reasoning_tokens,
        }
    }
}

/// Sums the log into today's and the trailing week's totals.
///
/// A missing log is not an error: the server creates it on its first request.
/// Unparsable lines are skipped, which also covers the last line while the
/// server is still writing it.
pub fn read(path: &Path, today: NaiveDate) -> std::io::Result<Usage> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Usage::default()),
        Err(error) => return Err(error),
    };

    let week_start = today - chrono::Duration::days(6);
    let mut usage = Usage::default();
    for line in BufReader::new(file).lines() {
        let line = line?;
        let Ok(record) = serde_json::from_str::<Record>(&line) else {
            continue;
        };
        let Ok(ts) = DateTime::parse_from_rfc3339(&record.ts) else {
            continue;
        };
        let day = ts.with_timezone(&Local).date_naive();
        if day == today {
            usage.today.add(&record);
        }
        if day >= week_start && day <= today {
            usage.week.add(&record);
        }
    }
    Ok(usage)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(text: &str) -> NaiveDate {
        text.parse().expect("valid date")
    }

    /// Builds a log line stamped at a local time, mirroring how the server
    /// writes timestamps for events that happened in the local day.
    fn line(local_ts: &str, model: &str, tier: &str, prompt: u64, completion: u64) -> String {
        let offset = Local::now().offset().to_string();
        format!(
            r#"{{"ts":"{local_ts}{offset}","model":"{model}","tier":"{tier}","prompt_tokens":{prompt},"cached_tokens":0,"completion_tokens":{completion},"reasoning_tokens":0}}"#
        )
    }

    fn log(contents: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("routing.jsonl");
        std::fs::write(&path, contents).expect("write log");
        (dir, path)
    }

    #[test]
    fn splits_routed_and_classifier_tokens() {
        let (_dir, path) = log(&format!(
            "{}\n{}\n",
            line("2026-09-28T10:00:00.000", "luna", "", 1_000, 200),
            line("2026-09-28T10:00:01.000", "terra", "classifier", 300, 10),
        ));

        let usage = read(&path, date("2026-09-28")).expect("read");

        assert_eq!(
            usage.today.requests(),
            1,
            "classifier calls are not requests"
        );
        assert_eq!(usage.today.routed["luna"].input, 1_000);
        assert_eq!(usage.today.classifier["terra"].input, 300);
        assert_eq!(usage.today.tokens(), 1_000 + 200 + 300 + 10);
    }

    #[test]
    fn separates_cache_reads_from_full_price_input() {
        let record: Record = serde_json::from_str(
            r#"{"ts":"2026-09-28T10:00:00.000Z","model":"m","prompt_tokens":1000,"cached_tokens":600,"completion_tokens":50,"reasoning_tokens":25}"#,
        )
        .expect("parse record");

        let tokens = record.tokens();

        assert_eq!(tokens.input, 400);
        assert_eq!(tokens.cached_input, 600);
        assert_eq!(tokens.output, 75, "reasoning tokens are billed as output");
        assert_eq!(tokens.total(), 1_075);
    }

    #[test]
    fn the_week_covers_seven_days() {
        let mut contents = String::new();
        for day in 18..=28 {
            contents.push_str(&line(
                &format!("2026-09-{day:02}T10:00:00.000"),
                "luna",
                "",
                100,
                10,
            ));
            contents.push('\n');
        }
        let (_dir, path) = log(&contents);

        let usage = read(&path, date("2026-09-28")).expect("read");

        assert_eq!(usage.today.requests(), 1);
        assert_eq!(usage.week.requests(), 7);
    }

    #[test]
    fn skips_lines_it_cannot_parse() {
        // The trailing line has no newline yet, as when the server is mid-write.
        let (_dir, path) = log(&format!(
            "not json\n{}\n{{\"ts\":\"2026-09-2",
            line("2026-09-28T10:00:00.000", "luna", "", 100, 10),
        ));

        let usage = read(&path, date("2026-09-28")).expect("read");

        assert_eq!(usage.today.requests(), 1);
    }

    #[test]
    fn a_missing_log_is_not_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");

        let usage = read(&dir.path().join("absent.jsonl"), date("2026-09-28")).expect("read");

        assert!(usage.today.is_empty());
    }
}
