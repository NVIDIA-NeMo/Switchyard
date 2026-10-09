// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module totals tokens for today and the past week from the server's routing log.

use std::collections::BTreeMap;
use std::fs::{File, Metadata};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use chrono::{DateTime, Local, NaiveDate};
use serde::Deserialize;

/// CLASSIFIER_TIER matches the tier recorded for classifier and judge calls.
const CLASSIFIER_TIER: &str = "classifier";

/// ModelTokens stores recorded token counts for one model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ModelTokens {
    pub requests: u64,
    /// This field counts input tokens priced at the full rate, including cache writes.
    pub input: u64,
    /// This field counts input tokens served from the provider cache.
    pub cached_input: u64,
    /// This field counts generated tokens, including reasoning.
    pub output: u64,
}

impl ModelTokens {
    /// This function returns the recorded token total, including cached reads.
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

/// Totals separates caller-facing calls from classifier calls for a period.
///
/// The split matters for savings: without Switchyard the `routed` calls would
/// still have happened, but the `classifier` calls would not exist at all.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    /// This field groups caller-facing calls by the model that answered.
    pub routed: BTreeMap<String, ModelTokens>,
    /// This field groups classifier and judge calls by model.
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

    fn merge(&mut self, other: &Self) {
        for (own, incoming) in [
            (&mut self.routed, &other.routed),
            (&mut self.classifier, &other.classifier),
        ] {
            for (model, tokens) in incoming {
                own.entry(model.clone()).or_default().add(tokens);
            }
        }
    }

    /// This function returns the number of caller-facing calls.
    pub fn requests(&self) -> u64 {
        self.routed.values().map(|tokens| tokens.requests).sum()
    }

    /// This function totals recorded tokens, including routing overhead.
    pub fn tokens(&self) -> u64 {
        self.routed
            .values()
            .chain(self.classifier.values())
            .map(ModelTokens::total)
            .sum()
    }

    /// This function returns true when neither group contains recorded calls.
    pub fn is_empty(&self) -> bool {
        self.routed.is_empty() && self.classifier.is_empty()
    }
}

/// Usage stores totals for the daily and weekly summaries.
#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub today: Totals,
    pub week: Totals,
}

/// Record contains the fields used to total one routing log entry.
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
}

impl Record {
    fn tokens(&self) -> ModelTokens {
        ModelTokens {
            requests: 1,
            // `prompt_tokens` is the whole input, cache reads included.
            input: self.prompt_tokens.saturating_sub(self.cached_tokens),
            cached_input: self.cached_tokens,
            // Completion tokens already include reasoning tokens.
            output: self.completion_tokens,
        }
    }
}

/// Reader keeps daily totals and reads only newly completed log lines.
#[derive(Default)]
pub struct Reader {
    metadata: Option<Metadata>,
    offset: u64,
    days: BTreeMap<NaiveDate, Totals>,
}

impl Reader {
    /// A missing log has no usage. Replacement or truncation starts fresh.
    pub fn read(&mut self, path: &Path, today: NaiveDate) -> std::io::Result<Usage> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                *self = Self::default();
                return Ok(Usage::default());
            }
            Err(error) => return Err(error),
        };
        let metadata = file.metadata()?;
        if metadata.len() < self.offset
            || !self
                .metadata
                .as_ref()
                .is_some_and(|previous| is_same_file(previous, &metadata))
        {
            self.offset = 0;
            self.days.clear();
        }
        self.metadata = Some(metadata);
        file.seek(SeekFrom::Start(self.offset))?;

        let week_start = today - chrono::Duration::days(6);
        self.days.retain(|day, _| *day >= week_start);
        let mut reader = BufReader::new(file);
        let mut line = Vec::new();
        loop {
            line.clear();
            let bytes = reader.read_until(b'\n', &mut line)?;
            // Leave an unfinished line at the offset so the next refresh retries it.
            if line.last() != Some(&b'\n') {
                break;
            }
            self.offset += bytes as u64;
            let Ok(record) = serde_json::from_slice::<Record>(&line) else {
                continue;
            };
            let Ok(ts) = DateTime::parse_from_rfc3339(&record.ts) else {
                continue;
            };
            let day = ts.with_timezone(&Local).date_naive();
            if day >= week_start {
                self.days.entry(day).or_default().add(&record);
            }
        }
        let mut usage = Usage::default();
        for (day, totals) in &self.days {
            if *day == today {
                usage.today = totals.clone();
            }
            if *day <= today {
                usage.week.merge(totals);
            }
        }
        Ok(usage)
    }
}

fn is_same_file(left: &Metadata, right: &Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        left.dev() == right.dev() && left.ino() == right.ino()
    }
    #[cfg(not(unix))]
    {
        left.created()
            .ok()
            .is_some_and(|created| right.created().ok() == Some(created))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(text: &str) -> NaiveDate {
        text.parse().expect("valid date")
    }

    /// This function builds a log line stamped at a local time, mirroring how the server
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
    fn separates_cache_reads_from_full_price_input() {
        let record: Record = serde_json::from_str(
            r#"{"ts":"2026-09-28T10:00:00.000Z","model":"m","prompt_tokens":1000,"cached_tokens":600,"completion_tokens":50,"reasoning_tokens":25}"#,
        )
        .expect("parse record");
        let tokens = record.tokens();
        assert_eq!(tokens.input, 400);
        assert_eq!(tokens.cached_input, 600);
        assert_eq!(tokens.output, 50);
        assert_eq!(tokens.total(), 1_050);
    }

    #[test]
    fn reads_appends_once_and_retries_an_unfinished_line() {
        use std::io::Write;

        let first = line("2026-09-28T10:00:00.000", "luna", "", 100, 10);
        let second = line("2026-09-28T10:00:01.000", "terra", "classifier", 30, 1);
        let (_dir, path) = log(&format!("not json\n{first}\n"));
        let today = date("2026-09-28");
        let mut reader = Reader::default();
        assert_eq!(reader.read(&path, today).expect("read").today.tokens(), 110);
        assert_eq!(
            reader.read(&path, today).expect("reread").today.tokens(),
            110
        );

        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open");
        let split = second.len() / 2;
        file.write_all(&second.as_bytes()[..split])
            .expect("partial append");
        assert_eq!(
            reader
                .read(&path, today)
                .expect("partial read")
                .today
                .tokens(),
            110
        );
        writeln!(file, "{}", &second[split..]).expect("finish append");
        let usage = reader.read(&path, today).expect("read append");
        assert_eq!(usage.today.tokens(), 141);
        assert_eq!(usage.today.requests(), 1);
        assert_eq!(usage.today.classifier["terra"].requests, 1);
    }

    #[test]
    fn resets_totals_when_the_log_is_replaced_or_truncated() {
        let (_dir, path) = log(&format!(
            "{}\n{}\n",
            line("2026-09-28T10:00:00.000", "luna", "", 100, 10),
            line("2026-09-28T10:00:01.000", "luna", "", 100, 10),
        ));
        let mut reader = Reader::default();
        let today = date("2026-09-28");
        assert_eq!(reader.read(&path, today).expect("read").today.requests(), 2);
        let replacement = format!("{}\n", line("2026-09-28T11:00:00.000", "sol", "", 200, 20));
        std::fs::write(&path, &replacement).expect("truncate");
        let usage = reader.read(&path, today).expect("read truncated");
        assert_eq!(usage.today.requests(), 1);
        assert_eq!(usage.today.tokens(), 220);

        std::fs::rename(&path, path.with_extension("old")).expect("rotate");
        std::fs::write(&path, replacement.repeat(3)).expect("replace");
        let usage = reader.read(&path, today).expect("read replacement");
        assert_eq!(usage.today.requests(), 3);
        assert_eq!(usage.today.tokens(), 660);
        std::fs::remove_file(&path).expect("remove log");
        let absent = reader.read(&path, today).expect("missing log");
        assert!(absent.today.is_empty() && absent.week.is_empty());
    }
}
