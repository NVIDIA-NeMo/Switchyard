// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module totals recorded model calls from the retained routing log.

use std::collections::BTreeMap;
use std::fs::{File, Metadata};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use crate::history::Entry;
use chrono::{DateTime, Local, NaiveDate};
use serde::{Deserialize, Serialize};

/// CLASSIFIER_TIER matches the tier recorded for classifier and judge calls.
const CLASSIFIER_TIER: &str = "classifier";

/// ModelTokens stores recorded token counts for one model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Totals {
    /// This field groups caller-facing calls by the model that answered.
    pub routed: BTreeMap<String, ModelTokens>,
    /// This field groups classifier and judge calls by model.
    pub classifier: BTreeMap<String, ModelTokens>,
    /// Each route includes its answer calls and routing overhead.
    pub routes: BTreeMap<String, ModelTokens>,
}

impl Totals {
    fn add(&mut self, record: &Entry) {
        let bucket = if record.tier == CLASSIFIER_TIER {
            &mut self.classifier
        } else {
            &mut self.routed
        };
        bucket
            .entry(record.model.clone())
            .or_default()
            .add(&ModelTokens::from(record));
        self.routes
            .entry(record.route_id.clone())
            .or_default()
            .add(&ModelTokens::from(record));
    }

    fn merge(&mut self, other: &Self) {
        for (own, incoming) in [
            (&mut self.routed, &other.routed),
            (&mut self.classifier, &other.classifier),
            (&mut self.routes, &other.routes),
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

/// Usage stores totals from the retained log and seven local calendar days.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Usage {
    pub today: Totals,
    pub week: Totals,
    pub all: Totals,
    pub days: BTreeMap<NaiveDate, Totals>,
    pub skipped: u64,
}

impl From<&Entry> for ModelTokens {
    fn from(record: &Entry) -> Self {
        Self {
            requests: 1,
            // The recorded input includes cached reads; output includes reasoning.
            input: record.prompt_tokens.saturating_sub(record.cached_tokens),
            cached_input: record.cached_tokens,
            output: record.completion_tokens,
        }
    }
}

/// Reader keeps model and route totals without retaining individual calls.
/// Each read consumes only newly completed log lines.
#[derive(Default)]
pub struct Reader {
    metadata: Option<Metadata>,
    offset: u64,
    days: BTreeMap<NaiveDate, Totals>,
    all: Totals,
    skipped: u64,
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
            self.all = Totals::default();
            self.skipped = 0;
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
            // Entry applies the same token and optional-ID rules as recent history.
            let Ok(record) = serde_json::from_slice::<Entry>(&line) else {
                self.skipped += 1;
                continue;
            };
            if record.model.trim().is_empty() {
                self.skipped += 1;
                continue;
            }
            let Ok(ts) = DateTime::parse_from_rfc3339(&record.ts) else {
                self.skipped += 1;
                continue;
            };
            let day = ts.with_timezone(&Local).date_naive();
            self.all.add(&record);
            if day >= week_start {
                self.days.entry(day).or_default().add(&record);
            }
        }
        let mut usage = Usage {
            all: self.all.clone(),
            skipped: self.skipped,
            ..Usage::default()
        };
        for (day, totals) in &self.days {
            if *day == today {
                usage.today = totals.clone();
            }
            if *day <= today {
                usage.week.merge(totals);
                usage.days.insert(*day, totals.clone());
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
    fn aggregate_skipped_records_count_once_and_reset_with_the_log() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("fixture");
        let path = dir.path().join("routing.jsonl");
        let valid = line("2026-10-09T12:00:00", "actual", "answer", 1, 2);
        let text = format!(
            "{valid}\nnot-json\n{{\"ts\":\"bad\",\"model\":\"actual\"}}\n{{\"ts\":\"2026-10-09T12:00:00Z\",\"model\":\"\"}}\nunfinished"
        );
        std::fs::write(&path, text).expect("log");
        let mut reader = Reader::default();
        assert_eq!(
            reader
                .read(&path, date("2026-10-09"))
                .expect("read")
                .skipped,
            3
        );
        assert_eq!(
            reader
                .read(&path, date("2026-10-09"))
                .expect("refresh")
                .skipped,
            3
        );
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("append")
            .write_all(b"\n")
            .expect("complete record");
        assert_eq!(
            reader
                .read(&path, date("2026-10-09"))
                .expect("read append")
                .skipped,
            4
        );
        std::fs::write(&path, format!("{valid}\n")).expect("truncate");
        assert_eq!(
            reader
                .read(&path, date("2026-10-09"))
                .expect("read reset")
                .skipped,
            0
        );
    }

    #[test]
    fn separates_cache_reads_from_full_price_input() {
        let record: Entry = serde_json::from_str(
            r#"{"ts":"2026-09-28T10:00:00.000Z","model":"m","prompt_tokens":1000,"cached_tokens":600,"completion_tokens":50,"reasoning_tokens":25}"#,
        )
        .expect("parse record");
        let tokens = ModelTokens::from(&record);
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

    // The old calls exceed the detail limit, so bounded history cannot supply these totals.
    #[test]
    fn totals_the_full_log_by_model_route_and_local_day() {
        let old = line("2026-08-01T12:00:00.000", "older-model", "", 100, 10);
        let answer = line("2026-09-28T12:00:00.000", "answer", "", 50, 5);
        let judge = line("2026-09-28T12:00:01.000", "answer", "classifier", 20, 2);
        let with_route =
            |text: &str| text.replace("\"model\":", "\"route_id\":\"router\",\"model\":");
        let (_dir, path) = log(&format!(
            "{}{}\n{}\n{}\n",
            format!("{old}\n").repeat(5_001),
            with_route(&answer),
            with_route(&judge),
            line("2026-09-28T12:00:02.000", " ", "", 900, 90)
        ));
        let mut reader = Reader::default();
        let usage = reader.read(&path, date("2026-09-28")).expect("read");
        assert_eq!(usage.all.requests(), 5_002);
        assert_eq!(usage.all.tokens(), 550_187);
        assert_eq!(usage.week.tokens(), 77);
        assert_eq!(usage.today.routes["router"].requests, 2);
        assert_eq!(usage.today.routes["router"].total(), 77);
        assert_eq!(usage.days[&date("2026-09-28")], usage.today);
        assert_eq!(
            reader
                .read(&path, date("2026-09-29"))
                .expect("next day")
                .all,
            usage.all
        );
        assert_eq!(
            crate::summary::tray_models(&usage.today),
            ["answer — 77 tokens · calls: 2"]
        );
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
        assert_eq!(usage.all.tokens(), 220);

        std::fs::rename(&path, path.with_extension("old")).expect("rotate");
        std::fs::write(&path, replacement.repeat(3)).expect("replace");
        let usage = reader.read(&path, today).expect("read replacement");
        assert_eq!(usage.today.requests(), 3);
        assert_eq!(usage.today.tokens(), 660);
        assert_eq!(usage.all.tokens(), 660);
        std::fs::remove_file(&path).expect("remove log");
        let absent = reader.read(&path, today).expect("missing log");
        assert!(absent.today.is_empty() && absent.week.is_empty() && absent.all.is_empty());
    }
}
