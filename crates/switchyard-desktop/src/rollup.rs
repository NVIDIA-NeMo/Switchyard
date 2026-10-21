// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module totals recorded model calls from the retained routing log.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use crate::history::Entry;
use chrono::{DateTime, Local, NaiveDate};
use serde::{Deserialize, Serialize};
use switchyard_costs::CostTotals;

/// CLASSIFIER_TIER matches the tier recorded for classifier and judge calls.
pub(crate) const CLASSIFIER_TIER: &str = "classifier";

/// ModelTokens stores recorded token counts and costs for one model.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelTokens {
    pub requests: u64,
    /// This field counts input tokens that were neither read from nor written to a cache.
    pub input: u64,
    /// This field counts input tokens served from the provider cache.
    pub cached_input: u64,
    /// This field counts input tokens written to the provider cache.
    pub cache_write: u64,
    /// This field counts the cache writes that use the one-hour rate.
    pub cache_write_1h: u64,
    /// This field counts generated tokens, including reasoning.
    pub output: u64,
    /// This field adds up the costs that the server recorded for these calls.
    pub cost: CostTotals,
    /// This field counts calls recorded without token counts, which cannot be priced.
    pub uncounted: u64,
}

impl ModelTokens {
    /// This function returns the recorded token total, including cached reads.
    pub fn total(&self) -> u64 {
        self.input + self.cached_input + self.cache_write + self.output
    }

    pub fn add(&mut self, other: &Self) {
        self.requests += other.requests;
        self.input += other.input;
        self.cached_input += other.cached_input;
        self.cache_write += other.cache_write;
        self.cache_write_1h += other.cache_write_1h;
        self.output += other.output;
        self.cost.merge(&other.cost);
        self.uncounted += other.uncounted;
    }

    /// This function returns the token counts as usage that a price table can price.
    pub fn usage(&self) -> switchyard_protocol::Usage {
        let mut cache = switchyard_protocol::Usage::cache_details(
            Some(self.cached_input),
            Some(self.cache_write),
        );
        if let Some(cache) = cache.as_mut() {
            cache.cache_creation_1h_input_tokens = Some(self.cache_write_1h);
        }
        switchyard_protocol::Usage {
            input_tokens: Some(self.input),
            output_tokens: Some(self.output),
            cache,
            ..Default::default()
        }
    }
}

/// Totals separates caller-facing calls from classifier calls for a period.
///
/// The split matters for savings: without Switchyard the `routed` calls would
/// still have happened, but the `classifier` calls would not exist at all.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
        let tokens = ModelTokens::from(record);
        bucket.entry(record.model.clone()).or_default().add(&tokens);
        self.routes
            .entry(record.route_id.clone())
            .or_default()
            .add(&tokens);
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

    /// This function adds up the recorded cost of every call, including routing overhead.
    pub fn cost(&self) -> CostTotals {
        let mut cost = CostTotals::default();
        for tokens in self.routed.values().chain(self.classifier.values()) {
            cost.merge(&tokens.cost);
        }
        cost
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
}

impl From<&Entry> for ModelTokens {
    fn from(record: &Entry) -> Self {
        let mut cost = CostTotals::default();
        cost.add(record.cost);
        Self {
            requests: 1,
            // The recorded input includes cache reads and writes; output includes reasoning.
            input: record
                .prompt_tokens
                .saturating_sub(record.cached_tokens)
                .saturating_sub(record.cache_creation_tokens),
            cached_input: record.cached_tokens,
            cache_write: record.cache_creation_tokens,
            cache_write_1h: record.cache_creation_1h_tokens,
            output: record.completion_tokens,
            cost,
            uncounted: u64::from(
                record.completion_tokens == 0
                    || record.prompt_tokens == 0
                    || record
                        .prompt_tokens
                        .checked_sub(record.cached_tokens)
                        .and_then(|input| input.checked_sub(record.cache_creation_tokens))
                        .is_none()
                    || record.cache_creation_1h_tokens > record.cache_creation_tokens,
            ),
        }
    }
}

/// Reader keeps model and route totals without retaining individual calls.
/// Each read consumes only newly completed log lines.
#[derive(Default)]
pub struct Reader {
    file_id: Option<(u64, u64)>,
    offset: u64,
    days: BTreeMap<NaiveDate, Totals>,
    all: Totals,
}

impl Reader {
    /// A missing log has no usage. Replacement or truncation starts fresh.
    /// An invalid complete record returns an error because skipping it would understate usage.
    pub fn read(&mut self, path: &Path, today: NaiveDate) -> std::io::Result<Usage> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                *self = Self::default();
                return Ok(Usage::default());
            }
            Err(error) => return Err(error),
        };
        let length = file.metadata()?.len();
        let id = file_id(&file)?;
        if length < self.offset || self.file_id != Some(id) {
            self.offset = 0;
            self.days.clear();
            self.all = Totals::default();
        }
        self.file_id = Some(id);
        file.seek(SeekFrom::Start(self.offset))?;

        let week_start = today - chrono::Duration::days(6);
        self.days.retain(|day, _| *day >= week_start);
        let mut reader = BufReader::new(file);
        let mut line = Vec::new();
        loop {
            line.clear();
            let bytes = reader.read_until(b'\n', &mut line)?;
            // The offset stays before an unfinished line so the next refresh retries it.
            if line.last() != Some(&b'\n') {
                break;
            }
            let invalid = |message| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid routing record at byte {}: {message}", self.offset),
                )
            };
            let record: Entry =
                serde_json::from_slice(&line).map_err(|error| invalid(error.to_string()))?;
            if record.model.trim().is_empty() {
                return Err(invalid("model must not be empty".into()));
            }
            let ts = DateTime::parse_from_rfc3339(&record.ts)
                .map_err(|error| invalid(format!("invalid timestamp: {error}")))?;
            self.offset += bytes as u64;
            let day = ts.with_timezone(&Local).date_naive();
            self.all.add(&record);
            if day >= week_start {
                self.days.entry(day).or_default().add(&record);
            }
        }
        let mut usage = Usage {
            all: self.all.clone(),
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

/// This function returns the device and inode numbers that identify an open file.
/// A replaced log has a new inode, so the reader starts its totals over.
#[cfg(unix)]
fn file_id(file: &File) -> std::io::Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file.metadata()?;
    Ok((metadata.dev(), metadata.ino()))
}

/// This function returns the volume serial number and file index that identify an open file.
/// NTFS can give a recreated file the old creation time, so creation time cannot detect a
/// replaced log.
#[cfg(windows)]
fn file_id(file: &File) -> std::io::Result<(u64, u64)> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
    };
    let mut info = BY_HANDLE_FILE_INFORMATION::default();
    // The handle stays open for the call because `file` borrows it.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let index = (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow);
    Ok((u64::from(info.dwVolumeSerialNumber), index))
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
    fn baseline_usage_rejects_zero_missing_and_inconsistent_counts() {
        for (usage, uncounted) in [
            (r#""prompt_tokens":0,"completion_tokens":1"#, 1),
            (r#""prompt_tokens":10,"completion_tokens":0"#, 1),
            (r#""prompt_tokens":10,"completion_tokens":1"#, 0),
            (r#""prompt_tokens":10"#, 1),
            (r#""completion_tokens":10"#, 1),
            (
                r#""prompt_tokens":10,"completion_tokens":1,"cached_tokens":11"#,
                1,
            ),
            (
                r#""prompt_tokens":10,"completion_tokens":1,"cached_tokens":6,"cache_creation_tokens":5"#,
                1,
            ),
            (
                r#""prompt_tokens":10,"completion_tokens":1,"cache_creation_tokens":4,"cache_creation_1h_tokens":5"#,
                1,
            ),
        ] {
            let record: Entry = serde_json::from_str(&format!(
                r#"{{"ts":"2026-09-28T10:00:00Z","model":"m",{usage}}}"#
            ))
            .expect("record");
            assert_eq!(ModelTokens::from(&record).uncounted, uncounted, "{usage}");
        }
    }

    #[test]
    fn reads_appends_once_and_retries_an_unfinished_line() {
        use std::io::Write;

        let first = line("2026-09-28T10:00:00.000", "luna", "", 100, 10);
        let second = line("2026-09-28T10:00:01.000", "terra", "classifier", 30, 1);
        let (_dir, path) = log(&format!("{first}\n"));
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
            "{}{}\n{}\n",
            format!("{old}\n").repeat(5_001),
            with_route(&answer),
            with_route(&judge),
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

        // The new first record has the same length as the old log but different counts,
        // so a reader that kept its offset would skip it and keep the old 220 tokens.
        std::fs::rename(&path, path.with_extension("old")).expect("rotate");
        let first = format!("{}\n", line("2026-09-28T11:00:00.000", "sol", "", 400, 40));
        assert_eq!(first.len(), replacement.len());
        std::fs::write(&path, first + &replacement.repeat(2)).expect("replace");
        let usage = reader.read(&path, today).expect("read replacement");
        assert_eq!(usage.today.requests(), 3);
        assert_eq!(usage.today.tokens(), 880);
        assert_eq!(usage.all.tokens(), 880);
        std::fs::remove_file(&path).expect("remove log");
        let absent = reader.read(&path, today).expect("missing log");
        assert!(absent.today.is_empty() && absent.week.is_empty() && absent.all.is_empty());
    }
}
