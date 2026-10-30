// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module reads recent completed model calls and counts them by hour.

use serde::Deserialize;
use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MAX_BYTES: u64 = 8 * 1024 * 1024;
const MAX_RECORDS: usize = 5000;

#[derive(Clone, Debug, Deserialize, serde::Serialize)]
pub struct Entry {
    pub ts: String,
    pub model: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub route_id: String,
    #[serde(default)]
    pub tier: String,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub cache_creation_tokens: u64,
    #[serde(default)]
    pub cache_creation_1h_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    /// The server records the charge or estimate when it has rates; older records have none.
    #[serde(default)]
    pub cost: Option<switchyard_costs::RecordedCost>,
}

#[derive(Default)]
pub struct History {
    pub entries: Vec<Entry>,
    pub limited: bool,
    pub skipped: usize,
}

// `hourly_calls` uses bounded recent history, so its buckets may omit calls from a busy day.
pub fn hourly_calls(entries: &[Entry], now: chrono::DateTime<chrono::Utc>) -> [u64; 24] {
    let mut buckets = [0; 24];
    for entry in entries {
        if let Ok(time) = chrono::DateTime::parse_from_rfc3339(&entry.ts) {
            let age = now.signed_duration_since(time);
            if (chrono::Duration::zero()..chrono::Duration::hours(24)).contains(&age) {
                buckets[23 - age.num_seconds() as usize / 3600] += 1;
            }
        }
    }
    buckets
}

pub fn load(path: &Path) -> Result<History, String> {
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(History::default()),
        Err(e) => return Err(e.to_string()),
    };
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    let offset = size.saturating_sub(MAX_BYTES);
    // The preceding byte distinguishes a partial record from a complete first
    // record when the byte limit starts immediately after a newline.
    let mut previous = *b"\n";
    if offset > 0 {
        file.seek(SeekFrom::Start(offset - 1))
            .map_err(|e| e.to_string())?;
        file.read_exact(&mut previous).map_err(|e| e.to_string())?;
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take(MAX_BYTES)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let mut history = parse(&bytes, offset > 0 && previous[0] != b'\n');
    history.limited |= offset > 0;
    Ok(history)
}

fn parse(bytes: &[u8], truncated: bool) -> History {
    let mut result = History {
        limited: truncated,
        ..History::default()
    };
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    if truncated {
        lines.next();
    }
    let mut entries = VecDeque::new();
    for line in lines {
        if !line.ends_with(b"\n") {
            break;
        }
        match serde_json::from_slice::<Entry>(line) {
            Ok(entry)
                if !entry.model.trim().is_empty()
                    && chrono::DateTime::parse_from_rfc3339(&entry.ts).is_ok() =>
            {
                if entries.len() == MAX_RECORDS {
                    entries.pop_front();
                    result.limited = true;
                }
                entries.push_back(entry);
            }
            _ => result.skipped += 1,
        }
    }
    result.entries = entries.into();
    result
}

impl History {
    /// The Calls page cycles through these unique, nonempty session IDs in sorted order.
    pub fn sessions(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter_map(|entry| entry.session_id.as_deref())
            .filter(|session| !session.trim().is_empty())
            .map(str::to_string)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_BYTES, MAX_RECORDS, load};

    #[test]
    fn retains_recent_records_and_complete_byte_limited_records() {
        let directory = tempfile::tempdir().expect("directory");
        let path = directory.path().join("log");
        let line = b"{\"ts\":\"2026-10-08T12:00:00Z\",\"model\":\"recent\"}\n";
        let mut contents = b"{\"ts\":\"2026-10-08T11:00:00Z\",\"model\":\"old\"}\n".to_vec();
        contents.extend(line.repeat(MAX_RECORDS));
        std::fs::write(&path, contents).expect("record-limited log");
        let history = load(&path).expect("record-limited history");
        assert_eq!(history.entries.len(), MAX_RECORDS);
        assert!(history.entries.iter().all(|entry| entry.model == "recent"));
        assert!(history.limited);

        for prefix in [
            b"discarded prefix\n".as_slice(),
            b"partial prefix".as_slice(),
        ] {
            let mut contents = prefix.to_vec();
            contents.extend_from_slice(line);
            contents.resize(prefix.len() + MAX_BYTES as usize, b' ');
            std::fs::write(&path, contents).expect("byte-limited log");
            let history = load(&path).expect("byte-limited history");
            assert!(history.limited);
            if prefix.ends_with(b"\n") {
                assert_eq!(history.entries.len(), 1);
                assert_eq!(history.entries[0].model, "recent");
            } else {
                assert!(history.entries.is_empty());
            }
        }
    }
}
