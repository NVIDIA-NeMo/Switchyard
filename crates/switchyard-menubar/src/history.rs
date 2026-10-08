// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module reads recent completed model calls and filters them by recorded IDs.

use serde::Deserialize;
use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MAX_BYTES: u64 = 8 * 1024 * 1024;
const MAX_RECORDS: usize = 5000;

#[derive(Debug, Deserialize)]
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
    pub completion_tokens: u64,
}

#[derive(Default)]
pub struct History {
    pub entries: Vec<Entry>,
    pub limited: bool,
    pub skipped: usize,
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
    pub fn sessions(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter_map(|e| e.session_id.as_deref())
            .filter(|s| !s.trim().is_empty())
            .map(str::to_string)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn display(&self, session: Option<&str>) -> String {
        let mut lines = vec![format!("{} completed requests{} · {} unreadable records skipped", self.entries.len(), if self.limited { " in the recent log window" } else { "" }, self.skipped), "Input includes cached reads. Classifier calls are routing overhead. Each line is one model call; a user turn can contain several calls.".into()];
        let entries: Vec<_> = self
            .entries
            .iter()
            .filter(|e| session.is_none_or(|s| e.session_id.as_deref() == Some(s)))
            .collect();
        let mut models: BTreeMap<&str, (u64, u64, u64)> = BTreeMap::new();
        for entry in &entries {
            let totals = models.entry(&entry.model).or_default();
            totals.0 = totals.0.saturating_add(entry.prompt_tokens);
            totals.1 = totals.1.saturating_add(entry.cached_tokens);
            totals.2 = totals.2.saturating_add(entry.completion_tokens);
        }
        lines.push("\nModels in this view".into());
        for (model, (input, cached, output)) in models {
            lines.push(format!(
                "{model}    Input {input}    Cached {cached}    Output {output}"
            ));
        }
        lines.push("\nRecent calls (newest first)".into());
        for entry in entries.into_iter().rev() {
            lines.push(format!("\n{}    {}{}\nSession: {}    Turn: {}\nRoute: {}    Input: {}    Cached: {}    Output: {}", entry.ts, entry.model, if entry.tier == "classifier" { " (routing overhead)" } else { "" }, entry.session_id.as_deref().filter(|s| !s.trim().is_empty()).unwrap_or("Not recorded"), entry.turn_id.as_deref().filter(|s| !s.trim().is_empty()).unwrap_or("Not recorded"), entry.route_id, entry.prompt_tokens, entry.cached_tokens, entry.completion_tokens));
        }
        if self.entries.is_empty() {
            lines.push("No completed calls recorded. Run a coding session through Switchyard, then click Refresh.".into());
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn groups_recorded_ids_and_never_displays_content() {
        let bytes = b"{\"ts\":\"2026-10-08T12:00:00Z\",\"model\":\"actual\",\"session_id\":\"s\",\"turn_id\":\"t\",\"prompt_tokens\":12,\"completion_tokens\":3,\"messages\":\"SECRET\"}\n{\"ts\":\"2026-10-08T12:00:01Z\",\"model\":\"judge\",\"tier\":\"classifier\"}\ninvalid\n{\"model\":\"unfinished\"}";
        let history = parse(bytes, false);
        assert_eq!(history.sessions(), vec!["s"]);
        assert_eq!(history.entries.len(), 2);
        assert_eq!(history.skipped, 1);
        let text = history.display(Some("s"));
        assert!(text.contains("Turn: t"));
        assert!(text.contains("Input: 12"));
        assert!(!text.contains("judge"));
        assert!(!text.contains("SECRET"));
        assert!(history.display(None).contains("Not recorded"));
    }
    #[test]
    fn reads_a_bounded_tail_and_keeps_only_recent_records() {
        let dir = tempfile::tempdir().expect("directory");
        let path = dir.path().join("log");
        let line = "{\"ts\":\"2026-10-08T12:00:00Z\",\"model\":\"m\"}\n";
        std::fs::write(&path, line.repeat(MAX_RECORDS + 1)).expect("log");
        let history = load(&path).expect("load");
        assert_eq!(history.entries.len(), MAX_RECORDS);
        assert!(history.limited);
        let mut contents = vec![b'x'; MAX_BYTES as usize];
        contents.extend_from_slice(b"\n");
        contents.extend_from_slice(line.as_bytes());
        std::fs::write(&path, contents).expect("large log");
        let history = load(&path).expect("load tail");
        assert_eq!(history.entries.len(), 1);
        assert!(history.limited);
    }
    // The cutoff falls after a newline, so the first complete record must stay.
    #[test]
    fn a_byte_limit_aligned_to_a_record_keeps_that_record() {
        let dir = tempfile::tempdir().expect("directory");
        let path = dir.path().join("log");
        let line = b"{\"ts\":\"2026-10-08T12:00:00Z\",\"model\":\"first\"}\n";
        let mut contents = b"discarded prefix\n".to_vec();
        contents.extend_from_slice(line);
        contents.resize(contents.len() + MAX_BYTES as usize - line.len(), b' ');
        std::fs::write(&path, contents).expect("log");
        let history = load(&path).expect("load");
        assert_eq!(history.entries.len(), 1);
        assert_eq!(history.entries[0].model, "first");
        assert!(history.limited);
    }
}
