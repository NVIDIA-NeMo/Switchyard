// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module reads the transcripts that Claude Code and Codex keep for their sessions and
//! renders a bounded hand-off that another harness can read when it starts.
//!
//! Switchyard never replays a transcript into another harness: the tools, system prompts, and
//! signed reasoning differ between harnesses. It extracts text only: the goal, recent turns,
//! files touched, commands, the last error, and any compaction summary. Reads are bounded to a
//! head window and a tail window, so describing a live multi-megabyte transcript stays cheap.
//! Tool outputs other than the last error never appear in a hand-off.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// The harnesses whose transcripts this module reads.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Claude,
    Codex,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Claude => "Claude Code",
            Kind::Codex => "Codex",
        }
    }

    /// The directory under the harness config directory that holds transcripts.
    pub fn transcript_dir(self, config_dir: &Path) -> PathBuf {
        match self {
            Kind::Claude => config_dir.join("projects"),
            Kind::Codex => config_dir.join("sessions"),
        }
    }
}

/// Byte and item limits for one read. Each limit keeps one part of the hand-off bounded.
#[derive(Clone, Debug)]
pub struct Limits {
    pub head_bytes: u64,
    pub tail_bytes: u64,
    pub turns: usize,
    pub turn_bytes: usize,
    pub goal_bytes: usize,
    pub summary_bytes: usize,
    pub files: usize,
    pub commands: usize,
    pub error_lines: usize,
    pub max_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            head_bytes: 256 * 1024,
            tail_bytes: 2 * 1024 * 1024,
            turns: 8,
            turn_bytes: 2048,
            goal_bytes: 2048,
            summary_bytes: 8192,
            files: 50,
            commands: 20,
            error_lines: 40,
            max_bytes: 32 * 1024,
        }
    }
}

/// Listing reads only enough of each transcript to describe its session.
const LIST_HEAD_BYTES: u64 = 64 * 1024;
const MAX_WALK_DEPTH: usize = 4;
const TITLE_BYTES: usize = 120;

/// The size and modification time of a transcript when it was read. A later read with a
/// different fingerprint means the session continued.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Fingerprint {
    pub len: u64,
    pub modified_ms: i64,
}

pub fn fingerprint(path: &Path) -> Result<Fingerprint, String> {
    let metadata =
        std::fs::metadata(path).map_err(|error| format!("Read {}: {error}", path.display()))?;
    Ok(Fingerprint {
        len: metadata.len(),
        modified_ms: modified_ms(&metadata),
    })
}

fn modified_ms(metadata: &std::fs::Metadata) -> i64 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_millis() as i64)
}

fn modified_rfc3339(metadata: &std::fs::Metadata) -> Option<String> {
    metadata.modified().ok().map(|time| {
        chrono::DateTime::<chrono::Utc>::from(time)
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
    })
}

/// What a transcript says about its session, read from its head window only.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Session {
    pub kind: Kind,
    pub native_id: String,
    pub transcript: PathBuf,
    pub cwd: Option<PathBuf>,
    pub branch: Option<String>,
    pub commit: Option<String>,
    pub started: Option<String>,
    /// The transcript's modification time, as RFC 3339.
    pub last_activity: Option<String>,
    pub title: Option<String>,
    pub model: Option<String>,
    pub bytes: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Turn {
    pub prompt: String,
    pub answer: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Excerpt {
    pub session: Session,
    pub goal: Option<String>,
    pub summary: Option<String>,
    pub turns: Vec<Turn>,
    pub files: Vec<String>,
    pub commands: Vec<String>,
    pub last_error: Option<String>,
    pub notes: Vec<String>,
    /// True when a limit dropped text or when the windows did not cover the whole file.
    pub truncated: bool,
    pub fingerprint: Fingerprint,
    pub extracted_at: String,
}

/// Sessions found under one harness config directory.
#[derive(Debug, Default)]
pub struct Listing {
    pub sessions: Vec<Session>,
    /// Files that were not readable transcripts, such as compressed rollouts.
    pub skipped: usize,
}

/// This lists the sessions whose transcripts live under `config_dir`, newest first.
pub fn list(kind: Kind, config_dir: &Path) -> Result<Listing, String> {
    let root = kind.transcript_dir(config_dir);
    let mut files = Vec::new();
    match kind {
        Kind::Claude => {
            for project in directories(&root)? {
                for entry in std::fs::read_dir(&project).map_err(|error| error.to_string())? {
                    let path = entry.map_err(|error| error.to_string())?.path();
                    if is_regular_file(&path) && path.extension().is_some_and(|e| e == "jsonl") {
                        files.push(path);
                    }
                }
            }
        }
        Kind::Codex => walk_rollouts(&root, MAX_WALK_DEPTH, &mut files)?,
    }
    let mut listing = Listing::default();
    for path in files {
        match describe_with(kind, &path, LIST_HEAD_BYTES) {
            Ok(session) => listing.sessions.push(session),
            Err(_) => listing.skipped += 1,
        }
    }
    listing
        .sessions
        .sort_by(|left, right| right.last_activity.cmp(&left.last_activity));
    Ok(listing)
}

/// This finds the transcript of one session by its harness-native ID.
pub fn locate(kind: Kind, config_dir: &Path, native_id: &str) -> Result<PathBuf, String> {
    if native_id.is_empty()
        || !native_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("Choose a session.".into());
    }
    let root = kind.transcript_dir(config_dir);
    match kind {
        Kind::Claude => {
            for project in directories(&root)? {
                let candidate = project.join(format!("{native_id}.jsonl"));
                if is_regular_file(&candidate) {
                    return Ok(candidate);
                }
            }
        }
        Kind::Codex => {
            let mut files = Vec::new();
            walk_rollouts(&root, MAX_WALK_DEPTH, &mut files)?;
            let suffix = format!("-{native_id}.jsonl");
            if let Some(path) = files
                .into_iter()
                .find(|path| path.to_string_lossy().ends_with(&suffix))
            {
                return Ok(path);
            }
        }
    }
    Err(format!(
        "No {} transcript for session {native_id} under {}.",
        kind.label(),
        root.display()
    ))
}

fn directories(root: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("Read {}: {error}", root.display())),
    };
    let mut directories = Vec::new();
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        if std::fs::symlink_metadata(&path).is_ok_and(|m| m.is_dir() && !m.file_type().is_symlink())
        {
            directories.push(path);
        }
    }
    directories.sort();
    Ok(directories)
}

fn is_regular_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file() && !m.file_type().is_symlink())
}

fn walk_rollouts(dir: &Path, depth: usize, files: &mut Vec<PathBuf>) -> Result<(), String> {
    if depth == 0 {
        return Ok(());
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("Read {}: {error}", dir.display())),
    };
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            walk_rollouts(&path, depth - 1, files)?;
        } else if metadata.is_file() {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default();
            if name.starts_with("rollout-") && name.ends_with(".jsonl") {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(())
}

/// This describes a session from the head of its transcript.
pub fn describe(kind: Kind, path: &Path) -> Result<Session, String> {
    describe_with(kind, path, Limits::default().head_bytes)
}

fn describe_with(kind: Kind, path: &Path, head_bytes: u64) -> Result<Session, String> {
    let (head, metadata) = read_head(path, head_bytes)?;
    let records: Vec<Value> = records(&head, metadata.len() > head_bytes).collect();
    let (session, _) = describe_records(kind, path, &metadata, &records)?;
    Ok(session)
}

/// Returns the session plus the goal found in the head records.
fn describe_records(
    kind: Kind,
    path: &Path,
    metadata: &std::fs::Metadata,
    records: &[Value],
) -> Result<(Session, Option<String>), String> {
    let mut session = Session {
        kind,
        native_id: String::new(),
        transcript: path.to_path_buf(),
        cwd: None,
        branch: None,
        commit: None,
        started: None,
        last_activity: modified_rfc3339(metadata),
        title: None,
        model: None,
        bytes: metadata.len(),
    };
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut goal = None;
    match kind {
        Kind::Claude => {
            session.native_id = stem;
            let mut title_rank = 0;
            for record in records {
                match record["type"].as_str() {
                    Some("user" | "assistant") => {
                        if session.cwd.is_none() {
                            session.cwd = string(record, "cwd").map(PathBuf::from);
                        }
                        if session.branch.is_none() {
                            session.branch = string(record, "gitBranch");
                        }
                        if session.started.is_none() {
                            session.started = string(record, "timestamp");
                        }
                        if session.model.is_none() {
                            session.model = record["message"]["model"].as_str().map(str::to_string);
                        }
                        if goal.is_none()
                            && let Some(prompt) = claude_prompt(record)
                        {
                            if title_rank < 1 {
                                session.title = Some(title_of(&prompt));
                                title_rank = 1;
                            }
                            goal = Some(prompt);
                        }
                    }
                    Some("summary") if title_rank < 2 => {
                        if let Some(text) = string(record, "summary") {
                            session.title = Some(title_of(&text));
                            title_rank = 2;
                        }
                    }
                    Some("ai-title") if title_rank < 3 => {
                        if let Some(text) = string(record, "aiTitle") {
                            session.title = Some(title_of(&text));
                            title_rank = 3;
                        }
                    }
                    Some("custom-title") => {
                        if let Some(text) = string(record, "customTitle") {
                            session.title = Some(title_of(&text));
                            title_rank = 4;
                        }
                    }
                    _ => {}
                }
            }
        }
        Kind::Codex => {
            // Rollout names end in the thread ID, which stands in until `session_meta` is read.
            session.native_id = stem
                .rsplit('-')
                .next()
                .filter(|tail| looks_like_uuid(tail))
                .unwrap_or(&stem)
                .to_string();
            for record in records {
                let payload = &record["payload"];
                match record["type"].as_str() {
                    Some("session_meta") => {
                        if let Some(id) = string(payload, "id") {
                            session.native_id = id;
                        }
                        session.cwd = string(payload, "cwd").map(PathBuf::from);
                        session.started =
                            string(payload, "timestamp").or_else(|| string(record, "timestamp"));
                        session.branch = string(&payload["git"], "branch");
                        session.commit = string(&payload["git"], "commit_hash");
                    }
                    Some("turn_context") => {
                        if session.model.is_none() {
                            session.model = string(payload, "model");
                        }
                        if session.cwd.is_none() {
                            session.cwd = string(payload, "cwd").map(PathBuf::from);
                        }
                    }
                    Some("event_msg") if goal.is_none() => {
                        if payload["type"] == "user_message"
                            && let Some(message) = string(payload, "message")
                        {
                            session.title = Some(title_of(&message));
                            goal = Some(message);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if records.is_empty() {
        return Err(format!("{} is not a JSONL transcript.", path.display()));
    }
    Ok((session, goal))
}

/// This extracts a bounded excerpt: the goal from the head window and the recent turns, files,
/// commands, last error, and compaction summary from the tail window.
pub fn excerpt(kind: Kind, path: &Path, limits: &Limits) -> Result<Excerpt, String> {
    let (head, metadata) = read_head(path, limits.head_bytes)?;
    let head_records: Vec<Value> = records(&head, metadata.len() > limits.head_bytes).collect();
    let (session, goal) = describe_records(kind, path, &metadata, &head_records)?;
    let (tail, tail_start) = read_tail(path, limits.tail_bytes)?;
    let tail_records: Vec<Value> = records(&tail, false).collect();
    let mut collector = Collector::new(limits);
    match kind {
        Kind::Claude => collector.claude(&tail_records),
        Kind::Codex => collector.codex(&tail_records),
    }
    if kind == Kind::Codex
        && head_records
            .iter()
            .any(|r| r["type"] == "session_meta" && !r["payload"]["history_base"].is_null())
    {
        collector.notes.push(
            "Earlier history of this Codex session is stored in another file and was not read."
                .into(),
        );
        collector.truncated = true;
    }
    // The head window ends at its last complete line. The windows cover the whole file only
    // when the tail window starts at or before that line, so a record that straddles the gap
    // between them counts as cut.
    let head_end = if metadata.len() > limits.head_bytes {
        head.iter()
            .rposition(|byte| *byte == b'\n')
            .map_or(0, |position| position as u64 + 1)
    } else {
        head.len() as u64
    };
    let windows_cover_file = tail_start <= head_end;
    let (goal, goal_cut) = goal
        .map(|text| truncate(&text, limits.goal_bytes))
        .map_or((None, false), |(text, cut)| (Some(text), cut));
    let truncated = collector.truncated || goal_cut || !windows_cover_file;
    Ok(Excerpt {
        session,
        goal,
        summary: collector.summary,
        turns: collector.turns,
        files: collector.files,
        commands: collector.commands,
        last_error: collector.last_error,
        notes: collector.notes,
        truncated,
        fingerprint: Fingerprint {
            len: metadata.len(),
            modified_ms: modified_ms(&metadata),
        },
        extracted_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
    })
}

/// This renders the hand-off as Markdown of at most `limits.max_bytes` bytes.
pub fn render(excerpt: &Excerpt, limits: &Limits) -> String {
    let session = &excerpt.session;
    let mut text = format!(
        "# Hand-off from {}\n\n\
         Continue this task in your own tools. First summarize the state in at most five lines, \
         confirm the next step, then continue. Do not redo completed work.\n\n\
         ## Origin\n\n\
         - Harness: {}\n\
         - Session: {}\n",
        session.kind.label(),
        session.kind.label(),
        session.native_id
    );
    if let Some(cwd) = &session.cwd {
        text.push_str(&format!("- Directory: {}\n", cwd.display()));
    }
    if let Some(branch) = &session.branch {
        text.push_str(&format!("- Branch: {branch}"));
        if let Some(commit) = &session.commit {
            text.push_str(&format!(
                " (commit {})",
                commit.chars().take(12).collect::<String>()
            ));
        }
        text.push('\n');
    }
    if let Some(started) = &session.started {
        text.push_str(&format!("- Started: {started}\n"));
    }
    if let Some(last) = &session.last_activity {
        text.push_str(&format!("- Last activity: {last}\n"));
    }
    if let Some(model) = &session.model {
        text.push_str(&format!("- Model: {model}\n"));
    }
    text.push_str(&format!(
        "- Transcript: {} ({} bytes{})\n",
        session.transcript.display(),
        session.bytes,
        if excerpt.truncated {
            "; this hand-off is an excerpt"
        } else {
            ""
        }
    ));
    if let Some(goal) = &excerpt.goal {
        text.push_str(&format!("\n## Goal\n\n{goal}\n"));
    }
    if let Some(summary) = &excerpt.summary {
        text.push_str(&format!("\n## Earlier summary\n\n{summary}\n"));
    }
    if !excerpt.turns.is_empty() {
        text.push_str("\n## Recent turns\n");
        for (index, turn) in excerpt.turns.iter().enumerate() {
            let prompt = if turn.prompt.is_empty() {
                "(prompt outside the read window)"
            } else {
                turn.prompt.as_str()
            };
            text.push_str(&format!("\n### Turn {} — user\n\n{prompt}\n", index + 1));
            if !turn.answer.is_empty() {
                text.push_str(&format!(
                    "\n### Turn {} — assistant\n\n{}\n",
                    index + 1,
                    turn.answer
                ));
            }
        }
    }
    if !excerpt.files.is_empty() {
        text.push_str("\n## Files touched\n\n");
        for file in &excerpt.files {
            text.push_str(&format!("- {file}\n"));
        }
    }
    if !excerpt.commands.is_empty() {
        text.push_str("\n## Commands\n\n");
        for command in &excerpt.commands {
            text.push_str(&format!("- `{}`\n", command.replace('`', "'")));
        }
    }
    if let Some(error) = &excerpt.last_error {
        text.push_str(&format!("\n## Last error\n\n```\n{error}\n```\n"));
    }
    if !excerpt.notes.is_empty() || excerpt.truncated {
        text.push_str("\n## Notes\n\n");
        for note in &excerpt.notes {
            text.push_str(&format!("- {note}\n"));
        }
        if excerpt.truncated {
            text.push_str("- Parts of the transcript were cut to keep this hand-off short.\n");
        }
    }
    cap(text, limits.max_bytes)
}

fn cap(text: String, max_bytes: usize) -> String {
    const MARKER: &str = "\n\n_(hand-off cut at the size limit)_\n";
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes.saturating_sub(MARKER.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut cut = text[..end].to_string();
    cut.push_str(MARKER);
    cut
}

struct Collector<'a> {
    limits: &'a Limits,
    turns: Vec<Turn>,
    files: Vec<String>,
    commands: Vec<String>,
    last_error: Option<String>,
    summary: Option<String>,
    notes: Vec<String>,
    truncated: bool,
    /// Tool call IDs mapped to the command or tool name that made them.
    calls: HashMap<String, String>,
}

impl<'a> Collector<'a> {
    fn new(limits: &'a Limits) -> Self {
        Self {
            limits,
            turns: Vec::new(),
            files: Vec::new(),
            commands: Vec::new(),
            last_error: None,
            summary: None,
            notes: Vec::new(),
            truncated: false,
            calls: HashMap::new(),
        }
    }

    fn prompt(&mut self, prompt: &str) {
        let (prompt, cut) = truncate(prompt, self.limits.turn_bytes);
        self.truncated |= cut;
        self.turns.push(Turn {
            prompt,
            answer: String::new(),
        });
        if self.turns.len() > self.limits.turns {
            self.turns.remove(0);
            self.truncated = true;
        }
    }

    fn answer(&mut self, text: &str) {
        if self.turns.is_empty() {
            self.turns.push(Turn::default());
        }
        let turn = self.turns.last_mut().expect("a turn exists");
        if !turn.answer.is_empty() {
            turn.answer.push('\n');
        }
        turn.answer.push_str(text.trim());
        if turn.answer.len() > self.limits.turn_bytes {
            let (cut, _) = truncate(&turn.answer, self.limits.turn_bytes);
            turn.answer = cut;
            self.truncated = true;
        }
    }

    /// Codex records an answer twice: as an `agent_message` event and as a `response_item`
    /// message. Whichever comes second is skipped when its text repeats the answer so far.
    fn answer_once(&mut self, text: &str) {
        let text = text.trim();
        if text.is_empty()
            || self
                .turns
                .last()
                .is_some_and(|turn| turn.answer.ends_with(text))
        {
            return;
        }
        self.answer(text);
    }

    fn file(&mut self, path: String) {
        if path.is_empty() || self.files.contains(&path) {
            return;
        }
        if self.files.len() >= self.limits.files {
            self.truncated = true;
            return;
        }
        self.files.push(path);
    }

    fn command(&mut self, command: String) {
        if command.is_empty() {
            return;
        }
        if self.commands.len() >= self.limits.commands {
            self.commands.remove(0);
            self.truncated = true;
        }
        self.commands.push(command);
    }

    fn error(&mut self, context: Option<&str>, output: &str) {
        let (lines, cut) = first_lines(output, self.limits.error_lines);
        self.truncated |= cut;
        self.last_error = Some(match context {
            Some(context) if !context.is_empty() => format!("$ {context}\n{lines}"),
            _ => lines,
        });
    }

    fn compaction(&mut self, summary: &str) {
        let (summary, cut) = truncate(summary, self.limits.summary_bytes);
        self.truncated |= cut;
        self.summary = Some(summary);
        // The summary stands in for the turns before it.
        self.truncated |= !self.turns.is_empty();
        self.turns.clear();
    }

    fn claude(&mut self, records: &[Value]) {
        for record in records {
            if record["isSidechain"] == true || record["isMeta"] == true {
                continue;
            }
            match record["type"].as_str() {
                Some("system") => {
                    if record["subtype"] == "compact_boundary" {
                        self.truncated |= !self.turns.is_empty();
                        self.turns.clear();
                    }
                }
                Some("user") => {
                    if record["isCompactSummary"] == true {
                        if let Some(text) = record["message"]["content"].as_str() {
                            self.compaction(text);
                        } else if let Some(text) = text_blocks(&record["message"]["content"]) {
                            self.compaction(&text);
                        }
                        continue;
                    }
                    if let Some(prompt) = claude_prompt(record) {
                        self.prompt(&prompt);
                        continue;
                    }
                    if let Some(blocks) = record["message"]["content"].as_array() {
                        for block in blocks {
                            if block["type"] == "tool_result" && block["is_error"] == true {
                                let context = block["tool_use_id"]
                                    .as_str()
                                    .and_then(|id| self.calls.get(id).cloned());
                                let output = block["content"]
                                    .as_str()
                                    .map(str::to_string)
                                    .or_else(|| text_blocks(&block["content"]))
                                    .unwrap_or_default();
                                self.error(context.as_deref(), &output);
                            }
                        }
                    }
                }
                Some("assistant") => {
                    let Some(blocks) = record["message"]["content"].as_array() else {
                        continue;
                    };
                    for block in blocks {
                        match block["type"].as_str() {
                            Some("text") => {
                                if let Some(text) = block["text"].as_str() {
                                    self.answer(text);
                                }
                            }
                            Some("tool_use") => {
                                let name = block["name"].as_str().unwrap_or_default();
                                let input = &block["input"];
                                let context = match name {
                                    "Edit" | "MultiEdit" | "Write" | "NotebookEdit" => {
                                        let path = string(input, "file_path")
                                            .or_else(|| string(input, "notebook_path"))
                                            .unwrap_or_default();
                                        self.file(path.clone());
                                        format!("{name} {path}")
                                    }
                                    "Bash" => {
                                        let command = string(input, "command").unwrap_or_default();
                                        self.command(command.clone());
                                        command
                                    }
                                    _ => name.to_string(),
                                };
                                if let Some(id) = block["id"].as_str() {
                                    self.calls.insert(id.to_string(), context);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn codex(&mut self, records: &[Value]) {
        for record in records {
            let payload = &record["payload"];
            match record["type"].as_str() {
                Some("event_msg") => match payload["type"].as_str() {
                    Some("user_message") => {
                        if let Some(message) = payload["message"].as_str() {
                            self.prompt(message);
                        }
                    }
                    Some("agent_message") => {
                        if let Some(message) = payload["message"].as_str() {
                            self.answer_once(message);
                        }
                    }
                    Some("exec_command_end") => {
                        let failed = payload["exit_code"].as_i64().is_some_and(|code| code != 0);
                        if failed {
                            let command = command_text(&payload["command"]);
                            let output = payload["aggregated_output"].as_str().unwrap_or_default();
                            self.error(Some(&command), output);
                        }
                    }
                    Some("patch_apply_end") => {
                        if let Some(changes) = payload["changes"].as_object() {
                            for path in changes.keys() {
                                self.file(path.clone());
                            }
                        }
                    }
                    _ => {}
                },
                Some("response_item") => match payload["type"].as_str() {
                    Some("function_call") if payload["name"] == "exec_command" => {
                        let arguments = payload["arguments"]
                            .as_str()
                            .and_then(|text| serde_json::from_str::<Value>(text).ok())
                            .unwrap_or(Value::Null);
                        let command = command_text(&arguments["cmd"]);
                        let command = if command.is_empty() {
                            command_text(&arguments["command"])
                        } else {
                            command
                        };
                        self.command(command);
                    }
                    Some("custom_tool_call") if payload["name"] == "apply_patch" => {
                        for path in patch_files(payload["input"].as_str().unwrap_or_default()) {
                            self.file(path);
                        }
                    }
                    Some("message")
                        if payload["role"] == "assistant" && payload["phase"] != "commentary" =>
                    {
                        if let Some(text) = text_blocks(&payload["content"]) {
                            self.answer_once(&text);
                        }
                    }
                    _ => {}
                },
                Some("compacted") => {
                    if let Some(message) = payload["message"].as_str() {
                        self.compaction(message);
                    }
                }
                _ => {}
            }
        }
    }
}

/// A user prompt typed by the user: string content, or text blocks only.
fn claude_prompt(record: &Value) -> Option<String> {
    if record["type"] != "user"
        || record["isMeta"] == true
        || record["isSidechain"] == true
        || record["isCompactSummary"] == true
    {
        return None;
    }
    let content = &record["message"]["content"];
    if let Some(text) = content.as_str() {
        return (!text.trim().is_empty()).then(|| text.trim().to_string());
    }
    let blocks = content.as_array()?;
    if blocks.is_empty() || blocks.iter().any(|block| block["type"] != "text") {
        return None;
    }
    text_blocks(content).filter(|text| !text.trim().is_empty())
}

/// Joins the text of `text`, `input_text`, and `output_text` blocks; other block kinds are
/// dropped.
fn text_blocks(content: &Value) -> Option<String> {
    let blocks = content.as_array()?;
    let text = blocks
        .iter()
        .filter(|block| {
            matches!(
                block["type"].as_str(),
                Some("text" | "input_text" | "output_text")
            )
        })
        .filter_map(|block| block["text"].as_str())
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

fn command_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.trim().to_string(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

fn patch_files(input: &str) -> Vec<String> {
    input
        .lines()
        .filter_map(|line| {
            [
                "*** Add File: ",
                "*** Update File: ",
                "*** Delete File: ",
                "*** Move to: ",
            ]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix))
            .map(|path| path.trim().to_string())
        })
        .filter(|path| !path.is_empty())
        .collect()
}

fn looks_like_uuid(text: &str) -> bool {
    text.len() == 36
        && text.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
}

fn string(value: &Value, key: &str) -> Option<String> {
    value[key]
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

fn title_of(text: &str) -> String {
    let first = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    truncate(first.trim(), TITLE_BYTES).0
}

/// Cuts `text` at a character boundary so that it fits in `max_bytes`.
fn truncate(text: &str, max_bytes: usize) -> (String, bool) {
    if text.len() <= max_bytes {
        return (text.to_string(), false);
    }
    // The ellipsis takes three bytes, so the kept text stops three bytes early.
    let mut end = max_bytes.saturating_sub('…'.len_utf8());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    if end == 0 {
        return (String::new(), true);
    }
    (format!("{}…", &text[..end]), true)
}

fn first_lines(text: &str, lines: usize) -> (String, bool) {
    let mut taken: Vec<&str> = text.lines().take(lines).collect();
    let cut = text.lines().nth(lines).is_some();
    if cut {
        taken.push("…");
    }
    (taken.join("\n"), cut)
}

/// Reads the first `limit` bytes. The caller drops a trailing partial line when the file is
/// longer than the window.
fn read_head(path: &Path, limit: u64) -> Result<(Vec<u8>, std::fs::Metadata), String> {
    let mut file =
        std::fs::File::open(path).map_err(|error| format!("Read {}: {error}", path.display()))?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.is_file() {
        return Err(format!("{} is not a file.", path.display()));
    }
    let mut bytes = Vec::new();
    file.by_ref()
        .take(limit)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if let Some(first) = bytes.iter().find(|byte| !byte.is_ascii_whitespace())
        && *first != b'{'
    {
        return Err(format!(
            "{} is not a JSONL transcript. Compressed rollouts are not supported.",
            path.display()
        ));
    }
    Ok((bytes, metadata))
}

/// Reads the last `limit` bytes, starting at a line boundary. Returns the bytes and the file
/// offset of their first byte.
fn read_tail(path: &Path, limit: u64) -> Result<(Vec<u8>, u64), String> {
    let mut file =
        std::fs::File::open(path).map_err(|error| format!("Read {}: {error}", path.display()))?;
    let size = file.metadata().map_err(|error| error.to_string())?.len();
    let offset = size.saturating_sub(limit);
    // The preceding byte distinguishes a partial record from a complete first record when the
    // window starts immediately after a newline.
    let mut previous = *b"\n";
    if offset > 0 {
        file.seek(SeekFrom::Start(offset - 1))
            .map_err(|error| error.to_string())?;
        file.read_exact(&mut previous)
            .map_err(|error| error.to_string())?;
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    file.take(limit)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    let mut start = offset;
    if offset > 0 && previous[0] != b'\n' {
        if let Some(newline) = bytes.iter().position(|byte| *byte == b'\n') {
            bytes.drain(..=newline);
            start += newline as u64 + 1;
        } else {
            start += bytes.len() as u64;
            bytes.clear();
        }
    }
    Ok((bytes, start))
}

/// Parses complete JSON object lines. A trailing line without a newline counts only when the
/// window reached the end of the file.
fn records(bytes: &[u8], complete_only: bool) -> impl Iterator<Item = Value> + '_ {
    bytes
        .split_inclusive(|byte| *byte == b'\n')
        .take_while(move |line| !complete_only || line.ends_with(b"\n"))
        .filter_map(|line| serde_json::from_slice::<Value>(line).ok())
        .filter(Value::is_object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write_lines(path: &Path, lines: &[Value]) {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("directories");
        let text = lines
            .iter()
            .map(|line| serde_json::to_string(line).expect("json"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(path, text).expect("transcript");
    }

    fn claude_user(uuid: &str, content: Value) -> Value {
        json!({"type":"user","uuid":uuid,"parentUuid":null,"isSidechain":false,"userType":"external",
               "cwd":"/work/repo","gitBranch":"feature","sessionId":"ignored-by-reader",
               "version":"2.1.300","timestamp":"2026-10-20T10:00:00.000Z",
               "message":{"role":"user","content":content}})
    }

    fn claude_assistant(id: &str, block: Value) -> Value {
        json!({"type":"assistant","uuid":format!("a-{id}"),"cwd":"/work/repo","gitBranch":"feature",
               "timestamp":"2026-10-20T10:01:00.000Z",
               "message":{"id":id,"role":"assistant","model":"claude-opus-5-5","content":[block]}})
    }

    fn claude_fixture(dir: &Path) -> PathBuf {
        let session = "0f6a1c2e-1111-4222-8333-444455556666";
        let path = dir
            .join("projects/-work-repo")
            .join(format!("{session}.jsonl"));
        let summary = format!("Summary so far: {}", "s".repeat(20 * 1024));
        write_lines(
            &path,
            &[
                json!({"type":"queue-operation","operation":"enqueue"}),
                json!({"type":"custom-title","customTitle":"Fix the parser"}),
                claude_user("u1", json!("Make the parser accept trailing commas")),
                claude_assistant(
                    "msg_1",
                    json!({"type":"text","text":"I will edit the parser."}),
                ),
                claude_assistant(
                    "msg_1",
                    json!({"type":"thinking","thinking":"THINK-SENTINEL","signature":"x"}),
                ),
                claude_assistant(
                    "msg_1",
                    json!({"type":"tool_use","id":"toolu_1","name":"Edit",
                    "input":{"file_path":"/work/repo/src/parser.rs","old_string":"x","new_string":"y"}}),
                ),
                claude_assistant(
                    "msg_1",
                    json!({"type":"tool_use","id":"toolu_2","name":"Bash",
                    "input":{"command":"cargo test -p parser"}}),
                ),
                claude_user(
                    "u2",
                    json!([{"type":"tool_result","tool_use_id":"toolu_1","content":"OKBODY-SENTINEL"}]),
                ),
                claude_user(
                    "u3",
                    json!([{"type":"tool_result","tool_use_id":"toolu_2","is_error":true,
                    "content":"error[E0308]: mismatched types\n --> src/parser.rs:10\nmore"}]),
                ),
                json!({"type":"attachment","attachment":{"type":"skill_listing","content":"ATTACH-SENTINEL"}}),
                json!({"type":"user","isSidechain":true,"message":{"role":"user","content":"SIDECHAIN-SENTINEL"}}),
                json!({"type":"user","isMeta":true,"message":{"role":"user","content":"META-SENTINEL"}}),
                json!({"type":"file-history-snapshot","snapshot":{"trackedFiles":"FHS-SENTINEL"}}),
                json!({"type":"system","subtype":"compact_boundary","compactMetadata":{"trigger":"auto"}}),
                json!({"type":"user","isCompactSummary":true,"message":{"role":"user","content":summary}}),
                claude_user("u4", json!("Now add a test for trailing commas")),
                claude_assistant(
                    "msg_2",
                    json!({"type":"text","text":"Added the test; it passes."}),
                ),
            ],
        );
        path
    }

    #[test]
    fn claude_transcripts_describe_the_session_and_excerpt_only_text() {
        let dir = tempfile::tempdir().expect("dir");
        let path = claude_fixture(dir.path());
        let session = describe(Kind::Claude, &path).expect("describe");
        assert_eq!(session.native_id, "0f6a1c2e-1111-4222-8333-444455556666");
        assert_eq!(session.cwd.as_deref(), Some(Path::new("/work/repo")));
        assert_eq!(session.branch.as_deref(), Some("feature"));
        assert_eq!(session.started.as_deref(), Some("2026-10-20T10:00:00.000Z"));
        assert_eq!(session.title.as_deref(), Some("Fix the parser"));
        assert_eq!(session.model.as_deref(), Some("claude-opus-5-5"));
        assert!(session.last_activity.is_some());

        let limits = Limits::default();
        let excerpt = excerpt(Kind::Claude, &path, &limits).expect("excerpt");
        assert_eq!(
            excerpt.goal.as_deref(),
            Some("Make the parser accept trailing commas")
        );
        let summary = excerpt.summary.as_deref().expect("summary");
        assert!(summary.starts_with("Summary so far"));
        assert!(summary.len() <= limits.summary_bytes);
        assert!(excerpt.truncated);
        assert_eq!(excerpt.turns.len(), 1);
        assert_eq!(
            excerpt.turns[0].prompt,
            "Now add a test for trailing commas"
        );
        assert_eq!(excerpt.turns[0].answer, "Added the test; it passes.");
        assert_eq!(excerpt.files, vec!["/work/repo/src/parser.rs"]);
        assert_eq!(excerpt.commands, vec!["cargo test -p parser"]);
        let error = excerpt.last_error.as_deref().expect("error");
        assert!(error.starts_with("$ cargo test -p parser\nerror[E0308]"));
        assert_eq!(
            excerpt.fingerprint,
            fingerprint(&path).expect("fingerprint")
        );

        let text = render(&excerpt, &limits);
        assert!(text.len() <= limits.max_bytes);
        for sentinel in [
            "OKBODY-SENTINEL",
            "ATTACH-SENTINEL",
            "SIDECHAIN-SENTINEL",
            "META-SENTINEL",
            "FHS-SENTINEL",
            "THINK-SENTINEL",
            "ignored-by-reader",
        ] {
            assert!(
                !text.contains(sentinel),
                "{sentinel} leaked into the hand-off"
            );
        }
        for expected in [
            "# Hand-off from Claude Code",
            "- Directory: /work/repo",
            "- Branch: feature",
            "## Goal",
            "## Earlier summary",
            "### Turn 1 — user",
            "- /work/repo/src/parser.rs",
            "- `cargo test -p parser`",
            "error[E0308]",
        ] {
            assert!(text.contains(expected), "missing {expected}");
        }
    }

    #[test]
    fn claude_listing_reads_cwd_from_records_and_ignores_subagent_files() {
        let dir = tempfile::tempdir().expect("dir");
        let first = claude_fixture(dir.path());
        let colliding = dir
            .path()
            .join("projects/-work-repo-/2f6a1c2e-1111-4222-8333-444455557777.jsonl");
        write_lines(
            &colliding,
            &[
                json!({"type":"ai-title","aiTitle":"Second session"}),
                claude_user("u1", json!("Second goal")),
            ],
        );
        let subagent = first.with_extension("").join("subagents/agent-abc.jsonl");
        write_lines(&subagent, &[claude_user("u9", json!("SUBAGENT-SENTINEL"))]);
        std::fs::create_dir_all(dir.path().join("projects/-work-repo/memory")).expect("memory");
        std::fs::write(
            dir.path().join("projects/-work-repo/memory/notes.md"),
            "notes",
        )
        .expect("notes");
        std::fs::write(dir.path().join("projects/-work-repo/scratch.txt"), "{}").expect("txt");

        let listing = list(Kind::Claude, dir.path()).expect("list");
        assert_eq!(listing.skipped, 0);
        let mut ids: Vec<_> = listing
            .sessions
            .iter()
            .map(|s| s.native_id.as_str())
            .collect();
        ids.sort();
        assert_eq!(
            ids,
            vec![
                "0f6a1c2e-1111-4222-8333-444455556666",
                "2f6a1c2e-1111-4222-8333-444455557777"
            ]
        );
        assert!(
            listing
                .sessions
                .iter()
                .all(|s| s.cwd.as_deref() == Some(Path::new("/work/repo")))
        );
        let second = listing
            .sessions
            .iter()
            .find(|s| s.native_id.ends_with("7777"))
            .expect("second");
        assert_eq!(second.title.as_deref(), Some("Second session"));
        assert_eq!(
            locate(
                Kind::Claude,
                dir.path(),
                "2f6a1c2e-1111-4222-8333-444455557777"
            )
            .expect("locate"),
            colliding
        );
        assert!(locate(Kind::Claude, dir.path(), "../etc/passwd").is_err());
        assert!(locate(Kind::Claude, dir.path(), "missing").is_err());
        assert!(
            list(Kind::Claude, &dir.path().join("absent"))
                .expect("absent")
                .sessions
                .is_empty()
        );
    }

    fn codex_fixture(dir: &Path, id: &str) -> PathBuf {
        let path = dir.join(format!(
            "sessions/2026/10/20/rollout-2026-10-20T10-00-00-{id}.jsonl"
        ));
        write_lines(
            &path,
            &[
                json!({"timestamp":"2026-10-20T10:00:00.000Z","type":"turn_context",
                       "payload":{"turn_id":"t1","cwd":"/work/repo","model":"gpt-6-sol"}}),
                json!({"timestamp":"2026-10-20T10:00:00.000Z","type":"session_meta",
                       "payload":{"id":id,"timestamp":"2026-10-20T10:00:00.000Z","cwd":"/work/repo",
                                  "originator":"codex_cli_rs","cli_version":"0.160.0","source":"cli",
                                  "git":{"commit_hash":"abc123def456789","branch":"feature"}}}),
                json!({"type":"compacted","payload":{"message":"Earlier: set up the project."}}),
                json!({"type":"event_msg","payload":{"type":"user_message","message":"Make the parser accept trailing commas"}}),
                json!({"type":"response_item","payload":{"type":"message","role":"user",
                       "content":[{"type":"input_text","text":"# AGENTS.md instructions INJECT-SENTINEL"}]}}),
                json!({"type":"response_item","payload":{"type":"reasoning","encrypted_content":"REASONING-SENTINEL"}}),
                json!({"type":"response_item","payload":{"type":"function_call","name":"exec_command",
                       "arguments":"{\"cmd\":\"cargo test -p parser\"}","call_id":"c1"}}),
                json!({"type":"response_item","payload":{"type":"function_call_output","call_id":"c1","output":"OUTPUT-SENTINEL"}}),
                json!({"type":"event_msg","payload":{"type":"exec_command_end","command":["bash","-lc","cargo test -p parser"],
                       "exit_code":101,"aggregated_output":"error[E0308]: mismatched types\nline two"}}),
                json!({"type":"response_item","payload":{"type":"custom_tool_call","name":"apply_patch",
                       "input":"*** Begin Patch\n*** Update File: src/parser.rs\n@@\n-x\n+y\n*** End Patch","call_id":"c2"}}),
                json!({"type":"event_msg","payload":{"type":"agent_message","message":"I edited the parser and ran the tests."}}),
                json!({"type":"response_item","payload":{"type":"message","role":"assistant","phase":"final_answer",
                       "content":[{"type":"output_text","text":"I edited the parser and ran the tests."}]}}),
            ],
        );
        let mut text = std::fs::read_to_string(&path).expect("read");
        text.push_str("not json at all\n");
        std::fs::write(&path, text).expect("malformed line");
        path
    }

    #[test]
    fn codex_rollouts_describe_the_thread_and_excerpt_only_text() {
        let dir = tempfile::tempdir().expect("dir");
        let id = "019a6c2e-1111-7222-8333-444455556666";
        let path = codex_fixture(dir.path(), id);
        let session = describe(Kind::Codex, &path).expect("describe");
        assert_eq!(session.native_id, id);
        assert_eq!(session.cwd.as_deref(), Some(Path::new("/work/repo")));
        assert_eq!(session.branch.as_deref(), Some("feature"));
        assert_eq!(session.commit.as_deref(), Some("abc123def456789"));
        assert_eq!(session.model.as_deref(), Some("gpt-6-sol"));
        assert_eq!(
            session.title.as_deref(),
            Some("Make the parser accept trailing commas")
        );

        let limits = Limits::default();
        let excerpt = excerpt(Kind::Codex, &path, &limits).expect("excerpt");
        assert_eq!(
            excerpt.goal.as_deref(),
            Some("Make the parser accept trailing commas")
        );
        assert_eq!(excerpt.turns.len(), 1);
        assert_eq!(
            excerpt.turns[0].answer,
            "I edited the parser and ran the tests."
        );
        assert_eq!(excerpt.files, vec!["src/parser.rs"]);
        assert_eq!(excerpt.commands, vec!["cargo test -p parser"]);
        assert!(
            excerpt
                .last_error
                .as_deref()
                .expect("error")
                .contains("error[E0308]")
        );
        assert!(
            excerpt
                .last_error
                .as_deref()
                .expect("error")
                .starts_with("$ bash -lc cargo test -p parser")
        );
        // The compaction summary stands in for the history before it; the turn after it stays.
        assert_eq!(
            excerpt.summary.as_deref(),
            Some("Earlier: set up the project.")
        );
        assert!(!excerpt.truncated);

        let text = render(&excerpt, &limits);
        for sentinel in ["INJECT-SENTINEL", "REASONING-SENTINEL", "OUTPUT-SENTINEL"] {
            assert!(
                !text.contains(sentinel),
                "{sentinel} leaked into the hand-off"
            );
        }
        assert!(text.contains("# Hand-off from Codex"));
        assert!(text.contains("(commit abc123def456)"));

        let listing = list(Kind::Codex, dir.path()).expect("list");
        assert_eq!(listing.sessions.len(), 1);
        assert_eq!(listing.skipped, 0);
        assert_eq!(locate(Kind::Codex, dir.path(), id).expect("locate"), path);
    }

    #[test]
    fn unreadable_rollouts_are_counted_not_fatal() {
        let dir = tempfile::tempdir().expect("dir");
        codex_fixture(dir.path(), "019a6c2e-1111-7222-8333-444455556666");
        let compressed = dir
            .path()
            .join("sessions/2026/10/21/rollout-2026-10-21T10-00-00-019a6c2e-1111-7222-8333-444455559999.jsonl");
        std::fs::create_dir_all(compressed.parent().expect("parent")).expect("dir");
        std::fs::write(&compressed, [0x1f, 0x8b, 0x08, 0x00]).expect("gzip bytes");
        std::fs::write(
            dir.path()
                .join("sessions/2026/10/21/rollout-decoy.jsonl.zst"),
            [0x28, 0xb5, 0x2f, 0xfd],
        )
        .expect("zstd bytes");
        let listing = list(Kind::Codex, dir.path()).expect("list");
        assert_eq!(listing.sessions.len(), 1);
        assert_eq!(listing.skipped, 1);
        assert!(
            describe(Kind::Codex, &compressed)
                .expect_err("compressed")
                .contains("Compressed rollouts are not supported")
        );
    }

    #[test]
    fn long_transcripts_read_only_the_head_and_tail_windows() {
        let dir = tempfile::tempdir().expect("dir");
        let path = dir
            .path()
            .join("projects/-work-repo/aaaaaaaa-0000-4000-8000-000000000000.jsonl");
        let mut lines = vec![
            json!({"type":"queue-operation","operation":"enqueue"}),
            claude_user("u0", json!("GOAL-FIRST-PROMPT")),
        ];
        for index in 0..40_000 {
            lines.push(claude_user(
                &format!("u{index}"),
                json!(format!("FILLER-SENTINEL-{index:05} {}", "x".repeat(80))),
            ));
            lines.push(claude_assistant(
                &format!("m{index}"),
                json!({"type":"text","text":format!("FILLER-ANSWER-{index:05}")}),
            ));
        }
        lines.push(claude_user("last", json!("LAST-PROMPT")));
        lines.push(claude_assistant(
            "last",
            json!({"type":"text","text":"LAST-ANSWER"}),
        ));
        write_lines(&path, &lines);
        let limits = Limits::default();
        assert!(
            fingerprint(&path).expect("fingerprint").len > limits.head_bytes + limits.tail_bytes
        );

        let excerpt = excerpt(Kind::Claude, &path, &limits).expect("excerpt");
        assert_eq!(excerpt.goal.as_deref(), Some("GOAL-FIRST-PROMPT"));
        assert!(excerpt.truncated);
        assert_eq!(excerpt.turns.len(), limits.turns);
        let last = excerpt.turns.last().expect("last turn");
        assert_eq!(last.prompt, "LAST-PROMPT");
        assert_eq!(last.answer, "LAST-ANSWER");
        let text = render(&excerpt, &limits);
        assert!(text.len() <= limits.max_bytes);
        assert!(!text.contains("FILLER-SENTINEL-00001"));
        assert!(text.contains("LAST-ANSWER"));

        let before = fingerprint(&path).expect("before");
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open");
        std::io::Write::write_all(&mut file, b"{\"type\":\"user\"}\n").expect("append");
        assert_ne!(before, fingerprint(&path).expect("after"));
    }

    #[test]
    fn rendering_caps_the_size_at_a_character_boundary() {
        let limits = Limits {
            max_bytes: 400,
            ..Limits::default()
        };
        let session = Session {
            kind: Kind::Codex,
            native_id: "id".into(),
            transcript: PathBuf::from("/tmp/t.jsonl"),
            cwd: None,
            branch: None,
            commit: None,
            started: None,
            last_activity: None,
            title: None,
            model: None,
            bytes: 0,
        };
        let excerpt = Excerpt {
            session,
            goal: Some("é".repeat(600)),
            summary: None,
            turns: Vec::new(),
            files: Vec::new(),
            commands: vec!["echo `x`".into()],
            last_error: None,
            notes: vec!["note".into()],
            truncated: false,
            fingerprint: Fingerprint {
                len: 0,
                modified_ms: 0,
            },
            extracted_at: String::new(),
        };
        let text = render(&excerpt, &limits);
        assert!(text.len() <= limits.max_bytes);
        assert!(text.ends_with("_(hand-off cut at the size limit)_\n"));
        assert_eq!(
            patch_files("*** Add File: a.rs\n*** Move to: b.rs\n*** Delete File: \n"),
            vec!["a.rs", "b.rs"]
        );
        assert_eq!(first_lines("a\nb\nc", 2), ("a\nb\n…".to_string(), true));
        assert_eq!(truncate("héllo", 4), ("h…".to_string(), true));
        assert!(truncate(&"é".repeat(10), 7).0.len() <= 7);
        assert_eq!(truncate("abc", 2), (String::new(), true));
        assert!(looks_like_uuid("019a6c2e-1111-7222-8333-444455556666"));
        assert!(!looks_like_uuid("rollout-2026"));
    }
}
