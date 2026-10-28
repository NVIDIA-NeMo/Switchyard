// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A hand-off is the text a destination harness reads to continue a session that another
//! harness started. Each one lives in its own directory under `~/.switchyard/handoffs`: the
//! Markdown the destination reads, a record of where it came from and where it went, and the
//! destination's private harness settings. Hand-offs hold conversation excerpts, so the files
//! are private to the user and old ones are removed.

use serde_json::Value;
use std::path::{Path, PathBuf};

/// Hand-offs older than this are removed when the app starts.
pub const RETENTION_DAYS: i64 = 7;
pub const FILE: &str = "HANDOFF.md";
pub const RECORD: &str = "handoff.json";

pub fn root() -> Result<PathBuf, String> {
    crate::config::expand_home(Path::new("~/.switchyard/handoffs"))
}

pub fn new_id() -> String {
    format!("hd-{}", uuid::Uuid::now_v7())
}

fn validate_id(id: &str) -> Result<(), String> {
    let suffix = id.strip_prefix("hd-").ok_or("Choose a hand-off.")?;
    uuid::Uuid::parse_str(suffix).map_err(|_| "Choose a hand-off.")?;
    Ok(())
}

/// The directory of one hand-off. It must be a directory the app created, not a symlink.
pub fn directory(id: &str) -> Result<PathBuf, String> {
    validate_id(id)?;
    let path = root()?.join(id);
    if std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err("Hand-off directories must not be symlinks.".into());
    }
    Ok(path)
}

fn private_dir(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|error| format!("Create {}: {error}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn private_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|error| format!("Write {}: {error}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// This writes a hand-off's directory, Markdown, and record, and returns the directory.
pub fn create(id: &str, markdown: &str, record: &Value) -> Result<PathBuf, String> {
    let root = root()?;
    if std::fs::symlink_metadata(&root).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(format!(
            "{} is a symlink. Edit its target yourself.",
            root.display()
        ));
    }
    private_dir(&root)?;
    let dir = directory(id)?;
    private_dir(&dir)?;
    private_file(&dir.join(FILE), markdown.as_bytes())?;
    update(id, record)?;
    Ok(dir)
}

/// This replaces a hand-off's record.
pub fn update(id: &str, record: &Value) -> Result<(), String> {
    let dir = directory(id)?;
    private_file(
        &dir.join(RECORD),
        &serde_json::to_vec_pretty(record).map_err(|error| error.to_string())?,
    )
}

/// This lists the recorded hand-offs, newest first. Unreadable records are skipped.
pub fn list() -> Result<Vec<Value>, String> {
    let root = root()?;
    let entries = match std::fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("Read {}: {error}", root.display())),
    };
    let mut records = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if validate_id(&name).is_err() || directory(&name).is_err() {
            continue;
        }
        let Ok(bytes) = std::fs::read(entry.path().join(RECORD)) else {
            continue;
        };
        if let Ok(record) = serde_json::from_slice::<Value>(&bytes)
            && record.is_object()
            && record["id"] == name
        {
            records.push(record);
        }
    }
    records.sort_by(|left, right| {
        right["created_at"]
            .as_str()
            .cmp(&left["created_at"].as_str())
    });
    Ok(records)
}

/// This removes hand-offs created more than `RETENTION_DAYS` before `now` and returns how
/// many it removed. A hand-off without a readable record is removed by its directory age.
pub fn purge(now: chrono::DateTime<chrono::Utc>) -> Result<usize, String> {
    let root = root()?;
    let entries = match std::fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(format!("Read {}: {error}", root.display())),
    };
    let cutoff = now - chrono::Duration::days(RETENTION_DAYS);
    let mut removed = 0;
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let Ok(dir) = directory(&name) else {
            continue;
        };
        let Ok(metadata) = std::fs::symlink_metadata(&dir) else {
            continue;
        };
        if !metadata.is_dir() {
            continue;
        }
        let created = std::fs::read(dir.join(RECORD))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .and_then(|record| {
                record["created_at"]
                    .as_str()
                    .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
                    .map(|ts| ts.with_timezone(&chrono::Utc))
            })
            .or_else(|| {
                metadata
                    .modified()
                    .ok()
                    .map(chrono::DateTime::<chrono::Utc>::from)
            });
        if created.is_some_and(|created| created < cutoff) {
            std::fs::remove_dir_all(&dir).map_err(|error| error.to_string())?;
            removed += 1;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // The subprocess isolates HOME so the test never touches the user's hand-offs.
    #[cfg(unix)]
    #[test]
    fn handoffs_are_private_listed_newest_first_and_purged_by_age() {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(home) = std::env::var("SWITCHYARD_HANDOFF_FIXTURE") {
            let home = PathBuf::from(home);
            let old = new_id();
            let recent = new_id();
            let created = |ts: &str, id: &str| json!({"id": id, "created_at": ts});
            let dir = create(&old, "# old", &created("2026-10-01T00:00:00Z", &old)).expect("old");
            assert_eq!(
                std::fs::metadata(&dir).expect("dir").permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(
                std::fs::metadata(dir.join(FILE))
                    .expect("file")
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
            create(
                &recent,
                "# recent",
                &created("2026-10-25T00:00:00Z", &recent),
            )
            .expect("recent");
            std::fs::create_dir(home.join(".switchyard/handoffs/not-a-handoff")).expect("decoy");
            std::fs::write(home.join(".switchyard/handoffs/hd-decoy"), "file").expect("decoy");
            let listed = list().expect("list");
            assert_eq!(
                listed
                    .iter()
                    .map(|r| r["id"].as_str().unwrap())
                    .collect::<Vec<_>>(),
                vec![recent.as_str(), old.as_str()]
            );
            assert!(directory("../etc").is_err());
            let now = chrono::DateTime::parse_from_rfc3339("2026-10-26T00:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc);
            assert_eq!(purge(now).expect("purge"), 1);
            assert!(!root().unwrap().join(&old).exists());
            assert!(root().unwrap().join(&recent).join(FILE).exists());
            assert!(home.join(".switchyard/handoffs/not-a-handoff").exists());
            return;
        }
        let home = tempfile::tempdir().expect("home");
        let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "handoffs::tests::handoffs_are_private_listed_newest_first_and_purged_by_age",
            ])
            .env("SWITCHYARD_HANDOFF_FIXTURE", home.path())
            .env("HOME", home.path())
            .output()
            .expect("child");
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
