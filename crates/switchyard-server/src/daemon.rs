// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Detached execution for `switchyard-server`.
//!
//! The server is otherwise a foreground process that drains on `SIGTERM`/
//! `SIGINT`. To run it as a managed background service that outlives the
//! launching terminal, call [`detach_into_background`] *before* the Tokio
//! runtime does significant work: it re-executes the current binary under the
//! system `setsid` in a new session with stdio disconnected, so the original
//! process exits and the child keeps serving. Spawning before the async
//! runtime boots avoids the hazard of `fork`/`setsid` after an OS-thread/
//! signal-handler runtime has initialised.
//!
//! The re-exec drops `--detach` so the child serves normally instead of
//! detaching again (which would re-exec repeatedly). The server process writes
//! its own real PID from inside the detached child, because `setsid` forks and
//! the spawned wrapper's PID is not the server's.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Internal env marker: present only on the detached child, signalling it to
/// record its own PID. A normal foreground run never sees it, so it never
/// writes a pidfile unexpectedly.
const DETACH_ENV: &str = "SWITCHYARD_SERVER_DETACHED";

/// Internal env carrying the pidfile path from parent to detached child, since
/// `--pidfile` is stripped from the re-executed arguments.
const PIDFILE_ENV: &str = "SWITCHYARD_SERVER_PIDFILE";

/// Re-exec the current binary under `setsid` in a detached background session.
///
/// Returns `Ok(())` in the detached child (the caller should then boot the
/// server); the parent process exits successfully after spawning the child.
#[cfg(unix)]
pub(crate) fn detach_into_background(pidfile: &Path) -> std::io::Result<()> {
    let current = std::env::current_exe()?;
    // Re-exec with `--detach` (and `--pidfile`) removed so the child parses a
    // normal foreground invocation; pass the pidfile path via the environment
    // instead so the child can record its own real PID.
    let mut command = Command::new("setsid");
    command.arg(&current).args(detached_args());
    command
        .env(DETACH_ENV, "1")
        .env(PIDFILE_ENV, pidfile)
        // Disconnect stdio so the child is independent of the launching
        // terminal.
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    // Parent spawns the setsid wrapper and exits; `setsid` forks the actual
    // server session, which re-enters `main` without `--detach`.
    let _ = command.spawn()?;
    std::process::exit(0);
}

/// Original CLI arguments with `--detach`/`--pidfile` stripped, so the
/// re-executed child behaves as a normal foreground server. The pidfile path
/// travels via [`PIDFILE_ENV`] instead.
fn detached_args() -> Vec<String> {
    let mut out = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--detach" || arg.starts_with("--detach=") {
            continue;
        }
        if arg == "--pidfile" {
            // Drop the flag and its value; the path is carried via env.
            let _ = args.next();
            continue;
        }
        if arg.starts_with("--pidfile=") {
            continue;
        }
        out.push(arg);
    }
    out
}

/// If this process is the detached child, return the pidfile path it should
/// record itself in. Returns `None` for a normal foreground run.
pub(crate) fn detached_pidfile() -> Option<PathBuf> {
    if std::env::var_os(DETACH_ENV).is_some() {
        std::env::var_os(PIDFILE_ENV).map(PathBuf::from)
    } else {
        None
    }
}

/// Write `pid` to `path`, creating parent directories as needed.
///
/// Refuses to follow a symlink at `path` and refuses to overwrite an existing
/// pidfile, so a pre-placed symlink or another process's pidfile cannot be
/// clobbered.
pub(crate) fn write_pidfile(path: &Path, pid: u32) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    // Reject a symlink at the target so we never write through it.
    #[cfg(unix)]
    if let Ok(meta) = std::fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "pidfile path is a symlink",
            ));
        }
    }
    // create_new => O_CREAT|O_EXCL: fail if the file already exists, avoiding
    // clobbering another process's pidfile.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    writeln!(file, "{pid}")?;
    file.flush()?;
    Ok(())
}
