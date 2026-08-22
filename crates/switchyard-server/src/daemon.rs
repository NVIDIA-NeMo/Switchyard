// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Detached execution for `switchyard-server`.
//!
//! The server is otherwise a foreground process that drains on `SIGTERM`/
//! `SIGINT`. To run it as a managed background service that outlives the
//! launching terminal, call [`detach_into_background`] *before* the Tokio
//! runtime does significant work: it re-executes the current binary under the
//! system `setsid` in a new session with stdio disconnected and writes a
//! pidfile, so the original process exits and the child keeps serving. Spawning
//! before the async runtime boots avoids the hazard of `fork`/`setsid` after an
//! OS-thread/signal-handler runtime has initialised.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// Re-exec the current binary under `setsid` in a detached background session.
///
/// Returns `Ok(())` in the detached child (the caller should then boot the
/// server); the parent process exits successfully after spawning the child.
#[cfg(unix)]
pub(crate) fn detach_into_background(pidfile: &Path) -> std::io::Result<()> {
    let current = std::env::current_exe()?;
    // Use the system `setsid` to spawn a detached session (stable, no unstable
    // std features). stdio is disconnected so the child is independent of the
    // launching terminal. Drop `--detach` from the re-exec args so the child
    // serves normally instead of recursing into another detach.
    let mut child = Command::new("setsid");
    child.arg(&current);
    for arg in std::env::args().skip(1) {
        if arg != "--detach" {
            child.arg(arg);
        }
    }
    child.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());

    let handle = child.spawn()?;
    write_pidfile(pidfile, handle.id())?;
    // Parent exits; the detached child continues and serves in its own session.
    std::process::exit(0);
}

/// Write `pid` to `path`, creating parent directories as needed.
pub(crate) fn write_pidfile(path: &Path, pid: u32) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "{pid}")?;
    file.flush()?;
    Ok(())
}
