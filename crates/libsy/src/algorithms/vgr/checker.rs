// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A checker that runs a task's own tests against a pinned, verified snapshot.
//!
//! This is the evidence source the checks regime commits on. It executes an
//! operator-supplied command and reports the exit code, but the part that makes
//! its verdict worth trusting is the manifest: the test suite is copied into a
//! private snapshot at construction, hashed file by file, and re-verified both
//! before the run and again before any pass is reported.
//!
//! # What this owns, and what it does not
//!
//! Resource limits, network isolation and filesystem confinement are **not**
//! implemented here. They belong to the deployment sandbox the router and the
//! agent both run inside, which can enforce them at the kernel level; a
//! same-uid child process cannot meaningfully confine itself. The reference
//! implementation reaches the same conclusion and says so explicitly.
//!
//! What no deployment sandbox can provide is the manifest. The attempt runs as
//! the same user, inside the same sandbox, with the same view of the filesystem
//! as the tests it is judged by — so nothing but re-hashing stops an attempt
//! from editing those tests and passing. That is this module's job.
//!
//! # Fail-closed
//!
//! Only a clean exit with the manifest verified twice reports a pass. A
//! non-zero exit is a fail. Everything else — a command that could not be
//! spawned, a snapshot that no longer matches, a run the caller's deadline cut
//! short — is indeterminate, which commits nothing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tempfile::TempDir;

use super::config::Checker;

/// Characters of the attempt and task written into the run directory.
///
/// The command reads them from files rather than argv, so a large attempt
/// cannot overflow the argument list.
const ATTEMPT_FILE: &str = "attempt.txt";
const TASK_FILE: &str = "task.txt";

/// The operator's attestation that a deployment sandbox confines this checker.
///
/// Required verbatim, because everything this module does not enforce — memory,
/// CPU, network, filesystem reach — is only enforced if something else is doing
/// it. Making that an explicit claim keeps a checker from being configured on a
/// bare host under the impression it is contained.
pub const SANDBOX_ATTESTATION: &str = "vgr-checker-runs-in-deployment-sandbox";

/// How an operator configures the checker.
#[derive(Clone, Debug)]
pub struct CheckerConfig {
    /// Directory holding the task's test suite. Copied at construction.
    pub tests_dir: PathBuf,
    /// The command to run, as an argv list. Never a shell string.
    ///
    /// Two placeholders are substituted per run: `{tests}` becomes the pinned
    /// snapshot path, `{workdir}` the per-run directory holding the attempt.
    pub command: Vec<String>,
    /// How long one run may take before it is abandoned.
    pub timeout: Duration,
    /// Extra environment entries, applied over the scrubbed base.
    pub env: Vec<(String, String)>,
    /// The operator's [`SANDBOX_ATTESTATION`], recorded verbatim.
    pub sandbox_attestation: String,
}

impl CheckerConfig {
    /// A configuration running `command` against the suite in `tests_dir`.
    pub fn new(tests_dir: impl Into<PathBuf>, command: Vec<String>) -> Self {
        Self {
            tests_dir: tests_dir.into(),
            command,
            timeout: Duration::from_secs(120),
            env: Vec::new(),
            sandbox_attestation: String::new(),
        }
    }
}

/// Why a checker could not be built.
#[derive(Debug, thiserror::Error)]
pub enum CheckerSetupError {
    /// The command was empty, so there is nothing to run.
    #[error("checker command must be a non-empty argv list")]
    EmptyCommand,
    /// The operator did not attest that a deployment sandbox confines this checker.
    #[error("checker requires the sandbox attestation {SANDBOX_ATTESTATION:?}")]
    NotAttested,
    /// The test suite could not be snapshotted.
    #[error("checker could not snapshot its test suite: {0}")]
    Snapshot(#[source] std::io::Error),
    /// The suite holds something the manifest cannot represent.
    #[error(
        "checker test suite entry {0:?} is neither a regular file nor a directory, \
         so it cannot be snapshotted or hashed"
    )]
    UnsupportedEntry(PathBuf),
}

/// The pinned record of what the checker will run, and against what.
///
/// Its `sha` covers every test file's path and contents together with the
/// command, so a decision can record exactly which suite licensed it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    /// Snapshot-relative path to the SHA-256 of that file's contents.
    files: BTreeMap<String, String>,
    /// One hash over the whole suite and the command.
    pub sha: String,
}

/// Runs a task's tests against a pinned, hash-verified snapshot.
///
/// Named for what it actually guarantees. It does not sandbox: confinement is
/// the deployment sandbox's job, and a type named for a property it does not
/// enforce would be read as a safety claim it cannot honour.
pub struct PinnedChecker {
    config: CheckerConfig,
    /// The private snapshot. Held so it outlives every run and is removed on drop.
    snapshot: TempDir,
    manifest: Manifest,
}

impl PinnedChecker {
    /// Snapshots the test suite and pins its manifest.
    ///
    /// The snapshot is taken once, at construction, so that later edits to the
    /// operator's original directory cannot change what is being verified
    /// against mid-flight.
    pub fn new(config: CheckerConfig) -> Result<Self, CheckerSetupError> {
        if config.command.is_empty() {
            return Err(CheckerSetupError::EmptyCommand);
        }
        if config.sandbox_attestation != SANDBOX_ATTESTATION {
            return Err(CheckerSetupError::NotAttested);
        }
        let snapshot = TempDir::with_prefix("vgr-checker-").map_err(CheckerSetupError::Snapshot)?;
        let tests_path = snapshot.path().join("tests");
        copy_tree(&config.tests_dir, &tests_path)?;
        // Read-only is a courtesy, not the control: the same uid can undo it.
        // The manifest re-check is what actually catches a modified suite.
        set_read_only(&tests_path).map_err(CheckerSetupError::Snapshot)?;

        let files = hash_tree(&tests_path).map_err(CheckerSetupError::Snapshot)?;
        let manifest = Manifest {
            sha: manifest_sha(&files, &config.command),
            files,
        };
        Ok(Self {
            config,
            snapshot,
            manifest,
        })
    }

    /// The pinned manifest this checker verifies against.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The snapshot the tests were pinned into.
    fn tests_path(&self) -> PathBuf {
        self.snapshot.path().join("tests")
    }

    /// Whether the snapshot still matches what was pinned.
    fn verifies(&self, stage: &'static str) -> bool {
        match hash_tree(&self.tests_path()) {
            Ok(current) if current == self.manifest.files => true,
            Ok(_) => {
                tracing::warn!(
                    target: "libsy",
                    stage,
                    "vgr checker test snapshot no longer matches its pinned manifest"
                );
                false
            }
            Err(error) => {
                tracing::warn!(
                    target: "libsy",
                    stage,
                    kind = ?error.kind(),
                    "vgr checker could not re-hash its test snapshot"
                );
                false
            }
        }
    }

    /// Runs the command once and reports the exit status.
    async fn run(&self, task_text: &str, attempt: &str) -> std::io::Result<bool> {
        // A fresh directory per run, removed when this returns, so one run
        // cannot see or corrupt another's working state.
        let workdir = TempDir::with_prefix("vgr-checker-run-")?;
        let work = workdir.path();
        std::fs::write(work.join(ATTEMPT_FILE), attempt)?;
        std::fs::write(work.join(TASK_FILE), task_text)?;

        let tests = self.tests_path();
        let substituted: Vec<String> = self
            .config
            .command
            .iter()
            .map(|argument| {
                argument
                    .replace("{tests}", &tests.to_string_lossy())
                    .replace("{workdir}", &work.to_string_lossy())
            })
            .collect();
        let (program, arguments) = substituted
            .split_first()
            .ok_or_else(|| std::io::Error::other("checker command is empty"))?;

        let mut command = tokio::process::Command::new(program);
        command
            .args(arguments)
            .current_dir(work)
            // A whitelist, not the router's environment: the child has no
            // reason to inherit credentials the router holds.
            .env_clear()
            .env(
                "PATH",
                std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin".into()),
            )
            .env("HOME", work)
            .env("TMPDIR", work)
            .env("LANG", "C.UTF-8")
            .env("TESTS_DIR", &tests)
            .env("ATTEMPT_FILE", work.join(ATTEMPT_FILE))
            .env("TASK_FILE", work.join(TASK_FILE))
            .stdin(std::process::Stdio::null())
            // Discarded rather than captured. Output derived from the attempt
            // must not reach the router's logs, and nothing reads it: the
            // verdict is the exit status alone.
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            // Killed if this future is dropped, which is how the router's
            // deadline stops a run rather than merely stopping waiting for one.
            .kill_on_drop(true)
            // Its own process group, so the whole tree can be signalled. A test
            // suite is a process *tree* — a runner that forks workers is the
            // normal case, not the exotic one — and killing only the direct
            // child would leave those workers running on the host after the
            // router has stopped waiting for them.
            .process_group(0);
        for (key, value) in &self.config.env {
            command.env(key, value);
        }

        let mut child = command.spawn()?;
        // Read before waiting: the id is gone once the child is reaped, and the
        // group must still be reachable at that point. `process_group(0)` makes
        // the child its own group leader, so its pid is the group id.
        let group = child.id();
        // Fires however the run ends — clean exit, timeout, or this future being
        // dropped by the router's deadline — so nothing the suite spawned
        // outlives the decision it was gathering evidence for.
        let _reaper = ProcessGroupReaper(group);

        let status = tokio::time::timeout(self.config.timeout, child.wait())
            .await
            .map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::TimedOut, "checker timed out")
            })??;

        // Only the exit status is recorded. The command's output is a function of
        // the attempt and the task, so it is conversation content by another
        // route — a suite that printed the attempt would put it in the router's
        // logs. Nothing here is worth that, since the verdict is the exit code.
        tracing::debug!(
            target: "libsy",
            status = status.code(),
            "vgr checker run"
        );
        Ok(status.success())
    }
}

#[async_trait::async_trait]
impl Checker for PinnedChecker {
    async fn check(&self, task_text: &str, attempt: &str) -> Option<bool> {
        // Before: the suite must be what was pinned, or there is nothing
        // trustworthy to run.
        if !self.verifies("pre-run") {
            return None;
        }
        let passed = match self.run(task_text, attempt).await {
            Ok(passed) => passed,
            Err(error) => {
                tracing::warn!(
                    target: "libsy",
                    kind = ?error.kind(),
                    "vgr checker run did not complete"
                );
                return None;
            }
        };
        // After: a pass certifies the tests were byte-identical before *and*
        // after, so an attempt that edited them to make them pass reports
        // nothing rather than success. A failing run is still a fail — only a
        // pass needs certifying.
        if passed && !self.verifies("post-run") {
            return None;
        }
        Some(passed)
    }
}

/// Signals a checker run's whole process group when the run goes out of scope.
///
/// The router can cancel a decision at any point, and a suite that forked
/// workers would otherwise leave them running: `kill_on_drop` reaches the direct
/// child only. Being able to stop work it started is the router's own
/// responsibility, and is separate from confining that work, which is the
/// deployment sandbox's.
struct ProcessGroupReaper(Option<u32>);

impl Drop for ProcessGroupReaper {
    fn drop(&mut self) {
        let Some(group) = self.0 else {
            return;
        };
        // Best effort, and deliberately not through a new dependency: sending a
        // signal to a process group needs `libc::kill` with a negative pid,
        // which is an `unsafe` call this repository has no precedent for in
        // production code. `kill` is POSIX and reaches the same syscall.
        //
        // Failure is ignored because there is nothing useful to do about it and
        // the common case is benign: the group is already empty because the
        // suite exited cleanly and was reaped.
        let _ = std::process::Command::new("kill")
            .args(["-KILL", "--", &format!("-{group}")])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}

/// Copies `from` to `to`, creating directories as needed.
///
/// Only regular files and directories are representable in the manifest, so
/// anything else is an error rather than an omission.
fn copy_tree(from: &Path, to: &Path) -> Result<(), CheckerSetupError> {
    std::fs::create_dir_all(to).map_err(CheckerSetupError::Snapshot)?;
    for entry in std::fs::read_dir(from).map_err(CheckerSetupError::Snapshot)? {
        let entry = entry.map_err(CheckerSetupError::Snapshot)?;
        let kind = entry.file_type().map_err(CheckerSetupError::Snapshot)?;
        let target = to.join(entry.file_name());
        if kind.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if kind.is_file() {
            std::fs::copy(entry.path(), &target).map_err(CheckerSetupError::Snapshot)?;
        } else {
            // Symlinks, sockets, devices and FIFOs. Skipping them silently would
            // snapshot an incomplete suite that still passes, and following a
            // symlink would let the manifest hash content living outside the
            // snapshot that can change underneath it. Neither is safe, so an
            // unrepresentable suite is refused rather than approximated.
            return Err(CheckerSetupError::UnsupportedEntry(entry.path()));
        }
    }
    Ok(())
}

/// Marks every file in the tree read-only.
fn set_read_only(root: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            set_read_only(&entry.path())?;
        } else if kind.is_file() {
            let mut permissions = entry.metadata()?.permissions();
            permissions.set_readonly(true);
            std::fs::set_permissions(entry.path(), permissions)?;
        }
    }
    Ok(())
}

/// Hashes every file in the tree, keyed by its path relative to `root`.
///
/// Ordered, so two trees with the same contents always produce the same map
/// and the same manifest hash.
fn hash_tree(root: &Path) -> std::io::Result<BTreeMap<String, String>> {
    let mut hashed = BTreeMap::new();
    hash_into(root, root, &mut hashed)?;
    Ok(hashed)
}

fn hash_into(
    root: &Path,
    current: &Path,
    hashed: &mut BTreeMap<String, String>,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(current)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_dir() {
            hash_into(root, &path, hashed)?;
        } else if kind.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| std::io::Error::other("snapshot entry escaped its root"))?
                .to_string_lossy()
                .into_owned();
            let contents = std::fs::read(&path)?;
            let digest = hex(ring::digest::digest(&ring::digest::SHA256, &contents));
            let stamp = inode_stamp(&entry.metadata()?);
            hashed.insert(relative, format!("{digest}{stamp}"));
        }
    }
    Ok(())
}

/// One hash over the whole suite and the command it will be run with.
///
/// Lengths are folded in alongside the values so that no two different suites
/// can produce the same digest by shifting a boundary between fields.
fn manifest_sha(files: &BTreeMap<String, String>, command: &[String]) -> String {
    let mut context = ring::digest::Context::new(&ring::digest::SHA256);
    for (path, digest) in files {
        context.update(&(path.len() as u64).to_le_bytes());
        context.update(path.as_bytes());
        context.update(digest.as_bytes());
    }
    for argument in command {
        context.update(&(argument.len() as u64).to_le_bytes());
        context.update(argument.as_bytes());
    }
    hex(context.finish())
}

/// Identity of a file beyond its contents: inode, size, and inode-change time.
///
/// Content hashing alone cannot see a suite that was edited, run against, and
/// restored to its original bytes — both hashes match and the run passes. `ctime`
/// closes that: the kernel updates it on every write and on every metadata
/// change, and unlike `mtime` it cannot be set from userspace, so restoring
/// content or backdating with `touch` does not hide the edit. The inode catches
/// a file replaced wholesale rather than modified in place.
///
/// Preventing the edit outright would need a read-only bind mount, which needs a
/// user namespace the router cannot rely on having. Detecting it is enough here,
/// because an unverifiable snapshot yields no verdict and escalates.
#[cfg(unix)]
fn inode_stamp(metadata: &std::fs::Metadata) -> String {
    use std::os::unix::fs::MetadataExt;
    format!(
        ":{}:{}:{}.{}",
        metadata.ino(),
        metadata.size(),
        metadata.ctime(),
        metadata.ctime_nsec()
    )
}

/// Contents alone off Unix, where there is no `ctime` to consult.
#[cfg(not(unix))]
fn inode_stamp(_metadata: &std::fs::Metadata) -> String {
    String::new()
}

/// Renders a digest as lower-case hex.
fn hex(digest: ring::digest::Digest) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests;
