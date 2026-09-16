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
use std::fmt;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tempfile::TempDir;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use self::platform::{
    MutationStamp, ProcessTreeReaper, configure_process_tree, is_directory, is_regular_file,
    mutation_stamp_file, mutation_stamp_path, open_regular_file, os_str_bytes,
};
use super::config::{Checker, CheckerRequest};

mod platform;

/// Characters of the attempt and task written into private control files.
///
/// The command reads them from files rather than argv, so a large attempt
/// cannot overflow the argument list.
const ATTEMPT_FILE: &str = "attempt.txt";
const TASK_FILE: &str = "task.txt";
const WORKSPACE_ENV: &str = "WORKSPACE_DIR";
const HASH_CHUNK_BYTES: usize = 64 * 1024;
const DEFAULT_MAX_ENTRIES: usize = 100_000;
const DEFAULT_MAX_BYTES: u64 = 1024 * 1024 * 1024;
#[cfg(unix)]
const PROTECTED_ENV: [&str; 8] = [
    "TESTS_DIR",
    "ATTEMPT_FILE",
    "TASK_FILE",
    "HOME",
    "TMPDIR",
    "PATH",
    "LANG",
    WORKSPACE_ENV,
];
#[cfg(windows)]
const WINDOWS_HOST_ENV: [&str; 5] = ["SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "PSMODULEPATH"];
#[cfg(windows)]
const PROTECTED_ENV: [&str; 16] = [
    "TESTS_DIR",
    "ATTEMPT_FILE",
    "TASK_FILE",
    "HOME",
    "TMPDIR",
    "PATH",
    "LANG",
    WORKSPACE_ENV,
    "TEMP",
    "TMP",
    "USERPROFILE",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "PSMODULEPATH",
];

/// The operator's attestation that a deployment sandbox confines this checker.
///
/// Required verbatim, because everything this module does not enforce — memory,
/// CPU, network, filesystem reach — is only enforced if something else is doing
/// it. Making that an explicit claim keeps a checker from being configured on a
/// bare host under the impression it is contained.
pub const SANDBOX_ATTESTATION: &str = "vgr-checker-runs-in-deployment-sandbox";

/// An RAII-owned candidate source tree materialized for one checker run.
pub struct CandidateWorkspace {
    root: TempDir,
}

impl CandidateWorkspace {
    /// Owns a newly materialized workspace until the checker run finishes.
    pub fn new(root: TempDir) -> std::io::Result<Self> {
        let metadata = std::fs::symlink_metadata(root.path())?;
        if !is_directory(&metadata) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "candidate workspace root is not a directory",
            ));
        }
        Ok(Self { root })
    }

    /// The candidate source directory used as the checker's working directory.
    pub fn path(&self) -> &Path {
        self.root.path()
    }
}

/// Trusted host materialization for the candidate source tree under test.
///
/// Implementations may interpret the candidate text, but the checker does not.
/// The identity must be stable for equivalent materialization behavior and must
/// not contain request data; it is included in the public manifest digest.
#[async_trait::async_trait]
pub trait WorkspaceProvider: Send + Sync {
    /// Stable, non-sensitive identity of this materialization contract.
    fn manifest_identity(&self) -> &str;

    /// Materializes one candidate source tree and returns its owned lifetime.
    async fn materialize(
        &self,
        task_text: &str,
        attempt: &str,
    ) -> std::io::Result<CandidateWorkspace>;
}

/// Fixed operator configuration for [`CommandWorkspaceProvider`].
#[derive(Clone)]
pub struct CommandWorkspaceProviderConfig {
    /// Trusted operator command that populates the candidate workspace.
    pub command: Vec<String>,
    /// Maximum time allowed for one materialization command.
    pub timeout: Duration,
    /// Extra environment entries applied before protected host values.
    pub env: Vec<(String, String)>,
}

impl CommandWorkspaceProviderConfig {
    /// Configures a materializer argv and its execution timeout.
    pub fn new(command: Vec<String>, timeout: Duration) -> Self {
        Self {
            command,
            timeout,
            env: Vec::new(),
        }
    }
}

impl fmt::Debug for CommandWorkspaceProviderConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let env_keys = self
            .env
            .iter()
            .map(|(key, _value)| key.as_str())
            .collect::<Vec<_>>();
        formatter
            .debug_struct("CommandWorkspaceProviderConfig")
            .field("command", &self.command)
            .field("timeout", &self.timeout)
            .field("env_keys", &env_keys)
            .finish()
    }
}

/// Why a command-backed workspace provider could not be built.
#[derive(Debug, thiserror::Error)]
pub enum CommandWorkspaceProviderSetupError {
    /// The materializer command was empty.
    #[error("candidate materialize command must be a non-empty argv list")]
    EmptyCommand,
    /// A protected environment value was supplied by the operator.
    #[error("candidate materializer environment key {0:?} is reserved for the host")]
    ReservedEnvironmentKey(String),
    /// An environment key or value cannot be passed to a process.
    #[error("candidate materializer environment entry {0:?} is invalid")]
    InvalidEnvironmentEntry(String),
}

/// Materializes each candidate with a trusted operator command.
///
/// Request content reaches the command only through private control files. The
/// command receives no inherited environment and never receives the pinned test
/// directory. Its stable identity covers only fixed operator configuration.
pub struct CommandWorkspaceProvider {
    config: CommandWorkspaceProviderConfig,
    manifest_identity: String,
    protected_path: String,
}

impl CommandWorkspaceProvider {
    /// Validates and builds a command-backed materialization contract.
    pub fn new(
        config: CommandWorkspaceProviderConfig,
    ) -> Result<Self, CommandWorkspaceProviderSetupError> {
        validate_materializer_config(&config)?;
        let protected_path = current_path();
        let manifest_identity = materializer_manifest_identity(&config, &protected_path);
        Ok(Self {
            config,
            manifest_identity,
            protected_path,
        })
    }

    /// Runs one materializer in a fresh, owned candidate workspace.
    async fn materialize_once(
        &self,
        task_text: &str,
        attempt: &str,
    ) -> std::io::Result<CandidateWorkspace> {
        let workspace = TempDir::with_prefix("vgr-candidate-")?;
        let control = TempDir::with_prefix("vgr-materializer-control-")?;
        let tmp = control.path().join("tmp");
        tokio::fs::create_dir(&tmp).await?;
        let attempt_file = control.path().join(ATTEMPT_FILE);
        let task_file = control.path().join(TASK_FILE);
        tokio::fs::write(&attempt_file, attempt).await?;
        tokio::fs::write(&task_file, task_text).await?;

        let (program, arguments) = self
            .config
            .command
            .split_first()
            .ok_or_else(|| std::io::Error::other("candidate materialize command is empty"))?;
        let mut command = tokio::process::Command::new(program);
        command
            .args(arguments)
            .current_dir(workspace.path())
            .env_clear();
        for (key, value) in &self.config.env {
            command.env(key, value);
        }
        // These values are intentionally last so only the host can select the
        // control files and candidate workspace. The pinned tests stay absent.
        command
            .env("PATH", &self.protected_path)
            .env("HOME", workspace.path())
            .env("TMPDIR", &tmp)
            .env("LANG", "C.UTF-8")
            .env("ATTEMPT_FILE", &attempt_file)
            .env("TASK_FILE", &task_file)
            .env(WORKSPACE_ENV, workspace.path())
            .env_remove("TESTS_DIR")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        #[cfg(windows)]
        command
            .env("USERPROFILE", workspace.path())
            .env("TEMP", &tmp)
            .env("TMP", &tmp);
        #[cfg(windows)]
        for (key, value) in windows_host_environment() {
            command.env(key, value);
        }
        configure_process_tree(&mut command);

        let mut child = command.spawn()?;
        let _reaper = ProcessTreeReaper::attach(&child)?;
        let status = child.wait().await?;
        tracing::debug!(
            target: "libsy",
            status = status.code(),
            "vgr candidate materializer run"
        );
        if !status.success() {
            return Err(std::io::Error::other(
                "candidate materializer exited unsuccessfully",
            ));
        }
        CandidateWorkspace::new(workspace)
    }
}

#[async_trait::async_trait]
impl WorkspaceProvider for CommandWorkspaceProvider {
    fn manifest_identity(&self) -> &str {
        &self.manifest_identity
    }

    async fn materialize(
        &self,
        task_text: &str,
        attempt: &str,
    ) -> std::io::Result<CandidateWorkspace> {
        match tokio::time::timeout(
            self.config.timeout,
            self.materialize_once(task_text, attempt),
        )
        .await
        {
            Ok(result) => result,
            Err(_elapsed) => Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "candidate materializer timed out",
            )),
        }
    }
}

/// How an operator configures the checker.
#[derive(Clone)]
pub struct CheckerConfig {
    /// Directory holding the task's test suite. Copied at construction.
    pub tests_dir: PathBuf,
    /// The command to run, as an argv list. Never a shell string.
    ///
    /// Two placeholders are substituted per run: `{tests}` becomes the private
    /// per-run test copy, and `{workdir}` becomes the candidate workspace.
    pub command: Vec<String>,
    /// How long materialization, verification, and command execution may take.
    pub timeout: Duration,
    /// Extra environment entries applied before the protected host values.
    pub env: Vec<(String, String)>,
    /// Maximum namespace entries, including the test tree root.
    pub max_snapshot_entries: usize,
    /// Maximum regular-file bytes read during each copy or verification pass.
    pub max_snapshot_bytes: u64,
    /// Trusted host materialization for the candidate sources under test.
    pub workspace_provider: Arc<dyn WorkspaceProvider>,
    /// The operator's [`SANDBOX_ATTESTATION`], recorded verbatim.
    pub sandbox_attestation: String,
}

impl CheckerConfig {
    /// A configuration running `command` against the suite in `tests_dir`.
    pub fn new(
        tests_dir: impl Into<PathBuf>,
        command: Vec<String>,
        workspace_provider: Arc<dyn WorkspaceProvider>,
    ) -> Self {
        Self {
            tests_dir: tests_dir.into(),
            command,
            timeout: Duration::from_secs(120),
            env: Vec::new(),
            max_snapshot_entries: DEFAULT_MAX_ENTRIES,
            max_snapshot_bytes: DEFAULT_MAX_BYTES,
            workspace_provider,
            sandbox_attestation: String::new(),
        }
    }
}

impl fmt::Debug for CheckerConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let env_keys = self
            .env
            .iter()
            .map(|(key, _value)| key.as_str())
            .collect::<Vec<_>>();
        formatter
            .debug_struct("CheckerConfig")
            .field("tests_dir", &self.tests_dir)
            .field("command", &self.command)
            .field("timeout", &self.timeout)
            .field("env_keys", &env_keys)
            .field("max_snapshot_entries", &self.max_snapshot_entries)
            .field("max_snapshot_bytes", &self.max_snapshot_bytes)
            .field(
                "workspace_provider",
                &self.workspace_provider.manifest_identity(),
            )
            .field("sandbox_attestation", &self.sandbox_attestation)
            .finish()
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
    /// A protected environment value was supplied by the operator.
    #[error("checker environment key {0:?} is reserved for the host")]
    ReservedEnvironmentKey(String),
    /// An environment key or value cannot be passed to a process.
    #[error("checker environment entry {0:?} is invalid")]
    InvalidEnvironmentEntry(String),
    /// The materialization contract has no safe stable identity.
    #[error("checker workspace provider requires a stable non-empty manifest identity")]
    InvalidWorkspaceIdentity,
    /// Snapshot resource limits must both be non-zero.
    #[error("checker snapshot byte and entry limits must be non-zero")]
    InvalidSnapshotLimits,
    /// The test suite could not be snapshotted.
    #[error("checker could not snapshot its test suite: {0}")]
    Snapshot(#[source] std::io::Error),
    /// The suite holds something the manifest cannot represent.
    #[error(
        "checker test suite entry {0:?} is neither a regular file nor a directory, \
         so it cannot be snapshotted or hashed"
    )]
    UnsupportedEntry(PathBuf),
    /// The suite contains more namespace entries than the configured bound.
    #[error("checker test suite exceeds the {0}-entry snapshot limit")]
    SnapshotEntryLimit(usize),
    /// The suite contains more file bytes than the configured bound.
    #[error("checker test suite exceeds the {0}-byte snapshot limit")]
    SnapshotByteLimit(u64),
}

/// The pinned record of what the checker will run, and against what.
///
/// Its stable `sha` covers paths, entry kinds, file contents, command,
/// materialization contract, limits, and effective environment. Volatile
/// inode and ctime stamps are kept separately and never enter this identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    entries: BTreeMap<PathBuf, StableEntry>,
    /// One stable public audit hash over the complete checker contract.
    pub sha: String,
}

/// Runs a task's tests against a pinned, hash-verified snapshot.
///
/// Named for what it actually guarantees. It does not sandbox: confinement is
/// the deployment sandbox's job, and a type named for a property it does not
/// enforce would be read as a safety claim it cannot honour.
pub struct PinnedChecker {
    config: CheckerConfig,
    /// The private pinned source. Commands only see a private copy of it.
    snapshot: TempDir,
    pinned: TreeSnapshot,
    manifest: Manifest,
    admission: Arc<Semaphore>,
    protected_path: String,
}

impl PinnedChecker {
    /// Snapshots the test suite and pins its manifest.
    ///
    /// The snapshot is taken once, at construction, so that later edits to the
    /// operator's original directory cannot change what is being verified
    /// against mid-flight.
    pub fn new(config: CheckerConfig) -> Result<Self, CheckerSetupError> {
        validate_config(&config)?;
        let limits = SnapshotLimits::from(&config);
        let snapshot = TempDir::with_prefix("vgr-checker-").map_err(CheckerSetupError::Snapshot)?;
        let tests_path = snapshot.path().join("tests");
        copy_tree(&config.tests_dir, &tests_path, limits).map_err(CheckerSetupError::from)?;
        // Read-only is a courtesy, not the control: the same uid can undo it.
        // Stable namespace/content plus mutation stamps are the actual control.
        set_read_only(&tests_path).map_err(CheckerSetupError::from)?;
        let pinned = snapshot_tree(&tests_path, limits).map_err(CheckerSetupError::from)?;
        let entries = pinned.entries.clone();
        let protected_path = current_path();
        let manifest = Manifest {
            sha: manifest_sha(&entries, &config, &protected_path),
            entries,
        };
        Ok(Self {
            config,
            snapshot,
            pinned,
            manifest,
            admission: Arc::new(Semaphore::new(1)),
            protected_path,
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

    /// Stable public identity of the pinned checker contract.
    pub fn manifest_identity(&self) -> &str {
        &self.manifest.sha
    }

    /// Runs the checker through its public request contract in unit tests.
    #[cfg(test)]
    async fn check(&self, task_text: &str, attempt: &str) -> Option<bool> {
        let deadline = std::time::Instant::now() + self.config.timeout;
        <Self as Checker>::check(
            self,
            CheckerRequest {
                task_text,
                attempt,
                deadline,
                remaining: self.config.timeout,
                manifest_identity: self.manifest_identity(),
            },
        )
        .await
    }

    /// Runs one admitted checker operation from materialization through teardown.
    async fn check_once(&self, task_text: &str, attempt: &str) -> Option<bool> {
        let permit = match Arc::clone(&self.admission).acquire_owned().await {
            Ok(permit) => permit,
            Err(_closed) => return None,
        };
        let workspace = match self
            .config
            .workspace_provider
            .materialize(task_text, attempt)
            .await
        {
            Ok(workspace) => workspace,
            Err(error) => {
                tracing::warn!(
                    target: "libsy",
                    kind = ?error.kind(),
                    "vgr checker could not materialize its candidate workspace"
                );
                return None;
            }
        };

        let control = match TempDir::with_prefix("vgr-checker-control-") {
            Ok(control) => control,
            Err(error) => {
                tracing::warn!(
                    target: "libsy",
                    kind = ?error.kind(),
                    "vgr checker could not create its private control directory"
                );
                return None;
            }
        };
        let tmp = control.path().join("tmp");
        if let Err(error) = tokio::fs::create_dir(&tmp).await {
            tracing::warn!(
                target: "libsy",
                kind = ?error.kind(),
                "vgr checker could not create its private temporary directory"
            );
            return None;
        }
        let attempt_file = control.path().join(ATTEMPT_FILE);
        let task_file = control.path().join(TASK_FILE);
        if let Err(error) = tokio::fs::write(&attempt_file, attempt).await {
            tracing::warn!(
                target: "libsy",
                kind = ?error.kind(),
                "vgr checker could not write its attempt control file"
            );
            return None;
        }
        if let Err(error) = tokio::fs::write(&task_file, task_text).await {
            tracing::warn!(
                target: "libsy",
                kind = ?error.kind(),
                "vgr checker could not write its task control file"
            );
            return None;
        }

        let pinned_path = self.tests_path();
        let pinned = self.pinned.clone();
        let limits = SnapshotLimits::from(&self.config);
        let (permit, prepared) = match blocking_with_permit(permit, move || {
            prepare_run_tests(&pinned_path, &pinned, limits)
        })
        .await
        {
            Ok(result) => result,
            Err(_join) => {
                tracing::warn!(
                    target: "libsy",
                    "vgr checker snapshot worker did not complete"
                );
                return None;
            }
        };
        let run_tests = match prepared {
            Ok(run_tests) => run_tests,
            Err(error) => {
                log_tree_error("pre-run", &error);
                return None;
            }
        };

        let passed = match self
            .run_command(&workspace, &run_tests, &attempt_file, &task_file, &tmp)
            .await
        {
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

        let pinned_path = self.tests_path();
        let run_tests_path = run_tests.path().to_path_buf();
        let run_baseline = run_tests.baseline.clone();
        let pinned = self.pinned.clone();
        let (_permit, verified) = match blocking_with_permit(permit, move || {
            verify_after_run(
                &pinned_path,
                &pinned,
                &run_tests_path,
                &run_baseline,
                limits,
            )
        })
        .await
        {
            Ok(result) => result,
            Err(_join) => {
                tracing::warn!(
                    target: "libsy",
                    "vgr checker verification worker did not complete"
                );
                return None;
            }
        };
        match verified {
            Ok(()) => Some(passed),
            Err(error) => {
                log_tree_error("post-run", &error);
                None
            }
        }
    }

    /// Runs the configured argv in the candidate workspace.
    async fn run_command(
        &self,
        workspace: &CandidateWorkspace,
        run_tests: &RunTests,
        attempt_file: &Path,
        task_file: &Path,
        tmp: &Path,
    ) -> std::io::Result<bool> {
        let work = workspace.path();
        let tests = run_tests.path();
        let substituted = self
            .config
            .command
            .iter()
            .map(|argument| {
                argument
                    .replace("{tests}", &tests.to_string_lossy())
                    .replace("{workdir}", &work.to_string_lossy())
            })
            .collect::<Vec<_>>();
        let (program, arguments) = substituted
            .split_first()
            .ok_or_else(|| std::io::Error::other("checker command is empty"))?;

        let mut command = tokio::process::Command::new(program);
        command
            .args(arguments)
            .current_dir(work)
            // A whitelist, not the router's environment: the child has no
            // reason to inherit credentials the router holds.
            .env_clear();
        for (key, value) in &self.config.env {
            command.env(key, value);
        }
        // Protected values are intentionally last as defense in depth. Setup
        // rejects these keys too, but no future construction path may reverse
        // host ownership of the execution contract.
        command
            .env("PATH", &self.protected_path)
            .env("HOME", work)
            .env("TMPDIR", tmp)
            .env("LANG", "C.UTF-8")
            .env("TESTS_DIR", tests)
            .env("ATTEMPT_FILE", attempt_file)
            .env("TASK_FILE", task_file)
            .env(WORKSPACE_ENV, work)
            .stdin(std::process::Stdio::null())
            // Discarded rather than captured. Output derived from the attempt
            // must not reach the router's logs.
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        #[cfg(windows)]
        command
            .env("USERPROFILE", work)
            .env("TEMP", tmp)
            .env("TMP", tmp);
        #[cfg(windows)]
        for (key, value) in windows_host_environment() {
            command.env(key, value);
        }
        configure_process_tree(&mut command);

        let mut child = command.spawn()?;
        // Declared after the child so it terminates descendants before
        // `kill_on_drop` handles the direct child during cancellation.
        let _reaper = ProcessTreeReaper::attach(&child)?;
        let status = child.wait().await?;
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
    async fn check(&self, request: CheckerRequest<'_>) -> Option<bool> {
        if request.manifest_identity != self.manifest.sha {
            tracing::warn!(target: "libsy", "vgr checker manifest identity mismatch");
            return None;
        }
        let timeout = self.config.timeout.min(request.remaining);
        match tokio::time::timeout(timeout, self.check_once(request.task_text, request.attempt))
            .await
        {
            Ok(verdict) => verdict,
            Err(_elapsed) => {
                tracing::warn!(target: "libsy", "vgr checker timed out");
                None
            }
        }
    }
}

/// Holds the private tests used by exactly one run and their mutation baseline.
struct RunTests {
    owner: TempDir,
    path: PathBuf,
    baseline: TreeSnapshot,
}

impl RunTests {
    fn path(&self) -> &Path {
        &self.path
    }
}

/// Moves admission into blocking work so cancellation cannot release it early.
async fn blocking_with_permit<T, F>(
    permit: OwnedSemaphorePermit,
    work: F,
) -> Result<(OwnedSemaphorePermit, T), tokio::task::JoinError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let result = work();
        (permit, result)
    })
    .await
}

/// Builds and verifies the one test copy exposed to a command.
fn prepare_run_tests(
    pinned_path: &Path,
    pinned: &TreeSnapshot,
    limits: SnapshotLimits,
) -> Result<RunTests, TreeError> {
    ensure_matches(snapshot_tree(pinned_path, limits)?, pinned)?;
    let owner = TempDir::with_prefix("vgr-checker-tests-").map_err(TreeError::Io)?;
    let tests_path = owner.path().join("tests");
    copy_tree(pinned_path, &tests_path, limits)?;
    set_read_only(&tests_path)?;
    let baseline = snapshot_tree(&tests_path, limits)?;
    if baseline.entries != pinned.entries {
        return Err(TreeError::Mismatch);
    }
    // A source mutation racing the copy is caught even if the copied bytes look
    // self-consistent.
    ensure_matches(snapshot_tree(pinned_path, limits)?, pinned)?;
    Ok(RunTests {
        owner,
        path: tests_path,
        baseline,
    })
}

/// Verifies both the private run copy and the unexposed pinned source.
fn verify_after_run(
    pinned_path: &Path,
    pinned: &TreeSnapshot,
    run_tests_path: &Path,
    run_baseline: &TreeSnapshot,
    limits: SnapshotLimits,
) -> Result<(), TreeError> {
    ensure_matches(snapshot_tree(run_tests_path, limits)?, run_baseline)?;
    ensure_matches(snapshot_tree(pinned_path, limits)?, pinned)
}

fn ensure_matches(current: TreeSnapshot, expected: &TreeSnapshot) -> Result<(), TreeError> {
    if current == *expected {
        Ok(())
    } else {
        Err(TreeError::Mismatch)
    }
}

fn validate_config(config: &CheckerConfig) -> Result<(), CheckerSetupError> {
    if config.command.is_empty() {
        return Err(CheckerSetupError::EmptyCommand);
    }
    if config.sandbox_attestation != SANDBOX_ATTESTATION {
        return Err(CheckerSetupError::NotAttested);
    }
    if config.max_snapshot_entries == 0 || config.max_snapshot_bytes == 0 {
        return Err(CheckerSetupError::InvalidSnapshotLimits);
    }
    let identity = config.workspace_provider.manifest_identity();
    if identity.is_empty() || identity.len() > 4096 || identity.contains('\0') {
        return Err(CheckerSetupError::InvalidWorkspaceIdentity);
    }
    for (key, value) in &config.env {
        if is_protected_environment_key(key) {
            return Err(CheckerSetupError::ReservedEnvironmentKey(key.clone()));
        }
        if key.is_empty() || key.contains('=') || key.contains('\0') || value.contains('\0') {
            return Err(CheckerSetupError::InvalidEnvironmentEntry(key.clone()));
        }
    }
    Ok(())
}

fn validate_materializer_config(
    config: &CommandWorkspaceProviderConfig,
) -> Result<(), CommandWorkspaceProviderSetupError> {
    if config.command.is_empty() {
        return Err(CommandWorkspaceProviderSetupError::EmptyCommand);
    }
    for (key, value) in &config.env {
        if is_protected_environment_key(key) {
            return Err(CommandWorkspaceProviderSetupError::ReservedEnvironmentKey(
                key.clone(),
            ));
        }
        if key.is_empty() || key.contains('=') || key.contains('\0') || value.contains('\0') {
            return Err(CommandWorkspaceProviderSetupError::InvalidEnvironmentEntry(
                key.clone(),
            ));
        }
    }
    Ok(())
}

fn is_protected_environment_key(key: &str) -> bool {
    #[cfg(unix)]
    {
        PROTECTED_ENV.contains(&key)
    }
    #[cfg(windows)]
    {
        PROTECTED_ENV
            .iter()
            .any(|protected| protected.eq_ignore_ascii_case(key))
    }
}

#[derive(Clone, Copy)]
struct SnapshotLimits {
    entries: usize,
    bytes: u64,
}

impl From<&CheckerConfig> for SnapshotLimits {
    fn from(config: &CheckerConfig) -> Self {
        Self {
            entries: config.max_snapshot_entries,
            bytes: config.max_snapshot_bytes,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TreeSnapshot {
    entries: BTreeMap<PathBuf, StableEntry>,
    stamps: BTreeMap<PathBuf, MutationStamp>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum StableEntry {
    Directory,
    File(String),
}

/// Volatile tamper evidence, deliberately excluded from the public manifest.
#[derive(Debug)]
enum TreeError {
    Io(std::io::Error),
    Unsupported(PathBuf),
    EntryLimit(usize),
    ByteLimit(u64),
    ChangedDuringSnapshot,
    Mismatch,
}

impl From<std::io::Error> for TreeError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<TreeError> for CheckerSetupError {
    fn from(error: TreeError) -> Self {
        match error {
            TreeError::Io(error) => Self::Snapshot(error),
            TreeError::Unsupported(path) => Self::UnsupportedEntry(path),
            TreeError::EntryLimit(limit) => Self::SnapshotEntryLimit(limit),
            TreeError::ByteLimit(limit) => Self::SnapshotByteLimit(limit),
            TreeError::ChangedDuringSnapshot => Self::Snapshot(std::io::Error::other(
                "test suite changed while it was being snapshotted",
            )),
            TreeError::Mismatch => {
                Self::Snapshot(std::io::Error::other("test suite snapshot did not match"))
            }
        }
    }
}

fn log_tree_error(stage: &'static str, error: &TreeError) {
    let reason = match error {
        TreeError::Io(error) => match error.kind() {
            std::io::ErrorKind::NotFound => "not_found",
            std::io::ErrorKind::PermissionDenied => "permission_denied",
            _ => "io",
        },
        TreeError::Unsupported(_) => "unsupported_entry",
        TreeError::EntryLimit(_) => "entry_limit",
        TreeError::ByteLimit(_) => "byte_limit",
        TreeError::ChangedDuringSnapshot => "changed_during_snapshot",
        TreeError::Mismatch => "manifest_mismatch",
    };
    tracing::warn!(
        target: "libsy",
        stage,
        reason,
        "vgr checker test snapshot could not be verified"
    );
}

#[derive(Clone, Copy)]
struct Usage {
    limits: SnapshotLimits,
    entries: usize,
    bytes: u64,
}

impl Usage {
    fn new(limits: SnapshotLimits) -> Self {
        Self {
            limits,
            entries: 0,
            bytes: 0,
        }
    }

    fn add_entry(&mut self) -> Result<(), TreeError> {
        self.entries = self
            .entries
            .checked_add(1)
            .ok_or(TreeError::EntryLimit(self.limits.entries))?;
        if self.entries > self.limits.entries {
            return Err(TreeError::EntryLimit(self.limits.entries));
        }
        Ok(())
    }

    fn check_file_size(&self, bytes: u64) -> Result<(), TreeError> {
        let total = self
            .bytes
            .checked_add(bytes)
            .ok_or(TreeError::ByteLimit(self.limits.bytes))?;
        if total > self.limits.bytes {
            return Err(TreeError::ByteLimit(self.limits.bytes));
        }
        Ok(())
    }

    fn add_bytes(&mut self, bytes: usize) -> Result<(), TreeError> {
        self.bytes = self
            .bytes
            .checked_add(bytes as u64)
            .ok_or(TreeError::ByteLimit(self.limits.bytes))?;
        if self.bytes > self.limits.bytes {
            return Err(TreeError::ByteLimit(self.limits.bytes));
        }
        Ok(())
    }
}

/// Copies a regular-file/directory tree with bounded streaming I/O.
fn copy_tree(from: &Path, to: &Path, limits: SnapshotLimits) -> Result<(), TreeError> {
    let root_metadata = std::fs::symlink_metadata(from)?;
    if !is_directory(&root_metadata) {
        return Err(TreeError::Unsupported(from.to_path_buf()));
    }
    std::fs::create_dir(to)?;
    let mut usage = Usage::new(limits);
    usage.add_entry()?;
    let mut directories = vec![(from.to_path_buf(), to.to_path_buf())];
    while let Some((source, target)) = directories.pop() {
        let before = std::fs::symlink_metadata(&source)?;
        if !is_directory(&before) {
            return Err(TreeError::Unsupported(source));
        }
        let before_stamp = mutation_stamp_path(&source, &before)?;
        for entry in std::fs::read_dir(&source)? {
            let entry = entry?;
            let source_path = entry.path();
            let target_path = target.join(entry.file_name());
            let metadata = std::fs::symlink_metadata(&source_path)?;
            usage.add_entry()?;
            if is_directory(&metadata) {
                std::fs::create_dir(&target_path)?;
                directories.push((source_path, target_path));
            } else if is_regular_file(&metadata) {
                copy_file(&source_path, &target_path, &metadata, &mut usage)?;
            } else {
                return Err(TreeError::Unsupported(source_path));
            }
        }
        let after = std::fs::symlink_metadata(&source)?;
        if mutation_stamp_path(&source, &after)? != before_stamp {
            return Err(TreeError::ChangedDuringSnapshot);
        }
    }
    Ok(())
}

fn copy_file(
    source: &Path,
    target: &Path,
    metadata: &std::fs::Metadata,
    usage: &mut Usage,
) -> Result<(), TreeError> {
    usage.check_file_size(metadata.len())?;
    let expected_stamp = mutation_stamp_path(source, metadata)?;
    let mut input = open_regular_file(source)?;
    let opened = input.metadata()?;
    let opened_stamp = mutation_stamp_file(&input)?;
    if !is_regular_file(&opened) || opened_stamp != expected_stamp {
        return Err(TreeError::ChangedDuringSnapshot);
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)?;
    let mut buffer = [0_u8; HASH_CHUNK_BYTES];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        usage.add_bytes(read)?;
        output.write_all(&buffer[..read])?;
    }
    std::fs::set_permissions(target, metadata.permissions())?;
    if mutation_stamp_file(&input)? != opened_stamp {
        return Err(TreeError::ChangedDuringSnapshot);
    }
    Ok(())
}

/// Marks every regular file in the tree read-only and rejects every other kind.
fn set_read_only(root: &Path) -> Result<(), TreeError> {
    let metadata = std::fs::symlink_metadata(root)?;
    if !is_directory(&metadata) {
        return Err(TreeError::Unsupported(root.to_path_buf()));
    }
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path)?;
            if is_directory(&metadata) {
                directories.push(path);
            } else if is_regular_file(&metadata) {
                let mut permissions = metadata.permissions();
                permissions.set_readonly(true);
                std::fs::set_permissions(path, permissions)?;
            } else {
                return Err(TreeError::Unsupported(path));
            }
        }
    }
    Ok(())
}

/// Hashes the complete namespace and captures separate mutation stamps.
fn snapshot_tree(root: &Path, limits: SnapshotLimits) -> Result<TreeSnapshot, TreeError> {
    let root_metadata = std::fs::symlink_metadata(root)?;
    if !is_directory(&root_metadata) {
        return Err(TreeError::Unsupported(root.to_path_buf()));
    }
    let mut usage = Usage::new(limits);
    let mut entries = BTreeMap::new();
    let mut stamps = BTreeMap::new();
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        let relative = relative_path(root, &directory)?;
        let before = std::fs::symlink_metadata(&directory)?;
        if !is_directory(&before) {
            return Err(TreeError::Unsupported(directory));
        }
        usage.add_entry()?;
        entries.insert(relative.clone(), StableEntry::Directory);
        let before_stamp = mutation_stamp_path(&directory, &before)?;
        stamps.insert(relative, before_stamp);
        for entry in std::fs::read_dir(&directory)? {
            let entry = entry?;
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path)?;
            if is_directory(&metadata) {
                directories.push(path);
            } else if is_regular_file(&metadata) {
                usage.add_entry()?;
                let relative = relative_path(root, &path)?;
                let (digest, stamp) = hash_file(&path, &metadata, &mut usage)?;
                entries.insert(relative.clone(), StableEntry::File(digest));
                stamps.insert(relative, stamp);
            } else {
                return Err(TreeError::Unsupported(path));
            }
        }
        let after = std::fs::symlink_metadata(&directory)?;
        if mutation_stamp_path(&directory, &after)? != before_stamp {
            return Err(TreeError::ChangedDuringSnapshot);
        }
    }
    Ok(TreeSnapshot { entries, stamps })
}

fn relative_path(root: &Path, path: &Path) -> Result<PathBuf, TreeError> {
    path.strip_prefix(root)
        .map(Path::to_path_buf)
        .map_err(|_| TreeError::Io(std::io::Error::other("snapshot entry escaped its root")))
}

fn hash_file(
    path: &Path,
    metadata: &std::fs::Metadata,
    usage: &mut Usage,
) -> Result<(String, MutationStamp), TreeError> {
    usage.check_file_size(metadata.len())?;
    let expected_stamp = mutation_stamp_path(path, metadata)?;
    let mut file = open_regular_file(path)?;
    let opened = file.metadata()?;
    let stamp = mutation_stamp_file(&file)?;
    if !is_regular_file(&opened) || stamp != expected_stamp {
        return Err(TreeError::ChangedDuringSnapshot);
    }
    let mut context = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buffer = [0_u8; HASH_CHUNK_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        usage.add_bytes(read)?;
        context.update(&buffer[..read]);
    }
    if mutation_stamp_file(&file)? != stamp {
        return Err(TreeError::ChangedDuringSnapshot);
    }
    Ok((hex(context.finish()), stamp))
}

/// One stable hash over the suite and complete execution contract.
fn manifest_sha(
    entries: &BTreeMap<PathBuf, StableEntry>,
    config: &CheckerConfig,
    protected_path: &str,
) -> String {
    let mut context = ring::digest::Context::new(&ring::digest::SHA256);
    digest_field(&mut context, b"vgr-pinned-checker-manifest-v2");
    for (path, entry) in entries {
        digest_field(&mut context, &os_str_bytes(path.as_os_str()));
        match entry {
            StableEntry::Directory => digest_field(&mut context, b"directory"),
            StableEntry::File(digest) => {
                digest_field(&mut context, b"file");
                digest_field(&mut context, digest.as_bytes());
            }
        }
    }
    for argument in &config.command {
        digest_field(&mut context, argument.as_bytes());
    }
    digest_field(
        &mut context,
        config.workspace_provider.manifest_identity().as_bytes(),
    );
    digest_field(
        &mut context,
        config.timeout.as_nanos().to_string().as_bytes(),
    );
    digest_field(
        &mut context,
        config.max_snapshot_entries.to_string().as_bytes(),
    );
    digest_field(
        &mut context,
        config.max_snapshot_bytes.to_string().as_bytes(),
    );
    for (key, value) in manifest_environment(config, protected_path) {
        digest_field(&mut context, key.as_bytes());
        digest_field(&mut context, value.as_bytes());
    }
    hex(context.finish())
}

fn manifest_environment(config: &CheckerConfig, protected_path: &str) -> BTreeMap<String, String> {
    let mut environment = config.env.iter().cloned().collect::<BTreeMap<_, _>>();
    environment.insert("PATH".into(), protected_path.to_string());
    environment.insert("LANG".into(), "C.UTF-8".into());
    environment.insert("HOME".into(), "{workspace}".into());
    environment.insert("TMPDIR".into(), "{control}/tmp".into());
    environment.insert("TESTS_DIR".into(), "{private-tests}".into());
    environment.insert("ATTEMPT_FILE".into(), "{control}/attempt.txt".into());
    environment.insert("TASK_FILE".into(), "{control}/task.txt".into());
    environment.insert(WORKSPACE_ENV.into(), "{workspace}".into());
    #[cfg(windows)]
    {
        environment.insert("TEMP".into(), "{control}/tmp".into());
        environment.insert("TMP".into(), "{control}/tmp".into());
        environment.insert("USERPROFILE".into(), "{workspace}".into());
        environment.extend(windows_host_environment());
    }
    environment
}

fn current_path() -> String {
    std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin".into())
}

#[cfg(windows)]
fn windows_host_environment() -> BTreeMap<String, String> {
    WINDOWS_HOST_ENV
        .into_iter()
        .filter_map(|key| {
            std::env::var(key)
                .ok()
                .map(|value| (key.to_string(), value))
        })
        .collect()
}

fn materializer_manifest_identity(
    config: &CommandWorkspaceProviderConfig,
    protected_path: &str,
) -> String {
    let mut context = ring::digest::Context::new(&ring::digest::SHA256);
    digest_field(&mut context, b"vgr-command-workspace-provider-v1");
    for argument in &config.command {
        digest_field(&mut context, argument.as_bytes());
    }
    digest_field(
        &mut context,
        config.timeout.as_nanos().to_string().as_bytes(),
    );
    for (key, value) in materializer_manifest_environment(config, protected_path) {
        digest_field(&mut context, key.as_bytes());
        digest_field(&mut context, value.as_bytes());
    }
    format!("command-workspace-provider-v1:{}", hex(context.finish()))
}

fn materializer_manifest_environment(
    config: &CommandWorkspaceProviderConfig,
    protected_path: &str,
) -> BTreeMap<String, String> {
    let mut environment = config.env.iter().cloned().collect::<BTreeMap<_, _>>();
    environment.remove("TESTS_DIR");
    environment.insert("PATH".into(), protected_path.to_string());
    environment.insert("LANG".into(), "C.UTF-8".into());
    environment.insert("HOME".into(), "{workspace}".into());
    environment.insert("TMPDIR".into(), "{control}/tmp".into());
    environment.insert("ATTEMPT_FILE".into(), "{control}/attempt.txt".into());
    environment.insert("TASK_FILE".into(), "{control}/task.txt".into());
    environment.insert(WORKSPACE_ENV.into(), "{workspace}".into());
    #[cfg(windows)]
    {
        environment.insert("TEMP".into(), "{control}/tmp".into());
        environment.insert("TMP".into(), "{control}/tmp".into());
        environment.insert("USERPROFILE".into(), "{workspace}".into());
        environment.extend(windows_host_environment());
    }
    environment
}

fn digest_field(context: &mut ring::digest::Context, value: &[u8]) {
    context.update(&(value.len() as u64).to_le_bytes());
    context.update(value);
}

/// Renders a digest as lower-case hex.
fn hex(digest: ring::digest::Digest) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(all(test, unix))]
mod tests;
#[cfg(all(test, windows))]
#[path = "checker/tests_windows.rs"]
mod tests;
