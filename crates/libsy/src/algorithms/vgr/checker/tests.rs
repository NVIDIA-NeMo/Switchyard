// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the pinned-manifest checker.

use std::future::Future as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Poll;
use std::time::Duration;

use tempfile::TempDir;
use tokio::sync::Notify;

use super::*;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// A suite directory holding one script, and the temp dir owning it.
fn suite(script: &str) -> std::io::Result<TempDir> {
    let dir = TempDir::with_prefix("vgr-checker-suite-")?;
    std::fs::write(dir.path().join("run.sh"), script)?;
    Ok(dir)
}

/// A host provider that creates a workspace with fixed candidate files.
struct TestProvider {
    identity: String,
    files: Vec<(String, String)>,
}

#[async_trait::async_trait]
impl WorkspaceProvider for TestProvider {
    fn manifest_identity(&self) -> &str {
        &self.identity
    }

    async fn materialize(
        &self,
        _task_text: &str,
        _attempt: &str,
    ) -> std::io::Result<CandidateWorkspace> {
        let root = TempDir::with_prefix("vgr-candidate-")?;
        for (name, contents) in &self.files {
            std::fs::write(root.path().join(name), contents)?;
        }
        CandidateWorkspace::new(root)
    }
}

fn provider(identity: &str) -> Arc<dyn WorkspaceProvider> {
    Arc::new(TestProvider {
        identity: identity.to_string(),
        files: Vec::new(),
    })
}

fn provider_with_file(identity: &str, name: &str, contents: &str) -> Arc<dyn WorkspaceProvider> {
    Arc::new(TestProvider {
        identity: identity.to_string(),
        files: vec![(name.to_string(), contents.to_string())],
    })
}

/// A config running `argv` against `tests`, attested and quick to time out.
fn config(tests: &TempDir, argv: &[&str]) -> CheckerConfig {
    CheckerConfig {
        timeout: Duration::from_secs(10),
        sandbox_attestation: SANDBOX_ATTESTATION.to_string(),
        ..CheckerConfig::new(
            tests.path(),
            argv.iter().map(|part| (*part).to_string()).collect(),
            provider("test-provider-v1"),
        )
    }
}

/// Runs `sh` over the private per-run copy of the snapshot's script.
fn shell_argv() -> Vec<&'static str> {
    vec!["/bin/sh", "{tests}/run.sh"]
}

#[tokio::test]
async fn a_clean_exit_with_an_intact_suite_passes() -> TestResult {
    let tests = suite("exit 0\n")?;
    let checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, Some(true));
    Ok(())
}

#[tokio::test]
async fn a_nonzero_exit_is_a_fail_not_an_indeterminate() -> TestResult {
    let tests = suite("exit 1\n")?;
    let checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, Some(false));
    Ok(())
}

#[tokio::test]
async fn a_run_that_edits_the_private_suite_reports_nothing() -> TestResult {
    let tests =
        suite("chmod u+w \"$TESTS_DIR/run.sh\"; echo tampered >> \"$TESTS_DIR/run.sh\"; exit 0\n")?;
    let checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, None);
    // The shared pinned source was never exposed to the command.
    assert_eq!(
        std::fs::read_to_string(checker.tests_path().join("run.sh"))?,
        "chmod u+w \"$TESTS_DIR/run.sh\"; echo tampered >> \"$TESTS_DIR/run.sh\"; exit 0\n"
    );
    Ok(())
}

#[tokio::test]
async fn a_run_that_edits_and_restores_file_bytes_reports_nothing() -> TestResult {
    const EXPECTED: &str = "expected answer\n";
    let tests = suite(
        "chmod u+w \"$TESTS_DIR/expected.txt\"\n\
         printf 'anything goes\\n' > \"$TESTS_DIR/expected.txt\"\n\
         printf 'expected answer\\n' > \"$TESTS_DIR/expected.txt\"\n\
         touch -d '2020-01-01' \"$TESTS_DIR/expected.txt\"\n\
         exit 0\n",
    )?;
    std::fs::write(tests.path().join("expected.txt"), EXPECTED)?;
    let checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, None);
    Ok(())
}

#[tokio::test]
async fn a_run_that_renames_replaces_and_restores_the_tests_root_reports_nothing() -> TestResult {
    let tests = suite(
        "original=\"${TESTS_DIR}.original\"\n\
         mv \"$TESTS_DIR\" \"$original\"\n\
         mkdir \"$TESTS_DIR\"\n\
         printf 'exit 0\\n' > \"$TESTS_DIR/replacement.sh\"\n\
         /bin/sh \"$TESTS_DIR/replacement.sh\" || exit 1\n\
         rm -rf \"$TESTS_DIR\"\n\
         mv \"$original\" \"$TESTS_DIR\"\n\
         exit 0\n",
    )?;
    let checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, None);
    Ok(())
}

#[tokio::test]
async fn a_run_that_adds_an_empty_directory_reports_nothing() -> TestResult {
    let tests = suite("mkdir \"$TESTS_DIR/empty\"; exit 0\n")?;
    let checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, None);
    Ok(())
}

#[tokio::test]
async fn a_fifo_added_before_verification_is_rejected() -> TestResult {
    let tests = suite("mkfifo \"$TESTS_DIR/pipe\"; exit 0\n")?;
    let checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, None);
    Ok(())
}

#[tokio::test]
async fn a_command_that_cannot_be_spawned_reports_nothing() -> TestResult {
    let tests = suite("exit 0\n")?;
    let checker = PinnedChecker::new(config(&tests, &["/nonexistent/checker-binary"]))?;
    assert_eq!(checker.check("task", "attempt").await, None);
    Ok(())
}

#[tokio::test]
async fn a_run_that_outlives_its_timeout_reports_nothing() -> TestResult {
    let tests = suite("sleep 30\n")?;
    let mut checker_config = config(&tests, &shell_argv());
    checker_config.timeout = Duration::from_millis(150);
    let checker = PinnedChecker::new(checker_config)?;
    assert_eq!(checker.check("task", "attempt").await, None);
    Ok(())
}

#[tokio::test]
async fn a_timeout_kills_the_whole_process_tree() -> TestResult {
    let marker = TempDir::with_prefix("vgr-checker-marker-")?;
    let alive = marker.path().join("still-alive");
    let tests = suite(&format!(
        "sh -c 'while true; do touch {}; sleep 0.02; done' &\n\
         while [ ! -e {} ]; do :; done\n\
         sleep 30\n",
        alive.display(),
        alive.display()
    ))?;
    let mut checker_config = config(&tests, &shell_argv());
    checker_config.timeout = Duration::from_millis(200);
    let checker = PinnedChecker::new(checker_config)?;
    assert_eq!(checker.check("task", "attempt").await, None);
    assert!(alive.exists(), "the background grandchild never ran");

    tokio::time::sleep(Duration::from_millis(200)).await;
    let settled = std::fs::metadata(&alive)?.modified()?;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let after = std::fs::metadata(&alive)?.modified()?;
    assert_eq!(settled, after, "the checker grandchild survived timeout");
    Ok(())
}

#[tokio::test]
async fn the_attempt_and_task_reach_the_command_as_files() -> TestResult {
    let tests = suite(
        "grep -q 'the attempt text' \"$ATTEMPT_FILE\" || exit 1\n\
         grep -q 'the task text' \"$TASK_FILE\" || exit 1\n\
         exit 0\n",
    )?;
    let checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    assert_eq!(
        checker.check("the task text", "the attempt text").await,
        Some(true)
    );
    Ok(())
}

#[tokio::test]
async fn the_command_tests_materialized_candidate_sources() -> TestResult {
    let tests = suite(
        "[ \"$(cat candidate.txt)\" = 'candidate source' ] || exit 1\n\
         [ \"$PWD\" = \"$WORKSPACE_DIR\" ] || exit 1\n\
         exit 0\n",
    )?;
    let mut checker_config = config(&tests, &shell_argv());
    checker_config.workspace_provider =
        provider_with_file("candidate-copy-v1", "candidate.txt", "candidate source");
    let checker = PinnedChecker::new(checker_config)?;
    assert_eq!(checker.check("task", "model text").await, Some(true));
    Ok(())
}

#[tokio::test]
async fn command_provider_materializes_request_data_only_through_files() -> TestResult {
    let provider = CommandWorkspaceProvider::new(CommandWorkspaceProviderConfig::new(
        vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            concat!(
                "[ -z \"$NVIDIA_API_KEY\" ] || exit 1\n",
                "[ -z \"${TESTS_DIR+x}\" ] || exit 1\n",
                "cp \"$ATTEMPT_FILE\" \"$WORKSPACE_DIR/candidate.txt\"\n",
                "cp \"$TASK_FILE\" \"$WORKSPACE_DIR/task.txt\"\n",
            )
            .to_string(),
        ],
        Duration::from_secs(10),
    ))?;
    let workspace = provider
        .materialize("trusted task", "untrusted attempt")
        .await?;

    assert_eq!(
        std::fs::read_to_string(workspace.path().join("candidate.txt"))?,
        "untrusted attempt"
    );
    assert_eq!(
        std::fs::read_to_string(workspace.path().join("task.txt"))?,
        "trusted task"
    );
    Ok(())
}

#[tokio::test]
async fn command_provider_applies_protected_environment_last() -> TestResult {
    let mut provider = CommandWorkspaceProvider::new(CommandWorkspaceProviderConfig::new(
        vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            concat!(
                "[ \"$HOME\" = \"$WORKSPACE_DIR\" ] || exit 1\n",
                "[ \"$TMPDIR\" != '/untrusted' ] || exit 1\n",
                "[ \"$PATH\" != '/untrusted' ] || exit 1\n",
                "[ \"$LANG\" = 'C.UTF-8' ] || exit 1\n",
                "[ \"$ATTEMPT_FILE\" != '/untrusted' ] || exit 1\n",
                "[ \"$TASK_FILE\" != '/untrusted' ] || exit 1\n",
                "[ \"$WORKSPACE_DIR\" != '/untrusted' ] || exit 1\n",
                "[ -z \"${TESTS_DIR+x}\" ] || exit 1\n",
            )
            .to_string(),
        ],
        Duration::from_secs(10),
    ))?;
    for key in PROTECTED_ENV {
        provider
            .config
            .env
            .push((key.to_string(), "/untrusted".to_string()));
    }

    provider.materialize("task", "attempt").await?;
    Ok(())
}

#[test]
fn command_provider_rejects_empty_argv_and_reserved_environment() -> TestResult {
    assert!(matches!(
        CommandWorkspaceProvider::new(CommandWorkspaceProviderConfig::new(
            Vec::new(),
            Duration::from_secs(10),
        )),
        Err(CommandWorkspaceProviderSetupError::EmptyCommand)
    ));

    for key in PROTECTED_ENV {
        let mut config = CommandWorkspaceProviderConfig::new(
            vec!["/bin/true".to_string()],
            Duration::from_secs(10),
        );
        config.env.push((key.to_string(), "override".to_string()));
        assert!(matches!(
            CommandWorkspaceProvider::new(config),
            Err(CommandWorkspaceProviderSetupError::ReservedEnvironmentKey(rejected))
                if rejected == key
        ));
    }
    Ok(())
}

#[test]
fn command_provider_identity_covers_only_fixed_configuration() -> TestResult {
    let base = CommandWorkspaceProvider::new(CommandWorkspaceProviderConfig::new(
        vec!["/bin/true".to_string()],
        Duration::from_secs(10),
    ))?;
    let command = CommandWorkspaceProvider::new(CommandWorkspaceProviderConfig::new(
        vec!["/bin/false".to_string()],
        Duration::from_secs(10),
    ))?;
    let timeout = CommandWorkspaceProvider::new(CommandWorkspaceProviderConfig::new(
        vec!["/bin/true".to_string()],
        Duration::from_secs(11),
    ))?;
    let mut environment_config =
        CommandWorkspaceProviderConfig::new(vec!["/bin/true".to_string()], Duration::from_secs(10));
    environment_config
        .env
        .push(("MATERIALIZER_FEATURE".to_string(), "enabled".to_string()));
    let environment = CommandWorkspaceProvider::new(environment_config)?;

    assert_ne!(base.manifest_identity(), command.manifest_identity());
    assert_ne!(base.manifest_identity(), timeout.manifest_identity());
    assert_ne!(base.manifest_identity(), environment.manifest_identity());
    assert!(!base.manifest_identity().contains("request"));
    Ok(())
}

#[tokio::test]
async fn a_materializer_timeout_kills_its_whole_process_tree() -> TestResult {
    let marker = TempDir::with_prefix("vgr-materializer-marker-")?;
    let alive = marker.path().join("still-alive");
    let provider = CommandWorkspaceProvider::new(CommandWorkspaceProviderConfig::new(
        vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            format!(
                "sh -c 'while true; do touch {}; sleep 0.02; done' &\n\
                 while [ ! -e {} ]; do :; done\n\
                 sleep 30\n",
                alive.display(),
                alive.display()
            ),
        ],
        Duration::from_millis(200),
    ))?;

    let error = provider
        .materialize("task", "attempt")
        .await
        .err()
        .ok_or("materializer unexpectedly completed")?;
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    assert!(alive.exists(), "the materializer grandchild never ran");

    tokio::time::sleep(Duration::from_millis(200)).await;
    let settled = std::fs::metadata(&alive)?.modified()?;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let after = std::fs::metadata(&alive)?.modified()?;
    assert_eq!(
        settled, after,
        "the materializer grandchild survived timeout"
    );
    Ok(())
}

#[tokio::test]
async fn the_command_does_not_inherit_the_router_environment() -> TestResult {
    let tests = suite("[ -z \"$NVIDIA_API_KEY\" ] || exit 1\nexit 0\n")?;
    let checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, Some(true));
    Ok(())
}

#[tokio::test]
async fn protected_environment_is_applied_last_at_runtime() -> TestResult {
    let tests = suite(
        "[ \"$TESTS_DIR\" != '/untrusted' ] || exit 1\n\
         [ \"$ATTEMPT_FILE\" != '/untrusted' ] || exit 1\n\
         [ \"$TASK_FILE\" != '/untrusted' ] || exit 1\n\
         [ \"$HOME\" = \"$WORKSPACE_DIR\" ] || exit 1\n\
         [ \"$TMPDIR\" != '/untrusted' ] || exit 1\n\
         [ \"$PATH\" != '/untrusted' ] || exit 1\n\
         [ \"$LANG\" = 'C.UTF-8' ] || exit 1\n\
         [ \"$WORKSPACE_DIR\" != '/untrusted' ] || exit 1\n\
         exit 0\n",
    )?;
    let mut checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    for key in PROTECTED_ENV {
        checker
            .config
            .env
            .push((key.to_string(), "/untrusted".to_string()));
    }
    assert_eq!(checker.check("task", "attempt").await, Some(true));
    Ok(())
}

#[tokio::test]
async fn nothing_derived_from_the_attempt_reaches_the_command_line() -> TestResult {
    let tests = suite(
        "case \"$*\" in *marker-from-attempt*) exit 1 ;; esac\n\
         case \"$0\" in *marker-from-attempt*) exit 1 ;; esac\n\
         exit 0\n",
    )?;
    let checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    assert_eq!(
        checker
            .check(
                "{tests} marker-from-attempt",
                "{workdir} marker-from-attempt"
            )
            .await,
        Some(true)
    );
    Ok(())
}

#[test]
fn a_checker_without_the_sandbox_attestation_does_not_build() -> TestResult {
    let tests = suite("exit 0\n")?;
    let mut checker_config = config(&tests, &shell_argv());
    checker_config.sandbox_attestation = "approved".to_string();
    assert!(matches!(
        PinnedChecker::new(checker_config),
        Err(CheckerSetupError::NotAttested)
    ));
    Ok(())
}

#[test]
fn an_empty_command_does_not_build() -> TestResult {
    let tests = suite("exit 0\n")?;
    let checker_config = CheckerConfig {
        sandbox_attestation: SANDBOX_ATTESTATION.to_string(),
        ..CheckerConfig::new(tests.path(), Vec::new(), provider("test-provider-v1"))
    };
    assert!(matches!(
        PinnedChecker::new(checker_config),
        Err(CheckerSetupError::EmptyCommand)
    ));
    Ok(())
}

#[test]
fn every_host_owned_environment_key_is_rejected_at_setup() -> TestResult {
    let tests = suite("exit 0\n")?;
    for key in PROTECTED_ENV {
        let mut checker_config = config(&tests, &shell_argv());
        checker_config
            .env
            .push((key.to_string(), "override".to_string()));
        assert!(matches!(
            PinnedChecker::new(checker_config),
            Err(CheckerSetupError::ReservedEnvironmentKey(rejected)) if rejected == key
        ));
    }
    Ok(())
}

#[test]
fn a_suite_holding_a_symlink_does_not_build() -> TestResult {
    let tests = suite("exit 0\n")?;
    std::os::unix::fs::symlink("/etc/hostname", tests.path().join("linked.txt"))?;
    let result = PinnedChecker::new(config(&tests, &shell_argv()));
    assert!(matches!(
        result,
        Err(CheckerSetupError::UnsupportedEntry(path)) if path.ends_with("linked.txt")
    ));
    Ok(())
}

#[test]
fn a_suite_holding_a_socket_does_not_build() -> TestResult {
    let tests = suite("exit 0\n")?;
    let _listener = std::os::unix::net::UnixListener::bind(tests.path().join("socket"))?;
    assert!(matches!(
        PinnedChecker::new(config(&tests, &shell_argv())),
        Err(CheckerSetupError::UnsupportedEntry(path)) if path.ends_with("socket")
    ));
    Ok(())
}

#[test]
fn empty_directories_are_part_of_the_stable_manifest() -> TestResult {
    let first = suite("exit 0\n")?;
    let second = suite("exit 0\n")?;
    std::fs::create_dir(second.path().join("empty"))?;
    let first = PinnedChecker::new(config(&first, &shell_argv()))?;
    let second = PinnedChecker::new(config(&second, &shell_argv()))?;
    assert_ne!(first.manifest().sha, second.manifest().sha);
    Ok(())
}

#[test]
fn the_manifest_is_stable_across_checker_restarts() -> TestResult {
    let tests = suite("exit 0\n")?;
    std::fs::create_dir(tests.path().join("empty"))?;
    std::fs::write(tests.path().join("data.txt"), "same bytes")?;
    let first = PinnedChecker::new(config(&tests, &shell_argv()))?;
    let first_identity = first.manifest().sha.clone();
    drop(first);
    let second = PinnedChecker::new(config(&tests, &shell_argv()))?;
    assert_eq!(first_identity, second.manifest().sha);
    assert_eq!(second.manifest_identity(), first_identity);
    Ok(())
}

#[test]
fn command_materialization_and_environment_change_manifest_identity() -> TestResult {
    let tests = suite("exit 0\n")?;
    let base = PinnedChecker::new(config(&tests, &shell_argv()))?;

    let command = PinnedChecker::new(config(&tests, &["/bin/sh", "-c", "exit 0"]))?;
    assert_ne!(base.manifest().sha, command.manifest().sha);

    let mut materialization = config(&tests, &shell_argv());
    materialization.workspace_provider = provider("test-provider-v2");
    let materialization = PinnedChecker::new(materialization)?;
    assert_ne!(base.manifest().sha, materialization.manifest().sha);

    let mut environment = config(&tests, &shell_argv());
    environment
        .env
        .push(("CHECKER_FEATURE".to_string(), "enabled".to_string()));
    let environment = PinnedChecker::new(environment)?;
    assert_ne!(base.manifest().sha, environment.manifest().sha);
    Ok(())
}

#[test]
fn snapshot_entry_and_byte_limits_fail_closed() -> TestResult {
    let tests = suite("exit 0\n")?;
    let mut entry_limited = config(&tests, &shell_argv());
    entry_limited.max_snapshot_entries = 1;
    assert!(matches!(
        PinnedChecker::new(entry_limited),
        Err(CheckerSetupError::SnapshotEntryLimit(1))
    ));

    let mut byte_limited = config(&tests, &shell_argv());
    byte_limited.max_snapshot_bytes = 1;
    assert!(matches!(
        PinnedChecker::new(byte_limited),
        Err(CheckerSetupError::SnapshotByteLimit(1))
    ));
    Ok(())
}

struct PendingProvider;

#[async_trait::async_trait]
impl WorkspaceProvider for PendingProvider {
    fn manifest_identity(&self) -> &str {
        "pending-provider-v1"
    }

    async fn materialize(
        &self,
        _task_text: &str,
        _attempt: &str,
    ) -> std::io::Result<CandidateWorkspace> {
        std::future::pending().await
    }
}

#[tokio::test]
async fn the_checker_deadline_includes_materialization() -> TestResult {
    let tests = suite("exit 0\n")?;
    let mut checker_config = config(&tests, &shell_argv());
    checker_config.timeout = Duration::from_millis(20);
    checker_config.workspace_provider = Arc::new(PendingProvider);
    let checker = PinnedChecker::new(checker_config)?;
    assert_eq!(checker.check("task", "attempt").await, None);
    Ok(())
}

struct BlockingProvider {
    calls: AtomicUsize,
    entered: Notify,
    release: Notify,
}

#[async_trait::async_trait]
impl WorkspaceProvider for BlockingProvider {
    fn manifest_identity(&self) -> &str {
        "blocking-provider-v1"
    }

    async fn materialize(
        &self,
        _task_text: &str,
        _attempt: &str,
    ) -> std::io::Result<CandidateWorkspace> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.notify_one();
        self.release.notified().await;
        CandidateWorkspace::new(TempDir::with_prefix("vgr-candidate-")?)
    }
}

#[tokio::test]
async fn one_checker_instance_never_materializes_overlapping_runs() -> TestResult {
    let tests = suite("exit 0\n")?;
    let provider = Arc::new(BlockingProvider {
        calls: AtomicUsize::new(0),
        entered: Notify::new(),
        release: Notify::new(),
    });
    let mut checker_config = config(&tests, &shell_argv());
    checker_config.workspace_provider = provider.clone();
    let checker = Arc::new(PinnedChecker::new(checker_config)?);

    let first_checker = Arc::clone(&checker);
    let first = tokio::spawn(async move { first_checker.check("task one", "attempt one").await });
    provider.entered.notified().await;

    let mut second = Box::pin(checker.check("task two", "attempt two"));
    let first_poll =
        std::future::poll_fn(|context| Poll::Ready(second.as_mut().poll(context))).await;
    assert!(matches!(first_poll, Poll::Pending));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);

    provider.release.notify_one();
    assert_eq!(first.await?, Some(true));
    provider.release.notify_one();
    assert_eq!(second.await, Some(true));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test]
async fn cancelled_blocking_verification_holds_admission_until_it_stops() -> TestResult {
    let admission = Arc::new(Semaphore::new(1));
    let permit = Arc::clone(&admission).acquire_owned().await?;
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let worker = tokio::spawn(blocking_with_permit(permit, move || {
        let _ignored = started_tx.send(());
        let _ignored = release_rx.recv();
    }));
    started_rx.await?;

    worker.abort();
    assert!(worker.await.is_err(), "the blocking join was not cancelled");
    assert_eq!(
        admission.available_permits(),
        0,
        "cancellation admitted a second run while blocking work remained"
    );

    release_tx.send(())?;
    let returned = tokio::time::timeout(Duration::from_secs(1), admission.acquire()).await??;
    drop(returned);
    assert_eq!(admission.available_permits(), 1);
    Ok(())
}

#[test]
fn editing_the_operator_directory_after_construction_does_not_change_the_snapshot() -> TestResult {
    let tests = suite("exit 0\n")?;
    let checker = PinnedChecker::new(config(&tests, &shell_argv()))?;
    let pinned = checker.manifest().sha.clone();
    std::fs::write(tests.path().join("run.sh"), "exit 1\n")?;
    assert_eq!(checker.manifest().sha, pinned);
    Ok(())
}
