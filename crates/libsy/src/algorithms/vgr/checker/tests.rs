// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the pinned-manifest checker.
//!
//! The verdict mapping is the whole contract: only a clean exit with the
//! manifest verified twice may report a pass, and every other outcome must be
//! indeterminate rather than a fail. The tampering cases are the reason this
//! module exists, so they are tested against a command that actually edits the
//! suite it is being judged by.

use std::time::Duration;

use tempfile::TempDir;

use super::*;

/// A suite directory holding one script, and the temp dir owning it.
fn suite(script: &str) -> TempDir {
    let dir = TempDir::with_prefix("vgr-checker-suite-").expect("suite dir");
    std::fs::write(dir.path().join("run.sh"), script).expect("write script");
    dir
}

/// A config running `argv` against `tests`, attested and quick to time out.
fn config(tests: &TempDir, argv: &[&str]) -> CheckerConfig {
    CheckerConfig {
        timeout: Duration::from_secs(10),
        sandbox_attestation: SANDBOX_ATTESTATION.to_string(),
        ..CheckerConfig::new(
            tests.path(),
            argv.iter().map(|part| (*part).to_string()).collect(),
        )
    }
}

/// Runs `sh` over the snapshot's script.
fn shell_argv() -> Vec<&'static str> {
    vec!["/bin/sh", "{tests}/run.sh"]
}

#[tokio::test]
async fn a_clean_exit_with_an_intact_suite_passes() {
    let tests = suite("exit 0\n");
    let checker = PinnedChecker::new(config(&tests, &shell_argv())).expect("builds");
    assert_eq!(checker.check("task", "attempt").await, Some(true));
}

#[tokio::test]
async fn a_nonzero_exit_is_a_fail_not_an_indeterminate() {
    // A failing test run is evidence, and the branch is entitled to act on it.
    let tests = suite("exit 1\n");
    let checker = PinnedChecker::new(config(&tests, &shell_argv())).expect("builds");
    assert_eq!(checker.check("task", "attempt").await, Some(false));
}

#[tokio::test]
async fn a_run_that_edits_the_suite_to_pass_reports_nothing() {
    // The reason the manifest exists. The attempt runs as the same user as the
    // tests, so read-only permissions alone cannot stop this; only re-hashing
    // after the run catches it.
    let tests =
        suite("chmod u+w \"$TESTS_DIR/run.sh\"; echo tampered >> \"$TESTS_DIR/run.sh\"; exit 0\n");
    let checker = PinnedChecker::new(config(&tests, &shell_argv())).expect("builds");
    assert_eq!(checker.check("task", "attempt").await, None);

    // Not vacuous: the edit really landed, and the run really exited clean, so
    // the only thing standing between it and a pass was the manifest.
    let script = std::fs::read_to_string(checker.tests_path().join("run.sh")).expect("read");
    assert!(
        script.contains("tampered"),
        "the suite was not modified: {script}"
    );
}

#[tokio::test]
async fn a_run_that_adds_a_test_file_reports_nothing() {
    // Additions change the tree even though every original file is untouched.
    let tests = suite("echo extra > \"$TESTS_DIR/added.txt\"; exit 0\n");
    let checker = PinnedChecker::new(config(&tests, &shell_argv())).expect("builds");
    assert_eq!(checker.check("task", "attempt").await, None);
    assert!(
        checker.tests_path().join("added.txt").exists(),
        "the run did not actually add a file"
    );
}

#[tokio::test]
async fn a_command_that_cannot_be_spawned_reports_nothing() {
    let tests = suite("exit 0\n");
    let checker =
        PinnedChecker::new(config(&tests, &["/nonexistent/checker-binary"])).expect("builds");
    assert_eq!(checker.check("task", "attempt").await, None);
}

#[tokio::test]
async fn a_run_that_outlives_its_timeout_reports_nothing() {
    // Never a fail: a suite that did not finish has not said anything about
    // the attempt.
    let tests = suite("sleep 30\n");
    let mut config = config(&tests, &shell_argv());
    config.timeout = Duration::from_millis(150);
    let checker = PinnedChecker::new(config).expect("builds");

    let started = std::time::Instant::now();
    assert_eq!(checker.check("task", "attempt").await, None);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn the_attempt_and_task_reach_the_command_as_files() {
    let tests = suite(
        "grep -q 'the attempt text' \"$ATTEMPT_FILE\" || exit 1\n\
         grep -q 'the task text' \"$TASK_FILE\" || exit 1\n\
         exit 0\n",
    );
    let checker = PinnedChecker::new(config(&tests, &shell_argv())).expect("builds");
    assert_eq!(
        checker.check("the task text", "the attempt text").await,
        Some(true)
    );
}

#[tokio::test]
async fn the_command_does_not_inherit_the_router_environment() {
    // The router holds provider credentials; a task's tests have no business
    // seeing them.
    unsafe {
        std::env::set_var("VGR_CHECKER_LEAK_PROBE", "secret");
    }
    let tests = suite("[ -z \"$VGR_CHECKER_LEAK_PROBE\" ] || exit 1\nexit 0\n");
    let checker = PinnedChecker::new(config(&tests, &shell_argv())).expect("builds");
    let verdict = checker.check("task", "attempt").await;
    unsafe {
        std::env::remove_var("VGR_CHECKER_LEAK_PROBE");
    }
    assert_eq!(verdict, Some(true));
}

#[test]
fn a_checker_without_the_sandbox_attestation_does_not_build() {
    // Nothing here enforces memory, CPU, network or filesystem reach. A
    // deployment that has not said something else does must not get a checker.
    let tests = suite("exit 0\n");
    let mut config = config(&tests, &shell_argv());
    config.sandbox_attestation = "approved".to_string();
    assert!(matches!(
        PinnedChecker::new(config),
        Err(CheckerSetupError::NotAttested)
    ));
}

#[test]
fn an_empty_command_does_not_build() {
    let tests = suite("exit 0\n");
    let config = CheckerConfig {
        sandbox_attestation: SANDBOX_ATTESTATION.to_string(),
        ..CheckerConfig::new(tests.path(), Vec::new())
    };
    assert!(matches!(
        PinnedChecker::new(config),
        Err(CheckerSetupError::EmptyCommand)
    ));
}

#[test]
fn the_manifest_covers_the_command_as_well_as_the_suite() {
    // Two checkers over identical tests but different commands are not
    // interchangeable, so a decision record naming one must not match the other.
    let tests = suite("exit 0\n");
    let first = PinnedChecker::new(config(&tests, &shell_argv())).expect("builds");
    let second = PinnedChecker::new(config(&tests, &["/bin/sh", "-c", "exit 0"])).expect("builds");
    assert_ne!(first.manifest().sha, second.manifest().sha);
}

#[test]
fn suites_differing_only_in_content_hash_differently() {
    let first = suite("exit 0\n");
    let second = suite("exit 1\n");
    let first = PinnedChecker::new(config(&first, &shell_argv())).expect("builds");
    let second = PinnedChecker::new(config(&second, &shell_argv())).expect("builds");
    assert_ne!(first.manifest().sha, second.manifest().sha);
}

#[test]
fn editing_the_operator_directory_after_construction_does_not_change_the_snapshot() {
    // The snapshot is taken once, so the suite cannot be swapped underneath a
    // running deployment.
    let tests = suite("exit 0\n");
    let checker = PinnedChecker::new(config(&tests, &shell_argv())).expect("builds");
    let pinned = checker.manifest().sha.clone();

    std::fs::write(tests.path().join("run.sh"), "exit 1\n").expect("rewrite");
    assert_eq!(checker.manifest().sha, pinned);
    assert!(checker.verifies("post-edit"), "the snapshot is unaffected");
}
