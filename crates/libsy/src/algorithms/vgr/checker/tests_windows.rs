// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Windows tests for the pinned-manifest checker.

use std::sync::Arc;
use std::time::Duration;

use tempfile::TempDir;

use super::*;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn suite(script: &str) -> std::io::Result<TempDir> {
    let dir = TempDir::with_prefix("vgr-checker-suite-")?;
    std::fs::write(dir.path().join("run.ps1"), script)?;
    Ok(dir)
}

struct TestProvider;

#[async_trait::async_trait]
impl WorkspaceProvider for TestProvider {
    fn manifest_identity(&self) -> &str {
        "windows-test-provider-v1"
    }

    async fn materialize(
        &self,
        _task_text: &str,
        _attempt: &str,
    ) -> std::io::Result<CandidateWorkspace> {
        CandidateWorkspace::new(TempDir::with_prefix("vgr-candidate-")?)
    }
}

fn config(tests: &TempDir, argv: &[&str]) -> CheckerConfig {
    CheckerConfig {
        timeout: Duration::from_secs(10),
        sandbox_attestation: SANDBOX_ATTESTATION.to_string(),
        ..CheckerConfig::new(
            tests.path(),
            argv.iter().map(|part| (*part).to_string()).collect(),
            Arc::new(TestProvider),
        )
    }
}

fn powershell_argv() -> Vec<&'static str> {
    vec![
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "& (Join-Path $env:TESTS_DIR 'run.ps1')",
    ]
}

fn powershell_command(script: &str) -> Vec<String> {
    [
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        script,
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn ps_literal(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

#[tokio::test]
async fn a_clean_exit_with_an_intact_suite_passes() -> TestResult {
    let tests = suite("exit 0\r\n")?;
    let checker = PinnedChecker::new(config(&tests, &powershell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, Some(true));
    Ok(())
}

#[tokio::test]
async fn a_nonzero_exit_is_a_fail_not_an_indeterminate() -> TestResult {
    let tests = suite("exit 1\r\n")?;
    let checker = PinnedChecker::new(config(&tests, &powershell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, Some(false));
    Ok(())
}

#[tokio::test]
async fn a_run_that_edits_and_restores_file_bytes_reports_nothing() -> TestResult {
    let tests = suite(
        "$path = Join-Path $env:TESTS_DIR 'expected.txt'\r\n\
         $item = Get-Item -LiteralPath $path\r\n\
         $writeTime = $item.LastWriteTimeUtc\r\n\
         $item.IsReadOnly = $false\r\n\
         [IO.File]::WriteAllText($path, \"anything goes`r`n\")\r\n\
         [IO.File]::WriteAllText($path, \"expected answer`r`n\")\r\n\
         (Get-Item -LiteralPath $path).LastWriteTimeUtc = $writeTime\r\n\
         exit 0\r\n",
    )?;
    std::fs::write(tests.path().join("expected.txt"), "expected answer\r\n")?;
    let checker = PinnedChecker::new(config(&tests, &powershell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, None);
    Ok(())
}

#[tokio::test]
async fn a_command_that_cannot_be_spawned_reports_nothing() -> TestResult {
    let tests = suite("exit 0\r\n")?;
    let checker = PinnedChecker::new(config(&tests, &["Z:\\definitely-missing\\checker.exe"]))?;
    assert_eq!(checker.check("task", "attempt").await, None);
    Ok(())
}

#[tokio::test]
async fn a_run_that_outlives_its_timeout_reports_nothing() -> TestResult {
    let tests = suite("Start-Sleep -Seconds 30\r\n")?;
    let mut checker_config = config(&tests, &powershell_argv());
    checker_config.timeout = Duration::from_millis(300);
    let checker = PinnedChecker::new(checker_config)?;
    assert_eq!(checker.check("task", "attempt").await, None);
    Ok(())
}

#[tokio::test]
async fn a_timeout_kills_the_whole_process_tree() -> TestResult {
    let marker = TempDir::with_prefix("vgr-checker-marker-")?;
    let alive = marker.path().join("still-alive");
    let alive_literal = ps_literal(&alive);
    let tests = suite(&format!(
        "$childScript = \"while (`$true) {{ \
             [IO.File]::WriteAllText('{alive_literal}', 'alive'); \
             Start-Sleep -Milliseconds 20 \
         }}\"\r\n\
         Start-Process powershell.exe -ArgumentList @('-NoLogo', '-NoProfile', \
             '-NonInteractive', '-Command', $childScript) | Out-Null\r\n\
         while (!(Test-Path -LiteralPath '{alive_literal}')) {{ }}\r\n\
         Start-Sleep -Seconds 30\r\n"
    ))?;
    let mut checker_config = config(&tests, &powershell_argv());
    checker_config.timeout = Duration::from_secs(3);
    let checker = PinnedChecker::new(checker_config)?;
    assert_eq!(checker.check("task", "attempt").await, None);
    assert!(alive.exists(), "the background grandchild never ran");

    tokio::time::sleep(Duration::from_millis(300)).await;
    let settled = std::fs::metadata(&alive)?.modified()?;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let after = std::fs::metadata(&alive)?.modified()?;
    assert_eq!(settled, after, "the checker grandchild survived timeout");
    Ok(())
}

#[tokio::test]
async fn the_attempt_and_task_reach_the_command_as_files() -> TestResult {
    let tests = suite(
        "if ((Get-Content -Raw -LiteralPath $env:ATTEMPT_FILE) -ne 'the attempt text') { exit 1 }\r\n\
         if ((Get-Content -Raw -LiteralPath $env:TASK_FILE) -ne 'the task text') { exit 1 }\r\n\
         exit 0\r\n",
    )?;
    let checker = PinnedChecker::new(config(&tests, &powershell_argv()))?;
    assert_eq!(
        checker.check("the task text", "the attempt text").await,
        Some(true)
    );
    Ok(())
}

#[tokio::test]
async fn command_provider_materializes_request_data_only_through_files() -> TestResult {
    let provider = CommandWorkspaceProvider::new(CommandWorkspaceProviderConfig::new(
        powershell_command(
            "if ($env:NVIDIA_API_KEY) { exit 1 }; \
             if ($env:TESTS_DIR) { exit 1 }; \
             Copy-Item -LiteralPath $env:ATTEMPT_FILE \
                 -Destination (Join-Path $env:WORKSPACE_DIR 'candidate.txt'); \
             Copy-Item -LiteralPath $env:TASK_FILE \
                 -Destination (Join-Path $env:WORKSPACE_DIR 'task.txt')",
        ),
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
            vec!["powershell.exe".to_string()],
            Duration::from_secs(10),
        );
        config.env.push((key.to_string(), "override".to_string()));
        assert!(matches!(
            CommandWorkspaceProvider::new(config),
            Err(CommandWorkspaceProviderSetupError::ReservedEnvironmentKey(rejected))
                if rejected == key
        ));

        let mut lowercase = CommandWorkspaceProviderConfig::new(
            vec!["powershell.exe".to_string()],
            Duration::from_secs(10),
        );
        lowercase
            .env
            .push((key.to_ascii_lowercase(), "override".to_string()));
        assert!(matches!(
            CommandWorkspaceProvider::new(lowercase),
            Err(CommandWorkspaceProviderSetupError::ReservedEnvironmentKey(rejected))
                if rejected == key.to_ascii_lowercase()
        ));
    }
    Ok(())
}

#[test]
fn checker_rejects_case_insensitive_protected_environment() -> TestResult {
    let tests = suite("exit 0\r\n")?;
    for key in PROTECTED_ENV {
        let mut checker_config = config(&tests, &powershell_argv());
        checker_config
            .env
            .push((key.to_ascii_lowercase(), "override".to_string()));
        assert!(matches!(
            PinnedChecker::new(checker_config),
            Err(CheckerSetupError::ReservedEnvironmentKey(rejected))
                if rejected == key.to_ascii_lowercase()
        ));
    }
    Ok(())
}

#[tokio::test]
async fn the_command_does_not_inherit_the_router_environment() -> TestResult {
    let tests = suite("if ($env:NVIDIA_API_KEY) { exit 1 }\r\nexit 0\r\n")?;
    let checker = PinnedChecker::new(config(&tests, &powershell_argv()))?;
    assert_eq!(checker.check("task", "attempt").await, Some(true));
    Ok(())
}

#[test]
fn the_manifest_is_stable_across_checker_restarts() -> TestResult {
    let tests = suite("exit 0\r\n")?;
    std::fs::create_dir(tests.path().join("empty"))?;
    std::fs::write(tests.path().join("data.txt"), "same bytes")?;
    let first = PinnedChecker::new(config(&tests, &powershell_argv()))?;
    let first_identity = first.manifest().sha.clone();
    drop(first);
    let second = PinnedChecker::new(config(&tests, &powershell_argv()))?;
    assert_eq!(second.manifest_identity(), first_identity);
    Ok(())
}

#[test]
fn snapshot_entry_and_byte_limits_fail_closed() -> TestResult {
    let tests = suite("exit 0\r\n")?;
    let mut entry_limited = config(&tests, &powershell_argv());
    entry_limited.max_snapshot_entries = 1;
    assert!(matches!(
        PinnedChecker::new(entry_limited),
        Err(CheckerSetupError::SnapshotEntryLimit(1))
    ));

    let mut byte_limited = config(&tests, &powershell_argv());
    byte_limited.max_snapshot_bytes = 1;
    assert!(matches!(
        PinnedChecker::new(byte_limited),
        Err(CheckerSetupError::SnapshotByteLimit(1))
    ));
    Ok(())
}
