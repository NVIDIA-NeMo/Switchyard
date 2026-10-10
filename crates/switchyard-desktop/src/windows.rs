// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coding tools run in their own Windows console with explicit arguments and account directories.

use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

/// This function reads current-user MSIX registrations without launching or changing the Codex app.
pub fn codex_app() -> Option<PathBuf> {
    use windows::{Management::Deployment::PackageManager, core::HSTRING};
    // An empty SID restricts discovery to packages registered for the current user.
    let packages = PackageManager::new()
        .ok()?
        .FindPackagesByUserSecurityId(&HSTRING::new())
        .ok()?;
    for package in packages {
        let Some(name) = package.Id().ok().and_then(|id| id.Name().ok()) else {
            continue;
        };
        if name == "OpenAI.Codex" {
            return Some(PathBuf::from(package.InstalledPath().ok()?.to_os_string()));
        }
    }
    None
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalJob {
    pub binary: PathBuf,
    pub args: Vec<String>,
    pub directory: Option<PathBuf>,
    pub environment: Vec<(String, String)>,
    pub remove_environment: Vec<String>,
}
/// The child reads structured launch settings so arguments and account directories remain separate.
pub fn open_terminal(job: TerminalJob) -> Result<(), String> {
    use std::io::Write;
    let mut file = tempfile::Builder::new()
        .prefix("switchyard-terminal-")
        .suffix(".json")
        .tempfile()
        .map_err(|e| e.to_string())?;
    file.write_all(&serde_json::to_vec(&job).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let path = file.into_temp_path();
    let mut child = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    child.arg("--terminal-job").arg(&path);
    child.spawn().map_err(|error| error.to_string())?;
    path.keep().map_err(|error| error.to_string())?;
    Ok(())
}
pub fn attach_console() {
    unsafe {
        windows_console(false);
    }
}
pub fn allocate_console() {
    unsafe {
        windows_console(true);
    }
}
unsafe fn windows_console(allocate: bool) {
    // The GUI has no console; terminal mode and coding-tool jobs allocate one only when needed.
    unsafe extern "system" {
        fn AllocConsole() -> i32;
        fn AttachConsole(pid: u32) -> i32;
    }
    unsafe {
        if allocate {
            AllocConsole();
        } else {
            AttachConsole(u32::MAX);
        }
    }
}
pub fn terminal_job() -> Option<Result<(), String>> {
    use std::io::IsTerminal;
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.first().is_none_or(|v| v != "--terminal-job") {
        return None;
    }
    Some((|| {
        if args.len() != 2 {
            return Err("The terminal job requires one saved launch file.".into());
        }
        let path = Path::new(&args[1]);
        let job: TerminalJob =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        std::fs::remove_file(path).map_err(|e| e.to_string())?;
        allocate_console();
        let mut child = Command::new(&job.binary);
        child.args(&job.args).envs(job.environment);
        for name in job.remove_environment {
            child.env_remove(name);
        }
        if let Some(directory) = job.directory {
            child.current_dir(directory);
        }
        let status = child
            .status()
            .map_err(|e| format!("Could not start the coding tool: {e}"))?;
        println!("Coding tool exited with {status}. Press Enter to close this window.");
        if std::io::stdin().is_terminal() {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
        }
        if status.success() {
            Ok(())
        } else {
            Err(format!("Coding tool exited with {status}."))
        }
    })())
}
